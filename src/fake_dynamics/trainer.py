import json
import math
import sys
import time
from pathlib import Path

import torch

from .backend import make_backend
from .checkpoint import resume, save_checkpoint
from .data import PromptStream
from .diffusion import ca_time, uniform_time
from .ema import ema_master, release_ema
from .objectives import (
    consistency_loss,
    corrected_fake,
    direct_gradient,
    direct_proxy,
    guided,
    mixed_target,
)
from .runtime import (
    JsonLogger,
    Runtime,
    environment_manifest,
    restore_rng,
    rng_state,
    unwrap,
    write_json,
)
from .sampling import PAIRS, backsimulate, fake_transition, generator_x0, sample_generator
from .telemetry import TensorBoardLogger, event_time


@torch.no_grad()
def update_ema(ema, generator, decay):
    release_ema(ema)
    ema = ema_master(ema)
    source = dict(unwrap(generator).named_parameters())
    for name, parameter in ema.named_parameters():
        if parameter.dtype != torch.float32:
            raise ValueError("EMA state must remain FP32")
        parameter.mul_(decay).add_(
            source[name].detach().to(device=parameter.device, dtype=torch.float32), alpha=1 - decay
        )
    source_buffers = dict(unwrap(generator).named_buffers())
    for name, buffer in ema.named_buffers():
        buffer.copy_(source_buffers[name].to(buffer.device))


class Trainer:
    def __init__(self, config, resume_path=None):
        config.validate()
        config.require_runtime()
        self.config = config
        self.runtime = Runtime(config)
        output = Path(config.output_dir)
        if not resume_path and (output / "run_manifest.json").exists():
            raise ValueError("Run already exists. Resume it or choose a new output_dir.")
        self.logger = JsonLogger(output / f"metrics_rank{self.runtime.rank}.jsonl")
        self.tensorboard = TensorBoardLogger(config, self.runtime.rank)
        self.backend = make_backend(config, self.runtime.device, import_weights=resume_path is None)
        self.backend.generator.train()
        self.backend.fake.train()
        self.data = PromptStream(
            config.prompts_path,
            config.per_device_batch_size,
            self.runtime.rank,
            self.runtime.world_size,
            config.seed,
        )
        self.tensorboard.text(
            "run/fixed_prompts",
            json.dumps(
                [
                    {"id": row["id"], "prompt": row["prompt"], "seed": config.probe_seed + i}
                    for i, row in enumerate(self.data.records[: config.fixed_sample_count])
                ],
                ensure_ascii=False,
            ),
        )
        self.shape = (
            config.per_device_batch_size,
            4,
            config.resolution // 8,
            config.resolution // 8,
        )
        self.optimizers = {}
        for name, lr in (("generator", config.generator_lr), ("fake", config.fake_lr)):
            model = getattr(self.backend, name)
            self.optimizers[name] = self.runtime.make_optimizer(model, lr)
            setattr(self.backend, name, self.runtime.wrap(model))
        if self.runtime.world_size > 1:
            if not resume_path:
                ema_master(self.backend.ema).load_state_dict(
                    unwrap(self.backend.generator).state_dict(), strict=True
                )
            if config.backend == "toy":
                for value in self.backend.teacher.state_dict().values():
                    torch.distributed.broadcast(value, src=0)
        self.scalers = {
            name: torch.amp.GradScaler("cuda", enabled=config.mixed_precision == "fp16")
            for name in self.optimizers
        }
        self.state = {
            "generator_updates": 0,
            "fake_updates": 0,
            "generator_attempts": 0,
            "fake_attempts": 0,
            "fake_in_cycle": 0,
            "warmup_completed": 0,
            "failed_generator": 0,
            "failed_fake": 0,
            "forward_counts": {"generator": 0, "fake": 0, "teacher": 0, "ema": 0},
            "elapsed_training_seconds": 0.0,
        }
        self.probe_state = {}
        self.assert_ownership()
        if resume_path:
            resume(self, resume_path)
            if config.total_generator_updates <= self.state["generator_updates"]:
                raise ValueError(
                    "Resume needs a total G budget larger than restored k_G; choose the next stage"
                )
        if self.runtime.rank == 0:
            write_json(
                output / ("resume_manifest.json" if resume_path else "run_manifest.json"),
                {
                    "config": config.as_dict(),
                    "config_hash": config.digest(),
                    "environment": environment_manifest(),
                    "initialization": self.backend.provenance,
                    "scheduler": self.backend.scheduler_config,
                    "global_batch_size": self.runtime.effective_batch_size,
                    "global_microbatch_size": config.per_device_batch_size
                    * self.runtime.world_size,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "batch_reduction": "mean over ranks and microbatches; clip/step/EMA once per optimizer window",
                    "world_size": self.runtime.world_size,
                    "prompt_sha256": self.data.file_hash,
                    "normalization": "dmd2_x0_branchwise_cfg_reference_adapter",
                    "update_order": f"{config.fake_updates_per_generator} successful F updates then 1 successful G update; retry skipped steps",
                    "lr_schedule": "constant; no inherited 19000-step scheduler state",
                    "resume_from": str(resume_path) if resume_path else None,
                },
            )
        self.runtime.barrier()

    def log(self, event, beta=None, lambda_cd=None, aggregate=False, **values):
        schedule_beta, schedule_cd = self.config.weights(self.state["generator_updates"])
        record = dict(
            event=event,
            unix_time=event_time(),
            rank=self.runtime.rank,
            k_G=self.state["generator_updates"],
            k_F=self.state["fake_updates"],
            beta=schedule_beta if beta is None else beta,
            lambda_cd=schedule_cd if lambda_cd is None else lambda_cd,
            **values,
        )
        self.logger.log(**record)
        dashboard_record = self.runtime.aggregate_metrics(record) if aggregate else record
        self.tensorboard.record(event, dashboard_record)

    def assert_ownership(self):
        identities = {}
        for name in ("generator", "fake"):
            expected = {id(p) for p in getattr(self.backend, name).parameters()}
            actual = {
                id(p) for group in self.optimizers[name].param_groups for p in group["params"]
            }
            if actual != expected:
                raise ValueError(f"Wrong optimizer parameter ownership: {name}")
            identities[name] = actual
        if identities["generator"] & identities["fake"]:
            raise ValueError("Generator and fake share parameters")
        for model in (self.backend.teacher, self.backend.ema):
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise ValueError("Teacher and EMA must be frozen and in eval mode")

    @torch.no_grad()
    def teacher_pair(self, x, times, c):
        self.state["forward_counts"]["teacher"] += 2
        return (
            self.backend.teacher(x, times, c),
            self.backend.teacher(x, times, c.unconditional()),
        )

    def fresh_input(self):
        c = self.backend.encode(self.data.next())
        tau, h = backsimulate(
            unwrap(self.backend.generator),
            self.backend.diffusion,
            c,
            self.shape,
            self.runtime.device,
            anchor_index=(
                self.state["generator_updates"] % 4
                if self.config.debug_anchor_cycle
                else getattr(self, "_window_anchor_index", None)
            ),
        )
        self.state["forward_counts"]["generator"] += (999 - tau) // 250
        return c, tau, h

    def optimizer_step(self, name, loss, *, backward_done=False, window_finite=True):
        optimizer = self.optimizers[name]
        scaler = self.scalers[name]
        parameters = list(getattr(self.backend, name).parameters())
        self.state[f"{name}_attempts"] += 1
        succeeded, norm = False, float("nan")
        finite = self.runtime.all_true(window_finite and bool(torch.isfinite(loss.detach()).all()))
        if not backward_done and (finite or self.runtime.world_size > 1):
            # Even invalid DDP losses must complete reducer hooks on every rank before
            # discarding the window. Otherwise the next forward can hang/fail.
            scaler.scale(loss).backward()
        if finite:
            scaler.unscale_(optimizer)  # exactly once
            norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
            norm = float(norm_tensor.item())
            if self.runtime.all_true(bool(torch.isfinite(norm_tensor))):
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # AdamW.step returns None even on success; AMP state determines skips.
                succeeded = scaler.get_scale() >= scale_before
            elif scaler.is_enabled():
                # All ranks skip together, including ranks with finite local gradients.
                scaler.update(new_scale=scaler.get_scale() / 2)
        elif backward_done and scaler.is_enabled():
            # Initialize scaler state even if the first microbatch was nonfinite.
            scaler.update(new_scale=scaler.get_scale() / 2)
        optimizer.zero_grad(set_to_none=True)
        if succeeded:
            self.state[f"{name}_updates"] += 1
            self.state[f"failed_{name}"] = 0
            if name == "generator":
                ema_started = time.monotonic()
                update_ema(
                    self.backend.ema, self.backend.generator, self.config.generator_ema_decay
                )
                self.state["last_ema_update_seconds"] = time.monotonic() - ema_started
        else:
            self.state[f"failed_{name}"] += 1
            self.log(
                "optimizer_step_skipped",
                network=name,
                loss=float(loss.detach()),
                grad_norm=norm,
                consecutive=self.state[f"failed_{name}"],
                amp_scale=scaler.get_scale(),
            )
            if self.state[f"failed_{name}"] >= self.config.max_consecutive_failed_updates:
                # All ranks reach this path together. Keep a resumable failure checkpoint.
                save_checkpoint(self, "failed")
                raise FloatingPointError(
                    f"Repeated nonfinite {name} updates. See metrics_rank*.jsonl"
                )
        return succeeded, norm

    def accumulated_step(self, name, microbatch):
        """One logical optimizer update; no model/EMA/schedule changes within its window."""
        count = self.config.gradient_accumulation_steps
        if count > 1 and not self.config.debug_anchor_cycle:
            # DMD2 uses ONE shared anchor per optimizer minibatch. Hold it across
            # ranks AND microbatches instead of silently mixing all four anchors.
            selected = torch.randint(0, 4, (1,), device=self.runtime.device)
            if self.runtime.world_size > 1:
                torch.distributed.broadcast(selected, src=0)
            self._window_anchor_index = int(selected.item())
        self.optimizers[name].zero_grad(set_to_none=True)
        rows = []
        window_finite = True
        for index in range(count):
            with self.runtime.accumulation_context(getattr(self.backend, name), index == count - 1):
                loss, metrics = microbatch()
                rows.append(metrics)
                if count == 1:
                    success, norm = self.optimizer_step(name, loss)
                else:
                    window_finite = self.runtime.all_true(bool(torch.isfinite(loss.detach())))
                    # Mean reduction, not a sum. Backward immediately releases this graph;
                    # never retain 32/64 SDXL graphs. Keep even rejected DDP hooks paired.
                    self.scalers[name].scale(loss / count).backward()
            del loss
            if not window_finite:
                break
        metrics = self.reduce_microbatch_metrics(rows, count)
        self._window_anchor_index = None
        if count > 1:
            success, norm = self.optimizer_step(
                name,
                torch.tensor(metrics["loss"], device=self.runtime.device),
                backward_done=True,
                window_finite=window_finite,
            )
        metrics.update(
            success=success,
            grad_norm=norm,
            gradient_clipped=norm > self.config.max_grad_norm,
            microbatches_processed=len(rows),
            accumulation_steps=count,
            local_samples_processed=len(rows) * self.config.per_device_batch_size,
            global_samples_processed=len(rows)
            * self.config.per_device_batch_size
            * self.runtime.world_size,
            global_batch_size=self.runtime.effective_batch_size,
        )
        return success, metrics

    def reduce_microbatch_metrics(self, rows, count):
        result = {}
        for key in set().union(*(row.keys() for row in rows)):
            if key in ("anchor", "cd_active", "cd_to_direct_y_gradient_ratio"):
                continue
            values = [row[key] for row in rows if row.get(key) is not None]
            if key.startswith("samples_"):
                result[key] = sum(values)
            elif key.startswith("corrected_error_"):
                weights = [row[key.replace("corrected_error_", "samples_")] for row in rows]
                total = sum(weights)
                result[key] = (
                    sum((row[key] or 0.0) * weight for row, weight in zip(rows, weights)) / total
                    if total
                    else None
                )
            elif key in ("direct_y_gradient_norm", "weighted_cd_y_gradient_norm"):
                # Output blocks are disjoint across microbatches, so combine their
                # gradient norms in quadrature, including the loss/count reduction.
                result[key] = math.sqrt(sum(value**2 for value in values)) / count
            elif key in ("dm_output_norm", "ca_output_norm"):
                result[key] = math.sqrt(sum(value**2 for value in values))
            else:
                result[key] = sum(values) / len(values) if values else None
        anchors = {row["anchor"] for row in rows}
        result["anchor"] = next(iter(anchors)) if len(anchors) == 1 else None
        for anchor in self.config.generator_anchors:
            result[f"anchor_samples_{anchor}"] = (
                sum(row["anchor"] == anchor for row in rows) * self.config.per_device_batch_size
            )
        if "cd_active" in rows[0]:
            result["cd_active"] = any(row["cd_active"] for row in rows)
            result["cd_active_fraction"] = sum(row["cd_active"] for row in rows) / len(rows)
            result["cd_to_direct_y_gradient_ratio"] = result["weighted_cd_y_gradient_norm"] / max(
                result["direct_y_gradient_norm"], 1e-12
            )
        return result

    def fake_microbatch(self, beta):
        d = self.backend.diffusion
        with self.runtime.autocast():
            with torch.no_grad():
                c, tau, h = self.fresh_input()
                y = generator_x0(unwrap(self.backend.generator), d, h, tau, c)
                self.state["forward_counts"]["generator"] += 1
                t = uniform_time(0, 999, y)
                noise = torch.randn_like(y)
                x = d.renoise(y, t, noise)
                teacher_cfg = None
                if beta > 0:
                    tc, tu = self.teacher_pair(x, t, c)
                    teacher_cfg = guided(tc, tu, self.config.teacher_cfg)
                target = mixed_target(noise, teacher_cfg, beta)
            fake = self.backend.fake(x, t, c)
            self.state["forward_counts"]["fake"] += 1
            loss = (fake.float() - target).square().mean()
        metrics = {
            "loss": float(loss.detach()),
            "loss_fake_mixed": float(loss.detach()),
            "anchor": tau,
        }
        # For beta=0 teacher_cfg is irrelevant; this is exact ordinary fake error.
        with torch.no_grad():
            estimate = (
                fake.float() if beta == 0 else corrected_fake(fake.float(), teacher_cfg, beta)
            )
            error = (estimate - noise).square().flatten(1).mean(1)
            for low, high in ((0, 249), (250, 499), (500, 749), (750, 999)):
                mask = (t >= low) & (t <= high)
                metrics[f"corrected_error_{low}_{high}"] = (
                    float(error[mask].mean()) if mask.any() else None
                )
                metrics[f"samples_{low}_{high}"] = int(mask.sum())
        return loss, metrics

    def fake_step(self, beta):
        started = time.monotonic()
        success, metrics = self.accumulated_step("fake", lambda: self.fake_microbatch(beta))
        # This training-window statistic is not a held-out oracle estimate of critic lag.
        metrics.update(
            step_seconds=time.monotonic() - started,
            optimizer_attempt=self.state["fake_attempts"],
            learning_rate=self.optimizers["fake"].param_groups[0]["lr"],
        )
        self.log("fake_step", beta=beta, aggregate=True, **metrics)
        return success

    def generator_microbatch(self, beta, cd_weight):
        d = self.backend.diffusion
        with self.runtime.autocast():
            c, tau, h = self.fresh_input()
            y = generator_x0(self.backend.generator, d, h, tau, c)
            self.state["forward_counts"]["generator"] += 1
            with torch.no_grad():
                t_dm, t_ca = uniform_time(20, 980, y), ca_time(tau, y)
                x_dm = d.renoise(y.detach(), t_dm, torch.randn_like(y))
                x_ca = d.renoise(y.detach(), t_ca, torch.randn_like(y))
                tc_dm, tu_dm = self.teacher_pair(x_dm, t_dm, c)
                tc_ca, tu_ca = self.teacher_pair(x_ca, t_ca, c)
                fake = unwrap(self.backend.fake)(x_dm, t_dm, c)
                self.state["forward_counts"]["fake"] += 1
                gradient, g_dm, g_ca = direct_gradient(
                    d,
                    y.detach(),
                    x_dm,
                    t_dm,
                    fake,
                    tc_dm,
                    tu_dm,
                    x_ca,
                    t_ca,
                    tc_ca,
                    tu_ca,
                    self.config.teacher_cfg,
                    beta,
                )
                target_cd = None
                if cd_weight > 0 and tau in PAIRS:
                    end, hs = fake_transition(unwrap(self.backend.fake), d, h, tau, c)
                    target_cd = generator_x0(self.backend.ema, d, hs, end, c)
                    self.state["forward_counts"]["fake"] += 2
                    self.state["forward_counts"]["ema"] += 1
            loss_direct = direct_proxy(y, gradient)
            loss_cd = consistency_loss(y, target_cd)
            loss = loss_direct + cd_weight * loss_cd
        release_ema(self.backend.ema)
        direct_output_norm = float(gradient.norm()) / y.numel()
        cd_output_gradient = (
            torch.zeros_like(y)
            if target_cd is None
            else (2 * cd_weight * (y.detach() - target_cd) / y.numel())
        )
        metrics = dict(
            anchor=tau,
            loss=float(loss.detach()),
            loss_direct=float(loss_direct.detach()),
            loss_cd=float(loss_cd.detach()),
            dm_output_norm=float(g_dm.norm()),
            ca_output_norm=float(g_ca.norm()),
            direct_y_gradient_norm=direct_output_norm,
            weighted_cd_y_gradient_norm=float(cd_output_gradient.norm()),
            cd_active=target_cd is not None,
            **(
                {"cd_endpoint_mean": float(hs.mean()), "cd_endpoint_std": float(hs.std())}
                if target_cd is not None
                else {}
            ),
        )
        return loss, metrics

    def generator_step(self, beta, cd_weight):
        started = time.monotonic()
        success, metrics = self.accumulated_step(
            "generator", lambda: self.generator_microbatch(beta, cd_weight)
        )
        self.log(
            "generator_step",
            aggregate=True,
            beta=beta,
            lambda_cd=cd_weight,
            schedule_for_next_update=self.config.weights(self.state["generator_updates"]),
            step_seconds=time.monotonic() - started,
            ema_update_seconds=self.state.get("last_ema_update_seconds", 0.0) if success else 0.0,
            optimizer_attempt=self.state["generator_attempts"],
            learning_rate=self.optimizers["generator"].param_groups[0]["lr"],
            forward_counts=dict(self.state["forward_counts"]),
            **metrics,
        )
        return success

    @torch.no_grad()
    def probe(self):
        # Every probe uses an independent generator so it cannot perturb training RNG.
        d, device = self.backend.diffusion, self.runtime.device
        c = self.backend.encode(
            [r["prompt"] for r in self.data.records[: self.config.per_device_batch_size]]
        )
        seed = self.config.probe_seed
        noise_rng = torch.Generator(device=device).manual_seed(seed)
        with self.runtime.autocast():
            h = torch.randn(self.shape, device=device, generator=noise_rng)
            times = torch.full((h.shape[0],), 999, device=device, dtype=torch.long)
            f = unwrap(self.backend.fake)(h, times, c)
            y = generator_x0(unwrap(self.backend.generator), d, h, 999, c)
            end, hs = fake_transition(unwrap(self.backend.fake), d, h, 999, c)
            target = generator_x0(self.backend.ema, d, hs, end, c)
            _, reference_h = backsimulate(
                unwrap(self.backend.generator),
                d,
                c,
                self.shape,
                device,
                anchor_index=1,
                noise_generator=torch.Generator(device=device).manual_seed(seed),
            )
            full = sample_generator(
                unwrap(self.backend.generator),
                d,
                c,
                self.shape,
                device,
                torch.Generator(device=device).manual_seed(seed),
            )
            # Fresh held-out generated/noised probe, independent of optimizer minibatches.
            probe_t = torch.randint(0, 1000, (h.shape[0],), device=device, generator=noise_rng)
            probe_noise = torch.randn(self.shape, device=device, generator=noise_rng)
            probe_x = d.renoise(full, probe_t, probe_noise)
            probe_fake = unwrap(self.backend.fake)(probe_x, probe_t, c)
            beta, cd_weight = self.config.weights(self.state["generator_updates"])
            tc, tu = self.teacher_pair(probe_x, probe_t, c)
            qhat = corrected_fake(probe_fake, guided(tc, tu, self.config.teacher_cfg), beta)
        release_ema(self.backend.ema)
        self.state["forward_counts"]["generator"] += 6
        self.state["forward_counts"]["fake"] += 4
        self.state["forward_counts"]["ema"] += 1
        metrics = {
            "fixed_cd_residual": float((y - target).square().mean()),
            "heldout_corrected_error": float((qhat - probe_noise).square().mean()),
            "cd_endpoint_mean": float(hs.mean()),
            "cd_endpoint_std": float(hs.std()),
            "baseline_endpoint_mean": float(reference_h.mean()),
            "baseline_endpoint_std": float(reference_h.std()),
            # Output-space ratio is explicitly NOT a parameter-gradient alignment measure.
            "cd_output_gradient_norm": float((2 * cd_weight * (y - target) / y.numel()).norm()),
            "seed": seed,
        }
        probe_error = (qhat - probe_noise).square().flatten(1).mean(1)
        for low, high in ((0, 249), (250, 499), (500, 749), (750, 999)):
            mask = (probe_t >= low) & (probe_t <= high)
            metrics[f"heldout_corrected_error_{low}_{high}"] = (
                float(probe_error[mask].mean()) if mask.any() else None
            )
            metrics[f"heldout_samples_{low}_{high}"] = int(mask.sum())
        for key, tensor in (("fake", f), ("generator", full)):
            previous = self.probe_state.get(key)
            metrics[f"fixed_{key}_output_drift_mse"] = (
                None if previous is None else float((tensor.cpu() - previous).square().mean())
            )
            self.probe_state[key] = tensor.detach().float().cpu()
        self.log("fixed_probe", aggregate=True, **metrics)

    @torch.no_grad()
    def fixed_samples(self):
        if self.config.backend != "sdxl":
            self.log("fixed_samples_skipped", reason="Toy backend has no image decoder")
            return
        self.runtime.barrier()
        status = None
        if self.runtime.rank == 0:
            training_rng = rng_state()
            generator_mode = self.backend.generator.training
            try:
                unwrap(self.backend.generator).eval()
                device, d = self.runtime.device, self.backend.diffusion
                sample_records = self.data.records[: self.config.fixed_sample_count]
                with self.runtime.autocast():
                    for name in ("generator", "ema"):
                        model = unwrap(getattr(self.backend, name))
                        for index, record in enumerate(sample_records):
                            c = self.backend.encode([record["prompt"]])
                            rng = torch.Generator(device=device).manual_seed(
                                self.config.probe_seed + index
                            )
                            latent = sample_generator(
                                model,
                                d,
                                c,
                                (1, 4, self.config.resolution // 8, self.config.resolution // 8),
                                device,
                                rng,
                            )
                            image = self.backend.decode(latent)[0].permute(2, 0, 1)
                            self.tensorboard.image(
                                f"fixed_samples/{name}/{index}",
                                image,
                                self.state["generator_updates"],
                            )
                            self.state["forward_counts"][name] += 4
                self.log(
                    "fixed_samples",
                    count=len(sample_records),
                    seed_base=self.config.probe_seed,
                    storage="TensorBoard only; no duplicate PNG directory",
                )
                status = {"ok": True}
            except Exception as error:
                status = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            finally:
                release_ema(self.backend.ema)
                # VAE is used only for samples; release its resident allocation after this event.
                self.backend.vae = None
                unwrap(self.backend.generator).train(generator_mode)
                restore_rng(training_rng)
        status = self.runtime.broadcast(status)
        if not status["ok"]:
            raise RuntimeError(f"Fixed sample logging failed: {status['error']}")

    def performance(self, cycle_seconds):
        metrics = {
            "cycle_seconds": cycle_seconds,
            "global_batch_size": self.runtime.effective_batch_size,
            "global_microbatch_size": self.config.per_device_batch_size * self.runtime.world_size,
            "accumulation_steps": self.config.gradient_accumulation_steps,
            "successful_generator_samples": self.state["generator_updates"]
            * self.runtime.effective_batch_size,
            "successful_fake_samples": self.state["fake_updates"]
            * self.runtime.effective_batch_size,
            "generator_lr": self.optimizers["generator"].param_groups[0]["lr"],
            "fake_lr": self.optimizers["fake"].param_groups[0]["lr"],
        }
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metrics["host_process_peak_rss_gib"] = peak / (
                2**30 if sys.platform == "darwin" else 2**20
            )
        except ImportError:
            pass
        if self.runtime.device.type == "cuda":
            metrics.update(
                cuda_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                cuda_reserved_gib=torch.cuda.memory_reserved() / 2**30,
                cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                cuda_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
            torch.cuda.reset_peak_memory_stats()
        metrics["forward_counts"] = dict(self.state["forward_counts"])
        self.log("performance", aggregate=True, **metrics)

    def run(self, continue_after_warmup=False):
        started = time.monotonic()
        elapsed_at_start = self.state["elapsed_training_seconds"]
        try:
            while self.state["warmup_completed"] < self.config.fake_warmup_updates:
                if self.fake_step(0.0):
                    self.state["warmup_completed"] += 1
            if (
                self.config.fake_initialization == "teacher"
                and self.state["generator_updates"] == 0
                and not continue_after_warmup
            ):
                self.probe()
                save_checkpoint(self, "warmup_complete")
                self.log(
                    "fallback_warmup_review_required",
                    reason="Inspect tracking diagnostics before --continue-after-warmup",
                )
                return
            while self.state["generator_updates"] < self.config.total_generator_updates:
                cycle_started = time.monotonic()
                beta, cd_weight = self.config.weights(self.state["generator_updates"])
                while self.state["fake_in_cycle"] < self.config.fake_updates_per_generator:
                    if self.fake_step(beta):
                        self.state["fake_in_cycle"] += 1
                if not self.generator_step(beta, cd_weight):
                    continue  # retry G with the same schedule; do not add another 5 F updates
                self.state["fake_in_cycle"] = 0
                k = self.state["generator_updates"]
                self.state["elapsed_training_seconds"] = (
                    elapsed_at_start + time.monotonic() - started
                )
                if k % self.config.performance_interval_g_updates == 0:
                    self.performance(time.monotonic() - cycle_started)
                if (
                    self.config.probe_interval_g_updates
                    and k % self.config.probe_interval_g_updates == 0
                ):
                    self.probe()
                if (
                    self.config.sample_interval_g_updates
                    and k % self.config.sample_interval_g_updates == 0
                ):
                    self.fixed_samples()
                if (
                    k % self.config.checkpoint_interval_g_updates == 0
                    and k < self.config.total_generator_updates
                ):
                    save_checkpoint(self)
            self.state["elapsed_training_seconds"] = elapsed_at_start + time.monotonic() - started
            save_checkpoint(self, "final")
            self.log(
                "training_complete",
                elapsed_seconds=self.state["elapsed_training_seconds"],
                allocated_gpu_hours=(
                    self.runtime.world_size * self.state["elapsed_training_seconds"] / 3600
                    if self.runtime.device.type == "cuda"
                    else 0
                ),
            )
        finally:
            release_ema(self.backend.ema)
            self.tensorboard.close()
            self.runtime.close()

import time
from pathlib import Path

import torch

from .backend import make_backend
from .checkpoint import resume, save_checkpoint
from .data import PromptStream
from .diffusion import ca_time, uniform_time
from .objectives import (
    consistency_loss,
    corrected_fake,
    direct_gradient,
    direct_proxy,
    guided,
    mixed_target,
)
from .runtime import JsonLogger, Runtime, environment_manifest, unwrap, write_json
from .sampling import PAIRS, backsimulate, fake_transition, generator_x0, sample_generator


@torch.no_grad()
def update_ema(ema, generator, decay):
    source = dict(unwrap(generator).named_parameters())
    for name, parameter in ema.named_parameters():
        if parameter.dtype != torch.float32:
            raise ValueError("EMA state must remain FP32")
        parameter.mul_(decay).add_(source[name].detach().float(), alpha=1 - decay)
    source_buffers = dict(unwrap(generator).named_buffers())
    for name, buffer in ema.named_buffers():
        buffer.copy_(source_buffers[name])


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
        self.shape = (
            config.per_device_batch_size,
            4,
            config.resolution // 8,
            config.resolution // 8,
        )
        self.optimizers = {}
        for name, lr in (("generator", config.generator_lr), ("fake", config.fake_lr)):
            model = getattr(self.backend, name)
            self.optimizers[name] = torch.optim.AdamW(
                model.parameters(), lr=lr, betas=config.adam_betas, weight_decay=config.weight_decay
            )
            setattr(self.backend, name, self.runtime.wrap(model))
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
        if self.runtime.rank == 0:
            write_json(
                output / ("resume_manifest.json" if resume_path else "run_manifest.json"),
                {
                    "config": config.as_dict(),
                    "config_hash": config.digest(),
                    "environment": environment_manifest(),
                    "initialization": self.backend.provenance,
                    "scheduler": self.backend.scheduler_config,
                    "global_batch_size": config.per_device_batch_size * self.runtime.world_size,
                    "world_size": self.runtime.world_size,
                    "prompt_sha256": self.data.file_hash,
                    "normalization": "dmd2_x0_branchwise_cfg_reference_adapter",
                    "update_order": f"{config.fake_updates_per_generator} successful F updates then 1 successful G update; retry skipped steps",
                    "lr_schedule": "constant; no inherited 19000-step scheduler state",
                    "resume_from": str(resume_path) if resume_path else None,
                },
            )
        self.runtime.barrier()

    def log(self, event, beta=None, lambda_cd=None, **values):
        schedule_beta, schedule_cd = self.config.weights(self.state["generator_updates"])
        self.logger.log(
            event=event,
            rank=self.runtime.rank,
            k_G=self.state["generator_updates"],
            k_F=self.state["fake_updates"],
            beta=schedule_beta if beta is None else beta,
            lambda_cd=schedule_cd if lambda_cd is None else lambda_cd,
            **values,
        )

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
        )
        self.state["forward_counts"]["generator"] += (999 - tau) // 250
        return c, tau, h

    def optimizer_step(self, name, loss):
        optimizer = self.optimizers[name]
        scaler = self.scalers[name]
        parameters = list(getattr(self.backend, name).parameters())
        self.state[f"{name}_attempts"] += 1
        succeeded, norm = False, float("nan")
        if self.runtime.all_true(bool(torch.isfinite(loss.detach()).all())):
            scaler.scale(loss).backward()
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
        optimizer.zero_grad(set_to_none=True)
        if succeeded:
            self.state[f"{name}_updates"] += 1
            self.state[f"failed_{name}"] = 0
            if name == "generator":
                update_ema(
                    self.backend.ema, self.backend.generator, self.config.generator_ema_decay
                )
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

    def fake_step(self, beta):
        self.optimizers["fake"].zero_grad(set_to_none=True)
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
        success, norm = self.optimizer_step("fake", loss)
        metrics = {
            "loss_fake_mixed": float(loss.detach()),
            "grad_norm": norm,
            "anchor": tau,
            "success": success,
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
        # This minibatch statistic is not a held-out oracle estimate of critic lag.
        self.log("fake_step", beta=beta, **metrics)
        return success

    def generator_step(self, beta, cd_weight):
        self.optimizers["generator"].zero_grad(set_to_none=True)
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
        success, norm = self.optimizer_step("generator", loss)
        direct_output_norm = float(gradient.norm()) / y.numel()
        cd_output_gradient = (
            torch.zeros_like(y)
            if target_cd is None
            else (2 * cd_weight * (y.detach() - target_cd) / y.numel())
        )
        self.log(
            "generator_step",
            beta=beta,
            lambda_cd=cd_weight,
            schedule_for_next_update=self.config.weights(self.state["generator_updates"]),
            success=success,
            anchor=tau,
            loss=float(loss.detach()),
            loss_direct=float(loss_direct.detach()),
            loss_cd=float(loss_cd.detach()),
            grad_norm=norm,
            dm_output_norm=float(g_dm.norm()),
            ca_output_norm=float(g_ca.norm()),
            direct_y_gradient_norm=direct_output_norm,
            weighted_cd_y_gradient_norm=float(cd_output_gradient.norm()),
            cd_to_direct_y_gradient_ratio=float(cd_output_gradient.norm())
            / max(direct_output_norm, 1e-12),
            cd_active=target_cd is not None,
            forward_counts=dict(self.state["forward_counts"]),
            **(
                {"cd_endpoint_mean": float(hs.mean()), "cd_endpoint_std": float(hs.std())}
                if target_cd is not None
                else {}
            ),
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
        self.log("fixed_probe", **metrics)

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
                if (
                    self.config.probe_interval_g_updates
                    and k % self.config.probe_interval_g_updates == 0
                ):
                    self.probe()
                if k % self.config.checkpoint_interval_g_updates == 0:
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
            self.runtime.close()

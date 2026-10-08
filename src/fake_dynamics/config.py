import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    backend: str = "sdxl"
    backbone: str = "stabilityai/stable-diffusion-xl-base-1.0"
    backbone_revision: str = "462165984030d82259a11f4367a4eed129e94a7b"
    generator_checkpoint: str | None = None
    guidance_checkpoint: str | None = None
    fake_initialization: str = "paired"  # paired | teacher (explicit fallback)
    prompts_path: str | None = None
    output_dir: str = "runs/main"
    resolution: int = 1024
    prediction_type: str = "epsilon"
    generator_anchors: tuple[int, ...] = (999, 749, 499, 249)
    teacher_cfg: float = 8.0
    generator_lr: float = 5e-7
    fake_lr: float = 5e-7
    adam_betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    max_grad_norm: float = 10.0
    fake_updates_per_generator: int = 5
    fake_warmup_updates: int = 0
    fake_timestep_range_inclusive: tuple[int, int] = (0, 999)
    dm_timestep_range_inclusive: tuple[int, int] = (20, 980)
    teacher_anchor_beta_max: float = 0.05
    teacher_anchor_start_g_update: int = 100
    teacher_anchor_ramp_g_updates: int = 1000
    consistency_weight_max: float = 0.1
    consistency_ramp_g_updates: int = 100
    generator_ema_decay: float = 0.99
    # Overrides are only for branch coverage or deliberate ablations; saved in config.
    beta_override: float | None = None
    consistency_override: float | None = None
    per_device_batch_size: int | None = None
    total_generator_updates: int | None = None
    checkpoint_interval_g_updates: int | None = None
    probe_interval_g_updates: int | None = None
    gradient_accumulation_steps: int = 1
    target_global_batch_size: int | None = None  # assert physical batch * ranks * accumulation
    distributed_strategy: str = "single"  # single | ddp. No untested FSDP switch.
    optimizer_state_sharding: str = "none"  # none | zero1 (DDP, Torch 2.6 rank-local state)
    teacher_weight_dtype: str = "float32"
    ema_device: str = "gpu"  # gpu | cpu; master EMA arithmetic always FP32
    ema_forward_dtype: str = "float32"  # temporary staged copy only
    tensorboard: bool = False
    tensorboard_flush_seconds: int = 30
    sample_interval_g_updates: int | None = None
    fixed_sample_count: int = 8
    performance_interval_g_updates: int = 10
    checkpoint_keep_last: int | None = None  # opt-in pruning of this run's managed checkpoints
    mixed_precision: str = "bf16"  # no | bf16 | fp16
    gradient_checkpointing: bool = False
    conversion_dtype: str = "float64"
    seed: int = 10
    max_consecutive_failed_updates: int = 3
    probe_seed: int = 12345
    debug_anchor_cycle: bool = False  # deterministic four-anchor coverage for short smoke ONLY
    hardware_and_compute_budget: str | None = None

    @classmethod
    def load(cls, path: str | Path, overrides: list[str] | None = None):
        raw = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("Config must be a mapping")
        for item in overrides or []:
            key, separator, value = item.partition("=")
            if not separator:
                raise ValueError(f"Override must be FIELD=VALUE: {item}")
            raw[key] = yaml.safe_load(value)
        unknown = set(raw) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        for key in (
            "generator_anchors",
            "adam_betas",
            "fake_timestep_range_inclusive",
            "dm_timestep_range_inclusive",
        ):
            if key in raw:
                raw[key] = tuple(raw[key])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self):
        if self.backend not in ("sdxl", "toy"):
            raise ValueError("backend must be sdxl or toy")
        if self.prediction_type != "epsilon" or self.generator_anchors != (999, 749, 499, 249):
            raise ValueError("v0 requires epsilon prediction and anchors [999,749,499,249]")
        if self.resolution <= 0 or self.resolution % 8:
            raise ValueError("resolution must be a positive multiple of 8")
        if self.backend == "sdxl" and self.resolution != 1024:
            raise ValueError(
                "v0 SDXL uses native 1024 resolution; change requires an explicit adapter"
            )
        if self.distributed_strategy not in ("single", "ddp"):
            raise ValueError("Only single and ddp are implemented; FSDP is not supported")
        if self.optimizer_state_sharding not in ("none", "zero1"):
            raise ValueError("optimizer_state_sharding must be none or zero1")
        if self.optimizer_state_sharding == "zero1":
            if self.distributed_strategy != "ddp" or self.mixed_precision == "fp16":
                raise ValueError("zero1 requires DDP and bf16/no precision; fp16 is not supported")
        if self.teacher_weight_dtype not in ("float32", "bfloat16"):
            raise ValueError("teacher_weight_dtype must be float32 or bfloat16")
        if self.ema_device not in ("gpu", "cpu"):
            raise ValueError("ema_device must be gpu or cpu")
        if self.ema_forward_dtype not in ("float32", "bfloat16"):
            raise ValueError("ema_forward_dtype must be float32 or bfloat16")
        if self.ema_device == "gpu" and self.ema_forward_dtype != "float32":
            raise ValueError("GPU EMA keeps FP32 parameters; use autocast for its forward")
        if self.sample_interval_g_updates and not self.tensorboard:
            raise ValueError("Fixed sample logging requires tensorboard=true")
        if (
            type(self.gradient_accumulation_steps) is not int
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if self.mixed_precision not in ("no", "bf16", "fp16"):
            raise ValueError("mixed_precision must be no, bf16, or fp16")
        if self.conversion_dtype not in ("float32", "float64"):
            raise ValueError("conversion_dtype must be float32 or float64")
        if self.fake_initialization not in ("paired", "teacher"):
            raise ValueError("fake_initialization must be paired or teacher")
        if self.fake_initialization == "teacher" and self.fake_warmup_updates < 1:
            raise ValueError("Teacher fallback requires explicit fake-only warmup and diagnostics")
        if self.fake_initialization == "paired" and self.fake_warmup_updates < 0:
            raise ValueError("fake_warmup_updates cannot be negative")
        if self.fake_timestep_range_inclusive != (0, 999):
            raise ValueError("v0 fake time range must be inclusive [0,999]")
        if self.dm_timestep_range_inclusive != (20, 980):
            raise ValueError("v0 DM time range must be inclusive [20,980]")
        for key in (
            "fake_updates_per_generator",
            "teacher_anchor_ramp_g_updates",
            "consistency_ramp_g_updates",
            "max_consecutive_failed_updates",
            "tensorboard_flush_seconds",
            "fixed_sample_count",
            "performance_interval_g_updates",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        for key in (
            "per_device_batch_size",
            "total_generator_updates",
            "checkpoint_interval_g_updates",
            "probe_interval_g_updates",
            "sample_interval_g_updates",
            "checkpoint_keep_last",
            "target_global_batch_size",
        ):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{key} must be a positive integer or null")
        for key in (
            "teacher_anchor_beta_max",
            "consistency_weight_max",
            "teacher_anchor_start_g_update",
        ):
            if getattr(self, key) < 0:
                raise ValueError(f"{key} cannot be negative")
        for key in ("beta_override", "consistency_override"):
            if getattr(self, key) is not None and getattr(self, key) < 0:
                raise ValueError(f"{key} cannot be negative")
        if not 0 <= self.generator_ema_decay < 1:
            raise ValueError("EMA decay must be in [0,1)")
        if min(self.generator_lr, self.fake_lr, self.max_grad_norm) <= 0:
            raise ValueError("Learning rates and max_grad_norm must be positive")
        if len(self.adam_betas) != 2 or any(not 0 <= b < 1 for b in self.adam_betas):
            raise ValueError("adam_betas must contain two numbers in [0,1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")

    def missing_runtime_fields(self):
        keys = [
            "per_device_batch_size",
            "total_generator_updates",
            "checkpoint_interval_g_updates",
            "prompts_path",
        ]
        if self.backend == "sdxl":
            keys.append("generator_checkpoint")
            if self.fake_initialization == "paired":
                keys.append("guidance_checkpoint")
        return [key for key in keys if getattr(self, key) is None]

    def require_runtime(self):
        missing = self.missing_runtime_fields()
        if missing:
            raise ValueError(f"Set runtime fields before training: {', '.join(missing)}")
        if self.debug_anchor_cycle and (
            self.total_generator_updates > 32
            or not self.beta_override
            or not self.consistency_override
        ):
            raise ValueError(
                "debug_anchor_cycle is only for <=32 G smoke with nonzero beta/CD overrides"
            )

    def weights(self, generator_updates: int):
        k = generator_updates
        beta = self.teacher_anchor_beta_max * max(
            0.0,
            min(1.0, (k - self.teacher_anchor_start_g_update) / self.teacher_anchor_ramp_g_updates),
        )
        cd = self.consistency_weight_max * max(0.0, min(1.0, k / self.consistency_ramp_g_updates))
        return (
            beta if self.beta_override is None else self.beta_override,
            cd if self.consistency_override is None else self.consistency_override,
        )

    def as_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

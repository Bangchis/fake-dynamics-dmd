"""Backend boundary: identical conditioning is passed to T/F/G/EMA."""

import copy
from dataclasses import dataclass

import torch
from torch import nn

from .diffusion import Diffusion
from .ema import OffloadedEMA
from .weights import import_paired


@dataclass
class Condition:
    tokens: torch.Tensor
    pooled: torch.Tensor
    time_ids: torch.Tensor

    def unconditional(self):
        # DMD2 SDXL convention: zero token/pooled embeddings; preserve size/crop time IDs.
        return Condition(
            torch.zeros_like(self.tokens), torch.zeros_like(self.pooled), self.time_ids
        )


class SDXLEpsilon(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, times, c):
        dtype = next(self.unet.parameters()).dtype
        return self.unet(
            x.to(dtype),
            times,
            encoder_hidden_states=c.tokens.to(dtype),
            added_cond_kwargs={"text_embeds": c.pooled.to(dtype), "time_ids": c.time_ids.to(dtype)},
        ).sample.float()


class SDXLBackend:
    def __init__(self, config, device, *, training=True, import_weights=True):
        # Heavy dependencies are deliberately lazy; plan/toy tests do not import them.
        from diffusers import DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

        self.config, self.device = config, device
        common = {"revision": config.backbone_revision}
        scheduler = DDIMScheduler.from_pretrained(config.backbone, subfolder="scheduler", **common)
        if (
            scheduler.config.prediction_type != "epsilon"
            or scheduler.config.num_train_timesteps != 1000
        ):
            raise ValueError("Checkpoint scheduler is not native 1000-step epsilon SDXL")
        self.scheduler_config = dict(scheduler.config)
        self.diffusion = Diffusion(
            scheduler.alphas_cumprod.to(device), getattr(torch, config.conversion_dtype)
        )
        self.tokenizers = [
            CLIPTokenizer.from_pretrained(config.backbone, subfolder=name, **common)
            for name in ("tokenizer", "tokenizer_2")
        ]
        self.text_encoders = [
            CLIPTextModel.from_pretrained(config.backbone, subfolder="text_encoder", **common),
            CLIPTextModelWithProjection.from_pretrained(
                config.backbone, subfolder="text_encoder_2", **common
            ),
        ]
        for encoder in self.text_encoders:
            encoder.requires_grad_(False).eval().to(device, dtype=torch.float32)
        unet_config = UNet2DConditionModel.load_config(config.backbone, subfolder="unet", **common)
        g = UNet2DConditionModel.from_config(unet_config).float()
        self.provenance = {}
        if training:
            f = UNet2DConditionModel.from_config(unet_config).float()
            if import_weights:
                self.provenance = import_paired(g, f, config)
            self.teacher = SDXLEpsilon(
                UNet2DConditionModel.from_pretrained(
                    config.backbone,
                    subfolder="unet",
                    torch_dtype=getattr(torch, config.teacher_weight_dtype),
                    **common,
                )
            )
            self.teacher.requires_grad_(False).eval().to(device)
            self.fake = SDXLEpsilon(f).to(device)
            if config.gradient_checkpointing:
                g.enable_gradient_checkpointing()
                f.enable_gradient_checkpointing()
        self.generator = SDXLEpsilon(g)
        if training:
            if config.ema_device == "cpu":
                self.ema = OffloadedEMA(
                    self.generator, device, getattr(torch, config.ema_forward_dtype)
                )
            else:
                self.ema = (
                    copy.deepcopy(self.generator).float().requires_grad_(False).eval().to(device)
                )
        self.generator.to(device)
        self.vae = None  # Never needed by prompt-only latent training.

    @torch.no_grad()
    def encode(self, prompts):
        hidden = []
        pooled = None
        # Frozen encoders run in FP32 outside trainer autocast, matching DMD2 encoding.
        with torch.autocast(self.device.type, enabled=False):
            for tokenizer, encoder in zip(self.tokenizers, self.text_encoders):
                ids = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(self.device)
                output = encoder(ids, output_hidden_states=True)
                hidden.append(output.hidden_states[-2])
                pooled = output[0]
        r = self.config.resolution
        time_ids = torch.tensor([r, r, 0, 0, r, r], device=self.device, dtype=torch.float32).repeat(
            len(prompts), 1
        )
        return Condition(torch.cat(hidden, dim=-1).float(), pooled.float(), time_ids)

    @torch.no_grad()
    def decode(self, latents):
        if self.vae is None:
            from diffusers import AutoencoderKL

            self.vae = (
                AutoencoderKL.from_pretrained(
                    self.config.backbone,
                    subfolder="vae",
                    revision=self.config.backbone_revision,
                    torch_dtype=torch.float32,
                )
                .requires_grad_(False)
                .eval()
                .to(self.device)
            )
        with torch.autocast(self.device.type, enabled=False):
            image = self.vae.decode(latents.float() / self.vae.config.scaling_factor).sample
        # DMD2 evaluator quantizes by truncation (uint8), not round-to-nearest.
        return ((image.float() + 1) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu()


class ToyEpsilon(nn.Module):
    """Small conditional network for implementation checks, never an SDXL approximation."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 8, 1), nn.SiLU(), nn.Conv2d(8, 4, 1))
        self.condition_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, times, c):
        offset = c.tokens.mean(dim=(1, 2)).reshape(-1, 1, 1, 1)
        return self.net(x) + self.condition_scale * offset + times[:, None, None, None] / 10000


class ToyBackend:
    def __init__(self, config, device, **kwargs):
        self.config, self.device = config, device
        beta = torch.linspace(0.00085**0.5, 0.012**0.5, 1000).square()
        self.diffusion = Diffusion(
            (1 - beta).cumprod(0).to(device), getattr(torch, config.conversion_dtype)
        )
        self.generator = ToyEpsilon().to(device)
        self.fake = copy.deepcopy(self.generator)
        self.teacher = copy.deepcopy(self.generator).requires_grad_(False).eval()
        self.ema = copy.deepcopy(self.generator).float().requires_grad_(False).eval()
        if config.ema_device == "cpu":
            self.ema = OffloadedEMA(
                self.generator, device, getattr(torch, config.ema_forward_dtype)
            )
        self.provenance = {"kind": "synthetic_toy_no_pretrained_weights"}
        self.scheduler_config = {"kind": "synthetic_scaled_linear_for_tests"}

    def encode(self, prompts):
        values = [sum(p.encode()) % 101 / 101 for p in prompts]
        tokens = torch.tensor(values, device=self.device).reshape(-1, 1, 1)
        return Condition(tokens, tokens[:, 0], torch.zeros(len(prompts), 6, device=self.device))


def make_backend(config, device, **kwargs):
    return (SDXLBackend if config.backend == "sdxl" else ToyBackend)(config, device, **kwargs)

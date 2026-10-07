"""CPU FP32 EMA master with a temporary device snapshot for target forwards."""

import copy

import torch
from torch import nn


class OffloadedEMA(nn.Module):
    def __init__(self, source, device, forward_dtype=torch.bfloat16):
        super().__init__()
        # Copy on CPU before the online generator is placed on CUDA.
        self.master = copy.deepcopy(source).float().cpu().requires_grad_(False).eval()
        self.forward_device = device
        self.forward_dtype = forward_dtype
        self._snapshot = None
        self.eval()

    @torch.no_grad()
    def forward(self, x, times, condition):
        if self._snapshot is None:
            self._snapshot = {
                name: value.to(
                    device=self.forward_device,
                    dtype=self.forward_dtype if value.is_floating_point() else value.dtype,
                )
                for name, value in self.master.state_dict().items()
            }
        # functional_call restores the untouched CPU master after every forward.
        return torch.func.functional_call(
            self.master, self._snapshot, (x, times, condition), strict=True
        )

    def release(self):
        self._snapshot = None


def ema_master(model):
    return model.master if isinstance(model, OffloadedEMA) else model


def release_ema(model):
    if isinstance(model, OffloadedEMA):
        model.release()

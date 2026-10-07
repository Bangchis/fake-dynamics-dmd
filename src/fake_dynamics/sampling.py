"""G follows stochastic DMD2 renoising. Only CD follows raw F DDIM."""

import torch

from .diffusion import batch_time

ANCHORS = (999, 749, 499, 249)
PAIRS = {999: 749, 749: 499, 499: 249}


def generator_x0(model, diffusion, h, tau, condition):
    return diffusion.x0(h, model(h, batch_time(tau, h), condition), tau).float()


@torch.no_grad()
def backsimulate(
    model, diffusion, condition, shape, device, anchor_index=None, noise_generator=None
):
    if anchor_index is None:
        selected = torch.randint(0, 4, (1,), device=device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(selected, src=0)
        anchor_index = int(selected.item())
    if not 0 <= anchor_index < 4:
        raise ValueError("anchor_index must be 0..3")
    h = torch.randn(shape, device=device, generator=noise_generator)
    for i in range(anchor_index):
        y = generator_x0(model, diffusion, h, ANCHORS[i], condition)
        fresh_noise = torch.randn(shape, device=device, generator=noise_generator)
        h = diffusion.renoise(y, ANCHORS[i + 1], fresh_noise)
    return ANCHORS[anchor_index], h


@torch.no_grad()
def sample_generator(model, diffusion, condition, shape, device, noise_generator=None):
    tau, h = backsimulate(model, diffusion, condition, shape, device, 3, noise_generator)
    return generator_x0(model, diffusion, h, tau, condition)


@torch.no_grad()
def fake_transition(fake, diffusion, h, tau, condition):
    if tau not in PAIRS:
        raise ValueError("CD has no transition from 249; do not add a clean boundary")
    end = PAIRS[tau]
    midpoint = (tau + end) // 2
    e = fake(h, batch_time(tau, h), condition)
    hm = diffusion.ddim(h, e, tau, midpoint)
    e_mid = fake(hm, batch_time(midpoint, hm), condition)
    return end, diffusion.ddim(hm, e_mid, midpoint, end)

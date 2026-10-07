"""Handoff sections 4–6. Estimates/targets detach; proxy loss differentiates through y."""

import torch


def guided(conditional, unconditional, scale):
    return unconditional + scale * (conditional - unconditional)


def mixed_target(noise, teacher_cfg, beta):
    if beta == 0:
        return noise.detach().float()
    if teacher_cfg is None:
        raise ValueError("Nonzero beta requires a teacher prediction at the same input")
    return ((noise.float() + beta * teacher_cfg.float()) / (1 + beta)).detach()


def corrected_fake(raw_fake, teacher_cfg, beta):
    return (1 + beta) * raw_fake - beta * teacher_cfg


@torch.no_grad()
def direct_gradient(
    diffusion, y, x_dm, t_dm, fake, tc_dm, tu_dm, x_ca, t_ca, tc_ca, tu_ca, scale, beta
):
    tc0 = diffusion.x0(x_dm, tc_dm, t_dm)
    tcfg0 = diffusion.x0(x_dm, guided(tc_dm, tu_dm, scale), t_dm)
    # Correct in x0 space using the exact same state, timestep and scheduler.
    q0 = corrected_fake(diffusion.x0(x_dm, fake, t_dm), tcfg0, beta)
    norm_dm = (y - tcfg0).abs().flatten(1).mean(1).clamp_min(1e-6)
    norm_dm = norm_dm.reshape(-1, *([1] * (y.ndim - 1)))
    g_dm = (q0 - tc0) / norm_dm
    ca_c0 = diffusion.x0(x_ca, tc_ca, t_ca)
    ca_u0 = diffusion.x0(x_ca, tu_ca, t_ca)
    ca_cfg0 = diffusion.x0(x_ca, guided(tc_ca, tu_ca, scale), t_ca)
    norm_ca = (y - ca_cfg0).abs().flatten(1).mean(1).clamp_min(1e-6)
    norm_ca = norm_ca.reshape(-1, *([1] * (y.ndim - 1)))
    g_ca = (scale - 1) * (ca_u0 - ca_c0) / norm_ca
    # Do not conceal nonfinite gradients with nan_to_num; trainer records/aborts them.
    return (g_dm + g_ca).float(), g_dm.float(), g_ca.float()


def direct_proxy(y, gradient):
    target = (y.detach().float() - gradient.detach().float()).detach()
    return 0.5 * (y.float() - target).square().mean()


def consistency_loss(y, target):
    return (
        y.new_zeros(()) if target is None else (y.float() - target.detach().float()).square().mean()
    )

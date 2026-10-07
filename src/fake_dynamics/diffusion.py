"""Native DDPM coefficients. Network outputs are always epsilon, never velocities."""

import torch


class Diffusion:
    def __init__(self, alphas_cumprod: torch.Tensor, dtype=torch.float64):
        a = alphas_cumprod.detach().to(dtype=dtype)
        if a.ndim != 1 or len(a) != 1000 or not torch.isfinite(a).all():
            raise ValueError("Expected 1000 finite native SDXL scheduler coefficients")
        if not ((a > 0) & (a <= 1)).all() or not (a[1:] <= a[:-1]).all():
            raise ValueError("alphas_cumprod must be positive and nonincreasing")
        self.alphas_cumprod = a
        self.dtype = dtype

    def coefficients(self, t: int | torch.Tensor, x: torch.Tensor):
        times = torch.as_tensor(t, device=x.device, dtype=torch.long).reshape(-1)
        if times.numel() not in (1, x.shape[0]):
            raise ValueError("Timesteps must be scalar or one per sample")
        a = self.alphas_cumprod.to(x.device)[times].reshape(-1, *([1] * (x.ndim - 1)))
        return a.sqrt(), (1 - a).sqrt()

    def x0(self, x, epsilon, t):
        alpha, sigma = self.coefficients(t, x)
        return (x.to(self.dtype) - sigma * epsilon.to(self.dtype)) / alpha

    def renoise(self, y, t, noise):
        alpha, sigma = self.coefficients(t, y)
        return (alpha * y.to(self.dtype) + sigma * noise.to(self.dtype)).float()

    def ddim(self, x, epsilon, start, end):
        """Deterministic eta=0 transition with the SAME predicted epsilon."""
        return self.renoise(self.x0(x, epsilon, start), end, epsilon)


def batch_time(tau: int, x: torch.Tensor):
    return torch.full((x.shape[0],), tau, dtype=torch.long, device=x.device)


def uniform_time(low, high, x):
    return torch.randint(low, high + 1, (x.shape[0],), device=x.device)


def ca_time(tau, x):
    return uniform_time(20, min(980, tau - 1), x)

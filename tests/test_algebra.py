import torch

from fake_dynamics.config import Config
from fake_dynamics.diffusion import Diffusion, ca_time, uniform_time
from fake_dynamics.objectives import (
    corrected_fake,
    direct_gradient,
    direct_proxy,
    guided,
    mixed_target,
)


def diffusion():
    beta = torch.linspace(0.00085**0.5, 0.012**0.5, 1000).square()
    return Diffusion((1 - beta).cumprod(0))


def test_mixed_target_reduces_to_noise_and_detaches():
    noise = torch.randn(3, 4, 2, 2, requires_grad=True)
    teacher = torch.randn_like(noise, requires_grad=True)
    assert torch.equal(mixed_target(noise, None, 0), noise.detach())
    beta = 0.05
    target = mixed_target(noise, teacher, beta)
    assert not target.requires_grad
    torch.testing.assert_close((1 + beta) * target - beta * teacher.detach(), noise.detach())


def test_epsilon_and_x0_corrections_are_equivalent():
    d = diffusion()
    x, f, t = (torch.randn(3, 4, 2, 2, dtype=torch.float64) for _ in range(3))
    times = torch.tensor([20, 499, 980])
    beta = 0.05
    eps_path = d.x0(x, corrected_fake(f, t, beta), times)
    x0_path = corrected_fake(d.x0(x, f, times), d.x0(x, t, times), beta)
    torch.testing.assert_close(eps_path, x0_path, atol=1e-10, rtol=1e-10)


def test_same_state_same_time_beta_zero_matches_dmd2_guided_direction():
    d = diffusion()
    y, x, f, tc, tu = (torch.randn(3, 4, 2, 2) for _ in range(5))
    t = torch.tensor([20, 500, 980])
    scale = 8
    result, _, _ = direct_gradient(d, y, x, t, f, tc, tu, x, t, tc, tu, scale, 0)
    teacher_cfg_x0 = d.x0(x, guided(tc, tu, scale), t)
    normalizer = (y - teacher_cfg_x0).abs().flatten(1).mean(1).reshape(3, 1, 1, 1).clamp_min(1e-6)
    expected = ((d.x0(x, f, t) - teacher_cfg_x0) / normalizer).float()
    torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)


def test_proxy_gradient_sign_and_mean_reduction():
    y = torch.randn(2, 4, 2, 2, requires_grad=True)
    g = torch.randn_like(y, requires_grad=True)
    direct_proxy(y, g).backward()
    torch.testing.assert_close(y.grad, g.detach() / y.numel())
    assert g.grad is None


def test_batched_coefficients_broadcast_correctly():
    d = diffusion()
    y = torch.ones(3, 4, 2, 2)
    t = torch.tensor([0, 499, 999])
    a, s = d.coefficients(t, y)
    assert a.shape == s.shape == (3, 1, 1, 1)
    noisy = d.renoise(y, t, torch.zeros_like(y))
    torch.testing.assert_close(noisy[:, 0, 0, 0], a[:, 0, 0, 0].float())
    torch.testing.assert_close(d.x0(noisy, torch.zeros_like(y), t).float(), y)


def test_integer_time_ranges_and_ca_direction():
    x = torch.empty(10000, 1, 1, 1)
    torch.manual_seed(17)
    for tau in Config().generator_anchors:
        t = ca_time(tau, x)
        assert int(t.min()) >= 20
        assert int(t.max()) <= min(980, tau - 1)
    # High is inclusive: degenerate interval must produce the endpoint.
    assert torch.equal(uniform_time(980, 980, x), torch.full((10000,), 980))


def test_ramps_use_new_successful_generator_counter():
    cfg = Config()
    assert cfg.weights(0) == (0, 0)
    assert cfg.weights(100) == (0, 0.1)
    assert cfg.weights(1100) == (0.05, 0.1)
    assert cfg.weights(19000) == (0.05, 0.1)
    assert cfg.weights(101)[0] == 0.00005

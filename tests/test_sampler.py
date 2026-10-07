import pytest
import torch
from test_algebra import diffusion

from fake_dynamics.backend import Condition
from fake_dynamics.sampling import backsimulate, fake_transition, sample_generator


class Recorder:
    def __init__(self, epsilon=0):
        self.calls = []
        self.epsilon = epsilon

    def __call__(self, x, times, condition):
        self.calls.append(int(times[0]))
        return torch.full_like(x, self.epsilon)


def condition():
    return Condition(torch.zeros(2, 1, 1), torch.zeros(2, 1), torch.zeros(2, 6))


def test_generator_only_visits_four_anchors_and_uses_fresh_noise():
    d, model = diffusion(), Recorder()
    shape = (2, 4, 2, 2)
    seed = 42
    actual = sample_generator(
        model, d, condition(), shape, "cpu", torch.Generator().manual_seed(seed)
    )
    assert model.calls == [999, 749, 499, 249]
    rng = torch.Generator().manual_seed(seed)
    h = torch.randn(shape, generator=rng)
    for start, end in ((999, 749), (749, 499), (499, 249)):
        y = d.x0(h, torch.zeros_like(h), start).float()
        h = d.renoise(y, end, torch.randn(shape, generator=rng))
    expected = d.x0(h, torch.zeros_like(h), 249).float()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("index,tau", [(0, 999), (1, 749), (2, 499), (3, 249)])
def test_backsimulation_returns_original_anchor_input(index, tau):
    model = Recorder()
    result, h = backsimulate(model, diffusion(), condition(), (2, 4, 2, 2), "cpu", index)
    assert result == tau
    assert model.calls == [999, 749, 499][:index]
    assert h.shape == (2, 4, 2, 2)
    assert not h.requires_grad


@pytest.mark.parametrize("tau,end,midpoint", [(999, 749, 874), (749, 499, 624), (499, 249, 374)])
def test_fake_ddim_two_substeps_and_constant_epsilon_solution(tau, end, midpoint):
    d, fake = diffusion(), Recorder(0.1)
    h = torch.randn(2, 4, 2, 2, requires_grad=True)
    result_end, hs = fake_transition(fake, d, h, tau, condition())
    assert result_end == end
    assert fake.calls == [tau, midpoint]
    assert not hs.requires_grad
    expected = d.ddim(h, torch.full_like(h, 0.1), tau, end)
    torch.testing.assert_close(hs, expected, atol=2e-6, rtol=2e-6)


def test_cd_has_no_249_to_clean_transition():
    with pytest.raises(ValueError, match="no transition"):
        fake_transition(Recorder(), diffusion(), torch.randn(2, 4, 2, 2), 249, condition())

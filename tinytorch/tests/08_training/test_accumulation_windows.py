"""Accumulation must match the same samples in a single batch (2026-09-11)."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss
from tinytorch.core.optimizers import SGD
from tinytorch.core.training import Trainer


def make_trainer(lr=0.1, clip=None):
    model = Linear(1, 1)
    model.weight.data[:] = 1.0
    model.bias.data[:] = 0.0
    return Trainer(model, SGD(model.parameters(), lr=lr), MSELoss(), grad_clip_norm=clip)


@pytest.mark.parametrize("sizes,window", [([1], 4), ([2, 1], 2), ([2, 1], 4), ([2, 1, 2], 2)])
@pytest.mark.parametrize("clip", [None, 0.5])
def test_partial_and_unequal_batches_match_full_batch_updates(sizes, window, clip):
    xs = np.arange(1, sum(sizes) + 1, dtype=np.float32).reshape(-1, 1)
    ys = np.zeros_like(xs)
    cuts = np.cumsum([0] + sizes)
    batches = [(Tensor(xs[a:b]), Tensor(ys[a:b])) for a, b in zip(cuts[:-1], cuts[1:])]
    accumulated, reference = make_trainer(clip=clip), make_trainer(clip=clip)
    observed_loss = accumulated.train_epoch(iter(batches), accumulation_steps=window)
    loss_sum = 0.0
    for start in range(0, len(sizes), window):
        a, b = cuts[start], cuts[min(start + window, len(sizes))]
        loss_sum += reference.train_epoch([(Tensor(xs[a:b]), Tensor(ys[a:b]))]) * (b - a)
    for actual, expected in zip(accumulated.model.parameters(), reference.model.parameters()):
        np.testing.assert_allclose(actual.data, expected.data, atol=1e-6)
        assert actual.grad is None or np.all(actual.grad == 0)
    assert observed_loss == pytest.approx(loss_sum / len(xs))
    assert accumulated.step == (len(sizes) + window - 1) // window


@pytest.mark.parametrize("window", [1, 2, 4])
def test_epoch_loss_is_sample_weighted_independently_of_batch_partition(window):
    trainer = make_trainer(lr=0.0)
    loss = trainer.train_epoch([
        (Tensor([[1.0], [1.0]]), Tensor([[0.0], [0.0]])),
        (Tensor([[3.0]]), Tensor([[0.0]])),
    ], accumulation_steps=window)
    assert loss == pytest.approx(11 / 3)


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_invalid_accumulation_window_is_rejected(window):
    trainer = make_trainer()
    with pytest.raises(ValueError, match="positive integer"):
        trainer.train_epoch([], accumulation_steps=window)
    assert trainer.epoch == 0

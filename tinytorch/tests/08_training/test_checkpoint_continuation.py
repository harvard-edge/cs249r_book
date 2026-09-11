"""Checkpoint correctness means the next update matches uninterrupted training."""
import numpy as np
import pytest
from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss
from tinytorch.core.optimizers import SGD, Adam, AdamW
from tinytorch.core.training import Trainer


@pytest.mark.parametrize('optimizer_cls,options', [
    (SGD, {'momentum': 0.8, 'weight_decay': 0.02}),
    (Adam, {'betas': (0.7, 0.95), 'eps': 1e-6, 'weight_decay': 0.02}),
    (AdamW, {'betas': (0.7, 0.95), 'eps': 1e-6, 'weight_decay': 0.02}),
])
def test_checkpoint_resumes_same_next_update(tmp_path, optimizer_cls, options):
    model = Linear(2, 1)
    model.weight.data[:] = [[0.4], [-0.3]]
    model.bias.data[:] = 0.1
    trainer = Trainer(model, optimizer_cls(model.parameters(), lr=0.03, **options), MSELoss())
    batches = [(Tensor([[1., 2.], [-2., 1.]]), Tensor([[0.5], [-0.2]]))]
    for _ in range(3):
        trainer.train_epoch(batches)
    trainer.evaluate(batches)
    path = tmp_path / 'resume.pkl'
    trainer.save_checkpoint(path)

    restored_model = Linear(2, 1)
    # Defaults intentionally differ: saved configuration must win on resume.
    restored = Trainer(restored_model, optimizer_cls(restored_model.parameters()), MSELoss())
    restored.load_checkpoint(path)
    assert restored.model.training is False
    assert restored.optimizer.step_count == trainer.optimizer.step_count
    actual_loss = restored.train_epoch(batches)
    expected_loss = trainer.train_epoch(batches)
    assert actual_loss == pytest.approx(expected_loss)
    for actual, expected in zip(restored.model.parameters(), trainer.model.parameters()):
        np.testing.assert_allclose(actual.data, expected.data, rtol=1e-6, atol=1e-7)


def test_evaluation_loss_does_not_depend_on_batch_partition():
    model = Linear(1, 1)
    model.weight.data[:] = 1
    model.bias.data[:] = 0
    trainer = Trainer(model, SGD(model.parameters()), MSELoss())
    x, y = np.array([[1.], [1.], [3.]]), np.zeros((3, 1))
    whole, _ = trainer.evaluate([(Tensor(x), Tensor(y))])
    split, _ = trainer.evaluate([(Tensor(x[:2]), Tensor(y[:2])), (Tensor(x[2:]), Tensor(y[2:]))])
    assert split == pytest.approx(whole)
    assert split == pytest.approx(11 / 3)


def test_clipping_large_finite_gradients_preserves_direction():
    from tinytorch.core.training import clip_grad_norm
    p = Tensor([0., 0.])
    p.grad = np.array([3e20, 4e20], dtype=np.float32)
    norm = clip_grad_norm([p], max_norm=1.)
    assert norm == pytest.approx(5e20, rel=1e-6)
    np.testing.assert_allclose(p.grad, [0.6, 0.8], atol=1e-6)


@pytest.mark.parametrize('limit', [-1., float('nan'), float('inf')])
def test_clipping_rejects_invalid_norm(limit):
    from tinytorch.core.training import clip_grad_norm
    with pytest.raises(ValueError, match='finite and nonnegative'):
        clip_grad_norm([], max_norm=limit)


def test_checkpoint_rejects_incompatible_model_before_copying_weights():
    model = Linear(2, 1)
    trainer = Trainer(model, SGD(model.parameters()), MSELoss())
    before = [p.data.copy() for p in model.parameters()]
    for state in [{0: np.zeros((2, 1))}, {0: np.zeros((3, 1)), 1: np.zeros((1,))}]:
        with pytest.raises(ValueError, match='Checkpoint parameter'):
            trainer._set_model_state(state)
        for actual, expected in zip(model.parameters(), before):
            np.testing.assert_array_equal(actual.data, expected)

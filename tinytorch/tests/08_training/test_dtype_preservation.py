"""float32 parameters stay float32 through clipping, scheduling, and optimizer steps (2026-09-28).

Under NumPy 2 promotion, an np.float64 clip coefficient or learning rate turned
float32 weights into float64 after one step while Tensor.dtype still said
float32, doubling memory_footprint().
"""
import numpy as np
import pytest

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss
from tinytorch.core.optimizers import SGD, Adam, AdamW
from tinytorch.core.training import Trainer, CosineSchedule, clip_grad_norm


def test_cosine_schedule_returns_python_float():
    lr = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10).get_lr(3)
    assert type(lr) is float


def test_clip_grad_norm_keeps_float32_gradients():
    p = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
    p.grad = np.full(3, 100.0, dtype=np.float32)
    clip_grad_norm([p], max_norm=1.0)
    assert p.grad.dtype == np.float32


@pytest.mark.parametrize("opt_cls", [SGD, Adam, AdamW])
@pytest.mark.parametrize("lr", [0.01, np.float64(0.01)])
def test_optimizer_step_preserves_param_dtype(opt_cls, lr):
    p = Tensor(np.ones(4, dtype=np.float32), requires_grad=True)
    p.grad = np.full(4, 0.5, dtype=np.float64)  # a float64 gradient must not leak in
    kwargs = {"weight_decay": 0.01} if opt_cls is not SGD else {"momentum": 0.9, "weight_decay": 0.01}
    opt = opt_cls([p], lr=lr, **kwargs)
    opt.step()
    assert p.data.dtype == np.float32
    assert p.data.nbytes == p.memory_footprint() == 16


@pytest.mark.parametrize("opt_cls", [SGD, Adam, AdamW])
def test_trainer_with_clipping_and_schedule_keeps_float32(opt_cls):
    model = Linear(4, 3)
    footprint = [p.data.nbytes for p in model.parameters()]
    trainer = Trainer(model, opt_cls(model.parameters(), lr=0.01), MSELoss(),
                      scheduler=CosineSchedule(0.1, 0.01, 5), grad_clip_norm=1e-3)
    x = Tensor(np.ones((2, 4), dtype=np.float32))
    y = Tensor(np.zeros((2, 3), dtype=np.float32))
    for _ in range(2):
        trainer.train_epoch([(x, y)])
    for p, nbytes in zip(model.parameters(), footprint):
        assert p.data.dtype == np.float32 == p.dtype
        assert p.data.nbytes == nbytes

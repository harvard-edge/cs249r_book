"""Reject corrupt optimizer input and preserve complete checkpoint continuation."""
import numpy as np
import pytest

from tinytorch.core.optimizers import SGD, Adam, AdamW
from tinytorch.core.tensor import Tensor
from tinytorch.core.training import Trainer


@pytest.mark.parametrize("optimizer_cls", [SGD, Adam, AdamW])
@pytest.mark.parametrize("name", ["lr", "weight_decay"])
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_invalid_common_hyperparameters_fail_before_mutating_parameter(optimizer_cls, name, value):
    parameter = Tensor([1.0], requires_grad=False)
    with pytest.raises(ValueError, match=name):
        optimizer_cls([parameter], **{name: value})
    assert not parameter.requires_grad


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
@pytest.mark.parametrize("kwargs", [
    {"betas": (1.0, 0.999)}, {"betas": (0.9, -0.1)},
    {"betas": (np.nan, 0.999)}, {"betas": (0.9, np.inf)},
    {"betas": (0.9,)}, {"eps": -1.0}, {"eps": np.nan}, {"eps": np.inf},
])
def test_invalid_adam_hyperparameters_raise(optimizer_cls, kwargs):
    with pytest.raises(ValueError):
        optimizer_cls([Tensor([1.0])], **kwargs)


@pytest.mark.parametrize("momentum", [-0.1, np.nan, np.inf])
def test_invalid_sgd_momentum_raises(momentum):
    with pytest.raises(ValueError, match="momentum"):
        SGD([Tensor([1.0])], momentum=momentum)


@pytest.mark.parametrize("optimizer_cls", [SGD, Adam, AdamW])
def test_wrong_shape_restore_is_atomic(optimizer_cls):
    parameters = [Tensor([1.0, 2.0]), Tensor([3.0, 4.0])]
    kwargs = {"momentum": 0.9} if optimizer_cls is SGD else {}
    optimizer = optimizer_cls(parameters, **kwargs)
    for parameter in parameters:
        parameter.grad = np.ones_like(parameter.data)
    optimizer.step()
    before = optimizer.get_momentum_state()
    # First entry is valid but different; the second could silently broadcast.
    if optimizer_cls is SGD:
        corrupt = [np.zeros(2), np.zeros((2, 2))]
    else:
        corrupt = [(np.zeros(2), np.zeros(2)), (np.zeros((2, 2)), np.zeros((2, 2)))]
    with pytest.raises(ValueError, match="shape"):
        optimizer.set_momentum_state(corrupt)
    for actual, expected in zip(optimizer.get_momentum_state(), before):
        np.testing.assert_array_equal(actual, expected)
    optimizer.step()
    for parameter in parameters:
        assert parameter.data.shape == parameter.shape == (2,)


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_half_initialized_adam_state_is_rejected(optimizer_cls):
    optimizer = optimizer_cls([Tensor([1.0])])
    with pytest.raises(ValueError, match="both"):
        optimizer.set_momentum_state([(None, np.zeros(1))])
    assert optimizer.m_buffers == [None]
    assert optimizer.v_buffers == [None]


@pytest.mark.parametrize("optimizer_cls", [Adam, AdamW])
def test_full_trainer_optimizer_state_resumes_delayed_parameters(optimizer_cls):
    parameters = [Tensor([1.0]), Tensor([2.0])]
    optimizer = optimizer_cls(parameters, lr=0.1, weight_decay=0.02)
    for first, second in [(1.0, None), (2.0, None), (3.0, 0.5)]:
        for parameter, gradient in zip(parameters, [first, second]):
            parameter.grad = None if gradient is None else np.array([gradient], dtype=np.float32)
        optimizer.step()

    # Exercise the full state interface used by checkpointing; the buffer-only
    # get/set_momentum_state methods intentionally do not own moment ages.
    trainer = Trainer.__new__(Trainer)
    trainer.optimizer = optimizer
    saved = trainer._get_optimizer_state()
    resumed_parameters = [Tensor(parameter.data.copy()) for parameter in parameters]
    resumed = optimizer_cls(resumed_parameters, lr=0.001)
    restored_trainer = Trainer.__new__(Trainer)
    restored_trainer.optimizer = resumed
    restored_trainer._set_optimizer_state(saved)
    assert resumed.update_counts == [3, 1]
    for gradients in [(1.0, 2.0), (None, -1.0), (-0.5, 0.25)]:
        for original, restored, gradient in zip(parameters, resumed_parameters, gradients):
            original.grad = None if gradient is None else np.array([gradient], dtype=np.float32)
            restored.grad = None if gradient is None else original.grad.copy()
        optimizer.step()
        resumed.step()
        for original, restored in zip(parameters, resumed_parameters):
            np.testing.assert_allclose(restored.data, original.data, rtol=1e-7, atol=1e-7)
        assert resumed.update_counts == optimizer.update_counts
        assert resumed.step_count == optimizer.step_count

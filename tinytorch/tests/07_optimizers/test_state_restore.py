"""Loading uninitialized momentum must clear later optimization history."""
import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import SGD


def test_restoring_none_momentum_clears_existing_buffer():
    p = Tensor([1.])
    optimizer = SGD([p], lr=0.1, momentum=0.9)
    state = optimizer.get_momentum_state()
    p.grad = np.array([2.], dtype=np.float32)
    optimizer.step()
    optimizer.set_momentum_state(state)
    assert optimizer.momentum_buffers == [None]
    p.grad = np.array([1.], dtype=np.float32)
    before = p.data.copy()
    optimizer.step()
    np.testing.assert_allclose(p.data, before - 0.1)


def test_adamw_decay_does_not_shrink_adaptive_update():
    from tinytorch.core.optimizers import AdamW
    p = Tensor([2.])
    optimizer = AdamW([p], lr=0.1, betas=(0., 0.), eps=1e-8, weight_decay=0.5)
    p.grad = np.array([1.], dtype=np.float32)
    optimizer.step()
    # Decay removes 0.1 from the OLD weight; the adaptive step removes 0.1.
    np.testing.assert_allclose(p.data, [1.8], atol=1e-7)


import pytest
from tinytorch.core.optimizers import Adam, AdamW


@pytest.mark.parametrize('optimizer_cls', [Adam, AdamW])
def test_skipped_parameter_uses_its_own_moment_age(optimizer_cls):
    active, delayed, reference = Tensor([1.]), Tensor([1.]), Tensor([1.])
    optimizer = optimizer_cls([active, delayed], lr=0.1, weight_decay=0)
    reference_optimizer = optimizer_cls([reference], lr=0.1, weight_decay=0)
    for _ in range(4):
        active.grad = np.array([0.5], dtype=np.float32)
        optimizer.step()
    for gradient in [2., -1., 0.5]:
        delayed.grad = np.array([gradient], dtype=np.float32)
        reference.grad = delayed.grad.copy()
        optimizer.step()
        reference_optimizer.step()
        np.testing.assert_allclose(delayed.data, reference.data, atol=1e-7)
    assert optimizer.step_count == 7
    assert optimizer.update_counts == [7, 3]


def test_duplicate_parameters_cannot_be_updated_twice():
    p = Tensor([1.])
    with pytest.raises(ValueError, match='duplicates'):
        SGD([p, p])


def test_optimizer_owns_parameter_list():
    p = Tensor([1.])
    supplied = [p]
    optimizer = Adam(supplied)
    supplied.clear()
    assert optimizer.params == [p]

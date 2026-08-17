"""
Tests for SGD's momentum-state checkpointing API.

SGD.has_momentum(), get_momentum_state(), and set_momentum_state() exist
specifically to let training checkpoints (Module 08) save and restore
optimizer momentum without hasattr() checks, the same category of bug as
the corrupted-checkpoint crash fixed for Trainer.load_checkpoint. Despite
that, no test (pytest or inline) exercised any of the three methods before
this file: not the has_momentum()/momentum=0 branches, not a save/restore
round-trip, and not the length-mismatch error path.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.optimizers import SGD
from tinytorch.core.autograd import enable_autograd

enable_autograd()


def make_sgd(momentum=0.9):
    params = [Tensor(np.array([1.0, 2.0]), requires_grad=True)]
    return SGD(params, lr=0.1, momentum=momentum), params


class TestHasMomentum:
    def test_true_when_momentum_positive(self):
        optimizer, _ = make_sgd(momentum=0.9)
        assert optimizer.has_momentum() is True

    def test_false_when_momentum_zero(self):
        optimizer, _ = make_sgd(momentum=0.0)
        assert optimizer.has_momentum() is False


class TestGetMomentumState:
    def test_none_when_momentum_disabled(self):
        optimizer, _ = make_sgd(momentum=0.0)
        assert optimizer.get_momentum_state() is None

    def test_returns_buffers_after_step(self):
        optimizer, params = make_sgd(momentum=0.9)
        params[0].grad = np.array([1.0, 1.0])
        optimizer.step()

        state = optimizer.get_momentum_state()
        assert state is not None
        assert len(state) == 1
        np.testing.assert_allclose(state[0], optimizer.momentum_buffers[0])

    def test_returns_a_copy_not_a_reference(self):
        """A caller must be able to keep get_momentum_state()'s result around
        across further step() calls without it silently changing underneath
        them, since the whole point is using it as a checkpoint snapshot."""
        optimizer, params = make_sgd(momentum=0.9)
        params[0].grad = np.array([1.0, 1.0])
        optimizer.step()

        snapshot = optimizer.get_momentum_state()
        snapshot_before = snapshot[0].copy()

        params[0].grad = np.array([5.0, 5.0])
        optimizer.step()

        np.testing.assert_allclose(snapshot[0], snapshot_before)


class TestSetMomentumState:
    def test_none_state_is_a_noop(self):
        optimizer, params = make_sgd(momentum=0.9)
        params[0].grad = np.array([1.0, 1.0])
        optimizer.step()
        before = optimizer.momentum_buffers[0].copy()

        optimizer.set_momentum_state(None)

        np.testing.assert_allclose(optimizer.momentum_buffers[0], before)

    def test_noop_when_momentum_disabled(self):
        optimizer, _ = make_sgd(momentum=0.0)
        # Even a well-formed state must be ignored when momentum is off,
        # since momentum_buffers stay [None, ...] for this optimizer.
        optimizer.set_momentum_state([np.array([9.0, 9.0])])
        assert len(optimizer.momentum_buffers) == 1
        assert optimizer.momentum_buffers[0] is None

    def test_round_trip_restores_exact_values(self):
        optimizer, params = make_sgd(momentum=0.9)
        params[0].grad = np.array([1.0, 2.0])
        optimizer.step()
        params[0].grad = np.array([3.0, -1.0])
        optimizer.step()

        saved_state = optimizer.get_momentum_state()

        fresh_params = [Tensor(np.array([1.0, 2.0]), requires_grad=True)]
        fresh_optimizer = SGD(fresh_params, lr=0.1, momentum=0.9)
        fresh_optimizer.set_momentum_state(saved_state)

        np.testing.assert_allclose(fresh_optimizer.momentum_buffers[0], optimizer.momentum_buffers[0])

    def test_length_mismatch_raises_value_error(self):
        optimizer, params = make_sgd(momentum=0.9)
        params[0].grad = np.array([1.0, 1.0])
        optimizer.step()
        state = optimizer.get_momentum_state()

        mismatched_state = state + [np.array([0.0, 0.0])]

        with pytest.raises(ValueError, match="length mismatch"):
            optimizer.set_momentum_state(mismatched_state)

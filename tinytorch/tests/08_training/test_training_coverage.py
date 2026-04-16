"""
Module 08: Training - Coverage Tests
======================================

Tests for the parts of Module 08 that are implemented but have no test coverage:
- CosineSchedule correctness
- clip_grad_norm behaviour
- Trainer.save_checkpoint / load_checkpoint round-trip
- Trainer.evaluate (loss and accuracy)
- Scheduler integration inside Trainer.train_epoch
- Gradient clipping integration inside Trainer.train_epoch
- Trainer train → eval mode switching
"""

import numpy as np
import os
import pickle
import tempfile
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.autograd import enable_autograd

enable_autograd()

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss, CrossEntropyLoss
from tinytorch.core.optimizers import SGD
from tinytorch.core.training import Trainer, CosineSchedule, clip_grad_norm


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def simple_model():
    """Linear(2→1) model with known initial weights for deterministic tests."""
    layer = Linear(2, 1)
    layer.weight.data = np.array([[0.5], [0.5]])
    layer.bias.data = np.array([0.0])
    return layer


def simple_trainer(lr=0.01, scheduler=None, grad_clip=None):
    model = simple_model()
    opt = SGD(model.parameters(), lr=lr)
    return Trainer(model, opt, MSELoss(), scheduler=scheduler, grad_clip_norm=grad_clip), model


# ─────────────────────────────────────────────
# CosineSchedule
# ─────────────────────────────────────────────

class TestCosineSchedule:
    """CosineSchedule returns correct learning rates."""

    def test_start_equals_max_lr(self):
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        assert abs(s.get_lr(0) - 0.1) < 1e-9

    def test_end_equals_min_lr(self):
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        assert abs(s.get_lr(100) - 0.01) < 1e-9

    def test_midpoint_is_between_min_and_max(self):
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        mid = s.get_lr(50)
        assert 0.01 < mid < 0.1

    def test_midpoint_formula(self):
        """get_lr(50) == (max_lr + min_lr) / 2 for a 100-epoch schedule."""
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        expected = (0.1 + 0.01) / 2
        assert abs(s.get_lr(50) - expected) < 1e-6

    def test_monotonically_decreasing(self):
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        lrs = [s.get_lr(e) for e in range(101)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1], (
                f"LR should be non-increasing: lr[{i}]={lrs[i]:.6f} > lr[{i+1}]={lrs[i+1]:.6f}"
            )

    def test_past_total_epochs_returns_min_lr(self):
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=50)
        assert abs(s.get_lr(999) - 0.01) < 1e-9

    def test_single_epoch_schedule(self):
        """Edge case: total_epochs=1."""
        s = CosineSchedule(max_lr=0.5, min_lr=0.05, total_epochs=1)
        assert abs(s.get_lr(0) - 0.5) < 1e-9
        assert abs(s.get_lr(1) - 0.05) < 1e-9


# ─────────────────────────────────────────────
# clip_grad_norm
# ─────────────────────────────────────────────

class TestClipGradNorm:
    """clip_grad_norm clips gradient magnitudes and returns the original norm."""

    def _params_with_grads(self, grad_values):
        """Create Tensor params with preset gradients."""
        params = []
        for v in grad_values:
            p = Tensor(np.zeros_like(v), requires_grad=True)
            p.grad = np.array(v, dtype=np.float64)
            params.append(p)
        return params

    def test_returns_original_norm(self):
        params = self._params_with_grads([[3.0, 4.0]])  # norm = 5
        original_norm = clip_grad_norm(params, max_norm=10.0)
        assert abs(original_norm - 5.0) < 1e-6

    def test_clips_large_gradients(self):
        params = self._params_with_grads([[3.0, 4.0]])  # norm = 5
        clip_grad_norm(params, max_norm=1.0)
        clipped_norm = np.linalg.norm(params[0].grad)
        assert abs(clipped_norm - 1.0) < 1e-6

    def test_does_not_clip_small_gradients(self):
        params = self._params_with_grads([[0.1, 0.1]])  # norm ≈ 0.14
        original_grad = params[0].grad.copy()
        clip_grad_norm(params, max_norm=1.0)
        np.testing.assert_allclose(params[0].grad, original_grad)

    def test_clips_across_multiple_params(self):
        """Global norm is computed over all params together."""
        params = self._params_with_grads([[3.0, 4.0], [0.0, 0.0]])
        # global norm = 5; max_norm = 1 → scale = 0.2
        clip_grad_norm(params, max_norm=1.0)
        expected = np.array([3.0, 4.0]) * (1.0 / 5.0)
        np.testing.assert_allclose(params[0].grad, expected, rtol=1e-5)

    def test_direction_preserved_after_clipping(self):
        """Clipping scales magnitude but preserves gradient direction."""
        params = self._params_with_grads([[3.0, 4.0]])
        original_dir = params[0].grad / np.linalg.norm(params[0].grad)
        clip_grad_norm(params, max_norm=1.0)
        clipped_dir = params[0].grad / np.linalg.norm(params[0].grad)
        np.testing.assert_allclose(clipped_dir, original_dir, atol=1e-6)

    def test_zero_gradients_no_division_by_zero(self):
        """All-zero gradients should not cause division by zero."""
        params = self._params_with_grads([[0.0, 0.0, 0.0]])
        norm = clip_grad_norm(params, max_norm=1.0)
        assert np.isfinite(norm)
        np.testing.assert_allclose(params[0].grad, np.zeros(3))


# ─────────────────────────────────────────────
# Checkpoint round-trip
# ─────────────────────────────────────────────

class TestCheckpointing:
    """save_checkpoint / load_checkpoint preserve all training state."""

    def test_checkpoint_file_is_created(self):
        trainer, _ = simple_trainer()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            assert os.path.exists(path)
        finally:
            os.remove(path)

    def test_checkpoint_contains_required_keys(self):
        trainer, _ = simple_trainer()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            with open(path, "rb") as f:
                ckpt = pickle.load(f)
            for key in ("epoch", "step", "model_state", "optimizer_state", "history"):
                assert key in ckpt, f"Missing key: {key}"
        finally:
            os.remove(path)

    def test_epoch_and_step_restored(self):
        trainer, _ = simple_trainer()
        trainer.epoch = 42
        trainer.step = 1337

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            trainer.epoch = 0
            trainer.step = 0
            trainer.load_checkpoint(path)
            assert trainer.epoch == 42
            assert trainer.step == 1337
        finally:
            os.remove(path)

    def test_history_restored(self):
        trainer, _ = simple_trainer()
        trainer.history["train_loss"] = [0.9, 0.7, 0.5]
        trainer.history["eval_loss"] = [0.8, 0.6]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            trainer.history = {"train_loss": [], "eval_loss": [], "learning_rates": []}
            trainer.load_checkpoint(path)
            assert trainer.history["train_loss"] == [0.9, 0.7, 0.5]
            assert trainer.history["eval_loss"] == [0.8, 0.6]
        finally:
            os.remove(path)

    def test_model_weights_restored(self):
        trainer, model = simple_trainer()
        original_weights = model.parameters()[0].data.copy()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            # Corrupt the weights
            model.parameters()[0].data[:] = 999.0
            trainer.load_checkpoint(path)
            np.testing.assert_allclose(model.parameters()[0].data, original_weights)
        finally:
            os.remove(path)

    def test_training_continues_after_load(self):
        """Model can keep training after loading a checkpoint."""
        trainer, model = simple_trainer(lr=0.01)
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]

        trainer.train_epoch(data)
        weights_after_first_epoch = model.parameters()[0].data.copy()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            trainer.load_checkpoint(path)
            trainer.train_epoch(data)
            weights_after_second_epoch = model.parameters()[0].data.copy()
            # Weights should change after resuming
            assert not np.allclose(weights_after_first_epoch, weights_after_second_epoch)
        finally:
            os.remove(path)

    def test_checkpoint_creates_parent_directory(self):
        """save_checkpoint creates intermediate directories if needed."""
        trainer, _ = simple_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "deep", "ckpt.pkl")
            trainer.save_checkpoint(path)
            assert os.path.exists(path)


# ─────────────────────────────────────────────
# Trainer.evaluate
# ─────────────────────────────────────────────

class TestTrainerEvaluate:
    """Trainer.evaluate computes correct metrics without modifying the model."""

    def test_returns_finite_loss(self):
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        loss, _ = trainer.evaluate(data)
        assert np.isfinite(loss), f"Expected finite loss, got {loss}"

    def test_returns_float(self):
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        loss, acc = trainer.evaluate(data)
        assert isinstance(loss, (float, np.floating))
        assert isinstance(acc, (float, np.floating))

    def test_model_set_to_eval_mode(self):
        trainer, model = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        assert model.training is False
        assert trainer.training_mode is False

    def test_weights_unchanged_after_evaluate(self):
        trainer, model = simple_trainer()
        weights_before = model.parameters()[0].data.copy()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))] * 5
        trainer.evaluate(data)
        np.testing.assert_array_equal(model.parameters()[0].data, weights_before)

    def test_eval_loss_recorded_in_history(self):
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        assert len(trainer.history["eval_loss"]) == 1

    def test_eval_loss_recorded_each_call(self):
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        trainer.evaluate(data)
        assert len(trainer.history["eval_loss"]) == 2

    def test_classification_accuracy(self):
        """Accuracy is 1.0 when argmax predictions match integer targets."""
        class PerfectClassifier:
            training = True

            def forward(self, x):
                # Always predicts class 0 with very high confidence
                batch = x.data.shape[0]
                logits = np.zeros((batch, 3))
                logits[:, 0] = 10.0
                return Tensor(logits)

            def parameters(self):
                return []

        loss_fn = CrossEntropyLoss()
        opt = SGD([], lr=0.01)
        trainer = Trainer(PerfectClassifier(), opt, loss_fn)

        data = [(Tensor([[1.0, 0.0]]), Tensor(np.array([0])))]
        _, accuracy = trainer.evaluate(data)
        assert accuracy == 1.0, f"Perfect classifier should have accuracy=1.0, got {accuracy}"

    def test_zero_accuracy_for_wrong_predictions(self):
        """Accuracy is 0.0 when predictions are always wrong."""
        class WrongClassifier:
            training = True

            def forward(self, x):
                batch = x.data.shape[0]
                logits = np.zeros((batch, 3))
                logits[:, 1] = 10.0   # always predicts class 1
                return Tensor(logits)

            def parameters(self):
                return []

        loss_fn = CrossEntropyLoss()
        opt = SGD([], lr=0.01)
        trainer = Trainer(WrongClassifier(), opt, loss_fn)

        data = [(Tensor([[1.0, 0.0]]), Tensor(np.array([0])))]  # target is class 0
        _, accuracy = trainer.evaluate(data)
        assert accuracy == 0.0, f"Wrong classifier should have accuracy=0.0, got {accuracy}"


# ─────────────────────────────────────────────
# Scheduler integration
# ─────────────────────────────────────────────

class TestSchedulerIntegration:
    """CosineSchedule is applied correctly during train_epoch."""

    def test_lr_recorded_in_history_when_scheduler_present(self):
        scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)
        trainer, _ = simple_trainer(scheduler=scheduler)
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.train_epoch(data)
        assert len(trainer.history["learning_rates"]) == 1

    def test_no_lr_in_history_without_scheduler(self):
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.train_epoch(data)
        assert len(trainer.history["learning_rates"]) == 0

    def test_optimizer_lr_updated_by_scheduler(self):
        """After train_epoch, optimizer lr should match scheduler output."""
        scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)
        trainer, _ = simple_trainer(lr=0.1, scheduler=scheduler)
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.train_epoch(data)
        expected_lr = scheduler.get_lr(1)   # epoch 1 after first epoch
        assert abs(trainer.optimizer.lr - expected_lr) < 1e-9

    def test_lr_decreases_over_epochs(self):
        scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=20)
        trainer, _ = simple_trainer(lr=0.1, scheduler=scheduler)
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]

        for _ in range(5):
            trainer.train_epoch(data)

        lrs = trainer.history["learning_rates"]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1], (
                f"LR should decrease: lrs[{i}]={lrs[i]:.6f} > lrs[{i+1}]={lrs[i+1]:.6f}"
            )


# ─────────────────────────────────────────────
# Gradient clipping integration
# ─────────────────────────────────────────────

class TestGradientClippingIntegration:
    """Gradient clipping actually limits gradient norms during training."""

    def test_training_completes_with_grad_clip(self):
        """Training should not crash when grad_clip_norm is set."""
        trainer, _ = simple_trainer(lr=0.01, grad_clip=1.0)
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        loss = trainer.train_epoch(data)
        assert np.isfinite(loss), f"Loss should be finite with grad clipping, got {loss}"

    def test_weights_update_with_grad_clip(self):
        """Weights still change when clipping is active."""
        trainer, model = simple_trainer(lr=0.1, grad_clip=0.01)
        weights_before = model.parameters()[0].data.copy()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.train_epoch(data)
        assert not np.allclose(model.parameters()[0].data, weights_before)

    def test_very_tight_clip_limits_updates(self):
        """Extremely small max_norm keeps weight updates very small."""
        trainer_clipped, model_clipped = simple_trainer(lr=0.1, grad_clip=1e-6)
        trainer_free, model_free = simple_trainer(lr=0.1)

        # Same initial weights
        w0 = np.array([[0.5], [0.5]])
        model_clipped.parameters()[0].data[:] = w0
        model_clipped.parameters()[1].data[:] = 0.0
        model_free.parameters()[0].data[:] = w0
        model_free.parameters()[1].data[:] = 0.0

        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer_clipped.train_epoch(data)
        trainer_free.train_epoch(data)

        update_clipped = np.abs(model_clipped.parameters()[0].data - w0).max()
        update_free = np.abs(model_free.parameters()[0].data - w0).max()
        assert update_clipped < update_free, (
            "Tightly clipped update should be smaller than unclipped update"
        )


# ─────────────────────────────────────────────
# Train / eval mode switching
# ─────────────────────────────────────────────

class TestTrainEvalMode:
    """Trainer correctly switches model between train and eval mode."""

    def test_model_in_train_mode_during_train_epoch(self):
        """model.training should be True at the end of train_epoch."""
        trainer, model = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.train_epoch(data)
        assert model.training is True
        assert trainer.training_mode is True

    def test_model_in_eval_mode_during_evaluate(self):
        trainer, model = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        assert model.training is False
        assert trainer.training_mode is False

    def test_train_after_eval_restores_train_mode(self):
        """Calling train_epoch after evaluate re-enables training mode."""
        trainer, model = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        assert model.training is False
        trainer.train_epoch(data)
        assert model.training is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

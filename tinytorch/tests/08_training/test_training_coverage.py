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

import tinytorch.core.autograd  # completes every operation with its backward half
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
        """
        WHAT: Cosine schedule begins at max_lr.

        WHY: The first epoch should train at the full learning rate you asked for. A
        schedule that starts below it silently wastes the early epochs, where the loss
        falls fastest.

        STUDENT LEARNING: get_lr(0) is the rate for the first epoch, not the rate before
        training.
        """
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        assert abs(s.get_lr(0) - 0.1) < 1e-9

    def test_end_equals_min_lr(self):
        """
        WHAT: Cosine schedule ends at min_lr.

        WHY: Late training needs small steps to settle into a minimum instead of
        bouncing around it. A schedule that never reaches min_lr leaves accuracy on the
        table.

        STUDENT LEARNING: The floor matters as much as the ceiling.
        """
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        assert abs(s.get_lr(100) - 0.01) < 1e-9

    def test_midpoint_is_between_min_and_max(self):
        """
        WHAT: The halfway rate lies strictly between the two bounds.

        WHY: It rules out the two ways a schedule degenerates: a constant rate, and a
        step that jumps straight from max to min.

        STUDENT LEARNING: A schedule is a curve, not a switch.
        """
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        mid = s.get_lr(50)
        assert 0.01 < mid < 0.1

    def test_midpoint_formula(self):
        """
        WHAT: The halfway rate is the exact cosine value, not an approximation.

        WHY: Cosine annealing spends more time near max_lr and near min_lr than a
        straight line would. Getting the midpoint right is what makes the curve cosine
        rather than linear.

        STUDENT LEARNING: At the halfway point cosine gives the average of the two
        bounds.
        """
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        expected = (0.1 + 0.01) / 2
        assert abs(s.get_lr(50) - expected) < 1e-6

    def test_monotonically_decreasing(self):
        """
        WHAT: The rate never rises from one epoch to the next.

        WHY: An off-by-one or a sign error can make the curve rise partway through,
        which undoes the annealing and destabilizes late training.

        STUDENT LEARNING: Monotonicity is cheaper to test than the whole curve and
        catches most errors in it.
        """
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=100)
        lrs = [s.get_lr(e) for e in range(101)]
        for i in range(len(lrs) - 1):
            assert lrs[i] >= lrs[i + 1], (
                f"LR should be non-increasing: lr[{i}]={lrs[i]:.6f} > lr[{i+1}]={lrs[i+1]:.6f}"
            )

    def test_past_total_epochs_returns_min_lr(self):
        """
        WHAT: Asking past the end returns min_lr rather than overshooting.

        WHY: Training loops often run one epoch longer than planned. Without a clamp the
        cosine keeps going and the rate climbs back up.

        STUDENT LEARNING: Schedules need defined behavior outside their range.
        """
        s = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=50)
        assert abs(s.get_lr(999) - 0.01) < 1e-9

    def test_single_epoch_schedule(self):
        """
        WHAT: A one-epoch schedule does not divide by zero.

        WHY: total_epochs - 1 is zero here, which is the natural denominator in the
        cosine formula. The degenerate case is the one nobody runs until a student does.

        STUDENT LEARNING: Check the boundary where your denominator vanishes.
        """
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
        """
        WHAT: Clipping reports the norm it saw before clipping.

        WHY: That returned value is the diagnostic. Logging it is how you find out
        whether gradients are exploding, and a function that returned the post-clip norm
        would always report the ceiling.

        STUDENT LEARNING: The measurement must describe the input, not the output.
        """
        params = self._params_with_grads([[3.0, 4.0]])  # norm = 5
        original_norm = clip_grad_norm(params, max_norm=10.0)
        assert abs(original_norm - 5.0) < 1e-6

    def test_clips_large_gradients(self):
        """
        WHAT: A gradient above the threshold is scaled down to exactly the threshold.

        WHY: Exploding gradients take one enormous step and destroy the weights a model
        spent epochs learning. Clipping bounds the damage a single batch can do.

        STUDENT LEARNING: Clipping caps the step size; it does not stop the step.
        """
        params = self._params_with_grads([[3.0, 4.0]])  # norm = 5
        clip_grad_norm(params, max_norm=1.0)
        clipped_norm = np.linalg.norm(params[0].grad)
        assert abs(clipped_norm - 1.0) < 1e-6

    def test_does_not_clip_small_gradients(self):
        """
        WHAT: A gradient below the threshold passes through untouched.

        WHY: Clipping that always fires is a learning-rate change in disguise, and it
        slows normal training. The threshold has to be a ceiling, not a target.

        STUDENT LEARNING: An always-on safety mechanism is a bug.
        """
        params = self._params_with_grads([[0.1, 0.1]])  # norm ≈ 0.14
        original_grad = params[0].grad.copy()
        clip_grad_norm(params, max_norm=1.0)
        np.testing.assert_allclose(params[0].grad, original_grad)

    def test_clips_across_multiple_params(self):
        """
        WHAT: The norm is global across all parameters, not per tensor.

        WHY: Clipping each tensor separately changes the direction of the overall
        update, because it shrinks some layers more than others. Global clipping is what
        preserves the descent direction.

        STUDENT LEARNING: One norm over the whole model, one scale factor applied to
        every gradient.
        """
        params = self._params_with_grads([[3.0, 4.0], [0.0, 0.0]])
        # global norm = 5; max_norm = 1 → scale = 0.2
        clip_grad_norm(params, max_norm=1.0)
        expected = np.array([3.0, 4.0]) * (1.0 / 5.0)
        np.testing.assert_allclose(params[0].grad, expected, rtol=1e-5)

    def test_direction_preserved_after_clipping(self):
        """
        WHAT: Clipping changes magnitude only, never direction.

        WHY: The gradient direction is the information backpropagation computed.
        Clipping is meant to shorten the step, and any change of direction would be
        discarding that information.

        STUDENT LEARNING: Scaling a vector by a positive number leaves its direction
        alone.
        """
        params = self._params_with_grads([[3.0, 4.0]])
        original_dir = params[0].grad / np.linalg.norm(params[0].grad)
        clip_grad_norm(params, max_norm=1.0)
        clipped_dir = params[0].grad / np.linalg.norm(params[0].grad)
        np.testing.assert_allclose(clipped_dir, original_dir, atol=1e-6)

    def test_zero_gradients_no_division_by_zero(self):
        """
        WHAT: An all-zero gradient does not divide by zero.

        WHY: The scale factor is max_norm divided by the observed norm, and a zero norm
        is reachable: a frozen layer, a dead ReLU, or the very first step of a masked
        batch.

        STUDENT LEARNING: Guard every denominator you did not prove non-zero.
        """
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
        """
        WHAT: Saving writes a file where you asked.

        WHY: Everything else about checkpointing is worthless if the file is not there
        after a crash.

        STUDENT LEARNING: Start with the claim you would check first at three in the
        morning.
        """
        trainer, _ = simple_trainer()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            trainer.save_checkpoint(path)
            assert os.path.exists(path)
        finally:
            os.remove(path)

    def test_checkpoint_contains_required_keys(self):
        """
        WHAT: The checkpoint carries every field a resume needs.

        WHY: A checkpoint missing the optimizer state or the epoch number loads without
        error and then resumes wrongly, which is harder to notice than a crash.

        STUDENT LEARNING: A partial checkpoint is more dangerous than none.
        """
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
        """
        WHAT: Epoch and step counters come back at the values they had.

        WHY: The scheduler reads the epoch. Resuming at zero restarts the learning-rate
        curve and undoes the annealing already paid for.

        STUDENT LEARNING: Position in training is training state, not bookkeeping.
        """
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
        """
        WHAT: The recorded history survives the round trip.

        WHY: History is how anyone judges whether the run was working. Losing it on
        resume means losing the evidence for every epoch before the crash.

        STUDENT LEARNING: Save what you will need to explain the run afterwards.
        """
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
        """
        WHAT: Weights come back bit for bit.

        WHY: This is the whole point. Weights that come back close but not equal mean
        the resumed run is a different run.

        STUDENT LEARNING: Compare weights exactly, not approximately.
        """
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
        """
        WHAT: A restored trainer can keep training.

        WHY: Restoring the numbers is not enough if the object cannot take another step,
        which is how a broken optimizer state shows up.

        STUDENT LEARNING: Test the resume, not just the load.
        """
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
        """
        WHAT: Saving creates missing parent directories.

        WHY: The first checkpoint of a run usually targets a directory that does not
        exist yet. Failing there loses the run it was meant to protect.

        STUDENT LEARNING: The first save is the one most likely to fail.
        """
        trainer, _ = simple_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "deep", "ckpt.pkl")
            trainer.save_checkpoint(path)
            assert os.path.exists(path)


class TestSaveCheckpointIsAtomic:
    """save_checkpoint must never leave a corrupted or partially-written
    file at the target path, the classic "process killed mid-write" or
    "disk full during save" failure mode. It writes to a temp file and
    atomically replaces the target, so the target is always either the
    old complete checkpoint or the new one, never a partial write."""

    def test_failure_during_write_leaves_previous_checkpoint_intact(self, monkeypatch):
        """
        WHAT: A failed save does not destroy the last good checkpoint.

        WHY: Writing in place means a crash mid-write leaves a truncated file where the
        last known-good state used to be. That converts a recoverable crash into a lost
        run.

        STUDENT LEARNING: Write to a temporary file, then rename. Rename is atomic;
        write is not.
        """
        import tinytorch.core.training as training_module

        trainer, _ = simple_trainer()
        trainer.epoch = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pkl")
            trainer.save_checkpoint(path)
            with open(path, "rb") as f:
                original_bytes = f.read()

            trainer.epoch = 999  # state that must NOT end up on disk

            def failing_dump(obj, f):
                f.write(b"partial garbage")
                raise OSError("simulated disk full")

            monkeypatch.setattr(training_module.pickle, "dump", failing_dump)

            with pytest.raises(OSError):
                trainer.save_checkpoint(path)

            with open(path, "rb") as f:
                after_bytes = f.read()
            assert after_bytes == original_bytes, (
                "A failed save must not corrupt the previous checkpoint"
            )

    def test_failure_during_write_does_not_leave_temp_file_behind(self, monkeypatch):
        """
        WHAT: A failed save cleans up its temporary file.

        WHY: Long runs checkpoint often. Leaked temporary files fill the disk, and a
        full disk is what causes the next failure.

        STUDENT LEARNING: Clean up on the error path, not only the success path.
        """
        import tinytorch.core.training as training_module

        trainer, _ = simple_trainer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pkl")

            def failing_dump(obj, f):
                raise OSError("simulated disk full")

            monkeypatch.setattr(training_module.pickle, "dump", failing_dump)

            with pytest.raises(OSError):
                trainer.save_checkpoint(path)

            assert not os.path.exists(path)
            # The contract is "a failed save leaves nothing behind", not a
            # particular temp filename, so assert on the directory rather
            # than on the reference implementation's ".tmp" suffix.
            assert os.listdir(tmpdir) == []

    def test_successful_save_still_produces_a_loadable_checkpoint(self):
        """
        WHAT: The atomic path still produces a loadable file.

        WHY: Safety machinery is worth nothing if it breaks the normal case. This is the
        test that stops the fix from becoming the bug.

        STUDENT LEARNING: When you harden a path, re-test the happy one.
        """
        trainer, model = simple_trainer()
        trainer.epoch = 5
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pkl")
            trainer.save_checkpoint(path)

            assert os.path.exists(path)
            # Same here: only the checkpoint should survive a successful save,
            # whatever the implementation named its temp file.
            assert os.listdir(tmpdir) == [os.path.basename(path)]

            trainer.epoch = 0
            trainer.load_checkpoint(path)
            assert trainer.epoch == 5


# ─────────────────────────────────────────────
# Trainer.evaluate
# ─────────────────────────────────────────────

class TestTrainerEvaluate:
    """Trainer.evaluate computes correct metrics without modifying the model."""

    def test_returns_finite_loss(self):
        """
        WHAT: Evaluation returns a finite number.

        WHY: NaN loss is the classic silent failure. It propagates through every later
        average and comparison without raising.

        STUDENT LEARNING: Assert finiteness explicitly; NaN compares false against
        everything and hides.
        """
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        loss, _ = trainer.evaluate(data)
        assert np.isfinite(loss), f"Expected finite loss, got {loss}"

    def test_returns_float(self):
        """
        WHAT: Evaluation returns a plain float, not a tensor.

        WHY: Callers log it, compare it, and store it in JSON. A tensor works until one
        of those does something surprising.

        STUDENT LEARNING: Decide what a function returns and hold it to that.
        """
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        loss, acc = trainer.evaluate(data)
        assert isinstance(loss, (float, np.floating))
        assert isinstance(acc, (float, np.floating))

    def test_model_set_to_eval_mode(self):
        """
        WHAT: Evaluation switches the model to eval mode.

        WHY: Dropout and batch norm behave differently in the two modes. Evaluating in
        train mode reports a number that is not the model's real accuracy.

        STUDENT LEARNING: Mode is part of the contract of evaluation.
        """
        trainer, model = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        assert model.training is False
        assert trainer.training_mode is False

    def test_weights_unchanged_after_evaluate(self):
        """
        WHAT: Evaluation does not change the weights.

        WHY: A stray optimizer step during evaluation means the model trains on the test
        set, which inflates every number after it and is nearly invisible.

        STUDENT LEARNING: Measurement must not modify what it measures.
        """
        trainer, model = simple_trainer()
        weights_before = model.parameters()[0].data.copy()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))] * 5
        trainer.evaluate(data)
        np.testing.assert_array_equal(model.parameters()[0].data, weights_before)

    def test_eval_loss_recorded_in_history(self):
        """
        WHAT: Evaluation loss is appended to the history.

        WHY: Validation loss over time is what tells you the model has started
        overfitting. It is only visible if it is recorded.

        STUDENT LEARNING: Record the series, not just the latest value.
        """
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        assert len(trainer.history["eval_loss"]) == 1

    def test_eval_loss_recorded_each_call(self):
        """
        WHAT: Each evaluation adds an entry rather than overwriting.

        WHY: A history that keeps only the last value cannot show a trend, which is the
        entire reason to keep it.

        STUDENT LEARNING: Append, do not assign.
        """
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        trainer.evaluate(data)
        assert len(trainer.history["eval_loss"]) == 2

    def test_classification_accuracy(self):
        """
        WHAT: Accuracy is computed correctly on known-correct predictions.

        WHY: Loss alone does not say whether the model is useful. Accuracy has to be
        pinned against predictions you worked out by hand.

        STUDENT LEARNING: Test a metric against a case where you already know the
        answer.
        """
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
        """
        WHAT: Accuracy is zero when every prediction is wrong.

        WHY: It is the other end of the range. A metric that never reaches zero is
        usually counting something other than what you think.

        STUDENT LEARNING: Check both ends of a bounded metric.
        """
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
        """
        WHAT: The learning rate is logged when a schedule is running.

        WHY: When a run goes wrong the first question is what the learning rate was
        doing. Unlogged, that is unanswerable after the fact.

        STUDENT LEARNING: Log the inputs to training, not only its outputs.
        """
        scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)
        trainer, _ = simple_trainer(scheduler=scheduler)
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.train_epoch(data)
        assert len(trainer.history["learning_rates"]) == 1

    def test_no_lr_in_history_without_scheduler(self):
        """
        WHAT: No learning-rate series is recorded without a schedule.

        WHY: A constant column adds noise to every plot and suggests a schedule that is
        not there.

        STUDENT LEARNING: Absence is part of the contract too.
        """
        trainer, _ = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.train_epoch(data)
        assert len(trainer.history["learning_rates"]) == 0

    def test_optimizer_lr_updated_by_scheduler(self):
        """
        WHAT: The schedule actually writes the rate into the optimizer.

        WHY: A scheduler that computes the right curve and never applies it is the quiet
        failure here: the log shows the intended rate while training uses the old one.

        STUDENT LEARNING: Check the effect on the optimizer, not the value the schedule
        returned.
        """
        scheduler = CosineSchedule(max_lr=0.1, min_lr=0.01, total_epochs=10)
        trainer, _ = simple_trainer(lr=0.1, scheduler=scheduler)
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        # train_epoch applies scheduler.get_lr(self.epoch) at the start of the epoch,
        # when self.epoch == 0, then increments epoch to 1 at the end.
        expected_lr = scheduler.get_lr(0)
        trainer.train_epoch(data)
        assert abs(trainer.optimizer.lr - expected_lr) < 1e-9

    def test_lr_decreases_over_epochs(self):
        """
        WHAT: The applied rate falls as epochs advance.

        WHY: It ties the schedule's own curve to what training really saw, across
        several epochs rather than one.

        STUDENT LEARNING: End-to-end after unit: verify the wiring, not just the parts.
        """
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
        """
        WHAT: Training runs to completion with clipping enabled.

        WHY: Clipping sits inside the update path, so a shape or type error there breaks
        every step rather than degrading quality.

        STUDENT LEARNING: A feature in the hot path must first not break the path.
        """
        trainer, _ = simple_trainer(lr=0.01, grad_clip=1.0)
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        loss = trainer.train_epoch(data)
        assert np.isfinite(loss), f"Loss should be finite with grad clipping, got {loss}"

    def test_weights_update_with_grad_clip(self):
        """
        WHAT: Weights still change when clipping is on.

        WHY: The failure to fear is clipping to zero, which stops learning while the
        loop keeps running and reporting.

        STUDENT LEARNING: Confirm the safety mechanism did not disable the thing it
        protects.
        """
        trainer, model = simple_trainer(lr=0.1, grad_clip=0.01)
        weights_before = model.parameters()[0].data.copy()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.train_epoch(data)
        assert not np.allclose(model.parameters()[0].data, weights_before)

    def test_very_tight_clip_limits_updates(self):
        """
        WHAT: A very small threshold produces correspondingly small updates.

        WHY: It shows clipping is doing its job proportionally rather than being ignored
        or applied once.

        STUDENT LEARNING: Push a parameter to its extreme and check the effect follows.
        """
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
        """
        WHAT: The model is in train mode while training.

        WHY: Dropout and batch norm need their training behavior. Training in eval mode
        trains a different model than the one you designed.

        STUDENT LEARNING: Mode is set by the loop, not left to the caller.
        """
        trainer, model = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.train_epoch(data)
        assert model.training is True
        assert trainer.training_mode is True

    def test_model_in_eval_mode_during_evaluate(self):
        """
        WHAT: The model is in eval mode while evaluating.

        WHY: The counterpart of the previous check, and the more common mistake, because
        evaluation is often added later.

        STUDENT LEARNING: Every mode switch needs its opposite tested.
        """
        trainer, model = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        assert model.training is False
        assert trainer.training_mode is False

    def test_train_after_eval_restores_train_mode(self):
        """
        WHAT: Training after evaluation returns the model to train mode.

        WHY: Evaluating mid-run is normal. A mode left in eval silently disables dropout
        for the rest of training.

        STUDENT LEARNING: State a function changes, it must restore.
        """
        trainer, model = simple_trainer()
        data = [(Tensor([[1.0, 0.5]]), Tensor([[2.0]]))]
        trainer.evaluate(data)
        assert model.training is False
        trainer.train_epoch(data)
        assert model.training is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

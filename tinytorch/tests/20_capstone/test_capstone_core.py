"""
Module 20: Capstone Core Tests
===============================

These tests verify the capstone submission and reporting system.

WHY THESE TESTS MATTER:
-----------------------
The capstone is where students prove their TinyTorch implementation works.
These tests verify:
1. BenchmarkReport can aggregate all metrics
2. Submission harness validates student work
3. The complete system integrates correctly

WHAT THIS MODULE TIES TOGETHER:
-------------------------------
- All modules (01-19) must work for capstone to pass
- Benchmarking (Module 19) provides metrics
- Optimization modules (14-18) show performance gains
"""

import pytest
import numpy as np
rng = np.random.default_rng(7)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.perf.benchmarking import BenchmarkSuite, BenchmarkResult


class TestBenchmarkSuite:
    """Test the benchmark suite functionality."""

    def test_benchmark_suite_import(self):
        """Verify BenchmarkSuite can be imported."""
        assert BenchmarkSuite is not None

    def test_benchmark_suite_can_instantiate(self):
        """Verify BenchmarkSuite can be created with models and datasets."""
        # BenchmarkSuite requires models and datasets lists
        class MockModel:
            def __init__(self, name):
                self.name = name
            def forward(self, x):
                return x

        models = [MockModel("test_model")]
        datasets = [{"name": "test", "data": rng.standard_normal((10, 4))}]
        suite = BenchmarkSuite(models=models, datasets=datasets)
        assert suite is not None

    def test_benchmark_result_import(self):
        """Verify BenchmarkResult can be imported."""
        assert BenchmarkResult is not None


class TestCapstoneValidation:
    """Test the capstone validation components."""

    def test_benchmarking_available(self):
        """Verify benchmarking module is available."""
        from tinytorch.perf import benchmarking
        assert benchmarking is not None

    def test_validate_tensor_operations(self):
        """
        WHAT: Verify basic tensor operations work.

        WHY: If tensors don't work, nothing else will.
        This is the most fundamental check.
        """
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])

        # Basic arithmetic
        c = a + b
        assert np.allclose(c.data, [5.0, 7.0, 9.0]), "Tensor addition broken"

        d = a * b
        assert np.allclose(d.data, [4.0, 10.0, 18.0]), "Tensor multiplication broken"

    def test_validate_gradient_flow(self):
        """
        WHAT: Verify gradients flow through a simple computation.

        WHY: This is the core of training. If gradients don't flow,
        the model cannot learn.
        """
        from tinytorch.core.autograd import enable_autograd
        enable_autograd()

        x = Tensor([2.0], requires_grad=True)
        y = x * x  # y = x^2
        y.backward()

        # dy/dx = 2x = 4.0
        assert x.grad is not None, "x didn't receive gradient"
        assert np.isclose(x.grad[0], 4.0), (
            f"Gradient should be 4.0 (2*x where x=2), got {x.grad[0]}"
        )

    def test_validate_layer_forward(self):
        """
        WHAT: Verify Linear layer produces output.

        WHY: Layers are the building blocks of neural networks.
        """
        layer = Linear(4, 2)
        x = Tensor(rng.standard_normal((1, 4)))

        output = layer.forward(x)

        assert output.shape == (1, 2), f"Wrong output shape: {output.shape}"


class TestEndToEndIntegration:
    """Test complete end-to-end functionality."""

    def test_simple_training_loop(self):
        """
        WHAT: Verify a complete training loop works.

        WHY: This is the ultimate integration test.
        If this works, the student's TinyTorch is complete.
        """
        from tinytorch.core.autograd import enable_autograd
        from tinytorch.core.optimizers import SGD
        enable_autograd()

        # Simple model
        layer = Linear(2, 1)
        # Use small learning rate to avoid gradient explosion
        optimizer = SGD(layer.parameters(), lr=0.01)

        # Fake data: y = x1 + x2 (simple linear pattern)
        x = Tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        target = Tensor([[3.0], [5.0], [7.0]])

        initial_loss = None
        final_loss = None

        # Training loop - more epochs with lower LR
        for epoch in range(50):
            optimizer.zero_grad()

            # Forward
            pred = layer.forward(x)

            # Loss (MSE) - use mean instead of sum to normalize
            diff = pred - target
            loss = (diff * diff).sum() / 3  # Divide by batch size

            if initial_loss is None:
                initial_loss = float(loss.data)

            # Backward
            loss.backward()

            # Update
            optimizer.step()

            final_loss = float(loss.data)

        # Loss should decrease
        assert final_loss < initial_loss, (
            f"Training didn't reduce loss!\n"
            f"  Initial: {initial_loss}\n"
            f"  Final: {final_loss}\n"
            "This means the training loop is broken."
        )


class TestValidateSubmissionSchema:
    """Test validate_submission_schema is importable and rejects bad input.

    This function previously wasn't exported to the built tinytorch package
    at all (missing #| export marker), so it was unreachable from pytest or
    from anything outside the source notebook. These tests both confirm the
    import path works and exercise its actual validation logic, which had
    never been tested against malformed input before.
    """

    def _valid_submission(self):
        from tinytorch.olympics import validate_submission_schema
        return validate_submission_schema, {
            'tinytorch_version': '0.1.0',
            'submission_type': 'capstone_benchmark',
            'timestamp': '2026-01-01T00:00:00',
            'system_info': {'platform': 'test', 'python_version': '3.11'},
            'baseline': {
                'model_name': 'baseline_model',
                'metrics': {
                    'parameter_count': 1024,
                    'model_size_mb': 4.0,
                    'accuracy': 0.9,
                    'latency_ms_mean': 12.5,
                },
            },
        }

    def test_importable_from_package(self):
        """WHAT: Verify the function can actually be imported from tinytorch.olympics."""
        from tinytorch.olympics import validate_submission_schema
        assert validate_submission_schema is not None

    def test_valid_submission_passes(self):
        """WHAT: Verify a well-formed submission validates successfully."""
        validate_submission_schema, submission = self._valid_submission()
        assert validate_submission_schema(submission) is True

    def test_missing_required_field_raises(self):
        """WHAT: Verify a submission missing a required top-level field is rejected."""
        validate_submission_schema, submission = self._valid_submission()
        del submission['tinytorch_version']

        with pytest.raises(AssertionError):
            validate_submission_schema(submission)

    def test_accuracy_above_valid_range_raises(self):
        """WHAT: Verify an out-of-range accuracy value (> 1) is rejected."""
        validate_submission_schema, submission = self._valid_submission()
        submission['baseline']['metrics']['accuracy'] = 1.5

        with pytest.raises(AssertionError):
            validate_submission_schema(submission)

    def test_accuracy_below_valid_range_raises(self):
        """WHAT: Verify an out-of-range accuracy value (< 0) is rejected."""
        validate_submission_schema, submission = self._valid_submission()
        submission['baseline']['metrics']['accuracy'] = -0.1

        with pytest.raises(AssertionError):
            validate_submission_schema(submission)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear


class TestBenchmarkReport:
    """Test the benchmark report generation."""

    def test_report_import(self):
        """Verify BenchmarkReport can be imported."""
        try:
            from tinytorch.perf.benchmarking import BenchmarkReport
            assert BenchmarkReport is not None
        except ImportError as e:
            pytest.skip(f"BenchmarkReport not yet exported: {e}")

    def test_report_can_instantiate(self):
        """Verify BenchmarkReport can be created."""
        try:
            from tinytorch.perf.benchmarking import BenchmarkReport
            report = BenchmarkReport()
            assert report is not None
        except ImportError:
            pytest.skip("BenchmarkReport not yet exported")

    def test_report_can_add_metrics(self):
        """
        WHAT: Verify report can record benchmark metrics.

        WHY: The report aggregates all performance data.
        Students need to see their results.
        """
        try:
            from tinytorch.perf.benchmarking import BenchmarkReport
        except ImportError:
            pytest.skip("BenchmarkReport not yet exported")

        report = BenchmarkReport()

        # Add some metrics
        if hasattr(report, 'add_metric'):
            report.add_metric("latency_ms", 15.5)
            report.add_metric("throughput", 1000)
            report.add_metric("memory_mb", 256)

            # Verify metrics were recorded
            if hasattr(report, 'get_metric'):
                assert report.get_metric("latency_ms") == 15.5

    def test_report_can_generate_summary(self):
        """
        WHAT: Verify report can generate a summary.

        WHY: Students need a readable summary of their results.
        """
        try:
            from tinytorch.perf.benchmarking import BenchmarkReport
        except ImportError:
            pytest.skip("BenchmarkReport not yet exported")

        report = BenchmarkReport()

        if hasattr(report, 'summary'):
            summary = report.summary()
            assert isinstance(summary, (str, dict)), (
                "summary() should return string or dict"
            )


class TestSubmissionHarness:
    """Test the submission harness for capstone validation."""

    def test_submission_harness_import(self):
        """Verify submission harness can be imported."""
        try:
            from tinytorch.perf.benchmarking import SubmissionHarness
            assert SubmissionHarness is not None
        except ImportError:
            # This might be named differently
            pytest.skip("SubmissionHarness not yet exported")

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
        x = Tensor(np.random.randn(1, 4))

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Exercise the authored module directly, without relying on its generated export."""

from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.fixture(scope="module")
def losses_source():
    source = Path(__file__).resolve().parents[2] / "src/04_losses/04_losses.py"
    return runpy.run_path(str(source))


def test_authored_module_integration(losses_source):
    """Run the same implementation checks that learners execute in the notebook."""
    losses_source["test_module"]()


def test_bce_positive_target_decreases_and_clipping_is_flat(losses_source):
    """The illustrated monotone curve and saturated regions match the actual loss."""
    operation = losses_source["BinaryCrossEntropyFunction"]()
    target = np.array([1.0], dtype=np.float32)
    values = [operation.forward(np.array([p], dtype=np.float32), target)
              for p in [0.01, 0.2, 0.8, 0.99]]
    assert np.all(np.diff(values) < 0)
    np.testing.assert_allclose(values, -np.log([0.01, 0.2, 0.8, 0.99]), rtol=1e-5)
    low = operation.forward(np.array([4e-8], dtype=np.float32), target)
    high = operation.forward(np.array([6e-8], dtype=np.float32), target)
    assert low == high
    assert np.isfinite(low)


def test_sensitivity_analysis_uses_valid_probability_axis(losses_source, capsys):
    """The reported BCE optimum must be a probability, not the regression grid's 1.02."""
    losses_source["analyze_loss_sensitivity"]()
    binary_report = capsys.readouterr().out.split("Binary Cross-Entropy Loss:")[1]
    assert "Minimum at prediction = 1.00" in binary_report

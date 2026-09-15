"""Check authored compression code, including composition and actual storage."""

from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.fixture(scope="module")
def compression_source():
    source = Path(__file__).resolve().parents[2] / "src/16_compression/16_compression.py"
    return runpy.run_path(str(source))


@pytest.mark.parametrize("nested_head", [False, True])
def test_nested_pruning_preserves_terminal_classifier(compression_source, nested_head):
    """Nested containers must not turn the classifier into a prunable hidden layer."""
    Linear = compression_source["Linear"]
    Sequential = compression_source["Sequential"]
    hidden, head = Linear(2, 4), Linear(4, 2)
    hidden.weight.data[:] = 1
    hidden.bias.data[:] = 1
    head.weight.data[:] = 2
    head.bias.data[:] = 3
    tail = Sequential(head) if nested_head else head
    model = Sequential(Sequential(hidden, compression_source["ReLU"]()), tail)
    original_head = [p.data.copy() for p in head.parameters()]

    report = compression_source["compress_model"](model, {"structured_prune": 1.0})

    assert not np.any(hidden.weight.data)
    assert not np.any(hidden.bias.data)
    for param, original in zip(head.parameters(), original_head):
        np.testing.assert_array_equal(param.data, original)
    assert report["final_sparsity"] == 50.0
    # The retained classifier still produces its bias, not erased class outputs.
    output = model(compression_source["Tensor"]([[1., 2.]]))
    np.testing.assert_array_equal(output.data, [[3., 3.]])


def test_low_rank_factors_release_full_svd_storage(compression_source):
    """Small returned factors must own small allocations and retain the SVD result."""
    matrix = np.diag(np.arange(50, 0, -1, dtype=np.float32))
    matrix = np.vstack([matrix, np.zeros_like(matrix)])
    factors = compression_source["low_rank_approximate"](matrix, rank_ratio=0.1)
    u, s, vt = factors

    assert all(factor.flags.owndata for factor in factors)
    assert sum(factor.nbytes for factor in factors) == 5 * (100 + 50 + 1) * 4
    assert sum(factor.nbytes for factor in factors) < matrix.nbytes
    expected = np.zeros_like(matrix)
    expected[np.arange(5), np.arange(5)] = np.arange(50, 45, -1)
    np.testing.assert_allclose(u @ np.diag(s) @ vt, expected)


def test_authored_module_integration(compression_source):
    compression_source["test_module"]()

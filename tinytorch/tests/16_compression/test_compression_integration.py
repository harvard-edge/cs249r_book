#!/usr/bin/env python3
"""
Integration tests for Module 16: Compression.

Pruning is only useful if it removes the weights it claims to remove, keeps the
model runnable, and removes the *smallest* ones. These tests check all three.
"""

import numpy as np

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear, Sequential
from tinytorch.core.activations import ReLU
from tinytorch.perf.compression import measure_sparsity, magnitude_prune


def _model(seed=0):
    rng = np.random.default_rng(seed)
    model = Sequential(Linear(16, 12), ReLU(), Linear(12, 6))
    # Deterministic weights so sparsity targets are exact.
    for layer in (model.layers[0], model.layers[2]):
        layer.weight.data = rng.standard_normal(layer.weight.data.shape).astype(np.float32)
    return model


def test_fresh_model_reports_no_sparsity():
    assert measure_sparsity(_model()) == 0.0, (
        "A freshly initialised model reported nonzero sparsity; "
        "biases initialised to zero are probably being counted as pruned weights"
    )


def test_magnitude_prune_hits_the_requested_sparsity():
    for target in (0.25, 0.5, 0.9):
        model = _model()
        magnitude_prune(model, sparsity=target)
        actual = measure_sparsity(model)
        assert abs(actual - target * 100) < 2.0, (
            f"Asked for {target:.0%} sparsity, measured {actual:.1f}%"
        )


def test_pruning_removes_the_smallest_weights_first():
    """The defining property of magnitude pruning."""
    model = _model(seed=3)
    before = [layer.weight.data.copy() for layer in (model.layers[0], model.layers[2])]

    magnitude_prune(model, sparsity=0.5)

    for original, layer in zip(before, (model.layers[0], model.layers[2])):
        now = layer.weight.data
        pruned = now == 0
        kept = ~pruned
        if pruned.any() and kept.any():
            largest_pruned = np.abs(original[pruned]).max()
            smallest_kept = np.abs(original[kept]).min()
            assert largest_pruned <= smallest_kept + 1e-6, (
                "Magnitude pruning removed a weight larger than one it kept: "
                f"pruned {largest_pruned:.6f} but kept {smallest_kept:.6f}"
            )
        # Surviving weights must be untouched, not rescaled.
        np.testing.assert_allclose(now[kept], original[kept], rtol=1e-6, atol=1e-6)


def test_pruned_model_still_runs():
    rng = np.random.default_rng(4)
    model = _model()
    x = Tensor(rng.standard_normal((3, 16)).astype(np.float32))
    reference_shape = model(x).data.shape

    magnitude_prune(model, sparsity=0.5)
    out = model(x)

    assert out.data.shape == reference_shape, "Pruning changed the output shape"
    assert np.isfinite(out.data).all(), "Pruned model produced NaN or inf"


if __name__ == "__main__":
    test_fresh_model_reports_no_sparsity()
    test_magnitude_prune_hits_the_requested_sparsity()
    test_pruning_removes_the_smallest_weights_first()
    test_pruned_model_still_runs()
    print("✅ Compression integration tests passed")

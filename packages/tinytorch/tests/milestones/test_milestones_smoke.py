"""
Milestone Smoke Tests — Model Construction
===========================================

Lightweight tests that verify every milestone script can at least
import its dependencies and construct its model. No data downloads,
no training — just "does the code not crash on import?"

These catch API drift between milestone scripts and the modules they
import (e.g., pool_size vs kernel_size — see GitHub issue #1278).

Run time: < 5 seconds total.

Usage:
    pytest tests/milestones/test_milestones_smoke.py -v
"""

import sys
import os
import importlib
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

# Setup paths
TINYTORCH_ROOT = Path(__file__).parent.parent.parent
MILESTONES_DIR = TINYTORCH_ROOT / "milestones"

sys.path.insert(0, str(TINYTORCH_ROOT))
sys.path.insert(0, str(MILESTONES_DIR))


def _import_milestone(script_path: Path):
    """Import a milestone script as a module without executing main()."""
    spec = importlib.util.spec_from_file_location(
        script_path.stem, script_path
    )
    module = importlib.util.module_from_spec(spec)

    # Suppress print output during import
    with patch("builtins.print"):
        # Prevent auto-execution of main() during import
        old_name = None
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            pass  # Some scripts call sys.exit in __main__ guard

    return module


class TestMilestoneImports:
    """Verify all milestone scripts can be imported without errors."""

    @pytest.mark.parametrize("script", sorted(MILESTONES_DIR.rglob("*.py")), ids=lambda p: str(p.relative_to(MILESTONES_DIR)))
    def test_milestone_imports(self, script):
        """Each milestone script should import without errors."""
        # Skip non-milestone files
        if script.name in ("data_manager.py", "networks.py", "__init__.py"):
            pytest.skip("Utility file, not a milestone script")
        if script.parent.name == "datasets":
            pytest.skip("Dataset directory")

        module = _import_milestone(script)
        assert module is not None, f"Failed to import {script.name}"


class TestModelConstruction:
    """Verify model classes can be instantiated (catches API mismatches)."""

    def test_milestone_01_perceptron(self):
        """Milestone 01: Perceptron model constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "01_1958_perceptron" / "01_rosenblatt_forward.py"
        )
        # Perceptron uses direct tensor ops, no model class to construct
        assert module is not None

    def test_milestone_02_xor_crisis(self):
        """Milestone 02 Part 1: XOR crisis script loads."""
        module = _import_milestone(
            MILESTONES_DIR / "02_1969_xor" / "01_xor_crisis.py"
        )
        assert module is not None

    def test_milestone_02_xor_solved(self):
        """Milestone 02 Part 2: XOR network constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "02_1969_xor" / "02_xor_solved.py"
        )
        if hasattr(module, "XORNetwork"):
            with patch("builtins.print"):
                model = module.XORNetwork()
            assert model is not None

    def test_milestone_03_mlp(self):
        """Milestone 03: DigitMLP constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "03_1986_mlp" / "01_rumelhart_tinydigits.py"
        )
        if hasattr(module, "DigitMLP"):
            with patch("builtins.print"):
                model = module.DigitMLP()
            assert model is not None

    def test_milestone_04_cnn_tinydigits(self):
        """Milestone 04 Part 1: SimpleCNN constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "04_1998_cnn" / "01_lecun_tinydigits.py"
        )
        if hasattr(module, "SimpleCNN"):
            with patch("builtins.print"):
                model = module.SimpleCNN()
            assert model is not None

    def test_milestone_04_cnn_cifar(self):
        """Milestone 04 Part 2: CIFARCNN constructs (no data needed).

        This is the exact test that would have caught issue #1278.
        """
        module = _import_milestone(
            MILESTONES_DIR / "04_1998_cnn" / "02_lecun_cifar10.py"
        )
        if hasattr(module, "CIFARCNN"):
            with patch("builtins.print"):
                model = module.CIFARCNN()
            assert model is not None

    def test_milestone_05_transformer(self):
        """Milestone 05: Transformer script loads."""
        module = _import_milestone(
            MILESTONES_DIR / "05_2017_transformer" / "01_vaswani_attention.py"
        )
        assert module is not None

    def test_milestone_06_networks(self):
        """Milestone 06: All network classes in networks.py construct."""
        module = _import_milestone(
            MILESTONES_DIR / "06_2018_mlperf" / "networks.py"
        )
        with patch("builtins.print"):
            if hasattr(module, "Perceptron"):
                assert module.Perceptron(input_size=10, num_classes=2) is not None
            if hasattr(module, "DigitMLP"):
                assert module.DigitMLP() is not None
            if hasattr(module, "SimpleCNN"):
                assert module.SimpleCNN() is not None


# ---------------------------------------------------------------------------
# Training regressions
#
# Construction smoke tests above prove a milestone imports and builds. They do
# not prove it *learns*, and Milestone 04 Part 2 shipped for a long time in a
# state where it built fine and trained nothing: the loss was computed in NumPy
# and re-wrapped in a fresh Tensor, and `flatten` did the same to the feature
# map, so no gradient reached the convolutional half of the network. The run
# printed a falling-looking loss and an accuracy, and every number was noise.
#
# This test trains on synthetic data so it needs no CIFAR-10 download.
# ---------------------------------------------------------------------------

def test_cifar_cnn_actually_trains():
    """Milestone 04 Part 2: every parameter must receive a gradient."""
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.optimizers import Adam
    from tinytorch.core.losses import CrossEntropyLoss

    script = MILESTONES_DIR / "04_1998_cnn" / "02_lecun_cifar10.py"
    if not script.exists():
        pytest.skip(f"{script} not present")
    cifar = _import_milestone(script)

    model = cifar.CIFARCNN()
    params = model.parameters()
    assert len(params) == 12, (
        f"CIFARCNN.parameters() returned {len(params)} tensors, expected 12 "
        "(conv1 w/b, bn1 gamma/beta, conv2 w/b, bn2 gamma/beta, fc1 w/b, fc2 w/b)"
    )

    optimizer = Adam(params, lr=0.001)
    criterion = CrossEntropyLoss()

    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((16, 3, 32, 32)).astype(np.float32))
    y = Tensor(rng.integers(0, 10, size=16).astype(np.int64))

    before = [p.data.copy() for p in params]
    losses = []
    model.train()
    for _ in range(8):
        loss = criterion(model(x), y)
        losses.append(float(loss.data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    stuck = [i for i, (b, p) in enumerate(zip(before, params)) if np.allclose(b, p.data)]
    assert not stuck, (
        f"Parameters {stuck} never changed. Gradients are not reaching them: check that "
        "flatten() uses Tensor.reshape and that the loss comes from CrossEntropyLoss, "
        "not a NumPy value re-wrapped in a new Tensor."
    )
    assert losses[-1] < losses[0], (
        f"Loss did not fall over 8 steps on a fixed batch: {losses}"
    )

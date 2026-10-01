"""
Milestone Smoke Tests: Model Construction
===========================================

Lightweight tests that verify every milestone script can at least
import its dependencies and construct its model. No data downloads,
no training: just "does the code not crash on import?"

These catch API drift between milestone scripts and the modules they
import (e.g., pool_size vs kernel_size; see GitHub issue #1278).

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
    """Import a milestone script as a module without executing main().

    The scripts guard their entry point with ``if __name__ == "__main__"``, so
    importing under the file stem runs only module-level code. A script that
    calls sys.exit during import is broken and fails the test; it is not
    silently treated as imported.
    """
    spec = importlib.util.spec_from_file_location(
        script_path.stem, script_path
    )
    module = importlib.util.module_from_spec(spec)

    # Suppress print output during import
    with patch("builtins.print"):
        try:
            spec.loader.exec_module(module)
        except SystemExit as exc:
            pytest.fail(f"{script_path.name} called sys.exit({exc.code!r}) at import time")

    return module


def _construct(module, class_name: str, *args, **kwargs):
    """Assert the milestone defines class_name, then construct it."""
    assert hasattr(module, class_name), (
        f"{module.__name__} no longer defines {class_name}; the smoke test "
        f"would otherwise skip model construction silently"
    )
    with patch("builtins.print"):
        return getattr(module, class_name)(*args, **kwargs)


def _param_count(params) -> int:
    return sum(int(np.prod(p.shape)) for p in params)


def _assert_build_model(module, **kwargs):
    assert callable(getattr(module, "build_model", None)), (
        f"{module.__name__} no longer defines build_model()"
    )
    with patch("builtins.print"):
        model, total_params = module.build_model(**kwargs)
    assert total_params > 0
    assert _param_count(model.parameters()) > 0
    return model


class TestMilestoneImports:
    """Verify all milestone scripts can be imported without errors."""

    @pytest.mark.parametrize("script", sorted(MILESTONES_DIR.rglob("*.py")), ids=lambda p: str(p.relative_to(MILESTONES_DIR)))
    def test_milestone_imports(self, script):
        """Each milestone script imports cleanly and exposes an entry point."""
        # Skip non-milestone files
        if script.name in ("data_manager.py", "networks.py", "try_it.py", "__init__.py"):
            pytest.skip("Utility file, not a milestone script")
        if script.parent.name == "datasets":
            pytest.skip("Dataset directory")

        module = _import_milestone(script)
        entry_points = [
            name for name in vars(module)
            if (name == "main" or name.startswith("train_")) and callable(getattr(module, name))
        ]
        assert entry_points, f"{script.name} defines no main() or train_*() entry point"


class TestModelConstruction:
    """Verify model classes can be instantiated (catches API mismatches)."""

    def test_milestone_01_perceptron(self):
        """Milestone 01: Perceptron model constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "01_1958_perceptron" / "01_rosenblatt_forward.py"
        )
        model = _construct(module, "Perceptron", input_size=2, output_size=1)
        from tinytorch.core.tensor import Tensor
        out = model(Tensor(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]])))
        assert out.shape == (3, 1)
        assert np.all((out.data > 0) & (out.data < 1)), "Sigmoid output must lie in (0, 1)"

    def test_milestone_02_xor_crisis(self):
        """Milestone 02 Part 1: XOR crisis single-layer perceptron constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "02_1969_xor" / "01_xor_crisis.py"
        )
        model = _construct(module, "SingleLayerPerceptron")
        from tinytorch.core.tensor import Tensor
        model.set_weights(1.0, 1.0, -1.5)  # an AND gate
        out = model(Tensor(np.array([[0.0, 0.0], [1.0, 1.0]])))
        expected = 1.0 / (1.0 + np.exp(-np.array([[-1.5], [0.5]])))
        np.testing.assert_allclose(out.data, expected, rtol=1e-5)

    def test_milestone_02_xor_solved(self):
        """Milestone 02 Part 2: XOR network constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "02_1969_xor" / "02_xor_solved.py"
        )
        model = _construct(module, "XORNetwork")
        assert _param_count(model.parameters()) > 0

    def test_milestone_03_mlp(self):
        """Milestone 03: DigitMLP constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "03_1986_mlp" / "01_rumelhart_tinydigits.py"
        )
        model = _construct(module, "DigitMLP")
        assert _param_count(model.parameters()) > 0

    def test_milestone_04_cnn_tinydigits(self):
        """Milestone 04 Part 1: SimpleCNN constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "04_1998_cnn" / "01_lecun_tinydigits.py"
        )
        model = _construct(module, "SimpleCNN")
        assert _param_count(model.parameters()) > 0

    def test_milestone_04_cnn_cifar(self):
        """Milestone 04 Part 2: CIFARCNN constructs (no data needed).

        This is the exact test that would have caught issue #1278.
        """
        module = _import_milestone(
            MILESTONES_DIR / "04_1998_cnn" / "02_lecun_cifar10.py"
        )
        model = _construct(module, "CIFARCNN")
        assert _param_count(model.parameters()) > 0

    def test_milestone_05_transformer_tinygpt(self):
        """Milestone 05: TinyGPT model constructs and script loads."""
        module = _import_milestone(
            MILESTONES_DIR / "05_2017_transformer" / "01_tinygpt_shakespeare.py"
        )
        _assert_build_model(module, vocab_size=50, embed_dim=32, num_layers=1, num_heads=2, max_seq_len=32)

    def test_milestone_05_sequence_routing(self):
        """Milestone 05 Part 2: Sequence routing model constructs."""
        module = _import_milestone(
            MILESTONES_DIR / "05_2017_transformer" / "02_vaswani_attention.py"
        )
        model = _construct(module, "AttentionTransformer", vocab_size=30, embed_dim=32, num_heads=2, seq_len=8, num_layers=1)
        assert _param_count(model.parameters()) > 0

    def test_milestone_05_tinycopilot(self):
        """Milestone 05 Part 3: TinyCopilot code generation model constructs and script loads."""
        module = _import_milestone(
            MILESTONES_DIR / "05_2017_transformer" / "03_tinycopilot.py"
        )
        _assert_build_model(module, vocab_size=50, embed_dim=32, num_layers=1, num_heads=2, max_seq_len=32)

    def test_milestone_05_conversational_chat(self):
        """Milestone 05 Part 4: Conversational Q&A model constructs and script loads."""
        module = _import_milestone(
            MILESTONES_DIR / "05_2017_transformer" / "04_tinygpt_chat.py"
        )
        _assert_build_model(module, vocab_size=50, embed_dim=32, num_layers=1, num_heads=2, max_seq_len=32)

    def test_milestone_06_networks(self):
        """Milestone 06: All network classes in networks.py construct."""
        module = _import_milestone(
            MILESTONES_DIR / "06_2018_mlperf" / "networks.py"
        )
        for class_name, kwargs in (
            ("Perceptron", dict(input_size=10, num_classes=2)),
            ("DigitMLP", {}),
            ("SimpleCNN", {}),
            ("MinimalTransformer", {}),
            ("TinyGPT", {}),
        ):
            model = _construct(module, class_name, **kwargs)
            assert _param_count(model.parameters()) > 0, f"{class_name} has no parameters"

    def test_milestone_07_kernels(self):
        """Milestone 07: Custom GPU kernels script loads and exposes main()."""
        module = _import_milestone(
            MILESTONES_DIR / "07_2024_kernels" / "01_custom_kernels.py"
        )
        assert callable(getattr(module, "main", None))

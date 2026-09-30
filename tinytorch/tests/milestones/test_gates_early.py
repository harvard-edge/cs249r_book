"""
Early-milestone gates must fail when the student's code is broken.
==================================================================

Each milestone script runs as a subprocess, the way ``tito milestone run``
runs it. A sabotage is injected through a ``sitecustomize`` module placed
first on PYTHONPATH: it breaks one piece of the exported ``tinytorch``
package before the script starts, so the script sees broken student code.

History (2026-09-29, release audit):
  * 01_rosenblatt_forward.py never set an exit code, so it passed when
    Linear returned zeros.
  * 01_xor_crisis.py always returned 0, even after "Found a solution!",
    which only broken code can produce.
  * 02_xor_solved.py passed with backprop stopped at the output layer: its
    fixed seed drew hidden features that already separated XOR, and its gate
    read the last pre-update training accuracy.
  * 02_xor_solved.py and 03_1986_mlp/01_rumelhart_tinydigits.py passed with
    the Module 04 loss forward returning 0: the gradient comes from the
    backward pass, so training worked while every printed loss was wrong.
    Each now checks one batch against a NumPy loss before training.

Each script runs in about a second, so these stay in the fast suite.

Set TT_GATES_MILESTONES_DIR to point the tests at another copy of the
milestones directory (used to show the tests fail on older versions).
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

TINYTORCH_ROOT = Path(__file__).resolve().parents[2]
MILESTONES_DIR = Path(os.environ.get("TT_GATES_MILESTONES_DIR", TINYTORCH_ROOT / "milestones"))

PERCEPTRON = "01_1958_perceptron/01_rosenblatt_forward.py"
XOR_CRISIS = "02_1969_xor/01_xor_crisis.py"
XOR_SOLVED = "02_1969_xor/02_xor_solved.py"
MLP_DIGITS = "03_1986_mlp/01_rumelhart_tinydigits.py"

SITECUSTOMIZE = textwrap.dedent('''
    """Test-only sabotage hook: TT_SABOTAGE=<name> breaks one student module."""
    import os
    import sys

    _name = os.environ.get("TT_SABOTAGE", "")

    if _name:
        import numpy as np

        if _name == "linear_zero":
            # Linear ignores its weights and input and returns zeros.
            import tinytorch.core.layers as L
            from tinytorch.core.tensor import Tensor

            def _forward(self, x):
                return Tensor(np.zeros(x.shape[:-1] + (self.weight.data.shape[1],),
                                       dtype=np.float32))

            L.Linear.forward = _forward
            L.Linear.__call__ = lambda self, x: _forward(self, x)

        elif _name == "frozen_hidden":
            # MatMul backward returns no gradient for its INPUT, so backprop
            # stops at the last layer and the hidden layer never learns.
            import tinytorch.core.autograd as ag
            _orig = ag.MatMul.backward

            def _backward(self, grad, _orig=_orig):
                ga, gb = _orig(self, grad)
                return (None if ga is None else np.zeros_like(ga)), gb

            ag.MatMul.backward = _backward

        elif _name == "opt_noop":
            # The optimizer step does nothing: parameters never change.
            import tinytorch.core.optimizers as o
            for _cls in (o.SGD, o.Adam, o.AdamW):
                _cls.step = lambda self: None

        elif _name in ("loss_zero", "loss_sum"):
            # The loss forward returns 0 (or sums instead of averaging) while
            # its backward stays correct, so training still works.
            import tinytorch.core.losses as LS
            for _cls in (LS.CrossEntropyFunction, LS.BinaryCrossEntropyFunction):
                def _forward(self, *args, _orig=_cls.forward, **kwargs):
                    value = _orig(self, *args, **kwargs)
                    if _name == "loss_zero":
                        return value * 0.0
                    return value * np.asarray(args[0]).shape[0]
                _cls.forward = _forward

        else:
            raise SystemExit(f"unknown sabotage {_name}")

        sys.stderr.write(f"[SABOTAGE] {_name}\\n")
''')


@pytest.fixture(scope="module")
def sabotage_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("sabotage")
    (path / "sitecustomize.py").write_text(SITECUSTOMIZE)
    return path


def run_script(script, sabotage_dir, sabotage="", args=()):
    env = dict(os.environ)
    env.update(
        TINYTORCH_NON_INTERACTIVE="1",
        CI="true",
        PYTHONPATH=f"{sabotage_dir}{os.pathsep}{TINYTORCH_ROOT}",
        TT_SABOTAGE=sabotage,
        PYTHONUNBUFFERED="1",
        COLUMNS="160",
    )
    proc = subprocess.run(
        [sys.executable, str(MILESTONES_DIR / script), *args],
        cwd=TINYTORCH_ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=120,
    )
    if sabotage:
        assert f"[SABOTAGE] {sabotage}" in proc.stderr, "sabotage hook did not load"
    return proc


def describe(proc):
    return f"rc={proc.returncode}\n--- stdout (tail) ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-2000:]}"


@pytest.mark.parametrize("script", [PERCEPTRON, XOR_CRISIS, XOR_SOLVED, MLP_DIGITS])
def test_correct_code_passes(script, sabotage_dir):
    proc = run_script(script, sabotage_dir)
    assert proc.returncode == 0, describe(proc)


@pytest.mark.parametrize("script", [PERCEPTRON, XOR_CRISIS])
def test_linear_returning_zeros_fails_forward_milestones(script, sabotage_dir):
    proc = run_script(script, sabotage_dir, "linear_zero")
    assert proc.returncode == 1, describe(proc)
    assert "Forward pass check FAILED" in proc.stdout, describe(proc)


def test_frozen_hidden_layer_fails_xor_solved(sabotage_dir):
    proc = run_script(XOR_SOLVED, sabotage_dir, "frozen_hidden")
    assert proc.returncode == 1, describe(proc)


def test_frozen_hidden_layer_fails_even_on_a_seed_whose_features_separate_xor(sabotage_dir):
    # Seed 1986 (the old default) draws hidden features that already separate
    # XOR, so the truth table alone passes; the hidden-gradient gate must not.
    proc = run_script(XOR_SOLVED, sabotage_dir, "frozen_hidden", ["--seed", "1986"])
    assert proc.returncode == 1, describe(proc)


def test_noop_optimizer_step_fails_xor_solved(sabotage_dir):
    proc = run_script(XOR_SOLVED, sabotage_dir, "opt_noop")
    assert proc.returncode == 1, describe(proc)


@pytest.mark.parametrize("script", [XOR_SOLVED, MLP_DIGITS])
@pytest.mark.parametrize("sabotage", ["loss_zero", "loss_sum"])
def test_wrong_loss_forward_fails(script, sabotage, sabotage_dir):
    proc = run_script(script, sabotage_dir, sabotage)
    assert proc.returncode == 1, describe(proc)
    assert "YOUR loss value is wrong" in proc.stdout, describe(proc)
    assert "check Module 04" in proc.stdout, describe(proc)


def test_noop_optimizer_step_fails_mlp(sabotage_dir):
    proc = run_script(MLP_DIGITS, sabotage_dir, "opt_noop")
    assert proc.returncode == 1, describe(proc)


# --------------------------------------------------------------------------
# The loss-forward consistency helpers, called directly (no subprocess)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xor_solved():
    from test_milestones_smoke import _import_milestone
    return _import_milestone(MILESTONES_DIR / XOR_SOLVED)


@pytest.fixture(scope="module")
def mlp_digits():
    from test_milestones_smoke import _import_milestone
    return _import_milestone(MILESTONES_DIR / MLP_DIGITS)


class _ConstantLoss:
    """A stand-in for a broken student loss that returns a fixed value."""

    def __init__(self, value):
        self.value = value

    def __call__(self, outputs, targets):
        from tinytorch.core.tensor import Tensor
        return Tensor(np.asarray(self.value, dtype=np.float32))


def _bce_batch():
    from tinytorch.core.tensor import Tensor
    rng = np.random.default_rng(0)
    # Include exact 0 and 1 so the clip matters.
    p = np.concatenate([[[0.0], [1.0]], rng.uniform(0.01, 0.99, (30, 1))]).astype(np.float32)
    t = np.concatenate([[[1.0], [0.0]], rng.integers(0, 2, (30, 1))]).astype(np.float32)
    return Tensor(p), Tensor(t)


def _ce_batch(scale=1.0):
    from tinytorch.core.tensor import Tensor
    rng = np.random.default_rng(1)
    logits = (rng.normal(size=(32, 10)) * scale).astype(np.float32)
    labels = (np.arange(32) % 10).astype(np.int64)
    return Tensor(logits), Tensor(labels)


def test_reference_bce_matches_module_04(xor_solved):
    from tinytorch.core.losses import BinaryCrossEntropyLoss
    p, t = _bce_batch()
    assert xor_solved.loss_forward_failure(BinaryCrossEntropyLoss(), p, t,
                                           xor_solved.reference_bce) is None
    assert xor_solved.reference_bce(np.array([0.5]), np.array([1.0])) == pytest.approx(np.log(2))


@pytest.mark.parametrize("scale", [1.0, 1000.0])
def test_reference_cross_entropy_matches_module_04(mlp_digits, scale):
    from tinytorch.core.losses import CrossEntropyLoss
    logits, labels = _ce_batch(scale)
    ref = mlp_digits.reference_cross_entropy(logits.data, labels.data)
    assert np.isfinite(ref)
    assert mlp_digits.loss_forward_failure(CrossEntropyLoss(), logits, labels,
                                           mlp_digits.reference_cross_entropy) is None
    # Uniform logits: the loss is log(num_classes).
    assert mlp_digits.reference_cross_entropy(np.zeros((4, 10)), np.arange(4)) == pytest.approx(np.log(10))


@pytest.mark.parametrize("which", ["xor", "mlp"])
@pytest.mark.parametrize("value, expect", [
    (0.0, "returns 0.0000"),
    (float("nan"), "returns nan"),
    (np.zeros(3), "array of shape (3,)"),
])
def test_loss_forward_failure_flags_wrong_values(xor_solved, mlp_digits, which, value, expect):
    module, batch, ref = ((xor_solved, _bce_batch(), xor_solved.reference_bce) if which == "xor"
                          else (mlp_digits, _ce_batch(), mlp_digits.reference_cross_entropy))
    message = module.loss_forward_failure(_ConstantLoss(value), *batch, ref)
    assert message and expect in message and "check Module 04" in message


def test_loss_forward_failure_flags_sum_reduction(mlp_digits):
    logits, labels = _ce_batch()
    summed = mlp_digits.reference_cross_entropy(logits.data, labels.data) * 32
    assert mlp_digits.loss_forward_failure(_ConstantLoss(summed), logits, labels,
                                           mlp_digits.reference_cross_entropy)

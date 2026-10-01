"""Milestone 04 pass gates: a CNN passes only if its convolution learned.

2026-09-29: Part 1 (TinyDigits) passed at 81% with every Conv2d filter
gradient zeroed, because a Linear head on random 3x3 filters already
classifies 8x8 digits. Part 2 (CIFAR-10) printed SUCCESS at any accuracy.

Fast tests call the extracted gate functions on small synthetic arrays and
never load CIFAR-10. The slow tests run Part 1 end to end (~2.5 min each):
once as shipped (must exit 0) and once with the conv filter gradients zeroed
by a sitecustomize hook (must exit 1). A third slow test zeroes the Module 04
loss forward, which passed Part 1 until 2026-09-29; it exits before training.
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from test_milestones_smoke import MILESTONES_DIR, _import_milestone
from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import SGD
from tinytorch.core.dataloader import DataLoader, TensorDataset

TINYTORCH_ROOT = MILESTONES_DIR.parent
DIGITS = MILESTONES_DIR / "04_1998_cnn" / "01_lecun_tinydigits.py"
CIFAR = MILESTONES_DIR / "04_1998_cnn" / "02_lecun_cifar10.py"


@pytest.fixture(scope="module")
def digits():
    return _import_milestone(DIGITS)


@pytest.fixture(scope="module")
def cifar():
    return _import_milestone(CIFAR)


# --------------------------------------------------------------------------
# Part 1: TinyDigits conv-learning gate (fast)
# --------------------------------------------------------------------------

def test_relative_change(digits):
    before = np.ones((2, 2))
    assert digits.relative_change(before, before) == 0.0
    assert digits.relative_change(before, before * 1.5) == pytest.approx(0.5)
    # A zero-initialized tensor falls back to absolute change.
    assert digits.relative_change(np.zeros(4), np.full(4, 0.5)) == pytest.approx(1.0)


def test_conv_gate_passes_measured_correct_run(digits):
    # Numbers from a correct 50-epoch run (2026-09-29): grad 0.25, moved 76%.
    assert digits.conv_learning_failures([0.25], [0.76]) == []


@pytest.mark.parametrize("grad, moved, expect", [
    (0.0, 0.0, "never received a non-zero filter gradient"),    # frozen filters
    (4.2e-2, 0.145, "moved only"),                              # grad scaled 0.1
    (4.5e-3, 0.017, "moved only"),                              # grad scaled 0.01
    (4.1e-4, 0.002, "moved only"),                              # grad scaled 0.001
    (float("nan"), float("nan"), "NaN"),
])
def test_conv_gate_fails_measured_sabotage(digits, grad, moved, expect):
    failures = digits.conv_learning_failures([grad], [moved])
    assert failures and any(expect in f for f in failures)


def test_conv_gate_requires_a_conv_layer(digits):
    assert digits.conv_learning_failures([], [])


def test_threshold_separates_measured_runs(digits):
    # Correct runs moved 76%; filter gradients scaled by 0.1 moved 14.5%.
    assert 0.145 < digits.MIN_CONV_RELATIVE_CHANGE < 0.76
    assert digits.MIN_TEST_ACCURACY == 75.0


def _tiny_batches(n=12, seed=0):
    rng = np.random.default_rng(seed)
    images = Tensor(rng.normal(size=(n, 1, 8, 8)).astype(np.float32))
    labels = Tensor((np.arange(n) % 10).astype(np.int64))
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)


def test_monitor_sees_real_conv_gradient(digits):
    model = digits.SimpleCNN()
    convs = digits.conv_layers(model)
    assert convs == [model.conv1]
    monitor = digits.ConvGradientMonitor(convs)
    digits.train_epoch(model, _tiny_batches(), CrossEntropyLoss(),
                       SGD(model.parameters(), lr=0.01), on_backward=monitor)
    assert monitor.max_abs_grad[0] > 0


def test_monitor_catches_zeroed_conv_gradient(digits, monkeypatch):
    import tinytorch.core.spatial as sp
    original = sp.Conv2dFunction.backward

    def frozen(self, grad_output):
        out = original(self, grad_output)
        return tuple([out[0]] + [None if g is None else np.zeros_like(g) for g in out[1:]])

    monkeypatch.setattr(sp.Conv2dFunction, "backward", frozen)
    model = digits.SimpleCNN()
    convs = digits.conv_layers(model)
    before = [c.weight.data.copy() for c in convs]
    monitor = digits.ConvGradientMonitor(convs)
    digits.train_epoch(model, _tiny_batches(), CrossEntropyLoss(),
                       SGD(model.parameters(), lr=0.01), on_backward=monitor)
    moved = [digits.relative_change(b, c.weight.data) for b, c in zip(before, convs)]
    assert monitor.max_abs_grad == [0.0]
    assert moved == [0.0]
    assert digits.conv_learning_failures(monitor.max_abs_grad, moved)


def test_loss_check_accepts_module_04_cross_entropy(digits):
    logits = Tensor(np.random.default_rng(2).normal(size=(32, 10)).astype(np.float32) * 50)
    labels = Tensor((np.arange(32) % 10).astype(np.int64))
    assert digits.loss_forward_failure(CrossEntropyLoss(), logits, labels,
                                       digits.reference_cross_entropy) is None


def test_loss_check_flags_zero_loss(digits):
    logits = Tensor(np.zeros((4, 10), dtype=np.float32))
    labels = Tensor(np.arange(4, dtype=np.int64))
    message = digits.loss_forward_failure(lambda o, t: Tensor(np.float32(0.0)), logits, labels,
                                          digits.reference_cross_entropy)
    assert message and "returns 0.0000" in message and f"{np.log(10):.4f}" in message


# --------------------------------------------------------------------------
# Part 2: CIFAR-10 gate (fast, synthetic numbers only; never loads CIFAR-10)
# --------------------------------------------------------------------------

def test_cifar_floors_are_above_chance(cifar):
    assert cifar.MIN_TEST_ACCURACY >= 2.5 * 10
    assert cifar.MIN_TEST_ACCURACY_QUICK >= 2 * 10
    assert cifar.EXIT_GATE_FAILED == 1


def test_cifar_gate_passes_learning_model(cifar):
    moved = {"conv1": 0.3, "conv2": 0.2, "fc1": 0.05, "fc2": 0.4}
    assert cifar.cifar_gate_failures(40.0, moved) == []


def test_cifar_gate_fails_chance_accuracy(cifar):
    moved = {"conv1": 0.3, "conv2": 0.2, "fc1": 0.05, "fc2": 0.4}
    failures = cifar.cifar_gate_failures(11.0, moved)
    assert len(failures) == 1 and "below" in failures[0]


def test_cifar_gate_fails_frozen_conv_even_at_high_accuracy(cifar):
    moved = {"conv1": 0.0, "conv2": 0.2, "fc1": 0.05, "fc2": 0.4}
    failures = cifar.cifar_gate_failures(60.0, moved)
    assert len(failures) == 1 and failures[0].startswith("conv1.weight did not move")


def test_cifar_gate_fails_nan(cifar):
    moved = {"conv1": float("nan"), "conv2": 0.2, "fc1": 0.05, "fc2": 0.4}
    assert len(cifar.cifar_gate_failures(float("nan"), moved)) == 2


def test_cifar_quick_floor_is_applied(cifar):
    moved = {"conv1": 0.3, "conv2": 0.2, "fc1": 0.05, "fc2": 0.4}
    assert cifar.cifar_gate_failures(22.0, moved)
    assert cifar.cifar_gate_failures(22.0, moved,
                                     min_accuracy=cifar.MIN_TEST_ACCURACY_QUICK) == []


def test_cifar_gated_weights_cover_conv_and_linear(cifar):
    model = cifar.CIFARCNN()
    names = [name for name, _ in cifar.gated_weights(model)]
    assert names == ["conv1", "conv2", "fc1", "fc2"]
    assert cifar.relative_change(model.fc2.weight.data, model.fc2.weight.data) == 0.0


def test_cifar_success_message_is_behind_the_gate():
    source = CIFAR.read_text(encoding="utf-8")
    gate = source.index("failures = cifar_gate_failures(")
    assert source.index("SUCCESS! CIFAR-10 CNN Milestone Complete") > gate
    assert source.index("YOUR computer vision works!") > gate
    assert "SMOKE CHECK ONLY" in source


# --------------------------------------------------------------------------
# Part 1 end to end (slow): correct code passes, frozen filters fail
# --------------------------------------------------------------------------

FREEZE_CONV_HOOK = textwrap.dedent("""
    import numpy as np
    import tinytorch.core.spatial as sp
    _ob = sp.Conv2dFunction.backward
    def _bw(self, g, _ob=_ob):
        out = _ob(self, g)
        return tuple([out[0]] + [None if t is None else np.zeros_like(t) for t in out[1:]])
    sp.Conv2dFunction.backward = _bw
""")


def _run_digits(extra_pythonpath=None):
    env = dict(os.environ, TINYTORCH_NON_INTERACTIVE="1", CI="true")
    if extra_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(extra_pythonpath)] + [p for p in [env.get("PYTHONPATH")] if p])
    result = subprocess.run([sys.executable, str(DIGITS)], cwd=TINYTORCH_ROOT, env=env,
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=900)
    return result.returncode, result.stdout + result.stderr


@pytest.mark.slow
def test_digits_correct_run_passes():
    code, out = _run_digits()
    assert code == 0, out[-3000:]
    assert "Success! Your CNN Learned" in out


@pytest.mark.slow
def test_digits_frozen_conv_fails(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(FREEZE_CONV_HOOK)
    code, out = _run_digits(tmp_path)
    assert code == 1, out[-3000:]
    assert "YOUR convolution did not learn" in out
    assert "Success!" not in out


ZERO_LOSS_HOOK = textwrap.dedent("""
    import tinytorch.core.losses as LS
    _of = LS.CrossEntropyFunction.forward
    LS.CrossEntropyFunction.forward = lambda self, *a, _of=_of, **k: _of(self, *a, **k) * 0.0
""")


@pytest.mark.slow
def test_digits_zero_loss_forward_fails(tmp_path):
    (tmp_path / "sitecustomize.py").write_text(ZERO_LOSS_HOOK)
    code, out = _run_digits(tmp_path)
    assert code == 1, out[-3000:]
    assert "YOUR loss value is wrong" in out and "check Module 04" in out
    assert "Success!" not in out

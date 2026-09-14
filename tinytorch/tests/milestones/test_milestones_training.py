"""Slow milestone training regressions, separated from construction smoke tests.

Run with: python3 -m pytest tests/milestones/test_milestones_training.py -v
"""
import numpy as np
import pytest
from test_milestones_smoke import MILESTONES_DIR, _import_milestone


# ---------------------------------------------------------------------------
# Training regressions
#
# Construction tests in test_milestones_smoke.py prove a milestone imports and builds. They do
# not prove it *learns*, and Milestone 04 Part 2 shipped for a long time in a
# state where it built fine and trained nothing: the loss was computed in NumPy
# and re-wrapped in a fresh Tensor, and `flatten` did the same to the feature
# map, so no gradient reached the convolutional half of the network. The run
# printed a falling-looking loss and an accuracy, and every number was noise.
#
# This test trains on synthetic data so it needs no CIFAR-10 download.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cifar_cnn_actually_trains():
    """Milestone 04 Part 2: every parameter must receive a gradient."""
    from tinytorch.core.tensor import Tensor
    from tinytorch.core.optimizers import Adam
    from tinytorch.core.losses import CrossEntropyLoss

    script = MILESTONES_DIR / "04_1998_cnn" / "02_lecun_cifar10.py"
    assert script.exists(), f"Required milestone script missing: {script}"
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

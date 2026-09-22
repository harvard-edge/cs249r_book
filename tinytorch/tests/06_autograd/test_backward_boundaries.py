"""Backward edge cases checked against source, without changing generated exports."""

from pathlib import Path
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def source_autograd(monkeypatch):
    """Load independent source classes, restoring imported modules after the test."""
    root = Path(__file__).resolve().parents[2]
    for folder, target in (
        ('01_tensor', 'tensor'), ('02_activations', 'activations'),
        ('03_layers', 'layers'), ('04_losses', 'losses'), ('06_autograd', 'autograd'),
    ):
        source = root / 'src' / folder / f'{folder}.py'
        module = types.ModuleType(f'tinytorch.core.{target}')
        module.__file__ = str(source)
        monkeypatch.setitem(sys.modules, module.__name__, module)
        exec(compile(source.read_text(encoding="utf-8"), str(source), 'exec'), module.__dict__)
    return module


@pytest.mark.parametrize('shape', [(2, 2, 2), (2, 3, 4)])
def test_negative_permute_axes_preserve_gradient_values(source_autograd, shape):
    x = source_autograd.Tensor(np.arange(np.prod(shape)).reshape(shape), requires_grad=True)
    output = x.permute(1, -1, 0)
    upstream = np.arange(output.data.size, dtype=np.float32).reshape(output.shape)
    output.backward(upstream)
    np.testing.assert_array_equal(x.grad, np.transpose(upstream, (2, 0, 1)))


@pytest.mark.parametrize('shape,axis', [((0, 3), 1), ((2, 0, 3), (0, -1)), ((0, 3), ())])
def test_mean_backward_with_empty_output(source_autograd, shape, axis):
    x = source_autograd.Tensor(np.empty(shape), requires_grad=True)
    output = x.mean(axis=axis)
    output.backward(np.empty(output.shape))
    assert x.grad.shape == shape
    assert x.grad.size == 0


def test_gelu_backward_at_infinite_and_large_finite_inputs(source_autograd):
    limit = np.finfo(np.float32).max
    x = source_autograd.Tensor([-np.inf, -limit, 0, limit, np.inf], requires_grad=True)
    with np.errstate(over='raise', invalid='raise'):
        output = source_autograd.GELUFunction.apply(x)
        output.backward(np.ones(5, dtype=np.float32))
    np.testing.assert_array_equal(x.grad, [0, 0, 0.5, 1, 1])


def test_bce_backward_matches_flat_clipped_forward(source_autograd):
    tensor = source_autograd.Tensor
    operation = source_autograd.BinaryCrossEntropyFunction
    targets = tensor([1.0])
    p = tensor([5e-8], requires_grad=True)
    operation.apply(p, targets).backward()
    step = 1e-8
    numeric = (operation.apply(tensor([5e-8 + step]), targets).data
               - operation.apply(tensor([5e-8 - step]), targets).data) / (2 * step)
    np.testing.assert_array_equal(p.grad, [0])
    np.testing.assert_array_equal(p.grad, numeric.reshape(1))
    # Upper clipping plateau and exact endpoints must also have zero slope.
    p = tensor([0.0, 1.0], requires_grad=True)
    operation.apply(p, tensor([1.0, 0.0])).backward()
    np.testing.assert_array_equal(p.grad, [0, 0])


def test_bce_backward_inside_clip_interval_keeps_chain_rule(source_autograd):
    tensor = source_autograd.Tensor
    p = tensor([0.25, 0.75], requires_grad=True)
    source_autograd.BinaryCrossEntropyFunction.apply(p, tensor([1., 0.])).backward(np.array(3.))
    np.testing.assert_allclose(p.grad, [-6., 6.])

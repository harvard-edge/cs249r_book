"""
Property-based fuzz tests for Tensor construction and reshape.

Hand-written tests check specific, chosen inputs. Fuzz testing generates
many randomized inputs (shapes, values, edge cases like empty arrays or
single elements) to find the input nobody thought to write a test for.
This complements, rather than replaces, the hand-written and mutation
tests elsewhere in this suite: SQLite's own testing philosophy notes
fuzzing and high branch coverage catch different classes of bug.

Requires the optional `fuzz` dependency group (`pip install -e .[fuzz]`);
skips cleanly if hypothesis isn't installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis", reason="fuzz tests need the optional 'fuzz' dependency group")
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

from tinytorch.core.tensor import Tensor


# Small shapes keep each example fast; hypothesis compensates with example count.
shape_strategy = array_shapes(min_dims=0, max_dims=4, min_side=0, max_side=6)
float_values = st.floats(
    allow_nan=False, allow_infinity=False, width=32,
    min_value=-1e6, max_value=1e6,
)


@given(arrays(dtype=np.float32, shape=shape_strategy, elements=float_values))
@settings(max_examples=200)
def test_construction_preserves_shape_size_and_values(data):
    """Tensor(x) must report the same shape/size as the input array and
    must not silently alter the values (beyond the documented float32 cast)."""
    t = Tensor(data)

    assert t.shape == data.shape
    assert t.size == data.size
    np.testing.assert_array_equal(t.data, data.astype(np.float32))


@given(
    arrays(dtype=np.float32, shape=shape_strategy, elements=float_values),
    st.data(),
)
@settings(max_examples=200)
def test_reshape_to_any_valid_shape_preserves_total_elements_and_values(data_array, data_strategy):
    """Any target shape whose product equals the tensor's size must
    succeed and preserve the flattened element order."""
    t = Tensor(data_array)
    size = t.size

    if size == 0:
        # Degenerate: any shape containing a 0 dimension is valid for an
        # empty tensor; skip generating a matching factorization for it.
        return

    # Build a random factorization of `size` into 1-3 dimensions by drawing
    # divisors, so the product always equals size exactly.
    ndims = data_strategy.draw(st.integers(min_value=1, max_value=3))
    dims = []
    remaining = size
    for i in range(ndims - 1):
        divisors = [d for d in range(1, remaining + 1) if remaining % d == 0]
        d = data_strategy.draw(st.sampled_from(divisors))
        dims.append(d)
        remaining //= d
    dims.append(remaining)

    reshaped = t.reshape(*dims)

    assert reshaped.shape == tuple(dims)
    assert reshaped.size == size
    np.testing.assert_array_equal(reshaped.data.flatten(), t.data.flatten())


@given(arrays(dtype=np.float32, shape=shape_strategy, elements=float_values), st.data())
@settings(max_examples=100)
def test_reshape_negative_one_infers_matching_dimension(data_array, data_strategy):
    """reshape(..., -1) must infer a dimension that makes the total
    element count match, for any valid partial shape."""
    t = Tensor(data_array)
    size = t.size

    if size == 0:
        return

    divisors = [d for d in range(1, size + 1) if size % d == 0]
    known_dim = data_strategy.draw(st.sampled_from(divisors))
    inferred_dim = size // known_dim

    reshaped = t.reshape(known_dim, -1)

    assert reshaped.shape == (known_dim, inferred_dim)
    np.testing.assert_array_equal(reshaped.data.flatten(), t.data.flatten())


@given(
    arrays(dtype=np.float32, shape=array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=6),
           elements=float_values),
)
@settings(max_examples=100)
def test_reshape_to_flat_then_back_is_identity(data_array):
    """Reshaping to 1D and back to the original shape must round-trip
    exactly, for any non-empty shape."""
    t = Tensor(data_array)
    original_shape = t.shape

    flat = t.reshape(-1)
    restored = flat.reshape(*original_shape)

    assert restored.shape == original_shape
    np.testing.assert_array_equal(restored.data, t.data)

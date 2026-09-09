#!/usr/bin/env python3
"""
Integration tests for Module 17: Acceleration.

Every acceleration technique in this module is an optimisation, which means its
one non-negotiable property is that it does not change the answer. These tests
pin each fast path to a reference computation.
"""

import numpy as np

from tinytorch.core.tensor import Tensor
from tinytorch.perf.acceleration import (
    vectorized_matmul,
    tiled_matmul,
    fused_gelu,
    unfused_gelu,
)


def test_vectorized_matmul_matches_the_reference_product():
    rng = np.random.default_rng(0)
    a_data = rng.standard_normal((17, 23)).astype(np.float32)
    b_data = rng.standard_normal((23, 11)).astype(np.float32)

    actual = vectorized_matmul(Tensor(a_data), Tensor(b_data))
    expected = a_data @ b_data

    assert actual.data.shape == expected.shape, (
        f"Shape mismatch: {actual.data.shape} vs {expected.shape}"
    )
    np.testing.assert_allclose(actual.data, expected, rtol=1e-5, atol=1e-5)


def test_tiled_matmul_agrees_at_every_tile_size():
    """Tiling changes the summation order, not the result."""
    rng = np.random.default_rng(1)
    a_data = rng.standard_normal((40, 40)).astype(np.float32)
    b_data = rng.standard_normal((40, 40)).astype(np.float32)
    expected = a_data @ b_data

    for tile_size in (8, 16, 32, 64):
        actual = tiled_matmul(Tensor(a_data), Tensor(b_data), tile_size=tile_size)
        assert actual.data.shape == expected.shape, (
            f"tile_size={tile_size} changed the output shape"
        )
        # float32 reassociation tolerance, scaled by the reduction depth.
        np.testing.assert_allclose(
            actual.data, expected, rtol=1e-4, atol=1e-4,
            err_msg=f"tiled_matmul disagreed with the reference at tile_size={tile_size}",
        )


def test_tiled_matmul_handles_a_partial_final_tile():
    """A size that is not a multiple of the tile is where tiling bugs hide."""
    rng = np.random.default_rng(2)
    a_data = rng.standard_normal((13, 7)).astype(np.float32)
    b_data = rng.standard_normal((7, 5)).astype(np.float32)

    actual = tiled_matmul(Tensor(a_data), Tensor(b_data), tile_size=4)
    np.testing.assert_allclose(actual.data, a_data @ b_data, rtol=1e-4, atol=1e-4)


def test_fusion_does_not_change_the_activation():
    """The fused and unfused GELU must be the same function."""
    rng = np.random.default_rng(3)
    x = Tensor(rng.standard_normal((64, 32)).astype(np.float32) * 2.0)

    fused = fused_gelu(x)
    unfused = unfused_gelu(x)

    assert fused.data.shape == x.data.shape, "fused_gelu changed the shape"
    np.testing.assert_allclose(
        fused.data, unfused.data, rtol=1e-5, atol=1e-5,
        err_msg="fused_gelu and unfused_gelu computed different functions",
    )

    # GELU is not the identity and not ReLU; a stub returning either would
    # otherwise pass a shape-only check.
    assert not np.allclose(fused.data, x.data), "fused_gelu returned its input unchanged"
    assert not np.allclose(fused.data, np.maximum(x.data, 0)), (
        "fused_gelu returned ReLU, not GELU"
    )
    # GELU sends small negatives slightly negative, unlike ReLU which zeroes them.
    assert float(fused_gelu(Tensor(np.array([-0.5], dtype=np.float32))).data[0]) < 0.0


if __name__ == "__main__":
    test_vectorized_matmul_matches_the_reference_product()
    test_tiled_matmul_agrees_at_every_tile_size()
    test_tiled_matmul_handles_a_partial_final_tile()
    test_fusion_does_not_change_the_activation()
    print("✅ Acceleration integration tests passed")

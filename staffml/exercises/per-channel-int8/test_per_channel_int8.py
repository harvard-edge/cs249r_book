"""Tests for the per-channel INT8 exercise (see README.md)."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _impl import load  # noqa: E402

impl = load(Path(__file__).resolve().parent)

N, IN, OUT = 16, 256, 64


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    # Rows with very different magnitudes, as in real weight matrices.
    w = rng.standard_normal((OUT, IN)) * np.geomspace(0.01, 1.0, OUT)[:, None]
    x = rng.standard_normal((N, IN))
    return x, w


def test_shapes_and_dtypes(data):
    x, w = data
    w_q, w_s = impl.quantize_per_channel(w)
    x_q, x_s = impl.quantize_per_tensor(x)
    assert w_q.dtype == np.int8 and w_q.shape == (OUT, IN)
    assert w_s.shape == (OUT,)                       # one scale per output channel
    assert x_q.dtype == np.int8 and np.ndim(x_s) == 0
    assert np.abs(w_q.astype(int)).max() <= 127 and np.abs(x_q.astype(int)).max() <= 127
    assert impl.int8_matmul_int32(x_q, w_q).dtype == np.int32


def test_each_channel_uses_its_full_range(data):
    _, w = data
    w_q, _ = impl.quantize_per_channel(w)
    np.testing.assert_array_equal(np.abs(w_q.astype(int)).max(axis=1), 127)


def test_round_trip_error_is_at_most_half_a_step(data):
    _, w = data
    w_q, w_s = impl.quantize_per_channel(w)
    err = np.abs(impl.dequantize_per_channel(w_q, w_s) - w)
    assert np.all(err <= w_s[:, None] / 2 + 1e-12)


def test_zero_channel():
    w = np.zeros((2, 4))
    w[1] = [1.0, -2.0, 0.5, 0.0]
    w_q, w_s = impl.quantize_per_channel(w)
    assert np.all(w_q[0] == 0) and np.all(np.isfinite(w_s))


def test_matmul_error_is_within_the_quantization_bound(data):
    # With |dw| <= s_w / 2 and |dx| <= s_x / 2 element-wise, each output's error is
    # at most sum_k (|x_k| s_w / 2 + |w_k| s_x / 2 + s_x s_w / 4).
    x, w = data
    w_q, w_s = impl.quantize_per_channel(w)
    x_q, x_s = impl.quantize_per_tensor(x)
    y = impl.int8_linear(x_q, x_s, w_q, w_s)
    err = np.abs(y - x @ w.T)
    bound = (np.abs(x).sum(axis=1)[:, None] * w_s[None, :] / 2
             + x_s / 2 * np.abs(w).sum(axis=1)[None, :]
             + IN * x_s * w_s[None, :] / 4)
    assert np.all(err <= bound + 1e-9)
    rel = np.linalg.norm(y - x @ w.T) / np.linalg.norm(x @ w.T)
    assert rel < 0.02


def test_per_channel_beats_per_tensor_on_small_rows(data):
    # One scale for the whole matrix wastes most of the INT8 range on the small rows.
    x, w = data
    w_q, w_s = impl.quantize_per_channel(w)
    t_q, t_s = impl.quantize_per_tensor(w)
    small = slice(0, OUT // 4)
    err_channel = np.abs(impl.dequantize_per_channel(w_q, w_s) - w)[small].mean()
    err_tensor = np.abs(t_q.astype(np.float64) * t_s - w)[small].mean()
    assert err_channel < err_tensor / 10


def test_int32_accumulator_does_not_overflow():
    # Worst case: K products of 127 * 127 each. INT32 holds this for K up to ~133k.
    k = 4096
    ones = np.full((1, k), 127, dtype=np.int8)
    acc = impl.int8_matmul_int32(ones, ones)
    assert acc[0, 0] == k * 127 * 127


def test_weight_bytes_7b():
    # 7B parameters: 28 GB in FP32, 7 GB in INT8; per-channel FP16 scales add little.
    assert impl.weight_bytes(7_000_000_000, 32) == 28_000_000_000
    assert impl.weight_bytes(7_000_000_000, 8) == 7_000_000_000
    assert impl.weight_bytes(4096 * 4096, 8, n_channels=4096) == 4096 * 4096 + 4096 * 2

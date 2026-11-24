"""Per-channel INT8 weights: reference solution.

Symmetric quantization: q = round(w / scale), clipped to [-127, 127].  Weights get
one scale per output channel (row); activations get one scale per tensor.  The
matmul multiplies INT8 values, accumulates in INT32, and rescales once at the end.
"""
import numpy as np

QMAX = 127


def _scale(max_abs):
    # An all-zero channel quantizes to zeros with any scale; use 1 to avoid 0 / 0.
    return np.where(max_abs > 0, max_abs / QMAX, 1.0)


def quantize_per_channel(w: np.ndarray):
    """``w`` (out, in) -> (int8 values (out, in), float scales (out,))."""
    scale = _scale(np.abs(w).max(axis=1))
    q = np.clip(np.rint(w / scale[:, None]), -QMAX, QMAX).astype(np.int8)
    return q, scale


def quantize_per_tensor(x: np.ndarray):
    """``x`` -> (int8 values, one float scale)."""
    scale = float(_scale(np.abs(x).max()))
    q = np.clip(np.rint(x / scale), -QMAX, QMAX).astype(np.int8)
    return q, scale


def dequantize_per_channel(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float64) * scale[:, None]


def int8_matmul_int32(x_q: np.ndarray, w_q: np.ndarray) -> np.ndarray:
    """``x_q`` (n, in) times ``w_q`` (out, in) transposed, accumulated in INT32."""
    return x_q.astype(np.int32) @ w_q.astype(np.int32).T


def int8_linear(x_q, x_scale, w_q, w_scale) -> np.ndarray:
    """Quantized ``x @ w.T``: INT32 accumulation, then one rescale per output channel."""
    return int8_matmul_int32(x_q, w_q) * (x_scale * w_scale)[None, :]


def weight_bytes(n_params: int, bits: int, n_channels: int = 0, scale_bytes: int = 2) -> int:
    """Bytes for ``n_params`` weights at ``bits`` each, plus one scale per channel."""
    return n_params * bits // 8 + n_channels * scale_bytes

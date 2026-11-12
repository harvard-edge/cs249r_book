"""Per-channel INT8 weights: starter.

Implement the TODOs, then run

    STAFFML_IMPL=starter pytest staffml/exercises/per-channel-int8

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
    # TODO: one scale per row, max|w_row| / QMAX (use _scale); round, clip, cast to int8.
    raise NotImplementedError


def quantize_per_tensor(x: np.ndarray):
    """``x`` -> (int8 values, one float scale)."""
    # TODO: one scale for the whole tensor; round, clip, cast to int8.
    raise NotImplementedError


def dequantize_per_channel(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    # TODO: multiply each row by its scale.
    raise NotImplementedError


def int8_matmul_int32(x_q: np.ndarray, w_q: np.ndarray) -> np.ndarray:
    """``x_q`` (n, in) times ``w_q`` (out, in) transposed, accumulated in INT32."""
    # TODO: multiply as int32 so the products and their sums cannot overflow int8.
    raise NotImplementedError


def int8_linear(x_q, x_scale, w_q, w_scale) -> np.ndarray:
    """Quantized ``x @ w.T``: INT32 accumulation, then one rescale per output channel."""
    # TODO: INT32 accumulate, then rescale by x_scale * w_scale per output channel.
    raise NotImplementedError


def weight_bytes(n_params: int, bits: int, n_channels: int = 0, scale_bytes: int = 2) -> int:
    """Bytes for ``n_params`` weights at ``bits`` each, plus one scale per channel."""
    # TODO: the napkin-math formula.
    raise NotImplementedError
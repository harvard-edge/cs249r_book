# Per-channel INT8 weights

**Task.** Quantize a weight matrix to INT8 with one scale per output channel, an
activation tensor with one scale, and compute the layer with an INT8 × INT8
matmul that accumulates in INT32 and rescales once per output channel. Implement
the `TODO`s in `starter.py`.

Symmetric quantization: `scale = max|w| / 127`, `q = clip(round(w / scale), -127, 127)`.

**What the tests check.**
- Dtypes and shapes: INT8 values, one scale per output channel, an INT32 accumulator.
- Every channel uses its full range, and the round-trip error is at most half a step.
- The matmul error stays within the analytic bound
  `Σ_k (|x_k| s_w/2 + |w_k| s_x/2 + s_x s_w/4)`, and under 2% overall.
- On rows whose magnitudes differ by 100×, per-channel scales are over 10× more
  accurate than a single per-tensor scale for the small rows.
- 4,096 products of 127 × 127 accumulate without overflow.

## Napkin math

- **Bytes.** INT8 weights are a quarter of FP32 and half of FP16: 7B parameters
  take 28 GB in FP32 and 7 GB in INT8. A 4096 × 4096 layer adds 4,096 FP16
  scales (8 KB) to 16 MB of weights, about 0.05%. At batch 1, decode is
  bandwidth-bound, so halving the bytes from FP16 roughly doubles tokens/s.
- **Why per channel.** One scale for the whole matrix is set by its largest row,
  so a row 100× smaller only reaches about ±1 of the ±127 levels and rounds mostly to 0.
  A per-channel scale gives every row all 255 levels.
- **Why INT32.** One product is up to 127² = 16,129, so a dot product over K
  inputs needs about 14 + log2(K) bits: INT16 overflows at the third product, while
  INT32 is safe up to K ≈ 133,000.

# Math Audit: vol1/backmatter/appendix_algorithm.qmd

Source audited: `book/quarto/contents/vol1/backmatter/appendix_algorithm.qmd`

## Summary

The appendix is mostly mathematically coherent: matrix multiplication notation, the square-GEMM intensity derivation, the broadcasting example, the backpropagation equations under a row-vector convention, the GPT-2 training-memory arithmetic, and the idealized fusion traffic reduction are largely correct. The main issues are prose-equation consistency and quantitative thresholds: a setup comment is off by large factors, the complexity table mixes bias conventions, reverse-mode AD is described as closer to two inference passes than the appendix's own later accounting supports, an activation-memory formula labels elements as bytes, and the sparse-memory threshold in the summary confuses memory savings with performance usefulness.

No source `.qmd` files were modified.

## Findings

### 1. Setup comments state incorrect sparse-memory, ridge-point, and small-GEMM efficiency values

- Lines 45-48 describe the examples as showing `~4 GB dense vs. ~8 MB sparse at 1% density`, a ridge point of `~208`, and a `64x64` GEMM at `~1% efficiency`.
- Lines 84-98 compute the actual values used for rendering.
- Severity: Low.
- Issue: Direct arithmetic gives `100,000 * 10,000 = 1e9` elements. Dense FP32 storage is `1e9 * 4 = 4e9` bytes, or `4 GB`. At 1 percent density, there are `1e7` nonzeros, and storing an FP32 value plus one int32 column index uses `1e7 * (4 + 4) = 80e6` bytes, or `80 MB`, before CSR row-pointer overhead. The A100 ridge point from local constants is `312 TFLOP/s / 2.039 TB/s = 153 FLOP/byte`, not 208. A `64x64` FP16 GEMM has intensity `64/3 = 21.3 FLOP/byte`; relative to a 153 FLOP/byte ridge point, the roofline fraction is about `13.9%`, not 1 percent.
- Proposed correction: Update the comments to `~4 GB dense vs. ~80 MB sparse at 1% density; ridge point ~153; small 64x64 GEMM at ~14% roofline efficiency` or remove the stale expected-output values.

### 2. CSR storage complexity omits row-pointer storage

- Line 183 says CSR uses `O(K)` storage instead of `O(N)` for a matrix with `N` elements and `K` nonzeros.
- Severity: Low.
- Issue: CSR stores values and column indices for the `K` nonzeros, plus a row-pointer array of length `num_rows + 1`. The exact asymptotic storage is `O(K + R)` for `R` rows, not just `O(K)`. In the later example, the omitted row-pointer term is small (`100,001 * 4 bytes`, about `0.4 MB`) compared with the `80 MB` value/index arrays, but the general statement is incomplete.
- Proposed correction: Change the sentence to: `CSR uses O(K + R) storage, where R is the number of rows, instead of O(N); when K is large relative to R this is often summarized as O(K).`

### 3. Conv2D parameter count uses a different bias convention than Linear

- Lines 205-210 define the computational complexity table.
- Line 207 includes Linear bias parameters as `(N_in + 1) * N_out`.
- Line 208 gives Conv2D parameters as `K^2 * C_in * C_out`, omitting the optional `C_out` bias terms.
- Severity: Medium.
- Issue: The table's `Parameters (P)` column is inconsistent across rows. If biases are included for Linear, a bias-enabled Conv2D has `(K^2 * C_in + 1) * C_out` parameters. If the table intentionally omits Conv2D biases because many modern convolution blocks disable bias before normalization, it should say so; otherwise readers may apply different conventions when estimating model size.
- Proposed correction: Either change Conv2D parameters to `(K^2 * C_in + 1) * C_out` and clarify that FLOPs are dominated by the kernel multiplications, or add a note that Conv2D bias terms are omitted while Linear includes bias.

### 4. Reverse-mode AD compute cost is understated relative to the later backprop accounting

- Line 256 says training, meaning one forward plus one backward pass, has compute cost similar to two inference passes.
- Lines 299-311 later explain that each dense layer's backward pass performs two matrix multiplications versus one in the forward pass.
- Line 262 also says training cost is roughly `2--3x` inference.
- Severity: Medium.
- Issue: For a dense layer `y = h W`, the forward pass computes one GEMM. Backward computes `dW = h^T dy` and `dh = dy W^T`, two GEMMs of comparable scale. Thus forward plus backward is closer to three forward-pass GEMMs for that layer, before loss, activation, and optimizer overheads. The important reverse-mode point is one backward traversal for all parameters, not that the total training step is about two inference passes.
- Proposed correction: Replace the end of line 256 with: `This is why training has the cost of a small constant multiple of inference, often about 2--3x for dense neural networks, rather than N passes.`

### 5. Activation-memory formula labels element counts as bytes

- Line 426 says per-layer transformer activation memory is approximately `12 * B * S * d` bytes in BF16.
- Line 428 correctly multiplies `layers * 12 * B * S * d * bf16 bytes`.
- Severity: Medium.
- Issue: `12 * B * S * d` is an approximate count of retained BF16 elements, not bytes. The byte count is `12 * B * S * d * bytes_per_element`, which is twice as large for BF16/FP16. The rendered worked example uses the correct multiplication, so this is a prose-equation mismatch rather than a numerical-output error.
- Proposed correction: Change line 426 to: `Per-layer activation storage is approximately 12 * B * S * d BF16 elements, or 12 * B * S * d * bytes_per_element bytes.`

### 6. Accelerator capacity mixes GiB and decimal GB without clear labeling

- Line 365 converts `A100_MEM_CAPACITY = 80 GiB` to `GB`.
- Lines 402-404 format the decimal-GB values without rounding.
- Lines 422 and 430 render the capacity as `GB`, while line 390's guard message says `80 GB`.
- Severity: Low.
- Issue: `80 GiB = 85.9 GB` in decimal units. The arithmetic is valid, but the prose and guard message mix the marketed `80 GB`/`80 GiB` convention with a decimal conversion. This can render as an awkward `85.899... GB` capacity and makes the remaining-memory number look inconsistent with the common "80 GB A100" label.
- Proposed correction: Keep the calculation in GiB and render `80 GiB`, or round decimal GB consistently to `86 GB` and change the guard message accordingly.

### 7. Gradient-checkpointing asymptotic memory claim is too general

- Line 430 says gradient checkpointing trades about 33 percent more compute for `O(sqrt(L))` activation memory.
- Line 438 says checkpointing reduces memory from `O(N)` to `O(sqrt(N))` at the cost of about 33 percent more compute.
- Severity: Low.
- Issue: The `O(sqrt(L))` memory result applies to a sequential chain with an appropriate checkpoint schedule and uniform-ish layer costs. Activation checkpointing in practice can also use fixed intervals, selective recomputation, or graph-specific policies, with different memory/compute trade-offs. The appendix states the sequential-chain result as if it were universal.
- Proposed correction: Qualify the statement: `For an L-layer sequential network, optimal checkpoint schedules can reduce activation memory from O(L) to O(sqrt(L)) with extra recomputation; common schedules often add a modest constant-factor compute overhead.`

### 8. Summary overstates the sparsity required for memory savings

- Line 470 says sparsity usually needs to exceed `90-95 percent` to be worthwhile for performance.
- Line 485 says sparse formats only reduce memory when sparsity exceeds roughly `90-95 percent`.
- Severity: Medium.
- Issue: The performance statement on line 470 is plausible as a rule of thumb, but the memory statement on line 485 is too strong. With FP32 values and one int32 index per nonzero, sparse storage is roughly `density * N * (4 + 4)` bytes, ignoring row pointers. Dense storage is `4N` bytes, so memory starts improving when `density < 0.5`, i.e. sparsity exceeds about 50 percent, plus row-pointer overhead. Very high sparsity may be needed for speedups or large net wins, but not merely to reduce bytes.
- Proposed correction: Change the summary bullet to: `Sparse storage starts reducing memory only after metadata overhead is outweighed (about >50% zeros for FP32 values plus int32 indices in the simplest case, higher with row/index overhead), while performance wins often require much higher sparsity such as 90-95% or structured hardware-supported patterns.`

## Checked But No Issue

- Lines 137-143: Matrix multiplication in Einstein notation is correct: `C_ij = sum_k A_ik B_kj`, corresponding to `ik,kj->ij`.
- Lines 153-155: The dot-product identity `a · b = |a||b|cos(theta)` and the attention use of query-key dot products are mathematically standard.
- Lines 159-165: GEMM FLOPs and the square FP16 intensity formula are internally correct under the simplified one-pass traffic model: `2n^3 / (3n^2 * 2 bytes) = n/3 FLOP/byte`.
- Lines 185-189: The rendered sparse example arithmetic is correct apart from omitted row-pointer overhead: `1e9` FP32 elements are `4 GB`; 1 percent density gives `10 million` nonzeros; value plus column index storage is about `80 MB`; the dense-to-sparse ratio is `50x`.
- Lines 225-236: Broadcasting rules and the example `(32, 1, 64)` with `(1, 128, 64)` yielding `(32, 128, 64)` are correct.
- Lines 250-256: The chain-rule/reverse-mode contrast is directionally correct: for a scalar loss with many parameters, reverse mode computes all parameter gradients in one reverse traversal, while forward-mode or finite-difference-style per-parameter seeding would scale with the number of parameters.
- Lines 299-309: The backpropagation equations are shape-consistent for row-vector activations: if `y = h W2`, then `dW2 = h^T dy` and `dh = dy W2^T`; similarly `dW1 = x^T dh`.
- Lines 317-324: The training-memory decomposition correctly includes weights, gradients, optimizer state, and activations.
- Lines 360-386 and 419-430: The GPT-2 XL memory arithmetic is internally consistent using decimal GB: weights `3.0 GB`, gradients `3.0 GB`, Adam master/moments `18.0 GB`, model state `24 GB`, batch-8 activations about `15 GB`, total about `39 GB`, and batch-64 activations about `121 GB`.
- Lines 458-462: The idealized fusion traffic reduction is correct for `k` equal-sized elementwise operations: unfused read/write traffic is `2kN` bytes and fused traffic is `2N` bytes.

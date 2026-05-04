# Math Audit: vol1/optimizations/model_compression.qmd

Source audited: `book/quarto/contents/vol1/optimizations/model_compression.qmd`

## Summary

The chapter's core compression formulas are mostly directionally correct: basic FP16/INT8/INT4 storage ratios, pruning masks, distillation temperature softmax, sparsity definitions, depthwise-separable convolution complexity, and Amdahl's Law arithmetic are largely coherent. The main issues are in quantitative examples and prose-equation consistency: several speedup/memory examples overstate the consequence of the arithmetic, one NAS cost conversion confuses GPU-hours with GPU-days, the low-rank section reverses the compute effect for rank-128 factorization, and a few tables/footnotes attach incorrect ratios or storage implications to the stated quantities.

No source `.qmd` files were modified.

## Findings

### 1. NAS search cost confuses GPU-hours and GPU-days

- Line 181 says the NAS example cost 22,400 GPU-hours and parenthetically gives "800 GPUs for twenty-eight days."
- Severity: High.
- Issue: `800 GPUs * 28 days * 24 hours/day = 537,600 GPU-hours`. The value 22,400 is GPU-days, not GPU-hours.
- Proposed correction: Change the cost to `537,600 GPU-hours` or change the unit to `22,400 GPU-days`.

### 2. Bandwidth-bound INT8 speedup is stated as 4x after comparing FP16 to INT8

- Lines 3501-3503 compute A100 FP16-vs-INT8 throughput and set `bandwidth_bound_speedup = 4`.
- Lines 3525-3529 discuss switching from FP16 to INT8, then say the bandwidth-bound case achieves up to 4x because halving data size nearly doubles effective throughput.
- Severity: High.
- Issue: For weight bandwidth alone, FP16 to INT8 halves bytes per weight, so the bandwidth-bound speedup is about 2x, not 4x. A 4x bandwidth-bound speedup applies to FP16 to INT4, as in the earlier 7B LLM napkin calculation, or FP32 to INT8.
- Proposed correction: Either set `bandwidth_bound_speedup = 2` for the FP16-to-INT8 callout, or rewrite the callout to compare FP16 with INT4 / FP32 with INT8.

### 3. Low-rank factorization section says FLOPs increase when the shown rank reduces them

- Lines 2253-2267 compute a `4096 x 4096` matrix factored with rank `128`, giving `16x` fewer stored parameters.
- Line 2297 says FLOPs "actually increases slightly" because two smaller matrix multiplies replace one large one.
- Line 2303 says the factorization adds a computation of runtime `O(mkn)`.
- Severity: High.
- Issue: Applying the original matrix to a vector costs `4096^2 = 16,777,216` multiply-scale operations. Applying the factored form costs `2 * 4096 * 128 = 1,048,576`, a 16x reduction, not an increase. More generally, dense application drops from `O(mn)` to `O(k(m+n))` for a vector input when `k` is small. The `O(mkn)` term is the cost to explicitly form `UV`, which should not be done during inference.
- Proposed correction: Replace the claim with: `For inference using the factors directly, both storage and matrix-vector compute fall from O(mn) to O(k(m+n)); only explicitly materializing UV would add O(mkn) work and defeat the purpose.`

### 4. TF32 table assigns a storage reduction it does not provide

- Lines 3537-3546 list numerical formats.
- Line 3542 lists TF32 as 19-bit with storage reduction "Similar to FP16."
- Severity: Medium.
- Issue: TF32 is primarily an internal Tensor Core compute format for FP32 inputs; tensors are generally stored as FP32, so it does not provide a 2x model-storage reduction like FP16. The 19-bit arithmetic format improves compute throughput, not parameter storage.
- Proposed correction: Change TF32 storage reduction to `None / stored as FP32` or `No model-size reduction`, while keeping the Ampere compute-speed note.

### 5. MobileNet pruning example rounds below the stated 2 MB device limit

- Lines 613-615 set `85%` pruning, `14MB` original size, and `2MB` pruned size.
- Line 626 says removing 85% of weights produces a model that fits in 2 MB.
- Severity: Medium.
- Issue: Direct proportional pruning gives `14 MB * (1 - 0.85) = 2.1 MB`, which exceeds a strict 2 MB flash limit before sparse-format metadata or alignment overhead. The rendered `2MB` is a rounded value that contradicts the hard device limit.
- Proposed correction: Use an original size of about `13.3 MB`, a pruning level of at least `86%`, or state that the result is `about 2.1 MB` and requires a slightly larger flash budget.

### 6. DistilBERT memory numbers do not scale with the stated parameter reduction

- Line 2197 says DistilBERT has 40 percent fewer parameters than BERT-Base.
- Line 2199 gives `66M vs. 110M` parameters, then says memory drops from `1.35 GB` to `0.54 GB`.
- Severity: Medium.
- Issue: `66M / 110M = 0.60`, so parameter memory should be about 60 percent of BERT-Base, not 40 percent. If `1.35 GB` is the baseline memory, proportional parameter scaling gives about `0.81 GB`, not `0.54 GB`. The stated `0.54 GB` is a 60 percent reduction, inconsistent with 40 percent fewer parameters unless additional assumptions are introduced.
- Proposed correction: Either change the student memory to approximately `0.81 GB`, or explicitly say the `0.54 GB` includes additional runtime optimizations beyond the 66M-vs-110M parameter reduction.

### 7. Low-rank "97 percent information content" confuses parameter-count reduction with information retained

- Line 2220 says a `4096 x 4096` matrix with effective rank 128 means 97 percent of its information content can be captured by a smaller pair of matrices.
- Severity: Medium.
- Issue: The arithmetic supports a parameter-count reduction, not a general information-retention percentage. Rank 128 has `2 * 4096 * 128 = 1,048,576` parameters versus `16,777,216` in the full matrix, so it uses `6.25%` as many parameters, a `93.75%` storage reduction. The retained information depends on the singular value spectrum, not just rank.
- Proposed correction: Replace with: `...may be approximated with only 6.25 percent as many parameters; the fraction of information retained depends on the singular-value spectrum.`

### 8. Generic N:M sparsity definition hard-codes the 2:4 zero count

- Line 6348 defines `$N$:$M$` sparsity as exactly `N` nonzeros in every `M` consecutive elements, "and the other two are zero."
- Severity: Medium.
- Issue: For generic `N:M`, the number of zeros is `M - N`, not necessarily two. "The other two" is only correct for the special case `2:4`.
- Proposed correction: Change to: `exactly N are nonzero and the remaining M - N are zero; for 2:4 sparsity, the remaining two are zero.`

### 9. BERT pruning-stage percentages do not imply 75 percent parameter reduction

- Line 7093 says stage one removes 30 percent of attention heads and 40 percent of intermediate FFN dimensions, resulting in 75 percent parameter reduction.
- Severity: Medium.
- Issue: Those structural reductions do not directly imply a 75 percent total parameter reduction. Even in a simplified transformer block, reducing attention heads by 30 percent and FFN intermediate width by 40 percent would usually leave a substantial fraction of the attention and FFN weights, and embeddings are not reduced by either operation. The later `440 MB to 28 MB` and `16x` combined reduction assumes a 4x pruning/distillation reduction before INT8, but the stated per-structure pruning percentages do not justify it.
- Proposed correction: Either change the stage-one reduction to a value supported by the stated structural cuts, or add an explicit distillation/student-architecture size reduction before claiming the 75 percent parameter reduction.

### 10. Energy table and prose mix DRAM access granularity with per-byte energy

- Lines 101-105 load `ENERGY_DRAM_ACCESS_PJ = 640 pJ` and `ENERGY_DRAM_PJ_PER_BYTE = 160 pJ/byte`.
- Lines 199-202 say fetching a 32-bit float from DRAM costs the 640 pJ access value, while fetching an 8-bit integer costs the 160 pJ per-byte value.
- Lines 205-209 list a 64-bit DRAM read as `40,000x` relative energy.
- Severity: Low.
- Issue: The prose compares a fixed DRAM access value for FP32 with a per-byte value for INT8. If the constants are used consistently, 32-bit is `4 bytes * 160 pJ/byte = 640 pJ`, 8-bit is `1 byte * 160 pJ/byte = 160 pJ`, and a 64-bit read is `8 bytes * 160 = 1280 pJ`. The table's `40,000x` is also not derivable from the local constants: relative to an INT8 add at `0.03 pJ`, a 64-bit DRAM read is about `42,667x`; relative to an 8-bit integer add of roughly `0.03 pJ` it rounds to `~43,000x`.
- Proposed correction: State the DRAM energy consistently as `160 pJ/byte`, then derive `640 pJ` for 32-bit, `160 pJ` for 8-bit, and either change the table to `~43,000x` or label it as an illustrative rounded hardware-era value.

### 11. Quantization-speedup example ignores KV-cache traffic in the throughput calculation

- Lines 319-332 set `kv_cache_gb = 1.0`, compute `fp16_total_gb = fp16_size_gb + kv_cache_gb`, but compute latency only from `fp16_size_gb` and `int4_size_gb`.
- Lines 371-379 present the KV cache as part of total memory, then present per-token bandwidth cost from weights alone.
- Severity: Low.
- Issue: If the 1 GB KV cache must also be read or written in the per-token path, the speedup is not exactly `14 / 3.5 = 4x`. Including the same 1 GB cache traffic on both sides gives `(14 + 1) / (3.5 + 1) = 3.33x`. If the example intentionally models weight streaming only, the prose should say so.
- Proposed correction: Add `for weight-loading traffic only` to the bandwidth step, or include KV-cache traffic in both latency calculations and change the speedup to about `3.3x`.

### 12. SIMD callout overstates "operations per cycle" from register packing alone

- Lines 3318-3321 compute 16 FP32 lanes and 64 INT8 lanes in a 512-bit register.
- Lines 3343-3350 call these "operations per cycle" and conclude a 4x compute-bound speedup solely from vector packing.
- Severity: Low.
- Issue: The lane-count ratio is correct, but operations per cycle also depends on instruction throughput and whether the hardware has comparable INT8 arithmetic instructions for the operation being performed. Register packing gives 4x more elements per vector instruction, not universally 4x more operations per cycle.
- Proposed correction: Replace "operations per cycle" with `elements per vector instruction` and qualify the conclusion: `up to 4x on hardware whose INT8 vector instructions have comparable throughput and whose kernels are compute-bound.`

## Checked But No Issue

- Lines 93-95, 127, 131, and 143: FP16 storage for 7B and 175B parameters is arithmetically correct under decimal GB: `7B * 2 bytes = 14 GB`, `175B * 2 bytes = 350 GB`; INT8 storage for 175B is `175 GB`.
- Lines 315-338 and 370-381: The weight-only FP16-to-INT4 LLM example is internally correct: `7B * 2 = 14 GB`, `7B * 0.5 = 3.5 GB`, and bandwidth-only latency at `50 GB/s` gives `280 ms` vs. `70 ms`, or `4x`.
- Lines 453-482 and 516: The model-vs-device table ratios are consistent with the code constants: `3.5 MiB / 512 KiB = 7x` for INT8 MobileNetV2 vs TinyML, and `500 KiB < 512 KiB` for DS-CNN.
- Lines 663-666: The pruning example keeps exactly four nonzero weights because `abs(weight) >= 0.5` retains `0.8, -0.7, -0.9, -0.6`.
- Lines 1110-1122: The unstructured pruning mask equation and threshold definition are mathematically standard.
- Lines 2182-2190: Temperature-scaled softmax and the `alpha T^2 KL(...)` distillation-loss form are standard and internally consistent.
- Lines 2240-2267: The low-rank storage calculation itself is correct: full FP32 storage is `4096^2 * 4 / 2^20 = 64 MiB`, factored storage is `2 * 4096 * 128 * 4 / 2^20 = 4 MiB`, for `16x` less storage.
- Lines 3241-3262 and 3271-3279: The 8B FP16-to-INT4 storage example is correct: `16 GB` to `4 GB`, a `4x` reduction.
- Lines 3672-3675 and 3689: The INT8-vs-FP32 MAC energy ratio computes correctly from the local constants: `(3.7 + 0.9) / (0.2 + 0.03) = 20.0x`.
- Lines 4882-4891: Compound scaling FLOPs are consistent with EfficientNet-style scaling: convolutional compute scales approximately as `d * w^2 * r^2`.
- Lines 4903-4911: Depthwise-separable convolution complexity is correct: standard convolution is `O(h w C_in C_out k^2)`, while depthwise plus pointwise is `O(h w C_in k^2) + O(h w C_in C_out)`.
- Lines 5094-5102 and 5156-5169: The idealized fusion memory-transfer reduction is correct for equal-sized intermediates: `2NM` unfused traffic falls to `2M`, so Conv-BN-ReLU drops from six transfers to two.
- Lines 5662-5671: The sparsity formulas correctly count exact zeros and epsilon-near-zero entries using an indicator and L0 count.
- Lines 5693-5704: The sparse matrix-vector example is correct: the dense `4 x 4` multiply has 16 possible multiplications, while the shown matrix has six nonzero multiplications.
- Lines 7117-7119: The Amdahl's Law example is correct: if model inference is 20 percent of total latency, eliminating it entirely gives maximum speedup `1 / 0.8 = 1.25x`.
- Lines 7133: The ResNet-50 INT8 measurement arithmetic is internally consistent: `4.2 / 1.3 = 3.23x`, `98 / 25 = 3.92x`, `0.31 / 0.08 = 3.875x`, and the accuracy drop is `0.3` percentage points.

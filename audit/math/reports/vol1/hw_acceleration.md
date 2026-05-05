# Math Audit: vol1/hw_acceleration/hw_acceleration.qmd

Source audited: `book/quarto/contents/vol1/hw_acceleration/hw_acceleration.qmd`

## Summary

The chapter's main analytical framework is sound: Amdahl's Law, the roofline equations, bandwidth-bound throughput, arithmetic intensity, and the basic dense/conv memory traffic formulas are dimensionally coherent in most places. The largest problems are localized but important: the transformer QKV example undercounts FLOPs by 2x relative to the chapter's own FLOP convention, the accelerator economics table mixes precision modes while labeling the column FP16, the small-batch inference pitfall quotes throughput numbers that do not follow from the stated roofline arithmetic, and the carbon ROI example uses daily energy numbers not derived from its own power values.

No source `.qmd` files were modified.

## Findings

### 1. Transformer QKV projection undercounts FLOPs by 2x

- Lines 3482-3491 compute QKV projection FLOPs as `3 * batch * seq * hidden * hidden`.
- Lines 3519-3521 present the result as FLOPs and use it to compute arithmetic intensity.
- Severity: High.
- Issue: Elsewhere in the chapter, dense and convolution examples count a multiply-add as 2 FLOPs (for example, lines 3588 and 3677). Under that convention, the QKV projection should be `2 * 3 * batch * seq * hidden * hidden`, not `3 * batch * seq * hidden * hidden`. The current value is a MAC count, not a FLOP count. This halves both reported FLOPs and reported arithmetic intensity.
- Proposed correction: Change line 3483 to `qkv_flops_value = 2 * 3 * t_batch_value * t_seq_value * t_hidden_value * t_hidden_value`, and update the prose formula on line 3519 to include the leading `2 x`. Alternatively, relabel the value as MACs and keep FLOP terminology out of the QKV arithmetic-intensity calculation.

### 2. Accelerator economics table mixes precision modes under an FP16 heading

- Lines 2287-2293 label the peak column `Peak FLOPS (FP16)`.
- Line 2291 reports H100 TF32 throughput in that FP16 column.
- Line 2293 reports Gaudi 2 INT8 throughput in that FP16 column.
- Lines 2243-2252 compute H100 price/performance from TF32 throughput, while lines 2257-2258 separately format H100 FP16 throughput.
- Severity: High.
- Issue: The table compares V100/A100 FP16, H100 TF32, TPU BF16, and Gaudi INT8 as though they are one price/performance metric. This makes the H100 and Gaudi price/TFLOP entries not comparable to the FP16 column label. Directly, if H100 FP16 peak is used instead of TF32, its dollars per FP16 TFLOP would be about half the TF32-based value.
- Proposed correction: Either rename the column to `Representative Peak Throughput (precision shown)` and explicitly state that price/performance is not apples-to-apples across precisions, or compute every row for a common precision class where possible. For H100, use `h100_tflops_fp16` if the column remains FP16.

### 3. Small-batch inference throughput claims do not follow from the stated roofline math

- Lines 5414-5426 compute a dense-layer batch-1 AI of about 1 FLOP/byte for `M=N=2048`.
- Line 5438 says that at batch 1 an A100 achieves 4 TFLOPS and a T4 achieves 3.5 TFLOPS due to memory bottlenecks.
- Severity: High.
- Issue: With AI approximately 1 FLOP/byte, the roofline memory-bound ceiling is `bandwidth * AI`. An A100 with about 2 TB/s bandwidth gives roughly 2 TFLOPS, not 4 TFLOPS. A T4 with roughly 320 GB/s bandwidth gives roughly 0.32 TFLOPS, not 3.5 TFLOPS. The T4 value is more than 10x above its own memory-bound roofline for the stated AI.
- Proposed correction: Recompute the attainable throughput from the same formula used throughout the roofline section, e.g. `A100 ~= 2 TFLOPS` and `T4 ~= 0.3 TFLOPS` for AI near 1, or change the example to describe measured cached/kernel-specific behavior and no longer present it as a direct roofline prediction from the displayed AI.

### 4. Carbon ROI energy totals are not derived from the stated power values

- Lines 5571-5578 define one CPU server as 100 W, one NPU chip as 5 W, and then set daily fleet energy to 2400 kWh/day and 12 kWh/day.
- Lines 5581-5587 compute the 200x efficiency gap and annual CO2 savings from those hardcoded daily energy values.
- Lines 5614-5623 present the power values and daily energy values in one calculation.
- Severity: High.
- Issue: A 100 W device running continuously for one day consumes `100 W * 24 h / 1000 = 2.4 kWh/day`, not 2400 kWh/day. A 5 W device consumes `0.12 kWh/day`, not 12 kWh/day. The hardcoded energy values imply about 1000 CPU servers and 100 NPU chips, but the example never states those fleet sizes or derives them from the workload. As written, the carbon savings are off by orders of magnitude relative to the stated per-device powers.
- Proposed correction: Either derive fleet sizes from an explicit per-inference compute requirement and the TFLOPS values, or state the assumed fleet counts. For a single CPU server vs. single NPU chip, the annual savings would be `(2.4 - 0.12) * 365 * 0.4 / 1000 ~= 0.33 metric tons CO2/year`, not about 350.

### 5. LayerNorm FLOP prose sums to 7, but the code uses 6

- Lines 3771-3772 set `ln_flops_per = 6`.
- Lines 3802-3804 explain the count as mean `(1 ADD)`, variance `(2 ADD, 1 MUL)`, and normalize `(1 ADD, 1 MUL, 1 DIV)`.
- Severity: Medium.
- Issue: The prose terms sum to `1 + 3 + 3 = 7` operations per element, not 6. The final classification remains memory-bound either way, but the displayed arithmetic is internally inconsistent. Using 7 would raise AI by about 17 percent, from roughly 1.5 to roughly 1.75 FLOP/byte.
- Proposed correction: Either change `ln_flops_per` to 7 or adjust the prose decomposition so it contains six operations.

### 6. Dense-layer weight reuse prose overstates "only once per batch element"

- Lines 3671-3684 analyze a dense layer with batch size 32 and correctly compute AI near 32 FLOP/byte.
- Line 3729 says each weight element is used only once per batch element.
- Severity: Medium.
- Issue: In the analyzed batch-32 GEMM, each weight is reused once for each batch element, i.e. 32 times per layer invocation. That reuse is exactly why the batch-size formula on lines 3851-3859 gives AI approximately equal to `B` when weight traffic dominates. The intended contrast with convolution is spatial reuse: dense weights are reused across batch elements but not across spatial positions.
- Proposed correction: Replace with: `Each dense weight is reused across the batch but lacks the additional spatial reuse of convolutional filters, so small-batch dense layers have much lower arithmetic intensity than convolutions.`

### 7. Hardware mapping formula uses inconsistent symbols for arithmetic intensity

- Line 3972 says a GEMM with arithmetic intensity `AI` runs at `min(R_peak, BW x I)`.
- Severity: Low.
- Issue: The sentence introduces `AI` but then uses `I` in the formula. This is likely the same quantity, but the notation shift is a prose-equation consistency problem in a definition callout.
- Proposed correction: Use one symbol consistently, e.g. `min(R_peak, BW x AI)`, or define `I = AI` before using it.

### 8. Matrix hierarchy example is shape-ambiguous relative to common framework convention

- Lines 1130-1137 show `layer = nn.Linear(256, 512)`, `output = layer(input_batch)`, then `Z = matmul(weights, input)` transforming `[256 x 32]` input to `[512 x 32]` output.
- Severity: Low.
- Issue: The math is valid under a column-batch convention with weights shaped `[512 x 256]` and input shaped `[256 x 32]`. However, common PyTorch `nn.Linear` examples use row-batch input `[32 x 256]` and compute `input @ weight.T -> [32 x 512]`. The chapter later uses both row-style and column-style examples, so this snippet may confuse readers about matrix orientation even though its internal dimensions can be made consistent.
- Proposed correction: Add a short convention note in the code comment, such as `# column-batch convention`, or rewrite as `Z = input_batch @ weights.T` with `[32 x 256] -> [32 x 512]` to match PyTorch's usual layout.

### 9. Memory-energy pitfall uses DRAM energy values inconsistent with the chapter's own constants

- Lines 2365-2370 use 640 pJ for a DRAM read and 5 pJ for an SRAM read.
- Lines 5309-5318 compute a DRAM/SRAM ratio from those constants.
- Line 5331 says DRAM costs 100--200 pJ per access versus 1--10 pJ for on-chip memory.
- Severity: Low.
- Issue: The qualitative claim is correct, but the numeric prose does not match the chapter's own displayed figure and constants. Using the figure values gives `640 / 5 = 128x`; using 100--200 pJ and 1--10 pJ gives a much wider 10--200x range.
- Proposed correction: Align the prose with the figure, e.g. `the example energy hierarchy uses 640 pJ for DRAM versus 5 pJ for SRAM, a 128x gap`, or explicitly state that line 5331 is a different representative range.

## Checked But No Issue

- Lines 93-99: Amdahl's Law is stated correctly; with `p=0.9` and `S -> infinity`, the maximum speedup is `1/(1-p) = 10x`.
- Lines 214-259 and 266-279: The H100 Amdahl examples are arithmetically consistent: `p=0.95, S=500` gives about `19.3x`, and `p=0.80, S=500` gives about `5.0x`.
- Lines 1710-1796: The systolic energy example is internally consistent under its assumptions: vector energy is `(4 * 640) + 1 = 2561 pJ/op`, systolic energy is `(2/128 * 640) + 1 = 11 pJ/op`, and the ratio is about `233x`, matching a rounded `~200x` claim.
- Lines 1858-1886: The tiling example is correct: a `4096 x 4096` operation tiled onto a `128 x 128` array has `(4096/128)^2 = 1024` tiles.
- Lines 2393-2396: The energy hierarchy annotation is correct: `640 pJ / 5 pJ = 128x`.
- Lines 2680-2707: The KWS tensor size is correct: `16,000` FP16 samples require `32,000` bytes, or `31.25 KiB`.
- Lines 3200-3206: The bandwidth taper example is directionally and numerically reasonable: `2000 GB/s / 64 GB/s ~= 31x` for HBM to PCIe, and the broader 30--100x range is plausible when PCIe Gen4 or HDR network bandwidth is used.
- Lines 3539-3543: The required-bandwidth and attainable-throughput equations are dimensionally correct.
- Lines 3576-3629: The Conv2D roofline arithmetic is consistent under ideal input/weight/output traffic: output elements are about `25.7M`, FLOPs are about `59.2 GFLOPs`, memory traffic is about `77.7 MB`, and AI is about `761 FLOP/byte`, which is compute-bound on A100.
- Lines 3671-3725: The dense-layer roofline arithmetic is correct: `2 * 32 * 2048 * 2048 = 268M FLOPs`, memory traffic is about `8.25 MB`, AI is about `32.5 FLOP/byte`, and attainable A100 throughput is about `65 TFLOPS`.
- Lines 3851-3859: The batch-size arithmetic-intensity formula is correct for FP16 dense layers. For `M=N=2048`, batch 1 gives about `1`, batch 32 gives about `32`, and batch 256 gives about `205 FLOP/byte`.
- Lines 3901-3953: The GPT-2 XL batch-1 throughput ceiling is internally consistent: `2 * 1.5B` parameters gives about `3 GFLOPs`, FP16 weights are about `3 GB`, AI is about `1 FLOP/byte`, and A100 utilization is below 1 percent.
- Lines 5477-5529: The feasibility checks are arithmetically coherent: a 7B FP16 model is about 14 GB, a 70B FP16 model is about 140 GB, `140 GB / 1 TB/s = 140 ms`, and a 30 FPS frame budget is about `33 ms`.

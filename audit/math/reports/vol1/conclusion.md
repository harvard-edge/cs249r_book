# Math Audit Report: `book/quarto/contents/vol1/conclusion/conclusion.qmd`

## Checked scope

Audited `book/quarto/contents/vol1/conclusion/conclusion.qmd` for equations, generated numeric examples, unit conversions, complexity/scaling claims, invariant counts, and prose-equation consistency. Direct reasoning only; no Gemini or external verification used. No source `.qmd` files were modified.

Checked items include:

- Lines 50-113 and 565-582: Llama-2-70B/H100 roofline calculation.
- Lines 118, 156, 170-180: MobileNetV2 latency, quantization, and FLOP-reduction claims.
- Lines 184-208, 228-244, 550, 787-797: invariant counts and equations.
- Lines 604-644: P99/mean latency ratio calculation.
- Lines 656-669: deployment-context scaling, speculative decoding, and TinyML unit claims.
- Lines 697-713, 721-745: trillion-parameter, exascale, petabyte, GPU-count, and fleet-scale claims.
- Lines 755-781: fallacy/pitfall percentage, latency, Amdahl's Law, and disparity examples.

## Findings

### 1. The conclusion counts twelve invariants, but the Part IV principles define thirteen total principles

- **Lines:** 40, 158, 184-206, 228-234, 550, 787-792
- **Severity:** Medium
- **Issue:** The conclusion repeatedly says the book has twelve quantitative invariants and the table lists 2 Foundations + 2 Build + 4 Optimize + 4 Deploy = 12. However, `book/quarto/contents/vol1/parts/deploy_principles.qmd` defines five Part IV principles: Verification Gap, Statistical Drift, Training-Serving Skew, Latency Budget, and Bias Feedback. Across the four part-principle files, the count is therefore `2 + 2 + 4 + 5 = 13`, not 12. The conclusion also says "invariants nine through twelve" are the deployment invariants on line 550, which omits Bias Feedback if Part IV's own principle list is authoritative.
- **Proposed correction:** Either add Bias Feedback as a thirteenth row in `@tbl-twelve-principles` and change "twelve" to "thirteen" throughout the conclusion, or explicitly state that the conclusion's "twelve invariants" exclude Bias Feedback and explain why that Part IV principle is not counted in the synthesized invariant set.

### 2. MobileNetV2 FLOP reduction says 8--9x versus ResNet-50, but the book's own earlier comparison gives about 14x

- **Lines:** 168-175
- **Severity:** Medium
- **Issue:** The table states "Depthwise Separable Convolutions: 8--9x reduction in FLOPs vs. ResNet-50." The 8--9x factor is a reasonable per-layer depthwise-separable-vs-standard-convolution intuition for common `3 x 3` convolutions, but it is not the full MobileNetV2-vs-ResNet-50 model comparison used elsewhere in the book. The earlier Lighthouse setup uses ResNet-50 at about `4.1 GFLOPs` and MobileNetV2 at about `0.3 GFLOPs`, giving `4100 / 300 = 13.7x`, which rounds to about `14x`.
- **Proposed correction:** Change the table entry to either "about 14x fewer FLOPs vs. ResNet-50" for the full-model comparison, or "8--9x reduction for representative depthwise-separable convolution layers versus standard convolutions" if the intended claim is the layer-level mechanism.

### 3. Llama/H100 roofline comments claim 40x memory-bound, but the code computes about 295x

- **Lines:** 50-61, 80-113, 565-582
- **Severity:** Low
- **Issue:** The setup comment says the goal is to show Llama-2-70B is `40x` memory-bound on H100. With the constants used in the code, the ratio is much larger. The model bytes are `70B params x 2 bytes = 140 GB`; H100 bandwidth is `3.35 TB/s = 3350 GB/s`, so `T_mem = 140 / 3350 = 0.0418 s = 41.8 ms`. The compute is `2 x 70B = 140 GFLOPs`; H100 peak is `989 TFLOP/s = 989,000 GFLOP/s`, so `T_comp = 140 / 989000 = 0.000142 s = 0.142 ms`. The ratio is `41.8 / 0.142 ~= 295x`, matching the H100 ridge point for arithmetic intensity `1 FLOP/byte`, not `40x`.
- **Proposed correction:** Update the comment to say "about 295x memory-bound under peak FP16 Tensor Core throughput" or, if the intended pedagogical ratio is `40x`, add the assumed effective sustained compute throughput that makes the ratio `40x` and use it consistently in the calculation.

### 4. Training-serving skew equation equates output divergence with accuracy loss without a task/loss mapping

- **Lines:** 199-202, 232, 240, 558-561
- **Severity:** Low
- **Issue:** The table gives `Delta Acc approx E[|f_serve(x) - f_train(x)|]`. The expectation is in model-output units unless `f` is defined as a scalar correctness indicator. For probabilities, logits, embeddings, rankings, or class labels, output divergence is not itself an accuracy change. The prose also states that subtle differences "silently degrade accuracy," but a serving/training difference more precisely invalidates the evaluated-performance assumption and may degrade accuracy depending on task distribution and decision boundaries.
- **Proposed correction:** Define the equation through a task loss or disagreement event, for example `|Delta L| <= C E[d(f_serve(x), f_train(x))]` under an explicit loss/distance assumption, or phrase it qualitatively as "output divergence is a risk signal for accuracy degradation."

### 5. "Wastes 100 percent of their effort" overstates the roofline implication

- **Lines:** 577-582
- **Severity:** Low
- **Issue:** The roofline example correctly shows memory dominates compute time, but optimizing compute kernels is not literally a 100 percent waste. Under the code's own numbers, eliminating compute entirely would reduce per-token time from about `41.94 ms` to `41.79 ms`, a maximum improvement of roughly `0.34 percent`. The engineering point is correct, but the numeric wording is absolute in a way the calculation does not support.
- **Proposed correction:** Replace with a quantitative statement such as "can save at most about 0.3 percent in this setup" or "spends effort on a term that contributes less than 1 percent of total token time."

## Verified calculations and consistency checks

- Lines 25 and 40-47: `\mlsysstack{30}{30}{30}{30}{30}{30}{30}{30}` has eight equal entries; the learning-objective list has seven bullets.
- Lines 85-113 and 571-582: Aside from the stale `40x` comment, the Llama/H100 generated arithmetic is internally consistent: `140 GB / 3350 GB/s ~= 41.8 ms`; `140 GFLOPs / 989 TFLOP/s ~= 0.14 ms`; ratio `~= 295x`; arithmetic intensity is `2 FLOPs / 2 bytes = 1 FLOP/byte`.
- Lines 118, 156, 175, and 177: The latency and compression examples are dimensionally clear: `P99 < 50 ms` is a latency constraint. In this table, "INT8 quantization: 4x memory reduction" is acceptable as a common FP32-to-INT8 storage comparison; if the source representation is FP16, the same quantization would be a `2x` storage reduction.
- Lines 189-202: The iron-law and roofline equations are dimensionally coherent: `D_vol / BW`, `O / (R_peak * eta_hw)`, and `L_lat` are times; `I x BW` has units of operations per second when `I` is operations per byte.
- Lines 197 and 224: The Energy-Movement Invariant's `100--1,000x` movement-vs-compute range is stated as an order-of-magnitude qualitative ratio and is internally consistent with the prose.
- Lines 604-644: Tail-latency ratio is correct: `2000 ms / 50 ms = 40x`.
- Lines 661 and 667: The `100--1,000x`, `4-8` token, `2--3x`, and `70 percent` claims are qualitative or externally sourced elsewhere; no internal arithmetic contradiction was found in this file.
- Lines 709 and 713: "Thousand-fold" means `1000x`, "exascale" is correctly expressed as `>= 10^18 FLOP/s`, and the FLOP/s unit is correct.
- Lines 721-745: Petabytes, thousands of GPUs, months of training, and failures per hour are scale claims without local equations; units are used consistently.
- Lines 761 and 773: `50 percent` latency reduction, `95 percent` vs. `93 percent` accuracy, `500 ms` vs. `50 ms`, and `40x` error-rate disparity are internally clear as illustrative comparisons.
- Lines 779-781: The Amdahl example is correct. If data loading is `90 percent` of end-to-end latency and inference is the remaining `10 percent`, a `10x` inference speedup gives new time `0.90 + 0.10/10 = 0.91`, so total speedup is `1 / 0.91 = 1.10x`, i.e. about `1.1x`.

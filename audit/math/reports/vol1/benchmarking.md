# Math Audit Report: `book/quarto/contents/vol1/benchmarking/benchmarking.qmd`

## Checked scope

Audited benchmark metrics, equations, numeric examples, unit conversions, throughput/latency/energy calculations, statistical claims, and prose-equation consistency using direct reasoning only. No Gemini or external model verification was used. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 2790-2802 and 2835-2837: EdgeTPU energy conclusion is under-specified and contradicts the reported per-inference energy table.**  
  The table computes CPU energy as `120 mW x 15 ms / 1000 = 1.8 mJ` and EdgeTPU energy as `500 mW x 2 ms / 1000 = 1.0 mJ`, so the EdgeTPU is lower energy per active inference. The deployment decision then says the CPU is "more power-efficient overall" for infrequent use. That may be true only if the EdgeTPU adds idle/leakage, wake-up, host-transfer, or standby overhead, but none of those terms are modeled. As written, the statement reverses the only energy calculation shown.
  - Proposed correction: Either remove the CPU-is-more-efficient claim, or add an explicit duty-cycle energy model such as `E_day = E_inference x N + P_idle x t_idle + E_wake x N` with CPU and EdgeTPU idle/wake assumptions. If using only the displayed calculation, say EdgeTPU has lower active per-inference energy, while total daily battery impact depends on idle and wake overheads.

- **Lines 431-447 and 452: the A100 FP32 value is sourced from a V100 constant.**  
  `A100Roofline._a100_fp32` is assigned `V100_FLOPS_FP32`, but the prose presents the result as "FP32" performance for an NVIDIA A100. A V100 FP32 peak is not the A100 FP32 peak, so the interpolated A100 spec will be wrong even though the FP16 roofline threshold uses the A100 peak.
  - Proposed correction: Replace `V100_FLOPS_FP32` with the A100 FP32 constant or derive it from `Hardware.A100` if available. If the chapter only needs the FP16 Tensor Core roofline, omit the FP32 parenthetical rather than mixing hardware generations.

### Medium Severity

- **Lines 1643, 1739, 1955, and 2238: training-memory and GPT-3 energy examples use inconsistent scales for "large training" overhead.**  
  Line 1643 says training creates only `3--4x` memory overhead relative to inference. That can be true for simple mixed-precision training without Adam states and large activations, but the chapter's surrounding large-model examples discuss Adam-style training where FP16 weights + FP16 gradients + FP32 Adam moments already produce about `6x` FP16 inference weight memory before activations. This understates the memory burden for the same class of systems the chapter is describing.
  - Proposed correction: Qualify the line as "often 3--4x for simpler optimizers/workloads, and 6x or more for Adam-style large-model training before activations." Keep the exact multiplier tied to what is counted: weights, gradients, optimizer state, activations.

- **Lines 2100-2127 and 2205-2234: the MobileNet energy table labels MAC energy but computes with FLOP constants.**  
  The code sets `m_macs = 300` and the rendered table says "per multiply-accumulate operation" and "300 M MACs," but the constants are named and loaded as `ENERGY_FLOP_FP32_PJ` and `ENERGY_FLOP_INT8_PJ` in pJ/flop. If one MAC is counted as two FLOPs, the compute-energy rows are low by `2x`; if the constants are really per MAC, the variable names and units are misleading. The total savings ratio may be close, but the absolute microjoule values and prose are not unit-clean.
  - Proposed correction: Choose one convention. Either rename the operation count to FLOPs and use `600M` FLOPs for `300M` MACs, or use explicit per-MAC energy constants and label them as such. State the convention in the table heading.

- **Lines 3058, 3115, 3253, and 3351: MLPerf power efficiency units alternate between samples/watt and samples/joule.**  
  The figure caption says energy efficiency is "samples per watt," but the axes say "Samples/Joule." These are not equivalent: watt is power (`J/s`), while samples/J is work per unit energy. `samples/W` would have units `samples x s / J` and depends on time normalization. For energy efficiency trends, the chart should use samples/J or inferences/J.
  - Proposed correction: Change the caption to "samples per joule" or "samples/J." If MLPerf reports a power-normalized throughput such as samples/s/W, write it explicitly as `samples/s/W`, which simplifies dimensionally to samples/J.

- **Lines 958-965 and 2671-2673: H100 INT8 TOPS is exported from an FP8 TFLOPS value.**  
  `h100_tflops_int8_str` is assigned `fmt(_h100_fp8, ...)`, then the TOPS footnote uses it as H100 INT8 throughput. H100 dense FP8 Tensor Core throughput and INT8 TOPS may be numerically similar for some spec-sheet modes, but the source value is still the FP8 constant and the variable name says TFLOPS rather than TOPS. This obscures the precision and operation-count convention the footnote is warning readers to check.
  - Proposed correction: Use an explicit H100 INT8 TOPS constant, or rename the value to indicate it is the FP8 dense Tensor Core spec. In the footnote, separate `FLOPS` and `OPS/TOPS` terminology.

- **Lines 4503-4589 and 4645: the battery-life multiplier ignores the throughput loss if the comparison is per unit work.**  
  The example compares `1200 QPS at 420 W` with `1000 QPS at 180 W`. The power ratio is `420/180 = 2.33x`, matching the displayed `2.3x` longer active runtime. But if the goal is fixed useful work, energy per query is `420/1200 = 0.35 J/query` versus `180/1000 = 0.18 J/query`, so the lower-power system provides about `0.35/0.18 = 1.94x` more queries per joule, not `2.3x`.
  - Proposed correction: Say "2.3x longer active wall-clock runtime at 17 percent lower throughput" or change the battery-efficiency claim to about `1.9x` more inferences per joule for a fixed workload.

### Low Severity

- **Lines 406-408 and 491-502: ResNet utilization uses the max endpoint while prose gives a range.**  
  The code computes `resnet_perf_tflops = peak_flops * 90%`, so the rendered achieved TFLOPS corresponds only to the top of the stated `85-90 percent` range. The prose says "85-90 percent ... approximately X TFLOPS," which makes the single value look representative of the whole range.
  - Proposed correction: Render an achieved range, e.g. `0.85 x peak` to `0.90 x peak`, or state that the displayed TFLOPS uses the upper end of the range.

- **Lines 1527-1548: MobileNet parameter-efficiency improvement is rounded as `7.5x`, but the stated parameter/accuracy numbers imply about `6.9x` by accuracy per parameter.**  
  Using the prose values, `(72 / 3.5) / (76 / 25.6) = 6.9x`. The raw parameter-count ratio is `25.6 / 3.5 = 7.3x`. Neither directly gives `7.5x`; the number is close but the metric name "parameter-to-accuracy ratio" is ambiguous.
  - Proposed correction: Use a precise metric. For parameter count, say MobileNet uses about `7.3x` fewer parameters. For accuracy per parameter, say about `6.9x` higher top-1 accuracy per million parameters.

- **Lines 1966-1969: the time-to-accuracy equation should specify the admissible training-time domain.**  
  `T_train = argmin_t { accuracy(t) >= target accuracy }` is mathematically understandable, but `argmin` over time should be constrained to observed training checkpoints or `t >= 0`. Without that, interpolation/noise can make the first crossing ambiguous.
  - Proposed correction: Write `T_train = min { t >= 0 : accuracy(t) >= target accuracy }`, optionally "over evaluated checkpoints after smoothing or benchmark-defined convergence criteria."

- **Lines 4046-4050: MobileNet INT8 edge-case drop formatting hides precision.**  
  `edge_drop = 68.2 - 61.4 = 6.8` percentage points, but `mv2_edge_drop_str` is formatted with zero decimals, rendering `7% drop`. That is acceptable as an approximation, but the table mixes one-decimal accuracies with an integer drop and may imply an exact 7.0 point drop.
  - Proposed correction: Format the edge-case drop with one decimal place and call it "percentage points": `6.8 pp drop`.

## Verified Correct

- Lines 1672-1694: `E = P x t` is applied correctly. `300 W x 0.01 s = 3 J`, and `3 J / 3600 = 0.00083 Wh`, which rounds to the displayed `0.0008 Wh`.
- Lines 2000-2053: the strong-scaling example is internally consistent. `24 h / (8 x 4 h) = 0.75`, so efficiency is `75 percent`; ideal time is `24/8 = 3 h`; overhead loss is `25 percent`.
- Lines 2494-2549: the Amdahl latency example is arithmetically correct. Baseline latency is `8 + 10 = 18 ms`; a `5x` inference speedup gives `8 + 2 = 10 ms`; end-to-end improvement is `18/10 = 1.8x`; infinite-inference ceiling is `1 / (8/18) = 2.25x`.
- Lines 2665: the cold-start lower bound is correct. A 7B-parameter FP16 model is about `14 GB`; at `25 GB/s`, weight transfer alone is `14/25 = 0.56 s`, or about `560 ms`.
- Lines 2790-2835: the active EdgeTPU arithmetic itself is correct: inference speedup `15/2 = 7.5x` rounds to `8x`; end-to-end speedup `18/6 = 3x`; power ratio `500/120 = 4.17x` rounds to `4x`; active energy ratio `1.8/1.0 = 1.8x`.
- Lines 3989-4016 and 4023-4044: the MobileNet INT8 compression arithmetic is internally consistent. Size reduction is `14.0 MB / 3.5 MB = 4x`; top-1 drop is `71.8 - 70.9 = 0.9` percentage points; ECE increases from `0.031` to `0.089`; edge-case drop is `6.8` percentage points.
- Lines 4102-4116: the LLM throughput example is correct. `750 / 25 = 30 s` and `750 / 100 = 7.5 s`.
- Lines 4585-4589 and 4636-4655: the fallacies calculations are mostly internally consistent: accuracy drop to the high end of production range is `92 - 82 = 10` points; latency multiplier uses the low p99 endpoint `150/15 = 10x`; throughput degradation uses the high production endpoint, `1 - 500/800 = 37.5 percent`, rounded to `38 percent`; `99.9 percent` monthly availability permits about `43.2` minutes downtime in a 30-day month.

# Math Audit Report: `book/quarto/contents/vol2/performance_engineering/performance_engineering.qmd`

## Checked scope

Audited performance, roofline, throughput, latency, memory traffic, precision/quantization, communication overlap, profiling, cost/scaling, and prose-equation consistency using direct reasoning only. No Gemini or external model verification was used. Source `.qmd` files were not modified.

## Findings

### High Severity

- **Lines 361 and 363: the B200 ridge point is off by about 3.6-4x, and the caption contradicts the prose about naive attention.**  
  The table at lines 247-252 gives B200 FP16 peak `4,500 TFLOPS` and HBM bandwidth `8.0 TB/s`, so the ridge point is `4,500 / 8 = 562.5 FLOP/byte`, matching the table's `563`. The figure code uses `5,000 / 8 = 625 FLOP/byte`. Neither supports the prose/caption value of `2,250 FLOP/byte` or the claimed `16x` increase from V100. The increase is `563/139 = 4.0x` using the table, or `625/139 = 4.5x` using the figure code. The figure caption also says operations like naive attention "were compute-bound on V100," but line 361 correctly says naive attention at about `10 FLOP/byte` is below the V100 ridge point `139 FLOP/byte`, so it is memory-bound even on V100.
  - Proposed correction: Use one B200 spec consistently. If using the table, change line 361 and the figure caption to "139 to 563 FLOP/byte, a 4x increase." If using the figure code, change the table/figure to `625 FLOP/byte` and say `4.5x`. Remove the claim that naive attention was compute-bound on V100.

- **Lines 614, 2050, and 2052: the 70B weight-shard and INT4 memory arithmetic is inconsistent.**  
  A 70B FP16 model is `70B x 2 bytes = 140 GB`, not `80 GB` as stated on line 614. In the 8-GPU case study, the FP16 shard is correctly `140/8 = 17.5 GB/GPU`, but INT4 storage is `35/8 = 4.375 GB/GPU`, not `8.75 GB/GPU`; `8.75 GB/GPU` is the 4-GPU value. Therefore available memory after INT4 on an 80 GB H100 is about `80 - 4.375 = 75.6 GB/GPU`, not `71 GB/GPU`. Also, FP16 to INT4 reduces weight bytes by `16/4 = 4x`; line 2050 says effective bandwidth only doubles.
  - Proposed correction: For the 8-GPU case, write "140 GB to 35 GB total, or 17.5 GB/GPU to 4.4 GB/GPU; available KV memory rises from about 62.5 GB/GPU to 75.6 GB/GPU." If a 2x speedup is intended after dequantization overhead, state that separately rather than saying the byte reduction is 2x.

- **Lines 725, 729, and 735: fusion traffic savings do not match the tensor size stated in the worked example.**  
  A `4096 x 2048` FP16 tensor is `4096 x 2048 x 2 = 16,777,216` bytes, about `16 MiB` or `16.8 MB`. Two intermediates `Z1` and `Z2` each require one write and one read if unfused, so the avoidable intermediate traffic is `2 tensors x 2 transfers x 16 MiB = 64 MiB`, not `32 MB` as line 725 says. The figure caption's `64 MB` saving is consistent with the transfer count but inconsistent with its own `88 MB` vs `24 MB` totals if each activation tensor is 16 MB.
  - Proposed correction: Change line 725 to say the unfused execution reads and writes `64 MB` of avoidable intermediate data. Recompute the figure totals with the same tensor size, or clarify that the figure uses a smaller illustrative tensor where each activation is 8 MB.

- **Lines 771-838 versus 857-861: the FlashAttention savings calculation changes definitions and gives different answers.**  
  The code computes naive traffic as only write+read of `S`: `2 x N^2 x bytes x heads`. For `N=8192`, FP16, and 32 heads, this is about `8.6 GB`; Flash traffic is `4 x N x d x bytes x heads`, about `268 MB`, giving `32x`. The later prose counts fuller naive traffic per head: QKV read `6 MB`, S write/read `256 MB`, P write/read `256 MB`, O write `2 MB`, totaling about `520 MB`; Flash is `8 MB`, giving `65x`. Both can be valid under different traffic definitions, but the chapter presents them as the same quantity.
  - Proposed correction: Choose one definition. If counting only score-matrix materialization, use `32x` and avoid saying `520 MB per head`. If counting all listed naive transfers, update the code to include `Q,K,V,O,S,P` traffic and render about `520 MB/head`, `16.6 GB total`, `268 MB Flash`, and `62-65x`.

- **Lines 1432 and 1430-1432: the communication example's GEMM timing is too large for the stated GEMM.**  
  The stated GEMM shape `64 x 8192 x 1024` costs `2 x 64 x 8192 x 1024 = 1.07e9 FLOPs`. On an H100 at `989 TFLOP/s`, the ideal time is about `1.1 us`; even at 40 percent utilization it is about `2.7 us`, not `20 us`. The communication time for `2 MB` at `900 GB/s` is about `2.2 us`, so the compute/communication ratio for that single GEMM is near `1.2x` at 40 percent MFU, not `9x`.
  - Proposed correction: Either change the compute time to roughly `2-3 us` for the stated GEMM, or state that the `20 us` includes multiple layer kernels/GEMMs rather than the single matrix multiply shown.

- **Lines 1862-1923: the fleet MFU notebook produces impossible or mismatched utilization values.**  
  The code uses `tokens_per_step = 2048 x 32 = 65,536` for both the 8-GPU local baseline and the 128-GPU fleet case. Useful FLOPs are about `6 x 70B x 65,536 = 2.75e16`. The local denominator is `8 x 989e12 x 0.180 = 1.42e15`, producing about `1,930 percent MFU`, which is impossible. The 128-GPU denominator is `128 x 989e12 x 0.245 = 3.10e16`, producing about `89 percent`, not the comment's intended `48 percent`. The scaling tax is therefore not meaningful.
  - Proposed correction: Use consistent global batch/tokens for each GPU count. For example, if weak scaling from 8 to 128 GPUs, local tokens should be `65,536 / 16 = 4,096`. Then recompute step times or token counts so rendered MFU values are physically below 100 percent and match the intended local/fleet comparison.

### Medium Severity

- **Lines 145-149: the cost-per-token examples do not follow from the stated throughputs without an unstated GPU hourly price and throughput convention.**  
  Configuration A can be made plausible if the 8-GPU model produces about `80 tokens/s` total and each H100 costs roughly `$4.30/hour`: `8 x 4.30 / (80 x 3600) x 1000 = $0.119/1K tokens`. Configuration B says each H100 serves `4,000 tokens/s`; with 4 GPUs that is `16,000 tokens/s`, giving about `$0.0003/1K` at the same hourly price, not `$0.002/1K`. If `4,000 tokens/s` is total across the 4-GPU deployment, the cost is about `$0.0012/1K`, still below `$0.002`.
  - Proposed correction: State the assumed GPU hourly price and whether throughput is per GPU or per deployment. Adjust Configuration B's cost or throughput so the `60x` cost ratio is derived from the same assumptions as Configuration A.

- **Lines 604-610: the batch-size arithmetic is right, but the explanation says the wrong denominator term becomes significant.**  
  The formula gives `I_decode(batch) = 2 params batch / (params bytes_per_param + batch d bytes_per_elem)`. For realistic LLMs, `params` is much larger than `batch x d`, so even at batch 256 the weight term still dominates. The approximation `I ~= 2 x 256 / bytes_per_param = 256 FLOP/byte` relies on the input/activation term remaining negligible, not becoming significant.
  - Proposed correction: Change "the input term becomes significant" to "the weight term still dominates, so the same weight bytes are amortized across 256 tokens."

- **Lines 853 and 865: the FlashAttention exactness claims are too strong for floating-point execution.**  
  The online softmax recurrence is mathematically exact in real arithmetic, but tiled accumulation changes floating-point operation order. "Bit-identical results to standard attention" is generally not guaranteed across kernels, tile sizes, accumulation precision, and FP8/FP16 modes. FlashAttention is exact in the algorithmic sense, not an approximation to attention, but numerical roundoff can differ.
  - Proposed correction: Replace "bit-identical results" with "mathematically equivalent results up to floating-point roundoff for the chosen precision."

- **Lines 873 and 886-920: the FlashAttention memory-savings figure compares score-matrix storage to running statistics, not total attention memory or traffic.**  
  The code uses `mem_standard = N^2 x 2 x heads` and `mem_flash = N x 4 x heads`, omitting bytes per running statistic and omitting head dimension/QKV/O storage. This makes the annotated ratio `N/2`, yielding `4096x` at 8K and `32768x` at 64K. That is not the same metric as lines 857-861, where FlashAttention still moves QKV and O and the 8K traffic ratio is about `65x`. The figure alt text also says the O(N) curve "stays flat," but an O(N) curve grows linearly on a log-log plot.
  - Proposed correction: Label the figure explicitly as "score matrix vs online-softmax state memory," or include QKV/O and head dimension in the FlashAttention curve. Change "stays flat" to "grows linearly."

- **Lines 1004 and 1006: quantization metadata overhead changes baseline midstream.**  
  With block size `B=64` and one FP16 scale, metadata adds `16/64 = 0.25 bits/weight`, so the effective width is correctly `4.25 bits/weight`. But this overhead is `0.25/4 = 6.25 percent` relative to the 4-bit quantized weights, not `1.6 percent`; `1.6 percent` is relative to the original 16-bit weight storage. At `B=16`, metadata adds `1 bit/weight`, which is `25 percent` overhead relative to 4-bit weights, though `6.25 percent` relative to the FP16 baseline.
  - Proposed correction: State both baselines explicitly, e.g. "0.25 bits/weight, a 6.25 percent overhead on the INT4 payload or 1.6 percent of the original FP16 size." For `B=16`, say "25 percent overhead on the INT4 payload."

- **Lines 1053-1141: the KV cache callout relies on sharding assumptions that are not stated clearly.**  
  The code computes total FP16 KV cache as `2 x 80 x 8 x 128 x 4096 x 2 = 1.34 GB/request`, then divides by 4 GPUs for per-GPU cache. This is valid only if the KV heads/cache are sharded across tensor-parallel GPUs. Without that assumption, the per-GPU cache is `1.34 GB`, not `0.34 GB`, and the max batch values are about 4x smaller. The code comments also mention `1.1 GB` and max batch `166`, but the actual formula gives about `1.3 GB` and `134`.
  - Proposed correction: Add "assuming tensor-parallel KV head sharding across 4 GPUs" to the problem, and remove the stale comments. If KV cache is replicated, do not divide by 4.

- **Lines 1438-1524: the overlap notebook comments and rendered arithmetic are not aligned with the intended 1.50x example.**  
  The comments say gradients are `14 GB`, backward is `~31 ms`, AllReduce is `~31 ms`, and overlap improves `93 ms` to `62 ms` (`1.50x`). But the code's backward FLOPs are `2 x (2 x 7e9 x 2048) = 5.73e13 FLOPs`. At `989 TFLOP/s` and 40 percent MFU, backward time is about `145 ms`, not `31 ms`; forward is about `72 ms`; AllReduce is about `31 ms` using the approximate ring factor 2. The resulting overlap speedup is roughly `(72+145+31)/(72+145) = 1.14x`, not `1.50x`.
  - Proposed correction: If the pedagogical target is `1.50x`, reduce `tokens_per_gpu` or change the hardware/MFU assumptions so backward and AllReduce are both near `31 ms`. Otherwise update the comments and prose to the rendered `~1.14x` speedup.

- **Lines 2004-2010 versus 2048-2067: the heuristic optimization order conflicts with the case study order.**  
  The heuristic says apply `torch.compile` first and weight quantization second. The case study applies precision engineering first, then `torch.compile`/fusion. Both orders can be defensible, but the chapter presents each as the systematic sequence.
  - Proposed correction: Either make the heuristic conditional ("for low-risk first pass, try torch.compile; for memory-capacity-limited 70B serving, quantize first") or align the case study with the stated heuristic.

### Low Severity

- **Lines 90, 127, and 1700: "zero improvement" is too absolute for non-dominant-term optimization.**  
  In a max-model, optimizing the smaller term gives no improvement only while it remains below the dominant term and overhead is unchanged. Real workloads often have partial overlap, non-ideal ceilings, and changing overheads, so the claim is directionally useful but too categorical.
  - Proposed correction: Replace "zero improvement" with "little or no improvement until that term becomes dominant" where the statement is conceptual rather than exactly tied to the simplified equation.

- **Lines 218-224: clock-cycle-to-nanosecond conversions imply different GPU frequencies.**  
  Line 220 says `20-30 clock cycles (~20 ns)`, implying roughly `1-1.5 ns/cycle`. Line 222 says `200 clock cycles (~130 ns)`, implying `0.65 ns/cycle`. These are both rough, but presented together they look more precise than warranted.
  - Proposed correction: Use less precise latency ranges, e.g. "tens of nanoseconds" for shared memory and "around 100-200 ns" for L2, or state that these are architecture/workload-dependent approximations.

- **Line 224: reading the full H100 memory capacity takes closer to 24-27 ms depending on GB convention.**  
  `80 GB / 3.35 TB/s = 23.9 ms` using decimal units, while `80 GiB / 3.35 TB/s = 25.6 ms`, and if the H100 memory is 80 GB but accessible bandwidth is lower in practice, the number may be higher. The stated `24 ms` is correct under decimal units but should not be read as a binary GiB calculation.
  - Proposed correction: No mathematical change needed if decimal units are intended; optionally write "about 24 ms using decimal GB/TB units."

- **Lines 226: the home-power equivalence is plausible but underspecified.**  
  Saving `320 kW` and calling it roughly `250 homes` implies about `1.28 kW/home` average load. That is plausible for average continuous household power, but readers may interpret "powering homes" as peak service capacity.
  - Proposed correction: Say "roughly 250 average U.S. homes at about 1.3 kW continuous average load" or omit the equivalence.

- **Lines 737 and 761: kernel-launch overhead percentages mix denominator conventions.**  
  Line 737 says `50 x 10 us = 500 us`; if arithmetic execution takes `2 ms`, then overhead is `0.5/(2+0.5) = 20 percent` of total wall time, which is correct. If "actual arithmetic execution" is interpreted as already including overhead, it would be 25 percent. The later CUDA Graphs example is consistent at the order-of-magnitude level.
  - Proposed correction: Write "2 ms of arithmetic plus 0.5 ms launch overhead" to make the 20 percent denominator explicit.

- **Lines 1801-1808: "remaining 53 percent" is a throughput gap, while the trace percentages are time shares.**  
  The observed rate `45 tokens/s` versus theoretical `96 tokens/s` means the system achieves `46.9 percent` of the theoretical rate and loses `53.1 percent` of possible throughput. The trace then says GEMM is `42 percent` of step time. These are not directly additive quantities, so the transition can confuse readers.
  - Proposed correction: State that the 53 percent is the lost throughput relative to an idealized bandwidth-only GEMM bound, then separately analyze the observed step-time shares.

## Verified Correct

- Lines 145-149: the stated `60x` cost ratio and `6x` latency ratio follow from `$0.12/$0.002 = 60` and `120 ms / 20 ms = 6`, independent of the unstated cost model.
- Lines 218-224: `256 KB/SM x 132 SMs = 33,792 KB`, about `33 MB`, and `228 KB/SM` shared memory plus `50 MB` L2 are internally consistent as approximate H100 hierarchy figures.
- Lines 247-254: the table values are internally consistent: `125/0.9 = 139`, `312/2.0 = 156`, `989/3.35 = 295`, and `4500/8 = 563`; compute increases `4500/125 = 36x`, bandwidth increases `8/0.9 = 8.9x`, and ridge point increases about `4x`.
- Lines 294-298: the H100 FP16 ridge calculation is correct: `989 TFLOP/s / 3.35 TB/s = 295 FLOP/byte`.
- Lines 557-583: the `4096^3` GEMM example is correct under the stated memory model: `2 x 4096^3 = 137.4B FLOPs`, bytes are `3 x 4096^2 x 2 = 100.7 MB`, and arithmetic intensity is about `1365 FLOP/byte`.
- Lines 562-587: the element-wise and batch-1 decode intensity examples are correct under their simplified assumptions: GELU is `5/(2 x 2) = 1.25 FLOP/byte`, and a batch-1 FP16 linear decode step is about `1 FLOP/byte`.
- Lines 622-624: if preallocation wastes `1 - 500/4096 = 87.8 percent`, describing the waste as about `88 percent` is correct.
- Lines 679-695: the 8-GPU 70B decode roofline diagnostic is consistent: each GPU reads `17.5B x 2 = 35 GB`, performs `35B FLOPs`, has `AI = 1`, and the bandwidth floor is `35 GB / 3.35 TB/s = 10.4 ms`, about `96 tokens/s` before overheads.
- Lines 857-861: the detailed per-head FlashAttention memory-traffic example is internally consistent: QKV/O total about `8 MB`, naive listed traffic about `520 MB`, and `520/8 = 65x`.
- Lines 1022: one FP32 scale per 128 INT8 weights adds `32/(128 x 8) = 3.125 percent`, so "only 3 percent" is correct.
- Lines 1307-1362: the compilation dividend arithmetic is consistent: baseline `120 tokens/s` is `8.33 ms/token`; post-compile time is `4.17 + 0.87 + 0.25 = 5.29 ms`, giving about `189 tokens/s` and a `1.58x` speedup.
- Lines 2079-2085: the speculative decoding ITL arithmetic is consistent if the expected tokens per verification round include the sampled target token: `(4 ms + 8 ms)/3.5 = 3.43 ms/token`, and `32/3.4 = 9.4x`.

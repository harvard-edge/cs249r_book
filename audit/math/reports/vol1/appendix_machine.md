# Math Audit Report: `book/quarto/contents/vol1/backmatter/appendix_machine.qmd`

## Checked scope

Audited hardware, memory, throughput, latency, energy equations, numeric examples, unit conversions, complexity/scaling claims, and prose-equation consistency using direct reasoning only. No Gemini or external model verification was used. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 129, 161, and 166: the DRAM energy ratio is computed per FP16 FLOP but described as an FP16 multiply-add.**  
  The code computes `ENERGY_DRAM_ACCESS_PJ / ENERGY_FLOP_FP16_PJ = 640 / 1.1 ~= 581`, so the displayed `~600x` or `~580x` ratio is a DRAM access versus one FP16 FLOP. Line 129 says "FP16 multiply-add"; elsewhere in the book a multiply-add is commonly counted as 2 FLOPs. Under that convention, the ratio versus one FP16 multiply-add would be `640 / (2 x 1.1) ~= 291x`, not `~600x`.
  - Proposed correction: Say "FP16 FLOP" or "FP16 operation" on line 129, or if the intended unit is one MAC/multiply-add, halve the displayed ratio and state the MAC convention explicitly.

- **Lines 131, 191, and 196: the 16 bytes/parameter training-memory decomposition omits gradients.**  
  The prose says `2B FP16 weights + 4B FP32 master weights + 8B Adam states = 16B`, but those terms sum to `14B`, not `16B`. The usual 16 B/parameter mixed-precision Adam rule is `2B FP16 weights + 2B gradients + 4B FP32 master weights + 8B Adam moments = 16B`, before activations and fragmentation.
  - Proposed correction: Add the missing `2B` gradients term in lines 131 and 196. For example: `FP16 weights (2B) + gradients (2B) + FP32 master weights (4B) + Adam moments (8B) = 16B/parameter`.

- **Lines 239-245: the compute table labels H100 FP8 throughput as INT8 and the caption's 28x ratio does not match the code constants.**  
  Line 241 places `NumbersToKnow.flops_h100_fp8` in the `INT8` column. The variable is derived from `H100_FLOPS_FP8_TENSOR`, not an INT8 TOPS constant. Separately, `dc_mobile_ratio = int(flops_h100_fp16 / flops_mobile_int8)`, and the local constants give `989 / 50 = 19.78`, rendered as `19x` in line 194. The caption's hardcoded "~28x" does not follow from those values. If comparing H100 FP8/INT8-class peak to the same mobile INT8 constant, the ratio would be `1979 / 50 ~= 40x`.
  - Proposed correction: Use an explicit H100 INT8 TOPS value if the column remains `INT8`, or relabel it as `FP8`. Update the caption to match the chosen ratio, e.g. `~20x` for H100 FP16 versus mobile INT8 using the current constants.

### Medium Severity

- **Lines 172-181: the memory-hierarchy prose says each level costs 10--100x more latency, but the displayed relationships exceed that range.**  
  With the appendix constants, HBM versus L1 is `300 ns / 1 ns = 300x`, SSD versus L1 is `100,000 ns / 1 ns = 100,000x`, and network versus local HBM is `5,000 ns / 300 ns ~= 16.7x`. These are not all adjacent levels, and two rows are outside the stated `10--100x` range.
  - Proposed correction: Change the prose/caption to "latency grows by orders of magnitude across the hierarchy" or make the table compare adjacent levels only and revise the range.

- **Lines 307-312: the batch-size arithmetic-intensity explanation omits input/output traffic.**  
  The FLOP count for `Y = XW` is correct, but the bytes bullet counts only weights. A fuller FP16 traffic model is approximately `2 bytes x (B D_in + D_in D_out + B D_out)`, ignoring caching details. Doubling `B` keeps weight traffic fixed, but input and output traffic also double, so arithmetic intensity does not literally double for all shapes and batch sizes.
  - Proposed correction: Qualify this as the weight-dominated approximation, or include the input/output terms and then explain that for large `D_in x D_out` and small-to-moderate `B`, weight reuse is the dominant effect.

- **Lines 832-843 and 863: the latency "relative distance" analogies are not scaled consistently.**  
  If register access at `0.3 ns` maps to `10 seconds`, then L1 at `1 ns` maps to about `33 seconds`, HBM at `300 ns` maps to about `2.8 hours`, and NVMe at `100,000 ns` maps to about `39 days`. The table gives L1 as `1 minute`, HBM as `5 hours`, and SSD as `3 months`. These are directionally useful but not a coherent scale.
  - Proposed correction: Either label the analogy column as qualitative, or recompute it from one fixed baseline. For example, using register `0.3 ns = 10 seconds`, SSD is about `39 days`, not `3 months`.

- **Lines 1002-1003 and 1034-1045: the 1 GB over 10 Gbps example hides the byte-to-bit conversion in the rendered equation.**  
  The code correctly computes `1 GB x 8 / 10 Gbps = 0.8 s = 800 ms`, plus `10 ms` latency for `810 ms`. But the rendered equation string is `1GB / 10Gbps ~= 800ms`. Dimensionally, `GB/Gbps` mixes bytes and bits; without the factor of 8, a reader may infer `100 ms`.
  - Proposed correction: Render the equation as `(1 GB x 8 bits/byte) / 10 Gbps ~= 800 ms`, then `10 ms + 800 ms = 810 ms`.

- **Line 1123: the precision discussion refers to the iron-law data-movement term as being in the denominator.**  
  The iron-law equation on lines 452-456 has the data term as `D_vol / BW`; reducing precision halves `D_vol`, the numerator of that term. It does not halve a denominator. The throughput conclusion is reasonable for memory-bound workloads, but the prose points to the wrong side of the equation.
  - Proposed correction: Replace "halves the Data Movement term in the denominator" with "halves `D_vol` in the data-movement term."

- **Lines 1143-1144: asymmetric quantization formula maps to unsigned 8-bit values while the table labels INT8 as -128 to 127.**  
  The formula `(x - x_min) / alpha x 255` maps into `[0, 255]`, which is a `uint8` representation. Signed INT8 asymmetric quantization needs a zero point and usually clips to `[-128, 127]` or stores an unsigned integer with a zero point. As written, the asymmetric equation is inconsistent with the INT8 range in line 1068.
  - Proposed correction: Either label the asymmetric formula as `uint8`, or write the more general form `q = clamp(round(x / s) + z, q_min, q_max)` with `q_min`, `q_max`, scale `s`, and zero point `z`.

- **Line 1161: the FP32 versus BF16 energy claim is too specific for the appendix's own constants.**  
  FP32 uses `2x` the memory bytes of BF16, so the bandwidth statement is fine for raw data movement. But the energy constants used earlier give FP32 compute energy `3.7 pJ` versus FP16 compute energy `1.1 pJ`, about `3.4x`, not `2x`. BF16 compute energy is not separately shown here.
  - Proposed correction: Say "FP32 consumes 2x memory bandwidth and often higher compute energy than BF16" or provide a BF16 energy constant if the text wants a precise energy multiplier.

### Low Severity

- **Line 163: the explanation for FP32 versus FP16 energy says halving bits roughly halves energy, but the displayed ratio is about 3.4x.**  
  The local constants give `3.7 / 1.1 ~= 3.36`. The qualitative direction is correct, but the explanation understates the nonlinear impact of arithmetic bit width.
  - Proposed correction: Replace with "narrower arithmetic reduces switching and datapath energy" rather than implying exactly linear scaling.

- **Lines 253-256 and 301-304: ridge points are labeled `ops/byte` in one table and defined as `FLOP/byte` in the equation.**  
  This is harmless if "ops" means FLOPs, but the appendix also discusses INT8 TOPS and FP8 throughput, where operation conventions differ.
  - Proposed correction: Use `FLOP/byte` for FP16 roofline rows, or explicitly define `ops` as the operation unit used by the corresponding peak-throughput number.

- **Lines 853 and 375-380: HBM capacity is extracted as GiB but displayed as GB.**  
  The code uses `.m_as(GiB)` for H100 and TPU memory capacity but the table labels the result `GB`. This is a small unit mismatch; for 80 GiB, the decimal equivalent is about `85.9 GB`.
  - Proposed correction: Label the table as `GiB` or extract decimal `GB`.

- **Lines 1064-1066: FP16 dynamic range is approximate but mixes normal and rounded values.**  
  FP16's maximum finite value is about `6.55 x 10^4`. The smallest normal positive value is about `6.1 x 10^-5`, while subnormals extend to about `6 x 10^-8`. The table's lower bound `~10^-5` is not clearly one or the other.
  - Proposed correction: Use `~6 x 10^-5` for normal range, or state that subnormals extend lower.

## Verified Correct

- Lines 146-153: the speed-of-light round-trip examples are order-of-magnitude consistent with `~200 km/ms` in fiber. A U.S. cross-country distance of about `4,000 km` gives about `20 ms` one-way and `40 ms` round trip.
- Lines 189-193: the scaling-rule examples are arithmetically correct once the missing gradient term is added to the training-memory explanation: `7B x 2 bytes = 14 GB`, `7B x 1 byte = 7 GB`, `7B x 16 bytes = 112 GB`, `2 x 7B = 14 GFLOPs/token`, and `6 x 7B x 1T = 4.2 x 10^22 FLOPs`.
- Lines 204-207: the braking-distance conversion is correct: `100 km/h = 27.78 m/s`, so `10 ms` is about `0.278 m`, or `28 cm`.
- Lines 301-305 and 438: the roofline equations are dimensionally correct. `312 TFLOP/s / 2.039 TB/s ~= 153 FLOP/byte` for A100 using the local constants.
- Lines 357-359 and 442-444: the GEMM and ReLU roofline examples are internally consistent under the stated FP16 traffic assumptions. Square GEMM has intensity `n/3 = 4096/3 ~= 1365 FLOP/byte`; FP16 ReLU has `1 op / (2-byte read + 2-byte write) = 0.25 op/byte`.
- Lines 452-462: the iron-law dimensional analysis is correct: each term reduces to seconds.
- Lines 478-564: Amdahl's Law and the `s=0.05`, `n=8` example are correct. `1 / (0.05 + 0.95/8) ~= 5.9x`, and the infinite-processor cap is `20x`.
- Lines 576-665: Gustafson's Law examples are correct. With `s=0.05`, `n=8` gives `8 - 0.05 x 7 = 7.65x`, and `n=1000` gives about `950x`.
- Lines 704-747: the training-time equation example is internally consistent. `6 x 1B x 20B = 1.2 x 10^20 FLOPs`; one A100 at `312 TFLOP/s` and `40%` MFU gives about `1.25 x 10^14 FLOP/s`, so training time is about `9.6 x 10^5 s`, or `11 days`.
- Lines 767-820: Little's Law is applied correctly. `1000 QPS x 0.05 s = 50` concurrent requests; a `24 GB` accelerator with `1 GB/request` allows `24` concurrent requests and caps throughput at `24 / 0.05 = 480 QPS`.
- Lines 947-1046: aside from the rendered unit omission noted above, the bandwidth-latency arithmetic is correct: `1 KB x 8 / 10 Gbps = 0.8 us`, and `1 GB x 8 / 10 Gbps + 10 ms = 810 ms`.

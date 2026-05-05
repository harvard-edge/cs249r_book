# Math Audit: `book/quarto/contents/vol2/backmatter/appendix_dam.qmd`

Audit method: direct reasoning over the assigned source file only; no Gemini or external validation. Scope included all single-machine/DAM equations, numeric examples embedded in setup code, unit conversions, complexity/scaling claims, and prose-equation consistency.

## Findings

### 1. The "sum to max" pipelining equation is stated as an unconditional transformation

- **Lines:** 112-118, 127
- **Severity:** Medium
- **Issue:** The transformation from
  `T_sequential = D_vol/BW + O/(R_peak eta_hw) + L_lat`
  to
  `T_pipelined = max(D_vol/BW, O/(R_peak eta_hw)) + L_lat`
  is valid only for ideal overlap in steady state. It requires enough independent work, buffering, scheduling support, and no shared-resource contention. Finite batches also pay fill/drain overhead. As written, "transforms the sum into a max" and "converting the iron law's additive terms into overlapped terms" can overstate the achievable speedup.
- **Proposed correction:** Qualify the right-hand side as an ideal lower bound or best case. For example:
  `T_pipelined >= max(D_vol/BW, O/(R_peak eta_hw)) + L_lat`, with equality only under perfect overlap and no contention. Alternatively, state that this is the steady-state approximation after pipeline fill.

### 2. The arithmetic-intensity threshold is not the actual memory/compute boundary

- **Lines:** 131-138, 207
- **Severity:** Medium
- **Issue:** The footnote correctly defines the boundary by comparing arithmetic intensity to the hardware ridge point `R_peak/BW`. The later heuristic says `I < 100 FLOPs/byte` implies memory-bound. That may be a useful warning threshold, but it is not the mathematical boundary on many current accelerators. For example, using the file's H100-class FP16 peak value from lines 48 and 64, a typical H100 FP16 tensor peak near `989 TFLOP/s` with HBM bandwidth around `3.35 TB/s` gives a ridge point near `295 FLOPs/byte`; workloads between `100` and `295 FLOPs/byte` can still be memory-bound.
- **Proposed correction:** Reword the bullet as a conservative heuristic: "`I < 100 FLOPs/byte` is almost certainly memory-bound on modern datacenter GPUs; for a precise diagnosis, compare `I` to the device-specific ridge point `R_peak/BW`." If the appendix wants one H100-specific number, use roughly `300 FLOPs/byte` for FP16 tensor peak versus HBM bandwidth.

### 3. GPU utilization heuristics are too strong for D/A/M classification

- **Lines:** 135-136
- **Severity:** Medium
- **Issue:** Low GPU utilization does not by itself identify a Data bottleneck. It can also result from CPU launch overhead, small kernels, synchronization, insufficient batching, algorithmic serial work, or framework overhead. Conversely, high "GPU utilization" from coarse tools such as `nvidia-smi` does not prove a Machine/compute bottleneck; a workload can keep the GPU busy while being limited by HBM bandwidth or memory stalls.
- **Proposed correction:** Make the bullets explicitly provisional and add the needed confirming counters. For example: low utilization plus high data-wait time or low H2D/HBM feed rate suggests Data/CPU starvation; high SM occupancy plus high achieved FLOP/s relative to peak suggests compute-bound Machine saturation; high HBM bandwidth with low FLOP/s suggests memory-bound behavior.

### 4. The compute-bound row says memory bandwidth is "already saturated"

- **Lines:** 144-150
- **Severity:** Medium
- **Issue:** In a compute-bound workload, compute throughput is the binding resource; memory bandwidth is typically not the saturated limit. The table's wasted-optimization note, "More memory bandwidth (already saturated)," reverses the roofline interpretation.
- **Proposed correction:** Replace the wasted optimization with "More memory bandwidth (compute is the binding limit)" or "More memory bandwidth (not the limiting resource)." If a workload is saturating memory bandwidth, classify it as memory-bound, not compute-bound.

### 5. The latency-bound row overstates that neither compute nor bandwidth can help

- **Lines:** 148-150
- **Severity:** Low
- **Issue:** The table says that for latency-bound workloads, "Neither compute nor bandwidth" helps because overhead dominates. This is true only when the measured `L_lat` term is actually dominant. In many batch-1 inference cases, latency includes a mixture of launch overhead, serial dependencies, memory movement, and compute. Batching, fusion, and async dispatch target overhead, but faster kernels or reduced memory movement can still help if those terms are not negligible.
- **Proposed correction:** Change the wasted-optimization cell to "Blindly increasing compute or bandwidth before measuring the exposed `L_lat` term." This preserves the warning against optimizing the wrong term without claiming hardware resources never affect latency.

### 6. MFU is described as FLOPs divided by peak FLOPs rather than rates over the same interval

- **Lines:** 166-174
- **Severity:** Low
- **Issue:** MFU is a utilization ratio, so the numerator and denominator should be comparable rates, or totals over the same time interval. The prose says "achieved model FLOPs to the hardware's theoretical peak FLOPs," and the table writes `Achieved FLOPs / Peak FLOPs`. Read literally, that compares total work to a peak-work quantity that is not defined without a time interval.
- **Proposed correction:** Write MFU as `Achieved model FLOP/s / Peak hardware FLOP/s`, or as `Model FLOPs per step / (Peak FLOP/s * step time)`. The footnote can still note that the numerator counts useful model computation and excludes non-model overhead.

### 7. "Active Params" is not a general Algorithm-axis pass/fail grade

- **Lines:** 166-176, 209
- **Severity:** Medium
- **Issue:** The scorecard labels `Non-Zero Params / Total Params = 100 percent (Dense)` as failing and `< 50 percent (Sparse)` as passing. Dense models are not inherently algorithmically immature or inefficient, and sparse models are not automatically faster unless the sparsity pattern is exploitable by the runtime and hardware. This metric is a sparsity/compression metric, not a universal Algorithm-axis health score.
- **Proposed correction:** Rename the metric to "Sparsity / Compression Opportunity" and remove generic pass/fail language. If the goal is a general Algorithm score, use operation count or achieved quality per unit compute relative to a baseline. If the sparsity metric remains, qualify `<50 percent` as beneficial only when kernels/hardware exploit the sparse structure.

### 8. The strict MECE claim conflicts with shared terms and boundary cases

- **Lines:** 80-88, 90, 106-110, 131-138
- **Severity:** Low
- **Issue:** The text says every bottleneck maps to exactly one D/A/M axis, but the iron-law mapping explicitly gives Algorithm and Machine a shared compute term, and the arithmetic-intensity section identifies a Data/Machine boundary. Real incidents often decompose across multiple terms, even if one term is dominant.
- **Proposed correction:** Soften the taxonomy claim to first-order decomposition: "D/A/M provides a collectively exhaustive decomposition of bottleneck terms; in practice, incidents can involve interactions across axes, and the dominant term should be identified by measurement."

### 9. The embedded 7B/H100 setup computes low utilization but lacks the bandwidth data needed for a bottleneck diagnosis

- **Lines:** 43-65, 71
- **Severity:** Low
- **Issue:** The hidden setup computes a 7B FP16 forward pass as `2 * 7B = 14e9 FLOPs = 0.014 TFLOPs`, model size as `14 GB`, and at `50 ms` an achieved rate of `0.28 TFLOP/s`, less than `10 percent` of H100 FP16 peak. These calculations are internally consistent. However, low compute utilization alone does not distinguish memory-bound execution from overhead/latency-bound execution. The same `50 ms` implies only `14 GB / 0.050 s = 280 GB/s` of weight traffic if each parameter is streamed once, far below H100-class HBM bandwidth.
- **Proposed correction:** If these values are rendered elsewhere as a memory-bound example, add a measured HBM-bandwidth counter or change the conclusion to "low machine utilization; likely batch-1 latency/overhead or memory movement, requiring profiling to separate." If the code is unused, no arithmetic correction is needed.

## Checked Without Findings

- Lines 43-65: The embedded 7B FP16 arithmetic is internally consistent: `7B parameters * 2 FLOPs/parameter = 14e9 FLOPs = 0.014 TFLOPs`; `0.014 TFLOPs / 0.050 s = 0.28 TFLOP/s`; `7B * 2 bytes = 14 GB`.
- Lines 51-68: The loss-improvement calculation is correct: `(0.45 - 0.42) / 0.45 = 0.0667`, or about `6.7 percent`.
- Lines 96-102: The D/A/M reference table is dimensionally coherent as a high-level mapping: bandwidth constrains data movement, operation count constrains algorithmic work, and peak throughput constrains machine execution.
- Lines 108-110: The main iron-law equation is dimensionally consistent: `D_vol/BW`, `O/(R_peak eta_hw)`, and `L_lat` all have units of time when `eta_hw` is dimensionless.
- Lines 131-133: Arithmetic intensity as `FLOPs/byte` and the ridge-point comparison `R_peak/BW` are mathematically correct.
- Lines 170-174: The I/O overhead and active-parameter ratios are dimensionless and syntactically well formed, independent of the concern above about whether the active-parameter ratio is a universal health grade.
- Lines 184-198: The node-to-fleet scaling discussion is qualitative and does not contain a numeric scaling formula requiring correction. The claim that single-node MFU and fleet scaling efficiency should both be considered is prose-consistent.

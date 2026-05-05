# Math Audit Report: `book/quarto/contents/vol2/introduction/introduction.qmd`

## Checked scope

Audited `book/quarto/contents/vol2/introduction/introduction.qmd` for equations, generated numeric examples, unit conversions, scaling/reliability claims, complexity claims, and prose-equation consistency. Direct reasoning only; no Gemini or external verification used.

Checked items include:

- Lines 28, 103, 154-164, 172-227, 326, 368-370, 414-416, 797-804, 814-816, 847-849, 894, 902, 908-1011: scale, training-compute, failure-rate, bandwidth, and scaling-law claims.
- Lines 1019-1025, 1359-1404, 1417-1423, 1502-1510, 1554-1577, 1583-1669: displayed equations and generated worked examples.
- Lines 1753-1757, 1873-1906, 1916-1934, 1940-1969, 1977-1989, 1999-2004: framework counts, archetype numeric claims, complexity claims, and fallacy examples.

## Findings

### 1. Total training work is repeatedly labeled as FLOPS instead of FLOPs

- **Lines:** 103, 215, 219-227, 908, 963, 991, 1753, 1757
- **Severity:** Medium
- **Issue:** The chapter repeatedly uses "FLOPS" for total work such as `$10^{18}$`, `$10^{25}$`, `$10^{26}$`, and `3.14 x 10^23` operations. FLOPS is a rate, while these quantities are total floating-point operations, i.e. FLOPs. The footnote at line 991 correctly explains the distinction, but the surrounding prose and tables violate it.
- **Proposed correction:** Change total-work labels to "FLOPs" throughout these lines. Reserve "FLOPS" for throughput values such as A100 TFLOPS/s or TPU pod exaFLOPS.

### 2. Cluster-size trend prose says 4x/year, but the plotted data fit gives about 2x/year

- **Lines:** 233, 260-309, 326
- **Severity:** Medium
- **Issue:** The figure caption says the dashed trend line indicates approximately `4x` annual growth. Using the listed verified points and the code's own log-linear fit gives a per-year factor of about `2.00x`, not `4x`. Including the GPT-3 estimated point still gives about `2.01x`. The separate prose claim that size grew from 2 GPUs to 16,384 GPUs is about `8192x`, or `3.91` orders of magnitude, so "roughly four orders of magnitude" is fine.
- **Proposed correction:** Change the caption/commentary to "approximately 2x annual growth" unless the dataset or fitting window is changed to one that actually supports 4x/year.

### 3. GPT-3 synchronization worked example contradicts its own efficiency interpretation

- **Lines:** 1583-1669
- **Severity:** High
- **Issue:** The code computes a 700 GB synchronization and divides by 200 Gbps InfiniBand, i.e. `25 GB/s`, giving `700 / 25 = 28 s` communication time. With `t_compute_iter = 1.2 s`, the stated efficiency formula gives `1.2 / (1.2 + 28) = 4.1 percent`, not a regime where "most of the time is spent computing." For 100G Ethernet, the same calculation gives `56 s` communication time and `2.1 percent` efficiency. The generated percentages will therefore be very low while the prose says high bandwidth yields compute-dominated execution.
- **Proposed correction:** Either revise the prose to match the calculation ("even 200G HDR is insufficient for a 700 GB per-step transfer without parallel links, overlap, or sharding"), or revise the model to use aggregate effective bisection/injection bandwidth and realistic overlap, then recompute the efficiencies.

### 4. Network energy example says one million steps consume gigajoules, but the code gives megajoules

- **Lines:** 1594-1596, 1622-1642, 1668-1669
- **Severity:** Medium
- **Issue:** The code computes `700 GB * 8 bits/byte * 15 pJ/bit = 84 J` per synchronization. Over one million steps this is `84,000,000 J = 84 MJ = 0.084 GJ`, not "gigajoules" in the plural. The code comment also says the example should show about `50 MJ`, but the exported value is per-step joules, not training-run megajoules.
- **Proposed correction:** Change the prose to "tens of megajoules" for one million steps under the current formula, or explicitly multiply by the number of workers/links if the intended quantity is aggregate fabric-wide energy and then update the formula, units, and exported variable names.

### 5. Wall-clock time/utilization relation is inverted

- **Lines:** 991
- **Severity:** Medium
- **Issue:** The footnote says dividing work by throughput gives wall-clock time "only if multiplied by utilization." If `O` is work, `R_peak` is peak throughput, and utilization is `eta < 1`, actual time is `O / (R_peak * eta)`, equivalently `(O / R_peak) / eta`. Multiplying the ideal time by `eta` would make lower utilization appear faster.
- **Proposed correction:** Replace with "Dividing work by peak throughput gives ideal wall-clock time; actual time divides effective throughput by utilization, `T = O / (R_peak eta_hw)`."

### 6. Communication Intensity thresholds are written as dimensionless ratios

- **Lines:** 1417-1423
- **Severity:** Low
- **Issue:** The equation defines `CI = Bytes Transferred / FLOPs Executed`, which has units of bytes/FLOP. The thresholds `CI < 0.01` and `CI > 0.1` are presented as dimensionless. That is workable only if the units are made explicit or CI is normalized by a hardware balance point.
- **Proposed correction:** Write thresholds as `0.01 B/FLOP` and `0.1 B/FLOP`, or define a dimensionless normalized CI such as `(bytes/FLOP) / (network_balance bytes/FLOP)`.

### 7. Iron law of scale can produce impossible negative communication time unless overlap is capped

- **Lines:** 1554-1563
- **Severity:** Low
- **Issue:** `T_step(N) = T_compute/N + T_comm(N) - T_overlap` is dimensionally valid, but only if `T_overlap <= T_comm(N)` or if overlap cannot hide more communication than exists. As written, large `T_overlap` can make the communication contribution negative and overstate speedup.
- **Proposed correction:** Use `T_step(N) = T_compute/N + max(T_comm(N) - T_overlap, 0) + T_coord(N)`, or state the constraint `0 <= T_overlap <= T_comm(N)`.

### 8. Archetype A parameter scale is internally inconsistent

- **Lines:** 1916, 1932, 1950
- **Severity:** Medium
- **Issue:** The prose first describes GPT-4/Llama-3 scale as "hundreds of billions to trillions of parameters," but then says the fleet challenge is partitioning `100+ trillion` or `100--trillion-parameter` models. Those are different scales by roughly two orders of magnitude at the lower end.
- **Proposed correction:** If the intended archetype is GPT-4/Llama-3 scale, use "100B+ to trillion-scale parameters" or "hundreds of billions to trillions of parameters." Reserve "100T+" only for a separate future-model archetype and align the table accordingly.

### 9. Linear-scaling fallacy example arithmetic does not match the stated overhead

- **Lines:** 1987-1989
- **Severity:** Low
- **Issue:** The example says a team predicts a `3x` improvement but achieves only `1.3x` because coordination and communication overhead consumes 40 percent of compute time. If 40 percent of runtime is non-scaling overhead and the other 60 percent improves by `3x`, Amdahl-style speedup is `1 / (0.4 + 0.6/3) = 1.67x`, not `1.3x`. A `1.3x` result would require about 65 percent non-scaling overhead under the same model.
- **Proposed correction:** Change `1.3x` to about `1.7x`, or change the overhead to about 65 percent, or state that overhead grows with scale rather than remaining a fixed 40 percent.

### 10. Structured pruning speedup is overstated as guaranteed

- **Lines:** 1977
- **Severity:** Low
- **Issue:** Reducing structured work by 50 percent can expose an ideal `2x` arithmetic speedup, but wall-clock speedup is not guaranteed because kernels, memory traffic, launch overhead, and hardware support still matter. This sentence conflicts with the paragraph's broader point that FLOP reductions do not necessarily predict deployment latency.
- **Proposed correction:** Replace "guarantees 2x speedup" with "can expose up to a 2x arithmetic speedup on supporting kernels."

## Verified calculations and consistency checks

- Lines 23 and 38-43: `\mlfleetstack{30}{30}{30}{30}` has four arguments, and the learning objectives list six principles consistently with the later six-item list on lines 1899-1906.
- Lines 154-164: `25,000 * 0.08 = 2,000` annual failures; `2,000 / 365 = 5.48` failures/day; `24 / 5.48 = 4.38` hours MTBF.
- Line 172: `64 TPUs * 4 days * 24 hours/day = 6,144` chip-hours.
- Lines 215-227: `10^25 / 10^18 = 10^7`, so the "10-million-fold" and "seven orders of magnitude" claims are internally consistent when comparing AlexNet to GPT-4.
- Lines 326: `16,384 / 2 = 8192`, which is about `10^3.91`, so "roughly four orders of magnitude" is correct.
- Lines 388-416: `200 Gb/s / 8 = 25 GB/s` and `400 Gb/s / 8 = 50 GB/s`, so the InfiniBand conversions are correct.
- Lines 882-887 and 894: `175B parameters * 4 bytes = 700 GB`, correct for FP32 gradients. Lines 1627-1629 also compute `2 * 175B * 2 bytes = 700 GB`, correct for a ring all-reduce approximation with FP16 payloads.
- Lines 1009: `D proportional to N^0.74` supports the prose "roughly three-quarters the rate" in log-scaling terms.
- Lines 1021-1025: `L(N) = A N^-alpha + B` is prose-consistent: larger positive `alpha` decays faster toward the floor `B`.
- Lines 1359-1404: `(0.999)^1000 * 100 = 36.77 percent`; `(0.999)^10000 * 100 = 0.00452 percent`; `(0.9999)^10000 * 100 = 36.79 percent`, matching the reliability-collapse prose and figure caption.
- Lines 1504-1508: The Fleet Law and fleet-efficiency equation are dimensionally coherent; all numerator and denominator terms are times.
- Lines 1566-1573: Fleet energy efficiency has units of FLOPs/J, and scaling efficiency `T_compute / (N * T_step)` is dimensionless if `T_compute` denotes single-device compute time for the same work.
- Lines 1575-1578: If an irreducible communication component is 20 percent of runtime, Amdahl's Law gives maximum speedup `1 / 0.2 = 5x`, as stated.
- Lines 1873-1879 and 1899-1906: The Five-Pillar Framework and Six Systems Engineering Principles counts match their lists.
- Lines 1985: `120 km/h = 33.33 m/s`; over `100 ms`, travel distance is `3.33 m`, correct.
- Lines 1977: Reducing a model from 10B to 3B parameters is a `70 percent` parameter/FLOP reduction under a proportional-FLOP assumption.

# Math Audit: `book/quarto/contents/vol2/backmatter/appendix_c3.qmd`

Audit method: direct reasoning over the assigned source file only; no Gemini or external validation. Scope included all C3 taxonomy equations, embedded numeric examples, unit conversions, complexity/scaling claims, and prose-equation consistency.

## Findings

### 1. Fleet efficiency is defined as compute-time fraction, then used as if it includes MFU

- **Lines:** 230-234, 392-394, 425-429
- **Severity:** High
- **Issue:** The appendix defines `eta_fleet = T_Computation / T_step`. This is an exposed compute-time fraction: it measures how much step wall time is spent in local arithmetic rather than communication or coordination. It does not measure whether that local arithmetic uses the GPU efficiently. A run with poor kernels could have `T_Computation / T_step = 1` and still have low MFU. Later, line 392 says scaling laws assume `eta_fleet = 1.0`, meaning "no communication overhead, no coordination losses, and perfect MFU"; line 394 then says effective work survives MFU, scaling efficiency, and goodput losses. That is inconsistent with the earlier equation because MFU is not part of `T_Computation / T_step`.
- **Proposed correction:** Rename the line-232 metric to something like `f_compute = T_Computation / T_step` or `eta_time`, and reserve fleet/effective efficiency for the multiplicative product:
  `eta_effective = MFU * eta_scaling * goodput`.
  Then rewrite line 392 to say scaling laws assume `eta_effective = 1`, or explicitly say they assume both `f_compute = 1` and `MFU = 1`.

### 2. The Fleet Law is additive only for exposed, non-overlapped critical-path time

- **Lines:** 216-220, 240-244, 463, 471-473
- **Severity:** Medium
- **Issue:** The equation `T_step = T_Computation + T_Communication + T_Coordination` is dimensionally valid as a decomposition of exposed critical-path time, but the appendix later discusses compute-communication overlap where a sequential sum can become closer to `max(T_comp, T_comm)`. Exercise 1 explicitly contrasts `T_comp + T_comm` with overlapped `max(T_comp, T_comm)`. Without qualifying the Fleet Law, readers may treat all communication as additive even when some or all of it is hidden under backward computation.
- **Proposed correction:** State the Fleet Law as an exposed-time decomposition, for example:
  `T_step = T_comp + T_comm,exposed + T_coord,exposed`,
  where overlapped communication is charged only to the visible critical path. Alternatively, keep the original equation but add that each term represents measured non-overlapped wall-clock contribution after overlap.

### 3. Case 3's coordination-overhead breakdown sums to 23 percent, not 40 percent

- **Lines:** 74-82, 109, 334, 338, 344-347
- **Severity:** High
- **Issue:** Case 3 sets `goodput_ratio = 0.60`, so `case3_coord_fraction = 1 - 0.60 = 0.40`, rendered as 40 percent coordination overhead. But the detailed overhead constants cited in the diagnosis are 10 percent failure recovery, 5 percent pipeline bubbles, 3 percent checkpoints, and 5 percent maintenance. These sum to 23 percent, leaving 17 percentage points unexplained. The separate `goodput_all` calculation on lines 88-91 uses this same 23 percent overhead sum and therefore gives 77 percent goodput, not the 60 percent used in Case 3.
- **Proposed correction:** Either change Case 3 goodput to 77 percent if the four listed overhead constants are intended to be exhaustive, or add an explicit additional 17 percent category such as scheduler/preemption/straggler idle time. The prose should make the overhead components sum to the stated 40 percent coordination loss.

### 4. Effective-FLOPS section applies an 8,192-GPU scaling-efficiency constant to a 100,000-GPU cluster

- **Lines:** 84-96, 149-150, 416-421, 431-437
- **Severity:** Medium
- **Issue:** The 100,000-GPU example computes peak throughput correctly as `100,000 * 989 TFLOP/s = 98,900 PFLOP/s`. It then multiplies by `SCALING_EFF_8192GPU = 0.35` and describes that factor as "scaling efficiency at this cluster size." But the constant is explicitly the 8,192-GPU reference value, not a 100,000-GPU value. The arithmetic is internally consistent under the chosen factor: `98,900 * 0.50 * 0.35 * 0.77 = 13,327.8 PFLOP/s`, an effective fraction of `13.5%` and tax of about `7.4x`. The problem is the cluster-size label and implied interpretation.
- **Proposed correction:** Either describe the factor as an illustrative 35 percent fleet-scale scaling-efficiency assumption, or introduce a separate `SCALING_EFF_100KGPU` value. If the text keeps the 8,192-GPU constant, do not call it "at this cluster size."

### 5. FLOP budgets and FLOP/s rates are mixed in the scaling-law discussion

- **Lines:** 394, 425-437, 513, 519
- **Severity:** Medium
- **Issue:** The effective-FLOPS callout computes a throughput rate in PFLOP/s. Scaling-law compute budgets such as `10^24` are cumulative FLOPs, not FLOPS rates. Exercise 3 asks how many "raw peak FLOPS" must be provisioned to deliver `10^24` FLOPS of training compute; read literally, this asks for a rate to satisfy a cumulative-work target. The multiplicative C3 tax can apply to either rates or total work, but the units should stay consistent.
- **Proposed correction:** Use "FLOPs" for cumulative training compute budgets and "FLOP/s" or "PFLOP/s" for throughput. In Exercise 3, write: "If a scaling law predicts `10^24` FLOPs of effective training compute, how many raw peak FLOPs must be budgeted?" The answer should say `4.7 * 10^24` raw peak FLOPs, not raw peak FLOPS.

### 6. Strict MECE language conflicts with the appendix's own intersection examples

- **Lines:** 176-180, 185, 216-228, 236-256, 353-360
- **Severity:** Medium
- **Issue:** The appendix says every fleet-scale problem maps to "mutually exclusive and collectively exhaustive" axes. But the Fleet Law components and the later intersection section acknowledge coupled cases: AllReduce is both data transfer and synchronization, stragglers are Communication intersecting Coordination, and pipeline bubbles are Computation intersecting Coordination. The taxonomy can still be useful, but the axes are not strictly mutually exclusive at the incident level unless each measured wall-clock interval is assigned to exactly one exposed term by convention.
- **Proposed correction:** Soften the claim to "collectively exhaustive first-order axes" and say incidents may involve intersections. If strict additivity is desired, define an attribution convention for measured time, such as assigning each exposed interval to its dominant limiting mechanism.

### 7. The communication-computation ratio rule assumes perfect overlap and sufficient compute efficiency

- **Lines:** 240-244
- **Severity:** Low
- **Issue:** The rule says when `rho = T_comm / T_comp < 1`, the network transfer can be overlapped, the system is compute-bound, and healthy. `rho < 1` is only a necessary condition for full hiding under ideal overlap; it also requires the framework to launch communication early enough, independent resources, and no contention. It also does not prove the system is healthy, because `T_comp` can be long due to poor MFU or inefficient kernels.
- **Proposed correction:** Reword as: "`rho < 1` means communication is in principle hideable if overlap is implemented and compute is efficient; remaining diagnosis should check MFU and timeline overlap. `rho > 1` means some communication remains exposed even under ideal overlap."

### 8. Exercise 2's "near zero" overlap claim overstates the maximum speedup

- **Lines:** 467-473
- **Severity:** Low
- **Issue:** The exercise has `T_comp = 100 ms`, `T_comm = 60 ms`, and `T_coord = 40 ms`. It says overlapping AllReduce could reduce the visible 60 ms to near zero and push `eta_fleet` toward `100/140 = 0.71`. The final number is correct for fully hiding communication while leaving coordination exposed: `100 / (100 + 40) = 0.714`. But "visible 60 ms to near zero" is only possible if the full 60 ms fits under the 100 ms compute window and there is no resource contention; otherwise the exposed communication is `max(0, T_comm - overlap_window)`.
- **Proposed correction:** Qualify the sentence: "with perfect overlap, the visible communication term could fall to zero because 60 ms fits within the 100 ms compute window, yielding `100/140 = 0.71`; partial overlap would give a smaller gain."

### 9. Exercise 3 uses a 1,024-GPU scaling-efficiency reference for a 2,048-GPU cluster

- **Lines:** 475-519
- **Severity:** Low
- **Issue:** The exercise states that the team provisions 2,048 H100 GPUs, but the hidden computation uses `SCALING_EFF_1024GPU = 0.50`. The rendered prompt says the cluster achieves 50 percent scaling efficiency, so the arithmetic `0.50 * 0.50 * 0.85 = 0.2125` is correct. The mismatch is that the code source for the 50 percent value is the 1,024-GPU reference constant, which can confuse maintenance and makes the example inconsistent with its own cluster size.
- **Proposed correction:** Either change the exercise to 1,024 GPUs or define the 50 percent scaling efficiency directly in the exercise code as a 2,048-GPU assumption.

## Checked Without Findings

- Lines 84-96 and 431-437: Given the stated constants, the 100,000-GPU effective-throughput arithmetic is correct: peak is `98,900 PFLOP/s`; goodput from listed overheads is `1 - (0.05 + 0.03 + 0.10 + 0.05) = 0.77`; effective throughput is `98,900 * 0.50 * 0.35 * 0.77 = 13,327.8 PFLOP/s`; effective fraction is `13.5%`; C3 tax is `98,900 / 13,327.8 = 7.4x`.
- Lines 100-107 and 308: Case 1's MFU improvement calculation is correct: raising MFU from 15 percent to 50 percent gives `0.50 / 0.15 = 3.33x` more useful model work if other factors are unchanged.
- Lines 104-107 and 328: Case 2's Amdahl-style upper bound is correct under the stated communication fraction: if 55 percent of step time is communication and 45 percent remains, eliminating communication entirely gives `1 / 0.45 = 2.22x`, rendered as `2.2x`.
- Lines 218 and 226-228: The three Fleet Law terms are dimensionally time quantities and the activity mapping is conceptually coherent once interpreted as exposed wall-clock contributions.
- Lines 266-272 and 380-386: The traffic-light and scorecard thresholds are dimensionless percentages/fractions. The table entries are internally consistent as heuristics, independent of the concerns above about metric naming and strict MECE language.
- Lines 497-519: Exercise 3's numeric tax calculation is correct for the rendered assumptions: `0.50 * 0.50 * 0.85 = 0.2125`, or `21.25%` effective fraction, and `1 / 0.2125 = 4.7059x`, rendered as about `4.7x`.

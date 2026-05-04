# Math Audit: `book/quarto/contents/vol2/fault_tolerance/fault_tolerance.qmd`

Scope: reliability, MTBF/FIT, checkpointing, recovery-time, probability, availability, scaling claims, unit conversions, numeric examples, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. FIT conversion is off by 1,000x in the failure-rate footnote

- **Lines:** 147, 320-324
- **Severity:** High
- **Issue:** The footnote says a 50,000-hour MTBF GPU has `20` FIT, but the FIT formula later correctly gives `10^9 / 50,000 = 20,000` FIT.
- **Explanation:** Since 1 FIT is one failure per billion device-hours, `lambda_FIT = 10^9 / MTBF_hours`. For 50,000 hours this is `20,000` FIT. A 10,000-GPU cluster then accumulates `200,000,000` FIT, not `200,000` FIT. The final expected failure interval of 5 hours is correct only with `200,000,000` FIT.
- **Proposed correction:** Change line 147 to `20,000 FIT` and `200,000,000 FIT`, while keeping the 5-hour cluster MTBF.

### 2. The "9s of reliability" example mixes availability with a once-per-year failure model

- **Lines:** 269-276, 294-302
- **Severity:** Medium
- **Issue:** The problem states each GPU has `99.99 percent` availability, but the calculation uses an hourly survival probability of `1 - 1/8760`, corresponding to one failure per year, not 99.99% availability.
- **Explanation:** Instantaneous independent availability would give `0.9999^10000 = 0.368`, similar but not identical to the displayed `0.32`. Survival for a one-hour interval requires a failure rate or MTBF, not just annual availability; availability also depends on repair time.
- **Proposed correction:** Either frame the example as "each GPU has 1-year MTBF" and remove the 99.99% availability wording, or compute cluster instantaneous availability as `0.9999^10000 = 37%` and avoid calling it a one-hour survival probability.

### 3. GPT-3 checkpoint byte accounting is internally inconsistent

- **Lines:** 75-80, 99-102, 117-120, 1915, 2388-2390, 2493
- **Severity:** High
- **Issue:** The setup comment says `weights (2 bytes FP16) + Adam m (4 bytes) + Adam v (4 bytes) = 12 bytes/param`, but that sum is `10 bytes/param`, not 12.
- **Explanation:** The computed `2.1 TB` checkpoint and `2.1 GB` shard values rely on `12 bytes/param`. That value could be valid for a richer checkpoint layout, such as FP16 weights plus FP32 master weights plus two FP32 Adam moments plus perhaps gradients/metadata depending on the convention, but it is not the layout described in the comment and prose.
- **Proposed correction:** If the intended checkpoint is weights plus two FP32 Adam moments, set `bytes_full_ckpt = 10`, yielding `1.75 TB` for 175B parameters and `1.75 GB` per 1,000-way shard. If `2.1 TB` is intended, explicitly state the additional `2 bytes/param` being checkpointed.

### 4. Young-Daly overhead percentages in the 16,000-GPU example are too low

- **Lines:** 1927-1946
- **Severity:** High
- **Issue:** The optimal interval calculation `sqrt(2 * 2 min * 180 min) = 26.8 min` is correct, but the listed overheads are not.
- **Explanation:** With the chapter's own cost model, total waste is `C/tau + tau/(2M)`. At 10 minutes, waste is `2/10 + 10/360 = 22.8%`, not `17%`. At 60 minutes, waste is `2/60 + 60/360 = 20.0%`, not `15%`. At the 26.8-minute optimum, waste is about `14.9%`, not `7%`.
- **Proposed correction:** Update the bullets to approximately `23%`, `20%`, and `15%` total overhead. If the intent is to report only one component of waste, label it explicitly and adjust the comparison.

### 5. One-hour checkpoint warning understates the rework waste

- **Lines:** 2084-2090, 2108-2112
- **Severity:** Medium
- **Issue:** The notebook computes the optimal total checkpoint tax as about `5.6%`, but the prose says checkpointing every hour would waste "nearly 10 percent."
- **Explanation:** With `T_save = 21 s`, `MTBF = 3.69 h`, and `tau = 1 h`, the total waste is `21/3600 + 3600/(2*13284) = 14.1%`. The expected rework term alone is `13.6%`.
- **Proposed correction:** Change "nearly 10 percent" to "about 14 percent" or recompute the example with assumptions that actually produce 10%.

### 6. Checkpoint-debug example uses BF16 optimizer states while describing Adam states generically

- **Lines:** 2221-2232, 2245, 2273
- **Severity:** Medium
- **Issue:** The 70B checkpoint calculation uses `140 GB` weights plus `280 GB` optimizer state for a `420 GB` total, which assumes the two Adam moment vectors are stored at 2 bytes each.
- **Explanation:** Elsewhere the chapter treats Adam `m` and `v` as 4 bytes each. Under that convention, a 70B checkpoint would be `140 GB + 560 GB = 700 GB`, not `420 GB`.
- **Proposed correction:** State that this example assumes BF16/compressed optimizer checkpointing, or change `optimizer_gb = weights_gb * 4` and update the total and write-time numbers.

### 7. Per-node checkpoint contention calculation overstates the serialized time

- **Lines:** 2238-2240, 2255-2256, 2288
- **Severity:** Medium
- **Issue:** The prose divides the full `420 GB` checkpoint by a single node's bandwidth share, producing about `358 minutes` per node. But in the stated 64-node write pattern, each node should write only its shard if the checkpoint is distributed.
- **Explanation:** With 64 equal shards, each node writes `420/64 = 6.56 GB`. At `1.25/64 = 0.0195 GB/s`, each shard takes `6.56/0.0195 = 336 s = 5.6 min`, the same as the aggregate lower bound. The full-checkpoint-per-node calculation would imply every node writes a complete 420 GB checkpoint, which is not what the example says.
- **Proposed correction:** Replace the serialized-time statement with per-shard math, or explicitly say the pathological case is each node redundantly writing the full checkpoint.

### 8. Recovery load time conflicts with the shard-size and bandwidth numbers

- **Lines:** 118-120, 2473-2481, 2497-2502, 2514
- **Severity:** High
- **Issue:** The recovery budget says each of 1,000 workers reads its `2.1 GB` shard at `5 GB/s`, but assigns `T_load = 21 seconds`.
- **Explanation:** `2.1 GB / 5 GB/s = 0.42 seconds`, not 21 seconds. The 21-second value matches reading a full `2.1 TB` checkpoint at `100 GB/s` aggregate, not per-worker local NVMe reads. Line 2514 then says warm restart saves a `50-second T_load`, introducing a third conflicting value.
- **Proposed correction:** Choose one recovery model. For sharded local reads at 5 GB/s, use `T_load ~= 0.4 s`. For aggregate shared-storage reads, keep `21 s` but remove the per-worker `5 GB/s` explanation. Update the warm-restart prose to the same value.

### 9. Recovery impact converts percent overhead to GPU-hours incorrectly

- **Lines:** 2473-2488, 2502-2504
- **Severity:** High
- **Issue:** The daily lost wall-clock time at 10,000 GPUs is correctly about `31 minutes`, but the equivalent wasted compute is listed as `210 GPU-hours every day`.
- **Explanation:** `31 minutes * 10,000 GPUs / 60 = 5,167 GPU-hours`, roughly `5,000 GPU-hours/day`. This also equals about `2.1%` of a `10,000 * 24 = 240,000 GPU-hour` day. `210 GPU-hours` is off by about 24x.
- **Proposed correction:** Change the value to approximately `5,000 GPU-hours every day`, or if `210` is intended, label it as average wasted GPUs over the day (`0.021 * 10,000 = 210 GPUs`), not GPU-hours.

### 10. Correlation-factor example treats grouped failures as reducing MTBF by the group size without defining the rate model

- **Lines:** 3354-3364
- **Severity:** Low
- **Issue:** The text says if failures correlate with factor 10, effective MTBF drops from 10 hours to 1 hour.
- **Explanation:** If the event rate stays the same but each event takes down 10 GPUs, the time between failure events remains 10 hours; the impact per event increases. MTBF drops to 1 hour only if the correlated domains also create ten times as many system-failing events, or if the metric is redefined as component-failure equivalents rather than event MTBF.
- **Proposed correction:** Clarify the assumed model, e.g. "if correlated domains make system-failing events 10x more likely" or "the lost-GPU-equivalent failure rate increases by 10x."

## Checks That Look Consistent

- Series reliability and MTBF equations are consistent under independent exponential failures: `R_system = exp(-N lambda t)` and `MTBF_system = MTBF_component/N` (lines 155-165).
- The detailed 10,000-GPU subsystem MTBF example is arithmetically consistent: per-GPU failure rate `2.708e-5/h`, cluster rate `0.2708/h`, and MTBF `3.69 h` (lines 359-375).
- The HBM FIT example is internally consistent under its assumptions: `1,024 * 80 GB * 8 * 1,024 = 671M Mb`; at `250 FIT/Mb`, MTBF is about 21 seconds, and a 100x ECC reduction gives about 36 minutes (lines 339-343).
- The SDC probability example is consistent: `100,000 * 2/3600 = 55.6 GPU-hours`, and `1 - (1 - 1e-6)^55.6 = 0.0056%`, implying about one event per 18,000 two-second steps (lines 1711-1740).
- Serving availability replication math is correct for independent replicas: `99% -> 99.99%` with two replicas and `99.9999%` with three replicas (lines 2799-2809).

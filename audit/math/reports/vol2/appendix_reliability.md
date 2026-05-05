# Math Audit: `book/quarto/contents/vol2/backmatter/appendix_reliability.qmd`

Scope: reliability, availability, MTBF/MTTF/FIT, probability, checkpoint sizing, Young-Daly checkpoint intervals, recovery budgets, unit conversions, scaling claims, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. The Young-Daly formula is labeled as Young-Daly but implements only Young's first-order approximation

- **Lines:** 37-39, 112, 381-387, 449-501, 715
- **Severity:** Medium
- **Issue:** The text repeatedly calls `tau_opt = sqrt(2 delta M)` the "Young-Daly formula", but the helper being called implements Young's first-order formula, not Daly's corrected interval.
- **Explanation:** Daly's higher-order correction is commonly written as `tau = sqrt(2 delta M) + delta` under the same notation. With the appendix's 175B example, `delta = 28 s` and `M = 4.1379 h`, so Young gives `15.22 min`; adding `delta` gives about `15.69 min`. The numerical difference here is small, but the naming is mathematically imprecise and could confuse readers comparing formulas.
- **Proposed correction:** Rename the displayed equation and surrounding prose to "Young's first-order checkpoint interval" or explicitly note that this appendix uses the common Young approximation to the Young-Daly model and omits Daly's `+ delta` correction.

### 2. The Young-Daly boundary-condition explanation overstates what `delta >= M` implies

- **Lines:** 523
- **Severity:** Medium
- **Issue:** The prose says when `delta >= M`, the formula yields `tau_opt >= M`, "meaning you would lose more than one MTBF interval of work per failure."
- **Explanation:** The first clause is true: `sqrt(2 delta M) >= sqrt(2) M` when `delta >= M`. But the expected replayed work per failure is `tau/2`, not `tau`. At `delta = M`, expected replay is `sqrt(2)M/2 = 0.707M`, not more than one MTBF. Expected replay exceeds one MTBF only when `tau/2 > M`, i.e. `delta > 2M`.
- **Proposed correction:** Replace the explanation with: "When `delta` approaches `M`, the optimal interval is on the order of the MTBF and checkpoint overhead is so large that checkpoint/restart is no longer effective. Expected replay exceeds one MTBF only for `delta > 2M` under this simplified model."

### 3. Checkpoint sizing includes gradients, but the prose does not justify checkpointing ephemeral gradient buffers

- **Lines:** 395-403, 443
- **Severity:** Medium
- **Issue:** The 16 bytes/parameter calculation includes 2 bytes for BF16 gradients. In most training checkpoint layouts, gradients are transient and not required to restart from the saved model and optimizer state.
- **Explanation:** Without gradients, the listed state is `2 + 4 + 4 + 4 = 14 bytes/param`, giving a 175B checkpoint of `2,450 GB` and a 100 GB/s write time of `24.5 s`. Including gradients gives the appendix's `2,800 GB` and `28.0 s`. The arithmetic is internally consistent, but the model assumption is not explained and conflicts with the usual "weights + optimizer state" checkpoint interpretation.
- **Proposed correction:** Either state that this is a full training-state snapshot that intentionally persists gradient buffers, or remove gradients and use `14 bytes/param` with updated table and Young-example values.

### 4. The checkpoint-size table computes a 1T model row but does not render it

- **Lines:** 76-77, 421-443
- **Severity:** Low
- **Issue:** `R.model_sizes_params` and `R.model_labels` include `1e12` / `"1T"`, but @tbl-checkpoint-size renders only `ckpt_data[0]` through `ckpt_data[3]`.
- **Explanation:** The generated data include a 1T checkpoint size of `16,000 GB` and a write time of `160.0 s` under the appendix's 16 bytes/parameter and 100 GB/s assumptions. Omitting that row is inconsistent with the stated model list and with the comment saying the table covers `7B-1T` models.
- **Proposed correction:** Add a fifth table row for `ckpt_data[4]`, or remove the 1T entry from `model_sizes_params`, `model_labels`, and the table comments.

### 5. Goodput "ratio" equation has throughput units, not ratio units

- **Lines:** 580-584
- **Severity:** Medium
- **Issue:** The displayed equation defines `Goodput Ratio = Useful Steps / Wall-Clock Time`, which has units of steps/time.
- **Explanation:** `Useful Steps / Wall-Clock Time` is goodput throughput. A dimensionless goodput ratio would be something like `Useful Steps / Total Executed Steps`, or `Goodput Throughput / Rawput Throughput`.
- **Proposed correction:** Rename the equation to `Goodput = Useful Steps / Wall-Clock Time`, or define `Goodput Ratio = Useful Steps / Raw Executed Steps`.

### 6. The 5% checkpoint and 5% recovery overhead bullets are constants, not derived from the appendix's worked example

- **Lines:** 79-80, 588-592
- **Severity:** Low
- **Issue:** The goodput section lists `~5 percent` checkpoint overhead and `~5 percent` recovery overhead from hard-coded constants, but the worked 10K/175B Young calculation gives checkpoint overhead about `3.1%`.
- **Explanation:** With `delta = 28 s` and `tau = 15.22 min`, checkpoint overhead is `28 / (15.22*60) = 3.07%`. At the optimum, the expected lost-work term `tau/(2M)` is also `3.07%` before adding detection, rescheduling, and reload. The prose's `5% + 5%` may be a generic budget, but it is presented in the same appendix immediately after the worked example without saying the assumptions changed.
- **Proposed correction:** Either derive these bullets from the current scenario (`~3%` checkpoint write overhead and `~3%` expected replay overhead, plus fixed recovery phases), or label them explicitly as broad operational budget placeholders rather than values from the worked example.

## Checked But No Issue

- **FIT conversion:** Lines 93-100 and 195-198 correctly use `FIT = 10^9 / MTTF_hours` and `MTTF = 10^9 / FIT`.
- **Node MTBF cascade:** Lines 87-91 and 246-256 are consistent for independent exponential components. With 8 GPUs at 50,000 h, 2 NICs at 150,000 h, and 2 PSUs at 100,000 h, `MTBF_node = 1/(8/50000 + 2/150000 + 2/100000) = 5,172 h`.
- **10K cluster MTBF:** Lines 102-104 and 256 are consistent: `10,000 / 8 = 1,250` nodes, so `5,172 h / 1,250 = 4.14 h`.
- **Failure probabilities:** Lines 319 and 339-351 use the correct exponential formula `1 - exp(-T/MTBF)`. For the 10K cluster, the 24-hour probability is `1 - exp(-24/4.1379) = 99.7%`, so displaying `> 99.9%` for week/month but not for one day is consistent with the thresholds.
- **Availability stacking:** Lines 633-680 correctly compute independent replica availability. Starting from `A = 0.99`, two replicas give `99.99%` and `52.6 minutes/year` of downtime; three replicas give `99.9999%` and about `0.53 minutes/year`.
- **Fallacy calculation:** Lines 688-690 are arithmetically correct: `0.9999^10000 = 0.3679`, so the complement is about `63%`.

No source `.qmd` files were modified.

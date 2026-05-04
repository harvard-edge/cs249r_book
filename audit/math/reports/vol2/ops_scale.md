# Math Audit: `book/quarto/contents/vol2/ops_scale/ops_scale.qmd`

Scope: operations-at-scale, SLO/incident/reliability/cost calculations, unit conversions, scaling/statistical claims, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. Platform ROI notebook comment claims 12x, but the calculation and prose give 1.25x

- **Lines:** 297-332
- **Severity:** Low
- **Issue:** The notebook header says `roi_ratio ≈ 12x`, while the code computes `50 * 20 * $150 / $120,000 = 1.25`, and the prose correctly reports a 25 percent return.
- **Explanation:** A 12x ROI would require either far larger savings or much lower platform cost. The visible result is internally consistent at 1.25x; the stale comment is inconsistent with the executed example.
- **Proposed correction:** Change the line 298 comment to `Show: roi_ratio = 1.25x`.

### 2. Shared-GPU annual savings estimate is low for the stated utilization change

- **Lines:** 911-912
- **Severity:** Medium
- **Issue:** The text says moving a 100-GPU organization from 35 percent to 75 percent effective utilization at `$2/GPU-hour` saves approximately `$700,000` annually.
- **Explanation:** For a fixed workload of 35 active GPU-equivalents, dedicated capacity at 35 percent utilization needs 100 GPUs and costs `100 * $2 * 8760 = $1.752M/year`. At 75 percent utilization, the same workload needs `35/0.75 = 46.7` GPUs and costs about `$818K/year`, saving about `$934K/year`. If the organization keeps all 100 GPUs, there is no direct spend reduction; the benefit is increased effective capacity, not annual savings.
- **Proposed correction:** Either change the savings to about `$930K/year` under a capacity-reduction model, or reframe the claim as effective capacity gain rather than cash savings.

### 3. Shadow-deployment z-test denominator is computed for the wrong sample size

- **Lines:** 1679-1683
- **Severity:** High
- **Issue:** The example says each model processes `100K` requests, but computes the standard error as `0.00016` and `z = 31.25`.
- **Explanation:** With pooled proportion `0.0125` and `n_A = n_B = 100,000`, the standard error is `sqrt(0.0125 * 0.9875 * (2/100000)) = 0.000497`, so `z = 0.005 / 0.000497 = 10.1`. The displayed `0.00016` denominator corresponds roughly to `1,000,000` samples per variant, not `100,000`.
- **Proposed correction:** Either keep `100K` and change the denominator and z-score to approximately `0.00050` and `10.1`, or change the sample size to `1M` per model and keep the displayed z-score.

### 4. Full-shadow error count is off by 10x

- **Lines:** 1687-1701
- **Severity:** Medium
- **Issue:** During full shadow, the model processes all `5M` daily transactions and flags `0.02 percent` as errors, but the text says this is `100 transactions/day`.
- **Explanation:** `0.02 percent = 0.0002`; `5,000,000 * 0.0002 = 1,000`, not 100. The `100/day` number would match `0.002 percent` of 5M or `0.02 percent` of the earlier 500K sampled-shadow volume.
- **Proposed correction:** Change `100 transactions/day` to `1,000 transactions/day`, or change the error rate to `0.002 percent`.

### 5. Risk category caption introduces numeric thresholds not present in the table or formula

- **Lines:** 2028-2049
- **Severity:** Low
- **Issue:** The rollout-risk equation is numeric, but the table columns contain qualitative labels; the caption then claims low risk is under `0.1` and critical risk above `0.75`.
- **Explanation:** The chapter does not define numeric scales or normalization for `P_regression`, `I_regression`, or `E_exposure`, so the thresholds `<0.1` and `>0.75` are not reproducible from the table.
- **Proposed correction:** Add a numeric scoring rubric for the three factors, or remove the threshold values from the caption.

### 6. Drift detection notebook comment claims 9,600 samples, but the code and prose use about 2,200

- **Lines:** 2368-2407
- **Severity:** Low
- **Issue:** The notebook header says `n_samples ≈ 9,600`, while the formula with `p1 = 0.95`, `p2 = 0.93`, `z_a = 1.96`, and `z_b = 0.84` gives about `2,207` samples. The prose then reports about `2.2 hours` at 1,000 labels/hour, matching the code rather than the header.
- **Explanation:** `(1.96 + 0.84)^2 * (0.95*0.05 + 0.93*0.07) / 0.02^2 = 2,207`. A 9,600-sample result would require different assumptions.
- **Proposed correction:** Change the header comment to `n_samples ≈ 2,200`, or revise the scenario and prose to match 9,600 samples.

### 7. GPU cluster efficiency example labels utilized GPU-hour value as current cost

- **Lines:** 2840-2858
- **Severity:** Medium
- **Issue:** The example says a 100-GPU cluster at `$2.50/GPU-hour` has current cost `100 * 24 * 0.65 * $2.50 = $3,900/day`.
- **Explanation:** If the cluster has 100 allocated GPUs for 24 hours, the daily bill is `100 * 24 * $2.50 = $6,000/day`, independent of utilization. Multiplying by utilization gives utilized GPU-hour value, not cost. The next line correctly describes `$4,800/day` as effective value from the same cost, but line 2857 calls the analogous quantity "Current cost."
- **Proposed correction:** Change line 2857 to `Current utilized GPU-hour value: ... = $3,900/day`, and optionally state `actual cluster cost = $6,000/day`.

### 8. Inference annual cost equation multiplies by 24 one time too many

- **Lines:** 3263-3273
- **Severity:** High
- **Issue:** The inference cost equation uses daily query volume and `L_avg` in hours, then multiplies by `24 * 365`.
- **Explanation:** `Q_daily * L_avg` already has units of GPU-hours per day before utilization and batching adjustments. Multiplying by `$ / GPU-hour` gives dollars per day. Annualizing requires multiplying by `365`, not `24 * 365`. The alternative formulation in lines 3279-3283 is dimensionally consistent because it uses annual queries and seconds-to-hours conversion.
- **Proposed correction:** Change the equation to `C_infer = (Q_daily * L_avg)/(U_GPU * B_eff) * R_GPU/hr * 365`, or redefine `Q` and `L` so the `24` factor is justified.

### 9. TCO comparison rounds `20.6x` down to `20x` in the interpretation

- **Lines:** 3327-3342
- **Severity:** Low
- **Issue:** The table computes total TCO scaling as `980,000 / 47,500 = 20.6x`, but the caption and insight say the `100x` user increase yields only `20x` TCO.
- **Explanation:** This is a small rounding issue, but the table is precise to one decimal place, so the prose should match that precision or explicitly round.
- **Proposed correction:** Use `about 21x` or `20.6x` in the caption and insight.

### 10. Silent-failure prose calls a 0.5 percentage-point CTR drop a "0.5 percent regression"

- **Lines:** 1474-1501
- **Severity:** Low
- **Issue:** The math correctly uses a CTR drop from `5.0 percent` to `4.5 percent`, but the systems insight calls this a `"0.5 percent regression"`.
- **Explanation:** The absolute drop is `0.5 percentage points`; the relative drop is `10 percent`. Saying `0.5 percent regression` can be read as a relative 0.5 percent drop, which would be much smaller.
- **Proposed correction:** Change the phrase to `0.5 percentage-point absolute regression` or `10 percent relative regression`.

## Checks That Look Consistent

- The multi-tenant sharing calculation is arithmetically consistent in the rendered example: `1 - 0.30/0.70 = 57.1%`, and `57% * $1.75M ≈ $1.0M` (lines 132-159). Only the hidden comment saying `≈40%` is stale.
- The 30-day training-run cost is correct: `2,048 GPUs * 24 h/day * 30 days * $2/GPU-hour = $2.95M` (line 549).
- The canary duration example is consistent if `10,000` treatment samples are needed and the control receives the remaining traffic: at 1 percent of `1M requests/hour`, the canary receives `10,000 requests/hour`; at 5 percent, it receives `50,000 requests/hour`, so `0.2 h = 12 min`.
- The alert-volume example is consistent for its stated 2-sigma assumptions: `100 * 10 * 0.05 * (24*60/5) = 14,400` false alerts/day, and 99 percent deduplication leaves 144/day (lines 2218-2227).
- The cost anomaly false-positive math is consistent: two-sided 3-sigma gives about `0.0027` per daily check and about one false alert per year; two-sided 2-sigma gives about 16-17 per year (lines 2558-2567).
- The feature-store point-in-time storage example is consistent: `1e8 * 1e3 * 8760 = 8.76e14` values, and 100 bytes/value is about `87 PB` before compression (lines 3578-3586).
- The data-quality revenue impact is consistent: `0.08 * $15M/week * 5/7 = $857K`, and reducing detection from five days to four hours cuts impact by about 97 percent (lines 3698-3717).

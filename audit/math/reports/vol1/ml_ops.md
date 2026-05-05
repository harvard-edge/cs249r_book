# Math Audit Report: `book/quarto/contents/vol1/ml_ops/ml_ops.qmd`

## Checked scope

Audited operations, SLO/SLA examples, drift and monitoring metrics, retraining economics, statistical sample-size claims, cost calculations, unit conversions, scaling claims, and prose-equation consistency using direct reasoning only. No Gemini or external model was used. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 1387-1392, 1415-1435, 1441-1455: the retraining economics formula uses ambiguous "accuracy point" units, which can change the optimal interval by 10x.**  
  The staleness integral uses fractional accuracy loss, `A_0 - A(t)`, so `V` must be dollars per query per unit accuracy fraction. But the prose defines `V` as "Financial Value of one percent Accuracy" and the table says "Value per accuracy point." If `V = $0.50` is per percentage point, the denominator should include a factor of 100 and the example gives `T* ≈ 0.10 days`, not `1 day`. If `V = $0.50` is per full 1.0 accuracy fraction, the arithmetic is internally consistent but the prose label is wrong.
  - Proposed correction: Define `V` explicitly as either dollars per query per unit accuracy fraction or dollars per query per percentage point. For the current numeric result, change the prose to "value per unit accuracy fraction" or set `V = $50` per unit accuracy if the intended business value is `$0.50` per percentage point.

- **Lines 1663-1668: the A/B test sample-size example is off by a large factor unless "two percent lift" means something other than the usual relative lift.**  
  For baseline conversion `p = 0.05`, 95 percent confidence, 80 percent power, and a two percent relative lift, the absolute effect is `delta = 0.001`. The standard two-sample approximation gives about `2(1.96+0.84)^2 p(1-p)/delta^2 ≈ 745,000` users per variant, not `25,000`. If the intended effect is two percentage points absolute (`5% -> 7%`), the requirement is only about `1,900` per variant. A `25,000` per-variant requirement corresponds to roughly a `0.55` percentage-point absolute effect, or about an 11 percent relative lift from a 5 percent baseline.
  - Proposed correction: Specify the effect size precisely. For "two percent relative lift," change the sample size to roughly `745,000 users per variant`; for `25,000 users per variant`, describe the detectable effect as about `0.5-0.6 percentage points absolute`.

### Medium Severity

- **Lines 1461-1469: the sensitivity caption contradicts the square-root law.**  
  The table correctly says a `4x` increase in query volume halves `T*`, because `T* ∝ 1/sqrt(Q)`. The caption then says "Doubling query volume halves the optimal interval," which would be linear scaling and is inconsistent with the formula.
  - Proposed correction: Change "Doubling query volume halves" to "Quadrupling query volume halves" or "Doubling query volume shortens the interval by a factor of `sqrt(2)`."

- **Lines 1816-1844: the latency-budget worked example omits request parsing and response serialization after listing them as budget components.**  
  The table includes six latency contributors, but the 100 ms worked allocation reserves budget only for network, feature fetch, inference, and post-processing, summing to the full 100 ms. That leaves zero budget for request parsing and response serialization, which the same table lists as nonzero contributors.
  - Proposed correction: Either include parsing and serialization in the worked allocation, e.g. `10 + 20 + 10 + 40 + 10 + 10 = 100 ms`, or state that the worked example folds parsing into network and serialization into post-processing.

- **Lines 2301-2310: the PSI table mixes percent displays with proportion arithmetic.**  
  The PSI contributions are computed with proportions: for the first row, `(0.12 - 0.15) * ln(0.12/0.15) = 0.0067`. But the table columns are labeled as percentages and display `15.0` and `12.0`, while the difference column displays `-0.03`. If the displayed values are percentages, the difference would be `-3.0` percentage points; if they are proportions, the training and serving values should be `0.15` and `0.12`.
  - Proposed correction: Rename the columns to "Training Proportion" and "Serving Proportion" and display `0.15`, `0.12`, etc., or keep percent displays and change the difference column to percentage points while noting that PSI uses proportions internally.

- **Lines 2389-2391 and 2420-2452: the monitoring-budget code computes about `$261/month`, not the stated `~$153/month`.**  
  The datapoint count is `3 * 50 * 4 * 60 * 24 * 30 = 25.92M`, ingestion is about `$7.78`, storage is about `$0.21`, and query compute is `2 * 3 * 12 * 8 * 22 * $0.02 = $253.44`. The total is therefore about `$261/month`. The code will render the larger total, while the hidden calculation note says the example shows `~$153/month`.
  - Proposed correction: Update the note to `~$261/month`, or reduce the query assumptions, e.g. fewer users, fewer dashboards, or a lower per-query cost, if `$153/month` is the intended target.

- **Lines 2720-2722 and 2804-2843: the fraud-detection business framing mixes accuracy, detection rate, and false positives without a consistent confusion-matrix basis.**  
  The prose first gives "five percent accuracy improvement" as reducing false fraud alerts from `1,000` to `800` daily, which is a 20 percent reduction in false positives, not directly implied by accuracy. The later example says improving detection from `92%` to `94%` both prevents `$2M` annual fraud losses and reduces false positive alerts by `50,000/month`, but no base transaction count, fraud prevalence, threshold movement, or FP/FN rates are given. A higher detection rate usually trades off against false positives unless the model/threshold improvement is specified.
  - Proposed correction: Reframe these as illustrative business outputs from an evaluated confusion matrix, or provide a small confusion-matrix example showing how the 92 to 94 percent detection change, `$2M` prevented loss, and `50,000/month` fewer false positives are simultaneously obtained.

### Low Severity

- **Lines 1281-1284 and 1268-1273: fraud-detection cadence is described as both weekly and daily without explaining the parameter difference.**  
  The schedule table lists fraud detection as weekly, but the triggered-retraining prose immediately says a fraud model with 2 percent daily decay has daily retraining as the economic sweet spot. Both can be true for different fraud workloads, but adjacent presentation makes the cadence appear contradictory.
  - Proposed correction: Add a qualifier such as "the weekly table is a starting point; the high-volume example below uses faster drift and higher business value, yielding daily retraining."

- **Lines 2013-2072: the drift-detection delay calculation assumes immediate labels for every request but the surrounding prose discusses delayed ground truth.**  
  The arithmetic `1000 samples / 1 QPS = 1000 seconds ≈ 17 minutes` is correct if all requests become labeled immediately. In many monitoring contexts, labeled samples arrive hours or days later, so the detection delay is at least `sample accumulation time + label latency`.
  - Proposed correction: State the assumption explicitly: "assuming labels are available immediately." Then add that delayed-label domains must add the feedback-loop delay to these times.

- **Lines 2090 and 2096: the degradation equation uses `lambda` for divergence sensitivity shortly after `lambda` was used for temporal exponential decay.**  
  The text notes the reuse earlier, but the same symbol is then used in `Accuracy(t) ≈ Accuracy_0 - lambda D(P_t || P_0)`, where it is not a temporal decay rate. This is mathematically legal but easy to confuse because `lambda` has units of `1/time` in the exponential model and accuracy-per-divergence in the drift equation.
  - Proposed correction: Use a different symbol for divergence sensitivity, such as `kappa`, in the degradation equation, or add a local reminder that this `lambda` is not the temporal decay rate.

- **Lines 3156-3176 and 3190-3198: "correlation accuracy" and "correlation threshold" conflate correlation with accuracy.**  
  The Oura case describes `62 percent correlation`, `79 percent correlation accuracy`, and a `79 percent correlation threshold`. Correlation and classification accuracy are different metrics; if the case uses agreement/accuracy against PSG labels, call it accuracy, and if it uses correlation, avoid appending "accuracy."
  - Proposed correction: Use one metric name consistently, e.g. "79 percent agreement/accuracy" or "correlation of 0.79 with PSG labels," depending on the source metric.

## Verified Correct

- Lines 1201-1250: the silent-failure cost example is arithmetically consistent. `50M * 0.05 * 28/365 ≈ $191,781`, daily detection costs about `$6,849`, and four incidents save about `$739,726`, matching "nearly `$740,000`."
- Lines 1350-1377 and 1451-1455: under the interpretation that `V` is per unit accuracy fraction, the retraining interval calculation is internally consistent: `sqrt(10000 / (1,000,000 * 0.50 * 0.95 * 0.02)) ≈ 1.03 days`.
- Lines 1951-2003: cost per 1K inferences is correct: `$3/hour / 50,000 inferences/hour * 1000 = $0.06`.
- Lines 2589-2597: the slice-analysis overall accuracy rounds correctly: `0.45*94% + 0.30*92% + 0.20*88% + 0.05*62% = 90.6%`, displayed as `91%`.
- Lines 3067-3118: the single-model ROI example is internally consistent: `4 * $25K + 20 * 12 * $150 = $136K`, and `$136K / $30K ≈ 4.5x`.

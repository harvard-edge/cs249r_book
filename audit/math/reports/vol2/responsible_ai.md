# Math Audit: `book/quarto/contents/vol2/responsible_ai/responsible_ai.qmd`

Scope: responsible-AI, fairness, governance, statistical, privacy, risk, scaling, numeric examples, unit conversions, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. Equal-opportunity threshold example uses tail probabilities that do not match the stated normal distributions

- **Lines:** 671-687
- **Severity:** High
- **Issue:** The initial TPRs at threshold `0.60` are consistent with the stated normal distributions, but the proposed Group B threshold `tau_B = 0.52`, false-positive rates, and precision values are not.
- **Explanation:** For Group B positives, `s | Y=1 ~ N(0.65, 0.16)`. To set `TPR_B = P(s >= tau_B) ≈ 0.70`, the threshold should satisfy `Phi((tau_B - 0.65)/0.16) = 0.30`, giving `tau_B ≈ 0.566`, not `0.52`. At `tau_B = 0.52`, the TPR is about `0.79`, not `0.70`. For Group B negatives, `s | Y=0 ~ N(0.40, 0.17)`: the FPR at `0.60` is about `0.12`, not `0.28`; the FPR at `0.52` is about `0.24`, not `0.38`. Using the stated base counts, precision is therefore about `0.78` at `0.60` and `0.69` at `0.52`, not `0.69` and `0.61`.
- **Proposed correction:** Either change Group B's equal-opportunity threshold to about `0.57` and recompute FPR/precision (`FPR_B ≈ 0.16`, precision about `0.74`), or adjust the means/standard deviations so that `tau_B = 0.52`, `FPR 0.38`, and precision `0.61` are reproducible.

### 2. Fairlearn example computes approval rate with `accuracy_score`

- **Lines:** 1123-1146
- **Severity:** High
- **Issue:** The listing labels the metric as `"approval_rate"` but maps it to `accuracy_score`.
- **Explanation:** Approval rate is `mean(y_pred == 1)`, independent of `y_true`. `accuracy_score` computes `mean(y_pred == y_true)`. The printed values described as approval rates by ethnicity could therefore be accuracies, not favorable-outcome rates, which undermines the fairness metric being demonstrated.
- **Proposed correction:** Replace `"approval_rate": accuracy_score` with a metric such as `lambda y_true, y_pred: np.mean(y_pred == 1)` or Fairlearn's `selection_rate`.

### 3. Production monitor's equalized-odds metric ignores false-positive-rate disparities

- **Lines:** 1619-1649
- **Severity:** High
- **Issue:** The code computes `fpr_by_group`, but `eq_odds_diff` is assigned only the maximum pairwise TPR difference. This makes `equalized_odds_diff` identical to the equality-of-opportunity gap, except for sign/implementation details.
- **Explanation:** Equalized odds requires both TPR and FPR equality. A model with equal TPRs but sharply different FPRs would produce `eq_odds_diff = 0` in this implementation and would not trigger the intended alert.
- **Proposed correction:** Include FPR in the metric, for example `max(max pairwise TPR gap, max pairwise FPR gap)` or the chapter's earlier average-odds definition `0.5 * (TPR gap + FPR gap)`.

### 4. SISA unlearning caption conflicts with the notebook and with its own cost arithmetic

- **Lines:** 1806, 1822-1845, 1858-1862
- **Severity:** Medium
- **Issue:** The figure caption says SISA with `20` shards costs about `$90K` and takes `1.7 days`; the notebook uses `100` shards and reports about `$46K` and `8 hours`.
- **Explanation:** With the stated full retraining cost of `$4.6M`, 20 shards imply `$4.6M / 20 = $230K`, not `$90K`; the time `34 days / 20 = 1.7 days` is consistent. The notebook's 100-shard arithmetic is internally consistent: `$4.6M / 100 = $46K` and `34*24/100 = 8.16 hours`.
- **Proposed correction:** Choose one scenario. For 20 shards, change the cost to about `$230K`. For 100 shards, change the caption to about `8 hours` and `$46K`.

### 5. SHAP/explainability overhead is expressed as incompatible percentages and multipliers

- **Lines:** 99, 111, 1044-1050, 2530, 2546, 2583
- **Severity:** Medium
- **Issue:** Several places say on-demand SHAP/explainability costs `50--1000x` more compute than inference, while the overhead table and the fallacy section cite SHAP inference cost as `+50 percent to +200 percent`.
- **Explanation:** `+50 percent to +200 percent` means `1.5x--3x` total inference cost, not `50x--1000x`. The line 2546 example is consistent with the table if a 100 ms baseline receives an added 50--200 ms, but it is inconsistent with the chapter-wide `50--1000x` claim.
- **Proposed correction:** Separate the cases explicitly: use `+50--200 percent` for approximate/streaming explanations on selected requests, and reserve `50--1000x` for exact or exhaustive explanation workloads if that is intended. Update the takeaways to use the same convention.

### 6. Summary fairness table implies equal base rates while the caption invokes differing base rates

- **Lines:** 2570-2577
- **Severity:** Medium
- **Issue:** The metrics in the summary table are mutually consistent only if both groups have the same positive base rate, but the caption concludes with a statement about base rates differing between groups.
- **Explanation:** For Group A, `approval = TPR*p + FPR*(1-p)` gives `0.55 = 0.90p + 0.20(1-p)`, so `p = 0.50`. For Group B, `0.40 = 0.60p + 0.20(1-p)` also gives `p = 0.50`. The listed PPVs then follow: `0.90*0.5/0.55 ≈ 82%` and `0.60*0.5/0.40 = 75%`. Thus the table does not illustrate a base-rate-difference impossibility case.
- **Proposed correction:** Either remove the phrase "when base rates differ" from this caption, or revise the table so the implied base rates differ while the approval, TPR, FPR, and PPV values remain algebraically consistent.

### 7. "4 percent accuracy" should be percentage points or a relative percentage

- **Lines:** 220-276, 640-661
- **Severity:** Low
- **Issue:** Accuracy drops from `85 percent` to `81 percent`, and the prose calls this a `4 percent accuracy` cost/profit loss.
- **Explanation:** The absolute change is `4 percentage points`; the relative accuracy reduction is `4/85 = 4.7 percent`. Saying `4 percent accuracy` is ambiguous and can be read as either.
- **Proposed correction:** Use `4 percentage points` for the absolute accuracy drop, or `about a 4.7 percent relative accuracy reduction` if the relative loss is intended.

## Checks That Look Consistent

- The confusion-matrix fairness example is arithmetically consistent: approval gaps are `70/100 - 40/100 = 30 pp`, TPRs are `40/45 ≈ 0.89` and `30/50 = 0.60`, and FPRs are `30/55 ≈ 0.55` and `10/50 = 0.20` (lines 336-386).
- The disparate-impact, SPD, EOD, AOD, and PPV calculations follow from the same loan example: `0.40/0.70 = 0.57`, `0.70 - 0.40 = 0.30`, `0.89 - 0.60 = 0.29`, `0.5*(0.29+0.35)=0.32`, and `40/70=0.571`, `30/40=0.750` (lines 544-630).
- The two-proportion z-test for the approval-rate gap is correct: pooled rate `0.55`, denominator `sqrt(0.55*0.45*(1/100+1/100)) ≈ 0.070`, and `z ≈ 4.29` (lines 697-710).
- The privacy-cost notebook is internally consistent: strong privacy multiplies `$4.6M` by `3.0` to get about `$13.8M`, and the displayed moderate-overhead and accuracy-drop values are direct constants (lines 1721-1778).
- The representation-tax arithmetic is consistent: `10 * 100,000 * $125 = $125M`, and adding 30--50 percent harmonization overhead gives `$162.5M--$187.5M`, reasonably rounded to `$160--190M` (lines 2329-2340).
- The fleet-level safety statement is directionally correct: `0.1 percent` failure at billions of daily requests is millions of failures per day, so the prose claim of at least thousands of daily safety incidents is conservative (line 2524).

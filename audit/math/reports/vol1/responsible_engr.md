# Math Audit Report: `book/quarto/contents/vol1/responsible_engr/responsible_engr.qmd`

## Checked scope

Audited fairness/statistical metrics, risk thresholds, numeric examples, unit conversions, cost and carbon calculations, scaling claims, percent versus percentage-point language, and prose-equation consistency using direct reasoning only. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 632 and 659-662: the loan example's claimed aggregate accuracy does not match the confusion matrices.**  
  The text says the loan model reports 85 percent accuracy across all applicants. Group A has `(4,500 + 4,000) / 10,000 = 85 percent` accuracy, but Group B has `(600 + 800) / 2,000 = 70 percent` accuracy. Across both groups, aggregate accuracy is `(4,500 + 4,000 + 600 + 800) / 12,000 = 77.5 percent`, not 85 percent.
  - Proposed correction: Either change line 632 to say the model reports 85 percent accuracy for the majority group, or adjust Group B's confusion matrix so the combined accuracy is 85 percent. If preserving the fairness pattern, a simple prose correction is: "A loan approval model reports 85 percent accuracy on the majority group and 77.5 percent across these evaluated applicants."

- **Lines 537 and 516-517: 1,000 subgroup examples are not enough to detect a one percentage-point performance gap with 95 percent confidence in general.**  
  The notebook states that 1,000 images for a 1 percent subgroup are statistically valid to detect a one percent performance gap with 95 percent confidence. For a single binomial accuracy estimate, a 95 percent margin of error of one percentage point requires about `0.25 * (1.96 / 0.01)^2 = 9,604` examples in the worst case. Even at 95 percent accuracy, the one-sample margin is about `1.96 * sqrt(0.95 * 0.05 / 1000) = 1.35` percentage points, before accounting for a two-group gap or power. The random-sampling multiplier arithmetic is right, but the statistical-validity claim is too strong.
  - Proposed correction: Either change the target to a larger subgroup sample size, e.g. "roughly 10,000 examples for a worst-case one-percentage-point margin," which would imply 1,000,000 random samples for a 1 percent subgroup, or soften the claim to "a minimum illustrative subgroup test set" and avoid the one-percentage-point/95-percent guarantee.

### Medium Severity

- **Lines 1458-1497 and 1509: the TCO summary caption conflicts with the implemented 20 percent quantization scenario and the actual inference/training ratio.**  
  The code defines `quant_reduction_pct = 0.20`, and downstream prose uses `{python} TCOSummary.quant_reduction_pct_str` as 20 percent. The caption instead says a 30 percent latency reduction. The same caption says the inference/training ratio is about 10:1, but the chapter's own values are `$1.52M / $38.4K = 39.6:1`, which rounds to 40:1.
  - Proposed correction: Change the caption to "The ~40:1 ratio..." and "A 20 percent reduction..." or change `quant_reduction_pct` to 0.30 and recompute the generated savings. The ratio should still be about 40:1 unless the underlying TCO assumptions are changed.

- **Lines 2116 and 1328-1339: the cost ratio is not the same as the carbon/emissions ratio in the TCO example.**  
  Line 2116 says the same inference-to-training ratio applies to energy consumption and carbon emissions. In the TCO code, training carbon is `9,600 GPU-hr * 0.16 = 1,536 kg`, while inference carbon is about `555.6 GPU-hr/day * 365 * 3 * 0.16 = 97,333 kg`. That is about `63:1`, not the financial `39.6:1`, because training and inference use different dollar rates and the training hyperparameter line is priced per experiment.
  - Proposed correction: Replace "The same ratio applies" with "The same dominance pattern applies; under this example the carbon ratio is even larger, about 63:1."

- **Lines 870-892 and 897-912: the "Price of Fairness" utility loss is asserted rather than computed from the displayed assumptions.**  
  The code comments compute an illustrative loss of `2,500 / (50,000 + 2,500) ≈ 4.8 percent`, then set `utility_loss_pct = 3` by adding an unstated 30 percent disadvantaged-group share. The rendered callout presents 3 percent as "The Calculation" from a 5 percent false-positive increase, `$100k` successful-hire value, and `$50k` bad-hire cost, but those visible assumptions alone do not produce 3 percent. The text also uses "20 percent TPR gap" and "5% increase" where absolute metric changes are intended.
  - Proposed correction: Either expose the additional group-share/base-rate assumptions and compute the 3 percent value from them, or change the displayed utility loss to the directly computed 4.8 percent. Use "20 percentage-point TPR gap" and "5 percentage-point increase in false positives" if the quantities are absolute rate changes.

- **Lines 717 and 96: the fairness impossibility theorem is stated without its required non-degenerate conditions.**  
  The chapter says it is mathematically impossible to satisfy multiple fairness metrics simultaneously when base rates differ. The standard result requires additional conditions, such as imperfect prediction and nontrivial classifiers. Perfect prediction can satisfy calibration and equalized error rates even with different base rates. The broad statement is directionally useful but mathematically overstated.
  - Proposed correction: Qualify the statement: "Except in degenerate cases such as perfect prediction or equal base rates, calibrated risk scores cannot generally satisfy equalized odds when base rates differ."

### Low Severity

- **Lines 1412-1419: retraining frequency is rendered under a carbon column as if it were kg CO2.**  
  The "Retraining frequency" row places `{python} TCOCalc.t_cycles_str`, which renders as `12`, in the `Carbon (kg CO2)` column. That value is a cycle count, not a carbon quantity. The total-carbon row below is correct, so this is a table-label/unit issue rather than a calculation error.
  - Proposed correction: Put `-` in the carbon column for retraining frequency, or relabel the column entry as "12 cycles" outside the carbon column.

- **Lines 897, 907, 2122, and 2127: several absolute metric differences use "percent" where "percentage points" would be clearer.**  
  The chapter correctly uses "percentage point" in many fairness calculations, but some absolute gaps still use percent language: a 20-point TPR gap, a 5-point false-positive increase, and a "fairness gap <five percent" requirement. In fairness and risk contexts, "5 percent" can mean a relative change, while the intended threshold is often an absolute change in a rate.
  - Proposed correction: Use "percentage points" for absolute differences between rates and reserve "percent" for relative changes or raw rates.

## Verified correct

- Lines 93-94 and 119: COMPAS ratios are consistent with the prose: `44.9 / 23.5 ≈ 1.91`, i.e. nearly twice, and `47.7 / 28.0 ≈ 1.70`, i.e. far more often.
- Lines 279-312: the fairness-frontier figure is internally consistent; point B at disparity `0.05` gives accuracy `0.85 + 0.10 * (1 - exp(-1)) ≈ 0.913`, matching the 91 percent label, and its disparity is about `0.18 / 0.05 = 3.6x` lower than point A, reasonably rounded as 4x fairer.
- Lines 359-390 and 401-424: Gender Shades arithmetic is consistent: `34.7 / 0.8 = 43.4x`, `7.1 / 0.8 = 8.9x`, `12.0 / 0.8 = 15.0x`, and accuracies are `99.2 percent` and `65.3 percent`.
- Lines 516-545: the random-sampling representation multiplier is arithmetically correct for the stated target: `1,000 / 0.01 = 100,000`, a `100x` multiplier.
- Lines 659-679 and 706-759: loan fairness metrics other than the aggregate accuracy claim are consistent: approval rates are 55 percent and 40 percent, TPRs are 90 percent and 60 percent, FPRs are both 20 percent, FNRs are 10 percent and 40 percent, and the minority false-negative rate is 4x the majority rate.
- Lines 1255-1257: carbon conversion is dimensionally correct: `0.4 kW * 1 hour * 0.4 kg/kWh = 0.16 kg CO2eq/GPU-hour`.
- Lines 1308-1400 and 1421-1448: TCO component calculations are internally consistent: training is `$3,200` per cycle and `$38,400` over 12 cycles; inference is about `556 GPU-hr/day`, `$507K/year`, and `$1.52M` over three years; operations are `$510K`; percentages round to 2 percent, 73 percent, and 25 percent.
- Lines 1548-1578: carbon scale arithmetic is correct: `1,300 MWh = 1,300,000 kWh`; at `0.4 kg/kWh` this is `520,000 kg = 520 metric tons`; `520 / 4.6 ≈ 113` passenger-car years, so a one percent efficiency gain saves about `5.2 tons`, roughly one car-year.

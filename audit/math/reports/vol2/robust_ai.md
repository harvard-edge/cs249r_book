# Math Audit: `book/quarto/contents/vol2/robust_ai/robust_ai.qmd`

Scope: robustness, adversarial, statistical drift, poisoning, reliability/risk equations, numeric examples, unit conversions, scaling claims, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. V100 bandwidth is converted to bits per second incorrectly

- **Lines:** 318, 337, 347
- **Severity:** High
- **Issue:** The text says a V100-class `~900 GB/s` memory system processes `10^11 bits per second`.
- **Explanation:** `900 GB/s` is `900 * 10^9 bytes/s`, or about `7.2 * 10^12 bits/s`, not `10^11 bits/s`. With a bit error rate of `10^-17` errors/bit, that gives `7.2e-5 errors/s = 0.259 errors/hour` for one device, which is below "multiple potential faults per hour" unless aggregating across many devices or using a higher error rate.
- **Proposed correction:** Change the throughput to approximately `7 * 10^12 bits/s`. Then either say this is about `0.26 potential faults per device-hour` at `10^-17`, or explicitly aggregate over a cluster, e.g. `about 2.6 potential faults per hour across 10 such GPUs`.

### 2. Exponent-bit flip multiplier is misstated

- **Lines:** 349
- **Severity:** Medium
- **Issue:** The figure caption says flipping bit `k` of an IEEE 754 exponent multiplies the weight by `2^k`.
- **Explanation:** Floating-point values scale as `2^e`. Flipping an exponent bit changes the stored exponent by the place value of that bit, so the numeric multiplier is `2^{+/- 2^k}` if `k` is the bit position within the exponent field. If `k` means the exponent delta itself, the wording should not say "bit `k`."
- **Proposed correction:** Use wording such as: "Flipping an exponent bit can change the exponent by a power-of-two amount, multiplying the weight by `2^{Delta e}` and potentially pushing it far outside the normal range."

### 3. The SDC-at-scale prose overstates the probability at 10K GPUs

- **Lines:** 361-384, 389-391, 400-401, 445-448
- **Severity:** Medium
- **Issue:** The code and annotation correctly compute `P = 1 - (1 - 10^-4)^10000 ~= 0.632`, but comments/prose describe this as "near 1.0" and "effectively certain" at around 10K GPUs or beyond a few thousand devices.
- **Explanation:** At `p = 10^-4` per device-hour, `P >= 0.95` requires `N >= ln(0.05)/ln(0.9999) ~= 29,956` devices. At 10K devices it is more likely than not, but not near-certain.
- **Proposed correction:** Change the framing to "more likely than not by about 7K GPUs and about 63% at 10K; effectively certain near 30K GPUs." Keep the formula.

### 4. The distribution-shift z-score is rounded too high

- **Lines:** 685-693, 710-715
- **Severity:** Low
- **Issue:** The notebook states the shift is `5.5 standard errors`, but the stated numbers give about `5.27`.
- **Explanation:** `SE = 0.3 / sqrt(1000) = 0.00949`, and `0.05 / 0.00949 = 5.27`. The p-value claim `< 0.001` remains correct.
- **Proposed correction:** Change `5.5 standard errors` to `5.3 standard errors`, or adjust the standard deviation to about `0.287` if `5.5` is intended.

### 5. The PSI figure indexes weeks off by one

- **Lines:** 849, 876-886, 906-932
- **Severity:** Low
- **Issue:** The caption says major drift is detected at Week 35 and retraining at Week 40, but the code uses zero-based indices `shift_idx = 35` and `retrain_idx = 40` into `weeks = 1..52`.
- **Explanation:** `weeks[35]` is Week 36, and `weeks[40]` is Week 41.
- **Proposed correction:** Either set `shift_idx = 34` and `retrain_idx = 39`, or update the caption to Week 36 and Week 41.

### 6. PSI is described as log-odds rather than a log ratio

- **Lines:** 818-824
- **Severity:** Low
- **Issue:** PSI is described as using a "symmetric log-odds formulation."
- **Explanation:** The displayed formula uses `ln(p_i/q_i)`, which is a log ratio of bin proportions. It is symmetric under swapping `P` and `Q` because `(p-q) ln(p/q)` is unchanged by the swap, but it is not a log-odds expression.
- **Proposed correction:** Replace "symmetric log-odds formulation" with "symmetric log-ratio formulation."

### 7. The adversarial-training notebook mixes clean accuracy and robust accuracy

- **Lines:** 598-611, 617-626
- **Severity:** Medium
- **Issue:** The code defines `robust_acc = 0.70` as "Accuracy against worst-case attack," but the notebook says clean-data accuracy drops from `95 percent` to that value and calls it a `25 percent drop in general performance`.
- **Explanation:** Robust accuracy under attack and clean accuracy are different metrics. A drop from `95%` clean accuracy to `70%` adversarial accuracy is not a clean-data accuracy drop; it compares different evaluation distributions. If the intended claim is a clean-accuracy trade-off, `robust_acc` should be clean accuracy after adversarial training.
- **Proposed correction:** Either relabel the result as "worst-case/adversarial accuracy is 70%" or change the variable and prose to a clean robust-model accuracy. If comparing `95%` to `70%`, call the change `25 percentage points`, not simply `25 percent`.

### 8. Adversarial-training compute overhead is inconsistent across the chapter

- **Lines:** 620-626, 1949, 1999, 2029, 2384, 2432, 2469
- **Severity:** Medium
- **Issue:** The chapter gives several incompatible overhead ranges: PGD-7 costs `8x`, intrinsic robustness costs `8x-10x`, adversarial training costs `3x-10x`, the FGSM implementation costs `2x-3x`, the minimax footnote says `5x-10x`, a pitfall says only `10--20 percent compute overhead`, and the summary says `2x-10x`.
- **Explanation:** Different attack generators can justify different costs, but the text often uses the broad phrase "adversarial training" without specifying FGSM, PGD-k, certified randomized smoothing, or detection-only defenses. The `10--20 percent` claim is especially inconsistent with the nearby `8x`/`5x-10x` statements.
- **Proposed correction:** Normalize the claims by defense type, e.g. "FGSM adversarial training: about `2x`; PGD-7: about `8x`; stronger PGD/certified methods: `5x-10x` or higher; lightweight detection/preprocessing: can be `10--20%` overhead."

### 9. Randomized smoothing radius formula omits the runner-up class condition

- **Lines:** 1965-1967
- **Severity:** Medium
- **Issue:** The text states the certified radius as `R = sigma Phi^{-1}(p_A)` based only on the top-class probability.
- **Explanation:** The general randomized smoothing certificate depends on both the lower bound for the top-class probability and an upper bound for the runner-up class: `R = (sigma/2) [Phi^{-1}(p_A) - Phi^{-1}(p_B)]`. The simplified `sigma Phi^{-1}(p_A)` follows only when `p_B` is bounded by `1 - p_A`, which should be stated. The numeric example with `p_A = 0.999` and `sigma = 0.5` gives `0.5 * Phi^{-1}(0.999) ~= 1.55`, so the arithmetic is fine under that simplifying assumption.
- **Proposed correction:** Present the general formula, then say the simplified bound is obtained when the remaining class probability is bounded by `1 - p_A`.

### 10. The adversarial-training code line references are stale

- **Lines:** 2007-2024, 2029
- **Severity:** Low
- **Issue:** The prose says gradients are computed at line `2190`, the sign function at line `2196`, and mixing at lines `2199-2200`, but the actual listing is at lines `2007-2024`.
- **Explanation:** In the current source, `loss.backward()` is line `2015`, `data.grad.sign()` is line `2018`, and the concatenation is lines `2021-2022`.
- **Proposed correction:** Update the prose line references to the current listing line numbers, or avoid hard-coded line numbers.

### 11. The Huber-loss footnote confuses squared loss with gradient magnitude

- **Lines:** 2382
- **Severity:** Medium
- **Issue:** The footnote says an outlier with `100x` normal error contributes `10,000x` normal gradient magnitude under MSE.
- **Explanation:** For MSE, the loss contribution scales as error squared, so `100x` error gives `10,000x` loss. But the gradient with respect to the prediction is proportional to error (`2e`), so the gradient magnitude is `100x`, not `10,000x`.
- **Proposed correction:** Change the sentence to "contributes `10,000x` the loss and `100x` the prediction-level gradient magnitude," or specify a parameter-gradient scenario where another factor justifies the larger number.

### 12. The influence-function formula is for upweighting, not removing, a point

- **Lines:** 2376
- **Severity:** Medium
- **Issue:** The prose says influence functions approximate the effect of removing a training point, but the displayed sign corresponds to the standard upweighting influence on test loss.
- **Explanation:** The standard influence of upweighting `z` on the loss at `z_test` is `- grad L(z_test)^T H^{-1} grad L(z)`. Removing a point has the opposite sign, with a dataset-size scaling factor depending on convention.
- **Proposed correction:** Either say "upweighting a training point" for the displayed formula, or give the removal form explicitly, e.g. `I_remove(z, z_test) approx (1/n) grad L(z_test)^T H^{-1} grad L(z)`.

## Checks That Look Consistent

- The PSI worked example is arithmetically consistent: the listed contributions sum to `0.0652`, matching the rounded `0.065` (lines 1043-1055).
- The Gaussian KL worked example is correct: `ln(1.3/1.1) + (1.1^2 + 0.3^2)/(2*1.3^2) - 0.5 ~= 0.052` (lines 1061-1074).
- The KS critical value is correct: `1.36 * sqrt(20000 / 10000^2) ~= 0.0192`, so `D = 0.089` is significant (lines 1078-1086).
- The FGSM formula has the standard untargeted sign for increasing loss under an `L_infinity` perturbation budget (lines 1207-1218).
- The MC dropout latency example is internally consistent: `50` passes at `2 ms` each adds `100 ms` (line 1989).

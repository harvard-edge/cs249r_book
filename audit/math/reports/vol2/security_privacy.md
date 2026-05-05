# Math Audit: `book/quarto/contents/vol2/security_privacy/security_privacy.qmd`

Scope: security/privacy/differential-privacy/statistical/risk equations, numeric examples, unit conversions, scaling claims, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. DP-SGD privacy example recommends reducing noise to improve privacy

- **Lines:** 3091-3110
- **Severity:** High
- **Issue:** The worked example computes a privacy loss above the target, then says that to reach target $\epsilon=3.0$ the system should "reduce $\sigma$ to 0.7 with more noise." This reverses the relationship between $\sigma$ and privacy.
- **Explanation:** In the chapter's own RDP approximation, $\epsilon_{\text{total}} \propto 1/\sigma^2$. Reducing $\sigma$ from 1.3 to 0.7 reduces noise and increases privacy loss by about $(1.3/0.7)^2 \approx 3.45\times$. It cannot lower $\epsilon$ from about 9-10 to 3.
- **Proposed correction:** Change the remedy to increasing the noise multiplier, reducing epochs/steps, or reducing the sampling rate. Under the displayed approximation, reaching $\epsilon \approx 3$ requires substantially larger $\sigma$, not $\sigma=0.7$.

### 2. DP-SGD RDP optimum is arithmetically inconsistent with the preceding bullets

- **Lines:** 3100-3108
- **Severity:** High
- **Issue:** The example says optimal $\alpha \approx 3.5$ yields $\epsilon \approx 8.4$, but the displayed formula gives about 9.6 at $\alpha=3.5$.
- **Explanation:** With $\varepsilon_{\text{total}}(\alpha)=1.42\alpha$ and $\ln(10^5)=11.51$, $\epsilon(3.5)=1.42(3.5)+11.51/(2.5)=4.97+4.61=9.58$. Optimizing continuously gives $\alpha \approx 3.85$ and $\epsilon \approx 9.5$, consistent with the line's own $\alpha=4$ calculation of 9.51.
- **Proposed correction:** Replace "optimal $\alpha \approx 3.5$ yields $\epsilon \approx 8.4$" with "optimal $\alpha \approx 3.8$ yields $\epsilon \approx 9.5$."

### 3. Moments-accountant conversion formula has the wrong sign

- **Lines:** 3056-3059
- **Severity:** High
- **Issue:** The equation subtracts $\ln(1/\delta)$ when converting accumulated moments to an $\epsilon(\delta)$ bound.
- **Explanation:** Since $\delta<1$, $\ln(1/\delta)>0$. The usual conversion adds the failure-probability penalty: $\epsilon \approx (\alpha_{\mathcal M}(\lambda) + \ln(1/\delta))/(\lambda-1)$, up to convention-specific indexing of $\lambda$. Subtracting it would make the bound smaller when $\delta$ becomes stricter, which is backwards.
- **Proposed correction:** Change the numerator to add the $\ln(1/\delta)$ term, or restate the exact moments-accountant convention being used and verify that stricter $\delta$ increases $\epsilon$.

### 4. Gaussian mechanism notation mixes absolute noise and noise multiplier

- **Lines:** 2997-3012, 3038
- **Severity:** High
- **Issue:** The Gaussian mechanism first defines the mechanism as adding $\mathcal{N}(0,\sigma^2(\Delta_2 f)^2)$, where $\sigma$ is a dimensionless noise multiplier, but the privacy-loss denominator and calibration formula then treat $\sigma$ as the absolute noise standard deviation.
- **Explanation:** If the absolute standard deviation is $\tau$, the mechanism is $f(D)+\mathcal{N}(0,\tau^2)$ and the denominator in the privacy-loss expression is $2\tau^2$. If instead $\sigma$ is the multiplier and $\tau=\sigma\Delta_2 f$, the denominator should include $(\Delta_2 f)^2$, and the sufficient condition becomes $\sigma \ge \sqrt{2\ln(1.25/\delta)}/\epsilon$.
- **Proposed correction:** Use separate symbols, e.g. $\tau$ for absolute standard deviation and $z=\tau/\Delta_2 f$ for the noise multiplier, then update the formulas consistently.

### 5. Gaussian noise multiplier for $\delta=10^{-7}$ is understated

- **Lines:** 3038
- **Severity:** Medium
- **Issue:** The text says $\epsilon=1,\delta=10^{-7}$ requires $\sigma \approx 4.45\cdot\Delta_2 f$ under the displayed Gaussian mechanism bound.
- **Explanation:** The displayed bound gives $\sqrt{2\ln(1.25/10^{-7})}=\sqrt{2\ln(1.25\times10^7)}\approx 5.72$, not 4.45.
- **Proposed correction:** Change the multiplier to about `5.72`, or change the stated $\delta$ to a value consistent with `4.45`.

### 6. Confidence-rounding information reduction is off by about 3x or more

- **Lines:** 1237-1244, 3280
- **Severity:** Medium
- **Issue:** The text says rounding probabilities from 6 decimals to 2 decimals reduces leakage by about 4 bits per class, and later describes the same result as about 4 bits per token.
- **Explanation:** Removing four decimal digits removes about $\log_2(10^4)=13.3$ bits of decimal precision per reported value. If comparing a 32-bit float to 101 possible two-decimal probability bins, the reduction is about $32-\log_2(101)\approx25.3$ bits. The later "per token" wording is also not the same as the ImageNet per-class probability example.
- **Proposed correction:** State the chosen model explicitly. For decimal precision, use about `13 bits per reported probability`; for 32-bit float quantization, use about `25 bits per reported probability`. Avoid carrying the ImageNet per-class result over to LLM tokens without recalculating.

### 7. Top-k truncation bit calculation does not match the information being removed

- **Lines:** 1245-1251
- **Severity:** Medium
- **Issue:** The text says top-5 truncation on a 1000-class problem reduces information leakage by $\log_2(1000/5)\approx7.6$ bits per query.
- **Explanation:** That ratio is not the information content of a full probability distribution. If the comparison is only top-1 class identity versus five candidates, top-1 has $\log_2(1000)\approx10$ bits and the top-5 set has $\log_2\binom{1000}{5}\approx43$ bits before probabilities. If the comparison is full logits/probabilities versus five rounded probabilities, the reduction is dominated by omitting 995 numeric values.
- **Proposed correction:** Replace the 7.6-bit claim with a qualitative statement, or define a precise encoding model and compute the bit reduction under that model.

### 8. Roadmap DP accuracy claims conflict with the chapter's own trade-off table

- **Lines:** 3150-3159, 3278, 3296, 3312
- **Severity:** Medium
- **Issue:** The roadmap says $\epsilon\le8$ has 3-5 percent degradation and $\epsilon\le1$ has 10-15 percent degradation, citing the DP trade-off section. The cited table shows much wider variation: at $\epsilon=8$, CIFAR-10 loses 10.5 points and ImageNet loses 28.3 points; at $\epsilon=1$, CIFAR-10 loses 26.5 points.
- **Explanation:** The 3-5 and 10-15 percent ranges may apply to some easier or text benchmarks, but they are not supported as general claims by the cited table.
- **Proposed correction:** Qualify these as task-dependent and cite benchmark-specific ranges, e.g. "low single-digit loss for simple tasks, double-digit loss for complex vision tasks."

### 9. Practical API extraction example has a query-count explanation error

- **Lines:** 1335-1342
- **Severity:** Low
- **Issue:** The example says 5M extraction queries correspond to "0.5 percent of ImageNet per class."
- **Explanation:** For 1000 classes, 5M queries is 5,000 queries per class. If ImageNet has roughly 1,000-1,300 images per class, 0.5 percent per class would be only about 5-7 examples per class, not 5,000.
- **Proposed correction:** Change the parenthetical to "about 5,000 queries per class" or remove it.

### 10. Defense-selection table and later roadmap use incompatible DP overhead ranges

- **Lines:** 2789-2791, 3185-3187
- **Severity:** Low
- **Issue:** The defense-selection table says cloud DP training time increases 30-120 percent, while the DP decision framework says DP training typically requires 2-5x more compute.
- **Explanation:** A 30-120 percent training-time increase is 1.3-2.2x runtime, whereas 2-5x more compute is a 100-400 percent increase. These ranges overlap only at the low end.
- **Proposed correction:** Harmonize the ranges or distinguish scenarios, e.g. "optimized large-batch DP-SGD may be 1.3-2.2x, while per-sample-gradient implementations can require 2-5x compute."

## Checks That Look Consistent

- Lines 157-166 and 184-190: For salaries bounded in `[0, 200000]`, Laplace noise scale for the sum at $\epsilon=1$ is `$200,000`; dividing by `N=1000` gives `$200` mean error scale, and `N=100` gives `$2,000`.
- Lines 213-234: Multi-tenant isolation arithmetic is consistent: `(1000 - 850) / 1000 = 15%`.
- Lines 1227-1231: Adaptive limiting example is consistent: $e^{-0.8}\approx0.45$, which is a 55 percent rate reduction.
- Lines 1271-1278: Economic deterrent threshold is consistent: `$5M / 100M queries = $0.05/query`.
- Lines 1317-1323 and 1367: Latency overhead arithmetic is consistent: `2.0 ms + 0.5 ms = 2.5 ms`, which rounds to `3%` of a `100 ms` baseline.
- Lines 1359-1361: API pricing examples are consistent: `5M * $0.001 = $5,000` and `300 * 30 * $0.001 = $9/month`.
- Lines 2402: TrustZone cycle conversion is consistent: at `500 MHz`, one cycle is `2 ns`, so `300-1000` cycles is `0.6-2.0 us`.
- Lines 2427-2458: FHE latency example is consistent: `20 ms * 10,000 = 200,000 ms = 200 s`; AES latency is `20.5 ms`.
- Lines 3031-3032: For $\delta=10^{-5}$, $\sqrt{2\ln(1.25/\delta)}\approx4.8$, so the strong/weak privacy noise values `4.8` and `0.48` are arithmetically consistent.
- Lines 3134-3142 and 3312: Simple composition examples are consistent: 10 mechanisms at $\epsilon=1$ consume total $\epsilon=10`; monthly retraining for two years consumes $\epsilon=24$ under simple composition.

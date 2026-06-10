# Float-explanation worklist — robust_ai.qmd (vol2)

## Summary
| type | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|
| figure | 22 | 21 | 1 | 0 |
| table | 5 | 5 | 0 | 0 |
| listing | 1 | 1 | 0 | 0 |
| algorithm | 0 | — | — | — |
| equation | 0 | — | — | — |
| **total** | **28** | **27** | **1** | **0** |

## Findings (⚠️ and 🛑 only — ✅ floats are tallied above, not expanded)

### ⚠️ `fig-adversarial-googlenet` — def L1357  (Thin — mismatched claim)
- **Caption:** **Adversarial Perturbations**: Subtle, intentionally crafted noise added to an image can cause a trained deep neural network (GoogLeNet) to misclassify it, even though the perturbed image remains visually indistinguishable to humans. This vulnerability underscores the lack of robustness in many machine learning models and motivates research into adversarial training and defense mechanisms. Source: [@goodfellow2015explaining].
- **Ref(s):** L1355 `@fig-adversarial-googlenet`: "The reason this defense budget matters is that adversarial attacks extend far beyond simple misclassification (@fig-adversarial-googlenet). These vulnerabilities create systemic risks across deployment domains."
- **Context checked:** ref ✗ (claim mismatch — figure shows misclassification; prose claims figure illustrates something beyond misclassification) · prev ¶ ✓ (defense families named) · next ¶ ✓ (payoff pivots to stop-sign physical attacks) · caption ✓ (names the panda/GoogLeNet result and vulnerability implication)
- **Issue:** The parenthetical ref appends the figure to support the claim that attacks go "far beyond simple misclassification," but the Goodfellow panda figure is the canonical *example of misclassification*. The sentence's implied meaning and the figure's actual content point in different directions, leaving the reader with no clear instruction about what to take away from the figure itself.
- **Suggested rewrite (flag-only):**
  ```diff
  - The reason this defense budget matters is that adversarial attacks extend far beyond simple misclassification (@fig-adversarial-googlenet). These vulnerabilities create systemic risks across deployment domains.
  + The Goodfellow panda result (@fig-adversarial-googlenet) makes the mechanism concrete: imperceptible noise, invisible to a human reviewer, flips GoogLeNet's classification with high confidence. That single example scales to systemic risk across deployment domains, because the same geometry that fools an image classifier applies to medical scans, traffic signs, and speech recognition pipelines.
  ```

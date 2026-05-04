# Math Audit Report: `book/quarto/contents/vol2/parts/responsible_fleet_principles.qmd`

## Checked scope

Audited `book/quarto/contents/vol2/parts/responsible_fleet_principles.qmd` for responsible-fleet, governance, sustainability, security, privacy, robustness, fairness, and feedback-loop numeric claims; displayed equations; scaling/statistical claims; and prose-equation consistency using direct reasoning only. No Gemini assistance was used.

The file contains four displayed equations/relations and several quantitative or quasi-quantitative claims: positive mutual information leakage, differential-privacy budget accounting, an `8x-10x` adversarial-robustness compute multiplier, a Jevons-style `10x` efficiency / `100x` usage example, base-rate-conditioned fairness impossibility, and a distribution feedback equation.

## Findings

### High: The mutual-information privacy equation is too strong as a universal invariant

- **Lines:** 7-11
- **Current text/equation:** "Every model output potentially leaks information about its training data. Perfect privacy is mathematically impossible if the model remains useful." The displayed equation is `I(\text{TrainingData}; \text{ModelOutput}) > 0`.
- **Issue:** The prose says "potentially leaks," but the equation asserts strictly positive mutual information for every useful model/output. That is stronger than the claim supports. A model output can be independent of a particular training dataset in trivial cases, in deliberately private mechanisms, or when it encodes population-level information rather than identifiable dependence on the realized training sample. For privacy risk, the more relevant object is often leakage about individual records or membership, not mutual information with the entire training dataset. Differential privacy also does not make `epsilon` a direct mutual-information cap "per query"; `epsilon` bounds a worst-case likelihood ratio between neighboring datasets, and privacy loss composes across mechanisms/queries.
- **Proposed correction:** Recast the equation as a risk bound rather than a universal strict inequality. For example: "Useful learned models can leak information about training records, so systems should bound the influence of any one record." Then use a DP-style relation such as `Pr[M(D) \in S] \le e^\epsilon Pr[M(D') \in S] + \delta` for neighboring datasets `D,D'`, and describe the privacy budget as composing across accesses rather than as mutual information per query.

### Medium: The adversarial-robustness compute multiplier is over-specific

- **Lines:** 14-17
- **Current text:** "Achieving intrinsic adversarial robustness requires training on perturbations, which demands ~8x-10x more compute per epoch than standard optimization."
- **Issue:** The direction is reasonable for common multi-step adversarial training, but the `8x-10x` multiplier is not a mathematical invariant. The overhead depends on the attack/training method, number of perturbation steps, whether extra forward/backward passes are reused, model architecture, threat model, and whether the defense is PGD-style adversarial training, single-step training, randomized smoothing, certified training, data augmentation, or another method. A `k`-step inner maximization can plausibly cost several times standard training, but not universally `8x-10x` per epoch.
- **Proposed correction:** Scope the number as an illustrative range. For example: "Multi-step adversarial training often increases per-epoch compute by several-fold, and PGD-style configurations can reach roughly 8x-10x depending on the number of attack steps and implementation."

### Medium: The Jevons example turns an elasticity possibility into a numeric prediction

- **Lines:** 20-24
- **Current text/equation:** The callout states `Efficiency up -> Cost down -> Demand up up`, then says "Making models 10x more efficient will likely lead to 100x more usage, not 10x energy savings."
- **Issue:** The qualitative Jevons mechanism is directionally coherent, but the `10x` efficiency to `100x` usage claim is not implied by the equation. Total energy is proportional to `usage * energy_per_use`; a `10x` efficiency improvement reduces energy per use to `1/10`, so total energy falls only if usage rises by less than `10x`, stays flat at `10x`, and increases if usage rises by more than `10x`. A `100x` usage response would make total energy about `100/10 = 10x` higher, but that requires a very high demand response and is not a general consequence of Jevons' paradox.
- **Proposed correction:** Make the rebound arithmetic explicit and avoid the fixed prediction. For example: "If efficiency improves by `10x`, total energy decreases only if demand rises by less than `10x`; a rebound to `100x` usage would increase total energy by about `10x`. Sustainability strategies therefore need absolute energy/carbon constraints in addition to rate efficiency."

### Medium: The fairness impossibility statement needs non-degeneracy and metric definitions

- **Lines:** 27-31
- **Current text/equation:** "It is mathematically impossible to simultaneously satisfy Calibration, Equalized Odds, and Demographic Parity when base rates differ between groups." The displayed implication is `P(Y=1|A=a) \neq P(Y=1|A=b) \implies \text{Trade-off Required}`.
- **Issue:** The broad direction is consistent with standard fairness impossibility results, but the statement omits important conditions. These impossibility theorems depend on how calibration is defined, whether the object is a score or a binary decision, and whether degenerate or perfect predictors are excluded. Equalized odds plus differing base rates conflicts with demographic parity for any nontrivial classifier with unequal true- and false-positive rates, because each group's positive prediction rate is `TPR * pi_A + FPR * (1 - pi_A)`. Calibration has its own incompatibility with equalized-odds-style balance under differing base rates for imperfect predictors. As written, the equation says base-rate difference alone is sufficient for all fairness trade-offs without stating the required nontriviality/imperfect-prediction assumptions.
- **Proposed correction:** Add the missing scope. For example: "When groups have different base rates, imperfect non-degenerate predictors generally cannot simultaneously satisfy calibration within groups, equalized-odds error-rate parity, and demographic parity; at least one metric must be relaxed or prioritized."

### Low: The sociotechnical feedback equation has a type/scope mismatch

- **Lines:** 34-38
- **Current text/equation:** "The probability distribution of future data `P_{t+1}(X)` is a function of the model's past decisions `f_t(X)`." The displayed equation is `P_{t+1}(X) = g(P_t(X), f_t(X))`.
- **Issue:** The equation is useful as a schematic, but it mixes a distribution `P_t(X)` with a pointwise model output `f_t(X)`. Future data distribution normally depends on the deployed policy's induced action distribution, user responses, selection effects, and exogenous shocks, not just the function evaluated on a symbolic `X`. The equality also makes the transition look deterministic and complete.
- **Proposed correction:** Clarify that it is a schematic or use policy/distribution notation. For example: `P_{t+1}(X) = g(P_t(X), \pi_t, E_t)`, where `\pi_t` is the deployed decision policy and `E_t` represents external factors, or state that `g` is an abstract transition model rather than a closed-form equality.

### Low: Section identifier uses a conflicting part number

- **Lines:** 1, 41
- **Current text:** The file title and roadmap heading say "Part IV," but the explicit section id includes `part-viii`: `#sec-principles-responsible-part-viii-roadmap-governance-responsibility-e8f1`.
- **Issue:** The rendered heading text is internally consistent, but the identifier encodes a different part number. This can create stale cross-reference labels and is a prose/count consistency defect for maintainers.
- **Proposed correction:** Rename the id to use `part-iv` if this file is intended to be Part IV of Volume 2, or update the visible part numbering if the global numbering scheme requires Part VIII.

## No-Issue Checks

- **Lines 3-5:** The governance/security/sustainability framing is qualitative. It contains no numeric conversion or equation requiring correction.
- **Lines 20-22:** The displayed Jevons relation is a qualitative monotonic chain, not an algebraic equation. The issue is the later `10x` to `100x` prediction, not dimensional consistency.
- **Lines 43-48:** The roadmap says the part secures and governs the fleet through four chapters, and the numbered list contains exactly four items.

# Math Audit Report: `book/quarto/contents/vol1/parts/deploy_principles.qmd`

## Checked scope

Audited `book/quarto/contents/vol1/parts/deploy_principles.qmd` for mathematical statements, displayed equations, numeric claims, unit consistency, complexity claims, and prose-equation consistency using direct reasoning only. The file contains four displayed equations, several latency and percentage examples, and qualitative claims about statistical drift, training-serving skew, and feedback loops. No Gemini assistance was used.

## Findings

### Low: Verification probability uses an undefined approximate-equality event

- **Lines:** 6-7
- **Current text/equation:** Machine-learning behavior is verified using statistical bounds:
  `$$ P(f(X) \approx Y) > 1 - \epsilon $$`
- **Issue:** The expression is directionally understandable, but `f(X) \approx Y` is not a well-defined event without a task-specific tolerance, distance, or loss threshold. For classification, approximate equality may mean exact class agreement or top-k agreement; for regression, it requires a tolerance such as `|f(X)-Y| \le \delta`; for structured outputs, it requires a metric. The probability is also implicitly over the joint distribution of `(X,Y)`, which is not stated.
- **Proposed correction:** Define the event through a metric or loss. For example:
  `$$ \Pr_{(X,Y)\sim P}\!\left[d(f(X),Y) \le \delta\right] \ge 1-\epsilon $$`
  or, for loss-based wording:
  `$$ \Pr_{(X,Y)\sim P}\!\left[\ell(f(X),Y) \le \tau\right] \ge 1-\epsilon $$`

### Medium: Drift equation overstates what distribution distance alone can determine

- **Lines:** 17-19
- **Current text/equation:** Accuracy degradation is described as governed by:
  `$$ \text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda \cdot D(P_t \| P_0) $$`
- **Issue:** The units can be made consistent if accuracy and `D(P_t\|P_0)` are dimensionless and `\lambda` converts divergence units into accuracy points. The mathematical problem is that a marginal distribution distance by itself does not determine accuracy change. Accuracy depends on the model, the label relationship, class priors, support overlap, and decision boundaries. A distribution shift can reduce accuracy, leave it unchanged, or even improve observed accuracy if the current population becomes easier. The text partly limits the equation as a first-order linearization for small shifts, but the preceding prose says degradation is "governed by" the equation, which is stronger than the approximation supports. The linear form can also predict negative accuracy for large drift unless it is explicitly local or clipped.
- **Proposed correction:** Weaken the invariant to a local risk model. For example: "Accuracy often degrades with task-relevant drift, and a local first-order approximation is..." Then write:
  `$$ \text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda\,D_{\text{task}}(P_t,P_0) \quad \text{for small, harmful shifts} $$`
  Add that the approximation is local and should be bounded to the valid accuracy range `[0,1]`.

### Medium: Training-serving skew claim says divergence can only hurt accuracy

- **Lines:** 27-29
- **Current text/equation:** The text states that effective accuracy degrades proportionally to:
  `$$ \Delta \text{Accuracy} \propto \mathbb{E}[|f_{\text{serve}}(x) - f_{\text{train}}(x)|] $$`
  and then says any divergence "can only hurt accuracy, never help it."
- **Issue:** The expected output difference is not, by itself, an accuracy loss. Its units are output units, while accuracy is dimensionless, so a proportionality constant with appropriate units and task dependence is implicit. More importantly, the universal "can only hurt" statement is false as a mathematical claim. A serving-time difference could coincidentally improve predictions on a shifted population, leave class decisions unchanged away from boundaries, or change confidence without changing accuracy. The safer statement is that skew invalidates the trained/evaluated performance guarantee and can cause unpredictable performance changes.
- **Proposed correction:** Replace the universal monotonic claim with a risk or bound statement. For example:
  `$$ |\Delta L| \lesssim C\,\mathbb{E}[d(f_{\text{serve}}(x), f_{\text{train}}(x))] $$`
  under an appropriate Lipschitz-style loss assumption. In prose: "Any divergence breaks the assumption that serving matches evaluated training behavior and can degrade accuracy, especially near decision boundaries."

### Low: Feature-store prose promises exact functional equivalence

- **Lines:** 31
- **Current text/equation:** Feature stores are described as consistency engines that "guarantee `$f_{\text{serve}} \equiv f_{\text{train}}$`."
- **Issue:** Exact functional equivalence is stronger than feature stores can generally guarantee. A feature store can reduce skew by sharing feature definitions, materialization, and point-in-time semantics, but equality can still fail because of model-code differences, library versions, precision, request-time transforms, missing values, feature freshness, or environment-dependent behavior. This also conflicts with the next sentence, which lists PIL vs. OpenCV and float64 vs. float32 differences as possible sources of degradation.
- **Proposed correction:** Change "guarantee" to a scoped claim, such as: "Feature stores help enforce shared feature definitions and point-in-time consistency so that `$f_{\text{serve}}` approximates `$f_{\text{train}}` as closely as the serving stack permits."

### Medium: P99 latency equation treats a tail constraint as a deterministic sum

- **Lines:** 37-40
- **Current text/equation:** The section says P99 latency is the hard constraint and gives:
  `$$ L_{\text{lat,total}} = L_{\text{lat,net}} + L_{\text{lat,pre}} + L_{\text{lat,infer}} + L_{\text{lat,post}} + L_{\text{lat,queue}} \leq \text{SLO} $$`
- **Issue:** The additive decomposition is dimensionally valid if each term is a latency duration. The inconsistency is that P99 latency is a quantile of a random end-to-end latency distribution, not simply a deterministic sum of component latencies. If each component is random, `P99(A+B)` is not generally equal to `P99(A)+P99(B)`, and correlations between queueing, inference time, and network time matter. The equation is fine as a per-request latency accounting identity, but it does not by itself express the P99 SLO stated in the prose.
- **Proposed correction:** Either scope the equation to per-request latency:
  `$$ L_{\text{total}}(r)=L_{\text{net}}(r)+L_{\text{pre}}(r)+L_{\text{infer}}(r)+L_{\text{post}}(r)+L_{\text{queue}}(r) $$`
  followed by `$$ Q_{0.99}(L_{\text{total}}) \le \text{SLO} $$`
  or explicitly state that the displayed sum is a budget allocation approximation for tail latency, not an exact quantile identity.

### Low: Latency section says the previous four principles all degrade accuracy

- **Lines:** 43
- **Current text:** "The previous four principles address failures that degrade *accuracy*; this final principle addresses a failure that degrades *equity*..."
- **Issue:** Of the previous four principles, the latency-budget principle does not primarily degrade accuracy. It degrades timeliness, availability, utility, or safety when predictions arrive after the deadline. This conflicts with lines 34-40, which correctly frame latency as a hard real-time constraint rather than an accuracy metric.
- **Proposed correction:** Revise the sentence to separate statistical correctness from operational usefulness. For example: "The previous principles address failures that degrade statistical reliability or operational usefulness; this final principle addresses a failure that degrades equity."

### Medium: Bias-feedback equation needs bounded-range and stability qualifications

- **Lines:** 48-50
- **Current text/equation:** For group `g`, disparity after `k` deployment cycles grows as:
  `$$ \Delta_g(k) \approx \Delta_g(0) \cdot \alpha^k, \quad \alpha > 1 $$`
- **Issue:** The recurrence is a reasonable local model for self-reinforcing disparity, but it is too absolute as written. If `\Delta_g` is a performance gap, error-rate gap, or approval-rate disparity, it is bounded by a finite interval, usually within `[-1,1]` or `[0,1]` depending on definition. Pure exponential growth will eventually exceed the feasible range unless the equation is identified as a local early-cycle approximation or clipped/saturating model. Also, feedback systems can have `\alpha < 1`, `\alpha = 1`, or `\alpha > 1`; `\alpha > 1` is the unstable/self-amplifying case, not a necessary property of every model-output feedback loop.
- **Proposed correction:** Make the stability condition explicit:
  `$$ \Delta_g(k+1) \approx \alpha\,\Delta_g(k), \qquad \alpha>1 \text{ indicates amplification} $$`
  Then add that the exponential form is valid only while disparities are small enough that saturation and intervention effects are negligible.

## No-Issue Checks

- **Lines 15 and 34:** The numeric examples, "two years later," "99 percent accuracy," "30 seconds per scan," "200 ms," and "50 ms," are illustrative and do not require unit conversion. The time units are internally clear.
- **Lines 18-19:** Aside from the scope issue noted above, the drift equation can be dimensionally consistent if `D(P_t\|P_0)` is dimensionless and `\lambda` is measured in accuracy points per divergence unit.
- **Line 28:** The absolute-value expectation is syntactically valid for scalar model outputs. For vector outputs, the expression would need a norm, but the local prose examples do not force a vector interpretation.
- **Line 38:** The latency-budget terms all have units of time, and comparing their sum to an SLO duration is dimensionally valid.
- **Lines 49-50:** The exponential feedback equation is algebraically consistent for a discrete-cycle multiplicative model once `\alpha` is interpreted as a dimensionless amplification factor.
- **Line 55:** The closing summary correctly counts five principles: verification gap, statistical drift, training-serving skew, latency budget, and bias feedback.

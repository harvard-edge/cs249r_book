# Float Exposition Audit — `ops_scale.qmd` (vol2)

**Standard:** FLOAT_EXPOSITION_STANDARD.md
**Float count:** 82 (20 eq 🔴 · 13 fig 🟠 · 10 lst 🟡 · 39 tbl 🟠)
**Dangling ref:** `@eq-distributed-training-scaling-efficiency` at L1604 (no def in this chapter)

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| eq   | 🔴    |     20 | 14 |  5 |  1 |
| fig  | 🟠    |     13 | 10 |  3 |  0 |
| lst  | 🟡    |     10 |  6 |  4 |  0 |
| tbl  | 🟠    |     39 | 35 |  4 |  0 |
| **Total** |  | **82** | **65** | **16** | **1** |

---

## Findings (⚠️ and 🛑 only)

---

### EQUATIONS 🔴

---

#### ⚠️ `eq-demographic-parity` (eq 🔴) — def L2142

**Verbatim ref sentence (L2138):**
> "@Eq-demographic-parity expresses one operational choice: the probability of a positive prediction must differ by less than threshold $\epsilon$ between protected groups $a$ and $b$."

**What is missing:** The cite sentence names what the equation says but delivers no lead-out. The prose never states the implication or regime: what a practitioner should do when the threshold is violated, what $\epsilon$ value is operationally meaningful, or why the difference-in-probabilities formulation (as opposed to equalized odds) carries different consequences for deployment. The payoff paragraph (L2150) pivots immediately to data validation without returning to the fairness equations. Removability test fails: deleting both fairness equations leaves the surrounding prose fully intact.

**Where takeaway currently lives:** Partially in the footnote about the Chouldechova-Kleinberg impossibility result — but footnotes do not count.

**Rule-compliant diff rewrite (add to the paragraph at L2138, after the two-equation citation):**

```diff
- @Eq-demographic-parity expresses one operational choice: the probability of a positive prediction must differ by less than threshold $\epsilon$ between protected groups $a$ and $b$. @Eq-equalized-odds encodes a stricter choice: prediction behavior must be similar across groups after conditioning on the true outcome.
+ @Eq-demographic-parity encodes demographic parity as an operational gate: the positive-prediction rate for group $a$ must stay within $\epsilon$ of the rate for group $b$, where the platform sets $\epsilon$ to reflect the organization's risk tolerance (a common starting value is 0.05). @Eq-equalized-odds is the stricter alternative, conditioning on the true outcome so that a model cannot meet the demographic-parity threshold merely by predicting positive uniformly across groups. The practical consequence is that demographic parity can be satisfied by a model that discriminates within each outcome class, while equalized odds cannot. Choosing between them is a policy decision the platform must encode before training, because swapping definitions after deployment invalidates all prior validation records.
```

---

#### ⚠️ `eq-equalized-odds` (eq 🔴) — def L2144

**Verbatim ref sentence (L2138):**
> "@Eq-equalized-odds encodes a stricter choice: prediction behavior must be similar across groups after conditioning on the true outcome."

**What is missing:** Symbol $Y$ (true outcome) and the false-positive/false-negative rate meaning are not named in prose. The Interpret move is absent: no statement of when equalized odds is legally or operationally required versus demographic parity, no worked regime example. The payoff paragraph does not return to this equation.

**Where takeaway currently lives:** Caption-equivalent prose inside the same sentence; no lead-out in body prose.

**Rule-compliant diff rewrite:** Addressed by the combined rewrite above for `eq-demographic-parity`. Both equations share one introduction paragraph; the fix merges their exposition into a single, complete Interpret move.

---

#### 🛑 `eq-tco-ml` (eq 🔴) — def L3631

**Verbatim ref sentence (L3629):**
> "@Eq-tco-ml expresses the total cost of ownership as the sum of four distinct cost components, each with different scaling characteristics and optimization levers:"

**What is missing:** The lead-in announces four components but the body prose delivers no symbol-level explanation and no stated consequence. The payoff paragraph (L3633) immediately pivots to `@fig-tco-iceberg` with only "hidden operational costs often constitute fully half of the actual budget" — a caption restatement, not an equation Interpret move. No sentence tells the reader what the additive structure implies (that components can be optimized independently, that the dominant term shifts with organizational maturity). The equation's purpose — distinguishing four cost regimes so practitioners know which lever to pull — is never articulated in body prose. Removability test: deleting the equation leaves the prose fully intact because the prose never actually uses it.

**Where takeaway currently lives:** Scattered across subsequent subsections (training cost model at L3741, inference cost at L3778, etc.) with no synthesis tied back to the top-level equation.

**Rule-compliant diff rewrite (replace L3629 paragraph and add a lead-out before the figure):**

```diff
- @Eq-tco-ml expresses the total cost of ownership as the sum of four distinct cost components, each with different scaling characteristics and optimization levers:
+ @Eq-tco-ml decomposes ML total cost of ownership into four independently optimizable components: $C_{\text{train}}$ scales with GPU count and experiment volume, $C_{\text{infer}}$ scales with query rate and model size, $C_{\text{data}}$ grows superlinearly with user base, and $C_{\text{iter}}$ reflects engineering investment in experimentation. The additive structure is the key engineering insight: because each term has a different scaling law and a different set of optimization levers, the dominant cost component shifts as organizations mature. Early-stage efforts are iteration-dominated; production deployments at scale become inference-dominated. Targeting the wrong component wastes optimization effort regardless of technique quality.
```

And after the equation, replace the current pivot sentence:

```diff
- As @fig-tco-iceberg illustrates, while GPU compute and storage are the visible costs, hidden operational costs often constitute fully half of the actual budget.
+ @Fig-tco-iceberg maps the four components against their organizational visibility: GPU compute and storage appear immediately in procurement budgets, while engineering labor, maintenance, and compliance — which can together constitute half the true budget — remain below the waterline until a formal TCO analysis is conducted.
```

---

#### ⚠️ `eq-canary-duration` (eq 🔴) — def L2231

**Verbatim ref sentence (L2229):**
> "@Eq-canary-duration relates stage duration to sample requirements, request rate, and traffic percentage, enabling precise calculation of minimum canary durations for statistical validity:"

**What is missing:** The lead-in names the relationship but the Interpret move is thin. The prose never states the key operational implication of the inverse relationship: increasing traffic percentage compresses stage duration proportionally, so the decision about what percentage to assign each stage is actually a decision about how long the platform must wait before promoting. The worked example at L2237 does numeric substitution but does not articulate the design principle. Removability test: the worked example would still make sense without the equation label, because the calculation is spelled out inline.

**Where takeaway currently lives:** Implicit in the worked example but not stated as a design principle.

**Rule-compliant diff rewrite (add one sentence after the where-clause at L2233):**

```diff
 where $T_{\text{stage}}$ is the duration required at a given percentage, $n_{\text{samples needed}}$ is the number of observations needed for statistical significance, $r_{\text{requests}}$ is the request rate, and $p_{\text{stage}}$ is the traffic percentage.
+
+ The inverse dependence on $p_{\text{stage}}$ means each traffic-percentage decision is simultaneously a duration decision: doubling the canary share halves the required soak time, but also doubles the user-facing blast radius if a regression is present. Platform teams use this trade-off to set stage percentages that bound the worst-case exposure while keeping the rollout within an operational shift.
```

---

#### ⚠️ `eq-rollout-risk` (eq 🔴) — def L2804

**Verbatim ref sentence (L2802):**
> "@Eq-rollout-risk formalizes deployment risk as the product of regression probability, impact severity, and exposure level, providing a quantitative foundation for risk-based rollout decisions:"

**What is missing:** The lead-in is a functional label, not an Interpret move. The immediate payoff (L2808) lists three mitigation strategies — reduce $p$, reduce $I$, reduce $E$ — but these are already printed as bulleted labels derived directly from the equation structure, not a prose explanation of why the multiplicative form matters. The key implication — that any factor can independently reduce total risk to zero, so a high-impact change can still proceed under low exposure — is not stated. Removability test: the three-bullet list at L2808 reads naturally without the equation because it only restates the factor names.

**Where takeaway currently lives:** In the three mitigation bullets, which are equation restatements rather than prose interpretation.

**Rule-compliant diff rewrite (replace L2808 paragraph):**

```diff
- The rollout risk framework suggests three mitigation strategies:
-
- - Reduce $p_{\text{regression}}$: More thorough testing before deployment
- - Reduce $I_{\text{regression}}$: Architectural patterns that limit blast radius
- - Reduce $E_{\text{exposure}}$: Slower rollouts with lower initial traffic percentages
+ The multiplicative structure of @Eq-rollout-risk reveals why staged rollout is a principled safety mechanism rather than just operational caution. Because all three factors multiply together, reducing any single factor proportionally reduces total risk, meaning even a high-probability regression can be tolerated under sufficiently low exposure. A model with a 40 percent estimated regression probability and high revenue impact becomes acceptable at 0.1 percent canary traffic because $E_{\text{exposure}} = 0.001$ drives the product below the platform's risk threshold. The three handles are not equivalent, however: reducing $I_{\text{regression}}$ through architectural blast-radius controls (serving isolation, circuit breakers) offers permanent, deployment-independent protection, while reducing $p_{\text{regression}}$ through testing requires repeating the testing investment for every release.
```

---

#### ⚠️ `eq-platform-utilization` (eq 🔴) — def L3426

**Verbatim ref sentence (L3424):**
> "@Eq-platform-utilization defines platform efficiency as the capacity-weighted average utilization across all resources:"

**What is missing:** The cite sentence is a definition label, not an Interpret move. The payoff (L3430) pivots to three dimensions of "effective utilization" (quality, fairness, cost) but never ties back to why the capacity-weighted average specifically matters over simple average utilization, or what its failure mode is. The equation's engineering relevance — that weighting by capacity prevents a single over-provisioned low-utilization resource from distorting the fleet-wide metric — is absent from prose. Removability test: L3430 reads cleanly without the equation.

**Where takeaway currently lives:** Not present in body prose; implied by the formula structure only.

**Rule-compliant diff rewrite (add after the where-clause at L3428):**

```diff
 where $U_i$ is the utilization of resource $i$ and $\text{Cap}_i$ is the capacity of resource $i$.
+
+ The capacity weighting matters because a naive arithmetic average would allow a small, fully-utilized resource to mislead a large, mostly-idle cluster into appearing healthy. By weighting each resource by its provisioned capacity, @Eq-platform-utilization ensures that a 100-GPU cluster sitting at 30 percent utilization dominates the metric even when every 2-GPU development node nearby runs at 100 percent. A low value of $U_{\text{platform}}$ therefore signals genuinely stranded capacity in the high-weight resources, which is where optimization work pays off.
```

---

### FIGURES 🟠

---

#### ⚠️ `fig-n-models-complexity` (fig 🟠) — def L191

**Verbatim ref sentence (L189):**
> "@Fig-n-models-complexity visualizes this superlinear growth across three complexity dimensions. Monitoring alerts grow linearly with model count, but dependency conflicts grow quadratically as models share features, data sources, and infrastructure. The total operational load crosses team capacity around 50 models, the empirical threshold where organizations discover they need platform engineering."

**What is missing:** The cite sentence delivers the linear vs. quadratic observation but the Interpret move is thin. The prose names the 50-model threshold but does not state what the figure demonstrates about the shape of the total-load curve or why the three distinct growth curves matter operationally. Specifically: the figure shows three separate lines converging into a total load, and the Interpret move should explain what that convergence implies for triage priority — at small scale, monitoring dominates; at mid-scale, coordination costs surge; at large scale, dependency conflicts dominate. The payoff paragraph (L268) is far downstream and covers the ROI equation, not the complexity figure. Removability test: the sentence at L189 could be deleted without breaking the preceding or following argument.

**Where takeaway currently lives:** Partially in the caption (the $\mathcal{O}(N \log N)$ annotation), which does not count.

**Rule-compliant diff rewrite (replace L189):**

```diff
- @Fig-n-models-complexity visualizes this superlinear growth across three complexity dimensions. Monitoring alerts grow linearly with model count, but dependency conflicts grow quadratically as models share features, data sources, and infrastructure. The total operational load crosses team capacity around 50 models, the empirical threshold where organizations discover they need platform engineering.
+ @Fig-n-models-complexity separates the three drivers that produce superlinear total load. Monitoring alerts track linearly because each new model adds a fixed alerting surface. Deployment coordination grows as $\mathcal{O}(N \log N)$ because coordinating $N$ models requires approximately $N \log N$ ordering and sequencing decisions. Dependency conflicts grow quadratically because each new model can potentially conflict with every other model sharing features or data sources. Below 10 models, alert volume dominates operational cost and ad hoc practices suffice. Near 50 models, all three curves converge and the total crosses team capacity simultaneously: the organization cannot scale monitoring, coordination, and dependency resolution independently. That convergence is the structural argument for platform investment — not that any single curve is steep, but that three separate cost drivers peak at the same scale.
```

---

#### ⚠️ `fig-interleaving-vs-ab` (fig 🟠) — def L2518

**Verbatim ref sentence (L2516):**
> "Interleaving is essential for recommendation systems where detecting small engagement changes quickly enables rapid iteration, as @fig-interleaving-vs-ab contrasts with traditional A/B testing."

**What is missing:** The cite sentence is a pure float-announcer ("contrasts with"). It names the topic but delivers no Interpret move: what the contrast shows, why within-user comparison provides higher sensitivity, or what the statistical consequence is. The payoff (L2571) discusses A/B testing challenges but never returns to interpret the figure's specific contrast. The mechanism — that within-user attribution removes user-preference variance as a confounder — is present only in the caption alt-text context, not in body prose.

**Where takeaway currently lives:** Caption only.

**Rule-compliant diff rewrite (replace L2516):**

```diff
- Interleaving is essential for recommendation systems where detecting small engagement changes quickly enables rapid iteration, as @fig-interleaving-vs-ab contrasts with traditional A/B testing.
+ Interleaving achieves higher sensitivity than A/B testing by turning each user into their own control, as @fig-interleaving-vs-ab shows. In A/B testing, users in the control group may simply prefer different content regardless of model quality, adding user-preference variance to the treatment effect estimate. In interleaving, both rankers contribute items to the same blended list served to the same user: a click on a ranker-B item while a ranker-A item was available is a direct quality comparison under identical user context. This within-user attribution removes the between-user variance that inflates A/B test confidence intervals, allowing teams to detect engagement differences of 0.1 to 0.5 percent that would require months of A/B traffic to reach significance.
```

---

#### ⚠️ `fig-ml-org-models` (fig 🟠) — def L4390

**Verbatim ref sentence (L4388):**
> "@Fig-ml-org-models places that pattern alongside embedded and hybrid alternatives, each trading off consistency against velocity."

**What is missing:** The citation is a placement statement without interpretation. The figure shows three org structures, but the body prose at L4388 never states which axis of the tradeoff is depicted in the figure, what the hybrid's structural difference from centralized is, or why the three patterns exist on a spectrum rather than being discrete choices. The payoff paragraphs (L4461 onward) provide good Interpret prose for centralized and embedded patterns but never connect back to the figure as a frame. Removability test: the figure can be removed from L4388 without changing the argument, because the prose simply continues describing the patterns independently.

**Where takeaway currently lives:** The payoff at L4461 provides interpretation but does not reference the figure.

**Rule-compliant diff rewrite (replace L4388):**

```diff
- A centralized ML platform team maximizes consistency by building and maintaining shared infrastructure while model teams focus on model development. @Fig-ml-org-models places that pattern alongside embedded and hybrid alternatives, each trading off consistency against velocity.
+ A centralized ML platform team maximizes consistency by building and maintaining shared infrastructure while model teams focus on model development. @Fig-ml-org-models maps all three patterns against the same consistency-velocity axis: centralized (left) pools expertise and enforces uniform standards, embedded (center) places specialists inside product teams for rapid domain response, and hybrid (right) maintains a shared infrastructure core while seeding model teams with platform-trained engineers. The figure's value is showing that hybrid is not a compromise between two extremes but a structural split: the core platform owns the invariants that recur across all teams (serving infrastructure, monitoring, governance), while embedded engineers own the domain-specific judgment that the core cannot generalize.
```

---

### LISTINGS 🟡

---

#### ⚠️ `lst-pipeline-params` (lst 🟡) — def L2030

**Verbatim ref sentence (L2028):**
> "Effective pipelines separate configuration from code, as illustrated in @lst-pipeline-params."

**What is missing:** This is a minimal float-announcer ("as illustrated in"). The prose delivers no mechanism framing: what the reader should notice in the YAML, which keys are doing the separating work (the `schema_version` pinning data contracts, the distinct `hyperparameters` section that can be overridden independently), or what the design choice is. The payoff at L2055 delivers a good consequence sentence but it arrives after the code and does not anchor back to what the listing specifically demonstrates. Medium-level standard requires orientation to the mechanism before the code.

**Where takeaway currently lives:** L2055 payoff (arrives post-code); no pre-code framing.

**Rule-compliant diff rewrite (replace L2028):**

```diff
- Effective pipelines separate configuration from code, as illustrated in @lst-pipeline-params.
+ Effective pipelines encode the separation of concerns in their file structure. @Lst-pipeline-params demonstrates this by placing data contract pins (the `schema_version` key), feature declarations, hyperparameters, and evaluation thresholds each in their own top-level YAML block. Notice that model code appears nowhere in the file: it is addressed by `model_type` label, which lets the platform substitute model implementations without touching the configuration record. The evaluation block's threshold keys are the structural detail worth examining — they turn a runtime quality check into a versioned artifact that the platform can compare across candidates.
```

---

#### ⚠️ `lst-cost-attribution` (lst 🟡) — def L3313

**Verbatim ref sentence (L3311):**
> "Effective cost management requires attributing costs to organizational units. Tag-based allocation assigns costs based on resource metadata, as shown in @lst-cost-attribution."

**What is missing:** The two-sentence lead-in establishes context but the second sentence is a bare pointer. The mechanism — which tags are doing the attribution work, what the platform reads to determine team vs. service vs. environment ownership, and what the design choice is between tag dimensions — is entirely in the code. A reader who has not read the code does not know what to look for. The payoff (L3337) pivots to shared-cost allocation policies without returning to interpret the listing.

**Where takeaway currently lives:** Code structure and tag names only.

**Rule-compliant diff rewrite (replace L3311):**

```diff
- Effective cost management requires attributing costs to organizational units. Tag-based allocation assigns costs based on resource metadata, as shown in @lst-cost-attribution.
+ Effective cost management requires attributing costs to organizational units. @Lst-cost-attribution implements tag-based allocation by enforcing four required tag dimensions on every resource: `team`, `service`, `environment`, and `cost_center`. The design choice worth examining is the separation of `team` from `cost_center`: a single cost center can contain multiple teams (relevant for shared-resource accounting), while a single team can own services across multiple environments with different budget envelopes. The `shared_cost_policy` block at the bottom — the part most configuration schemas omit — is where the platform encodes how undifferentiated shared infrastructure charges are distributed, either by proportional usage or flat allocation.
```

---

#### ⚠️ `lst-anomaly-attribution` (lst 🟡) — def L3079

**Verbatim ref sentence (L3077):**
> "@Lst-anomaly-attribution shows a fleet-wide correlation detector that attributes simultaneous anomalies to shared causes."

**What is missing:** The cite sentence is a minimal content label. No mechanism framing: what the correlation threshold `0.6` represents, why simultaneous detection of anomalies across models is the right signal for shared-infrastructure failures (as opposed to per-model thresholds), or what the reader should inspect in the code. The prior bullet list at L3073-3075 sets up the requirements but does not connect to what the listing's specific implementation choice is.

**Where takeaway currently lives:** Implicit in the function signature and comments only.

**Rule-compliant diff rewrite (replace L3077):**

```diff
- @Lst-anomaly-attribution shows a fleet-wide correlation detector that attributes simultaneous anomalies to shared causes.
+ @Lst-anomaly-attribution implements the correlation-before-attribution pattern: instead of firing a per-model alert when any model's metric crosses a threshold, the function first measures which models are anomalous simultaneously (using the `threshold=0.6` parameter for the cross-model correlation score) and routes correlated anomalies to a shared-infrastructure cause. The key design choice is the correlation coefficient threshold rather than a count threshold: high correlation across dissimilar models (recommendation, fraud, vision) indicates the anomaly originates in a shared layer such as the feature store or serving infrastructure, not in any individual model. Without this cross-model correlation step, the fleet would generate $N$ per-model alerts for what is actually one infrastructure incident.
```

---

#### ⚠️ `lst-nccl-debug` (lst 🟡) — def L4683

**Verbatim ref sentence (L4679):**
> "NCCL collective operations can fail silently or hang indefinitely, and @lst-nccl-debug shows how debug logging identifies blocked ranks."

**What is missing:** The cite sentence delivers a minimal content label but no mechanism framing. It does not tell the reader which environment variables to look at first, why the rank-verbose level is set differently from the collective verbose level, or what pattern in the log output signals a blocked rank versus a slow rank. The payoff at L4696 provides a diagnostic sequence but does not anchor it to the listing's specific configuration choices.

**Where takeaway currently lives:** Variable names visible in code only; no prose orientation.

**Rule-compliant diff rewrite (replace the second sentence at L4679):**

```diff
- NCCL collective operations can fail silently or hang indefinitely, and @lst-nccl-debug shows how debug logging identifies blocked ranks.
+ NCCL collective operations can fail silently or hang indefinitely, and @lst-nccl-debug shows the environment variable pattern that converts the silence into a rank-level trace. The key variable is `NCCL_DEBUG_SUBSYS=ALL` combined with `NCCL_DEBUG=TRACE`: the subsystem flag captures both collective and transport layers, while the trace level emits per-rank send and receive events so the log can show which rank sent its contribution and which rank never received it. When a collective hangs, a grep for `[rank N]` lines that stop before the matching receive confirms the blocked rank and narrows the investigation to that device's network interface or memory state before touching any other part of the cluster.
```

---

### TABLES 🟠

---

#### ⚠️ `tbl-ops-scale-telemetry` (tbl 🟠) — def L888

**Verbatim ref sentence (L890):**
> "Notice in @tbl-ops-scale-telemetry that volume grows from constant (metrics) through linear (logs) to super-linear (traces) with request rate, which is why effective platforms present high-level metric dashboards by default and enable investigation into lower levels only when anomalies are detected."

**What is missing:** The prose arrives after the table and reads as a caption restatement. The lead-in before the table is absent entirely — the table appears at L888 with no body-prose introduction before it (the bundle shows the prev paragraph is the caption line itself at L888). The table is cited only post-hoc at L890. This means there is no lead-in at all before the float.

**Where takeaway currently lives:** The post-table sentence (L890) partially delivers the Interpret move, but the Lead-in move is absent.

**Rule-compliant diff rewrite (add a lead-in sentence before the table at L888, where the caption currently is the first reference):**

The fix requires a body-prose sentence before the table definition. The table definition line at L888 is actually the caption, meaning the table appears in the flow with no prior introduction. Insert one sentence of prose before the caption:

```diff
+Telemetry channels differ not just in what they record but in how their volume scales with request rate, which determines how the platform must tier its monitoring infrastructure to remain operable as the fleet grows.
+
 : **Telemetry Paradigms at Scale**: ...
```

And tighten the post-table sentence to complete the Interpret move rather than repeat it:

```diff
- Notice in @tbl-ops-scale-telemetry that volume grows from constant (metrics) through linear (logs) to super-linear (traces) with request rate, which is why effective platforms present high-level metric dashboards by default and enable investigation into lower levels only when anomalies are detected.
+ @Tbl-ops-scale-telemetry shows the practical consequence: metrics can be continuously pulled without storage growth, logs must be sampled aggressively above moderate traffic, and traces require always-on sampling at below 1 percent for most fleets to avoid making monitoring itself a significant cost center.
```

---

#### ⚠️ `tbl-cost-anomaly-categories` (tbl 🟠) — def L3307

**Verbatim ref sentence (L3297):**
> "@Tbl-cost-anomaly-categories groups cost anomaly root-cause categories into five classes, each with distinct investigation paths:"

**What is missing:** The cite sentence is a pure list-announcer with a trailing colon, delivering no Interpret move. The prose never states which category is most common, which is most dangerous to miss, or what the table's decision value is. The key insight from the table — that traffic increases and efficiency regressions have opposite indicator patterns (QPS up vs. QPS unchanged but latency up), which is what makes them diagnosable — is absent from body prose. Removability test: the surrounding prose (the worked example at L3291 and the L3311 cost attribution paragraph) reads fine without the table.

**Where takeaway currently lives:** Cells only (the Indicators column).

**Rule-compliant diff rewrite (replace L3297):**

```diff
- @Tbl-cost-anomaly-categories groups cost anomaly root-cause categories into five classes, each with distinct investigation paths:
+ The five categories in @Tbl-cost-anomaly-categories are distinguished by the combination of metrics that move together. Traffic increases and efficiency regressions account for most cost spikes but require opposite responses: a traffic increase is healthy and calls for capacity planning, while an efficiency regression at unchanged QPS signals a model or infrastructure change that consumed more compute per request. The table's most useful column is the indicator pattern, not the category name — a practitioner who sees cost up with latency up and QPS flat can immediately skip the traffic and pricing rows and begin the deployment-review investigation path for efficiency regression.
```

---

#### ⚠️ `tbl-ops-scale-efficiency-metric-actions` (tbl 🟠) — def L3597

**Verbatim ref sentence (L3588):**
> "Efficiency metrics should be reviewed alongside model quality so the platform can identify waste that accuracy alone hides. @Tbl-ops-scale-efficiency-metric-actions connects each metric to the operational question it answers."

**What is missing:** The cite sentence (second sentence) is a connector label. The first sentence establishes context but the Interpret move is absent. The table lists four metrics but the prose never names which metric reveals which type of waste first, which metric the platform should establish before the others, or what the table's load-bearing contrast is (the "cost per accuracy point" metric catches a specific failure mode that GPU utilization cannot). The payoff (L3599) is a one-sentence summary that does not extract the table's takeaway.

**Where takeaway currently lives:** Cells only.

**Rule-compliant diff rewrite (replace L3588-L3589 and L3599):**

```diff
- Efficiency metrics should be reviewed alongside model quality so the platform can identify waste that accuracy alone hides. @Tbl-ops-scale-efficiency-metric-actions connects each metric to the operational question it answers.
+ Efficiency metrics should be reviewed alongside model quality so the platform can identify waste that accuracy alone hides. @Tbl-ops-scale-efficiency-metric-actions reveals the distinct failure mode each metric catches: cost per accuracy point surfaces models where quality improvements are bought with disproportionate compute, a failure invisible to GPU utilization monitoring because utilization may be high even when the work being done produces little accuracy return. Spot utilization rate catches a different failure — fault-tolerant jobs running on expensive on-demand capacity — that quality metrics and utilization dashboards both miss entirely. Reviewing all four metrics together is necessary because each covers a blind spot of the others.
```

And replace the thin payoff at L3599:

```diff
- Regular review of these metrics identifies systemic inefficiencies and guides platform improvements.
+ The prioritization order follows the organization's cost structure: when inference dominates, cost per inference and GPU utilization are reviewed weekly; when training dominates, spot utilization rate and experiments per production model reveal the largest optimization opportunities.
```

---

#### ⚠️ `tbl-feature-freshness-engagement` (tbl 🟠) — def L4088

**Verbatim ref sentence (L4079):**
> "A recommendation system uses user interaction features with different freshness levels. Testing on historical data produces the engagement lift in @tbl-feature-freshness-engagement:"

**What is missing:** The cite sentence is a bare pointer with a trailing colon, delivering the table's content label but no Interpret move. The payoff (L4090) states "the engagement difference between hourly and real-time features is 2.1 percentage points" — but this is a cell readout rather than an interpretation. The table's load-bearing insight — that the marginal gain diminishes as freshness improves, so the real-time-to-near-real-time gap (0.5 points) is much smaller than the daily-to-hourly gap (2.1 points) — is never articulated in body prose. This diminishing-returns pattern is what drives the infrastructure investment decision but it is not stated.

**Where takeaway currently lives:** Readable from cell arithmetic only.

**Rule-compliant diff rewrite (replace L4079 and L4090):**

```diff
- A recommendation system uses user interaction features with different freshness levels. Testing on historical data produces the engagement lift in @tbl-feature-freshness-engagement:
+ A recommendation system uses user interaction features with different freshness levels. @Tbl-feature-freshness-engagement shows the diminishing-returns pattern that governs the streaming investment decision: the largest marginal gain — 2.1 percentage points — comes from moving from daily to hourly features, while the subsequent step from hourly to near real-time adds 1.6 points, and the final step to real-time adds only 0.5 points. The step-size pattern matters more than the total lift because it determines which infrastructure tier earns its cost: a team that cannot afford full streaming should prioritize hourly batch over near-real-time streaming because hourly delivers more than four times the return per unit of freshness improvement.
```

And replace L4090:

```diff
- The engagement difference between hourly and real-time features is 2.1 percentage points. If this translates to \$10 million in annual engagement value, investing in real-time feature infrastructure may be justified if costs are below this value.
+ The \$10 million engagement value cited above corresponds to the daily-to-real-time total lift of 4.2 percentage points. The 2.1-point hourly-to-real-time gap — the marginal value of streaming over batch — is therefore worth roughly \$5 million annually on this platform. Whether a streaming feature pipeline costs less than \$5 million per year is the break-even question the platform team must answer before committing to the investment.
```

---

## ⚠️ Dangling reference

`@eq-distributed-training-scaling-efficiency` at L1604 has no matching definition in this chapter. The reference appears inside the body of a numbered list item. Either the equation should be defined in this chapter or the cross-reference should point to the chapter where it is defined.

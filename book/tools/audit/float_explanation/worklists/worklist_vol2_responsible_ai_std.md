# Float Exposition Worklist — `responsible_ai.qmd` (vol2)

Graded against the Float Exposition Standard (eq 🔴 / alg 🔴 / tbl 🟠 / fig 🟠 / lst 🟡).
Caption, fig-alt, in-figure labels, code comments, and callout interiors do not count toward prose.

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| fig  | 🟠    | 14     | 9  | 5  | 0  |
| tbl  | 🟠    | 7      | 5  | 2  | 0  |
| lst  | 🟡    | 2      | 2  | 0  | 0  |
| **Total** |  | **23** | **16** | **7** | **0** |

---

## Findings (⚠️ only — no 🛑)

---

### `fig-privacy-risk-flow` (fig 🟠) — def L728

**Ref sentence (L726):**
> "@Fig-privacy-risk-flow outlines key privacy checkpoints in the early stages of a data pipeline, highlighting where safeguards such as differential privacy, federated learning, and secure aggregation reduce attacker visibility into raw personal data. Actual implementations often involve more nuanced tradeoffs and context-sensitive decisions, including separate consent and governance controls, but this diagram provides a scaffold for identifying where privacy risks arise and how they can be mitigated through responsible design choices."

**Missing move:** The Interpret move is absent. The cite describes what the figure shows (sequential checkpoints, decreasing attacker visibility) but never delivers the takeaway: what the reader should conclude about the ordering of safeguards and why that pipeline structure matters. The payoff paragraph at L732 discusses weak governance consequences but never interprets the figure's specific mechanism (that each stage reduces attacker visibility additively, and that the order matters). The insight lives in the caption ("progressively reduce") rather than in body prose.

**Rule-compliant diff rewrite** (replace the existing L726 paragraph with the two sentences below; insert after the word "choices"):

```diff
- @Fig-privacy-risk-flow outlines key privacy checkpoints in the early stages of a data pipeline,
- highlighting where safeguards such as differential privacy, federated learning, and secure
- aggregation reduce attacker visibility into raw personal data. Actual implementations often involve
- more nuanced tradeoffs and context-sensitive decisions, including separate consent and governance
- controls, but this diagram provides a scaffold for identifying where privacy risks arise and how
- they can be mitigated through responsible design choices.
+ Three safeguards in @Fig-privacy-risk-flow reduce attacker visibility at successive stages of the
+ pipeline: differential privacy at the raw data boundary, federated learning across the aggregation
+ network, and secure aggregation at the model update layer. The ordering is load-bearing. Applying
+ only one safeguard leaves the other two stages exposed; the residual attack surface at each
+ unprotected stage is larger than the sum of individual omissions, because an attacker who
+ compromises an early stage bypasses all downstream protections. Actual implementations often
+ require separate consent and governance controls not shown here, but the pipeline structure
+ establishes the minimum sequence of checkpoints responsible design must cover.
```

---

### `fig-responsible-ai-architecture` (fig 🟠) — def L1061

**Ref sentence (L1059):**
> "Implementing responsible AI principles in production systems requires architectural patterns that integrate fairness monitoring, explainability, and privacy controls directly into the model serving infrastructure. @Fig-responsible-ai-architecture demonstrates how these responsible AI components integrate with existing ML systems infrastructure, showing the data flow from user requests through anonymization, model inference, fairness monitoring, and explanation generation."

**Missing move:** The cite names the figure and describes the data flow path ("demonstrates how... showing the data flow") but delivers no takeaway. The reader learns what the figure shows but not what the figure's architecture demonstrates about responsible AI system design. The payoff is at L1351, 290 lines later, where the three governance paths and their latency costs are finally explained. The immediate lead-out is a float-announcer.

**Rule-compliant diff rewrite** (replace the L1059 paragraph):

```diff
- Implementing responsible AI principles in production systems requires architectural patterns that
- integrate fairness monitoring, explainability, and privacy controls directly into the model
- serving infrastructure. @Fig-responsible-ai-architecture demonstrates how these responsible AI
- components integrate with existing ML systems infrastructure, showing the data flow from user
- requests through anonymization, model inference, fairness monitoring, and explanation generation.
+ Implementing responsible AI principles in production systems requires that fairness monitoring,
+ explainability, and privacy controls sit on the critical serving path rather than beside it.
+ @Fig-responsible-ai-architecture shows the consequence of that placement: every inference request
+ passes through data anonymization before the model sees it, and every prediction triggers both a
+ fairness monitor and an explanation engine before reaching the caller. The dashed feedback loop
+ from the fairness metrics database back to the model makes the key design choice visible: the
+ architecture does not treat fairness as a postdeployment audit but as an operational signal that
+ feeds directly into retraining. This embedding has a provisioning cost that must be budgeted into
+ the serving SLO from the start.
```

---

### `fig-machine-unlearning` (fig 🟠) — def L1612

**Ref sentence (L1610):**
> "Traditional approaches to data deletion assume that the full training dataset remains accessible and that models can be retrained from scratch after removing the targeted records. @Fig-machine-unlearning contrasts traditional model retraining with machine unlearning approaches: while retraining involves reconstructing the model from scratch using a modified dataset, unlearning aims to remove a specific datapoint's influence without repeating the entire learning process."

**Missing move:** The cite describes the contrast between the three approaches but omits the interpret move: which approach is preferable under which constraints, and why the three-way contrast (full retrain vs. gradient ascent vs. SISA) matters for systems design. The caption names the cost model; the payoff at L1616 delivers the "so what" (full retraining is impractical under real constraints), but the cite paragraph itself only paraphrases the figure labels without naming any conclusion.

**Rule-compliant diff rewrite** (replace the second sentence only):

```diff
- @Fig-machine-unlearning contrasts traditional model retraining with machine unlearning approaches:
- while retraining involves reconstructing the model from scratch using a modified dataset, unlearning
- aims to remove a specific datapoint's influence without repeating the entire learning process.
+ @Fig-machine-unlearning maps the cost-accuracy trade-off across three strategies: full retraining
+ gives exact removal at the highest cost, gradient ascent approximates removal in a single backward
+ pass at the risk of underfitting the forget set, and SISA training achieves exact per-shard removal
+ by paying only the cost of the affected shard. The figure's central lesson is that exact unlearning
+ is not a binary choice between full retraining and approximation. Shard isolation converts a
+ full-dataset problem into a bounded one, and choosing the right shard granularity at design time
+ determines whether a deletion request costs hours or minutes.
```

---

### `fig-monitoring-pipeline` (fig 🟠) — def L1883

**Ref sentence (L1881):**
> "Systems must log inputs, outputs, and contextual metadata in a structured and secure manner, feeding a continuous observability pipeline (@fig-monitoring-pipeline)."

**Missing move:** The cite is a bare parenthetical pointer. There is no lead-out: no naming of what the pipeline's architecture demonstrates, no statement of what the reader should conclude from its structure (subgroup metrics feeding a threshold check that triggers either alert or retraining). The payoff at L1887 discusses telemetry and drift detection but does not interpret the figure. The figure earns no prose of its own.

**Rule-compliant diff rewrite** (replace the L1881 sentence):

```diff
- Systems must log inputs, outputs, and contextual metadata in a structured and secure manner,
- feeding a continuous observability pipeline (@fig-monitoring-pipeline).
+ Implementing effective monitoring depends on robust infrastructure. Systems must log inputs,
+ outputs, and contextual metadata in a structured and secure manner. The fairness monitoring
+ pipeline in @Fig-monitoring-pipeline makes the feedback structure explicit: model predictions feed
+ subgroup metric computation, a threshold check determines whether those metrics indicate
+ regression, and the alert branch triggers retraining rather than simply notifying an operator.
+ That retraining loop is the critical design choice. A pipeline that only alerts without a
+ retraining path turns fairness monitoring into a reporting tool rather than a control mechanism.
```

---

### `fig-rlhf-pipeline` (fig 🟠) — def L1921

**Ref sentence (L1919):**
> "This requires a transition from *static dataset curation* to *dynamic behavioral shaping*, typically through a multi-stage alignment process (@fig-rlhf-pipeline)."

**Missing move:** The cite is a bare parenthetical pointer. No prose names what the six-stage structure reveals about the alignment problem or what the reader should conclude from seeing base model, SFT, human preferences, reward training, PPO, and aligned model laid out in sequence. The payoff at L1925 is rich and delivers cost numbers, but the figure itself receives no interpret move at the cite site.

**Rule-compliant diff rewrite** (replace the closing clause of the L1919 sentence):

```diff
- This requires a transition from *static dataset curation* to *dynamic behavioral shaping*,
- typically through a multi-stage alignment process (@fig-rlhf-pipeline).
+ This requires a transition from static dataset curation to dynamic behavioral shaping. The
+ six-stage process in @Fig-rlhf-pipeline shows why alignment is more expensive than fine-tuning:
+ human preference collection and reward model training interpose two full training passes between
+ supervised fine-tuning and the final aligned model, each stage introducing its own cost and
+ failure mode. The pipeline structure also makes the representativeness problem concrete. Because
+ every stage amplifies the biases present in the human preference labels, the quality of the
+ aligned model is bounded by the demographic coverage of the rater pool at stage three, not by
+ compute or model capacity at stage five.
```

---

### `tbl-practitioner-decision-framework` (tbl 🟠) — def L2366

**Ref sentence (L2337):**
> "@Tbl-practitioner-decision-framework provides a practitioner decision framework that guides context-sensitive choices, mapping deployment contexts to primary principles, implementation priorities, and acceptable trade-offs across high stakes individual decisions, safety-critical systems, privacy-sensitive applications, large-scale consumer systems, resource-constrained deployments, and research environments."

**Missing move:** The cite describes what the table maps (deployment contexts to principles) but delivers no takeaway: which row is the hardest trade-off, what the table reveals about when principles conflict, or what a reader should do differently after consulting it. The payoff at L2368 has rich interpretation but the cite paragraph itself is a "Table X provides guidance on Y" pattern.

**Rule-compliant diff rewrite** (append a second sentence to L2337 after the existing sentence):

```diff
  @Tbl-practitioner-decision-framework provides a practitioner decision framework that guides
  context-sensitive choices, mapping deployment contexts to primary principles, implementation
  priorities, and acceptable trade-offs across high stakes individual decisions, safety-critical
  systems, privacy-sensitive applications, large-scale consumer systems, resource-constrained
  deployments, and research environments.
+ The table's most useful column is acceptable trade-offs, because it names the principle a
+ responsible team may consciously defer given the deployment's binding constraints. Safety-critical
+ systems accept reduced explainability to hold latency; resource-constrained deployments accept
+ reduced fairness monitoring to fit compute budgets. Naming those deferrals in advance converts an
+ implicit compromise into an auditable design decision.
```

---

### `tbl-responsible-ai-fairness-summary` (tbl 🟠) — def L2623

**Ref sentence (L2559):**
> "@Tbl-responsible-ai-fairness-summary illustrates why fairness requires explicit trade-offs. Consider a loan approval system evaluated across two demographic groups:"

**Missing move:** The cite invokes the table as a "consider" setup and immediately defers to the reader to draw conclusions. No interpret move appears before or in the payoff (L2625 says only "The table makes the chapter's central constraint concrete"). The takeaway that one metric can appear satisfied while others expose substantial disparities is never stated in the cite move itself.

**Rule-compliant diff rewrite** (replace the L2559 two-sentence cite):

```diff
- @Tbl-responsible-ai-fairness-summary illustrates why fairness requires explicit trade-offs.
- Consider a loan approval system evaluated across two demographic groups:
+ @Tbl-responsible-ai-fairness-summary shows the incompatibility between fairness metrics on a
+ concrete loan approval system: equalized false positive rates are nearly satisfied while
+ demographic parity and equal opportunity both show substantial gaps. No single threshold
+ adjustment closes all three gaps simultaneously, because each metric equalizes a different
+ quantity. This is the chapter's central constraint made numerical: a system can be technically
+ compliant on one fairness metric while systematically violating two others, and the choice of
+ which metric to report is itself a policy decision. The two demographic groups in the table are:
```

---

## ⚠️ Dangling ref (no matching def)

- L70 `@fig-fleet-stack`: "In the fleet stack shown in @fig-fleet-stack, Responsible AI is the Governance Layer." This figure is defined in another chapter and carried forward here as a cross-chapter reference. Flag for verification that the target def exists in the referenced chapter.

# Float Exposition Worklist — vol1 / ml_ops.qmd

**Standard:** FLOAT_EXPOSITION_STANDARD.md
**Chapter:** vol1/ml_ops/ml_ops.qmd
**Date:** 2026-06-09
**Floats scanned:** 65 (16 eq · 12 fig · 6 lst · 31 tbl)

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| eq   | 🔴    |     16 | 14 |  2 |  0 |
| fig  | 🟠    |     12 |  9 |  3 |  0 |
| lst  | 🟡    |      6 |  6 |  0 |  0 |
| tbl  | 🟠    |     31 | 21 | 10 |  0 |
| **Total** | | **65** | **50** | **15** | **0** |

---

## Findings (⚠️ only — no 🛑)

---

### `eq-retrain-decision` (eq 🔴) — def L248

**Verbatim cite sentence (L247–248):**
> Cost-aware automation should balance computational costs against accuracy improvements. @Eq-retrain-decision models this trade-off:

**Missing move:** Lead-out (Interpret). The equation symbols $\Delta\text{Accuracy}$, "Value per Point," "Training Cost," and "Deployment Risk" are not defined or named anywhere in body prose. No consequence or regime is stated. The payoff paragraph defers interpretation to a later section rather than delivering the implication here.

**Where takeaway currently lives:** Partially in the payoff, which says only "the framework for making principled trade-off decisions remains constant" without unpacking the equation's decision structure.

**Rule-compliant diff rewrite** (add after the equation, replacing the opening of the payoff paragraph):

```diff
- This principle guides the design of retraining triggers, validation thresholds,
- and deployment strategies examined throughout this chapter.
+ Each term encodes a measurable quantity: $\Delta\text{Accuracy}$ is the
+ expected accuracy gain from a fresh model, "Value per Point" converts that
+ gain into revenue impact, and "Training Cost" plus "Deployment Risk" are
+ the combined compute and regression costs of the update. The inequality fires
+ a retrain only when the revenue gain clears the combined cost, making
+ retraining a threshold decision rather than a calendar event.
+ This principle guides the design of retraining triggers, validation thresholds,
+ and deployment strategies examined throughout this chapter.
```

---

### `eq-monitoring-cost` (eq 🔴) — def L2578

**Verbatim cite sentence (L2577–2578):**
> Monitoring costs break down into four categories, as @eq-monitoring-cost decomposes:

**Missing move:** Lead-out (Interpret). The cite sentence introduces the decomposition without stating the consequence. The payoff sentence points immediately to the next table rather than explaining what the decomposition implies.

**Where takeaway currently lives:** Nowhere in prose adjacent to the equation. The dominant cost driver (metric cardinality) is explained in a later cost-optimization section, not as a payoff here.

**Rule-compliant diff rewrite** (insert after the equation before the existing table citation):

```diff
- @Tbl-monitoring-cost-components provides typical unit costs for each component:
+ The decomposition isolates the cost driver most teams underestimate:
+ $C_{\text{ingest}}$ scales with metric cardinality (number of unique
+ label combinations), not with request volume, so high-cardinality labels
+ such as user-level IDs can dwarf compute costs. @Tbl-monitoring-cost-components
+ provides typical unit costs for each component:
```

---

### `fig-mlops-diagram` (fig 🟠) — def L101

**Verbatim cite sentence (L99):**
> Trace the infinity-loop structure in @fig-mlops-diagram to see how these phases feed back into one another continuously; the loop gives the discipline its operating shape.

**Missing move:** Lead-out (Interpret). The prose directs the reader to trace the figure but stops there. The payoff paragraph (L150) jumps to a retail failure scenario without stating what the loop structure demonstrates.

**Where takeaway currently lives:** The cite sentence names the operating shape but never states what that shape teaches: that no phase is terminal (deployment feeds back into design), so MLOps is not a pipeline but a closed feedback system.

**Rule-compliant diff rewrite** (replace the cite sentence):

```diff
- Trace the infinity-loop structure in @fig-mlops-diagram to see how these phases
- feed back into one another continuously; the loop gives the discipline its
- operating shape.
+ The infinity-loop structure in @fig-mlops-diagram makes this closure concrete:
+ deployment does not terminate the cycle but feeds back into design through
+ monitoring signals, making continuous improvement a structural property of
+ the discipline rather than an optional practice.
```

---

### `fig-ops-layers` (fig 🟠) — def L682

**Verbatim cite sentence (L680):**
> Examine the layered architecture in @fig-ops-layers, which organizes these components across ML models, frameworks, orchestration, infrastructure, and hardware.

**Missing move:** Lead-out (Interpret). The prose names the layers but the payoff paragraph pivots immediately to data handling requirements without stating what the layered view teaches.

**Where takeaway currently lives:** The caption notes that MLOps spans orchestration and infrastructure tasks, but that insight is in the caption, not body prose.

**Rule-compliant diff rewrite** (extend the existing cite sentence):

```diff
- Examine the layered architecture in @fig-ops-layers, which organizes these
- components across ML models, frameworks, orchestration, infrastructure, and
- hardware. Understanding how these layers interact enables practitioners to
- design systems that systematically address the technical debt patterns
- identified earlier while maintaining operational sustainability.
+ The layered architecture in @fig-ops-layers organizes these components across
+ ML models, frameworks, orchestration, infrastructure, and hardware. The key
+ structural insight is that MLOps work sits below the model tier: the discipline
+ spans orchestration (data management through model serving) and infrastructure
+ (job scheduling through monitoring), so improving MLOps means instrumenting
+ the layers the model depends on, not the model itself.
```

---

### `fig-business-cost-curve` (fig 🟠) — def L3006

**Verbatim cite sentence (L3004):**
> This connection is not linear. @Fig-business-cost-curve exposes this nonlinearity: the optimal operating point for a model is rarely the point of highest accuracy. It is the point where the combined cost of False Positives (for example, blocking a legitimate user) and False Negatives (for example, missing fraud) is minimized.

**Missing move:** Lead-out (Interpret). The prose states the general principle (minimize combined cost) but the specific takeaway that the cost asymmetry ($500 FN vs $100 FP, a 5× ratio) pulls the threshold left to approximately 0.34 lives only in the caption. Body prose never states this consequence.

**Where takeaway currently lives:** Caption only. The caption explains the $5× asymmetry and the resulting threshold shift.

**Rule-compliant diff rewrite** (add after the existing cite paragraph, before the figure):

```diff
+ In fraud detection, where a missed fraud costs five times a blocked
+ legitimate transaction, the U-shaped total cost curve reaches its minimum
+ well to the left of 0.5: the model must become more aggressive at flagging
+ suspicious transactions, accepting more false positives to avoid the
+ disproportionately expensive misses. MLOps operationalizes this by surfacing
+ the cost parameters and updating the threshold as fraud patterns and
+ business costs evolve.
```

---

### `tbl-degradation-types` (tbl 🟠) — def L243

**Verbatim cite sentence (L234):**
> ML systems must make silent failures visible through continuous measurement. Model performance degrades along a continuum rather than failing discretely, requiring the detection mechanisms and response strategies summarized in @tbl-degradation-types:

**Missing move:** Lead-out (Interpret). After the table the prose jumps directly to `eq-retrain-decision` with no sentence stating what the table teaches.

**Where takeaway currently lives:** Nowhere in adjacent body prose. The table cells encode the key finding (each degradation type requires a distinct detection mechanism), but no prose restates it.

**Rule-compliant diff rewrite** (insert after the table, before the cost-aware automation heading):

```diff
+ The four rows share no common detection mechanism: sudden drops need
+ threshold alerts while gradual drift requires trend analysis, and
+ no single monitoring approach covers all four failure modes. A production
+ system that monitors only aggregate accuracy will reliably catch sudden
+ drops while missing gradual drift and subgroup degradation until user
+ complaints surface the problem.
+
#### Cost-aware automation ...
```

---

### `tbl-mlops-principles-summary` (tbl 🟠) — def L260

**Verbatim cite sentence (L250):**
> @Tbl-mlops-principles-summary provides the quick reference:

**Missing move:** Lead-out (Interpret). The cite sentence is a bare pointer. The payoff moves directly to monitoring archetypes without stating what the table's arrangement implies.

**Where takeaway currently lives:** Nowhere. The table is positioned as a lookup tool without a prose statement of which principle is hardest to achieve or which drives the most failures.

**Rule-compliant diff rewrite** (add after the table, before the next section):

```diff
+ Of the five principles, observable degradation is the one most
+ organizations treat as optional until the first silent failure incident.
+ The others (reproducibility, separation of concerns, consistency, and
+ cost-aware automation) have tooling that enforces them as side effects
+ of normal development; observable degradation requires deliberate,
+ ongoing investment in monitoring infrastructure that produces no visible
+ output until something goes wrong.
```

---

### `tbl-training-serving-skew` (tbl 🟠) — def L793

**Verbatim cite sentence (L784):**
> @Tbl-training-serving-skew summarizes common causes and their detection methods:

**Missing move:** Lead-out (Interpret). The table is a bare-pointer citation. The payoff is a concrete example paragraph but it illustrates the skew concept rather than stating the table's key finding.

**Where takeaway currently lives:** Implicitly in the four rows, but no prose names which skew type is hardest to detect or most common in practice.

**Rule-compliant diff rewrite** (insert after the table, before the example paragraph):

```diff
+ Library version drift is the most insidious category because it passes
+ every schema and distribution check while silently altering numerical
+ outputs: a Pandas version update that changes floating-point rounding
+ behavior in a normalization step will not register as a schema violation
+ but will degrade predictions on any feature sensitive to that rounding.
```

---

### `tbl-rollback-patterns` (tbl 🟠) — def L1614

**Verbatim cite sentence (L1606):**
> @Tbl-rollback-patterns summarizes implementation patterns for each rollback type:

**Missing move:** Lead-out (Interpret). The cite sentence is a bare pointer. The preceding paragraph already describes the three tiers clearly, but after the table the prose moves to rollback testing without a sentence drawing a conclusion from the table itself.

**Where takeaway currently lives:** The preceding paragraph explains the content; the table adds the implementation column. No prose states what the implementation column teaches.

**Rule-compliant diff rewrite** (insert after the table, before the "Rollback testing" heading):

```diff
+ The state-handling column reveals the key distinction: immediate and
+ rapid rollbacks are stateless operations that succeed by keeping a
+ warm copy of the previous model, while delayed rollback is a stateful
+ migration requiring data replay and cache invalidation. Teams that
+ test only the first two tiers typically discover the state-migration
+ complexity of delayed rollback for the first time during a live incident.
```

---

### `tbl-gpu-memory-hierarchy` (tbl 🟠) — def L1972

**Verbatim cite sentence (L1962):**
> Model serving performance depends critically on memory hierarchy utilization. Data must flow through multiple memory levels with vastly different bandwidths (see @sec-appdx-machine-foundations-memory-hierarchy-2278 for a comprehensive latency hierarchy across the full storage spectrum), as @tbl-gpu-memory-hierarchy quantifies.

**Missing move:** Lead-out (Interpret). "Quantifies" is a bare pointer verb. The payoff at L2038 discusses KV-cache footprint, not the bandwidth ratios in the table. No prose states the central finding: NVMe swap is approximately 300× slower than L2 cache, making model weights that spill to disk catastrophic for inference latency.

**Where takeaway currently lives:** The caption notes the general trade-off; the magnitude of the bandwidth gap is not stated in body prose.

**Rule-compliant diff rewrite** (extend the existing cite sentence):

```diff
- Data must flow through multiple memory levels with vastly different
- bandwidths (see @sec-appdx-machine-foundations-memory-hierarchy-2278
- for a comprehensive latency hierarchy across the full storage spectrum),
- as @tbl-gpu-memory-hierarchy quantifies.
+ Data must flow through multiple memory levels with vastly different
+ bandwidths, as @tbl-gpu-memory-hierarchy quantifies. The bandwidth gap
+ from HBM to NVMe is roughly 300×: a model that fits in GPU memory serves
+ at terabyte-per-second bandwidth, while a model that spills to NVMe swap
+ serves at seven gigabytes per second, degrading inference latency by the
+ same ratio. Keeping model weights in HBM is therefore a hard constraint,
+ not a preference.
```

---

### `tbl-monitoring-cost-components` (tbl 🟠) — def L2589

**Verbatim cite sentence (L2580):**
> @Tbl-monitoring-cost-components provides typical unit costs for each component:

**Missing move:** Lead-out (Interpret). Bare pointer. The payoff says "Translating these unit costs into a concrete budget estimate clarifies the real expense of monitoring even a single production model" — forward-pointing rather than interpretive. The dominant cost driver (metric cardinality) is identified only in the later cost-optimization paragraph, not as a payoff for this table.

**Where takeaway currently lives:** Downstream cost-optimization section, not adjacent to the table.

**Rule-compliant diff rewrite** (replace the payoff sentence):

```diff
- Translating these unit costs into a concrete budget estimate clarifies the
- real expense of monitoring even a single production model.
+ The ingestion row dominates total cost in high-cardinality deployments:
+ at $0.10–0.50 per million data points, a system with 10,000 unique metric
+ series sampled every 15 seconds generates roughly 57 billion data points
+ per month, pushing ingestion costs above storage and compute combined.
+ Translating these unit costs into a concrete budget estimate clarifies
+ the real expense of monitoring even a single production model.
```

---

### `tbl-slice-analysis-example` (tbl 🟠) — def L2875

**Verbatim cite sentence (L2818):**
> Slice analysis exposes that masking, and @tbl-slice-analysis-example illustrates how overall accuracy can hide severe degradation in specific segments:

**Missing move:** Lead-out (Interpret). The prose states the general principle but the specific finding — tablet users at 62 percent accuracy on 5 percent of traffic, masked by 91 percent overall — lives only in the caption. No body prose names these numbers or states the implication (a cohort receiving sub-par predictions is large enough to matter commercially yet small enough to disappear in aggregate metrics).

**Where takeaway currently lives:** Caption only.

**Rule-compliant diff rewrite** (insert after the table, before the SHAP listing citation):

```diff
+ The tablet cohort in this example carries the worst accuracy (62 percent)
+ on only 5 percent of traffic, which is enough to suppress overall accuracy
+ from 91 percent to a weighted average that still reads as acceptable.
+ A recommendation system degrading for one in twenty users on a specific
+ device class will not trip a system-wide accuracy alert, which is why
+ slice analysis is a required complement to aggregate monitoring rather
+ than an optional debugging step.
```

---

### `tbl-ml-roles-matrix` (tbl 🟠) — def L2990

**Verbatim cite sentence (L2980):**
> @Tbl-ml-roles-matrix maps these roles to their primary responsibilities:

**Missing move:** Lead-out (Interpret). Bare pointer. The payoff addresses handoff risks in general without drawing a conclusion from the table's structure.

**Where takeaway currently lives:** The payoff paragraph describes handoff risks but these are not connected back to what the table's "Collaboration Points" column encodes.

**Rule-compliant diff rewrite** (insert after the table, before the existing handoff paragraph):

```diff
+ The Collaboration Points column identifies where organizational coupling
+ creates reliability risk: the Data Scientist to ML Engineer handoff (notebook
+ to production) and the ML Engineer to SRE handoff (deployment to on-call)
+ are the two transitions most likely to silently break reproducibility because
+ each specialist optimizes for different success criteria. Data Scientists
+ optimize for model accuracy; ML Engineers optimize for pipeline reliability;
+ SREs optimize for uptime. None of these goals naturally includes documenting
+ the other's failure modes.
```

---

### `tbl-technical-debt-summary` (tbl 🟠) — def L3156

**Verbatim cite sentence (L3143):**
> @Tbl-technical-debt-summary consolidates the debt patterns discussed throughout this chapter, providing the reference that the assessment rubric below builds on.

**Missing move:** Lead-out (Interpret). Reference-summary pointer with no takeaway. The prose moves immediately to the ML Test Score rubric without stating which debt pattern is most prevalent, most costly, or hardest to remediate.

**Where takeaway currently lives:** Nowhere in adjacent prose.

**Rule-compliant diff rewrite** (insert after the table, replacing the opening of the next paragraph):

```diff
- With those debt patterns in one place, awareness alone is insufficient;
+ Of the eight patterns, feedback loops and boundary erosion are the
+ hardest to detect before they compound: feedback loops are invisible
+ until the model's own predictions begin corrupting future training data,
+ and boundary erosion produces no error messages because the implicit
+ coupling runs through data distributions, not code interfaces.
+ With those debt patterns in one place, awareness alone is insufficient;
```

---

### `tbl-mlops-single-model-investment` (tbl 🟠) — def L3347

**Verbatim cite sentence (L3337):**
> @Tbl-mlops-single-model-investment summarizes the main cost categories:

**Missing move:** Lead-out (Interpret). Bare pointer. The payoff pivots to the ROI equation without stating what the cost table implies (the total range, which category dominates, or what "one-time vs. recurring" implies for budget planning).

**Where takeaway currently lives:** Nowhere adjacent. The ROI equation that follows uses a $30K figure that appears in the code, not derived from the table's ranges.

**Rule-compliant diff rewrite** (insert after the table, before the "Single-model ROI calculation" heading):

```diff
+ The table separates one-time setup cost from recurring annual spend:
+ CI/CD pipeline setup ($10–30K) is a fixed investment that amortizes
+ across the model's lifetime, while monitoring, feature store, and model
+ registry costs recur annually at $7–35K combined. For a model with a
+ multi-year production lifetime, recurring spend dominates total cost of
+ ownership, making monitoring and feature store selection more consequential
+ than the initial pipeline build.
```

---

*End of findings.*

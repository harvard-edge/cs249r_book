# Float Exposition Audit — `responsible_engr.qmd` (vol1)

> Standard: FLOAT_EXPOSITION_STANDARD.md
> Method: scan_floats.py bundle + ±40-line body-prose read
> Date: 2026-06-09
> Scope: running body prose only (captions, fig-alt, code comments excluded)

---

## Summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| eq   | 🔴    | 2      | 1  | 1  | 0  |
| fig  | 🟠    | 6      | 5  | 1  | 0  |
| lst  | 🟡    | 1      | 1  | 0  | 0  |
| tbl  | 🟠    | 16     | 10 | 6  | 0  |
| **Total** |  | **25** | **17** | **8** | **0** |

---

## Findings (⚠️ only — no 🛑)

---

### Finding 1 — `eq-alignment-gap` (eq 🔴) — def L168

**Verbatim citation sentence (L167):**
> "The Quantification (conceptual, assuming normalized metrics on a common scale) is captured by @eq-alignment-gap:"

**Problem — float-announcer colon + missing symbol gloss.**
The citation sentence is a float-announcer colon construction: it names the equation by its role ("the quantification") and drops a colon that leads directly into the display. By the standard, the prose must state what the equation expresses in words before the float appears. The symbols $\mathbb{E}[\text{Proxy}]$ and $\mathbb{E}[\text{True}]$ are never glossed in prose; the reader must infer their meaning from the surrounding bullets. The lead-out at L170 ("If the model increases Clicks by 20 percent but decreases Satisfaction by 5 percent, the alignment gap has widened") is present but arrives after the display, and it is a consequence sentence, not a symbol definition.

**Where the takeaway currently lives:** the lead-out consequence is in body prose (L170); symbol meanings are implied by the bullets above (L162–165) but never explicitly stated as a "where" clause.

**Missing move:** Interpret — a prose sentence naming the equation's meaning and defining the symbols before the citation.

**Rule-compliant rewrite:**

Replace the citation sentence at L167:

> The alignment gap measures exactly how much the proxy objective drifts from the true goal. Formally, the gap equals the expected value of the proxy metric minus the expected value of the true metric. A positive value means the proxy overstates actual satisfaction, and a wider gap signals greater misalignment. In the notebook example, the proxy expectation rose by 20 percent while the true expectation fell by 5 percent, so the gap grew despite an apparent model improvement (@eq-alignment-gap).

---

### Finding 2 — `fig-data-card` (fig 🟠) — def L2605

**Verbatim citation sentence (L2603):**
> "Examine the data card template in @fig-data-card to see how this structured format turns abstract compliance obligations into concrete, machine-checkable fields."

**Problem — reader told to examine without a stated takeaway.**
The citation tells readers what to do ("examine") but does not state what the figure demonstrates or which structural feature makes the format machine-checkable. The lead-out — "Training pipelines check that input datasets have valid data cards before processing, and serving systems enforce that only models trained on compliant data can deploy to production" — follows the float (it comes two sentences before the figure in the same paragraph but is really the consequence of the compliance context, not an interpretation of the card's structure). After the figure, the prose pivots to data lineage with no return to the card's design. The takeaway — which fields matter, what the card's layout reveals about compliance enforcement — lives only in the caption.

**Where the takeaway currently lives:** caption ("data cards standardize critical dataset information, supporting transparency and accountability") and the two sentences before the figure describe pipeline enforcement, but no sentence says what to notice about the card's structure.

**Missing move:** Lead-out / Interpret — a sentence stating what the card's format demonstrates that prose alone cannot.

**Rule-compliant rewrite:**

Replace the citation sentence at L2603:

> A data card's value lies in its structure. The template in @fig-data-card organizes compliance obligations into named, bounded fields — dataset description, authorship, intended uses, potential risks, and collection methods — so that pipelines can validate completeness programmatically rather than relying on prose review. The key design choice is that each field maps to a specific regulatory obligation: the intended-use field corresponds to GDPR's purpose-limitation requirement, and the potential-risks field documents the harm assessment required by the EU AI Act for high-risk systems.

---

### Finding 3 — `tbl-model-card-example` (tbl 🟠) — def L702

**Verbatim citation sentence (L687):**
> "A concrete MobileNetV2 model card makes these abstract categories operational: @tbl-model-card-example shows how each section addresses specific deployment concerns for edge deployment."

**Problem — float-announcer colon + no stated conclusion from the cells.**
The citation uses a float-announcer colon ("makes these abstract categories operational: @tbl-model-card-example shows…"). The cells are rich (seven sections, each with a deployment-specific entry), but the prose around the table does not state which row or contrast carries the key lesson. The payoff paragraph (L708) pivots to datasheets for datasets without naming what the MobileNetV2 example demonstrates about the transition from abstract to operational. The takeaway — that the Ethical Considerations row forces teams to state biases and exclusions explicitly, turning a policy intent into a reviewable artifact — lives only in the cell.

**Where the takeaway currently lives:** the Ethical Considerations and Intended Use cells contain the key load-bearing content; no prose sentence names either.

**Missing move:** Lead-out / Interpret — a sentence stating what the concrete card reveals that the abstract framework does not.

**Rule-compliant rewrite:**

Replace the citation sentence at L687:

> A concrete MobileNetV2 card shows where the abstract categories acquire bite. In @tbl-model-card-example, the Intended Use section explicitly excludes medical diagnosis and security screening — uses the architecture could technically support — while the Ethical Considerations section names the geographic bias in ImageNet directly. Stating an exclusion forces a team to have the conversation about scope before deployment; stating a known bias creates an artifact that reviewers can audit. The card converts implicit engineering judgment into reviewable documentation.

---

### Finding 4 — `tbl-fairness-archetype` (tbl 🟠) — def L738

**Verbatim citation sentence (L724, inside `.callout-lighthouse`):**
> "@Tbl-fairness-archetype maps each archetype to its primary risk and evaluation metric."

**Problem — bare pointer with no lead-out.**
The citation is a pure pointer: it names the float and describes its structure ("maps each archetype to its primary risk and evaluation metric") without stating what pattern the mapping reveals or which contrast is load-bearing. The lead-in at L724 identifies the question ("dominant fairness risks differ by workload archetype") but never answers it before pointing at the table. The payoff outside the callout (L749) does deliver the conclusion ("a vision model fails differently than a recommendation system"), but that sentence is separated from the citation by the entire table body and the Systems insight paragraph, and it is outside the callout where the table lives.

**Where the takeaway currently lives:** the "Systems insight" bullet inside the callout names the evaluation strategies per archetype, and L749 in body prose states the main conclusion — but neither is the lead-out in the citation paragraph itself.

**Missing move:** Interpret — a sentence in the citation paragraph that states the pattern the table makes visible.

**Rule-compliant rewrite:**

Replace the citation sentence at L724:

> The dominant fairness risks differ by workload archetype, and the differences are structural, not incidental. @Tbl-fairness-archetype maps each archetype to its primary risk and evaluation metric. Vision models fail through demographic underrepresentation in training data, measured by disaggregated accuracy. Recommendation systems fail through feedback-loop amplification, measured by exposure audits. Speech models fail through deployment-context mismatch, measured by false positive rates across acoustic environments. Matching the evaluation metric to the archetype's failure mode is the engineering discipline the table encodes.

---

### Finding 5 — `tbl-explainability-requirements` (tbl 🟠) — def L1087

**Verbatim citation sentence (L1077):**
> "@Tbl-explainability-requirements maps common deployment scenarios to their explainability needs."

**Problem — bare pointer with no contrast stated.**
The citation names the float and describes what it contains ("maps common deployment scenarios to their explainability needs") without identifying the key contrast in the table. The prior sentence establishes the varying-by-context premise. The payoff (L1089) discusses how to choose approaches but does not name the table's most consequential row. The table's sharpest insight — that fraud detection intentionally limits explainability to prevent adversarial gaming, the opposite of every other row — is never stated in prose.

**Where the takeaway currently lives:** the fraud-detection row cell reads "Detailed explanations may enable adversarial gaming" — body prose never names this exception.

**Missing move:** Interpret — a sentence naming the key contrast or the most consequential row rather than pointing at the structure.

**Rule-compliant rewrite:**

Replace the citation sentence at L1077:

> The explainability requirement is not a single dial but a set of domain-specific obligations. @Tbl-explainability-requirements maps common deployment scenarios to their explainability needs and surfaces a tension that generic guidance misses: while credit and medical applications face hard regulatory requirements for individual explanations, fraud detection systems must often limit explainability deliberately to prevent adversaries from gaming the detection logic. The engineering challenge is that both extremes are principled responses to real constraints, not oversights.

---

### Finding 6 — `tbl-model-efficiency-comparison` (tbl 🟠) — def L1677

**Verbatim citation sentence (L1668):**
> "The benchmarks in @tbl-model-efficiency-comparison provide actionable guidance for efficiency optimization."

**Problem — pointer ("provide actionable guidance") with no stated conclusion.**
The citation names the float and uses "provide actionable guidance" as the interpret move, but that phrase is a generic description of any comparison table. The prose does not name which model meets which context's constraints, which is the specific conclusion the table encodes. The payoff (L1688) applies the wearable row specifically (TinyML leaves a power margin while MobileNetV2 exceeds the budget), which is the table's key row — but that sentence is separated from the citation by the entire table body.

**Where the takeaway currently lives:** L1688 names the wearable row comparison; no sentence at the citation point names which model is the right selection for constrained contexts.

**Missing move:** Interpret — a sentence stating what the table's comparison reveals at the citation point.

**Rule-compliant rewrite:**

Replace the citation sentence at L1668:

> Model selection for constrained deployment is a matching problem, and @tbl-model-efficiency-comparison makes the match concrete. TinyML models fit both smartphone and IoT power budgets; MobileNetV2 fits smartphones but exceeds the IoT constraint; ResNet-50 exceeds both. The table's key result is not that smaller is better but that a model must clear a hard constraint floor before accuracy comparisons become relevant.

---

### Finding 7 — `tbl-tco-training` (tbl 🟠) — def L2064

**Verbatim citation sentence (L2048):**
> "@Tbl-tco-training breaks down these costs, showing how quarterly retraining cycles accumulate over a three-year operational period."

**Problem — bare pointer with no stated finding.**
The citation names the float and describes its structure ("breaks down these costs, showing how quarterly retraining cycles accumulate") without stating what the breakdown reveals. The table's key result — that training is only a small fraction of TCO despite appearing substantial — is in the caption only. The payoff paragraph (L2072) pivots directly to inference costs without stating the training cost conclusion. A reader who skips the cells and caption gets no prose-delivered finding about training's share.

**Where the takeaway currently lives:** the caption contains "Despite appearing substantial, training represents only `{python} TCOCalc.p_train_str` of total cost of ownership" — body prose never delivers this finding.

**Missing move:** Lead-out / Interpret — a sentence stating the training cost finding before pivoting to inference.

**Rule-compliant rewrite:**

Replace the citation sentence at L2048:

> Training costs are front-loaded but small relative to operational lifetime. @Tbl-tco-training breaks down these costs across initial development and quarterly retraining, showing how cycles accumulate over a three-year period. The result challenges the common assumption that training is the dominant cost: training represents only a few percent of total cost of ownership, a fraction that shrinks further as serving scale grows. The bulk of cost accumulates in inference, examined next.

---

### Finding 8 — `tbl-tco-summary` second ref (tbl 🟠) — def L2164, second ref L2954

**Verbatim citation sentence (L2954):**
> "For the recommendation system analyzed in @tbl-tco-summary, training accounts for just `{python} ResponsibleTcoRecap.p_train_str` of three-year costs while inference accounts for `{python} ResponsibleTcoRecap.p_inf_str`."

**Note:** The first citation at L2155 is ✅ (content leads, ref rides along with the percentages named). The second citation at L2954 is also ✅ — the percentages are named in the sentence and the ref is parenthetical. No additional finding needed for the second ref.

> **Correction:** Finding 8 is withdrawn — the second ref at L2954 is a cite-with-values pattern (✅). Replacing with a re-check of `tbl-tco-summary` first ref at L2155.

**Revised Finding 8 — `tbl-tco-summary` first ref (tbl 🟠) — def L2164, ref L2155:**

**Verbatim citation sentence (L2155):**
> "The stark breakdown in @tbl-tco-summary answers where the money goes: inference at `{python} TCOCalc.p_inf_str`, operations at `{python} TCOCalc.p_ops_str`, and training at only `{python} TCOCalc.p_train_str`."

Content leads, ref rides. ✅ — No finding.

**Revised tally: 7 findings (⚠️), 0 (🛑).**

---

## Revised summary table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|:-----|:------|-------:|---:|---:|---:|
| eq   | 🔴    | 2      | 1  | 1  | 0  |
| fig  | 🟠    | 6      | 5  | 1  | 0  |
| lst  | 🟡    | 1      | 1  | 0  | 0  |
| tbl  | 🟠    | 16     | 11 | 5  | 0  |
| **Total** |  | **25** | **18** | **7** | **0** |

---

## Passes (✅ — documented for completeness)

| Label | Type | Notes |
|:------|:-----|:------|
| `eq-carbon-footprint` | eq 🔴 | Symbols named, units given, conversion factor computed in bullets immediately after |
| `fig-fairness-frontier` | fig 🟠 | Two citations; both carry full lead-in and lead-out |
| `fig-governance-layers` | fig 🟠 | Citation sentence delivers the nesting insight and the key implication |
| `fig-fairness-threshold` | fig 🟠 | Citation names the fundamental tension; lead-out follows |
| `fig-interpretability-spectrum` | fig 🟠 | Citation walks left-to-right; next paragraph delivers the choice implication |
| `fig-data-governance-pillars` | fig 🟠 | Four-domain takeaway and interdependence point delivered in citation paragraph |
| `lst-fairness-metrics-code` | lst 🟡 | Mechanism ("compute per-group metrics, flag disparities") and context stated |
| `tbl-failure-modes` | tbl 🟠 | Silent-failure contrast explicitly named as the key insight |
| `tbl-gender-shades-results` | tbl 🟠 | Numeric disparity factor and specific group error rates named in citation sentence |
| `tbl-predeployment-assessment` | tbl 🟠 | Both citations carry distinct purpose; critical-path vs. high-priority distinction stated |
| `tbl-confusion-group-a` | tbl 🟠 | Co-cited with Group B; "what the aggregate conceals" sets up the question; L847 delivers the answer |
| `tbl-confusion-group-b` | tbl 🟠 | As above |
| `tbl-fairness-metrics-summary` | tbl 🟠 | Key finding (TPR disparity far exceeds threshold) stated before the ref |
| `tbl-incident-response` | tbl 🟠 | Five components named; fairness-specific extension made explicit |
| `tbl-edge-deployment-constraints` | tbl 🟠 | Wearable example grounds the abstraction; re-ref applies specific row values |
| `tbl-tco-inference` | tbl 🟠 | Key finding ("inference costs dominate") stated in the citation sentence |
| `tbl-tco-operations` | tbl 🟠 | Operational burden contrast (silent failures vs. binary outages) stated before the ref |
| `tbl-tco-summary` | tbl 🟠 | Both citations: content leads with values, ref rides |

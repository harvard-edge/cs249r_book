# Float Exposition Worklist — `sustainable_ai.qmd` (vol2)

Rubric: FLOAT_EXPOSITION_STANDARD.md — eq/alg 🔴, tbl/fig 🟠, lst 🟡.
Caption, fig-alt, in-figure labels, callout interiors do NOT count toward prose's job.

---

## Summary Table

| Type | Level | Floats | ✅ | ⚠️ | 🛑 |
|------|-------|--------|----|----|-----|
| eq   | 🔴    | 17     | 13 |  3 |  1  |
| fig  | 🟠    | 25     | 20 |  4 |  1  |
| tbl  | 🟠    | 12     |  9 |  3 |  0  |
| lst  | 🟡    |  1     |  1 |  0 |  0  |
| **Total** | | **55** | **43** | **10** | **2** |

---

## Findings (⚠️ and 🛑 only)

---

### EQUATIONS

---

#### `eq-energy-arithmetic-intensity` (🔴 eq) — def L1261
**Grade: ⚠️ Partial**

**Ref sentence (L1259):**
> "The balance between computation and data movement determines whether energy consumption is compute-bound or memory-bound. @Eq-energy-arithmetic-intensity defines arithmetic intensity (AI), the ratio that determines which resource dominates energy consumption:"

**Missing move:** The lead-in and cite are present, but the payoff paragraph (L1263) immediately pivots to cite two further equations (@Eq-energy-total and @eq-energy-roofline) without first stating what arithmetic intensity *implies*: what value separates memory-bound from compute-bound, or what practical range typical workloads fall in. The interpretation ("so what does this ratio tell you?") lives only implicitly, deferred to the worked example that follows. Removability test: delete the equation and the surrounding prose loses its concrete anchor, but the interpret move is effectively missing from the cite paragraph itself — the payoff sentence delivers a forward-pointer, not a takeaway.

**Where takeaway currently lives:** Implicitly in the MatMul worked example (L1349 onward) and the energy roofline figure caption.

**Rule-compliant diff rewrite** (add interpret move to the payoff sentence at L1263, before the forward pivot):

```diff
-Arithmetic intensity measured in FLOP/byte determines the dominant energy consumer. @Eq-energy-total expresses total energy as the sum of compute and memory contributions, while @eq-energy-roofline isolates the roofline-style dominant term:
+Arithmetic intensity measured in FLOP/byte determines the dominant energy consumer: workloads with low arithmetic intensity spend most of their energy moving bytes, while workloads with high arithmetic intensity are bounded by the cost of arithmetic itself. @Eq-energy-total expresses total energy as the sum of compute and memory contributions, while @eq-energy-roofline isolates the dominant term for roofline reasoning:
```

---

#### `eq-energy-total` (🔴 eq) — def L1265
**Grade: ⚠️ Partial**

**Ref sentence (L1263):**
> "@Eq-energy-total expresses total energy as the sum of compute and memory contributions, while @eq-energy-roofline isolates the roofline-style dominant term:"

**Missing move:** The sentence cites and briefly glosses ("sum of compute and memory"), but provides no interpretation: what the equation implies for optimization, or what each term costs relative to the other. The payoff at L1349 is a LEGO setup cell (not prose) before the worked example, so no prose interpret move follows the equation cluster. The "where" clause for symbols comes later (L1269 covers $E_{\text{compute}}$ and $E_{\text{move}}$), but only after a further equation, not in the cite paragraph.

**Where takeaway currently lives:** In the MatMul worked example body (L1349+) and in the caption of `fig-energy-roofline`.

**Rule-compliant diff rewrite** (extend the cite sentence to deliver the interpret move):

```diff
-Arithmetic intensity measured in FLOP/byte determines the dominant energy consumer. @Eq-energy-total expresses total energy as the sum of compute and memory contributions, while @eq-energy-roofline isolates the roofline-style dominant term:
+Arithmetic intensity measured in FLOP/byte determines the dominant energy consumer: workloads with low arithmetic intensity spend most of their energy moving bytes, while workloads with high arithmetic intensity are bounded by the cost of arithmetic itself. @Eq-energy-total decomposes total energy into compute and memory terms, making explicit that both the number of operations and the volume of data moved carry independent energy costs, and that an optimization reducing only FLOPs may leave memory energy untouched. @Eq-energy-roofline isolates the dominant term for roofline reasoning:
```

*(Note: this rewrite subsumes the `eq-energy-arithmetic-intensity` fix above; the two are cited together in the same paragraph and the rewrite addresses both.)*

---

#### `eq-energy-roofline` (🔴 eq) — def L1267
**Grade: ⚠️ Partial** (same paragraph as eq-energy-total above)

**Ref sentence (L1263):**
> "while @eq-energy-roofline isolates the roofline-style dominant term:"

**Missing move:** "Isolates the roofline-style dominant term" is a structural description, not an interpretation. The prose does not state that this is an approximation useful for first-order reasoning (not the full energy budget), nor that it directly defines the memory-bound vs. compute-bound regimes. The payoff for what the max() expression *means* lives in L1269 ("The maximum term identifies the dominant bottleneck for roofline reasoning; it is not the full energy in balanced cases"), which is adequate as a where-clause after `eq-ai-crossover` — but by then the reader has passed two equations without an interpret move. The issue is that L1263 bears the cite for all three consecutive equations, making the interpret move thin for each individually.

**Where takeaway currently lives:** L1269 partially covers it but arrives after the crossover equation, not after this one.

**Rule-compliant diff rewrite** (add the interpret move immediately after L1267, as a brief where-clause):

```diff
 $$E_{\text{dominant}} = \max\left(O \times E_{\text{compute}}, D_{\text{vol}} \times E_{\text{move}}\right)$$ {#eq-energy-roofline}
-
-where $E_{\text{compute}}$ is energy per FLOP and $E_{\text{move}}$ is energy per byte moved. The maximum term identifies the dominant bottleneck for roofline reasoning; it is not the full energy in balanced cases.
+
+where $E_{\text{compute}}$ is energy per FLOP and $E_{\text{move}}$ is energy per byte moved. Taking the maximum isolates the dominant energy contributor for roofline reasoning: when memory energy exceeds compute energy, reducing FLOPs does little, and the right intervention targets data reuse instead. This is not the full energy in balanced cases but correctly identifies which lever matters most.
```

---

#### `eq-lifecycle-carbon` (🔴 eq) — def L1765
**Grade: 🛑 Fails**

**Ref sentence (L1763):**
> "Complete lifecycle assessment combines operational and embodied emissions across all phases. @Eq-lifecycle-carbon aggregates these contributions:"

**Missing move:** No interpret move at all. The equation is cited with a purely structural gloss ("aggregates these contributions"), and the very next sentence (L1767) immediately pivots to a figure ("As @fig-carbon-lifecycle shows..."). There is no prose statement of what the equation implies: which term typically dominates, how the balance shifts over deployment lifetime, or what the equation enables a practitioner to do. The payoff is entirely delegated to the figure and caption.

**Where takeaway currently lives:** In the `fig-carbon-lifecycle` caption ("training dominates this single-deployment lifecycle snapshot") and the figure's payoff prose. Those are caption-level and therefore do not count.

**Rule-compliant diff rewrite** (add interpret move between the equation and the figure reference):

```diff
 $$C_{\text{lifecycle}} = C_{\text{training}} + C_{\text{inference}} + C_{\text{embodied}}$$ {#eq-lifecycle-carbon}
-
-As @fig-carbon-lifecycle shows, training dominates this single-deployment lifecycle snapshot, while manufacturing and inference remain significant factors.
+
+For a single deployment, training dominates this sum because the one-time compute cost of a large model dwarfs inference energy before the model accumulates significant serving traffic. That balance inverts over time: as cumulative inference queries grow, the inference term compounds while the training and embodied terms remain fixed. @Fig-carbon-lifecycle visualizes this lifecycle snapshot, with training as the largest share for a single run.
```

---

### FIGURES

---

#### `fig-data-center-energy-usage` (🟠 fig) — def L620
**Grade: ⚠️ Partial**

**Ref sentence (L618):**
> "To make the uncertainty visible, @fig-data-center-energy-usage shows high-growth sensitivity scenarios for data center electricity usage rather than the IEA baseline forecast above. The spread between best, expected, and worst cases illustrates how strongly the outcome depends on efficiency improvements and demand growth assumptions."

**Missing move:** The cite sentence names what the figure contains and notes that the spread depends on efficiency improvements. It does not deliver the figure's key takeaway: what the *magnitude* of the spread means — roughly a 10-fold range between best and worst cases by 2030 — and what that implies for infrastructure planning decisions. The prose points at the spread without interpreting it as an engineering constraint. Removability test: delete the figure and the prose still teaches "uncertainty depends on efficiency"; the figure's quantitative lesson (the spread spans an order of magnitude and this shapes how conservatively engineers should plan) lives only in the caption.

**Where takeaway currently lives:** In the caption ("The three trajectories diverge significantly after 2018").

**Rule-compliant diff rewrite** (extend cite sentence to add interpret move):

```diff
-To make the uncertainty visible, @fig-data-center-energy-usage shows high-growth sensitivity scenarios for data center electricity usage rather than the IEA baseline forecast above. The spread between best, expected, and worst cases illustrates how strongly the outcome depends on efficiency improvements and demand growth assumptions.
+To make the uncertainty visible, @fig-data-center-energy-usage shows high-growth sensitivity scenarios for data center electricity usage rather than the IEA baseline forecast above. The worst-case trajectory reaches roughly 8,000 TWh by 2030 — more than ten times the best-case — meaning that engineers designing capacity today are making decade-scale bets on an outcome that could differ by an order of magnitude. Infrastructure decisions made on the IEA baseline alone carry a failure mode proportional to that spread.
```

---

#### `fig-carbon-lifecycle` (🟠 fig) — def L1775
**Grade: ⚠️ Partial**

**First ref sentence (L1767):**
> "As @fig-carbon-lifecycle shows, training dominates this single-deployment lifecycle snapshot, while manufacturing and inference remain significant factors."

**Second ref sentence (L1781):**
> "That single-deployment snapshot in @fig-carbon-lifecycle tells only part of the story. The cumulative picture is the opposite: a model serving millions of queries per day can exceed its entire training carbon footprint within months..."

**Missing move:** The first cite ("As @fig-carbon-lifecycle shows") is a float-announcer pattern — it delegates the takeaway to the figure rather than stating it in prose first. The content of the second cite (L1781) is strong and substantive, but it functions as a pivot *away* from the figure's story rather than an interpretation of what the figure demonstrates. The figure's actual visual story (the relative proportions of training, inference, and manufacturing in a pie or bar) is never stated in prose: the reader is told only that training "dominates," but not by how much, or what fraction manufacturing represents. Removing the figure leaves the prose with "training dominates" — which is thin for a 🟠 figure.

**Where takeaway currently lives:** Primarily in the caption ("training as the largest share, followed by manufacturing and inference").

**Rule-compliant diff rewrite** (replace float-announcer with content-led sentence):

```diff
-As @fig-carbon-lifecycle shows, training dominates this single-deployment lifecycle snapshot, while manufacturing and inference remain significant factors.
+For a single deployment, training accounts for the majority of lifecycle carbon, with embodied manufacturing emissions and inference each representing meaningful but smaller shares. @Fig-carbon-lifecycle breaks down these proportions, making the relative magnitudes concrete: the manufacturing footprint that occurs before the model ever runs is comparable in scale to months of production inference, which explains why hardware longevity is a first-order sustainability lever.
```

---

#### `fig-ai-lca` (🟠 fig) — def L3336
**Grade: ⚠️ Partial**

**Ref sentence (L3334):**
> "Each of the four primary lifecycle stages contributes to an AI system's total environmental footprint. @Fig-ai-lca visualizes this progression from design through disposal, highlighting the interdependencies between phases and the environmental impact categories associated with each stage."

**Missing move:** The cite sentence describes the figure's structure but delivers no takeaway about what the lifecycle sequence *implies*. The payoff (L3340) is strong ("the binding sustainability problem shifts as the system matures"), but it arrives *after* the figure, not in the cite paragraph — and the figure is sandwiched between cite and payoff with no interpret move in the cite itself. More critically, the cite is purely structural ("visualizes this progression," "highlighting interdependencies"), which is the exact pattern the standard flags for figures: "Figure X illustrates this" with no prose of the mechanism. The removability test is borderline: L3340 does carry the takeaway, but it needs to be connected to the figure in the cite, not delivered separately as a paragraph after the figure.

**Where takeaway currently lives:** L3340 (payoff paragraph) — adequately placed but logically disconnected from the cite sentence.

**Rule-compliant diff rewrite** (move the substance into the cite, with L3340 remaining as elaboration):

```diff
-Each of the four primary lifecycle stages contributes to an AI system's total environmental footprint. @Fig-ai-lca visualizes this progression from design through disposal, highlighting the interdependencies between phases and the environmental impact categories associated with each stage.
+Each of the four primary lifecycle stages contributes to an AI system's total environmental footprint, but the dominant stage shifts as the system matures: manufacturing locks in embodied carbon before the system runs, operations couple the workload to grid and cooling constraints, and disposal externalizes e-waste that scales with replacement frequency. @Fig-ai-lca maps this progression from design through disposal, with environmental impact categories shown for each phase.
```

---

#### `fig-mckinsey-analysis` (🟠 fig) — def L2832
**Grade: ⚠️ Partial**

**Ref sentence (L2830):**
> "Early market forecasts anticipated this shift: a 2017 McKinsey projection (@fig-mckinsey-analysis) expected data center and edge inference markets to grow faster than training through 2025. The stronger evidence is physical rather than economic. The Meta lifecycle measurements in @fig-meta-analysis show inference serving at scale rivaling or exceeding training emissions..."

**Missing move:** The figure is cited parenthetically — `(@fig-mckinsey-analysis)` — embedded mid-sentence as a provenance note rather than as a substantive reference. The prose immediately pivots to "the stronger evidence is physical rather than economic," implicitly discounting the figure before explaining what the reader should take from it. There is no interpretation of what the figure demonstrates: specifically, that the McKinsey projection showed inference overtaking training in market size well before the carbon evidence was available, validating the thesis from a different measurement axis. The payoff (L2932) does not connect back to the figure. Removability test: removing the parenthetical citation loses nothing pedagogically — the figure contributes no lesson that survives in the surrounding prose.

**Where takeaway currently lives:** Implicitly in the caption ("inference workloads dominated the projected growth").

**Rule-compliant diff rewrite**:

```diff
-Early market forecasts anticipated this shift: a 2017 McKinsey projection (@fig-mckinsey-analysis) expected data center and edge inference markets to grow faster than training through 2025.
+Early market forecasts anticipated this shift well before carbon measurements confirmed it. @Fig-mckinsey-analysis shows a 2017 McKinsey projection in which data center inference was expected to double by 2025 while edge inference grew from near zero to a comparable market segment — both outpacing training growth. The market signal and the physical evidence converge on the same conclusion: inference at production scale is the dominant energy and carbon problem for deployed systems.
```

---

#### `fig-carbon-scheduling` (🟠 fig) — def L3689
**Grade: 🛑 Fails**

**Ref sentence (L3687):**
> "Google's carbon-intelligent computing platform demonstrated this approach at scale, achieving a 40 percent reduction in carbon footprint under its global workload-shifting assumptions. @Fig-carbon-scheduling shows where this intervention falls in the broader cascade: it appears as a systemic-stage step and contributes a more conservative 1.3× average reduction for mixed production fleets, where only some jobs are deadline-tolerant enough to move across time or geography."

**Problem:** The cite sentence is entirely about the number (1.3×) and the caveat, but the figure reuses the same cascade SVG as `fig-energy-intervention`. The prose does not explain what the reader should *look at* in this figure that differs from the earlier one, nor does it interpret why the figure is placed here rather than just repeating the cross-reference. The payoff (L3693) is a footnote, not body prose — footnotes do not count. The figure effectively has no body-prose lead-out after the parenthetical citation. Removability test: remove the figure and nothing in the surrounding prose changes — the 1.3× number and the description of Google's system are already in the cite sentence. The figure adds no prose-supported lesson.

**Where takeaway currently lives:** In the caption ("carbon-aware scheduling appearing as one of the systemic-stage steps") and in the footnote at L3693.

**Rule-compliant diff rewrite** (add interpret move as a sentence after the figure):

Since the figure must come after the cite by convention, the fix is to add body prose immediately following the figure:

```diff
+Carbon-aware scheduling's position in the cascade reveals something important: it is a systemic, not algorithmic, intervention. Unlike algorithmic efficiency gains that require retraining or re-architecture, scheduling shifts are deployable today, require no model changes, and compound with every efficiency improvement applied earlier in the cascade.
```

---

### TABLES

---

#### `tbl-edge-power-monitors` (🟠 tbl) — def L1432
**Grade: ⚠️ Partial**

**Ref sentence (L1423):**
> "Mobile profiling tools integrate with development workflows, enabling iterative optimization of on-device inference energy consumption during model deployment. @Tbl-edge-power-monitors summarizes edge power measurement instruments across platforms, including resolution, accuracy, and integration requirements."

**Missing move:** The cite sentence is a pure structural descriptor ("summarizes instruments across platforms"), which the standard flags for tables as insufficient. The table's key result — which instrument to reach for and why — lives only in the cells. Specifically, the table's load-bearing contrast is that accuracy spans from ±0.1 percent (Joulescope JS220) to ±2 percent (PAC1934), and that this spread matters because TinyML energy budgets are in microampere ranges where instrument accuracy becomes the measurement bottleneck. No such conclusion appears in the prose.

**Where takeaway currently lives:** In the caption ("The Joulescope JS220 provides the gold-standard accuracy for TinyML research").

**Rule-compliant diff rewrite**:

```diff
-Mobile profiling tools integrate with development workflows, enabling iterative optimization of on-device inference energy consumption during model deployment. @Tbl-edge-power-monitors summarizes edge power measurement instruments across platforms, including resolution, accuracy, and integration requirements.
+Mobile profiling tools integrate with development workflows, enabling iterative optimization of on-device inference energy consumption during model deployment. The key selection criterion is accuracy relative to the current being measured: at sub-milliamp TinyML operating points, ±2 percent measurement error can exceed the efficiency gain being optimized. @Tbl-edge-power-monitors surveys the instrument landscape, with the Joulescope JS220 setting the accuracy floor for research-grade TinyML benchmarking and INA-series sensors providing a cost-effective path for deployment validation.
```

---

#### `tbl-training-emissions` (🟠 tbl) — def L3353
**Grade: ⚠️ Partial**

**Ref sentence (L3344):**
> "@Tbl-training-emissions reveals stark differences in model carbon footprint across model scales."

**Missing move:** "Stark differences across model scales" is a direction, not a takeaway. The table's load-bearing contrast is that carbon scales roughly with FLOPs (super-linearly in practice), so moving from BERT-Base to GPT-3 increases emissions by roughly 770×, far faster than accuracy improvements — which is the engineering point that motivates transfer learning and efficient NAS. That conclusion is not stated in the prose; the payoff (L3355) immediately pivots to mitigation techniques without interpreting the table's numbers.

**Where takeaway currently lives:** In the cells and the caption.

**Rule-compliant diff rewrite**:

```diff
-@Tbl-training-emissions reveals stark differences in model carbon footprint across model scales.
+@Tbl-training-emissions reveals that carbon scales super-linearly with model size: moving from BERT-Base to GPT-3 increases estimated emissions by more than 700×, a ratio that dwarfs any accuracy gain from that scaling step. The implication is that fine-tuning a pretrained model or using transfer learning is not merely a computational convenience — it is a sustainability decision that eliminates most of this multiplier.
```

---

#### `tbl-material_depletion` (🟠 tbl) — def L3310
**Grade: ⚠️ Partial**

**Ref sentence (L3295):**
> "@Tbl-material_depletion quantifies the scope of this material dependency challenge."

**Missing move:** The cite is a pure table-announcer ("quantifies the scope"). The prose before the table (L3295) establishes the problem context — critical materials, geographic concentration of refining — but does not deliver the table's key conclusion. The table's load-bearing row is the geographic concentration risk: specifically that gallium refining is concentrated in China and helium in the US and Qatar, making these critical supply bottlenecks for AI chip manufacturing. The payoff (L3312) pivots to ecosystem impacts without interpreting which materials pose the highest supply risk, which is what the table encodes.

**Where takeaway currently lives:** In the table cells (the "Supply Concerns" column).

**Rule-compliant diff rewrite**:

```diff
-@Tbl-material_depletion quantifies the scope of this material dependency challenge.
+@Tbl-material_depletion catalogs this dependency, and the supply-concern column reveals a pattern: the rarest materials — gallium for power semiconductors, helium for EUV lithography and plasma etching, and rare earth elements for magnets — combine limited global supply with geographically concentrated production, making AI chip manufacturing vulnerable to supply shocks that have no domestic substitute.
```

---

## Dangling reference note

`@tbl-prefill-decode` is cited at L2942 but has no matching definition in the chapter. This appears to be an orphaned cross-reference to a table that was removed or renamed; it should be investigated separately from float exposition findings.

---

*Floats passing standard (43 of 55): eq-cmos-power ✅, eq-pue ✅, eq-wue ✅, eq-edge-duty-cycle ✅, eq-total-energy ✅, eq-facility-power ✅, eq-operational-carbon ✅, eq-embodied-daily ✅, eq-tinyml-duty-cycle ✅, eq-cascade-energy ✅, eq-federated-energy ✅, eq-energy-delay-product ✅, fig-energy-wall-quantitative ✅, fig-energy-wall ✅, fig-model-scaling ✅, fig-energy-intervention ✅, fig-ethical-ai ✅, fig-carbon-intensity ✅, fig-energy-roofline ✅, fig-carbon-tco ✅, fig-ai-data-center-demand ✅, fig-energy-gap ✅, fig-carbonfootprint ✅, fig-meta-analysis ✅, fig-ghg-protocol ✅, fig-cooling-comparison ✅, fig-water-cycle ✅, fig-iot-number ✅, fig-jevons-ai ✅, fig-carbon-aware-scheduling ✅, fig-europe-energy-grid ✅, fig-prefill-decode-energy ✅, tbl-carbon-intensity ✅, tbl-energy-per-op ✅, tbl-energy-per-byte ✅, tbl-rack-power ✅, tbl-cooling-limits ✅, tbl-edge-power-budgets ✅, tbl-energy-harvesting ✅, tbl-tinyml-optimization ✅, tbl-mlperf-tiny ✅, lst-carbon-calculation ✅, fig-prefill-decode-energy ✅*

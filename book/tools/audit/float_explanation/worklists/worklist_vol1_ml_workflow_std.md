# Float exposition eval — ml_workflow.qmd (vol1)
Standard: FLOAT_EXPOSITION_STANDARD.md (caption excluded from prose budget)

## Summary
| type | level | floats | ✅ | ⚠️ | 🛑 |
|---|---|---|---|---|---|
| equation | 🔴 | 0 | 0 | 0 | 0 |
| algorithm | 🔴 | 0 | 0 | 0 | 0 |
| table | 🟠 | 3 | 3 | 0 | 0 |
| figure | 🟠 | 6 | 6 | 0 | 0 |
| listing | 🟡 | 0 | 0 | 0 | 0 |
| **total** | | **9** | **9** | **0** | **0** |

## Findings (⚠️ / 🛑 only)

None. All 9 floats meet the standard.

---

## Per-float grading notes (✅ cases)

### `fig-ml-lifecycle` (figure 🟠) — def L101 ✅
- **Lead-in:** The `dfn-ml-workflow-machine-learning-lifecycle` callout directly above establishes the lifecycle as a closed feedback loop.
- **Citation (L99):** Names both pipelines, lists all ten stages, and names the three feedback arrows (deployment-to-collection, data-fixes, data-needs).
- **Lead-out (L99):** "These feedback paths create the continuous improvement cycles that distinguish ML from traditional linear development." Prose delivers the takeaway without relying on the figure.
- **Removability test:** Delete the figure and the argument still holds. ✅
- **Note:** Referenced six times total; each later citation (L285, L2165, L2209, L2213, L2225) adds a distinct interpretive layer — iteration cycles, pitfall analysis, and the summary reread. Exceptionally well-integrated.

### `fig-ds-time` (figure 🟠) — def L236 ✅
- **Lead-in / Citation (L234):** Delivers all key numbers in prose (cleaning %, collection %, combined data-work %), names the practical lesson ("data preparation can dominate ML engineering effort"), and names the "long tail" before and after the figure ("The long tail of @fig-ds-time is as telling as its dominant slice: the model-focused activities… together drew only [model_focused_str]%").
- **Lead-out:** The conclusion that "In ML projects, the 'source code' is the data" is stated explicitly in body prose.
- **Removability test:** Delete the figure and all key proportions remain in the text. ✅

### `fig-lifecycle-overview` (figure 🟠) — def L459 ✅
- **Citation (L457):** Names all six stages, identifies the feedback loop as "the key insight in the diagram," and explains what it encodes (production signals flow back to inform earlier phases). The prose does not merely describe; it extracts the insight.
- **Second citation (L820):** Adds the iron law interpretation — each stage maps to a specific performance-equation term. This is a substantive lead-out beyond the first citation.
- **Third citation (L884):** Uses the figure to explain *why* ML projects experience iteration cycles (downstream violation of upstream contract = forced backtrack).
- **Removability test:** The argument about cyclical ML development is carried entirely in prose. ✅

### `fig-eye-dr` (figure 🟠) — def L859 ✅
- **Citation (L855):** "look closely at @fig-eye-dr: the clinical challenge is detecting characteristic hemorrhages (dark red spots) that indicate disease progression." The prose names the visual feature the reader is meant to notice.
- **Lead-out (L865):** Explains what the figure's context means at the systems level — laboratory success must integrate with data quality challenges, infrastructure constraints in rural clinics, regulatory requirements, and workflow integration. The payoff is in body prose, not caption.
- **Removability test:** The constraint-propagation argument proceeds without the figure. The figure is concrete reinforcement. ✅
- **Note:** The scanner flagged this as an orphan due to a trailing colon in the ref token; it is not a genuine orphan (verified against source, consistent with prior `worklist_vol1_ml_workflow.md` note).

### `fig-ml-lifecycle-feedback` (figure 🟠) — def L1226 ✅
- **Lead-in (L1220–L1222):** Two paragraphs establish the data quality stakes (blurry images corrupting the training distribution) and the local-vs-centralized validation design.
- **Citation (L1224):** Before the figure appears, prose walks through three concrete DR-specific feedback arrow examples: older fundus cameras triggering targeted collection, cataract patients driving augmentation, equipment upgrades triggering preprocessing updates. The figure is reinforcement for walkthrough already delivered in prose.
- **Lead-out (L1538):** "data collection does not end when training begins" — brief but accurate distillation of the diagram's central claim.
- **Removability test:** The three feedback examples are all stated in prose; deleting the figure preserves the argument. ✅

### `fig-mlops-returns` (figure 🟠) — def L1652 ✅
- **Citation (L1650):** Names the coordination tax, names the flywheel effect, explains the axes are relative units, and frames the shape question ("show, not absolute throughput"). All framing is in body prose before the figure.
- **Lead-out (L1750):** A full mechanistic explanation of both curves — red saturates because of combinatorial synchronization overhead; blue escapes because the platform absorbs coordination into reusable infrastructure. The "widening gap" is named and its meaning stated. This is the H&P "key result" sentence, delivered in body prose.
- **Removability test:** Delete the figure and the two-curve argument is still fully stated. ✅

### `tbl-sw-ml-cycles` (table 🟠) — def L429 ✅
- **Lead-in (L414):** Delivers the critical conclusion before the citation: "Unlike traditional software where later phases rarely influence earlier ones, ML systems require continuous feedback loops: deployment insights reshape data collection, monitoring drives model updates, and production data reveals distributional properties invisible in development." This is the table's key row stated as a prose conclusion.
- **Citation (L416):** "@Tbl-sw-ml-cycles contrasts these differences across six development dimensions, from problem definition through maintenance." Scope-setting only, but the takeaway was already in the lead-in.
- **Lead-out:** The next paragraph (L433) pivots to the OS-assumptions argument, which is a substantive consequence of the table's conclusion.
- **Removability test:** The six-dimension ML-vs-traditional contrast is stated in prose at L414. ✅
- **Note:** The payoff prose at L433 does not restate the table's key row, but the lead-in delivers it so robustly that the standard is met. If ever revised, a one-sentence explicit consequence after the table would make this a model case.

### `tbl-lighthouse-workflow-comparison` (table 🟠) — def L845 ✅
- **Lead-in / Citation (L822):** "The binding constraint differs dramatically across workload archetypes, causing each lifecycle stage to optimize different iron law terms." Names all three archetypes and identifies what each one stresses (accelerator utilization, memory bandwidth for embeddings, energy and memory for TinyML). This is the table's conclusion stated in body prose before the cells are seen.
- **Lead-out (L847):** "Production systems rarely fall neatly into a single archetype… Understanding *how* the same workflow framework adapts to each archetype, and *how* a single project can span multiple archetypes simultaneously, is essential for making sound engineering decisions." States the implication and its practical consequence.
- **Removability test:** The archetype-constraint mapping is in prose; the table adds detail but is not load-bearing for the argument. ✅

### `tbl-stage-interface` (table 🟠) — def L882 ✅
- **Citation at def (L871):** "Each lifecycle stage operates as a distinct engineering phase with defined inputs, outputs, and quality invariants. Think of these as *API contracts*… @Tbl-stage-interface formalizes these contracts, making explicit what each stage must receive and produce. This specification transforms the abstract lifecycle diagram into actionable engineering requirements. When a stage's output fails to meet its contract, the deficiency propagates forward, compounding costs at each subsequent stage." The decision the table drives is stated — use these contracts to catch violations early.
- **Lead-out (L884):** Explains the iteration-cycle consequence of contract violations and introduces auditing stage transitions as the operational practice.
- **Removability test:** The contract concept, the propagation consequence, and the auditing practice are all in prose. The table supplies detail for practitioners. ✅
- **Note:** Referenced five times; every citation site carries substantive body-prose interpretation. Exemplary float integration.

# Float-explanation worklist (std) — conclusion.qmd (vol2)

Graded against FLOAT_EXPOSITION_STANDARD.md. Caption, fig-alt, in-figure labels, code comments,
and callout interiors do not count toward the prose's job. Only running body prose is judged.
Removability test applied to every float.

---

## Summary

| type      | level | floats | ✅ | ⚠️ | 🛑 |
|:----------|:------|-------:|---:|---:|---:|
| figure    | 🟠    |      1 |  1 |  0 |  0 |
| table     | 🟠    |      1 |  0 |  1 |  0 |
| listing   | 🟡    |      0 |  0 |  0 |  0 |
| algorithm | 🔴    |      0 |  0 |  0 |  0 |
| equation  | 🔴    |      0 |  0 |  0 |  0 |
| **total** |       |  **2** |**1**|**1**|**0**|

---

## Passing floats

**`fig-fleet-stack-conclusion`** (figure 🟠) — def L65

The reference at L63 delivers all three prose moves in body text.

- Lead-in (L61): establishes the fleet law, C³ taxonomy, and the shift from single-node to
  distributed realities — the reader knows why a layered architecture diagram is coming.
- Citation: "These principles form a layered architecture that mirrors the fleet stack synthesized
  in @fig-fleet-stack-conclusion."
- Lead-out (L63, same paragraph): names each layer with its governing principle — hardware physics
  sets hard ceilings at the foundation; communication dominates and failure is routine in the
  operational middle; responsibility and sustainability act as normative constraints at governance;
  scale creates qualitative change as the emergent sixth principle.

Removability test: delete the figure, and the prose still teaches the layered architecture concept
with full specificity. The figure serves as concrete reinforcement, not a substitute for
explanation. ✅

---

## Findings

### ⚠️ `tbl-vol2-principles` (table 🟠) — def L109

**Verbatim reference sentence (L98):**

> @Tbl-vol2-principles is therefore a decision map rather than a glossary: the six principles of
> distributed ML systems link each principle to the question, metric, and development context an
> engineer uses when diagnosing a fleet-scale design.

**What is missing:**

The Table standard requires the prose to deliver the conclusion the table encodes — the
load-bearing contrast, the specific row(s) that matter, or the decision the table drives
(H&P "the key result is…" move). The reference sentence correctly names the table's *function*
(decision map, not glossary) and *structure* (principle → question → metric → context), but it
does not state the takeaway from the cells. The non-obvious insight in the table is that
communication and failure are the two operational-layer principles that engineering interventions
most frequently need to address first — before governance and before scaling effects — yet that
ordering lives only in the cells and in the preceding body prose, never brought back into a
payoff sentence at the table's cite point.

**Where the takeaway currently lives:** The six principles are discussed individually in the
surrounding body prose (L158 onward), but those paragraphs are after the table, not at the cite
sentence. The cite sentence (L98) carries orientation only; the payoff paragraph after the table
(L158) starts with principle one without connecting back to what the table's structure reveals
as a whole.

**Removability test result:** Delete the table. The cite sentence still works as a transition
("a decision map rather than a glossary"), but the reader loses the mapping without ever having
been told which row(s) to consult first or what the table reveals that the prose paragraphs
alone did not. The prose is leaning on the cells to deliver the insight.

**Rule-compliant diff rewrite** (adds the missing Interpret move at L98; no em-dash/hyphen,
content leads, ref rides along, at most one colon per paragraph):

```diff
-@Tbl-vol2-principles is therefore a decision map rather than a glossary: the six principles of
-distributed ML systems\index{Six Principles of Distributed ML Systems} link each principle to the
-question, metric, and development context an engineer uses when diagnosing a fleet-scale design.
+The table that follows is a decision map rather than a glossary (@Tbl-vol2-principles). Each row
+names the principle, the diagnostic question that activates it, and the metric that makes it
+measurable. The two operational-layer principles (communication dominates and failure is routine)
+appear at the top of most diagnostic workflows because they bind first in synchronous training and
+large-scale inference; the governance-layer principles (responsibility constrains design,
+sustainability is a first-order cost) set the ceiling on what the system is allowed to do once
+the operational rates are established. Reading the table as a stack, not a flat list, is the
+practical lesson: the row that matters most is the one whose metric is currently the binding
+constraint.
```

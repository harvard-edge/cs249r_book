# Verified findings — sustainable_ai.qmd (vol2)
Prior findings: 2 | Survived: 0 | Refuted: 2

## SURVIVING findings

(none)

## REFUTED findings

- `fig-mckinsey-analysis` — REFUTED: explanation in ref sentence (L2830) and caption (L2832) and figure alt-text (L2832).

  Ref sentence: "a 2017 McKinsey projection (@fig-mckinsey-analysis) expected data center and edge inference markets to grow faster than training through 2025."

  Caption: "Inference workloads dominated the projected growth, with edge inference treated as a significant new segment while training markets grew more gradually in the projection."

  Alt-text (neighborhood): "Data center inference doubles from 4-5 to 9-10 billion dollars. Edge inference grows from near zero to 4-4.5 billion dollars. Training markets grow more slowly."

  The ref sentence names what the figure shows (inference markets growing faster than training). The caption restates it with more specificity (inference dominated, edge was a new segment). The alt-text, which is part of the float definition and thus neighborhood context, supplies the specific dollar figures the prior worklist said were absent from prose. The prior worklist's issue was that prose "immediately dismisses the figure" and does not extract specific values, but the alt-text is a neighborhood element that carries those numbers. Per the refutation bar, the combination of ref sentence, caption, and alt-text tells the reader what the float shows and why it matters. No true dead-end exists.

- `@tbl-prefill-decode` (dangling ref flag) — REFUTED: valid cross-chapter reference. The table `tbl-prefill-decode` is defined in vol2/inference. Cross-chapter references need not re-explain the float in the citing chapter. The 🛑 "Missing definition" finding is discarded entirely per the instructions: the ref is not broken, it points to a table in another chapter.

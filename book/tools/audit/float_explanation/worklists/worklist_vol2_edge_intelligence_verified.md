# Verified findings — edge_intelligence.qmd (vol2)
Prior findings: 1 | Survived: 0 | Refuted: 1

## SURVIVING findings

(none)

## REFUTED findings

- `fig-fl-communication-computation` — REFUTED: explanation in caption (L2942): "As network bandwidth decreases (Fast to Slow), the optimal number of local epochs shifts rightward to amortize the high cost of communication over more computation. However, excessive local computation eventually increases total time due to model drift (requiring more global rounds to converge)."

  The caption delivers both the "what" (optimal E shifts rightward on slower networks) and the "why it matters" (the U-curve: too few epochs wastes bandwidth, too many epochs triggers model drift). The prior worklist credited the caption ✓ and the refutation bar requires only that ANY neighborhood element carry the takeaway. The ref sentence at L2940 is an announcer-shape pivot, but that shape is permissible when the caption carries the substance. No true dead-end exists here.

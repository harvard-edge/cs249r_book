# Verified findings — benchmarking.qmd (vol1)
Prior findings: 3 | Survived: 0 | Refuted: 3

## SURVIVING findings

_(none)_

## REFUTED findings

- `tbl-benchmarking-vendor-claims` — REFUTED: explanation is in the callout frame (L515-517) + caption. The ref lives inside a `.callout-checkpoint` block whose opening sentence (L517) already states the evaluative purpose: "check whether the claim identifies the workload, measurement boundary, and operating conditions." The prior paragraph (L513) supplies the systemic cause ("The peak-vs.-sustained gap is structurally guaranteed by the memory wall"). The table's four rows are a lookup operationalizing the checklist, not free-floating content. Callout-resident tables carry their explanatory context from the callout itself; the ref sentence does not need to repeat it.

- `tbl-edge-vs-cloud-constraints` — REFUTED: explanation is in caption (strong) + payoff ¶ L1795. Caption states the inversion explicitly: "Cloud systems treat power as an operational cost and latency as a UX metric, leaving accuracy as the primary optimization target; edge systems must treat power and latency as hard physical limits, leaving accuracy as the residual variable to optimize." Payoff ¶ (L1795) quantifies the constraint concretely (30 fps deadline, <1 W limit, MobileNetV3 vs. ResNet-50 side-by-side) and closes with the systemic reason: "That thermal mechanism, not measurement sloppiness, makes edge benchmarking a categorically different exercise than cloud benchmarking." Both caption and payoff independently satisfy the refutation bar.

- `tbl-benchmarking-edgetpu-validation` — REFUTED: explanation is in payoff ¶ L2931 ("**What this reveals**") + caption. The "**What this reveals**" paragraph immediately after the table states: "The [X] inference speedup is real, but end-to-end improvement is only ~[Y] because preprocessing (image capture, resize, normalize) runs on the CPU in both cases." This is the exact finding the first pass said was absent. Caption also states "showing how preprocessing overhead narrows the headline accelerator speedup." The ref sentence at L2920 is mechanical, but the finding is thoroughly delivered by the adjacent paragraph, which is the strongest possible form of payoff.

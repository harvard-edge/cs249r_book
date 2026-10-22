# How does batching reuse the same weights?

Keep weights resident while distinct input rows update output accumulators and commit their own Z rows.

Scope: Two examples, two inputs and outputs, active two-by-two corner of source four-by-four PE array; no bias or activation.

1. Inspect one dot product.
2. Show X/W/Z as the same batched operation.
3. Stage one tile and stream the first row through reduction.
4. Commit row one, stream row two with unchanged weights, then commit row two.

Guard: Conceptual panel arrows are not physical copies. Tile-value/MAC counts are not measured bandwidth or performance.

Run `python3 books/_prototypes/vol1-web-animations/original-v1-neuron-to-machine/check.py`. The fixture uses the full owning chapter from the canonical art index. It tests exact endpoints, repeatable menu jumps, keyboard, real-motion freeze/resume in Expand, reduced motion, overflow, and print player hiding in Chromium/WebKit, desktop/phone, light/dark. Production ordinary visible playback and complete static fallback require the root integration build. Source PNGs: `/tmp/v1-neuron-to-machine.png`. QA PNGs: `/tmp/original-v1-neuron-to-machine-review`. No real student study is claimed.

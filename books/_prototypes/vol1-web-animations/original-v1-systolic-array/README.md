# How does a skewed wavefront reuse neighboring values?

A weight-stationary two-row, two-column slice is overlaid inside the original array. Inputs move horizontally, partial sums vertically, and completed reductions drain to bottom accumulators. An explicit one-clock row skew aligns each incoming input with the matching partial sum. The second clock advances multiple independent cells of one wavefront. Phone inspection crops to the actual slice, with a Whole original toggle.

## Scope and invariant

Declared weight-stationary 2×2 slice inside the original schematic: row inputs 2 and 3, every PE weight 1, two-term column reductions. Horizontal values and vertical partial sums advance on explicit skewed clocks. Other source FIFOs/accumulators illustrate the surrounding array and are outside this slice. No full-array utilization or DRAM-saving model.

Clock-skewed weight-stationary two-row two-column slice; partial sums retained to matching reduction arrival; no premature output, no DRAM or throughput multiplier.

## Review fixture

Run from the repository root:

```sh
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-systolic-array/check.py
```

The check builds the complete owning-chapter QA fixture through `build_original_figure_fixture.py`, preserving prose, source figure, caption, references, navigation, and actual theme assets. It exercises every endpoint on desktop and phone in light and dark, sequence jumps, keyboard stepping/restart, reduced motion, paused motion and resume, Expand/Escape, Diagram switching, print visibility, and a JavaScript-disabled static view. Endpoint guards check the mechanism-specific retention/conservation rules. Screenshots stay in `/tmp/original-v1-systolic-array-review/`; fixture timing comes from shared YAML and the lesson has no local timing defaults.

Normal-motion control checks use the expanded viewport after endpoint screenshots, preventing offscreen IntersectionObserver pauses from being mistaken for playback failures. This local fixture establishes neither production enrollment nor PDF/EPUB build correctness. Those remain root integration checks. No student testing is claimed.

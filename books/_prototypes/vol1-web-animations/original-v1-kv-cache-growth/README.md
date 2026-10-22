# How do active requests compete with context length?

Every original canonical curve, axis, legend, and capacity annotation remains visible. Exact source-calculator inputs drive a marker, resident-budget fill, and four groups of eight independent request histories. Context growth appends storage; release removes completed histories; an over-budget extension remains an unallocated ghost. The plot and capacity bar use the same retained state. Groups represent separate allocations rather than shared payload.

## Scope and invariant

Canonical 70B-class GQA model, FP16 KV, INT4 weights and the original H100 budget. A uniform-context batch is used for clarity. Capacity excludes runtime workspace, as in the source. Markers use the original curves and axes; motion is illustrative, not latency.

Exact canonical model budget. Rejecting a target never commits over-budget records. Markers and capacity bar show same retained state; no runtime workspace, token eviction, or timing model.

Canonical inputs: `PYTHONPATH=mlsysim /tmp/venv/bin/python books/_extensions/mlsysbook-ext/figure-explore/original-v1-kv-cache-growth.calc.py`.

## Review fixture

Run from the repository root:

```sh
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-kv-cache-growth/check.py
```

The check builds the complete owning-chapter QA fixture through `build_original_figure_fixture.py`, preserving prose, source figure, caption, references, navigation, and actual theme assets. It exercises every endpoint on desktop and phone in light and dark, sequence jumps, keyboard stepping/restart, reduced motion, paused motion and resume, Expand/Escape, Diagram switching, print visibility, and a JavaScript-disabled static view. Endpoint guards check the mechanism-specific retention/conservation rules. Screenshots stay in `/tmp/original-v1-kv-cache-growth-review/`; fixture timing comes from shared YAML and the lesson has no local timing defaults.

Normal-motion control checks use the expanded viewport after endpoint screenshots, preventing offscreen IntersectionObserver pauses from being mistaken for playback failures. This local fixture establishes neither production enrollment nor PDF/EPUB build correctness. Those remain root integration checks. No student testing is claimed.

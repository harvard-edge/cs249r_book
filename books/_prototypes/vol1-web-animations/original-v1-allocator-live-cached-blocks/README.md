# Why can six free units fail a six-unit request?

The three original snapshots become one fixed-address temporal pool. Releasing A or B reveals cached storage in place; D fills only the first half of B’s cached block. A six-unit dashed footprint crosses the live D block and fails despite six reusable units. Reservation, live bytes, reusable bytes, and largest gap derive from the same block state.

## Scope and invariant

Original fixed virtual-address pool, expressed in the same twelve illustrative units. No growth, compaction, remapping, or pending device consumers. Snapshot (c) is split into two causal actions.

Reserved = live + reusable; request accepted only if largest contiguous free block fits.

## Review fixture

Run from the repository root:

```sh
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-allocator-live-cached-blocks/check.py
```

The check builds the complete owning-chapter QA fixture through `build_original_figure_fixture.py`, preserving prose, source figure, caption, references, navigation, and actual theme assets. It exercises every endpoint on desktop and phone in light and dark, sequence jumps, keyboard stepping/restart, reduced motion, paused motion and resume, Expand/Escape, Diagram switching, print visibility, and a JavaScript-disabled static view. Endpoint guards check the mechanism-specific retention/conservation rules. Screenshots stay in `/tmp/original-v1-allocator-live-cached-blocks-review/`; fixture timing comes from shared YAML and the lesson has no local timing defaults.

Normal-motion control checks use the expanded viewport after endpoint screenshots, preventing offscreen IntersectionObserver pauses from being mistaken for playback failures. This local fixture establishes neither production enrollment nor PDF/EPUB build correctness. Those remain root integration checks. No student testing is claimed.

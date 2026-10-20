# Which intermediates cross into HBM?

Both original forward paths remain the main drawing. Dynamic labels identify actual HBM intermediates, replaceable SRAM tiles, retained online-softmax state, and completed output. The declared toy row computes a changed running maximum and rescales the old normalizer/numerator before adding the second block. Both alternatives produce the same computed output. Phone inspection follows the active original panel, with a Both original paths toggle.

## Scope and invariant

Same two forward paths and memory boundary as the original. A declared one-query/four-key, two-block toy row (scores 0, 0, ln 2, ln 2; values 2, 4, 6, 8) keeps running row statistics and partial output; the original matrix boxes stand for the full intermediates of this row trace, not a quadratic-capacity measurement; all backward/optimizer memory is omitted. Tiled execution is a separate alternative to the materialized path.

Forward lifetime only; neither path represents total training memory. Online-softmax statistics survive tile replacement and output normalizes only after final tile.

## Review fixture

Run from the repository root:

```sh
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-training-attention-storage/check.py
```

The check builds the complete owning-chapter QA fixture through `build_original_figure_fixture.py`, preserving prose, source figure, caption, references, navigation, and actual theme assets. It exercises every endpoint on desktop and phone in light and dark, sequence jumps, keyboard stepping/restart, reduced motion, paused motion and resume, Expand/Escape, Diagram switching, print visibility, and a JavaScript-disabled static view. Endpoint guards check the mechanism-specific retention/conservation rules. Screenshots stay in `/tmp/original-v1-training-attention-storage-review/`; fixture timing comes from shared YAML and the lesson has no local timing defaults.

Normal-motion control checks use the expanded viewport after endpoint screenshots, preventing offscreen IntersectionObserver pauses from being mistaken for playback failures. This local fixture establishes neither production enrollment nor PDF/EPUB build correctness. Those remain root integration checks. No student testing is claimed.

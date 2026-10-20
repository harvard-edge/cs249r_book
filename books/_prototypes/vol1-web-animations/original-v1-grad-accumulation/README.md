# What survives between micro-batches?

The original horizontal three-row drawing is the desktop stage. A narrow-screen adaptation keeps the same three loss/gradient branches, common accumulator, and downstream weights. Saved activations appear for one micro-batch and disappear after backward; gradient contributions persist. One optimizer update follows all three contributions. Clearing the accumulator preserves the historical completed rows.

## Scope and invariant

Same three micro-batches as the original. Equal-size micro-batches; each gradient is from a sum-reduced loss, so their sum gives the full-batch sum gradient. No parameter update between micro-batches. Saved-activation labels indicate schematic storage, not a complete memory budget.

Sum-reduced equal micro-batches; no claim that batch-dependent layers, dropout, or arithmetic order are identical to a large batch.

## Review fixture

Run from the repository root:

```sh
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-grad-accumulation/check.py
```

The check builds the complete owning-chapter QA fixture through `build_original_figure_fixture.py`, preserving prose, source figure, caption, references, navigation, and actual theme assets. It exercises every endpoint on desktop and phone in light and dark, sequence jumps, keyboard stepping/restart, reduced motion, paused motion and resume, Expand/Escape, Diagram switching, print visibility, and a JavaScript-disabled static view. Endpoint guards check the mechanism-specific retention/conservation rules. Screenshots stay in `/tmp/original-v1-grad-accumulation-review/`; fixture timing comes from shared YAML and the lesson has no local timing defaults.

Normal-motion control checks use the expanded viewport after endpoint screenshots, preventing offscreen IntersectionObserver pauses from being mistaken for playback failures. This local fixture establishes neither production enrollment nor PDF/EPUB build correctness. Those remain root integration checks. No student testing is claimed.

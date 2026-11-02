# Where does a discarded image stop costing work?

The original four-stage data-path PNG anchors the desktop stage; a vertical adaptation preserves the same storage/host/PCIe/accelerator chain on narrow screens. Actual A/C packet copies follow its edges. In the default stream alternative, all A/B/C/D are read, decoded, and copied through PCIe before B/D branch to rejection at accelerator admission. Decode, transfer, rejection, and training arithmetic are separate actions. Source physical annotations remain the full-epoch reference, while record identities are an explicitly scoped toy trace.

## Scope and invariant

Same four stages and source full-epoch reference. A four-record toy trace retains A and C. The materialized coreset exists before the epoch; its construction cost is outside this replay. The default stream alternative follows the figure caption: all A–D are read, decoded, and copied before B/D are rejected from accelerator training arithmetic. Earlier metadata filtering is a separate possible design. No wall-clock speedup is inferred.

Follow original caption default: stream filtering saves training arithmetic only; full read/decode/transfer costs already occurred. Materialization construction/scoring excluded from replay, never claimed free.

## Review fixture

Run from the repository root:

```sh
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-data-selection-resnet-data-path/check.py
```

The check builds the complete owning-chapter QA fixture through `build_original_figure_fixture.py`, preserving prose, source figure, caption, references, navigation, and actual theme assets. It exercises every endpoint on desktop and phone in light and dark, sequence jumps, keyboard stepping/restart, reduced motion, paused motion and resume, Expand/Escape, Diagram switching, print visibility, and a JavaScript-disabled static view. Endpoint guards check the mechanism-specific retention/conservation rules. Screenshots stay in `/tmp/original-v1-data-selection-resnet-data-path-review/`; fixture timing comes from shared YAML and the lesson has no local timing defaults.

Normal-motion control checks use the expanded viewport after endpoint screenshots, preventing offscreen IntersectionObserver pauses from being mistaken for playback failures. This local fixture establishes neither production enrollment nor PDF/EPUB build correctness. Those remain root integration checks. No student testing is claimed.

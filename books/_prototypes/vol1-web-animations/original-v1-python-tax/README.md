# What keeps the next kernel from starting?

The original canonical timeline is retained. Revealed dispatch/compute intervals are synchronized with an actual host/submission/device/storage flow. Eager operations read and write intermediate x identities through global storage. The fused alternative keeps internal values on-chip and writes final x5 once in the same kernel action. The phone arrangement stacks the same resources vertically. No universal speedup is inferred from the scenario.

## Scope and invariant

The original five-operation timeline and its canonical Ops dispatch/launch values. One-microsecond kernels and the fused efficiency are the chapter’s declared scenario, not measurements or general compilation gains. Eager and compiled/fused rows are alternatives; the source chart omits per-kernel driver-launch cost on the eager row.

Exact source plot preserved; mask reveals completed dispatch/compute intervals, source scenario only. Eager intermediate global traffic and fused on-chip retention distinguished; no arbitrary fusion speed claims. The fused region computes and writes final output in one action; no duplicate output step.

Canonical inputs: `PYTHONPATH=mlsysim /tmp/venv/bin/python books/_extensions/mlsysbook-ext/figure-explore/original-v1-python-tax.calc.py`.

## Review fixture

Run from the repository root:

```sh
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-v1-python-tax/check.py
```

The check builds the complete owning-chapter QA fixture through `build_original_figure_fixture.py`, preserving prose, source figure, caption, references, navigation, and actual theme assets. It exercises every endpoint on desktop and phone in light and dark, sequence jumps, keyboard stepping/restart, reduced motion, paused motion and resume, Expand/Escape, Diagram switching, print visibility, and a JavaScript-disabled static view. Endpoint guards check the mechanism-specific retention/conservation rules. Screenshots stay in `/tmp/original-v1-python-tax-review/`; fixture timing comes from shared YAML and the lesson has no local timing defaults.

Normal-motion control checks use the expanded viewport after endpoint screenshots, preventing offscreen IntersectionObserver pauses from being mistaken for playback failures. This local fixture establishes neither production enrollment nor PDF/EPUB build correctness. Those remain root integration checks. No student testing is claimed.

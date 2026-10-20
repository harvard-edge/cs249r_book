# How do sparse values find their matching activations?

Follow retained weight identity and its original activation position through packing, selection, and accumulation.

Scope: Same valid already-pruned eight-element row in dense and sparse paths. Metadata positions are local to each group.

1. Compute the dense reference for the same row.
2. Pack retained values and group-relative tags.
3. Use metadata to select matching dense activations.
4. Retain partial sums until all nonzero products arrive; compare equal outputs.

Guard: No pruning-quality or end-to-end-speedup claim. Dense reads preserve source activations; metadata adds traffic.

Run `python3 books/_prototypes/vol1-web-animations/original-v1-2-4-gemm/check.py`. The fixture uses the full owning chapter from the canonical art index. It tests exact endpoints, repeatable menu jumps, keyboard, real-motion freeze/resume in Expand, reduced motion, overflow, and print player hiding in Chromium/WebKit, desktop/phone, light/dark. Production ordinary visible playback and complete static fallback require the root integration build. Source PNGs: `/tmp/v1-2-4-gemm.png`. QA PNGs: `/tmp/original-v1-2-4-gemm-review`. No real student study is claimed.

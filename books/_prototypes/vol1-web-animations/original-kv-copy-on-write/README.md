# Original paged-KV copy-on-write review

The `original-kv-copy-on-write` factory clones the actual verified Volume III
SVG for `fig-vol3-pagedattention-mapping`. It retains both panels: the logical
row, software block table, scattered physical block pool, and the A-above/B-below
fork with full prefix blocks visible.

The sequence locates position 9 through logical block 2, physical block 9, and
slot 1; forks B by sharing table references; reserves block 14; copies the two
initialized tail records separately; remaps only B; then writes newly computed
KV for position 10 through a separate payload inlet. Metadata movement is purple
and explicitly labeled by ID; KV payload movement is green. Digits in the
original slots name token positions, not scalar key/value contents.

A remains unchanged. Allocation ownership holds block 14 even while it has no
table references. Copying leaves both candidates pointing to block 9 until the
replacement is ready. Empty green-outlined slots are allocated but unwritten;
the unallocated block 14 has a gray dashed outline. No concurrent consumers,
eviction, hardware page table, byte address calculation, or measured execution
time is modeled.

Narrow screens keep the original full-size artwork in a local scroll surface.
The camera follows the active reference or tail, and A/B labels within the table
rows remain visible in the crop. The final crop includes the table, tail, and
complete new-KV inlet. The shared player owns playback, timing, keyboard
shortcuts, reduced motion, and expanded inspection.

From the worktree root:

```sh
python3 books/_prototypes/vol1-web-animations/original-kv-copy-on-write/build-preview.py
python3 books/_prototypes/vol1-web-animations/original-kv-copy-on-write/check-preview.py --engine chromium --screenshots /tmp/original-kv-copy-on-write-chromium
python3 books/_prototypes/vol1-web-animations/original-kv-copy-on-write/check-preview.py --engine webkit --screenshots /tmp/original-kv-copy-on-write-webkit
```

The builder uses `/tmp/figures-animation-art-current/index.json`, copies an
existing complete chapter build into an ignored local output, and mounts the
lesson there for review. It does not edit chapter sources, captions, enrollment,
the registry, or the publication build. Its default URL expects the worktree’s
`books` directory at `http://127.0.0.1:8882`.

Browser checks cover every endpoint, candidate references, immutable A and full
prefix blocks, allocation/copy/remap/append ordering, interrupted transient
cleanup, frozen motion on pause, keyboard, reduced motion, expanded state and
focus restoration, mobile fit, no chapter overflow, static print, and no-JavaScript
fallback. Browser print does not establish a successful Quarto PDF/EPUB build.
The review uses agent feedback and visual inspection, not a student study.

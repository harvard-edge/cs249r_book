# Original context-prefix invalidation review

The `original-context-invalidation` factory enhances the canonical Volume III
`fig-vol3-context-zones` SVG. It retains the three consecutive request bars,
root/trunk/leaf layout, r17/r18 labels, first-change boundaries, output reserve,
and source legend.

Selected token chips make the dependency visible: the trunk starts with `u`,
`u`, then `v`; the leaf starts with `a`, `b`, then `c`; the selected trailing token
`T` stays identical at the same position. Other serialized tokens are elided.
The letter chips are illustrative token identities, not a tokenizer trace.

Thin KV bands use existing space beneath each input bar. C0/C1/C2 identify the
state derived for T under each request prefix, not numerical K/V values. Reused
prefix bindings precede recomputation of the changed suffix. The third request
first recomputes its trunk, then processes the leaf and unchanged T. Prior
request rows remain historical records; the animation does not model physical
cache erasure, allocation, or cleanup. The original input-limit label moves
slightly downward to clear the added band. The output reserve is unchanged.

The example assumes exact serialized-prefix equality and compatible resident
cache under matching model, position, and attention configuration. Real block
alignment and serving policy can shorten reuse. Stable zone placement never
justifies stale source content.

The shared original player owns timing, controls, keyboard shortcuts, reduced
motion, and expanded inspection. Narrow screens provide a labeled local scroll
surface that follows the active cutoff or suffix and keeps T beside its derived
state. The full original geometry remains available through panning and Expand.

From the worktree root:

```sh
python3 books/_prototypes/vol1-web-animations/original-context-invalidation/build-preview.py
python3 books/_prototypes/vol1-web-animations/original-context-invalidation/check-preview.py --engine chromium --screenshots /tmp/original-context-chromium
python3 books/_prototypes/vol1-web-animations/original-context-invalidation/check-preview.py --engine webkit --screenshots /tmp/original-context-webkit
```

The builder reads the local canonical-art index, copies an existing complete
chapter site into ignored build output, and mounts this lesson in that chapter.
It does not edit source prose, captions, enrollment, registry, or compiler. The
default browser-test URL assumes the worktree's `books` directory is served at
`http://127.0.0.1:8882`.

Checks independently validate every endpoint, immutable token identities,
prefix-only reuse, pending suffix state, historical provenance, untouched output
reserve, pause/resume, transient cleanup, keyboard, reduced motion, expanded
state/focus restoration, phone fit, static browser print, and no-JavaScript
fallback. Paused destination narration remains conditional until processing or
binding finishes; C1/C2 cannot appear in the KV bands before endpoint commit.
Browser print does not establish a Quarto PDF/EPUB build. Independent
agent reviews and visual inspection are not a student study.

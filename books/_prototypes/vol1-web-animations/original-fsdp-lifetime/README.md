# Original FSDP lifetime grid

The `original-fsdp-lifetime` factory enhances the verified SVG for
`vol2:fig-fsdp-parameter-lifetime` without replacing its six phase columns,
allocation rows, persistent shards, or dashed optional prefetch cells.

Eight paused states establish persistent ownership, gather forward parameters,
produce saved activations, release gathered parameters, gather backward
parameters, produce a full gradient, release consumed backward state, and
ReduceScatter the gradient to its owned shard. The last two actions share the
original final phase column. Its existing gradient label temporarily reads
`full` after backward state is released, then changes to `shard` after reduction.
The final shard is ready for the local optimizer update; the walkthrough does
not claim to execute an unshown numerical optimizer update.

A compact `Live now` view sits beside the original history on desktop and above
it on phones. Its fixed rows use the source object's identity and semantic color.
Stable object nodes appear on allocation, remain while needed, and disappear
on release. In particular, saved activations remain through the forward weight
release; the full gradient remains through backward cleanup and then becomes
the reduced shard. Empty slots do not imply byte sizes or memory addresses.

Rows preserve object identity. Columns are execution phases, not physical
memory locations or proportional durations. Existing source cell groups change
visibility as their phase becomes current. Past cells remain dim history, and
future cells remain pale previews. Persistent parameter and optimizer shard
rows remain present. Prefetch retains its original dashed optional cells but
is not executed by this walkthrough. No byte counts or speedups are inferred.

On narrow screens the source row names remain readable during local scrolling.
The current phase is centered at step boundaries and after expansion/resizing.
Manual scrolling is retained between those boundaries. Shared original-player
controls supply keyboard navigation, pause/resume, reduced motion, and expanded
inspection. Timing is required from the build; there are no local defaults.

## Focused review

```sh
python3 books/_prototypes/vol1-web-animations/original-fsdp-lifetime/check-browser.py
python3 books/_prototypes/vol1-web-animations/original-fsdp-lifetime/check-browser.py --engine webkit
```

The fixture mounts the factory inside the actual rendered distributed-training
chapter, with its theme and surrounding material. It checks every declared
allocation state in both history and the live view, stable current object appearance/release, persistent ownership, optional-prefetch retention, exact
jumps, keyboard stepping, frozen animation while paused, resumption, playback,
expanded state, mobile phase visibility, local manual scrolling, and no added
chapter overflow. Fixture pacing is explicitly accelerated test input.
Screenshots are written to `/tmp/original-fsdp-review`.

This focused fixture does not replace a production Quarto HTML/PDF/EPUB build
or the final packaged-chapter visual review.

# Original-art ring AllReduce review

The `original-ring-allreduce` factory enhances the verified SVG for
`vol2:fig-ring-allreduce`. It uses the chapter's four GPU cards, four horizontal
chunk tiles per card, two phase panels, and wrap-around links. It does not fetch
or substitute a different drawing.

Seven endpoints establish local contributions, complete three ReduceScatter
rounds, and complete three AllGather rounds. Chunk A has a persistent provenance
explanation. Every link carries its scheduled chunk on each round. The source
chunk rectangles and labels update at endpoints; temporary packets follow the
source links during movement.

Contribution identities replace arbitrary numeric examples. Reduction combines
disjoint rank sets using only the previous round's snapshot. AllGather begins
with one complete shard per rank and copies complete sets without addition.
Partially reduced scratch cells in the upper panel are not AllGather inputs.
The original miniature link tiles are revealed as their rounds complete; they
record round history rather than extra simultaneous packets.

The shared original player supplies timing, step controls, keyboard hints,
pause/resume, expansion, reduced motion, and focus behavior. Timing must be
provided by the build. On narrow displays the original geometry scrolls inside
the figure; the chapter itself does not acquire new overflow.

## Focused browser check

With the local artwork index and rendered Volume 2 chapter available:

```sh
python3 books/_prototypes/vol1-web-animations/original-ring-allreduce/check-browser.py
python3 books/_prototypes/vol1-web-animations/original-ring-allreduce/check-browser.py --engine webkit
```

The check mounts the factory inside the actual rendered collective-communication
chapter, retaining its CSS and surrounding material. It checks all contribution
endpoints, manual stopping, Back, direct menu selection, restart shortcut,
real movement, frozen coordinates during pause, resume, playback completion,
expansion/Escape with retained state, and local scrolling. Screenshots are
written to `/tmp/original-ring-review` for desktop and phone in both themes.
The short timing values in this test are explicit fixture inputs, not runtime
defaults or production pacing.

This focused harness is not a full Quarto HTML/PDF/EPUB acceptance build. Source
integration and format fallbacks belong to the extension/runtime checks.

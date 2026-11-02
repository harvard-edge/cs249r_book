# Original-artwork stream capture review

`original-stream-ingestion.js` animates the canonical Volume III figure
`fig-vol3-stream-ingestion`. It clones the runtime's verified artwork, retains
the child/adapter, stream strip, four-cell head, eight-sector tail ring,
observation fields, and optional authorized-record path, and adds synchronized
ASCII bytes to those objects.

The example reads `A` through `P` in one illustrative observed readiness order.
The first four bytes remain in the head. The tail stores the most recent eight
later bytes using cyclic physical slots. Further bytes replace the oldest tail
entry and increase the computed omission count. Serialization reads the tail
in chronological order. The example authorizes no full record; notice/status
framing, pipe buffers, and the next call's token budget are separate.

The shared original player owns playback, keyboard operation, timing, reduced
motion, and expanded inspection. On narrow screens the original diagram has a
local scroll surface, a visible swipe cue, and camera movement synchronized with
the byte packet. The readable state summary supplements the original diagram.

Run from the worktree root:

```sh
python3 books/_prototypes/vol1-web-animations/original-stream-ingestion/build-preview.py
python3 books/_prototypes/vol1-web-animations/original-stream-ingestion/check-preview.py
python3 books/_prototypes/vol1-web-animations/original-stream-ingestion/check-preview.py --engine webkit --screenshots /tmp/original-stream-ingestion-webkit
```

The preview builder reads the local canonical-art index at
`/tmp/figures-animation-art-current/index.json`, copies the existing complete
chapter site into an ignored build directory, and inserts the lesson there.
It does not enroll the manuscript or alter the original figure/caption. The
default browser-test URL assumes the worktree's `books` directory is served at
`http://127.0.0.1:8882`.

Checks independently derive all capture states, physical ring slots,
chronological tail order, bounded occupancy, omission accounting, endpoint-only
writes, and serialization. They exercise pause/resume, Back/Restart/keyboard,
reduced motion, same-state expanded inspection, focus restoration, print-static
fallback, and narrow-screen panning without chapter overflow. Browser print is
not a Quarto PDF/EPUB build. These checks do not establish student learning gains.

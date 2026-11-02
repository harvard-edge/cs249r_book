# Which ready batch keeps the training stage busy?

Show ready-batch handoff and overlapping independent host/GPU work in the actual Gantt figure.

Scope: Canonical illustrative minute axis and exact source stage durations; readiness and consumption are distinct.

1. Complete the open before its reads.
2. Make B1 ready before Train 1 consumes it.
3. Prepare B2 while B1 trains; retain B2 until Train 2 is ready.
4. Prepare B3 while B2 trains, then consume and finish B3.

Guard: No same-batch read/train dependency is violated. Bars retain canonical durations; movement pace is pedagogical, not measured.

Run `python3 books/_prototypes/vol1-web-animations/original-v1-fetching-optimized/check.py`. The fixture uses the full owning chapter from the canonical art index. It tests exact endpoints, repeatable menu jumps, keyboard, real-motion freeze/resume in Expand, reduced motion, overflow, and print player hiding in Chromium/WebKit, desktop/phone, light/dark. Production ordinary visible playback and complete static fallback require the root integration build. Source PNGs: `/tmp/v1-fetching-optimized.png`. QA PNGs: `/tmp/original-v1-fetching-optimized-review`. No real student study is claimed.

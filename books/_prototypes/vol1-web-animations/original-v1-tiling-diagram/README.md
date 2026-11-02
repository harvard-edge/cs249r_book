# What stays resident while the K tiles change?

Retain the same C partial while matching A/B tiles advance, then write and release in dependency order.

Scope: A selected 2 × 2 output tile reduces K = 6 through three K tiles of width 2. The source layout includes a resident accumulator inset.

1. Stage matching A/B tiles for the current K interval.
2. Add each reduction term to the same resident C partial.
3. Retire operands while preserving C across three slabs.
4. Write the complete output before releasing its accumulators.

Guard: Fast accumulator state is distinct from output memory. Counts are toy logical work, not measured traffic or speed.

Run `python3 books/_prototypes/vol1-web-animations/original-v1-tiling-diagram/check.py`. The fixture uses the full owning chapter from the canonical art index. It tests exact endpoints, repeatable menu jumps, keyboard, real-motion freeze/resume in Expand, reduced motion, overflow, and print player hiding in Chromium/WebKit, desktop/phone, light/dark. Production ordinary visible playback and complete static fallback require the root integration build. Source PNGs: `/tmp/v1-tiling-diagram.png`. QA PNGs: `/tmp/original-v1-tiling-diagram-review`. No real student study is claimed.

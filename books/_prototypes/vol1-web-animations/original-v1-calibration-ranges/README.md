# How do range and zero-point change the stored value?

Observe the same real value clipping, encoding, and reconstructing differently in the original two mappings.

Scope: Original signed INT8 endpoints; same inputs and declared affine rounding/clipping policy.

1. Present real zero and trace both zero-point mappings.
2. Compare an interior input and an input clipped only by the symmetric range.
3. Compare saturation in both ranges, then reconstruct stored codes.

Guard: Asymmetric rounded zero-point shifts representable endpoints slightly; no exact endpoint or accuracy promise.

Run `python3 books/_prototypes/vol1-web-animations/original-v1-calibration-ranges/check.py`. The fixture uses the full owning chapter from the canonical art index. It tests exact endpoints, repeatable menu jumps, keyboard, real-motion freeze/resume in Expand, reduced motion, overflow, and print player hiding in Chromium/WebKit, desktop/phone, light/dark. Production ordinary visible playback and complete static fallback require the root integration build. Source PNGs: `/tmp/v1-calibration-ranges.png`. QA PNGs: `/tmp/original-v1-calibration-ranges-review`. No real student study is claimed.

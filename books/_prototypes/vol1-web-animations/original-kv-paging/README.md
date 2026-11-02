# Original-art KV paging

This factory animates `fig-serving-paged-kv-mapping` in Volume 1's serving chapter. It requires verified source SVG and the shared original player; production enrollment belongs to the registry/manifest integration owner.

The source starts with E already allocated, with ordered pages mapped to physical blocks 0, 1, 4, and 5. C and D retain blocks 2–3 and 6–7. Each page has separate lookup, resolution, and nondestructive read actions. The last consumer finishes before a separate release action. The lower source panel is an after-release reference until that endpoint; it is not another initial allocation. Two records per block is a visible toy assumption. Letters name K/V records, not scalar values or sizes.

The original geometry, logical/table connections, pool labels, owner identity, and hatched unused half of E's last block remain. On release, E's table mapping becomes historical, the upper blocks become free, and the tail annotation changes. A mobile follow view crops around the active logical page and physical block, with a whole-diagram option and shared Expand control. Movement duration is exclusively supplied by the shared build configuration.

Run from the repository root:

```sh
python3 books/_prototypes/vol1-web-animations/original-kv-paging/check.py
```

The focused test uses a previously rendered complete owning chapter from the canonical art index, injects only this factory and the shared player, and leaves manuscript source untouched. It checks all fifteen endpoint layouts, mapping/ownership invariants, keyboard stepping, real-motion pause/resume in the supported expanded view, reduced motion, expansion, print hiding, and overflow in Chromium and WebKit at desktop/phone widths in light/dark. The fixture suppresses the old chapter's release-manifest network lookup; it is unrelated to the lesson. Production build/static-format acceptance is separate. PNGs default to `/tmp/original-kv-paging-review`.

The older chapter fixture clips the ordinary phone player for IntersectionObserver after repeated full-component screenshots and seeks. The guard correctly pauses playback; real-motion pause/resume is therefore exercised in Expand rather than disabling the guard. Ordinary visible playback and the canonical static fallback must also be checked in the new production chapter build.

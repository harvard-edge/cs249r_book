# Original capture-buffer figure review

The factory `original-capture-buffer` enhances the verified SVG for
`vol4:fig-08-shared-capture-buffer`. It retains the actual three-panel drawing,
adds frame identity and reference counts inside its existing boxes, and uses
shared `MLSysOriginalPlayer` controls and build-time timing.

Panel a and panel b are alternatives, not consecutive stages. Copy operations
leave the source payload in place. Three admitted consumers alias physical A;
conversion explicitly writes separate B for the same frame F. CPU, ISP, and GPU
release their illustrative references individually. Only the final release makes
A eligible; the next step explicitly requeues it. B retains F independently.
Nothing models bandwidth, latency, device overlap, or a later capture.

Mobile inspection follows the active region of the original drawing. Expand
provides more space; the final `Inspect retained B` button changes only the
viewport, never ownership state or the sequence position.

Run from the repository root:

```sh
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-capture-buffer/check.py
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-capture-buffer/check.py --engine webkit --screenshots /tmp/original-capture-buffer-webkit
```

The check constructs a disposable fixture from the complete existing Volume IV
chapter recorded in `/tmp/figures-animation-art-current/index.json`, preserving
its prose, caption, navigation and theme. Explicit short timing is fixture-only;
the factory has no timing defaults. It checks all ownership/identity states,
copy-source persistence, exact sequence jumps, paused motion, keyboard control,
reduced motion, expansion/Escape, print/static visibility, and chapter overflow
on desktop/mobile in both themes.

This fixture does not establish production enrollment or PDF/EPUB correctness.
Those require the normal source integration and full target builds. Its print
rule only simulates the production runtime's retained static view.

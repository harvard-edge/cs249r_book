# Intersample velocity on the original planning diagram

`original-intersample-overshoot` enhances the existing Volume IV two-panel SVG.
A shared time cursor connects progressively shaded signed acceleration area to
velocity on the original bronze and pine trajectories. Endpoint samples remain
fixed and valid while the bronze interpolant crosses the ceiling. The pine
comparison changes only the baseline velocity; its acceleration is identical.

The numerical exporter reads the existing normalized lesson from
`mechanisms/vol4-b.calc.py`, computes coordinate correspondence, and verifies it
against the exact original SVG paths. Values are normalized illustrations, not
physical machine measurements. Playback speed is not a simulation time scale.

The source assumes equal endpoint velocities, bounded acceleration, and an
instantaneous sign reversal. No finite jerk limit or general motion-safety
certification follows from the velocity reserve. Static chapter assets remain
unchanged.

Run from the repository root:

```sh
python3 books/_extensions/mlsysbook-ext/figure-explore/original-intersample-overshoot.calc.py
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-intersample-overshoot/check.py
/tmp/venv/bin/python books/_prototypes/vol1-web-animations/original-intersample-overshoot/check.py --engine webkit --screenshots /tmp/original-intersample-overshoot-webkit
```

The check makes a disposable fixture from the existing complete chapter listed
in `/tmp/figures-animation-art-current/index.json`. It retains local prose,
caption, navigation, and actual volume theme, and adds fixture-only short timing.
Production timing comes solely from the shared YAML metadata; the factory has
no fallback.

Checks cover every endpoint and original sample location, threshold equality
versus violation, normalized signed areas, unchanged acceleration in the pine
comparison, pause/freeze, sequence jumps, keyboard controls, reduced motion,
play-once/replay/restart, expansion/Escape, print fallback, and chapter overflow.
Live marker, time, velocity classification, and signed-area readouts update from the same interpolated state and freeze together. A registered-frame replay in the test fixture checks a tiny overspeed that would otherwise round to the ceiling; the UI preserves the positive excess explicitly.

Mobile follows the original time cursor; source panel titles remain visible in
the viewport and a short key identifies the original dashed ceilings. Expanded
inspection uses the same objects and state.

This is component verification inside an actual-chapter fixture. Production
enrollment, static targets, and complete volume builds are separate checks.

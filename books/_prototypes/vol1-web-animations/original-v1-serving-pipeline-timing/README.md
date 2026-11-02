# How do different requests overlap on the same CPU and GPU?

Make request identity, readiness, and concurrent resource progress visible within the original schedule.

Scope: Original alternative schedules, symbolic equal-width slots, one GPU; source A has two requests and B four.

1. Prepare R1 before the GPU consumes it.
2. Prepare R2 while the same GPU executes R1.
3. Keep each next request ready until its own handoff.
4. Drain all four illustrated requests without comparing unequal-workload counts.

Guard: No measured timing, automatic full utilization, or matched-workload speedup ratio. GPU1…4 are requests, not devices.

Run `python3 books/_prototypes/vol1-web-animations/original-v1-serving-pipeline-timing/check.py`. The fixture uses the full owning chapter from the canonical art index. It tests exact endpoints, repeatable menu jumps, keyboard, real-motion freeze/resume in Expand, reduced motion, overflow, and print player hiding in Chromium/WebKit, desktop/phone, light/dark. Production ordinary visible playback and complete static fallback require the root integration build. Source PNGs: `/tmp/v1-serving-pipeline-timing.png`. QA PNGs: `/tmp/original-v1-serving-pipeline-timing-review`. No real student study is claimed.

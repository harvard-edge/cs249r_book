# When can the candidate bundle be released after rollback?

This QA prototype enhances `vol1:fig-rollback-compatible-bundles` in the ML operations chapter (*Stateful vs. stateless rollback*). The verified original SVG supplies the router bar, both bundles, the green solid and gray dashed arrows, the red dashed invalid-pairing box, the two notes, and the degraded-window log bar. Toy request markers add a time layer to those same arrows. The static figure explains that both arrows exist. Explore shows that they belong to different admission times, and that Bundle 2 stays resident until its last pinned request is gone.

## Question and storyboard

The question is: when we roll back, what happens to requests already running on the candidate, and when can the candidate bundle actually be released? The lesson has nine states, each with one causal change.

1. Bundle 2 is the routing target. R1–R4 carry weights 2, schema 2, and namespace 2, and Bundle 1 is a loaded standby. The green "after rollback" path is faded.
2. Cutover selects Bundle 1. The target chip moves, and no request moves with it.
3. R5 and R6 travel down the green arrow and pin to Bundle 1. Both bundles now serve.
4. R1 and R2 finish on Bundle 2. In flight drops from 4 to 2.
5. The mixed pairing is refused. A ghost of R3 approaches the invalid box, which lights up, and R3 stays on Bundle 2. No counter changes.
6. Cancellation of R4 is requested. R4 still counts (dashed outline), so the count stays at 2.
7. R4's cancellation is confirmed. In flight drops from 2 to 1.
8. R3, the last pinned request, finishes. In flight reaches 0, and the bundle is marked `0 · releasable`.
9. Bundle 2 is released. The log bar stays flagged with the R1–R4 records, because reverting routing repaired no data. The cost is that both bundles stayed loaded through the drain.

The four claims in the caption map to states 2, 3–8, 5, and 9.

## Fidelity guard

All request markers, counts, and their completion order are declared toy values. The section gives no request counts, rates, durations, or drain timeout, and it has no LEGO cell, so no real number appears in this view. Movement speed encodes no latency. No request changes bundle after admission. Cancellation is not instantaneous. Bundle 2 is never released before its in-flight count reaches 0, and routing never repairs logs. Degraded-window records are counted for requests admitted to Bundle 2. The view keeps the stateful namespace framing and does not reduce it to a stateless blue-green flip.

`original-v1-rollback-compatible-bundles.calc.py` derives every counter from the per-request ledger and asserts these invariants. It reads the source rectangles from the verified SVG, checks that the token slots clear the boxes and arrows, and verifies that the ledger and geometry embedded in the factory are current:

```sh
python3 books/_extensions/mlsysbook-ext/figure-explore/original-v1-rollback-compatible-bundles.calc.py
```

On screens 700 px wide or narrower, a responsive redraw keeps the same objects, wording, arrow semantics, and state. Both renderings, the readout line, and the toy request ledger table derive from one committed state. While a move is in progress or paused, the counters keep the last committed values and the narration is prefixed "On completion".

## Run the check

```sh
python3 books/_prototypes/vol1-web-animations/original-v1-rollback-compatible-bundles/check.py
```

The check serves the actual rendered chapter from `/tmp/figures-animation-art-current/index.json` and mounts the factory in place on the verified artwork. It runs Chromium and WebKit at 1440 px and 390 px in light and dark. For every state it covers menu jumps, the ledger invariants, the overlay labels in the visible layout, and `expected_final`. It also checks repeatable jumps, N/P/R/K keys, absence of autoplay, chapter and component overflow, and Expand. With real motion it confirms that a paused admission freezes the drawing, token position, and ledger, and that Play then resumes it. It confirms reduced-motion commits and the player hidden in print, and fails on any page error. The one exception: during WebKit's emulated-print relayout it tolerates the shared, documented `ResizeObserver loop` delivery warning, and only that warning. Screenshots go to `/tmp/original-v1-rollback-compatible-bundles-review/`.

This is a local review fixture. It does not establish production enrollment or a PDF/EPUB build, and it is not a student study.

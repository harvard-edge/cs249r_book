# Why does a closed-loop client record fewer delayed requests?

This QA prototype enhances `vol1:fig-benchmark-coordinated-omission` in the benchmark-harness section of the benchmarking chapter. The complete original SVG supplies the shaded server pause, both policy rows, the six drawn arrival-to-response intervals, the notes, the axis, and the legend. One time cursor advances along that same axis. Requests the cursor has not reached are faded in place; issued requests show their arrival dot and a strong trace that grows until the drawn response; completed intervals return to full strength. Nothing is added to the drawing except the cursor, the traces, a pause outline while the server is paused, committed counts beneath each row label, and one short cue inside the pause.

## Question and storyboard

The static figure shows the finished intervals. It does not show that the closed-loop client's silence happens during the pause, or that the open-loop backlog builds and then drains. The ten endpoints, one causal change each:

1. Establish the axis, the pause, and two empty rows.
2. The server pauses; nothing has been sent.
3. Each policy sends one request inside the pause: 1 outstanding each.
4. Independent arrivals continue to 4 outstanding; the closed-loop client is blocked at 1.
5. The pause ends; every request is still outstanding.
6. Each policy's first response returns and records one sample.
7. Only then does the closed-loop client send its next request, after the pause.
8. The open-loop backlog drains by two responses.
9. The last responses return; nothing is outstanding.
10. Compare: 4 requests offered during the pause against 1, with the following paragraph's caveat that the closed-loop result is valid for a population of waiting clients and does not estimate latency for independent arrivals.

## Fidelity guard

The figure is schematic and has no LEGO cell or MLSysIM value. Every displayed number is a count of drawn requests (sent during the pause, outstanding, recorded, recorded from pause arrivals) at a cursor position on the drawing's own axis. The intervals come from the TikZ coordinates exactly: pause 4.0 to 8.4; open-loop arrivals 4.3, 5.5, 6.7, 7.9 with responses 9.1, 9.7, 10.3, 10.9; closed loop 4.3 to 9.1 and 9.6 to 10.9. There is no millisecond axis, no queue glyph, no service-time bar, and no percentile. Response order follows the drawing, not a queueing model. Animation speed represents no duration.

While a step is moving or paused, the cursor and traces follow the interpolated position, but the counts in the figure, the text ledger, `data-ledger`, and the narration describe only the last committed step; narration marks the pending step as in progress.

## Calculations

`books/_extensions/mlsysbook-ext/figure-explore/original-v1-benchmark-coordinated-omission.calc.py` parses the TikZ block from the chapter, reads the transform and arrowhead positions from the verified artwork (SHA-1 `cd6012765d36…`), computes the per-step ledgers, asserts the invariants (sent = outstanding + recorded per policy; the waiting client never has two requests in flight) and the endpoints (4 vs 1 offered during the pause; 4 vs 2 recorded; 4 vs 1 recorded from pause arrivals), and checks that the constant embedded in the browser module is current. The browser recomputes the same counts from that constant; `check.py` compares every step's ledger with the script's output.

```sh
python3 books/_extensions/mlsysbook-ext/figure-explore/original-v1-benchmark-coordinated-omission.calc.py
```

## Responsive view

Below 700 px the factory crops the original SVG to its timeline (dropping the row-label column and legend), adds compact "Open loop" and "Closed loop" labels and the axis label, and hides the in-figure counts. The text ledger below the figure carries the same committed counts at every width and in both themes. Diagram retains the unchanged full drawing.

## Review check

```sh
python3 books/_prototypes/vol1-web-animations/original-v1-benchmark-coordinated-omission/check.py [--engine chromium|webkit] [--width 1440|390]
```

The check serves the actual rendered chapter from `/tmp/figures-animation-art-current/index.json`, mounts the factory on the verified SVG with the shared player, and runs Chromium and WebKit at 1440 and 390 px in light and dark. For each profile it verifies no autoplay and an untouched static figure; every endpoint's `data-ledger` against the calc script, with the in-figure counters and text ledger matching it; `expected_final`; state-independent menu jumps; P/N/R/K keys; reduced-motion immediate commit; no chapter overflow and no ledger overflow; pause mid-motion freezes cursor, ledger, counts, and narration while the cursor sits strictly between endpoints; resume completes the step; Expand/Escape; print hides the player; no page errors. Screenshots of every state and the paused and expanded views go to `/tmp/original-v1-benchmark-coordinated-omission-review/`.

WebKit reports `ResizeObserver loop completed with undelivered notifications` when print media is emulated. The accepted `original-v1-benchmark-async-timing` module reproduces it identically in the same chapter, so the check tolerates exactly that message, in WebKit, during the print step only; any error before that fails the profile.

This is a local review artifact. It establishes neither production enrollment nor PDF/EPUB correctness, and no student testing is claimed.

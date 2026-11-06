# How much do I clip, and how coarse is the step?

This QA prototype enhances `vol1:fig-resnet-activations-histogram` in the range-calibration section of Model Compression. The prose describes a continuous clipping-versus-resolution trade-off (max, entropy, percentile) and the figure freezes one candidate threshold. Explore moves one clipping threshold β across the original histogram so the reader sees the UINT8 step shrink while the clipped share of the plotted mass grows.

The complete verified SVG supplies the title, axes, blue bulk, red outlier tail, green candidate line at 1.1, and both annotations. Three objects are added: the β line (broken around the printed annotations so it never covers their text), hatching over each bar that lies beyond β, and a UINT8 code ruler under the axis that lays the 255 steps on the same activation scale (a tick every 17 codes). Committed values appear as text cards: β, step s = β/255, rounding bound s/2, clipped share of plotted mass, and the largest clipped-value error, 2.1 − β.

## Storyboard (6 endpoints)

1. The printed histogram: unchanged figure, no threshold, empty ruler.
2. Max method, β = 2.1 (top edge of the last occupied bin): nothing clipped, largest step.
3. Lower β to 1.6: five tail bars clipped, step shrinks.
4. Lower β to the printed candidate, 1.1: all nine red bars and the blue 1.1 to 1.2 bar clipped; a clipped value can be off by up to 1.0.
5. Push β into the bulk, 0.5: the clipped share jumps (the "truncates signal" side).
6. Compare the two error terms: back at 1.1, with a reader-controlled β slider on the bin edges 0.1 to 2.5 (above 2.1 the ruler shows unused codes) and a table of the visited thresholds. It is framed as the trade-off that calibration methods resolve differently, not as an accuracy prediction.

Each step changes only β. Between endpoints the line glides and bars become hatched as the line passes them. The readout keeps the last committed values, the tag reads "β → target", and the narration states that the move is in progress. Motion encodes no time.

## Guard

- Bar heights are read from the figure's own TikZ tables in the chapter source (digitized and adapted from wu2020integer, normalized to a peak of 1). They are relative frequencies, not counts, so the clipped share is labeled illustrative. No percentile rank, KL or MSE optimum, or accuracy claim is made.
- Grid: all plotted activations are non-negative, so α = 0 on the chapter's affine UINT8 grid (eq-quant-scale, LEGO `QuantizationMathCalc`): s = β/255, z = 0, codes 0 to 255. This is stated in the scope because fig-calibration-ranges uses signed INT8 mappings.
- Thresholds sit on the 0.1 bin edges, so a bar is either kept or clipped. "Max" means the top edge of the last occupied bin (2.1), which is the histogram's resolution of the extremum.
- The default state is the printed figure; Diagram, caption, and print are unchanged.

## Calculations

`books/_extensions/mlsysbook-ext/figure-explore/original-v1-activation-clipping.calc.py` parses the TikZ tables and computes every threshold from 0.1 to 2.5 with MLSysIM's `QuantizationGridScenario` (bits = 8, α = 0). It asserts the endpoints (max clips nothing, s = 2.1/255; 1.1 clips 10 bars; 1.6 clips 5; 0.5 clips 16), the monotone trade-off, and the SVG axis mapping against drawn bar and line positions. It embeds the result in the factory between `BEGIN/END calc.py output` markers; the browser computes no reported value.

```sh
PYTHONPATH=mlsysim ../MLSysBook/.venv/bin/python books/_extensions/mlsysbook-ext/figure-explore/original-v1-activation-clipping.calc.py --check   # or --write
```

## Review fixture

Run from the repository root with a Python that has Playwright (the calculation step uses `../MLSysBook/.venv/bin/python`, or set `AC_CALC_PYTHON`):

```sh
python3 books/_prototypes/vol1-web-animations/original-v1-activation-clipping/check.py [--engine chromium|webkit] [--width 1440|390]
```

The check runs `calc.py --check`, then builds the full owning-chapter fixture through `build_original_figure_fixture.py` (served from a private port) and exercises Chromium and WebKit at 1440 and 390 px in light and dark. Coverage:

- every endpoint, with the ledger equal to calc.py and the hatched-bar count equal to the ledger;
- `expected_final`, the monotone trade-off, slider extremes, and a final-state reset after a menu jump;
- repeatable menu jumps, N/P/R keyboard, and reduced motion committing directly;
- Expand/Escape, and paused motion freezing the line, hatching, ledger, and pending narration before resuming;
- added-label overlap and canvas bounds, stage and card overflow, and no chapter horizontal scroll;
- Diagram view, print hiding the player, the JavaScript-disabled static figure, and no page errors.

Screenshots (2× device scale) go to `/tmp/original-v1-activation-clipping-review/`. This is author QA, not production enrollment; PDF/EPUB builds are unaffected and remain root integration checks. No student testing is claimed.

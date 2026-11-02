# Where do the errors go when the boundary moves?

This QA prototype enhances `vol1:fig-fairness-threshold` in the fairness section of the responsible-engineering chapter. It uses the redesigned toy from commit `ce4d703324`: the left dashed line (0.30) carries 81.25% and the right line (0.63) carries 75%. The complete original SVG (SHA-1 `1c925a5d3062…`) supplies both subgroup rows, all sixteen drawn applicants, the two dashed candidate boundaries with their labels, and the legend. A boundary (a solid line with light approval shading to its left) moves along the same axis. It is drawn as two segments, one per row. The segments coincide while the boundary is shared and separate when Subgroup B gets its own cutoff, so the split is visibly the same object coming apart. Misclassified markers get a dashed ring with a cross glyph above it, so error marking does not depend on color, which already encodes subgroup. Committed counts sit under each row title and the accuracy sits top left. The approve/deny tags under row B take the longest wording whose measured extent clears both dashed candidate lines (for example `◀ B approves`, then `◀ B`, then a bare arrow), so they never cross a dashed line, a marker, or a ring.

## Question and storyboard

The static figure shows two snapshots and leaves the reader to count misclassifications by eye. The Explore view moves the boundary one marker at a time in the caption's direction (75% to 81.25%), confirms that no shared cutoff beats 81.25%, and then shows the threshold adjustment that the paragraph after the figure describes. There are twelve states, each with one causal change:

1. Establish the rows, the sixteen toy applicants, and both candidate lines. Nothing is applied. The narration states the score direction once: markers further left have higher scores, so a boundary approves everything to its left and moving it right is more lenient (the scope and the phone legend repeat this).
2. Apply the 75% boundary to both rows. Approvals only (A 7 of 8, B 3 of 8); outcomes are not yet checked.
3. Check decisions against outcomes: A has 3 approved defaulters, B has 1 denied repayer. 12 of 16 correct (75%).
4. Move 1 (left, stricter) denies a B repayer: A 3, B 2, 68.75%.
5. Move 2 denies an A defaulter: A 2, B 2, 75%.
6. Move 3 denies another B repayer: A 2, B 3, 68.75%.
7. Move 4 denies another A defaulter: A 1, B 3, 75%.
8. Reach the 81.25% line: A 0, B 3 (all denied repayers), 13 of 16.
9. Compare the candidates and sweep all 17 shared positions. 81.25% is reached only here, and every other position scores at most 75%. B's TPR is 0.25 against A's 1.00.
10. Give B its own cutoff while A stays on the 81.25% line. B's cutoff moves right (more lenient) and regains the two repayers lost in Moves 1 and 3: B has 1 error (a denied repayer), TPR 0.75, FPR 0.00. Overall accuracy is already 93.75%, above every shared boundary, with no false positive; only TPR is still unequal (0.75 against 1.00). This is the one step that crosses two markers, both B repayers.
11. B's cutoff approves a defaulter: B has 2 errors, FPR 0.25, TPR still 0.75, accuracy 87.5%. B's last denied repayer sits right of this defaulter, so equal TPR cannot be reached without approving it.
12. B's cutoff approves its last repayer. TPR is 1.00 for both groups, but B's FPR is 0.25 against A's 0.00. B's one error is now an approved defaulter instead of a denied repayer. Overall accuracy is 15 of 16, the same as at B cutoff 1, which had FPR 0.00 but TPR 0.75: the false positive is the price of equal TPR, not of a separate cutoff, so accuracy does not settle the choice. The step carries the prose's legal caveat and closes with "a policy decision, not a metric".

The text ledger below the figure shows the committed step's approvals, repayers approved (TPR, as a count of 4 with the rate, for example `3 of 4 (0.75)`), defaulters approved (FPR, same form), and misclassifications per subgroup, plus the overall accuracy. From step 9 it also shows the sweep result, a strip with the correct count (of 16) at each of the 17 shared positions (the 81.25% and 75% lines' positions shaded, the unique best in bold), and a history table of every boundary visited.

## Fidelity guard

The figure is an illustrative toy with no LEGO cell or MLSysIM value. Every displayed number is a count of drawn markers on one side of a boundary, or a ratio of those counts out of 16 (accuracy) or 4 (TPR and FPR). The view is labeled as a toy in the ledger note and in the assumptions, and it shows none of the LoanFairness credit-service metrics from the chapter table. Marker positions, boundary positions, colors, shapes, and the approve-to-the-left rule come exactly from the TikZ source. Intermediate stops are midpoints between consecutive drawn positions (Subgroup B positions only, for the B-only cutoff). No score value is displayed, and the dashed labels remain accuracies, not cutoffs. Animation speed represents nothing.

The split phase states plainly that the boundary is no longer shared. A stays fixed on the 81.25% line. The FPR cost and the legal restriction are stated in the final step, and the 93.75% toy accuracy is never presented as a recommendation. B reaches its own minimum of 1 error in two places: left of its 0.69 defaulter (a denied repayer) and right of its 0.76 repayer (an approved defaulter). The lesson visits both, because the move from one to the other is the TPR-for-FPR trade.

While a step is moving or paused, the boundary segments and shading follow the interpolated position. The rings, counts, ledger, `data-ledger`, and narration describe only the last committed step, and the narration marks the pending step as in progress.

## Calculations

`books/_extensions/mlsysbook-ext/figure-explore/original-v1-fairness-threshold.calc.py` does the following:

- Parses the four marker lists, both boundary coordinates, and both labels from the chapter's TikZ.
- Reads the transform from the verified artwork and matches all sixteen drawn marker paths to their TikZ positions.
- Computes the per-step ledgers.
- Asserts the caption's endpoints (75%: A 3, B 1, 12 of 16; 81.25%: A 0, B 3, 13 of 16, all denied repayers, B TPR 0.25).
- Asserts the conservation invariants (approved = repayers approved + defaulters approved; correct + errors = 16).
- Asserts that each shared move crosses exactly one marker.
- Checks every distinct shared cutoff: 81.25% is the unique best, every other cutoff is at most 75%, and no shared cutoff reaches both subgroups' own minima (A 0, B 1).
- Asserts B's two minimum-error regions and the split-step TPR/FPR/error sequence.
- Checks that the constant embedded in the browser module is current.

The browser recomputes the same counts from that constant, and `check.py` compares every step's ledger with the script's output.

```sh
python3 books/_extensions/mlsysbook-ext/figure-explore/original-v1-fairness-threshold.calc.py [--check]
```

`--check` is accepted for parity with the sibling calc scripts; the embedded constant is always checked.

## Responsive view

Below 700 px the factory makes these changes:

- Crops the original SVG to the two score rows, keeping the whole score span through the ring on the last B marker.
- Adds compact A and B row labels.
- Hides the in-figure counts and the approve/deny tags.
- Wraps the sweep strip onto two lines of position/count cells.
- Draws the error rings and crosses heavier.
- Shows a legend line above the ledger that points to Expand.

The ledger carries the same committed counts at every width and in both themes. Diagram keeps the full drawing unchanged.

## Review check

```sh
python3 books/_prototypes/vol1-web-animations/original-v1-fairness-threshold/check.py [--engine chromium|webkit] [--width 1440|390]
```

The check serves the actual rendered chapter from `/tmp/figures-animation-art-current/index.json`. Once the chapter is enrolled, the check removes the figure's Explore enrollment attributes from that one div as it serves the page, so its own mounted copy runs on an unenrolled host; production enrollment is checked by `binder/tools/scripts/check_original_figures.py`. It mounts the factory on the verified SVG with the shared player and runs Chromium and WebKit at 1440 and 390 px in light and dark. Each profile verifies:

- no autoplay, and the static figure is untouched;
- every endpoint's `data-ledger` against the calc script, with both boundary segments, rings, in-figure counters, ledger cells, sweep visibility, and history rows matching;
- `expected_final`;
- the compare narration (all 17 positions, "only here", at most 75%, TPR 0.25), and the sweep strip's counts against the calc's sweep;
- the step-10 narration (accuracy already 93.75%, FPR stays 0.00) and the step-12 comparison with B cutoff 1;
- in every state, the approve/deny tags' boxes intersect no dashed line, drawn marker, legend glyph, or error ring;
- the final narration (TPR equalized, FPR cost, error kind changed, legal caveat);
- state-independent menu jumps, the P/N/R/K keys, and reduced-motion immediate commit;
- no chapter, ledger, or figure overflow;
- a pause mid-motion in the expanded view (shared step) and in the page (split step) freezes the segments, ledger, rings, counts, and narration while the moving segment sits strictly between endpoints and A's segment stays put;
- resume completes the step;
- Expand/Escape;
- print hides the player;
- no page errors.

Screenshots of every state and of the paused and expanded views go to `/tmp/original-v1-fairness-threshold-review/`.

As in the sibling modules, WebKit's `ResizeObserver loop completed with undelivered notifications` message under print-media emulation is tolerated in the print step only.

This is a local review artifact. It establishes neither production enrollment nor PDF/EPUB correctness, and no student testing is claimed.

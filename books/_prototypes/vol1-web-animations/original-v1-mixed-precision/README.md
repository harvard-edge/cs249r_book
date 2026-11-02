# Why scale the loss, and what if S is wrong?

This QA prototype enhances `vol1:fig-mixed-precision` in the mixed-precision section of `books/vol1/08_training/08_training.qmd` (loss scaling, `sec-model-training-loss-scaling-f9f5`). The verified original SVG supplies the six boxes, their FP32/FP16/scaled colors, the numbered arrows, and the dashed update edge. Explore adds only value tags beside each box, a loss-scale panel in the empty center of the cycle, an orange outline on the active box and step label, and a value token that moves along the active arrow. The static Diagram view and caption are unchanged.

## Learner question

The static cycle names Scale and Unscale but shows no values, so the reason for scaling lives in two prose paragraphs. Here the reader follows one tracked gradient and the largest entry of the same gradient tensor through the cycle. A too-small S loses the small value, a too-large S overflows the large value, and the dynamic scaler's skip-and-halve rule finds a factor that fits both.

## Storyboard (16 endpoints)

1. Declare the toy values: FP32 master w = 0.100000001, gradients 2⁻²⁶ (tracked) and 3 (largest entry), FP16 limits.
2. **1. Cast:** w rounds to 0.0999755859375 (0x2E66); the FP32 master is unchanged.
3. **2. Forward:** FP16 toy loss L = 0.75.
4. **3. Scale** by S = 1.
5. **4. Backprop:** 2⁻²⁶ is below half of FP16's smallest subnormal (2⁻²⁴), so it rounds to 0.
6. Compare S = 2¹⁰ (scaled loss 768).
7. **4. Backprop:** S·g = 2⁻¹⁶ is subnormal. IEEE keeps it; flush-to-zero hardware would store 0.
8. Compare S = 2¹⁵ on the same forward pass, a larger what-if factor (scaled loss 24,576).
9. **4. Backprop:** S·g = 2⁻¹¹ is normal, but 3 × 32,768 = 98,304 > 65,504 overflows to inf.
10. **5. Copy & Unscale at 2¹⁵:** the tracked value returns to exactly 2⁻²⁶, but inf ÷ S is still inf. The nonfinite entry is found in these unscaled FP32 gradients, before the update.
11. Dynamic scaler: only the update (step 6) is skipped, so the inf never reaches the master weight, and S halves to 2¹⁴.
12. Next iteration (toy assumption: same values): scale by 2¹⁴ (12,288).
13. **4. Backprop:** 2⁻¹² and 49,152 are both finite FP16 values.
14. **5. Copy & Unscale:** exactly 2⁻²⁶ and 3 in FP32.
15. **6. Update:** with toy η = 0.5, w moves by −2⁻²⁷ (one FP32 step) to 0.099999994. FP16 spacing near 0.1 is 2⁻¹⁴, so an FP16 master would not move.
16. BF16 comparison: step 3 is dropped and step 5 only copies to FP32. 2⁻²⁶ and 3 are representable unscaled; the trade-off is mantissa precision (0.100000001 becomes 0.100097656). The tags switch to this BF16 run; the boxes keep the figure's FP16 labels.

## Fidelity guard

The six boxes, labels, arrows, and colors are the TikZ source's, and tags never cover a label. Every displayed value comes from `original-v1-mixed-precision.calc.py`, computed with `numpy.float16` (IEEE binary16, subnormals included) and a round-to-nearest-even bfloat16 conversion. The script asserts each storyboard endpoint and embeds the result in the factory, and the browser performs no rounding. The FP16 limits (2⁻¹⁴ ≈ 6.1 × 10⁻⁵, 65,504, about 6 × 10⁻⁸) are IEEE format constants. The script asserts them against the chapter's prose literals because no LEGO cell or MLSysIM constant owns them. S = 2¹⁰ and 2¹⁵ are the prose's examples, and 2¹⁴ is one halving. Weight, loss, gradients, and learning rate are declared toy values, labeled as toy in the tags and scope. The claim is per value, not about a whole gradient distribution. Skip-and-halve is the prose's dynamic-scaler rule, not a seventh figure step. The lesson includes no GradScaler API detail and makes no speedup or memory claim. The caption's hedge that framework implementations may vary stays in the scope and the final state. Motion is illustrative and does not encode time.

The moving token swerves to one side of its arrow while it passes the step-label plate, so the label stays readable at every paused frame. While an action is moving or paused, tags and the ledger still show the last committed state, and the narration labels the next state as pending. On phones (≤ 600 px), the drawing stays full width, and the same node values move to a 2 × 3 grid of cards that follows the drawing's layout. `data-ledger` on the root exports the committed numeric state, and the shared player sets `data-step`.

## Run the check

From the repository root, with a Python that has Playwright (numpy is found in `../MLSysBook/.venv` or via `MP_CALC_PYTHON`):

```sh
python3 books/_prototypes/vol1-web-animations/original-v1-mixed-precision/check.py            # chromium + webkit
python3 books/_prototypes/vol1-web-animations/original-v1-mixed-precision/check.py --engine webkit --width 390
/path/to/numpy/python books/_extensions/mlsysbook-ext/figure-explore/original-v1-mixed-precision.calc.py --check
```

The check first runs `calc.py --check`. It then serves `books/` on a private port and builds the full owning-chapter fixture through `build_original_figure_fixture.py`, which keeps the prose, static figure, caption, navigation, and theme assets. It covers Chromium and WebKit at 1440 px and 390 px in light and dark themes. In each configuration it checks:

- no autoplay on opening Explore
- all 16 menu endpoints, with mechanism invariants (underflow, subnormal, overflow, inf found at unscale, a skip of only the update that leaves the master weight untouched, exact unscale, a one-ulp update) and `expected_final`; the dashed skip marks are exactly plate 6 (skip) and plate 3 (BF16)
- menu jumps that are independent of history
- N/P/R keyboard stepping, and reduced motion committing without an intermediate state
- tag and panel text inside their plates, and no overflow of the stage, cards, or chapter
- Expand and Escape
- in the light runs, that pause freezes the token, ledger, and narration, that the paused token does not cover its step label, and that Play resumes
- Diagram switching, and that print hides the player
- no page errors

WebKit's print-media reflow makes the chapter's own Quarto sidebar ResizeObserver report a benign "loop completed" notice. This also happens with the figure left in Diagram view. The check tolerates only that message and only at that point. A JavaScript-disabled load keeps the static figure and caption. Screenshots go to `/tmp/original-v1-mixed-precision-review/`.

This is a local QA fixture. It establishes neither production enrollment nor PDF or EPUB build correctness, and no student testing is claimed.

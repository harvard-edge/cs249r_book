# Why does training keep every state that inference frees?

This QA prototype enhances `vol1:fig-rnn` in the algorithmic-structure section of `books/vol1/06_nn_architectures/06_nn_architectures.qmd` (`sec-network-architectures-algorithmic-structure-9dea`). The verified original SVG supplies the compact loop, the unfold arrow, the three unrolled columns, the green hidden, purple input, and orange output circles, and every W_hx, W_hh, and W_yh label. Explore adds only a status ring and a small badge beside each unrolled hidden node, orange outlines on the label plates a step reads, orange traces along the arrows a step uses, and two short chips (the zero initial state and the backward gradient). A panel below the drawing lists the same statuses with retained-memory and MAC ledgers. The static Diagram view and caption are unchanged.

## Learner question

The static unfolding draws all three hidden states at once. That matches the training footprint, but it hides the inference behavior the prose stresses, where only the current state is kept. The difference between the two regimes is lifetime: which states exist, when each one is freed, and why. The serial dependency is also temporal. The figure draws the arrows, but it cannot show step t waiting for h(t−1) while the input projections are free to run ahead.

## Storyboard (12 endpoints)

1. **Inference:** the three matrices exist once each. The figure repeats their labels, not their storage. No state exists, and no input has arrived.
2. Step t−1 reads the zero initial state and x(t−1), then writes h(t−1): 16,384 + 12,800 = 29,184 MACs. 128 values are kept.
3. Step t reads h(t−1) through the same W_hh copy and writes h(t). That read was the last use of h(t−1), so it is freed. Still 128 values.
4. Step t+1 writes h(t+1) and frees h(t). Still 128 values after 3 × 16,384 = 49,152 recurrent MACs, run in order.
5. **Training (BPTT):** the whole sequence is known, so the three input projections may run together (3 × 12,800 = 38,400 MACs). The recurrent products still wait for their predecessors.
6. Forward t−1 writes h(t−1) and keeps it: 128 values.
7. Forward t keeps h(t−1) for the backward pass, where inference would free it: 256 values.
8. Forward t+1 keeps all three states, the 384-value peak. The forward work equals inference's 87,552 MACs.
9. Backward through t+1: the gradient enters from y(t+1). Nothing later needs h(t+1), so it is freed: 256 values.
10. Backward through t: ∂L/∂h moves back through W_hh, a Jacobian product, and joins the gradient from y(t). h(t) is freed: 128 values.
11. Backward through t−1 frees h(t−1). Each shared matrix receives one gradient summed over the three steps.
12. **Compare:** inference keeps 128 values at any length. BPTT keeps 384 at toy T = 3 and 128,000 values (512,000 bytes in FP32) at the scenario's 1,000 steps. Both run the recurrence in order. Recomputation trades work for memory, stated as the prose does, without a quantity.

## Fidelity guard

The circles, labels, arrows, colors, and compact loop are the TikZ source's. Badges sit to the upper left of each node, and a check asserts that none covers a label plate. Retained states use the memory blue (`#006395`), and compute reads use the compute orange (`#E67817`). Waiting and freed states are neutral gray. Red is not used.

Every count comes from `original-v1-rnn-state-lifetime.calc.py`. The script loads `Scenarios.ElmanRnnStep`, the MLSysIM scenario that the chapter's `RNNCompute` LEGO cell uses. It asserts the caption's literals (128-dimensional state, 100-dimensional input, 16,384 recurrent and 12,800 input MACs) and the prose's 1,000-step sequence. It then derives each state's ledger and asserts the endpoints: inference holds 128 values throughout, training holds 0, 128, 256, 384, 256, 128, and 0, and both total 87,552 MACs. The script embeds the result in the factory, and the browser does no arithmetic. The figure's three drawn steps are treated as a complete toy sequence (T = 3) from a zero initial state. The scope says so.

Counts follow the chapter's h₁ … h_T list. Inputs, outputs, other saved tensors, output-projection MACs, and backward-pass MACs are not counted. The scope notes that during an inference update, the old and new state briefly coexist. The only byte figure is the MLSysIM scenario's FP32 state size (512 bytes), used once in the comparison. The lesson adds no LSTM gates, truncated BPTT, checkpointing quantity, or latency claim. Motion duration is illustrative and does not encode time.

While an action is moving or paused, the badges, slots, and ledgers still show the last committed state, and the narration labels the next state as pending. On phones (≤ 600 px), the drawing is cropped to the three unrolled columns, and the SVG badges are hidden because they would render below 7 px. The three HTML slots below carry the same statuses. `data-ledger` on the root exports the committed state, and the shared player sets `data-step`.

## Run the check

From the repository root, with a Python that has Playwright (the MLSysIM calculation runs through `../MLSysBook/.venv/bin/python` when the current Python lacks numpy, or through `RNN_CALC_PYTHON`):

```sh
python3 books/_prototypes/vol1-web-animations/original-v1-rnn-state-lifetime/check.py            # chromium + webkit
python3 books/_prototypes/vol1-web-animations/original-v1-rnn-state-lifetime/check.py --engine webkit --width 390
PYTHONPATH=mlsysim ../MLSysBook/.venv/bin/python books/_extensions/mlsysbook-ext/figure-explore/original-v1-rnn-state-lifetime.calc.py --check
```

The check first runs `calc.py --check` and confirms that the numbers restated in `entry.json` match it. It then serves `books/` on a private port and builds the full owning-chapter fixture through `build_original_figure_fixture.py`. The fixture keeps the prose, static figure, caption, navigation, and theme assets. The check covers Chromium and WebKit at 1440 px and 390 px in light and dark themes. In each configuration it checks:

- no autoplay on opening Explore
- all 12 menu endpoints, with these mechanism invariants:
  - inference keeps exactly one state
  - a state is freed only after its reader exists
  - training's input projections run ahead
  - backward frees right to left after each state's own step
  - a node waits only while its predecessor is unwritten
  - recurrent MACs equal W_hh reads × 16,384
  - the expected retained sequence
  - `expected_final`
- that the panel restates the ledger's retained count
- menu jumps that are independent of history
- N/P/R keyboard stepping, and reduced motion committing without an intermediate state
- badge text inside its badge, no badge over a label plate or another badge, and no overflow of the stage, panel, or chapter
- Expand and Escape
- in the light runs, that pause freezes the motion layer, ledger, and narration while the committed count stays put, and that Play resumes
- Diagram switching, and that print hides the player
- no page errors, and a JavaScript-disabled load that keeps the static figure and caption

Screenshots go to `/tmp/original-v1-rnn-state-lifetime-review/`.

This is a local QA fixture. It establishes neither production enrollment nor PDF or EPUB build correctness, and no student testing is claimed.

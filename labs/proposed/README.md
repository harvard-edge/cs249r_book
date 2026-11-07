# Proposed Co-Labs

Labs here follow the Co-Lab format and get the same per-lab checks as `vol1/` and
`vol2/` (`labs/tests/conftest.py` includes them in `ALL_LABS`), but do not yet have a
chapter slot. The catalog, track arcs, and site navigation are unchanged until the maintainers place them.

## The Same-Bits Contract (`lab_same_bits.py`)

**Driving question:** You send the same prompt at temperature 0, twice, and get two
different answers. Nothing was sampled. Where did the difference come from, and what
does it cost to remove it?

| Part | Prediction | What students measure |
|---|---|---|
| A · One sum, five orders | How many float32 results from 5 orders? | 4 distinct sums of the same 4,096 numbers |
| B · The batch picks the kernel | Alone vs batched: how many outputs change? | 108–113 of 128 outputs change bits (float16) |
| C · From bits to words | How many of 64 users see different text? | Up to 7 of 64 at temperature 0 |
| D · Batch-invariant kernels | What does a fixed split cost? | 0 of 64 changed, plus the extra partial-sum additions |
| E · Integer arithmetic | What do exact int32 sums give up? | 0 of 64 changed at any batch, 94.5% token agreement with float |
| Synthesis | Choose a determinism contract | For the track's stakeholder: RL trainer, incident review, support, firmware validation |

Every number is computed by `mlsysim.labs.experiments.same_bits_experiments` with real
IEEE 754 float16/float32 arithmetic, rounded after each operation, and explicit reduction
orders. The language model is a seeded toy recurrent model, so counts illustrate the
mechanism rather than measure a production model.

**Possible placements:** Volume II, Chapter 10 (Inference at Scale) as a follow-on to
*The Inference Economy*, or Volume I, Chapter 12 (Benchmarking), where reproducible
measurement is the theme.

```bash
python3 -m pytest labs/tests/test_static.py labs/tests/test_protocol.py labs/tests/test_widget.py labs/tests/test_engine.py -v -k proposed
python3 -m pytest mlsysim/tests/test_same_bits_experiments.py -v
marimo edit labs/proposed/lab_same_bits.py
```

The published browser build loads the committed `labs/wheels/mlsysim-*.whl`, which needs
its usual refresh to include the new experiments module.

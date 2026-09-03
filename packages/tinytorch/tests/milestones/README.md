# TinyTorch Milestone Tests

This directory validates the historical milestone scripts that students run with
`tito milestone`.

The canonical milestone definitions live in `tito.commands.milestone.MILESTONE_SCRIPTS`.
The compatibility tracker in `milestone_tracker.py` mirrors that table for older
hooks and tests.

## Test Layers

### Smoke Tests

Fast import and model-construction checks:

```bash
pytest tests/milestones/test_milestones_smoke.py -v
```

Use these in regular development. They catch API drift between milestone scripts
and exported TinyTorch modules without downloading data or running training.

### Full Milestone Runs

End-to-end milestone execution:

```bash
pytest tests/milestones/test_milestones_run.py -v
```

Use these for release validation. They run milestone scripts through
`tito milestone run --skip-checks`, including training where applicable, and can
take several minutes.

## Current Milestones

| ID | Name | Default Script | Purpose |
|----|------|----------------|---------|
| 01 | Perceptron (1958) | `01_1958_perceptron/01_rosenblatt_forward.py` | Forward pass with random weights |
| 02 | XOR Crisis (1969) | `02_1969_xor/01_xor_crisis.py` | Single-layer XOR limitation |
| 03 | MLP Revival (1986) | `02_xor_solved.py`, `01_rumelhart_tinydigits.py` | Hidden layers plus TinyDigits |
| 04 | CNN Revolution (1998) | `04_1998_cnn/01_lecun_tinydigits.py` | Convolutions on TinyDigits |
| 05 | Transformer Era (2017) | `05_2017_transformer/01_vaswani_attention.py` | Attention sequence task |
| 06 | MLPerf Benchmarks (2018) | `06_2018_mlperf/*.py` | Optimization and speedup demos |

## Required Setup

The tests assume the generated package exists:

```bash
tito dev export --all
```

If pytest reports that `tinytorch/core/*.py` files are missing, export the modules
first. Generated notebooks and exported package files are ignored by git.

## Progress Files

The CLI uses:

- `.tito/progress.json` for completed modules
- `.tito/milestones.json` for ready/completed milestones

Do not use a home-directory progress file in new code.

## Maintenance Notes

When adding or renaming a milestone:

1. Update `MILESTONE_SCRIPTS` in `tito/commands/milestone.py`.
2. Update or add the milestone script under `milestones/`.
3. Add smoke coverage in `test_milestones_smoke.py`.
4. Add release-run coverage in `test_milestones_run.py` if the script should run end to end.
5. Update student-facing docs and Quarto pages that mention the command or script name.

# 04 — Result Comparison

## Learning Goal

Distinguish quality compatibility from performance compatibility and explain
why the dashboard sometimes blocks a comparison instead of drawing a chart.

## Runtime and Hardware

Use the laptop-capable image-classification workload. This example requires two
authoritative runs with the same assets and device. Runtime varies by system.

## Create a Compatible Pair

```bash
uv run mlperf fetch --workload image-classification --profile max
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/04-comparison/baseline
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/04-comparison/current
uv run mlperf report \
  submissions/04-comparison/current/image-classification_max_report.json \
  --baseline submissions/04-comparison/baseline/image-classification_max_report.json \
  --format html --output submissions/04-comparison/compatible.html --open
```

## Create an Intentionally Incompatible Pair

```bash
MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE=64 \
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/04-comparison/batch-64
uv run mlperf report \
  submissions/04-comparison/batch-64/image-classification_max_report.json \
  --baseline submissions/04-comparison/baseline/image-classification_max_report.json \
  --format html --output submissions/04-comparison/batch-size-blocked.html --open
```

## Allowed Changes

The first pair must be reruns of the same configuration. The second pair changes
only batch size to demonstrate the compatibility boundary. Do not alter the
dataset, checkpoint, evaluator, or quality target.

## Read the Report

The compatible pair can show both quality and performance comparisons. The
batch-size pair can still compare the unchanged quality contract, but the
standard performance comparison is blocked because the configurations differ.
Use the plan workflow from Example 02 when the configuration difference is the
declared independent variable.

## Interpretation Questions

1. Which fields must match for quality comparison?
2. Which additional fields must match for performance comparison?
3. Why is “blocked” different from “worse”?
4. When should a controlled experiment plan replace a standard rerun comparison?

## Suggested Rubric

| **Item** | **Points** | **Evidence** |
|:---|---:|:---|
| Compatible pair | 3 | Valid quality and performance comparison |
| Boundary diagnosis | 3 | Correct explanation of the batch-size block |
| Claim discipline | 2 | No performance conclusion from incompatible reruns |
| Workflow choice | 2 | Correct use of rerun comparison versus experiment plan |

Submit all three source JSON reports and manifests (`baseline`, `current`, and
`batch-64`), `compatible.html`, `batch-size-blocked.html`, and `answers.md`.
The report labels a structurally compatible comparison as exploratory unless
both source manifests verify.

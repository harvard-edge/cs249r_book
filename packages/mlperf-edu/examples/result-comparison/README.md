# Result Comparison Example

This legacy path is retained for existing links. New courses should use the
[numbered result-comparison lab](../04-result-comparison/README.md).

## Learning Goal

This example teaches the difference between a quality comparison and a
performance comparison. A result can use the same dataset, evaluator, and
quality target while still being an invalid performance baseline because its
checkpoint, configuration, software, or hardware differs.

## Create Two Results

Run the same laptop-capable workload twice and retain the first report as the
provisional baseline.

```bash
uv run mlperf fetch --workload image-classification --profile max
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/comparison/baseline
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/comparison/current
```

This is a one-result classroom comparison, not a promoted stability baseline.

## Open the Comparison Dashboard

```bash
uv run mlperf report \
  submissions/comparison/current/image-classification_max_report.json \
  --baseline submissions/comparison/baseline/image-classification_max_report.json \
  --format html --open
```

The dashboard shows the two quality values against one target. If the complete
comparison fingerprint matches, it also shows paired performance bars and a
direction-aware improvement percentage.

## Inspect an Incompatible Change

Change the batch size for the current run and render the comparison again.

```bash
MLPERF_EDU_IMAGE_CLASSIFICATION_MAX_BATCH_SIZE=64 \
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/comparison/batch-size-64
uv run mlperf report \
  submissions/comparison/batch-size-64/image-classification_max_report.json \
  --baseline submissions/comparison/baseline/image-classification_max_report.json \
  --format html --open
```

Quality remains interpretable when the task, dataset, evaluator, and target
contract match. The performance comparison is blocked because the runs use
different configurations. A later research experiment plan can declare an
allowed independent variable and define how to analyze that controlled change.

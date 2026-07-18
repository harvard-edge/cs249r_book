# 02 — Inference Tradeoff

## Learning Goal

Change one systems variable, preserve the benchmark quality contract, and
explain a baseline-versus-candidate result without overstating one-run timing
evidence.

## Runtime and Hardware

The MLPerf Tiny ResNet8 workload is laptop-capable on CPU, Apple Silicon, and
CUDA. The first run downloads pinned CIFAR-10 and MLPerf Tiny assets. Runtime
varies by device, so instructors should trial the example on the course image.

## Inspect and Run

```bash
uv run mlperf run --plan examples/02-inference-tradeoff/plan.yaml --dry-run
uv run mlperf fetch --workload image-classification --profile max
uv run mlperf run --plan examples/02-inference-tradeoff/plan.yaml
```

The plan runs one baseline condition and one candidate condition. Each executes
the authoritative accuracy contract once and writes separate evidence under
the shared experiment directory.

## Allowed Changes

Students may change only the candidate batch size. The checkpoint, dataset,
accuracy subset, evaluator, target, device, input representation, and baseline
condition remain fixed. The plan loader rejects attempts to override the
quality target.

The plan hash records the submitted design but does not by itself prove that it
matches the instructor's handout. Instructors should diff the submitted plan
against the reference and accept changes only to the candidate batch-size
value. If device or another control differs between conditions, the dashboard
blocks the throughput comparison.

Students should include the repository diff with the submission:

```bash
git diff -- examples/02-inference-tradeoff/plan.yaml \
  > submissions/02-inference-tradeoff/plan.diff
```

## Read the Report

Read quality before throughput. Both cards should use the same top-1 accuracy
target. The condition chart then shows samples per second and a descriptive
candidate delta against the declared baseline. The plan hash, independent
variable, controls, hardware, and condition settings explain what the bars
mean.

## Interpretation Questions

1. Did both conditions meet the unchanged accuracy gate?
2. What throughput delta did the candidate show on this machine?
3. Which part of the measurement changed, and which parts stayed controlled?
4. Why is the delta useful for a lab discussion but insufficient for a stable
   performance claim?

## Suggested Rubric

| **Item** | **Points** | **Evidence** |
|:---|---:|:---|
| Controlled design | 3 | One declared independent variable and correct controls |
| Quality interpretation | 3 | Correct metric, target, direction, and decisions |
| Systems interpretation | 2 | Correct throughput delta and plausible explanation |
| Claim discipline | 2 | Explicit one-run limitation and proposed next measurement |

Submit the plan YAML, `plan.diff`, `answers.md`, and the complete
`submissions/02-inference-tradeoff/` evidence tree. The tree must include the
aggregate JSON, HTML, CSV, aggregate manifest, child reports, and child
manifests. The plan hash proves what ran; the written responses supply the
reasoning evidence used by the rubric.

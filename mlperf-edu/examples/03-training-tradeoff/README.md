# 03 — Training Tradeoff and Checkpoint Lineage

## Learning Goal

Connect a controlled training change to its checkpoint, quality decision, and
downstream inference eligibility. A trained checkpoint is not acceptable merely
because training finished.

## Runtime and Hardware

This is the longest classroom example. An accelerator is recommended, and an
instructor should schedule it as a take-home. Each
condition performs the pinned 5,001-step NanoGPT training contract once. No
five-run stability campaign is required for this milestone.

Existing Apple M5 Max evidence is roughly 34–36 minutes per training condition,
or about 70 minutes for the two-condition plan before inference. This is a
planning reference, not a promised runtime. Instructors should record CPU,
MPS, or CUDA time, peak memory, download size, and disk use on the course image.
The current plan runner does not import a precomputed baseline into a
candidate-only aggregate. Until that workflow exists, do not advertise a
precomputed-baseline shortcut; assign the full plan only when the course budget
can accommodate it.

## Inspect and Run

```bash
uv run mlperf fetch --workload causal-language-modeling --profile max
uv run mlperf run --plan examples/03-training-tradeoff/plan.yaml --dry-run
uv run mlperf run --plan examples/03-training-tradeoff/plan.yaml
```

Run decode inference only from a condition whose training result passed the
fixed validation-loss gate and whose manifest verifies.

```bash
ACCEPTED_RUN=01-baseline-lr-1e-3  # or the passing candidate directory
uv run mlperf run --workload causal-language-modeling --profile max \
  --mode inference --phase decode \
  --output-dir "submissions/03-training-tradeoff/runs/${ACCEPTED_RUN}"
```

The inference runner finds the training checkpoint, training report, and
provenance manifest in the same directory. It refuses a failed or unverified
source result.

## Allowed Changes

Students may change only `MLPERF_EDU_MAX_LR` in the candidate. Architecture,
dataset split, seed, iteration count, evaluator, and quality target remain
fixed. Shortening training creates a different experiment and may cause the
checkpoint to miss the unchanged gate.

Instructors should compare the submitted plan with the reference plan and
accept only the candidate learning-rate edit. The current assignment grader
does not enforce experiment-plan diffs automatically.

```bash
git diff -- examples/03-training-tradeoff/plan.yaml \
  > submissions/03-training-tradeoff/plan.diff
```

## Read the Report

Compare best validation loss first, then training tokens per second. Inspect
the model-lineage section for the training, checkpoint, inference, and
evaluation stages. The candidate result is informative even if it misses the
gate, but a missed checkpoint cannot support an accepted inference result.
The plan may therefore return a nonzero status after still writing useful
diagnostic artifacts.

## Interpretation Questions

1. Which condition met the published validation-loss reproduction point?
2. How did the learning-rate change affect loss and training throughput?
3. Which digest binds the checkpoint to the training report?
4. Why does inference reject a checkpoint whose source quality did not pass?

## Suggested Rubric

| **Item** | **Points** | **Evidence** |
|:---|---:|:---|
| Training contract | 3 | Correct controls and unchanged quality target |
| Quality interpretation | 3 | Correct loss direction, target, and condition decisions |
| Lineage | 2 | Verified training → checkpoint → inference explanation |
| Systems reasoning | 2 | Throughput interpretation and next experiment |

Submit the plan aggregate and provenance manifest, both condition manifests,
the accepted condition's decode report and manifest, the checkpoint digest,
`plan.diff`, and `answers.md`. If no condition passes, submit the diagnostic
aggregate and explain why downstream inference was correctly withheld.

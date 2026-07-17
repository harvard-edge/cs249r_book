# Pro Experiment Plan Example

## Research Question

This example asks whether inference batch size changes image-classification
throughput without changing accuracy. The plan pins the workload, checkpoint,
dataset, evaluator, device policy, and outer-run count while changing one
declared environment setting.

One authoritative result per condition is enough for the current quality and
workflow milestone. The plan defaults to one outer run. A later stability study
can increase `repetitions` without changing the research question.

## Inspect and Run

Preview the fully resolved workload, execution mode, device, and repetition
count before downloading data or executing a model.

```bash
uv run mlperf run --plan examples/research/pro-collection/plan.yaml --dry-run
uv run mlperf fetch --workload image-classification --profile max
uv run mlperf run --plan examples/research/pro-collection/plan.yaml
```

The plan writes each condition into a separate `runs/` directory and produces
one suite-level JSON, CSV, and HTML report. The aggregate records the exact plan
hash, normalized study design, resolved run settings, hardware fingerprint, and
child result artifacts.

## Interpret the Result

Check quality first. Both conditions should execute the same accuracy contract
and meet the same target. Then inspect throughput. A difference is descriptive
evidence for this machine, not a stable performance claim, because this starter
plan runs each condition only once.

The normal baseline comparison command intentionally rejects a paired
performance claim when batch sizes differ. The suite report still preserves the
declared independent variable so a later research analysis can model the
controlled change without pretending the two configurations are identical.

# MLPerf EDU Example Path

Set up first if you have not already. The [project README](../README.md) has
the quickstart, and these examples additionally need the dev extra:

```bash
uv sync --locked --extra dev
```

Run everything below from the `mlperf-edu/` project directory. The
[classroom guide](GETTING_STARTED.md) covers teaching context and grading.

The numbered examples form one classroom sequence. They use registered
workloads and product CLI artifacts rather than standalone toy measurements.

| **Example** | **Question** | **Primary Artifact** |
|:---|:---|:---|
| [01 health check](01-health-check/README.md) | Is this installation ready for benchmark work? | Suite health HTML |
| [02 inference tradeoff](02-inference-tradeoff/README.md) | What did a controlled batch-size change do? | Pro condition report |
| [03 training tradeoff](03-training-tradeoff/README.md) | Did a training change produce an acceptable checkpoint? | Training and inference lineage |
| [04 result comparison](04-result-comparison/README.md) | Which comparisons are valid, and why? | Compatibility-checked HTML |
| [05 assignment package](05-assignment-package/README.md) | Can the result be verified and graded elsewhere? | Portable ZIP and grade JSON |

Each README states the learning goal, hardware expectations, allowed changes,
report sections to inspect, interpretation questions, and a suggested rubric.
Unless an instructor supplies a different template, students should submit an
`answers.md` file with numbered responses beside the generated artifacts. The
CLI grader checks the benchmark contract; the rubric also grades those written
responses.
The separate `research/pro-collection` example uses the same plan mechanism for
a research-facing study.

Before choosing a `max` workload, read the
[fourteen-workload readiness matrix](../docs/internal/STATUS.md). Every
workload runs locally, but they separate into those that reproduce their
inherited target and those recorded as a miss, and the matrix states the next
quality task for each. Instructors should publish course-machine
`max` runtime, memory, download, and disk budgets because those costs are
hardware dependent. The initial [course-image budget](../docs/internal/COURSE_BUDGETS.md)
already covers every functional `min` path on CPU and the available MPS paths.

The three legacy `lab*.py` files remain standalone teaching experiments. They
do not emit canonical benchmark artifacts and should not be presented as
registered MLPerf EDU results.

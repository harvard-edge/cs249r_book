# MLPerf EDU Example Path

Run these examples from the `mlperf-edu/` project directory. From the parent
repository checkout, prepare the environment first.

```bash
cd mlperf-edu
uv sync --locked --extra dev
```

See [Getting Started](GETTING_STARTED.md) for platform checks and installation
alternatives.

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
[fourteen-workload readiness matrix](../READINESS.md#portfolio-status). It
separates target-passing, target-gap, and research-environment workloads and
states the next quality task for each. Instructors should publish course-machine
`max` runtime, memory, download, and disk budgets because those costs are
hardware dependent. The initial [course-image budget](../COURSE_BUDGETS.md)
already covers every functional `min` path on CPU and the available MPS paths.

The three legacy `lab*.py` files remain standalone teaching experiments. They
do not emit canonical benchmark artifacts and should not be presented as
registered MLPerf EDU results.

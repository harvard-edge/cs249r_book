# Assignment Package Example

This legacy path is retained for existing links. New courses should use the
[numbered assignment-package lab](../05-assignment-package/README.md).

## Learning Goal

This lab verifies that a student can run one authoritative quality contract,
interpret the result, and submit a portable package whose workload, profile,
configuration, quality decision, and provenance match the instructor's
assignment contract.

The example uses the MLPerf Tiny ResNet8 image-classification workload. It is a
laptop-capable `max` run. Runtime depends on the local device and excludes asset
fetching and report generation.

## Student Flow

Run the suite health check first, then fetch and execute the pinned benchmark.

```bash
uv run mlperf health
uv run mlperf fetch --workload image-classification --profile max
uv run mlperf run --workload image-classification --profile max \
  --output-dir submissions/image-classification-quality-lab
uv run mlperf package \
  submissions/image-classification-quality-lab/image-classification_max.provd.json \
  --output submissions/image-classification-quality-lab.zip
```

The run writes its HTML dashboard without opening a browser. Add
`--open-report` when the student wants to inspect it immediately. Before
submitting, the student should identify the observed top-1 accuracy, target,
target decision, measured region, checkpoint source, dataset split, and device.

## Instructor Flow

Grade the package directly without unpacking it.

```bash
uv run mlperf grade \
  submissions/image-classification-quality-lab.zip \
  --assignment examples/assignment-package/assignment.yaml \
  --output submissions/image-classification-quality-lab-grade.json
```

The grader rejects path traversal, symbolic links, duplicate archive members,
unindexed files, digest or size mismatches, invalid provenance, missing quality
evidence, configuration drift, and unexpected result cardinality.

## Suggested Rubric

| **Item** | **Points** | **Evidence** |
|:---|---:|:---|
| Setup and execution | 2 | Health report and one complete `max` result |
| Quality interpretation | 3 | Correct metric, direction, target, and decision |
| Systems interpretation | 3 | Explanation of the measured region, throughput, device, and one proposed change |
| Reproducibility | 2 | Package verifies and passes the assignment contract |

The default assignment does not permit configuration changes. An instructor
can create a second contract for an optimization experiment after deciding
which fields students may change and which comparison remains valid.

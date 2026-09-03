# 05 — Assignment Package

## Learning Goal

Produce one portable functional-readiness result whose workload, profile,
claim boundary, and provenance can be verified and graded on another machine.
This lab isolates artifact portability; Examples 02 and 03 carry the
authoritative quality work.

## Runtime and Hardware

The assignment uses the fast image-classification `min` path and requires no
authoritative dataset download. It runs on a laptop CPU, Apple Silicon, or CUDA
system. The artifact proves setup and execution only; it does not claim that the
85% `max` quality gate was evaluated.

## Student Flow

```bash
uv run mlperf health
uv run mlperf run --workload image-classification --profile min \
  --output-dir submissions/05-assignment
uv run mlperf package \
  submissions/05-assignment/image-classification_min.provd.json \
  --output submissions/05-assignment.zip
uv run mlperf grade submissions/05-assignment.zip \
  --assignment examples/05-assignment-package/assignment.yaml \
  --output submissions/05-assignment-grade.json
```

The last command is the same fail-closed contract check the instructor runs.
Before submission, identify the functional decision, why the displayed max
target is context only, checkpoint source, executed device, and provenance
status in the HTML report.

## Instructor Flow

```bash
uv run mlperf grade submissions/05-assignment.zip \
  --assignment examples/05-assignment-package/assignment.yaml \
  --output submissions/05-assignment-grade.json
```

The grader accepts a directory, manifest, or portable ZIP. Package verification
rejects traversal, symbolic links, duplicate members, unindexed files, digest
or size mismatches, invalid provenance, false quality claims, configuration
drift, and unexpected result cardinality.

## Allowed Changes

This contract permits no workload or profile changes. Students may change the
output location and choose an available device. Score-bearing result packaging
remains disabled where doing so would redistribute fetch-only or
release-review dataset bytes.

## Interpretation Questions

1. Which artifact binds the report, weights, and dataset evidence?
2. What does a passing package verification establish, and why does it not
   establish the 85% quality gate?
3. Which assignment fields catch an unexpected workload or profile?
4. Why should an instructor still rerun a sample of submissions?

## Suggested Rubric

| **Item** | **Points** | **Evidence** |
|:---|---:|:---|
| Setup and execution | 2 | Complete packaged `min` artifact |
| Claim boundary | 3 | Correct explanation that quality was not evaluated |
| Systems interpretation | 2 | Measured region, throughput, and device explanation |
| Reproducibility | 3 | Verified package that passes the assignment contract |

Submit the ZIP, self-check grade JSON, and `answers.md`. The instructor should
regenerate the authoritative grade JSON rather than trusting the submitted
self-check.

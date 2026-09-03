# 01 — Suite Health Check

## Learning Goal

Confirm that the installation, device selection, all fourteen functional
paths, report export, and provenance verification work before attempting an
authoritative quality run.

## Runtime and Hardware

This example is intended for a laptop CPU, Apple Silicon, or CUDA system.
Runtime depends on the environment and first-run package initialization. No
authoritative dataset download is required for the functional `min` paths.

## Run the Check

```bash
uv run mlperf doctor
uv run mlperf health --output-dir submissions/01-health
```

The second command writes the suite health HTML without opening a browser. Add
`--open-report` only when you want to view it immediately.

`doctor` reports both the fourteen-workload registry and the four-workload
starter selection. `health` deliberately expands to all fourteen functional
paths.

## Allowed Changes

Students may select an available device and change the output directory. They
must not remove workloads from the full health run when submitting setup
evidence.

## Read the Report

Verify that the page lists fourteen functional paths, fourteen verified
manifests, zero failures, and zero warnings. A passing health report proves that
the workflow is connected. It does not prove that any authoritative quality
target was evaluated.

## Interpretation Questions

1. What do the Overall, Environment, Paths Passed, Manifests Verified, Warnings,
   and Duration cards establish?
2. Why is a diagnostic metric from a `min` run not a benchmark score?
3. Which path would you inspect first if its manifest failed verification?
4. Which `max` workload fits the machine and the next assignment goal?

## Suggested Rubric

| **Item** | **Points** | **Evidence** |
|:---|---:|:---|
| Complete execution | 3 | All fourteen paths appear in the report |
| Provenance | 3 | All fourteen manifests verify |
| Claim boundary | 2 | Explanation that `min` makes no quality claim |
| Next-step reasoning | 2 | A justified laptop-capable `max` workload choice |

Submit the validation JSON and HTML, `min-all/grade.json`, the complete
`min-all/` child report and manifest tree, and `answers.md` containing the four
numbered responses. The summary is the grading view; the child tree lets a TA
reverify every manifest.

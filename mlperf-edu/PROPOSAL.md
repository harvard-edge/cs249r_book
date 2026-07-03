# Proposal: MLPerf EDU

## Executive Summary

MLPerf EDU is a laptop-scale educational benchmark suite for teaching and
researching ML systems evaluation. The goal is not to replace MLPerf Training,
Inference, Tiny, or Client. The goal is to provide a runnable, inspectable
on-ramp that lets students and researchers learn MLPerf-style methodology on a
local machine before they move to production-scale submissions.

The current harness installs as `mlperf`, defaults to the `mlperf-edu` suite,
and provides three benchmark profiles:

| Profile | Purpose |
|---|---|
| `min` | Fast setup and artifact validation for every registered workload |
| `max` | Course-scale benchmark runs with comparable metrics and reports |
| `pro` | User-configured research envelope for repetitions, larger models, backend studies, pruning, quantization, and ablations |

It also provides four validation presets:

| Validation | Purpose | Current Fresh-Install Result |
|---|---|---:|
| `smoke` | Doctor plus default `min` run | 12 manifests, 12.3 s |
| `coverage` | All registered `min` workloads | 30 manifests, 26.5 s |
| `max` | Course-scale `max` validation | 30 manifests, 93.4 s |
| `release` | All registered `min` and `max` workloads | 60 manifests, 278.3 s |

Each validation run writes JSON, HTML, and CSV artifacts, including per-suite and
per-workload timing. This makes the suite usable in courses, CI, and
research workflows without requiring cluster hardware, gated datasets, or a
production MLPerf submission environment.

## Why MLPerf EDU Should Exist

MLPerf is the industry standard for fair, representative, reproducible ML
systems benchmarking. That strength also makes it difficult to use as a first
teaching tool: official submissions require careful rules compliance, large
assets, review, and domain-specific harness knowledge. MLPerf EDU fills the
gap between toy assignments and full MLPerf participation.

MLPerf EDU gives students and researchers:

- A working `mlperf` command that runs from a fresh clone.
- A standard benchmark vocabulary: suites, profiles, validation, reports,
  provenance, verification, and grading.
- Small but representative workloads for training, inference, TinyML,
  recommender systems, SLM serving, agents, quantization, pruning, LoRA,
  distributed training, and backend comparisons.
- Report artifacts that are easy to inspect in a browser or spreadsheet.
- A safe path for course setup checks and autograding.
- A research path for architecture-style studies in pruning, quantization,
  memory systems, accelerators, serving, compiler backends, and workload
  characterization.

The intended outcome is benchmarking literacy: students learn to reason from
model, data, hardware, scenario, constraints, metrics, and artifacts rather
than treating performance numbers as ad hoc script output.

## Relationship to Existing MLCommons Work

MLCommons already organizes benchmark development through working groups that
define, develop, and conduct MLPerf benchmarks and research projects. MLPerf EDU
should follow that model: start as an educational/community suite, define a
clear rules document, and graduate only the portions that meet MLCommons review
expectations.

Relevant precedents:

- MLCommons working groups coordinate benchmark definition and execution.
- MLPerf submission and publication are governed by review rules and
  suite-specific policies.
- MLPerf Client demonstrates the value of local client-system benchmarking for
  laptops, desktops, and workstations.
- MLPerf Tiny demonstrates that constrained-device benchmarking can be part of
  the MLPerf ecosystem while using its own appropriate rules and scale.

MLPerf EDU should be careful about naming and messaging. Early releases should
say "aligned with MLPerf methodology" or "candidate MLCommons educational
suite" until MLCommons explicitly approves stronger wording.

## Scope

The proposed suite has domain workload families, individual workload IDs, and
three run profiles:

| Suite | Scope |
|---|---|
| `language` | NanoGPT, BERT, MoE, LoRA, and white-box language workloads |
| `slm` | Off-the-shelf small language model decode, quantized decode, LoRA, and serving studies |
| `vision` | CNN/mobile vision training, compression, quantization, and backend comparisons |
| `recommender` | DLRM-style sparse lookup and memory-system behavior |
| `tiny` | TinyML keyword spotting, anomaly detection, and visual wake-word studies |
| `agent` | RAG, code generation, ReAct, and tool-call systems measurements |
| `distributed` | Local multi-process training and communication/computation studies |
| `graph` | Sparse graph and message-passing workloads |
| `timeseries` | Sequence and forecasting workloads |
| `rl` | Reinforcement-learning control-flow workloads |

The registry currently contains 30 workload rows with complete `min` and `max`
coverage. The release validation checks 60 manifests across those two standardized
profiles, while `pro` provides an opt-in research envelope.

MLPerf EDU separates workload usefulness from public score claims:

| Public status | Current count | Meaning |
|---|---:|---|
| `score-bearing` | 5 | Real-data quality target plus comparable performance metrics |
| `performance-bearing` | 4 | Comparable performance metrics with a functional check, but no public task-quality score |
| `systems-only` | 21 | Runnable systems and research workloads for architecture, backend, pruning, quantization, distributed, or agent studies |

This separation is essential for credibility. A workload can support an
excellent class or architecture paper without being advertised as a public
MLPerf EDU score.

## Out-of-the-Box Contract

A fresh local install should support:

```bash
pip install -e ".[dev]"
mlperf doctor
mlperf audit
mlperf validate smoke
mlperf validate coverage
mlperf validate max
mlperf validate release --output-dir submissions/validation
pytest
```

The harness should produce:

- Report JSON for every workload and suite.
- HTML reports by default for browser inspection.
- CSV exports by default for spreadsheet analysis.
- Provenance manifests with artifact hashes and machine metadata.
- Local verification and grading results.
- Validation summaries with pass/fail totals, per-suite artifacts, per-workload
  timing, and a stable `mlperf_suite: "mlperf-edu"` identifier.

## Proposed MLCommons Endorsement Path

### Phase 0: Community Preview

Publish MLPerf EDU as an explicitly unofficial educational preview. The goal is
to gather feedback on workload scope, CLI vocabulary, report schema, and
course usability.

Exit criteria:

- Fresh clone passes `mlperf validate release`.
- `mlperf audit` passes.
- All examples in `README.md` are runnable or clearly marked optional.
- Every workload has `min` coverage and provenance.
- Every public score-bearing workload has a documented `max` target, data
  policy, scenario, and verified baseline.

### Phase 1: MLCommons Working-Group Review

Bring the proposal to the relevant MLCommons benchmark/education stakeholders.
The ask is review and sponsorship, not immediate publication of official
competitive results.

Review topics:

- Whether `MLPerf EDU` is an acceptable name.
- Which existing working group should sponsor the effort.
- Which workloads should be in the first endorsed course-scale validation path.
- What rules, disclaimers, and result-messaging language are required.
- Whether the reports need a formal schema version and validator.

### Phase 2: Rules and Reference Release

Define a lightweight rules package:

- Valid system configurations for laptop and course runs.
- Required commands for `smoke`, `coverage`, `max`, and `release`.
- Rules for synthetic data in `min` and real or micro-sharded data in `max`.
- Allowed off-the-shelf SLMs and license requirements.
- Required report/provenance fields.
- Accuracy, quality, and pass/fail semantics.
- Score-bearing, performance-bearing, systems-only, and experimental result
  categories.
- The educational scenario subset: `single_stream`, `offline`, and `server`
  for public score/performance results.
- Result messaging that distinguishes educational validation from official
  competitive MLPerf results.

Exit criteria:

- Rules document reviewed by MLCommons stakeholders.
- Reference release passes on macOS and Linux.
- CI runs unit tests plus `mlperf validate smoke`.
- Longer validation suites are reproducible locally and available as manual or scheduled CI.

### Phase 3: Endorsed Educational Suite

If MLCommons approves, label the suite as an endorsed educational benchmark or
MLCommons educational project. The initial endorsed scope should be
methodology, tooling, and course reproducibility, not competitive ranking.

Exit criteria:

- Public release artifacts are versioned.
- Results messaging is approved.
- A small set of pilot courses run the suite and report setup friction.
- Workloads have stable IDs, schemas, and regression tests.

### Phase 4: Optional Official Track

Only after the educational suite is stable should MLPerf EDU consider an
official submission-style track. That track could use a subset of workloads,
stricter data requirements, and stronger review rules. It should not block the
course-first release.

## What We Need from MLCommons

1. Guidance on naming: whether `MLPerf EDU` is acceptable or should be
   positioned as `MLCommons EDU Benchmark` until formal endorsement.
2. A sponsoring working group or review group.
3. Feedback on the profile/validation vocabulary: `min`, `max`, `pro` and
   `smoke`, `coverage`, `max`, `release`.
4. Agreement on result-messaging boundaries so educational reports are not
   confused with official MLPerf benchmark submissions.
5. Review of the first course-scale workload list and SLM model policy.

## Near-Term Engineering Plan

The work should follow the stakeholder iteration loop in
[`ITERATION_LOOP.md`](ITERATION_LOOP.md): start with a specific audience
question, create a small review packet, use parallel private reviews when
helpful, synthesize findings into one bounded implementation slice, validate,
and then record the decision.

The implementation loop is:

1. Select one workload.
2. Confirm registry metadata, data policy, backend policy, metric, target, and
   runtime budget.
3. Implement or harden `min`.
4. Implement or harden `max`.
5. Add or harden `pro` sweeps when a research use case needs repetitions,
   backend comparisons, larger models, pruning, or quantization.
6. Emit report, provenance, verifier, package, and grade artifacts.
7. Add tests for registry, fetch, run, report, verify, package, and grade.
8. Run `mlperf validate smoke`, targeted suite validation, and finally
   `mlperf validate release`.
9. Update docs and runtime budgets from measured results.

This loop keeps the suite from becoming a catalog of aspirational benchmarks.
A workload is not done until it runs, reports, verifies, grades, and passes the
appropriate validation suite.

## References

- [MLCommons Working Groups](https://mlcommons.org/working-groups/)
- [MLCommons benchmark submission overview](https://mlcommons.org/benchmarks/)
- [MLPerf submission rules](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc)
- [MLPerf Inference submission guide](https://docs.mlcommons.org/inference/submission/)
- [MLPerf Client](https://mlcommons.org/benchmarks/client/)
- [MLPerf Tiny](https://mlcommons.org/working-groups/benchmarks/tiny/)

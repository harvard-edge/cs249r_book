# MLPerf EDU Public Rules

MLPerf EDU is intended to be a SPEC-like educational benchmark suite for ML
systems: small enough to run on course laptops, strict enough to produce
comparable artifacts, and transparent enough for students and researchers to
inspect the full stack.

These rules define what can be advertised as a public MLPerf EDU result.

## Public Status

Every workload in `workloads.yaml` declares a `public.status` value.

| Status | Meaning | Public result use |
|---|---|---|
| `score-bearing` | Real data, explicit quality target, verified baseline, and comparable performance metrics | Public quality-plus-performance result |
| `performance-bearing` | Standardized model or checkpoint path with a functional check, but no task-quality score | Public performance result |
| `systems-only` | Runnable and useful for architecture, kernel, backend, or course experiments | Research/teaching result, not a public score |
| `experimental` | Under development or unstable | Not release-ready |

A workload can be valuable and still be `systems-only`. This is deliberate:
quantization, pruning, distributed training, agent orchestration, and
microarchitectural studies often need controlled scaffolds before they are
ready to carry public task scores.

## Profiles

| Profile | Role | Contract |
|---|---|---|
| `min` | Setup and correctness | Runs quickly, may use synthetic or tiny deterministic data, emits reports and provenance |
| `max` | Standard benchmark | Runs on laptop-class hardware, emits comparable metrics and artifacts |
| `pro` | Research envelope | Repetitions, backend sweeps, pruning, quantization, larger models, and ablations |

The public release validation sequence is:

```bash
mlperf doctor
mlperf audit
mlperf audit --policy public
mlperf validate smoke
mlperf validate coverage
mlperf validate max
mlperf validate release
pytest
```

## Dataset Release Status

Dataset dossiers separate normal benchmark metadata from public-release
readiness. The public CLI surfaces both `license_status` and
`public_release_status` in `mlperf info --dataset`, `mlperf fetch --dry-run`,
`mlperf cache list --format json`, reports, CSV, HTML, package artifacts, and
public audits.

| Release status | Meaning | Public audit behavior |
|---|---|---|
| `public-ok-bundled` | Asset is maintained by MLPerf EDU and can ship with the project | No warning |
| `public-ok-with-attribution` | Asset can be used when attribution and license metadata are preserved | No warning |
| `public-ok-fetch-only` | Asset can be used only by fetching from upstream, not by redistribution | No warning if the fetch-only policy is documented |
| `restricted-needs-approval` | Terms restrict redistribution, commercial use, endorsement, or publication | Public policy warning until MLCommons/maintainer approval is recorded |
| `needs-release-decision` | Source terms are incomplete, ambiguous, or not yet reviewer-approved | Public policy warning until resolved |

Current dataset release decisions:

| Dataset | License status | Release status | Action |
|---|---|---|---|
| `prompt-suite-local` | `bundled-project-asset` | `public-ok-bundled` | Keep prompts in project-controlled artifacts |
| `mnist` | `cc-by-sa-3.0` | `public-ok-with-attribution` | Preserve attribution and license metadata in reports/packages |
| `tinyshakespeare` | `public-domain-us` | `public-ok-fetch-only` | Generate the tiny excerpt from Project Gutenberg eBook 100 and preserve the recipe/source metadata |
| `fashion-mnist` | `mit` | `public-ok-with-attribution` | Preserve MIT license and attribution metadata in reports/packages |
| `movielens-100k` | `noncommercial-research-education` | `restricted-needs-approval` | Use fetch-only and get explicit approval before public score-bearing use |
| `cifar100` | `source-citation-no-license` | `needs-release-decision` | Keep out of default score-bearing public rows unless terms are resolved |

## Scenario Scope

MLPerf EDU uses MLCommons-style scenarios as teaching concepts, not as a
mandate to clone every official scenario.

Public `score-bearing` and `performance-bearing` workloads should use one of:

| Scenario | Educational purpose |
|---|---|
| `single_stream` | Latency for one request, sample, token path, or training example |
| `offline` | Throughput-oriented batch or dataset processing |
| `server` | Serving, queueing, decode, retrieval, and latency-throughput tradeoffs |

The suite also allows `training` and `inference` as systems-only labels while a
workload is still a teaching scaffold. A workload should not be promoted to a
public score-bearing result until its scenario is one of the public result
scenarios above.

## Score-Bearing Requirements

A `score-bearing` workload must declare:

- Real dataset name and source.
- Explicit quality metric and target.
- Verified baseline evidence.
- Reference-run protocol when `target_basis` is `reference_runs`.
- `min` and `max` runners.
- Standard report JSON, HTML, and CSV exports.
- Provenance manifest with dataset/model hashes where locally available.
- Local verifier and grading compatibility.
- Scenario in `single_stream`, `offline`, or `server`.

## Reference-Run Comparability

When a score-bearing target uses `target_basis: reference_runs`, the registry
must declare `quality_target.reference_protocol`. The protocol is the minimum
information needed for another maintainer, instructor, or paper artifact
reviewer to reproduce the target-setting process.

Required fields:

| Field | Meaning |
|---|---|
| `profile` | Profile used to set the target, normally `max` |
| `backend` | Reference backend path or policy for declaring an alternate backend |
| `machine_class` | Hardware class and fingerprinting expectation |
| `dataset_mode` | Real dataset, split, preprocessing, and fallback policy |
| `seeds` | One seed per reference run |
| `aggregation` | Statistic used to turn multiple runs into the target evidence |
| `artifact_policy` | Artifacts that must be preserved for audit |
| `rerun_policy` | Changes that force a fresh reference sweep |

MLPerf EDU currently requires at least three reference runs for public
score-bearing targets and uses five runs for the first score-bearing workload
set. Reports and validation CSVs surface the reference protocol so a public
artifact can be inspected without opening `workloads.yaml`.

## Performance-Bearing Requirements

A `performance-bearing` workload must declare:

- Standard model source, shared checkpoint, or dataset path.
- Functional check that prevents empty or invalid runs from passing.
- Comparable performance metrics.
- `min` and `max` runners.
- Standard reports, provenance, verification, and grading.
- Scenario in `single_stream`, `offline`, or `server`.

Checkpoint-backed inference workloads must also preserve lineage to the
training workload that provides task quality. Reports should expose
`shared_checkpoint`, `quality_dependency`, and a structured
`checkpoint_provenance` block with the source workload, source run selector,
source quality metric, target, target basis, reference runs, verified baseline,
and artifact policy. This keeps prefill/decode/serving performance results tied
to the training evidence that produced the weights.

## Systems-Only Requirements

A `systems-only` workload must still be runnable and honest:

- `min` and `max` runners are required.
- Reports must label synthetic, random, tiny, or micro-sharded data clearly.
- Quality checks that are not meaningful must be disabled or explicitly marked.
- The public rationale must explain what systems question the workload teaches.

Systems-only workloads are appropriate for pruning, quantization, LoRA,
speculative decoding, distributed training, kernel studies, memory hierarchy
studies, and agent orchestration until their task-quality contract is strong
enough for promotion.

## Promotion Loop

Promote one workload at a time:

1. Choose the workload and intended public status.
2. Define model, data, scenario, metric, target, runtime budget, and backend
   policy in `workloads.yaml`.
3. Make `mlperf fetch --workload <id> --profile max --dry-run` truthful.
4. Make `min` pass with reports, provenance, verifier, and grade output.
5. Make `max` pass on laptop-class hardware.
6. Run `mlperf audit`, fix every local blocker, then run
   `mlperf audit --policy public` and explicitly triage every public-release
   warning.
7. Run targeted validation for the workload's suite.
8. Run `mlperf validate release`.
9. Update README/SPEC/PROPOSAL only with commands that passed.

Promotion is conservative:

- `experimental` to `systems-only`: `min` and `max` run with honest artifacts.
- `systems-only` to `performance-bearing`: standardized model/checkpoint path
  and comparable performance metrics are stable.
- `performance-bearing` to `score-bearing`: real-data quality target and
  verified baseline are stable.

## Release Bar

A public release candidate must satisfy:

- `mlperf audit` passes for development checks.
- `mlperf audit --policy public` passes for endorsement/release checks.
- `mlperf validate release` passes.
- `pytest` passes.
- Local `mlperf grade` and `mlperf validate` outputs are clean execution and
  quality checks; public-release warnings are reviewed through
  `mlperf audit --policy public`, with release-blocking warnings resolved or
  explicitly accepted by the maintainers and MLCommons reviewers.
- Every public result has JSON, HTML, CSV, provenance, verification, and grade
  artifacts.
- Documentation distinguishes MLPerf EDU educational results from official
  competitive MLPerf submissions unless MLCommons approves stronger language.

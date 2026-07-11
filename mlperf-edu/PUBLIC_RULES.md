# MLPerf EDU Public-Candidate Rules

MLPerf EDU is an independent educational benchmark preview. It is not an
official MLCommons benchmark and is not endorsed by MLCommons. The terms
`score-bearing` and `performance-bearing` below are local candidate
classifications used to decide what is ready for external review. They do not
authorize official MLPerf result claims.

## Registry Status

Every row in the native registry declares `public.status`. The current counts
are five score-bearing, three performance-bearing, and 22 systems-only rows.

| **Status** | **Required Meaning** | **Allowed Claim Before External Approval** |
|:---|:---|:---|
| `score-bearing` | Real data, explicit quality target, five-seed reference protocol, comparable performance, and complete artifacts. | MLPerf EDU score candidate for review. |
| `performance-bearing` | Standard model or quality-approved checkpoint, nonempty functional or task-quality gate, repeatable timing, and complete artifacts. | MLPerf EDU performance candidate for review. |
| `systems-only` | Runnable teaching or research workload whose current data, quality, or comparison contract is insufficient for a public score. | Classroom or systems measurement only. |
| `experimental` | Unstable or incomplete workload. | No release claim. |

A registry label alone is not evidence. The final `max` report, release
validation, retained reference packet, asset policy, and reviewer decisions
must all agree.

## Profile Semantics

| **Profile** | **Role** | **Evidence Boundary** |
|:---|:---|:---|
| `min` | Fast correctness and artifact plumbing | May use synthetic or tiny deterministic inputs. It cannot establish a public quality or performance result. |
| `max` | Candidate comparable scale | Must satisfy the row's data, quality, measurement, provenance, and report-level review contract. |
| `pro` | Controlled research envelope | Useful for repeated or variant studies only under an explicitly recorded protocol. It is not automatically public evidence. |

`smoke`, `coverage`, `max`, and `release` are validation presets, not profiles.
`smoke` executes the default `min` collection, `coverage` executes every `min`
row, `max` executes every `max` row, and `release` executes both complete sets.
A dry run prints selection only and never counts as execution evidence.

## Public-Candidate Report Contract

`src/mlperf/contracts.py` evaluates every public-candidate `max` report. An
eligible report must satisfy all common checks.

- Runner status is `passed`.
- The report records an integer seed.
- The declared metric resolves to one finite numeric report value.
- Quality or functional enforcement is enabled and its target is met.
- Report and provenance artifact paths exist.
- The `data_mode` is eligible for the candidate status.

Score-bearing rows require `data_mode: real`. Performance-bearing rows may use
`checkpoint-backed`, `local-prompt`, `local-prompt-batch`, or
`local-prompt-long-context` when the registry declares the corresponding
contract.

Performance-bearing reports have additional requirements.

- At least one warmup and at least three measured runs.
- Declared latency statistics for the measured samples.
- A checkpoint SHA-256 digest when the row depends on shared training weights.
- A pinned model revision when the row uses an external model source.
- A passing task-quality evaluation for external-model results.

`mlperf validate max` and `mlperf validate release` collect failed
`review_contract` blocks and fail the validation item. A locally passing runner
cannot bypass this report-level gate.

## Score-Bearing Rule

The current score candidates are NanoGPT training, Micro-DLRM training, MNIST
anomaly detection, ResNet-18 training, and MobileNetV2 training. Each must
declare the following registry fields and execution behavior.

- Real dataset, source, split, preprocessing, and fallback policy.
- Numeric metric, direction, threshold, and tolerance.
- `target_basis: reference_runs` with five declared runs.
- Seeds `0,1,2,3,4` and median aggregation.
- Backend and machine-class disclosure.
- Artifact and full-sweep rerun policies.
- `min` and `max` runners.
- Explicit `training` scenario under the proposed MLPerf EDU vocabulary, pending MLCommons reviewer acceptance.

The release evidence tool runs each seed in a fresh process through the product
runner, report enrichment, provenance verification, and grading path. All five
individual reports must pass their targets. The median must also pass. Each
attempt is create-once and receives an evidence summary plus an unauthenticated
SHA-256 digest sidecar.

The current thresholds are listed in
[QUALITY_TARGET_REVIEW.md](QUALITY_TARGET_REVIEW.md). The five score rows now
have committed reference summaries from clean source commit `318cd842`; older
development calibration fields are historical rationale rather than the
authoritative reference result.

## Performance-Bearing Rule

The current performance candidates are NanoGPT prefill, NanoGPT decode, and the
pinned SmolLM2 baseline.

Each declares five reference executions. Every execution must pass its report
contract and retain its within-run samples. Five reference executions do not
mean five timed samples. They wrap the repeated timing protocol declared below.

Checkpoint-backed NanoGPT inference must retain all of the following evidence.

- Source training workload and quality dependency.
- Exact checkpoint path and SHA-256 digest.
- Source quality metric, target, target basis, and reference protocol in the
  enriched report.
- Fixed prompt shape and configured prefill or decode work.
- Synchronized repeated timing with raw or aggregate latency samples.
- Positive throughput and completed functional work.

The prefill default uses three discarded warmups and ten measurements. The
decode default uses one discarded warmup and five measured requests. Prefill
reports median, p90, and p99 latency. Decode reports TTFT and inter-token
median, p90, and p99 latency across the measured requests.

The SmolLM2 baseline must retain its pinned revision, model metadata, bundled
four-case fixture digest, continuation-only NLL and perplexity, generation
length, and repeated timing. Its default gate requires at least eight generated
tokens and perplexity at most 10. Its default protocol uses one warmup and five
measured requests with separate prefill and generation median, p90, and p99
latencies.

The dynamic-int8 SLM path is systems-only because its current calibration fails
the task-quality parity limits. Completing generation is not enough to promote
it.

## Current Committed Reference Set

`reference_results/index.json` contains eight content-addressed summaries from
clean source commit `318cd842efe3b90cbf56a109797d2bed4ad3dc09`. All eight
record `status: valid`, `eligible_for_public_baseline: true`, seeds 0 through 4,
passing individual contracts, and passing aggregate acceptance.

The score medians are `2.0568` NanoGPT cross-entropy loss, `0.7041` DLRM
accuracy, `0.9666` anomaly AUROC, `0.8750` ResNet-18 top-1 accuracy, and
`0.8089` MobileNetV2 top-1 accuracy. The performance medians are `117797.22`
NanoGPT prefill tokens/s, `175.8925` NanoGPT decode tokens/s, and `127.9239`
SmolLM2 output tokens/s. These are project candidate observations on the
recorded Apple M5 Max system, not official MLPerf results or cross-system
performance claims.

The repository commits compact summaries and digests, not every raw artifact.
Complete create-once attempts are retained for local handoff, and
reviewer-facing public URLs remain unassigned. The DLRM raw packet is local-only
while the MovieLens policy decision remains open. A summary does not close
hosted CI, independent reproduction, representative browser inspection,
dataset policy, target and scenario review, naming, or result-wording approval.

## Systems-Only Rule

A systems-only row must still be honest and executable.

- Both `min` and `max` runners exist.
- Reports label synthetic, random, tiny, local, or micro-sharded inputs.
- Meaningless quality gates are disabled or explicitly described as functional
  checks.
- The registry explains the systems question and why the row is not a public
  candidate.
- Documentation does not reuse its configured teaching threshold as a released
  score.

Systems-only rows are appropriate for architecture, memory, precision,
quantization, compression, distributed, agent, and control-flow studies while
their public task contract remains incomplete.

## Dataset and Model Policy

Dataset dossiers expose `license_status` and `public_release_status`. The
structured statuses are project policy inputs, not legal conclusions.

| **Release Status** | **Project Behavior** |
|:---|:---|
| `public-ok-bundled` | The project intends to ship the asset with attribution and an applicable component license. |
| `public-ok-with-attribution` | Preserve the named upstream attribution and license fields in reports and packages. |
| `public-ok-fetch-only` | Fetch from upstream and preserve recipe and source metadata. Do not redistribute the upstream asset. |
| `restricted-needs-approval` | Block strict public policy until written approval, an accepted policy, or row demotion. |
| `systems-only-with-attribution` | Keep the row systems-only and preserve attribution. |
| `systems-only-review-pending` | Keep the row systems-only while source or release review remains incomplete. |

The complete 14-entry catalog and the MovieLens decision paths are in
[DATASET_RELEASE_REVIEW.md](DATASET_RELEASE_REVIEW.md). External model rows must
also pin their revision and preserve model-license metadata.

## Provenance and Package Integrity

Every canonical run emits JSON, HTML, CSV, and `.provd.json` artifacts. The
manifest records source-tree evidence, weights, dataset files, seed, hardware,
optional roofline sidecar, canonical report content, and exact report bytes
when available.

The manifest integrity record is an unauthenticated digest. It detects changes
but does not prove the identity of the producer. Documentation and reports must
not call it a digital signature.

`mlperf verify` recomputes available evidence from disk. `mlperf package`
requires a verified manifest, rewrites machine-local paths to archive-relative
paths, includes referenced artifacts, indexes SHA-256 and byte size, and
verifies the package after extraction in a clean temporary directory.
Packaging fails closed when a known dataset dossier forbids redistribution or
still requires a release decision. MovieLens evidence remains locally
verifiable, but its dataset bytes cannot enter an MLPerf EDU archive.

Verification establishes internal integrity. It does not establish fair
measurement, legal permission, independent reproduction, or MLCommons
acceptance.

## Audit and Validation Boundary

```bash
uv run mlperf audit
uv run mlperf audit --policy public --format json
uv run mlperf validate smoke
uv run mlperf validate coverage
uv run mlperf validate max --keep-going
uv run mlperf validate release --keep-going
uv run pytest
uv run python tools/build_wheel.py
```

The development audit checks registry consistency and currently serves as a
blocking local gate. The strict public audit fails on unresolved release-policy
warnings, including the current MovieLens-100K warning. Audit does not execute
models. Actual validation must run, grade, and verify the artifacts on the same
source revision used for review.

## Promotion Rule

Promotion happens one row at a time.

1. Define the model, data, scenario, metric, target, runtime budget, and backend
   policy.
2. Make both profiles execute with honest data labels and complete artifacts.
3. Calibrate the target without presenting calibration as released evidence.
4. Produce the complete final-source reference packet.
5. Pass development audit, strict public audit, targeted validation, full
   release validation, package portability, and documentation checks.
6. Obtain the applicable dataset, target, scenario, naming, and result-wording
   decisions.

`experimental` may become `systems-only` after honest execution is stable.
`systems-only` may become `performance-bearing` after its standardized work and
timing contract are reviewable. `performance-bearing` may become
`score-bearing` only after a real-data task-quality contract and complete
reference evidence are stable.

## Release Claim Rule

An independent preview may show implemented machinery and explicit open gates.
It may not claim an official score, endorsement, or MLCommons publication. A
public candidate release requires every in-repository gate in
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) and all applicable external
decisions. The current eight committed summaries satisfy the reference-evidence
step only. Stronger wording requires written MLCommons approval.

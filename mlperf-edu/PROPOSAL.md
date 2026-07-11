# Proposal for MLPerf EDU Review

## Review Status

This document proposes a bounded technical review. MLPerf EDU is currently an
independent project preview. It is not an official MLCommons benchmark and is
not endorsed by MLCommons. The project should use stronger naming or result
language only after written approval.

## Executive Summary

MLPerf EDU is a laptop-scale benchmark and teaching harness for ML systems. It
aims to give students, instructors, researchers, and artifact evaluators a
small but disciplined path through workload selection, task quality,
repeatable measurement, provenance, verification, and packaging.

The current native registry contains 30 executable rows in 23 workload
families across 10 suites. Every row declares `min` and `max` runners. The
project classifies five score-bearing candidates, three performance-bearing
candidates, and 22 systems-only rows. Those classifications are internal
review labels rather than accepted MLCommons result categories.

The initial request to MLCommons is feedback and a review home. It is not a
request to publish competitive results. The most useful first review would
address the project name, sponsoring group, candidate workload set, quality
targets, inference scenarios, asset policy, report schema, and result wording.

The repository retains eight committed, content-addressed summaries produced
from clean source commit `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`.
They are current review evidence for local handoff under the present contracts:
five score-bearing candidates and three performance-bearing candidates, each
with five fresh-process executions, provenance verification, grading, immutable
attempts, artifact indexes, and digest sidecars.

## Problem the Project Addresses

Production benchmark suites teach important discipline, but their scale,
assets, optimized implementations, and submission processes can be difficult
to use as a first classroom or academic artifact. Smaller course exercises are
easier to run, yet they often lack fixed quality gates, environment disclosure,
repeated timing, checkpoint lineage, and portable evidence.

MLPerf EDU explores a middle layer. The core path requires no cluster or paid
API, keeps many teaching models inspectable, and produces artifacts that a
reviewer can verify. The project does not seek to replace MLPerf Training,
Inference, Tiny, Client, or official submission rules.

## Current Implementation Snapshot

| **Area** | **Implemented Surface** | **Review Boundary** |
|:---|:---|:---|
| Registry | 30 rows, 23 families, 10 suites, `min` and `max` runners, native YAML plus generated mirrors | Final generated-file and full execution gates must pass on one frozen revision. |
| CLI | `doctor`, `init`, `list`, `show`, `info`, `fetch`, `run`, `report`, `verify`, `package`, `grade`, `validate`, `audit`, `cache` | Supported preview install is a locked source checkout; no package-index release is claimed. |
| Reports | JSON, HTML, CSV, quality state, hardware and software fingerprint, dataset and model dossiers | Reports remain project artifacts until the release and external policy gates close. |
| Provenance | Exact report bytes and semantics, source, dataset, weights, seed, hardware, optional sidecars, unauthenticated digest | Integrity does not authenticate the producer or establish result acceptance. |
| Packaging | Schema 0.2 archive, relative paths, complete SHA-256 and byte-size index, clean-extraction verification | The retained NanoGPT lineage package passed all 56 verification checks; public distribution and reviewer URL remain undecided. |
| Reference evidence | Eight committed summaries from clean source `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`, each with five fresh-process runs, grading, verification, immutable attempts, artifact indexes, and digest sidecars | Current local-handoff summaries are committed; public URLs, hosted CI evidence, and independent reproduction remain open. |
| Inference | Checkpoint-backed NanoGPT prefill and decode; pinned SmolLM2 with continuation-perplexity gate | Scenario, fixture, target, and timing policy need domain review. |
| Education | Three CPU and network-free lab smokes plus one implemented tutorial with provenance verification | Longer tutorial program remains roadmap work. |
| Website | Registry and dataset generated pages, CLI reference, Quarto build, link checks, preview and guarded live workflows | Workflow presence does not prove deployment, and representative desktop and narrow in-app browser inspection is not yet recorded. |
| Validation | Fast blocking workflow plus scheduled or manual actual `max` and `release` workflow with a five-hour timeout | Green hosted same-revision workflow artifacts remain a release requirement. |

## Candidate Result Set

The score-bearing candidates use real datasets and seeds 0 through 4. Each
individual run must pass its target, provenance verification, grading, and
report contract. The five-run median must also pass.

| **Candidate** | **Data or Model** | **Gate** | **Current Evidence Boundary** |
|:---|:---|:---|:---|
| NanoGPT training | Deterministic excerpt generated from Project Gutenberg eBook 100 | cross-entropy loss `<= 2.30` | Current five-run packet committed. |
| Micro-DLRM training | MovieLens-100K official split, without rating-derived aggregate features | fixed-final-epoch ROC AUC `>= 0.76` | Current five-run packet committed; raw MovieLens-derived artifacts remain local-only pending policy review. |
| MNIST anomaly autoencoder | Digit 5 normal; digits 3, 8, and 9 anomalous | macro AUROC `>= 0.93`, worst-class AUROC `>= 0.90`, and learned-control margin `>= 0.20` | Current five-run packet committed. |
| ResNet-18 training | Fashion-MNIST | top-1 accuracy `>= 0.85` | Current five-run packet committed. |
| MobileNetV2 training | Fashion-MNIST | top-1 accuracy `>= 0.78` | Current five-run packet committed. |
| NanoGPT prefill | Quality-approved NanoGPT checkpoint | positive checkpoint-backed prefill throughput with fresh cache materialization | Current five-run packet committed under the twenty-sample protocol. |
| NanoGPT decode | Same quality-approved checkpoint | 64 decode steps and positive throughput | Current five-run packet committed under causal TTFT and ITL boundaries. |
| SmolLM2 baseline | Pinned 135M model revision and 28 attributed continuation cases across seven categories | at least eight tokens, token-weighted perplexity `<= 7`, and worst-category perplexity `<= 24` | Current five-run packet committed; cross-execution CV is below the `5%` limit. |

Current committed evidence from source
`86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`:

| **Workload** | **Evidence ID** | **Primary Metric Median** | **Minimum** | **Maximum** | **CV** |
|:---|:---|---:|---:|---:|---:|
| `anomaly-ae-train` | `anomaly-ae-train_max_20260711T204007.498158Z` | `4.3060` | `4.2488` | `4.5012` | n/a |
| `micro-dlrm-train` | `micro-dlrm-train_max_20260711T205839.712863Z` | `2.0394` | `1.9475` | `2.0924` | n/a |
| `mobilenetv2-train` | `mobilenetv2-train_max_20260711T204653.574587Z` | `103.6699` | `103.4276` | `104.7116` | n/a |
| `nanogpt-decode` | `nanogpt-decode_max_20260711T210118.136527Z` | `316.7836` | `313.2178` | `319.4718` | `0.75%` |
| `nanogpt-prefill` | `nanogpt-prefill_max_20260711T205947.060179Z` | `3859.7758` | `3849.9537` | `3868.4721` | `0.19%` |
| `nanogpt-train` | `nanogpt-train_max_20260711T202223.716219Z` | `117.5981` | `117.0730` | `118.2963` | n/a |
| `resnet18-train` | `resnet18-train_max_20260711T204117.913822Z` | `58.4192` | `57.3181` | `59.5123` | n/a |
| `slm-decode` | `slm-decode_max_20260711T210317.517476Z` | `61.5742` | `60.4083` | `61.9703` | `1.11%` |

The dynamic-int8 SLM variant remains systems-only because its historical v1
calibration failed quality parity and a v2 calibration is still required. This
is an intentional example of quality taking precedence over a performance
label.

## Measurement and Evidence Model

The candidate contract has four layers.

1. The registry defines the model, data, scenario, metric, target, runner,
   reference protocol, asset policy, and result classification.
2. The runner performs the work and writes a structured report plus provenance
   manifest.
3. The report-level contract rejects ineligible data modes, missing seeds,
   disabled quality gates, absent timing protocols, unidentified checkpoints,
   unpinned external models, failed task quality, and missing artifacts.
4. Verification, grading, portable packaging, and release validation test the
   evidence as a whole.

The manifest digest is deliberately described as unauthenticated. Anyone can
recompute it. It detects accidental or post-publication changes but is not a
producer signature. Independent reproduction and governance remain necessary.

## Classroom and Notebook Contract

The standard smoke path and three lab smokes run on CPU without a network after
dependencies are installed. Candidate training runs fetch datasets, and the
SLM candidate fetches pinned model weights before measurement. A GPU is
optional.

The project avoids a universal runtime promise because laptop hardware and
framework behavior vary. Reports retain measured durations and fingerprints.
The long CI workflow has a five-hour safety limit, while normal course use can
select one workload or the fast `min` profile.

The implemented labs cover training-loop optimization, naive versus KV-cache
decode with token parity, and dense versus sparse training on identical
batches. Their outputs are explicitly noncanonical classroom measurements.
The product CLI currently uses registered runners and does not accept an
arbitrary `--sut` plugin file.

## Scope of the First Review

The first review should stay narrower than the 30-row research registry. The
proposed focus is the eight public candidates, the common artifact contract,
and the educational workflow. Systems-only rows remain available for teaching
and architecture experiments without being presented as accepted scores.

The requested review topics follow.

- Whether `MLPerf EDU` is an acceptable name during and after review.
- Which MLCommons group, if any, is the right review or sponsorship home.
- Whether score-bearing and performance-bearing are useful internal terms or
  should be renamed.
- Whether Fashion-MNIST, the MNIST anomaly protocol, TinyShakespeare recipe,
  and MovieLens path are suitable for the first candidate set.
- Whether median-of-five plus individual-run target enforcement is sufficient
  for course-scale training targets.
- Whether NanoGPT prefill and decode scenarios, warmups, repetitions, and
  latency statistics are appropriate.
- Whether the pinned SmolLM2 model, attributed 28-case continuation fixture,
  token-weighted aggregation, overall and weakest-category perplexity limits,
  and token floor are adequate for a first serving candidate.
- Which report fields and schema-stability guarantees reviewers require.
- Which wording cleanly separates independent educational candidate results
  from official competitive MLPerf results.

## Known Gates Before an External Review Packet

The project should not arrive with avoidable in-repository failures. The packet
must include all of the following, with measurements anchored to source commit
`86738e46` and any later evidence-only commit identified separately.

- Complete tests and generated-source checks.
- Actual `smoke`, `coverage`, `max`, and `release` validation artifacts.
- The eight committed summaries, their digest index, and a handoff map for all
  create-once raw attempts. Public artifact URLs may remain explicitly
  unassigned for the bounded review.
- The verified NanoGPT training package, checkpoint digest, prefill packet, and
  decode packet.
- A replacement pinned SmolLM2 packet with v2 task-quality evidence, exact
  fixture digest, and overall plus weakest-category gate results.
- Provenance verification, grading, and portable package examples.
- Clean wheel installs on Linux and the primary macOS laptop environment.
- Generated review packets, rendered website, link checks, and representative
  browser visual review. The latter remains an explicit local verification gap.
- A synchronized, verified companion-paper PDF.
- The strict public audit JSON showing either no warning or the explicit raw-package
  publication and MovieLens policy warnings.

External gates should be presented rather than concealed. They include the
component license, MovieLens policy, independent reproduction, target and
scenario approval, naming, sponsorship, and result wording.

## Proposed Review Path

### Phase 0 Independent Review Preview

Freeze a defensible candidate revision, retain all in-repository evidence, and
publish only independent-preview language. Invite course and artifact-review
feedback.

Exit evidence includes a clean release ledger, source install, portable wheel,
complete validation, score packets, inference chain, rendered site, and
verified paper.

### Phase 1 Bounded MLCommons Technical Review

Present the eight-candidate packet and ask for a sponsor or review home. The
request is methodology and governance feedback rather than immediate result
publication.

Exit decisions cover name, scope, asset policy, targets, scenarios, report
schema, and result language.

### Phase 2 Rules and Reference Release

Revise the candidate set and publish a lightweight rules package. Define valid
systems, required commands, dataset and model handling, quality and timing
semantics, artifact fields, allowed variation, and result presentation.

Exit evidence includes macOS and Linux reproduction, stable schemas, reviewed
assets, versioned artifacts, and pilot-course reports.

### Phase 3 MLCommons-Reviewed Educational Project

Use an endorsed or sponsored label only if MLCommons approves it. The initial
scope should emphasize education, reproducibility, and methodology rather than
competitive ranking.

### Phase 4 Optional Submission Track

Consider a stricter submission-style subset only after the educational release
is stable and independently reproduced. This phase should not block the
course-first project.

## Non-Goals for the First Release

- Replacing any official MLPerf suite.
- Claiming hardware leadership or a global leaderboard.
- Treating synthetic `min` results as task-quality evidence.
- Promoting every systems-only row at once.
- Requiring a cluster, paid API, or vendor-specific backend for the standard
  path.
- Calling integrity digests producer signatures.
- Publishing a package or site before its guarded release gates pass.

## Materials for Review

- [README.md](README.md) for the supported user path and current boundary.
- [PUBLIC_RULES.md](PUBLIC_RULES.md) for candidate result rules.
- [QUALITY_TARGET_REVIEW.md](QUALITY_TARGET_REVIEW.md) for target and inference
  evidence.
- [DATASET_RELEASE_REVIEW.md](DATASET_RELEASE_REVIEW.md) for all dataset
  classifications and MovieLens options.
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) for the executable gate ledger.
- `review_packets/` for generated row-level packets.
- `site/` for the generated review website.
- `paper/` for the draft companion paper and evidence snapshot.

## Reference Context

- [MLCommons working groups](https://mlcommons.org/working-groups/)
- [MLCommons benchmarks](https://mlcommons.org/benchmarks/)
- [MLPerf submission rules](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc)
- [MLPerf Inference submission guide](https://docs.mlcommons.org/inference/submission/)
- [MLPerf Client](https://mlcommons.org/benchmarks/client/)
- [MLPerf Tiny](https://mlcommons.org/working-groups/benchmarks/tiny/)

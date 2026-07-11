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
| Packaging | Schema 0.2 archive, relative paths, complete SHA-256 and byte-size index, clean-extraction verification | Must be demonstrated from a final candidate run. |
| Score evidence | Fresh-process five-seed tool with grading, verification, immutable attempts, artifact indexes, and digest sidecar | Final-source packets remain to be retained and reviewed. |
| Inference | Checkpoint-backed NanoGPT prefill and decode; pinned SmolLM2 with continuation-perplexity gate | Scenario, fixture, target, and timing policy need domain review. |
| Education | Three CPU and network-free lab smokes plus one implemented tutorial with provenance verification | Longer tutorial program remains roadmap work. |
| Website | Registry and dataset generated pages, CLI reference, Quarto build, link checks, preview and guarded live workflows | Workflow presence does not prove that the final revision has deployed. |
| Validation | Fast blocking workflow plus scheduled or manual actual `max` and `release` workflow with a five-hour timeout | Green same-revision workflow artifacts remain a release requirement. |

## Candidate Result Set

The score-bearing candidates use real datasets and seeds 0 through 4. Each
individual run must pass its target, provenance verification, grading, and
report contract. The five-run median must also pass.

| **Candidate** | **Data or Model** | **Gate** | **Current Evidence Boundary** |
|:---|:---|:---|:---|
| NanoGPT training | Deterministic excerpt generated from Project Gutenberg eBook 100 | cross-entropy loss `<= 2.30` | Five-seed development calibration recorded; clean final-revision packet pending. |
| Micro-DLRM training | MovieLens-100K | best validation accuracy `>= 0.70` | Five-seed calibration recorded; clean packet and dataset-policy decision pending. |
| MNIST anomaly autoencoder | MNIST, zero treated as normal | anomaly AUROC `>= 0.95` | Five-seed full-test development calibration recorded; clean packet pending. |
| ResNet-18 training | Fashion-MNIST | top-1 accuracy `>= 0.85` | Five-seed full-test development calibration recorded; clean packet pending. |
| MobileNetV2 training | Fashion-MNIST | top-1 accuracy `>= 0.78` | Five-seed full-test development calibration recorded; clean packet pending. |
| NanoGPT prefill | Quality-approved NanoGPT checkpoint | positive checkpoint-backed prefill throughput | Five reference executions; each has a checkpoint digest, three warmups, ten measured runs, median, p90, p99. Final chain pending. |
| NanoGPT decode | Same quality-approved checkpoint | 64 decode steps and positive throughput | Five reference executions; each has a checkpoint digest, one warmup, five measured requests, TTFT and inter-token percentiles. Final chain pending. |
| SmolLM2 baseline | Pinned 135M model revision and four bundled continuation cases | at least eight tokens and perplexity `<= 10` | Five development executions passed; output throughput median `74.04` tokens/s, range `73.67`-`87.12`. Clean final packet pending. |

The dynamic-int8 SLM variant remains systems-only because its current
calibration completes generation but fails the quality-parity limit. This is an
intentional example of quality taking precedence over a performance label.

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
- Whether the pinned SmolLM2 model, four-case continuation fixture, perplexity
  limit, and token floor are adequate for a first serving candidate.
- Which report fields and schema-stability guarantees reviewers require.
- Which wording cleanly separates independent educational candidate results
  from official competitive MLPerf results.

## Known Gates Before an External Review Packet

The project should not arrive with avoidable in-repository failures. The packet
must include all of the following from one frozen source revision.

- Complete tests and generated-source checks.
- Actual `smoke`, `coverage`, `max`, and `release` validation artifacts.
- Five create-once score evidence directories and digest sidecars.
- One quality-passing NanoGPT checkpoint followed by prefill and decode reports.
- One pinned SmolLM2 baseline report with task-quality evidence.
- Provenance verification, grading, and portable package examples.
- Clean wheel installs on Linux and the primary macOS laptop environment.
- Generated review packets, rendered website, link checks, and visual review.
- A synchronized, verified companion-paper PDF.
- The strict public audit JSON showing either no warning or the explicit
  unresolved MovieLens warning.

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

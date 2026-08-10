# MLPerf EDU Independent Audit

> **Superseded in part, 2026-08-04.** Recommendation moved from DLRM on Criteo
> Terabyte to MLPerf Training v0.5 NCF on MovieLens-20M, reinforcement
> learning moved from a CUDA and TensorFlow 1.x container to a PyTorch
> adapter, and the timing protocol dropped from five runs to one. No
> workload is environment-gated. Statements below about gated execution,
> licensed Criteo data, or five-run promotion describe the state at the
> time of the audit and are retained as a record rather than corrected.
> Current state: [WORKLOAD_STATUS.md](WORKLOAD_STATUS.md) and
> [MISS_DIAGNOSIS.md](MISS_DIAGNOSIS.md).


## Verdict

The fourteen-workload portfolio is coherent and covers distinct systems
behaviors. The target contracts are now acceptable for initial classroom and
research use, with three explicitly conditional interpretations. This is not a
claim that all workloads currently pass. Eight have complete target-passing
results, four have measured quality gaps, and two require external research
environments.

This review was conducted as an internal simulated independent audit on
2026-07-18. It represents three reader roles rather than an external
endorsement.

- A student checked installation, terminology, report readability, and the
  fourteen-workload `min` journey.
- An instructor checked assignment feasibility, grading boundaries, failure
  behavior, and laptop versus research requirements.
- A benchmark reviewer checked workload rationale, target authority, dataset
  identity, evaluator fidelity, and provenance against primary sources.

## Target Sign-Off

| **Workload** | **Target Review** | **Current Result State** | **Decision** |
|:---|:---|:---|:---|
| `image-classification` | MLPerf Tiny inherited gate | Passing | Accepted |
| `keyword-spotting` | MLPerf Tiny inherited gate | Passing | Accepted |
| `anomaly-detection` | MLPerf Tiny inherited gate | Passing | Accepted |
| `visual-wake-words` | MLPerf Tiny inherited gate | Passing | Accepted |
| `causal-language-modeling` | Published nanoGPT result | Passing | Conditional. Describe it as a reproduction point rather than a universal threshold. |
| `text-classification` | Pinned model-index result | Passing | Accepted |
| `information-retrieval` | Published NanoBEIR evaluator result | Passing | Accepted |
| `graph-node-classification` | Published OGB mean and standard deviation | Passing | Conditional. The one-sided tolerance needs domain approval before publication. |
| `time-series-forecasting` | Published PatchTST result | Missed | Accepted after correction to the strict 0.290 reproduction point. |
| `code-generation` | Published Qwen2.5-Coder HumanEval+ result | Missed | Accepted |
| `function-calling` | Frozen BFCL V4 leaderboard result | Missed | Accepted after pinning the leaderboard/evaluator revisions and regenerating the complete 1,150-case artifact ledger. |
| `recommendation` | MLPerf Inference inherited gate | Environment gated | Accepted |
| `image-generation` | Published NVIDIA EDM result | Missed | Accepted after defining one result as the upstream three-trial minimum. |
| `reinforcement-learning` | Historical MLPerf Training gates | Environment gated | Accepted |

The two blocking contract defects found during review were corrected. PatchTST
no longer borrows an MLPerf Inference quality fraction for a training-paper
MSE. EDM now produces three independent 50,000-image trials and uses their
minimum as one authoritative result packet. These changes do not convert the
measured misses into passes.

## Dataset and Provenance Findings

- OGB identifies `ogbn-arxiv` as ODC-By. The catalog now records attributed
  fetch-only use instead of an unresolved license.
- HumanEval+, SST-2, and NanoBEIR reports now bind every consumed checkpoint,
  tokenizer, vocabulary, and configuration file rather than only the weight
  tensor.
- The earlier BFCL ledger contained one 40-character identifier in a list
  labeled as SHA-256. The current packet replaces it with six verified
  per-category SHA-256 digests and a complete canonical sample artifact.
- HumanEval+ generated-code execution and EDM pickle loading remain security
  boundaries. The generated-code container now adds a host-user mapping, init,
  and core-file and open-file limits to its existing isolation. EDM rehashes
  the same open file immediately before deserialization. These controls are
  suitable for a controlled source-checkout preview, not an unattended
  multi-tenant service. [SECURITY_REVIEW.md](../../SECURITY_REVIEW.md) records the
  production gates.
- DLRM and MiniGo retain their full upstream contracts. They fail closed when
  the licensed data, memory, legacy runtime, or GPU environment is absent.

## Simulated User Feedback

The student and instructor reviews approved the current `min` health run for an
introductory classroom pilot. Their highest-priority findings were addressed.

- The report no longer pairs a functional diagnostic number with an unrelated
  max-profile quality target.
- `mlperf health` now runs all fourteen `min` paths, verifies provenance, keeps
  going after a failure, and always writes a suite report.
- Suite-filtered validation now checks the selected workloads rather than a
  starter subset.
- Doctor checks identify the profile-aware DLRM and MiniGo environment gates.
- The report clearly labels `min` as setup evidence with no baseline or quality
  claim.

The first review identified missing assignment contracts, packaged-result
grading, complete examples, and compatibility-checked comparison. Those product
features are now implemented.

## Repeated Product Review

A second simulated review used the rendered dashboard, numbered labs, generated
website, CLI dry runs, and desktop and narrow layout evidence. The reviewers did
not receive a promised verdict; they were asked to find blockers.

The student, TA, and research reviewers initially returned no-go decisions. The
highest-risk findings were valid and are now enforced in code and tests.

- A plan cannot change the registry quality target, inherit an ambient target
  override, or report a different metric, target, direction, tolerance, or
  decision.
- A controlled performance delta requires every participating child manifest
  to verify. The aggregate fails closed unless it binds every nonfailed child
  manifest.
- Assignment grading rechecks the canonical registry quality contract and
  recomputes the decision instead of trusting a submitted `target_met` field.
- The dashboard leads with child-manifest verification and experimental public
  status. It no longer calls a passing value an authoritative public result.
- The portable-package lab now uses an artifact-free `min` result. The package
  command rejects fetch-only and release-review dataset bytes instead of
  contradicting the dataset policy.
- The health, inference, comparison, and packaging labs now request complete
  evidence trees or unambiguous source artifacts. All five labs are rendered in
  the website navigation.

The repeated review now supports a supervised pilot for all five labs. Lab 03
can bind a provenance-verified instructor baseline into a candidate-only plan,
and the CLI enforces allowed plan edits against the instructor reference.
Measured functional CPU and accelerator-requested resource budgets are
published for the first course system. Authoritative max budgets on each
selected course image remain open. This limit is not a reason to change the
fourteen benchmark identities or lower their quality targets.

## Setup-to-Results Usability Review

A third review followed the complete setup, first-run, first-quality-result,
and research-plan journey. Independent simulated student, TA, and research
readers reviewed the source instructions and rendered desktop and narrow pages.
The review also compared the entry path with the progressive-disclosure
patterns used in the official uv, PyTorch, Hugging Face, and Gymnasium guides.

The review found two correctness defects, which are now covered by tests. The
time-series page had preserved a pass decision from an earlier provisional
target instead of recomputing against the current 0.290 contract. Packaging
instructions also recommended a score-bearing max artifact whose restricted
dataset policy required the CLI to refuse it. The page now reports the current
time-series miss, and the portable exercise uses a dataset-free min artifact
without making a quality claim.

The usability findings led to the following changes.

- The homepage and Getting Started page use one canonical sequence based on
  `uv sync --locked` and `mlperf init`.
- Prerequisites, expected files, command roles, and success states are visible
  before the first quality run.
- Image classification replaces causal language modeling as the first max
  workload.
- Student, instructor, and researcher routes lead to purpose-specific guidance.
- The results guide begins with an execution-to-publication claim ladder.
- A troubleshooting page covers environment selection, assets, devices, target
  misses, provenance, packaging policy, interrupted runs, and TA diagnostics.
- The research guide states the one-run evidence boundary and preserves plans,
  child reports, and manifests as one review packet.
- DLRM and MiniGo pages now lead with environment preflight and handoff commands
  instead of generic local max and pro commands.
- The documentation states that general `run --resume` behavior is planned but
  not implemented.

All 40 rendered pages pass HTTP, console, content-shell, horizontal-overflow,
and table-scroll-cue checks at desktop and 390-pixel viewports. The review still
does not replace observation of students and instructors in a real course.
Authoritative max budgets, general resume behavior, and local DLRM and MiniGo
backends remain open.

## Primary Sources

- [MLPerf Tiny rules](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/MLPerfTiny_Rules.adoc)
- [nanoGPT reference implementation](https://github.com/karpathy/nanoGPT)
- [NanoBEIR cross-encoder evaluator](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html)
- [OGB node property datasets and licenses](https://ogb.stanford.edu/docs/nodeprop/)
- [PatchTST paper and review record](https://openreview.net/forum?id=Jbdc0vTOcol)
- [Qwen2.5-Coder technical report](https://arxiv.org/abs/2409.12186)
- [NVIDIA EDM reference implementation](https://github.com/NVlabs/edm)
- [MLPerf Training benchmark paper](https://arxiv.org/abs/1910.01500)
- [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)
- [PyTorch quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [Hugging Face Transformers quicktour](https://huggingface.co/docs/transformers/quicktour)
- [Gymnasium basic usage](https://gymnasium.farama.org/introduction/basic_usage/)

## Sign-Off Boundary

The suite is approved for the next spiral as an experimental, supervised
classroom and research design preview. This approval covers workload selection,
initial target contracts, one-result quality evaluation, the `min` health
journey, provenance-gated controlled comparisons, canonical assignment grading,
and fail-closed authoritative runners. It does not cover an unsupervised course
release, promoted performance baselines, timing stability, public production
release, or external benchmark governance.

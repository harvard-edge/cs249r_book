# MLPerf EDU Independent Audit

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
| `function-calling` | Frozen BFCL V4 leaderboard result | Missed | Conditional. Freeze the mutable leaderboard snapshot and regenerate the complete artifact ledger. |
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
- The BFCL ledger contained one 40-character identifier in a list labeled as
  SHA-256. The unverified per-category list is removed until a complete current
  run regenerates it.
- HumanEval+ generated-code execution and EDM pickle loading remain security
  review boundaries. They are suitable for a controlled source-checkout
  preview, not an unattended multi-tenant service.
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

The remaining classroom blockers are product features rather than benchmark
identity defects. The suite still needs an assignment contract, packaged-result
grading, examples, and compatibility-checked baseline comparison.

## Primary Sources

- [MLPerf Tiny rules](https://github.com/mlcommons/tiny/blob/4addd0fa08d216e20637637874e084895f289da4/benchmark/MLPerfTiny_Rules.adoc)
- [nanoGPT reference implementation](https://github.com/karpathy/nanoGPT)
- [NanoBEIR cross-encoder evaluator](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html)
- [OGB node property datasets and licenses](https://ogb.stanford.edu/docs/nodeprop/)
- [PatchTST paper and review record](https://openreview.net/forum?id=Jbdc0vTOcol)
- [Qwen2.5-Coder technical report](https://arxiv.org/abs/2409.12186)
- [NVIDIA EDM reference implementation](https://github.com/NVlabs/edm)
- [MLPerf Training benchmark paper](https://arxiv.org/abs/1910.01500)

## Sign-Off Boundary

The suite is approved for the next spiral as an experimental classroom and
research preview. This approval covers workload selection, initial target
contracts, one-result quality evaluation, the `min` health journey, and
fail-closed authoritative runners. It does not cover promoted performance
baselines, timing stability, public production release, or external benchmark
governance.

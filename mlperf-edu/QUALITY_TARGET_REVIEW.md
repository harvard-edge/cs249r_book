# MLPerf EDU v0.1 Quality Target Review

## Review Boundary

The v0.1 portfolio contains seven workloads and ten evidence cases. Every
target below comes from an authoritative upstream result or rule. None was
invented to make a local implementation pass.

The evidence campaign is bound to clean source revision
`3cc071737454494d6a14d58fb5dc74d190d6cf7a`. Exact five-run values, evidence
IDs, and SHA-256 digests are generated from `reference_results/index.json`.
This document reviews the target basis and acceptance logic rather than
duplicating mutable result tables.

## Score-Bearing Targets

| **Workload** | **Model and Data** | **Quality Gate** | **Authority and Rationale** |
|:---|:---|:---|:---|
| `image-classification` | Official float ResNet8 and the 200-example MLPerf Tiny CIFAR-10 accuracy set | top-1 accuracy at least 0.85 | MLPerf Tiny fixes the model, accuracy set, metric, and threshold. The PyTorch adapter must reproduce that result without changing preprocessing. |
| `keyword-spotting` | Official DS-CNN and the 1,000-example EEMBC MFCC accuracy set | top-1 accuracy at least 0.90 | MLPerf Tiny fixes the task and threshold. The adapter preserves the quantized input convention and model graph. |
| `causal-language-modeling` | nanoGPT Shakespeare character configuration and Tiny Shakespeare split | best validation cross-entropy at most 1.4697 | The threshold is nanoGPT's published result for the exact 5,000-iteration recipe. |
| `text-classification` | Pinned DistilBERT SST-2 checkpoint and GLUE development split | accuracy at least 0.9105504587155964 | The pinned model-index metadata publishes this exact verified accuracy for the GLUE SST-2 validation split. The complete split is evaluated. |
| `information-retrieval` | Pinned MiniLM cross-encoder and the documented three-dataset NanoBEIR subset | mean nDCG@10 equal to 0.60716840988382 within the registry tolerance | Sentence Transformers publishes the exact evaluator example and score. |
| `graph-node-classification` | Official OGB GCN recipe and `ogbn-arxiv` split | test accuracy within 0.0029 of 0.7174 | The correct OGB GCN reference is 71.74% with a published 0.29-point standard deviation. The previously quoted 72.51% belongs to a different leaderboard section and is not used. |
| `time-series-forecasting` | Official PatchTST ETTm1 recipe and split | test MSE at most 0.29292929292929293 | PatchTST publishes the 0.290 result. The gate divides that lower-is-better reference by the MLPerf 0.99 quality fraction, an explicit direction-aware MLPerf EDU policy inference. |

The MLPerf policy input is pinned to inference-policies commit
`c547732b539cb3a14cc5680597714c8c1df4cad0`. The referenced
`inference_rules.adoc` bytes have SHA-256
`4a42bec8ab869b78b41dc00e94da18113ab4fffa32aa19a8dccc814c5d12897e`.

## Performance-Bearing Phase Gates

`causal-language-modeling` adds full, prefill, and decode inference. These
three cases report timing only after every run passes its functional contract.
They do not introduce a second language-quality benchmark. All phases must use
one package selecting the median-quality committed training run.

| **Phase** | **Primary Metric** | **Functional Requirement** |
|:---|:---|:---|
| full | output tokens per second | Complete the declared prompt and generation path with finite outputs and the expected token count. |
| prefill | prefill tokens per second | Complete the declared prompt prefill with a valid cache and finite output. |
| decode | output tokens per second | Complete cache-backed autoregressive decode with the expected token count and finite output. |

## Ten-Case Evidence Closure

The required cases are:

1. `image-classification__max__inference`
2. `keyword-spotting__max__inference`
3. `causal-language-modeling__max__training`
4. `causal-language-modeling__max__inference__full`
5. `causal-language-modeling__max__inference__prefill`
6. `causal-language-modeling__max__inference__decode`
7. `text-classification__max__inference`
8. `information-retrieval__max__inference`
9. `graph-node-classification__max__training`
10. `time-series-forecasting__max__training`

Every case needs five passing fresh-process runs and timing CV no greater than
5%. Score-bearing cases require every individual quality value and the median
to pass. Phase cases require five functional passes.

## Target Review Questions

Domain reviewers should confirm:

- The upstream artifact revision and data split remain the strongest practical authority.
- The PyTorch adapter preserves preprocessing, model semantics, and evaluator behavior.
- The tolerance reflects the upstream reference rather than observed local convenience.
- The measured region is long enough for stable laptop timing.
- CPU and accelerator numeric differences cannot silently weaken the quality gate.
- The target remains appropriate when upstream dependencies change.

A source, preprocessing, evaluator, model, optimizer, schedule, or target
change invalidates the relevant evidence and requires a fresh five-run packet.

## Deferred and Rejected Coverage

Machine-sound anomaly detection and visual wake words remain deferred because
the authoritative MLPerf Tiny accuracy inputs are not directly available as a
thin laptop adapter. Reinforcement learning remains deferred to a future
MiniGo contract. Recommendation, diffusion, agent, retrieval-augmented
generation, and code-agent proposals are rejected or deferred when they would
require project-created tasks, judges, datasets, or quality targets.

The selection ledger is the authoritative record of those decisions.

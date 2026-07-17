# MLPerf EDU v0.1 Quality Target Review

## Review Boundary

The v0.1 portfolio contains fourteen workloads. The current evidence scope
contains nine workloads and twelve evidence cases. Every target below comes
from an authoritative upstream result or rule. None was invented to make a
local implementation pass.

The draft evidence campaign is bound to clean source revision
`163d42ee3df54ab122543469ccf2b6b3bd119455`. Exact run counts, values, evidence
classes, and SHA-256 digests are generated from
`provisional_results/index.json`. Six cases have complete five-run evidence;
six remain explicitly provisional. This document reviews the target basis and
acceptance logic rather than duplicating mutable result tables.

## Score-Bearing Targets

| **Workload** | **Model and Data** | **Quality Gate** | **Authority and Rationale** |
|:---|:---|:---|:---|
| `image-classification` | Official float ResNet8 and the 200-example MLPerf Tiny CIFAR-10 accuracy set | top-1 accuracy at least 0.85 | MLPerf Tiny fixes the model, accuracy set, metric, and threshold. The PyTorch adapter must reproduce that result without changing preprocessing. |
| `keyword-spotting` | Official DS-CNN and the 1,000-example EEMBC MFCC accuracy set | top-1 accuracy at least 0.90 | MLPerf Tiny fixes the task and threshold. The adapter preserves the quantized input convention and model graph. |
| `anomaly-detection` | Official MLPerf Tiny autoencoder and the 248-recording ToyCar accuracy set | ROC AUC at least 0.85 | MLPerf Tiny fixes the model, accuracy-set construction, reconstruction-error evaluator, metric, and threshold. The adapter preserves the fused network and feature pipeline. |
| `visual-wake-words` | Official MLPerf Tiny MobileNetV1 0.25 and the 1,000-example EEMBC accuracy set | top-1 accuracy at least 0.80 | MLPerf Tiny fixes the model, labeled accuracy set, preprocessing contract, metric, and threshold. |
| `causal-language-modeling` | nanoGPT Shakespeare character configuration and Tiny Shakespeare split | best validation cross-entropy at most 1.4697 | The threshold is nanoGPT's published result for the exact 5,000-iteration recipe. |
| `text-classification` | Pinned DistilBERT SST-2 checkpoint and GLUE development split | accuracy at least 0.9105504587155964 | The pinned model-index metadata publishes this exact verified accuracy for the GLUE SST-2 validation split. The complete split is evaluated. |
| `information-retrieval` | Pinned MiniLM cross-encoder and the documented three-dataset NanoBEIR subset | mean nDCG@10 equal to 0.60716840988382 within the registry tolerance | Sentence Transformers publishes the exact evaluator example and score. |
| `graph-node-classification` | Official OGB GCN recipe and `ogbn-arxiv` split | test accuracy within 0.0029 of 0.7174 | The correct OGB GCN reference is 71.74% with a published 0.29-point standard deviation. The previously quoted 72.51% belongs to a different leaderboard section and is not used. |
| `time-series-forecasting` | Official PatchTST ETTm1 recipe and split | test MSE at most 0.29292929292929293 | PatchTST publishes the 0.290 result. The gate divides that lower-is-better reference by the MLPerf 0.99 quality fraction, an explicit direction-aware MLPerf EDU policy inference. |

## Functional-Stage Quality Backlog

These five workloads run bounded functional probes but remain outside the
promotion importer. Their published targets are retained as future conformance
gates rather than weakened to match local observations.

| **Workload** | **Authoritative Target** | **Current Boundary** |
|:---|:---|:---|
| `code-generation` | Qwen2.5-Coder HumanEval+ pass@1 of 0.573 | Autoregressive CLI integration works; complete EvalPlus reproduction remains pending. |
| `function-calling` | Qwen3-1.7B BFCL V4 Non-Live AST accuracy of 0.8292 | Grammar-constrained generation and AST-evaluator integration works; the complete local audit reached 0.7852. |
| `recommendation` | Meta DLRM Criteo Terabyte ROC AUC of 0.8025 | Dense-sparse execution works; the unchanged Criteo contract remains outside the practical laptop boundary. |
| `image-generation` | NVIDIA EDM CIFAR-10 FID of 1.79 | Iterative denoising works; three official 50,000-image trials reached a best FID of 1.8015540749984766. |
| `reinforcement-learning` | MiniGo professional-move prediction of 0.40 and upstream playoff rule | Policy-value self-play and a training step work; the unchanged self-play volume remains impractical locally. |

The MLPerf policy input is pinned to inference-policies commit
`c547732b539cb3a14cc5680597714c8c1df4cad0`. The referenced
`inference_rules.adoc` bytes have SHA-256
`4a42bec8ab869b78b41dc00e94da18113ab4fffa32aa19a8dccc814c5d12897e`.

## Performance-Bearing Phase Gates

`causal-language-modeling` adds full, prefill, and decode inference. These
three cases report timing only after every run passes its functional contract.
They do not introduce a second language-quality benchmark. All phases must use
one content-addressed training package. Promotion requires the package to
select the median-quality execution from a passing five-run training campaign;
the draft phases use the separately labeled provisional package.

| **Phase** | **Primary Metric** | **Functional Requirement** |
|:---|:---|:---|
| full | output tokens per second | Complete the declared prompt and generation path with finite outputs and the expected token count. |
| prefill | prefill tokens per second | Complete the declared prompt prefill with a valid cache and finite output. |
| decode | output tokens per second | Complete cache-backed autoregressive decode with the expected token count and finite output. |

## Twelve-Case Evidence Closure

The required cases are:

1. `image-classification__max__inference`
2. `keyword-spotting__max__inference`
3. `anomaly-detection__max__inference`
4. `visual-wake-words__max__inference`
5. `causal-language-modeling__max__training`
6. `causal-language-modeling__max__inference__full`
7. `causal-language-modeling__max__inference__prefill`
8. `causal-language-modeling__max__inference__decode`
9. `text-classification__max__inference`
10. `information-retrieval__max__inference`
11. `graph-node-classification__max__training`
12. `time-series-forecasting__max__training`

Promotion requires five passing fresh-process runs and timing CV no greater
than 5%. Score-bearing cases require every individual quality value and the
median to pass. Phase cases require five functional passes. The current draft
has six cases that meet this standard and six provisional cases that establish
execution and gate passage only.

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

## Functional and Rejected Coverage

The five functional-stage candidates remain experimental until their
authoritative quality contracts pass. End-to-end RAG, ReAct agents, and
distributed training remain rejected because they require unstable project
choices or fall outside the single-node boundary.

The selection ledger is the authoritative record of those decisions.

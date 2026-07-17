# MLPerf EDU v0.1 Quality Target Review

## Review Boundary

The v0.1 portfolio contains fourteen workloads. The current evidence scope
contains nine workloads and twelve evidence cases. Every target below is bound
to an upstream rule, published result, or predeclared interpretation. None is
fitted to make a local implementation pass.

The draft evidence campaign is bound to clean source revision
`163d42ee3df54ab122543469ccf2b6b3bd119455`. Exact run counts, values, evidence
classes, and SHA-256 digests are generated from
`provisional_results/index.json`. Six cases have complete five-run evidence;
six remain explicitly provisional. This document reviews the target basis and
acceptance logic rather than duplicating mutable result tables.

## Current One-Run Acceptance

The current milestone evaluates quality separately from stability. One complete
authoritative run is sufficient to decide whether a workload meets its quality
target. The run must use the pinned model or training recipe, the complete
required evaluation set, the authoritative evaluator, and every declared
quality gate. A functional probe cannot satisfy this milestone.

Repeated fresh-process runs, timing variation, and promotion evidence remain a
later stabilization phase. The registry records `acceptance_runs: 1` for all
fourteen workloads while retaining the five-run measurement protocol for that
later phase.

## Target Design Standard

A quality target is acceptable for this milestone only when the model or
training recipe, evaluation data, evaluator, metric direction, and source value
all refer to the same upstream contract. A local observation may confirm or
miss that target, but it cannot define or relax it. Discrete targets also record
their task-count interpretation when rounding would otherwise hide the actual
gate.

The registry distinguishes three target kinds so that a student can tell what a
number means.

| **Target Kind** | **Workloads** | **Interpretation** |
|:---|:---|:---|
| **Inherited acceptance gate** | Image classification, keyword spotting, anomaly detection, visual wake words, recommendation, and reinforcement learning | An upstream benchmark already defines the pass threshold. These are the strongest targets because MLPerf EDU does not choose the boundary. |
| **Published reference reproduction** | Causal language modeling, text classification, information retrieval, time-series forecasting, code generation, function calling, and image generation | The target is a published point result for the exact pinned recipe. It is a strict conformance target rather than a claim that the upstream project defined a universal pass threshold. |
| **Published mean with tolerance** | Graph node classification | The official OGB GCN mean is paired with its published standard deviation. The one-sided tolerance remains a domain-review item rather than a locally fitted margin. |

This distinction is important for the four current near-misses. PatchTST,
HumanEval+, BFCL, and EDM remain below strict published-reference points. Their
results do not justify lowering the gates. The causal-language-modeling point,
graph tolerance, and mutable BFCL leaderboard snapshot remain conditional until
their interpretation or frozen source is independently approved.

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
| `time-series-forecasting` | Official PatchTST ETTm1 recipe and split | test MSE at most 0.290 | PatchTST publishes the 0.290 result. MLPerf EDU treats it as a strict reproduction point and does not transfer an unrelated MLPerf Inference tolerance to this training-paper metric. |

## New-Workload Quality Backlog

These five workloads remain outside the promotion importer. Their published
targets are retained as conformance gates rather than weakened to match local
observations. Every workload now has a complete authoritative `max` runner.
Some runs remain pending because their current result misses the target or the
required external execution environment is unavailable.

| **Workload** | **Authoritative Target** | **Current Boundary** |
|:---|:---|:---|
| `code-generation` | Qwen2.5-Coder HumanEval+ pass@1 of 0.573 | The complete 164-task run passed 91 tasks, or 0.554878. The unchanged gate requires at least 94. |
| `function-calling` | Qwen3-1.7B BFCL V4 Non-Live AST accuracy of 0.8292 | The pinned 1,150-case runner and official evaluator are ready. A complete artifact from the current runner remains pending; the earlier full audit reached 0.7852. |
| `recommendation` | MLPerf Inference v1.0.1 DLRM Criteo Terabyte ROC AUC of 0.8025 | The complete historical accuracy adapter is ready. Execution requires licensed Criteo data, the roughly 90 GB checkpoint, a legacy runtime, and a 256-GB-class system. |
| `image-generation` | NVIDIA EDM CIFAR-10 minimum FID of 1.79 across three trials | One acceptance result now contains three independent 50,000-image trials, matching the upstream score definition. Prior trials reached a minimum FID of 1.8015540749984766, and a current three-trial artifact remains pending. |
| `reinforcement-learning` | MiniGo professional-move prediction of 0.40 and upstream playoff rule | The complete resumable self-play, training, and evaluation loop is ready. Execution requires a reviewed immutable legacy GPU image and a suitable NVIDIA system. |

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
change invalidates the relevant evidence and requires a fresh authoritative
quality run. A promoted timing baseline additionally requires a new stability
packet.

## Functional and Rejected Coverage

The five quality-conformance candidates remain experimental until their
authoritative quality contracts pass. End-to-end RAG, ReAct agents, and
distributed training remain rejected because they require unstable project
choices or fall outside the single-node boundary.

The selection ledger is the authoritative record of those decisions.

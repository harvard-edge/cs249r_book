# MLPerf EDU v0.1 Quality Target Review

## Review Boundary

The v0.1 portfolio contains fourteen workloads. The current evidence scope
contains nine workloads and twelve evidence cases. Every target below is bound
to an upstream rule, published result, or predeclared interpretation. None is
fitted to make a local implementation pass.

The draft evidence campaign is bound to clean source revision
`163d42ee3df54ab122543469ccf2b6b3bd119455`. Exact run counts, values, evidence
classes, and SHA-256 digests are generated from
`provisional_results/index.json`. Six cases have complete repeated timing evidence;
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
fourteen workloads while retaining the repeated-timing measurement protocol for that
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
results do not justify lowering the gates. The graph tolerance and the
causal-language-modeling reproduction-point interpretation remain conditional
until independently approved. The BFCL data, leaderboard result, and evaluator
are now pinned to immutable revisions.

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

> **Tolerance convention.** When a source publishes a mean together with a
> standard deviation, the contract carries both: the mean as the target and the
> published deviation as the tolerance. Graph node classification does this with
> the OGB leaderboard's 0.0029, and time-series forecasting now does it with
> PatchTST's 0.002. Carrying a mean while discarding the spread stated beside it
> makes a contract stricter than the source it inherits from. Applying the
> convention never moves a recorded verdict; see
> [MISS_DIAGNOSIS.md](MISS_DIAGNOSIS.md).

| **Workload** | **Authoritative Target** | **Current Boundary** |
|:---|:---|:---|
| `code-generation` | Qwen2.5-Coder HumanEval+ pass@1 of 0.573 | 91 of 164 in the container and 92 on the host, against a 94-task gate. The evaluator self-check passes 163 of 164, so the gap is in generation. |
| `function-calling` | Qwen3-1.7B BFCL V4 Non-Live AST accuracy of 0.8292 | The complete provenance-bound 1,150-case packet rescored at 0.785208 and did not meet the unchanged gate. |
| `recommendation` | MLPerf Training v0.5 NCF on MovieLens-20M, HR@10 of 0.635 | Trains locally in roughly half an hour. HR@10 peaks at 0.6232 on epoch 7 and declines after, so the shortfall is not an epoch budget. |
| `image-generation` | NVIDIA EDM CIFAR-10 minimum FID of 1.79 across three trials | The complete packet binds and rehashes three independent 50,000-image trials and rescored at 1.8015540749997014, missing the unchanged gate. |
| `reinforcement-learning` | MiniGo professional-move prediction of 0.40 and upstream playoff rule | Runs locally through a PyTorch adapter over the pinned reference; only the network is replaced. No result is recorded yet beyond a smoke run. |

## Performance-Bearing Phase Gates

`causal-language-modeling` adds full, prefill, and decode inference. These
three cases report timing only after every run passes its functional contract.
They do not introduce a second language-quality benchmark. All phases must use
one content-addressed training package. Promotion requires the package to
select the median-quality execution from a passing repeated-timing training campaign;
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

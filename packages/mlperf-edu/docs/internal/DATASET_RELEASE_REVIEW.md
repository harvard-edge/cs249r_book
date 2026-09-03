# MLPerf EDU v0.1 Dataset Release Review

> **Superseded in part, 2026-08-04.** Recommendation moved from DLRM on Criteo
> Terabyte to MLPerf Training v0.5 NCF on MovieLens-20M, reinforcement
> learning moved from a CUDA and TensorFlow 1.x container to a PyTorch
> adapter, and the timing protocol dropped from five runs to one. No
> workload is environment-gated. Statements below about gated execution,
> licensed Criteo data, or five-run promotion describe the state at the
> time of the audit and are retained as a record rather than corrected.
> Current state: [WORKLOAD_STATUS.md](WORKLOAD_STATUS.md) and
> [MISS_DIAGNOSIS.md](MISS_DIAGNOSIS.md).


## Policy

Canonical runs fetch pinned assets from their authoritative upstream source.
The repository records versions, file hashes, splits, licenses, and release
status. It does not redistribute dataset bytes unless the applicable policy
permits packaging.

A missing redistribution decision does not allow a synthetic or reduced
substitute. The workload remains fetch-only, deferred, or unpublished until
the decision is resolved.

## Active Dataset Catalog

| **Dataset** | **Workload** | **Pinned Boundary** | **Current Release Treatment** |
|:---|:---|:---|:---|
| CIFAR-10 | `image-classification` | Pinned test Parquet plus the MLPerf Tiny 200-sample accuracy index | Fetch-only pending dataset-specific redistribution review. |
| CIFAR-10 | `image-generation` | Official EDM 50,000-image generation and FID reference-statistics boundary | Fetch-only inputs; generated image retention remains local-review material. |
| MLPerf Tiny keyword-spotting accuracy set | `keyword-spotting` | Pinned EEMBC MFCC files and labels | Fetch-only pending MLCommons and EEMBC release review. |
| MLPerf Tiny ToyCar accuracy set | `anomaly-detection` | Exact 248-recording index reconstructed from the pinned ToyADMOS archive | Fetch-only with CC BY 4.0 attribution. |
| MLPerf Tiny visual-wake-words accuracy set | `visual-wake-words` | Balanced 1,000-example EEMBC selection from the pinned COCO-derived archive | Fetch-only pending COCO and MLCommons release review. |
| Tiny Shakespeare | `causal-language-modeling` | Exact corpus bytes and upstream 90/10 split | Fetch-only; char-rnn repository is MIT and the underlying text is public domain in the United States. |
| Deterministic prompt suite | `causal-language-modeling` inference | Bundled project prompt asset | Bundling depends on the final component license. |
| GLUE SST-2 | `text-classification` | Official development split and pinned archive | Fetch-only pending redistribution review. |
| NanoBEIR English subset | `information-retrieval` | Twelve pinned corpus, query, candidate, and relevance files | Fetch-only; component source licenses remain applicable. |
| OGB `ogbn-arxiv` | `graph-node-classification` | Official archive, time split, and evaluator | ODC-By 1.0 fetch-only with OGB and Microsoft Academic Graph attribution. |
| ETTm1 | `time-series-forecasting` | Pinned CSV and official 12/4/4-month split | Fetch-only pending dataset-specific review under the source repository terms. |
| HumanEval+ | `code-generation` | Complete 164-task EvalPlus release | Fetch-only; code execution must remain sandboxed. |
| BFCL V4 Non-Live AST | `function-calling` | Complete 1,150-example six-category split | Fetch-only pending BFCL component review. |
| Criteo Terabyte | `recommendation` | Canonical Meta DLRM preprocessing and split | External terms acceptance and fetch instructions only. |
| MiniGo self-play stream | `reinforcement-learning` | Run-generated self-play plus upstream professional-move and playoff evaluation | Generated data remains local; upstream inputs require release review. |

`datasets.yaml` is the structured public catalog. Asset dossiers in
`src/mlperf/assets.py` add file-level provenance and packaging policy.

## Required Controls

- Fetch verifies pinned revisions and file digests before measurement.
- A canonical `max` run fails if the required real asset is unavailable.
- No canonical runner silently falls back to synthetic data.
- Reports name the dataset, split, data mode, and file hashes.
- Provenance manifests bind every consumed dataset file.
- Portable packages reject bytes whose dossier is not approved for redistribution.
- Raw promotion packets remain outside Git because they may include dataset-derived artifacts.

## Release Decisions

The project needs a recorded decision for each asset whose status is
`needs-release-decision` or equivalent. An accepted decision may authorize
fetch-only use, attribution, or redistribution. Until then, documentation must
describe the asset conservatively and packages must fail closed.

The user may resolve licenses and permissions separately from implementation,
but technical readiness does not erase those external obligations.

## Functional-Stage Dataset Coverage

HumanEval+, BFCL, Criteo Terabyte, CIFAR-10 EDM inputs, and MiniGo self-play are
registered because their workload identities and upstream contracts are known.
Every workload now has a fail-closed max runner for its complete canonical
contract. HumanEval+, BFCL, and EDM consume their complete local evaluation
assets. DLRM and MiniGo currently require external research environments, and
their planned local backends must not fall back to reduced substitutes.
End-to-end RAG and ReAct agent datasets remain
outside the portfolio because no stable upstream tuple fixes the complete task
and evaluator without project choices.

## Reviewer Checklist

- [x] Every active dataset has an authoritative upstream source.
- [x] Every revision, file, and split is pinned and verified, or is generated
  by the pinned run contract.
- [ ] License and attribution text is accurate.
- [x] Fetch-only and redistribution rules are explicit and conservative while
  external decisions remain open.
- [x] Generated reports disclose data mode and asset hashes.
- [x] Package tests reject restricted bytes.
- [x] No removed workload dataset remains on the public benchmark site.

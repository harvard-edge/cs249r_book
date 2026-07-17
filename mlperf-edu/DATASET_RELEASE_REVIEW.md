# MLPerf EDU v0.1 Dataset Release Review

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
| OGB `ogbn-arxiv` | `graph-node-classification` | Official archive, time split, and evaluator | Fetch-only pending OGB and source-data terms review. |
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
Their current bounded probes do not consume the complete canonical assets and
therefore cannot support quality or timing claims. End-to-end RAG and ReAct
agent datasets remain outside the portfolio because no stable upstream tuple
fixes the complete task and evaluator without project choices.

## Reviewer Checklist

- [ ] Every active dataset has an authoritative upstream source.
- [ ] Every revision, file, and split is pinned and verified.
- [ ] License and attribution text is accurate.
- [ ] Fetch-only and redistribution rules are explicit.
- [ ] Generated reports disclose data mode and asset hashes.
- [ ] Package tests reject restricted bytes.
- [ ] No removed workload dataset remains on the public benchmark site.

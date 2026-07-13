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
| MLPerf Tiny keyword-spotting accuracy set | `keyword-spotting` | Pinned EEMBC MFCC files and labels | Fetch-only pending MLCommons and EEMBC release review. |
| Tiny Shakespeare | `causal-language-modeling` | Exact corpus bytes and upstream 90/10 split | Fetch-only; char-rnn repository is MIT and the underlying text is public domain in the United States. |
| Deterministic prompt suite | `causal-language-modeling` inference | Bundled project prompt asset | Bundling depends on the final component license. |
| GLUE SST-2 | `text-classification` | Official development split and pinned archive | Fetch-only pending redistribution review. |
| NanoBEIR English subset | `information-retrieval` | Twelve pinned corpus, query, candidate, and relevance files | Fetch-only; component source licenses remain applicable. |
| OGB `ogbn-arxiv` | `graph-node-classification` | Official archive, time split, and evaluator | Fetch-only pending OGB and source-data terms review. |
| ETTm1 | `time-series-forecasting` | Pinned CSV and official 12/4/4-month split | Fetch-only pending dataset-specific review under the source repository terms. |

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

## Deferred Dataset Coverage

The MLPerf Tiny machine-sound anomaly and visual wake-word tasks remain
deferred because their complete authoritative accuracy inputs are not directly
available as a thin laptop adapter. Rebuilding them from larger source corpora
would add a substantial project-defined data pipeline.

Agent, retrieval-augmented generation, code-generation, diffusion,
recommendation, and reinforcement-learning datasets are not admitted in v0.1
without a complete authoritative model, data, evaluator, target, and laptop
contract.

## Reviewer Checklist

- [ ] Every active dataset has an authoritative upstream source.
- [ ] Every revision, file, and split is pinned and verified.
- [ ] License and attribution text is accurate.
- [ ] Fetch-only and redistribution rules are explicit.
- [ ] Generated reports disclose data mode and asset hashes.
- [ ] Package tests reject restricted bytes.
- [ ] No removed workload dataset remains on the public benchmark site.

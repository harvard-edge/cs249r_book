# MLPerf EDU Dataset Release Review

Local catalog review date is 2026-07-11. This document mirrors the 14 entries
in `datasets.yaml` and separates project classifications from legal or
MLCommons approval. It is a release-engineering ledger, not legal advice.

The current strict public audit has one dataset-policy warning. The
score-bearing `micro-dlrm-train` row uses MovieLens-100K, whose catalog status
is `restricted-needs-approval`.

## Public-Candidate Assets

| **Dataset** | **Candidate Use** | **Catalog License Status** | **Catalog Release Status** | **Current Decision Boundary** |
|:---|:---|:---|:---|:---|
| TinyShakespeare | `nanogpt-train` | `public-domain-us` | `public-ok-fetch-only` | The harness generates a deterministic excerpt from Project Gutenberg eBook 100. Keep the source recipe, upstream terms, and asset digest in the packet. |
| Fashion-MNIST | `resnet18-train`, `mobilenetv2-train` | `mit` | `public-ok-with-attribution` | The project selected it for the first vision candidates. Preserve upstream attribution and seek reviewer acceptance of the educational task choice. |
| MNIST | `anomaly-ae-train` | `cc-by-sa-3.0` | `public-ok-with-attribution` | Preserve attribution and review the zero-versus-nonzero anomaly protocol separately from the dataset terms. |
| MovieLens-100K | `micro-dlrm-train` | `noncommercial-research-education` | `restricted-needs-approval` | Do not redistribute. Obtain a written fetch-only policy decision or demote the row and select an open replacement. |
| MLPerf EDU prompt suite | NanoGPT prefill/decode and SmolLM2 baseline | `bundled-project-asset` | `public-ok-bundled` | The fixture is technically bundled and digested, but public distribution still depends on an authoritative component license. |

The SmolLM2 candidate also depends on a model dossier. Its default
`HuggingFaceTB/SmolLM2-135M-Instruct` revision is pinned in the registry and is
classified there as Apache-2.0. Model policy review is separate from this
dataset table.

## Systems-Only Assets

These assets do not currently support public score claims. Their incomplete
release review does not justify silently promoting the corresponding rows.

| **Catalog ID** | **Used By** | **License Status** | **Release Status** | **Required Before Promotion** |
|:---|:---|:---|:---|:---|
| `cartpole_local` | `micro-rl-train` | `project-license-pending` | `systems-only-review-pending` | Component license and environment-methodology review. |
| `cifar10` | `micro-diffusion-train` | `source-citation-no-license` | `systems-only-review-pending` | Source-terms and generative-quality review. |
| `cora` | `micro-gnn-train` | `needs-review` | `systems-only-review-pending` | Upstream terms, split, attribution, and target review. |
| `etth1` | `micro-lstm-train` | `needs-review` | `systems-only-review-pending` | Upstream terms, fixed split, and forecasting target review. |
| `mbpp` | `nano-codegen-agent` | `needs-review` | `systems-only-review-pending` | Source and bundled-subset license review plus capability methodology. |
| `react_traces` | RAG, ReAct, tool-call agents | `project-license-pending` | `systems-only-review-pending` | Component license, trace provenance, and evaluation policy. |
| `speech_commands_v2` | `dscnn-kws-train` | `cc-by-4.0` | `systems-only-with-attribution` | Real-data runner, attribution, split, and target evidence. |
| `sst2` | `micro-bert-train` | `needs-review` | `systems-only-review-pending` | Checkout asset provenance, upstream terms, split, and target evidence. |
| `wake_vision` | `wake-vision-vww` | `needs-review` | `systems-only-review-pending` | Upstream terms, real-data runner, split, and target evidence. |

The prompt suite is also used by systems-only inference variants. Its candidate
classification appears in the first table because three performance-bearing
rows rely on it.

## Implemented Project Decisions

These statements describe repository behavior. They do not claim external
approval.

- NanoGPT training fetches Project Gutenberg source and applies a deterministic
  TinyShakespeare recipe instead of redistributing an unexplained corpus file.
- Fashion-MNIST is the default real dataset for both score-bearing vision
  candidates. CIFAR data is not used by a default score-bearing vision row.
- MNIST anomaly scoring uses the complete 10,000-example test split and reports
  discrimination metrics rather than grading training MSE.
- MovieLens remains fetch-only and raises the strict public audit warning.
- Reports and generated dataset pages expose source, license status,
  public-release status, and next-step metadata.
- Synthetic and micro-sharded modes are labeled and are ineligible for
  score-bearing `max` review.

## Source Records

- Project Gutenberg eBook 100 source
  <https://www.gutenberg.org/files/100/100-0.txt>
- Project Gutenberg license and terms
  <https://www.gutenberg.org/policy/license.html>
- Fashion-MNIST project and MIT license
  <https://github.com/zalandoresearch/fashion-mnist>
- Original MNIST dataset page
  <https://yann.lecun.org/exdb/mnist/>
- TensorFlow MNIST loader documentation and license record
  <https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data>
- MovieLens-100K dataset page
  <https://grouplens.org/datasets/movielens/100k/>
- MovieLens-100K README and terms
  <https://files.grouplens.org/datasets/movielens/ml-100k-README.txt>
- CIFAR-10 and CIFAR-100 official page
  <https://www.cs.toronto.edu/~kriz/cifar.html>

The release packet should preserve the reviewed terms or stable citations used
for each decision. A link list alone is not a rights review.

## MovieLens Decision Paths

| **Path** | **Required Action** | **Registry Outcome** |
|:---|:---|:---|
| Written approval | Record permission or an accepted MLCommons fetch-only policy that covers the intended result publication. | Keep `micro-dlrm-train` score-bearing and retain the policy record with the release. |
| Conservative demotion | Change `micro-dlrm-train` to systems-only while preserving it as a classroom recommender experiment. | Strict public audit can clear the current warning, but the first release has four score-bearing rows. |
| Open replacement | Add a recommender dataset with suitable terms, then recalibrate the model, split, target, and five-seed evidence. | Restore a score-bearing recommender only after the replacement contract passes all gates. |

## External Review Questions

1. May the intended public-candidate result use MovieLens-100K under a
   fetch-only workflow without redistributing data?
2. Do reviewers accept Fashion-MNIST as a course-scale vision-training task for
   the first candidate release?
3. Do reviewers accept the MNIST zero-versus-nonzero anomaly protocol and its
   attribution package?
4. What component license should cover the bundled prompt fixture, CartPole
   implementation, curated traces, and other project-authored assets?
5. Which source and redistribution records must accompany external review
   packets even when a runner remains systems-only?

`uv run mlperf audit --policy public --format json` is the machine-readable
check for the current candidate rows. It checks the structured project policy.
It does not replace legal or MLCommons review.

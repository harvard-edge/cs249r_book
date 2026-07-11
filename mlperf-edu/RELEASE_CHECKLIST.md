# MLPerf EDU Release Readiness Ledger

This ledger is the source of truth for deciding whether the current revision is
ready to show MLCommons reviewers. It separates implemented machinery from
fresh evidence and external decisions. A feature is not release-verified merely
because its code or workflow exists.

The current reference evidence contains eight retained result summaries from
clean source commit `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`. Complete local
validation, hosted CI, independent reproduction, and external decisions remain
separate gates.

Do not change a row to `Verified` without recording the source revision,
command, exit status, date, platform, and retained artifact path in the review
notes or release packet.

## Status Key

| **Status** | **Meaning** |
|:---|:---|
| Implemented | The repository contains the feature and a check for it. The check must still run on the final revision. |
| Verified | The exact final revision passed the named gate and retained its evidence. |
| Pending | In-repository work or evidence remains. |
| External | A maintainer, rights holder, or MLCommons decision is required. |

## Implemented Repository Work

These rows describe code present in the worktree. They remain subject to the
final gate sequence below.

| **Status** | **Item** | **Implementation Evidence** | **Final Gate** |
|:---|:---|:---|:---|
| Implemented | Public CLI and source install | `pyproject.toml` exposes `mlperf`; `INSTALL.md` uses the locked checkout path. | Clean checkout and wheel smoke. |
| Implemented | Native registry and package mirrors | `registry/`, `workloads.yaml`, and `src/mlperf_edu/workloads.yaml`. | Both registry export checks. |
| Implemented | Complete registry execution surface | 30 rows declare `min` and `max` runners. | Actual `coverage`, `max`, and `release` validation. |
| Implemented | Candidate result classification | Five score-bearing, three performance-bearing, 22 systems-only. | Registry audit and generated-page drift checks. |
| Verified | Strong score gates | NanoGPT loss `<= 2.30`; DLRM fixed-final-epoch ROC AUC `>= 0.76`; anomaly macro AUROC `>= 0.93`, worst-class AUROC `>= 0.90`, and learned-control margin `>= 0.20`; ResNet top-1 `>= 0.85`; MobileNetV2 top-1 `>= 0.78`. | Five clean public-candidate packets from source `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`. |
| Implemented | Report-level public review contract | `src/mlperf/contracts.py` checks data mode, seed, quality, timing, checkpoint or model lineage, and artifacts. | Actual `max` and `release` validation with zero contract failures. |
| Verified | Repeatable inference timing | NanoGPT prefill and decode plus pinned SmolLM2 baseline record warmups, measured runs, median, p90, p99, and cross-execution CV. | Three committed five-execution summaries from clean source `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`; CVs are `0.19%`, `0.75%`, and `1.11%`, below the `5%` limit. |
| Verified | Training-to-inference lineage | NanoGPT inference requires the training checkpoint and records its SHA-256 digest and quality dependency. | Verified with median-quality training seed 2 and both inference packets. |
| Verified | SLM task-quality fixture | Pinned SmolLM2 revision, attributed 28-case v2 suite, token-weighted NLL, and overall plus weakest-category gates. | Current clean five-execution packet passes the v2 fixture and timing contract. |
| Implemented; replacement calibration pending | Honest quantized boundary | Dynamic-int8 SLM row remains systems-only after failing the former quality-parity calibration. | Rerun the quantized path on v2 and require overall, weakest-category, and NLL-parity gates before promotion. |
| Implemented | Exact report provenance | Manifest binds canonical report content and exact report bytes with SHA-256 and size evidence. | Manifest and package portability tests. |
| Implemented | Portable package schema 0.2 | Relative paths, complete artifact index, digest and byte-size checks, and clean-extraction verification. | NanoGPT lineage package SHA-256 `0b0173d78e2c3315c4687b6319beb8a2826c98bce7f52710542f4b496edadd20` passed all 56 archive checks; 35 policy-permitted run packages were also created and verified. |
| Verified | Five-seed sweep tool | Fresh-process seeds, seed/report/manifest agreement, separate performance and quality fields, grading, immutable attempts, artifact index, digest sidecar, cooldowns, and portable NanoGPT training-lineage staging. | Eight fresh public-candidate summaries from source `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`. |
| Implemented | Systems-only execution boundaries | Every systems-only row declares the exact max data mode, whether fetched or declared assets are used, and whether a quality target is enforced. | Registry validation and generated-page truth tests. |
| Implemented | Classroom entry points | Three labs accept literal `--smoke`; Tutorial 01 has a noninteractive provenance smoke. | Four commands pass on CPU without network. |
| Implemented | Generated review site | Registry, dataset, benchmark, and CLI pages are generated; Quarto and link gates exist. | Regenerate, render, link-check, and visually inspect. |
| Implemented | Blocking and long CI | Development workflow runs tests, smoke, labs, wheel, registry, docs, and site checks; scheduled/manual workflow executes actual `max` or `release` work with a five-hour limit. | Green workflow runs on the final pushed revision. |
| Implemented | Guarded site publication | Live preview requires recent development and full benchmark validation plus manual `PUBLISH` confirmation. | Confirm deployment only after the guarded workflow succeeds. |
| Implemented | Evidence-synchronized paper build | Paper Makefile generates a registry snapshot and verifies the PDF. | Clean build, verification, and visual inspection. |

## Current Committed Reference Evidence

The retained evidence set was collected on July 11, 2026, from clean source
commit `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`. Its embedded validity fields
describe the current contracts. `reference_results/index.json` retains the
exact IDs and digests.

| **Workload** | **Evidence ID** | **Primary Metric Median** | **Minimum** | **Maximum** | **CV** |
|:---|:---|---:|---:|---:|---:|
| `anomaly-ae-train` | `anomaly-ae-train_max_20260711T204007.498158Z` | `4.3060` | `4.2488` | `4.5012` | n/a |
| `micro-dlrm-train` | `micro-dlrm-train_max_20260711T205839.712863Z` | `2.0394` | `1.9475` | `2.0924` | n/a |
| `mobilenetv2-train` | `mobilenetv2-train_max_20260711T204653.574587Z` | `103.6699` | `103.4276` | `104.7116` | n/a |
| `nanogpt-decode` | `nanogpt-decode_max_20260711T210118.136527Z` | `316.7836` | `313.2178` | `319.4718` | `0.75%` |
| `nanogpt-prefill` | `nanogpt-prefill_max_20260711T205947.060179Z` | `3859.7758` | `3849.9537` | `3868.4721` | `0.19%` |
| `nanogpt-train` | `nanogpt-train_max_20260711T202223.716219Z` | `117.5981` | `117.0730` | `118.2963` | n/a |
| `resnet18-train` | `resnet18-train_max_20260711T204117.913822Z` | `58.4192` | `57.3181` | `59.5123` | n/a |
| `slm-decode` | `slm-decode_max_20260711T210317.517476Z` | `61.5742` | `60.4083` | `61.9703` | `1.11%` |

Complete create-once attempt directories remain outside the checkout under the
local reference root and are available by local handoff. Thirty-five
policy-permitted run packages have been created and verified; the package tool
correctly blocked the five DLRM runs because their MovieLens-derived artifacts
cannot be redistributed under the current policy. The raw packets do not yet
have reviewer-facing public URLs. The MovieLens packet remains local-only until
its redistribution and candidate-policy decision is resolved. The later
promotion revision adds the exact summaries, fail-closed cross-checks,
documentation, and packaging without changing runner, data, model, measurement,
grading, or report-contract behavior. A post-freeze change to any of those
measurement-bearing surfaces requires new sweeps rather than editing these
records.

## Current Local Verification Snapshot

The promotion revision recorded in
`review-f917f4823a/handoff_manifest.json` completed the local release matrix on
July 11, 2026, on macOS 26.4 with an Apple M5 Max. That revision remains the
exact clean executable source named by the retained validation reports and
handoff manifest; the commit that records this ledger changes documentation
only.

| **Artifact** | **Result** | **SHA-256** |
|:---|:---|:---|
| `validation-smoke` | 12 of 12 default `min` workloads passed | `eeeb344053c06784dca86a68a41b744b9fe05867b0e72a99240d7c44c69ca898` |
| `validation-coverage` | 30 of 30 `min` workloads passed | `373442fcd7bcecee8cf976e77a0b68ce162cbc9ce063d5617e61269ac674c6dc` |
| `validation-max` | 30 of 30 `max` workloads passed at reference seed 0 | `c8e9a4fe5f7d8936a49cfd9564a01d287d052a970f4dd839f97929ee5f657573` |
| `validation-release` | 60 of 60 `min` and `max` workloads passed | `5c54d3d984141b6c8e757b319c21648127717dffb0733d484742ec8c214b4c01` |
| Complete test suite | 382 tests passed | Recorded in the local review log for the promotion revision |
| Clean Python 3.12 wheel | Installed outside the checkout, audited, executed a workload, and verified provenance | `24eef2fe9217df32648f7d92b60972df424c59843046010e157caf7969a4d75c` |
| Reviewer handoff manifest | 8 summaries, 40 valid attempts, 35 reverified packages, and 5 policy-blocked DLRM attempts | `8d7674df0e3fe21088b7c6ea19a69e1121250d05693cccfe524099462ef21a2b` |
| Strict public audit | 0 internal blockers and 9 explicit external-publication warnings | `4ed5bfa6a34d9f27f9699bda9f937246411813acfcf4b8038bd14d703e5403e0` |
| Review paper PDF | 6 visually inspected and mechanically verified pages | `99e149460d4faa5bdbf37a1298b1242c8383d8e64a0bc250f90b152d205b00c6` |

## Repository Verification Status

| **Status** | **Gate** | **Required Evidence** |
|:---|:---|:---|
| Verified | Evidence source stability | Source lock regenerated for `86738e4654d8f77ef1cec4698b30e0ebd20dd2b3`; every imported packet came from a clean source snapshot. |
| Verified | Complete local tests | All 382 tests passed on the promotion revision recorded in the handoff manifest. |
| Verified | Five score packets | Fresh create-once seeds 0–4 packets are committed for NanoGPT, DLRM, anomaly detection, ResNet-18, and MobileNetV2. |
| Verified | Inference chain | One NanoGPT training lineage package produced current prefill and decode packets; SmolLM2 baseline packet is committed under the current protocol. |
| Verified | Actual validation presets | Fresh `smoke`, `coverage`, `max`, and `release` executions passed with zero run, grade, provenance, or review-contract failures. |
| Verified | Package portability | All 35 policy-permitted packages reverified; all 5 MovieLens-derived packages failed closed as required. |
| Verified | Clean wheel environment on macOS | The final wheel installed under Python 3.12 outside the checkout, passed its packaged audit, ran `nano-rag-agent`, and verified its provenance. |
| Pending | Clean wheel environment on Linux | The hosted Linux wheel job must pass on the pushed review revision. |
| Verified | Lab and tutorial regression | All three lab smokes and Tutorial 01 passed; the tutorial provenance verified. |
| Verified | Automated site review | Generated 37 benchmark pages, rendered all 43 site pages, passed internal links, and passed 86 layout checks across two viewports. |
| Verified | Site visual review | Representative desktop and narrow pages were visually inspected after rendering. |
| Verified | Paper review | The six-page paper rebuilt, passed both mechanical checks, and was visually inspected page by page. |
| Pending | Hosted CI evidence | Development and full benchmark workflows are green for the same review revision and their artifacts are retained. Local evidence does not imply this hosted gate passed. |
| Verified | Review notes | Registry mirrors, review packets, generated pages, proposal, quality review, paper snapshot, and release notes identify the promoted evidence. |

## External Decisions

| **Status** | **Decision** | **Owner and Required Outcome** |
|:---|:---|:---|
| External | Component license | Maintainers provide an authoritative publishable license covering code and bundled project assets. |
| External | MovieLens-100K | GroupLens terms and MLCommons policy review either permit the fetch-only score-bearing path or require demotion and an open replacement. |
| External | Name and relationship | MLCommons decides whether `MLPerf EDU` is acceptable and whether any working group will sponsor review. |
| External | Result wording | MLCommons approves language that distinguishes educational candidate results from official competitive MLPerf submissions. |
| External | Target and scenario review | Domain reviewers accept or revise the first eight public-candidate contracts. |
| External | Independent reproduction | At least one reviewer runs install, fetch, benchmark, verify, package, and report steps on a separate machine. |

## Executable Final Gate

Run this block from `mlperf-edu` on the final review revision. It stops at the
first failure.

```bash
set -euo pipefail

test -z "$(git status --porcelain)"
git rev-parse HEAD | tee /tmp/mlperf-edu-review-revision.txt

uv sync --locked --extra dev --extra tutorial
uv run pytest

uv run python tools/export_registry_layout.py --check
uv run python tools/export_flat_registry.py --check
uv run python tools/sync_verified_baselines.py --check
uv run python tools/check_taxonomy.py
uv run python tools/check_reference_claims.py --check
uv run python tools/generate_review_packets.py --check
uv run python tools/generate_docs.py --check

uv run mlperf doctor
uv run mlperf audit
mkdir -p submissions/release-audit
set +e
uv run mlperf audit --policy public --format json \
  > submissions/release-audit/public-audit.json
PUBLIC_AUDIT_EXIT=$?
set -e
echo "strict public audit exit: $PUBLIC_AUDIT_EXIT"

uv run mlperf validate smoke \
  --output-dir submissions/release-smoke
uv run mlperf validate coverage \
  --output-dir submissions/release-coverage
uv run mlperf validate max --keep-going \
  --output-dir submissions/release-max
uv run mlperf validate release --keep-going \
  --output-dir submissions/release-full

uv run python examples/lab1_optimization.py --smoke
uv run python examples/lab2_inference_sut.py --smoke
uv run python examples/lab3_arch_comparison.py --smoke
uv run python tutorials/smoke_first_benchmark.py

uv run python tools/build_wheel.py
WHEEL=$(find dist -maxdepth 1 -name '*.whl' -print -quit)
test -n "$WHEEL"
uv venv /tmp/mlperf-edu-review-wheel --python 3.12
uv pip install --python /tmp/mlperf-edu-review-wheel/bin/python "$WHEEL"
(cd /tmp && /tmp/mlperf-edu-review-wheel/bin/mlperf audit)
quarto render site
python3 ../shared/scripts/check-internal-links.py site --quiet
make -C paper clean all
make -C paper check
```

The strict public audit is expected to fail until the raw reference packages
have public URLs and the MovieLens decision is closed. For an independent
preview review packet, retain that failing JSON audit as an explicit external
blocker and inspect every warning. Do not silently remove the command or
represent the gate as passed. A public or endorsed release still requires exit
zero.

## Score Reference Commands

Run these commands only after the source-cleanliness guard passes. The tool's
default output root is `~/.mlperf-edu/reference_runs`. These are reproduction
commands for the current committed summaries; retain a new create-once attempt
rather than overwriting the evidence from `86738e46`.

```bash
uv run python tools/run_reference_sweep.py \
  --workload nanogpt-train --profile max --seeds 0,1,2,3,4 \
  --evidence-tier public-candidate

uv run python tools/run_reference_sweep.py \
  --workload micro-dlrm-train --profile max --seeds 0,1,2,3,4 \
  --evidence-tier public-candidate

uv run python tools/run_reference_sweep.py \
  --workload anomaly-ae-train --profile max --seeds 0,1,2,3,4 \
  --evidence-tier public-candidate

uv run python tools/run_reference_sweep.py \
  --workload resnet18-train --profile max --seeds 0,1,2,3,4 \
  --evidence-tier public-candidate

uv run python tools/run_reference_sweep.py \
  --workload mobilenetv2-train --profile max --seeds 0,1,2,3,4 \
  --evidence-tier public-candidate
```

Each command must exit zero and its `evidence_summary.json` must say
`eligible_for_public_baseline: true`. Retain the adjacent `.sha256` sidecar and
the complete create-once attempt directory. Copying only the aggregate numbers
into YAML is insufficient.

## Inference Evidence Commands

Package one passing run from the clean NanoGPT training sweep. The committed
inference chain uses the median-quality seed-2 training run. The inference
sweep verifies that package, rejects unsafe or unindexed ZIP members, stages the
checkpoint, report, and manifest inside its create-once attempt, and records
only attempt-relative lineage paths. Replace the manifest placeholder below
with the retained seed-2 directory or another explicitly justified passing
training run from a new sweep.

```bash
NANOGPT_TRAIN_MANIFEST="<clean-training-attempt>/seed_2/nanogpt-train_max.provd.json"
NANOGPT_LINEAGE_PACKAGE="/tmp/nanogpt-training-lineage.zip"
uv run mlperf package "$NANOGPT_TRAIN_MANIFEST" \
  --output "$NANOGPT_LINEAGE_PACKAGE"

uv run python tools/run_reference_sweep.py \
  --workload nanogpt-inference --variant prefill --profile max \
  --seeds 0,1,2,3,4 --evidence-tier public-candidate \
  --nanogpt-lineage-package "$NANOGPT_LINEAGE_PACKAGE"

uv run python tools/run_reference_sweep.py \
  --workload nanogpt-inference --variant decode --profile max \
  --seeds 0,1,2,3,4 --evidence-tier public-candidate \
  --nanogpt-lineage-package "$NANOGPT_LINEAGE_PACKAGE"

uv run python tools/run_reference_sweep.py \
  --workload smollm2-chat-inference --variant baseline --profile max \
  --seeds 0,1,2,3,4 --evidence-tier public-candidate
```

Inspect each `review_contract` block and require `status: passed`. Then verify
and package every candidate manifest. Retain all raw within-run timing samples
and summarize variation across the five executions. `mlperf package` performs
its own clean-extraction check, refuses an unverifiable manifest, and refuses
known restricted or unresolved dataset bytes. MovieLens remains a local-only
evidence path until its external policy decision closes.

## Reviewer Handoff Manifest

Build the handoff manifest only after the promotion revision is committed. The
command verifies the committed summaries against the retained historical
attempts, rechecks the shared NanoGPT lineage, hashes every indexed byte in the
portable packages, and records policy-blocked attempts without creating a
prohibited archive.

```bash
: "${MLPERF_EDU_REFERENCE_ROOT:?set the retained reference-attempt root}"
: "${MLPERF_EDU_REFERENCE_PACKAGE_ROOT:?set the portable-package root}"
: "${MLPERF_EDU_HANDOFF_OUTPUT:?set an output path outside the checkout}"

uv run python tools/build_handoff_manifest.py \
  --evidence-root "$MLPERF_EDU_REFERENCE_ROOT" \
  --portable-package-root "$MLPERF_EDU_REFERENCE_PACKAGE_ROOT" \
  --lineage-archive \
    "$MLPERF_EDU_REFERENCE_ROOT/nanogpt-training-lineage-median.zip" \
  --promotion-git-sha "$(git rev-parse HEAD)" \
  --output "$MLPERF_EDU_HANDOFF_OUTPUT"

uv run python tools/build_handoff_manifest.py \
  --evidence-root "$MLPERF_EDU_REFERENCE_ROOT" \
  --portable-package-root "$MLPERF_EDU_REFERENCE_PACKAGE_ROOT" \
  --lineage-archive \
    "$MLPERF_EDU_REFERENCE_ROOT/nanogpt-training-lineage-median.zip" \
  --promotion-git-sha "$(git rev-parse HEAD)" \
  --output "$MLPERF_EDU_HANDOFF_OUTPUT" --check
```

## Review-Ready Decision

The repository is ready to show MLCommons when every in-repository `Pending`
row is `Verified`, the retained packet names the exact source revision, and the
external rows are presented as explicit questions rather than concealed gaps.
It is ready for a public or endorsed release only after the applicable external
rows are closed in writing.

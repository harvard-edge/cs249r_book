# MLPerf EDU Release Readiness Ledger

This ledger is the source of truth for deciding whether the current revision is
ready to show MLCommons reviewers. It separates implemented machinery from
fresh evidence and external decisions. A feature is not release-verified merely
because its code or workflow exists.

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
| Implemented | Strong score gates | NanoGPT `<=2.30`, DLRM `>=0.70`, anomaly AUROC `>=0.95`, ResNet `>=0.85`, MobileNetV2 `>=0.78`. | Verified by five committed summaries from clean source `0ec4d3e1`. |
| Implemented | Report-level public review contract | `src/mlperf/contracts.py` checks data mode, seed, quality, timing, checkpoint or model lineage, and artifacts. | Actual `max` and `release` validation with zero contract failures. |
| Implemented | Repeatable inference timing | NanoGPT prefill and decode plus pinned SmolLM2 baseline record warmups, measured runs, median, p90, p99, and cross-execution CV. | Verified by three committed five-execution summaries from clean source `0ec4d3e1`; all three CVs are at or below `5%`. |
| Implemented | Training-to-inference lineage | NanoGPT inference requires the training checkpoint and records its SHA-256 digest and quality dependency. | Verified with median-quality training seed 4, checkpoint SHA-256 `a0d2f31a747355d47d11c6aa77eb09faf2232f84cb519accb286a78159fb2d8a`, and both inference packets. |
| Implemented | SLM task-quality fixture | Pinned SmolLM2 revision and bundled four-case continuation suite with perplexity gate. | Verified in five executions. Every execution passed the token and perplexity gates with the pinned revision and fixture digest retained. |
| Implemented | Honest quantized boundary | Dynamic-int8 SLM row is systems-only after failing the current quality-parity calibration. | Registry, docs, and packet regeneration. |
| Implemented | Exact report provenance | Manifest binds canonical report content and exact report bytes with SHA-256 and size evidence. | Manifest and package portability tests. |
| Implemented | Portable package schema 0.2 | Relative paths, complete artifact index, digest and byte-size checks, and clean-extraction verification. | NanoGPT lineage package SHA-256 `0b0173d78e2c3315c4687b6319beb8a2826c98bce7f52710542f4b496edadd20` passed all 56 archive checks; 35 policy-permitted run packages were also created and verified. |
| Implemented | Five-seed sweep tool | Fresh-process seeds, seed/report/manifest agreement, grading, immutable attempts, artifact index, digest sidecar, and portable NanoGPT training-lineage staging. | Eight clean public-candidate summaries are committed for the five score and three performance candidates. |
| Implemented | Systems-only execution boundaries | Every systems-only row declares the exact max data mode, whether fetched or declared assets are used, and whether a quality target is enforced. | Registry validation and generated-page truth tests. |
| Implemented | Classroom entry points | Three labs accept literal `--smoke`; Tutorial 01 has a noninteractive provenance smoke. | Four commands pass on CPU without network. |
| Implemented | Generated review site | Registry, dataset, benchmark, and CLI pages are generated; Quarto and link gates exist. | Regenerate, render, link-check, and visually inspect. |
| Implemented | Blocking and long CI | Development workflow runs tests, smoke, labs, wheel, registry, docs, and site checks; scheduled/manual workflow executes actual `max` or `release` work with a five-hour limit. | Green workflow runs on the final pushed revision. |
| Implemented | Guarded site publication | Live preview requires recent development and full benchmark validation plus manual `PUBLISH` confirmation. | Confirm deployment only after the guarded workflow succeeds. |
| Implemented | Evidence-synchronized paper build | Paper Makefile generates a registry snapshot and verifies the PDF. | Clean build, verification, and visual inspection. |

## Committed Reference Evidence

The current evidence set was collected on July 11, 2026, from clean source
commit `0ec4d3e1c415944227d0754d170edb0addc1d925`. The reference sweep commands
below exited zero for every candidate, and all eight summaries report
`status: valid` and `eligible_for_public_baseline: true`. The recorded platform
is an Apple M5 Max laptop with CPU or MPS selected by the declared protocol.
`reference_results/index.json` retains the exact evidence IDs, summary digests,
aggregates, and source revision.

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

## Local Verification Snapshot

The promotion revision recorded by `review-ece36ac566/handoff_manifest.json`
completed the local release matrix on July 11, 2026, on macOS 26.4 with an
Apple M5 Max. A later ledger-only commit may record these results, but the
promotion revision remains the exact executable source named by the retained
validation reports and handoff manifest.

| **Artifact** | **Result** | **SHA-256** |
|:---|:---|:---|
| `validation-smoke` | 12 of 12 default `min` workloads passed | `6954a13d3cd99fb66cd0faee4793c91540e522c3b46eb70602f172b839354eb9` |
| `validation-coverage` | 30 of 30 `min` workloads passed | `33d32ff1256ec4411c7c430579bfa9e4903ac05d1dce3694d4c5ccad58175b60` |
| `validation-max` | 30 of 30 `max` workloads passed | `bfcd71c94d4ed7602d8dec75a8d5037a7203c7d53c66365494cb7c184967aca7` |
| `validation-release` | 60 of 60 `min` and `max` workloads passed | `e40edcea473c1982a272d69bee558ea67e0b32963c7f8e57663b48cb4890493a` |
| Clean Python 3.12 wheel | Installed outside the checkout, audited, and executed a verified smoke workload | `94a28beab06a8f70ae85793460ba7e74d9d3a0e31603908ae2c019c541c11bb4` |
| Reviewer handoff manifest | 8 summaries, 40 valid attempts, 35 reverified packages, and 5 policy-blocked DLRM attempts | `0fabaaed4cf71754b9a7074729b7ff267799a450d678d6bdbb4960116c461a6a` |
| Strict public audit | 0 blockers and 9 expected external warnings | `e80a606345c6f2e0d47abc33f789201e2961aaeffeb7ee27eb4f778e57965613` |
| Review paper PDF | 6 visually inspected and mechanically verified pages | `22c12622281d8c31976f4b1c29dc922ae91964fd32c05f3401b36a6ef6b8a510` |

## Repository Verification Status

| **Status** | **Gate** | **Required Evidence** |
|:---|:---|:---|
| Verified | Evidence source stability | Reference collection began from clean commit `0ec4d3e1c415944227d0754d170edb0addc1d925`; every summary records empty source-status and source-patch digests, and source lock SHA-256 `42cf76614351260bf946633ab9b23341d6053a491d0632258d49b53d36a66e20` binds 28 files and eight candidate contracts. |
| Verified | Complete local tests | The promotion revision recorded by the handoff manifest passed all 241 tests. |
| Verified | Generated-file consistency | Native layout, flat mirrors, verified baselines, taxonomy, claims, review packets, and generated docs passed together. |
| Verified | Five score packets | Five committed summaries cover valid create-once seeds 0–4 attempts for NanoGPT, DLRM, anomaly detection, ResNet-18, and MobileNetV2. All individual runs and medians passed. |
| Verified | Inference chain | The committed NanoGPT training, prefill, and decode summaries share the verified seed-4 checkpoint lineage; prefill CV is `4.60%`, decode CV is `2.09%`, and the committed SLM summary records five passing outer executions with CV `0.86%`. |
| Verified | Actual validation presets | Fresh `smoke`, `coverage`, `max`, and `release` executions passed 12, 30, 30, and 60 workload-profile checks with zero run, grade, provenance, or review-contract failures. |
| Verified | Package portability | The retained NanoGPT lineage archive uses package schema 0.2 and passed 56 index, hash, size, path, extraction, and source-verification checks. Thirty-five policy-permitted run packages verified; five DLRM packages were correctly policy-blocked. |
| Verified | Clean wheel environment on macOS | The wheel installed in a clean Python 3.12.13 environment outside the checkout; dependency checks, packaged evidence checks, installed audit, installed smoke, and provenance verification passed. |
| Pending | Clean wheel environment on Linux | The hosted Linux wheel job must pass on the pushed review revision. |
| Verified | Lab and tutorial regression | All three lab smokes and Tutorial 01 smoke passed on CPU without network access. |
| Verified | Automated site review | All 43 pages rendered; required outputs, internal links, and 47 supported external HTTP links passed with zero errors. |
| Pending | Site visual review | Representative desktop and narrow browser views still need inspection. The in-app browser was unavailable during the local review. |
| Verified | Paper review | The PDF built from synchronized evidence; both verifier passes and visual inspection of all six pages succeeded. |
| Pending | Hosted CI evidence | Development and full benchmark workflows are green for the same review revision and their artifacts are retained. Local evidence does not imply this hosted gate passed. |
| Verified | Review notes | `README.md`, `PROPOSAL.md`, `PUBLIC_RULES.md`, and this ledger state the independent-preview status, supported install, evidence revision, known limitations, and non-endorsement. |

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
rather than overwriting the evidence from `0ec4d3e1`.

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
inference chain uses the median-quality seed-4 training run. The inference
sweep verifies that package, rejects unsafe or unindexed ZIP members, stages the
checkpoint, report, and manifest inside its create-once attempt, and records
only attempt-relative lineage paths. Replace the manifest placeholder below
with the retained seed-4 directory or another explicitly justified passing
training run from a new sweep.

```bash
NANOGPT_TRAIN_MANIFEST="<clean-training-attempt>/seed_4/nanogpt-train_max.provd.json"
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

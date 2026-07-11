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
| Implemented | Strong score gates | NanoGPT `<=2.30`, DLRM `>=0.70`, anomaly AUROC `>=0.95`, ResNet `>=0.85`, MobileNetV2 `>=0.78`. | Five clean reference packets. |
| Implemented | Report-level public review contract | `src/mlperf/contracts.py` checks data mode, seed, quality, timing, checkpoint or model lineage, and artifacts. | Actual `max` and `release` validation with zero contract failures. |
| Implemented | Repeatable inference timing | NanoGPT prefill and decode plus pinned SmolLM2 baseline record warmups, measured runs, and median, p90, p99 statistics. | Five retained reference executions per candidate. |
| Implemented | Training-to-inference lineage | NanoGPT inference requires the training checkpoint and records its SHA-256 digest and quality dependency. | One same-revision train, prefill, and decode chain. |
| Implemented | SLM task-quality fixture | Pinned SmolLM2 revision and bundled four-case continuation suite with perplexity gate. | Baseline run, model dossier, fixture digest, and report contract. |
| Implemented | Honest quantized boundary | Dynamic-int8 SLM row is systems-only after failing the current quality-parity calibration. | Registry, docs, and packet regeneration. |
| Implemented | Exact report provenance | Manifest binds canonical report content and exact report bytes with SHA-256 and size evidence. | Manifest and package portability tests. |
| Implemented | Portable package schema 0.2 | Relative paths, complete artifact index, digest and byte-size checks, and clean-extraction verification. | Package a fresh public-candidate run and verify after extraction. |
| Implemented | Five-seed sweep tool | Fresh-process seeds, seed/report/manifest agreement, grading, immutable attempts, artifact index, digest sidecar, and portable NanoGPT training-lineage staging. | Five clean score and performance packets from the final revision. |
| Implemented | Systems-only execution boundaries | Every systems-only row declares the exact max data mode, whether fetched or declared assets are used, and whether a quality target is enforced. | Registry validation and generated-page truth tests. |
| Implemented | Classroom entry points | Three labs accept literal `--smoke`; Tutorial 01 has a noninteractive provenance smoke. | Four commands pass on CPU without network. |
| Implemented | Generated review site | Registry, dataset, benchmark, and CLI pages are generated; Quarto and link gates exist. | Regenerate, render, link-check, and visually inspect. |
| Implemented | Blocking and long CI | Development workflow runs tests, smoke, labs, wheel, registry, docs, and site checks; scheduled/manual workflow executes actual `max` or `release` work with a five-hour limit. | Green workflow runs on the final pushed revision. |
| Implemented | Guarded site publication | Live preview requires recent development and full benchmark validation plus manual `PUBLISH` confirmation. | Confirm deployment only after the guarded workflow succeeds. |
| Implemented | Evidence-synchronized paper build | Paper Makefile generates a registry snapshot and verifies the PDF. | Clean build, verification, and visual inspection. |

## Pending Repository Evidence

| **Status** | **Gate** | **Required Evidence** |
|:---|:---|:---|
| Pending | Final source stability | One identified commit with no uncommitted or untracked source changes before evidence collection. |
| Pending | Complete local tests | Full `uv run pytest` output from the final revision. |
| Pending | Generated-file consistency | Native layout, flat mirrors, taxonomy, review packets, and docs checks all pass together. |
| Pending | Five score packets | One valid create-once seeds 0–4 packet for each score-bearing candidate. |
| Pending | Inference chain | One quality-passing NanoGPT checkpoint followed by five eligible prefill and five eligible decode executions; promote the completed five-run SLM development calibration through a clean public-candidate packet. |
| Pending | Actual validation presets | Fresh `smoke`, `coverage`, `max`, and `release` executions with zero run, grade, provenance, or review-contract failures. |
| Pending | Package portability | Fresh manifest packaged, indexed, extracted in a temporary directory, and verified there. |
| Pending | Clean wheel environments | Wheel installed outside the checkout on supported Python and at least Linux plus the primary macOS laptop environment. |
| Pending | Lab and tutorial regression | All three lab smokes and Tutorial 01 smoke pass on CPU without network access. |
| Pending | Site review | Generated docs current, Quarto render complete, links clean, and representative desktop and narrow views visually inspected. |
| Pending | Paper review | PDF builds from synchronized registry evidence, verifier passes, and every page receives visual inspection. |
| Pending | CI evidence | Development and full benchmark workflows are green for the same review revision and their artifacts are retained. |
| Pending | Release notes | Notes state independent-preview status, supported install, evidence revision, known limitations, and non-endorsement. |

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
uv run python tools/check_taxonomy.py
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
quarto render site
python3 ../shared/scripts/check-internal-links.py site --quiet
make -C paper clean all
make -C paper check
```

The strict public audit is expected to fail until the MovieLens decision is
closed. For an independent preview review packet, retain that failing JSON
audit as an explicit external blocker and inspect every warning. Do not silently
remove the command or represent the gate as passed. A public or endorsed
release still requires exit zero.

## Score Reference Commands

Run these commands only after the source-cleanliness guard passes. The tool's
default output root is `~/.mlperf-edu/reference_runs`.

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

Package one passing run from the clean NanoGPT training sweep. The inference
sweep verifies that package, rejects unsafe or unindexed ZIP members, stages the
checkpoint, report, and manifest inside its create-once attempt, and records
only attempt-relative lineage paths. Replace the manifest placeholder below
with one retained seed directory from the completed training packet.

```bash
NANOGPT_TRAIN_MANIFEST="<clean-training-attempt>/seed_0/nanogpt-train_max.provd.json"
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

## Review-Ready Decision

The repository is ready to show MLCommons when every in-repository `Pending`
row is `Verified`, the retained packet names the exact source revision, and the
external rows are presented as explicit questions rather than concealed gaps.
It is ready for a public or endorsed release only after the applicable external
rows are closed in writing.

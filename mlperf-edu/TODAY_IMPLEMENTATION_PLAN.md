# MLPerf EDU Implementation Plan

## Objective

This plan is about producing the highest-quality implementation artifacts we
can, not narrowing the project to a small demo. The work should move MLPerf EDU
toward its north star: a SPEC-like academic ML systems benchmark suite that
researchers can actually run, cite, compare against, and extend. The work may
take a day, a week, or longer; the priority is that each completed artifact is
runnable, auditable, and defensible.

The immediate implementation focus is the user-facing workflow:

```bash
mlperf init
mlperf list
mlperf fetch
mlperf audit
mlperf run
```

These commands must become the clean path for preparing the local environment,
getting workloads, checking that they are properly sourced and runnable, and
producing benchmark evidence. Supporting commands such as `doctor`, `info`,
`report`, `grade`, and `cache` should make the experience discoverable without
adding new public concepts beyond suite, profile, workload, and variant.

## Live Status Board

This is the current checklist to keep the work visible while implementation
continues. Update this board every time a workstream or workload moves.

### Workstream Checklist

| Status | Workstream | Current checkpoint |
|---|---|---|
| Done | Public CLI vocabulary | User-facing selection is suite/profile/workload/variant with profiles `min`, `max`, and `pro` |
| Done | Canonical CLI selection | Canonical workloads expand to variants; `--variant` narrows to one variant; dry runs preview selections |
| Done | Default report artifacts | Runs emit JSON, HTML, CSV, and `.provd.json` artifacts by default |
| Done | Dataset/public-release policy | Asset dossiers and reports carry release status, policy, and next step |
| Done | Reference-run comparability policy | Score-bearing workloads require reference-run protocol metadata |
| Done | Canonical list UX | `mlperf list` leads with public workload selectors and keeps internal IDs as metadata |
| Done | Training-to-inference provenance | Checkpoint-backed inference reports include source workload quality and artifact policy |
| Done | Expert-review packets | Nine public-result candidate packets are generated under `review_packets/` |
| Done | Expert-review packet freshness check | `python3 tools/generate_review_packets.py --check` verifies generated packets are current |
| Done | Paper and README alignment | Public docs now describe 30 workload rows, suite/profile/workload/variant selection, public-result status, and the current native registry source of truth |
| Done | Native registry layout reader | `load_registry()` now accepts `suites/<suite>/<workload>/workload.yaml` plus `variants/*.yaml`; current full test baseline is 52 passing tests |
| Done | Native registry generated mirror and default loader | `registry/suites/...` contains all 30 current rows; default local registry load prefers native files; registry tests, list, audit, and export check pass |
| Done | Native source-of-truth cleanup | `registry/` is documented as the edit source; `tools/export_flat_registry.py --check` verifies `workloads.yaml` as the compatibility mirror |
| Done | Native registry mirror checks | `export_registry_layout.py --check`, `export_flat_registry.py --check`, review packet check, and focused registry/review tests pass |
| Done | Final pytest pass | Full `pytest -q` passed 52 tests after the latest runnable-suite changes |
| Done | Fresh full validation pass | Historical pass superseded by the Post-Gutenberg validation below |
| Done | Post-native-default validation | Historical pass superseded by the Post-Gutenberg validation below |
| Done | Public-audit blocker routing | Default `mlperf audit` is the clean local development contract; `mlperf audit --policy public --format json` is the stricter endorsement check and fails on unresolved public-release warnings such as MovieLens-100K approval |
| Done | MovieLens local fetch/run verification | `mlperf fetch --workload micro-dlrm-train --profile max` downloads MovieLens-100K; current `mlperf run --workload micro-dlrm-train --profile max` passed with accuracy 0.705 against target 0.70 in 2.6s and `mlperf verify` passed |
| Done | Recommender suite MovieLens verification | `micro-dlrm-dram-train` also passed on the fetched MovieLens-100K path with accuracy 0.6921 against target 0.65, 16 MiB working set, and verified provenance |
| Done | Public vision dataset decision | `resnet18-train` and `mobilenetv2-train` now use MIT-licensed Fashion-MNIST by default; ResNet default passed at 0.773 accuracy in 7.4s, MobileNet default passed at 0.727 accuracy in 24.2s |
| Done | Public language dataset decision | `nanogpt-train` TinyShakespeare now regenerates from Project Gutenberg eBook 100 with a recorded recipe; max run passed with val_loss 2.045 against target 2.3 in 86.4s and fetch reports `public-ok-fetch-only` |
| Done | Post-Gutenberg full validation | Historical checkpoint superseded by the current 52-test baseline and final 30/30 max validation |
| Done | Post-documentation cleanup | Historical checkpoint superseded by the current public-text synchronization row |
| Done | DLRM pass-margin polish | Fixed MovieLens text occupation encoding, saves the best validation checkpoint, and grades `best_accuracy`; seeds 0-4 now all pass target 0.70 with best_accuracy in [0.7019, 0.7094] |
| Done | Post-DLRM full validation | Historical checkpoint superseded by the later SLM/model-release/full-validation passes |
| Done | SLM functional-margin polish | Max SLM runs now request 16 decode tokens while retaining an 8-token functional target; real SmolLM2 baseline, dynamic-int8, batched, and long-context variants all passed grade and provenance verification |
| Done | Post-SLM full validation | Historical checkpoint superseded by the current packaged 30/30 max validation and local/public warning split |
| Done | Post-best-checkpoint full validation | Historical checkpoint superseded by the current packaged 30/30 max validation and DLRM `best_accuracy` grading |
| Done | SLM model release UX | Hugging Face models with permissive licenses now report `release=public-ok-with-attribution`; `mlperf audit --suite slm --format json` passes with 0 warnings |
| Done | Post-model-release full validation | Historical checkpoint superseded by the current packaged 30/30 max validation; SLM model reports still carry `public-ok-with-attribution` |
| Done | First-user CLI journey | `mlperf init --suite slm --profile min` passes doctor, runs min-profile smoke validation, writes JSON/HTML/CSV/provenance artifacts, and prints next `fetch`, `run`, and `report` commands |
| Done | CLI help clarity | `--profile`, `--suite`, and `--workload` help now explain min/max/pro and suite/workload selection consistently across user commands |
| Done | Post-help full validation | `pytest -q` passed 52 tests; registry mirror and review packet checks pass; local audit passed; `mlperf validate max --keep-going --skip-doctor --output-dir submissions/validation` passed 30/30 in 93.4s |
| Done | Public-text synchronization | README, SPEC, PROPOSAL, and paper text now use the current 30-workload/60-manifest counts, canonical workload selectors, and registry source-of-truth wording |
| Done | Plain pytest entry point | Plain `pytest -q` now passes 52/52 from the repo root, with repo and `src` import paths declared in `pyproject.toml` |
| Done | Local/public warning split | Packaged `mlperf audit` passes with 0 warnings; packaged `mlperf audit --policy public` fails with the expected single MovieLens-100K endorsement warning |
| Done | Packaged first-user journey | Packaged `mlperf init --profile min` regenerated fresh JSON/HTML/CSV/provenance artifacts; `mlperf grade` passed 12/12 with 0 warnings and `mlperf verify` passed |
| Done | Current smoke validation | Packaged `mlperf validate smoke --keep-going --skip-doctor` passed 12/12 in 11.9s with 0 warnings |
| Done | Current full max validation | Packaged `mlperf validate max --keep-going --skip-doctor` passed 30/30 in 95.3s with 0 warnings |
| Done | Current release validation | Packaged `mlperf validate release --keep-going --skip-doctor` passed 60/60 in 115.9s with 0 warnings |
| Done | uv packaging contract | `INSTALL.md` documents `uv sync`, `uv tool install`, and `uv build`; packaged `src/mlperf_edu/workloads.yaml` is now a real wheel data file checked by `tools/export_flat_registry.py --check` |
| Done | Release/quality review docs | `RELEASE_CHECKLIST.md` and `QUALITY_TARGET_REVIEW.md` capture the remaining non-runtime work for public release and expert target review |
| Open | Public dataset release decisions | `DATASET_RELEASE_REVIEW.md` records the remaining MovieLens-100K decision needed for endorsement; MovieLens remains runnable locally |

Last validation artifacts:

- First-user init directory: `submissions/rocksolid-init-min-current/`
- First-user aggregate HTML: `submissions/rocksolid-init-min-current/mlperf_edu_min_20260629_012126.html`
- First-user grade summary: `submissions/rocksolid-init-min-current/grade.json`
- Smoke validation JSON/HTML/CSV: `submissions/rocksolid-smoke-current/mlperf_validate_smoke_20260629_012200.*`
- Smoke workload CSV: `submissions/rocksolid-smoke-current/mlperf_validate_workloads_smoke_20260629_012200.csv`
- Max aggregate JSON/HTML/CSV: `submissions/rocksolid-max-current/max-all/mlperf_edu_max_20260629_012341.*`
- Max grade summary: `submissions/rocksolid-max-current/max-all/grade.json`
- Max validation JSON/HTML/CSV: `submissions/rocksolid-max-current/mlperf_validate_max_20260629_012349.*`
- Max workload CSV: `submissions/rocksolid-max-current/mlperf_validate_workloads_max_20260629_012349.csv`
- Release aggregate directories: `submissions/rocksolid-release-current/min-all/` and `submissions/rocksolid-release-current/max-all/`
- Release grade summaries: `submissions/rocksolid-release-current/min-all/grade.json` and `submissions/rocksolid-release-current/max-all/grade.json`
- Release validation JSON/HTML/CSV: `submissions/rocksolid-release-current/mlperf_validate_release_20260629_013241.*`
- Release workload CSV: `submissions/rocksolid-release-current/mlperf_validate_workloads_release_20260629_013241.csv`
- Result: 60/60 release manifests passed in 115.9 seconds with 0 blockers and 0 local warnings; `mlperf audit --policy public` still reports the single MovieLens-100K public-release decision.

### Workload Checklist

Run state tracks whether the workload is selectable and covered by the latest
recorded validation loop in this plan. Ship state tracks whether the workload is
ready to be treated as a public-result candidate.

| Run state | Ship state | Workload selector | Suite | Role | Profiles | Current note |
|---|---|---|---|---|---|---|
| Done | Systems-only | `nano-rag-agent` | agent | agent | min, max, pro | Runnable teaching/research row; public-result quality contract still deferred |
| Done | Systems-only | `nano-codegen-agent` | agent | agent | max, pro | Runnable teaching/research row; public-result quality contract still deferred |
| Done | Systems-only | `nano-react-agent` | agent | agent | max, pro | Runnable teaching/research row; public-result quality contract still deferred |
| Done | Systems-only | `nano-toolcall-agent` | agent | agent | max, pro | Runnable teaching/research row; public-result quality contract still deferred |
| Done | Systems-only | `micro-dlrm-distributed` | distributed | distributed | min, max, pro | Runnable systems row; distributed comparability policy still deferred |
| Done | Systems-only | `micro-gnn-train` | graph | training | min, max, pro | Runnable systems row; public-result review packet not yet targeted |
| Done | Public candidate | `nanogpt-train` | language | training | min, max, pro | Score-bearing; reference protocol and review packet generated |
| Done | Public candidate | `nanogpt-inference --variant prefill` | language | inference | min, max, pro | Performance-bearing; checkpoint lineage and review packet generated |
| Done | Public candidate | `nanogpt-inference --variant decode` | language | inference | min, max, pro | Performance-bearing; checkpoint lineage and review packet generated |
| Done | Systems-only | `micro-bert-train` | language | training | max, pro | Runnable systems row; public-result quality contract still deferred |
| Done | Systems-only | `nano-lora-finetune` | language | optimization | max, pro | Runnable optimization row; public-result quality contract still deferred |
| Done | Systems-only | `nano-moe-train` | language | optimization | max, pro | Runnable optimization row; public-result quality contract still deferred |
| Done | Systems-only | `nanogpt-inference --variant fp16-b16` | language | optimization | max, pro | Runnable optimization variant; public-result packet not targeted |
| Done | Systems-only | `nanogpt-inference --variant fp32-b16` | language | optimization | max, pro | Runnable optimization variant; public-result packet not targeted |
| Done | Systems-only | `nanogpt-inference --variant speculative` | language | test-time compute | max, pro | Runnable TTC variant; public-result contract still deferred |
| Done | Public candidate | `micro-dlrm-train` | recommender | training | min, max, pro | Score-bearing; text occupation encoding fixed; default max reaches best_accuracy 0.705 against target 0.70 in 2.4s; seeds 0-4 all pass using best validation checkpoint; MovieLens release approval still needed |
| Done | Systems-only | `micro-dlrm-dram-train` | recommender | optimization | max, pro | Runnable memory-pressure variant; fetched MovieLens max run passed with verified provenance; public-result packet not targeted |
| Done | Systems-only | `micro-rl-train` | rl | training | min, max, pro | Runnable systems row; public-result quality contract still deferred |
| Done | Public candidate | `smollm2-chat-inference --variant baseline` | slm | inference | min, max, pro | Performance-bearing; max default generates 16 tokens against an 8-token functional target; permissive HF model dossier and review packet generated |
| Done | Public candidate | `smollm2-chat-inference --variant quantized-int8` | slm | optimization | max, pro | Performance-bearing optimization variant; max default generates 16 tokens against an 8-token functional target; review packet generated |
| Done | Systems-only | `smollm2-chat-inference --variant batched-b4` | slm | inference | max, pro | Runnable serving variant; max default generates 16 tokens per request against an 8-token functional target; public-result packet not targeted |
| Done | Systems-only | `smollm2-chat-inference --variant long-context` | slm | inference | max, pro | Runnable long-context variant; max default generates 16 tokens against an 8-token functional target; public-result packet not targeted |
| Done | Systems-only | `micro-lstm-train` | timeseries | training | min, max, pro | Runnable systems row; public-result quality contract still deferred |
| Done | Public candidate | `anomaly-ae-train` | tiny | training | min, max, pro | Score-bearing; reference protocol and review packet generated |
| Done | Systems-only | `dscnn-kws-train` | tiny | training | max, pro | Runnable systems row; Speech Commands public-result policy still deferred |
| Done | Systems-only | `wake-vision-vww` | tiny | inference | max, pro | Runnable systems row; large dataset policy still deferred |
| Done | Public candidate | `resnet18-train` | vision | training | min, max, pro | Score-bearing; default max now uses Fashion-MNIST and passed target 0.75 with measured 0.773 accuracy; review packet generated |
| Done | Public candidate | `mobilenetv2-train` | vision | training | max, pro | Score-bearing; default max now uses Fashion-MNIST and passed target 0.70 with measured 0.727 MPS accuracy; CPU fallback optimization remains open |
| Done | Systems-only | `mobilenet-cifar100-composed-fp16` | vision | optimization | max, pro | Runnable optimization row; public-result packet not targeted |
| Done | Systems-only | `micro-diffusion-train` | vision | training | max, pro | Runnable systems row; public-result quality contract still deferred |

## Implementation Contract

The public user model stays simple:

- `suite`: workload domain, such as `slm`, `vision`, `language`, or `tiny`.
- `profile`: run scale and research surface: `min`, `max`, `pro`.
- `workload`: one benchmark ID.

The profiles mean:

| Profile | Meaning | Implementation direction |
|---|---|---|
| `min` | Minimum representative benchmark path, ideally one workload from each major suite | Used for install checks, CI smoke, demos |
| `max` | Full MLPerf EDU suite at comparable scale | Used for papers, assignments, artifact evaluation |
| `pro` | Research envelope exposing controlled variants and optimization knobs | Used for architecture, systems, compiler, pruning, quantization, serving, and test-time compute studies |

The implementation may use internal collections and validation presets, but
those should not leak into the public CLI.

Selection semantics:

- With no `suite` or `workload`, `profile` selects the default workload set:
  `min`, `max`, or `pro`.
- With `suite`, the suite selects the workload domain and `profile` selects the
  run scale/context.
- With an exact workload ID, the command targets that one registry row.
- With a canonical workload ID and no `variant`, the command targets the
  workload family, meaning every variant under that canonical workload.
- With a canonical workload ID and `variant`, the command targets exactly that
  variant.

Variants are subordinate to workloads. Quantization, pruning, sparsity, precision,
batching, long-context settings, backend changes, energy measurement, and
test-time compute budgets should normally be variants or measured modes under a
canonical workload, not separate top-level workloads. Create a new top-level
workload only when the model/task/dataset identity changes enough that a user
would reasonably expect a distinct benchmark.

## Quality Bar

Every artifact we produce should satisfy this bar:

1. **Runnable:** command works from the repo without hidden manual steps.
2. **Auditable:** source, license, dataset, model, and quality status are explicit.
3. **Creditable:** off-the-shelf code, models, datasets, and papers are recorded.
4. **Reproducible:** reports include enough provenance to rerun or inspect.
5. **Comparable:** metrics, data mode, profile, backend, and hardware are recorded.
6. **Honest:** synthetic or micro-sharded data is labeled and never presented as a public score.
7. **Extensible:** new training, inference, and test-time compute workloads can be added through the same registry path.

## North Star Goals

MLPerf EDU should feel like the academic ML systems benchmark suite people wish
MLPerf had been for classroom and research use: easy enough to run on a laptop,
structured enough to compare across papers, and transparent enough that students
can understand what happened.

The durable goals are:

1. **SPEC-like usability:** a new user can install, fetch, audit, run, and inspect
   results without building a private engineering stack.
2. **MLPerf-shaped credibility:** workloads should map to canonical ML systems
   shapes: vision, language, recommender, tiny/edge, agents, serving, training,
   inference, and test-time compute.
3. **Academic relevance:** `pro` should expose the research surfaces that show up
   in ISCA, MICRO, HPCA, ASPLOS, MLSys, NeurIPS systems papers, and artifact
   evaluation: quantization, pruning, sparsity, batching, serving, compiler
   backends, memory behavior, distributed training, RAG, tool use, and
   speculative decoding.
4. **Classroom reliability:** `min` must be the path an instructor can assign
   with confidence; it should run quickly, produce a clear report, and explain
   synthetic or tiny assets honestly.
5. **Public release discipline:** every public workload must have source,
   license, citation, dataset, model, profile, quality, and report metadata.
6. **Research extensibility:** new workloads should be added through registry
   metadata and runner interfaces, not one-off scripts.
7. **Provenance first:** if inference consumes trained weights, the report should
   show the training run; if it consumes external weights, the report should show
   the upstream model provider, revision, tokenizer, license, and rationale.

## Core Workflow UX

The CLI should support a short primary path and a small set of discoverability
commands.

Primary path:

```bash
mlperf init
mlperf list
mlperf fetch
mlperf audit
mlperf run
```

Supporting path:

```bash
mlperf doctor
mlperf info
mlperf report
mlperf grade
mlperf cache
```

The user should never have to know internal registry terms to answer basic
questions: what can I run, what will it download, what variants exist, what did
the report mean, and whether this result is publishable.

### `mlperf list`

Purpose:

- Show available suites, profiles, workloads, variants, and public-result status.
- Make the benchmark discoverable before the user fetches assets.
- Explain enough naming context that users understand what they are about to run.

Required user paths:

```bash
mlperf list
mlperf list suites
mlperf list profiles
mlperf list workloads
mlperf list workloads --suite slm
mlperf list matrix
mlperf list matrix --profile max
mlperf list --profile min
mlperf list --profile max
mlperf list --profile pro
mlperf list --workload nanogpt-inference
mlperf list --workload nanogpt-inference --variant prefill
mlperf list --workload nanogpt-inference --profile min
mlperf list variants --workload smollm2-chat-inference
mlperf list --format json
```

Implementation requirements:

- `list workloads` shows workload ID, suite, model/task, type, default profile
  membership, public status, and whether assets are local.
- `list variants --workload <id>` shows baseline and optimization variants under
  that canonical workload.
- `list matrix` shows workload, canonical run selector, suite, default profile
  membership, role, public-result status, dataset, and quality target.
- `list --profile min|max|pro` shows exactly what the default run for that
  profile would select.
- `list --workload <canonical>` expands to variants under that canonical
  workload; `--variant` narrows to one variant. `--profile` records the
  requested run context but does not silently filter an explicit workload or
  suite selection.
- `suite`, `profile`, `workload`, and `variant` filters compose consistently
  across `list`, `list matrix`, `run`, `fetch`, and `audit`.
- JSON output is stable enough for tests and docs generation.

### `mlperf info`

Purpose:

- Explain one suite, profile, workload, variant, model, dataset, or run artifact.
- Give the user a readable dossier before they run anything.

Required user paths:

```bash
mlperf info --suite slm
mlperf info --profile pro
mlperf info --workload smollm2-chat-inference
mlperf info --workload smollm2-chat-inference --variant quantized-int8
mlperf info --model smollm2-135m
mlperf info --dataset tinyshakespeare
mlperf info --run /tmp/mlperf-edu-min/mlperf_edu_min_<timestamp>.json
```

Implementation requirements:

- Workload info includes task, suite, variants, profiles, runner, scenario,
  model/dataset rationale, source/license/citation status, and expected outputs.
- Variant info explains what changes relative to baseline and which fields must
  appear in reports, such as bit-width, backend, calibration data, batch size,
  context length, or TTC budget.
- Run info summarizes report artifacts, provenance manifests, warnings,
  blockers, and grade status.

### `mlperf doctor`

Purpose:

- Check local environment readiness without running a benchmark.
- Explain missing dependencies, backend availability, hardware detection, cache
  paths, and optional asset requirements.

Required user paths:

```bash
mlperf doctor
mlperf doctor --profile min
mlperf doctor --suite slm --profile pro
mlperf doctor --workload smollm2-chat-inference --variant quantized-int8
mlperf doctor --profile min --format json
```

Implementation requirements:

- `doctor` checks Python, package imports, PyTorch, optional backends, disk/cache
  paths, network availability when needed, and hardware capabilities.
- `doctor` accepts the same public selection concepts as the main workflow:
  `suite`, `profile`, `workload`, and `variant`.
- `doctor --format json` emits stable checks and selected-workload metadata for
  CI, course setup scripts, and packaging smoke tests.
- Missing optional acceleration should be a warning; missing required runtime
  dependencies should be a blocker.

### `mlperf init`

Purpose:

- Prepare a fresh checkout for MLPerf EDU without requiring the user to know
  where caches, datasets, model weights, reports, or hardware metadata live.
- Run environment checks and create stable local directories.
- Optionally perform a smoke run so the user knows the install is real.
- Make the next commands obvious: `fetch`, `audit`, and `run`.

Required user paths:

```bash
mlperf init
mlperf init --profile min
mlperf init --profile max
mlperf init --suite slm --profile max
mlperf init --workload nanogpt-train --profile max
mlperf init --no-smoke
```

Implementation requirements:

- `init` runs `doctor`-style environment checks.
- `init` prints cache locations for datasets, models, reports, and submissions.
- `init` records hardware/backend capabilities.
- `init` invokes or previews the relevant fetch plan.
- `init --profile min` should be the cleanest first-run path for a new user.
- `init --no-smoke` prepares directories and assets without running workloads.
- `init` should never hide license, account, restricted-model, or large-download requirements.

Acceptance checks:

```bash
mlperf init --profile min --output-dir /tmp/mlperf-edu-init
mlperf init --suite slm --profile min --output-dir /tmp/mlperf-edu-init-slm
mlperf init --workload nanogpt-train --profile min --output-dir /tmp/mlperf-edu-init-nanogpt
```

### `mlperf fetch`

Purpose:

- Resolve assets needed by a workload, suite, or profile.
- Download or verify datasets, model weights, prompt sets, tokenizer files, and local micro-shards.
- Show exactly what will be fetched before doing work.
- Record source metadata and cache locations.

Required user paths:

```bash
mlperf fetch --profile min
mlperf fetch --profile max
mlperf fetch --profile pro
mlperf fetch --suite slm --profile max
mlperf fetch --suite slm --profile pro --model smollm2-135m
mlperf fetch --workload nanogpt-train --profile max
mlperf fetch --dry-run --profile max
```

Implementation requirements:

- `--dry-run` prints asset plan without mutating local state.
- Fetch output includes workload, asset type, URL/source, cache path, expected size if known, and license if known.
- Fetch uses deterministic cache locations under the configured MLPerf EDU data/model cache.
- Fetch never silently downgrades from real data to synthetic data in `max`.
- Fetch is allowed to use synthetic/tiny assets for `min`, but must label that in the later report.
- Fetch supports Hugging Face model aliases for SLM workloads.
- Fetch records model/dataset source metadata so reports and provenance can include it.

Acceptance checks:

```bash
mlperf fetch --dry-run --profile min
mlperf fetch --dry-run --profile max
mlperf fetch --dry-run --suite slm --profile max --model smollm2-135m
mlperf fetch --dry-run --workload nanogpt-train --profile max
```

### `mlperf audit`

Purpose:

- Check whether workload registry entries are complete and honest.
- Check public result eligibility.
- Check source, license, citation, model, dataset, runner, profile, and quality metadata.
- Catch missing credit before anything is packaged as MLPerf EDU.

Required user paths:

```bash
mlperf audit
mlperf audit --suite slm
mlperf audit --status score-bearing
mlperf audit --workload smollm2-chat-inference
mlperf audit --workload smollm2-chat-inference --variant quantized-int8
mlperf audit --format json
```

Implementation requirements:

- Score-bearing workloads must declare real dataset source, quality metric,
  target value, target direction, target basis, reference-run count, variance
  summary, reviewer notes, verified baseline, source metadata, and scenario.
- Score-bearing target basis is controlled vocabulary:
  `reference_runs`, `literature`, `mlcommons_derived`, or
  `pedagogical_baseline`.
- Reference-run score-bearing targets require at least three reference runs and
  a variance summary with `runs`, `statistic`, and `acceptance_rule`.
- Performance-bearing workloads must declare a deterministic functional check,
  performance metric, scenario, source metadata, and source license where a
  model source is used.
- Performance-bearing workloads that consume a shared checkpoint must declare a
  `quality_dependency` so the training-to-inference chain is auditable.
- Systems-only workloads must still declare source metadata and explain why they are systems-only.
- Experimental workloads must not pass as public-release-ready.
- Audit JSON should be stable enough for CI and future MLCommons review.
- Audit should distinguish warnings from blockers.
- Audit should fail on missing required source/license/citation metadata for public workloads.

Acceptance checks:

```bash
mlperf audit
mlperf audit --suite slm --format json
mlperf audit --status score-bearing --format json
pytest tests/test_registry.py -q
```

### `mlperf run`

Purpose:

- Execute a workload, suite, or profile.
- Produce benchmark evidence: JSON, HTML, CSV, provenance manifest, and optional package/grade outputs.
- Support training, inference, and test-time compute.

Required user paths:

```bash
mlperf run --profile min
mlperf run --profile max --dry-run
mlperf run --profile max
mlperf run --profile pro
mlperf run --suite slm --profile max --model smollm2-135m
mlperf run --suite slm --profile pro --model smollm2-135m
mlperf run --workload nanogpt-train --profile max
mlperf run --workload smollm2-chat-inference --variant baseline
mlperf run --workload smollm2-chat-inference --variant quantized-int8
mlperf run --workload nanogpt-inference --variant decode
```

Implementation requirements:

- `run --profile min` uses the representative cross-suite path.
- `run --profile max` runs the full suite at comparable scale.
- `run --profile pro` uses research-envelope behavior and records variant metadata.
- Suite and workload filters override the default profile selection.
- A canonical workload without `--variant` runs every variant under that
  workload family; use `--variant` for exactly one variant.
- `run --dry-run` prints the selected workloads and canonical run selectors
  without running models or writing artifacts.
- Run auto-fetches missing required assets only when the fetch plan is deterministic and safe.
- Run reports must include source, license, citation, model source, dataset source, data mode, hardware, backend, profile, metrics, and quality status.
- Reports must distinguish required quality checks from informational targets,
  especially for `min` smoke runs where `quality_required=false`.
- Every `run`, including a single-workload run, writes JSON, HTML, and CSV by
  default unless the user explicitly disables a format.
- Run always writes `.provd.json` manifests for workload evidence.
- Run must label synthetic, tiny, random, or micro-sharded data clearly.
- The default aggregate run should make the HTML report the primary human
  artifact. `--open-report` should open it after generation when the environment
  supports that, while JSON and CSV remain first-class machine-readable outputs.
- The terminal summary must print the exact output paths for HTML, JSON, CSV,
  logs, and provenance manifests.

Acceptance checks:

```bash
mlperf run --profile min --output-dir /tmp/mlperf-edu-min
mlperf run --workload nanogpt-train --profile min --output-dir /tmp/mlperf-edu-one
mlperf run --suite slm --profile min --output-dir /tmp/mlperf-edu-slm-min
mlperf report /tmp/mlperf-edu-min/mlperf_edu_min_<timestamp>.json --format html
mlperf grade /tmp/mlperf-edu-min --output /tmp/mlperf-edu-min/grade.json
```

### `mlperf report`

Purpose:

- Regenerate or open human- and machine-readable reports from run artifacts.
- Make HTML the default inspection path while preserving JSON and CSV exports.

Required user paths:

```bash
mlperf report /tmp/mlperf-edu-min/mlperf_edu_min_<timestamp>.json
mlperf report /tmp/mlperf-edu-min/mlperf_edu_min_<timestamp>.json --format html --open
mlperf report /tmp/mlperf-edu-min/mlperf_edu_min_<timestamp>.json --format csv
mlperf report /tmp/mlperf-edu-min/mlperf_edu_min_<timestamp>.json --format json
```

Implementation requirements:

- `report` reads existing run artifacts without re-running workloads.
- `report --open` opens the HTML artifact when supported and prints the path.
- Report generation must preserve provenance, execution status, quality status,
  and any policy metadata already present in the source artifact.
- `report` should be idempotent: re-running it on the same run directory should
  update derived HTML/CSV views without changing benchmark measurements.
- `mlperf report <run-directory>` auto-selects the latest aggregate JSON in that
  directory, falling back to the latest workload report when no aggregate report
  exists.

### `mlperf grade`

Purpose:

- Decide whether a run is locally complete and artifact-evaluation-ready.

Required user paths:

```bash
mlperf grade /tmp/mlperf-edu-min
mlperf grade /tmp/mlperf-edu-min --output /tmp/mlperf-edu-min/grade.json
```

Implementation requirements:

- `grade` should keep local verification pass/fail separate from
  public-release eligibility.
- `grade` must not hide failed quality targets or missing provenance needed to
  verify a run.
- Public-release warnings about license approval or endorsement readiness
  belong in `mlperf audit --policy public`, not in ordinary local grading.

### `mlperf cache`

Purpose:

- Let users inspect and verify downloaded assets without learning cache paths.

Required user paths:

```bash
mlperf cache list
mlperf cache list --profile min
mlperf cache list --profile max
mlperf cache list --suite slm
mlperf cache list --workload smollm2-chat-inference --variant quantized-int8
mlperf cache verify
mlperf cache verify --profile max
mlperf cache verify --workload smollm2-chat-inference
```

Implementation requirements:

- `cache list` shows asset type, workload, source, size if known, local path, and
  verification status.
- `cache verify` checks expected files, hashes where available, and manifest
  consistency.
- Any destructive cache cleanup should require an explicit subcommand and should
  print the exact files it will remove before removing them.

## User Journey Simulation

We should test the benchmark through the eyes of the people who must trust it,
not only through unit tests.

| User | Goal | Expected first successful path | What the report must answer |
|---|---|---|---|
| Student on a laptop | Prove setup works and understand benchmark outputs | `mlperf init --profile min`, `mlperf fetch --profile min`, `mlperf run --profile min` | What ran, how long it took, whether it passed, whether data was real or synthetic, and what command to try next |
| Instructor | Assign a lab that works across many machines | `mlperf init --profile min --no-smoke`, `mlperf audit`, `mlperf run --profile min` | Which workloads are classroom-safe, which downloads are needed, expected runtime, and where reports are written |
| Researcher | Run comparable experiments and modify one systems variable | `mlperf fetch --profile max`, `mlperf run --suite slm --profile pro --model smollm2-135m` | Baseline, variant, hardware, backend, dataset, model, quality delta, speedup, and provenance |
| Artifact evaluator | Re-run a paper artifact with enough evidence to judge it | `mlperf audit --format json`, `mlperf run --profile max`, `mlperf grade <run-dir>` | Source/license/citation completeness, quality target, reproducibility metadata, warnings, and blockers |
| MLPerf EDU maintainer | Decide whether a workload is public-release-ready | `mlperf audit --workload <id>`, `mlperf run --workload <id>`, report inspection | Missing metadata, failed checks, profile membership, report completeness, and public-result eligibility |

Simulation acceptance checks:

```bash
mlperf init --profile min --output-dir /tmp/mlperf-edu-journey-min
mlperf fetch --dry-run --profile min
mlperf run --profile min --output-dir /tmp/mlperf-edu-journey-min --open-report
mlperf audit --format json
mlperf run --suite slm --profile pro --model smollm2-135m --output-dir /tmp/mlperf-edu-journey-pro
```

The journey fails if the user cannot answer these questions from the terminal
output and generated report:

- What was selected by `suite`, `profile`, and `workload`?
- What assets were fetched, from where, under what license, and into which cache?
- Which data was real, public, synthetic, tiny, restricted, account-based, or micro-sharded?
- Which model was used, why that size was selected, and where the weights came from?
- Did training produce a checkpoint, or did inference use external pretrained weights?
- What quality target or functional check was used?
- What hardware/backend executed the run?
- What is the comparable metric, and what is only a systems or smoke metric?
- Which warnings block a public result, and which are ordinary caveats?

## Report UX Goals

The default report should be useful immediately after a run. HTML is the primary
human artifact; JSON and CSV are stable export formats.

Required report sections:

1. **Run summary:** command, timestamp, profile, suite/workload filters, pass/fail
   status, total runtime, and output directory.
2. **Hardware and backend:** CPU, GPU/MPS/CUDA availability, memory if available,
   Python/PyTorch/runtime versions, backend path, and precision.
3. **Workload table:** workload ID, suite, model/task, profile membership, type
   tags, data mode, runtime, primary metric, quality target, and status.
4. **Training-to-inference chain:** checkpoint origin, parent training run,
   external pretrained source, adapter source, or synthetic/min marker.
5. **Model dossier:** name, parameter count, size class, weight size, tokenizer or
   preprocessor, license/access, backend support, and rationale.
6. **Dataset dossier:** public source, license/terms, citation, size, version/hash,
   cache path, data mode, and score eligibility.
7. **Optimization and TTC details:** quantization/pruning/sparsity settings,
   batch/context shape, RAG retrieval depth, tool calls, speculative draft steps,
   and compute budget.
8. **Warnings and blockers:** missing metadata, synthetic data, restricted assets,
   failed quality targets, unsupported backend, and public-result eligibility.
9. **Export links:** JSON, CSV, provenance manifests, logs, and grade output.
10. **Run fingerprint:** machine-readable hardware/software/backend/data-mode
    summary with a stable fingerprint hash, visible in JSON and HTML and bound
    into the workload provenance manifest.

The report should not just say "passed." It should tell a student what happened,
tell an instructor whether it is assignable, and tell a researcher whether the
artifact is comparable enough to cite.

## Canonical Harness And Load Generator

MLPerf EDU cannot be a collection of loose run scripts. The actual benchmark code
must live in the repo as a canonical harness, with clear extension points
for datasets, models, runners, scenarios, metrics, reports, and provenance.

MLPerf Inference has a useful separation that we should preserve in an
education-friendly form:

- **Query/sample library:** owns dataset or prompt samples, loading policy,
  preprocessing boundaries, sample IDs, and data provenance.
- **System under test:** owns the model/backend path and receives issued queries
  or batches.
- **Load generator:** creates deterministic traffic, runs scenarios, measures
  timing, and records enough settings to reproduce the run.
- **Accuracy or quality evaluator:** computes correctness, quality, pass/fail, or
  functional checks from captured outputs.
- **Reporter:** writes HTML, JSON, CSV, logs, grade output, and provenance
  manifests.

MLPerf EDU should implement this as a first-class Python harness before trying
to wrap the official MLPerf LoadGen directly. The educational harness should be
simple, inspectable, and portable on laptops, while leaving a future path to
official MLPerf LoadGen compatibility.

Methodology references to keep in view:

- MLPerf Inference LoadGen separates benchmark traffic generation from the
  system under test, which is the right architectural idea even if MLPerf EDU
  needs a smaller educational implementation.
- MLPerf-style runs separate performance measurement from accuracy or quality
  evaluation; MLPerf EDU should preserve that separation for training,
  inference, and test-time compute.
- MLPerf EDU should keep the benchmark harness canonical in this repository and
  treat external scripts as sources to wrap, not as the public execution
  contract.

Current implementation:

- Source lives under `src/mlperf/` with a compatibility CLI entry at
  `mlperf_edu.cli`.
- A deterministic harness/load-generation contract lives in
  `src/mlperf/harness.py`, with compatibility exports from
  `src/mlperf_edu/harness.py`.
- The current registry source of truth is the native `registry/suites/...`
  layout. `workloads.yaml` is a generated compatibility mirror checked by
  `tools/export_flat_registry.py --check`.
- The package build now includes the registry, `mlperf_edu` compatibility
  modules, and reference workload code, so the CLI works from an installed wheel
  outside the source tree.
- Canonical workload and variant names are exposed through the CLI so users can
  run selectors such as `smollm2-chat-inference --variant quantized-int8` and
  `nanogpt-inference --variant decode` directly.
- A first structured asset dossier catalog exists for the datasets and models
  fetched today. `mlperf fetch`, `mlperf cache`, JSON reports, CSV reports, and
  HTML reports now surface source, license/terms status, and public-result use
  without requiring users to inspect registry internals.
- The public audit now requires score-bearing and performance-bearing workloads
  with datasets to have a structured asset dossier, so free-text
  `dataset_source` alone is no longer sufficient for public-result paths.
- `mlperf audit --policy public` reports and fails on public-release caveats
  such as `restricted-needs-approval` or `needs-release-decision`, separating
  development pass/fail from endorsement readiness.
- `mlperf grade` and `mlperf validate` now stay focused on local verification
  and required quality checks; endorsement-only warnings stay in the explicit
  public policy audit path.
- Validation workload CSVs now preserve canonical workload selectors, dataset
  terms, and shared checkpoint dependencies for downstream spreadsheet analysis.
- `mlperf info --dataset <name>` now prints the structured dataset dossier plus
  matching workloads.
- `mlperf info --model <alias>` now resolves Hugging Face model aliases from the
  registry and prints model source/license metadata plus matching workloads.
- `mlperf audit` and `mlperf grade` now preserve canonical workload selectors
  and variants in machine-readable output, so public review artifacts can use
  `smollm2-chat-inference --variant ...` even while internal runner IDs remain
  flat.
- Discovery and reports expose canonical workload identity while retaining
  internal runner IDs as compatibility metadata. For example,
  `slm-decode` reports as `smollm2-chat-inference --variant baseline`, and
  `slm-quantized-decode` reports as
  `smollm2-chat-inference --variant quantized-int8`.
- `mlperf init` now prints local data/model/report paths plus copyable next
  commands for `fetch`, `run`, and `report`, so a first-time user is not left
  guessing what to do after setup.
- `mlperf fetch --dry-run` now explains shared checkpoint dependencies, such as
  NanoGPT prefill/decode inheriting quality from `nanogpt-train`, instead of
  presenting those inference phases as missing-dataset cases.
- Shared-checkpoint workloads now carry `shared_checkpoint` and
  `quality_dependency` through workload JSON, CSV, and HTML reports.

Roadmap target repo structure:

```text
mlperf-edu/
  src/mlperf_edu/
    harness/
      loadgen.py          # deterministic query, batch, server-lite, and TTC drivers
      scenarios.py        # offline, single-stream, server-lite, training, TTC
      qsl.py              # dataset/prompt sample libraries
      sut.py              # system-under-test interfaces
      metrics.py          # latency, throughput, quality, tokens, energy hooks
      report.py           # HTML/JSON/CSV/provenance emitters
      provenance.py       # source, model, dataset, checkpoint, and license manifests
    registry.py           # suite/profile/workload/variant resolution
    edu_cli.py            # mlperf init/fetch/audit/run/report/grade
  suites/
    <suite>/
      suite.yaml
      workloads/
        <canonical-workload>/
          workload.yaml   # model/task/dataset identity and shared assets
          runner.py       # canonical runner for baseline and variants
          assets.yaml     # base model, dataset, tokenizer, prompts, checkpoints
          variants/
            baseline.yaml
            quantized-int8.yaml
            batched.yaml
            long-context.yaml
          tests/
```

Variants inherit the canonical workload unless they explicitly override a field.
For example, `smollm2-chat-inference --variant quantized-int8` should normally
use the same task, prompt set, tokenizer, and base model family as
`smollm2-chat-inference --variant baseline`, while adding quantization settings,
calibration/provenance metadata, quality tolerance, and performance/energy
metrics. It should not become a new top-level workload folder unless the
benchmark identity changes.

Harness acceptance checks:

```bash
mlperf run --workload smollm2-chat-inference --profile min
mlperf run --workload smollm2-chat-inference --profile pro --variant quantized-int8
mlperf run --workload nanogpt-train --profile min
mlperf audit --workload smollm2-chat-inference
pytest tests/test_harness.py tests/test_registry.py tests/test_edu_cli.py -q
pytest tests/test_harness.py tests/test_registry.py tests/test_manifest.py tests/test_edu_cli.py -q
```

Best-practice rules for the harness:

- Keep sample selection deterministic and record random seeds.
- Separate untimed asset loading/preprocessing from timed execution.
- Use the same runner path for quality and performance modes.
- Record scenario settings: sample count, warmup, duration, target QPS if any,
  batch size, context length, output length, and TTC budget.
- Prevent accidental result caching unless a workload explicitly declares a cache
  experiment.
- Store every output under the run directory with stable names.
- Make the runner interface small enough that students can read and implement a
  new workload without learning official MLPerf internals first.

## Workload, Variant, And Phase Model

The user-facing benchmark name should be a recognizable workload, not an
internal kernel phase. Users should see names like `nanogpt-train`,
`resnet18-train`, `micro-dlrm-train`, `smollm2-chat-inference`,
`qwen-chat-inference`, or `smollm2-rag`.

Definitions:

| Concept | Meaning | User-facing examples |
|---|---|---|
| `suite` | Workload domain | `slm`, `vision`, `language`, `recommender`, `tiny`, `agent` |
| `workload` | Canonical benchmark identity | `smollm2-chat-inference`, `resnet18-train`, `nanogpt-train` |
| `variant` | Controlled optimization, backend, scenario, or research knob under a workload | `baseline`, `fp16`, `int8`, `batched`, `long-context`, `speculative`, `mlx`, `onnx` |
| `phase` | Measured subpart inside a run | prefill, decode, forward, backward, optimizer step, retrieval, tool call |
| `profile` | Scale and research surface | `min`, `max`, `pro` |

Naming rule:

- Prefer `mlperf run --workload smollm2-chat-inference --variant baseline`.
- Prefer `mlperf run --workload smollm2-chat-inference --variant quantized-int8`.
- Avoid making `slm-decode`, `nanogpt-prefill`, or `fp16-b16` the primary public
  workload identity unless the workload is explicitly a microbenchmark suite.
- Still record prefill/decode/batch/precision as metrics and variant metadata in
  the report.

This means internal IDs such as `slm-decode` and `slm-quantized-decode` should
remain implementation metadata, not public benchmark names. The public shape is
the canonical workload `smollm2-chat-inference` with variants such as
`baseline`, `batched-b4`, `quantized-int8`, and `long-context`.

## Workload Sourcing Policy

MLPerf EDU should prefer off-the-shelf implementations and assets. The job is
benchmark packaging, not reinventing every model. At the same time, the
canonical benchmark code path must live with MLPerf EDU: registry entries,
wrappers, fetch logic, runner interfaces, report schemas, provenance manifests,
and tests should be in this repository. External code can be wrapped or vendored
when appropriate, but it must be pinned, credited, and audited.

Preferred order:

1. Stable library API.
2. Official model or dataset package.
3. Well-maintained public repository with a clear license.
4. Paper artifact repository with a clear license.
5. Original MLPerf EDU implementation only when wrapping a public source is worse.

Open-source requirements:

- Default workloads must use open-source code paths.
- Public default datasets should be public-domain, permissively licensed, or
  otherwise clearly usable for academic benchmarking.
- Restricted or account-based assets may exist only as optional `pro` paths with
  explicit manifest requirements.
- Any vendored code must include license text, source URL, version/commit, and
  a summary of local modifications.
- Any wrapped external dependency must be pinned tightly enough for reproducible
  artifact evaluation.
- The canonical MLPerf EDU behavior is the wrapper and registry contract in this
  repo, not an undocumented upstream script.

Every workload should eventually include registry metadata like:

```yaml
source:
  implementation: external | wrapped | vendored | original
  repo: https://example.org/repo
  commit: abc123
  license: Apache-2.0
  citation: "@paper2026"
  model_source: HuggingFaceTB/SmolLM2-135M-Instruct
  dataset_source: https://...
  modifications:
    - Wrapped in MLPerf EDU runner interface.
    - Added deterministic profile-scale controls.
```

The implementation should include schema and audit checks even if some entries
are initially marked incomplete. The audit should make the gaps visible.

## Model And Dataset Selection Rationale

MLPerf EDU should not pick models only because they are convenient. Every model
and dataset should have an explicit rationale that a student, reviewer, or
researcher can understand.

For every model family we consider, create a model dossier before locking it
into the default suite:

| Field | Why it matters |
|---|---|
| Recognizable name | Researchers should understand what class of model is being benchmarked, like NanoGPT, ResNet-18, DLRM, Qwen, LLaMA, or SmolLM2 |
| Architecture family | Captures whether this is decoder-only transformer, CNN, recommender, encoder, diffusion, etc. |
| Parameter count | Determines laptop feasibility, memory pressure, and relevance to systems work |
| Weight size | Determines fetch time, cache size, CI feasibility, and classroom usability |
| Runtime footprint | Determines CPU/MPS/CUDA practicality and whether `max` can run locally |
| License/access | Determines whether the model can be part of the public default path |
| Source URL and revision | Makes the result reproducible and creditable |
| Tokenizer/preprocessor | Often dominates compatibility and reproducibility for language workloads |
| Backend support | PyTorch, Transformers, ONNX Runtime, MLX, llama.cpp, or other paths |
| Training provenance | Whether weights are trained inside MLPerf EDU, externally pretrained, externally fine-tuned, or synthetic/tiny |
| Research value | Which papers or systems questions this model helps study |
| Public-result suitability | Whether it can be score-bearing, performance-bearing, or systems-only |

The model size ladder should be deliberate:

| Size class | Purpose | Expected use |
|---|---|---|
| tiny synthetic/local | Proves install, runner, report, and provenance paths | `min`, CI, no-network smoke |
| small public model | Runs on laptop CPU/MPS with acceptable time | default `max` candidate |
| medium local model | Stresses memory, batching, quantization, and backend behavior | `pro` and research sweeps |
| restricted or large model | Only when license/access and hardware assumptions are explicit | optional `pro`, never default course path |

For SLM specifically, we should evaluate a model ladder rather than choosing by
intuition. Candidate families include SmolLM2-class, Qwen-class, and
LLaMA-family models, but the plan must verify current license, access,
download size, backend support, and expected laptop runtime before making any
one of them the default. The rationale should explain not only "what model,"
but "why this size, why this license, why this task, and why this is useful for
systems research."

## Benchmark Qualification Loop

Every benchmark should go through the same qualification loop before it is
treated as min-ready, artifact-evaluation-ready, or public-release-ready.
This applies to training, inference, optimization variants, and test-time compute
workloads.

The loop is:

1. **Define the canonical workload:** suite, workload ID, task, model family,
   dataset, runner, scenario, expected runtime envelope, and default variants.
2. **Confirm source provenance:** upstream code, local wrapper, license, citation,
   model source, dataset source, and any local modifications.
3. **Establish the reference run:** execute the baseline workload with the
   intended `min`, `max`, and `pro` scale controls, recording hardware/backend,
   seeds, metrics, quality, and runtime.
4. **Set the quality target:** choose a target from measured reference behavior,
   published expectations, or a documented expert decision. The target must be
   reachable on the reference path and strict enough to prevent trivial or broken
   implementations from passing.
5. **Validate robustness:** run enough repeated trials to understand variance,
   stochastic failures, dataset sensitivity, and runtime spread.
6. **Check classroom feasibility:** confirm `min` runs quickly and produces a
   comprehensible report on a laptop-class machine.
7. **Check research usefulness:** confirm `max` and `pro` expose meaningful
   systems dimensions such as batching, precision, quantization, sparsity,
   backend choice, memory behavior, energy, TTC budget, or distributed behavior.
8. **Review with experts:** ask domain reviewers whether the workload, dataset,
   quality metric, and target are credible for the intended use.
9. **Promote status:** move from experimental to systems-only,
   performance-bearing, or score-bearing only after audit, run, report, and
   expert-review evidence agree.
10. **Record the decision:** write the target rationale, reviewer feedback,
    accepted caveats, and known limitations into registry metadata and reports.

Quality target policy:

| Workload type | Primary quality evidence | Target-setting method |
|---|---|---|
| Training | final loss, accuracy, AUC, perplexity, reward, or task metric | Reference runs plus reachable threshold; avoid targets that require long tuning |
| Inference | functional equivalence, accuracy preservation, perplexity drift, pass/fail checks | Match the baseline model within an allowed tolerance |
| Optimization variant | quality delta relative to baseline plus performance/energy change | Require bounded degradation and record compression/speedup/energy claims |
| Test-time compute | answer quality, pass@k, exact match, verifier success, retrieval correctness, or task-specific score | Report quality per token, per sample, per tool call, and per latency budget |
| Systems-only | deterministic functional check plus performance metric | No public score until a credible quality target exists |

Target metadata required in each workload:

```yaml
quality_target:
  metric: accuracy | loss | perplexity | auc | pass_at_1 | exact_match | functional
  value: 0.0
  direction: higher | lower
  target_basis: reference_runs | literature | mlcommons_derived | pedagogical_baseline
  tolerance: 0.0
  reference_runs: 5
  variance_summary:
    runs: 5
    statistic: median
    acceptance_rule: median metric must satisfy target
    spread_note: short text explaining observed spread and caveats
  reviewer_notes: []
functional_check:
  metric: generated_tokens | decode_steps | forward_pass | task_success
  condition: deterministic pass/fail condition for performance-bearing work
  reference_runs: 5
  reviewer_notes: []
public:
  status: experimental | systems-only | performance-bearing | score-bearing
```

Expert-review questions:

- Is this the right dataset or task for the workload?
- Is the model size appropriate for `min`, `max`, and `pro`?
- Is the quality metric meaningful or only convenient?
- Is the target reachable without long manual tuning?
- Would a broken implementation fail?
- Does the report expose enough information to compare papers?
- Are any variants, such as quantization or pruning, allowed to reuse the same
  target with a tolerance, or do they need a separate target?

Promotion checklist:

- `mlperf info --workload <id>` explains the quality target and target basis.
- `mlperf audit --workload <id>` passes required metadata checks.
- `mlperf run --workload <id> --variant baseline` reaches the target on the
  reference path.
- Key variants either reach the same target within tolerance or clearly report
  degradation.
- The HTML report shows target, actual metric, target basis, tolerance, status,
  access restrictions, and known caveats.
- Expert feedback has been recorded and unresolved objections are visible.

## Training-To-Inference Provenance Chain

Whenever possible, MLPerf EDU should connect training and inference. The best
benchmark story is:

```text
public dataset -> training run -> checkpoint/provenance -> inference workload
```

For example, a NanoGPT training workload can produce a checkpoint, and prefill
or decode workloads can consume that checkpoint. That gives students and
researchers a complete path from training to serving.

The registry and reports should explicitly state weight origin:

| Weight origin | Meaning | Example |
|---|---|---|
| `mlperf_edu_trained` | Weights produced by an MLPerf EDU training workload | NanoGPT checkpoint consumed by NanoGPT decode |
| `external_pretrained` | Weights downloaded from a public model provider | SmolLM2/Qwen/LLaMA-family inference |
| `external_finetuned` | Weights come from a documented external fine-tuning run | public adapter or instruction-tuned checkpoint |
| `mlperf_edu_finetuned` | Base weights fetched externally, adapter/checkpoint produced by MLPerf EDU | LoRA fine-tuning workload |
| `synthetic_min` | Tiny/random/local weights used only to validate plumbing | `min` smoke path |

Inference workloads should prefer MLPerf EDU-trained checkpoints when that is
scientifically meaningful and feasible. When it is not feasible, such as
off-the-shelf SLM inference, the benchmark must record the external model
provider, revision, license, tokenizer, and intended use. This keeps the suite
honest without forcing us to pretrain every model.

This is also how training and inference suites should relate:

| Training workload | Inference or test-time workload | Provenance expectation |
|---|---|---|
| `nanogpt-train` | NanoGPT prefill/decode/speculative decode | consume MLPerf EDU-produced or explicitly fetched checkpoint |
| `qwen-lora-finetune` | Qwen chat/RAG/code generation | record base model plus MLPerf EDU adapter provenance |
| `resnet18-train` | vision inference/compression variants | consume trained checkpoint where feasible |
| `micro-dlrm-train` | recommender inference/memory-system variants | consume trained or deterministic baseline embeddings where feasible |

## Public Dataset Policy

Public datasets are part of the benchmark contract. We should choose datasets
that are accessible, stable, legally usable, and small enough for the relevant
profile.

Dataset selection rationale should include:

- Public source URL.
- License or terms-of-use summary.
- Citation.
- Expected download size.
- Cache location.
- Integrity hash or version.
- Whether it is used in `min`, `max`, or `pro`.
- Whether it supports a score-bearing quality target.
- Whether a micro-shard or synthetic stand-in exists, and exactly which
  profiles may use it.

Data policy by profile:

| Profile | Dataset expectation |
|---|---|
| `min` | synthetic, tiny public sample, or deterministic local fixture is allowed, but must be labeled |
| `max` | public real dataset or documented public micro-shard; no silent fallback |
| `pro` | public real dataset preferred; larger or restricted datasets allowed only with explicit manifest requirements |

The suite should prefer public datasets for default paths. Restricted, account-based,
or license-click datasets can exist in `pro`, but they should never be required
for the first successful run.

## Training, Inference, And Test-Time Compute

The registry should make workload type explicit:

| Type | Examples | Required evidence |
|---|---|---|
| Training | NanoGPT, DLRM, ResNet, MobileNet, BERT, LoRA | loss/accuracy, steps, throughput, checkpoint, data source |
| Inference | prefill, decode, classification, retrieval | latency, throughput, batch/context shape, model source |
| Test-time compute | RAG, reranking, self-consistency, speculative decode, agent loops, tool calls | quality per token, tokens per answer, latency per answer, tool/retrieval steps |
| Optimization | quantization, pruning, sparsity, backend swaps | baseline, variant metadata, compression ratio, speedup, quality delta |

Make the registry and report schema ready for these categories even if not every
workload has a mature implementation yet.

## Full Workload Matrix Checklist

This matrix is the working checklist for the current registry. It is not proof
that the workloads are correct. Each row must eventually be validated by
`init`, `fetch`, `audit`, `run`, `verify`, `grade`, report inspection, model
rationale review, dataset review, and source/license/citation review.

Current verified state:

- Installed-wheel smoke passed from `/tmp` on June 28, 2026: the installed
  package resolved `mlperf_edu/workloads.yaml`, imported packaged reference code,
  ran `mlperf doctor`, listed `--profile min`, and executed
  `nanogpt-train --profile min` with aggregate JSON/HTML/CSV plus workload
  JSON/HTML/CSV/provenance.
- `pytest -q` passed locally on June 28, 2026.
- `mlperf validate coverage --keep-going` passed locally on June 28, 2026:
  all 30 registered workloads ran at `min`, wrote JSON/HTML/CSV/provenance
  artifacts, and graded successfully.
- `mlperf validate max --keep-going` passed locally on June 28, 2026:
  all 30 registered workloads ran at `max`, wrote JSON/HTML/CSV/provenance
  artifacts, and graded successfully.
- `mlperf validate release --keep-going` passed locally on June 28, 2026:
  all 30 workloads ran at both `min` and `max`, wrote validation
  JSON/HTML/CSV plus workload CSVs, and graded 60/60 workload runs.
- Every aggregate and per-workload report now includes a
  `mlperf-edu-run-fingerprint/0.1` block with hardware, software, backend,
  profile, workload, data-mode, and seed context. Per-workload JSON reports are
  enriched before the paired `.provd.json` measurement hash is recomputed, so
  `mlperf verify` validates the exact JSON file users inspect.
- Per-workload reports now include a structured dataset/model asset dossier
  where the registry has a known source. HTML reports include an **Assets and
  Provenance** section, CSV reports include dataset/model terms columns, and
  `fetch --dry-run`/`cache list` show the same license-status language.
- This success means execution, imports, runner paths, manifests, grading, and
  report generation now work across the current registry. It does not mean every
  workload is scientifically final, publicly score-bearing, or ready for
  MLCommons endorsement; the remaining work is quality target hardening, source
  and dataset provenance, canonical workload/variant structure, expert review,
  and public-result policy.

Legend:

- `Y`: primary workload role.
- `Phase`: measured phase or downstream use, not necessarily top-level identity.
- `Opt`: optimization or systems variant.
- `TTC`: test-time compute or agentic inference.
- `Gap`: declared but needs audit, provenance, or rationale hardening.

| Workload | Suite | Recognizable model/task | Dataset | Training | Inference | TTC | Opt/systems | Weight provenance target | Public dataset target | Source/code provenance target | Current checklist status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `slm-decode` -> `smollm2-chat-inference` | `slm` | SmolLM2 local chat inference | prompt-suite-local |  | Y |  | Phase | external pretrained or tiny synthetic for `min` | public prompt set | Transformers wrapper, model revision, tokenizer, license | Run OK at min/max; canonical workload plus baseline variant is registry-owned; gap: model-size rationale and expert review |
| `slm-quantized-decode` -> `smollm2-chat-inference --variant quantized-int8` | `slm` | SmolLM2 quantized chat inference | prompt-suite-local |  | Y |  | Opt | external pretrained plus quantized variant metadata | public prompt set | Transformers quantization path, model revision, tokenizer, license | Run OK at min/max; quantization is a registry-owned variant; gap: quantization policy and expert review |
| `nanogpt-train` | `language` | NanoGPT training | tinyshakespeare | Y |  |  |  | MLPerf EDU trained checkpoint | public TinyShakespeare | canonical local training code, dataset source, citation | Run OK at min/max; shared checkpoint chain is reported in JSON/CSV/HTML; gap: public checkpoint provenance hardening |
| `nano-moe-train` | `language` | Nano-MoE training | tinyshakespeare | Y |  |  | Opt | MLPerf EDU trained checkpoint | public TinyShakespeare | canonical local MoE code plus sparse-routing rationale | Run OK at min/max; gap: quality target and public status audit |
| `micro-dlrm-train` | `recommender` | Micro-DLRM recommendation training | movielens-100k | Y | Phase |  |  | MLPerf EDU trained checkpoint or deterministic baseline | public MovieLens-100K | canonical DLRM wrapper/source, dataset license/citation | Run OK at min/max; gap: verify public dataset and checkpoint provenance |
| `micro-dlrm-dram-train` | `recommender` | DLRM memory-system variant | movielens-100k | Y | Phase |  | Opt | MLPerf EDU trained or deterministic systems baseline | public MovieLens-100K | canonical variant code and scaling rationale | Run OK at min/max; gap: explain systems-only status and memory-size rationale |
| `nano-lora-finetune` | `language` | NanoGPT LoRA fine-tuning | none declared | Y | Phase |  | Opt | external or MLPerf EDU base plus MLPerf EDU adapter | public or declared synthetic fine-tune data | LoRA implementation source and adapter provenance | Run OK at min/max; gap: declare dataset, source, and adapter provenance |
| `mobilenet-cifar100-composed-fp16` | `vision` | MobileNetV2 compression/FP16 variant | none declared |  | Y |  | Opt | derived from trained or fetched MobileNet checkpoint | CIFAR-100 or declared tensor shard | compression pipeline source and modification log | Run OK at min/max; gap: connect to trained checkpoint and dataset source |
| `micro-dlrm-distributed` | `distributed` | Local DDP DLRM | none declared | Y |  |  | Opt | MLPerf EDU trained or deterministic DDP baseline | synthetic or MovieLens-derived shard declared | torch.distributed wrapper and communication rationale | Run OK at min/max; gap: declare data source and distributed reproducibility metadata |
| `nanogpt-decode-fp32-b16` -> `nanogpt-inference --variant fp32-b16` | `language` | NanoGPT fp32 batch inference | prompt-suite-local |  | Y |  | Opt | consume NanoGPT checkpoint | bundled deterministic token prompt fixture | canonical inference runner and dtype metadata | Run OK at min/max; dtype/batch is a registry-owned variant; prompt fixture is declared; gap: checkpoint provenance |
| `nanogpt-decode-fp16-b16` -> `nanogpt-inference --variant fp16-b16` | `language` | NanoGPT fp16 batch inference | prompt-suite-local |  | Y |  | Opt | consume NanoGPT checkpoint | bundled deterministic token prompt fixture | canonical inference runner and dtype metadata | Run OK at min/max; dtype/batch is a registry-owned variant; prompt fixture is declared; gap: checkpoint provenance |
| `nanogpt-decode-spec` -> `nanogpt-inference --variant speculative` | `language` | NanoGPT speculative inference | prompt-suite-local |  | Y | Y | Opt | consume NanoGPT target/draft checkpoints | bundled deterministic token prompt fixture | speculative decode implementation and acceptance metrics | Run OK at min/max; speculative decoding is a registry-owned TTC variant; prompt fixture is declared; gap: draft/target provenance and acceptance-target review |
| `nanogpt-prefill` -> `nanogpt-inference --variant prefill` | `language` | NanoGPT prefill serving phase | prompt-suite-local |  | Y |  | Phase | consume NanoGPT checkpoint | bundled deterministic token prompt fixture | canonical inference runner and prompt source | Run OK at min/max; phase is a registry-owned variant; prompt fixture and checkpoint dependency are reported |
| `nanogpt-decode` -> `nanogpt-inference --variant decode` | `language` | NanoGPT decode serving phase | prompt-suite-local |  | Y |  | Phase | consume NanoGPT checkpoint | bundled deterministic token prompt fixture | canonical inference runner and prompt source | Run OK at min/max; phase is a registry-owned variant; prompt fixture and checkpoint dependency are reported |
| `micro-diffusion-train` | `vision` | Micro-diffusion U-Net training | cifar10 | Y | Phase |  |  | MLPerf EDU trained checkpoint | public CIFAR-10 | canonical U-Net or wrapped public implementation with citation | Run OK at min/max; gap: dataset/source/license rationale |
| `micro-gnn-train` | `graph` | Micro-GNN/Cora node classification | cora | Y | Phase |  |  | MLPerf EDU trained checkpoint | public Cora | canonical GNN code or cited public algorithm | Run OK at min/max; gap: dataset source/citation and public status audit |
| `micro-bert-train` | `language` | Micro-BERT fine-tuning | sst2 | Y | Phase |  |  | MLPerf EDU trained checkpoint | public SST-2/GLUE access path | canonical BERT-style code/source and tokenizer rationale | Run OK at min/max; gap: dataset access/license and model rationale |
| `micro-lstm-train` | `timeseries` | Micro-LSTM forecasting | etth1 | Y | Phase |  |  | MLPerf EDU trained checkpoint | public ETTh1 | canonical LSTM code and dataset citation | Run OK at min/max; gap: dataset source/citation and forecast quality rationale |
| `micro-rl-train` | `rl` | Micro-RL CartPole policy training | cartpole_local | Y | Phase | Y |  | MLPerf EDU trained policy | local/public environment spec | canonical environment and policy code | Run OK at min/max; gap: clarify environment provenance and stochastic-run policy |
| `resnet18-train` | `vision` | ResNet-18 Fashion-MNIST training | fashion-mnist | Y | Phase |  |  | MLPerf EDU trained checkpoint | MIT-licensed Fashion-MNIST | canonical local ResNet-18 code or source rationale | Run OK at min/max; gap: five-seed reference sweep and checkpoint handoff to inference/compression variants |
| `mobilenetv2-train` | `vision` | MobileNetV2 Fashion-MNIST training | fashion-mnist | Y | Phase |  |  | MLPerf EDU trained checkpoint | MIT-licensed Fashion-MNIST | canonical local MobileNetV2 code or source rationale | Run OK at min/max on MPS; gap: CPU fallback optimization and link to compression/edge inference variants |
| `dscnn-kws-train` | `tiny` | DS-CNN keyword spotting | speech_commands_v2 | Y | Phase |  | Opt | MLPerf EDU trained checkpoint | public Speech Commands v2 or declared micro-shard | DS-CNN source/citation and preprocessing provenance | Run OK at min/max; gap: public dataset fetch and synthetic micro-shard policy |
| `anomaly-ae-train` | `tiny` | MNIST anomaly autoencoder | mnist | Y | Phase |  |  | MLPerf EDU trained checkpoint | public MNIST | canonical AE code and thresholding rationale | Run OK at min/max; gap: threshold/provenance and public score audit |
| `wake-vision-vww` | `tiny` | Visual wake word/person detection | wake_vision | Y | Phase |  | Opt | MLPerf EDU trained checkpoint | public Wake Vision or declared proxy | MicroNet/VWW source, dataset license/citation | Run OK at min/max; gap: dataset access and proxy policy |
| `nano-rag-agent` | `agent` | NanoRAG retrieval-generation | react_traces |  | Y | Y |  | external/tiny language model or MLPerf EDU trained agent | public traces or declared local corpus | RAG source/citation, retrieval corpus provenance | Run OK at min/max; gap: clarify dataset, model source, and TTC budget |
| `nano-codegen-agent` | `agent` | NanoCodeGen MBPP code generation | mbpp |  | Y | Y |  | external/tiny language model or MLPerf EDU trained agent | public MBPP subset | codegen source/citation and test harness provenance | Run OK at min/max; gap: public subset definition and pass@1 confidence |
| `nano-react-agent` | `agent` | NanoReAct tool/reasoning traces | react_traces |  | Y | Y |  | external/tiny language model or MLPerf EDU trained agent | public or generated ReAct traces | ReAct source/citation and trace provenance | Run OK at min/max; gap: trace source and reasoning-vs-imitation caveat |
| `nano-toolcall-agent` | `agent` | Nano tool-calling | react_traces |  | Y | Y |  | external/tiny language model or MLPerf EDU trained agent | public or generated tool traces | tool schema, trace source, dispatcher provenance | Run OK at min/max; gap: quality target and trace provenance |

## SLM Expansion Direction

SLM is the highest-priority research suite gap. Current SLM coverage is too
thin for the north star.

The top-level workload names should describe the benchmark task and recognizable
model family that a user understands, not an internal transformer phase. This
should feel like `nanogpt-train`, `resnet18-train`, or `micro-dlrm-train`: the
name tells the user what benchmark they are running.

For SLM, users should see names and examples built around model families such
as LLaMA, Qwen, and SmolLM2-135M. Metrics inside that workload can then report
prefill, decode, TTFT, inter-token latency, KV-cache size, batching, and memory
pressure.

In other words:

- `qwen-chat-inference` or `smollm2-chat-inference` is a workload name users recognize.
- `llama-chat-inference` can be a `pro` workload when license/access rules allow.
- `qwen-rag` or `smollm2-rag` is a workload.
- `qwen-lora-finetune` or `smollm2-lora-finetune` is a workload.
- Prefill and decode are measured phases inside inference workloads.

Target SLM workloads and measured phases:

| Workload | Profile role | Measured phases / research dimensions |
|---|---|---|
| `smollm2-chat-inference` | `min`, `max`, `pro` | TTFT/prefill, decode throughput, inter-token latency, prompt length, output length |
| `qwen-chat-inference` | `max`, `pro` | same chat-serving metrics on a Qwen-family model |
| `llama-chat-inference` | `pro` | LLaMA-family local serving where license/access rules allow |
| `smollm2-chat-inference --variant batched-b4` | `pro` | batch scaling, throughput/latency tradeoff, memory footprint |
| `smollm2-chat-inference --variant long-context` | `pro` | context length scaling, KV-cache memory, prefill cost |
| `qwen-long-context` | `pro` | context length scaling, KV-cache memory, prefill cost |
| `smollm2-quantized-inference` | `max`, `pro` | int8, int4, weight-only quantization, quality/performance delta |
| `qwen-lora-finetune` | `max`, `pro` | adapter fine-tuning, rank/alpha sweeps, trainable parameter fraction |
| `smollm2-rag` | `max`, `pro` | retrieval depth, generation cost, quality per token, latency per answer |
| `qwen-code-generation` | `max`, `pro` | code prompt throughput, pass/fail or syntax validity, test-time samples |
| `smollm2-speculative-generation` | `pro` | draft/target model tradeoffs, accepted tokens, speedup, quality parity |
| `slm-backend-compare` | `pro` | PyTorch, ONNX Runtime, MLX, llama.cpp, future compiler paths |

Candidate model ladder:

| Alias class | Examples | Role |
|---|---|---|
| tiny local | deterministic tiny/random model | `min` profile and CI |
| small permissive/off-the-shelf SLM | SmolLM2-135M, Qwen 0.5B/0.6B class models | default local `max` where license and download path are acceptable |
| LLaMA-family model | LLaMA-style local model where license and access allow | `pro` and research comparisons |

The exact model should remain visible through both the workload identity and the
`--model` choice. The workload should name the recognizable benchmark family;
`--model` should allow controlled substitutions inside that family.

The next implementation milestone should produce:

- Registry slots or roadmap entries for the SLM expansion.
- Source/provenance policy for off-the-shelf SLM models.
- A concrete first implementation slice for `smollm2-chat-inference`, with baseline, quantized, and batched variants plus prefill/decode/batching metrics recorded in reports.

## Completion Evidence

The job is done only when the implementation produces evidence that a new user
can run MLPerf EDU through the intended workflow and understand the output.

Required evidence:

- `mlperf init --profile min` completes from a fresh checkout.
- `mlperf fetch --dry-run --profile min` and `mlperf fetch --dry-run --profile max`
  explain assets, licenses, model rationale, dataset rationale, and cache paths.
- `mlperf audit --format json` reports local blockers with stable machine
  output; `mlperf audit --policy public --format json` additionally reports
  endorsement/release warnings.
- `mlperf run --profile min` produces HTML, JSON, CSV, logs, and provenance
  manifests automatically.
- `mlperf run --workload <id>` produces the same artifact family automatically,
  even when only one workload is executed.
- `mlperf run --workload <canonical-workload> --variant <variant>` works for at
  least one baseline workload and one optimized variant.
- The default HTML report answers the user-journey questions in this plan.
- Tests cover registry metadata, CLI workflow, harness behavior, report output,
  and audit rules.
- The workload matrix marks remaining gaps honestly rather than implying that
  declared workloads are already release-ready.

Verification commands:

```bash
PYTHONPATH=src python3 -m mlperf_edu.cli doctor
PYTHONPATH=src python3 -m mlperf_edu.cli init --profile min
PYTHONPATH=src python3 -m mlperf_edu.cli fetch --dry-run --profile min
PYTHONPATH=src python3 -m mlperf_edu.cli fetch --dry-run --profile max
PYTHONPATH=src python3 -m mlperf_edu.cli audit --format json
PYTHONPATH=src python3 -m mlperf_edu.cli run --profile min --output-dir /tmp/mlperf-edu-min
PYTHONPATH=src python3 -m mlperf_edu.cli report /tmp/mlperf-edu-min/mlperf_edu_min_<timestamp>.json --format html
PYTHONPATH=src python3 -m mlperf_edu.cli grade /tmp/mlperf-edu-min --output /tmp/mlperf-edu-min/grade.json
PYTHONPATH=src python3 -m mlperf_edu.cli validate coverage --output-dir /tmp/mlperf-edu-coverage --keep-going
PYTHONPATH=src python3 -m mlperf_edu.cli validate max --output-dir /tmp/mlperf-edu-max --keep-going
PYTHONPATH=src python3 -m mlperf_edu.cli validate release --output-dir /tmp/mlperf-edu-release --keep-going
python3 -m build --sdist --wheel
python3 -m pip install --no-deps --target /tmp/mlperf-edu-wheel-target dist/mlperf_edu-0.1.0-py3-none-any.whl
PYTHONPATH=/tmp/mlperf-edu-wheel-target python3 -m mlperf_edu.cli doctor
PYTHONPATH=/tmp/mlperf-edu-wheel-target python3 -m mlperf_edu.cli run --workload nanogpt-train --profile min --output-dir /tmp/mlperf-edu-wheel-run
pytest tests/test_harness.py tests/test_registry.py tests/test_manifest.py tests/test_edu_cli.py -q
pytest -q
```

Evidence ledger from June 28, 2026:

Historical rows are retained for traceability. The live status board at the top
of this file supersedes older workload counts and runtimes.

| Check | Result | Notes |
|---|---|---|
| Installed wheel from `/tmp` | Passed | Registry, reference packages, `doctor`, `list --profile min`, and `nanogpt-train --profile min` work outside the source tree |
| Installed wheel after report/audit updates | Passed | Built sdist/wheel, installed to `/tmp`, ran `doctor`, `nanogpt-train --profile min`, `verify`, and `report <run-directory> --format html` outside the source tree |
| `pytest -q` | Passed | Includes CLI, registry, harness, manifest, power, and workload path coverage |
| `validate coverage` | Passed | 30/30 `min` workload runs passed and graded after report fingerprinting |
| `validate max` | Passed | 30/30 registered `max` workload runs passed and graded; latest full validation duration recorded as 93.4 seconds |
| `validate release` | Passed | 60/60 combined `min`+`max` workload runs passed and graded |
| Report artifacts | Passed | Aggregate and workload JSON/HTML/CSV are emitted by default; `.provd.json` manifests are emitted for workload evidence |
| Run fingerprints | Passed | Coverage workload reports plus aggregate reports include `run_fingerprint`; provenance verification still passes |
| Warning hygiene | Passed | `validate coverage` and `validate max` pass with empty stderr after filtering known third-party DDP, CIFAR, HF loading, and qnnpack noise |
| Asset dossiers | Passed for first slice | `fetch --dry-run`, `cache list`, workload JSON, CSV, and HTML expose dataset/model source and license-status fields for the currently fetchable TinyShakespeare, MovieLens-100K, MNIST, CIFAR-100, prompt-suite, and Hugging Face model assets |
| Asset-dossier smoke | Passed | `nanogpt-train --profile min` generated JSON/HTML/CSV with `dataset_asset`, CSV terms fields, and the HTML **Assets and Provenance** section; `mlperf verify` still passed |
| Public audit asset rule | Passed | `mlperf audit --format json` passes with zero issues after requiring structured asset dossiers for public result datasets |
| Public policy audit warnings | Passed | `mlperf audit --policy public --status score-bearing --format json` reports unresolved dataset license/release-policy warnings while default local audit remains clean |
| Canonical SLM discovery | Passed | `mlperf list --suite slm`, JSON list output, `info`, workload reports, CSV, and HTML expose `smollm2-chat-inference` plus baseline/quantized variants while preserving current runner IDs |
| Directory report UX | Passed | `mlperf report <run-directory>` selects the latest aggregate report and can regenerate HTML without requiring users to copy timestamped JSON filenames |
| Init UX | Passed | `mlperf init --profile min --no-smoke` prints cache/report paths and copyable next commands; `init --profile min` still runs the smoke path and writes JSON/HTML/CSV/provenance |
| Asset sizes and hash policy | Passed for first slice | Fetchable asset dossiers include stable expected byte counts where known plus a hash-policy field; run manifests remain the source of truth for computed local hashes |
| Shared checkpoint fetch UX | Passed | `mlperf fetch --profile min --dry-run` explains `nanogpt-prefill` and `nanogpt-decode` as shared-checkpoint workloads with `nanogpt-train` quality dependency |
| Shared checkpoint report chain | Passed | NanoGPT prefill/decode reports include `shared_checkpoint` and `quality_dependency` in JSON/CSV/HTML while provenance verification still passes |
| NanoGPT inference prompt fixture | Passed for current slice | NanoGPT inference variants now declare the bundled `prompt-suite-local` deterministic token prompt fixture; fetch output, JSON reports, CSV reports, HTML reports, and validation workload CSVs expose the prompt fixture and checkpoint dependency |
| Post-prompt-fixture smoke validation | Passed | `mlperf validate smoke --keep-going --skip-doctor` passed 12/12; validation workload CSV shows `prompt-suite-local` plus `nanogpt-train` dependency for NanoGPT prefill/decode |
| Post-prompt-fixture max validation | Historical pass | Superseded by the current packaged 30/30 max validation with 0 local warnings; validation workload CSV still shows `prompt-suite-local` for NanoGPT inference variants |
| Cache profile selection | Passed | Bare `mlperf cache list` now defaults to the `min` profile, prints the selected workload count plus `--profile max` hint, records selection metadata in JSON, and `--profile max` inspects all 30 workload asset rows |
| Post-cache/profile wheel smoke | Passed | Rebuilt wheel, installed into a throwaway venv, confirmed packaged `mlperf list profiles` reports `min=12`, `max=30`, `pro=12`, packaged `cache list --format json` reports profile `min`, and packaged `validate smoke --dry-run` emits `min-default` |
| Local grade hygiene | Passed | `mlperf grade` stays focused on verification and quality, while public-release warnings are handled by `mlperf audit --policy public` |
| Post-warning `validate coverage` | Passed | 28/28 `min` workloads passed; grade summary reported 5 non-blocking public-release warnings |
| Validation hygiene | Passed | `mlperf validate smoke` keeps local execution/quality status in JSON, CSV, and HTML while endorsement-only warnings stay in `mlperf audit --policy public` |
| Validation workload CSV provenance | Passed | Per-workload validation CSV rows include canonical selector, dataset terms, and shared checkpoint dependency fields |
| Docs alignment | Passed for current slice | README, SPEC, and PUBLIC_RULES document report-directory UX, canonical SLM selectors, asset/provenance reports, and blocker-vs-warning policy |
| Dataset info UX | Passed | `mlperf info --dataset tinyshakespeare` prints source URL, citation, license status, expected size, hash policy, and matching workloads |
| Dataset release policy | Passed for current slice | Dataset dossiers expose `public_release_status`, `public_release_policy`, and `release_next_step`; default local audit/validation are clean, while `mlperf audit --policy public` reports unresolved public-release warnings |
| Reference-run protocol | Passed for current slice | Score-bearing workloads now declare `quality_target.reference_protocol`; registry contract tests enforce required protocol fields; workload JSON/CSV/HTML and validation workload CSV expose reference runs, statistic, and protocol summary |
| Training-to-inference checkpoint lineage | Passed for current slice | Checkpoint-backed NanoGPT inference reports now include structured `checkpoint_provenance`; CSV/HTML show checkpoint source, inherited source quality, and artifact policy |
| Workload review packets | Passed for current slice | `tools/generate_review_packets.py` generates nine public-result Markdown packets under `review_packets/` with summary, commands, quality/functional contract, assets, checkpoint lineage, warnings, and source provenance |
| Model info UX | Passed | `mlperf info --model qwen3-0.6b` resolves to `Qwen/Qwen3-0.6B`, shows source/license metadata, explains selected/default/size/backend rationale, and lists matching SLM workloads |
| SLM model rationale artifacts | Passed for current slice | SLM workload reports carry `selected_model_rationale`, `selection_rationale`, `size_rationale`, and `backend_rationale` in `model_asset`; HTML/CSV report views and validation workload CSVs include `model_rationale`, so reports remain defensible without reading `workloads.yaml` |
| SLM rationale validation | Passed | `mlperf validate smoke --suite slm --keep-going --skip-doctor` passed 4/4; validation workload CSV includes canonical selectors and model rationale for SLM variants |
| Post-model-rationale tests | Passed | Focused model-info/SLM report tests and full `pytest -q` passed after adding structured SLM model rationale metadata and report/CSV rationale fields |
| Canonical audit/grade metadata | Passed | `audit --suite slm --format json` and graded SLM variant artifacts include `canonical_workload`, `variant`, and `run_selector` |
| Post-canonical `pytest -q` | Passed | Full test suite passed after canonical audit/grade metadata changes |
| Post-canonical `validate coverage` | Passed | 28/28 `min` workloads passed; validation workload CSV preserves `smollm2-chat-inference --variant baseline` selector |
| Run selection preview | Passed | `mlperf run --profile min`, `slm-decode`, and `smollm2-chat-inference --variant quantized-int8` print selected workload counts plus `run as` selectors before execution |
| Post-selection `pytest -q` | Passed | Full test suite passed after adding explicit run-selection preview output |
| Quality-required naming | Passed for current slice | Reports, CSV, HTML, grade JSON, validation CSV, and validation HTML expose `quality_required`; public docs avoid the old internal wording |
| Runner quality-key migration | Passed for current slice | Runner-generated reports now emit `quality_required`; the CLI keeps one centralized compatibility reader for older artifacts |
| Post-quality-key smoke validation | Historical pass | Superseded by the current packaged 12/12 smoke validation with 0 local warnings; aggregate and workload JSON contain `quality_required` and no legacy quality field |
| Post-quality-key max validation | Historical pass | Superseded by the current packaged 30/30 max validation with 0 local warnings; aggregate and workload JSON contain `quality_required` and no legacy quality field |
| Profile/fetch discovery | Passed | `mlperf info --profile min` and `mlperf fetch --dry-run --profile min` show the selected workload set and `run as` selectors before listing assets |
| Profile discovery counts | Passed | `mlperf list profiles` shows registry-derived workload counts for `min` (12), `max` (30), and `pro` (12); JSON output carries the same counts |
| Validation preset wording | Passed | CLI help, README, and SPEC now say validation `presets` rather than validation `suites`, keeping `suite` reserved for workload domains such as `slm`, `vision`, and `language` |
| Post-discovery CLI tests | Passed | `pytest tests/test_edu_cli.py -q` passed after quality-required naming and profile/fetch discovery changes |
| Workload matrix CLI | Passed | `mlperf list matrix` exposes workload roles, canonical selectors, default profile membership, datasets, public status, and quality summaries; JSON tests cover SLM quantization and speculative decode |
| Workload-filtered matrix UX | Passed | `mlperf list matrix --workload nanogpt-inference` now filters to the canonical workload variants, and `--variant prefill` narrows to one row instead of silently returning the full matrix |
| Workload-filtered list UX | Passed | `mlperf list --workload nanogpt-inference` filters to the canonical NanoGPT inference variants, `--variant prefill` narrows to `nanogpt-prefill`, and explicit suite/workload selections are not silently narrowed by `--profile` |
| Post-workload-list `pytest -q` | Passed | Full test suite passed after sharing selection semantics across `mlperf list` and `mlperf list matrix` |
| Doctor selector and JSON UX | Passed | `mlperf doctor` now accepts `--profile`, `--suite`, `--workload`, `--variant`, and `--format json`; JSON includes cache paths, registry status, selected workloads, and clean selection failures |
| Post-doctor `pytest -q` | Passed | Full test suite passed after adding doctor selectors and JSON output, including validation smoke preflight coverage |
| Audit JSON failure reasons | Passed | `mlperf audit --policy public --format json` now exposes top-level `issues`, `warnings`, `blocker_count`, and `warning_blocked` fields so MLCommons/public-release review can see why a policy audit failed without scanning every workload row |
| Post-audit-json `pytest -q` | Passed | Full test suite passed after adding top-level audit warning/blocker fields |
| Canonical workload selection semantics | Passed | Canonical `--workload <id>` now means the workload family across `doctor`, `list`, `fetch`, `cache`, `audit`, `init`, and `run`; `--variant` narrows to exactly one variant |
| AGY/Gemini selection review | Passed | External review flagged inconsistent canonical workload/profile semantics; the CLI now uses a single rule: bare `--profile` selects the default profile set, while explicit `--suite` or `--workload` controls workload selection and `profile` records run/audit context |
| Post-canonical-semantics `pytest -q` | Passed | Full test suite passed after canonical workload family execution and profile/audit semantics changes |
| Post-canonical-semantics wheel smoke | Passed | Rebuilt the wheel, installed it in a throwaway venv, and from `/tmp` verified packaged canonical family selection for `doctor`, `list`, `audit --profile min`, and `run --workload nanogpt-inference --profile min` |
| Run dry-run UX | Passed | `mlperf run --dry-run` previews selected workloads and canonical run selectors without running models or writing artifacts |
| Post-run-dry-run `pytest -q` | Passed | Full test suite passed after adding `run --dry-run` and validation-internal namespace compatibility |
| Post-run-dry-run wheel smoke | Passed | Rebuilt the wheel, installed it in a throwaway venv, and confirmed packaged `mlperf run --workload nanogpt-inference --profile min --dry-run` previews five variants and writes no artifacts |
| Post-matrix `pytest -q` | Passed | Full test suite passed after workload matrix, quality-required naming, and profile/fetch discovery changes |
| Post-prompt-fixture matrix tests | Passed | Focused NanoGPT prompt-fixture/matrix tests and full `pytest -q` passed after declaring `prompt-suite-local` for NanoGPT inference variants |
| Post-matrix `validate coverage` | Passed | 28/28 `min` workloads passed and graded; validation workload CSV includes `quality_required` and preserves `smollm2-chat-inference --variant baseline` |
| Validation naming cleanup | Passed | `mlperf validate max --dry-run` now reports `max-all` / all workloads, matching the full-suite max profile contract |
| Agent review follow-up: portable provenance | Passed for current slice | New `.provd.json` manifests use portable `sha256-merkle-root-v1` signatures and source-tree verification compares tree/patch hashes when a git checkout is available |
| Agent review follow-up: public quality naming | Passed for current slice | New report JSON/CSV/HTML and grade JSON use `quality_required`; the grader still understands older artifact metadata as fallback input |
| Agent review follow-up: validation report directories | Passed | `mlperf report <validation-output-dir>` resolves latest `mlperf_validate_*.json` and summarizes workload counts |
| Agent review follow-up: endorsement audit policy | Passed | `mlperf audit --policy public` fails on unresolved public-release warnings while default development audit stays clean for local users |
| Agent review follow-up: SLM pro matrix | Passed | `mlperf list matrix --suite slm --profile pro` shows baseline and quantized SLM variants as `max, pro` capable |
| Registry-owned canonical variants | Passed | `smollm2-chat-inference` and `nanogpt-inference` aliases, variants, and default variants now live in `workloads.yaml`; CLI list/run/audit/report paths derive selectors from registry metadata |
| Post-registry metadata `pytest -q` | Passed | Full test suite passed after moving canonical aliases, variants, and default variants into `workloads.yaml` |
| SLM batched pro variant | Passed | Added `smollm2-chat-inference --variant batched-b4`; direct `min` run produced JSON/HTML/CSV/provenance with batch size, requests/sec, total generated tokens, and portable verification |
| SLM long-context pro variant | Passed | Added `smollm2-chat-inference --variant long-context`; direct `min` run produced JSON/HTML/CSV/provenance with configured context tokens, measured context tokens, prompt size, prefill latency, and portable verification |
| Post-SLM-long-context `pytest -q` | Passed | Full test suite passed after adding `slm-long-context-decode`, registry-owned SLM variant metadata, and default `pro` profile selection |
| Post-SLM-long-context `validate coverage` | Passed | 30/30 `min` workloads passed and graded; coverage duration recorded as 26.5 seconds on the local Apple Silicon machine |
| Post-SLM-long-context `validate release` | Passed | 60/60 combined `min`+`max` workload runs passed and graded; release duration recorded as 278.3 seconds on the local Apple Silicon machine |
| Post-SLM-long-context wheel smoke | Passed | Rebuilt sdist/wheel, installed in a throwaway venv, and ran packaged `mlperf doctor`, `list variants --workload smollm2-chat-inference`, `smollm2-chat-inference --variant long-context --profile min`, and `verify` from `/tmp` |
| SLM serving metrics | Passed | SLM reports now include `time_to_first_token_s`, `inter_token_latency_s`, `prefill_tokens_per_sec`, and `total_context_tokens`; focused SLM tests, direct long-context run, provenance verification, and `pytest -q` passed |
| Post-serving-metrics wheel smoke | Passed | Rebuilt sdist/wheel, installed in a throwaway venv, ran packaged `smollm2-chat-inference --variant long-context --profile min`, confirmed serving metrics in JSON, and verified provenance from `/tmp` |
| SLM serving report UX | Passed | Workload and aggregate HTML reports now include a dedicated **Serving Metrics** table for generation-style metrics; `mlperf report` terminal summaries print the key SLM serving metrics; focused report/SLM tests and full `pytest -q` passed |
| Post-serving-report wheel smoke | Passed | Rebuilt sdist/wheel, installed in a throwaway venv, ran packaged `smollm2-chat-inference --variant long-context --profile min`, regenerated HTML with `mlperf report`, confirmed the serving metrics table, and verified provenance from `/tmp` |
| Matrix quality completeness | Passed | Every workload in `mlperf list matrix --format json` now has either a quality target or functional-check summary; metadata-only fixes covered LoRA, DDP, NanoGPT dtype/speculative decode, MobileNet compression, and tool-call agent workloads; focused registry/audit tests and full `pytest -q` passed |
| Full-suite max profile | Passed | `mlperf run/fetch/list --profile max` and `mlperf validate max` now select all 30 workloads; latest packaged release validation passed the max half 30/30 manifests with 0 local warnings in 91.1 seconds |
| Functional-check grade metrics | Passed | `mlperf grade` now displays functional-check metrics for systems-only workloads when no score-bearing quality metric exists; focused grade tests and full `pytest -q` passed |
| Post-max-contract wheel smoke | Passed | Rebuilt sdist/wheel, installed in a throwaway venv, confirmed packaged `mlperf list --profile max` reports 30 workloads, `validate max --dry-run` emits `max-all`, and packaged grading reports `effective_compression_ratio` for MobileNet compression |
| Cross-suite min starter | Passed | Default `min` covers 12 workloads across agent, distributed, graph, language, recommender, RL, SLM, timeseries, tiny, and vision; packaged smoke passed 12/12 in 11.9 seconds, and packaged release passed all 30 min manifests in 24.8 seconds |
| Post-starter-contract wheel smoke | Passed | Rebuilt sdist/wheel, installed in a throwaway venv, confirmed packaged `mlperf list --profile min` reports 12 workloads, packaged `mlperf list --profile max` reports 30 workloads, and `validate smoke --dry-run` resolves `min-default` |
| Post-SLM-batched `pytest -q` | Passed | Full test suite passed after adding `slm-batched-decode`, registry-owned SLM variant metadata, and default `pro` profile selection |
| Post-SLM-batched `validate coverage` | Passed | 29/29 `min` workloads passed and graded; coverage duration recorded as 24.6 seconds on the local Apple Silicon machine |
| Post-SLM-batched `validate release` | Passed | 58/58 combined `min`+`max` workload runs passed and graded; release duration recorded as 268.0 seconds on the local Apple Silicon machine |
| Post-SLM-batched wheel smoke | Passed | Rebuilt sdist/wheel, installed in throwaway venvs, ran packaged `mlperf doctor`, `list variants --workload smollm2-chat-inference`, `smollm2-chat-inference --variant batched-b4 --profile min`, `list --profile pro`, and `verify` from `/tmp` |
| Post-agent `pytest -q` | Passed | Full test suite passed after portable provenance, public quality naming, validation-directory report resolution, endorsement audit policy, and SLM pro matrix fixes |
| Post-agent `validate coverage` | Passed | 28/28 `min` workloads passed and graded with stricter source/signature verification and public `quality_required` report fields |
| Post-agent `validate max` | Historical pass | Earlier max-subset validation passed and graded with stricter source/signature verification and public `quality_required` report fields before the profile was expanded to all workloads |
| Post-agent wheel smoke | Passed | Rebuilt sdist/wheel, installed into a throwaway venv, ran packaged `mlperf doctor`, `list matrix --suite slm --profile pro`, `nanogpt-train --profile min`, and `verify` outside the source tree |

Status reporting should include:

- Commands that now work.
- Commands that still fail.
- Workload download gaps.
- Missing source/license/citation metadata.
- Missing model-size rationale and dataset rationale.
- Training-to-inference provenance gaps.
- Report/provenance improvements.
- Profile semantic gaps.
- Next workload implementation slice.

## Continuous Agentic Review Loop

For each implementation slice:

1. Codex implements the slice.
2. Run local tests and command checks.
3. Ask one or more independent review agents for focused critique when useful.
4. Convert useful feedback into concrete changes.
5. Re-run tests.
6. Record evidence in the implementation status.
7. Move to the next workload, variant, or user-journey gap.

Do not delegate source-of-truth decisions to agents. Agents provide critique;
the implementation remains grounded in the repo, commands, tests, and user
north star.

For major planning or architecture changes, run a 10-round convergence loop
before moving on:

1. Write the current plan or implementation summary.
2. Ask independent review agents for critique from different perspectives:
   student, instructor, architecture researcher, artifact evaluator, MLCommons
   reviewer, and maintainer.
3. Extract concrete objections, missing requirements, ambiguous names, UX
   friction, and implementation risks.
4. Patch the plan or code.
5. Re-run relevant local checks.
6. Repeat until at least 10 review/improvement rounds have been completed or the
   remaining feedback is only duplicate or explicitly deferred.
7. Record convergence evidence: what changed, what feedback was rejected, what
   remains open, and which checks passed.

The loop is not a substitute for implementation. It runs alongside
implementation, one benchmark or user journey at a time:

```text
plan -> implement -> run -> report -> review -> patch -> rerun -> repeat
```

Latest review convergence summary:

- A 10-role review loop converged on the same main risks: public comparability
  policy, dataset/model license disclosure, externally reproducible reference
  runs, canonical SLM workload names, warning hygiene, and stronger
  machine-readable provenance.
- Implemented responses so far:
  1. run-fingerprint/report-provenance patch: reports expose
     hardware/software/backend/data-mode context directly, and the workload
     provenance manifest is recomputed after report enrichment;
  2. first asset-dossier patch: `fetch`, `cache`, JSON, CSV, and HTML expose
     structured source/license-status/public-use metadata for fetchable assets;
  3. first public-audit asset rule: score-bearing and performance-bearing
     workloads with datasets must now have a structured asset dossier;
  4. public-audit warning patch: release-policy caveats are warnings, not hidden
     in prose and not confused with implementation blockers;
  5. first canonical SLM UX patch: list/info/report surfaces show
     `smollm2-chat-inference` and variant selectors even while the underlying
     registry IDs remain `slm-decode` and `slm-quantized-decode`;
  6. report-directory UX patch: `mlperf report <run-directory>` auto-selects
     the latest aggregate report and can regenerate HTML/CSV/JSON from the
     directory path;
  7. init UX patch: `mlperf init` prints local paths and exact next commands for
     fetch, run, and report;
  8. expected-size dossier patch: TinyShakespeare, MovieLens-100K, MNIST,
     CIFAR-100, and the bundled prompt suite carry expected byte counts where
     stable plus explicit hash-policy wording;
  9. shared-checkpoint fetch patch: inference phases that consume a trained
     checkpoint explain the source training workload and quality dependency;
  10. shared-checkpoint report patch: JSON, CSV, and HTML reports show the
      checkpoint source and inherited quality dependency for prefill/decode;
  11. local/public warning split: local grade and validation stay focused on
      verification and required quality checks, while public-release warnings
      are routed through `mlperf audit --policy public`;
  12. validation hygiene patch: validation summaries preserve execution and
      quality status without surfacing endorsement-only policy warnings;
  13. validation-workload CSV patch: per-workload CSV exports retain canonical
      selectors, dataset terms, and shared checkpoint dependencies;
  14. docs-alignment patch: README, SPEC, and PUBLIC_RULES reflect the current
      CLI/report/audit behavior for this implementation slice;
  15. dataset-info patch: `mlperf info --dataset` exposes asset dossiers without
      requiring users to inspect JSON reports or registry internals;
  16. model-info patch: `mlperf info --model` resolves registry aliases and
      shows source/license metadata for off-the-shelf SLMs;
  17. canonical audit/grade patch: public review JSON carries canonical
      workload selectors and variants in addition to internal runner IDs;
  18. dataset-release-status patch: dataset dossiers now classify
      `public-ok-bundled`, `public-ok-with-attribution`,
      `restricted-needs-approval`, and `needs-release-decision`, and reports
      carry release status plus next-step fields.
- Remaining high-priority follow-up slices are:
  1. get MLCommons/dataset-owner review on the remaining `restricted` row,
     MovieLens-100K, or replace it with a clearly open recommender dataset;
  2. complete the native registry migration to canonical workload folders with
     `baseline` and `quantized-int8` variants, so phase-like SLM IDs no longer
     appear as primary public IDs;
  3. a published reference-run/comparability policy for at least one score-bearing
     workload before any public-result claim;
  4. keep warning routing strict: first-run terminal output should stay quiet
     unless a benchmark-controlled warning is actionable.

## Active Open Checklist

This is the visible working list for the next implementation passes. Keep it
current as items move from open to done.

## Live Worklog Checklist

I will keep this section current as implementation proceeds. Each row is a
single work slice with an observable artifact or test result.

| Status | Slice | Observable checkpoint |
|---|---|---|
| Done | Selection vocabulary cleanup | Public CLI/docs use suite/profile/workload/variant and profiles `min`, `max`, `pro`; no public `gate`, `set`, `std`, `classroom`, or `research` profile vocabulary |
| Done | Canonical CLI selection | `--workload smollm2-chat-inference` selects all variants; `--variant quantized-int8` selects one row; `run --dry-run` previews without artifacts |
| Done | Default report artifacts | `mlperf run` emits JSON, HTML, CSV, and `.provd.json`; `mlperf report <run-dir>` can regenerate HTML |
| Done | Dataset/public-release policy | Dataset dossiers expose `public_release_status`, `public_release_policy`, and `release_next_step`; current full `pytest -q` baseline is 52 passing tests |
| Done | Reference-run comparability policy | Registry, reports, validation CSVs, public rules, focused tests, and full `pytest -q` are clean |
| Done | Canonical list UX | `mlperf list` leads with canonical workload selectors, keeps internal runner IDs as metadata, and full `pytest -q` is clean |
| Done | Training-to-inference provenance hardening | Checkpoint lineage is implemented for shared-checkpoint workloads and full `pytest -q` is clean |
| Done | Native registry layout migration | Native `registry/suites/...` layout is the source of truth; `workloads.yaml` is a generated compatibility mirror |
| Done | Workload expert-review packets | Generated nine public-result review packets, added regression test, and full `pytest -q` passed |
| Done | Public text update | README, SPEC, PROPOSAL, and paper text are aligned with the current CLI, registry layout, validation counts, and honest review-status language |

| Status | Item | Why it matters | Low-compute action |
|---|---|---|---|
| Done | Dataset/license public-release policy | Public MLPerf EDU cannot rely on vague release warnings | Added release-status taxonomy, source-specific warnings, report/CSV/HTML fields, and regression coverage |
| Done | Reference-run comparability policy | Papers need a defensible target-setting method, not just runnable scripts | Added `reference_protocol`, public rules, report/CSV/HTML fields, validation CSV fields, and tests |
| Done | Canonical list UX | Public names should be canonical workloads with variants, not internal runner IDs | List UX now uses canonical selectors first; full `pytest -q` passed |
| Done | Training-to-inference provenance hardening | Inference results need clear upstream checkpoint lineage | Structured checkpoint provenance added; full `pytest -q` passed |
| Done | Native registry layout migration | Public registry layout should be canonical workloads with variant rows | Native registry layout is active and documented; generated mirror check preserves `workloads.yaml` compatibility |
| Done | Workload-specific expert review packets | MLCommons endorsement needs reviewer-readable evidence | Generated packet set and test; full `pytest -q` passed |
| Done | Paper update | Paper should describe working behavior after CLI/report semantics settle | Paper text now describes working behavior, native registry source, and planned review methodology without unsupported completed-study claims |

## Paper Update Policy

Update `paper/paper.tex` near the end of the implementation cycle, after the
CLI, harness, workload naming, report schema, provenance model, and validation
evidence are stable enough that the paper describes working behavior rather than
aspirational behavior.

The paper update should include:

- The final public workflow: `mlperf init`, `mlperf fetch`, `mlperf audit`,
  `mlperf run`, report generation, and grading.
- The settled `min`, `max`, and `pro` profile semantics.
- The canonical workload/variant/phase model.
- The workload matrix and which rows are public-release-ready versus still
  systems-only or experimental.
- The model and dataset rationale policy.
- The training-to-inference provenance chain.
- The public dataset policy and any restricted optional `pro` exceptions.
- The report artifacts: HTML, JSON, CSV, logs, grade output, and `.provd.json`
  manifests.
- The harness/load-generator design and how it relates to MLPerf-style
  measurement without requiring the full official MLPerf stack for classroom
  use.

## Non-Blocking Questions For The User

These are useful but should not block implementation:

1. For SLM default models, should the first public path be one named family
   such as SmolLM2-135M, a Qwen-class model, or an explicit model ladder with
   rationale for each size?
2. Should source/license/citation audit failures block all public statuses now,
   or should the first pass report them as warnings while metadata is filled in?
3. Which paper target should guide `pro` first: quantization, pruning,
   SLM serving, test-time compute, or memory-system studies?
4. Which workloads should require inference to consume an MLPerf EDU-trained
   checkpoint, and which are allowed to use external pretrained weights as the
   base artifact?

## No-Stop Rule

If a command fails:

1. Capture the exact command and failure.
2. Classify it: fetch, audit, run, report, provenance, quality, test, or docs.
3. Fix the smallest root cause.
4. Re-run the command.
5. Continue to the next acceptance check.

The goal is to keep moving until the job is done and leave behind artifacts that
are better than what we started with: clearer commands, stronger audit, better
provenance, better reports, and a more credible workload suite.

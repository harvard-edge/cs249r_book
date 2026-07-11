# MLPerf EDU Product Contract

This document defines the working contract for turning MLPerf EDU from a
research prototype into a runnable teaching and research benchmark harness.
It is intentionally operational: every command listed here should eventually
be executable from a fresh clone on laptop-class hardware.

## Product Goal

MLPerf EDU should become the runnable academic research substrate for ML
systems papers: SPEC-like in usability and citation value, but MLPerf-like in
discipline around fixed workloads, scenarios, quality targets, reproducible
measurement, hardware fingerprints, and verifiable artifacts. A student,
instructor, reviewer, or architecture researcher should be able to run
MLPerf-style training and inference experiments locally without becoming a
benchmark-infrastructure expert.

The reference target is a laptop-class machine. Apple Silicon with PyTorch/MPS
is the first local target because it is common in courses, but the harness
must keep CPU and CUDA paths viable. Backend-specific optimizations are allowed
only when they are explicit in the run manifest.

The north star is captured in [`NORTH_STAR.md`](NORTH_STAR.md): two years from
now, an ML systems paper should be expected to answer whether it ran MLPerf EDU
and, if not, why not. Product iteration follows
[`ITERATION_LOOP.md`](ITERATION_LOOP.md): every change should start from a
stakeholder question, collect feedback by audience lane, implement one bounded
slice, and validate only what changed before broadening the suite.

## Relationship To The Paper

The paper explains the research and pedagogical case for MLPerf EDU. This
contract defines the runnable product surface. When the two disagree, this
document should guide implementation until the paper and README are updated to
match commands that pass local validation.

## Audiences

The project should optimize for these audiences in order:

1. Harvard course use.
2. MLSysBook readers.
3. MLCommons review.
4. Architecture and ML systems researchers.

This order matters. A workload that is impressive for research but fragile in
class should start as experimental, not as part of the course core.

## Profiles

Profiles define run scale. They are separate from suites.

| Profile | Purpose | Runtime Target | Data Policy | Result Use |
|---|---|---:|---|---|
| `min` | Minimum representative run, likely one small path from each major suite | Seconds per selected workload; smoke validation should stay comfortably under 1 min on a laptop | Synthetic or tiny deterministic data allowed | CI, install checks, student setup, demos |
| `max` | Full MLPerf EDU benchmark suite at comparable scale | Full-suite validation should stay practical on a laptop | Real data or documented micro-shards | Course grading, artifact evaluation, paper baselines |
| `pro` | Research envelope exposing controlled variants and optimization knobs | User-configured | Real data, hydrated model weights, and explicit variant metadata | Papers and extended systems studies |

The CLI accepts only the canonical profile names above: `min`, `max`, and `pro`.

The initial `pro` implementation may conservatively reuse the matching `max`
runner, aggregate one or more sub-runs, and bind the sub-run reports/manifests
by hash. That is only a bootstrap path. The north-star `pro` profile should
expose controlled research dimensions such as quantization, pruning, sparsity,
LoRA, backend comparisons, SLM serving shape, batch/context scaling, memory
stressors, distributed/local communication variants, and power or energy
measurement. `MLPERF_EDU_PRO_REPETITIONS` controls the default repeat count
until richer sweep syntax lands.

## Suites And Workloads

Suites define workload domains. Workloads are individual benchmark IDs. Profiles
define run scale. These are the only user-facing selectors.

| Suite | Scope |
|---|---|
| `language` | NanoGPT training/serving, BERT, MoE, LoRA, and other white-box language workloads |
| `slm` | Off-the-shelf small language model inference, LoRA, quantization, and serving |
| `vision` | CNN and mobile vision training/inference, pruning, quantization, and compile studies |
| `recommender` | DLRM-style sparse lookup and memory-system behavior |
| `tiny` | TinyML-style KWS, anomaly detection, and compact models |
| `agent` | RAG, tool calling, ReAct, and code-generation systems cost |
| `distributed` | Local multi-process distributed training and communication/computation studies |
| `graph` | Sparse graph and message-passing workloads |
| `timeseries` | Sequence and forecasting workloads |
| `rl` | Reinforcement-learning control-flow workloads |

Readiness is a maintainer concern, not a user selector. Public commands should
present suite, profile, workload, and variant. Internal readiness review decides
whether a workload belongs in the default `min`/`max` path, should appear only
through `pro` or direct workload runs, or should stay out of release validation.

Public result status is also separate from suite membership:

| Public status | Meaning |
|---|---|
| `score-bearing` | Real dataset, explicit quality target, verified baseline, and comparable performance metrics |
| `performance-bearing` | Standardized functional check and comparable performance metrics, but no public task-quality score |
| `systems-only` | Runnable workload for architecture, backend, kernel, pruning, quantization, distributed, agent, or course studies |
| `experimental` | Under development and excluded from public release |

Run `mlperf audit` before release candidates. It audits public status,
scenario labels, runner coverage, and quality/data contracts.

## Scenario Contract

MLPerf EDU uses MLCommons-style scenarios as a teaching vocabulary, not as a
mandate to reproduce every official MLPerf scenario. The educational
public-result subset is:

| Scenario | Use in MLPerf EDU |
|---|---|
| `single_stream` | One sample, request, token path, or step; latency and step-level behavior |
| `offline` | Throughput-oriented batch or dataset processing |
| `server` | Serving, queueing, decode, retrieval, and latency-throughput tradeoffs |
| `training` | Fixed-budget training followed by a real-data quality gate |

Current score-bearing rows use `training`; performance-bearing inference rows
use `single_stream`, `offline`, or `server`. `inference` remains available only
for systems-only teaching scaffolds. These are MLPerf EDU scenario semantics,
not a claim that `training` is an official MLPerf Inference scenario. A workload
should not be promoted until MLCommons reviewers accept its scenario and
measurement semantics.

## CLI Contract

The CLI should be explicit, composable, and registry-driven. It should never
depend on hard-coded workload names in command handlers.

The canonical executable is `mlperf`. In this package, `mlperf` defaults to the
`mlperf-edu` benchmark suite. Subcommand `--suite` selects a domain such as
`language`, `slm`, `vision`, or `tiny`; `--workload` selects one workload by
registry ID; `--profile` selects `min`, `max`, or `pro`. With no `--suite` or
`--workload`, commands use the default fast path for the selected profile.
If `--workload` names a canonical workload, commands target every variant under
that workload family; add `--variant` to target exactly one variant. Once
`--suite` or `--workload` is explicit, `--profile` is the run/audit context and
not a second hidden workload filter.
`mlperf-edu` may remain as a compatibility alias, but examples and course
instructions should use `mlperf`.

### Setup

```bash
mlperf doctor
mlperf audit
mlperf init --profile min
mlperf init --profile max
mlperf init --profile pro
mlperf init --profile min --output-dir submissions/init_smoke
```

`doctor` performs a non-mutating environment check:

- Python version.
- PyTorch import and available backends.
- Optional backend availability: CUDA, MPS, ONNX Runtime, MLX, llama.cpp.
- Disk space.
- Writable cache and submission directories.
- Optional power hooks.
- Registry validity.

`init` may mutate local state:

- Create cache directories.
- Fetch or verify profile-required datasets.
- Hydrate profile-required model weights.
- Record local machine capabilities.
- Run the `min` profile smoke validation unless `--no-smoke` is passed, writing the
  same aggregate JSON, HTML, and CSV reports as `run`.

### Data And Model Fetching

```bash
mlperf fetch --profile max
mlperf fetch --suite slm --profile max
mlperf fetch --suite slm --profile max --model smollm2-135m
mlperf fetch --suite slm --profile pro --model qwen3-0.6b
mlperf fetch --workload nanogpt-train --profile max
mlperf fetch --dry-run --profile max
```

Fetch should be automatic when a run needs missing assets, but the CLI must
print what it is downloading and where it will be cached. A benchmark run
should never silently switch from real data to synthetic data. Fallbacks are
allowed only in `min` profile and must be recorded in the manifest.

Assets that require accounts, license click-through, or tokens are not part of
the default course path. They may be available in `pro` profile when
the manifest records the requirement.

### Running Benchmarks

```bash
mlperf run --profile min
mlperf run --profile max --dry-run
mlperf run --profile max
mlperf run --profile max --power
mlperf run --suite slm --profile min
mlperf run --suite slm --profile max --model smollm2-135m
mlperf run --suite slm --profile max
mlperf run --suite agent --profile min
mlperf run --workload nanogpt-train --profile max
mlperf run --workload smollm2-chat-inference --variant baseline --model qwen3-0.6b --profile pro
```

Every run produces:

- A report JSON.
- A human-readable HTML report for course inspection.
- A CSV report for spreadsheets and lightweight analysis.
- A provenance manifest.
- Hardware fingerprint.
- Dataset/model hashes.
- Runtime/backend configuration.
- Quality and performance metrics.
- Optional power telemetry.

`run --dry-run` prints the selected workloads and canonical run selectors
without writing artifacts.

`--power` records aggregate estimated average watts and joules in the suite
report. The first implementation is an explicit nominal estimate that works
without sudo; hardware counter integrations can replace the source while
preserving the report schema.

### Artifacts

```bash
mlperf verify submissions/latest.provd.json
mlperf report submissions/latest.json
mlperf report submissions/latest.json --format html --open
mlperf report submissions/latest.json --format csv
mlperf report submissions/latest.json --format json
mlperf report submissions/ --format html --open
mlperf package submissions/latest.provd.json
mlperf grade submissions/
```

There should be one submission schema and one verifier. The verifier must
recompute every hash it can locally recompute and explain missing evidence
clearly. The course grader should consume verifier output instead of
implementing a parallel integrity path.

Suite runs should write HTML and CSV sidecars by default next to the canonical
JSON report. Browser launch should be opt-in (`--open-report` or
`report --format html --open`) so batch, CI, and autograder jobs stay
non-interactive.

`package` should refuse unverified manifests and then bundle the `.provd.json`,
measurement report, package index, and available optional evidence such as
checkpoints or report sidecars. `grade` should scan a submissions directory for
`.provd.json` files, run the same verifier as `verify`, summarize quality
status from the measurement report, preserve public-release warning counts, and
optionally write a grading JSON.

### Bundled Validation

```bash
mlperf validate smoke
mlperf validate coverage
mlperf validate max
mlperf validate release --output-dir submissions/validation
mlperf validate max --suite vision --suite recommender
mlperf audit
```

`validate` is the one-command confidence path for CI, course setup checks,
and release candidates. It expands to registry-driven `run` and `grade` commands
using stable output directories named for the validation item, such as
`submissions/validation/min-default`.

When no seed environment variable is present, `max` validation uses seed 0
from the declared five-seed reference protocol. Validation reports record the
resolved seed and whether it came from the protocol default or an explicit
environment override.

Benchmark profiles remain `min`, `max`, and `pro`. Validation presets name the
scale they exercise:

| Preset | Expansion |
|---|---|
| `smoke` | `doctor` plus default `min` run |
| `coverage` | every workload at `min` |
| `max` | every workload at `max` |
| `release` | every workload at `min` followed by every workload at `max` |

`--dry-run` prints the planned validation items without writing artifacts.
`--keep-going` continues after a failed item so release candidates can report
all failures in one pass. `--skip-doctor` and `--skip-grade` are reserved for
CI jobs that already ran those checks explicitly.
Each non-dry-run validation writes one top-level JSON, HTML, and CSV summary next to
the per-suite artifacts.

Current observed fresh-install validation budgets on Apple Silicon are:

| Validation | Expected Scope | Current Observation |
|---|---|---:|
| `smoke` | Doctor plus default 12-workload `min` run | 12.3 s |
| `coverage` | All 30 registered `min` manifests | 26.5 s |
| `max` | All 30 registered `max` manifests | 93.4 s |
| `release` | All 30 `min` plus all 30 `max` manifests | 278.3 s |

Validation JSON must include top-level `duration_seconds`, per-suite
`duration_seconds`, pass/fail totals, warning totals, per-suite report links,
the `mlperf_suite` identifier, and a normalized workload breakdown. Validation
runs also write `mlperf_validate_workloads_<preset>_<timestamp>.csv` for
spreadsheet-friendly per-workload analysis, including canonical selectors,
dataset terms, and shared checkpoint dependencies.

### Discovery

```bash
mlperf list
mlperf list matrix
mlperf list matrix --profile max
mlperf list --suite vision
mlperf list --profile min
mlperf list --public-status score-bearing
mlperf show nanogpt-train
mlperf show smollm2-chat-inference --variant baseline
mlperf audit
mlperf audit --profile min --format json
mlperf audit --suite slm --format json
```

`list` and `show` read only the workload registry. They should report runtime
targets, data requirements, supported profiles, supported backends, and known
caveats.

`audit` is the release-facing registry audit. It does not run benchmarks;
it verifies that every workload has a public status, rationale, scenario,
required runner coverage, and the data/quality contract required by its public
status. It should separate local development failures from MLCommons
endorsement readiness. `mlperf audit` is the clean local contract for students,
instructors, and maintainers; `mlperf audit --policy public` fails until
public-release warnings are resolved or explicitly accepted.

## Backend Policy

PyTorch is the first-class backend because it is the most teachable and the
closest match to the existing white-box reference code. Other backends are
supported when they serve a clear systems question:

| Backend | Role |
|---|---|
| PyTorch CPU | Portability and baseline correctness |
| PyTorch MPS | Primary Apple Silicon course path |
| PyTorch CUDA | Optional accelerator path |
| ONNX Runtime | Portable inference and deployment comparison |
| MLX | Apple Silicon SLM inference and LoRA research path |
| llama.cpp | Quantized local SLM serving path |

Backend choice must be explicit in the run manifest. Backend-specific kernels
are allowed in optimization studies, but the reference `max` path should
remain understandable.

## Model Policy

MLPerf EDU should reuse standard networks and off-the-shelf small language
models where that improves credibility and reduces maintenance. It should not
hand-build every model solely for purity.

The suite should use two model classes:

1. **White-box pedagogical models** for teaching internals: NanoGPT,
   ResNet/MobileNet, Micro-DLRM, DS-CNN, anomaly detection, and similar small
   networks.
2. **Hydrated off-the-shelf models** for SLM inference, LoRA, quantization,
   pruning, serving, and backend comparison.

Default SLM choices should prefer permissive and easy-to-fetch models. Good
initial candidates include Qwen, Gemma, Phi, and other small open models whose
licenses and download paths fit course use. Llama-family support may be
added when licensing and access are appropriate, but it should not be required
for the default course path.

The first implemented SLM public selector is `smollm2-chat-inference`, with
`baseline`, `quantized-int8`, `batched-b4`, and `long-context` variants. The
native registry keeps compatibility runner IDs such as `slm-decode`,
`slm-quantized-decode`, `slm-batched-decode`, and `slm-long-context-decode` as
metadata, while public commands and reports use the canonical workload/variant
selectors. Their `min` profiles use a deterministic tiny local Transformers
GPT-2 configuration so setup tests never depend on network access.
Their `max` profiles default to `HuggingFaceTB/SmolLM2-135M-Instruct`; the
quantized variant applies CPU dynamic int8 quantization before decode,
`batched-b4` records batch size, requests/sec, and aggregate output tokens/sec,
and `long-context` records configured/measured context length, prompt size, and
prefill latency. `--model` supports aliases such as `smollm2-135m`,
`qwen2.5-0.5b`, and `qwen3-0.6b`.

Full local SLM pretraining is out of scope for default profiles. SLM work
should focus on:

- Inference.
- Quantization.
- Pruning or structured sparsity when supported.
- LoRA or adapter fine-tuning.
- KV-cache behavior.
- Batching and serving policy.
- Backend comparison.

## Workload Registry Contract

Each workload entry should define enough information for the CLI to run it
without task-specific command branches:

```yaml
id: nanogpt-train
suite: language
task: language_modeling
mode: train
scenario: single_stream
public:
  status: score-bearing
  rationale: Real TinyShakespeare data with a fixed cross-entropy target and verified baseline.
profiles:
  min:
    runtime_budget_seconds: 60
    data: tinyshakespeare-micro
  max:
    runtime_budget_seconds: 900
    data: tinyshakespeare
  pro:
    runtime_budget_seconds: null
    data: tinyshakespeare
runner:
  module: reference.cloud.nanogpt_train
  function: run
backends:
  default: pytorch
  supported: [pytorch-cpu, pytorch-mps, pytorch-cuda]
quality:
  metric: cross_entropy_loss
  direction: lower
  threshold: 2.3
artifacts:
  emits_checkpoint: true
  emits_report: true
  emits_provenance: true
```

The registry must validate before any benchmark runs.

## Initial Public Candidate Workloads

The first base suite should be small enough to finish on a laptop and broad
enough to cover the major systems regimes:

| Workload | Systems Lesson |
|---|---|
| `nanogpt-train` | Training loop, attention cost, convergence-to-quality |
| `nanogpt-inference --variant prefill` / `nanogpt-inference --variant decode` | LLM serving, TTFT, ITL, KV-cache, bandwidth pressure |
| `resnet18-train` / `mobilenetv2-train` | Vision training, data pipeline, pruning, quantization |
| `micro-dlrm-train` | Sparse lookup, embedding behavior, memory hierarchy |
| `dscnn-kws` or `anomaly-ae` | Tiny/edge compression and quality retention |

Additional workloads stay in a maintainer-only promotion stage until their data, runtime, and
quality targets are reliable across common laptop setups.

## Development Loop

Work proceeds one benchmark at a time. A benchmark is not "working" because
one script ran once; it is working when it has assets, runners, reports,
provenance, verification, tests, docs, and repeatable local validation.
The stakeholder feedback rhythm is defined in
[`ITERATION_LOOP.md`](ITERATION_LOOP.md); this section defines the engineering
bring-up loop once a specific workload or harness slice has been selected.

### Benchmark Bring-Up Loop

For each workload, repeat this loop until all validation passes:

1. **Select the next workload.**
   Work in this order unless there is a blocking dependency:
   `nanogpt-train`, `nanogpt-inference --variant prefill`,
   `nanogpt-inference --variant decode`,
   `micro-dlrm-train`, `resnet18-train`, `anomaly-ae-train`,
   then the remaining workloads still in maintainer bring-up.

2. **Define the contract.**
   Add or normalize the registry entry before touching runner code:
   suite, profiles, data assets, model assets, runner module,
   backend support, public status, scenario, quality target, metrics, runtime
   budgets, and artifacts.

3. **Make assets deterministic.**
   `mlperf fetch --workload <id> --profile min` must be no-network or
   tiny. `max` may download real data or micro-shards, but it must cache them,
   hash them, and never silently fall back to synthetic data. `max` may use
   larger or restricted assets only when the manifest records the requirement.

4. **Implement the `min` runner.**
   The `min` profile should validate imports, instantiate the model or backend,
   run a tiny deterministic forward/training/inference path, and emit report
   plus provenance. It is allowed to use synthetic or tiny data, but the report
   must label that fact. Until a workload has a `min` runner, suite-level
   reports may mark it as `definition_valid`, but that is an inventory state,
   not a passing benchmark state.

5. **Implement the `max` runner.**
   The `max` profile should run the default comparable benchmark with real metrics
   and a quality check. It should fit the workload's runtime budget on a
   laptop-class machine and produce comparable results.

6. **Implement the `pro` runner only when useful.**
   The `pro` profile is for repetitions, ablations, larger SLMs, backend
   comparisons, quantization/pruning sweeps, and research experiments. It is
   not required before a default workload can become course-ready.

7. **Emit the standard artifacts.**
   Every run writes a report JSON and `.provd.json` manifest containing:
   workload id, profile, suite, backend, hardware fingerprint, dataset hashes,
   model/checkpoint hashes when applicable, seeds, quality metrics,
   performance metrics, and optional power metrics.
   Suite-level runs also write HTML and CSV summaries by default. `report`
   should be able to convert either single-workload or aggregate reports to
   `summary`, `json`, `csv`, or `html`.

8. **Make verification authoritative.**
   `mlperf verify <manifest>` must recompute every locally available hash
   and fail clearly when evidence is missing or modified. Grading and reports
   consume verifier output rather than implementing separate integrity logic.

9. **Add tests at the right level.**
   Each workload gets:
   import test, registry test, fetch dry-run test, `min` smoke test, artifact
   schema test, verifier test, and one `max` validation test if runtime allows.
   Slow `max` and `pro` validation can be marked separately, but must be runnable.

10. **Run local validation.**
    At minimum:

    ```bash
    mlperf doctor
    mlperf audit
    mlperf fetch --workload <id> --profile min --dry-run
    mlperf run --workload <id> --profile min
    mlperf verify submissions/latest.provd.json
    pytest
    ```

    For default `max` workloads, also run:

    ```bash
    mlperf fetch --workload <id> --profile max
    mlperf run --workload <id> --profile max
    mlperf report submissions/latest.json
    ```

11. **Document only passing commands.**
    README and examples may mention the workload only after the matching local
    command passes. If a command is planned but not implemented, it belongs in
    this spec, not in the quick start.

12. **Promote deliberately.**
    A workload should enter the default `min` path only after it runs with
    artifacts and verifier. It should enter the default `max` path only when it
    passes repeatedly on laptop-class hardware and has a stable quality target
    or functional check. Public status promotion is stricter: `systems-only`
    can become `performance-bearing` only when it has a stable standardized
    functional check, and `score-bearing` only when real-data quality is
    verified.

13. **Continue to the next workload.**
    Do not broaden the suite while a selected benchmark has failing starter validation.
    If blocked by data licensing, runtime, or a missing backend, mark the
    blocker in the registry and move the workload back to `experimental`.

### Implementation Slice Loop

Each code slice should follow the same loop:

1. Pick one workload or harness capability.
2. Define its registry entry and profile budgets.
3. Implement the smallest runner that produces report plus provenance.
4. Add `min` profile coverage.
5. Add `max` profile coverage if the workload is a base candidate.
6. Run local validation on laptop hardware.
7. Update docs only for commands that pass.
8. Promote only after repeated clean runs.

The first implementation slice is:

```text
Make the harness honest:
  - executable name: mlperf
  - default MLPerf suite: mlperf-edu
  - commands: doctor, list, show, fetch, init, run, verify, report, package, grade, validate
  - profiles: min, max, pro
  - suites: language, slm, vision, recommender, tiny, agent, distributed, graph, timeseries, rl
  - registry-driven command dispatch
  - tests import the installed package that actually exists
```

The second implementation slice is:

```text
Make one workload complete, then repeat:
  - nanogpt-train
  - min profile with deterministic execution
  - report and provenance manifest
  - verifier
  - pytest coverage for CLI, registry, artifacts, and verification
  - max profile with real TinyShakespeare quality check
  - README command that passes from a fresh clone once max is stable
```

Current bring-up ledger:

| Workload | `min` Runner | Verifier | Tests | `max` Runner | Next Step |
|---|---|---|---|---|---|
| `nanogpt-train` | Implemented | Implemented | Implemented | Implemented | Replace generic `pro` repetition fallback with SLM-scale sweeps and power telemetry |
| `nanogpt-inference --variant prefill` | Implemented | Implemented | Implemented | Implemented | Replace generic `pro` repetition fallback with larger-context sweeps and power telemetry |
| `nanogpt-inference --variant decode` | Implemented | Implemented | Implemented | Implemented | Replace generic `pro` repetition fallback with batched serving sweeps and power telemetry |
| `micro-dlrm-train` | Implemented | Implemented | Implemented | Implemented | Replace generic `pro` repetition fallback with sparse-table scaling sweeps |
| `resnet18-train` | Implemented | Implemented | Implemented | Implemented | Replace generic `pro` repetition fallback with pruning/quantization studies |
| `anomaly-ae-train` | Implemented | Implemented | Implemented | Implemented | Replace generic `pro` repetition fallback with ToyADMOS audio-data extension |

## Release Validation

### Validation: Harness

```bash
pip install -e ".[dev]"
mlperf audit
mlperf validate smoke
pytest
```

### Validation: Complete Min Coverage

```bash
mlperf validate coverage --output-dir submissions/validation
```

This validation should pass without network access or privileged machine setup. It is
the out-of-the-box course confidence path: every registered workload must
construct its model, execute a tiny deterministic path, emit JSON/HTML/CSV
reports, write a provenance manifest, verify locally, and grade cleanly.

### Validation: Max Path

```bash
mlperf run --profile max --output-dir submissions/validation/max-all
```

### Validation: Max Validation

```bash
mlperf validate max --output-dir submissions/validation
```

This validation covers all registered workloads at `max` scale. Some max workloads use
synthetic micro-shards or local tensor-shard overrides where real datasets are
too large for course setup; those reports must label `data_mode` clearly and
mark non-score-bearing quality checks as disabled.

### Validation: Complete Max Coverage

```bash
mlperf validate release --output-dir submissions/validation
```

This validation covers every registered workload at `min` and `max`, including the
workloads still in maintainer bring-up. Those `max` workloads may use deterministic
micro-shards until their real-data quality checks are promoted; reports must
label those runs as `synthetic-micro-shard` and keep quality checks disabled.

### Validation: Research Envelope

```bash
MLPERF_EDU_PRO_REPETITIONS=1 mlperf run --profile pro --output-dir submissions/pro-default
mlperf grade submissions/pro-default --output submissions/pro-default/grade.json
```

### Validation: Vision Compression

```bash
mlperf validate release --suite vision --output-dir submissions/validation
mlperf run --workload mobilenet-cifar100-composed-fp16 --profile max
for manifest in submissions/mobilenet*_*.provd.json; do mlperf verify "$manifest"; done
```

### Validation: Recommender Systems

```bash
mlperf validate release --suite recommender --output-dir submissions/validation
mlperf run --workload micro-dlrm-train --profile max
for manifest in submissions/micro-dlrm*_*.provd.json; do mlperf verify "$manifest"; done
```

### Validation: TinyML

```bash
mlperf validate release --suite tiny --output-dir submissions/validation
mlperf run --workload anomaly-ae-train --profile max
for manifest in submissions/*_min.provd.json submissions/anomaly-ae-train_max.provd.json; do mlperf verify "$manifest"; done
```

### Validation: Agent Systems

```bash
mlperf validate release --suite agent --output-dir submissions/validation
```

This validation must run the RAG, iterative code-generation, ReAct, and structured
tool-call workloads locally without external APIs. It validates the systems loop
and provenance path. Current `max` runs are comparable systems measurements;
higher-fidelity task quality checks remain future `max`/`pro` work.

### Validation: Research SLM

```bash
mlperf validate release --suite slm --model smollm2-135m --output-dir submissions/validation
MLPERF_EDU_PRO_REPETITIONS=1 mlperf run --suite slm --profile pro --model smollm2-135m
mlperf grade submissions --output submissions/slm_grade.json
```

## Non-Goals For The First Stable Release

- Full official MLPerf equivalence.
- Cluster-scale training.
- Mandatory external accounts or restricted model access.
- Training off-the-shelf SLMs from scratch.
- Treating synthetic-data fallbacks as benchmark-valid results.
- Supporting every experimental workload in the default course run.

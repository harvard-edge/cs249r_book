# Changelog

<!--
Format note
===========
Each release entry follows the structure below. The mlsysim-pypi-publish
workflow extracts the `## vX.Y.Z (YYYY-MM-DD)` section and wraps it with
install, links, and an "About MLSys·im" footer to produce the GitHub
Release body. Omit any section that has no entries for a given release.

    ## vX.Y.Z (YYYY-MM-DD) — Optional Short Theme

    [Narrative opening — 1–3 sentences describing the release.]

    ### Highlights
    [Optional; 2–4 bullets for feature+ releases.]

    ### Solvers, Models & Taxonomy
    ### Hardware Registry
    ### Workload & Model Registry
    ### CLI
    ### Python API
    ### Documentation
    ### Packaging & Dependencies
    ### Bug Fixes
    ### Internal

    ### Breaking Changes    (only if present)
    ### Security            (only if present)
    ### Deprecations           (only if present)

    ### Contributors
    - @profvjreddi
-->

## Unreleased

### Solvers, Models & Taxonomy

- `ContinuousBatchingModel` now derives static and paged capacity from a
  request-length distribution instead of assuming static batching reaches 60%
  of the paged batch. Static allocation reserves `max_seq_len` per request;
  paged allocation holds `ceil(S/p)` blocks averaged over exponential request
  lengths capped at `max_seq_len` with mean `mean_request_tokens`. Page size
  now changes fragmentation and capacity even when the context divides evenly
  by the page size, and `speedup_vs_static` compares memory-bound decode
  throughput at the two concurrencies. Provenance now cites Kwon et al. (2023)
  Fig. 2 correctly (20.4% to 38.2% of KV memory holds token states under
  contiguous pre-allocation).
- `ServingCapacityModel` accepts `mean_request_tokens` and evaluates base
  latency at the mean request length.
- Added `calc_capped_exponential_scale` and `calc_expected_paged_kv_tokens` to
  `mlsysim.physics`.
- Added the `mlsysim.Agents` registry (`Coding.SWE_Bench_Runner`,
  `Deliberation.TreeSearch`, `MultiAgent.SupervisorWorker`,
  `Interactive.StreamingVoice`) and the `mlsysim.Embodied` registry
  (`Quadruped.Spot`, `Humanoid.Atlas`, `Humanoid.Unitree_H1`,
  `Manipulator.Panda`, `Drone.DJI_Matrice`, `AMR.LogisticsAMR`,
  `Vehicle.Robotaxi`). Provenance states what is sourced. The Panda record is
  a datasheet. Spot, Atlas, Unitree H1, and the DJI Matrice 350 RTK are
  estimates that link their spec pages and name the unsourced fields. The AMR
  and robotaxi are estimated class profiles, and the four agent profiles are
  illustrative teaching assumptions.
- Added `mlsysim.physics.agents` (trajectory step time and reliability, Pareto
  trajectory-length tail, radix prefix-cache latency, test-time compute cost,
  multi-agent coordination overhead, speedup, and optimal concurrency) and
  `mlsysim.physics.robotics` (sensor-to-actuator latency, stopping distance
  and maximum permitted velocity, kinetic energy, reflected inertia and seam
  torque, inverted-pendulum fall time, actuator Joule heating, and
  action-chunk cadence). Each docstring has a `Source:` line that cites a
  checked reference or names the formula as a modeling assumption.

### Documentation

- Align website tutorials and landing pages with canonical nested registry paths
  (`Hardware.Cloud.*`, `Models.Language.*`, etc.).
- Add Zoo pages for Platforms, Datasets, Literature, and Ops; document
  `Infrastructure.Pricing` and provenance audit workflow on the public site.
- Add CI gate `test_doc_registry_paths.py` for docs and tutorial markdown.
- Document physics module layout, formatting helpers, and import surface in
  `api-stability.md` and `contributing.qmd`.

### Internal

- Converged book-facing provenance on `Provenance` + `Sourced` (`sourced()` factory);
  removed `TraceableConstant`. Appendix lineage and `audit_provenance` use one type.
- Removed duplicate `GPU_UNIT_COST_*` (use `Hardware.Cloud.*.unit_cost`).
- `Metadata` accepts only `provenance` (dropped `source` / `source_url` coalesce fields).
- Removed `mlsysim.core.defaults`; reorganized into `Literature`, `Systems`, `Infrastructure`,
  `Ops.Monitoring`, and `core.calibration` (solver/engine parameters only).
- Added `Infrastructure.Pricing` (`Cloud`, `Storage`, `Labeling`, `Fleet`, `Capital`).
  Appendix lineage audits registry paths and rejects stale `defaults.*` references.

### Breaking Changes

- `ContinuousBatchingModel.solve()` and `ServingCapacityModel.solve()` rename
  `seq_len` to `max_seq_len`.
- `ContinuousBatchingResult.memory_fragmentation_pct` is replaced by
  `paged_internal_fragmentation` and `static_internal_fragmentation`
  (fractions in [0, 1]). New fields: `static_max_active_requests`,
  `static_throughput_tokens_per_sec`, `static_kv_cache_size`, and
  `mean_request_tokens`.

## v0.1.2 (2026-05-17) — CLI & Website Release Polish

Patch release focused on first-run usability for students, instructors, and
automation, plus a backward-compatible serving-model extension for current
LLM inference scheduling practice and three small first-order modeling additions.

### Solvers, Models & Taxonomy

- `ServingModel.solve()` now accepts optional `prefill_chunk_tokens` to estimate
  a chunked-prefill stall proxy. The default remains unchunked and preserves
  existing TTFT/ITL behavior.
- `ServingResult` now reports `prefill_chunks`, `prefill_chunk_time`, and
  `decode_stall_bound` so users can reason about Sarathi-Serve-style decode
  stall bounds without replacing the two-phase serving model or implementing a
  full scheduler.
- Added `TrainingMemoryModel` for per-accelerator training memory breakdowns:
  weights, gradients, optimizer state, activations, and communication buffers.
- Added `ServingCapacityModel` to compose serving latency, continuous-batching
  capacity, and tail-latency queueing into a first-pass replica estimate.
- Added `MoERoutingModel` and `DistributedModel.solve(...,
  moe_routing_imbalance_factor=...)` for first-order MoE hot-expert routing
  sensitivity.

### CLI

- `-o/--output` now works both globally and after subcommands, so documented
  examples such as `mlsysim eval Llama3_8B H100 -o json` and
  `mlsysim zoo hardware -o json` are executable as written.
- `mlsysim serve` now exposes `--prefill-chunk-tokens` and includes chunked
  prefill metrics in JSON/text output when the option is provided.
- `mlsysim audit -o json` now emits a single JSON object instead of mixing
  human-readable banners into stdout.
- `mlsysim schema -o json` is accepted for consistency; schema output remains
  JSON by design.

### Documentation

- Clarified CLI output-flag placement in the CLI reference.
- Added website math documentation for chunked prefill and prefill/decode
  interference, grounded in Sarathi-Serve, Splitwise, and DistServe.
- Updated website citation snippets and instructor version-pinning guidance for
  the 0.1.2 release.
- Added tutorial-style documentation for training memory, serving capacity, MoE
  routing imbalance, validation boundaries, and efficiency calibration.
- Updated the paper text and math documentation so all new modeling additions
  are documented with verified references.
- Removed experimental internal automation docs from the public site so the release
  documentation stays focused on student and community workflows.
- Updated MLSysBook browser wheel references to `mlsysim-0.1.2-py3-none-any.whl`.

### Packaging & Dependencies

- Version bumped to `0.1.2` across `pyproject.toml`, `mlsysim/__init__.py`,
  and `CITATION.cff`; `date-released` updated to `2026-05-17`.

### Internal

- Added CLI contract tests for command-local output flags and audit JSON purity.
- Added solver tests for optional chunked prefill behavior and validation.
- Added solver tests for training memory accounting, serving capacity planning,
  and MoE routing imbalance.
- Removed an unused solver import so `ruff check .` is clean.
- Verified `quarto render docs` completes for the full MLSys·im website.
- Verified package tests, website render, paper build, and book unit tests
  against the updated wheel/version references.

### Contributors

- @profvjreddi

## v0.1.1 (2026-04-24) — Paper Title Correction

Metadata-only patch release. No code or API changes; safe drop-in
replacement for 0.1.0. Corrects the paper title cited in three places
to match the actual title of the companion paper.

### Documentation

- **Paper title corrected** across `CITATION.cff`, the BibTeX snippet in
  `README.md`, and the reference docstring in `mlsysim/core/walls.py`.
  Was: *"A Composable Analytical Framework for Machine Learning Systems."*
  Now: *"MLSys·im: First-Principles Infrastructure Modeling for Machine
  Learning Systems."*

### Packaging & Dependencies

- Version bumped to `0.1.1` across `pyproject.toml`, `mlsysim/__init__.py`,
  and `CITATION.cff`; `date-released` updated to `2026-04-24`.

### Contributors

- @profvjreddi

## v0.1.0 (2026-04-01)

**Initial release** of MLSysim — the first-principles analytical modeling engine for ML systems.

### Core Framework
- 22-wall taxonomy organizing every constraint that bounds ML system performance
- 20+ analytical solvers (Models, Solvers, Optimizers) covering all 22 walls
- Pint unit system with dimensional analysis throughout
- TraceableConstant pattern — key defaults carry citations for every assumption
- Pipeline composer for chaining solvers with `explain()` and `run()`
- 3-tier evaluation scorecard: Feasibility → Performance → Macro/Economics
- Design Space Exploration (DSE) engine with constraint evaluation

### Hardware Registry
- 15+ accelerators: V100, A100, H100, H200, B200, GB200 NVL72, MI300X, TPUv5p, T4, Cerebras CS-3, Jetson Orin NX, ESP32-S3, nRF52840, Himax WE-I Plus, DGX Spark, MacBook M3 Max, iPhone 15 Pro, Pixel 8
- Full precision support: FP32, TF32, BF16, FP16, FP8, INT8, INT4
- Multi-level memory hierarchy: HBM + SRAM + Flash (TinyML)
- Hardware specs include manufacturer datasheet references where available

### Model Registry
- GPT-2/3/4, LLaMA-2/3 (7B/8B/70B), BERT Base/Large, ResNet-50, MobileNetV2, AlexNet, Mamba, Stable Diffusion v1.5, DS-CNN, WakeVision
- HuggingFace model importer for custom workloads

### Analytical Models
- **SingleNodeModel**: Roofline analysis with SRAM/flash-aware bandwidth selection
- **DistributedModel**: 4D parallelism (DP/TP/PP/EP) with correct activation-based TP communication, gradient accumulation, straggler effects
- **ServingModel**: Prefill/decode with attention O(S²), batch amortization, speculative decoding, disaggregated serving
- **ContinuousBatchingModel**: PagedAttention with KV cache compression
- **WeightStreamingModel**: Cerebras-style with prefill/decode phases
- **TailLatencyModel**: Erlang C (M/M/c) with log-space computation for large clusters
- **ReliabilityModel**: Compound MTBF with correlated failures, goodput ratio
- **CheckpointModel**: Distributed writing with filesystem bandwidth limits
- **SustainabilityModel**: Energy-proportional power, embodied carbon, PUE/WUE/carbon intensity
- **EconomicsModel**: Amortized CapEx, infrastructure multiplier, maintenance
- **CompressionModel**: Quantization (FP8/INT8/INT4) + pruning (unstructured/structured/N:M) with inference speedup
- **ScalingModel**: Chinchilla compute-optimal scaling
- **TopologyModel**: Ring, torus, fat-tree, dragonfly bisection analysis
- **SensitivitySolver**: Numerical partial derivatives for binding constraint identification
- **SynthesisSolver**: Inverse Roofline for hardware spec derivation
- **ParallelismOptimizer**, **BatchingOptimizer**, **PlacementOptimizer**: Design-space search

### CLI
- `mlsysim eval` — evaluate workload on hardware
- `mlsysim zoo` — explore hardware/model registries
- `mlsysim schema` — export solver schemas
- `mlsysim optimize` — design-space search
- `mlsysim audit` — system audit

### Testing
- 367 tests, 100% pass rate
- Direct formula unit tests with known-answer validation
- Solver suite covering all major models
- Physics bounds validation across all hardware
- Wall taxonomy completeness tests
- Pipeline composition tests

### Packaging & Tooling
- Standard nested package layout (`mlsysim/mlsysim/...`) so `pip install -e .` works out of the box without the prefix-add `sources` rewrite that broke the editables backend
- Wheel and sdist contain only the package and project metadata (no `tests/`, `docs/`, `examples/`, `paper/`, `vscode-ext/`)
- Project-wide ruff configuration: `[tool.ruff]` block in `pyproject.toml` with sensible per-file ignores for `__init__.py` re-export patterns, `core/constants.py` unit-registry star imports, and tests/examples idioms; `ruff check .` reports zero issues
- Real bug fixes uncovered by lint cleanup: removed unused `BaseModel` import in `core/solver.py`, fixed `Fleet` shadowing bug in `sim/simulations.py`, narrowed three bare `except:` clauses to specific exception types, and added missing speculative-decoding ITL assertion in `tests/test_sota.py`

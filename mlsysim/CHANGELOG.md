# Changelog

## v0.1.1 (2026-04-24)

**Metadata-only patch release.** No code or API changes; safe drop-in
replacement for 0.1.1.

- Corrected paper title in citations: now reads "MLSys·im:
  First-Principles Infrastructure Modeling for Machine Learning
  Systems" (was "A Composable Analytical Framework for Machine Learning
  Systems"). Updated in `CITATION.cff`, the BibTeX snippet in
  `README.md`, and the reference in `mlsysim/core/walls.py`.

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
- All specs verified against manufacturer datasheets

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

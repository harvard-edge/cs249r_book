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

## v0.3.0 (2026-10-24)

MLSys·im 0.3.0 reorganizes the package around nested registries and a dedicated
`mlsysim.engine` layer, adds Physical AI and agentic modeling (embodied
platforms, sensors, actuators, robotics and drivetrain physics), and corrects
several distributed-training, serving, and hardware-registry numbers. It
removes a number of top-level convenience imports, so code written against
0.1.1 needs the import updates listed below.

> **PyPI note:** 0.1.2 and 0.2.0 were tagged but never published to PyPI, so
> on PyPI 0.3.0 directly follows 0.1.1. If you are upgrading from the PyPI
> 0.1.1 release, the v0.1.2 entry below also applies to you.

### Breaking Changes

**Top-level imports.** The following names are no longer importable from the
top-level `mlsysim` package. Import them from their defining module instead.

| 0.1.1 import | 0.3.0 import |
|---|---|
| `from mlsysim import Workload, TransformerWorkload, CNNWorkload` | `from mlsysim.models.types import Workload, TransformerWorkload, CNNWorkload` |
| `from mlsysim import HardwareNode` | `from mlsysim.hardware.types import HardwareNode` |
| `from mlsysim import Fleet, Node, NetworkFabric` | `from mlsysim.systems.types import Fleet, Node, NetworkFabric` |
| `from mlsysim import DeploymentTier` | `from mlsysim.systems.types import DeploymentTier` |
| `from mlsysim import PerformanceProfile` | `from mlsysim.engine.engine import PerformanceProfile` |
| `from mlsysim import SystemEvaluator, SystemEvaluation` | `from mlsysim.engine.evaluation import SystemEvaluator, SystemEvaluation` |
| `from mlsysim import SimulationConfig, load_config` | `from mlsysim.engine.config import SimulationConfig, load_config` |
| `from mlsysim import SingleNodeModel` (and every other solver or optimizer class) | `from mlsysim.solvers import SingleNodeModel` |
| `from mlsysim import Infra` | `from mlsysim import Infrastructure` (`Infra.Grids`, `Infra.Racks` become `Infrastructure.Grids`, `Infrastructure.Racks`; `Infra.Quebec` becomes `Infrastructure.Grids.Quebec`, likewise `US_Avg`, `Poland`) |
| `from mlsysim import infra` (subpackage) | `from mlsysim import infrastructure` |
| `from mlsysim import Applications` | `from mlsysim import Scenarios` (`Applications.Doorbell` becomes `Scenarios.SmartDoorbell`, `AutoDrive` becomes `AutonomousVehicle`, `Workstation` becomes `LocalTraining`, `Frontier` becomes `FrontierTraining`, `Chatbot` becomes `ChatbotServing`, `KWS` becomes `KeywordSpotting`) |
| `from mlsysim import Archetypes` | `from mlsysim import Scenarios` (same mapping as `Applications`; `Archetypes.Cloud_V100` becomes `Scenarios.FrontierTraining`, `Archetypes.TinyML_M7` becomes `Scenarios.SmartDoorbell`) |
| `from mlsysim import show` | `import mlsysim.show` or `from mlsysim import show` still works as an explicit submodule import; it is no longer imported eagerly |
| `mlsysim.viz` attribute access without import | `import mlsysim.viz` (top-level `plot_roofline` and `plot_evaluation_scorecard` are unchanged) |
| `from mlsysim import ureg` | Unchanged; `ureg` is still exported (now from `mlsysim.core.units`) |
| `from mlsysim import Tiers` | Removed, no replacement registry. The `DeploymentTier` type remains in `mlsysim.systems.types`. |
| `from mlsysim import sim` (`Persona`, `BaseSimulation`, `SystemLedger`, ...) | Removed, no replacement. |
| `from mlsysim import constants` / `mlsysim.core.constants` | Removed. Units and `Q_` live in `mlsysim.core.units`; physical constants in `mlsysim.physics.constants`; specs in the registries. |

**Moved modules.** `mlsysim.core.engine`, `mlsysim.core.solver`,
`mlsysim.core.evaluation`, `mlsysim.core.scenarios`, and `mlsysim.core.config`
moved to `mlsysim.engine.*`. Use `mlsysim.solvers` as the stable import path
for solver classes. `mlsysim.core.defaults` is removed (see Internal).

**Registry paths.** Flat registry aliases are removed; use the nested paths.

| 0.1.1 | 0.3.0 |
|---|---|
| `Hardware.H100`, `H200`, `B200`, `A100`, `V100`, `T4`, `MI250X`, `MI300X`, `Gaudi2`, `Gaudi3`, `TPUv5p`, `TPUv6`, `Trainium2` | `Hardware.Cloud.<same name>` |
| `Hardware.NVL72` | `Hardware.Cloud.GB200_NVL72` |
| `Hardware.CerebrasCS3` | `Hardware.Cloud.Cerebras_CS3` |
| `Hardware.DGXSpark`, `Hardware.MacBook` | `Hardware.Workstation.DGX_Spark`, `Hardware.Workstation.MacBookM3Max` |
| `Hardware.Jetson` | `Hardware.Edge.JetsonOrinNX` |
| `Hardware.iPhone`, `Hardware.Snapdragon` | `Hardware.Mobile.iPhone15Pro`, `Hardware.Mobile.Snapdragon8Gen3` |
| `Hardware.ESP32`, `Hardware.Himax` | `Hardware.Tiny.ESP32_S3`, `Hardware.Tiny.HimaxWE1` |
| `Hardware.Networks.*` | `Systems.Fabrics.*` |
| `Models.GPT2`, `GPT3`, `GPT4`, `Llama2_70B`, `Llama3_8B`, `Llama3_70B` | `Models.Language.<same name>` |
| `Models.ResNet50`, `MobileNetV2`, `AlexNet` | `Models.Vision.<same name>` |
| `Models.WakeVision` | `Models.Tiny.WakeVision` |
| `Models.DLRM` | `Models.Recommendation.DLRM` |
| `Models.Mamba_2_8B` | `Models.StateSpace.Mamba_2_8B` |
| `Models.StableDiffusion_v1_5` | `Models.GenerativeVision.StableDiffusion_v1_5` |
| `Systems.Cloud`, `Edge`, `Mobile`, `Tiny`, `Systems.Tiers` | Removed; use the `Hardware.*` entries directly |

**Behavior changes.**

- `Hardware.TPUv4` and `Hardware.Cloud.TPUv4` were aliases of TPU v5p in
  0.1.1. `Hardware.Cloud.TPUv4` is now a distinct TPU v4 record (275 TFLOP/s,
  32 GiB), so results computed against it change. To reproduce 0.1.1 numbers,
  use `Hardware.Cloud.TPUv5p`; the flat `Hardware.TPUv4` alias is removed.
- `Engine.solve` raises `ValueError` for an unknown `precision` name instead of
  silently using FP16 byte widths.
- `Infrastructure.FacilityCooling.UptimeSurvey2022Average` is removed; use
  `Infrastructure.FacilityCooling.Typical` (now 1.55).
- `ContinuousBatchingModel.solve()` and `ServingCapacityModel.solve()` rename
  `seq_len` to `max_seq_len`.
- `ContinuousBatchingResult.memory_fragmentation_pct` is replaced by
  `paged_internal_fragmentation` and `static_internal_fragmentation`
  (fractions in [0, 1]). New fields: `static_max_active_requests`,
  `static_throughput_tokens_per_sec`, `static_kv_cache_size`, and
  `mean_request_tokens`.
- `Metadata` accepts only `provenance` (the `source` and `source_url` fields
  are dropped); `TraceableConstant` is removed in favor of `sourced()`.

### Physical AI & Embodied Modeling

- **Embodied, Sensors, and Actuators registries (`mlsysim.Embodied`,
  `mlsysim.Sensors`, `mlsysim.Actuators`).** `Embodied` covers
  `Vehicle.UberATG_VolvoXC90`, `Vehicle.Robotaxi`, `AMR.LogisticsAMR`,
  `AMR.WarehouseAMR`, `Manipulator.Panda`, `Quadruped.Spot`,
  `Humanoid.Atlas`, `Humanoid.Unitree_H1`, and `Drone.DJI_Matrice`, with mass,
  velocity, braking, sensing latency, and power envelopes. `Sensors` covers
  `Camera`, `IMU`, and `LiDAR`; `Actuators` covers `FramelessMotor`,
  `HarmonicDrive`, `HubMotor`, `QuasiDirectDrive`, `Servo`, and the composed
  `PandaClassJoint`. Provenance states what is sourced: the Panda record is a
  datasheet; Spot, Atlas, Unitree H1, and the DJI Matrice 350 RTK are
  estimates that link their spec pages and name the unsourced fields; the AMR
  and robotaxi are estimated class profiles.
- **Robotics physics (`mlsysim.physics.robotics`).** Closed-form models for
  stopping and safety envelopes (`calc_safe_stopping_distance`,
  `calc_max_permitted_velocity`, `calc_watchdog_lease_bound`,
  `calc_cbf_qp_orthogonal_projection`), contact and impact
  (`calc_kinetic_energy`, `calc_transient_impact_force`,
  `calc_tripwire_contact_force_accumulation`), robot buses
  (`calc_canfd_bus_utilization`, `calc_ethercat_cycle_time`), perception and
  mapping budgets (`calc_teleop_ingestion_budget`,
  `calc_tsdf_voxel_grid_budget`), validation statistics
  (`calc_clopper_pearson_zero_failure_bound`, `calc_zero_failure_sample_size`),
  and dynamics such as reflected inertia, seam torque, inverted-pendulum fall
  time, and action-chunk cadence. Each docstring has a `Source:` line that
  cites a checked reference or names the formula as a modeling assumption.
- **Drivetrain braking physics (`mlsysim.physics.drivetrain`).** Checks a
  requested deceleration for a wheeled base against three independent
  ceilings: traction (`calc_traction_deceleration_limit`,
  `calc_required_friction`), tip-over (`calc_tipover_deceleration_limit`,
  `calc_max_cg_height_ratio`), and drive torque
  (`calc_drive_torque_deceleration_limit`), plus `calc_drive_voltage_headroom`.
  `calc_braking_limits` combines them and reports the binding ceiling.
- **Physical AI SI units (`mlsysim.core.units`).** Exported `millimeter`/`mm`,
  `newton`/`N`, `volt`/`V`, `millivolt`/`mV`, `ampere`/`A`,
  `milliampere`/`mA`, `ohm`, `milliohm`, `radian`/`rad`, and `hertz`/`Hz`.
- **Provenance catalog (`mlsysim.core.provenance_catalog`)** carries the
  sources and datasheets behind the embodied platforms and parameters.

### Solvers, Models & Taxonomy

- Added the `mlsysim.Agents` registry (`Coding.SWE_Bench_Runner`,
  `Deliberation.TreeSearch`, `MultiAgent.SupervisorWorker`,
  `Interactive.StreamingVoice`); the four agent profiles are illustrative
  teaching assumptions.
- Added `mlsysim.physics.agents`: trajectory step time and reliability,
  Pareto trajectory-length tail, radix prefix-cache latency, test-time compute
  cost, multi-agent coordination overhead, speedup, and optimal concurrency.
- `ContinuousBatchingModel` derives static and paged capacity from a
  request-length distribution instead of assuming static batching reaches 60%
  of the paged batch. Static allocation reserves `max_seq_len` per request;
  paged allocation holds `ceil(S/p)` blocks averaged over exponential request
  lengths capped at `max_seq_len` with mean `mean_request_tokens`. Page size
  now changes fragmentation and capacity even when the context divides evenly
  by the page size, and `speedup_vs_static` compares memory-bound decode
  throughput at the two concurrencies. Provenance cites Kwon et al. (2023)
  Fig. 2 (20.4% to 38.2% of KV memory holds token states under contiguous
  pre-allocation).
- `ServingCapacityModel` accepts `mean_request_tokens` and evaluates base
  latency at the mean request length.
- Added `calc_capped_exponential_scale` and `calc_expected_paged_kv_tokens` to
  `mlsysim.physics`.

### Hardware Registry

- `Hardware.Cloud.TPUv4` records its inter-chip interconnect (ICI) as 300 GB/s
  per direction, six links at 50 GB/s (Jouppi et al. 2023, Table 4), and its
  provenance carries the paper's actual title.
- `Hardware.Cloud.TPUv3`: peak 123 TFLOP/s per chip (BF16) and TDP 450 W per
  chip (Jouppi et al. 2020, CACM, Table 3), replacing 105 TFLOP/s and 250 W;
  ICI 280 GB/s per direction, four links at 70 GB/s (Jouppi et al. 2023,
  Table 4); `release_year` 2018. Provenance points to the CACM article
  (DOI 10.1145/3360307).
- `Hardware.Cloud.TPUv2`: peak 46 TFLOP/s per chip (BF16) and TDP 280 W per
  chip (Jouppi et al. 2020, CACM, Table 3), replacing 45 TFLOP/s and 200 W;
  `release_year` 2017. Provenance points to the CACM article.

### Workload & Model Registry

- `Models.Language.GPT3` reference run is 10,000 V100 GPUs for about 14.8 days
  (Patterson et al. 2021, Table 4), replacing 1,024 accelerators for 25 days,
  which implied more than 100 percent MFU. It gains
  `training_hardware_label: V100`.
- `Models.Language.GPT4` reference duration is 100 days (SemiAnalysis 2023
  estimate of 90 to 100 days on about 25,000 A100s), matching its
  `training_gpu_days` anchor.
- `TransformerWorkload` validates that `training_accelerators_ref` times
  `training_days_ref` agrees with `training_gpu_days` within 1 percent when all
  three are set.
- `Scenarios.FrontierTraining` keeps its 8,192-H100 fleet; its description now
  states that this is the H100 restatement of a GPT-4-class run.

### Infrastructure & Systems Registry

- Each `Infrastructure.FacilityCooling` tier cites its own primary source:
  liquid-cooled 1.06 (NREL ESIF), best air-cooled 1.12 (Microsoft design PUE,
  2022), industry average 1.55 (Uptime 2022) and 1.58 (Uptime 2023, display
  name "Industry Average (2023)", formerly "Legacy"), simple air 1.5 (LBNL
  small server rooms, 2013), state of the art 1.10 (Google fleet TTM,
  Q4 2023). `FacilityCooling.Typical` is 1.55 (Uptime 2022), replacing 1.40.
- Added `Infrastructure.Pricing` (`Cloud`, `Storage`, `Labeling`, `Fleet`,
  `Capital`).
- `Node` gains `psu_redundancy` (M in an N+M supply), `psu_rating`, and
  `max_power`, validated so that the N required PSUs can carry `max_power`.
  The new `series_psus` property counts the PSUs whose single failure stops
  the node. Defaults leave custom nodes unchanged.
- `Systems.Nodes.DGX_H100` (six 3.3 kW PSUs, 4+2, 10.2 kW max),
  `DGX_A100` (six 3 kW PSUs, 3+3, 6.5 kW max), and `DGX_B200` (six 3.3 kW
  PSUs, 5+1, 14.3 kW max) record their supplies from the NVIDIA user guides.
  `ReliabilityModel` now sums `series_psus` rather than `psus_per_node`, so
  node MTBF on these three rises from 4,285.7 h to 4,687.5 h and every fleet
  built on them (including `Clusters.Training_1K_A100`, `Lab_64_A100`, and
  `Meta_RSC_A100_16K`) gains 9.4 percent in fleet MTBF. `HGX_H100_EPYC`,
  `GrandTeton_H100`, `Kempner_H100_4GPU`, and custom nodes keep the
  non-redundant two-PSU default.

### Bug Fixes

- `CompressionModel` measured every ratio against a hard-coded FP32 baseline,
  so INT4 always reported 8x compression, even for FP16/BF16-served models.
  `solve`, `candidate`, and `sweep` now take `baseline_precision` (default
  `"fp32"`, so existing results are unchanged); pass `"fp16"` for FP16/BF16
  models (INT4 = 4x). `CompressionResult` and `CompressionCandidate` record
  the baseline, and the Wall 13 equation reads `r = b_base/b`.
- `DistributedModel` tensor parallelism: a training step now pays 4 separate
  AllReduces per layer (two forward, two backward; Shoeybi et al. 2019, p.4),
  each with its own ring latency term, instead of 2 priced as one collective of
  twice the size. The bandwidth term of `tp_communication_latency` doubles and
  the latency term quadruples.
- `DistributedModel` local step: it priced the whole unsharded model on one
  accelerator with one token per sequence, so multi-billion-parameter
  transformers fell to PCIe offload bandwidth and every TP/PP/DP split looked
  alike. The step is now one model-parallel replica: step FLOPs
  (`inference_flops * seq_len * local_batch`, 3x for training, 4x with
  recomputation) shared across `tp * pp * ep` accelerators, with feasibility
  from `TrainingMemoryModel` under the same sharding, recomputation, and 1F1B
  in-flight microbatches. Tensor-parallel communication covers one pipeline
  stage's layers; the pipeline bubble is added as `compute * b / (1 - b)`;
  `effective_throughput` multiplies by the DP replica count.
- `DistributedModel` data-parallel AllReduce: each rank now holds a `tp * pp`
  shard, a node holds `accelerators_per_node // (tp * pp * ep)` ranks, and the
  collective is an intra-node ring, a fabric ring, or hierarchical
  accordingly. Previously gradients were sized over TP only and the ring was
  split by accelerators per node.
- `SensitivitySolver` returned all-zero sensitivities with `peak_flops` named
  as binding when the configuration did not fit in memory. `SensitivityResult`
  now carries `feasible`; an infeasible baseline reports
  `binding_constraint="memory_capacity"` with the Memory Wall failure in
  `constraint_trace`.
- `ParallelismOptimizer` drops candidates whose replica does not fit once
  activations are counted, prices pipelined candidates with one-sample
  microbatches, takes `seq_len` and `activation_recomputation`, and names the
  memory remedies when nothing fits.
- `TrainingMemoryModel` now divides activations by the tensor-parallel degree
  when `sequence_parallel=True` (the default; Korthikanti et al. 2023).
  `tp_size=1` results are unchanged.
- `WeightStreamingModel.wafer_memory_utilization` divided GB by GiB without
  reducing units, overstating utilization by 7.4%.

### Documentation

- Website tutorials and landing pages use canonical nested registry paths
  (`Hardware.Cloud.*`, `Models.Language.*`, and so on), enforced by the CI
  gate `test_doc_registry_paths.py`.
- Added Zoo pages for Platforms, Datasets, Literature, and Ops; documented
  `Infrastructure.Pricing` and the provenance audit workflow.
- Documented the physics module layout, formatting helpers, and import surface
  in `api-stability.md` and `contributing.qmd`.

### Packaging & Dependencies

- Version bumped to `0.3.0` across `pyproject.toml`, `mlsysim/__init__.py`,
  and `CITATION.cff`.
- Raised dependency floors: `numpy>=2.0` (was 1.24), `pint>=0.24.4`,
  `pydantic>=2.10.5`, `typer>=0.25.1`, `rich>=15.0.0`. The floors match the
  Pyodide bundles used by the browser labs.
- New `mcp` extra; `full` now includes it. The `labs` extra no longer pulls in
  `plotly`.

### Internal

- Book-facing provenance converged on `Provenance` + `Sourced` (`sourced()`
  factory). Appendix lineage and `audit_provenance` use one type, audit
  registry paths, and reject stale `defaults.*` references.
- Removed duplicate `GPU_UNIT_COST_*` constants (use
  `Hardware.Cloud.*.unit_cost`).
- Removed `mlsysim.core.defaults`; its contents moved to `Literature`,
  `Systems`, `Infrastructure`, `Ops.Monitoring`, and `core.calibration`
  (solver and engine parameters only).

### Contributors

- @profvjreddi

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

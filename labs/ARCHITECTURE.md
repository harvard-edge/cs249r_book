# Labs Architecture: The 22-Wall Laboratory System

**Status:** Architecture Complete | Implementation: Summer 2026
**Backend:** `mlsysim` analytical simulator (22 walls, 20+ solvers)
**Format:** Marimo reactive notebooks (.py), 2-Act structure, 30–45 min each

---

## Design Philosophy

Each lab is a **quantitative trade-off exploration** powered by `mlsysim` solvers.
Students never write code — they manipulate design knobs (sliders, toggles, selectors)
and observe how physical constraints bind. Every number has a traceable source.

The 22-wall taxonomy provides the analytical backbone:

```
Domain 1 — Node       : Walls 1–7   (compute, memory, software, serving, batching, streaming, tail latency)
Domain 2 — Data       : Walls 8–10  (ingestion, transformation, locality)
Domain 3 — Algorithm  : Walls 11–13 (complexity, reasoning, fidelity)
Domain 4 — Fleet      : Walls 14–16 (communication, fragility, multi-tenant)
Domain 5 — Operations : Walls 17–20 (capital, sustainability, checkpoint, safety)
Domain 6 — Analysis   : Walls 21–22 (sensitivity, synthesis)
```

---

## Master Wall × Lab Mapping

### Volume 1: Single-Machine Foundations (16 Labs, including Lab 06)

| # | Lab Title | Chapter | Primary Walls | Solvers | Duration |
|---|-----------|---------|---------------|---------|----------|
| 01 | The Magnitude Gap | `@sec-ml-intro` | — (Orientation) | Hardware Registry | 45 min |
| 02 | The Physics of Performance | `@sec-ml-systems` | W1 (Compute), W2 (Memory) | `SingleNodeModel` | 45 min |
| 03 | Iteration Velocity | `@sec-ml-workflow` | W8 (Ingestion), W9 (Transform) | `DataModel`, `TransformationModel` | 35 min |
| 04 | Data Gravity | `@sec-data-engineering` | W8 (Ingestion), W10 (Locality) | `DataModel`, `TopologyModel` | 35 min |
| 05 | The Transistor Tax | `@sec-neural-computation` | W1 (Compute), W2 (Memory) | `SingleNodeModel` | 40 min |
| 06 | The Architecture Bottleneck | `@sec-network-architectures` | W1 (Compute), W2 (Memory) | `SingleNodeModel`, `EfficiencyModel` | 40 min |
| 07 | The Kernel Fusion Gap | `@sec-ml-frameworks` | W3 (Software) | `EfficiencyModel` | 40 min |
| 08 | The MFU Gap | `@sec-model-training` | W1 (Compute), W2 (Memory), W3 (Software) | `SingleNodeModel`, `EfficiencyModel` | 40 min |
| 09 | Diminishing Returns | `@sec-data-selection` | W11 (Complexity), W13 (Fidelity) | `ScalingModel`, `CompressionModel` | 40 min |
| 10 | The Compression Pareto | `@sec-model-compression` | W13 (Fidelity) | `CompressionModel` | 40 min |
| 11 | The Memory Wall | `@sec-hw-acceleration` | W1 (Compute), W2 (Memory) | `SingleNodeModel`, `EfficiencyModel` | 45 min |
| 12 | Amdahl's Ceiling | `@sec-perf-benchmarking` | W3 (Software), W21 (Sensitivity) | `EfficiencyModel`, `SensitivitySolver` | 40 min |
| 13 | The Batching Paradox | `@sec-model-serving` | W4 (Serving), W5 (Batching), W7 (Tail Latency) | `ServingModel`, `ContinuousBatchingModel`, `TailLatencyModel` | 45 min |
| 14 | Silent Degradation | `@sec-ml-ops` | W7 (Tail Latency), W19 (Checkpoint) | `TailLatencyModel`, `CheckpointModel` | 40 min |
| 15 | The Fairness Tax | `@sec-responsible-engineering` | W20 (Safety) | `ResponsibleEngineeringModel` | 40 min |
| 16 | The Architect's Synthesis | `@sec-ml-conclusion` | W21 (Sensitivity), W22 (Synthesis) | `SensitivitySolver`, `SynthesisSolver` | 45 min |

### Volume 2: Fleet-Scale Systems (17 Labs)

| # | Lab Title | Chapter | Primary Walls | Solvers | Duration |
|---|-----------|---------|---------------|---------|----------|
| 01 | The Scale Illusion | `@sec-v2-ml-intro` | W14 (Communication) | `DistributedModel` | 45 min |
| 02 | The Interconnect Wall | `@sec-compute-infra` | W1 (Compute), W2 (Memory), W10 (Locality) | `SingleNodeModel`, `TopologyModel` | 45 min |
| 03 | Bisection Bandwidth | `@sec-network-fabrics` | W10 (Locality), W14 (Communication) | `TopologyModel`, `DistributedModel` | 45 min |
| 04 | Data Gravity at Scale | `@sec-data-storage` | W8 (Ingestion), W10 (Locality) | `DataModel`, `TopologyModel` | 40 min |
| 05 | The Parallelism Paradox | `@sec-dist-training` | W14 (Communication), W3 (Software) | `DistributedModel`, `EfficiencyModel` | 45 min |
| 06 | The Bandwidth Invariant | `@sec-collective-comms` | W14 (Communication), W10 (Locality) | `DistributedModel`, `TopologyModel` | 45 min |
| 07 | The Checkpoint Reckoning | `@sec-fault-tolerance` | W15 (Fragility), W19 (Checkpoint) | `ReliabilityModel`, `CheckpointModel` | 45 min |
| 08 | The Utilization Trap | `@sec-fleet-orchestration` | W16 (Multi-tenant), W17 (Capital) | `OrchestrationModel`, `EconomicsModel` | 45 min |
| 09 | Performance Engineering | `@sec-perf-engineering` | W21 (Sensitivity), W3 (Software) | `SensitivitySolver`, `EfficiencyModel` | 45 min |
| 10 | The KV-Cache Wall | `@sec-dist-inference` | W4 (Serving), W5 (Batching), W7 (Tail Latency) | `ServingModel`, `ContinuousBatchingModel`, `TailLatencyModel` | 45 min |
| 11 | The Federation Paradox | `@sec-edge-intelligence` | W10 (Locality), W14 (Communication) | `TopologyModel`, `DistributedModel` | 40 min |
| 12 | SLO Composition | `@sec-ops-at-scale` | W7 (Tail Latency), W16 (Multi-tenant) | `TailLatencyModel`, `OrchestrationModel` | 45 min |
| 13 | The Privacy Gap | `@sec-security-privacy` | W20 (Safety) | `ResponsibleEngineeringModel` | 40 min |
| 14 | The Robustness Tax | `@sec-robust-ai` | W20 (Safety), W13 (Fidelity) | `ResponsibleEngineeringModel`, `CompressionModel` | 40 min |
| 15 | Jevons Paradox | `@sec-sustainable-ai` | W18 (Sustainability), W17 (Capital) | `SustainabilityModel`, `EconomicsModel` | 45 min |
| 16 | Fairness at Scale | `@sec-responsible-ai` | W20 (Safety), W16 (Multi-tenant) | `ResponsibleEngineeringModel`, `OrchestrationModel` | 40 min |
| 17 | The Fleet Architect | `@sec-v2-ml-conclusion` | W21 (Sensitivity), W22 (Synthesis) | `SensitivitySolver`, `SynthesisSolver` | 45 min |

---

## Wall Coverage Analysis

Every wall is exercised by at least one lab per volume:

| Wall | Name | Vol 1 Labs | Vol 2 Labs |
|------|------|------------|------------|
| W1  | Compute | 02, 05, 06, 08, 11 | 02 |
| W2  | Memory | 02, 05, 06, 08, 11 | 02 |
| W3  | Software | 07, 08, 12 | 05, 09 |
| W4  | Serving | 13 | 10 |
| W5  | Batching | 13 | 10 |
| W6  | Streaming | — | (advanced extension) |
| W7  | Tail Latency | 13, 14 | 10, 12 |
| W8  | Ingestion | 03, 04 | 04 |
| W9  | Transformation | 03 | — |
| W10 | Locality | 04 | 02, 03, 04, 06, 11 |
| W11 | Complexity | 09 | — |
| W12 | Reasoning | — | (advanced extension) |
| W13 | Fidelity | 09, 10 | 14 |
| W14 | Communication | — | 01, 03, 05, 06, 11 |
| W15 | Fragility | — | 07 |
| W16 | Multi-tenant | — | 08, 12, 16 |
| W17 | Capital | — | 08, 15 |
| W18 | Sustainability | — | 15 |
| W19 | Checkpoint | 14 | 07 |
| W20 | Safety | 15 | 13, 14, 16 |
| W21 | Sensitivity | 12, 16 | 09, 17 |
| W22 | Synthesis | 16 | 17 |

**Note:** W6 (Streaming/Wafer-Scale) and W12 (Reasoning/Inference-Time Compute) are
advanced topics covered in textbook content but reserved as optional lab extensions.

---

## Progressive Instrument Disclosure

New visualization instruments are introduced at specific labs and reused thereafter:

| Lab | New Instrument | Description |
|-----|----------------|-------------|
| V1-01 | `LandscapeRadar` | Log-scale hardware comparison radar chart |
| V1-02 | `RooflinePlot` | Dynamic Roofline with hardware ridge point |
| V1-05 | `MemoryLedger` | Stacked bar chart for memory decomposition |
| V1-06 | `WorkloadSignatureComparator` | Side-by-side arithmetic intensity + bottleneck regime comparison |
| V1-08 | `PipelineBreakdown` | Horizontal bar showing iteration time phases |
| V1-09 | `ScalingCurve` | Diminishing returns / power-law curve |
| V1-10 | `ParetoCurve` | Accuracy–Compression Pareto frontier |
| V1-13 | `LatencyThroughputKnee` | Queuing theory L-shaped curve |
| V1-16 | `SensitivityTornado` | Tornado chart showing binding constraints |
| V2-03 | `BisectionHeatmap` | Network topology bandwidth visualization |
| V2-05 | `ParallelismMap` | 3D parallelism (TP/PP/DP) configuration space |
| V2-07 | `YoungDalyCurve` | Checkpoint interval vs. wasted work U-curve |
| V2-10 | `KVCacheHeatmap` | Memory occupancy and fragmentation |
| V2-15 | `CarbonWaterfall` | Embodied + operational carbon breakdown |
| V2-17 | `FleetSynthesisRadar` | 22-wall composite radar (capstone) |

---

## 2-Act Lab Structure (from PROTOCOL.md)

Every lab follows this exact structure:

```
┌─────────────────────────────────────────────────┐
│  PREDICTION LOCK (radio buttons or numeric)     │
│  Students commit BEFORE seeing any data         │
├─────────────────────────────────────────────────┤
│  ACT 1: CALIBRATION (12–15 min)                 │
│  • Single concept, one instrument               │
│  • Prediction reveal with gap annotation        │
│  • Structured reflection (not free text)         │
├─────────────────────────────────────────────────┤
│  PREDICTION LOCK #2                             │
├─────────────────────────────────────────────────┤
│  ACT 2: DESIGN CHALLENGE (20–25 min)            │
│  • Multi-knob optimization                      │
│  • Failure states (OOM, SLA violation, etc.)     │
│  • Two deployment contexts compared             │
│  • Design Ledger output recorded                │
└─────────────────────────────────────────────────┘
```

### Failure States (Mandatory)
Every lab must include at least one reversible failure state:
- **OOM** — Memory exceeds device budget
- **SLA Violation** — Latency exceeds target threshold
- **Thermal Throttle** — Power budget exceeded, clock drops
- **Negative ROI** — Cost exceeds economic feasibility
- **Pipeline Starvation** — GPU idle waiting for data
- **Gradient Collapse** — Loss plateaus due to vanishing gradients

---

## Design Ledger Schema

Each lab writes a JSON record to the persistent Design Ledger (`labs/core/state.py`).
Records carry forward — later labs read earlier decisions as starting conditions.

### Key Forward Dependencies

```
Lab 05 (activation_choice, max_width) ──→ Lab 06 (architecture baseline)
                                      └──→ Lab 08 (gradient stability baseline)
                                      └──→ Lab 10 (compression starting point)

Lab 06 (bottleneck_regime, intensity) ──→ Lab 08 (workload profile for training)
                                      └──→ Lab 11 (roofline operating points)

Lab 08 (optimizer, precision, MFU)    ──→ Lab 11 (roofline analysis baseline)
                                      └──→ Lab 10 (precision baseline)

Lab 10 (compression_ratio, method)    ──→ Lab 13 (serving model size)

Lab 13 (batch_size, SLA_target)       ──→ Lab 14 (monitoring baseline)

Lab 16 (full system config)           ──→ V2-01 (Vol 2 entry state)

V2-05 (parallelism config)            ──→ V2-07 (checkpoint sizing)
                                      └──→ V2-08 (fleet scheduling)

V2-07 (checkpoint strategy)           ──→ V2-09 (performance engineering)

V2-10 (serving config)                ──→ V2-12 (SLO composition)

V2-15 (sustainability config)         ──→ V2-17 (capstone synthesis)
```

---

## Solver Integration Reference

Labs import solvers from `mlsysim` and use them as analytical backends:

```python
from mlsysim import (
    # Layer A: Workload
    TransformerWorkload, ConvWorkload,
    # Layer B: Hardware
    H100, A100, JetsonOrinNX, iPhone15Pro, CortexM7,
    # Layer E: Solvers
    SingleNodeModel, DistributedModel, ServingModel,
    CompressionModel, EfficiencyModel, ScalingModel,
    TailLatencyModel, ContinuousBatchingModel,
    ReliabilityModel, SustainabilityModel, EconomicsModel,
    OrchestrationModel, TopologyModel, DataModel,
    TransformationModel, CheckpointModel,
    ResponsibleEngineeringModel, SensitivitySolver, SynthesisSolver,
    # Registry
    PerformanceProfile, Engine,
)
```

### Solver → Lab Mapping (Reverse Index)

| Solver | Labs Using It |
|--------|--------------|
| `SingleNodeModel` | V1-02, V1-05, V1-06, V1-08, V1-11, V2-02 |
| `EfficiencyModel` | V1-06, V1-07, V1-08, V1-12, V2-05, V2-09 |
| `ServingModel` | V1-13, V2-10 |
| `ContinuousBatchingModel` | V1-13, V2-10 |
| `TailLatencyModel` | V1-13, V1-14, V2-10, V2-12 |
| `CompressionModel` | V1-09, V1-10, V2-14 |
| `ScalingModel` | V1-09 |
| `DataModel` | V1-03, V1-04, V2-04 |
| `TransformationModel` | V1-03 |
| `TopologyModel` | V1-04, V2-02, V2-03, V2-04, V2-06, V2-11 |
| `DistributedModel` | V2-01, V2-03, V2-05, V2-06, V2-11 |
| `ReliabilityModel` | V2-07 |
| `CheckpointModel` | V1-14, V2-07 |
| `OrchestrationModel` | V2-08, V2-12, V2-16 |
| `EconomicsModel` | V2-08, V2-15 |
| `SustainabilityModel` | V2-15 |
| `ResponsibleEngineeringModel` | V1-15, V2-13, V2-14, V2-16 |
| `SensitivitySolver` | V1-12, V1-16, V2-09, V2-17 |
| `SynthesisSolver` | V1-16, V2-17 |

---

## Deployment Context Pairs

Every lab provides two deployment contexts (not four personas).
Students toggle between them to discover scope-dependent constraint binding.

| Lab | Context A | Context B |
|-----|-----------|-----------|
| V1-02 | H100 (Cloud, 80 GB) | CortexM7 (TinyML, 256 KB) |
| V1-05 | H100 (Training Node) | Mobile GPU (2 GB) |
| V1-06 | H100 (Compute-bound CNN) | H100 (Memory-bound LLM) |
| V1-08 | H100 (Training Node) | Laptop GPU (8 GB) |
| V1-10 | H100 (Accuracy-first) | iPhone15Pro (Latency-first) |
| V1-11 | H100 (Compute-bound) | JetsonOrinNX (Memory-bound) |
| V1-13 | H100 (Throughput SLA) | JetsonOrinNX (Latency SLA) |
| V2-05 | 8× H100 NVLink (Intra-node) | 64× H100 Ethernet (Inter-node) |
| V2-07 | 16K GPU cluster (MTBF = 4h) | 8 GPU pod (MTBF = 6 months) |
| V2-10 | H100 × 8 (KV-cache abundant) | A100 × 4 (KV-cache constrained) |
| V2-15 | Coal-grid region (CI = 800 g/kWh) | Hydro-grid region (CI = 20 g/kWh) |

---

## Capstone Labs (V1-16 and V2-17)

Both capstone labs use the `SensitivitySolver` and `SynthesisSolver` to perform
**inverse design**: given an SLA target, derive the minimum hardware specification.

- **V1-16:** Single-machine synthesis — students combine all Vol 1 optimizations
  into one coherent system configuration and defend it.
- **V2-17:** Fleet-scale synthesis — students navigate all 22 walls simultaneously,
  using the `FleetSynthesisRadar` to visualize binding constraints across domains.

The capstone Design Ledger is the student's "portfolio piece" — a complete
system specification with every number traceable to a wall equation.

---

## File Structure

```
labs/
├── ARCHITECTURE.md          ← This file
├── LABS_SPEC.md             ← Gold-standard spec (2-Act, failure states, etc.)
├── PROTOCOL.md              ← 7 invariants every lab must satisfy
├── core/
│   ├── state.py             ← DesignLedger persistence
│   ├── style.py             ← COLORS, LAB_CSS, apply_plotly_theme
│   └── components.py        ← MathPeek, MetricRow, ComparisonRow
├── plans/
│   ├── vol1/                ← 16 detailed mission plans (lab_XX_name.md)
│   └── vol2/                ← 17 detailed mission plans (lab_XX_name.md)
├── vol1/
│   └── lab_XX_name.py       ← Marimo notebook implementations
└── vol2/
    └── lab_XX_name.py       ← Marimo notebook implementations
```

---

## Implementation Priority

### Phase 1: Core Infrastructure (P0)
- [ ] Update `labs/core/` components to use `mlsysim` solver API
- [ ] Build reusable Marimo instrument components (RooflinePlot, MemoryLedger, etc.)
- [ ] Implement Design Ledger forward-passing between labs

### Phase 2: Volume 1 Labs (P1)
- [ ] V1-02 (Iron Law / Roofline) — most foundational instrument
- [ ] V1-05 (Transistor Tax) — introduces memory decomposition
- [ ] V1-08 (MFU Gap) — introduces pipeline analysis
- [ ] V1-13 (Batching Paradox) — introduces serving/queuing
- [ ] V1-16 (Capstone) — validates full Design Ledger chain

### Phase 3: Volume 2 Labs (P2)
- [ ] V2-05 (Parallelism Paradox) — introduces distributed solver
- [ ] V2-07 (Checkpoint Reckoning) — introduces reliability
- [ ] V2-10 (KV-Cache Wall) — introduces fleet serving
- [ ] V2-15 (Jevons Paradox) — introduces sustainability
- [ ] V2-17 (Fleet Capstone) — validates 22-wall synthesis

### Phase 4: Remaining Labs (P3)
- [ ] All remaining Vol 1 labs
- [ ] All remaining Vol 2 labs
- [ ] Cross-volume Design Ledger integration testing

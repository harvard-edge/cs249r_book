# MLSysIM Data Model

Twelve **zoos** (typed registries) plus support layers. Book LEGO cells and
tutorials should prefer zoos + `mlsysim.physics.*` + explicit operands.
Measurement units live in `core/units.py`, physical constants in
`physics/constants.py`, and domain values in registries; the former
`core/constants.py` shim is deleted (no-backward-compat policy).

This document is the *registry-level* view. The *runtime* view (how workloads,
hardware, infrastructure, systems, and embodied platforms feed the solver layer) is the 6-layer
model in [architecture.qmd](architecture.qmd). The zoos below are the registries
that populate Layers A–D and Layer F of that stack.

## Zoos

| Zoo | Registry | Role |
|-----|----------|------|
| Hardware | `Hardware.Cloud.*`, `Hardware.Edge.*`, `Hardware.Mobile.*`, `Hardware.Tiny.*`, `Hardware.Workstation.*`, `Hardware.Tech.*` | Chip/board/appliance specs (datasheet truth) plus technology-class tiers. **Canonical paths only** (no bare `Hardware.H100`). |
| Embodied | `Embodied.Quadruped.*`, `Embodied.Humanoid.*`, `Embodied.Manipulator.*`, `Embodied.Drone.*`, `Embodied.AMR.*`, `Embodied.Vehicle.*`, `Embodied.MobileManipulator.*`, `Embodied.ContinuousProcess.*` | Cyber-physical machine platforms and archetypes across Physical AI classes. |
| Sensors | `Sensors.Camera.*`, `Sensors.LiDAR.*`, `Sensors.IMU.*`, `Sensors.SafetyScanner.*`, `Sensors.SafetyLightCurtain.*` | Perceptual and safety-rated sensing hardware specifications. |
| Actuators | `Actuators.HarmonicDrive.*`, `Actuators.QuasiDirectDrive.*`, `Actuators.Servo.*`, `Actuators.FramelessMotor.*`, `Actuators.PandaClassJoint.*`, `Actuators.HubMotor.*`, `Actuators.SafetyRelay.*` | Joint actuators, drivetrain motors, and safety switching specifications. |
| Agents | `Agents.Coding.*`, `Agents.Deliberation.*`, `Agents.MultiAgent.*`, `Agents.Interactive.*`, `Agents.Research.*`, `Agents.Remediation.*`, `Agents.Support.*` | Agentic ML system profiles, execution harnesses, and deliberation search architectures. |
| Models | `Models.*` | Workloads and architectures (parameters, layers, FLOPs). |
| Datasets | `Datasets.*` | Data zoo: ImageNet, MNIST, CIFAR, etc. |
| Platforms | `Platforms.*` | Abstract deployment envelopes (RAM, storage, latency ranges). Replaces `Systems.Tiers`. |
| Infrastructure | `Infrastructure.Grids.*`, `Infrastructure.Datacenters.*`, `Infrastructure.Pricing.*`, `Infrastructure.Capacity.*` | Site/energy/economics layer: utility grid, facility PUE, pricing, and capacity facts. **Not** GPU fleets or network fabrics. |
| Systems | `Systems.Nodes.*`, `Systems.Racks.*`, `Systems.Fabrics.*`, `Systems.Clusters.*`, `Systems.Pods.*`, `Systems.Storage.*` | Composed physical systems and topology. Fleets live in `Systems.Clusters` (type `Fleet`); rack-level aggregates live in `Systems.Racks`. |
| Ops | `Ops.Monitoring.*`, `Ops.RuntimeOverheads.*`, `Ops.MemoryProtection.*`, `Ops.TrainingRunOverheads.*`, `Ops.Preemption.*` | Operational policies, runtime dispatch/memory overheads, thresholds, and goodput-loss profiles. |
| Scenarios | `Scenarios.*` | Executable workload + system + constraint bundles. |

## Support (not zoos)

- **`mlsysim.core.units`**: pint units, byte/bit widths, precision map.
- **`mlsysim.physics.*`**: physical constants and formulas.
- **`Literature.*`**: cited appendix scalars (MFU bands, Chinchilla, communication, batch-size anchors).
- **`ReferenceStats.*`**: non-executable sourced anchors for real-world scenario and case-study statistics.
- **`Systems.Reliability` / `Orchestration`**: MTTF, recovery, scheduling assumptions.
- **`mlsysim.engine.calibration`**: solver/engine default kwargs (not appendix tables).
- **`Infrastructure.Pricing`**: cloud, storage, labeling, fleet economics (`PricePoint.rate`).
- **Regional carbon / PUE / fleet / fabrics**: `Infrastructure.Grids`, `FacilityCooling`, `Systems.Clusters`, `Systems.Fabrics`.

## Validation invariants

MLSys·im is used to generate textbook calculations, so registry and CLI data are
validated before they reach solver equations.

- **Explicit units are required for physical quantities.** A capacity must be
  written as `80 GB` or `80 GiB`, not `80`. A model size must be bytes, a power
  value must be watts, and a latency value must be time.
- **Compute work carries its own dimension.** FLOPs (and integer-op rates such
  as TOPS) live on a dedicated `[flop]` pint dimension (2026-06-06), so
  `1 TFLOP/s` can never silently add to or convert into `900 GB/s`; pint
  itself raises `DimensionalityError`. Bytes, counts, parameters, and dollars
  remain dimensionless aliases, and MLSys·im's unit-family checks keep those
  semantically distinct at every schema boundary.
- **Precision names are closed vocabulary.** Use the precision names in
  `core.units.PRECISION_MAP`; unsupported values fail instead of silently using
  FP16 storage.
- **Distributed topology must divide exactly.** Tensor, pipeline, and expert
  parallel groups must divide total accelerators without flooring. CLI fleet
  plans must likewise specify topology that divides cleanly.

## Relationships

```mermaid
flowchart TB
  subgraph zoos [Zoos]
    Hardware
    Embodied
    Sensors
    Actuators
    Agents
    Models
    Datasets
    Platforms
    Infrastructure
    Systems
    Ops
    Scenarios
  end
  subgraph support [Support]
    units[core/units.py]
    literature[Literature.*]
    referenceStats[ReferenceStats.*]
    calibration[engine/calibration.py]
    physics[physics.*]
  end
  Hardware --> Systems
  Platforms --> Systems
  Infrastructure --> Systems
  Sensors --> Embodied
  Actuators --> Embodied
  Systems --> Agents
  Systems --> Embodied
  Models --> Agents
  Models --> physics
  Datasets --> physics
  units --> physics
  literature --> physics
  Ops --> physics
  Scenarios --> physics
  calibration --> physics
  Systems --> physics
  Models --> Scenarios
  Hardware --> Scenarios
  Systems --> Scenarios
  referenceStats --> Scenarios
```

- **Fleet ≠ datacenter:** `Systems.Clusters.*` (Fleet) references optional `Infrastructure.Datacenters.*` / grid for carbon and PUE.
- **NVL72** is `Hardware.Cloud.GB200_NVL72`, not an Infrastructure rack entry.
- **Networks/fabrics:** interconnect specs on Hardware; topology instances under `Systems.Fabrics`.
- **Scenario ≠ model or hardware:** a scenario composes existing model and system facts with local constraints. It does not redefine GPT-4, H100, or a fleet.
- **Reference statistics are not scenarios:** `ReferenceStats.MobilePower.*` and `ReferenceStats.Workloads.*` are sourced anchors for book calculations, not runnable bundles.

## Ownership Rule

When adding a number, classify the semantic object before choosing a namespace:

| Question | Home |
|----------|------|
| Is this a datasheet fact about a chip, board, appliance, NIC, or storage device? | `Hardware.*` |
| Is this a composed physical setup such as a node, rack, cluster, fabric, or storage path? | `Systems.*` |
| Is this a grid, datacenter, price, capacity, or facility-envelope fact? | `Infrastructure.*` |
| Is this an operational threshold, run-overhead profile, or monitoring policy? | `Ops.*` |
| Is this a cited scalar from a paper/table used as a literature anchor? | `Literature.*` |
| Is this a runnable workload + system + constraint bundle? | `Scenarios.*` |
| Is this a non-executable sourced world statistic or case-study anchor? | `ReferenceStats.*` |
| Is this a reusable teaching/problem setting that is not a physical system or runnable scenario? | `ReferenceStats.*` or local LEGO, depending on reuse and provenance |
| Is this a one-off knob that defines a local exercise? | Keep it local in the LEGO cell and label it as a scenario assumption. |

`Provenance` is metadata attached to entries in any of these homes. It does not
decide the namespace; the type of thing being modeled does.

## Consumer Conventions

1. Use explicit zoo paths for registry operands.
2. Use `mlsysim.physics.*` for derived quantities; registries for operands.
3. Use `Scenario.evaluate()` when a runnable workload + system + constraint bundle is needed.
4. Use `ReferenceStats.*` for non-executable anchors. Do not route reference statistics through `Scenarios.*`.
5. Never retain compatibility shims (`Hardware.H100`, `Infrastructure.Quebec`, `Systems.Cloud`, etc.); always reference canonical zoo paths directly.

See `PROVENANCE.md` and `docs/contributing.qmd` for package-side provenance rules.

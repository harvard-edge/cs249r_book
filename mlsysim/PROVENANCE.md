# Provenance in `mlsysim`

Every **Tier A** number (registry entries and public `Sourced` scalars) must carry a `Provenance` record. Citation managers and prose live outside the package; MLSysIM stores structured lineage only.

## Audit gates

| Gate | What it catches | Tool |
|------|-----------------|------|
| 1. **Registry metadata** | Hardware/model nodes without `metadata.provenance` | `audit_registries` |
| 2. **Sourced scalars** | `Literature.*`, `Ops.*`, `ReferenceStats.*`, `Infrastructure.Capacity.*`, `engine.calibration.*` without notes/URL rules | `audit_literature_sourced`, `audit_infra_capacity`, sourced registry checks |
| 3. **Appendix lineage** | Stale `defaults.*` or registry paths in assumption appendices without provenance | `audit_appendix_*` |

Run the package gate:

```bash
python -m mlsysim.tools.audit_provenance --scope all --strict
```

## Where constants live

There is **no** `mlsysim.core.defaults` module. Organize by domain:

| Namespace | Examples |
|-----------|----------|
| **`Hardware.*` / `Hardware.Tech.*`** | Chip/board datasheets plus technology-class tiers (`Memory`, `Storage`, `Op`, `EffectiveCompute`, `Movement`, `Interconnect`, `Lithography`, `Battery`) |
| **`Models.*` / `Datasets.*` / `Platforms.*`** | Workload architectures, dataset sizes, abstract deployment envelopes |
| **`Embodied.*` / `Sensors.*` / `Actuators.*` / `Agents.*`** | Physical AI platforms, sensor/actuator datasheets, safety scanners/curtains/relays, agentic harnesses |
| **`Systems.Reliability`** | Component MTTF, recovery timeouts, checkpoint bandwidth |
| **`Systems.Nodes` / `Racks` / `Fabrics` / `Clusters` / `Pods` / `Storage`** | Server/rack profiles, network tiers, fleet sizes, storage subsystems |
| **`Systems.Orchestration`** | Target utilization, queue discipline |
| **`Literature.*`** | MFU bands, Chinchilla, communication and batch-size literature anchors |
| **`Infrastructure.Grids` / `Datacenters` / `FacilityCooling` / `Pricing` / `Capacity`** | Carbon, PUE, cloud \$ anchors, build-out lead times |
| **`Ops.Monitoring` / `RuntimeOverheads` / `MemoryProtection` / `TrainingRunOverheads` / `Preemption`** | PSI thresholds, runtime dispatch/protection overheads, goodput-loss profiles |
| **`Scenarios.*`** | Executable workload + system + constraint bundles |
| **`ReferenceStats.*`** | Non-executable sourced world statistics and case-study anchors |
| **`engine.calibration`** | Default kwargs and internal heuristics for solver/engine internals |

Do **not** duplicate registry fields across namespaces (chip `unit_cost` lives on `Hardware.Cloud.*` only).

## Downstream checks

Rendered-content checks live outside the package. Those checks should consume
MLSysIM through public registries such as
`Systems.Reliability.Gpu.mttf_hours` and `Literature.Training.MfuHigh`; the
package does not scan downstream content files.

## Colocation vs. Shared Catalog

- **Colocated provenance (default for components):** Hardware devices (YAML), server nodes, and custom systems define their `metadata.provenance` inline alongside the component, or inherit directly from their accelerator (`accelerator.metadata.provenance`).
- **Shared catalog (`mlsysim.core.provenance_catalog`):** Reserved for cross-cutting bibliographic literature, industry reports, grid carbon benchmarks, and pricing datasets that are referenced by multiple models or chapters.

# Provenance in `mlsysim`

Every **Tier A** number (registry entries and public `Sourced` scalars) must carry a `Provenance` record. Citation managers and prose live outside the package; MLSysIM stores structured lineage only.

## Audit gates

| Gate | What it catches | Tool |
|------|-----------------|------|
| 1. **Registry metadata** | Hardware/model nodes without `metadata.provenance` | `audit_registries` |
| 2. **Sourced scalars** | `Literature.*`, `Infrastructure.Capacity.*`, `core.calibration.*` without notes/URL rules | `audit_literature_sourced`, `audit_infra_capacity`, `audit_calibration_sourced` |
Run the package gate:

```bash
python -m mlsysim.tools.audit_provenance --scope all --strict
```

## Where constants live

There is **no** `mlsysim.core.defaults` module. Organize by domain:

| Namespace | Examples |
|-----------|----------|
| **`Systems.Reliability`** | Component MTTF, recovery timeouts, checkpoint bandwidth |
| **`Systems.Fabrics` / `Clusters` / `Pods`** | Network tiers, fleet sizes |
| **`Systems.Orchestration`** | Target utilization, queue discipline |
| **`Literature.*`** | MFU bands, Chinchilla, scaling η, overhead budgets, ring AllReduce factor |
| **`Infrastructure.Grids` / `FacilityCooling` / `Pricing` / `Capacity`** | Carbon, PUE, cloud \$ anchors, build-out lead times |
| **`Ops.Monitoring`** | PSI thresholds, KS coefficient (MLOps chapters) |
| **`core.calibration`** | Default kwargs and internal heuristics for solver/engine internals |

Do **not** duplicate registry fields across namespaces (chip `unit_cost` lives on `Hardware.Cloud.*` only).

## Downstream checks

Rendered-content checks live outside the package. Those checks should consume
MLSysIM through public registries such as
`Systems.Reliability.Gpu.mttf_hours` and `Literature.Training.MfuHigh`; the
package does not scan downstream content files.

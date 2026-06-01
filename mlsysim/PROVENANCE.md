# Provenance in `mlsysim`

Every **Tier A** number (registry entries and narrative-facing `Sourced` scalars) must carry a `Provenance` record. External citations stay with the prose; the package stores structured lineage only.

## Audit gates

| Gate | What it catches | Tool |
|------|-----------------|------|
| 1. **Registry metadata** | Hardware/model nodes without `metadata.provenance` | `audit_registries` |
| 2. **Sourced scalars** | `Literature.*`, `Infrastructure.Capacity.*`, `core.calibration.*` without notes/URL rules | `audit_literature_sourced`, `audit_infra_capacity`, `audit_calibration_sourced` |
| 3. **Appendix lineage** | Stale `defaults.*` or registry paths in assumption appendices without provenance | `audit_appendix_*` |

Run the narrative lineage gate:

```bash
python -m mlsysim.tools.audit_provenance --scope narrative --strict
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
| **`core.calibration`** | Default kwargs and internal heuristics for `core.solver` / `core.engine` only — not cited in assumption tables |

Do **not** duplicate registry fields across namespaces (chip `unit_cost` lives on `Hardware.Cloud.*` only).

## Appendix tables

Values in `appendix_assumptions.qmd` must resolve via `Sourced.provenance` or registry `metadata.provenance`. The book cites with `@citekey`; labs import registry paths such as `Systems.Reliability.Gpu.mttf_hours` and `Literature.Training.MfuHigh`.

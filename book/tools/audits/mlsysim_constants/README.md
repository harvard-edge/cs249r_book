# MLSysIM constants audit YAMLs (historical)

These files record the **2025–2026 constants → registry migration** audit for each
chapter: which LEGO cells used flat `mlsysim.core.constants` symbols and where they
should source values after migration.

**Current source of truth:**

| Concern | Location |
|---------|----------|
| Hardware specs | `mlsysim/hardware/registry.py` → `Hardware.*` |
| Model workloads | `mlsysim/models/registry.py` → `Models.*` |
| Datasets | `mlsysim/datasets/registry.py` → `Datasets.*` |
| Fabrics / topology | `mlsysim/systems/registry.py` → `Systems.*` |
| Cited appendix scalars | `mlsysim/literature/registry.py` → `Literature.*` |
| Reliability / orchestration | `mlsysim/systems/` → `Systems.Reliability.*`, `Systems.Orchestration` |
| MLOps thresholds | `mlsysim/ops/` → `Ops.Monitoring.*` |
| Solver/engine defaults | `mlsysim/core/calibration.py` → `mlsysim.engine.calibration.*` |
| Economics | `mlsysim/infra/pricing.py` → `Infrastructure.Pricing.*` |
| Physics / units only | `mlsysim/core/constants.py`, `mlsysim/physics/` |
| Symbol → replacement map | `book/tools/audit/migrate_constants_to_registry.py` |
| Live manifest | `book/tools/audit/artifacts/registry_migration_manifest.json` |

Regenerate `target_source` fields after map changes:

```bash
python3 book/tools/audit/refresh_mlsysim_constants_yamls.py
python3 book/tools/audit/refresh_mlsysim_constants_yamls.py --finalize
```

Verify appendix assumption-table LEGO cells against live registries:

```bash
python3 book/tools/audit/generate_appendix_constants.py --verify
```

End-to-end migration build check:

```bash
python3 book/tools/audit/check_registry_migration_build.py
```

Pre-commit gates: `mlsysim-check-registry-gates`, `book-check-registry-sources`.

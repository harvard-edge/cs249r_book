---
title: "API Stability Promise"
subtitle: "Semantic versioning, stable entry points, and experimental surfaces in pre-1.0 MLSys·im."
---

> **Applies to:** `mlsysim` v0.3.x (current release: `0.3.1`)

This document defines which parts of the `mlsysim` API are stable within a minor
release series, which are experimental, and what guarantees code built on top of
the framework can rely on.

---

## Versioning Policy

`mlsysim` follows [Semantic Versioning](https://semver.org/) with one important
caveat: **we are pre-1.0.** Under semver, this means:

| Version bump | What it means |
|-------------|---------------|
| `0.3.x` -> `0.3.y` (patch) | Bug fixes, provenance updates, and additive registry entries only. No breaking API changes. Safe to upgrade. |
| `0.3.x` -> `0.4.0` (minor) | Breaking changes allowed. Read the changelog before upgrading. |
| `1.0.0` | Full stability guarantee begins. Breaking changes require a major bump. |

**In practice:** code pinned to `mlsysim ~= 0.3.0` (any `0.3.x` release) will not
break. Upgrading to `0.4.0` may require updating imports or adjusting solver call
signatures.

---

## Stable API (will not break in v0.3.x)

These interfaces are locked for the entire `0.3.x` series. Bug fixes or first-principles
physics corrections may refine return *values*, but signatures and field names
will not change.

### Core Engine

```python
from mlsysim import Engine, Hardware, Models

result = Engine.solve(
    model=Models.Language.Llama3_8B,   # Validated Workload instance from Models.*
    hardware=Hardware.Cloud.H100,      # Validated HardwareNode instance from Hardware.*
    batch_size=32,                     # int
    precision="fp16",                  # str: "fp32", "tf32", "fp16", "bf16", "fp8", "int8", "int4"
    efficiency=0.45,                   # float: 0.0-1.0
)
```

All five core parameters to `Engine.solve()` are stable (`Engine.solve()` accepts typed
`Workload` and `HardwareNode` instances rather than raw string names; use the CLI for
short-name resolution).

### Curated Registries (The 12 Zoos)

```python
from mlsysim import (
    Actuators,
    Agents,
    Datasets,
    Embodied,
    Hardware,
    Infrastructure,
    Models,
    Ops,
    Platforms,
    Scenarios,
    Sensors,
    Systems,
)

gpu = Hardware.Cloud.H100            # All current entries in 0.3.x are stable
gpu = Hardware.Cloud.A100
gpu = Hardware.Edge.JetsonOrinNX
model = Models.Language.Llama3_70B
model = Models.Language.GPT2
```

New entries may be *added* in patch releases across all 12 curated registries
(plus `Literature` and `ReferenceStats`), but existing canonical entries will not be
removed or renamed within `0.3.x`.

### Registry paths

Use **nested canonical paths** in Python:

```python
mlsysim.Hardware.Cloud.H100
mlsysim.Models.Language.Llama3_8B
mlsysim.Models.Vision.ResNet50
```

Flat aliases at the registry root (for example bare `H100` or `ResNet50` leaf names) were removed in the registry migration.
The CLI still resolves short names (`mlsysim eval Llama3_8B H100`) for convenience.

Solvers not listed in `mlsysim.__init__` (for example `CompressionModel`, `MoERoutingModel`)
import from `mlsysim.solvers`. Workload types import from `mlsysim.models.types`.

### Scenario Registry

```python
from mlsysim import ReferenceStats, Scenarios
```

`Scenarios.*` is the executable scenario registry: each entry composes an
existing `Models.*` workload, a `Hardware.*` or `Systems.*` target, and
scenario-local constraints such as latency or power budget. `ReferenceStats.*`
holds non-executable sourced anchors, such as mobile power envelopes, Waymo
data-rate ranges, and TinyML case-study measurements.

There are no compatibility aliases between these namespaces. Use
`Scenarios.SmartDoorbell` for an executable case and
`ReferenceStats.MobilePower` for sourced non-executable anchors.

### PerformanceProfile Fields

The following fields on the result object returned by `Engine.solve()` are
stable:

| Field | Type | Description |
|-------|------|-------------|
| `latency` | `pint.Quantity` | Wall-clock time for one forward pass |
| `throughput` | `pint.Quantity` | Tokens/sec or samples/sec |
| `bottleneck` | `str` | `"Compute"` or `"Memory"` |
| `mfu` | `float` | Model FLOPs Utilization (0.0-1.0) |
| `feasible` | `bool` | Whether the workload fits in memory |
| `energy` | `pint.Quantity` | Energy consumption per forward pass |

### Unit Registry

```python
from mlsysim import ureg
```

The Pint unit registry instance is stable. All quantities returned by the
engine use this registry.

---

## Experimental API (may change in v0.4.0)

These interfaces work today and are covered by the test suite, but may evolve in
minor releases (`0.4.0`). Pin to `mlsysim ~= 0.3.0` (or `== 0.3.1` for exact value
reproducibility) when building automated grading or CI pipelines against them.

### Individual Solver Classes

```python
from mlsysim.solvers import DistributedModel, ForwardModel, ServingModel
```

Solver constructors and specialized keyword arguments may evolve across minor
releases. The `Engine.solve()` facade insulates single-node callers from those changes.

Solver classes are exported from `mlsysim.solvers`, not the package root. Use
`from mlsysim.solvers import ServingModel` so solver-specific dependencies stay
explicit and the root namespace remains reserved for registries, units, and
formatting helpers.

### Training Mode Parameter

```python
Engine.solve(..., is_training=True)  # experimental on Engine.solve()
```

The `is_training` flag on `Engine.solve()` may be complemented or replaced by dedicated
`Engine.train()` / `Engine.infer()` entry points in a future minor release.

### Pipeline Composition API

The API for composing multiple solver stages into a `Pipeline` with `EngineHooks`
callbacks is functional today, though hook signatures may be refined in `0.4.0`.

### Design Space Exploration (DSE) API

The `DSE`, `SynthesisSolver`, and `SensitivitySolver` search/sweep APIs are
actively expanding; parameter names and result schemas may evolve in `0.4.0`.

### CLI Commands and Flags

`mlsysim` CLI command names, subcommands, and flags may gain options or refine output
tables across minor releases. Shell scripts that parse CLI output should pass `--json`
and pin to `~= 0.3.0`.

### Solver-Specific Result Fields

Fields on specialized result types (`DistributedResult`, `ServingResult`, etc.)
beyond the six core `PerformanceProfile` fields listed above may be extended or
reorganized in `0.4.0`.

---

## Deprecated (scheduled for removal in v0.4.0)

Interfaces listed here still work in `0.3.x` with deprecation warnings and are
scheduled for removal in the next minor release.

No public import path is deprecated in `0.3.1`. Deprecations will be listed here
and in the changelog before `0.4.0`.

---

## How to Protect Downstream Code

1. **Pin the dependency:** `mlsysim ~= 0.3.0` (allows `0.3.x` patches, blocks `0.4.0`), or `mlsysim == 0.3.1` in course autograders that assert exact numerical outputs.
2. **Use `Engine.solve()` as the primary single-node interface.** It is the most stable entry point.
3. **Import domain solvers from `mlsysim.solvers`.** Keep solver imports explicit.
4. **Run with warnings enabled** (`python3 -W default`) to catch deprecation notices early.
5. **Read the changelog** before any minor version upgrade.

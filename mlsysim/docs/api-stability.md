# API Stability Promise

> **Applies to:** mlsysim v0.1.x

This document defines which parts of the mlsysim API are stable, which are
experimental, and what guarantees you can rely on when building on top of the
simulator.

---

## Versioning Policy

mlsysim follows [Semantic Versioning](https://semver.org/) with one important
caveat: **we are pre-1.0.** Under semver, this means:

| Version bump | What it means |
|-------------|---------------|
| `0.1.x` -> `0.1.y` (patch) | Bug fixes only. No API changes. Safe to upgrade. |
| `0.1.x` -> `0.2.0` (minor) | Breaking changes allowed. Read the changelog before upgrading. |
| `1.0.0` | Full stability guarantee begins. Breaking changes require a major bump. |

**In practice:** if you pin to `mlsysim ~= 0.1.0` (any 0.1.x), your code will
not break. If you upgrade to 0.2.0, expect to update imports and possibly
adjust call signatures.

---

## Stable API (will not break in v0.1.x)

These interfaces are locked for the entire 0.1.x series. Bug fixes may change
return *values* (e.g., correcting a formula), but signatures and field names
will not change.

### Core Engine

```python
from mlsysim import Engine

result = Engine.solve(
    model=...,        # ModelSpec or registry name
    hardware=...,     # HardwareSpec or registry name
    batch_size=32,    # int
    precision="fp16", # str: "fp32", "fp16", "bf16", "int8", "int4"
    efficiency=0.45,  # float: 0.0-1.0
)
```

All five parameters to `Engine.solve()` are stable. Their names, types, and
positions will not change.

### Hardware Registry

```python
from mlsysim import Hardware

gpu = Hardware.H100_SXM        # All current entries are stable
gpu = Hardware.A100_80GB
gpu = Hardware.RTX_4090
# ... every entry shipping in 0.1.0
```

New entries may be *added* in patch releases, but existing entries will not be
removed or renamed.

### Model Registry

```python
from mlsysim import Models

model = Models.LLAMA3_70B      # All current entries are stable
model = Models.GPT2
# ... every entry shipping in 0.1.0
```

Same guarantee as Hardware: additions are allowed, removals are not.

### Scenario Registry

```python
from mlsysim import Scenarios
```

All scenarios shipping in 0.1.0 are stable. Their names, parameters, and
behavior are fixed for the 0.1.x series.

### PerformanceProfile Fields

The following fields on the result object returned by `Engine.solve()` are
stable:

| Field | Type | Description |
|-------|------|-------------|
| `latency` | `pint.Quantity` | Wall-clock time for one forward pass |
| `throughput` | `pint.Quantity` | Tokens/sec or samples/sec |
| `bottleneck` | `str` | `"compute"` or `"memory"` |
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

## Experimental API (may change in v0.2.0)

These interfaces work today but are not yet finalized. Use them freely for
exploration, but do not build production tooling against them without
pinning to an exact version.

### Individual Solver Classes

```python
from mlsysim.solvers import ForwardModel, DistributedModel, ServingModel
```

The solver class hierarchy, their constructors, and their method signatures
may change. The `Engine.solve()` facade insulates you from these changes --
prefer it over direct solver instantiation.

### Training Mode Parameter

```python
Engine.solve(..., is_training=True)  # experimental
```

The `is_training` flag will likely be replaced by separate `Engine.train()`
and `Engine.infer()` methods in v0.2.0, or by a more expressive workload
specification.

### Pipeline Composition API

The API for composing multiple solver stages into a pipeline (e.g., prefill +
decode, or TP + PP) is experimental. The abstraction is correct but the
interface is still being refined.

### Design Space Exploration (DSE) API

The search/sweep API for exploring hardware-model combinations is experimental.
Parameter names and result formats may change.

### CLI Commands and Flags

All `mlsysim` CLI command names, subcommands, and flags are experimental.
Shell scripts that call the CLI should pin to an exact version.

### Solver-Specific Result Fields

Fields on specialized result types (`DistributedResult`, `ServingResult`, etc.)
beyond the six stable `PerformanceProfile` fields listed above are experimental.
They may be renamed, reorganized, or moved to nested objects.

---

## Deprecated (will be removed in v0.2.0)

These interfaces still work in v0.1.x but emit deprecation warnings and will
be removed in the next minor release.

### `mlsysim.BaseModel` Alias

```python
# Deprecated:
from mlsysim import BaseModel

# Use instead:
from mlsysim.solvers import ForwardModel
```

The top-level `BaseModel` name was ambiguous (conflicts with Pydantic's
`BaseModel` in many codebases). It is now an alias that emits a
`DeprecationWarning`.

### Direct Solver Imports from Top-Level

```python
# Deprecated:
from mlsysim import ForwardModel, DistributedModel

# Use instead:
from mlsysim.solvers import ForwardModel, DistributedModel
```

Solver classes should be imported from `mlsysim.solvers`, not from the
`mlsysim` top-level namespace. The top-level re-exports will be removed in
v0.2.0 to keep the public API surface clean.

---

## How to Protect Your Code

1. **Pin your dependency:** `mlsysim ~= 0.1.0` (allows 0.1.x patches, blocks 0.2.0).
2. **Use `Engine.solve()` as your primary interface.** It is the most stable entry point.
3. **Avoid importing from `mlsysim.solvers` unless you need solver-specific features.** The engine facade covers most use cases.
4. **Run with warnings enabled** (`python3 -W default`) to catch deprecation notices early.
5. **Read the changelog** before any minor version upgrade.

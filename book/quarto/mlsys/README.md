# The MLSys Physics Engine: Owner's Manual

This directory (`book/quarto/mlsys/`) is the **Physics Engine** — a 5-layer hierarchy of
typed Python objects that backs every quantitative claim in the textbook. Instead of
hardcoding numbers that rot, the engine derives them from first principles with unit
safety via `pint`.

**Labs authors:** Import from this package directly. Everything you need is re-exported
from the top-level `mlsys` namespace. See the [Quick Reference](#quick-reference) below.

---

## Architecture: 5-Layer Digital Twin

The package is organized as a composition hierarchy. Each layer builds on the one below.
No layer duplicates values from a lower layer.

```
L1  constants.py     ← raw hardware specs, physical limits, economic assumptions
     ↓
L2  hardware.py      ← HardwareSpec, NetworkSpec (typed accelerator entities)
    models.py        ← ModelSpec (parameter counts, FLOP estimates)
    deployment.py    ← DeploymentTier (RAM/storage/latency per tier class)
    formulas.py      ← canonical equations (no formatting, returns Quantities)
     ↓
L3  systems.py       ← SystemArchetype = HardwareSpec + DeploymentTier + power
    clusters.py      ← NodeSpec, ClusterSpec (multi-node distributed configs)
    datacenters.py   ← GridProfile, RackProfile (regional carbon + cooling)
     ↓
L4  scenarios.py     ← ApplicationScenario (Vol1), ClusterScenario (Vol2)
     ↓
L5  __init__.py      ← public API: Hardware, Models, Systems, Clusters,
                        Datacenters, Scenarios, Applications, Fleet, …
```

**Rule:** Always import from L5 (`from mlsys import …`). Never import directly from
internal modules in QMD cells.

---

## Module Reference

### L1 — `constants.py`

The single source of truth for every number. Never define derived values here.

```python
from mlsys.constants import ureg, Q_                # pint unit registry + Quantity
from mlsys.constants import H100_FLOPS_FP16_TENSOR  # 1979 TFLOPs/s
from mlsys.constants import CARBON_QUEBEC_GCO2_KWH  # 20 gCO₂/kWh
```

Key constant groups:
- Hardware: `H100_*`, `A100_*`, `B200_*`, `V100_*`, `T4_*`, `TPUV4_*`
- Memory: `BYTES_FP32`, `BYTES_FP16`, `BYTES_BF16`, `BYTES_FP8`, `BYTES_INT8`
- Models: `GPT3_PARAMS`, `GPT4_PARAMS`, `LLAMA2_70B_PARAMS`, `RESNET50_PARAMS`
- Carbon: `CARBON_QUEBEC_GCO2_KWH`, `CARBON_POLAND_GCO2_KWH`, `CARBON_US_AVG_GCO2_KWH`
- Cluster: `GPU_MTTF_HOURS`, `MFU_TRAINING_HIGH`, `SCALING_EFF_8192GPU`

---

### L2 — `hardware.py`

Typed accelerator and network entities. Fields are `pint` Quantities; all validated on
construction.

```python
from mlsys import Hardware

h = Hardware.H100
h.peak_flops          # → Quantity[TFLOP/s]
h.memory_bw           # → Quantity[TB/s]
h.memory_capacity     # → Quantity[GiB]
h.tdp                 # → Quantity[W]
h.ridge_point()       # → Quantity[FLOP/byte]  (Roofline ridge point)
```

**Named accelerators:**

| Name | Class | Object |
|------|-------|--------|
| `Hardware.H100` | Cloud | NVIDIA H100 SXM (2022) |
| `Hardware.A100` | Cloud | NVIDIA A100 SXM (2020) |
| `Hardware.B200` | Cloud | NVIDIA B200 (2024) |
| `Hardware.V100` | Cloud | NVIDIA V100 (2017) |
| `Hardware.TPUv4` | Cloud | Google TPU v4 (2021) |
| `Hardware.Cloud.T4` | Cloud | NVIDIA T4 inference (2018) |
| `Hardware.Edge.JetsonOrinNX` | Edge | NVIDIA Jetson Orin NX (2023) |
| `Hardware.Edge.Generic_Phone` | Mobile | Flagship smartphone (2024) |
| `Hardware.Tiny.ESP32` | TinyML | ESP32-CAM (2019) |
| `Hardware.Tiny.Generic_MCU` | TinyML | ARM Cortex-M7 (2020) |

---

### L2 — `models.py`

Typed model entities with parameter counts, FLOP budgets, and precision footprints.

```python
from mlsys import Models

m = Models.Language.Llama2_70B
m.params              # → Quantity[count]
m.flops_per_token     # → Quantity[FLOP]
m.memory_fp16         # → Quantity[byte]  (inference weight footprint)
```

**Named models:**

| Object | Family | Description |
|--------|--------|-------------|
| `Models.GPT3` | LLM | GPT-3 (175B params) |
| `Models.GPT4` | LLM | GPT-4 estimate (1.7T params) |
| `Models.Language.Llama2_70B` | LLM | Llama-2 70B |
| `Models.Language.Llama2_7B` | LLM | Llama-2 7B |
| `Models.Vision.YOLOv8_Nano` | CV | YOLOv8-Nano real-time detector |
| `Models.Vision.ResNet50` | CV | ResNet-50 classifier |
| `Models.Tiny.WakeVision` | TinyML | Person detection (WakeVision) |
| `Models.Tiny.DS_CNN` | TinyML | Keyword spotting (DS-CNN) |

---

### L2 — `formulas.py`

Pure calculation functions. Return `pint` Quantities (no display formatting).

```python
from mlsys.formulas import (
    dTime,                    # training time from FLOPs
    calc_training_time_days,  # convenience wrapper → float days
    calc_bottleneck,          # Roofline analysis dict
    model_memory,             # parameter × bytes → MB/GB
    calc_ring_allreduce_time, # communication overlap estimate
    calc_mtbf_cluster,        # MTBF_cluster = MTBF_node / N
    calc_young_daly_interval, # optimal checkpoint interval
    calc_effective_flops,     # peak × MFU × η × goodput
    calc_failure_probability, # 1 - e^(-T/MTBF)
    calc_kv_cache_size,       # KV cache for autoregressive inference
    calc_checkpoint_size,     # mixed-precision Adam checkpoint size
    calc_pipeline_bubble,     # GPipe/1F1B bubble fraction
    calc_amdahls_speedup,     # Amdahl speedup
    calc_fleet_tco,           # Total Cost of Ownership
    calc_monthly_egress_cost, # cloud egress cost
    calc_network_latency_ms,  # RTT from distance_km
)
```

---

### L3 — `systems.py`

`SystemArchetype` binds a hardware accelerator to its deployment environment
(tier, network bandwidth, power budget). Used for Vol1 single-machine scenarios.

```python
from mlsys import Systems, Archetypes

s = Systems.Edge                  # Jetson Orin NX archetype
s.hardware                        # → HardwareSpec
s.tier                            # → DeploymentTier
s.peak_flops                      # → Quantity (delegates to hardware)
s.memory_bw                       # → Quantity
s.network_bw                      # → Quantity[Gbps]
s.power_budget                    # → Quantity[W]
```

**Named system archetypes:**

| Object | Tier | Hardware |
|--------|------|---------|
| `Systems.Cloud` / `Archetypes.Cloud_H100` | Cloud | H100 SXM |
| `Archetypes.Cloud_A100` | Cloud | A100 SXM |
| `Systems.Edge` / `Archetypes.Edge_Robotics` | Edge | Jetson Orin NX |
| `Archetypes.Edge_Server` | Edge | Generic edge server |
| `Systems.Mobile` / `Archetypes.Mobile_Phone` | Mobile | Flagship smartphone |
| `Systems.Tiny` / `Archetypes.TinyML_MCU` | TinyML | ESP32-CAM |
| `Archetypes.TinyML_M7` | TinyML | Cortex-M7 MCU |

---

### L3 — `clusters.py`

Multi-node distributed cluster entities for Vol2. Derives FLOP counts, memory, MTBF,
and effective throughput by composing from `HardwareSpec` primitives.

```python
from mlsys import Clusters, Nodes, ClusterSpec, NodeSpec

c = Clusters.Frontier_8K
c.total_gpus              # → int   (8 192)
c.peak_flops_pflops       # → float (PFLOPs/s)
c.effective_flops_pflops  # → float (after MFU × scaling × goodput)
c.cluster_mtbf_days       # → float (expected days between any node failure)
c.total_memory_capacity   # → Quantity[byte]
c.aggregate_memory_bw     # → Quantity[byte/s]
c.n_nodes                 # → int
c.node                    # → NodeSpec
c.fabric                  # → NetworkSpec
```

**Named clusters:**

| Object | GPUs | Use case |
|--------|------|----------|
| `Clusters.Research_256` | 256 | University / mid-tier research |
| `Clusters.Production_2K` | 2 048 | Hyperscaler fine-tuning |
| `Clusters.Frontier_8K` | 8 192 | Llama-scale pre-training |
| `Clusters.Mega_100K` | 100 000 | GPT-4-scale frontier training |

**Named nodes:**

| Object | Config |
|--------|--------|
| `Nodes.DGX_H100` | 8× H100 + NVLink 4.0 + 8× ConnectX-7 |
| `Nodes.DGX_A100` | 8× A100 + NVLink 3.0 |
| `Nodes.DGX_B200` | 8× B200 + NVLink 5.0 |

---

### L3 — `datacenters.py`

Regional grid and rack profiles for carbon-aware scheduling chapters (Vol2).

```python
from mlsys import Datacenters

g = Datacenters.Quebec
g.carbon_intensity_g_kwh       # → float  (20 gCO₂/kWh)
g.carbon_intensity_kg_kwh      # → float  (0.020 kg CO₂/kWh)
g.pue                          # → float  (1.06)
g.wue                          # → float
g.carbon_kg(energy_kwh)        # → float  (includes PUE overhead)
g.carbon_tonnes(energy_kwh)    # → float
g.intensity_ratio_vs(Datacenters.Poland)  # → float  (≈61× cleaner)
```

**Named grids:**

| Object | Region | gCO₂/kWh | PUE | Source |
|--------|--------|----------|-----|--------|
| `Datacenters.Quebec` | Canada | 20 | 1.06 | Hydro |
| `Datacenters.Norway` | Norway | 26 | 1.06 | Hydro |
| `Datacenters.France` | France | 85 | 1.12 | Nuclear |
| `Datacenters.EU_Avg` | EU | 295 | 1.20 | Mixed |
| `Datacenters.US_Avg` | USA | 369 | 1.12 | Mixed |
| `Datacenters.Poland` | Poland | 773 | 1.58 | Coal |

**Named racks:**

| Object | Power | Cooling |
|--------|-------|---------|
| `Datacenters.Racks.Traditional` | 10 kW | Air |
| `Datacenters.Racks.AI_Standard` | 80 kW | Liquid |
| `Datacenters.Racks.AI_HighDensity` | 120 kW | Immersion |

---

### L4 — `scenarios.py`

Fully described deployments — a model bound to a system with a named mission. The
primary LEGO block for chapter prose.

Both `ApplicationScenario` (Vol1) and `ClusterScenario` (Vol2) expose the same
`.hardware`, `.mission_goal`, `.critical_constraint` interface, so LEGO blocks work
identically across volumes.

```python
from mlsys import Applications, Fleet

# Vol1: single-machine scenario
s = Applications.Doorbell
s.name                  # "Smart Doorbell (Wake Vision)"
s.hardware              # → HardwareSpec (ESP32-CAM)
s.tier                  # → DeploymentTier (TinyML)
s.latency_slo           # → Quantity[ms]  (200 ms)
s.accuracy_target       # → float  (0.85)
s.mission_goal          # → str
s.critical_constraint   # → str

# Vol2: distributed scenario
f = Fleet.Training
f.name                  # "Large-Scale Pre-Training (8 192 GPUs)"
f.hardware              # → HardwareSpec (H100 — lead accelerator)
f.cluster               # → ClusterSpec
f.total_gpus            # → int  (8 192)
f.latency_slo           # → Quantity[ms] (if applicable)
```

**Vol1 single-machine scenarios (`Applications.*`):**

| Alias | Scenario | Hardware | SLO |
|-------|----------|----------|-----|
| `Applications.Frontier` | Frontier Model Training | H100 | — |
| `Applications.AutoDrive` | Autonomous Vehicle Perception | Jetson Orin NX | 10 ms |
| `Applications.Assistant` | On-Device Language Assistant | Smartphone | 50 ms |
| `Applications.Doorbell` | Smart Doorbell (Wake Vision) | ESP32-CAM | 200 ms |
| `Applications.KWS` | Keyword Spotting (Always-On) | Cortex-M7 | 100 ms |

**Vol2 distributed scenarios (`Fleet.*`):**

| Alias | Scenario | Cluster | GPUs |
|-------|----------|---------|------|
| `Fleet.Research` | Research Cluster Training | Research_256 | 256 |
| `Fleet.Training` | Large-Scale Pre-Training | Frontier_8K | 8 192 |
| `Fleet.Frontier` | Frontier Model Training | Mega_100K | 100 000 |
| `Fleet.Inference` | Distributed LLM Inference Fleet | Production_2K | 2 048 |

---

### Support modules

| Module | Purpose |
|--------|---------|
| `formatting.py` | Display helpers: `fmt`, `sci`, `md_math`, `md_frac`, `check` |
| `viz.py` | Global matplotlib theme (import before any plot) |
| `engine.py` | `Engine` — scenario-level compute helpers |
| `registry.py` | `start_chapter` / `end_chapter` execution tape |

---

## Quick Reference

For labs notebooks, copy this import block:

```python
# ── mlsys standard import for labs ────────────────────────────────────────────
from mlsys import (
    Hardware,       # L2: accelerator specs (Hardware.H100, Hardware.ESP32, …)
    Models,         # L2: model specs (Models.Language.Llama2_70B, …)
    Systems,        # L3: single-node system archetypes
    Archetypes,     # L3: named system archetypes (full list)
    Clusters,       # L3: multi-node cluster configs (Clusters.Frontier_8K, …)
    Nodes,          # L3: named DGX nodes
    Datacenters,    # L3: regional grid profiles (Datacenters.Quebec, …)
    Tiers,          # L2: deployment tier objects (Tiers.Cloud, Tiers.Tiny, …)
    Applications,   # L4: Vol1 named deployment scenarios
    Fleet,          # L4: Vol2 distributed workload scenarios
    ureg, Q_,       # pint unit registry
)
from mlsys.formulas import (
    model_memory, calc_bottleneck, dTime,
    calc_training_time_days, calc_effective_flops,
    calc_failure_probability, calc_young_daly_interval,
)
from mlsys.formatting import fmt, sci, md_math
```

---

## Calculation Conventions (PIPO+)

These rules make calculations auditable and keep prose consistent across HTML/EPUB/PDF.

### 1) Compute anything derived
If a value is the result of combining inputs (ratio, conversion, multiplication, rounding), compute it in a Python block and reference the display string in the text or table.

### 2) Leave raw facts as literals
Dates, names, event years, and citations stay in prose. Do not over-encode non-derived facts.

### 3) Every table cell must be either:
- A computed display variable (preferred), or
- A literal fact (explicitly non-computed)

### 4) Tie calculations to their targets
Add a short comment in the calc block:
```python
# Used in: Table "Latency Numbers" (rows: compute + network)
```
This creates a traceable link from computation → narrative.

### 5) Use a consistent block structure (PIPO+)
**PIPO+** = **Purpose → Input → Process → Output**, with optional **Context** and **Checks** for clarity and auditability.
````markdown
```{python}
#| label: <calc-id>
#| echo: false

# =============================================================================
# PURPOSE
# =============================================================================
# Purpose: One-line description of the calculation.
# Used in: Section/Table/Figure reference.
# Context: (Optional) One sentence on why this matters.

# =============================================================================
# INPUT (SOURCES)
# =============================================================================
# from mlsys import Applications, Clusters, Datacenters
# sc = Applications.Doorbell

# =============================================================================
# INPUT (ASSUMPTIONS)
# =============================================================================
# a_value = 1.6  # x/year

# =============================================================================
# PROCESS
# =============================================================================
# ratio_value = a_value / b_value

# =============================================================================
# CHECKS
# =============================================================================
# assert a_value > 0
# assert 0 < ratio_value < 10

# =============================================================================
# OUTPUT
# =============================================================================
# ratio_str = f"{ratio_value:.2f}"
```
````

**Formatting note:** The separator line should be followed immediately by the header (no blank line between `# =============================================================================` and the next header).

### 6) Use display strings in text blocks
Inline prose should use `` `{python} <name>_str` `` or `` `{python} <name>_math` ``.
Do not hardcode derived numbers in prose.
Prefer `mlsys.formatting` helpers (`fmt`, `display_value`, `md_math`, `md_frac`) over inline f-strings.

### 6a) Use `Markdown()` for LaTeX output
If the output contains LaTeX (fractions, `\times`, exponents), return a `Markdown()` object (via `md_math`, `md_frac`, `md_sci`) and reference it inline:
```python
from mlsys.formatting import md_math
ratio_math = md_math(r"\frac{4.1 \times 10^{9}}{3.1 \times 10^{14}}")
```
Then in prose:
```
`{python} ratio_math`
```

### 7) Name by meaning, not formatting
Use explicit raw values (`latency_ms_value`) alongside display strings (`latency_ms_str`) for auditability.

### 7a) Make representation explicit in Markdown
Only reference explicit representation variables in prose/tables:
- `*_value` — raw numeric (keep for auditability; avoid inline use)
- `*_str` — plain text / formatted numbers
- `*_math` or `*_frac` — LaTeX via `Markdown()`
This makes the output type obvious at a glance.

### 8) Prefer one block per narrative chunk
Avoid mixing unrelated calculations into a single Python block.

### 9) Figures/Plots (PIPO + SETUP/DATA/PLOT)
Figure blocks can use `SETUP/DATA/PLOT`, but should still start with a PURPOSE header.
````markdown
```{python}
#| label: fig-<name>
#| echo: false
#| fig-cap: "<caption>"
#| fig-alt: "<alt>"

# =============================================================================
# PURPOSE
# =============================================================================
# Purpose: One-line description of the figure.
# Used in: Figure "<caption short name>".

# =============================================================================
# SETUP
# =============================================================================
# imports, viz setup

# =============================================================================
# DATA
# =============================================================================
# data tables or arrays

# =============================================================================
# PLOT
# =============================================================================
# plotting code
```
````

---

## Checklist

- Use the PIPO+ template exactly (Purpose, Input, Process, Output; optional Context/Checks).
- Keep one calc block per narrative chunk.
- Name outputs with explicit suffixes: `*_value`, `*_str`, `*_math`.
- Use `mlsys.formatting` helpers instead of inline f-strings.
- Inline prose should reference only `*_str` or `*_math`.

---

## Where should the code live?

**Default (recommended):** Keep calculations inline in the QMD for context.
Use `mlsys/` helpers for shared math, units, and formatting.

**Why:** It keeps prose clean, enables reuse, and makes tests easy.

### A) Chapter calculator modules (for shared calculations)
Create a chapter module:
```python
# book/quarto/mlsys/ch_introduction.py
def calc_intro_setup():
    return {"google_search_b": "8.5"}
```

In the QMD:
```python
from mlsys.ch_introduction import calc_intro_setup
_intro = calc_intro_setup()
google_search_b = _intro["google_search_b"]
```

### B) Inline QMD blocks (for local one-offs)
Inline blocks are fine for **small, local** calculations that are not reused and don't warrant a chapter module.
If a value appears in more than one place, move it into `mlsys/`.

---

## Naming conventions

- **Files:** `ch_<chapter>.py` (e.g., `ch_training.py`)
- **Functions:** `calc_<section>()` (e.g., `calc_gpt3_training()`)
- **Variables:** raw values use `_value` suffix; display strings use `_str`; LaTeX use `_math`
- **Provenance:** Add a comment describing where values are used:
  ```python
  # Used in: Table "Latency Numbers" (rows: compute + network)
  ```

---

## How to extend the engine

### 1. Adding a new accelerator
Open [constants.py](constants.py) and add the raw constants:
```python
H200_FLOPS_FP16_TENSOR = 1979 * TFLOPs / second  # same die as H100
```
Then open [hardware.py](hardware.py) and add a `HardwareSpec`:
```python
H200 = HardwareSpec("NVIDIA H200", 2024, H200_MEM_BW, H200_FLOPS_FP16_TENSOR, ...)
```

### 2. Adding a new scenario
Open [scenarios.py](scenarios.py) and instantiate an `ApplicationScenario` or
`ClusterScenario` from existing `Systems`/`Clusters` and `Models` objects.

### 3. Debugging narrative logic
If the text says "Edge is Cheaper" but the math says "Cloud is Cheaper", the
narrative invariant tests will fail.
1. Run tests: `pytest book/tests/test_narrative_invariants.py`
2. If it fails, **rewrite the prose**, not the test. The test protects the truth.

---

## Dependencies

- `pint` — unit safety and dimensional analysis
- `matplotlib` — charts (import `mlsys.viz` before plotting)
- `pandas` — data tables

## Common pitfalls

- **Dimensionality errors:** If `pint` raises "Cannot convert second to meter", you
  divided Distance by Time incorrectly.
- **Format strings:** Never use `f"{constant}"` on a Quantity directly. Use
  `f"{constant.magnitude}"` or `fmt(constant)`.
- **Mixed raw/Quantity:** `calc_failure_probability` requires both arguments to be
  the same type (both Quantities or both raw numbers) — mixed types raise `TypeError`.

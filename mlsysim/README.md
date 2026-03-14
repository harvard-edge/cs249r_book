# 🚀 mlsysim: The ML Systems Modeling Platform

`mlsysim` is the high-performance, physics-grounded analytical simulator powering the **Machine Learning Systems** textbook ecosystem. It provides a unified "Single Source of Truth" (SSoT) for modeling systems from sub-watt microcontrollers to exaflop-scale global fleets.

---

## 🏗 The 5-Layer Analytical Stack
`mlsysim` implements a "Progressive Lowering" architecture, separating high-level workloads from the physical infrastructure that executes them.

### Layer A: Workload Representation (`mlsysim.models`)
High-level model definitions (`TransformerWorkload`, `CNNWorkload`).
*   **Math:** FLOPs, parameter counts, and arithmetic intensity.
*   **Key Models:** `Models.Llama3_70B`, `Models.GPT3`, `Models.ResNet50`.

### Layer B: Hardware Registry (`mlsysim.hardware`)
Precise, concrete specifications for real-world silicon.
*   **Cloud:** `Hardware.H100`, `Hardware.H200`, `Hardware.B200`, `Hardware.NVL72`, `Hardware.MI300X`, `Hardware.TPUv5p`.
*   **Workstation:** `Hardware.DGXSpark`, `Hardware.MacBook`.
*   **Mobile:** `Hardware.iPhone`, `Hardware.Snapdragon`.
*   **Edge/Tiny:** `Hardware.Jetson`, `Hardware.ESP32`, `Hardware.Tiny.HimaxWE1`.

### Layer C: Infrastructure & Environment (`mlsysim.infra`)
Regional grid profiles and datacenter sustainability.
*   **Math:** PUE, Carbon Intensity (gCO2/kWh), WUE.
*   **Grids:** `Infra.Quebec`, `Infra.Poland`, `Infra.US_Avg`.

### Layer D: Systems & Topology (`mlsysim.systems`)
Fleet configurations, network fabrics, and narrative scenarios.
*   **Scenarios:** `Applications.Doorbell`, `Applications.AutoDrive`, `Applications.Frontier`.

### Layer E: Execution & Resolvers (`mlsysim.core.solver`)
The 3-tier mathematical engine that resolves the hierarchy of constraints.
*   **Tier 1: Models (`SingleNodeModel`, `DistributedModel`)**: Forward evaluation of physics ($Y = f(X)$).
*   **Tier 2: Solvers (`SynthesisSolver`, `SensitivitySolver`)**: Algebraic inversion and diagnostics ($X = f^{-1}(Y)$).
*   **Tier 3: Optimizers (`ParallelismOptimizer`, `BatchingOptimizer`)**: Design space search ($\max f(X)$).

---

## 🚀 Quick Usage: The Agent-Ready CLI

`mlsysim` is designed as an **Infrastructure-as-Code (IaC) Compiler** for ML systems. It features a stunning terminal UI for humans and a strict JSON API for CI/CD pipelines and AI agents.

### 1. Explore the Registry (The Zoo)
Discover built-in hardware, models, and infrastructure without reading source code:
```bash
mlsysim zoo hardware
mlsysim zoo models
```

### 2. Quick Evaluation (CLI Flags)
Evaluate the physics of a workload on a specific hardware node instantly:
```bash
mlsysim eval Llama3_8B H100 --batch-size 32
```

### 3. Deep Simulation (Infrastructure as Code)
Define your entire cluster and SLA constraints in a declarative `mlsys.yaml` file:

```yaml
# example_cluster.yaml
version: "1.0"
workload:
  name: "Llama3_70B"
  batch_size: 4096
hardware:
  name: "H100"
  nodes: 64
ops:
  region: "Quebec"
  duration_days: 14.0
constraints:
  assert:
    - metric: "performance.latency"
      max: 50.0
```

Then compile and evaluate the 3-lens scorecard (Feasibility, Performance, Macro):
```bash
mlsysim eval example_cluster.yaml
```

### 4. CI/CD & Agentic Automation
Every command supports strict, schema-validated JSON output. If an `assert` constraint is violated, the CLI returns a semantic `Exit Code 3`.
```bash
# Export the JSON Schema for your IDE or AI Agent
mlsysim schema > schema.json

# Run an evaluation in a CI pipeline
tco=$(mlsysim --output json eval example_cluster.yaml | jq .macro.metrics.tco_usd)
```

### 5. Design Space Search (Optimizers)
Use the Tier 3 Engineering Engine to automatically find the optimal configuration:
```bash
# Find the optimal (TP, PP, DP) split
mlsysim optimize parallelism example_cluster.yaml

# Find the cheapest, greenest datacenter location
mlsysim optimize placement example_cluster.yaml --carbon-tax 150
```

---

## 🛡 Stability & Integrity
Because this core powers a printed textbook, we enforce strict **Invariant Verification**. Every physical constant is traceable to a primary source (datasheet or paper), and dimensional integrity is enforced via `pint`.

## 🛠 Installation
```bash
pip install -e .
```

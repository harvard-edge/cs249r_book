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
*   **Cloud:** `Hardware.H100`, `Hardware.H200`, `Hardware.B200`, `Hardware.MI300X`, `Hardware.TPUv5p`.
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

## 🚀 Quick Usage: The CLI & System Evaluation

You can evaluate hardware and workloads directly from your terminal using the built-in CLI:

```bash
python3 -m mlsysim evaluate --model Llama3_8B --hardware H100 --batch-size 32
```

In Python, the primary way to use `mlsysim` is through the **Hierarchy of Constraints**:

```python
import mlsysim

# 1. Pick a Lighthouse Scenario
scenario = mlsysim.Applications.Doorbell

# 2. Run a Multi-Level Evaluation
evaluation = scenario.evaluate()

# 3. View the Scorecard
print(evaluation.scorecard())
```

**Example Scorecard Output:**
```text
=== SYSTEM EVALUATION: Smart Doorbell ===
Level 1: Feasibility -> [PASS]
   Model fits in memory (0.5 MB / 0.5 MB)
Level 2: Performance -> [PASS]
   Latency: 105.00 ms (Target: 200 ms)
Level 3: Macro/Economics -> [PASS]
   Annual Carbon: 5.1 kg | TCO: $31,501
```

---

## 🛡 Stability & Integrity
Because this core powers a printed textbook, we enforce strict **Invariant Verification**. Every physical constant is traceable to a primary source (datasheet or paper), and dimensional integrity is enforced via `pint`.

## 🛠 Installation
```bash
pip install -e .
```

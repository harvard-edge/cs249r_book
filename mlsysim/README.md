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
*   **Cloud:** `Hardware.H100`, `Hardware.H200`, `Hardware.MI300X`, `Hardware.TPUv5p`.
*   **Mobile/Workstation:** `Hardware.iPhone`, `Hardware.Snapdragon`, `Hardware.MacBookM3Max`.
*   **Edge/Tiny:** `Hardware.Jetson`, `Hardware.TeslaFSD`, `Hardware.ESP32`, `Hardware.Arduino`.

### Layer C: Infrastructure & Environment (`mlsysim.infra`)
Regional grid profiles and datacenter sustainability.
*   **Math:** PUE, Carbon Intensity (gCO2/kWh), WUE.
*   **Grids:** `Infra.Quebec`, `Infra.Poland`, `Infra.US_Avg`.

### Layer D: Systems & Topology (`mlsysim.systems`)
Fleet configurations, network fabrics, and narrative scenarios.
*   **Scenarios:** `Applications.Doorbell`, `Applications.AutoDrive`, `Applications.Frontier`.

### Layer E: Execution & Solvers (`mlsysim.core.solver`)
The physics-grounded solvers that resolve the hierarchy of constraints.
*   **`SingleNodeSolver`**: Roofline and Iron Law performance.
*   **`ServingSolver`**: LLM Pre-fill vs. Decoding and KV-Cache growth.
*   **`DistributedSolver`**: 3D Parallelism (TP/PP/DP) and Network Oversubscription.
*   **`SustainabilitySolver`**: Carbon Footprint and Water usage.

---

## 🚀 Quick Usage: The System Evaluation

The primary way to use `mlsysim` is through the **Hierarchy of Constraints**.

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

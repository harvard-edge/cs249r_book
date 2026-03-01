# 🚀 mlsysim
### The ML Systems Infrastructure & Modeling Platform

`mlsysim` is the high-performance, physics-grounded analytical engine powering the **Machine Learning Systems** textbook ecosystem (`mlsysbook.ai`). It provides a unified "Single Source of Truth" (SSoT) for modeling systems from sub-watt microcontrollers to exaflop-scale global fleets.

---

## 🏗 One Core, Multiple Worlds
`mlsysim` is designed to be the shared brain for every product in the ecosystem:
*   📚 **The Book**: Powers the precise "Napkin Math" and invariant checks in every chapter.
*   🧪 **The Labs**: Drives the interactive "Persona-based" simulations and trade-off explorers.
*   🛠 **The Kits**: Interfaces with physical hardware kits to bridge theory and measurement.
*   🔥 **Tito (TinyTorch)**: Provides the analytical baseline for custom framework profiling.

---

## 📐 Architecture (The 3-Layer Stack)
The package is organized into three professional domains:

1.  **`mlsysim.core` (The Physics & Definitions)**: 
    *   **Constants**: Immutable physical truths (H100 specs, Grid carbon intensity).
    *   **Formulas**: The "Iron Laws" of ML systems (Stateless math via `pint`).
    *   **Scenarios**: Definitive workloads like **Doorbell**, **AV**, and **GPT-4**.
    *   **Engine**: The analytical solver for single-node performance (Latency, MFU, Energy).
2.  **`mlsysim.sim` (The Analytical Simulator)**:
    *   **Personas**: Scale multipliers and constraints (Cloud Titan, Tiny Pioneer).
    *   **Simulations**: Domain logic (Sustainability, Reliability) that processes choices into ledgers.
    *   **Ledger**: The universal multi-dimensional scorecard.
3.  **`mlsysim.viz` (The Presentation)**:
    *   Presentation logic: LaTeX formatting, Markdown helpers, and professional plotting.

---

## 🚀 Getting Started

### Installation (Developer Mode)
To use `mlsysim` across the monorepo (Labs, Book, etc.), perform an editable install from the root:
```bash
pip install -e .
```

### Quick Usage
```python
import mlsysim
from mlsysim.sim import ResourceSimulation

# 1. Setup Scenario & Persona
scenario = mlsysim.Applications.Doorbell
persona = mlsysim.sim.Personas.TinyPioneer

# 2. Run an analytical simulation
sim = ResourceSimulation(scenario, persona)
ledger = sim.evaluate({"region": "Quebec", "duration_days": 365})

# 3. Inspect the results
print(f"Annual Carbon: {ledger.sustainability.carbon_kg:,.0f} kg CO2e")
```

---

## 🛡 Stability & Integrity
Because this core powers a printed textbook, we enforce strict **Invariant Verification**: All math cells in the book use `check()` guards. If a core formula change breaks the book's narrative, the build system will fail immediately.

---

## 👩‍💻 For Contributors & TAs
We built `mlsysim` to be extensible. To add a new domain lab, simply subclass `BaseSimulation` in the `sim` sub-package. 

See the [**Developer Documentation**](docs/index.qmd) for full API details and the "Wicked Sick" guide to building custom systems models.

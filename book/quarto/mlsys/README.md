# MLSys Developer Guide
## The Physics-Based ML Systems Simulator Package

This package is the "Single Source of Truth" for the **Machine Learning Systems** textbook. It provides a hierarchical analytical engine for modeling, simulating, and optimizing ML systems from microcontrollers to global fleets.

### 🏗 Architecture (The 5-Layer Stack)

The package is organized hierarchically. Each layer depends only on the ones below it.

1. **`mlsysim.constants` & `mlsysim.formulas`**: The "Iron Laws" and physical truth (H100 specs, Grid carbon intensity).
2. **`mlsysim.ledger`**: The "Universal Scorecard." Defines the `SystemLedger` data structure used for all results.
3. **`mlsysim.engine`**: The "Solver." Computes static performance (Latency, MFU, Energy) for a single node.
4. **`mlsysim.personas`**: The "Scale Layer." Defines personas (Cloud Titan, Tiny Pioneer) and their scale multipliers.
5. **`mlsysim.simulations`**: The "Decision Engine." Implements domain logic (Sustainability, Reliability) that processes choices into ledgers.

---

### 🚀 For TAs & Researchers: How to Extend

#### 1. Creating a Custom Simulation
To build a new interactive lab, subclass `BaseSimulation` and implement the `evaluate` method.

```python
from mlsysim import BaseSimulation, SystemLedger

class MyCustomSimulation(BaseSimulation):
    def evaluate(self, choice: dict) -> SystemLedger:
        # 1. Get baseline node performance
        system = self._get_system_archetype()
        perf = Engine.solve(self.scenario.model, system)

        # 2. Scale by persona
        scale = self.persona.scale_factor

        # 3. Add your custom physics logic here...

        # 4. Return a SystemLedger
        return SystemLedger(...)
```

#### 2. Adding a New Hardware "Twin"
Add a new entry to `mlsysim/hardware.py` using the `HardwareSpec` dataclass.

---

### 🧪 Integration Testing
You can run the analytical engine directly in any Python environment:

```python
from mlsysim import ResourceSimulation, Applications, Personas

sim = ResourceSimulation(Applications.Doorbell, Personas.TinyPioneer)
ledger = sim.evaluate({"region": "Quebec"})

print(f"Carbon Footprint: {ledger.sustainability.carbon_kg} kg")
```

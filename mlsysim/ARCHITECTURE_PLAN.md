# mlsysim: The Architecture & Development Plan

## Vision: The MIPS/SPIM for Machine Learning Systems
`mlsysim` is a first-order analytical simulator for AI infrastructure. Just as Hennessy and Patterson used the MIPS architecture and SPIM simulator to teach the physics of instruction pipelining, `mlsysim` teaches the physics of tensor movement, memory hierarchies, and distributed fleet dynamics.

It abstracts away the friction of PyTorch/CUDA and the extreme slowness of cycle-accurate simulators (like `gem5`), focusing entirely on macroscopic physical and economic limits. It is designed to be the de facto pedagogical tool for universities and a rapid prototyping engine for researchers.

---

## 1. Related Work & Our Unique Value Add

To succeed academically and practically, `mlsysim` must clearly differentiate itself from existing, heavy-duty simulators. If we are asked "Why not just use X?", our answer must be airtight.

**The Existing Landscape:**
*   **Intra-Accelerator / Cycle-Accurate (e.g., gem5, Accelergy, Timeloop):** These model the exact movement of bits across SRAM buffers and MAC arrays to output picojoules and exact clock cycles. They are notoriously slow (hours to run simple operations) and have an extreme learning curve.
*   **Distributed / Discrete-Event Simulators (e.g., ASTRA-sim, VIDUR):** ASTRA-sim (Intel/Meta/Georgia Tech) and VIDUR (MLSys 2024) are incredibly powerful tools for modeling packet-level network collisions and LLM continuous batching queues. However, they are complex C++ frameworks designed for deep industry research, not pedagogical intuition.
*   **Generic Performance Monitors (e.g., Datadog, AnyLogic):** Track live performance but cannot analytically predict the "what-if" scenarios of unbuilt hardware.

**The `mlsysim` Value Proposition (The "Blue Ocean"):**
`mlsysim` is not competing to be a cycle-accurate or packet-accurate simulator. It is the definitive **First-Order Analytical Simulator**.
1.  **Speed & Interpretability ("Glass Box"):** Because it uses closed-form physics equations (The Iron Law, Roofline, Young-Daly), it executes in milliseconds. A student or researcher can trace any output (e.g., a 14.2ms latency) directly back to a readable, textbook mathematical equation, rather than digging through C++ event queues.
2.  **Full-Stack Scope:** Existing tools specialize heavily (just the network, or just the silicon). `mlsysim` models the *entire macroscopic lifecycle*: from single-node memory walls, to distributed ring all-reduce, all the way up to the resulting Total Cost of Ownership ($ TCO) and Carbon Intensity.
3.  **Pedagogy-First & WASM-Native:** It requires zero compilation. It runs purely in Python and can execute natively in a browser via Pyodide/Marimo, making it the only tool capable of powering an interactive, zero-friction undergraduate textbook.
4.  **The Universal Interface / Control Plane:** `mlsysim` serves as the frontend interface to generate configurations for heavy simulators, bridging the gap between high-level analytical modeling and low-level cycle-accurate execution via a standardized Intermediate Representation (IR).

---

## 2. "What Else Should We Be Thinking About?" (The Research & ASPLOS/IISWC Angle)

To make this worthy of a top-tier systems architecture paper (IISWC/ASPLOS), the framework must be more than an educational toy. We must think about:

1. **Empirical Validation (The "Ground Truth" Gap):** An analytical model is only useful if it's accurate. The paper will need a section comparing `mlsysim` predictions against real-world benchmarks.
2. **Modeling Non-Linearities (The "Staircase" Effect):** Reality has step-functions (e.g., when a KV-cache exceeds SRAM and spills to HBM). `mlsysim` must gracefully handle memory hierarchy transitions.
3. **Economics as a First-Class Metric:** `mlsysim` must convert technical metrics natively into Total Cost of Ownership ($ TCO) and Carbon Intensity (kgCO2eq).
4. **Extensibility for Unreleased Hardware:** Researchers must be able to subclass `mlsysim.Hardware` to test new architectures without rewriting the core Engine.
5. **Standardized Intermediate Representation (IR):** The engine must serialize its hardware, workload, and system state into a standardized JSON/YAML schema. This allows external developers to write adapters for other simulators (like ASTRA-sim) just by parsing our IR.

---

## 3. Architectural Best Practices (The MLIR / TVM Influence)

While `mlsysim` is a macroscopic systems simulator rather than a strict AI compiler, its architecture heavily borrows from the best practices established by modern compiler frameworks like **MLIR** (Multi-Level Intermediate Representation) and **Apache TVM**.

To ensure extensibility and maintainability, `mlsysim` implements two core compiler principles:

1. **Progressive Lowering:** Just as MLIR lowers high-level graph concepts down to machine instructions, `mlsysim` progressively "lowers" a Workload.
   * A high-level `Transformer` object is lowered into a `Hardware-Agnostic Computation Graph` (total FLOPs, memory footprint).
   * That graph is then lowered onto a specific `HardwareNode`, which applies precision-specific throughput constraints.
   * Finally, it is lowered onto the `System` layer, which applies dispatch overheads and network latency.
2. **Domain-Specific "Dialects":** Instead of forcing everything into a single monolithic API, `mlsysim` separates concerns. The `infra` layer (Datacenters, Energy Grids) acts as its own dialect. A researcher only working on single-node Roofline optimization never has to interact with or instantiate the `infra` dialect.

---

## 4. Core Architecture (The 6 Layers for 5-Year Longevity)

To ensure the tool remains relevant, the package will be refactored into a rigorous object-oriented hierarchy, acting as both an analytical engine and a universal API.

### Layer A: Workload Representation (`mlsysim.models`)
* `Transformer(params, layers, heads, d_model)`
* `CNN(macs, parameter_bytes, activation_bytes)`

### Layer B: Hardware Registry (`mlsysim.hardware`)
* `ComputeCore(peak_flops, precision_matrix)`
* `MemoryHierarchy(sram_kb, hbm_gb, bandwidth_gbs)`
* `HardwareNode(compute, memory, tdp_watts, unit_cost_dollars)`

### Layer C: Infrastructure & Environment (`mlsysim.infra`)
* `Datacenter(pue, cooling_overhead)`
* `EnergyGrid(carbon_intensity_g_kwh, cost_per_kwh)`

### Layer D: Systems & Topology (`mlsysim.systems`)
* `NetworkFabric(topology="fat-tree", bisection_bw, latency)`
* `Fleet(node, count, fabric, region, mtbf_hours)`

### Layer E: Execution Backends (The Simulators)
This is the most powerful architectural decision: **`mlsysim` cleanly separates the system's *State* (Layers A-D) from its *Execution*.**

Instead of hardcoding analytical math into the core, the analytical models are simply the *default adapters* (backends). This pluggable architecture allows a researcher to define a system once, and simulate it across entirely different engines to compare theoretical bounds against cycle-accurate reality.

*   **The Default Backend (`backend="analytical"`):** The native, first-order physics engine (Iron Law, Roofline, Young-Daly). It runs in milliseconds and is used for the textbook labs.
*   **External Backends (`backend="astrasim"`, `backend="timeloop"`):** Plugins that serialize the system state into the required Intermediate Representation (IR), orchestrate the external C++ simulator, parse the output logs, and return it to the user.
*   **Custom Backends:** A researcher can write `MyCustomAnalyticalBackend` if they want to test a new mathematical theory of pipeline bubbles without altering the core `mlsysim` ontology.

**Example UX:**
```python
fleet = sysim.Fleet(...)
# Instantly get theoretical bounds
analytical_profile = fleet.simulate(backend="analytical")
# Wait 12 hours for cycle-accurate proof
astra_profile = fleet.simulate(backend="astrasim")
```

---

## 5. System Internals & Engineering Standards

To ensure `mlsysim` operates flawlessly as both a research engine and an educational tool, the internal engineering must adhere to strict standards:

1. **Strict Type Safety & Validation (`pydantic`):** All layers (Hardware, Workloads, Systems) will be built using `pydantic` models. This allows for rigorous pre-simulation validation. If a user attempts to connect 10,000 GPUs to a single PCIe switch, the topology layer will throw a descriptive `TopologyValidationError` before the solver ever runs.
2. **Absolute Determinism:** Autograders rely on exact answers. The engine must guarantee deterministic outputs. We will manage floating-point drift and use defined tolerances (`numpy.isclose`) in our testing to ensure a simulation run on an M3 Mac yields the exact same milliseconds result as one run on a Linux server.
3. **Caching & Memoization:** When researchers use `mlsysim` to sweep 100,000 architectural combinations, speed is paramount. The "Progressive Lowering" stages will use memoization (e.g., `@lru_cache`) so that if a Workload graph is lowered once, it isn't redundantly re-calculated on every sweep iteration.

### Extensibility & Ecosystem Integrations
A resilient, modern Python library must play well with the broader MLOps and infrastructure ecosystem.
* **Event-Driven Telemetry (Hooks):** `mlsysim` will implement an event-hook architecture (e.g., `on_simulation_start`, `on_bottleneck_detected`). This allows external tools like MLflow or Weights & Biases to effortlessly track our analytical sweeps without bloating our core codebase.
* **CLI Entry Point:** While Python is the primary API, `mlsysim` will ship with a robust CLI (`sysim run config.yaml`). This allows researchers to orchestrate massive sweeps on SLURM or Kubernetes clusters using standard YAML configurations, treating the simulator as a standalone binary.

---

## 6. Powering the "Gold Standard" Labs

The textbook's interactive labs are the primary vehicle for driving adoption of `mlsysim`. For these labs to be recognized globally as the "Gold Standard," the underlying package must provide native support for pedagogical workflows:

1. **Browser-Native Execution (WASM/Pyodide):** `mlsysim` will have zero heavy C++ dependencies. It must run perfectly in a browser environment (like Marimo/JupyterLite) so students can start learning without installing Python or Docker.
2. **Native Failure States:** Educational labs require "productive failure." The engine will natively throw constraint exceptions (e.g., `OOMError`, `ThermalThrottleWarning`, `SLAViolation`) that UI components can catch and render as visual red alerts.
3. **Autograding Hooks:** To win over university professors, `mlsysim` will include a `mlsysim.eval` module. This will allow instructors to programmatically verify if a student's `System` configuration successfully hit an optimal Pareto frontier (e.g., `assert profile.is_pareto_optimal()`).

---

## 7. Documentation-Driven Development (Quartodoc)

* **Docstrings First:** Exhaustive NumPy-style docstrings with LaTeX equations.
* **Quartodoc Integration:** Auto-generated API reference site matching the textbook.
* **Executable Examples:** `doctests` ensuring code and math remain synced.

---

## 8. Serving Volume 1 & Volume 2 (Client Zero)

The textbook is the ultimate integration test for the simulator. The book's quantitative claims must be generated *by* the simulator.
* **Vol 1 Labs:** Iron Law, Memory bottlenecks, Roofline plotting.
* **Vol 2 Labs:** Ring vs Tree AllReduce, Fault Tolerance/MTBF, Continuous Batching.

---

## 9. Development Roadmap: The Path to v0.1.0

**Migration Strategy: Package-First, Book-Last**
Focus entirely on developing `mlsysim` as a standalone framework. Only refactor the book once validation is complete.

### The v0.1.0 Focus: Full-Stack Analytical Simulation & IR
#### Phase 1: Core API & The Ontology (Weeks 1-2)
- [ ] Implement `HardwareNode`, `NetworkFabric`, Workload abstractions, and `Fleet`.
- [ ] Define the JSON **Intermediate Representation (IR)** schema.

#### Phase 2: The Multi-Scale Solvers (Weeks 3-4)
- [ ] Implement `SingleNodeSolver`, `DistributedSolver`, and `ReliabilitySolver`.

#### Phase 3: UX, Visualization, and Economics (Week 5)
- [ ] Actionable Errors (e.g., `OOMError`), `plot_roofline()`, and Economics calculators (TCO, Carbon).

#### Phase 4: Empirical Validation & Testing (Week 6)
- [ ] Implement strict `pytest` suite for textbook math and `test_empirical.py` against MLPerf.

#### Phase 5: Release v0.1.0 & The Great Book Refactor (Weeks 7-8)
- [ ] Publish to PyPI, refit Vol 1/2 Labs, and update book QMD files.

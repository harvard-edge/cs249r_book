# mlsysim: The Architecture & Development Plan

## Vision: The MIPS/SPIM for Machine Learning Systems
`mlsysim` is a first-order analytical simulator for AI infrastructure. Just as Hennessy and Patterson used the MIPS architecture and SPIM simulator to teach the physics of instruction pipelining, `mlsysim` teaches the physics of tensor movement, memory hierarchies, and distributed fleet dynamics.

---

## 1. Core Architecture (The 5-Layer Stack) - [COMPLETED]

*   **Layer A: Workload Representation**: High-level model definitions.
*   **Layer B: Hardware Registry**: Concrete specs for real-world devices (H100, iPhone, ESP32).
*   **Layer C: Infrastructure & Environment**: Regional grids and PUE models.
*   **Layer D: Systems & Topology**: Fleet configurations and narrative Scenarios.
*   **Layer E: Execution & Solvers**: Pluggable solvers for Performance, Serving, and Economics.

---

## 2. Systematic Record of Execution

### Phase 1: Core API & The Ontology [COMPLETED - 2025-03-06]
*   Migrated from monolithic `core` to 5-layer Pydantic-powered structure.
*   Implemented `Quantity` types with strict validation and JSON serialization.

### Phase 2: Volume 2 "Farm to Scale" Core [COMPLETED - 2025-03-06]
*   **3D Parallelism:** Implemented `DistributedSolver` with TP/PP/DP and Pipeline Bubble math.
*   **LLM Serving:** Implemented `ServingSolver` with KV-Cache footprint and Pre-fill/Decode phases.
*   **Network Physics:** Added Oversubscription Ratios and Bisection BW logic.
*   **Narrative Scenarios:** Implemented the "Lighthouse Archetypes" (Doorbell, AV, Frontier).
*   **Hierarchy of Constraints:** Implemented `SystemEvaluation` Scorecard (Feasibility -> Performance -> Macro).
*   **Concrete Registry:** Replaced generic placeholders with 15+ real-world devices (iPhone 15, H200, MI300X, etc).

---

## 3. The "No Hallucination" Validation Standard

1.  **Empirical Anchoring:** Every solver validated against **MLPerf**, **Megatron-LM**, or published training logs.
2.  **Dimensional Analysis:** Every formula proven via `pint` unit resolution.
3.  **Traceable Constants:** Every constant in `core.constants` cited to a specific datasheet or paper.

### Phase 3: Empirical Validation & Documentation [IN PROGRESS - 2025-03-06]
*   **Deep Narrative Analysis:** Completed 32-chapter audit. Integrated `plot_scorecard()` into Volume 1 and "Memory Wall" case study into Volume 2.
*   **Empirical Validation Suite:** Build `tests/test_empirical.py`.
*   **Goal:** Assert that simulator predictions match MLPerf results within 10%.

### Phase 4: Tail Latency & Straggler Physics
*   **Scope:** Probabilistic models for P99/P99.9 latencies in massive fleets.

### Phase 5: Automated Documentation (Quartodoc)
*   **Scope:** Generate the full API reference site directly from docstrings.

### Phase 6: Live Sourcing & Freshness (Thinking Ahead)
*   **Goal:** Move from hardcoded constants to a "Source-Anchored" registry.
*   **Action:** Implement a `ProvenanceMap` that links physical constants to public dashboards (e.g., Electricity Maps, AWS Pricing API).
*   **Outcome:** A "Verified" badge next to every number in the documentation with a link to the primary source.

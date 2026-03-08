# mlsysim: Volume 2 "Farm to Scale" Plan

This document tracks the systematic build-out of the advanced features for Volume 2 of the Machine Learning Systems textbook.

---

## 📅 Roadmap Overview

| Feature | Status | Priority | Goal |
| :--- | :---: | :---: | :--- |
| **LLM Serving & KV-Cache** | ✅ | P0 | Model TTFT, ITL, and memory footprint of LLM inference. |
| **3D Parallelism Solver** | ✅ | P1 | Model TP/PP bubbles for massive Frontier-scale training. |
| **Network Bisection & Oversubscription** | ✅ | P1 | Model congestion in non-blocking and oversubscribed fabrics. |
| **Concrete Hardware Registry** | ✅ | P1 | Replace generics with real-world devices (iPhone, H200, etc). |
| **Empirical Validation Suite** | ⬜ | P2 | Build `test_empirical.py` against MLPerf benchmarks. |
| **Tail Latency Physics** | ⬜ | P2 | Calculate P99/P99.9 using queueing theory. |

---

## ✅ Systematic Execution Log

### 2025-03-06: Infrastructure Foundations Complete
*   Completed the refactor to the 5-layer Pydantic stack (Layers A-E).
*   Implemented the baseline `DistributedSolver` and `EconomicsSolver`.
*   Fixed the `generate_appendix.py` to correctly extract data from the new registry.
*   Verified that all Volume 1 & 2 book invariants hold after the structural refactor.

### 2025-03-06: LLM Serving & KV-Cache [COMPLETED]
*   Implemented `ServingSolver` in `mlsysim.core.solver` supporting Pre-fill and Decoding phases.
*   Added `heads`, `kv_heads`, and `hidden_dim` to `TransformerWorkload`.
*   Implemented `get_kv_cache_size` method for dynamic memory calculation.
*   Verified against Llama-3-70B on H100 (detected infeasibility for single-node FP16).

### 2025-03-06: 3D Parallelism & Network Congestion [COMPLETED]
*   Upgraded `DistributedSolver` to support **Tensor Parallelism (TP)** and **Pipeline Parallelism (PP)**.
*   Implemented the **Pipeline Bubble** formula ($ (P-1)/(P-1+M) $).
*   Added `oversubscription_ratio` to `NetworkFabric` and integrated it into communication math.
*   Added comprehensive **NumPy-style docstrings** to all solvers in `mlsysim.core.solver`.
*   Verified against a Frontier-8K H100 cluster scenario.

### 2025-03-06: Concrete Hardware & Narrative Scenarios [COMPLETED]
*   Replaced generic placeholders with **15+ real-world devices** including iPhone 15 Pro, MacBook M3 Max, and NVIDIA H200.
*   Implemented the **Lighthouse Archetype** scenarios (Doorbell, AV, Frontier) with built-in SLA validation.
*   Created the **Hierarchy of Constraints** `SystemEvaluation` scorecard.
*   Established **Engineering & Modeling Best Practices** in `BEST_PRACTICES.md`.
*   Created **Hello World** and **Manual Sweep** tutorials for students.

---

## 🛠 Feature Specs

### [P0] LLM Serving & KV-Cache
- **Input:** `model: TransformerWorkload`, `hardware: HardwareNode`, `seq_len: int`, `batch_size: int`.
- **Output:** `latency_prefill`, `latency_decoding`, `total_kv_cache_gb`, `feasible_on_hardware`.
- **Validation:** Must match vLLM benchmark results for Llama-3-70B on H100 (within 10%).

---

## 🛡 Verification Standard ("No Hallucination")
1.  **Unit Tests:** Every feature must have a corresponding test in `mlsysim/tests/`.
2.  **Empirical Anchor:** Formulas must be cited from standard industry papers (e.g., "The Case for PagedAttention").
3.  **Dimensional Integrity:** `pint` must resolve all results to correct SI units.

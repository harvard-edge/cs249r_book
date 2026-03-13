# mlsysim: Volume 2 "Farm to Scale" Plan

This document tracks the systematic build-out of the advanced features for Volume 2 of the Machine Learning Systems textbook.

---

## 🚀 The "Robust Framework" Roadmap (CLI V3)

To elevate `mlsysim` from an agent-ready tool to a true industry-standard framework (like PyTorch or Terraform), the following architectural features are prioritized:

| Feature | Goal | UX Design |
| :--- | :--- | :--- |
| **Implicit Config Discovery** | Make the tool "directory-aware" like `docker-compose`. | `mlsysim eval` automatically finds and runs `./mlsys.yaml`. |
| **Traceability Engine** | Eliminate the "black box" by logging the exact mathematical sequence. | `mlsysim --debug eval` streams step-by-step solver calculations to `stderr`. |
| **Plugin Architecture** | Allow companies to model unreleased/proprietary hardware privately. | `mlsysim --include custom_chips.yaml eval` injects external definitions into the Zoo. |
| **Rich Export Formats** | Embed the tool into the social workflow of engineering (PRs, Design Docs). | `mlsysim eval -o markdown` generates a beautiful Markdown table for GitHub. |

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
*   Implemented the baseline `DistributedModel` and `EconomicsModel`.
*   Fixed the `generate_appendix.py` to correctly extract data from the new registry.
*   Verified that all Volume 1 & 2 book invariants hold after the structural refactor.

### 2025-03-06: LLM Serving & KV-Cache [COMPLETED]
*   Implemented `ServingModel` in `mlsysim.core.solver` supporting Pre-fill and Decoding phases.
*   Added `heads`, `kv_heads`, and `hidden_dim` to `TransformerWorkload`.
*   Implemented `get_kv_cache_size` method for dynamic memory calculation.
*   Verified against Llama-3-70B on H100 (detected infeasibility for single-node FP16).

### 2025-03-06: 3D Parallelism & Network Congestion [COMPLETED]
*   Upgraded `DistributedModel` to support **Tensor Parallelism (TP)** and **Pipeline Parallelism (PP)**.
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

### SOTA Paradigm Completeness (New Additions) [COMPLETED]
*   **ZeRO/FSDP:** Added `zero_stage` support in `TransformerWorkload.training_memory` and `DistributedModel`, appropriately sharding memory footprints for optimizer states and gradients.
*   **PEFT/LoRA:** Added `is_lora` flag to eliminate 99% of optimizer state footprint for fine-tuning workloads, making large-model single-node tuning mathematically feasible.
*   **Activation Recomputation:** Added `activation_recomputation` flag trading off +33% training FLOPS for massive activation memory savings via selective recalculation.
*   **Compute/Communication Overlap:** Added `overlap_comm` flag to `DistributedModel` changing the latency equation from strictly additive to mathematically hiding network latency behind backward pass computation.
*   **Speculative Decoding:** Upgraded `ServingModel` with `draft_model` and `draft_acceptance_rate` inputs, calculating probability-weighted expected latency improvements.
*   **Disaggregated Serving:** Added `decode_hardware` and `network_bandwidth` options to `ServingModel` to model split pre-fill and decode clusters exchanging KV-cache payloads over datacenter fabrics.

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

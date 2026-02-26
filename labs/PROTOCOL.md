# 📘 ML Systems: Interactive Lab Developer Guide

This document is the master instruction set for generating the interactive analytical laboratories that accompany the *Machine Learning Systems* textbook. Any LLM agent tasked with generating or modifying a lab must adhere strictly to this guide.

## 1. Context: The Textbook and The Ecosystem
*Machine Learning Systems* is a textbook teaching **AI Engineering**: the discipline of building stochastic systems with deterministic reliability.

*   **Volume 1 (Foundations):** Physics of the Node (D·A·M taxonomy, Iron Law, Memory/Power Walls).
*   **Volume 2 (Fleet Scale):** Orchestration of the Fleet (Distributed Training, Fault Tolerance, Serving at Scale).

**The Pedagogical Pipeline:**
1. **Read:** Theoretical principles.
2. **Build (TinyTorch):** Software mechanics.
3. **Analyze (The Labs):** **[YOUR DOMAIN]** Design Space Exploration (DSE).
4. **Deploy (The Kits):** Physical hardware realization.

---

## 2. Principle of Progressive Disclosure
Labs are synchronized with the reader's journey. **Never assume knowledge or provide instruments not yet introduced.**

*   **Instrument Gating:** Lab 2 has only the Waterfall. Lab 9 adds the Pareto Curve. Lab 13 adds the P99 Histogram.
*   **Narrative Persistence:** Decisions in Chapter 4 (e.g., Data Format) should impact Chapter 13 (e.g., Latency).
*   **Retrofit Option:** Provide a "System Reset" or "Expert Preset" for students starting mid-book.

---

## 3. The Structural Matrix: Rows & Columns
Every Marimo lab MUST dynamicially reskin itself based on the student's chosen **Persona Track** (Row) and the **Chapter Invariant** (Column).

### 3.1 The Narrative Tracks (The Rows)
Each track follows a **Fixed North Star Mission** that remains constant from Chapter 1 to 16.

| Track | Persona | Fixed Ultimate Goal (The Single-Node North Star) |
| :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving throughput on a single H100 node. |
| **Edge Guardian** | AV Systems Lead | Maintain sub-10ms deterministic safety loop for an Autonomous Vehicle. |
| **Mobile Nomad** | AR Glasses Dev | Run 60FPS AR filters on Meta Ray-Bans without thermal throttling. |
| **Tiny Pioneer** | Hearable Lead | Real-time speech isolation/translation in <10ms using <1mW of power. |

### 3.2 Narrative Continuity Rules
1. **The Node Invariant (Volume 1):** Every lab in Volume 1 MUST focus strictly on the physics of a **Single Node** (one accelerator, one robot, one phone, one MCU). Do not introduce multi-node orchestration, load balancing, or fleet-scale economics until Volume 2.
2. **The Introduction Hook:** Lab 01 must establish the universal D·A·M fundamentals *before* the student selects a track. Track selection acts as the "Call to Adventure" at the end of the lab.
2. **Persistent Struggle:** The "Arch Nemesis" constraint for each track must reappear in every lab, becoming harder to solve as the system scales.
3. **The Design Ledger:** Every lab must update the student's "System Specs" (the Ledger), creating a sense of cumulative progress.

### 3.2 The Master Slug List (The Columns)
Agents MUST ensure 1:1 path parity with the book contents. **Rule:** Lab files must be named `lab_XX_slug.py` (e.g., `lab_02_ml_systems.py`) to ensure they are valid Python modules.

**Volume 1 (Foundations)**
lab_01_ml_intro, lab_02_ml_systems, lab_03_ml_workflow, lab_04_data_engr, lab_05_nn_compute, lab_06_nn_arch, lab_07_ml_frameworks, lab_08_model_train, lab_09_data_selection, lab_10_model_compress, lab_11_hw_accel, lab_12_perf_bench, lab_13_model_serving, lab_14_ml_ops, lab_15_responsible_engr, lab_16_ml_conclusion.

**Volume 2 (Scale)**
01_ml_intro, 02_compute_infra, 03_network_fabrics, 04_data_storage, 05_dist_train, 06_collect_comms, 07_fault_tolerance, 08_fleet_orch, 09_perf_engr, 10_dist_inference, 11_edge_intel, 12_ops_scale, 13_sec_privacy, 14_robust_ai, 15_sustainable_ai, 16_responsible_ai, 17_ml_conclusion.

---

## 4. The Pedagogical Gearbox (Lock-and-Key)
Every Analysis Task MUST follow this loop:
1. **Predict:** Student types a hypothesis. UI is **locked/blurred** until input.
2. **Act:** Student manipulates sliders.
3. **Observe:** High-fidelity visual cards update.
4. **Reflect:** Student justifies choice to a **Stakeholder** using chapter Invariants.

---

## 5. Technical Implementation Specifications
*   **Format:** Single-file Marimo Notebook (`.py`).
*   **Engine:** 100% of math must call `mlsys.Engine.solve()`.
*   **WASM-First:** Zero local file I/O. Data must be in native Python dicts in the `mlsys` package.
*   **Unique Variable Naming:** Marimo treats the notebook as a single dataflow graph. Agents MUST NOT reuse variable names (e.g., `fig`, `reflection`, `selected`) across different cells. Use descriptive, cell-specific names or leading underscores for private variables.
*   **Units:** Mandatory use of `mlsys.ureg` (`pint`).
*   **Visuals:** Import components from `labs.core`. Use **BlueLine** (#006395), **RedLine** (#CB202D), and **GreenLine** (#008F45).

---

## 6. API & Component Contract

### 6.1 The Engine Interface (Strict Contract)
Agents MUST NOT hallucinate method names. The only valid way to compute physics is:

```python
from mlsys import Engine, Models, Systems

profile = Engine.solve(
    model=Models.ResNet50,      # Type: ModelSpec
    system=Systems.Mobile,      # Type: SystemArchetype
    batch_size=32,              # Type: int
    precision="int8",           # Options: "fp32", "fp16", "int8"
    efficiency=0.5              # Type: float (0.0 - 1.0)
)

# The 'profile' object (PerformanceProfile) guarantees these fields:
# .latency (Pint ms)
# .latency_compute (Pint ms)
# .latency_memory (Pint ms)
# .latency_overhead (Pint ms) - The Dispatch Tax
# .throughput (Pint samples/sec)
# .bottleneck (str: "Compute" or "Memory")
# .energy (Pint joule)
# .memory_footprint (Pint byte)
# .feasible (bool: memory_footprint <= system.ram)
```

### 6.2 The UI Boilerplate
* **The Lock:** `mo.stop(prediction == "", mo.md("⚠️ Enter prediction..."))`
* **The Math Peek:** Use `labs.core.components.MathPeek(formula, variables)` to reveal the underlying physics toggle. Mandatory for every chart card.
* **The Overhead Term:** Every Latency Waterfall MUST show the **Overhead** (Dispatch Tax) to teach the reality of Software 2.0.

### 6.3 Stakeholder Personas
* **CFO (Cloud):** Worried about "Samples per Dollar."
* **Safety Lead (Edge):** Demands "Deterministic Reliability."
* **UX Designer (Mobile):** Worried about "Hand Thermals."
* **Hardware Lead (Tiny):** Fights for every "KB of SRAM."

---

## 7. The Design Ledger (JSON Schema)
```json
{
  "chapter_id": int,
  "track": "string",
  "current_design": { "params": "..." },
  "reflections": [ { "task": "...", "text": "..." } ]
}
```

---

## 8. Developer Agent Workflow
1. **Read Chapter X:** Extract learning objectives and invariants.
2. **Context Audit:** Identify what knowledge/instruments are "Unlocked."
3. **Distill Struggle:** Identify the zero-sum trade-off.
4. **Grid Map:** Define scenarios for all 4 Tracks.
5. **Design KATs:** Outline 3 sequential Key Analysis Tasks.
6. **Generate Code:** Write the Marimo app using `labs.core`.

---
**[END OF GUIDE]**

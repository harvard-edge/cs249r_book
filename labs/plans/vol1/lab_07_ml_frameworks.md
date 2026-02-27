# 📐 Mission Plan: 07_ml_frameworks (Deep Analysis)

## 1. Chapter Context
*   **Chapter Title:** ML Frameworks: The Execution Engines.
*   **Core Invariant:** The Dispatch Tax (Fixed Software Overhead per Kernel).
*   **The Struggle:** Understanding that software flexibility isn't free. Students must navigate the "Compilation Continuum"—balancing the ease of **Eager Execution** against the throughput of **Graph Compilation**.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Framework" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Python Ceiling.** Eager overhead prevents H100 saturation at small batch sizes. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Jitter Tax.** Python's garbage collection and eager dispatch introduce non-deterministic latency spikes. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Fusion Gap.** Separate kernels drain battery by repeatedly reading/writing to memory; must fuse to survive. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Binary Bloat.** Full frameworks don't fit; must navigate the "Bare-Metal" trade-offs of TFLite Micro. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Dispatch Tax Audit (Exploration - 15 Mins)
*   **Objective:** Quantify the "Software Overhead" ($L_{lat}$) relative to actual "Math Time" ($T_{comp}$).
*   **The "Lock" (Prediction):** "For a small model (e.g., KWS), which will be larger: the time spent doing arithmetic or the time spent by the framework launching the kernels?"
*   **The Workbench:**
    *   **Action:** Toggle between **Eager** and **Compiled** modes. Slide the **Model Scale** (Layer count).
    *   **Observation:** The **Latency Waterfall**. Notice the "Overhead" bar ($L_{lat}$) remaining constant while the "Compute" bar shrinks.
*   **Reflect:** "Why does a faster GPU sometimes result in *lower* utilization for small models? Reference the Dispatch Tax in your answer."

### Part 2: The Compilation Break-even (Trade-off - 15 Mins)
*   **Objective:** Identify the "Compilation Sweet Spot" where the setup cost of a graph is justified by its execution gain.
*   **The "Lock" (Prediction):** "Will compilation yield more relative speedup for a model with 1,000 small layers or a model with 1 giant layer?"
*   **The Workbench:**
    *   **Action:** Adjust **Kernel Complexity** and **Operator Fusion** levels.
    *   **Observation:** The **Compilation Continuum Plot**. A curve showing "Throughput vs. Flexibility."
    *   **The 5-Move Rule:** Students must find the exact "Break-even" point where Operator Fusion reduces memory traffic enough to move the system from bandwidth-bound to compute-bound.
*   **Reflect:** "Reconcile the result with the 'Ladder of Abstraction.' When is 'High-level' Python actually a physical liability?"

### Part 3: Graph Surgery (Synthesis - 15 Mins)
*   **Objective:** Fix a "Graph Break" to enable end-to-end acceleration.
*   **The "Lock" (Prediction):** "If our model has a 'Data-Dependent Branch' (if/else based on tensor values), can the framework compile the entire model into a single kernel?"
*   **The Workbench:** 
    *   **Interaction:** A "Graph Trace" visualizer. The student identifies a "Red Node" (The Graph Break).
    *   **Action:** Select a "Surgery Option" (e.g., Rewrite branch as a mask, use symbolic shape).
    *   **The "Stakeholder" Challenge:** The **Software Lead** refuses to accept the rewritten code because it's "less readable." You must prove the **MFU (Model FLOPs Utilization)** gain is >2x to justify the change.
*   **Reflect (The Ledger):** "Describe the 'Software-Hardware Gap' you bridged. Did your fix improve the Data term or the Compute term of the Iron Law?"

---

## 4. Visual Layout Specification
*   **Primary:** `FrameworkWaterfall` (Compute vs. Memory vs. Framework Overhead).
*   **Secondary:** `GraphTraceMap` (Mermaid.js diagram showing nodes and the "Break" location).
*   **Math Peeking:** Toggle for the `Dispatch Tax` formula and `Break-even Analysis`.

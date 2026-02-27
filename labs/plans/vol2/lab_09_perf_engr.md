# 📐 Mission Plan: 09_perf_engr (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Performance Engineering: Analysis & Optimization at Scale.
*   **Core Invariant:** The Profiling Invariant (You cannot optimize what you cannot measure) and the **Iron Law of ML Performance**.
*   **The Struggle:** Understanding that at scale, "Optimizing everything is optimizing nothing." Students must navigate the trade-off between **Local Kernel Gains** (improving one layer) and **Global System Utilization** (MFU/MBU), specifically focusing on how the Memory Wall dictates the "Ridge Point" across a heterogeneous fleet.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Performance Missions)

| Track | Persona | Fixed North Star Mission | The "Performance" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Attention Bottleneck.** Your Llama-3 throughput is 50% below the A100 baseline. Your profile shows that 80% of time is spent loading KV-cache. You must implement 'FlashAttention'. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Jitter Audit.** A sporadic 50ms latency spike is causing AV phantom braking. You must use 'Trace-level Profiling' to find the exact kernel causing the jitter. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Thermal-Precision Trap.** Your 60FPS filter is causing the glasses to overheat. You must decide whether to use 'Operator Fusion' or 'Mixed-Precision' to save the thermal budget. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Memory-Math Ratio.** Your LSTM is memory-bound on the ESP32. You must 'Dimension' the hidden state size to align with the local cache line. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Diagnostic Challenge (Exploration - 15 Mins)
*   **Objective:** Diagnose a failing system using the Roofline Model and the Iron Law.
*   **The "Lock" (Prediction):** "If a layer sits on the *Sloped* section of the Roofline, will adding more TFLOPS speed it up?"
*   **The Workbench:**
    *   **Action:** Select layers from your Track's model. Toggle **Profiler Mode**.
    *   **Observation:** The **Live Roofline Dash**. Watch the "Red Dot" move based on Layer Intensity.
*   **Reflect:** "Patterson asks: 'Identify your binding constraint.' Is it $BW_{mem}$ or $R_{peak}$? Use the **MBU (Memory Bandwidth Utilization)** gauge to prove it."

### Part 2: The Fusion Gain (Trade-off - 15 Mins)
*   **Objective:** Quantify the reduction in "Memory Traffic" achieved through Operator Fusion.
*   **The "Lock" (Prediction):** "Does fusing 'Linear + ReLU' reduce the total number of operations ($O$) or the total data moved ($D_{vol}$)?"
*   **The Workbench:**
    *   **Interaction:** Toggle **Fusion Levels** (None -> Partial -> Full). Adjust **Batch Size**.
    *   **Instruments:** **Data Traffic Waterfall** (Bits saved vs. Operations).
    *   **The 10-Iteration Rule:** Students must find the "Fusion Set" that maximizes MFU without exceeding the track's fixed memory capacity.
*   **Reflect:** "Jeff Dean observes: 'Your kernels are too small, causing dispatch overhead to dominate.' Propose a 'Kernel Tiling' change to saturate the hardware."

### Part 3: Algorithmic Innovation (Synthesis - 15 Mins)
*   **Objective:** Implement 'Speculative Decoding' or 'MoE' to bypass physical walls.
*   **The "Lock" (Prediction):** "If we use a tiny 'Draft Model' to predict tokens, will it increase or decrease the total TFLOPS used per final word?"
*   **The Workbench:** 
    *   **Interaction:** **Speculative Decoding Toggle**. **Expert Selection (MoE)**. **Sparsity Scrubber**.
    *   **The "Stakeholder" Challenge:** The **Product Lead** demands a 2x throughput boost. You must prove that using **Mixture of Experts** (MoE) hits the target by reducing the *Active Parameter* count while the *Memory Footprint* grows.
*   **Reflect (The Ledger):** "Defend your final 'Performance Strategy.' Did you optimize the 'Machine' (Fusion) or the 'Algorithm' (Speculation)? Justify how you bridged the Systems Gap."

---

## 4. Visual Layout Specification
*   **Primary:** `DynamicRooflineVisualizer` (Plotting MFU and MBU in real-time).
*   **Secondary:** `OptimizationWaterfall` (Showing speedup from Precision vs Fusion vs Algorithmic tricks).
*   **Math Peek:** Toggle for `MFU = \frac{	ext{Observed Throughput}}{	ext{Theoretical Peak}}` and `MBU` formulas.

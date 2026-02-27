# 📐 Mission Plan: 02_ml_systems (Deep Analysis)

## 1. Chapter Context
*   **Chapter Title:** ML Systems: The Physics of Deployment.
*   **Core Invariant:** The Iron Law ($T = T_{data} + T_{memory} + T_{compute} + T_{overhead}$).
*   **The Struggle:** Identifying the "Binding Constraint" across disparate hardware regimes.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Arch Nemesis" Wall |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving on a single H100. | **Memory Bandwidth Wall** (HBM) |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop on NVIDIA Orin. | **The Determinism Wall** (Jitter) |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation on Meta Ray-Bans. | **The Power Wall** (Thermals) |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Capacity Wall** (SRAM) |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Term Audit (Exploration - 15 Mins)
*   **Objective:** Quantify the magnitude of the Iron Law terms ($T_{mem}, T_{comp}, T_{ovh}$) across the deployment spectrum.
*   **The "Lock" (Prediction):** "If you move your model from an ESP32 to an H100, which term of the Iron Law will shrink the MOST in relative terms?"
*   **The Workbench:**
    *   **Sliders:** Hardware Tier (Tiny -> Cloud), Batch Size (1-256), Model Scale.
    *   **Instruments:** `LatencyWaterfall` (Stacked components), `MetricRatioCounter`.
    *   **The 5-Move Rule:** Students must analyze at least 5 different hardware/batch combinations to find the "Ridge Point" where the bottleneck flips for each tier.
*   **The "Hook":** A "Magic Zero-Latency Network" button to show that even with no light-speed penalty, the Memory Wall remains.
*   **Reflect:** "Reconcile the $100,000x$ TFLOPS gap with the fact that throughput only increased $100x$. Where did the remaining $1,000x$ performance go? (Reference: The Bottleneck Principle)."

### Part 2: The Amdahl Wall (Trade-off - 20 Mins)
*   **Objective:** Prove that accelerating the parallel fraction yields diminishing returns if the serial tax remains high.
*   **The "Lock" (Prediction):** "If the serial overhead (dispatch tax) is 5ms, can a 1,000,000x faster GPU hit a 4ms total latency target?"
*   **The Workbench:**
    *   **Sliders:** "Magic Accelerator" Speedup (1x -> 10,000x), Framework Tax ($\mu s$ -> $ms$), Kernel Fusion Level.
    *   **Instruments:** `AmdahlHeatmap` (Speedup vs Serial %), `ThroughputCeilingPlot`.
    *   **The 15-Iteration Rule:** Students must find the "Critical Parallel Fraction" ($p$) required to hit their specific mission target (e.g., 60FPS for AR).
*   **Reflect:** "Your CEO wants to buy a faster GPU. Use the Amdahl Heatmap to prove why this investment is useless unless we first reduce the software overhead."

### Part 3: The Efficiency Synthesis (Synthesis - 10 Mins)
*   **Objective:** Optimize the system to hit the Mission Goal while maintaining >40% Model FLOPs Utilization (MFU).
*   **The "Lock" (Prediction):** "What combination of Batch Size and Precision will maximize MFU without violating the Latency SLA?"
*   **The Workbench:** All D·A·M sliders unlocked.
*   **The "Stakeholder" Challenge:** The **CFO** presents a counter-proposal: "Just use the cheapest hardware at Batch 1." The student must prove why this leads to a catastrophic "Low Utilization Tax."
*   **Reflect (The Ledger):** Justify your final $R_{peak}$, $BW$, and $O$ configuration. Explain the "Conservation of Complexity" in your design.

---

## 4. Visual Layout Specification
*   **Primary:** `LatencyWaterfall` (Now showing the $L_{lat}$ Overhead term explicitly).
*   **Secondary:** `RooflineVisualizer` (Log-Log scale).
*   **Transparency:** Toggle for `Speedup = 1 / ((1-p) + p/S)`.

# 📐 Mission Plan: 11_hw_accel (Hardware Acceleration)

## 1. Chapter Context
*   **Chapter Title:** Hardware Acceleration: The Silicon Foundation.
*   **Core Invariant:** The Memory Wall ($Bandwidth$ vs. $Compute$) and Arithmetic Intensity ($I$).
*   **The Struggle:** Understanding that silicon is not a "magic speedup." Students must navigate the **Memory Wall**—identifying that a 1000x faster processor ($R_{peak}$) provides zero gain if the system is limited by the rate of weight loading ($BW$).
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Silicon" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The HBM Bottle.** You have 2,000 TFLOPS of compute, but your model only achieves 10% utilization. Your HBM is 100% saturated. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Determinism Tax.** Your NPU is fast, but the 'Dispatch Overhead' of the OS is introducing 5ms of jitter, breaking safety constraints. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Energy Inversion.** Fetching a weight from DRAM costs 500x more energy than the actual multiplication. You must use **SRAM Tiling** or the glasses melt. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Parallelism Gap.** Your MCU lacks a Systolic Array. You must manually vectorize ops or miss the 10ms 'Echo Window.' |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Roofline Audit (Exploration - 15 Mins)
*   **Objective:** Map your model's layers onto the Hardware Roofline to identify the bottleneck.
*   **The "Lock" (Prediction):** "Calculate the Ridge Point ($R_{peak} / BW$) of your hardware. If your layer's Arithmetic Intensity is *below* this point, what is the primary bottleneck?"
*   **The Workbench:**
    *   **Action:** Select layers from your Lighthouse Model. Toggle between **CPU**, **GPU**, and **NPU**.
    *   **Observation:** The **Dynamic Roofline Plot**. Watch the "Model Dot" move from the Compute ceiling to the Bandwidth slope.
*   **Reflect:** "Patterson asks: 'Why did doubling the TFLOPS yield zero speedup for your Attention layer?' Use the term **'Sloped Region'** in your answer."

### Part 2: The Energy Inversion (Trade-off - 15 Mins)
*   **Objective:** Minimize DRAM traffic by optimizing the Dataflow Strategy (Weight-Stationary vs. Output-Stationary).
*   **The "Lock" (Prediction):** "Is it more energy-efficient to keep Weights stationary or Activations stationary for a layer with 10 million parameters and 1 thousand inputs?"
*   **The Workbench:**
    *   **Sliders:** SRAM Buffer Size (KB), Tiling Strategy, Dataflow Mode.
    *   **Instruments:** **Energy Waterfall** (SRAM Energy vs. DRAM Energy).
    *   **The 10-Iteration Rule:** Students must find the optimal "Tile Size" that maximizes SRAM reuse and minimizes the expensive "DRAM Energy Penalty."
*   **Reflect:** "Jeff Dean observes: 'Your system is fast but its carbon footprint is 5x too high.' Propose a tiling change to reduce total Joules per inference."

### Part 3: The Acceleration Ceiling (Synthesis - 15 Mins)
*   **Objective:** Apply Amdahl's Law to find the true limit of hardware speedups.
*   **The "Lock" (Prediction):** "If 10% of your code is non-parallelizable (Serial), what is the maximum possible speedup if you buy a 1,000,000x faster accelerator?"
*   **The Workbench:**
    *   **Interaction:** **Accelerator Speedup Slider (S)**. **Serial Fraction Slider (1-p)**.
    *   **The "Stakeholder" Challenge:** The **Software Lead** wants to spend the budget on a faster NPU. You must prove, using the **Amdahl Heatmap**, that spending the budget on "Reducing Driver Overhead" yields a 4x better ROI.
*   **Reflect (The Ledger):** "Defend your final Hardware Selection. Did you prioritize raw TFLOPS or Memory Bandwidth? Explain how the 'Memory Wall' influenced your final 'Silicon Contract'."

---

## 4. Visual Layout Specification
*   **Primary:** `RooflineVisualizer` (Log-Log plot showing Hardware Roof and Model Point).
*   **Secondary:** `EnergyWaterfall` (MAC pJ vs. SRAM pJ vs. DRAM pJ).
*   **Math Peek:** Toggle for `Speedup = 1 / ((1-p) + p/S)` and `Ridge Point = R_{peak} / BW`.

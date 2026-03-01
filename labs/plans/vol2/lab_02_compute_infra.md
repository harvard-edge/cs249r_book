# 📐 Mission Plan: 02_compute_infra (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Compute Infrastructure: The Warehouse-Scale Engine.
*   **Core Invariant:** Power Density Walls and **PUE** (Power Usage Effectiveness).
*   **The Struggle:** Understanding that at scale, "The Machine" is not a chip, but a **Warehouse**. Students must navigate the trade-off between **Compute Density** (TFLOPS per Rack) and **Cooling Efficiency** (PUE), specifically focusing on the transition from Air to Liquid cooling.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Infrastructure Missions)

| Track | Persona | Fixed North Star Mission | The "Infrastructure" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The 100kW Rack Wall.** Your new B200 nodes pull 120kW per rack. The datacenter's air cooling is failing. You must decide whether to de-clock or move to liquid cooling. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Field-Pod Transient.** Your 'Fleet Hub' (a local compute node) is overheating in a hot garage. You must balance perception accuracy against thermal throttling to prevent 'Safety Voids'. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Multi-Tenant Latency.** You are offloading AR compute to a shared Edge Server. Another app's noisy neighbor is stealing your cache, causing FPS drops. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Gateway Jam.** You are scaling from 1 to 1,000 hearables. The 'Machine' is now the gateway processor that must aggregate 1,000 streams without exploding its mW budget. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Silicon Contract (Selection - 15 Mins)
*   **Objective:** Select the optimal "Accelerator Class" (GPU, TPU, ASIC) for your fleet based on Arithmetic Intensity.
*   **The "Lock" (Prediction):** "Will a specialized ASIC always outperform a general-purpose GPU if the model's layers have highly irregular access patterns?"
*   **The Workbench:**
    *   **Action:** Toggle between **GPU**, **TPU**, and **NPU**. Apply your track's specific workload.
    *   **Observation:** The **Accelerator Efficiency Radar**. Compare 'Throughput-per-Watt' across the three architectures.
*   **Reflect:** "Patterson asks: 'Why is the Generality Tax higher for the Cloud Titan than for the Tiny Pioneer?' (Reference the silicon area spent on control logic)."

### Part 2: The Power Density Wall (Thermal Design - 15 Mins)
*   **Objective:** Optimize Rack Density while minimizing the "Cooling Tax" (PUE).
*   **The "Lock" (Prediction):** "If you increase rack density by 2x using Air cooling, does the PUE (Total Power / IT Power) improve, stay the same, or degrade?"
*   **The Workbench:**
    *   **Sliders:** Nodes per Rack, Ambient Temperature, Fan Speed (Air) vs Pump Flow (Liquid).
    *   **Instruments:** **PUE Gauge**. **Thermal Heatmap**.
    *   **The 10-Iteration Rule:** Students must find the "Knee of the Thermal Curve" where the energy cost of cooling exceeds the gain in compute density.
*   **Reflect:** "Jeff Dean observes: 'Your PUE is 1.8. You are wasting 80% of your energy on non-compute.' Propose a transition to Liquid Cooling and prove the TCO reduction."

### Part 3: The Scaling Cliff (Fleet Economics - 15 Mins)
*   **Objective:** Perform a "Build-vs-Buy" audit for your track's infrastructure.
*   **The "Lock" (Prediction):** "Over a 3-year period, is it cheaper to rent cloud TFLOPS or own a private pod, factoring in hardware failure (MTBF)?"
*   **The Workbench:**
    *   **Interaction:** **Ownership Toggle**. **MTBF Slider** (Hours between failures). **Electricity Price ($/kWh)**.
    *   **The "Stakeholder" Challenge:** The **CFO** demands a 20% reduction in OpEx. You must use the **TCO Waterfall** to prove that investing in "Efficient Power Delivery" saves more than "Cheaper Nodes."
*   **Reflect (The Ledger):** "Defend your final 'Fleet Infrastructure' choice. Did you prioritize 'Density' or 'Efficiency'? Justify why 'Availability' was your primary constraint."

---

## 4. Visual Layout Specification
*   **Primary:** `PUE_Waterfall` (Compute Power vs. Cooling Power vs. Distribution Loss).
*   **Secondary:** `ThermalSafeZonePlot` (Density vs. Temperature).
*   **Math Peek:** Toggle for `PUE = \frac{	ext{Total Facility Power}}{	ext{IT Equipment Power}}`.

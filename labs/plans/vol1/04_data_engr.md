# 📐 Mission Plan: 04_data_engr (Deep Analysis)

## 1. Chapter Context
*   **Chapter Title:** Data Engineering: Dataset Compilation.
*   **Core Invariant:** Data Gravity ($T = D_{vol}/BW$) and the Energy-Movement Invariant ($E_{move} \gg E_{comp}$).
*   **The Struggle:** Balancing the "Feeding Tax"—ensuring the data pipeline can keep up with the GPU's consumption rate without destroying the energy budget.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Data Gravity" |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving on a single H100. | **The Feeding Tax.** Disk I/O cannot keep up with HBM speeds. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop on NVIDIA Orin. | **The Ingestion Choke.** 8 raw 4K vision streams flood the bus. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation on Meta Ray-Bans. | **Transmission Energy.** Moving bits over Bluetooth drains glasses. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **SRAM Budget.** Buffering audio consumes 50% of total memory. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Data Gravity Audit (Exploration - 15 Mins)
*   **Objective:** Dimension the physical and economic cost of moving the mission's dataset.
*   **The "Lock" (Prediction):** "Will it be cheaper to stream your data over Fiber or ship a physical hard drive across the country?"
*   **The Workbench:**
    *   **Sliders:** Dataset Size (10GB -> 10PB), Distance (km), Link Bandwidth (10G -> 100G).
    *   **Instruments:** `TransferTimeRadar`, `SneakernetCrossoverPlot` (Time vs Distance).
    *   **The 5-Move Rule:** Students must analyze 5 different scale tiers to identify the "Distance Invariant" where each path wins.
*   **Reflect:** "Reconcile the transfer time with the 'Physics of Data Gravity' from the text. When does bit-volume become a physical barrier?"

### Part 2: The Feeding Tax Solver (Trade-off - 20 Mins)
*   **Objective:** Maximize GPU Model FLOPS Utilization (MFU) by optimizing the serialization pipeline.
*   **The "Lock" (Prediction):** "If you switch from JSON to Protobuf, will your GPU utilization increase more than if you upgrade to a faster SSD?"
*   **The Workbench:**
    *   **Sliders:** Serialization Format (CSV, JSON, Parquet, Protobuf), Worker Count (1-32), Disk Type (HDD -> NVMe).
    *   **Instruments:** `FeedingTaxGauge` (% GPU Idle), `MFU_vs_Ingestion_Plot`.
    *   **The 15-Iteration Rule:** Students must find the exact "Flow Equilibrium" where the CPU's pre-processing rate matches the GPU's consumption rate.
*   **Reflect:** "Your GPU is 80% idle. Prove whether the bottleneck is in the 'Blueprint' (Algorithm) or the 'Fuel' (Data pipeline) using the MFU plot."

### Part 3: The Zero-Waste Audit (Synthesis - 10 Mins)
*   **Objective:** Maximize 'Data Selection Gain' to hit accuracy targets within a carbon/energy budget.
*   **The "Lock" (Prediction):** "Is it more energy-efficient to use 1 million noisy samples or 10,000 curated 'Gold Standard' samples?"
*   **The Workbench:**
    *   **Sliders:** Filtering Ratio (0-90%), Label Quality (Low -> Expert), Processing Location (Local vs Cloud).
    *   **The "Stakeholder" Challenge:** The **Sustainability Lead** demands a 50% reduction in transmission energy. The student must use the **Energy-Movement Invariant** to propose an architectural change (e.g. local pre-processing).
*   **Reflect (The Ledger):** Justify your final Data/Compute energy ratio. Explain why "Signal-to-Noise Engineering" is more effective than raw scaling for this mission.

---

## 4. Visual Layout Specification
*   **Primary:** `IngestionWaterfall` (Storage BW vs. Network BW vs. Compute rate).
*   **Secondary:** `EnergyRadar` (MAC pJ vs. DRAM pJ vs. Network pJ).
*   **Transparency:** Toggle for `Data Selection Gain \propto \frac{\text{Entropy}}{\text{Gravity}}`.

# 📐 Mission Plan: 04_data_storage (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Data Storage: Feeding the Machine Learning Fleet.
*   **Core Invariant:** The Sequential Invariant (Random I/O is the enemy of throughput) and the **I/O Wall**.
*   **The Struggle:** Understanding that at scale, storage is about **IOPS** and **Bandwidth**, not just capacity. Students must navigate the trade-off between **Data Locality** (Local NVMe) and **Shared Scalability** (Object Stores/S3), specifically focusing on how random shuffling kills training performance.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Storage Missions)

| Track | Persona | Fixed North Star Mission | The "Storage" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Checkpoint Storm.** Your 1024-node cluster is trying to save a 350GB checkpoint simultaneously. The shared storage has collapsed under the write-pressure. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Black-Box Log.** Your 10,000-vehicle fleet is generating 5TB/hour of Lidar data. You must decide what to log locally vs. what to upload to the Cloud. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The App Cache Wall.** The 8GB glasses RAM is full. You must stream model weights from flash memory without causing a 50ms frame-skip. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Circular Buffer.** You have only 64KB of audio buffer. If your Flash-read latency is inconsistent, the audio 'glitches' for the user. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Access Pattern Audit (Exploration - 15 Mins)
*   **Objective:** Quantify the 100x performance difference between Sequential and Random I/O.
*   **The "Lock" (Prediction):** "If you randomly shuffle a 10TB dataset during each epoch, will your training throughput be limited by your GPU or your Storage IOPS?"
*   **The Workbench:**
    *   **Action:** Toggle between **Sequential Reading** and **Stochastic Shuffling**. Adjust **File Format** (Raw Files vs. TFRecord/WebDataset).
    *   **Observation:** The **I/O Waterfall** (Wait-Time vs. Load-Time). Watch the "I/O Wait" bar explode during random shuffling.
*   **Reflect:** "Patterson asks: 'Why is Sequential access the only way to hit the 'Machine' peak?' (Reference the disk-head/block-prefetching physics)."

### Part 2: Sizing the Pipeline (Trade-off - 15 Mins)
*   **Objective:** Dimension a tiered storage hierarchy (S3 -> NVMe -> DRAM) to hit a specific throughput target.
*   **The "Lock" (Prediction):** "Will adding more Local NVMe cache improve training speed if the bottleneck is the initial S3-to-Node network link?"
*   **The Workbench:**
    *   **Sliders:** Buffer Size (GB), Download BW (Gbps), Local NVMe BW (GB/s).
    *   **Instruments:** **Data Flow Gauge**. **Pipeline Saturation Plot**.
    *   **The 10-Iteration Rule:** Students must find the "Balanced Tiering" that keeps the GPU 95% utilized for their track's specific dataset size.
*   **Reflect:** "Jeff Dean observes: 'Your storage system is 50% idle while your GPUs are starving.' Identify the 'Impedance Mismatch' in your pipeline."

### Part 3: The Checkpoint Wall (Synthesis - 15 Mins)
*   **Objective:** Optimize the Checkpoint Interval to minimize the "Reliability Tax."
*   **The "Lock" (Prediction):** "Does saving a checkpoint every 10 minutes increase or decrease the total time to finish a 1-month training run?"
*   **The Workbench:**
    *   **Interaction:** **Checkpoint Frequency Slider**. **Write-Bandwidth Selector**. **MTBF (Mean Time Between Failures) Scrubber**.
    *   **The "Stakeholder" Challenge:** The **Ops Lead** warns that the MTBF of the cluster has dropped. You must use the **Young-Daly Plot** to find the optimal checkpoint frequency that minimizes "Wasted Work" without crashing the storage.
*   **Reflect (The Ledger):** "Defend your final 'Storage Strategy.' Did you choose 'Local-First' or 'Cloud-Native'? Justify how you solved the 'Feeding Problem' for your fleet."

---

## 4. Visual Layout Specification
*   **Primary:** `DataFlowSankey` (Visualizing bits moving from Cloud -> Disk -> GPU).
*   **Secondary:** `IOPS_vs_Throughput_Curve` (Showing the saturation point of different disk types).
*   **Math Peek:** Toggle for the `Data Pipeline Equation` and `Young-Daly Checkpoint Interval`.

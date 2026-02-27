# 📐 Mission Plan: 05_dist_train (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Distributed Training: The Parallelism Paradox.
*   **Core Invariant:** The Parallelism Paradox (More nodes, more overhead) and **MFU** (Model FLOPs Utilization).
*   **The Struggle:** Understanding that adding GPUs does not linearly reduce training time. Students must navigate the trade-off between **Model Parallelism** (fitting the model) and **Data Parallelism** (scaling the throughput), specifically focusing on the "Communication Wall."
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Distributed Missions)

| Track | Persona | Fixed North Star Mission | The "Parallelism" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The 3D Wall.** To train Llama-3, you must combine DP, PP, and TP. Your 'Pipeline Bubbles' are wasting 30% of your H100s. You must re-dimension the cluster. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Fleet Learning Gap.** You are training on logs from 10,000 cars. The 'Stale Gradients' from slow 5G links are causing the model to diverge during training. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Federated Power Wall.** You are training a personalized filter across a mesh of 1,000 Ray-Bans. The 'Global Sync' is draining batteries 5x faster than local inference. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Swarm Bottleneck.** You have 100 hearables collaborating to learn a noise profile. The BLE bandwidth is too low for a full weight-sync. You must use 'Gossip' protocols. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Communication Wall Audit (Exploration - 15 Mins)
*   **Objective:** Quantify the 'Scaling Efficiency' ($\eta_{scaling}$) as the fleet grows.
*   **The "Lock" (Prediction):** "If you move from 8 GPUs to 64 GPUs, will your training time decrease by exactly 8x? Why or why not?"
*   **The Workbench:**
    *   **Action:** Slide the **Number of Nodes** ($N$). Toggle between **Ethernet (10G)** and **InfiniBand (400G)**.
    *   **Observation:** The **Scaling Efficiency Curve**. Watch the throughput deviate from the "Ideal" line as $N$ increases.
*   **Reflect:** "Patterson asks: 'Identify the point of diminishing returns.' At what node count does adding a GPU provide less than a 0.5x marginal speedup?"

### Part 2: The Parallelism Puzzle (Trade-off - 15 Mins)
*   **Objective:** Balance Data Parallelism (DP) and Model Parallelism (PP/TP) to fit a model in memory while maximizing MFU.
*   **The "Lock" (Prediction):** "Does 'Pipeline Parallelism' (PP) increase or decrease the total memory footprint of your model weights across the cluster?"
*   **The Workbench:**
    *   **Sliders:** DP Degree, PP Degree, TP Degree. Micro-batch size.
    *   **Instruments:** **Parallelism Memory Heatmap**. **Pipeline Bubble Gauge**.
    *   **The 10-Iteration Rule:** Students must find the exact "3D Parallelism" configuration that fits their track's model into HBM while keeping "Bubble Time" below 10%.
*   **Reflect:** "Jeff Dean observes: 'Your MFU is only 20%.' Propose a change to the **Micro-batch** count to reclaim the idle GPU cycles."

### Part 3: The Synchronization Audit (Synthesis - 15 Mins)
*   **Objective:** Choose between Synchronous and Asynchronous SGD based on fleet-scale noise.
*   **The "Lock" (Prediction):** "In a fleet with 5% 'Straggler' nodes (slow GPUs), will Async SGD achieve a higher samples-per-second than Sync SGD?"
*   **The Workbench:** 
    *   **Interaction:** **Sync/Async Toggle**. **Straggler Intensity Slider**. **Check-pointing Cost**.
    *   **The "Stakeholder" Challenge:** The **Infrastructure Lead** warns that "Wait-for-All" synchronization is wasting $1M per month. You must prove that switching to **ZeRO-Redundancy Optimizer** (ZeRO) saves more than switching to Async updates.
*   **Reflect (The Ledger):** "Defend your final 'Distributed Strategy.' Did you prioritize 'Gradient Consistency' or 'Raw Throughput'? Justify how you bypassed the Parallelism Paradox."

---

## 4. Visual Layout Specification
*   **Primary:** `ScalingEfficiencyPlot` (Samples/sec vs. Number of Accelerators).
*   **Secondary:** `3DParallelismMap` (Visualizing how the model is sliced across Nodes, Racks, and Pods).
*   **Math Peek:** Toggle for `MFU = \frac{O_{actual}}{O_{peak}}` and `Pipeline Bubble = \frac{P-1}{P+M-1}`.

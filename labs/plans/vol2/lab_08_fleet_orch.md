# 📐 Mission Plan: 08_fleet_orch (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Fleet Orchestration: The Cluster Scheduler.
*   **Core Invariant:** The Allocation Invariant (Packing Efficiency vs. Interconnect Locality) and the **Utilization Paradox**.
*   **The Struggle:** Understanding that at scale, "Scheduling is the Compiler of the Cluster." Students must navigate the trade-off between **Bin Packing** (maximizing node density) and **Topology Awareness** (minimizing network hops), specifically focusing on how fragmentation kills large-job throughput.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Orchestration Missions)

| Track | Persona | Fixed North Star Mission | The "Orchestration" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Fragmentation Trap.** You have 100 free GPUs, but they are scattered across 50 racks. Your 64-GPU 'Gang Job' is stuck in the queue because it needs a contiguous 'Rail-Optimized' block. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Topology Lottery.** Your AV perception job was scheduled on two nodes separated by 3 network hops. The AllReduce latency just jumped from 1ms to 15ms, breaking the safety loop. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Elastic Preemption.** You are using 'Spot Instances' to save 80% on cloud rendering. The provider just reclaimed your nodes mid-frame. You must implement 'Dynamic Re-scaling'. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Heterogeneous Mess.** Your fleet has V1, V2, and V3 hearables. The scheduler is sending heavy models to low-power nodes, causing 100% 'Job Dropout'. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Fragmentation Audit (Exploration - 15 Mins)
*   **Objective:** Quantify the "Scheduling Delay" caused by resource fragmentation.
*   **The "Lock" (Prediction):** "If a cluster is 90% full, will a job requiring 8 contiguous GPUs wait longer or shorter than a job requiring 8 random GPUs?"
*   **The Workbench:**
    *   **Action:** Toggle between **Best-Fit** and **Random** placement. Adjust **Cluster Load** (0-100%).
    *   **Observation:** The **Cluster Heatmap**. Watch "Holes" appear in the racks.
*   **Reflect:** "Patterson asks: 'Why is fragmentation an economic killer for the Cloud Titan?' (Reference the opportunity cost of idle silicon)."

### Part 2: The Topology Penalty (Trade-off - 15 Mins)
*   **Objective:** Balance "Fairness" (sharing the cluster) vs. "Locality" (keeping jobs on one switch).
*   **The "Lock" (Prediction):** "Does 'Gang Scheduling' increase or decrease the average wait-time for all users in the fleet?"
*   **The Workbench:**
    *   **Sliders:** Locality Constraint (Strict -> Loose), Quota Limits, Priority Level.
    *   **Instruments:** **Throughput-vs-WaitTime Seesaw**. **Network Hop Gauge**.
    *   **The 10-Iteration Rule:** Students must find the "Scheduling Policy" that maximizes cluster MFU without letting "Low-Priority" jobs starve for more than 24 hours.
*   **Reflect:** "Jeff Dean observes: 'Your scheduler is fair, but our training goodput is down 40% due to cross-rack traffic.' Propose a 'Topology-Aware' placement rule."

### Part 3: The Elastic Buffer (Synthesis - 15 Mins)
*   **Objective:** Design an "Elastic Training" strategy that survives node preemption.
*   **The "Lock" (Prediction):** "If 20% of your nodes are 'Spot Instances' that can be killed at any time, should your 'AllReduce' ring be static or elastic?"
*   **The Workbench:**
    *   **Interaction:** **Spot Intensity Slider**. **Enable Elastic-Horovod**. **Preemption Frequency Scrubber**.
    *   **The "Stakeholder" Challenge:** The **CFO** wants to move 100% of training to Spot Instances to save $5M. You must prove that without **Elastic Checkpointing**, the "Time-to-Converge" will actually double due to constant restarts.
*   **Reflect (The Ledger):** "Defend your final 'Orchestration Policy.' Did you prioritize 'Packing Density' or 'Job Locality'? Justify how you bypassed the Utilization Paradox."

---

## 4. Visual Layout Specification
*   **Primary:** `ClusterHeatmap` (Visualizing GPU occupancy and Job fragmentation).
*   **Secondary:** `WaitTime_vs_MFU_Plot` (Showing the trade-off between user happiness and hardware efficiency).
*   **Math Peek:** Toggle for `Dominant Resource Fairness (DRF)` and `Bin Packing` complexity.

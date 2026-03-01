# 📐 Mission Plan: 03_network_fabrics (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Network Fabrics: The Cluster Interconnect.
*   **Core Invariant:** The Bisection Bandwidth Wall and the **Alpha-Beta Framework** ($T = \alpha + \beta \cdot M$).
*   **The Struggle:** Understanding that at scale, the "Wire" is the bottleneck. Students must navigate the trade-off between **Network Topology** (Fat-Tree, Torus, Rail-Optimized) and **Message Latency** ($\alpha$), specifically focusing on the "Bandwidth Cliff" between intra-node and inter-node communication.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Networking Missions)

| Track | Persona | Fixed North Star Mission | The "Fabric" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Bisection Choke.** Your 1024-GPU cluster is spending 60% of its time in 'AllReduce' because the top-of-rack switches are oversubscribed. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Incast Storm.** Your fleet of 50 drones is reporting back to a single Edge Hub. The resulting 'Incast' is causing a 50ms buffer-bloat spike, violating safety SLAs. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Global Metaverse Latency.** You are offloading AR rendering to a 5G Edge server. The 'Alpha' (start-up latency) of the cell tower is jittering between 10ms and 100ms. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Mesh Contention.** You have a mesh of 100 hearables sharing a single BLE channel. Collision overhead is killing the audio stream. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Bandwidth Cliff Audit (Exploration - 15 Mins)
*   **Objective:** Quantify the 10x-100x gap between NVLink (Intra-node) and InfiniBand/Ethernet (Inter-node) bandwidth.
*   **The "Lock" (Prediction):** "If moving a model weight between GPUs on the SAME node takes 1ms, how long will it take between GPUs on DIFFERENT racks?"
*   **The Workbench:**
    *   **Action:** Slide the **Communication Locality** (Percentage of data moved off-node).
    *   **Observation:** The **Bandwidth Cliff Plot**. Watch the transfer time jump as data hits the "Inter-Rack" hop.
*   **Reflect:** "Patterson asks: 'Why is data locality 10x more important in Volume 2 than it was in Volume 1?' (Reference the Alpha-Beta model)."

### Part 2: The Topology Race (Trade-off - 15 Mins)
*   **Objective:** Select the optimal Network Topology (Fat-Tree, Torus, Dragonfly) for your track's communication pattern.
*   **The "Lock" (Prediction):** "Which topology will have the lowest 'Bisection Bandwidth' for a 10,000-node cluster: 3D Torus or Non-blocking Fat-Tree?"
*   **The Workbench:**
    *   **Interaction:** Toggle between **Fat-Tree**, **Torus**, and **Rail-Optimized** fabrics. Adjust **Cluster Scale** ($N$).
    *   **Instruments:** **Fabric Throughput Gauge**. **Congestion Heatmap**.
    *   **The 10-Iteration Rule:** Students must find the exact "Oversubscription Ratio" that keeps the cost under budget without causing "AllReduce Saturation."
*   **Reflect:** "Jeff Dean observes: 'Your Fat-Tree is too expensive, but your Torus has too many hops.' Propose a 'Rail-Optimized' layout for your LLM pods and justify the cabling complexity."

### Part 3: Congestion Control Audit (Synthesis - 15 Mins)
*   **Objective:** Configure RDMA/RoCE congestion control (DCQCN vs. HPCC) to eliminate "PFC Storms."
*   **The "Lock" (Prediction):** "If one node is slow, will it cause 'Head-of-Line' blocking for all other nodes in a lossless network?"
*   **The Workbench:**
    *   **Interaction:** **RDMA Toggle**. **Congestion Control Level**. **Incast Intensity**.
    *   **The "Stakeholder" Challenge:** The **Networking Lead** warns that "Pause Frames" are killing the fleet throughput. You must tune the ECN (Explicit Congestion Notification) thresholds to maintain "Deterministic Jitter."
*   **Reflect (The Ledger):** "Defend your final 'Fabric Configuration.' Did you prioritize 'Peak Throughput' or 'Tail Latency'? Justify why the 'Alpha' term was the killer for your specific track."

---

## 4. Visual Layout Specification
*   **Primary:** `BandwidthCliffPlot` (Throughput vs. Communication Distance).
*   **Secondary:** `NetworkTopologyVisualizer` (Mermaid.js diagram of switches and nodes).
*   **Math Peek:** Toggle for `T_{comm} = \alpha + \frac{M}{\beta}` and `Bisection Bandwidth` formulas.

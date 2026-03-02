# 📐 Mission Plan: 06_collect_comms (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Collective Communication: The Orchestration of Data.
*   **Core Invariant:** The Communication-Compute Invariant (Ring vs. Tree AllReduce) and the **Alpha-Beta model**.
*   **The Struggle:** Understanding that "AllReduce" is not a single operation, but a choice between **Bandwidth-Optimal** (Ring) and **Latency-Optimal** (Tree) strategies. Students must navigate the trade-off between **Message Size** ($M$) and **Cluster Scale** ($N$) to find the "Algorithm Crossover Point."
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Communication Missions)

| Track | Persona | Fixed North Star Mission | The "Collective" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Parameter Flood.** Your Ring AllReduce is 100% bandwidth-saturated, but training is still slow. You must determine if a Tree topology can reduce the 'Alpha' overhead. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Jittery Mesh.** Your 10 vehicle-SoCs are syncing perception gradients over a noisy wireless link. Tree AllReduce is failing due to high 'Alpha' variance. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Battery Contention.** Sending full gradients over Bluetooth is killing the AR frames. You must implement '1-bit Compression' with error feedback. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Low-Power Gossip.** You have 50 hearables. A full AllReduce is too expensive. You must analyze the trade-off between 'Gossip' rounds and 'Convergence Quality'. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Algorithm Crossover (Exploration - 15 Mins)
*   **Objective:** Identify the "Crossover Point" ($M_{crossover} \approx N \alpha / \beta$) where Tree AllReduce beats Ring AllReduce.
*   **The "Lock" (Prediction):** "For a fixed cluster size, does the Tree algorithm become more or less attractive as the model size ($M$) increases?"
*   **The Workbench:**
    *   **Action:** Toggle between **Ring** and **Tree** algorithms. Slide the **Message Size** ($M$) and **Node Count** ($N$).
    *   **Observation:** The **AllReduce Performance Plot**. Watch the crossover point move as the Alpha (Latency) and Beta (Bandwidth) parameters change.
*   **Reflect:** "Patterson asks: 'Why is Ring AllReduce considered bandwidth-optimal but latency-inefficient?' (Reference the $2(N-1)\alpha$ term)."

### Part 2: The Hierarchical Multiplier (Trade-off - 15 Mins)
*   **Objective:** Optimize a 2-tier "Hierarchical AllReduce" (NVLink intra-node + InfiniBand inter-node).
*   **The "Lock" (Prediction):** "Will a hierarchical strategy reduce total sync time if the inter-node link is 100x slower than the intra-node link?"
*   **The Workbench:**
    *   **Interaction:** Adjust **Intra-node BW** vs. **Inter-node BW**. Toggle **Hierarchical Mode**.
    *   **Instruments:** **Communication Waterfall** (Intra-sync vs Inter-sync time).
    *   **The 10-Iteration Rule:** Students must find the optimal "Aggregation Factor" that minimizes the time spent on the slow inter-rack links.
*   **Reflect:** "Jeff Dean observes: 'Your intra-node link is 90% idle while the network is 100% saturated.' Propose a 'Gradient Accumulation' change to balance the two."

### Part 3: Compression & Error Feedback (Synthesis - 15 Mins)
*   **Objective:** Implement 1-bit Gradient Compression to bypass the bandwidth wall without breaking convergence.
*   **The "Lock" (Prediction):** "If you compress gradients to 1-bit (32x reduction), what happens to the 'Accuracy' of the next training step?"
*   **The Workbench:**
    *   **Interaction:** **Compression Ratio Slider**. **Error Feedback Toggle**. **Convergence Monitor**.
    *   **The "Stakeholder" Challenge:** The **ML Lead** refuses to use 1-bit compression because the model is diverging. You must prove that enabling **Error Feedback** (residual accumulation) recovers the accuracy loss while maintaining the 32x bandwidth gain.
*   **Reflect (The Ledger):** "Defend your final 'Communication Strategy.' Did you choose 'Raw Speed' (Ring) or 'Rigor' (Sync + Compression)? Justify using the Communication-to-Compute ratio."

---

## 4. Visual Layout Specification
*   **Primary:** `CommunicationCrossoverPlot` (Execution Time vs. Message Size for different algorithms).
*   **Secondary:** `NetworkTopologyHUD` (Visualizing the Ring or Tree data flow across nodes).
*   **Math Peek:** Toggle for `T_{ring} = 2(N-1)\frac{M}{N\beta} + 2(N-1)\alpha`.

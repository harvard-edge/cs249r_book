# 📐 Mission Plan: 01_ml_intro (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Introduction: Machine Learning at Scale.
*   **Core Invariant:** The Fleet-Node Divergence and the **Reliability Gap** ($P_{fleet} = P_{node}^N$).
*   **The Struggle:** Moving from "How do I make it work?" (Single Node) to "How do I make it scale and survive?" (The Fleet). Understanding that at scale, hardware failure is a statistical certainty, and communication replaces computation as the primary bottleneck.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Fleet Scale Expansion)

| Track | Persona | Fixed North Star Mission | The "Scale" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The 25k GPU Wall.** You are scaling from one node to a pod of 25,000 GPUs. The probability of zero failures during training is effectively zero. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Global Fleet Sync.** You are managing 500,000 autonomous taxis. One software bug now propagates across a continental-scale network in seconds. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The 1 Billion Device Update.** You are pushing a new model to 1 billion Ray-Bans. The "Fleet Law" dictates that your cloud egress will cost more than the model development. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Mesh Congestion.** You have deployed 10,000 smart rings in a stadium. The local gateway is crushed by 10,000 simultaneous telemetry heartbeats. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Reliability Gap (Exploration - 15 Mins)
*   **Objective:** Quantify the collapse of system availability as the number of nodes ($N$) increases.
*   **The "Lock" (Prediction):** "If a single node has a 99.9% uptime (Three Nines), what is the probability that all 1,000 nodes in your cluster are up simultaneously?"
*   **The Workbench:**
    *   **Action:** Slide the **Node Count** ($N$) from 1 to 10,000. Adjust the **Per-Node Reliability** ($P_{node}$).
    *   **Observation:** The **Fleet Availability Waterfall**. Watch the "Uptime" bar vanish exponentially as $N$ grows.
*   **Reflect:** "Reconcile the result with the **Reliability Gap** formula. Why does scaling a 'reliable' node lead to an 'unreliable' fleet?"

### Part 2: The Communication Intensity Wall (Trade-off - 15 Mins)
*   **Objective:** Identify the point where adding more compute nodes decreases total throughput (The Scaling Inversion).
*   **The "Lock" (Prediction):** "Will adding 2x more GPUs always result in 2x faster training?"
*   **The Workbench:**
    *   **Sliders:** Number of GPUs, Model Synchronization Frequency, Link Bandwidth ($BW_{network}$).
    *   **Instruments:** **Compute-Communication Seesaw**. A chart showing "Goodput" (useful math) vs "Rawput" (total math).
    *   **The 10-Iteration Rule:** Students must find the exact "Critical Node Count" where the communication overhead exceeds the compute gain for their track's specific model.
*   **Reflect:** "Patterson asks: 'What happens to the **CI Ratio** (Communication Intensity) as you scale?' Explain why the network, not the GPU, is now your primary constraint."

### Part 3: The C³ Design Review (Synthesis - 15 Mins)
*   **Objective:** Balance the **C³ Taxonomy** (Compute, Communication, Coordination) to hit a fleet-scale target.
*   **The "Lock" (Prediction):** "To reduce 'Coordination Overhead', should you use Synchronous (faster convergence) or Asynchronous (higher throughput) updates?"
*   **The Workbench:**
    *   **Interaction:** Toggle between **Sync/Async SGD**. Adjust **Topology** (Ring vs. Tree).
    *   **The "Stakeholder" Challenge:** **Jeff Dean** (Systems Lead) warns that the tail latency of the slowest GPU is killing the entire cluster. You must implement **Speculative Execution** or **Staggard Checkpoints** to save the mission.
*   **Reflect (The Ledger):** "Defend your final 'Fleet Architecture.' Did you prioritize 'Reliability' or 'Speed'? Justify how you bridged the **Fleet-Node Divergence**."

---

## 4. Visual Layout Specification
*   **Primary:** `AvailabilityCollapsePlot` (Availability vs. Node Count).
*   **Secondary:** `CI_Ratio_Gauge` (Calculating the ratio of bits moved to ops performed).
*   **Math Peek:** Toggle for the **Fleet Law** and the **C³ Taxonomy**.

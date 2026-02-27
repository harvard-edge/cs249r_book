# 📐 Mission Plan: 11_edge_intel (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Edge Intelligence: On-Device Learning & Federated Fleets.
*   **Core Invariant:** The Locality Invariant (Data locality is the only path to real-time privacy) and the **Memory Amplification Factor**.
*   **The Struggle:** Understanding that training on the edge is 4-8x more memory-intensive than inference. Students must navigate the trade-off between **Local Personalization** (LoRA/TinyTL) and **Global Convergence** (Federated Learning), specifically focusing on how the "Communication-to-Compute" ratio shifts in unreliable mesh networks.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Edge Missions)

| Track | Persona | Fixed North Star Mission | The "Edge Intel" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Personalization Flood.** You are managing 1 million custom LoRA adapters for different users. The 'Swap Latency' between HBM and SSD is killing your throughput. You must optimize the 'Adapter Cache'. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Fleet Learning Drift.** Your 500,000 AVs are learning to detect 'Dust Storms' locally. The 'Non-IID' data (different cities) is causing the global model to diverge. You must tune 'Federated Averaging'. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Battery-Privacy Paradox.** Users want their glasses to learn their specific accent locally. But on-device training drains the battery 5x faster than inference. You must use 'TinyTL' (Tiny Transfer Learning). |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Mesh Update.** 10,000 hearables in a mesh are trying to sync noise-profiles. The BLE bandwidth is too low for full sync. You must decide between 'Gossip' updates and 'Local-Only' learning. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Memory-Accuracy Frontier (Exploration - 15 Mins)
*   **Objective:** Quantify the "Memory Tax" of on-device adaptation (LoRA vs. Full Fine-Tuning).
*   **The "Lock" (Prediction):** "If inference requires 512MB, what is the minimum RAM required to fine-tune the last 3 layers of your model on-device?"
*   **The Workbench:**
    *   **Action:** Toggle between **Full Fine-Tuning**, **LoRA**, and **TinyTL**. Adjust **Rank ($r$)** of adapters.
    *   **Observation:** The **On-Device Memory Waterfall**. Watch the "Optimizer State" and "Gradients" bars dwarf the "Weights" bar.
*   **Reflect:** "Patterson asks: 'Why is the Memory Wall higher for Edge Intel than for Edge Inference?' (Reference the **Memory Amplification Factor**)."

### Part 2: The Non-IID Convergence (Trade-off - 15 Mins)
*   **Objective:** Balance local iteration speed vs. global model quality in a Federated Learning (FL) fleet.
*   **The "Lock" (Prediction):** "Will increasing the 'Communication Rounds' (syncing more often) always improve the accuracy of a fleet with extremely different local data distributions?"
*   **The Workbench:**
    *   **Sliders:** Local Epochs per Round, Number of Participating Clients, Link Reliability (%).
    *   **Instruments:** **Fleet Convergence Plot**. Watch the "Global Loss" curve jitter as data becomes more Non-IID (Independent and Identically Distributed).
    *   **The 10-Iteration Rule:** Students must find the "Optimal Round Frequency" that hits the mission accuracy target without exceeding the track's fixed 5G data cap.
*   **Reflect:** "Jeff Dean observes: 'Your fleet is wasting 80% of its power on failed syncs.' Propose a 'Client Selection' strategy to save the energy budget."

### Part 3: The Privacy Frontier (Synthesis - 15 Mins)
*   **Objective:** Implement Differential Privacy (DP) to satisfy strict data governance laws.
*   **The "Lock" (Prediction):** "Does adding noise ($\epsilon$) to your gradients increase or decrease the total compute required to reach convergence?"
*   **The Workbench:** 
    *   **Interaction:** **Privacy Epsilon ($\epsilon$) Slider**. **Clip-Norm Scrubber**. **Local vs. Global Aggregation Toggle**.
    *   **The "Stakeholder" Challenge:** The **Privacy Officer** demands $\epsilon < 1.0$. You must find an architecture that satisfies this without dropping the model's 'Safety-Critical' accuracy below the mission SPEC.
*   **Reflect (The Ledger):** "Defend your final 'Edge Intelligence Strategy.' Did you prioritize 'Personalization' or 'Privacy'? Justify why 'Data Locality' was your only path to real-time reliability."

---

## 4. Visual Layout Specification
*   **Primary:** `ConvergenceVsCommunicationPlot` (X-axis: GB Transferred, Y-axis: Fleet Accuracy).
*   **Secondary:** `OnDeviceMemoryMap` (Visualizing Weight/Gradient/Optimizer/Activation buffers in RAM).
*   **Math Peek:** Toggle for `Memory_{train} \approx Memory_{weights} \cdot (1 + 	ext{Multipliers})` and `FedAvg` logic.

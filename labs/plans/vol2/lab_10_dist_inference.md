# 📐 Mission Plan: 10_dist_inference (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Distributed Inference: Fleet-Scale Serving.
*   **Core Invariant:** The Serving Invariant (P99 Latency vs. Throughput Efficiency) and the **Serving Cost Dominance Law** (OpEx >> CapEx).
*   **The Struggle:** Understanding that at scale, "The Queue is the Model." Students must navigate the trade-off between **Request Isolation** (low latency) and **Batch Saturation** (low cost), specifically focusing on how **Continuous Batching** and **PagedAttention** bypass the KV-Cache Wall.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Inference Missions)

| Track | Persona | Fixed North Star Mission | The "Serving" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The KV-Cache Wall.** Your H100s are only 20% utilized because fragmentation in the KV-cache is causing premature OOM. You must implement 'PagedAttention' to reclaim 40% of your VRAM. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Fan-out Tail.** Your perception loop now queries 10 parallel sub-models. The slowest sub-model's jitter is causing the total response time to fail the 10ms SLA. You must use 'Speculative Execution'. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Offload Jitter.** You are offloading AR reasoning to a fleet of Edge nodes. The variable 'Alpha' (start-up latency) of the WiFi-6 mesh is causing AR frame-stutter. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Power-Latency Seesaw.** You are serving a noise-isolation fleet. Higher batching saves gateway power but adds 50ms of delay, causing 'Echo' for the user. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Throughput Knee (Exploration - 15 Mins)
*   **Objective:** Predict and measure the point of system collapse using Queuing Theory.
*   **The "Lock" (Prediction):** "If you increase the request rate ($\lambda$) to 90% of your maximum capacity, does the P99 latency increase linearly or exponentially?"
*   **The Workbench:**
    *   **Action:** Slide the **Arrival Rate** ($\lambda$). Adjust the **Batch Window**.
    *   **Observation:** The **Latency-Throughput Pareto Curve**. Watch the "Knee of the Curve" where latency explodes.
*   **Reflect:** "Patterson asks: 'Why is 80% utilization the practical ceiling for a responsive system?' (Reference the $M/M/1$ queue math)."

### Part 2: Sharding the Heavyweight (Trade-off - 15 Mins)
*   **Objective:** Balance Tensor Parallelism (TP) vs. Pipeline Parallelism (PP) for latency-sensitive serving.
*   **The "Lock" (Prediction):** "Does 'Tensor Parallelism' (sharding weights) reduce the latency of a single request more than 'Pipeline Parallelism' (sharding layers)?"
*   **The Workbench:**
    *   **Interaction:** Adjust **TP Degree** vs. **PP Degree**. Toggle **Continuous Batching**.
    *   **Instruments:** **Latency Component Waterfall** (Compute vs. Communication vs. Bubbles).
    *   **The 10-Iteration Rule:** Students must shard a 70B model across 8 GPUs to hit a 50ms 'Time-to-First-Token' (TTFT) target.
*   **Reflect:** "Jeff Dean observes: 'Your sharding strategy is fast, but your bisection bandwidth is 100% saturated.' Propose a 'Weight-Gather' optimization to reduce the network tax."

### Part 3: The Memory Wall (Synthesis - 15 Mins)
*   **Objective:** Optimize KV-Cache management to maximize user concurrency.
*   **The "Lock" (Prediction):** "If you use 'PagedAttention' to eliminate internal fragmentation, how many more concurrent users can you fit in 80GB of HBM?"
*   **The Workbench:**
    *   **Interaction:** **Fragmentation Slider**. **KV-Cache Eviction Policy**. **Request Preemption Budget**.
    *   **The "Stakeholder" Challenge:** The **CFO** demands a 50% reduction in 'Cost-per-User'. You must implement **Speculative Decoding** to reduce the 'Tokens-per-Second' cost without regressing on P99 latency.
*   **Reflect (The Ledger):** "Defend your final 'Fleet Serving Strategy.' Did you prioritize 'Throughput' (Continuous Batching) or 'Responsiveness' (Zero-Batching)? Justify how you solved the 'Tail at Scale' problem."

---

## 4. Visual Layout Specification
*   **Primary:** `LatencyThroughputFrontier` (X-axis: QPS, Y-axis: P99 Latency).
*   **Secondary:** `KVCacheHeatmap` (Visualizing memory occupancy and fragmentation).
*   **Math Peek:** Toggle for `Serving Cost Dominance Law` and `TTFT vs TPOT` metrics.

# 📐 Mission Plan: 13_model_serving (Model Serving)

## 1. Chapter Context
*   **Chapter Title:** Model Serving: Deployment at Scale.
*   **Core Invariant:** Little's Law ($L = \lambda \cdot W$) and the Batching Paradox.
*   **The Struggle:** Understanding that serving is a queuing problem, not just a compute problem. Students must navigate the trade-off between **Throughput** (maximizing users per second) and **Latency** (minimizing milliseconds per user), specifically focusing on the "Tail at Scale."
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Serving" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The TTFT Wall.** Your 'Time to First Token' is 5 seconds. Users are leaving. You must implement Continuous Batching and KV-Caching to save the UX. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Concurrency Jitter.** When multiple sensors trigger simultaneous vision requests, the queue depth (L) explodes, violating the 10ms safety window. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Frame-Skip Crisis.** Processing one frame takes 15ms, but wait-time in the queue adds 5ms. You are dropping to 30FPS. You must eliminate the serving overhead. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Memory-Queuing Wall.** You have zero buffer for requests. If a second audio frame arrives before the first is done, the system crashes (OOM). |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Concurrency Invariant (Exploration - 15 Mins)
*   **Objective:** Apply Little's Law to calculate the required "Machine Capacity" for a given "User Load."
*   **The "Lock" (Prediction):** "If you double the request rate ($\lambda$) and your processing time ($W$) stays the same, what happens to the number of concurrent requests ($L$) in your system?"
*   **The Workbench:**
    *   **Action:** Adjust **Request Arrival Rate** ($\lambda$) and **Model Execution Time** ($W$).
    *   **Observation:** The **Little's Law Gauge**. Watch the "In-Flight Requests" ($L$) rise and fall.
*   **Reflect:** "Patterson asks: 'Why is $L$ the most important number for memory dimensioning?' Reconcile this with your track's specific RAM limit."

### Part 2: The Batching Tax (Trade-off - 15 Mins)
*   **Objective:** Optimize the Batching Window to maximize throughput without violating Latency SLAs.
*   **The "Lock" (Prediction):** "Does increasing the Batch Size decrease or increase the *average* latency for a single user?"
*   **The Workbench:**
    *   **Sliders:** Batch Size (1-128), Max Batch Window (ms), Service Rate.
    *   **Instruments:** **Throughput-vs-Latency Seesaw**. A plot showing the "Knee of the Curve" where further batching causes latency to spike exponentially.
    *   **The 10-Iteration Rule:** Students must find the "Optimal Batch Size" that hits the mission's throughput target (e.g., 100 QPS) while staying in the "Green" latency zone.
*   **Reflect:** "Jeff Dean observes: 'Your batching strategy is efficient but your P99 latency is 10x the mean.' Use the queue depth data to explain why 'Batching Window' is a hidden tax on real-time users."

### Part 3: The Utilization Cliff (Synthesis - 15 Mins)
*   **Objective:** Design a serving architecture that maintains 80% utilization under "Bursty" traffic.
*   **The "Lock" (Prediction):** "If request traffic follows a Poisson distribution (Bursty), can you achieve 100% hardware utilization without causing infinite queues?"
*   **The Workbench:** 
    *   **Interaction:** **Poisson Traffic Toggle**. **KV-Cache Optimization Level**. **PagedAttention Level**.
    *   **The "Stakeholder" Challenge:** The **UX Designer** (Mobile) or **CFO** (Cloud) demands zero lag. You must implement **Continuous Batching** to fill the GPU bubbles.
*   **Reflect (The Ledger):** "Defend your final Serving Configuration. Did you prioritize 'Throughput' or 'Responsiveness'? Explain how Little's Law influenced your final 'Capacity Plan'."

---

## 4. Visual Layout Specification
*   **Primary:** `QueuingWaterfall` (Wait Time vs. Math Time vs. Transfer Time).
*   **Secondary:** `LittleLawCalculator` (Real-time $L = \lambda \cdot W$ visualization).
*   **Math Peek:** Toggle for `M/M/1 Queue` approximations and `Throughput = Batch / Latency`.

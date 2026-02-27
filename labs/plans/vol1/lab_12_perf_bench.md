# 📐 Mission Plan: 12_perf_bench (Performance Benchmarking)

## 1. Chapter Context
*   **Chapter Title:** Performance Benchmarking: The Evaluation Standard.
*   **Core Invariant:** The Benchmarking Paradox (Peak vs. Sustained Performance).
*   **The Struggle:** Understanding that standardized metrics (like Peak TFLOPS) rarely predict real-world success. Students must navigate the gap between **Benchmark Scores** and **Application Realities**, learning to audit systems using the MLPerf scenarios.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Benchmark" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Offline Mirage.** The H100 achieves 90% utilization in the 'Offline' benchmark, but drops to 15% in our 'Server' mode. You must fix the utilization gap. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The SingleStream Lie.** The benchmark says 5ms mean latency, but the 'SingleStream' scenario fails to model the 8-camera parallel load. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Thermal Peak.** The chip hits 60FPS for 30 seconds (Benchmark run), then throttles to 10FPS. You need a 'Sustained' audit. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Accuracy Bias.** The benchmark was trained on clean audio. In the field (the 'RealWorld' scenario), accuracy drops from 99% to 60%. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Scenario Selection (Exploration - 15 Mins)
*   **Objective:** Map your application to the correct MLPerf scenario (SingleStream, MultiStream, Server, Offline).
*   **The "Lock" (Prediction):** "Which MLPerf scenario is the most representative for an Autonomous Vehicle's safety-critical braking loop?"
*   **The Workbench:**
    *   **Action:** Toggle between the 4 MLPerf scenarios. Adjust the **Arrival Rate** of requests.
    *   **Observation:** The **Scenario Latency Plot**. Watch how the 'Offline' throughput looks great while the 'Server' latency becomes unusable.
*   **Reflect:** "Reconcile the difference between 'Throughput-Optimized' and 'Latency-Optimized' benchmarking. Why is Goodhart's Law relevant here?"

### Part 2: The Tail at Scale (Trade-off - 15 Mins)
*   **Objective:** Audit the system for P99 Tail Latency and identify "Outlier Killers."
*   **The "Lock" (Prediction):** "If the 'Mean' latency is 10ms and the 'Standard Deviation' is 5ms, what is the P99 latency likely to be?"
*   **The Workbench:**
    *   **Sliders:** Noise Level, OS Background Load, Context Switching Frequency.
    *   **Instruments:** **Latency Distribution Histogram** (Linear vs. Log Scale).
    *   **The 10-Iteration Rule:** Students must introduce 'Jitter' and find the exact P99 threshold that violates their track's safety/UX window.
*   **Reflect:** "Jeff Dean asks: 'Why is the P99 more important than the Mean for a fleet of 1,000 devices?' (Hint: See the 'Statistical Probability of Failure' math)."

### Part 3: The Comparative Audit (Synthesis - 15 Mins)
*   **Objective:** Perform a head-to-head audit of two hardware platforms using the 'Pareto Efficiency' metric.
*   **The "Lock" (Prediction):** "Will the chip with the highest 'Peak TFLOPS' necessarily deliver the highest 'Sustained Throughput' for your mission?"
*   **The Workbench:** 
    *   **Interaction:** Compare **System A** (High Peak, Low BW) vs. **System B** (Low Peak, High BW).
    *   **The "Stakeholder" Challenge:** The **Purchasing Lead** wants to buy System A because it has a better marketing spec. You must use the **Comparative Roofline** to prove that System B is 2x more efficient for your specific Lighthouse Model.
*   **Reflect (The Ledger):** "Define your final 'Benchmark Strategy.' Which specific metric ($/Token, FPS/Watt, or P99 ms) is your ultimate measure of success? Justify why you ignored the 'Peak' marketing numbers."

---

## 4. Visual Layout Specification
*   **Primary:** `LatencyHistogram` (Showing Mean vs. P95 vs. P99).
*   **Secondary:** `ScenarioComparisonChart` (Throughput vs. Latency for all 4 MLPerf modes).
*   **Math Peek:** Toggle for `P99 = \mu + 2.33\sigma` (assuming normal) vs. actual non-normal tail math.

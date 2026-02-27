# 📐 Mission Plan: 01_ml_intro (Deep Analysis)

## 1. Chapter Context
*   **Chapter Title:** Introduction to ML Systems.
*   **Core Invariants:** The AI Triad (D·A·M), The Bitter Lesson (Scale > Logic), and The Verification Gap.
*   **The Struggle:** Moving from "Software 1.0" (Explicit Rules) to "Software 2.0" (Learned from Data).
*   **Target Duration:** 45 Minutes.

---

## 2. Narrative Arc
You have claimed your track. Now, you must calibrate your "Architect's Intuition." You will witness the three physical laws that make AI Engineering a distinct discipline: the massive scaling gap, the take-off of learning systems, and the impossibility of brute-force testing.

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Magnitude Gap (The AI Triad - 15 Mins)
*   **Objective:** Quantify the 9-order-of-magnitude span of the ML landscape.
*   **The "Lock" (Prediction):** "By what factor (ratio) does an H100 (Cloud) exceed an ESP32 (TinyML) in peak compute performance?"
*   **The Workbench:**
    *   **Action:** A slider that sweeps from TinyML -> Mobile -> Edge -> Cloud.
    *   **Observation:** A **Comparison Radar Chart** and **Dynamic Ratio Gauge**.
    *   **The 5-Move Rule:** Students must compare their specific chosen track against all 3 other archetypes.
*   **Reflect:** "You say the gap is 10^9. Why does this physical asymmetry prevent a 'one-size-fits-all' software stack for AI? Justify using the D·A·M axes."

### Part 2: Proving the Bitter Lesson (Historical Audit - 15 Mins)
*   **Objective:** Contrast the "Feature Engineering" era with the "Deep Learning" era using real historical data.
*   **The "Lock" (Prediction):** "If we increase human tuning effort by 10x, will it beat a 10x increase in machine scale over a 5-year period?"
*   **The Workbench:**
    *   **Action:** A "Historical Scrubber" slider (1980 -> 2024).
    *   **Observation:** A **Historical Accuracy Plot** featuring actual model benchmarks (AlexNet, ResNet, GPT-3, GPT-4).
    *   **The 15-Iteration Rule:** Students must "Step through Time" to see where the curves for "Rules" and "Learning" diverge.
*   **Reflect:** "Reconcile this result with Richard Sutton's 'Bitter Lesson.' Why is human expertise a 'depreciating asset' in the current AI regime?"

### Part 3: The Verification Gap Audit (Untestable Space - 15 Mins)
*   **Objective:** Quantify the mathematical impossibility of exhaustive testing for your mission.
*   **The "Lock" (Prediction):** "Can a test suite with 1 billion samples achieve even 1% coverage of your model's input space?"
*   **The Workbench:**
    *   **Action:** A **Verification Calculator**. Input resolution (e.g., 224x224) and test rate (samples/sec).
    *   **Observation:** A **Time-to-Test Counter**. (Output: "Years to Test 1%: 10^300,000").
*   **Reflect:** (Stakeholder Quality Lead): "Our CI/CD pipeline passed 100%. Why should we still invest in 'Continuous Monitoring' in production? Prove the necessity using the Verification Gap math."

---

## 4. Visual Layout Specification
*   **Primary:** `LandscapeRadar` (Log-scale comparison).
*   **Secondary:** `ScalingLawPlot` (Actual historical data points).
*   **Math Peeking:** Toggle for the `Degradation Equation` math.

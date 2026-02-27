# 📐 Mission Plan: 14_ml_ops (ML Operations)

## 1. Chapter Context
*   **Chapter Title:** ML Operations: System Entropy & Maintenance.
*   **Core Invariant:** The Statistical Drift Invariant ($	ext{Accuracy}_t \approx 	ext{Accuracy}_0 - \lambda \cdot D(P_t \| P_0)$).
*   **The Struggle:** Understanding that ML systems are "rotting assets." Students must navigate the trade-off between **Retraining Cost** (Machine budget) and **Model Accuracy** (Data signal), specifically focusing on identifying "Silent Failures" before they become catastrophic.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Ops" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Economic Drift.** User prompts have shifted to a new slang/topic. Accuracy is down 10%, but retraining costs $2M. You must determine the exact ROI of a new training run. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Seasonal Drift.** Winter has arrived. Perception accuracy on snowy roads is 40% lower than the 'Gold Standard' sunny baseline. You must detect this 'Silent Failure' in real-time. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Firmware Skew.** A system update changed the NPU driver. Now, the 'Compiled' model from Ch 7 is producing garbage outputs. You must debug the 'Training-Serving Skew'. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Environmental Noise.** A new fan in the user's home creates a frequency spike that kills noise isolation. You have no labels for this new environment. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Drift Detective (Exploration - 15 Mins)
*   **Objective:** Detect "Silent Failure" by measuring statistical divergence (PSI) between training and production data.
*   **The "Lock" (Prediction):** "If the feature distribution ($P_t$) shifts by 20%, will your infrastructure monitoring (CPU/RAM usage) alert you to the failure?"
*   **The Workbench:**
    *   **Action:** Slide the **Environment Noise** or **Season Scrubber**.
    *   **Observation:** The **Drift Radar**. Watch the **PSI (Population Stability Index)** spike while the "Infrastructure Health" stays perfectly green.
*   **Reflect:** "Patterson asks: 'Why is MLOps more than just DevOps?' Use the term **'Silent Failure'** to justify why code-based monitoring failed here."

### Part 2: The Retraining ROI (Trade-off - 15 Mins)
*   **Objective:** Optimize the Retraining Interval to balance compute budget vs. business loss.
*   **The "Lock" (Prediction):** "Is it cheaper to retrain every day or retrain only when accuracy drops below a threshold?"
*   **The Workbench:**
    *   **Sliders:** Retraining Cost ($), Accuracy Decay Rate ($\lambda$), Drift Speed.
    *   **Instruments:** **Optimal Retraining Interval Plot**. A curve showing "Total Cost" (Compute + Opportunity Loss) with a clear minimum.
    *   **The 10-Iteration Rule:** Students must use the **Optimal Retraining Formula** ($T^* = \sqrt{2C/K}$) to find the "Sweet Spot" for their track's specific budget.
*   **Reflect:** "Jeff Dean observes: 'Your team is retraining 10x more than necessary.' Use the square-root law to propose a more efficient schedule."

### Part 3: Slice-Aware Deployment (Synthesis - 15 Mins)
*   **Objective:** Manage a "Shadow Deployment" to validate a new model on a specific failing slice of data.
*   **The "Lock" (Prediction):** "If the 'Aggregate' accuracy is 95%, is it safe to deploy to the entire fleet? (Hint: Check the 'Minority Slice' metrics)."
*   **The Workbench:** 
    *   **Interaction:** **Shadow Mode Toggle**. **Slice Selector** (e.g., Night-time, Low-battery, Accented Speech).
    *   **The "Stakeholder" Challenge:** The **Safety Director** (Edge) or **Product Manager** (Mobile) blocks the rollout. You must use the **Slice Metrics Card** to prove the new model fixed the specific drift issue without regressing on the majority.
*   **Reflect (The Ledger):** "Defend your final 'Rollout Strategy.' Why did you choose a 'Staged Canary' deployment over a 'Big Bang' update? Justify using the Degradation Equation."

---

## 4. Visual Layout Specification
*   **Primary:** `DriftRadar` (Comparing Feature Distributions: Train vs. Live).
*   **Secondary:** `RetrainingCostCurve` (Cost vs. Retraining Frequency).
*   **Math Peek:** Toggle for `PSI = \sum (O - E) \ln(O/E)` and the `Degradation Equation`.

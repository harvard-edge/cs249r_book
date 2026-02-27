# 📐 Mission Plan: 16_responsible_ai (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Responsible AI: Governance at Scale.
*   **Core Invariant:** The Governance Invariant (Accountability is a feedback loop) and the **Fairness Tax**.
*   **The Struggle:** Understanding that "Responsible AI is the System's Control Plane." Students must navigate the trade-off between **Global Capability** (accuracy) and **Algorithmic Accountability** (fairness, explainability, privacy), specifically focusing on how bias amplifies through fleet-scale feedback loops.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Responsible Missions)

| Track | Persona | Fixed North Star Mission | The "Governance" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Machine Unlearning Wall.** A user has exercised their 'Right to be Forgotten' (GDPR). You must remove their data from the trained weights of a 70B model without retraining the whole cluster. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Explainability Tax.** The **Safety Board** demands that every braking decision be accompanied by an 'Explainability Saliency Map'. This adds 15ms to your 10ms budget. You must optimize the 'Feature Trace'. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Bias Amplification.** Your AR translation is learning from user 'Corrections'. A small group of users is 'Poisoning' the global accent-model with offensive slang. You must implement 'Robust Aggregation'. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Demographic Fade.** Your low-power quantization works for the 'Majority' but has 10x higher error for female and child speakers. You must find a 'Fairness-Aware' bit-width. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Fairness Tax Audit (Exploration - 15 Mins)
*   **Objective:** Quantify the "Accuracy Penalty" required to achieve demographic parity.
*   **The "Lock" (Prediction):** "If you enforce 'Equal Opportunity' across all subgroups, will your 'Aggregate' accuracy increase, decrease, or stay the same?"
*   **The Workbench:**
    *   **Action:** Slide the **Fairness Constraint** ($\gamma$) from 0 to 1.0. 
    *   **Observation:** The **Fairness-Accuracy Pareto Frontier**. Watch the gap between subgroups shrink while the global average capability drops.
*   **Reflect:** "Patterson asks: 'Identify the exact Accuracy Loss ($pts$) required to hit your track's fairness SLA.' Use the term **'Pareto Suboptimality'** in your answer."

### Part 2: The Machine Unlearning Race (Trade-off - 15 Mins)
*   **Objective:** Balance "Data Deletion Speed" vs. "Model Integrity."
*   **The "Lock" (Prediction):** "Can we 'Remove' a data point's influence from a model faster than the time it took to train the model?"
*   **The Workbench:**
    *   **Interaction:** Toggle between **Full Retraining**, **Gradient Scrubbing**, and **Influence Masking**.
    *   **Instruments:** **Deletion Latency vs. Model Regret Plot**.
    *   **The 10-Iteration Rule:** Students must find the "De-learning" method that satisfies the **Compliance Officer** without causing the model's accuracy on unrelated tasks to collapse.
*   **Reflect:** "Jeff Dean observes: 'Your scrubbed model is starting to hallucinate.' Propose a 'Weight-Sparsity' strategy to isolate sensitive data influence."

### Part 3: Explainability Performance (Synthesis - 15 Mins)
*   **Objective:** Design an "Auditable Pipeline" that hits the 10ms budget.
*   **The "Lock" (Prediction):** "Does adding 'SHAP' or 'LIME' explainability increase the memory bandwidth requirement ($D_{vol}/BW$) of your serving system?"
*   **The Workbench:** 
    *   **Interaction:** **Explainability Level Toggle** (None -> Heatmap -> Full Trace). **Saliency Resolution Slider**.
    *   **The "Stakeholder" Challenge:** The **Regulatory Board** blocks the project because the model is a 'Black Box.' You must implement **Model Surgery** (distilling a transparent student model) to provide explanations within the 1mW/10ms power/latency wall.
*   **Reflect (The Ledger):** "Defend your final 'Responsible Architecture.' Did you prioritize 'Model Transparency' or 'Peak Utility'? Justify why 'Accountability' is a first-order design variable at scale."

---

## 4. Visual Layout Specification
*   **Primary:** `FairnessParetoPlot` (Accuracy vs. Disparity across subgroups).
*   **Secondary:** `GovernanceWaterfall` (Math Time vs. Security/Explainability Overhead).
*   **Math Peek:** Toggle for `Demographic Parity` and `Machine Unlearning Residuals`.

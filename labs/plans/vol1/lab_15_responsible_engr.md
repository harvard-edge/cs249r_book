# 📐 Mission Plan: 15_responsible_engr (Responsible Engineering)

## 1. Chapter Context
*   **Chapter Title:** Responsible Engineering: Ethics & Governance.
*   **Core Invariant:** Lifecycle TCO & Carbon Economics (Inference TCO $\gg$ Training TCO).
*   **The Struggle:** Understanding that "Responsibility" is an engineering optimization problem, not just a policy statement. Students must navigate the trade-off between **Fairness/Privacy** (Algorithmic constraint) and **Economic/Environmental TCO** (Machine constraint) over a 3-year operational horizon.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Responsibility" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Carbon Bill.** Your fleet of H100s consumes 50MWh/year. The **Sustainability Lead** demands a 30% reduction in carbon footprint without dropping model quality. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Disparity Crisis.** Disaggregated evaluation reveals the model is 40x more likely to misidentify pedestrians with dark skin tones in low-light. You must fix the 'Safety Gap'. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Privacy Paradox.** Users want real-time translation but won't accept local video storage. You must implement 'Differential Privacy' which adds 5ms of noise-overhead to your 10ms budget. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Demographic Soundstage.** Your noise isolation model works for studio-recorded male voices but fails for children and female speakers in home environments. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: Disparity Discovery (Exploration - 15 Mins)
*   **Objective:** Identify algorithmic bias through Disaggregated Evaluation.
*   **The "Lock" (Prediction):** "If the 'Global' accuracy is 98%, can any specific demographic subgroup have an error rate higher than 10%?"
*   **The Workbench:**
    *   **Action:** Select different **Data Slices** (Gender, Age, Skin Tone, Acoustic Environment).
    *   **Observation:** The **Disparity Bar Chart**. Watch the "Global" average hide the catastrophic failures in the "Minority" slices.
*   **Reflect:** "Reconcile the result with the 'Gender Shades' study from the text. Why is aggregate accuracy a misleading metric for safety-critical systems?"

### Part 2: The Price of Fairness (Trade-off - 15 Mins)
*   **Objective:** Balance Accuracy vs. Fairness using the Pareto Frontier.
*   **The "Lock" (Prediction):** "Will enforcing 'Equal Opportunity' (matching True Positive Rates) increase or decrease the overall throughput of your serving system?"
*   **The Workbench:**
    *   **Sliders:** Fairness Constraint Level (0-100%), Privacy Epsilon ($\epsilon$), Training Data Re-weighting.
    *   **Instruments:** **Fairness-Accuracy Pareto Plot**. Shows the "Cost of Fairness" in terms of percentage accuracy points lost.
    *   **The 10-Iteration Rule:** Students must find the exact "Fairness Threshold" that satisfies the **Compliance Officer** without dropping accuracy below the mission's "Utility Floor."
*   **Reflect:** "Patterson asks: 'Is a 1% fairness gain worth a 10% increase in TFLOPS?' Use the **RoC (Return on Compute)** gauge to defend your decision."

### Part 3: The Lifecycle Ledger (Synthesis - 15 Mins)
*   **Objective:** Calculate the 3-year TCO and Carbon Footprint of the entire system.
*   **The "Lock" (Prediction):** "Over a 3-year period, which will contribute more to the carbon footprint: Training the model once, or Serving it 1 million times?"
*   **The Workbench:**
    *   **Interaction:** **Deployment Duration Slider** (1mo -> 3yr). **Energy Grid Mix Selector** (Green vs. Coal).
    *   **The "Stakeholder" Challenge:** The **Sustainability Lead** (or CFO) demands a "Lifecycle Audit." You must prove that using a **Compressed Model** (from Lab 10) saves more in carbon over 3 years than it cost in "Accuracy Loss" at launch.
*   **Reflect (The Ledger):** "Defend your final 'Responsible Architecture.' Did you prioritize Carbon, Fairness, or Privacy? Justify your choice using the 3-year TCO projections."

---

## 4. Visual Layout Specification
*   **Primary:** `DisparityDiscoveryChart` (Disaggregated metrics across subgroups).
*   **Secondary:** `LifecycleTCORadar` (Accuracy, Fairness, Carbon, Dollar Cost).
*   **Math Peek:** Toggle for `TCO = Capex + (P \cdot T \cdot Cost_{kwh})` and `Carbon = Energy \cdot Intensity_{grid}`.

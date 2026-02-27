# 📐 Mission Plan: 14_robust_ai (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Robust AI: Reliability under Attack and Entropy.
*   **Core Invariant:** The Robustness Invariant ($Accuracy_{robust} \le Accuracy_{std}$) and the **Robustness Tax**.
*   **The Struggle:** Understanding that "Robustness is a Computational Tax." Students must navigate the trade-off between **Standard Accuracy** (benchmarks) and **Adversarial Robustness** (real-world survival), specifically focusing on how fleet-scale distribution shifts and Silent Data Corruption (SDC) degrade reliability.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Robustness Missions)

| Track | Persona | Fixed North Star Mission | The "Robustness" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The RAG Poisoning.** Your vector database has been injected with adversarial snippets that force the LLM into 'Negative Persona' loops. You must implement 'Semantic Sanitization' without killing throughput. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Blizzard Drift.** A sudden snowstorm has shifted the visual distribution. The car is 50% less confident in its lane detection. You must calibrate 'Uncertainty Thresholds' to trigger a safe takeover. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Adversarial Patch.** Users are wearing 'Cloaking Shirts' that cause your AR glasses to crash. You must implement 'Gradient Masking' within your 2W thermal budget. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Cosmic SDC.** Your cluster of 10,000 hearables is experiencing a 0.1% bit-flip rate due to radiation (SDC). You must implement 'Triple Modular Redundancy' (TMR) at the MCU level. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Robustness Frontier (Exploration - 15 Mins)
*   **Objective:** Quantify the "Standard vs. Robust" Accuracy gap.
*   **The "Lock" (Prediction):** "If you increase the 'Noise Radius' ($\sigma$) of your input defense, will the accuracy on clean (non-noisy) data increase or decrease?"
*   **The Workbench:**
    *   **Action:** Slide the **Attack Intensity** ($\epsilon$) and **Defense Strength**.
    *   **Observation:** The **Robustness Pareto Curve**. Watch the "Clean Accuracy" drop as the "Robust Accuracy" rises.
*   **Reflect:** "Patterson asks: 'Why is there no Free Lunch in Robustness?' (Reference the **Robustness Tax**)."

### Part 2: Fleet-Wide Drift Detection (Trade-off - 15 Mins)
*   **Objective:** Optimize the "Drift Sensitivity" (PSI) to balance False Positives vs. Fleet Safety.
*   **The "Lock" (Prediction):** "Will a 'Strict' PSI threshold (0.1) result in more or fewer 'Retraining Requests' than a 'Loose' threshold (0.2)?"
*   **The Workbench:**
    *   **Interaction:** Adjust **PSI Alert Threshold**. Slide **Environmental Noise** (Distribution Shift).
    *   **Instruments:** **Fleet Alert Dashboard**. **Retraining Cost Waterfall**.
    *   **The 10-Iteration Rule:** Students must find the exact "Confidence Window" that catches 99% of drifts without causing the "Retraining Budget" to explode.
*   **Reflect:** "Jeff Dean observes: 'Your drift monitor is crying wolf every 5 minutes.' Propose a 'Sequential Probability Ratio Test' (SPRT) to stabilize the alerts."

### Part 3: Defense-in-Depth Architecture (Synthesis - 15 Mins)
*   **Objective:** Design a "Resilient Pipeline" that survives a combined SDC and Adversarial attack.
*   **The "Lock" (Prediction):** "Which defense adds more latency to the 10ms budget: 'Input Pre-processing' or 'MC Dropout' (running the model 10 times for uncertainty)?"
*   **The Workbench:** 
    *   **Interaction:** **Defense Checklist** (Adversarial Training, TMR, MC Dropout, Input Smoothing).
    *   **The "Stakeholder" Challenge:** The **Safety Lead** demands 100% SDC resilience. You must prove that using **Error-Correcting Codes (ECC)** on weights is 100x more efficient than running the model three times (TMR).
*   **Reflect (The Ledger):** "Defend your final 'Robust Architecture.' Did you prioritize 'Model Integrity' or 'Latency Budget'? Justify how you bypassed the Robustness Invariant."

---

## 4. Visual Layout Specification
*   **Primary:** `RobustnessParetoPlot` (Accuracy vs. Noise/Attack Strength).
*   **Secondary:** `FleetDriftGauge` (Showing PSI across different cities/hardware-versions).
*   **Math Peek:** Toggle for `Accuracy_{robust} \le Accuracy_{std}` and `SDC Probability` formulas.

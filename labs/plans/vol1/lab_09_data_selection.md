# 📐 Mission Plan: 09_data_selection (Data Selection)

## 1. Chapter Context
*   **Chapter Title:** Data Selection: Signal-to-Noise Engineering.
*   **Core Invariant:** The Data Quality Multiplier ($N_{noisy} \propto 1/\epsilon^2$ vs $N_{clean} \propto 1/\epsilon$).
*   **The Struggle:** Understanding that "more data" is not always better. Students must navigate the **Data Wall**—the point where compute abundance meets high-quality data exhaustion—and learn to maximize the **Information-Compute Ratio (ICR)**.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Data" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Deduplication Tax.** Your web-scraped corpus is 50% redundant. You are wasting $5M in GPU hours training on identical tokens. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Hard-Negative Crisis.** The model keeps missing 'statue' edge cases. You have 1PB of raw video but only a $50k labeling budget. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Noise Penalty.** Your training data has 5% label noise, requiring 10x more training steps to converge, which exceeds your project deadline. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Synthetic Bridge.** You have only 500 real samples. You must use synthetic augmentation without creating a 'Domain Gap' that kills field accuracy. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Deduplication Audit (Exploration - 15 Mins)
*   **Objective:** Quantify the speedup of 'Static Pruning' (removing redundant data) on total training time.
*   **The "Lock" (Prediction):** "If you remove 30% of the most redundant samples using LSH/MinHash, what is the expected reduction in total training FLOPs?"
*   **The Workbench:**
    *   **Action:** Adjust the **Deduplication Threshold** (MinHash Similarity).
    *   **Observation:** The **ICR Curve (Information-Compute Ratio)**. Watch the learning signal per compute unit rise as redundant mass is removed.
*   **Reflect:** "Why does training on duplicate data decrease the efficiency ($\eta$) of your training system? Reconcile this with the Iron Law."

### Part 2: Active Learning ROI (Trade-off - 15 Mins)
*   **Objective:** Optimize the labeling budget using Uncertainty Sampling.
*   **The "Lock" (Prediction):** "Will uncertainty sampling reach 90% accuracy with more or fewer samples than random sampling?"
*   **The Workbench:**
    *   **Action:** Toggle between **Random Sampling** and **Active Learning**. Adjust the 'Selection Batch Size'.
    *   **Observation:** **Accuracy vs. Labeling Cost ($) Plot**. A Pareto frontier showing the ROI of expert labels.
    *   **The 10-Iteration Rule:** Students must find the exact 'Knee of the Curve' where the cost of running the active-learning model exceeds the savings in labeling fees.
*   **Reflect:** "Jeff Dean asks: 'Is the CPU cost of indexing the entire 1PB dataset higher than the GPU savings from training on fewer samples?' Prove your answer using the dashboard."

### Part 3: The Domain Gap Synthesis (Synthesis - 15 Mins)
*   **Objective:** Balance Synthetic and Real data to maximize generalization.
*   **The "Lock" (Prediction):** "What happens to your 'Safety Metric' if you move from 10% Synthetic data to 90% Synthetic data?"
*   **The Workbench:** 
    *   **Interaction:** **Data Mix Ratio Slider** (Synthetic vs. Real). **Domain Randomization Intensity**.
    *   **The "Stakeholder" Challenge:** The **Safety Lead** warns that the synthetic simulator doesn't model 'Rain' correctly. You must find a mix that hits the accuracy target while maintaining a 'FID Score' (Domain Gap) below the safety threshold.
*   **Reflect (The Ledger):** "Defend your final Data Acquisition Strategy. Did you prioritize 'Quantity' (Synthetic) or 'Quality' (Expert-Labeled Real)? Explain how the $1/\epsilon^2$ noise penalty influenced your choice."

---

## 4. Visual Layout Specification
*   **Primary:** `ICR_Curve` (Learning Progress vs. Compute FLOPs).
*   **Secondary:** `LabelingROIPlot` (Accuracy vs. Total Project Cost).
*   **Math Peek:** Toggle for the `Data Quality Multiplier` and `MinHash Probability` formulas.

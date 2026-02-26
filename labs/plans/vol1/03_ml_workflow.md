# 📐 Mission Plan: 03_ml_workflow (Deep Analysis)

## 1. Chapter Context
*   **Chapter Title:** ML Workflow: Orchestrating the Lifecycle.
*   **Core Invariant:** The Iteration Tax (Velocity compounds into Quality).
*   **The Struggle:** Balancing the speed of experimentation against the depth of each individual experiment.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Iteration Tax" |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving on a single H100. | **Data Egress.** $90k/PB moving between regions. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop on NVIDIA Orin. | **Real-world Testing.** Safety certs take 2 weeks per change. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation on Meta Ray-Bans. | **App Store Review.** 1-week cycle for every FW update. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **Data Collection.** Manual ear-sim tests take 1 month. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Pipeline Audit (Exploration - 15 Mins)
*   **Objective:** Map the time-on-task for each of the 6 lifecycle stages and identify the "Bottleneck Stage."
*   **The "Lock" (Prediction):** "Which stage currently accounts for >60% of your total development time?"
*   **The Workbench:**
    *   **Sliders:** Sensitivity of 6 Stages (Data Prep, Labeling, Training, Eval, Deploy, Monitoring).
    *   **Instruments:** `LifecycleWaterfall` (Time per stage), `CostPropagationGauge` (The $2^{N-1}$ multiplier).
    *   **The 5-Move Rule:** Students must simulate at least 5 different "Optimization Strategies" (e.g. automating labeling vs buying GPUs) to find the highest leverage point.
*   **Reflect:** "You automated 'Training' but the project iteration time only dropped 5%. Identify the 'Hidden Tax' in your lifecycle using the waterfall data."

### Part 2: The Quality-Velocity Frontier (Trade-off - 20 Mins)
*   **Objective:** Compare a "Heavy/SOTA" model vs. a "Light/Iterative" model over a 6-month window.
*   **The "Lock" (Prediction):** "If the SOTA model starts with 5% higher accuracy but iterates 10x slower, which model will win in Month 6?"
*   **The Workbench:**
    *   **Sliders:** Model Complexity (Large/Slow -> Small/Fast), Engineering Budget Allocation ($), Iteration Count.
    *   **Instruments:** `compound_accuracy_plot` (Time vs Accuracy growth), `IterationTaxMeter`.
    *   **The 15-Iteration Rule:** Students must find the "Knee of the Iteration Curve"—the point where further reducing model size results in "Speed without Signal."
*   **Reflect:** "Your CFO wants to buy the larger model because it has better 'Day 1' metrics. Use the 6-month projection to prove why iteration velocity is a higher-value feature for this mission."

### Part 3: The Design Review (Synthesis - 10 Mins)
*   **Objective:** Negotiate a "De-scoped" mission that is actually achievable within the project deadline.
*   **The "Lock" (Prediction):** "Given our current iteration tax, what is the maximum achievable accuracy before the 6-month deadline?"
*   **The Workbench:** All lifecycle variables unlocked.
*   **The "Stakeholder" Challenge:** The **Product Manager** demands "99% or bust." The student must use the **Constraint Propagation math** to prove that 99% requires more iterations than the calendar allows. Propose an achievable target.
*   **Reflect (The Ledger):** Define your final "Engineering Pace." How many iterations per week are you committing to? Justify the risk of the "Verification Gap" in your plan.

---

## 4. Visual Layout Specification
*   **Primary:** `IterationProjectionPlot` (Showing the compound effect of fast cycles).
*   **Secondary:** `LifecycleWaterfall` (Showing where the weeks go).
*   **Transparency:** Toggle for `Cost(N) = 2^{N-1}` propagation formula.

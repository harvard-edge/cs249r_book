# 📐 Mission Plan: 12_ops_scale (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Operations at Scale: Fleet Monitoring & Governance.
*   **Core Invariant:** The Operations Invariant (Complexity scales super-linearly with model heterogeneity: $O(N)$ alerts, $O(N \log N)$ deployment, $O(N^2)$ dependencies).
*   **The Struggle:** Understanding that at scale, "Manual Work is Toil." Students must navigate the trade-off between **Operational Toil** (manual fixes) and **Platform CapEx** (automation cost), specifically focusing on the break-even point for building an ML Platform.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Ops Missions)

| Track | Persona | Fixed North Star Mission | The "Ops" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The N-Models Explosion.** You are serving 50 different LLM versions for different customers. Your 'Alert Noise' has made it impossible to find real drift. You must automate 'Threshold Discovery'. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Rollout Jitter.** You are updating the perception model on 500,000 AVs. A 'Shadow Mode' test reveals that the new model has a 100ms latency tail on older hardware versions. You must 'Rollback' the fleet. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The App-OS Skew.** A system update changed the NPU power-policy. Now, the AR filter's 'FPS' is drifting differently across 10 regions. You must implement 'Slice-level Monitoring'. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Dependency Domino.** You updated the noise-isolation model, but it broke the downstream 'Wake-Word' detector. You must fix the 'Feature Store' versioning to prevent 'Training-Serving Skew'. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Platform ROI Audit (Exploration - 15 Mins)
*   **Objective:** Identify the "Break-even Point" ($N^*$) where building an ML Platform is cheaper than manual operations.
*   **The "Lock" (Prediction):** "How many concurrent models ($N$) can your team manage manually before the cost of 'Toil' exceeds the cost of a \$500k platform investment?"
*   **The Workbench:**
    *   **Action:** Slide the **Number of Models** ($N$) and **Model-Type Heterogeneity**.
    *   **Observation:** The **Ops Cost Curve**. Watch the "Manual Toil" line explode super-linearly while the "Platform" line remains stable.
*   **Reflect:** "Patterson asks: 'Why does heterogeneity ($O(N^2)$) kill productivity faster than simple scale ($O(N)$)?' (Reference the dependency entanglement)."

### Part 2: The Canary Duration (Trade-off - 15 Mins)
*   **Objective:** Optimize the "Canary Deployment" window to balance safety vs. release velocity.
*   **The "Lock" (Prediction):** "If you want to be 95% confident that your new model hasn't regressed on a '1-in-1,000' edge case, how many live requests must you monitor in 'Shadow Mode'?"
*   **The Workbench:**
    *   **Sliders:** Target Confidence (%), Expected Failure Rate, Fleet Request Rate.
    *   **Instruments:** **Confidence Clock**. **Risk-vs-Velocity Radar**.
    *   **The 10-Iteration Rule:** Students must find the exact "Shadow Duration" that satisfies the **Safety Lead** without letting the "Release Cycle" exceed 1 week.
*   **Reflect:** "Jeff Dean observes: 'Your shadow window is 3 months long. By the time you deploy, the world has already drifted.' Propose a 'Sequential Validation' strategy to speed up the loop."

### Part 3: The Ensemble Lockdown (Synthesis - 15 Mins)
*   **Objective:** Manage a "Model Handshake" between upstream and downstream dependencies in an ensemble.
*   **The "Lock" (Prediction):** "If you update the 'Feature Extractor' without retraining the 'Classifier', will the system accuracy drop even if the new extractor is 'better'?"
*   **The Workbench:** 
    *   **Interaction:** **Upstream Version Selector**. **Downstream Retrain Toggle**. **Feature Store Consistency Check**.
    *   **The "Stakeholder" Challenge:** The **Software Lead** wants to deploy the new extractor tonight. You must use the **Ensemble Trace** to prove that this will cause a 'Silent Failure' in the downstream wake-word detector.
*   **Reflect (The Ledger):** "Defend your final 'Governance Policy.' Did you choose 'Loose Versioning' or 'Strict Immutable Contracts'? Justify using the Training-Serving Skew Invariant."

---

## 4. Visual Layout Specification
*   **Primary:** `OpsBreakEvenPlot` (Total Cost vs Number of Models).
*   **Secondary:** `ShadowModeDashboard` (Real-time confidence metrics for a staged rollout).
*   **Math Peek:** Toggle for `ROI_{platform} = 	ext{Toil}_{manual} - (	ext{Capex}_{platform} + 	ext{Toil}_{platform})`.

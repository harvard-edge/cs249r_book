# 📐 Mission Plan: 13_sec_privacy (Volume 2: Fleet Scale)

## 1. Chapter Context
*   **Chapter Title:** Security & Privacy: The Responsible Fleet.
*   **Core Invariant:** The Privacy-Utility-Budget Triangle (Accuracy vs. Privacy $\epsilon$ vs. Cost $O$).
*   **The Struggle:** Understanding that "Privacy is a Finite Resource." Students must navigate the trade-off between **Model Utility** (accuracy) and **Privacy Guarantees** ($\epsilon$), specifically focusing on the "Defense Tax" (Latency/Energy overhead) of secure execution environments.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard (Security Missions)

| Track | Persona | Fixed North Star Mission | The "Security" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The PII Leak.** Your model is regurgitating PII from the training set. The board demands DP-SGD with $\epsilon < 1$. You must determine the exact 'TFLOPS Penalty' required to regain safety. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Adversarial Sticker.** A 50-cent sticker causes your AV to misidentify a 'Stop' sign as '80 MPH'. You must implement 'Adversarial Training' without breaking the 10ms safety window. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The FaceID Side-Channel.** Power analysis of the NPU reveals the user's biometric embeddings. You must move inference to a TEE (TrustZone) while staying under the 2W thermal cap. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Model Extraction.** Competitors are dumping your model weights via JTAG. You must implement 'Binary Encryption' and PUFs, adding 2ms to your 10ms 'Echo Window'. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Privacy-Utility Frontier (Exploration - 15 Mins)
*   **Objective:** Quantify the "Accuracy Collapse" as the Privacy Epsilon ($\epsilon$) is tightened.
*   **The "Lock" (Prediction):** "If you decrease $\epsilon$ from 8.0 (Weak) to 1.0 (Strong), will your training time double, triple, or stay the same to reach the same accuracy?"
*   **The Workbench:**
    *   **Action:** Slide the **Privacy Budget** ($\epsilon$). Adjust **Batch Size** (DP-SGD sensitivity).
    *   **Observation:** The **Privacy-Utility Pareto Plot**. Watch the "Utility Cliff" appear as noise overrides the gradient signal.
*   **Reflect:** "Patterson asks: 'Why does larger batching help recover the accuracy lost to DP noise?' (Reference the Signal-to-Noise ratio)."

### Part 2: Secure Aggregation Tax (Trade-off - 15 Mins)
*   **Objective:** Balance communication overhead vs. privacy in a distributed Federated Learning fleet.
*   **The "Lock" (Prediction):** "Does adding 'Pairwise Masking' (Secure Aggregation) increase network bandwidth consumption or compute latency more?"
*   **The Workbench:**
    *   **Interaction:** Toggle **Secure Aggregation**. Adjust **Number of Clients** ($N$) and **Secret Key Size**.
    *   **Instruments:** **Privacy-Communication Waterfall** (Masking Overhead vs. Math Time).
    *   **The 10-Iteration Rule:** Students must find the "Mask Complexity" that satisfies the **Privacy Officer** without causing the 5G sync to exceed the battery budget.
*   **Reflect:** "Jeff Dean observes: 'One slow node (straggler) is holding up the entire secure handshake.' Propose a 'Dropout-Resilient' aggregation strategy."

### Part 3: The Defense-in-Depth Audit (Synthesis - 15 Mins)
*   **Objective:** Design a "Hardened Architecture" that satisfies the Security Lead within the 10ms budget.
*   **The "Lock" (Prediction):** "Will running the model in a TEE (Trusted Execution Environment) be faster or slower than running it in plaintext with input sanitization?"
*   **The Workbench:** 
    *   **Interaction:** **TEE Toggle**. **Input Validator Toggle**. **Output Smoothing Slider**.
    *   **The "Stakeholder" Challenge:** The **Security Lead** rejects any design that doesn't use TEEs. You must prove that using **Model Sharding** (keeping only sensitive layers in TEE) hits the safety goal with 50% less latency than a full-TEE approach.
*   **Reflect (The Ledger):** "Defend your final 'Secure Design.' Did you prioritize 'Model Integrity' or 'User Experience'? Justify how you managed the 'Privacy-Utility-Budget Triangle'."

---

## 4. Visual Layout Specification
*   **Primary:** `PrivacyUtilityPareto` (Accuracy vs. Epsilon).
*   **Secondary:** `SecurityOverheadWaterfall` (Math vs. Encryption vs. Sanitization time).
*   **Math Peek:** Toggle for `DP-SGD Noise Scale` and `Secure Aggregation Complexity`.

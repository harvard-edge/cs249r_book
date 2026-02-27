# 📐 Mission Plan: 08_model_train (Model Training)

## 1. Chapter Context
*   **Chapter Title:** Model Training: The Scaled Process.
*   **Core Invariant:** Scaling Efficiency ($\eta_{scaling}$) and the Communication-to-Compute Ratio.
*   **The Struggle:** Understanding that training a model is not just about 'fitting' it, but 'manufacturing' it at scale. Students must navigate the **Network Wall** and the **Memory Wall** (Optimizer State) to achieve high **Model FLOPs Utilization (MFU)**.
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Training" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Utility Bill.** Training is 10x slower than estimated because of low MFU. We are burning $10k/hr on idle GPUs. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The 3AM Gradient Explosion.** Mixed-precision training is unstable on the Orin; loss NaN is killing the safety cert. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The On-Device Adaptation.** You need to fine-tune the filter for new lighting, but the optimizer state (Adam) exceeds the 512MB glasses RAM. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The Micro-Batch Wall.** To fit in SRAM, you must use Batch Size 1, but this drops throughput to 1% efficiency. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The State Management Audit (Exploration - 15 Mins)
*   **Objective:** Quantify the 'Hidden Memory' of training (Weights vs. Gradients vs. Optimizer State).
*   **The "Lock" (Prediction):** "If you are using the Adam optimizer in FP32, how many bytes of memory are required per parameter? (Hint: See the 4x multiplier in the text)."
*   **The Workbench:**
    *   **Action:** Toggle between **SGD**, **Momentum**, and **Adam**. Toggle **FP32** vs **Mixed-Precision**.
    *   **Observation:** The **Training Memory Waterfall**. Watch the "Optimizer State" bar grow to 3x the size of the "Weights" bar.
*   **Reflect:** "Why does training a model require 4–8x more memory than serving it? Reconcile this with your track's specific memory limit."

### Part 2: The Staged Pipeline Race (Trade-off - 15 Mins)
*   **Objective:** Identify the bottleneck in the Training Pipeline (Data Prep -> Ingestion -> Compute).
*   **The "Lock" (Prediction):** "Will adding a second GPU double your training speed if the CPU is already 100% saturated with data augmentation?"
*   **The Workbench:**
    *   **Sliders:** Worker Count (Data Loading), GPU Count, Augmentation Complexity.
    *   **Instruments:** **Pipeline Utilization Gauge**. A 3-box view showing Data, Bus, and Compute saturation.
    *   **The 5-Move Rule:** Students must find the exact "Balance Point" where the GPU is no longer "starved" by the CPU dataloader.
*   **Reflect:** "Reconcile the result with the 'Staged System Pipeline' diagram in @fig-staged-pipeline. Which stage is your 'Longest Pole'?"

### Part 3: The Scaling Wall (Synthesis - 15 Mins)
*   **Objective:** Optimize Batch Size and Precision to maximize MFU without exploding the loss.
*   **The "Lock" (Prediction):** "Does increasing the Batch Size improve or degrade the Communication-to-Compute ratio?"
*   **The Workbench:** 
    *   **Interaction:** All D·A·M variables unlocked + "Gradient Clipping" toggle.
    *   **The "Stakeholder" Challenge:** The **Infrastructure Lead** (Jeff Dean) warns that the network is 95% saturated. You must find a configuration that hits the target MFU (>40%) without exceeding the current network bandwidth.
*   **Reflect (The Ledger):** "Defend your final Training Configuration. Did you use Mixed-Precision or Gradient Checkpointing? Explain why raw TFLOPS wasn't the answer to this scaling mission."

---

## 4. Visual Layout Specification
*   **Primary:** `TrainingMemoryWaterfall` (Weights vs. Gradients vs. Opt-State vs. Activations).
*   **Secondary:** `ScalingEfficiencyPlot` (MFU vs. Number of Accelerators / Batch Size).
*   **Math Peek:** Toggle for the `Iron Law of Training` and `MFU = \frac{O_{total}}{T \cdot N \cdot R_{peak}}`.

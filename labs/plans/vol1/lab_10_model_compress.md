# 📐 Mission Plan: 10_model_compress (Model Compression)

## 1. Chapter Context
*   **Chapter Title:** Model Compression: The Silicon Contract.
*   **Core Invariant:** The Pareto Frontier ($Accuracy$ vs. $Bits$ vs. $Energy$).
*   **The Struggle:** Negotiating the 'Silicon Contract'—understanding that some accuracy must be sacrificed to satisfy physical laws. Students must navigate the trade-offs between **Precision** (Quantization), **Sparsity** (Pruning), and **Architecture** (Distillation).
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Compression" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The HBM Ceiling.** Your 70B model requires 140GB in FP16, but your H100 only has 80GB. You are 100% OOM. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The Latency Wall.** Your perception model takes 25ms. You must compress the 'Brains' to hit the 10ms 'Safety Window' without missing pedestrians. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Thermal Tax.** Running FP16 filters drains the glasses in 20 minutes and triggers thermal shutdown. You need 10x energy efficiency. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms under 1mW. | **The SRAM Crisis.** Your model is 1MB; your on-chip memory is 256KB. You must compress 4x just to 'boot' the system. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Quantization 'Free Lunch' (Exploration - 15 Mins)
*   **Objective:** Identify the 'Knee of the Curve' where bit-reduction saves memory without catastrophic accuracy loss.
*   **The "Lock" (Prediction):** "If you move from FP16 to INT8, by what factor does your memory footprint shrink? What happens to accuracy?"
*   **The Workbench:**
    *   **Action:** Toggle between **FP16**, **INT8**, **INT4**, and **Binary (1-bit)** weights.
    *   **Observation:** The **Accuracy-vs-Bits Pareto Plot**. Watch the model footprint shrink linearly while accuracy stays flat until the 'Quantization Cliff.'
*   **Reflect:** "Patterson asks: 'Why is INT8 often called a Free Lunch?' Identify the exact bit-width where your mission's accuracy falls below the safety threshold."

### Part 2: The Sparsity Frontier (Trade-off - 15 Mins)
*   **Objective:** Balance Pruning Ratio against Inference Speed.
*   **The "Lock" (Prediction):** "Does 50% unstructured pruning result in a 2x speedup on your specific hardware? (Hint: See the 'Software Tax' in Chapter 7)."
*   **The Workbench:**
    *   **Sliders:** Pruning Ratio (0-90%), Pruning Type (Unstructured vs. 2:4 Structured).
    *   **Instruments:** **Throughput-vs-Sparsity Waterfall**. Compare 'Theoretical Speedup' vs 'Realized Hardware Speedup.'
    *   **The 10-Iteration Rule:** Students must find the exact pruning ratio that hits their Latency target without triggering a 'Stakeholder Accuracy Veto.'
*   **Reflect:** "Jeff Dean observes: 'Your model is 50% smaller but only 5% faster.' Use the concept of 'Hardware-Algorithm Alignment' to explain this discrepancy."

### Part 3: The Distillation Bridge (Synthesis - 15 Mins)
*   **Objective:** Use a 'Teacher' model to recover the accuracy lost during aggressive compression.
*   **The "Lock" (Prediction):** "Can a 4-bit model trained with Distillation outperform an 8-bit model trained from scratch?"
*   **The Workbench:**
    *   **Interaction:** **Distillation Toggle**. **Temperature Slider**. **Student Model Selection**.
    *   **The "Stakeholder" Challenge:** The **UX Designer** (Mobile) or **Hardware Lead** (Tiny) demands a further 20% power reduction. You must use the 'Distillation Bridge' to move to an even smaller student model while staying 'In-Spec.'
*   **Reflect (The Ledger):** "Defend your final 'Silicon Contract.' Did you choose a 'Heavy' model with aggressive quantization or a 'Light' student model with high precision? Justify your choice using the Energy-per-Inference metric."

---

## 4. Visual Layout Specification
*   **Primary:** `ParetoFrontierPlot` (X-axis: Model Size, Y-axis: Accuracy, Bubbles: Energy/Latency).
*   **Secondary:** `CompressionWaterfall` (Weights vs Activations vs KV-Cache footprint).
*   **Math Peek:** Toggle for `Quantization Error` and `Sparsity Speedup` formulas.

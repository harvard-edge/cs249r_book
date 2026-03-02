# 📐 Mission Plan: 05_nn_compute (Deep Analysis)

## 1. Chapter Context
*   **Chapter Title:** Neural Computation: The Mechanics of ML.
*   **Core Invariant:** Arithmetic Intensity (FLOPs / Byte).
*   **The Struggle:** Understanding that not all neural network layers are created equal. Some layers starve the compute cores (Memory Bound), while others choke the arithmetic logic units (Compute Bound).
*   **Target Duration:** 45 Minutes.

---

## 2. The 4-Track Storyboard

| Track | Persona | Fixed North Star Mission | The "Intensity" Crisis |
| :--- | :--- | :--- | :--- |
| **Cloud Titan** | LLM Architect | Maximize Llama-3-70B serving. | **The Attention Sink.** The KV-cache forces attention layers into a brutally low Arithmetic Intensity. |
| **Edge Guardian** | AV Systems Lead | Deterministic 10ms safety loop. | **The High-Res Tax.** Processing 4K images makes early convolutions extremely compute-heavy. |
| **Mobile Nomad** | AR Glasses Dev | 60FPS AR translation. | **The Depthwise Compromise.** Standard convolutions burn too much power; must trade accuracy for intensity. |
| **Tiny Pioneer** | Hearable Lead | Neural isolation in <10ms. | **The Matrix Limit.** Standard matrix multiplications simply will not fit in the cache hierarchy. |

---

## 3. The 3-Part Mission (The KATs)

### Part 1: The Arithmetic Intensity Audit (Exploration - 15 Mins)
*   **Objective:** Calculate and compare the Arithmetic Intensity of different neural network primitives.
*   **The "Lock" (Prediction):** "Calculate the theoretical Arithmetic Intensity (FLOPs/Byte) of a Fully Connected (Linear) layer vs. a Convolutional layer. Which is more likely to hit the Memory Wall?"
*   **The Workbench:**
    *   **Action:** Select specific layers from your Track's Lighthouse Model (e.g., Llama Attention vs. FFN).
    *   **Observation:** The **Layer Profiler Card**. It displays exact MAC counts, Byte sizes, and calculates `Intensity = Ops / Bytes`.
*   **Reflect:** "Reconcile your prediction. Why do convolutions naturally have higher arithmetic intensity than dense linear layers? Use the concept of 'Weight Reuse' in your answer."

### Part 2: The Scaling Reality Check (Trade-off - 15 Mins)
*   **Objective:** Witness how sequence length and batch size non-linearly affect the compute-to-memory ratio.
*   **The "Lock" (Prediction):** "If you double the sequence length of an Attention block, does the Arithmetic Intensity increase, decrease, or stay the same?"
*   **The Workbench:**
    *   **Sliders:** Batch Size, Sequence Length (or Image Resolution for Vision tracks).
    *   **Instruments:** **Intensity Trajectory Plot**. Shows Ops growing quadratically ($O(N^2)$) while memory footprint grows differently.
    *   **The 5-Move Rule:** Students must slide the parameters to find the "Crossover Point" where the layer flips from memory-bound to compute-bound on their specific hardware.
*   **Reflect:** "At what sequence length/resolution did your model hit the hardware's 'Ridge Point'? Why does scaling this dimension eventually break the system?"

### Part 3: The Operator Budget (Synthesis - 15 Mins)
*   **Objective:** Redesign a network backbone by swapping operators to meet a strict hardware constraint.
*   **The "Lock" (Prediction):** "Replacing standard convolutions with Depthwise-Separable convolutions reduces FLOPs by ~8x. What does it do to Arithmetic Intensity?"
*   **The Workbench:** All sliders unlocked. A "Lego Toolkit" of operators (Linear, Conv2D, DW-Conv, FlashAttention).
*   **The "Stakeholder" Challenge:** The Hardware Lead informs you that the current architecture sits perfectly on the memory wall, but is exceeding the Thermal Budget (TDP). You must swap operators to reduce total energy without falling below an Intensity of 20 FLOPs/Byte.
*   **Reflect (The Ledger):** "Defend your final operator mix. Did you sacrifice Arithmetic Intensity to save power? Explain the 'Conservation of Complexity' in your design."

---

## 4. Visual Layout Specification
*   **Primary:** `LayerProfiler` (A table showing MACs, Bytes, and Intensity gauge).
*   **Secondary:** `IntensityTrajectoryPlot` (X-axis: Parameter Scale, Y-axis: Intensity).
*   **Math Peeking:** Toggle for `Intensity = \frac{2 	imes M 	imes N 	imes K}{(M 	imes K + K 	imes N + M 	imes N) 	imes 	ext{bytes}}`.

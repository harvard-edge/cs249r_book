# Mission Plan: lab_05_nn_compute

## 1. Chapter Alignment

- **Chapter:** Neural Computation (`@sec-neural-computation`)
- **Core Invariant:** The **Compute Graph** — every prediction is a chain of matrix multiplications interleaved with activation functions, and every design choice (layer width, activation type, batch size) has a direct, quantifiable cost in operations, memory, and energy.
- **The Chapter's Central Tension:** Neural networks are simple atoms (multiply, add, activate) repeated at enormous scale. The "bug" is never in the logic — it is always in the math: a saturated activation silently blocks learning, a memory footprint that fits in development exhausts the accelerator in production, a layer that dominates 92% of compute goes unnoticed until profiling.
- **Target Duration:** 35–40 minutes (2 acts)

---

## 2. The Two-Act Structure

This lab uses the **2-Act format**: one focused calibration followed by one open design challenge. No 3-KAT structure.

---

## 3. Act 1: The Transistor Tax (Calibration — 12 minutes)

### Pedagogical Goal
Students believe activation functions are "free" — just a nonlinearity tacked onto a matrix multiply. The chapter's central claim is the opposite: the choice of activation function is a *hardware design decision* with a 50× silicon cost difference. This act forces a wrong prediction, then reveals the gap.

### The Lock (Structured Prediction)
Present students with a **multiple-choice prediction** before any instruments unlock:

> "A ReLU unit and a Sigmoid unit both produce an output from the same input. In terms of transistor count, how much more expensive is Sigmoid than ReLU?"

Options:
- A) About the same (~1–2×)
- B) About 5×
- C) About 20×
- D) **About 50×** ← correct

Students must select one option. The selection is recorded and displayed throughout the rest of the act.

### The Instrument: Activation Cost Comparator

A side-by-side panel for four activation functions: **Sigmoid, Tanh, ReLU, GELU**.

For each, show:
- **Transistor count** (from chapter constants: ReLU ≈ 50, Sigmoid/Tanh ≈ 2,500)
- **Cycles per evaluation** (ReLU: 1 cycle; Sigmoid: 20–40 cycles)
- **Power ratio** relative to ReLU (1×, ~10×, ~10×, ~3×)
- **Saturation behavior**: a small activation curve with the gradient plotted below it — students can see where the gradient collapses to zero for sigmoid/tanh
- **Dying neuron risk** for ReLU (10–40% of neurons can die)

A **single slider** controls the network depth (1–20 layers). As depth increases, show:
- For sigmoid/tanh: gradient magnitude after backpropagation = $0.25^{\text{depth}}$, plotted on a log scale
- For ReLU: gradient stays near 1.0 for positive activations
- A threshold line at $10^{-6}$ labeled "Learning becomes impossible"

### The Reveal
After interaction, overlay the student's prediction on the actual 50× ratio:
> "You predicted [X]. The actual silicon cost ratio is **50×**. Your prediction was off by [Y]×."

Then surface the systems implication:
> "At 1,000 neurons per layer × 10 layers = 10,000 activation evaluations per forward pass. Choosing sigmoid over ReLU adds ~10,000 × 2,450 extra transistors of thermal load per inference."

### Reflection (Structured)
**Not free text.** Students complete the sentence:

> "Sigmoid blocks learning in deep networks because ______. This is why the chapter calls it a [silent / loud / gradual] failure."

Dropdown options for the second blank: `silent` / `loud` / `gradual` → only `silent` is correct; the lab explains why (gradient collapse produces no error message, loss just plateaus).

**Math Peek (collapsible):**
$$\sigma'(x) \leq 0.25 \implies \text{after } L \text{ layers: gradient} \approx 0.25^L$$

---

## 4. Act 2: The Memory Wall of Training (Design Challenge — 22 minutes)

### Pedagogical Goal
Students do not realize that training the same network they just profiled requires storing every intermediate activation for the backward pass. The chapter's claim is: training uses ~4× more memory than inference. This act makes students *discover* that number by building the memory budget themselves, then hit the wall when they try to scale.

### The Lock (Structured Prediction)
**Numeric range prediction** before instruments unlock:

> "The 784→128→64→10 MNIST network uses ~427 KB of memory at inference time. How much memory does training the same network require (batch size = 32)? Enter your estimate in KB."

Students type a number. The system records it. After the act, it shows: "You estimated [X] KB. The actual answer is [~Y] KB."

### The Instrument: The Training Memory Ledger

An interactive memory decomposition for the MNIST network (784→128→64→10, batch=32).

**Four stacked bars**, each toggleable:

| Component | Formula | Approximate Size |
|---|---|---|
| **Weights** | params × bytes_per_param | ~427 KB (FP32) |
| **Gradients** | same as weights | ~427 KB |
| **Optimizer State (Adam)** | 2× weights (momentum + velocity) | ~854 KB |
| **Activations** | sum of all layer outputs × batch_size | computed live |

Students control:
- **Batch size slider** (1, 8, 16, 32, 64, 128) — activations scale linearly with batch size
- **Precision toggle** (FP32 / FP16) — halves weights, gradients, optimizer state
- **Optimizer selector** (SGD vs. Adam) — SGD has no optimizer state beyond weights

The **Activations bar** is computed from the chapter's formula:
- Layer 0 (input): batch × 784 values
- Layer 1: batch × 128 values
- Layer 2: batch × 64 values
- Layer 3 (output): batch × 10 values
- Total activation memory = sum × bytes_per_value

A **red threshold line** appears at the device memory budget — students can select their deployment target:
- H100 (80 GB)
- Laptop GPU (8 GB)
- Mobile GPU (2 GB)
- Microcontroller (256 KB)

When total training memory exceeds the threshold, the bar chart turns red and shows: **"OOM: Training infeasible on this device."**

### The Scaling Challenge

A second panel below the bar chart: the same 784→**W**→**W/2**→10 architecture, where **W** (first hidden layer width) is a slider from 64 to 4096.

Students must answer: **"Find the maximum W where training with batch=32 fits on a Laptop GPU (8 GB)."**

The system tracks the student's moves and records the W they converge on.

Key discovery: the first layer dominates. At default (W=128), layer 1 accounts for 92% of forward-pass MACs. As W scales, the first layer grows quadratically in both compute and activation storage.

Show a **layer contribution pie chart** that updates live — students see the 92% claim from the chapter become visible in the chart before they understand why.

### The Backprop Reveal

After students find a valid configuration, surface the chapter's 3× claim:

> "You just sized the forward pass. Training requires: 1× forward pass + ~2× backward pass = **~3× total compute** per batch. Why? Because the backward pass must recompute gradients through every layer, using the activations you stored."

Show a **Waterfall chart**: three bars — Forward (1×), Backward (~2×), Optimizer Update (~0.1×). Total = ~3.1×.

**Math Peek:**
$$\text{Training Memory} \approx \underbrace{W}_{\text{weights}} + \underbrace{W}_{\text{gradients}} + \underbrace{2W}_{\text{Adam state}} + \underbrace{\sum_l B \cdot n_l}_{\text{activations}}$$

### Reflection (Structured)

Students select which statement is correct:

> "Activations must be stored during training because:"
- A) The optimizer needs them to update weights directly
- B) **The backward pass uses them to compute weight gradients via the chain rule** ← correct
- C) They reduce memory by avoiding recomputation
- D) The framework caches them for debugging

Then write one sentence:
> "If I increase batch size from 32 to 64, training memory grows by approximately ___× because ___."

(Expected answer: ~2×, because activations scale linearly with batch size while weights/gradients/optimizer state do not change.)

---

## 5. Visual Layout Specification

### Act 1: Transistor Tax
- **Primary:** Activation Cost Comparator (side-by-side stat cards per activation function)
- **Secondary:** Gradient Magnitude vs. Depth plot (log scale, threshold line at $10^{-6}$)
- **Prediction overlay:** Student's selected option highlighted, correct answer revealed with gap annotation

### Act 2: Training Memory Ledger
- **Primary:** Stacked bar chart (4 components × toggleable) with device threshold line
- **Secondary:** Layer contribution pie chart (live-updating, shows 92% layer-1 dominance)
- **Tertiary:** 3× compute waterfall (Forward / Backward / Update)
- **Failure state:** Full OOM crash visual when memory exceeds device budget

---

## 6. Two-Track Variant (Deployment Context Comparison)

Rather than 4 narrative personas, this lab offers **2 deployment contexts** as a comparison toggle (not a persistent identity):

| Context | Device Budget | Key Constraint |
|---|---|---|
| **Training Node** | H100 (80 GB) | Can batch=1024; optimizer state is affordable |
| **Edge Inference** | Mobile GPU (2 GB) | Inference only; no gradient storage needed |

Students switch between contexts mid-lab to discover: the *same* network that comfortably fits for inference is infeasible to *train* on a mobile device. This is a concrete discovery, not a narrative.

---

## 7. Design Ledger Output

At lab completion, the student's Design Ledger records:

```json
{
  "chapter": 5,
  "activation_choice": "relu",
  "max_trainable_width_laptop_gpu": <W>,
  "training_memory_estimate_error_kb": <estimate - actual>,
  "batch_size_chosen": 32
}
```

The `activation_choice` and `max_trainable_width_laptop_gpu` values feed forward to:
- **Lab 08 (Training):** The activation choice affects gradient stability displays
- **Lab 10 (Compression):** The max width becomes the starting point for compression targets

---

## 8. Connection to Chapter Content (Traceability)

| Lab Element | Chapter Section | Chapter Claim Being Tested |
|---|---|---|
| 50× transistor ratio | `@sec-neural-computation-artificial-neuron-computing-primitive-45b4` | "Selecting Sigmoid over ReLU increases silicon cost by 50×" |
| Sigmoid gradient collapse | footnote on gradient instabilities, line 146 | "$0.25^{20} \approx 10^{-12}$" |
| 92% first-layer dominance | `MNISTInference` class, line 2822 | "Layer 1 accounts for 92% of all operations" |
| 4× training memory | `MNISTMemory` class; training_ratio check | "Training requires ~4× more memory than inference" |
| 3× compute for training | line 488, checkpoint line 3113 | "roughly 3× the forward cost" |
| Activation storage requirement | eq-training-memory, lines 2975–2983 | "store every intermediate activation until the backward pass reaches that layer" |
| Memory wall / L1 cache miss | footnote line 539 | "L1 cache delivers data in ~1 ns; main memory takes ~100 ns — a 100× gap" |

# Mission Plan: lab_08_model_train

## 1. Chapter Alignment

- **Chapter:** Model Training (`@sec-model-training`)
- **Core Invariant:** The **Iron Law of Training Performance** — $T_{train} = O_{total} / (N \cdot R_{peak} \cdot \eta)$ — where the hardware peak ($R_{peak}$) is fixed and the only levers are total operations ($O_{total}$) and utilization ($\eta$). This lab is strictly single-machine scope: one accelerator, one training loop, no network bandwidth.
- **Central Tension:** Students believe that buying a faster GPU is the primary lever for training speed. The chapter's data reveals the opposite: real systems operate at 45–55% MFU, not 90–100%. The bottleneck is almost never peak FLOPS — it is the memory hierarchy (optimizer state, activation storage, pipeline stalls). Optimization means eliminating waste in $\eta$, not upgrading $R_{peak}$.
- **Target Duration:** 35–40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** The student has just read that Adam is the standard optimizer. They believe the cost is negligible — it is "just an optimizer." This act forces them to compute the actual memory breakdown for GPT-2 training: weights + gradients + Adam state = 4× the inference footprint, before activations even appear. The prediction question exploits the specific wrong prior that Adam costs "about the same" as SGD.

**Act 2 (Design Challenge, 22 min):** The student believes that saturating a GPU means high TFLOPS utilization. This act shows the GPT-2 training walkthrough from the chapter: the *baseline* system achieves only 45% MFU, with 40% of iteration time consumed by data loading — not by gradient computation. Students must diagnose the bottleneck using the pipeline breakdown, then apply optimizations (mixed precision, gradient checkpointing, prefetching) to hit a target MFU of 80%. Each optimization has a quantified cost and benefit.

---

## 3. Act 1: The Optimizer Memory Tax (Calibration — 12 minutes)

### Pedagogical Goal
Students treat optimizer choice as a convergence decision ("Adam converges faster than SGD") not a memory decision. The chapter's claim is precise: SGD requires 1× model memory, Momentum 2×, Adam 3× — *before* activations. For GPT-2 XL (1.5B parameters), switching from SGD to Adam adds 12 GB of GPU memory. This act makes students predict that number and then confront it.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "You are training GPT-2 XL (1.5 billion parameters) in FP32. You switch from SGD to Adam. How much additional GPU memory does Adam require compared to SGD?"

Options:
- A) About the same — the optimizer doesn't store model data (~0 GB extra)
- B) About 3 GB extra — it stores one extra vector
- C) **About 12 GB extra — it stores two extra vectors (momentum + velocity)** ← correct
- D) About 24 GB extra — it replicates the full model for safety

The correct answer requires knowing: Adam stores $m_t$ and $v_t$, each the same size as the gradient vector; at FP32 and 1.5B parameters, each vector = 6 GB → 12 GB total additional.

### The Instrument: Training Memory Ledger

A stacked bar chart for GPT-2 XL training memory decomposition. Four components, each toggleable:

| Component | Formula | GPT-2 XL (FP32) |
|---|---|---|
| **Parameters** | `params × bytes_per_param` | 6 GB |
| **Gradients** | same as parameters | 6 GB |
| **Adam State (m_t)** | same as parameters | 6 GB |
| **Adam State (v_t)** | same as parameters | 6 GB |
| **Activations** | `batch × sum_of_layer_outputs × bytes` | varies |

Controls:
- **Optimizer selector**: SGD (params + gradients only) → Momentum (+ 1 vector) → Adam (+ 2 vectors). Each selection adds/removes bars live.
- **Precision toggle**: FP32 (4 bytes) / FP16 (2 bytes) / Mixed (FP16 compute + FP32 master copy). Mixed precision halves the active compute buffers but keeps a FP32 master copy — students discover this is not simply "2× better."
- **Batch size slider**: 1, 8, 16, 32, 64 — only activations scale with batch size; all other bars are constant.

A **red threshold line** marks the device memory budget. Two contexts selectable:
- **Training Node**: H100 (80 GB)
- **Laptop GPU**: 8 GB

When total memory exceeds threshold: bars turn red, banner reads **"OOM — Optimizer state alone exceeds device memory."**

### The Reveal
After exploration, overlay the prediction:
> "You predicted [X] GB of additional memory. Adam actually adds **12 GB** for GPT-2 XL in FP32 — about [Y]× your estimate. This is why the chapter states Adam requires 3× SGD's memory: parameters (6 GB) + gradients (6 GB) + Adam state (12 GB) = 24 GB static memory before a single activation is stored."

Then surface the convergence trade-off:
> "GPT-2 converges in ~50,000 steps with Adam vs. ~150,000+ steps with SGD+Momentum. Is 12 GB of extra memory worth 3× fewer training steps? This is an engineering decision, not a preference."

### Reflection (Structured)
Students complete:

> "Mixed precision reduces memory by approximately ___× because ___."

Dropdown options for blank 1: `1.3×` / `1.5×` / **`2×`** / `4×`
Dropdown options for blank 2:
- **"active compute buffers (parameters, gradients) halve from FP32 to FP16, while a FP32 master copy is retained for optimizer updates"** ← correct
- "the GPU compresses all tensors automatically"
- "optimizer state is not needed in FP16 mode"
- "activation memory dominates and activations are always stored in INT8"

**Math Peek (collapsible):**
$$\text{Training Memory} = \underbrace{W}_{\text{params}} + \underbrace{W}_{\text{grads}} + \underbrace{2W}_{\text{Adam}} + \underbrace{\sum_l B \cdot n_l \cdot \text{bytes}}_{\text{activations}}$$

where $W = \text{params} \times \text{bytes\_per\_param}$, $B$ = batch size, $n_l$ = layer width.

---

## 4. Act 2: The MFU Gap (Design Challenge — 22 minutes)

### Pedagogical Goal
Students expect that a well-configured training run uses 80–95% of peak GPU FLOPS. The chapter's GPT-2 walkthrough shows the baseline reality is 45% MFU — with 40% of wall-clock time consumed by data loading and 25% by memory transfers, leaving only 35% for actual compute. Students must diagnose this using the pipeline breakdown, then apply three specific optimizations to reach the chapter's target of 85% MFU. Each optimization has a concrete cost: gradient checkpointing adds 33% recompute overhead; prefetching adds implementation complexity but recovers 35 percentage points of compute time.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "A GPT-2 XL training loop on an A100 is fully configured — correct batch size, no bugs, no data augmentation. In a single training iteration, what fraction of wall-clock time is the GPU actually performing tensor operations (not waiting for data or transfers)?"

Students type a percentage (0–100). The system records it. Expected wrong answers: 70–95%. Actual answer from chapter: 35% (compute) at baseline.

### The Instrument: Pipeline Breakdown Analyzer

A **stacked horizontal bar** showing one training iteration divided into three phases:

| Phase | Baseline | Optimized |
|---|---|---|
| **Data Loading** | 40% of iteration | 5% (overlapped prefetch) |
| **Memory Transfers** | 25% of iteration | 20% (mixed precision reduces volume) |
| **Compute** | 35% of iteration | 75% |

Controls:
- **Prefetching toggle** (Off / On): When On, data loading drops from 40% → 5% (overlapped with prior step's compute). This is the highest-leverage single change.
- **Mixed precision toggle** (FP32 / Mixed): Memory transfer bar shrinks because FP16 activations are half the volume; compute bar grows.
- **Gradient checkpointing toggle** (Off / On): Activation memory bar in the memory panel shrinks 4×; a new "Recompute Overhead" segment appears in the compute bar (+33% of compute time, but allows 4× larger batch or deeper network).
- **Batch size slider** (16, 32, 64, 128, 256): Shows the 60–70% → 90%+ utilization transition the chapter describes.

A **MFU gauge** (0–100%) updates live:
$$\text{MFU} = \frac{O_{total}}{T_{wall} \cdot R_{peak}}$$

Baseline: 45%. Target: ≥ 80%. Students must apply optimizations to cross the threshold.

**Secondary instrument:** The Optimizer Walkthrough Summary — a side panel showing the chapter's GPT-2 scenario numbers in a table:

| Metric | Baseline | After Optimization |
|---|---|---|
| Memory | 89 GB | 32 GB |
| MFU | 45% | 85% |
| Throughput | — | 1,200 tokens/sec |
| Data loading share | 40% | 5% |

Students can compare their optimized configuration against this reference.

### The Scaling Challenge
A second panel: **"Fit GPT-2 XL training on a Laptop GPU (8 GB)."**

The student starts with the baseline configuration (89 GB total) and must apply a combination of:
- FP16 mixed precision (halves active buffers)
- Switch optimizer from Adam to SGD (saves 12 GB, costs 3× more iterations)
- Gradient checkpointing (saves ~4× activation memory, adds 33% compute overhead)
- Reduce batch size to 1

They must find a valid configuration that fits within 8 GB. The system tracks their move sequence.

Key discovery: even with every optimization applied, GPT-2 XL cannot train in 8 GB at batch ≥ 4. The student must quantify why — and the answer is the static memory floor (parameters + gradients + optimizer state) is already 24 GB in the best case (SGD + FP16). To train GPT-2 XL on a laptop GPU, you would need parameter sharding — which requires multiple machines and is out of scope for Volume 1.

**Failure state:** When the student has applied all available optimizations and still cannot fit:
> "🔴 **Physical Limit Reached.** With all single-node optimizations applied, minimum static memory = 18 GB (SGD + FP16). GPT-2 XL cannot train on a Laptop GPU. This boundary is the entry condition for Volume 2: parameter sharding across nodes."

### Structured Reflection
Four-option multiple choice:

> "Your pipeline analysis shows 40% of time is spent on data loading. The most effective single fix is:"
- A) Buy a GPU with 2× more TFLOPS
- B) Switch from Adam to SGD to reduce memory pressure
- **C) Enable prefetching so data loading overlaps with the previous step's compute** ← correct
- D) Reduce batch size to minimize the memory transfer volume

Then complete the sentence:
> "Gradient checkpointing reduces activation memory by ___× at the cost of ___% additional compute, because ___."

Expected fill-in: 4×, 33%, "the backward pass must recompute intermediate activations that were discarded rather than stored."

**Math Peek:**
$$\text{MFU} = \frac{O_{total}}{T_{wall} \cdot N \cdot R_{peak}} \qquad \eta \approx 0.45 \text{ (GPT-3 baseline)}, \quad \eta > 0.55 \text{ (current target)}$$

---

## 5. Visual Layout Specification

### Act 1: Optimizer Memory Tax
- **Primary:** Stacked vertical bar chart — 4–6 components (params, grads, Adam m, Adam v, activations), toggleable per component. Y-axis: GB (0–100). Device threshold line in red.
- **Secondary:** Optimizer comparison table (SGD / Momentum / Adam) with memory multiplier column (1× / 2× / 3×) and convergence steps column (150K+ / 90K / 50K).
- **Prediction overlay:** Student's selected option highlighted; correct bar annotated with "12 GB gap."
- **Failure state:** All bars turn RedLine (#CB202D) when memory > device budget; banner appears.

### Act 2: MFU Gap
- **Primary:** Stacked horizontal bar (one iteration = 100%): Data Loading / Memory Transfers / Compute. Three segments, colors: OrangeLine / BlueLine / GreenLine.
- **Secondary:** MFU gauge (semicircle, 0–100%, threshold line at 80%).
- **Tertiary:** Optimizer Walkthrough Summary table (baseline vs. optimized reference numbers from chapter).
- **Laptop GPU scaling panel:** Memory bar starting at 89 GB, must reach ≤ 8 GB; failure state when all optimizations exhausted above 8 GB.

---

## 6. Deployment Context Definitions

| Context | Device | Memory | Power Budget | Key Constraint |
|---|---|---|---|---|
| **Training Node** | H100 (80 GB HBM3) | 80 GB | 700 W TDP | Maximize MFU; optimizer state is a manageable fraction |
| **Laptop GPU** | RTX 4060 (8 GB GDDR6) | 8 GB | 115 W TDP | Static memory floor of GPT-2 XL exceeds budget even at minimum configuration |

The two contexts isolate the chapter's key insight: single-node optimization can recover 40 percentage points of MFU, but cannot overcome the memory floor for large models. That boundary is a precise, computable number — not a vague "scale limit."

---

## 7. Design Ledger Output

```json
{
  "chapter": 8,
  "optimizer_chosen": "adam | sgd | momentum",
  "precision_chosen": "fp32 | mixed | fp16",
  "gradient_checkpointing": true,
  "final_mfu_pct": 82,
  "baseline_mfu_estimate_pct": 75,
  "laptop_fit_achieved": false,
  "laptop_minimum_memory_gb": 18
}
```

The `optimizer_chosen` and `precision_chosen` fields feed forward to:
- **Lab 10 (Compression):** The precision choice affects the quantization baseline comparison
- **Lab 11 (HW Acceleration):** The MFU value becomes the starting point for roofline analysis

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| SGD = 1×, Momentum = 2×, Adam = 3× memory | Line 992 | "Memory requirements increase progressively from SGD (1× model size) through Momentum (2×) to Adam (3×)" |
| GPT-2 XL = 1.5B params, 6 GB FP32 | Lighthouse spec, line 327 | "~6 GB (FP32) for weights alone" |
| Adam adds 12 GB for GPT-2 XL | Lines 1084–1085 | "1.5B × 8 bytes = [Adam state]"; each vector = 6 GB, two vectors = 12 GB |
| 50K steps (Adam) vs 150K steps (SGD) | Line 1095 | "GPT-2 converges in ~50K steps…~150K+ steps with SGD+Momentum" |
| Mixed precision saves ~50% memory | Line 3463 | "memory consumption decreases by approximately 50%" |
| Baseline MFU = 45% | Line 4612 | "accelerator utilization: 45%" |
| Data loading = 40% of iteration time (baseline) | Line 4613 | "Data loading: 40% of iteration time" |
| After optimization: compute = 75%, data loading = 5% | Lines 4628–4629 | "Data loading: 5%…Compute: 75% of iteration time (overlapped)" |
| Optimized MFU = 85% | Line 4626 | "accelerator utilization: 85%" |
| Memory reduction: 89 GB → 32 GB | Line 4636 | "Naive: 89 GB…Optimized: 32 GB…2.8× reduction" |
| Gradient checkpointing: 4× activation reduction | Line 4604 | "reduces activations by 4×" |
| Gradient checkpointing: +33% compute overhead | Line 4604 | "33% more compute for activation recomputation" |
| Batch ≥ 256 → >90% utilization | Line 908 | "batch sizes of 256 or higher typically achieve over 90% hardware utilization" |
| Batch 16–32 → 60–70% utilization | Line 908 | "smaller batches of 16–32 may only achieve 60–70% utilization" |
| GPT-3 baseline η ≈ 0.45 | Line 223 | "GPT-3 training achieved η ≈ 0.45" |
| Current systems target η > 0.55 | Line 223 | "current systems target η > 0.55" |

# Mission Plan: lab_11_edge_intel

## 1. Chapter Alignment

- **Chapter:** Edge Intelligence (`@sec-edge-intelligence`)
- **Core Invariant:** The **Memory Amplification Factor** -- on-device training requires 3-5x more memory than inference for the same model, and energy consumption balloons 10-50x, because backpropagation demands storing activations, gradients, and optimizer state simultaneously.
- **Central Tension:** Students believe that federated learning is "just distributed training on phones" and that more communication rounds always improve convergence. The chapter's convergence analysis demolishes both: non-IID data causes a 28x increase in required communication rounds compared to IID baselines, and aggressive local computation (large E) actually degrades convergence under heterogeneous data distributions. The federation paradox is that centralized training converges faster and cheaper, but privacy and bandwidth constraints force decentralized approaches that are communication-dominated.
- **Target Duration:** 35-40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that on-device training is a modest extension of on-device inference -- perhaps 1.5-2x more resource-intensive. The chapter's data shows memory amplification of 4-12x across model scales (1M to 100M parameters), with the dominant cost not in weights but in optimizer state and activations. This act forces students to predict the memory cost of fine-tuning a 10M parameter model on a smartphone, then reveals that the 40 MB inference footprint balloons to over 200 MB during backpropagation, competing directly with the OS and foreground apps for the device's 8 GB RAM.

**Act 2 (Design Challenge, 22 min):** Students confront the federation paradox directly. They must configure a federated learning system to reach a target accuracy within a fixed communication budget. The chapter's convergence bound shows that non-IID data introduces a heterogeneity penalty beta that can require 28x more communication rounds. Students discover that increasing local epochs (E) helps under IID conditions but degrades convergence under non-IID conditions beyond an optimal point of E=2-5. The design challenge is to find the operating point that satisfies both accuracy and communication constraints for a given data heterogeneity level.

---

## 3. Act 1: The Memory Amplification Tax (Calibration -- 12 minutes)

### Pedagogical Goal
Students dramatically underestimate the memory cost of on-device training versus inference. They think of training as "inference plus a bit extra for gradients." The chapter shows that for a 10M parameter model, 40 MB of FP32 weights spike to over 200 MB during backpropagation as activations, gradients, and optimizer states accumulate. The amplification factor ranges from 4x (1M params) to 12x (100M params). This act calibrates the student's intuition about why on-device learning is an engineering crisis, not a minor extension.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "A 10M parameter model requires 40 MB for inference (FP32 weights only). You want to fine-tune the last 3 layers on a smartphone. How much total memory does on-device training require at peak (during the backward pass)?"

Options:
- A) ~60 MB -- weights plus gradients for 3 layers (~1.5x)
- B) ~120 MB -- roughly 3x for weights, gradients, and optimizer state
- **C) ~200-360 MB -- 5-9x amplification from activations, gradients, and optimizer state** (correct)
- D) ~40 MB -- fine-tuning only a few layers keeps memory the same as inference

The correct answer is C because training requires storing the full activation cache (for backpropagation), gradients for all trainable parameters, and optimizer state (2x parameter size for Adam). Students who pick A or B underestimate the activation cache; students who pick D confuse parameter-efficient methods with zero overhead.

### The Instrument: Memory Amplification Waterfall

A **stacked bar chart** showing memory breakdown for inference vs. training:

- **X-axis:** Model scale categories: 1M Params, 10M Params, 100M Params
- **Y-axis:** Memory requirement (MB, log scale, range 1-10,000)
- **Bars per group:** Two bars -- Inference (weights only) and Training (weights + gradients + optimizer state + activations)
- **Stacked components for training bar:** Weights (gray), Gradients (orange), Optimizer State (dark orange), Activations (red)
- **Amplification labels:** "4.0x", "9.0x", "12.0x" annotated above each training bar

Controls:
- **Model scale slider:** 1M / 5M / 10M / 50M / 100M parameters (default: 10M)
- **Adaptation strategy toggle:** Full Fine-Tune / LoRA (rank-16) / Weight Freezing (last 3 layers)
  - Full Fine-Tune: all components scale fully
  - LoRA (rank-16): trainable parameters drop to ~0.7% of original (per chapter: 7B model goes from 14 GB to ~100 MB)
  - Weight Freezing: gradients and optimizer state only for frozen layers are zero
- **Device RAM indicator:** 8 GB total, ~300 MB available for ML (per chapter: smartphones allocate 200-300 MB to background apps)

### The Reveal
After interaction:
> "You predicted [X]. A 10M parameter model requires approximately **360 MB** at peak during full fine-tuning -- a **9x amplification** over the 40 MB inference footprint. On a smartphone with 300 MB available, this is an OOM crash. LoRA with rank-16 reduces trainable parameters to 0.7%, bringing peak memory to ~80 MB -- the only feasible path for on-device adaptation."

### Reflection (Structured)
Four-option multiple choice:

> "The chapter states that on-device training amplifies memory requirements by 3-5x (and up to 12x at scale). What is the single largest contributor to this amplification?"
- A) Gradients -- backpropagation requires storing a gradient for every weight
- B) Optimizer state -- Adam stores two moment vectors per parameter (3x weight memory)
- **C) Activation cache -- the forward pass must store intermediate outputs at every layer for backpropagation** (correct)
- D) Model weights -- training uses FP32 while inference can use INT8

### Math Peek (collapsible)
$$\text{Memory}_{\text{train}} = \underbrace{W}_{\text{weights}} + \underbrace{W}_{\text{gradients}} + \underbrace{2W}_{\text{optimizer (Adam)}} + \underbrace{A \cdot B \cdot L}_{\text{activations}}$$
where $W$ = parameter memory, $A$ = activation size per layer, $B$ = batch size, $L$ = number of layers. Activations dominate because they scale with batch size and depth, not just parameter count.

---

## 4. Act 2: The Federation Paradox (Design Challenge -- 22 minutes)

### Pedagogical Goal
Students believe that federated learning is simply "distributed training but private" and that more communication rounds monotonically improve convergence. The chapter's convergence analysis reveals that non-IID data introduces a heterogeneity penalty beta that can require 28x more communication rounds than IID baselines. Worse, increasing local epochs (E) -- which reduces communication frequency -- actually degrades convergence under heterogeneous data beyond E=2-5 due to client drift. Students must find the operating point where communication cost and convergence rate are jointly satisfactory for a given heterogeneity level.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "A federated learning system with IID data across 100 clients requires 200 communication rounds to reach 95% accuracy. The same system with highly non-IID data (heterogeneity factor beta=1.0) requires how many rounds to reach the same accuracy?"

Students type a number. Expected wrong answers: 300-600 (students expect a modest 1.5-3x increase). Actual: approximately **5,625 rounds** -- a 28x increase, because the convergence bound includes an additive beta^2/epsilon term that dominates under heterogeneity.

### The Instrument: Convergence vs. Communication Plot

A **dual-axis line plot** showing convergence behavior:

- **X-axis:** Communication rounds (0-10,000)
- **Y-axis (left):** Global model accuracy (50%-100%)
- **Y-axis (right):** Cumulative data transferred (GB)
- **Lines:**
  - IID convergence curve (solid BlueLine): reaches 95% by round 200
  - Non-IID convergence curve (dashed OrangeLine): reaches 95% by round ~5,600
  - Communication cost line (dotted RedLine): linear growth with rounds

Controls:
- **Data heterogeneity (beta):** 0.0 (IID) / 0.25 / 0.5 / 1.0 / 2.0 (default: 0.0)
- **Local epochs (E):** 1 / 2 / 5 / 10 / 20 (default: 5)
- **Number of participating clients (C):** 10 / 50 / 100 / 500 (default: 100)
- **Communication budget cap:** 100 GB / 500 GB / 2,000 GB / Unlimited (default: Unlimited)

**Key interaction:** When beta > 0 and E > 5, the non-IID convergence curve develops a plateau or oscillation -- convergence stalls or degrades because client drift overwhelms the aggregation signal. Students observe that reducing E from 10 to 2 under high beta actually improves final accuracy despite requiring more communication rounds.

**Secondary instrument: Adaptation Strategy Selector**

A comparison panel with three columns:

| Strategy | Memory | Accuracy | Communication |
|----------|--------|----------|---------------|
| Full Fine-Tune | [computed] | [computed] | [computed] |
| LoRA (rank-16) | [computed] | [computed] | [computed] |
| Weight Freezing | [computed] | [computed] | [computed] |

Students toggle adaptation strategies and observe that LoRA reduces communication per round (fewer parameters to transmit) but may require more rounds to converge.

### The Scaling Challenge
**"Find the configuration (beta, E, C) that achieves 90% accuracy within a 500 GB communication budget."**

Students must discover that under high heterogeneity (beta >= 1.0), the only path to meeting the budget is either: (a) reducing E to 2-3 (more rounds but each is productive), or (b) using LoRA to reduce per-round communication by 99.3%, or (c) increasing client participation C to improve per-round signal quality.

### The Failure State
**Trigger:** Communication budget exceeded before target accuracy reached.
**Visual:** Communication cost line turns red; accuracy curve freezes at last value; banner appears.
**Banner:** "BUDGET EXHAUSTED -- Federated training consumed [X] GB without reaching 90% accuracy. Current accuracy: [Y]%. With beta=[Z], each round transfers [W] MB across [C] clients. Reduce per-round cost (LoRA), reduce local epochs (less client drift), or increase client participation."

### Structured Reflection
Four-option multiple choice:

> "Under non-IID data (beta=1.0), increasing local epochs from 2 to 20 causes convergence to stall. Why?"
- A) More local epochs cause the model to overfit to each client's small dataset
- **B) Client drift: each client's local model diverges toward its own optimum, and averaging divergent models produces a worse global model** (correct)
- C) The learning rate is too high for 20 epochs and gradients explode
- D) Communication overhead increases proportionally with local epochs

### Math Peek (collapsible)
$$\epsilon \leq \frac{\sigma}{\sqrt{C \cdot E \cdot R}} + \frac{\beta^2}{\epsilon}$$
where $C$ = clients per round, $E$ = local epochs, $R$ = communication rounds, $\sigma$ = gradient variance, $\beta$ = heterogeneity factor. When $\beta = 0$ (IID), convergence scales as $1/\sqrt{CER}$. When $\beta > 0$, the additive term dominates and more computation cannot overcome the heterogeneity penalty.

---

## 5. Visual Layout Specification

### Act 1: Memory Amplification Tax
- **Primary:** Grouped bar chart -- inference vs. training memory, stacked by component (weights, gradients, optimizer, activations). Log-scale Y-axis (1-10,000 MB). OOM threshold line at 300 MB (smartphone available memory).
  - X: Model scale (1M, 10M, 100M)
  - Y: Memory (MB)
  - Failure state: bars turn red when total exceeds 300 MB
- **Secondary:** Adaptation strategy comparison table -- updates dynamically with toggle.

### Act 2: Federation Paradox
- **Primary:** Dual-axis convergence plot. X: communication rounds (0-10,000). Left Y: accuracy (50-100%). Right Y: cumulative GB transferred.
  - IID curve (BlueLine), Non-IID curve (OrangeLine), communication cost (RedLine)
  - Crossover annotation when budget is hit before accuracy target
- **Secondary:** Per-round communication cost breakdown bar -- shows bytes per client per round for each adaptation strategy.
- **Failure state:** Communication budget line turns RedLine; accuracy curve freezes; OrangeLine banner.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---------|--------|-----|--------------|----------------|
| **Centralized Datacenter** | H100 (80 GB HBM3) | 80 GB | 700 W | Communication bandwidth between clients and server; convergence speed is the bottleneck |
| **Federated Edge Fleet** | Smartphone (8 GB, NPU 16 TOPS) | 300 MB available | 10 W thermal | Memory amplification makes full fine-tuning infeasible; LoRA/sparse updates are mandatory; communication budget is finite |

The two contexts demonstrate the federation paradox: centralized training on the H100 converges in hours with IID data aggregation, but requires shipping raw data to the server (privacy violation). Federated training on the edge fleet preserves privacy but faces 28x communication overhead under non-IID conditions and must fit within the 300 MB memory envelope per device.

---

## 7. Design Ledger Output

```json
{
  "chapter": 11,
  "adaptation_strategy": "lora | full_finetune | weight_freezing",
  "lora_rank": 16,
  "memory_amplification_factor": 9.0,
  "federated_beta": 1.0,
  "optimal_local_epochs": 3,
  "communication_budget_gb": 500,
  "rounds_to_convergence": 5625
}
```

The `adaptation_strategy` and `memory_amplification_factor` fields feed forward to:
- **Lab 13 (Security & Privacy):** The adaptation strategy affects the differential privacy noise budget -- LoRA with fewer trainable parameters requires less noise per round.
- **Lab 14 (Robust AI):** The memory amplification factor constrains how much robustness overhead (redundancy, error correction) can be added to on-device models.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Memory amplification 3-5x (up to 12x at scale) | @sec-edge-intelligence, Learning Objectives, line 41 | "memory amplification (3-5x), compute costs (2-3x)" |
| Training amplifies constraints by 3-10x | @tbl-training-amplification, line 332 | "training amplifies each constraint dimension by 3 to 10 times" |
| Memory footprint 3-5x increase | @tbl-training-amplification, line 336 | "3-5x increase; forces aggressive compression" |
| Compute operations 2-3x increase | @tbl-training-amplification, line 337 | "2-3x increase; limits model complexity" |
| Energy per sample 10-50x increase | @tbl-training-amplification, line 339 | "10-50x increase; requires opportunistic scheduling" |
| 10M param model: 40 MB weights spike to 200+ MB during backprop | @sec-edge-intelligence, line 420 | "40 MB of FP32 weights might spike to over 200 MB during backpropagation" |
| Smartphones allocate 200-300 MB to background apps | @sec-edge-intelligence, line 195 | "Modern smartphones typically allocate 200-300 MB to background applications" |
| LoRA rank-16 reduces to 0.7% of original parameters | @sec-edge-intelligence, line 1371 | "LoRA with rank-16 reduces this to approximately 100 MB of trainable parameters (0.7% of original)" |
| Non-IID accuracy drops 10-50% vs IID baselines | [^fn-non-iid-federated], line 301 | "accuracy drops of 10-50% compared to IID baselines when label distributions are skewed" |
| 28x increase in communication rounds for non-IID | @sec-edge-intelligence, line 2222 | "This represents a 28x increase in communication rounds compared to the IID case" |
| Optimal local epochs E=2-5 for non-IID | @sec-edge-intelligence, line 2281 | "client drift causes convergence degradation beyond an optimal point (typically E=2-5)" |
| Convergence bound: epsilon <= sigma/sqrt(CER) + beta^2/epsilon | @sec-edge-intelligence, line 2192 | "heterogeneity penalty beta degrades convergence" |
| NPU speedup 20x, energy gain 50x vs CPU | EdgeNpuSpeedup class, lines 144-148 | "speedup = 20; energy_gain = 50" |
| Smartphone RAM 8 GB | EdgeIntelligenceSetup class, line 99 | "smartphone_ram_gb" |

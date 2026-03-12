# Mission Plan: lab_09_data_selection

## 1. Chapter Alignment

- **Chapter:** Data Selection (`@sec-data-selection`)
- **Core Invariant:** The **ICR Frontier** — the Information-Compute Ratio decays as $1/(O \cdot D)$, meaning marginal learning signal from additional data drops toward zero while compute cost grows linearly. A carefully selected 10% coreset can match the accuracy of the full dataset at 10% of the compute cost.
- **Central Tension:** Students believe "more data is always better" — that doubling data doubles model quality. The chapter's scaling law data demolishes this: loss scales as $L \propto D^{-0.095}$, meaning each doubling of data yields diminishing returns. Meanwhile, compute grows 10x every 3 years while quality data grows only 2x every 5 years, creating a Data Wall where selection matters more than collection. Students must discover that data selection is not a compromise but the highest-leverage optimization in the entire D-A-M stack.
- **Target Duration:** 35–40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that reducing a dataset by 50% will reduce accuracy proportionally — roughly a 50% degradation. The chapter's ICR framework shows the opposite: because most large datasets contain massive redundancy, a well-selected 50% coreset achieves nearly identical accuracy while halving compute cost. The ICR of the coreset is 1.8x higher than random sampling because it concentrates on high-information samples the model has not yet mastered. Students predict the accuracy impact of halving the dataset, then discover the coreset achieves comparable accuracy at half the FLOPs.

**Act 2 (Design Challenge, 23 min):** Students face the Data Wall directly: given a 10,000 H100 GPU cluster that can process 10T tokens in 3 months, but only 5T tokens of quality data exist, they must design a data strategy. They explore the full ICR curve from 10% to 100% of the dataset, discover the "knee" where adding more data yields near-zero learning per FLOP, and must find the optimal coreset size that maximizes accuracy per dollar. The scaling challenge forces them to hit a target accuracy within a fixed compute budget — achievable only by selecting data, not by throwing more compute at it.

---

## 3. Act 1: The Redundancy Surprise (Calibration — 12 minutes)

### Pedagogical Goal
Students dramatically overestimate the accuracy cost of dataset reduction. The chapter's coreset comparison shows that random sampling at 50% yields 4.0% accuracy gain per epoch while a 50% coreset yields 4.5% — a higher ICR because the coreset concentrates on difficult, high-information samples. Students will predict that halving the data halves the learning, then discover the coreset actually learns *more* per FLOP than the full dataset. The aha moment: redundancy in large datasets means you are paying full compute price for near-zero information on most samples.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "You have a 1M-sample ImageNet training set. You select a 10% coreset (100K samples) using gradient-based importance scoring. Compared to training on the full dataset for the same number of epochs, what happens to final accuracy?"

Options:
- A) Accuracy drops by ~10% — you removed 90% of the data, so you lose roughly proportional information
- B) Accuracy drops by ~5% — some redundancy exists, but 90% removal is too aggressive
- **C) Accuracy drops by < 2% — most samples are redundant; the coreset captures the decision boundary** (correct)
- D) Accuracy increases — the coreset removes noisy samples that were hurting training

Common wrong answer: **A** — students assume data reduction maps linearly to accuracy reduction.

Why wrong: The $1/(O \cdot D)$ ICR decay means that in a large dataset, most samples contribute near-zero gradient signal. The 10% coreset (cited in the chapter Purpose: "a carefully selected 10% of your data match the accuracy of 100%") retains the high-information samples that define the decision boundary.

### The Instrument: ICR Curve Explorer

A **dual-axis line chart** showing accuracy vs. dataset fraction:

- **X-axis:** Dataset fraction (10% to 100%, step 5%)
- **Y-axis (left):** Model accuracy (%)
- **Y-axis (right):** ICR (learning signal per PFLOP)
- **Line 1 (BlueLine):** Accuracy vs. dataset fraction — logarithmic curve following $L \propto D^{-0.095}$
- **Line 2 (GreenLine):** ICR vs. dataset fraction — decaying as $1/(O \cdot D)$
- **Annotation:** "The Knee" marked where ICR drops below a threshold

Controls:
- **Dataset fraction slider** (10%–100%, default 100%): Shows accuracy and ICR at the selected fraction
- **Selection method toggle** (Random / Coreset): Coreset curve sits above random at all fractions because it selects high-information samples
- **Context toggle** (H100 Cloud / Jetson Orin NX Edge): Changes the compute cost axis — same accuracy curve, different dollar cost

### The Reveal
After interaction:
> "You predicted [X] accuracy drop at 10%. The actual drop is **< 2%** with coreset selection. The ICR at 10% is **9.0x higher** than at 100% — meaning each FLOP contributes 9x more learning. The chapter's scaling law ($L \propto D^{-0.095}$) explains why: doubling data from 50% to 100% yields only a $2^{0.095} \approx 7\%$ improvement in loss, but doubles compute cost."

### Reflection (Structured)
Four-option multiple choice:

> "A training run costs $100M on 10,000 H100s. Data selection reduces the dataset by 50% while maintaining accuracy. What is the compute savings?"

- A) ~$10M — data is only part of the cost
- **B) ~$50M — halving data halves total training FLOPs, which linearly reduces compute cost** (correct)
- C) ~$25M — there are fixed costs that do not scale with data
- D) ~$75M — data selection also reduces memory overhead

### Math Peek (collapsible)
$$\text{ICR}(D) = \frac{d/dD \; I(D)}{d/dD \; C(D)} \approx \frac{1}{O \cdot D}$$
$$L \propto D^{-\alpha}, \quad \alpha \approx 0.095 \text{ (Kaplan et al.)}$$

---

## 4. Act 2: The Data Wall (Design Challenge — 23 minutes)

### Pedagogical Goal
Students believe that throwing more compute at a problem always helps. The chapter's Data Wall analysis shows that a 10,000 H100 cluster can process 10T tokens in 3 months, but only 5T quality tokens exist — a 2x compute-data gap. Training on duplicate data past epoch 2–3 yields diminishing returns. Students must find the optimal operating point on the ICR curve: the coreset fraction that maximizes accuracy within a fixed compute budget. The design challenge is: hit 95% of full-dataset accuracy using the minimum compute, then compare the cost on H100 (cloud) vs. Jetson Orin NX (edge inference after training).

### The Lock (Numeric Prediction)
Before instruments unlock:

> "Your cluster can process 10T tokens but only 5T quality tokens exist. If you train for 2 epochs on all 5T tokens (10T total token-passes), what percentage of the second epoch's compute is wasted (contributing < 1% of the first epoch's learning signal)?"

Students type a percentage (0–100%). Expected wrong answers: 10–30% (students underestimate diminishing returns). Actual: **~80%** — the chapter's ICR decay formula shows that repeating data yields rapidly diminishing information, with the second pass contributing roughly 1/D of the first pass's marginal learning.

### The Instrument: Data Budget Optimizer

**Primary chart:** A **cost-accuracy Pareto frontier** (scatter + connected line):
- **X-axis:** Total training cost in GPU-hours (log scale)
- **Y-axis:** Final model accuracy (%)
- **Points:** Each point = a (coreset fraction, epoch count) configuration
- **Pareto frontier line (GreenLine):** Connects the dominant configurations
- **Student's current config (OrangeLine marker):** Moves as sliders change
- **Budget line (RedLine, vertical):** Fixed compute budget threshold

Controls:
- **Coreset fraction slider** (10%–100%, step 5%, default 100%)
- **Number of epochs slider** (1–10, default 3)
- **Quality threshold slider** (0.0–1.0, default 0.5) — controls the aggressiveness of deduplication
- **Context toggle** (H100 Cloud / Jetson Orin NX Edge): H100 shows training cost; Jetson shows inference cost of the resulting model

**Secondary chart:** **ICR decay curve** showing marginal information per FLOP across epochs:
- **X-axis:** Epoch number (1–10)
- **Y-axis:** Marginal ICR (relative to epoch 1)
- Shows the steep drop-off after epoch 2–3 on redundant data

### The Scaling Challenge
**"Find the configuration (coreset fraction + epochs) that achieves >= 95% of full-dataset accuracy using <= 50% of the full-dataset compute budget."**

Students must discover that training on a 30–40% coreset for 2–3 epochs dominates training on 100% data for 1 epoch, because the coreset has higher per-sample ICR. The key insight: selection + fewer epochs beats quantity + more epochs.

### The Failure State
**Trigger:** `total_compute > budget_limit AND accuracy < target_accuracy`

**Visual:** Cost bar turns RedLine; budget line pulses.

**Banner:** "BUDGET EXCEEDED — Training cost of [X] GPU-hours exceeds budget of [Y] GPU-hours. Accuracy is [Z]%, below the 95% target. Reduce dataset size and increase selection quality to stay within budget."

**Second failure state (Data Tax):**
**Trigger:** `epoch > 3 AND coreset_fraction > 0.8`

**Banner:** "DATA TAX — Epoch [N] on [X]% of the dataset yields < 1% of Epoch 1's learning signal. You are paying $[cost] for near-zero information. Consider reducing dataset fraction or stopping earlier."

### Structured Reflection
Four-option multiple choice:

> "The chapter states that a 2x improvement in ICR is mathematically equivalent to a 2x improvement in hardware peak throughput ($R_{\text{peak}}$). Why is data selection often the higher-leverage optimization?"

- A) Data selection is always faster to implement than hardware upgrades
- B) Hardware improvements have reached their physical limits
- **C) Improving ICR reduces the Total Operations ($O$) term directly — the work itself shrinks — while hardware only increases the rate at which existing work executes** (correct)
- D) Data selection works across all hardware platforms simultaneously

### Math Peek
$$\text{ICR}(D) = \frac{1}{O \cdot D} \qquad \text{(diminishing returns law)}$$
$$\text{Cost Savings} = \text{Budget} \times (1 - \text{coreset fraction})$$
$$\text{Accuracy}(D) \approx A_0 - k \cdot D^{-\alpha}, \quad \alpha \approx 0.095$$

---

## 5. Visual Layout Specification

### Act 1: ICR Curve Explorer
- **Primary:** Dual-axis line chart. X: dataset fraction (10%–100%). Y-left: accuracy (%). Y-right: ICR (signal/PFLOP). Two lines: random sampling (BlueLine) vs. coreset (GreenLine). "The Knee" annotation where ICR flattens.
- **Secondary:** Coreset quality toggle overlay showing the 1.8x ICR advantage of coreset over random.
- **Prediction overlay:** Student's predicted accuracy drop shown as a horizontal dashed line; actual shown as solid.

### Act 2: Data Budget Optimizer
- **Primary:** Cost-accuracy Pareto scatter. X: GPU-hours (log). Y: accuracy (%). Pareto frontier in GreenLine. Student's config as OrangeLine dot. Budget limit as vertical RedLine.
- **Secondary:** ICR decay curve. X: epoch (1–10). Y: marginal ICR (relative). Steep drop after epoch 2–3 shown with RedLine shading.
- **Failure state:** Cost bar turns RedLine when budget exceeded. "DATA TAX" OrangeLine banner when training on redundant data.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---|---|---|---|---|
| **H100 (Cloud Training)** | NVIDIA H100 SXM | 80 GB HBM3 | 700 W | Compute budget ($100M training run); 10T token processing capacity exceeds 5T quality data supply — Data Wall forces selection |
| **Jetson Orin NX (Edge Inference)** | NVIDIA Jetson Orin NX | 16 GB LPDDR5 | 25 W | Memory-constrained; smaller model from coreset-trained pipeline must fit in 16 GB; ICR savings during training translate to smaller, more efficient models at edge |

The two contexts demonstrate that data selection benefits both ends of the deployment spectrum: at cloud scale, it saves millions of dollars in compute; at edge scale, the more efficient training produces models better suited to memory-constrained inference.

---

## 7. Design Ledger Output

```json
{
  "chapter": 9,
  "coreset_fraction": 0.35,
  "optimal_epochs": 3,
  "icr_improvement_factor": 2.8,
  "compute_savings_pct": 65,
  "selection_method": "coreset"
}
```

The `coreset_fraction` and `compute_savings_pct` fields feed forward to:
- **Lab 10 (Model Compression):** The coreset fraction informs baseline model quality — a model trained on a well-selected coreset has different redundancy characteristics for pruning
- **Lab 12 (Performance Benchmarking):** The compute savings estimate provides a baseline for total pipeline cost analysis

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| 10% coreset matching full accuracy | `@sec-data-selection` Purpose section | "a carefully selected 10% of your data match the accuracy of 100%" |
| $L \propto D^{-0.095}$ scaling law | `@sec-data-selection-data-selection-fundamentals-e839`, footnote `fn-scaling-laws-origin` | "loss scales as $L \propto D^{-\alpha}$ with $\alpha \approx 0.095$, meaning each doubling of data yields diminishing returns" |
| ICR decay as $1/(O \cdot D)$ | `@sec-data-selection-icr-frontier`, @eq-icr-decay | "ICR$(D) \approx 1/(O \cdot D)$" |
| 2x Data Wall (10T capacity, 5T data) | `@sec-data-selection-defining-data-selection-ef2f`, ComputeDataGap class | "tokens_capacity = 10e12, tokens_available = 5e12, gap_ratio = 2x" |
| Compute 10x/3yr vs. data 2x/5yr | `@sec-data-selection-data-selection-fundamentals-e839`, @tbl-scaling-asymmetry | "GPU Compute: ~10x / 3 years; Training Data (Web): ~2x / 5 years" |
| 2x ICR = 2x hardware speedup | `@sec-data-selection-icr-frontier` | "A 2x improvement in ICR is mathematically equivalent to a 2x improvement in hardware Peak Throughput ($R_{\text{peak}}$)" |
| Coreset 1.8x ICR over random | `@sec-data-selection-icr-frontier`, IcrCoresetComparison class | "coresets achieve 1.8x higher ICR by focusing on difficult samples" |
| $50M savings from 50% data reduction | `@sec-data-selection-systems-perspective-bd61`, IronLawSavings class | "50% reduction in dataset size...for a USD 100M training run, this translates to USD 50M in compute savings" |
| Coreset scoring: 2.8 hours for 1M samples | `@sec-data-selection-defining-data-selection-ef2f`, SelectionEconomicsAnchor | "scoring_time_hrs = 2.8, dataset_size_m = 1" |
| Multiplicative 8x from D-A-M | `@sec-data-selection-systems-perspective-bd61`, IronLawSavings class | "2x data selection * 2x compression * 2x hardware = 8x total" |

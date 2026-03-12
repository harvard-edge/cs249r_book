# Mission Plan: lab_05_dist_train (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Distributed Training Systems (`@sec-distributed-training-systems`)
- **Core Invariant:** The **Scaling Efficiency Bound** -- scaling efficiency degrades as communication overhead grows with device count: $\text{Efficiency}(N) = 1 / (1 + N(T_{\text{comm}} - T_{\text{overlap}}) / T_{\text{compute}})$. Adding GPUs beyond the Communication Wall yields diminishing or negative returns because synchronization cost eventually dominates compute savings.
- **Central Tension:** Students believe that distributed training scales linearly -- doubling GPUs halves training time. The chapter's quantitative analysis demolishes this: scaling GPT-2 from 8 GPUs (intra-node NVLink) to 32 GPUs (inter-node 10GbE) collapses parallel efficiency from 97% to 13% because communication overhead explodes from 5.6 ms to 4,805 ms per step. The bottleneck is not compute but the network between machines. The counterintuitive lesson: 8 GPUs with gradient accumulation outperform 32 GPUs with naive data parallelism at one-seventh the cost.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students confront the Communication Wall directly. They predict the scaling efficiency when moving from 8 intra-node GPUs (NVLink at 900 GB/s) to 32 GPUs across 4 nodes on a commodity 10GbE network (1.25 GB/s). Most students will predict efficiency stays high because "we just added more GPUs." The instrument reveals that parallel efficiency collapses from 97% to 13% -- communication consumes 73% of the step time. This calibrates the student's intuition: the interconnect, not the GPU count, determines scaling behavior.

**Act 2 (Design Challenge, 22 min):** Students must configure a distributed training system to minimize wall-clock training time for GPT-2 under a fixed dollar budget. They control GPU count, interconnect type (NVLink vs. InfiniBand vs. 10GbE), and gradient accumulation steps. The scaling challenge forces them to discover that 8 GPUs with gradient accumulation achieves the same effective batch size as 32 GPUs at 87% lower cost, and to find the exact GPU count where adding more hardware yields negative returns. A pipeline bubble calculator reveals the second tax: when model parallelism becomes necessary, bubble overhead wastes 18--48% of compute depending on the microbatch count.

---

## 3. Act 1: The Communication Wall (Calibration -- 12 minutes)

### Pedagogical Goal

Students dramatically overestimate the benefit of scaling to more GPUs. The chapter demonstrates that intra-node scaling on NVLink achieves 97% efficiency because gradient synchronization takes only 5.6 ms against 1,800 ms of compute. But crossing the node boundary on a commodity network (10GbE, 1.25 GB/s) inflates communication to 4,805 ms -- exceeding the compute time itself. Students who expect "4 nodes = 4x faster" discover that 32 GPUs deliver only 4.2x speedup (13% efficiency) instead of the expected 32x. The physics is the Communication-Computation Ratio: when $T_{\text{comm}} > T_{\text{compute}}$, the system is communication-bound and adding GPUs makes it worse.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "You are training GPT-2 (1.5B parameters) on 8 H100 GPUs within a single node using NVLink (900 GB/s). Parallel efficiency is 97%. You scale to 32 GPUs across 4 nodes connected by 10 Gigabit Ethernet (1.25 GB/s). What happens to parallel efficiency?"

Options:

- A) Stays above 90% -- more GPUs always help
- B) Drops to ~60% -- some overhead from the network, but still a net win
- **C) Drops to ~13% -- communication overhead dominates the entire step** (correct)
- D) Drops to ~40% -- the network is slower but parallelism compensates

**Why each distractor is plausible:**

- **A** reflects the naive "more GPUs = more speed" assumption, reinforced by marketing materials
- **B** is the "reasonable pessimist" answer -- students who know networks are slower but underestimate by how much
- **D** is the "split-the-difference" estimate that sounds moderate and safe
- **C is correct** because the gradient sync volume (5.6 GB) over 1.25 GB/s takes ~4,480 ms, exceeding the 1,800 ms compute time, making communication 73% of the total step

### The Instrument: Scaling Efficiency Explorer

A **dual-panel visualization** with a shared GPU-count x-axis:

**Panel 1: Speedup vs. GPU Count (log-log)**

- **X-axis:** Number of GPUs (1, 2, 4, 8, 16, 32, 64, 128), log scale
- **Y-axis:** Effective Speedup, log scale
- **Series:**
  - Ideal linear scaling (dashed black line)
  - Actual scaling with selected interconnect (solid colored line)
- **Annotation:** "Communication Wall" arrow at the point where efficiency drops below 50%

**Panel 2: Time Breakdown (stacked bar)**

- **X-axis:** Number of GPUs (matching Panel 1 selection)
- **Y-axis:** Time per step (ms)
- **Stacked segments:** Compute (BlueLine #006395), Communication (RedLine #CB202D), Overlap saved (GreenLine #008F45, negative)
- **Horizontal reference line:** Single-GPU baseline step time (1,800 ms)

Controls:

- **Interconnect selector** (toggle): NVLink (900 GB/s) / InfiniBand HDR (25 GB/s) / 10GbE (1.25 GB/s)
  - Default: NVLink
- **Model size selector** (dropdown): GPT-2 (1.5B) / BERT-Large (340M) / ResNet-50 (25M)
  - Default: GPT-2
- **Comm-Compute overlap slider**: 0--80% (how much communication is hidden behind compute)
  - Default: 0% (no overlap, worst case)

On each control change, both panels update simultaneously. Switching from NVLink to 10GbE with GPT-2 selected produces the dramatic efficiency collapse.

### The Reveal

After the student interacts with at least 2 interconnect settings:

> "You predicted [X] efficiency. The actual parallel efficiency for 32 GPUs on 10GbE is **13%**. You were off by [Y] percentage points."
>
> "At 32 GPUs on 10GbE, communication takes **4,805 ms** while compute takes **1,800 ms**. The GPUs spend 73% of each step waiting for gradients to arrive. The 720x bandwidth gap between NVLink (900 GB/s) and 10GbE (1.25 GB/s) is the root cause."

### Reflection (Structured)

Four-option multiple choice:

> "The chapter shows that 8 GPUs on NVLink achieve 97% efficiency while 32 GPUs on 10GbE achieve only 13% efficiency. What is the primary physical cause?"

- A) The GPUs on the commodity network are slower processors
- B) The gradient tensors are larger when using more GPUs
- **C) The inter-node bandwidth (1.25 GB/s) is 720x slower than NVLink (900 GB/s), making gradient synchronization take longer than the computation itself** (correct)
- D) Amdahl's Law limits speedup regardless of the network

### Math Peek (collapsible)

$$T_{\text{step}}(N) = \frac{T_{\text{compute}}}{N} + T_{\text{comm}}(N) - T_{\text{overlap}}$$

$$T_{\text{comm}} = \frac{2(N-1)}{N} \times \frac{M_{\text{bytes}}}{BW_{\text{net}}}$$

$$\text{Efficiency}(N) = \frac{T_{\text{compute}}}{N \times T_{\text{step}}(N)}$$

> For GPT-2 (1.5B params, 5.6 GB gradient payload) on 10GbE (1.25 GB/s): $T_{\text{comm}} \approx 5.6 / 1.25 \times 1000 \approx 4{,}480$ ms. With $T_{\text{compute}} = 1{,}800$ ms, communication exceeds compute by 2.5x. The chapter's Scaling32GPU LEGO cell validates this: efficiency is 13% because communication consumes 73% of the total step time.

---

## 4. Act 2: The Budget-Optimal Training Configuration (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students believe the fastest training configuration uses the most GPUs. The chapter's cost analysis reveals a counterintuitive result: 8 GPUs with gradient accumulation (4 steps) achieves the same effective batch size of 512 as 32 GPUs with naive data parallelism, but costs $422 instead of $3,021 -- an 87% savings. The chapter also introduces the pipeline bubble tax: when models require pipeline parallelism, bubble overhead wastes $(P-1)/(P-1+M)$ of compute, which is 18% for 8 stages and 32 microbatches but 48% for 16 stages and 16 microbatches. Students must navigate a three-way trade-off: GPU count (throughput vs. communication overhead), gradient accumulation (batch size vs. sync frequency), and pipeline stages (memory capacity vs. bubble waste).

### The Lock (Numeric Prediction)

Before instruments unlock:

> "You are training GPT-2 on a budget of $500. GPUs cost $16/GPU-hour. An 8-GPU node with NVLink achieves 97% parallel efficiency. A 32-GPU cluster on 10GbE achieves 13% efficiency. Single-GPU training takes 25 hours. What is the minimum wall-clock training time achievable within the $500 budget?"

Students enter a number in hours. Expected range: 1--25 hours.

**Common wrong answers:** Students will enter 0.8--1.5 hours (assuming 32 GPUs at near-linear scaling). The actual answer is approximately **3.3 hours** using the 8-GPU configuration with gradient accumulation, which costs approximately $422. The 32-GPU configuration costs $3,021 (exceeding the budget) because GPUs spend 73% of their time idle waiting for gradients.

### The Instrument: Distributed Training Cost Optimizer

**Primary Chart: Training Time vs. Cost (scatter)**

- **X-axis:** Total training cost (USD), range $0--$5,000
- **Y-axis:** Wall-clock training time (hours), range 0--30
- **Data points:** One point per configuration, colored by efficiency regime:
  - GreenLine (#008F45): efficiency > 80%
  - OrangeLine (#CC5500): efficiency 40--80%
  - RedLine (#CB202D): efficiency < 40%
- **Budget line:** Vertical dashed line at the student's budget
- **Pareto frontier:** Dashed curve connecting optimal configurations

**Secondary Chart: Step Time Breakdown (waterfall)**

- **X-axis:** Component (Compute, Intra-node Comm, Inter-node Comm, Overlap Savings, Total)
- **Y-axis:** Time (ms)
- **Updates dynamically** as sliders change

**Tertiary Chart: Pipeline Bubble Gauge**

- Circular gauge showing bubble fraction as a percentage
- Green zone: 0--10%, Orange zone: 10--25%, Red zone: 25--50%
- Updates when pipeline stages or microbatch count change

Controls:

- **GPU count slider**: 1, 2, 4, 8, 16, 32, 64
  - Default: 8
  - Step: powers of 2
- **Interconnect selector** (toggle): NVLink-only (max 8 GPUs) / InfiniBand HDR (25 GB/s) / 10GbE (1.25 GB/s)
  - Default: NVLink-only
  - Constraint: selecting NVLink caps GPU count at 8; selecting >8 GPUs forces InfiniBand or 10GbE
- **Gradient accumulation steps**: 1, 2, 4, 8, 16
  - Default: 1
- **Pipeline stages** (for model parallelism exploration): 1, 2, 4, 8, 16
  - Default: 1 (no pipeline parallelism)
- **Microbatches per stage**: 4, 8, 16, 32, 64
  - Default: 32
- **Budget slider**: $100--$5,000
  - Default: $500

### The Scaling Challenge

**"Find the configuration that minimizes training time within a $500 budget and achieves an effective batch size of at least 512."**

Students must discover:

1. 32 GPUs on 10GbE exceeds the budget ($3,021) and is infeasible
2. 8 GPUs with gradient accumulation (4 steps) achieves batch size 512 at $422 in ~3.3 hours
3. Adding pipeline stages (when enabled) introduces bubble overhead that further increases cost
4. The sweet spot is the highest GPU count where NVLink is available (8), combined with gradient accumulation to achieve the target batch size without crossing node boundaries

### The Failure State

**Trigger condition:** `total_cost > budget`

**Visual change:** The cost axis point turns RedLine; the budget line pulses; all chart elements beyond the budget dim to 30% opacity.

**Banner text:**

> "**Budget Exceeded** -- Configuration costs $[X] but budget is $[Y]. Reduce GPU count or switch to a cheaper interconnect. The chapter shows that 8 GPUs + gradient accumulation saves 87% vs. naive 32-GPU scaling."

**Second failure state -- Pipeline Bubble Overrun:**

**Trigger condition:** `bubble_fraction > 0.40`

**Visual change:** Pipeline Bubble Gauge enters red zone; bubble segments in the waterfall chart turn RedLine.

**Banner text:**

> "**Pipeline Bubble > 40%** -- With [P] stages and [M] microbatches, $(P-1)/(P-1+M) = [X]$% of GPU cycles are wasted idle. Increase microbatches to $M \gg P$ or reduce pipeline stages."

Both failure states are reversible: adjusting sliders immediately recalculates and clears the failure banners.

### Structured Reflection

Four-option multiple choice:

> "The chapter demonstrates that 8 GPUs with gradient accumulation costs $422 while 32 GPUs costs $3,021 for the same effective batch size. Why is the smaller configuration cheaper despite using fewer GPUs?"

- A) The 8-GPU configuration uses less electricity per GPU
- B) Gradient accumulation compresses the gradients, reducing network cost
- **C) The 8-GPU configuration keeps all communication on NVLink (900 GB/s), reducing overhead to 0.07% vs. 73% on 10GbE, so GPU-hours are spent computing rather than waiting** (correct)
- D) The 32-GPU configuration requires more training epochs to converge

### Math Peek (collapsible)

$$\text{Effective Batch Size} = N_{\text{GPUs}} \times B_{\text{local}} \times K_{\text{accum}}$$

$$\text{Comm Overhead (GA)} = \frac{T_{\text{comm}}}{K_{\text{accum}} \times T_{\text{compute}}}$$

$$\text{Bubble Fraction} = \frac{P - 1}{P - 1 + M}$$

$$\text{Cost} = N_{\text{GPUs}} \times T_{\text{wall}} \times \text{Rate}_{\text{USD/GPU-hr}}$$

> The gradient accumulation insight: synchronizing once every $K$ steps reduces communication overhead by factor $K$. With $K=4$ and NVLink overhead of 5.6 ms against 1,800 ms compute: overhead $= 5.6 / (4 \times 1800) = 0.08\%$. The GPUs spend 99.92% of their time computing.

---

## 5. Visual Layout Specification

### Act 1

1. **Scaling Efficiency Plot** (primary, log-log line chart)
   - X: Number of GPUs (1--128, log2 scale)
   - Y: Effective Speedup (1--128, log2 scale)
   - Series: Ideal (dashed black), Actual (solid, color by interconnect)
   - Failure annotation: "Communication Wall" arrow where efficiency < 50%

2. **Step Time Breakdown** (stacked bar chart)
   - X: GPU count (categorical: 1, 2, 4, 8, 16, 32, 64, 128)
   - Y: Time per step (ms), range 0--7,000
   - Segments: Compute (BlueLine #006395), Communication (RedLine #CB202D), Overlap (GreenLine #008F45, negative offset)
   - Reference line: 1,800 ms (single GPU baseline)
   - Failure state: Communication segment turns bright red when it exceeds compute segment

### Act 2

1. **Training Time vs. Cost** (scatter plot, primary)
   - X: Total cost (USD), range $0--$5,000
   - Y: Wall-clock time (hours), range 0--30
   - Points colored by efficiency regime (Green > 80%, Orange 40--80%, Red < 40%)
   - Budget constraint: vertical dashed line
   - Pareto frontier: dashed curve

2. **Step Time Waterfall** (waterfall chart)
   - X: Components (Compute, Intra-comm, Inter-comm, Overlap, Pipeline Bubble, Total)
   - Y: Time (ms)
   - Updates on slider change

3. **Pipeline Bubble Gauge** (circular gauge)
   - Range: 0--50%
   - Zones: Green 0--10%, Orange 10--25%, Red 25--50%
   - Label: "Bubble: [X]%"
   - Enters failure state when > 40%

---

## 6. Deployment Context Definitions

| Context | Device | Interconnect | Key Constraint |
|:--------|:-------|:-------------|:---------------|
| **Single-Node (NVLink)** | 8x H100 (80 GB each) with NVLink (900 GB/s) | Intra-node only | Communication overhead < 1%; memory is the scaling ceiling (model must fit in 8x 80 GB) |
| **Multi-Node (Commodity)** | 32x H100 across 4 nodes on 10GbE (1.25 GB/s) | Inter-node 10GbE | Communication overhead 73%; bandwidth is the wall; naive scaling wastes 87% of budget |

The two contexts demonstrate that identical GPUs produce radically different scaling behavior depending on the interconnect. The same 5.6 GB gradient payload takes 6 ms on NVLink but 4,480 ms on 10GbE -- a 720x difference that transforms a compute-bound job into a communication-bound one. This is the core lesson: distributed training performance is governed by the interconnect, not by the accelerator count.

---

## 7. Design Ledger Output

```json
{
  "chapter": "v2_05",
  "ch05_v2": {
    "gpu_count_chosen": 8,
    "interconnect": "nvlink",
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 512,
    "scaling_efficiency_pct": 97,
    "training_cost_usd": 422,
    "pipeline_stages": 1,
    "bubble_fraction_pct": 0,
    "comm_compute_ratio": 0.003
  }
}
```

**Downstream consumers:**

- **Lab V2-06 (Collective Communication):** Reads `interconnect` and `gpu_count_chosen` to initialize the AllReduce algorithm comparison with the student's chosen topology
- **Lab V2-07 (Fault Tolerance):** Reads `gpu_count_chosen` and `pipeline_stages` to calculate MTBF for the student's cluster configuration (failure probability scales with node count)

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|:---|:---|:---|
| NVLink bandwidth: 900 GB/s | `@sec-distributed-training-systems-systems-multimachine-scaling-fundamentals-ff96` (DistTrainConstants, line 105) | "nvlink_h100_gbs = NVLINK_H100_BW.m_as(GB / second)" yielding 900 GB/s |
| 8-GPU intra-node efficiency: 97% | `@sec-distributed-training-systems-systems-data-parallelism-tradeoffs` (Scaling8GPU LEGO, line 791) | "check(efficiency_8gpu_val > 95, ...)" and "NVLink enables efficient scaling within single nodes (97% efficiency)" (line 974) |
| 32-GPU inter-node efficiency: 13% | `@sec-distributed-training-systems-systems-data-parallelism-tradeoffs` (Scaling32GPU LEGO, line 877) | "check(comm_pct_val > 50, ...)" and "Inter-node communication kills efficiency (drops to 13%)" (line 975) |
| Communication overhead 10--40% for AllReduce | `@sec-distributed-training-systems-systems-distributed-training-efficiency-metrics-9488` (line 1069) | "AllReduce operations consume 10--40% of total training time in data parallel systems" |
| GPT-2 gradient sync size: ~5.6 GB | `@sec-distributed-training-systems-systems-data-parallelism-tradeoffs` (Scaling8GPU LEGO, line 780) | "sync_size_gb = (params_b * BILLION * 4) / BILLION" yielding ~5.6 GB for GPT-2 |
| Gradient accumulation saves 87% cost | `@sec-distributed-training-systems-systems-data-parallelism-tradeoffs` (GradAccumScenario LEGO, line 916) | "Savings: USD [ga_savings_str] ([ga_savings_pct_str]% reduction)" |
| Gradient accumulation comm overhead: ~0.07% | `@sec-distributed-training-systems-systems-data-parallelism-tradeoffs` (GradAccumScenario LEGO, line 958) | "Overhead = 5 ms / (4 x 1800 ms) ~ 0.07%" |
| Pipeline bubble fraction: $(P-1)/(P-1+M)$ | `@sec-distributed-training-systems-systems-pipeline-parallelism-8748` (PipelineBubble LEGO, line 1741) | "bubble_fraction = (p_stages - 1) / (p_stages - 1 + m_microbatches)" |
| Pipeline bubble 18% at P=8, M=32 | `@sec-distributed-training-systems-systems-pipeline-parallelism-8748` (PipelineBubble LEGO, line 1744) | "check(17 < bubble_pct < 19, ...)" yielding 17.9% |
| Pipeline bubble 48% at P=16, M=16 | `@sec-distributed-training-systems-systems-pipeline-parallelism-8748` (definition callout, line 1714) | "with p=16 stages and m=16 micro-batches, 48% of compute is wasted idle" |
| Scaling efficiency formula: Speedup = N/(1+(N-1)*r) | `@sec-distributed-training-systems-systems-physics-scaling-amdahls-law-communication-4d7f` (@fig-scaling-tax code, line 1124) | "speedup = N / (1 + (N - 1) * sc['r'])" with r=0.05 (compute-bound) and r=0.50 (bandwidth-bound) |
| 85--95% efficiency in linear scaling regime (2--32 GPUs) | `@sec-distributed-training-systems-systems-distributed-training-efficiency-metrics-9488` (line 1187) | "In the linear scaling regime of 2-32 GPUs, systems typically achieve 85--95% parallel efficiency" |
| Amdahl's ceiling at 30% serial fraction: ~3x max speedup | `@sec-distributed-training-systems-systems-distributed-training-fundamentals-97da` (definition callout, line 214) | "with 30% of a step's time spent on synchronization, the theoretical scaling ceiling is 1/0.30 ~ 3x" |
| Effective batch size = N x B x K_accum | `@sec-distributed-training-systems-systems-data-parallelism-tradeoffs` (GradAccumScenario, line 919) | "8 GPUs x batch 16 x 4 accumulation steps = 512" |
| Single GPU baseline: 25 hours, 1.8s/step | `@sec-distributed-training-systems-systems-data-parallelism-tradeoffs` (Scaling8GPU LEGO, lines 770--771) | "single_gpu_step_s = 1.8" and "training_hours_1gpu = 25" |
| Distributed step time equation | `@sec-distributed-training-systems-systems-physics-scaling-amdahls-law-communication-4d7f` (line 1163) | "$T_{\text{step}}(N) = T_{\text{compute}}/N + T_{\text{comm}}(N) - T_{\text{overlap}}$" |

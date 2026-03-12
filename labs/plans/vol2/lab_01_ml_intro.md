# Mission Plan: lab_01_ml_intro (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Introduction (`@sec-vol2-introduction`)
- **Core Invariant:** The **Iron Law of Scale** ($T_{\text{step}}(N) = T_{\text{compute}}/N + T_{\text{comm}}(N) - T_{\text{overlap}}$) — scaling efficiency collapses as communication overhead grows with node count; doubling GPUs never halves training time because the network synchronization tax is irreducible.
- **Central Tension:** Students believe that scaling from 8 GPUs to 64 GPUs will yield an 8x speedup. The chapter's quantitative analysis shows that for a 175B-parameter model, synchronization over 100G Ethernet collapses scaling efficiency to less than 30%, while even InfiniBand at 200 Gbps achieves only ~90%. The communication wall, not compute, is the binding constraint at fleet scale. Amdahl's Law limits maximum speedup to 5x when 20% of time is spent on network synchronization.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students confront the Reliability Gap. They believe a cluster of individually reliable GPUs (99.9% uptime each) will produce a reliable fleet. The Fleet Law ($P_{\text{fleet}} = (1 - p_{\text{fail}})^N$) shows that a 25,000-GPU cluster experiences a hardware failure every 4.4 hours. This act uses a node-count slider to reveal the exponential collapse of fleet availability, correcting the intuition that "reliable parts make reliable systems."

**Act 2 (Design Challenge, 23 min):** Students must configure a distributed training run for a 175B-parameter model across two interconnect regimes: 8 H100s on NVLink (intra-node) and 64 H100s on Ethernet (inter-node). They predict the scaling speedup, then discover that the Ethernet cluster achieves less than 30% efficiency due to the 700 GB gradient synchronization bottleneck. The design challenge requires finding the crossover point where adding GPUs actually decreases throughput.

---

## 3. Act 1: The Reliability Collapse (Calibration -- 12 minutes)

### Pedagogical Goal

Students dramatically overestimate fleet reliability. The chapter states that GPT-4 training on 25,000 A100s experiences a failure every 4.4 hours (MTBF), yet infrastructure dashboards show each individual GPU at 99.9% uptime. This act forces students to predict fleet MTBF before showing them the exponential collapse curve. The gap between "my GPU is reliable" and "my fleet is always broken" is the foundational insight of Volume 2.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "A cluster of 1,000 GPUs is used for a training run. Each GPU has 99.9% annual uptime (Three Nines). What is the probability that ALL 1,000 GPUs are operational simultaneously?"

Options:
- A) ~99.0% -- nearly all up, rare failures ← common wrong answer
- B) ~90.0% -- one in ten chance of a failure
- C) ~50.0% -- coin flip whether the cluster is fully healthy
- **D) ~37% -- more likely than not that at least one GPU is down** ← correct

The answer is $(0.999)^{1000} \approx 0.368$, or about 37%. Students almost universally pick A or B because they conflate per-node reliability with fleet reliability. The exponential decay is the core insight.

### The Instrument: Fleet Availability Curve

A **line chart** showing fleet availability as a function of node count:

- **X-axis:** Number of GPUs (1 -- 25,000, logarithmic scale)
- **Y-axis:** Probability all GPUs healthy simultaneously (0% -- 100%)
- **Line:** $(1 - p_{\text{fail}})^N$ where $p_{\text{fail}}$ is per-GPU failure probability

Controls:
- **Node Count slider** (1 -- 25,000, step: 100, default: 1,000): Adjusts the fleet size; the availability line updates in real time.
- **Per-GPU Reliability selector** (99%, 99.9%, 99.99%): Three Nines, Four Nines, Five Nines. Students observe that even Five Nines (99.99%) collapses to 8% at 25,000 GPUs.

Threshold annotations:
- At N=1,000, availability ~37% (Three Nines)
- At N=10,000, availability ~0.005% (Three Nines)
- At N=25,000, MTBF ~4.4 hours annotation appears

### The Reveal

After interaction:

> "You predicted [X]% fleet availability at 1,000 GPUs. The actual value is **37%** — the fleet is more likely to have at least one failed GPU than to be fully healthy. At 25,000 GPUs (GPT-4 scale), a failure occurs every **4.4 hours**. Your monitoring dashboard would show 100% uptime for each individual GPU while the fleet fails 5 times per day."

### Reflection (Structured)

Four-option multiple choice:

> "GPT-4 was trained on 25,000 A100s for 100 days. Meta's Llama 3 experienced 419 interruptions during 54 days on 16,384 H100s. What is the correct engineering response to this failure rate?"

- A) Use more reliable hardware to reduce per-GPU failure rate below 0.01%
- B) Reduce cluster size to fewer than 1,000 GPUs to keep availability above 90%
- **C) Accept failure as routine and engineer for fast recovery: automated checkpointing, redundant workers, and self-healing orchestration** ← correct
- D) Switch to asynchronous training to eliminate the impact of individual GPU failures

### Math Peek (collapsible)

$$P_{\text{all healthy}} = (1 - p_{\text{fail}})^N$$
$$\text{MTBF}_{\text{cluster}} = \frac{\text{MTTF}_{\text{gpu}}}{N}$$

For 25,000 GPUs at 8% annual failure rate:
$$\text{Failures/day} = 25{,}000 \times 0.08 / 365 \approx 5.5$$
$$\text{MTBF} = 24 / 5.5 \approx 4.4 \text{ hours}$$

---

## 4. Act 2: The Communication Wall (Design Challenge -- 23 minutes)

### Pedagogical Goal

Students believe that adding more GPUs proportionally reduces training time. The chapter's Iron Law of Scale ($T_{\text{step}}(N) = T_{\text{compute}}/N + T_{\text{comm}}(N) - T_{\text{overlap}}$) and the Amdahl's Distributed Pitfall reveal that communication overhead grows with N while compute per device shrinks. For a 175B-parameter model, synchronizing 700 GB of FP16 gradients via Ring AllReduce on 100G Ethernet takes longer than the computation itself, collapsing efficiency below 30%. Students must find the "Scaling Inversion Point" where adding more GPUs actually decreases total throughput.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "You are training GPT-3 (175B parameters) on 64 H100 GPUs connected by 100G Ethernet. Each GPU computes for 1.2 seconds per step. The Ring AllReduce must synchronize ~700 GB of FP16 gradients. What scaling efficiency (%) do you expect compared to a single GPU?"

Students type a percentage (0--100%). Expected wrong answers: 70--90% (students expect reasonable overhead). Actual: approximately **28%** on Ethernet vs **~90%** on InfiniBand. The chapter states: "Ethernet (100 Gbps) collapses Scaling Efficiency to less than 30%."

### The Instrument: Scaling Efficiency Comparator

A **dual-line speedup chart** showing actual vs. ideal throughput:

- **X-axis:** Number of GPUs (1 -- 128)
- **Y-axis:** Throughput (samples/second, left) and Scaling Efficiency % (right)
- **Ideal line** (dashed, GreenLine): Linear speedup $\text{Throughput} = N \times \text{Throughput}_1$
- **Actual line** (solid, BlueLine): $\text{Throughput} = N / (1 + N \times T_{\text{comm}} / T_{\text{compute}})$

Controls:
- **Number of GPUs slider** (1 -- 128, step: 1, default: 8): Adjusts fleet size
- **Interconnect toggle**: NVLink 900 GB/s (intra-node) vs 100G Ethernet 12.5 GB/s (inter-node)
- **Model size selector**: 1B / 7B / 70B / 175B parameters (changes gradient volume)
- **Overlap toggle**: Communication-computation overlap on/off (hides up to 50% of comm time)

**Secondary instrument: C3 Breakdown Bar**
A stacked horizontal bar showing the decomposition of one training step:

- $T_{\text{compute}}$ (BlueLine): Useful math
- $T_{\text{communication}}$ (OrangeLine): Gradient sync
- $T_{\text{coordination}}$ (RedLine): Barrier + scheduling overhead

The bar updates as sliders move. When communication exceeds 50%, the bar enters a warning state with an annotation: "Communication dominates — fleet is spending more time on the wire than on the math."

### The Scaling Challenge

**"Find the maximum GPU count where Ethernet scaling efficiency stays above 50% for a 175B model."**

Students must adjust the model size and GPU count sliders to discover that:
- At 175B params on Ethernet: efficiency drops below 50% at approximately 8--16 GPUs
- At 175B params on NVLink (intra-node, max 8): efficiency stays above 90%
- Switching to InfiniBand pushes the crossover to ~64 GPUs

The catch: enabling communication-computation overlap extends the useful range, but cannot save Ethernet at scale because the gradient volume (700 GB) is physically too large for the 12.5 GB/s pipe.

### The Failure State

**Trigger condition:** `scaling_efficiency < 0.5` (communication time exceeds compute time)

**Visual change:** The C3 Breakdown bar turns the communication segment RedLine. The speedup line on the chart flattens and begins to curve downward.

**Banner text:**
> "**Scaling Inversion -- Adding GPUs Decreases Efficiency.** At [N] GPUs on [interconnect], communication consumes [X]% of step time. Each additional GPU adds more synchronization overhead than it removes compute. The fleet is network-bound: upgrade the interconnect or reduce the gradient volume."

### Structured Reflection

Four-option multiple choice:

> "The chapter states: 'If a model spends 20% of its time waiting for the network, no amount of faster GPUs can make it more than 5x faster.' This is an application of:"

- A) Moore's Law -- hardware improvements eventually overcome software bottlenecks
- B) The Chinchilla Scaling Law -- optimal allocation requires balancing model and data size
- **C) Amdahl's Law -- the serial fraction (communication) caps the maximum speedup from parallelism** ← correct
- D) The Roofline Model -- arithmetic intensity determines whether computation or memory is the bottleneck

### Math Peek (collapsible)

$$T_{\text{step}}(N) = \frac{T_{\text{compute}}}{N} + T_{\text{comm}}(N) - T_{\text{overlap}}$$

$$\eta_{\text{scale}} = \frac{T_{\text{compute}}}{N \times T_{\text{step}}}$$

Ring AllReduce volume: $2 \times \frac{N-1}{N} \times P \times 2$ bytes (FP16)

For GPT-3 (175B params, FP16): $\approx 700$ GB

$$T_{\text{comm}} = \frac{700 \text{ GB}}{12.5 \text{ GB/s}} = 56 \text{ s (Ethernet)} \quad \text{vs} \quad \frac{700 \text{ GB}}{900 \text{ GB/s}} = 0.78 \text{ s (NVLink)}$$

---

## 5. Visual Layout Specification

### Act 1: Fleet Availability Curve
- **Chart type:** Line chart (semi-log)
- **X-axis:** Number of GPUs (1 -- 25,000, log scale)
- **Y-axis:** Fleet availability probability (0% -- 100%, linear)
- **Data series:** One line per reliability level (99%, 99.9%, 99.99%); active line highlighted
- **Threshold annotations:** MTBF callout at 25,000 GPUs ("Failure every 4.4 hours")
- **Failure state:** N/A (Act 1 is calibration only)

### Act 2: Scaling Efficiency Comparator
- **Chart type:** Dual-axis line chart
- **X-axis:** Number of GPUs (1 -- 128, linear)
- **Y-axis (left):** Throughput (samples/sec)
- **Y-axis (right):** Scaling Efficiency (0% -- 100%)
- **Data series:** Ideal (dashed green), NVLink (solid blue), Ethernet (solid orange)
- **Failure state:** When efficiency < 50%, the Ethernet line turns RedLine; banner appears

### Act 2: C3 Breakdown Bar
- **Chart type:** Horizontal stacked bar
- **Segments:** Compute (BlueLine), Communication (OrangeLine), Coordination (RedLine)
- **Updates on:** GPU count, interconnect, model size changes
- **Failure state:** Communication segment turns RedLine when > 50% of total

---

## 6. Deployment Context Definitions

| Context | Device | Interconnect | Bandwidth | Key Constraint |
|---|---|---|---|---|
| **8x H100 NVLink (intra-node)** | DGX H100 node | NVLink 4.0 | 900 GB/s | Scaling ceiling at 8 GPUs; efficiency >90% but limited parallelism |
| **64x H100 Ethernet (inter-node)** | 8 DGX nodes | 100G Ethernet | 12.5 GB/s | Communication wall collapses efficiency to <30% for large models; bandwidth is 72x slower than NVLink |

The two contexts demonstrate the bandwidth cliff between intra-node and inter-node communication. The same model and the same training code produce dramatically different scaling efficiency depending solely on the physical interconnect. This is the Scale Moment made visceral: the wire, not the chip, determines fleet performance.

---

## 7. Design Ledger Output

```json
{
  "chapter": "v2_01",
  "interconnect_chosen": "nvlink | ethernet",
  "gpu_count": 64,
  "scaling_efficiency_pct": 28,
  "communication_fraction_pct": 72,
  "model_size_b": 175,
  "reliability_mtbf_hours": 4.4
}
```

The `interconnect_chosen` and `scaling_efficiency_pct` fields feed forward to:
- **Lab V2-02 (Compute Infrastructure):** The bandwidth hierarchy visualization uses the student's chosen interconnect as the starting constraint.
- **Lab V2-05 (Distributed Training):** The communication fraction informs the parallelism strategy selection.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Fleet availability $(0.999)^{1000} \approx 0.368$ | @sec-vol2-introduction-scale-moment, line 152 | "If each individual GPU has an annual failure rate of 8%..." |
| MTBF ~4.4 hours at 25,000 GPUs | @sec-vol2-introduction-scale-moment, lines 157--158 | "MTBF ≈ 4.4 hours" |
| 5.5 failures per day at 25,000 GPUs | @sec-vol2-introduction-scale-moment, line 157 | "~5.5 failures per day" |
| 700 GB gradient sync for 175B params (FP16) | @sec-vol2-introduction-fleet-law, line 1615 | "Ring All-Reduce must move approximately 700 GB of data" |
| InfiniBand efficiency ~90% | @sec-vol2-introduction-fleet-law, line 1619 | "Scaling Efficiency of ~90%" |
| Ethernet efficiency <30% | @sec-vol2-introduction-fleet-law, line 1620 | "Scaling Efficiency to <30%" |
| Amdahl's 5x cap at 20% communication | @sec-vol2-introduction-fleet-law, line 1539 | "no amount of faster GPUs can make it more than 5x faster" |
| Communication up to 40% of iteration time | @sec-vol2-introduction-breed-apart, line 872 | "communication can consume up to 40% of the total iteration time" |
| 419 interruptions during Llama 3 training | @sec-vol2-introduction-scale-moment, fn-failure-rates-fleet | "419 unexpected interruptions during Llama 3's 54-day training" |
| NVLink 900 GB/s, InfiniBand NDR 400 Gbps | @sec-vol2-introduction-scale-moment, fn-infiniband-rdma-v2 | "HDR IB delivers 25 GB/s; NDR reaches 50 GB/s" / NVLink "900 GB/s" |

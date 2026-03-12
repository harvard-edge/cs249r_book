# Mission Plan: lab_13_model_serving

## 1. Chapter Alignment

- **Chapter:** Model Serving (`@sec-model-serving`)
- **Core Invariant:** The **Tail Latency Explosion**: P99 latency diverges nonlinearly from mean latency as system utilization approaches saturation. Using the M/M/1 approximation, P99 is approximately 4.6x the mean latency. At 70% utilization (the "knee"), tail latency begins exponential growth. Production systems must run at 40--60% utilization to absorb traffic spikes, not the 90--100% that maximizes throughput.
- **Central Tension:** Students believe that maximizing GPU utilization is always the correct serving strategy because it was the correct training strategy. The chapter's serving inversion demolishes this: training maximizes throughput (utilization -> 100%), but serving minimizes tail latency (utilization -> 40--60%). A system at 90% utilization looks efficient on a dashboard but delivers P99 latency that is 46x the service time, causing timeouts and system collapse under any traffic spike. More utilization is not better; headroom is the product.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that increasing utilization linearly increases latency. The M/M/1 queueing model shows that latency grows as 1/(1-rho), which is approximately linear below 50% but explodes exponentially past 70%. Students predict what happens to P99 at 85% utilization, discover it is 30x the service time (not 6x as they expect from linear extrapolation), and learn why production serving systems deliberately run at 40--60% utilization. The Black Friday traffic spike example grounds this: a 10x traffic spike does not cause 10x latency; it causes system collapse.

**Act 2 (Design Challenge, 23 min):** Students must design a serving configuration that meets two simultaneous SLAs: a throughput target (QPS) and a P99 latency budget. They manipulate batch size, utilization target, and number of replicas to find the throughput-latency knee. The instrument exposes the batching paradox: increasing batch size improves throughput but degrades individual request latency. Students must find the "sweet spot" batch size for each deployment context (H100 optimizing throughput vs. Jetson Orin NX optimizing latency) and discover that the optimal operating point differs radically between the two.

---

## 3. Act 1: The Utilization Trap (Calibration -- 12 minutes)

### Pedagogical Goal
Students transfer their training intuition (maximize utilization = maximize efficiency) to serving, which is exactly wrong. The chapter states: "In training, the goal is Utilization (keeping GPUs at 100% to saturate throughput). In serving, the goal is Headroom (keeping GPUs at 40--60% to absorb traffic spikes before tail latency explodes)." This act forces students to confront the nonlinear relationship between utilization and tail latency, correcting the linear mental model they bring from training.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "A serving system processes requests with a mean service time of 10 ms. At 50% utilization, mean latency is 20 ms and P99 is ~92 ms. You increase utilization to 85% to 'improve efficiency.' What happens to P99 latency?"

Options:
- A) P99 increases to ~140 ms -- roughly proportional to the utilization increase
- B) P99 increases to ~200 ms -- somewhat worse than linear
- **C) P99 increases to ~300 ms -- the tail explodes nonlinearly past the 'knee'** <-- correct
- D) P99 stays about the same -- tail latency is independent of utilization

The correct answer is C. Using M/M/1: mean at 85% = 1/(1-0.85) = 6.67x service time = 66.7 ms. P99 = 4.6 x 66.7 = 307 ms. Students who assume linear scaling expect ~140 ms. The 2x gap between their prediction and reality is the learning moment.

### The Instrument: Tail Latency Explorer

A **dual-axis line chart** showing latency vs. utilization:

- **X-axis:** System utilization (0% to 95%), labeled at 10% intervals
- **Y-axis (left):** Latency (ms), 0 to 500 ms
- **Data series:** Mean latency (BlueLine, dashed), P99 latency (RedLine, solid)
- **Shaded regions:** Safe Zone (green, 0--50%), Caution Zone (orange, 50--70%), Danger Zone (red, 70--95%)
- **"The Knee" annotation** at ~70% utilization with arrow

Controls:
- **Utilization slider** (0% to 95%, step 1%, default 50%): As students drag, a vertical marker moves along the X-axis. Current mean and P99 values display in a live readout panel. The student's prediction is shown as a horizontal dashed line on the chart.
- **Service time slider** (1 ms to 100 ms, step 1 ms, default 10 ms): Scales all latency values proportionally. Students see that the shape of the curve is invariant; only the scale changes.
- **Traffic pattern toggle** (Steady / Poisson Burst): Under Poisson, instantaneous utilization spikes above the average, triggering transient tail explosions even at moderate average utilization.

### The Reveal
After interaction:
> "You predicted P99 = [X] ms at 85% utilization. The actual value is **307 ms** (4.6 x mean of 66.7 ms). You were off by [Y]x. The curve is not linear: below 50% utilization, latency is manageable. Past 70% (the 'knee'), tail latency explodes. This is why production systems run at 40--60% utilization -- not to waste capacity, but to absorb the traffic spikes that would otherwise cause the Black Friday collapse the chapter describes."

### Reflection (Structured)
Four-option multiple choice:

> "A system runs at 50 ms mean latency and 1,000 QPS steady-state. On Black Friday, traffic spikes 10x to 10,000 QPS. What happens?"
- A) Latency increases 10x to 500 ms -- proportional to the traffic increase
- B) Latency increases 3x to 150 ms -- sublinear due to batching efficiency
- **C) The system collapses -- utilization hits 100%, queue lengths explode, and useful throughput drops to near zero** <-- correct
- D) Latency stays the same -- the serving infrastructure auto-scales instantly

**Math Peek (collapsible):**
$$\text{Mean Latency (M/M/1)} = \frac{1}{\mu - \lambda} = \frac{1}{\mu(1 - \rho)} \quad \text{where } \rho = \frac{\lambda}{\mu}$$
$$\text{P99} \approx 4.6 \times \text{Mean Latency (M/M/1 approximation)}$$
$$L = \lambda \times W \quad \text{(Little's Law: in-flight requests = arrival rate × wait time)}$$

---

## 4. Act 2: The Batching Paradox (Design Challenge -- 23 minutes)

### Pedagogical Goal
Students believe that larger batch sizes always improve system performance because batching amortizes overhead. The chapter shows the serving inversion: batching increases throughput but degrades individual request latency. The chapter's cost-of-latency calculation demonstrates that reducing latency from 10 ms to 5 ms increases hardware cost by 300%. Students must find the throughput-latency knee for each deployment context and discover that the optimal batch size is different for a throughput-SLA system (H100, maximize QPS) vs. a latency-SLA system (Jetson Orin NX, minimize P99).

### The Lock (Numeric Prediction)
Before instruments unlock:

> "A GPU server rents for $4/hour. Scenario A (batch=1): 5 ms latency, 200 req/s. Scenario B (batch=8): 10 ms latency, 800 req/s. What is the cost per million queries for Scenario A (the low-latency configuration)?"

Students type a dollar value. Expected wrong answers: $1--2 (students underestimate the cost premium of low latency). Actual: $5.56 per million queries (vs. $1.39 for Scenario B). The 4x cost premium for 2x latency reduction is the learning moment.

### The Instrument: Throughput-Latency Dashboard

**Primary chart -- Throughput vs. P99 Latency Curve:**
- **X-axis:** P99 Latency (ms), 0 to 500 ms
- **Y-axis:** Throughput (QPS), 0 to 10,000
- **Data series:** One curve per batch size (batch=1, 2, 4, 8, 16, 32, 64, 128) showing the throughput-latency operating envelope
- **Vertical red line:** Latency SLA (adjustable)
- **Horizontal green line:** Throughput target (adjustable)
- **"The Knee" annotation:** Where the curve bends; beyond this point, adding batch size increases latency faster than throughput

Controls:
- **Batch size slider** (1 to 128, step power-of-2, default 1): Moves the operating point along the curve. At batch=1, latency is minimal but throughput is low. At batch=128, throughput is high but P99 may exceed SLA.
- **Max batch wait time slider** (0 ms to 50 ms, step 5 ms, default 0 ms): How long the system waits to fill a batch. Longer wait = fuller batches = higher throughput but higher latency. At 0 ms, only requests that arrive simultaneously are batched.
- **Arrival rate slider** (100 to 10,000 QPS, step 100, default 1,000): Simulates traffic load. As arrival rate increases, utilization rises and tail latency grows.
- **Deployment context toggle** (H100: Throughput SLA / Jetson Orin NX: Latency SLA): H100 default SLA = throughput > 5,000 QPS with P99 < 200 ms. Jetson default SLA = P99 < 20 ms with throughput > 100 QPS.

**Secondary chart -- Cost Efficiency (Bar):**
- **X-axis:** Configuration (Batch 1, 4, 8, 16, 32)
- **Y-axis:** Cost per million queries (USD)
- Shows the cost premium of low-latency configurations

**Tertiary chart -- P99 Latency Histogram:**
- Distribution of request latencies for the current configuration
- Mean line (BlueLine), P95 line (OrangeLine), P99 line (RedLine)
- Visually demonstrates how the tail extends far beyond the mean

### The Scaling Challenge
**"For the H100 (throughput SLA), find the batch size that achieves >5,000 QPS while keeping P99 < 200 ms at 2,000 QPS arrival rate. Then switch to the Jetson Orin NX (latency SLA) and find the batch size that keeps P99 < 20 ms while maximizing throughput."**

Students discover:
- H100: batch=16--32 achieves the throughput target within latency budget. Cost per million: ~$0.50
- Jetson: batch=1--2 is the only option within the 20 ms P99 budget. Cost per million: ~$8.00
- The "best" batch size depends entirely on which SLA is binding

### The Failure State
**Trigger:** `p99_latency > sla_target_ms` OR `throughput < throughput_target_qps`

**Visual (Latency violation):** P99 histogram tail turns red; banner:
> "**SLA VIOLATED -- P99 exceeds budget.** P99 = [X] ms (budget: [Y] ms). Reduce batch size or increase replicas."

**Visual (Throughput violation):** Throughput bar turns red; banner:
> "**THROUGHPUT TARGET MISSED.** Achieved [X] QPS (target: [Y] QPS). Increase batch size, but watch the P99."

**Visual (System collapse):** When utilization > 95%:
> "**SYSTEM COLLAPSE -- Queue explosion.** Utilization at [X]%. Useful throughput has dropped to near zero. Load shed or scale out."

### Structured Reflection
Sentence completion with dropdown:

> "The serving inversion means that the optimal utilization target for a latency-sensitive serving system is _____, because _____."

Dropdown 1 options:
- A) 90--100% (maximize hardware ROI)
- B) 70--80% (moderate headroom)
- **C) 40--60% (absorb traffic spikes before tail latency explodes)** <-- correct
- D) 10--20% (minimize all latency)

Dropdown 2 options:
- A) higher utilization causes hardware failures
- **B) P99 latency grows as 1/(1-rho), exploding past the 70% knee** <-- correct
- C) GPU clock speeds decrease at high utilization
- D) batch sizes must increase to maintain throughput

**Math Peek:**
$$L = \lambda \times W \quad \text{(Little's Law)}$$
$$\text{Cost per query} = \frac{\text{GPU cost/hour}}{\text{QPS} \times 3600}$$
$$\text{Throughput} \approx \frac{\text{Batch size}}{\text{Batch latency}} \quad \text{subject to } P99 < \text{SLA}$$

---

## 5. Visual Layout Specification

### Act 1: Utilization Trap
- **Primary:** Dual-line chart -- X: Utilization (0--95%), Y: Latency (ms). Mean (BlueLine dashed), P99 (RedLine solid). Shaded zones: green (0--50%), orange (50--70%), red (70--95%). Prediction overlay as horizontal dashed line.
- **P99 Latency Histogram** available on hover: shows the full distribution at the current utilization point.

### Act 2: Batching Paradox
- **Primary:** Throughput vs. P99 Latency scatter with parametric batch-size curves. SLA box (green rectangle) defined by latency ceiling and throughput floor. Operating point turns red when outside the box.
- **Secondary:** Cost per million queries bar chart. Bars colored green (within SLA) or red (SLA violated).
- **Tertiary:** P99 Latency Histogram -- live updating with mean, P95, P99 vertical lines. Tail highlighted in red when exceeding SLA.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---|---|---|---|---|
| **Cloud (Throughput SLA)** | H100 (80 GB HBM3) | 80 GB | 700 W | Maximize QPS at acceptable P99; batching and high utilization are viable because latency budget is generous (200 ms) |
| **Edge (Latency SLA)** | Jetson Orin NX (8 GB) | 8 GB | 25 W | Minimize P99 latency (< 20 ms); batch size limited to 1--2; utilization must stay very low to guarantee deterministic response |

The two contexts demonstrate the serving inversion: the same model on the same architecture requires opposite optimization strategies depending on which SLA is binding (throughput vs. latency).

---

## 7. Design Ledger Output

```json
{
  "chapter": 13,
  "optimal_batch_h100": 16,
  "optimal_batch_jetson": 1,
  "utilization_target_pct": 55,
  "p99_at_target_ms": 46,
  "throughput_at_target_qps": 3200,
  "cost_per_million_queries_usd": 0.50,
  "sla_type": "throughput | latency"
}
```

The `utilization_target_pct` and `sla_type` feed forward to:
- **Lab 14 (ML Operations):** The P99 latency baseline from serving informs what "healthy" looks like in monitoring; drift detection uses this baseline to flag serving degradation
- **Lab 16 (Conclusion):** The throughput-latency operating point feeds the synthesis Roofline

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| P99 = 4.6x mean latency (M/M/1) | @sec-model-serving, @fig-tail-latency-explosion code (line 96) | "p99_latency = mean_latency * 4.6" |
| Utilization knee at ~70% | @sec-model-serving (line 181) | "latency remains manageable until utilization crosses roughly 70%, then explodes" |
| Production systems run at 40--60% | @sec-model-serving, Serving Inversion callout (line 59) | "keeping GPUs at 40--60% to absorb traffic spikes before tail latency explodes" |
| 10x traffic spike causes collapse, not 10x latency | @sec-model-serving, Black Friday callout (line 169) | "The system does not slow down 10x. It collapses. Latency hits 10 seconds, then requests start timing out" |
| Batch=1 at 200 req/s, 5 ms latency | @sec-model-serving, Cost of Latency callout (line 448) | "Latency: 5 ms. Throughput: 200 req/s" |
| Batch=8 at 800 req/s, 10 ms latency | @sec-model-serving, Cost of Latency callout (line 453) | "Latency: 10 ms (doubled due to batching overhead). Throughput: 800 req/s (quadrupled)" |
| GPU rental cost $4/hour | @sec-model-serving, Cost of Latency callout (line 443) | "a GPU server renting for USD 4/hour" |
| Low latency costs 300% more per query | @sec-model-serving, Cost of Latency callout (line 457) | "Reducing latency from 10 ms to 5 ms increases the hardware bill by 300%" |
| Little's Law: L = lambda * W | @sec-model-serving, Black Friday callout (line 171) | "This is Little's Law and queueing theory in action" |
| Training = Utilization; Serving = Headroom | @sec-model-serving, Serving Inversion callout (line 57--59) | "In training, the goal is Utilization...In serving, the goal is Headroom" |

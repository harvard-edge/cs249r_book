# Mission Plan: lab_08_fleet_orch (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Fleet Orchestration (`@sec-fleet-orchestration`)
- **Core Invariant:** The Utilization Trap: at 80% GPU utilization with heavy-tailed ML workloads (coefficient of variation C_s = 3), the expected queue wait time is 20x the average job duration -- 5x worse than uniform workloads at the same utilization. High utilization does not mean high productivity; it means long queues. The Pollaczek-Khinchine formula `W_q = rho/(1-rho) * (1+C_s^2)/(2*mu)` quantifies exactly how job-duration variance amplifies queuing delay.
- **Central Tension:** Students believe that maximizing GPU utilization (approaching 100%) is the primary scheduling objective. The chapter's queuing theory analysis demonstrates that for ML workloads with heavy-tailed job durations, pushing utilization above 60--70% produces catastrophic queue times. A cluster that reports 80% utilization can have researchers waiting days for their jobs to start. Simultaneously, fragmentation (stranded GPUs that individually satisfy no pending job) means reported utilization overstates productive work. The tension between throughput (favor large jobs) and latency (favor small jobs) has no single solution; every scheduling policy trades one for the other.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict the queue wait time at 80% GPU utilization, assuming uniform job durations. The instrument reveals that ML workloads have a coefficient of variation of 3--5 (mixing 10-minute debug jobs with 3-week training runs), which inflates queue wait time from 4x (uniform) to 20x (ML-typical) the average job duration. A queue wait of 20x means a researcher submitting an average-length job waits 20x that duration before it starts. This explains why ML clusters "feel broken" at 80% utilization while web server clusters feel fine at the same level.

**Act 2 (Design Challenge, 22 min):** Students manage a simulated 10,000-GPU cluster under two scheduling policies: FIFO (first-come-first-served) and Priority scheduling (with preemption). A stream of jobs arrives with heavy-tailed durations. Students must balance utilization against queue latency and discover that fragmentation (30% of capacity stranded as gaps no single job can fill) makes high utilization figures misleading. The challenge is to find a scheduling configuration that keeps both utilization above 60% and median queue wait below 2 hours.

---

## 3. Act 1: The Utilization Trap (Calibration -- 12 minutes)

### Pedagogical Goal

Students assume that high GPU utilization is always desirable and that 80% utilization means the cluster is performing well. The chapter's queuing theory analysis reveals that ML workloads have a heavy-tailed job duration distribution (C_s = 3--5) that turns the familiar "high utilization = good" heuristic into a trap. At 80% utilization with C_s = 3, the expected queue wait is 20x the mean job duration. A cluster where the average job takes 4 hours has researchers waiting 80 hours (over 3 days) for their jobs to start. This act calibrates student intuition by making them predict wait time at 80% utilization, then revealing the variance multiplier.

### The Lock (Structured Prediction)

Present a multiple-choice prediction before any instruments unlock:

> "A 10,000-GPU cluster is running at 80% utilization. Jobs have a heavy-tailed duration distribution: many short debugging runs (minutes) mixed with a few massive training jobs (weeks). The average job duration is 4 hours. How long does a newly submitted average-length job wait in the queue?"

Options:
- A) About 4 hours -- roughly equal to the average job duration
- B) About 16 hours -- a few times the average, standard queuing
- **C) About 80 hours (3+ days) -- the heavy tail inflates wait dramatically** (correct)
- D) About 320 hours (2 weeks) -- exponential blowup at high utilization

Common wrong answer: B. Students who know basic queuing theory (W_q ~ rho/(1-rho)) expect ~4x at 80% utilization but forget the variance multiplier (1+C_s^2)/2 which contributes another 5x for C_s = 3.

### The Instrument: Queuing Theory Dashboard

Controls:
- **Utilization slider (rho)**: 0.1 to 0.95 in steps of 0.05 (default: 0.8)
- **Workload type selector**: Uniform (C_s = 1) / ML-Typical (C_s = 3) / Extreme (C_s = 5)
- **Average job duration**: 1 hr / 4 hrs / 24 hrs / 168 hrs (default: 4 hrs)

Outputs:
- **Primary chart**: Queue wait time vs utilization. X-axis: utilization (0--1.0). Y-axis: wait time (hours, log scale). Three curves: Uniform (C_s = 1, BlueLine), ML-Typical (C_s = 3, OrangeLine), Extreme (C_s = 5, RedLine). Vertical dashed line at current utilization. Horizontal reference: "1 day", "1 week" lines.
- **Secondary metric cards**: "Wait time (W_q)", "Queue depth", "Effective throughput"
- **Annotation**: At 80% utilization, vertical callout showing 4x (uniform) vs 20x (ML) vs 52x (extreme).

Formulas:
- `W_q = rho / (1-rho) * (1 + C_s^2) / (2 * mu)` (Pollaczek-Khinchine)
- Where `mu = 1 / avg_job_duration`, `rho = lambda / mu`
- Queue depth: `L_q = lambda * W_q`

### The Reveal

After interaction:
> "You predicted [X]. At 80% utilization with ML-typical workloads (C_s = 3), the expected queue wait is **20x the average job duration** = 80 hours (3.3 days). For comparison, uniform workloads at the same utilization wait only 4x = 16 hours. The heavy tail of ML job durations acts as a 5x latency multiplier, forcing operators to run ML clusters at 60--70% utilization to maintain acceptable responsiveness."

### Reflection (Structured)

Four-option multiple choice:

> "The chapter states that improving utilization from 50% to 80% on a 10,000-GPU cluster effectively adds 6,000 GPUs of productive capacity. But at 80% utilization, queue wait is 20x average job duration. What is the fundamental trade-off?"

- A) Higher utilization is always better because GPU-hours cost $2/hour
- B) Lower utilization is always better because researcher time is more valuable
- **C) There is an optimal utilization point (60--70% for ML) that balances GPU cost against queue delay** (correct)
- D) The trade-off is irrelevant because preemption eliminates queuing delays

### Math Peek (collapsible)

$$W_q = \frac{\rho}{1 - \rho} \cdot \frac{1 + C_s^2}{2\mu}$$

$$\text{At } \rho = 0.8, \; C_s = 1: \quad W_q = 4 \times \frac{1}{2\mu} = \frac{4}{\mu} = 4 \times \bar{T}$$

$$\text{At } \rho = 0.8, \; C_s = 3: \quad W_q = 4 \times \frac{10}{2\mu} = \frac{20}{\mu} = 20 \times \bar{T}$$

---

## 4. Act 2: The Scheduling Trade-off (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students believe there is a "best" scheduling policy. The chapter shows that every policy trades throughput for latency or fairness for utilization. FIFO is simple but causes head-of-line blocking (a 1024-GPU 3-week job blocks hundreds of small experiments). Priority scheduling with preemption improves small-job latency but wastes compute (preempted jobs lose work since their last checkpoint). Fragmentation compounds the problem: even at 70% reported utilization, 30% of GPUs may be stranded in gaps that no pending job can fill. Students must configure a scheduling policy that satisfies competing objectives on a simulated cluster.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "A 64-node cluster (8 GPUs each, 512 total) is 70% utilized. Jobs request 6 GPUs each, occupying one node but wasting 2 GPUs per node. What percentage of the cluster's GPU capacity is lost to fragmentation?"

Students type a percentage. Expected wrong answers: 5--10% (students underestimate fragmentation). Actual: 2 wasted GPUs per node / 8 GPUs per node = 25% fragmentation. So 70% "utilization" means only 52.5% of GPUs are doing productive work.

### The Instrument: Fleet Scheduling Simulator

Controls:
- **Scheduling policy toggle**: FIFO / Priority (with preemption) / Backfill
- **Preemption checkbox**: On/Off (only available in Priority mode)
- **Job mix slider**: "Small-heavy" (90% small jobs) to "Large-heavy" (90% large jobs) (default: 70% small)
- **Cluster utilization target**: 50% / 60% / 70% / 80% / 90%
- **Deployment context toggle**: FIFO scheduling / Priority scheduling

Outputs:
- **Primary chart**: Cluster heatmap grid (nodes as rows, GPUs as columns). Colors: Green = active job, Gray = free, Red = stranded (fragmented). Updates live as policy changes.
- **Secondary chart**: Dual-axis time series. Left Y-axis: utilization % (green line). Right Y-axis: median queue wait (orange line). X-axis: simulation time (hours, 0--24).
- **Tertiary metrics**: "Effective utilization" (excluding fragmentation), "P50 queue wait", "P99 queue wait", "Jobs completed/hour", "Wasted GPU-hours ($)"

Formulas:
- `fragmentation_pct = stranded_gpus / total_gpus * 100`
- `effective_utilization = (utilized_gpus - stranded_gpus) / total_gpus`
- `daily_waste = stranded_gpus * 24 * $2/GPU-hour`
- `queue_wait = f(policy, rho, C_s)` via Pollaczek-Khinchine

### The Scaling Challenge

**"Configure the scheduling policy and utilization target to simultaneously achieve: (1) effective utilization > 60%, (2) P50 queue wait < 2 hours, and (3) daily GPU waste < $10,000."**

Students discover that:
- FIFO at 80%: utilization = 80%, but P50 wait = 20x ~ 80 hours. Fails criterion 2.
- FIFO at 60%: P50 wait = 6x ~ 24 hours. Still fails criterion 2.
- Priority + preemption at 70%: P50 wait drops to ~4 hours for small jobs, but preemption wastes ~5% of large job compute. Closer but still fails.
- Backfill + 65% target: small jobs fill gaps, effective utilization ~62%, P50 wait ~2 hours. Passes all three.

### The Failure State

**Trigger:** Utilization target set to 90% with ML-typical workload.

**Visual change:** Queue wait line goes vertical (approaching infinity on the plot). Heatmap shows near-solid green with tiny red fragments. Wait time card turns RedLine.

**Banner text:** "QUEUE COLLAPSE -- At 90% utilization with heavy-tailed ML workloads (C_s = 3), expected queue wait is 45x the average job duration = 180 hours (7.5 days). The cluster appears fully utilized but researchers cannot get work done. Reduce target utilization to 60--70% or implement priority scheduling with preemption."

### Structured Reflection

Sentence completion with dropdown:

> "A 10,000-GPU cluster at $2/GPU-hour costs $______ per day to operate. If 30% of GPUs are idle due to scheduling inefficiencies, the annual waste is approximately $______."

Dropdown options for daily cost: $240,000 / $480,000 / $960,000 / $1,200,000
Dropdown options for annual waste: **$53M** (correct: 480K * 0.30 * 365) / $26M / $106M / $175M

### Math Peek

$$W_q = \frac{\rho}{1 - \rho} \cdot \frac{1 + C_s^2}{2\mu} \qquad \text{(Pollaczek-Khinchine)}$$

$$\text{Daily cost} = N_{\text{GPUs}} \times 24 \times \$2 = 10{,}000 \times 24 \times 2 = \$480{,}000$$

$$\text{Annual waste at 30\% idle} = \$480{,}000 \times 0.30 \times 365 = \$52.6\text{M}$$

---

## 5. Visual Layout Specification

### Act 1: Queuing Theory Dashboard
- **Primary:** Multi-line plot. X-axis: utilization (0--1.0). Y-axis: wait time multiplier (log scale, 1x to 1000x). Three curves for C_s = 1, 3, 5. Vertical dashed line at current utilization. Horizontal reference lines at 1 day, 1 week.
- **Secondary:** Three metric cards showing W_q in hours, queue depth, and effective throughput.
- **Failure state:** None (calibration act).

### Act 2: Fleet Scheduling Simulator
- **Primary:** Cluster heatmap grid (64 rows x 8 columns for 512-GPU view, or 8x8 simplified). Green = active, Gray = free, Red = stranded/fragmented.
- **Secondary:** Dual-axis time series. Left: utilization %. Right: P50 queue wait (hours). Simulation runs over 24 simulated hours.
- **Tertiary:** Five metric cards: effective utilization, P50 wait, P99 wait, jobs completed/hr, daily waste ($).
- **Failure state:** Queue wait line goes vertical at high utilization. Banner appears.

---

## 6. Deployment Context Definitions

| Context | Policy | Queue Behavior | Key Constraint |
|---|---|---|---|
| **FIFO scheduling** | First-come-first-served, no preemption | Fair ordering but head-of-line blocking; large jobs block small experiments for days | Throughput-optimal for large jobs, latency-catastrophic for small jobs |
| **Priority scheduling** | Priority classes with preemption; small interactive jobs preempt large training runs | Low latency for priority jobs but wastes compute (preempted jobs lose work since last checkpoint) | Latency-optimal for small jobs, wastes 5--15% of large job compute due to preemption |

The two contexts demonstrate that scheduling is a zero-sum game: every policy that improves one objective degrades another. The chapter's queuing theory provides the quantitative framework for making these trade-offs explicit.

---

## 7. Design Ledger Output

```json
{
  "chapter": 8,
  "scheduling_policy": "fifo | priority | backfill",
  "target_utilization_pct": 65,
  "effective_utilization_pct": 62,
  "p50_queue_wait_hours": 2.0,
  "fragmentation_pct": 25,
  "daily_waste_usd": 7200
}
```

- `effective_utilization_pct` feeds forward to **Lab 09 (Performance Engineering)**: sets the baseline MFU context.
- `scheduling_policy` feeds forward to **Lab 10 (Distributed Inference)**: determines preemption behavior for serving replicas.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Pollaczek-Khinchine formula: W_q = rho/(1-rho) * (1+C_s^2)/(2*mu) | The Queuing Theory of GPU Clusters callout-notebook | "The Pollaczek-Khinchine formula defines the expected waiting time in the queue" |
| C_s = 1 -> W_q = 4x at 80% utilization | The Queuing Theory of GPU Clusters callout-notebook | "uniform workload (C_s = 1), W_q = 4 x the average job duration" |
| C_s = 3 -> W_q = 20x at 80% utilization | The Queuing Theory of GPU Clusters callout-notebook | "typical ML workload (C_s = 3), W_q = 20 x the average job duration" |
| 10,000-GPU cluster at $2/GPU-hour = $480K/day | `@sec-fleet-orchestration-introduction` | "10,000-GPU cluster at $2 per GPU-hour costs $480,000 per day" |
| 30% idle -> $53M annual waste | `@sec-fleet-orchestration-introduction` | "30 percent of GPUs idle...over $53 million annually" |
| Improving utilization 50% to 80% = +6,000 effective GPUs | `@sec-fleet-orchestration-introduction` | "improving utilization from 50 percent to 80 percent effectively adds 6,000 GPUs" |
| 6-GPU jobs on 8-GPU nodes -> 25% fragmentation | `@sec-fleet-orchestration-bin-packing` | "jobs request 6 GPUs each...wasting 2 GPUs per node, reducing effective capacity" |
| Gang scheduling: all-or-nothing allocation | `@sec-fleet-orchestration-gang-scheduling` | "either all N GPUs are allocated atomically, or the job remains in the queue" |
| ML job durations: heavy-tailed, C_s = 3--5 | The Queuing Theory of GPU Clusters callout-notebook | "ML clusters, job durations follow a heavy-tailed distribution...C_s values between 3 and 5" |

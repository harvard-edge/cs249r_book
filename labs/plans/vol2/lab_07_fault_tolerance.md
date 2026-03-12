# Mission Plan: lab_07_fault_tolerance (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Fault Tolerance and Reliability (`@sec-fault-tolerance-reliability`)
- **Core Invariant:** System MTBF scales as `MTBF_component / N`. A 10,000-GPU cluster with 50,000-hour per-GPU MTBF experiences a failure every 5 hours. The Young-Daly optimal checkpoint interval `tau_opt = sqrt(2 * T_write * MTBF)` balances checkpoint overhead against rework cost, and its square-root dependence means doubling cluster size only increases checkpoint frequency by sqrt(2) ~ 1.4x.
- **Central Tension:** Students believe hardware failure is a rare, exceptional event that systems handle through error recovery code paths. The chapter's mathematics prove that at fleet scale, failure is a continuous operating condition: a 16,384-GPU cluster experiences a failure every 2--3 hours (empirically validated by Meta's Llama 3 training logs). The optimal checkpoint interval is not a tuning parameter but a derived quantity from physics (storage bandwidth) and statistics (fleet MTBF). Students who set checkpoint intervals by intuition ("every hour seems fine") systematically waste compute.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict the MTBF of a 10,000-GPU cluster given individual GPU MTBF of 50,000 hours. Most students expect cluster MTBF to remain in the "months" range because they anchor on per-component reliability. The 1/N scaling law reveals that cluster MTBF is just 5 hours, and a 10,000-GPU cluster has only a 32% chance of surviving a single hour without a failure. This shatters the intuition that "reliable components make reliable systems" and establishes failure as a continuous condition at scale.

**Act 2 (Design Challenge, 22 min):** Students apply the Young-Daly formula to find optimal checkpoint intervals for two deployment contexts: an 8-GPU pod (MTBF = 6,250 hours) and a 16,384-GPU cluster (MTBF ~ 3 hours). The challenge reveals that the optimal interval for the small pod is ~35 hours (checkpoint once a day) while for the large cluster it is ~27 minutes (checkpoint constantly). Students must then find the storage bandwidth required to keep the "checkpoint tax" below 10% of total training time, discovering that multi-terabyte checkpoints at 27-minute intervals demand high-performance tiered storage.

---

## 3. Act 1: The 1/N Collapse (Calibration -- 12 minutes)

### Pedagogical Goal

Students dramatically overestimate fleet reliability because they think in terms of individual component MTBF (50,000 hours = 5.7 years) rather than system MTBF (50,000/N hours). The chapter proves that a 10,000-GPU cluster with 99.99%-reliable individual GPUs has only a 32% chance of surviving one hour without failure. The 9s of reliability at component level evaporate at fleet scale. This act forces students to predict cluster MTBF and discover the 1/N collapse.

### The Lock (Structured Prediction)

Present a multiple-choice prediction before any instruments unlock:

> "Each GPU in a 10,000-GPU training cluster has an individual MTBF of 50,000 hours (about 5.7 years). How often does the *cluster as a whole* experience a failure?"

Options:
- A) About once per year -- individual reliability aggregates well
- B) About once per month -- some degradation at scale
- C) About once per week -- significant but manageable
- **D) About once every 5 hours -- failure is a continuous condition** (correct)

Common wrong answer: B. Students reason that "10,000 GPUs means 10,000x more chances to fail" but estimate the impact linearly rather than computing MTBF_cluster = 50,000 / 10,000 = 5 hours.

### The Instrument: Fleet MTBF Calculator

Controls:
- **GPU count slider**: 8 / 64 / 512 / 1,000 / 4,000 / 10,000 / 16,384 / 25,000 (default: 10,000)
- **Per-GPU MTBF slider**: 10,000 / 25,000 / 50,000 / 100,000 hours (default: 50,000)

Outputs:
- **Primary chart**: Semi-log plot. X-axis: GPU count (log scale). Y-axis: Cluster MTBF (hours, log scale). The curve drops with 1/N. Reference lines at 1 day, 1 week, 3 months. Shaded red region where MTBF falls below training duration.
- **Secondary metric cards**: "Cluster MTBF", "Expected failures per day", "Probability of surviving 1 hour"
- **Reference overlay**: Meta Llama 3 data point (16,384 GPUs, 3.1 hours MTBF) and Kokolis et al. RSC data points

Formulas:
- `MTBF_cluster = MTBF_gpu / N`
- `failures_per_day = 24 / MTBF_cluster`
- `P_survive_1hr = (1 - 1/(MTBF_gpu))^N`

### The Reveal

After interaction:
> "You predicted [X]. The actual cluster MTBF for 10,000 GPUs at 50,000h per-GPU MTBF is **5 hours**. The cluster experiences **4.8 failures per day**. Meta's Llama 3 training on 16,384 H100 GPUs confirmed this: 419 failures in 54 days = one failure every 3.1 hours. At scale, failure is not an exception; it is the normal operating condition."

### Reflection (Structured)

Four-option multiple choice:

> "A training run on 1,024 A100 GPUs will last 3 months. The cluster MTBF (including GPU, memory, PCIe, power, and network) is approximately 3.7 hours. How many failures should the team expect?"

- A) 0--5 failures -- the hardware is enterprise-grade
- B) 10--20 failures -- a few per week
- C) About 100 failures -- roughly one per day
- **D) About 600 failures -- roughly one every 3.7 hours for 90 days** (correct)

### Math Peek (collapsible)

$$\text{MTBF}_{\text{cluster}} = \frac{\text{MTBF}_{\text{component}}}{N} = \frac{50{,}000}{10{,}000} = 5 \text{ hours}$$

$$P(\text{survive 1 hour}) = \left(1 - \frac{1}{\text{MTBF}_{\text{gpu}}}\right)^N \approx e^{-N/\text{MTBF}} \approx 0.32$$

$$\text{Failures per day} = \frac{24}{\text{MTBF}_{\text{cluster}}} = \frac{24}{5} = 4.8$$

---

## 4. Act 2: The Checkpoint Tax (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students believe checkpoint intervals should be set by intuition or convenience ("checkpoint every hour"). The Young-Daly formula `tau_opt = sqrt(2 * T_write * MTBF)` reveals that the optimal interval is a derived quantity. For the 16K-GPU cluster (MTBF ~ 3 hours, T_write ~ 2 minutes), the optimal interval is ~27 minutes. For the 8-GPU pod (MTBF ~ 6,250 hours, T_write ~ 30 seconds), the optimal interval is ~35 hours. Students must discover that the checkpoint tax (fraction of time spent writing checkpoints) grows as MTBF shrinks, creating a storage bandwidth crisis at large scale: checkpointing a 2.1 TB GPT-3 checkpoint every 27 minutes demands 1.3 GB/s of sustained write throughput per worker.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "A 16,384-GPU cluster has MTBF of 3 hours. Saving a full checkpoint takes 2 minutes. Using the Young-Daly formula, what is the optimal checkpoint interval in minutes?"

Students type a number in minutes. Expected wrong answers: 60--180 minutes (students anchor on "checkpoint every hour or two"). Actual: sqrt(2 * 2 * 180) = sqrt(720) ~ 27 minutes. The system overlays the student's prediction on the optimization curve.

### The Instrument: Young-Daly Optimizer

Controls:
- **Deployment context toggle**: 8-GPU pod (MTBF = 6,250 hrs) / 16K-GPU cluster (MTBF = 3 hrs)
- **Checkpoint write time slider**: 0.5 / 1 / 2 / 5 / 10 / 15 minutes (default: 2 min)
- **Cluster MTBF slider**: 0.5 / 1 / 3 / 5 / 10 / 50 / 100 / 1000 hours (default: 3 hrs for cluster context)
- **Model checkpoint size slider**: 0.5 / 1.0 / 2.1 / 5.0 / 10.0 TB (default: 2.1 TB for GPT-3)

Outputs:
- **Primary chart**: Three curves. X-axis: checkpoint interval (minutes, 1--360). Y-axis: fraction of time wasted (0--1.0). Blue dashed: checkpoint overhead (T_write / tau). Red dashed: expected rework (tau / 2*MTBF). Green solid: total wasted work. Optimal point annotated.
- **Secondary metric cards**: "Optimal interval (tau_opt)", "Total overhead %", "Required storage throughput (GB/s)"
- **Checkpoint tax gauge**: Semicircular gauge showing % of training time consumed by checkpointing. Green < 5%, Orange 5--15%, Red > 15%.

Formulas:
- `tau_opt = sqrt(2 * T_write * MTBF)`
- `overhead_checkpoint = T_write / tau`
- `overhead_rework = tau / (2 * MTBF)`
- `total_overhead = overhead_checkpoint + overhead_rework`
- `min_overhead = sqrt(2 * T_write / MTBF)` (at optimal)
- `required_throughput = checkpoint_size / T_write`

### The Scaling Challenge

**"Find the maximum checkpoint write time (T_write) that keeps total overhead below 10% for the 16K-GPU cluster (MTBF = 3 hours)."**

Students must solve: `sqrt(2 * T_write / MTBF) < 0.10`, so `T_write < 0.005 * MTBF = 0.005 * 180 = 0.9 minutes`. For a 2.1 TB checkpoint, this requires storage throughput of 2.1 TB / 54 seconds = 40 GB/s per checkpoint write. This reveals why production systems use tiered storage staging (HBM -> NVMe -> PFS) and sharded checkpointing (2.1 TB / 1000 workers = 2.1 GB/worker).

### The Failure State

**Trigger:** Total overhead exceeds 50% (more time spent checkpointing and recovering than computing).

**Visual change:** The total-overhead curve enters the red zone on the gauge. The checkpoint tax bar turns RedLine.

**Banner text:** "CHECKPOINT STORM -- Total overhead exceeds 50%. The system spends more time writing checkpoints and recovering from failures than performing useful computation. At T_write = [X] min and MTBF = [Y] hrs, the optimal interval is [Z] min with [W]% overhead. Reduce checkpoint size (sharding, compression) or increase storage bandwidth to make training feasible."

### Structured Reflection

Four-option multiple choice:

> "The Young-Daly formula shows that tau_opt = sqrt(2 * T_write * MTBF). If you double the cluster size (halving MTBF), by what factor does the optimal checkpoint frequency increase?"

- A) 2x -- checkpoint frequency scales linearly with cluster size
- **B) sqrt(2) ~ 1.4x -- the square root dampens the scaling** (correct)
- C) 4x -- it scales quadratically because both MTBF and risk compound
- D) 1x -- the formula is independent of cluster size

### Math Peek

$$\tau_{\text{opt}} = \sqrt{2 \cdot T_{\text{write}} \cdot \text{MTBF}}$$

$$\text{For 16K GPUs: } \tau_{\text{opt}} = \sqrt{2 \times 2 \times 180} = \sqrt{720} \approx 27 \text{ min}$$

$$\text{For 8 GPUs: } \tau_{\text{opt}} = \sqrt{2 \times 0.5 \times 375{,}000} = \sqrt{375{,}000} \approx 612 \text{ min} \approx 10 \text{ hrs}$$

$$\text{Min overhead} = \sqrt{\frac{2 \cdot T_{\text{write}}}{\text{MTBF}}} = \sqrt{\frac{2 \times 2}{180}} \approx 0.149 = 14.9\%$$

---

## 5. Visual Layout Specification

### Act 1: Fleet MTBF Calculator
- **Primary:** Semi-log plot. X-axis: GPU count (8 to 25,000, log scale). Y-axis: Cluster MTBF (hours, log scale, 0.1 to 100,000). Red curve: MTBF = 50,000/N. Horizontal dashed lines: 1 day, 1 week, 3 months. Scatter overlay: Meta Kokolis et al. data points (blue), Llama 3 data point (red star). Shaded red region below 1-day line.
- **Secondary:** Three metric cards: Cluster MTBF (hours), Expected failures/day, P(survive 1hr) as percentage.
- **Failure state:** None (calibration act).

### Act 2: Young-Daly Optimizer
- **Primary:** Three-curve plot. X-axis: checkpoint interval (1--360 min). Y-axis: fraction of time wasted (0--1.0). Blue dashed: T_write/tau. Red dashed: tau/(2*MTBF). Green solid: total. Black dot at optimum with annotation.
- **Secondary:** Checkpoint tax gauge (semicircular, 0--100%). Green/orange/red zones.
- **Tertiary:** Storage throughput card: "Required: [X] GB/s. Available: [Y] GB/s."
- **Failure state:** Gauge turns red and banner appears when overhead > 50%.

---

## 6. Deployment Context Definitions

| Context | Device | MTBF | Checkpoint Size | Key Constraint |
|---|---|---|---|---|
| **8-GPU pod** | 1 node, 8x H100 (80 GB each) | ~6,250 hours (260 days) | ~50 GB (7B model, FP16 + Adam) | Failures are rare; checkpoint interval is long; storage bandwidth is trivially sufficient |
| **16K-GPU cluster** | 2,048 nodes, 8x H100 each | ~3 hours (empirical: Meta Llama 3) | ~2.1 TB (175B model, FP16 + Adam) | Failures every 2--3 hours; must checkpoint every ~27 min; demands tiered storage at 40+ GB/s aggregate throughput |

The two contexts demonstrate that the same mathematical framework (Young-Daly) produces radically different engineering requirements depending on scale. The 8-GPU pod barely needs fault tolerance; the 16K-GPU cluster cannot operate without it.

---

## 7. Design Ledger Output

```json
{
  "chapter": 7,
  "cluster_size": 16384,
  "cluster_mtbf_hours": 3.0,
  "checkpoint_interval_min": 27,
  "checkpoint_overhead_pct": 14.9,
  "storage_throughput_gbs": 40
}
```

- `checkpoint_interval_min` feeds forward to **Lab 08 (Fleet Orchestration)**: determines how much idle time the scheduler can reclaim between checkpoints.
- `cluster_mtbf_hours` feeds forward to **Lab 08 (Fleet Orchestration)**: sets the failure rate input for the scheduling simulator.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| MTBF_cluster = MTBF_component / N | `@eq-system-mtbf` | "MTBF_system = 1/(N * lambda) = MTBF_component / N" |
| Per-GPU MTBF 50,000 hours | `@sec-fault-tolerance-reliability-reliability-mathematics-inevitable-failure-93ef` | "Modern GPUs in datacenter environments exhibit MTBF values ranging from 40,000 to 100,000 hours" |
| 10,000-GPU cluster MTBF = 5 hours | `@sec-fault-tolerance-reliability-reliability-mathematics-inevitable-failure-93ef` | "A 10,000 GPU cluster...system MTBF of 5 hours" |
| 10,000-GPU cluster 32% survival for 1 hour | Nines of Reliability callout-notebook | "10,000-GPU cluster has only a 32% chance of surviving a single hour" |
| Young-Daly: tau_opt = sqrt(2 * T_write * MTBF) | `@sec-fault-tolerance-young-daly` | "optimal checkpoint interval tau_opt = sqrt(2 * T_write * MTBF)" |
| Llama 3: 16,384 GPUs, 419 failures in 54 days, MTBF ~3.1 hrs | `@fig-published-failure-rates` | "Meta's Llama 3...419 failures across 54 days on 16,384 H100 GPUs" |
| GPT-3 checkpoint size 2.1 TB | fault-tolerance-setup LEGO cell | "gpt3_ckpt_tb = 2.1 TB (weights + Adam states)" |
| Llama 3 scenario: T_write = 2 min, MTBF = 3 hrs, tau_opt = 27 min | Optimal Checkpoint Interval callout-example | "tau_opt = sqrt(2 * 2 * 180) = sqrt(720) approximately 26.8 minutes" |
| Halving MTBF increases checkpoint frequency by sqrt(2) | `@sec-fault-tolerance-young-daly` (fn-young-daly-history) | "halving MTBF only increases optimal checkpoint frequency by sqrt(2) approximately 1.4x" |
| Cluster MTBF at 1024 GPUs ~3.7 hours (with all components) | Worked Example cluster MTBF calculation | "MTBF_cluster = 1/0.2708 = 3.69 hours" |

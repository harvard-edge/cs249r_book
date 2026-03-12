# Mission Plan: lab_10_dist_inference (Volume 2: Fleet Scale)

## 1. Chapter Alignment

- **Chapter:** Inference at Scale (`@sec-inference-scale`)
- **Core Invariant:** The **Serving Cost Dominance Law** -- over a successful model's lifetime, Inference OpEx exceeds Training CapEx by orders of magnitude (9x for a 70B LLM at 1M DAU in year one; 100--1000x for high-QPS recommendation systems). Every architectural decision -- batching strategy, KV cache management, load balancing -- is an economic optimization operating under this invariant. A 10% serving efficiency gain saves more than the entire training budget.
- **Central Tension:** Students believe that training is the expensive part of ML and that inference is a simple forward pass with a straightforward "bigger batch = better throughput" relationship. The chapter demolishes both assumptions. First, serving a 70B model to 1M DAU costs $18M/year vs $2M to train -- a 9x inversion. Second, the queuing hockey stick shows that pushing GPU utilization past 80% causes P99 latency to explode non-linearly: a batch size that yields faster per-request compute (B=8, 54 ms service time) produces *worse* total latency (123 ms) than a batch size with slower compute (B=32, 66 ms) because queuing delay dominates at high utilization. The engineering challenge is finding the narrow corridor where throughput is maximized without violating latency SLOs, while the KV cache memory wall constrains how large that batch can grow.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict which batch size minimizes total response time for a GPT-3 class model at 100 QPS. The intuitive answer -- small batch for fast per-request service -- is catastrophically wrong. Batch sizes 1 and 4 are *unstable* (utilization exceeds 100%), and batch size 8, though stable, suffers 42 ms of queuing delay. Batch size 32, despite 22% longer service time, achieves the lowest total latency because it drives utilization down to 20.6%, where queuing delay nearly vanishes. The queuing hockey stick is the aha moment: students see that the relationship between utilization and latency is exponential, not linear, and that "faster compute per request" does not mean "faster system."

**Act 2 (Design Challenge, 22 min):** Students configure a serving fleet for a 70B LLM under a dual constraint: a P99 latency SLO of 200 ms and a monthly GPU budget. They must select batch size, number of replicas, context length, and load balancing strategy while managing the KV cache memory wall. The KV cache consumes 2.6 MB per token per request for Llama-70B in FP16; at 4096-token context, each request requires ~10.65 GB of cache, limiting a 2-GPU replica to 1--2 concurrent requests. Students discover that the fleet cost is dominated by this memory constraint, not compute, and that switching from random to power-of-two-choices load balancing reduces P99 by 47%, potentially saving entire GPU nodes.

---

## 3. Act 1: The Queuing Hockey Stick (Calibration -- 12 minutes)

### Pedagogical Goal

Students assume that minimizing per-request service time minimizes total response time. The chapter's queuing theory analysis proves otherwise: at 67.5% utilization (B=8), queuing delay is 42.3 ms, making total latency 123.2 ms. At 20.6% utilization (B=32), queuing delay drops to 8.9 ms, making total latency 91.5 ms despite the longer 66 ms service time. The key insight: queuing delay follows a hockey-stick curve that explodes as utilization approaches 100%, and the dominant contribution to response time shifts from compute to queuing as the system loads up. This act forces students to predict the optimal batch size, then reveals the non-linear queuing penalty.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "A GPT-3 class model (175B parameters) is served at 100 QPS on 8x A100 GPUs. Service time follows S(B) = 50 + 0.5B ms. At which batch size is the total response time (including queuing delay) minimized?"

Options:

- A) Batch size 1 -- minimal service time per request means fastest response
- B) Batch size 8 -- balanced trade-off between service time and queuing
- **C) Batch size 32 -- despite longer service time, queuing delay is dramatically lower** (correct)
- D) Batch size 200 -- the square root law predicts the optimum at sqrt(2 * 50 * 100 / 0.25) = 200

Distractor rationale: **A** is the naive intuition (fast per-request = fast system), but B=1 yields 505% utilization and infinite queues. **B** seems like a reasonable middle ground but has 42.3 ms queuing delay. **D** is the mathematically unconstrained optimum from @eq-optimal-batch-approx, but memory constraints cap practical batch size at 32--64 for large LLMs.

### The Instrument: Batch Size vs. Total Latency Explorer

**Primary chart: Dual-axis latency-throughput curve**

- **X-axis:** System utilization (rho), range 0%--100%
- **Y-axis (left):** Total response time E[T] in ms, range 0--500 ms
- **Y-axis (right):** Throughput in req/s, range 0--600 req/s
- **Data series 1 (BlueLine):** Total latency curve (hockey-stick shape from Kingman's formula)
- **Data series 2 (GreenLine):** Throughput curve (rising, saturating)
- **Threshold line (RedLine, dashed):** Latency SLO at 200 ms P99
- **Annotation:** Current batch size marked on the curve with a large dot

Controls:

- **Batch size selector** (radio: 1, 2, 4, 8, 16, 32, 64): Each selection updates utilization, service time, queuing delay, and total latency. Values from @tbl-batch-size-comparison:
  - B=1: S=50.5 ms, rho=505% (UNSTABLE), throughput=19.8 req/s
  - B=4: S=52.0 ms, rho=130% (UNSTABLE), throughput=76.9 req/s
  - B=8: S=54.0 ms, rho=67.5%, E[W]=42.3 ms, E[T]=123.2 ms, throughput=148.1 req/s
  - B=16: S=58.0 ms, rho=36.2%, E[W]=16.4 ms, E[T]=92.4 ms, throughput=275.9 req/s
  - B=32: S=66.0 ms, rho=20.6%, E[W]=8.9 ms, E[T]=91.5 ms, throughput=484.8 req/s
- **Arrival rate slider** (lambda): 50--200 QPS, default 100, step 10

**Secondary chart: Latency decomposition stacked bar**

- **Chart type:** Stacked horizontal bar
- **X-axis:** Time (ms), range 0--200 ms
- **Segments:** Queuing delay E[W] (OrangeLine), Batch accumulation B/2lambda (BlueLine), Compute S(B) (GreenLine)
- Updates per batch size selection; at B=8 queuing dominates, at B=32 compute dominates

### The Reveal

After interaction:

> "You predicted batch size [X]. The optimal batch size is **32**. Despite 22% longer service time (66 ms vs 54 ms at B=8), total latency is **26% lower** (91.5 ms vs 123.2 ms) because queuing delay drops from 42.3 ms to 8.9 ms. The queuing hockey stick means reducing utilization from 67.5% to 20.6% has a larger effect on total response time than reducing per-batch compute time."

### Reflection (Structured)

Four-option multiple choice:

> "Batch sizes 1 and 4 are marked UNSTABLE for this system at 100 QPS. What does system instability mean in queuing theory?"

- A) The GPU runs out of memory and the process crashes
- B) The service time exceeds the SLO latency threshold
- **C) The arrival rate exceeds the effective service rate, so queues grow without bound and latency approaches infinity** (correct)
- D) The model produces incorrect outputs due to numerical precision loss at small batch sizes

### Math Peek (collapsible)

$$\mu_{eff}(B) = \frac{B}{S(B)} = \frac{B}{\alpha + \beta B} \quad \text{(effective service rate)}$$

$$E[T(B)] = \underbrace{\frac{\lambda \cdot S(B)^2}{2B(1 - \lambda S(B)/B)}}_{\text{Queuing delay}} + \underbrace{\frac{B}{2\lambda}}_{\text{Batch accumulation}} + \underbrace{S(B)}_{\text{Compute}}$$

Stable when $\mu_{eff}(B) > \lambda$: $B_{min} = \frac{\alpha\lambda}{1 - \beta\lambda}$. For $\alpha=50, \beta=0.5, \lambda=100$: $B_{min} = \frac{50 \times 100}{1 - 0.5 \times 100} = \frac{5000}{-49}$ -- negative denominator means B=1 through B=4 cannot stabilize this system at any batch size within that range. B must exceed ~6 for stability.

---

## 4. Act 2: The Fleet Designer (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students believe that fleet sizing is simple capacity planning: divide total QPS by per-replica throughput. The chapter reveals three interacting constraints: (1) the KV cache memory wall limits batch size for long-context requests -- a 70B model at 128K context can serve only 1 concurrent request on 8xH100 (640 GB total); (2) the serving tax consumes 10--30% of the latency budget in overhead; and (3) load balancing algorithm choice has quantitative P99 impact -- power-of-two-choices reduces P99 by 47% vs random assignment. Students must navigate all three constraints simultaneously, discovering that the fleet cost is dominated by the memory constraint rather than compute throughput.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "You are serving Llama-70B (140 GB FP16 weights) on a 2-GPU H100 replica (160 GB total). After weights and system overhead (~20 GB), 0 GB remains for... wait. After weights (140 GB) and overhead, there is no room for KV cache on 2 GPUs. You need at minimum 4 GPUs. On a 4-GPU replica (320 GB total), with 140 GB weights and 20 GB overhead, 160 GB remains for KV cache. At 2.6 MB per token per request, how many concurrent 4096-token requests can this replica serve?"

Students type an integer. Expected wrong answers: 30--50 (students divide 160 GB by a rough per-request estimate without computing carefully). Actual: 4096 tokens x 2.6 MB/token = 10.65 GB per request. 160 GB / 10.65 GB = **15** concurrent requests.

### The Instrument: Fleet Configuration Dashboard

**Primary chart: Fleet Cost vs. P99 Latency Pareto Frontier**

- **Chart type:** Scatter plot with Pareto frontier line
- **X-axis:** P99 latency (ms), range 50--500 ms
- **Y-axis:** Monthly fleet cost ($), range $10K--$500K
- **Shaded SLO region (GreenLine, alpha=0.1):** P99 < 200 ms
- **Shaded budget region (BlueLine, alpha=0.1):** Cost < $150K/month
- **Feasible region:** Intersection of both shaded areas
- **Current configuration:** Large highlighted dot; infeasible configurations shown in RedLine

Controls:

- **Replicas slider** (2--32, step 2, default 8): Number of 4-GPU replica groups
- **Batch size slider** (1, 4, 8, 16, 32, default 8): Per-replica max concurrent requests
- **Context length selector** (512 / 2048 / 4096 / 8192 / 16384 tokens, default 2048): Affects KV cache memory per request, constraining max batch size
- **Load balancing selector** (Random / Round-Robin / Power-of-Two-Choices / Least-Connections): Affects P99 latency per @tbl-lb-comparison ratios
- **Deployment context toggle** (see Section 6)

**Secondary chart: Per-Replica Memory Waterfall**

- **Chart type:** Stacked vertical bar
- **Y-axis:** Memory (GB), range 0--350 GB
- **Segments:** Weights (BlueLine, 140 GB), KV Cache (OrangeLine, variable), System Overhead (neutral, 20 GB), Free (GreenLine)
- **Capacity line (RedLine, dashed):** 320 GB (4x H100)
- Turns red when total exceeds capacity

**Tertiary display: Load Balancing Comparison**

Live-updating row from @tbl-lb-comparison, scaled to current fleet size:

| Algorithm | Max Queue | P99 Latency | Overhead |
|---|---|---|---|
| Random | 4.2 req | 45 ms | Minimal |
| Round-Robin | 2.8 req | 32 ms | Minimal |
| Two-Choices | 1.9 req | 24 ms | 2 probes/req |
| Least-Connections | 1.4 req | 19 ms | Global state |

### The Scaling Challenge

**"Find the cheapest fleet configuration that serves 1,000 QPS of Llama-70B inference with P99 latency < 200 ms, supporting context lengths up to 4,096 tokens."**

Students must discover:

1. At 4096-token context, KV cache per request is 10.65 GB, limiting each 4-GPU replica to ~15 concurrent requests
2. Each replica's throughput depends on batch size and service time: at B=15, throughput is ~227 req/s per replica (from the batching efficiency formula)
3. Meeting 1,000 QPS requires ~5 replicas minimum (5 x 227 = 1,135 QPS capacity), but this operates at high utilization
4. Adding replicas to reduce utilization (and thus queuing delay) is the key to meeting the P99 SLO
5. Power-of-Two-Choices load balancing reduces P99 by 47% compared to random, potentially saving 2--3 replicas worth of cost
6. Reducing context length to 2048 tokens doubles the per-replica batch capacity, halving fleet cost

### The Failure State

**Trigger condition:** `kv_cache_per_request * batch_size + weight_memory + overhead > replica_memory_capacity`

**Visual change:** Memory waterfall bar overflows past the capacity line; KV cache segment turns RedLine.

**Banner text:**

> "OOM -- KV Cache Memory Exceeded. At context length [X] tokens with batch size [Y], each request requires [Z] GB of KV cache. Total memory demand ([W] GB) exceeds replica capacity (320 GB). Reduce batch size or context length."

This failure state is **reversible**: pulling the batch size or context length slider back clears the OOM state immediately.

**Secondary failure state (SLO violation):**

**Trigger condition:** `p99_latency > 200`

**Visual change:** Current configuration dot on Pareto chart turns RedLine; SLO boundary flashes.

**Banner text:**

> "SLA Violated -- P99 latency [X] ms exceeds 200 ms target. Add replicas or switch to Power-of-Two-Choices load balancing to reduce tail latency."

### Structured Reflection

Four-option multiple choice:

> "You discovered that at 4096-token context, a 4-GPU Llama-70B replica can serve at most ~15 concurrent requests. What is the primary cause of this capacity limitation?"

- A) The H100 compute throughput is insufficient for 70B parameter forward passes
- B) Tensor parallelism communication overhead across 4 GPUs consumes the latency budget
- **C) The KV cache grows linearly with context length (2.6 MB/token), consuming the GPU memory that would otherwise hold additional concurrent requests** (correct)
- D) The attention mechanism becomes quadratically slow at 4096-token contexts, limiting throughput

### Math Peek (collapsible)

$$M_{KV} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times P_{prec} = 2 \times 80 \times 64 \times 128 \times 2 \approx 2.6 \text{ MB/token}$$

$$\text{Max Batch} = \left\lfloor \frac{M_{replica} - M_{weights} - M_{overhead}}{M_{KV} \times \text{Context Length}} \right\rfloor = \left\lfloor \frac{320 - 140 - 20}{0.0106 \times 1000} \right\rfloor$$

$$E[\text{max queue}]_{\text{two choices}} = \Theta(\log \log n) \quad \text{vs} \quad E[\text{max queue}]_{\text{random}} = \Theta\left(\frac{\log n}{\log \log n}\right)$$

---

## 5. Visual Layout Specification

### Act 1: Queuing Hockey Stick

1. **Primary: Dual-axis latency-throughput curve**
   - Chart type: Line chart, two Y-axes
   - X-axis: System utilization (rho), 0--100%
   - Y-axis left: Total response time E[T] (ms), 0--500 ms
   - Y-axis right: Throughput (req/s), 0--600 req/s
   - Series: Hockey-stick latency (BlueLine), throughput (GreenLine)
   - Threshold: SLO at 200 ms (RedLine, dashed)
   - Failure state: When rho > 100%, chart area turns red with "UNSTABLE" overlay text

2. **Secondary: Latency decomposition stacked bar**
   - Chart type: Stacked horizontal bar
   - X-axis: Time (ms), 0--200 ms
   - Segments: Queuing delay (OrangeLine), Batch accumulation (BlueLine), Compute (GreenLine)
   - Updates per batch size selection

### Act 2: Fleet Designer

1. **Primary: Cost vs. P99 Pareto scatter**
   - Chart type: Scatter with connected Pareto frontier
   - X-axis: P99 latency (ms), 50--500 ms
   - Y-axis: Monthly cost ($), $10K--$500K
   - Shaded feasible region (intersection of SLO and budget bounds)
   - Current config highlighted; infeasible configs in RedLine

2. **Secondary: Per-replica memory waterfall**
   - Chart type: Stacked vertical bar
   - Y-axis: Memory (GB), 0--350 GB
   - Segments: Weights (BlueLine, 140 GB fixed), KV Cache (OrangeLine, variable), Overhead (neutral, 20 GB), Free (GreenLine)
   - Capacity line at 320 GB (RedLine, dashed)
   - Failure state: KV cache overflows, turns red, OOM banner

3. **Tertiary: Load balancing comparison table**
   - 4-row table: Algorithm, Max Queue, P99 Latency, Overhead
   - Selected algorithm row highlighted in BlueLine

---

## 6. Deployment Context Definitions

| Context | Device | Total Replica Memory | Available for KV Cache | Key Constraint |
|---|---|---|---|---|
| **4x H100 NVLink** | 4x H100 (80 GB each), NVLink interconnect | 320 GB | 160 GB (after 140 GB weights + 20 GB overhead) | Communication overhead is small (~0.3 ms per AllReduce via NVLink at 900 GB/s). KV cache is the binding constraint. At 4096-token context, max batch = 15 concurrent requests per replica. |
| **2x H100 PCIe** | 2x H100 (80 GB each), PCIe interconnect | 160 GB | 0 GB (140 GB weights + 20 GB overhead = 160 GB; no room for KV cache) | Model barely fits; no memory for KV cache without quantization. Must use INT8 weights (70 GB) to free 70 GB for cache. Even then, max batch = 6 at 4096-token context. Forces cost-latency compromise: more, cheaper replicas vs fewer, larger ones. |

The two contexts demonstrate that the KV cache wall creates fundamentally different operating regimes. The 4-GPU NVLink replica is memory-rich and communication-efficient; the 2-GPU PCIe replica cannot even begin serving without weight quantization. Students see that "fitting the model" is necessary but nowhere near sufficient -- the KV cache budget determines the economics.

---

## 7. Design Ledger Output

```json
{
  "chapter": 10,
  "serving_batch_size": 15,
  "replicas_chosen": 8,
  "load_balancing_algorithm": "two_choices",
  "context_length_tokens": 4096,
  "p99_latency_ms": 175,
  "monthly_fleet_cost_usd": 96000,
  "kv_cache_gb_per_request": 10.65,
  "deployment_topology": "4x_h100_nvlink"
}
```

The `serving_batch_size` and `load_balancing_algorithm` fields feed forward to:

- **Lab 12 (Ops at Scale):** The fleet configuration informs monitoring threshold selection and SLO alerting design
- **Lab 15 (Sustainable AI):** The monthly fleet cost and GPU count feed into the energy and carbon efficiency analysis

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Service time model S(B) = 50 + 0.5B ms | `@sec-inference-scale-worked-example-gpt3-serving-100-qps-edcf`, lines 1007--1008 | "Weight loading overhead: alpha = 50 ms ... Per-token compute: beta = 0.5 ms per request" |
| B=1 unstable at rho=505%, B=4 unstable at rho=130% | `@tbl-batch-size-comparison`, lines 1032--1033 | "B=1: 505% (unstable) ... B=4: 130% (unstable)" |
| B=8: S=54 ms, rho=67.5%, E[W]=42.3 ms, E[T]=123.2 ms | `@tbl-batch-size-comparison`, line 1034 | "54.0 ... 148.1 ... 67.5% ... 42.3 ... 123.2" |
| B=32: S=66 ms, rho=20.6%, E[W]=8.9 ms, E[T]=91.5 ms | `@tbl-batch-size-comparison`, line 1036 | "66.0 ... 484.8 ... 20.6% ... 8.9 ... 91.5" |
| Square root law for optimal batch size | `@eq-optimal-batch-approx`, line 956 | "B ~ sqrt(2 * alpha * lambda / (1 - rho_target))" |
| Queuing hockey stick: latency explodes near 100% utilization | `@fig-queuing-hockey-stick`, line 1099 | "As utilization approaches 100%, the queuing delay increases exponentially" |
| KV cache = 2.6 MB/token for Llama-70B FP16 | `@sec-inference-scale-kv-cache-wall`, line 2063 | "M_KV = 2 x 80 x 64 x 128 x 2 = 2,621,440 bytes ~ 2.6 MB/token" |
| 128K context request needs 340 GB KV cache; max batch = 1 on 8xH100 | `@sec-inference-scale-kv-cache-wall`, lines 2067--2073 | "131,072 tokens x 2.6 MB/token ~ 340 GB/request ... Max Batch = floor(480/340) = 1" |
| 60--80% KV cache memory wasted from fragmentation (pre-allocated) | `@sec-inference-scale-fragmentation-problem-28be`, line 2101 | "Production systems report 60--80% memory waste from fragmentation" |
| PagedAttention enables 2--4x concurrent throughput | `@sec-inference-scale-pagedattention-3c94`, line 2109 | "it enables 2x to 4x higher Concurrent Throughput on the same hardware" |
| Serving cost 9x training cost (70B LLM, 1M DAU, year 1) | Serving Cost Multiplier callout, lines 228--234 | "Serving costs 9x more than training in just the first year" |
| Serving tax: 10--30% of latency budget | `@sec-inference-scale-serving-tax-overhead-distribution-5ef3`, line 193 | "The total serving tax often consumes 10--30% of the latency budget" |
| Power-of-two-choices: P99 = 24 ms vs random P99 = 45 ms (47% improvement) | `@tbl-lb-comparison`, lines 4013--4019 | "Two-choices: 1.9 req max queue, 24 ms P99 ... Random: 4.2 req, 45 ms" |
| Power-of-two-choices: max queue O(log log n) vs O(log n / log log n) | `@eq-two-choices-max-queue`, lines 3609--3617 | "Random max queue: ~4-5 requests; Two choices max queue: ~2 requests" |
| H100 HBM capacity: 80 GB | `@sec-inference-scale-singlemachine-serving-insufficient-3f13`, line 125 | "A single NVIDIA H100 GPU provides 80GB of HBM3 memory" |
| Llama-70B requires 140 GB FP16, minimum 2-way sharding | `@sec-inference-scale-sharding-becomes-necessary-81d4`, lines 2867--2870 | "70 x 10^9 x 16/8 = 140 GB ... exceeds the 80GB capacity ... requiring at minimum 2-way sharding" |
| NVLink H100: 900 GB/s; AllReduce ~0.3 ms per 8 MB activation | TP for Llama-70B callout, lines 3048, 3055 | "AllReduce (attention): 0.3 ms ... 900 GB/s NVLink bandwidth" |
| 8-way TP: 6.9x realized speedup vs 8x theoretical | TP for Llama-70B callout, line 3048 | "Total per layer: 30 ms sequential, 4.35 ms 8-way TP, speedup 6.9x" |
| Continuous batching: 37.5% avg latency reduction for variable outputs | `@sec-inference-scale-quantitative-analysis-traditional-vs-continuous-batching-ca70`, line 1518 | "W = 1 - 125/200 = 37.5% ... 37.5% reduction in average latency" |
| Cold start for Llama-70B: 5 min 20 sec total | `@sec-inference-scale-cold-start-problem-ecf9`, line 4388 | "Total cold start: 5 min 20 sec" |

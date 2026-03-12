# Mission Plan: lab_06_collect_comms (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Collective Communication (`@sec-collective-communication`)
- **Core Invariant:** The Ring AllReduce bandwidth term is `2(N-1)/N * M/beta`, which converges to `2M/beta` as N grows. Per-node communication volume is **constant** regardless of cluster size; the bottleneck is gradient volume (proportional to model size), not participant count.
- **Central Tension:** Students believe that adding more GPUs to a training cluster proportionally increases communication overhead per node. The alpha-beta model reveals two distinct regimes: small messages are latency-bound (dominated by alpha), large messages are bandwidth-bound (dominated by M/beta), and the crossover at `n* = alpha * beta` (~100 KB for InfiniBand NDR) determines which optimization strategy matters. Ring AllReduce is bandwidth-optimal but latency-inefficient; Tree AllReduce is latency-optimal but bandwidth-inefficient. Choosing the wrong algorithm for the wrong regime wastes engineering effort entirely.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students confront the alpha-beta model by predicting how long a 70B-model AllReduce takes on 64 GPUs. Most students dramatically underestimate the communication time because they focus on the latency term (microseconds) and ignore the bandwidth term (seconds). The instrument reveals that for large gradient payloads (280 GB), the bandwidth term dominates by a factor of 30,000x over the latency term. This calibrates the student's intuition: at LLM scale, communication is a bandwidth problem, not a latency problem.

**Act 2 (Design Challenge, 22 min):** Students toggle between Ring and Tree AllReduce algorithms across varying message sizes and cluster scales to find the crossover point where Tree beats Ring. The challenge forces them to discover that Ring AllReduce's latency penalty (2(N-1) * alpha) becomes catastrophic for small messages at large N, while its bandwidth advantage (2(N-1)/N converges to 2) makes it unbeatable for large messages. Students must configure a hierarchical strategy that uses Tree intra-node (small, latency-sensitive) and Ring inter-node (large, bandwidth-sensitive) to hit a target AllReduce time.

---

## 3. Act 1: The Bandwidth Surprise (Calibration -- 12 minutes)

### Pedagogical Goal

Students dramatically underestimate AllReduce communication time for large models because they anchor on the per-message latency (microseconds) rather than the per-byte bandwidth cost (seconds). The chapter's napkin math shows that a 70B-parameter model in FP32 generates 280 GB of gradients, and Ring AllReduce across 64 GPUs on InfiniBand NDR (50 GB/s) takes approximately 11,200 ms -- over 11 seconds of pure communication per training step. The latency term (0.4 ms) is negligible. This act forces students to predict the AllReduce time and discover that bandwidth, not latency, is the binding constraint for LLM-scale gradient synchronization.

### The Lock (Structured Prediction)

Present a multiple-choice prediction before any instruments unlock:

> "A 70B-parameter model trains with data parallelism across 64 GPUs connected by InfiniBand NDR (50 GB/s per port, 3 us latency). Gradients are FP32 (4 bytes per parameter). How long does one Ring AllReduce take?"

Options:
- A) About 0.5 ms -- network latency dominates, and 64 hops add up
- B) About 50 ms -- significant but not a bottleneck
- C) About 1,100 ms (~1 second) -- bandwidth starts to matter at this scale
- **D) About 11,000 ms (~11 seconds) -- bandwidth completely dominates** (correct)

Common wrong answer: B or C. Students who recall that InfiniBand latency is microseconds assume the total stays in the millisecond range. They forget to multiply 280 GB by the bandwidth factor 2(N-1)/N.

### The Instrument: AllReduce Cost Calculator

Controls:
- **Model Parameters slider**: 1B / 7B / 13B / 70B / 175B (default: 70B)
- **Precision selector**: FP32 (4 bytes) / BF16 (2 bytes) / FP8 (1 byte) (default: FP32)
- **GPU count slider**: 8 / 16 / 32 / 64 / 128 / 256 / 512 / 1024 (default: 64)
- **Interconnect toggle**: NVLink 4.0 (900 GB/s, 1 us) / InfiniBand NDR (50 GB/s, 3 us) / InfiniBand HDR (25 GB/s, 5 us)

Outputs:
- **Primary chart**: Stacked bar showing bandwidth term vs latency term of AllReduce time. X-axis: component (bandwidth, latency). Y-axis: time (ms, log scale).
- **Secondary metric**: "Bandwidth fraction" percentage badge showing what % of total time is bandwidth.
- **Gradient size readout**: total gradient payload in GB.

Formulas:
- `gradient_size = params * bytes_per_param`
- `T_bandwidth = 2 * (N-1)/N * gradient_size / beta` (ms)
- `T_latency = 2 * (N-1) * alpha` (ms)
- `T_total = T_bandwidth + T_latency`

### The Reveal

After interaction:
> "You predicted [X]. The actual AllReduce time for 70B FP32 on 64 GPUs over InfiniBand NDR is **11,200 ms** (11.2 seconds). The bandwidth term is 11,200 ms (99.997%). The latency term is 0.4 ms (0.003%). At LLM scale, communication is entirely a bandwidth problem."

### Reflection (Structured)

Four-option multiple choice:

> "The chapter states that gradient synchronization consumes 30--70% of training step wall-clock time at scale. Given that a single AllReduce takes 11 seconds, what is the primary technique for reducing this overhead?"

- A) Reduce the number of GPUs to lower the latency term 2(N-1) * alpha
- B) Switch from Ring to Tree AllReduce to reduce the bandwidth factor
- **C) Overlap AllReduce with the backward pass so communication hides behind computation** (correct)
- D) Increase the InfiniBand link count to 10x the bandwidth

### Math Peek (collapsible)

$$T_{\text{Ring}} = \underbrace{2 \cdot \frac{N-1}{N} \cdot \frac{M}{\beta}}_{\text{bandwidth term}} + \underbrace{2(N-1) \cdot \alpha}_{\text{latency term}}$$

$$\text{For } N=64,\; M=280\text{ GB},\; \beta=50\text{ GB/s},\; \alpha=3\;\mu\text{s}:$$
$$T_{\text{bw}} = 1.97 \times \frac{280}{50} \times 1000 \approx 11{,}032\text{ ms}$$
$$T_{\text{lat}} = 126 \times 0.003 \approx 0.4\text{ ms}$$

---

## 4. Act 2: Ring vs Tree -- The Algorithm Crossover (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students believe Ring AllReduce is always better because it is "bandwidth-optimal." The chapter shows that Ring's latency term scales as 2(N-1) * alpha, while Tree's scales as 2 * log2(N) * alpha. For small messages at large cluster sizes, Ring's latency penalty becomes catastrophic. The crossover message size `M_crossover = N * alpha * beta / (N - log2(N))` determines which algorithm wins. Students must find this crossover by manipulating message size and cluster scale, then design a hierarchical strategy that picks the right algorithm per regime.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "At 1024 GPUs on InfiniBand NDR (alpha = 3 us, beta = 50 GB/s), what is the maximum message size (in KB) where Tree AllReduce is faster than Ring AllReduce?"

Students type a number in KB. Expected wrong answers: 1--10 KB (too small) or 10+ MB (too large). Actual crossover: approximately 150 KB. The system will overlay the student's prediction on the crossover plot.

### The Instrument: Algorithm Crossover Comparator

Controls:
- **Message size slider**: 1 KB to 1 GB (log scale, default: 100 KB)
- **GPU count slider**: 8 / 64 / 256 / 1024 / 4096 (default: 64)
- **Algorithm toggle**: Ring / Tree / Hierarchical (Ring intra-node + Tree inter-node)
- **Deployment context toggle**: Ring topology (bandwidth-optimal) / Tree topology (latency-optimal)

Outputs:
- **Primary chart**: Dual-line plot. X-axis: message size (KB, log scale). Y-axis: AllReduce time (ms, log scale). Two lines: Ring (BlueLine) and Tree (GreenLine). Crossover annotation at intersection.
- **Secondary chart**: Communication breakdown waterfall showing bandwidth and latency components for each algorithm.
- **Crossover badge**: "Tree wins below [X] KB; Ring wins above [X] KB"

Formulas:
- Ring: `T_ring = 2(N-1)/N * M/beta + 2(N-1) * alpha`
- Tree: `T_tree = 2 * log2(N) * M/(N*beta) + 2 * log2(N) * alpha`
- Crossover: where `T_ring = T_tree`
- Hierarchical: `T_hier = T_tree_intra(n_per_node, M) + T_ring_inter(n_nodes, M)`

### The Scaling Challenge

**"Configure a hierarchical AllReduce strategy for a 70B model (280 GB FP32 gradients) on 512 GPUs (64 nodes of 8 GPUs) that achieves total AllReduce time under 6,000 ms."**

Students must select:
- Intra-node algorithm (NVLink, 900 GB/s): Ring or Tree for 8 GPUs
- Inter-node algorithm (InfiniBand NDR, 50 GB/s): Ring or Tree for 64 nodes
- Optional gradient compression: None / FP16 (2x) / FP8 (4x)

Key discovery: without compression, Ring-Ring hierarchy achieves ~5,800 ms. With BF16 compression (halving gradient size to 140 GB), it drops to ~2,900 ms. Tree inter-node at 64 nodes is slower than Ring inter-node because the message (140+ GB) is firmly in the bandwidth regime.

### The Failure State

**Trigger:** Student selects Tree algorithm for inter-node communication with message size > 1 GB.

**Visual change:** The Tree line on the chart turns RedLine. The AllReduce time bar extends past the right edge.

**Banner text:** "COMMUNICATION BOTTLENECK -- Tree AllReduce selected in bandwidth-bound regime. At 280 GB message size, Tree requires 64 sequential transfers instead of Ring's pipelined approach. AllReduce time: [X] ms (expected: [Y] ms with Ring). Switch to Ring for bandwidth-bound messages."

### Structured Reflection

Four-option multiple choice:

> "The chapter states that communication consumes 25% of step time at 8 GPUs (NVLink) but 65% at 4096 GPUs (InfiniBand). What is the primary reason?"

- A) Ring AllReduce's per-node bandwidth cost increases with N
- **B) The transition from NVLink (900 GB/s) to InfiniBand (50 GB/s) reduces available bandwidth by 18x** (correct)
- C) Tree AllReduce becomes mandatory at large scale, adding log(N) overhead
- D) Gradient size increases with the number of GPUs

### Math Peek

$$T_{\text{Ring}} = 2 \cdot \frac{N-1}{N} \cdot \frac{M}{\beta} + 2(N-1) \cdot \alpha$$
$$T_{\text{Tree}} = 2 \cdot \lceil\log_2 N\rceil \cdot \frac{M}{N \cdot \beta} + 2 \cdot \lceil\log_2 N\rceil \cdot \alpha$$
$$n^* = \alpha \cdot \beta \approx 3\;\mu\text{s} \times 50\;\text{GB/s} = 100\;\text{KB}$$

---

## 5. Visual Layout Specification

### Act 1: AllReduce Cost Calculator
- **Primary:** Stacked bar chart. X-axis: component (Bandwidth, Latency). Y-axis: time in ms (log scale). Bandwidth bar in OrangeLine, Latency bar in BlueLine. Shows relative dominance.
- **Secondary:** Gradient size readout card with model params, precision, and total GB.
- **Failure state:** None in Act 1 (calibration only).

### Act 2: Algorithm Crossover Comparator
- **Primary:** Dual-line plot. X-axis: message size (KB, log scale, 1 KB to 1 GB). Y-axis: AllReduce time (ms, log scale). Ring line in BlueLine, Tree line in GreenLine. Crossover point annotated. Student prediction overlaid as dashed vertical line.
- **Secondary:** Communication waterfall. X-axis: component (BW intra, Lat intra, BW inter, Lat inter). Y-axis: time (ms). Stacked for hierarchical view.
- **Failure state:** Tree line turns RedLine when message > 10x crossover size. Banner appears.

---

## 6. Deployment Context Definitions

| Context | Topology | Intra-node BW | Inter-node BW | Key Constraint |
|---|---|---|---|---|
| **Ring topology (bandwidth-optimal)** | 8 GPUs/node, NVLink 900 GB/s intra, IB NDR 50 GB/s inter | 900 GB/s | 50 GB/s | Large gradient payloads (140--280 GB); bandwidth dominates; Ring AllReduce is optimal inter-node |
| **Tree topology (latency-optimal)** | 8 GPUs/node, NVLink 900 GB/s intra, IB NDR 50 GB/s inter | 900 GB/s | 50 GB/s | Small activation tensors (1--100 KB for pipeline/MoE); latency dominates; Tree AllReduce is optimal inter-node |

The two contexts demonstrate that the same physical network demands different algorithms depending on message size. The chapter's alpha-beta model quantifies exactly where the crossover occurs.

---

## 7. Design Ledger Output

```json
{
  "chapter": 6,
  "allreduce_algorithm": "ring | tree | hierarchical",
  "gradient_compression": "none | fp16 | fp8",
  "crossover_msg_size_kb": 150,
  "allreduce_time_ms": 5800,
  "bandwidth_fraction_pct": 99.9
}
```

- `allreduce_algorithm` feeds forward to **Lab 07 (Fault Tolerance)**: determines checkpoint coordination strategy.
- `gradient_compression` feeds forward to **Lab 09 (Performance Engineering)**: affects the effective bandwidth in roofline analysis.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| 70B model gradient size = 280 GB FP32 | `@sec-communication-collective-operations-collective-operations-communication-fundamentals-44eb` | "70 billion parameter model...generates 140 GB of gradient data per worker per step" (280 GB in FP32 at 4 bytes/param) |
| Ring AllReduce bandwidth factor 2(N-1)/N | `@sec-communication-collective-operations-collective-operations-communication-fundamentals-44eb` | "ring AllReduce distributes the communication...each worker's per-step communication cost stays constant at 2 x (N-1)/N x gradient_size" |
| AllReduce total ~11,200 ms for 70B on 64 GPUs | AllReduce napkin math callout-notebook | "ring_total_ms approximately 11,200 ms" |
| InfiniBand NDR: alpha = 1--3 us, beta = 50 GB/s | `@tbl-interconnect-parameters` | "InfiniBand NDR 400G: Latency 1--3 us, Bandwidth 50 GB/s per port" |
| NVLink 4.0: alpha = 1--2 us, beta = 900 GB/s | `@tbl-interconnect-parameters` | "NVLink 4.0 (intra-node): Latency 1--2 us, Bandwidth 900 GB/s" |
| Critical message size n* = alpha * beta ~100 KB | `@sec-communication-collective-operations-collective-operations-alphabeta-model-f9b4` | "crossover size n* = alpha * beta approximately 100 KB" |
| Communication 25% at 8 GPUs, 65% at 4096 GPUs | `@fig-compute-comm-timeline` | "communication grows from approximately 25% to over 65% of each training step" |
| 30--70% of wall-clock time consumed by gradient sync | `@sec-communication-collective-operations-collective-operations-communication-fundamentals-44eb` | "gradient synchronization dominates...consuming 30--70% of wall-clock time" |
| Ring latency term: 2(N-1) * alpha | Ring AllReduce bandwidth formula | "T_latency = 2(N-1) * alpha" |

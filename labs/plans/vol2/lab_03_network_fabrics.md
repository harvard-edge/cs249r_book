# Mission Plan: lab_03_network_fabrics (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Network Fabrics (`@sec-network-fabrics`)
- **Core Invariant:** The **alpha-beta model** ($T(n) = \alpha + n/\beta$) governs all network communication, with the crossover point $n^* = \alpha \cdot \beta$ separating the latency-dominated regime (small messages, topology matters) from the bandwidth-dominated regime (large gradients, link speed matters). For InfiniBand NDR, $n^* \approx 75$ KB. Gradient AllReduce for large models is firmly bandwidth-dominated, but pipeline bubble signals and coordination messages are latency-dominated — optimizing the wrong regime wastes engineering effort.
- **Central Tension:** Students believe that upgrading from 200 Gbps to 400 Gbps InfiniBand will halve their AllReduce time for all workloads. The alpha-beta model reveals that for small messages (below the 75 KB crossover), the improvement is near zero because startup latency ($\alpha$) dominates, not bandwidth ($\beta$). Conversely, students underestimate the impact of oversubscription on large-message AllReduce: a 4:1 oversubscribed spine makes each gradient sync 4x slower, turning the network into the dominant bottleneck regardless of per-link speed.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict whether upgrading link speed from 200 Gbps to 400 Gbps will reduce transfer time for a 4 KB control message. The alpha-beta model reveals that both links take approximately the same time (~1.5 us) because the startup latency dominates. The aha moment: bandwidth upgrades are irrelevant in the latency-dominated regime. Students then observe the crossover point (~75 KB) where bandwidth begins to matter, calibrating their intuition for which optimization lever to pull.

**Act 2 (Design Challenge, 23 min):** Students configure a network topology for a 1,024-GPU AllReduce training cluster. They choose between a non-blocking fat-tree (1:1 subscription, full bisection bandwidth) and an oversubscribed fat-tree (4:1, quarter bisection bandwidth), then discover that the oversubscribed fabric makes each AllReduce step 4x slower for a 70B model. The design challenge requires finding the maximum oversubscription ratio that keeps AllReduce communication below 30% of step time.

---

## 3. Act 1: The Alpha-Beta Crossover (Calibration -- 12 minutes)

### Pedagogical Goal

Students assume that faster links always mean faster communication. The alpha-beta model ($T(n) = \alpha + n/\beta$) shows that two regimes exist: below the crossover $n^* = \alpha \cdot \beta$, latency ($\alpha$) dominates and faster bandwidth provides no benefit; above $n^*$, bandwidth ($\beta$) dominates and link speed is decisive. For NDR InfiniBand ($\alpha \approx 1.5$ us, $\beta = 50$ GB/s), the crossover is approximately 75 KB. Pipeline coordination messages (hundreds of bytes to a few KB) gain nothing from bandwidth upgrades; gradient AllReduce (hundreds of MB to GB) gains linearly.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "You upgrade your cluster's InfiniBand from HDR (200 Gbps, ~25 GB/s) to NDR (400 Gbps, ~50 GB/s). You send a 4 KB pipeline coordination message. By how much does the transfer time improve?"

Options:
- A) 2x faster -- double the bandwidth, half the time
- B) 1.5x faster -- some improvement, less than linear
- **C) Essentially no improvement -- the transfer time is dominated by startup latency, not bandwidth** ← correct
- D) Slightly slower -- the NDR protocol adds overhead that exceeds the bandwidth gain

The answer is C because $T(4\text{ KB}) = 1.5\text{ us} + 4\text{ KB} / 25\text{ GB/s} = 1.5\text{ us} + 0.16\text{ us} = 1.66\text{ us}$ (HDR) vs. $1.5\text{ us} + 0.08\text{ us} = 1.58\text{ us}$ (NDR). The improvement is 5%, not 50%. Students who pick A are applying bandwidth-regime reasoning to a latency-regime problem.

### The Instrument: Alpha-Beta Regime Visualizer

A **log-log line chart** showing transfer time vs. message size for two fabric types:

- **X-axis:** Message size (1 byte -- 10 GB, logarithmic)
- **Y-axis:** Transfer time (nanoseconds -- seconds, logarithmic)
- **Line 1 (BlueLine):** InfiniBand NDR ($\alpha = 1.5$ us, $\beta = 50$ GB/s)
- **Line 2 (OrangeLine):** RoCE/Ethernet ($\alpha = 5.0$ us, $\beta = 12.5$ GB/s)
- **Vertical marker:** Crossover point $n^* = \alpha \cdot \beta \approx 75$ KB (IB) annotated
- **Shaded regions:** Left = "Latency Dominated" (gray), Right = "Bandwidth Dominated" (white)

Controls:
- **Fabric selector**: InfiniBand NDR / InfiniBand HDR / 100G Ethernet (changes $\alpha$ and $\beta$)
- **Message size slider** (1 byte -- 10 GB, log scale, default: 4 KB): A vertical line tracks the current message size across both curves, showing the absolute transfer time and the gap between fabrics.

Annotations at key message sizes:
- 4 KB (pipeline signal): "Both fabrics ~1.5 us. Bandwidth irrelevant."
- 100 MB (gradient shard): "IB NDR: 2 ms. Ethernet: 8 ms. Bandwidth matters."
- 700 GB (full AllReduce): "IB NDR: 14 s. Ethernet: 56 s. Bandwidth decisive."

### The Reveal

After interaction:

> "You predicted [X]x improvement for a 4 KB message. The actual improvement is **5%** (1.66 us to 1.58 us). The bandwidth upgrade is invisible because startup latency ($\alpha = 1.5$ us) dominates. For a 100 MB gradient shard, the same upgrade delivers a **2x improvement** (4 ms to 2 ms) because the bandwidth term dominates. The crossover point is **~75 KB**: below this, optimize for topology (fewer hops reduce $\alpha$); above this, optimize for link speed (more bandwidth reduces $n/\beta$)."

### Reflection (Structured)

Four-option multiple choice:

> "Pipeline Parallelism sends small activation tensors (~200 MB) between stages, while Data Parallelism synchronizes large gradient vectors (~700 GB for 175B params). Which optimization strategy is correct for each?"

- A) Both need faster link speed (higher $\beta$) since both are above the crossover
- **B) Pipeline activations (200 MB) need bandwidth; gradient AllReduce (700 GB) needs even more bandwidth. Both are bandwidth-dominated, but the gradient case benefits from topology with more parallel paths** ← correct
- C) Pipeline signals need fewer hops (lower $\alpha$); gradient AllReduce needs faster links (higher $\beta$). Each is in a different regime
- D) Both need lower latency since AllReduce has 2(N-1) sequential phases

### Math Peek (collapsible)

$$T(n) = \alpha + \frac{n}{\beta}$$

Crossover: $n^* = \alpha \cdot \beta$

For InfiniBand NDR: $\alpha = 1.5 \text{ us}$, $\beta = 50 \text{ GB/s}$
$$n^* = 1.5 \times 10^{-6} \times 50 \times 10^9 = 75{,}000 \text{ bytes} \approx 75 \text{ KB}$$

Ring AllReduce cost: $T_{\text{ring}} = 2(p-1)\alpha + \frac{2(p-1)}{p} \cdot \frac{m}{\beta}$

---

## 4. Act 2: The Oversubscription Tax (Design Challenge -- 23 minutes)

### Pedagogical Goal

Students believe that a "faster switch" automatically means faster training. The chapter shows that the network topology — specifically the **bisection bandwidth** and **oversubscription ratio** — determines the effective bandwidth available to global collectives like AllReduce. A 1,024-GPU non-blocking fat-tree with 400 Gbps links provides 25.6 TB/s of bisection bandwidth; a 4:1 oversubscribed spine reduces this to 6.4 TB/s, making each AllReduce step 4x slower. Students must find the oversubscription ratio that keeps training efficient for their model size.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "A 1,024-GPU fat-tree cluster uses 400 Gbps InfiniBand links. The spine layer is oversubscribed 4:1 (each pod switch has 4x more downlinks than uplinks). You run AllReduce for a 70B model (280 GB of FP32 gradients). How long does AllReduce take compared to a non-blocking (1:1) fat-tree?"

Students type a multiplier (1x -- 10x). Expected wrong answers: 1.5--2x (students expect some degradation but not proportional). Actual: **~4x slower** because bisection bandwidth drops proportionally with oversubscription ratio, and AllReduce is a global communication pattern that must cross the bisection.

### The Instrument: Topology Bisection Bandwidth Comparator

A **comparison panel** with two topology diagrams and associated performance metrics:

**Top: Topology Selector**
- Fat-tree 1:1 (non-blocking)
- Fat-tree 2:1 (oversubscribed)
- Fat-tree 4:1 (heavily oversubscribed)
- Rail-optimized (for comparison)

**Middle: Simplified topology diagram** showing switch tiers (leaf, spine, core) with link counts. Oversubscribed tiers are highlighted in OrangeLine.

**Bottom: AllReduce Performance Chart**
- **X-axis:** Model size (1B -- 175B parameters)
- **Y-axis:** AllReduce time (ms, log scale)
- **Lines:** One per topology, colored by subscription ratio
- **Threshold line (RedLine):** "30% of step time" — when AllReduce exceeds this, the network is the bottleneck

Controls:
- **Topology selector** (fat-tree 1:1 / 2:1 / 4:1 / rail-optimized, default: 1:1)
- **Cluster size selector** (256 / 512 / 1024 / 4096 GPUs, default: 1024)
- **Model size slider** (1B -- 175B, step: powers of ~2, default: 70B)
- **GPU utilization slider** (30% -- 70%, step: 5%, default: 50%): Affects $T_{\text{compute}}$ and thus the "30% threshold" line

### The Scaling Challenge

**"Find the maximum oversubscription ratio that keeps AllReduce below 30% of step time for a 70B model on 1,024 GPUs."**

Students must discover:
- At 1:1 (non-blocking): AllReduce ~5% of step time. Efficient but expensive ($20--100M in switches).
- At 2:1: AllReduce ~10% of step time. Acceptable trade-off for most workloads.
- At 4:1: AllReduce ~20% of step time. Marginal for 70B models.
- At 4:1 with 175B model: AllReduce >50% of step time. Network-bound.

The catch: rail-optimized topology (designed for AllReduce traffic patterns) achieves near-1:1 performance at 2:1 cost by aligning network paths with the collective communication pattern.

### The Failure State

**Trigger condition:** `allreduce_time > 0.50 * step_time` (AllReduce consumes more than half the step)

**Visual change:** The AllReduce line on the chart crosses above the threshold and turns RedLine. The topology diagram highlights the oversubscribed spine tier in red. A congestion overlay shows packet queuing at the spine switches.

**Banner text:**
> "**Bisection Bandwidth Exhausted -- AllReduce Saturated.** At [X]:1 oversubscription with [Y]B parameters, AllReduce takes [Z] ms vs [W] ms compute. The spine switches are the bottleneck. Options: (1) Reduce oversubscription ratio, (2) Use gradient compression to reduce AllReduce volume, (3) Switch to pipeline parallelism to avoid global AllReduce."

### Structured Reflection

Four-option multiple choice:

> "A fat-tree built from radix-64 switches supports $k^3/4 = 65{,}536$ hosts with full bisection bandwidth. The chapter states that a 4,096-GPU non-blocking fat-tree needs ~2,048 switches costing $20--100M. An engineer proposes using 4:1 oversubscription to cut switch costs by 75%. What is the hidden cost?"

- A) No hidden cost -- oversubscription is a standard technique with no performance impact
- B) Latency increases by 4x for all messages, including small coordination signals
- **C) Bisection bandwidth drops to 1/4, making every global AllReduce 4x slower and potentially turning the network into the dominant training bottleneck** ← correct
- D) The fabric becomes blocking, causing packet drops that corrupt gradient synchronization

### Math Peek (collapsible)

Fat-tree host count: $\frac{k^3}{4}$ for radix-$k$ switches

Bisection bandwidth (1:1): $\frac{k}{2} \times \frac{k}{2} \times BW_{\text{link}} = \frac{k^2}{4} \times BW_{\text{link}}$

With oversubscription ratio $r$: $BW_{\text{bisect}} = \frac{1}{r} \times BW_{\text{bisect, 1:1}}$

AllReduce time: $T_{\text{AR}} \propto \frac{M}{BW_{\text{bisect}}} \propto r$

---

## 5. Visual Layout Specification

### Act 1: Alpha-Beta Regime Chart
- **Chart type:** Log-log line chart
- **X-axis:** Message size (1 B -- 10 GB, log)
- **Y-axis:** Transfer time (1 ns -- 100 s, log)
- **Data series:** Two lines (IB NDR, Ethernet); shaded latency/bandwidth regions
- **Annotations:** Crossover point with value (~75 KB); example message sizes (4 KB, 100 MB, 700 GB)
- **Failure state:** N/A (Act 1)

### Act 2: Topology Bisection Bandwidth Comparator
- **Chart type:** Multi-line chart with threshold
- **X-axis:** Model size (1B -- 175B, log)
- **Y-axis:** AllReduce time (ms, log)
- **Data series:** One line per topology/subscription ratio
- **Threshold line:** "30% of step time" (RedLine, dashed)
- **Failure state:** When AllReduce exceeds 50% of step time, line turns RedLine, topology diagram highlights congested spine

### Act 2: Topology Diagram
- **Type:** Simplified switch-tier diagram (leaf -- spine -- core)
- **Updates on:** Topology selector and cluster size
- **Oversubscribed tiers:** Highlighted in OrangeLine
- **Failure state:** Spine tier turns RedLine when saturated

---

## 6. Deployment Context Definitions

| Context | Configuration | Bisection BW | Switches | Key Constraint |
|---|---|---|---|---|
| **8-GPU pod (non-blocking)** | Single DGX H100 node, NVLink | 900 GB/s (NVLink mesh) | 0 (direct links) | All communication is intra-node; no topology concerns; TP=8 works perfectly |
| **1,024-GPU cluster (oversubscribed)** | 128 DGX nodes, 3-tier fat-tree, 4:1 spine | ~6.4 TB/s (1/4 of non-blocking) | ~512 switches | Bisection bandwidth is the ceiling for AllReduce; oversubscription directly multiplies gradient sync time |

The two contexts demonstrate that network topology is invisible at small scale (8 GPUs share a backplane) but becomes the dominant engineering decision at fleet scale. The same AllReduce algorithm produces 4x different performance depending on the spine oversubscription ratio.

---

## 7. Design Ledger Output

```json
{
  "chapter": "v2_03",
  "topology_chosen": "fat_tree_2_1",
  "oversubscription_ratio": 2,
  "cluster_size_gpus": 1024,
  "bisection_bw_tbs": 12.8,
  "allreduce_fraction_pct": 10,
  "crossover_point_kb": 75
}
```

The `topology_chosen` and `bisection_bw_tbs` fields feed forward to:
- **Lab V2-04 (Data Storage):** The network bandwidth constrains checkpoint write throughput for distributed checkpoints.
- **Lab V2-05 (Distributed Training):** The bisection bandwidth sets the ceiling for gradient synchronization in the 3D parallelism configurator.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Alpha-beta model $T(n) = \alpha + n/\beta$ | @sec-network-fabrics-performance-model, line 506 | "$T(n) = \alpha + n/\beta$" |
| Crossover point ~75 KB for IB NDR | @sec-network-fabrics-performance-model, line 574 | "crossover point $n = \alpha\beta \approx$ 75 KB" |
| IB NDR: $\alpha \approx 1.5$ us, $\beta \approx 50$ GB/s | @sec-network-fabrics-performance-model, lines 538--549 | Alpha and beta values from NetworkAlphaBeta LEGO |
| Bandwidth-dominated regime for gradients | @sec-network-fabrics-performance-model, line 593 | "Since gradients are megabytes to gigabytes, we are almost always in the bandwidth-dominated regime" |
| Fat-tree supports $k^3/4$ hosts | @sec-network-fabrics-fat-tree, line 847 | "A fat-tree built from radix-$k$ switches supports $k^3/4$ hosts" |
| 4,096-GPU fat-tree needs ~2,048 switches, $20--100M | @sec-network-fabrics-fat-tree, line 849 | "a 4,096-GPU non-blocking fat-tree needs roughly 2,048 switches, costing $20--100 million" |
| 1:1 bisection: 512 x 50 GB/s = 25.6 TB/s | @sec-network-fabrics, bisection bandwidth definition, line 803 | "512 x 50 GB/s = 25.6 TB/s of bisection bandwidth" |
| 4:1 oversubscription reduces to 6.4 TB/s | @sec-network-fabrics, bisection bandwidth definition, line 803 | "a 4:1 oversubscribed spine reduces this to 6.4 TB/s" |
| FEC adds 100--200 ns per hop | @sec-network-fabrics, fn-fec-latency, line 410 | "FEC...adds 100--200 ns per hop" |
| 70B model: AllReduce comm dominates by 3x at 1024 GPUs | @nb-allreduce-bottleneck, line 789 | "communication dominates computation by nearly 3x" |

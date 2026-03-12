# Mission Plan: lab_02_compute_infra (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Compute Infrastructure (`@sec-compute-infrastructure`)
- **Core Invariant:** The **Bandwidth Hierarchy** dictates parallelism strategy — each physical boundary (die, package, node, rack) introduces an order-of-magnitude bandwidth cliff that constrains which communication patterns are feasible. NVLink at 900 GB/s confines Tensor Parallelism to intra-node; InfiniBand at 50 GB/s relegates Data Parallelism to inter-node. The 18x gap between NVLink and InfiniBand is the single most important number in distributed training system design.
- **Central Tension:** Students believe that all GPUs in a cluster communicate at roughly the same speed — "they're all connected." The chapter's bandwidth staircase reveals that transferring 10 GB takes 11 ms within a node (NVLink) but 200 ms between nodes (InfiniBand), an 18x difference. This cliff is not an engineering failure but a physical consequence of signal propagation at different distance scales. Students who ignore the hierarchy will design parallelism strategies that idle 90% of their silicon.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict how long it takes to transfer a 10 GB buffer at each tier of the bandwidth hierarchy. They expect roughly similar transfer times across tiers, perhaps 2--3x differences. The instrument reveals a staircase of order-of-magnitude cliffs: 3 ms (HBM), 11 ms (NVLink), 200 ms (InfiniBand). The 18x gap between NVLink and InfiniBand is the aha moment — it explains why tensor parallelism is confined to within a single node and cannot simply "extend" across the cluster.

**Act 2 (Design Challenge, 23 min):** Students must partition a 175B-parameter model across 8 GPUs within a single node (NVLink) vs. 2 nodes of 8 GPUs each (Ethernet/InfiniBand inter-node link). They configure tensor parallelism degree and data parallelism degree, discovering that placing TP across the inter-node boundary collapses utilization because the per-layer AllReduce cannot complete within the compute window. The design challenge requires finding the maximum model size that achieves >80% efficiency on each topology.

---

## 3. Act 1: The Bandwidth Staircase (Calibration -- 12 minutes)

### Pedagogical Goal

Students treat cluster communication as uniform — "all the GPUs are connected." The chapter's bandwidth hierarchy shows this is catastrophically wrong. HBM delivers 3,350 GB/s, NVLink delivers 900 GB/s, and InfiniBand NDR delivers 50 GB/s. These are not incremental differences; they are order-of-magnitude cliffs driven by the physics of signal propagation at different distance scales (millimeters for HBM, centimeters for NVLink, meters for InfiniBand). This act makes the staircase visceral by showing transfer times for the same 10 GB buffer at each tier.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "You need to transfer a 10 GB tensor between two GPUs. Within a DGX H100 node, the transfer uses NVLink (900 GB/s). Between two nodes in the same rack, it uses InfiniBand NDR (50 GB/s). How much slower is the inter-node transfer compared to the intra-node transfer?"

Options:
- A) About 2x slower -- networking adds some overhead but modern fabrics are fast
- B) About 5x slower -- crossing the node boundary is noticeable but manageable
- **C) About 18x slower -- each physical boundary introduces an order-of-magnitude cliff** ← correct
- D) About 100x slower -- inter-node communication is essentially a different world

The answer is 18x because NVLink delivers 900 GB/s and InfiniBand NDR delivers 50 GB/s. Students who pick A or B are thinking about latency differences (microseconds) rather than bandwidth differences (GB/s). Students who pick D are confusing inter-node with WAN.

### The Instrument: Bandwidth Staircase Visualization

A **horizontal bar chart** showing transfer time for a 10 GB buffer at each tier:

- **Y-axis (categories):** HBM (Intra-Chip), NVLink (Intra-Node), PCIe Gen5 (CPU-GPU), InfiniBand NDR (Inter-Node)
- **X-axis:** Transfer time (ms, logarithmic scale)
- **Bars:** Each bar shows $T = \text{Data} / \text{BW}$ for 10 GB

Controls:
- **Transfer size slider** (1 MB -- 100 GB, log scale, step: powers of 2, default: 10 GB): Adjusts the data volume; all bars update proportionally.
- **Tier highlight toggle**: Click any bar to see the bandwidth value, latency, and physical description (e.g., "copper traces on silicon interposer, millimeter path lengths").

Reference values from the chapter:
- HBM: 10 GB / 3,350 GB/s = 3.0 ms
- NVLink: 10 GB / 900 GB/s = 11.1 ms
- PCIe Gen5: 10 GB / 64 GB/s = 156 ms
- InfiniBand NDR: 10 GB / 50 GB/s = 200 ms

### The Reveal

After interaction:

> "You predicted [X]x slower for inter-node vs intra-node. The actual gap is **18x** (NVLink 900 GB/s vs InfiniBand 50 GB/s). A 10 GB transfer takes 11 ms within the node but 200 ms between nodes. This is why Tensor Parallelism — which requires AllReduce after EVERY layer — is confined to the NVLink domain. Placing TP across the InfiniBand boundary would leave GPUs idle for 95% of each step."

### Reflection (Structured)

Four-option multiple choice:

> "The chapter states that the bandwidth hierarchy 'is the physical law that determines the topology of distributed training.' Given the 18x gap between NVLink and InfiniBand, which parallelism strategy is placed on the inter-node (InfiniBand) fabric?"

- A) Tensor Parallelism -- it splits individual matrix multiplications and needs per-layer sync
- **B) Data Parallelism -- it synchronizes gradients only once per step, tolerating lower bandwidth** ← correct
- C) Pipeline Parallelism -- it transfers full model copies between stages after each step
- D) All strategies perform equally because modern frameworks hide communication behind computation

### Math Peek (collapsible)

$$T_{\text{transfer}} = \frac{\text{Data (bytes)}}{\text{Bandwidth (bytes/sec)}}$$

| Tier | Bandwidth | 10 GB Transfer |
|------|-----------|----------------|
| HBM (H100) | 3,350 GB/s | 3.0 ms |
| NVLink 4.0 | 900 GB/s | 11.1 ms |
| PCIe Gen5 | 64 GB/s | 156 ms |
| InfiniBand NDR | 50 GB/s | 200 ms |

Gap: NVLink / InfiniBand = $900 / 50 = 18\times$

---

## 4. Act 2: The Parallelism Boundary (Design Challenge -- 23 minutes)

### Pedagogical Goal

Students believe that parallelism strategies can be applied uniformly across a cluster. The chapter shows that the bandwidth hierarchy creates hard boundaries: Tensor Parallelism within the node (NVLink), Pipeline Parallelism across nearby nodes (InfiniBand), and Data Parallelism across the full cluster. Violating these boundaries (e.g., placing TP on an inter-node link) collapses utilization because the per-layer AllReduce at 50 GB/s cannot complete within the compute window of a single layer. Students must find the correct parallelism-to-interconnect mapping that keeps efficiency above 80%.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "You place Tensor Parallelism (TP=8) across 8 GPUs connected by InfiniBand NDR (50 GB/s) instead of NVLink (900 GB/s). Each TP AllReduce exchanges a tensor of ~200 MB, 96 times per forward+backward pass (one per transformer layer). What fraction of step time will be spent on communication?"

Students type a percentage (0--100%). Expected wrong answers: 10--30% (students underestimate). Actual: approximately **85%** because $96 \times (200\text{ MB} / 50\text{ GB/s}) = 96 \times 4\text{ ms} = 384\text{ ms}$ of communication vs ~67 ms of compute (for a single layer at ~0.7 ms x 96 layers). On NVLink: $96 \times (200\text{ MB} / 900\text{ GB/s}) = 96 \times 0.22\text{ ms} = 21\text{ ms}$.

### The Instrument: Parallelism-Topology Configurator

A **split-panel comparison** showing single-node (NVLink) vs. multi-node (Ethernet/InfiniBand):

**Left panel (Single Node):**
- 8 GPU icons connected by NVLink lines (900 GB/s)
- TP degree slider (1, 2, 4, 8; default: 8)
- Shows per-step compute vs communication time

**Right panel (Multi-Node):**
- Two groups of 4 GPUs, connected by inter-node link
- Inter-node bandwidth toggle: 100G Ethernet (12.5 GB/s) / InfiniBand NDR (50 GB/s)
- TP degree slider (1, 2, 4, 8; default: 8)
- DP degree computed as complement

Controls:
- **Model size selector** (7B / 70B / 175B, default: 70B): Changes hidden dimension (4096 / 8192 / 12288) and activation tensor size
- **TP degree** (1 / 2 / 4 / 8, default: 8): Degree of tensor parallelism
- **Inter-node link toggle**: Ethernet (12.5 GB/s) / InfiniBand NDR (50 GB/s)
- **Layers slider** (24 / 48 / 96, default: 96): Number of transformer layers

**Output: Step Timeline Waterfall**
- Horizontal bars showing compute time vs. communication time per step
- Efficiency percentage prominently displayed
- Color: BlueLine for compute, OrangeLine for communication
- When communication > compute, communication bar turns RedLine

### The Scaling Challenge

**"Find the parallelism configuration that achieves >80% efficiency for a 70B model across 2 DGX H100 nodes (16 GPUs total) connected by InfiniBand NDR."**

Students must discover the correct hierarchical mapping:
- TP=8 within each node (uses NVLink 900 GB/s) -- per-layer AllReduce completes in ~0.2 ms
- DP=2 across nodes (uses InfiniBand 50 GB/s) -- gradient sync once per step
- This configuration achieves ~85% efficiency
- Placing TP=16 across both nodes collapses efficiency to ~15%

### The Failure State

**Trigger condition:** `communication_fraction > 0.80` (communication consumes more than 80% of step time)

**Visual change:** The communication bar in the timeline turns RedLine. The efficiency gauge drops into the red zone. The GPU icons in the affected panel show an "IDLE" overlay on 7 of 8 GPUs.

**Banner text:**
> "**Interconnect Violation -- TP Across Node Boundary.** Tensor Parallelism requires [X] AllReduce operations per step (one per layer). At InfiniBand bandwidth ([Y] GB/s), each operation takes [Z] ms. Total communication: [W] ms vs [V] ms compute. GPUs are idle [P]% of the time. Move TP within the NVLink domain."

### Structured Reflection

Four-option multiple choice:

> "Meta's LLaMA training, Google's PaLM training, and OpenAI's GPT-4 training all use the same hierarchical parallelism assignment. Why is this mapping (TP intra-node, PP inter-rack, DP cluster-wide) universal rather than workload-specific?"

- A) It is arbitrary convention adopted by the first successful distributed training paper
- B) The software frameworks only support this specific configuration
- **C) The bandwidth hierarchy is a physical law — the 18x gap between NVLink and InfiniBand forces TP to the fastest tier regardless of the model** ← correct
- D) Models are designed to match this parallelism pattern from the architecture stage

### Math Peek (collapsible)

Tensor Parallelism communication per layer:
$$T_{\text{TP-comm}} = \frac{2 \times B \times S \times H \times 2 \text{ (FP16)}}{\text{BW}}$$

For hidden dim H=8192, microbatch B=4, seq S=2048:
$$\text{Activation} = 4 \times 2048 \times 8192 \times 2 \approx 134 \text{ MB}$$

Per-step communication (96 layers, 2 AllReduces each):
$$T_{\text{total-comm}} = 192 \times \frac{134 \text{ MB}}{\text{BW}}$$

- NVLink: $192 \times 0.15 \text{ ms} = 29 \text{ ms}$
- InfiniBand: $192 \times 2.7 \text{ ms} = 518 \text{ ms}$

---

## 5. Visual Layout Specification

### Act 1: Bandwidth Staircase
- **Chart type:** Horizontal bar chart (log x-axis)
- **Y-axis:** Four bandwidth tiers (HBM, NVLink, PCIe, InfiniBand)
- **X-axis:** Transfer time (ms, log scale, 0.1 ms -- 10,000 ms)
- **Data series:** One bar per tier; color encodes tier (darker = faster)
- **Annotations:** Bandwidth value (GB/s) and ratio to previous tier on each bar
- **Failure state:** N/A (Act 1)

### Act 2: Step Timeline Waterfall (Left: NVLink, Right: InfiniBand)
- **Chart type:** Stacked horizontal bar (two panels side by side)
- **X-axis:** Time within one training step (ms)
- **Segments:** Compute (BlueLine), TP AllReduce (OrangeLine), DP AllReduce (GreenLine)
- **Annotations:** Efficiency % badge on each panel
- **Failure state:** When comm > 80% of step, communication segment turns RedLine; IDLE overlays on GPU icons

### Act 2: Efficiency Gauge
- **Chart type:** Semicircular gauge
- **Range:** 0% -- 100%
- **Zones:** 0--50% red, 50--80% orange, 80--100% green
- **Updates on:** Any control change

---

## 6. Deployment Context Definitions

| Context | Device | Interconnect | Bandwidth | Key Constraint |
|---|---|---|---|---|
| **Single Node (8 GPU, NVLink)** | 1x DGX H100 | NVLink 4.0 | 900 GB/s GPU-to-GPU | TP works perfectly; limited to 8-way parallelism; model must fit in 8x 80 GB = 640 GB aggregate HBM |
| **Multi-Node (2 nodes, InfiniBand)** | 2x DGX H100 | InfiniBand NDR | 50 GB/s inter-node | TP across node boundary collapses utilization; must use hierarchical TP-intra / DP-inter mapping |

The two contexts isolate the bandwidth cliff. The same model, same code, same number of GPUs (8 vs. 8+8) behaves fundamentally differently depending on whether parallelism respects or violates the node boundary. This is the defining insight of the compute infrastructure chapter.

---

## 7. Design Ledger Output

```json
{
  "chapter": "v2_02",
  "tp_degree_chosen": 8,
  "dp_degree_chosen": 2,
  "inter_node_bandwidth_gbs": 50,
  "efficiency_single_node_pct": 97,
  "efficiency_multi_node_pct": 85,
  "max_model_size_single_node_b": 70,
  "bandwidth_gap_nvlink_ib": 18
}
```

The `tp_degree_chosen` and `bandwidth_gap_nvlink_ib` fields feed forward to:
- **Lab V2-03 (Network Fabrics):** The inter-node bandwidth sets the starting point for topology analysis.
- **Lab V2-05 (Distributed Training):** The TP/DP split informs the 3D parallelism configurator defaults.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| NVLink 900 GB/s per GPU (H100) | @sec-compute-infrastructure, line 281 | "Hopper also introduced NVLink 4.0 at 900 GB/s per GPU" |
| InfiniBand NDR 50 GB/s | @tbl-bandwidth-hierarchy-compute | "~50 GB/s" for Inter-Node InfiniBand NDR |
| PCIe Gen5 64 GB/s | @tbl-bandwidth-hierarchy-compute | "~64 GB/s" for PCIe Gen5 x16 |
| HBM3 3,350 GB/s (H100) | @sec-compute-infrastructure, line 500 | "HBM3 in the H100 delivers approximately 3.35 TB/s of memory bandwidth" |
| 18x gap NVLink-to-InfiniBand | @sec-compute-infrastructure, line 1559 | "check(gap_nv_ib == 18.0)" in BandwidthStaircase LEGO |
| 10 GB transfer: 3 ms HBM, 11 ms NVLink, 200 ms IB | @sec-compute-infrastructure, lines 1576--1578 | "3 ms (HBM), 11 ms (NVLink), 200 ms (InfiniBand)" |
| TP confined to intra-node | @sec-compute-infrastructure, line 1606 | "Tensor parallelism...generates communication volume...hundreds of times per second" |
| Order-of-magnitude cliff per boundary | @sec-compute-infrastructure, line 1588 | "Effective Bandwidth drops by approximately one order of magnitude" |
| Activation tensor ~200 MB per stage boundary | @sec-compute-infrastructure, line 1612 | "activation tensor...approximately 200 MB" |
| Blackwell NVLink 1,800 GB/s | @sec-compute-infrastructure, line 316 | "fifth-generation NVLink at 1,800 GB/s per GPU" |

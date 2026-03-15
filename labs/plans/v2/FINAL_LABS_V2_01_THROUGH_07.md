# Volume 2 Labs: Final Cleaned Plans (V2-01 through V2-07)

Generated: 2026-03-15
Status: FINAL -- ready for lab-developer handoff
Numbering: Post-merger (old V2-03 + V2-06 merged into new V2-03; old V2-07 becomes V2-06; old V2-08 becomes V2-07)

---

## Lab V2-01: The Scale Illusion

**Story arc**: Students arrive believing that distributed training is "just more of the same" -- that 1,000 GPUs deliver 1,000x speedup and that hardware reliability at scale is a non-issue. Over five parts, they watch reliability collapse exponentially, discover that communication devours compute gains, learn that scaling laws punish naive resource allocation, confront Amdahl's Law with real communication overhead, and finally classify workloads by their dominant bottleneck. They leave understanding that scale creates qualitative change, not just quantitative increase.

**Time budget**: 55 min (12 + 12 + 10 + 12 + 8 = 54 min + 1 min transitions)

---

### Part A -- The Reliability Collapse (~12 min)

**Concept**: Fleet-wide availability decays exponentially with fleet size. A 1,000-GPU cluster with 99.9% per-node reliability is healthy only 36.8% of the time. At GPT-4 scale (25,000 GPUs), a hardware failure occurs every ~4.4 hours. Failure is the common case, not the exception.

**Prediction**: "Your cluster has 1,000 GPUs, each with 99.9% individual uptime. What fraction of the time is the entire cluster healthy?"

| Option | Value |
|--------|-------|
| A | ~99% -- nearly always healthy |
| B | ~90% -- healthy most of the time |
| C | ~60% -- healthy more often than not |
| **D (correct)** | **~37% -- healthy barely a third of the time** |

**Common wrong answer**: A or B. Students anchor on the per-node reliability (99.9%) and assume fleet reliability degrades linearly or gently.

**Why wrong**: The exponential in P_fleet = (P_node)^N makes even tiny per-node failure rates catastrophic at scale. (0.999)^1000 = 0.368.

**Instrument**:
- Slider 1: Fleet size N (1 to 25,000, step 100, default 1,000)
- Slider 2: Per-node reliability P_node (0.990 to 0.9999, step 0.001, default 0.999)
- Chart: Fleet availability vs. fleet size (line chart, with reference line at P_node = 0.9999)
- Metric row: Fleet availability %, MTBF (= GPU_MTTF_HOURS / N), failures per day

**mlsysim grounding**: `calc_failure_probability(mtbf, job_duration)` from `mlsysim.core.formulas`; `GPU_MTTF_HOURS` from `mlsysim.core.defaults` (50,000 hours). `calc_mtbf_cluster(GPU_MTTF_HOURS, N)` gives system MTBF.

**Transition to B**: "So the cluster fails every few hours. But when it is running, surely 1,000 GPUs give you 1,000x speedup? Let us check what happens to your training step time as you scale."

---

### Part B -- The Coordination Tax (~12 min)

**Concept**: The Fleet Law decomposes distributed step time into Compute + Communication + Coordination. The "Conservation of Overhead" means you cannot eliminate overhead, only redistribute it. At 1,000 GPUs, communication can consume 40% or more of step time, capping effective speedup far below 1,000x.

**Prediction**: "You scale a 175B model training job from 1 GPU to 256 GPUs on InfiniBand NDR. What fleet efficiency do you expect?"

| Option | Value |
|--------|-------|
| A | ~95% -- InfiniBand is fast enough |
| B | ~80% -- some communication overhead |
| **C (correct)** | **~55-65% -- communication is substantial** |
| D | ~30% -- communication dominates |

**Common wrong answer**: A or B. Students overestimate InfiniBand's ability to hide 175B-parameter gradient synchronization.

**Why wrong**: Ring AllReduce for 175B FP16 parameters transfers ~700 GB per step. Even at 50 GB/s (IB NDR), this takes ~14 seconds, a significant fraction of compute time.

**Instrument**:
- Slider 1: Number of GPUs (1 to 1,024, powers of 2)
- Slider 2: Model gradient size (select: 1B/7B/70B/175B, maps to bytes)
- Toggle: Network type (IB NDR 50 GB/s vs. IB HDR 25 GB/s vs. 100GbE 12.5 GB/s)
- Chart: Stacked bar of T_compute, T_communication, T_coordination per step
- Gauge: Fleet efficiency eta = T_compute / T_step

**mlsysim grounding**: `calc_ring_allreduce_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s)` from `mlsysim.core.formulas`. Fleet defined using `Fabrics.InfiniBand_NDR` (bandwidth=400 Gbps, latency=5 us) and `Nodes.DGX_H100` from `mlsysim.systems.registry`.

**Transition to C**: "Communication is expensive. But before you can even estimate costs, you need to know how much compute to buy. Scaling laws tell you the optimal allocation of compute between model size and data -- get it wrong and you waste millions."

---

### Part C -- The Scaling Law Budget Planner (~10 min)

**Concept**: Compute-optimal resource allocation (Chinchilla scaling) requires coordinated scaling of model size and dataset size. Scaling one dimension alone wastes resources. The optimal ratio is approximately D = 20P (20 tokens per parameter).

**Prediction**: "You have a fixed compute budget of 10^23 FLOPs. Which achieves lower loss: a 10B model trained on 200B tokens, or a 3B model trained on 600B tokens?"

| Option | Value |
|--------|-------|
| A | 10B on 200B tokens -- bigger models are always better |
| **B (correct)** | **3B on 600B tokens -- balanced allocation wins** |
| C | Both achieve the same loss -- total FLOPs is what matters |
| D | Neither -- you need at least 70B parameters |

**Common wrong answer**: A. Students assume that model size is the dominant factor in capability.

**Why wrong**: The Chinchilla scaling law shows that for a fixed compute budget, there is a unique optimal (P, D) pair. Over-allocating to model size under-trains the model; over-allocating to tokens under-parameterizes it.

**Instrument**:
- Slider 1: Model parameters P (1B to 100B, log scale)
- Slider 2: Training tokens D (10B to 10T, log scale)
- Constraint indicator: C = 6PD FLOPs (shows current compute use vs. budget)
- Chart: IsoFLOP loss curves with current allocation marked
- Metric: Distance from Chinchilla optimal point

**mlsysim grounding**: `CHINCHILLA_TOKENS_PER_PARAM` (value: 20) and `CHINCHILLA_COMPUTE_CONSTANT` (value: 6) from `mlsysim.core.defaults`. Loss approximation uses the Hoffmann et al. parametric form.

**Transition to D**: "Now you know how much compute you need. But how many GPUs should you buy? At some point, adding GPUs costs more than the speedup is worth. Let us find that point."

---

### Part D -- The Iron Law of Scale (~12 min)

**Concept**: Distributed training speedup is limited by an extended Amdahl's Law where the serial fraction includes communication overhead. Beyond a critical GPU count, adding hardware reduces cost-efficiency. The communication fraction r determines where the speedup curve bends.

**Prediction**: "For a workload with 20% communication overhead (r = 0.20), how many GPUs does it take before scaling efficiency drops below 50%?"

| Option | Value |
|--------|-------|
| A | ~512 GPUs -- efficiency holds a long time |
| B | ~128 GPUs -- moderate scale |
| **C (correct)** | **~32-64 GPUs -- surprisingly few** |
| D | ~8 GPUs -- almost immediately |

**Common wrong answer**: A or B. Students overestimate how far linear scaling extends.

**Why wrong**: At r = 0.20, the communication term grows linearly with N while compute per GPU shrinks as 1/N. The crossover arrives faster than intuition suggests. By 64 GPUs, the communication term consumes half of step time.

**Instrument**:
- Slider 1: Communication fraction r (0.01 to 0.50)
- Slider 2: Number of GPUs (1 to 512, log scale)
- Slider 3: Overlap percentage (0% to 80%)
- Chart: Log-log speedup chart (ideal linear vs. actual), echoing @fig-scaling-tax
- Chart 2: Cost per sample vs. GPU count (shows $ wasted on idle GPUs)
- Marker: Efficiency = 50% line

**mlsysim grounding**: Formulas: T_step(N) = T_compute/N + T_comm(N) - T_overlap. Scaling efficiency from `SCALING_EFF_32GPU` (0.90), `SCALING_EFF_256GPU` (0.70), `SCALING_EFF_1024GPU` (0.50) in `mlsysim.core.defaults`. GPU cost from `Hardware.H100.unit_cost` ($30,000).

**Transition to E**: "You have seen four failure modes of scale: reliability collapse, communication tax, misallocated compute budgets, and diminishing GPU returns. But in any real system, one of these dominates. The C-Cubed diagnostic tells you which."

---

### Part E -- The C-Cubed Diagnostic (~8 min)

**Concept**: The C-Cube taxonomy (Computation, Communication, Coordination) provides a diagnostic framework for identifying the dominant bottleneck in any distributed system. The Conservation of Overhead sits at the center: reducing one C causes another to become dominant.

**Prediction**: Students classify three archetype workloads by dominant bottleneck via radio buttons (not drag-and-drop):

"For each workload, select the dominant bottleneck: Computation, Communication, or Coordination."

| Workload | Student's likely guess | Actual dominant bottleneck |
|----------|----------------------|--------------------------|
| GPT-4 LLM training (175B, 25K GPUs) | Computation | **Communication** (gradient sync) |
| DLRM recommendation (embedding-heavy) | Communication | **Coordination** (All-to-All embedding lookups) |
| Federated MobileNet (edge devices) | Communication | **Coordination** (straggler handling, privacy overhead) |

**Common wrong answer**: Students guess Computation for LLM training because "it has the most FLOPs."

**Why wrong**: At 25K GPUs, the per-GPU compute is small but gradient synchronization of 175B parameters is massive. The Communication term dominates.

**Instrument**:
- Radio buttons: For each of 3 archetypes, select Computation / Communication / Coordination
- Reveal: Stacked bar showing actual time breakdown for each archetype
- Visual: C-Cube triangle with each workload plotted at its actual position

**mlsysim grounding**: Uses `calc_ring_allreduce_time()` for LLM communication estimate. Fleet configurations from `Clusters.Frontier_8K` and `Clusters.Mega_100K`. Compute time from `Engine.solve(model=..., hardware=Hardware.H100)`.

---

## Lab V2-02: The Compute Infrastructure Wall

**Story arc**: Students discover that even the fastest accelerator in the world spends most of its time waiting for data, that the roofline model reveals why, that bandwidth drops by orders of magnitude at each physical boundary, and that even a full DGX node cannot hold a frontier model without memory optimization. They close by discovering that the real cost of scale is not GPUs but everything around them.

**Time budget**: 57 min (12 + 12 + 12 + 12 + 9 = 57 min)

---

### Part A -- The Memory Wall (~12 min)

**Concept**: Token generation latency is dominated by memory bandwidth, not compute. Even with infinite compute throughput, token latency barely improves because data delivery from HBM is the binding constraint. At batch=1, an H100 achieves less than 1% of peak FLOPS.

**Prediction**: "During single-token generation of a 70B model on an H100, what fraction of the time are the arithmetic units idle?"

| Option | Value |
|--------|-------|
| A | ~10% idle -- GPUs are mostly computing |
| B | ~50% idle -- memory and compute are balanced |
| C | ~80% idle -- memory is a drag |
| **D (correct)** | **~99% idle -- the GPU is almost entirely waiting for data** |

**Common wrong answer**: A or B. Students assume GPUs are "compute machines."

**Why wrong**: 70B parameters at 2 bytes = 140 GB. At 3.35 TB/s HBM bandwidth, loading takes ~42 ms. Compute for one token takes ~0.07 ms. The GPU is 99.8% idle.

**Instrument**:
- Select: Model size (7B, 70B, 175B)
- Select: Accelerator (A100, H100, B200)
- Slider: Batch size (1 to 128, powers of 2)
- Chart: Latency waterfall (compute bar vs. memory bar) with ridge point marked
- Gauge: MFU (turning green as batch size increases past ridge point)

**mlsysim grounding**: `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100, batch_size=1, precision="fp16")` returns `PerformanceProfile` with `latency_compute`, `latency_memory`, `mfu`, and `bottleneck`. Hardware specs from `Hardware.H100` (3.35 TB/s HBM, 989 TFLOPS FP16).

**Transition to B**: "At batch=1, the H100 is a $30,000 space heater. The roofline model explains exactly why -- and reveals which workloads can actually use the hardware you paid for."

---

### Part B -- The Roofline Diagnostic (~12 min)

**Concept**: The Roofline Model determines whether a workload is compute-bound or memory-bound. The ridge point separates regimes. Fleet-scale workloads (175B training, not ResNet-50) demonstrate that even on frontier hardware, most LLM operations are memory-bound.

**Prediction**: "Place these fleet-scale workloads on the roofline: LLM decode at batch=1, LLM decode at batch=32, 175B LLM training (forward pass), 175B LLM prefill. Which falls above the ridge point?"

| Option | Value |
|--------|-------|
| A | All of them -- H100 is always compute-bound for large models |
| B | LLM training and prefill -- they have large batch sizes |
| **C (correct)** | **Only LLM prefill and training at large batch -- decode is always below** |
| D | None -- all LLM workloads are memory-bound |

**Common wrong answer**: A. Students conflate "large model" with "compute-bound."

**Why wrong**: Arithmetic intensity depends on the ratio of FLOPs to bytes moved, not model size alone. LLM decode at batch=1 has arithmetic intensity ~0.5 (each weight is loaded once for one multiply), far below the H100 ridge point of ~295 FLOPs/byte at FP16.

**Instrument**:
- Interactive roofline plot: Students drag workload dots to predicted positions
- Workloads: 175B decode B=1, 175B decode B=32, 175B training, 175B prefill, DLRM embedding lookup
- Hardware selector: V100, A100, H100, B200 (shifts the roofline and ridge point)
- Metric: Achieved TFLOPS vs. peak for each workload

**mlsysim grounding**: `Engine.solve()` for each workload returns `arithmetic_intensity` and `peak_flops_actual`. Hardware ridge points calculated as `H100_FLOPS_FP16_TENSOR / H100_MEM_BW` (~295). Model specs from `Models.Llama3_8B`, `Models.Llama3_70B`. Note: uses fleet-scale models (175B), NOT ResNet-50, to differentiate from V1-11.

**Transition to C**: "The roofline tells you about one chip. But a training cluster has thousands of chips connected by a bandwidth staircase where each step drops speed by 10-100x. Let us see how this hierarchy dictates which parallelism strategy goes where."

---

### Part C -- The Bandwidth Staircase (~12 min)

**Concept**: Data transfer speed drops by orders of magnitude at each physical boundary (HBM to NVLink to PCIe to InfiniBand), and this hierarchy dictates which parallelism strategy operates at which level. NVLink-to-IB is an 18x cliff.

**Prediction**: "How much slower is a 10 GB gradient AllReduce over InfiniBand NDR compared to NVLink 4.0?"

| Option | Value |
|--------|-------|
| A | ~2x slower -- InfiniBand is fast |
| B | ~5x slower -- significant but manageable |
| **C (correct)** | **~18x slower -- an order of magnitude gap** |
| D | ~100x slower -- completely different regime |

**Common wrong answer**: A or B. Students underestimate the NVLink-to-IB bandwidth gap.

**Why wrong**: NVLink 4.0 = 900 GB/s. IB NDR = 50 GB/s. Ratio = 18x. For a 10 GB transfer: 11 ms (NVLink) vs. 200 ms (IB NDR). This is why tensor parallelism is confined to within a node.

**Instrument**:
- Slider: Transfer size (1 MB to 10 GB, log scale)
- Staircase bar chart: Transfer time at HBM (3.35 TB/s), NVLink (900 GB/s), PCIe Gen5 (64 GB/s), IB NDR (50 GB/s), IB HDR (25 GB/s)
- Parallelism strategy mapping: TP -> NVLink, PP -> IB (small transfers), DP -> IB (large transfers with compression)

**mlsysim grounding**: `NVLINK_H100_BW` (900 GB/s), `INFINIBAND_NDR_BW` (400 Gbps = 50 GB/s), `PCIE_GEN5_BW` (64 GB/s), `H100_MEM_BW` (3.35 TB/s) from `mlsysim.core.constants`. `Nodes.DGX_H100.intra_node_bw` (900 GB/s).

**Transition to D**: "The bandwidth hierarchy tells you where to put each parallelism strategy. But before you parallelize, you need to know: does the model even fit? Let us check the memory budget for a frontier 175B model."

---

### Part D -- The Node Memory Budget (~12 min)

**Concept**: Training a 175B model requires careful memory budgeting. A single accelerator cannot hold the model; a full 8-GPU DGX H100 node barely suffices even with ZeRO-3, because activation memory pushes total past HBM limits.

**Prediction**: "Can a single 8-GPU DGX H100 node (640 GB total HBM) train a 175B model with Adam in FP16 without ZeRO?"

| Option | Value |
|--------|-------|
| A | Yes -- 640 GB is plenty for a 175B model |
| B | Barely -- it fits with ~50 GB headroom |
| **C (correct)** | **No -- static memory alone exceeds 640 GB** |
| D | No, but ZeRO-1 fixes it |

**Common wrong answer**: A. Students compute only weight memory (175B x 2 bytes = 350 GB) and think it fits.

**Why wrong**: Training memory = weights (350 GB) + gradients (350 GB) + Adam optimizer states (700 GB FP32) = 1,400 GB static memory. This is 2.2x the total HBM of a DGX H100 node.

**Instrument**:
- Slider: Model size (1B to 175B)
- Toggle: Precision (FP32 / FP16 / INT8)
- Select: Optimizer (SGD / Adam / Adafactor)
- Slider: GPUs per node (1, 2, 4, 8)
- Select: ZeRO stage (0, 1, 2, 3)
- Stacked bar: Per-GPU memory (weights, gradients, optimizer, activations) with HBM capacity line

**mlsysim grounding**: `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100, batch_size=1, precision="fp16", is_training=True, zero_stage=3, dp_size=64)` returns `memory_footprint` and `feasible`. Hardware from `Hardware.H100` (80 GB HBM).

**Transition to E**: "So you need multiple nodes, ZeRO optimization, and careful parallelism. But what does all of this cost? The GPU price tag is only the beginning."

---

### Part E -- TCO: The Hidden Cost of Scale (~9 min)

**Concept**: Total Cost of Ownership goes far beyond GPU purchase price. Power, cooling, networking, and utilization efficiency determine the real cost per useful FLOP. A 1,000-GPU H100 cluster costs ~$3M/year in electricity alone. Utilization rate often matters more than hardware generation.

**Prediction**: "For a 1,000-GPU H100 inference cluster, which costs more over 3 years: the GPUs themselves, or the electricity to run them?"

| Option | Value |
|--------|-------|
| A | GPUs by far -- $30M vs. ~$3M electricity |
| **B (correct)** | **GPUs cost more, but electricity is ~30% of total TCO -- surprisingly close** |
| C | Electricity is more expensive -- power costs dominate |
| D | They are roughly equal |

**Common wrong answer**: A. Students drastically underestimate operational costs.

**Why wrong**: 1,000 H100s at 700W = 700 kW. With PUE 1.12, facility power = 784 kW. At $0.12/kWh, annual electricity = $824K. Over 3 years: $2.5M electricity + maintenance + cooling + networking brings OpEx to ~40% of total.

**Instrument**:
- Configure: GPU count, GPU type (A100/H100/B200), networking tier, cooling type
- Slider: Utilization (30% to 90%)
- Slider: PUE (1.1 to 1.6)
- Chart: TCO breakdown (CapEx: GPUs, networking, storage; OpEx: power, cooling, staff) over 3-year lifecycle
- Metric: Cost per useful FLOP at current utilization

**mlsysim grounding**: `Hardware.H100.unit_cost` ($30,000), `Hardware.H100.tdp` (700W), `PUE_BEST_AIR` (1.12), `CLOUD_ELECTRICITY_PER_KWH` ($0.12). `calc_fleet_tco()` from `mlsysim.core.formulas`. Infrastructure costs via `Racks.AI_Standard.power_kw`.

---

## Lab V2-03: Communication at Scale

**Story arc**: This lab merges the old Network Fabrics (V2-03) and Collective Communication (V2-06) labs into a single narrative. Students start with the alpha-beta model to understand the fundamental physics of network transfer, then discover that the choice of AllReduce algorithm depends on a quantitative crossover point. They confront the NVLink-to-InfiniBand cliff and see how hierarchical communication exploits it. They learn when gradient compression helps (and when it hurts). Finally, they assemble a complete communication budget for a 70B model and try to get it under 20% of step time.

**Time budget**: 58 min (10 + 12 + 12 + 12 + 12 = 58 min)

---

### Part A -- The Network Time Budget (~10 min)

**Concept**: The alpha-beta model (T(n) = alpha + n/beta) separates network communication into a latency-dominated regime and a bandwidth-dominated regime. For LLM-scale gradients (hundreds of GB), the bandwidth term dominates by four to five orders of magnitude. This calibration is essential before any algorithm selection can be meaningful.

**Prediction**: "A 70B model trains with data parallelism across 64 GPUs on InfiniBand NDR (50 GB/s, 5 us latency). Gradients are FP32 (4 bytes per parameter). How long does one Ring AllReduce take?"

| Option | Value |
|--------|-------|
| A | ~0.5 ms -- network latency dominates |
| B | ~50 ms -- bandwidth matters, but IB is fast |
| C | ~1,100 ms (~1 second) -- the 280 GB payload starts to show |
| **D (correct)** | **~11,000 ms (~11 seconds) -- bandwidth completely dominates** |

**Common wrong answer**: B or C. Students anchor on IB latency (microseconds) and underestimate by 100-1000x.

**Why wrong**: 70B params x 4 bytes = 280 GB. Ring AllReduce transfers 2(N-1)/N x M bytes. T_bandwidth = 2 x (63/64) x 280 GB / 50 GB/s = 11,032 ms. T_latency = 2 x 63 x 5 us = 0.63 ms. Bandwidth is 99.994% of total.

**Instrument**:
- Select: Model parameters (1B, 7B, 13B, 70B, 175B)
- Toggle: Precision (FP32, BF16, FP8)
- Select: GPU count (8, 16, 32, 64, 128, 256, 512, 1024)
- Toggle: Interconnect (NVLink 4.0 / IB NDR / IB HDR)
- Chart: Stacked bar showing bandwidth term vs. latency term (log scale)
- Metric: Total AllReduce time, bandwidth fraction %

**mlsysim grounding**: `calc_ring_allreduce_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s)` from `mlsysim.core.formulas`. `Fabrics.InfiniBand_NDR` (bandwidth=400 Gbps, latency=5 us). `INFINIBAND_NDR_BW` (400 Gbps = 50 GB/s), `IB_NDR_LATENCY_US` (5 us) from constants/defaults.

**Transition to B**: "Now you know AllReduce is a bandwidth problem. But Ring AllReduce is not the only algorithm. Tree AllReduce trades bandwidth for latency. When does each win? The answer depends on a crossover formula."

---

### Part B -- Ring vs. Tree: The Algorithm Crossover (~12 min)

**Concept**: Ring AllReduce is bandwidth-optimal but latency-poor (O(N) steps). Tree AllReduce has logarithmic latency but O(log N) bandwidth overhead. The crossover point M_crossover = N x alpha x beta determines which algorithm wins. An algorithm that is optimal at 8 GPUs can be catastrophically wrong at 1,024 GPUs.

**Prediction**: "For 256 GPUs on InfiniBand NDR, which AllReduce algorithm is faster for a 1 MB message?"

| Option | Value |
|--------|-------|
| A | Ring -- it is always bandwidth-optimal |
| **B (correct)** | **Tree -- at 1 MB and 256 GPUs, Tree's O(log N) latency advantage wins** |
| C | They are identical for this message size |
| D | Neither -- you need hierarchical AllReduce |

**Common wrong answer**: A. Students learn that Ring is "bandwidth-optimal" and assume it always wins.

**Why wrong**: At 1 MB with 256 GPUs, Ring incurs 2 x 255 = 510 latency steps (2,550 us) while Tree incurs 2 x 8 = 16 steps (80 us). The bandwidth overhead of Tree (log_2(256) = 8x) on 1 MB is only 160 us vs. Ring's 40 us. Tree total: 240 us. Ring total: 2,590 us. Tree wins by >10x. At 10 GB, Ring wins because bandwidth dominates.

**Instrument**:
- Slider: Message size (1 KB to 10 GB, log scale)
- Select: GPU count (64, 256, 1024)
- Animated time bars: Ring vs. Tree completion time side-by-side
- Chart: Transfer time vs. message size (two curves with crossover point marked)
- Crossover marker: M_crossover = N x alpha x beta (draggable to verify formula)

**mlsysim grounding**: `calc_ring_allreduce_time()` and `calc_tree_allreduce_time()` from `mlsysim.core.formulas`. Constants: `IB_NDR_LATENCY_US` (5 us), `INFINIBAND_NDR_BW_GBS` (50 GB/s).

**Transition to C**: "So algorithm choice depends on message size and GPU count. But there is another variable: the physical topology. NVLink within a node is 18x faster than InfiniBand between nodes. What happens when the algorithm is aware of this hierarchy?"

---

### Part C -- Topology and Hierarchy Effects (~12 min)

**Concept**: The NVLink-to-InfiniBand bandwidth cliff (900 GB/s vs. 50 GB/s = 18x gap) means a flat AllReduce that ignores hierarchy wastes up to 50% of training throughput. Hierarchical AllReduce (local reduce within NVLink, then global AllReduce over IB) achieves 5-6x speedup by reducing inter-node traffic by a factor of G (GPUs per node). Fat-tree bisection bandwidth determines the inter-node ceiling; oversubscription creates multiplicative slowdown.

**Prediction**: "A flat Ring AllReduce across 64 GPUs (8 nodes x 8 GPUs) mixes NVLink and InfiniBand links. A hierarchical 2-level AllReduce does local reduce first, then inter-node AllReduce. What speedup does the hierarchical approach achieve?"

| Option | Value |
|--------|-------|
| A | ~1.5x -- marginal improvement |
| B | ~2x -- moderate improvement |
| **C (correct)** | **~5-6x -- dramatic improvement** |
| D | ~18x -- full NVLink/IB ratio |

**Common wrong answer**: A or B. Students underestimate the multiplicative effect of reducing inter-node traffic.

**Why wrong**: Hierarchical AllReduce reduces the data sent over InfiniBand by 8x (one reduce per node before global AllReduce). Since inter-node communication is the bottleneck at 50 GB/s, reducing it by 8x while the intra-node reduce at 900 GB/s is nearly free yields ~5-6x total speedup.

**Instrument**:
- Select: Topology (flat ring, hierarchical 2-level, hierarchical 3-level)
- Slider: GPUs per node (2, 4, 8)
- Slider: Oversubscription ratio (1:1, 2:1, 4:1)
- Chart: AllReduce time breakdown (intra-node component vs. inter-node component)
- Metric: Effective bandwidth, speedup vs. flat ring
- Failure state: Oversubscription ratio > 1 shows proportional slowdown on inter-node component

**mlsysim grounding**: `calc_hierarchical_allreduce_time(message_bytes, n_nodes, gpus_per_node, intra_bw, inter_bw, intra_latency, inter_latency)` from `mlsysim.core.formulas`. `Nodes.DGX_H100.intra_node_bw` (900 GB/s), `Fabrics.InfiniBand_NDR.bandwidth` (400 Gbps = 50 GB/s).

**Transition to D**: "Hierarchical AllReduce is a huge win. But what if you could make the data smaller before sending it? Gradient compression promises exactly this -- but the physics of convergence fight back."

---

### Part D -- Gradient Compression: When Does It Pay Off? (~12 min)

**Concept**: Gradient compression (quantization, sparsification) trades bandwidth savings for convergence slowdown. It is only worthwhile when the communication-to-computation ratio is high. On fast networks, compression hurts because the extra convergence steps outweigh per-step savings.

**Prediction**: "You apply INT8 gradient compression (4x bandwidth reduction) to a 70B model training on 64 GPUs with InfiniBand NDR. Does total training time decrease?"

| Option | Value |
|--------|-------|
| A | Yes, by ~4x -- you saved 75% of communication |
| B | Yes, by ~2x -- significant improvement |
| **C (correct)** | **It depends -- on IB NDR, compression barely helps because the extra convergence steps nearly cancel the per-step savings** |
| D | No -- compression always hurts convergence |

**Common wrong answer**: A or B. Students assume bandwidth savings translate directly to total time savings.

**Why wrong**: INT8 compression reduces per-step communication by ~4x, but typically requires 1.1-1.5x more training steps to converge. On IB NDR (50 GB/s), where communication is already only 30-40% of step time, the net improvement is small. On slow networks (100GbE, 12.5 GB/s), where communication is 70%+ of step time, compression provides substantial benefit.

**Instrument**:
- Select: Compression method (None, FP16, INT8, Top-K 1%, 1-bit)
- Slider: Network bandwidth (10 GB/s to 100 GB/s)
- Slider: Model size (7B, 70B, 175B)
- Chart 1: Per-step waterfall (compute vs. communication bars)
- Chart 2: Total training time curve accounting for extra convergence steps
- Toggle: Error feedback on/off (shows loss plateau when feedback is disabled)

**mlsysim grounding**: Communication time from `calc_ring_allreduce_time()` with modified message_bytes. Convergence penalty modeled as multiplicative step increase (1.0x for None, 1.05x for FP16, 1.15x for INT8, 1.3x for Top-K, 1.5x for 1-bit). Network bandwidth from `Fabrics.InfiniBand_NDR.bandwidth` and `Fabrics.Ethernet_100G.bandwidth`.

**Transition to E**: "You now have four tools: algorithm choice, hierarchical decomposition, compression, and overlap. Let us put them all together and build a complete communication budget for a production 70B training job."

---

### Part E -- Communication Budget Optimization (~12 min)

**Concept**: For a 70B model on 64 GPUs over IB NDR, raw Ring AllReduce takes ~11 seconds per step. Students assemble a communication strategy by toggling optimizations one at a time and watching each chip away at the budget. The goal: reduce communication to under 20% of total step time.

**Prediction**: "Starting from the raw 11-second AllReduce, how many optimizations must you stack to get communication under 20% of step time?"

| Option | Value |
|--------|-------|
| A | Just one -- hierarchical AllReduce is enough |
| B | Two -- hierarchical + FP16 gradients |
| **C (correct)** | **Three or four -- you need hierarchical + FP16 + overlap + bucket fusion** |
| D | It is impossible on IB NDR -- you need XDR |

**Common wrong answer**: A. Students overestimate the impact of a single optimization.

**Why wrong**: Hierarchical AllReduce gives ~5-6x reduction (11s to ~2s). FP16 halves it (~1s). Overlap hides 50-85% behind backward pass. Bucket fusion reduces latency overhead. You need all of them stacked to reach <20%.

**Instrument**:
- Starting point: 11-second AllReduce (from chapter napkin math)
- Checkboxes: Hierarchical AllReduce, FP16 gradients, Bucket fusion, Backward overlap
- Chart: Stacked step-time bar updating as each optimization is toggled
- Metric: Communication as % of total step time
- Target line: 20% threshold (green when met)
- Preset: "Megatron-LM configuration" button that toggles all optimizations

**mlsysim grounding**: `calc_ring_allreduce_time()` and `calc_hierarchical_allreduce_time()` for base and hierarchical times. `DEFAULT_OVERLAP_EFFICIENCY` (0.85) from `mlsysim.core.defaults`. `INFINIBAND_NDR_BW_GBS` (50 GB/s).

---

## Lab V2-04: The Data Pipeline Wall

**Story arc**: Students discover that storage -- the least glamorous infrastructure component -- can silently determine whether a training cluster is productive or an expensive space heater. The gap between compute consumption and storage delivery has widened 60x in seven years and is getting worse. Over five parts they watch the chasm widen, learn the pipeline equation, encounter the birthday problem in shard contention, diagnose stalls that prefetching cannot fix, and finally confront the checkpoint frequency trade-off. This is the best story arc in Vol 2.

**Time budget**: 58 min (12 + 12 + 10 + 12 + 12 = 58 min)

---

### Part A -- The Storage-Compute Chasm (~12 min)

**Concept**: Accelerator throughput has grown 236x (P100 to B200) while NVMe bandwidth grew only 4x over the same period. The resulting 60x widening gap means that data pipeline engineering is a first-order concern. Faster GPUs make the storage problem worse, not better.

**Prediction**: "Your current cluster is storage-bottlenecked at 30% GPU utilization. You upgrade from A100s to H100s (2x more TFLOPS). What happens to GPU utilization?"

| Option | Value |
|--------|-------|
| A | ~60% -- faster GPUs process data faster, so utilization improves |
| B | ~30% -- no change, storage is the bottleneck |
| **C (correct)** | **~15% -- utilization drops because GPUs are faster but storage isn't** |
| D | ~5% -- catastrophic collapse |

**Common wrong answer**: A. Students assume faster GPUs always improve the system.

**Why wrong**: If the GPU processes data 2x faster but the storage delivers at the same rate, the GPU waits twice as long relative to its compute time. Utilization = T_compute / (T_compute + T_IO). Halving T_compute while keeping T_IO constant reduces utilization.

**Instrument**:
- Select: GPU generation (V100, A100, H100, B200)
- Dual-axis timeline chart: Compute throughput (TFLOPS) vs. storage bandwidth (GB/s) across generations
- Metric: Compute-to-storage bandwidth ratio (showing 60x widening)
- Gauge: GPU utilization at current storage bandwidth

**mlsysim grounding**: Hardware specs from `Hardware.V100` through `Hardware.B200` (peak_flops and memory bandwidth). Storage bandwidth via `NVME_SEQUENTIAL_BW` from constants. Compute-to-storage ratio computed as accelerator HBM bandwidth / storage bandwidth.

**Transition to B**: "The chasm is real and getting worse. So how much storage bandwidth do you actually need? The pipeline equation tells you exactly -- and the answer depends on how many GPUs you are feeding."

---

### Part B -- The Data Pipeline Equation (~12 min)

**Concept**: Required storage bandwidth = N_GPUs x U_target x S_batch / T_iteration. Under-provisioning starves accelerators; over-provisioning wastes money. Doubling GPUs without upgrading storage causes a cliff-like utilization drop.

**Prediction**: "You have a 128-GPU cluster with 80% GPU utilization, well-provisioned with NVMe storage. You double to 256 GPUs without upgrading storage. What happens to utilization?"

| Option | Value |
|--------|-------|
| A | ~80% -- utilization is independent of GPU count |
| B | ~60% -- moderate drop from increased demand |
| **C (correct)** | **~40% -- the storage bandwidth is now split across twice as many GPUs** |
| D | ~20% -- catastrophic starving |

**Common wrong answer**: A. Students do not realize that storage bandwidth is a shared resource.

**Why wrong**: BW_required doubles when GPU count doubles, but BW_available stays constant. The utilization drops proportionally.

**Instrument**:
- Slider: GPU count (8 to 1024)
- Select: Model type (vision/language -- affects batch data size)
- Slider: Target utilization (50% to 95%)
- Chart: Data Stall Frontier S-curve (GPU utilization vs. storage bandwidth)
- Metric: Required bandwidth, current bandwidth, utilization, stall %
- Failure state: Banner when utilization drops below 50%

**mlsysim grounding**: Storage bandwidth from `NVME_SEQUENTIAL_BW` in constants. GPU specs from `Hardware.H100`. Pipeline throughput formula: BW_required = N x U x S_batch / T_iter.

**Transition to C**: "Even with enough aggregate bandwidth, random access patterns create a hidden bottleneck. When hundreds of GPUs independently read dataset shards, collisions are surprisingly common. This is the birthday problem at datacenter scale."

---

### Part C -- The Shard Contention Birthday Problem (~10 min)

**Concept**: Even with many dataset shards, random access by many GPUs creates surprisingly high collision probability (birthday problem), causing tail-latency spikes that stall the entire BSP-synchronized cluster. With 64 workers and 1,000 shards, collision probability exceeds 87%.

**Prediction**: "Your 256-GPU cluster reads from a dataset with 1,000 shards. Each GPU randomly selects a shard at the start of each step. What is the probability that at least two GPUs collide on the same shard?"

| Option | Value |
|--------|-------|
| A | ~10% -- 1,000 shards is plenty for 256 workers |
| B | ~50% -- borderline |
| **C (correct)** | **~100% (near certainty) -- collisions are essentially guaranteed** |
| D | ~75% -- high but not certain |

**Common wrong answer**: A. Students think 1,000 shards / 256 workers = ~4 shards per worker, so collisions should be rare.

**Why wrong**: P(collision) = 1 - e^(-n^2 / 2N). With n=256 and N=1,000: exponent = -256^2 / 2000 = -32.8. P = 1 - e^(-32.8) = ~100%. The birthday problem strikes at n = sqrt(N), which is 32 -- far below 256.

**Instrument**:
- Slider: GPU workers (8 to 256)
- Slider: Dataset shards (100 to 10,000)
- Animation: Workers selecting shards with collisions highlighted in red
- Probability meter: Theoretical collision probability
- Toggle: Random vs. deterministic shard assignment (shows collisions drop to zero)

**mlsysim grounding**: Birthday collision formula from textbook. Shard counts and worker counts grounded in typical ImageNet/C4 dataset partitioning.

**Transition to D**: "Collisions create tail latency. But even without contention, can prefetching eliminate stalls entirely? Only if I/O time is shorter than compute time. Let us check."

---

### Part D -- The Data Stall Diagnostic (~12 min)

**Concept**: Pipelining and prefetching can hide storage latency, but only when I/O time does not exceed compute time. When I/O exceeds compute, no amount of overlap eliminates the stall. Without pipelining: T_step = T_IO + T_compute. With pipelining: T_step = max(T_compute, T_IO).

**Prediction**: "Your training step has 200 ms compute and 300 ms I/O. You add prefetching with depth 4. Does the stall disappear?"

| Option | Value |
|--------|-------|
| A | Yes -- 4 batches of prefetch hide the 300 ms I/O |
| **B (correct)** | **No -- stall drops from 60% to 33% but never reaches zero because I/O > compute** |
| C | Partially -- stall drops to ~10% |
| D | No effect -- prefetching only helps random access patterns |

**Common wrong answer**: A. Students believe prefetching can always hide I/O latency.

**Why wrong**: With perfect pipelining, T_step = max(T_compute, T_IO) = max(200, 300) = 300 ms. Stall = (300 - 200) / 300 = 33%. The only fix is faster storage or slower compute (i.e., the pipeline can only hide I/O when T_IO < T_compute).

**Instrument**:
- Slider: Compute time (100-500 ms)
- Slider: I/O time (50-1000 ms)
- Slider: Prefetch buffer depth (0 to 8 batches)
- Timeline animation: Pipeline execution showing overlap (compute green, I/O wait red)
- Gauge: Stall percentage
- Metric: Effective step time, utilization

**mlsysim grounding**: Formulas from textbook: T_step_sequential = T_IO + T_compute; T_step_pipelined = max(T_compute, T_IO); Stall% = (T_step - T_compute) / T_step.

**Transition to E**: "I/O is not just about feeding the GPUs -- it is also about saving progress. Every checkpoint is a massive write that competes with training data reads. How often should you save?"

---

### Part E -- Checkpoint Economics (~12 min)

**Concept**: Checkpoint frequency trades recovery granularity against I/O overhead. Too frequent: checkpointing steals storage bandwidth from training. Too infrequent: failures waste millions in recomputation. The optimal frequency depends on MTBF (provided directly as a parameter, not requiring V2-01 completion) and checkpoint write time.

**Prediction**: "A 1,000-GPU cluster has MTBF of 5 hours (given). Checkpoints for a 175B model take 2 minutes to write. What is the optimal checkpoint interval?"

| Option | Value |
|--------|-------|
| A | Every 5 minutes -- minimize lost work |
| **B (correct)** | **Every ~27 minutes -- the Young-Daly sweet spot** |
| C | Every hour -- minimize I/O overhead |
| D | Every 2 hours -- checkpoints are expensive |

**Common wrong answer**: A. Students prioritize minimizing lost work without considering I/O cost.

**Why wrong**: Young-Daly optimal interval = sqrt(2 x T_write x MTBF) = sqrt(2 x 120s x 18000s) = sqrt(4,320,000) = ~2,078s = ~35 minutes. (Exact value depends on parameter choices; ~27 min for slightly different MTBF.) Too-frequent checkpointing saturates storage bandwidth.

**Instrument**:
- Slider: Cluster size (determines MTBF; or direct MTBF slider, default 5 hours)
- Slider: Checkpoint write time (30s to 5 min; depends on model size and storage BW)
- Slider: Checkpoint interval (1 min to 2 hours)
- Chart: U-shaped waste curve with three components: checkpoint overhead (decreasing), expected rework (increasing), total (U-curve)
- Metric: Optimal interval (Young-Daly), waste %, dollar cost per day
- Failure state: Banner when checkpoint write time exceeds interval

**mlsysim grounding**: `calc_young_daly_interval(checkpoint_cost_s, mtbf_s)` from `mlsysim.core.formulas`. `calc_checkpoint_size(n_params, bytes_per_param=14)` for 175B model. `calc_mtbf_cluster(GPU_MTTF_HOURS, N)` for deriving MTBF. Note: MTBF is provided as a given value (5 hours for 1,000 GPUs) so this lab does not depend on V2-01 completion. Forward reference to V2-06 (Fault Tolerance) for deeper treatment.

---

## Lab V2-05: The Parallelism Puzzle

**Story arc**: Students scale from a single GPU to a 1,024-GPU cluster by trying each parallelism strategy in sequence and discovering that each one solves one constraint while creating another. Data parallelism hits a communication wall. ZeRO trades communication for memory. Pipeline parallelism creates bubbles. Extended Amdahl's Law reveals the cost-efficiency optimum. Finally, 3D parallelism maps strategies to the bandwidth hierarchy. The Conservation of Overhead governs every choice.

**Time budget**: 60 min (12 + 12 + 12 + 12 + 12 = 60 min)

---

### Part A -- The Communication Wall (~12 min)

**Concept**: Data parallelism scales throughput linearly only until gradient synchronization dominates step time. Ring AllReduce communication overhead grows relative to shrinking per-GPU compute, creating a "Communication Wall." Even with InfiniBand, communication consumes 40%+ of step time for a 175B model at 256 GPUs.

**Prediction**: "You train a 175B model with pure data parallelism on 256 GPUs with InfiniBand NDR. What scaling efficiency do you achieve?"

| Option | Value |
|--------|-------|
| A | ~90% -- InfiniBand handles the gradients |
| B | ~70% -- moderate overhead |
| **C (correct)** | **~50-55% -- communication is nearly half of step time** |
| D | ~25% -- communication dominates |

**Common wrong answer**: A. Students who completed V2-03 may have better calibration, but many still overestimate.

**Why wrong**: 175B FP16 gradients = 350 GB. Ring AllReduce: 2(255/256) x 350 GB / 50 GB/s = ~14 seconds. If compute per GPU is ~10 seconds, efficiency = 10 / (10 + 14) = ~42%. With overlap, ~55%.

**Instrument**:
- Slider: GPU count (1 to 512, powers of 2)
- Select: Model size (1B, 7B, 70B, 175B)
- Toggle: Interconnect (100G Ethernet, IB HDR, IB NDR)
- Chart: Speedup (linear vs. actual) and efficiency gauge
- Metric: Communication fraction, step time breakdown

**mlsysim grounding**: `calc_ring_allreduce_time()` from formulas. `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100)` for per-GPU compute time. `Fabrics.InfiniBand_NDR` for network params. `SCALING_EFF_256GPU` (0.70) as reference.

**Transition to B**: "Communication is expensive because every GPU holds a full copy of the model. What if you sharded the model across GPUs so each holds only a fraction? That is ZeRO -- but it trades memory savings for more communication."

---

### Part B -- ZeRO: Trading Communication for Memory (~12 min)

**Concept**: ZeRO optimization shards optimizer states, gradients, and parameters across workers. Each stage reduces per-GPU memory but increases communication volume, embodying the Conservation of Overhead. Even ZeRO-3 on 64 A100s cannot train a 175B model because activation memory pushes total past 80 GB.

**Prediction**: "Can ZeRO-3 on 64 A100 GPUs (80 GB each) train a 175B model?"

| Option | Value |
|--------|-------|
| A | Yes -- ZeRO-3 shards everything across 64 GPUs |
| B | Yes, but only with FP16 precision |
| **C (correct)** | **No -- activation memory (~50 GB) pushes per-GPU total past 80 GB even with ZeRO-3** |
| D | No -- 64 GPUs is not enough for any ZeRO stage |

**Common wrong answer**: A. Students compute static memory only: 175B x 14 bytes / 64 = 38 GB, well within 80 GB.

**Why wrong**: ZeRO-3 shards static memory (parameters + gradients + optimizer) to 38 GB per GPU. But activations are NOT sharded -- each GPU stores its own activations for the micro-batch. At seq_len=2048, batch_size=1, activations ~50 GB for 175B model. Total = 38 + 50 = 88 GB > 80 GB HBM.

**Instrument**:
- Slider: Model size (1B to 175B)
- Slider: Number of GPUs (8 to 256)
- Select: ZeRO stage (0, 1, 2, 3)
- Stacked bar: Per-GPU memory (parameters, optimizer, gradients, activations) at each stage
- Second chart: Communication volume per step at each stage
- HBM capacity line (80 GB for A100, 80 GB for H100)
- Failure state: OOM banner when total exceeds HBM

**mlsysim grounding**: `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100, is_training=True, zero_stage=3, dp_size=64)` returns `memory_footprint` and `feasible`. Activation memory via `calc_activation_memory()`.

**Transition to C**: "ZeRO cannot do it alone for frontier models. Pipeline parallelism takes a different approach: split the model into stages across GPUs. But idle GPUs create 'bubbles' that waste compute."

---

### Part C -- Pipeline Bubbles: The Idle GPU Problem (~12 min)

**Concept**: Pipeline parallelism overlaps computation across model stages using microbatches, but fill and drain phases create idle GPU time (bubbles). Bubble fraction = (P-1)/(P+M-1). With 4 stages and 4 microbatches, bubble fraction = 43%. Reducing bubbles below 10% requires very large numbers of microbatches.

**Prediction**: "With 8 pipeline stages, how many microbatches do you need to keep bubble fraction below 10%?"

| Option | Value |
|--------|-------|
| A | 8 -- one per stage |
| B | 16 -- twice the stages |
| C | 32 -- four times the stages |
| **D (correct)** | **72+ microbatches -- far more than intuition suggests** |

**Common wrong answer**: A or B. Students think matching microbatches to stages is sufficient.

**Why wrong**: Bubble fraction = (P-1)/(P+M-1). For P=8 and target <10%: 7/(7+M) < 0.10, so M > 63. With M=72: bubble = 7/78 = 9%. This implies a very large global batch size (72 x micro_batch_size), which may hurt convergence.

**Instrument**:
- Select: Pipeline stages (2, 4, 8, 16)
- Slider: Microbatches per step (1 to 64)
- Animated Gantt chart: Forward and backward passes flowing through stages, bubbles in red
- Gauge: Bubble percentage
- Metric: Global batch size implied, per-GPU utilization

**mlsysim grounding**: `calc_pipeline_bubble(n_stages, n_microbatches)` from `mlsysim.core.formulas`. `OVERHEAD_PIPELINE_BUBBLE` (0.05) as reference for well-tuned systems.

**Transition to D**: "Remember the scaling tax from V2-01 Part D? Now let us apply it with real parallelism strategies. Each workload type has a different communication fraction, and that determines where the cost-efficiency optimum falls."

---

### Part D -- The Scaling Tax: Amdahl Meets Communication (~12 min)

**Concept**: Distributed training obeys an extended Amdahl's Law where the serial fraction includes communication overhead. The cost-efficiency optimal point differs dramatically by workload type. A bandwidth-bound embedding model hits diminishing returns at just 4 GPUs, while a compute-bound ResNet scales to 128+.

**Prediction**: "For a bandwidth-bound DLRM embedding model (r = 0.50), at what GPU count does cost-per-sample reach its minimum?"

| Option | Value |
|--------|-------|
| A | ~64 GPUs -- standard cluster size |
| B | ~16 GPUs -- moderate scale |
| **C (correct)** | **~4 GPUs -- essentially single-node** |
| D | ~1 GPU -- never parallelize |

**Common wrong answer**: A. Students assume all models benefit equally from parallelism.

**Why wrong**: At r = 0.50, maximum theoretical speedup = 1/r = 2x regardless of GPU count. The cost-efficiency optimum is where the marginal speedup gain equals the marginal GPU cost, which is ~4 GPUs for r=0.50.

**Instrument**:
- Select: Workload type with preset r values:
  - Compute-bound ResNet (r = 0.05)
  - Balanced LLM (r = 0.20)
  - Bandwidth-bound DLRM (r = 0.50)
- Slider: GPU count (1 to 512)
- Toggle: Communication-computation overlap (0%, 50%, 80%)
- Chart 1: Speedup vs. ideal (log-log)
- Chart 2: Cost per sample vs. GPU count (U-curve with optimum marked)

**mlsysim grounding**: Extended Amdahl's: T_step(N) = T_compute/N + T_comm(N) - T_overlap. Cost = N x T_step(N) x GPU_hourly_rate. GPU cost from `Hardware.H100.unit_cost`. Communication fraction from `calc_ring_allreduce_time()`.

**Transition to E**: "No single parallelism strategy works alone. Production training combines tensor, pipeline, and data parallelism -- each mapped to the bandwidth hierarchy you explored in V2-02 Part C. Now you design that mapping."

---

### Part E -- 3D Parallelism: The Hierarchy-Aware Design (~12 min)

**Concept**: Production training combines TP (within NVLink), PP (across nearby nodes on IB), and DP (across all nodes). The constraint is TP x PP x DP = total GPUs. The correct mapping to the bandwidth hierarchy can yield 2-3x efficiency improvement over naive DP.

**Prediction**: "For a 175B model on 256 H100 GPUs, which 3D configuration maximizes training efficiency?"

Students choose from 3-4 preset configurations to reduce cognitive load:

| Config | TP | PP | DP | Notes |
|--------|----|----|-----|-------|
| A | 1 | 1 | 256 | Pure DP |
| B | 8 | 1 | 32 | TP within node + DP |
| **C (correct)** | **8** | **4** | **8** | **Full 3D parallelism** |
| D | 8 | 32 | 1 | TP + aggressive PP |

**Common wrong answer**: A (pure DP) or B (TP + DP). Students either default to the simplest strategy or forget pipeline parallelism.

**Why wrong**: Pure DP requires AllReduce of 350 GB FP16 gradients across 256 GPUs -- prohibitively slow. TP=8 within NVLink is fast, but DP=32 still requires large AllReduce over IB. Adding PP=4 reduces the DP degree to 8, shrinking the AllReduce to 8 nodes while PP's small activation transfers (200 MB) tolerate IB latency.

**Instrument**:
- Select: TP degree (1, 2, 4, 8)
- Select: PP degree (1, 2, 4, 8, 16, 32)
- Auto-calculated: DP = 256 / (TP x PP)
- Topology diagram: Physical mapping of TP/PP/DP to nodes and racks
- Metrics: Per-GPU memory, communication volume per step, pipeline bubble fraction, scaling efficiency
- Preset buttons: 3-4 configurations (listed above) to reduce cognitive load

**mlsysim grounding**: `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100, is_training=True, zero_stage=3, dp_size=DP)` for memory. `calc_ring_allreduce_time()` for DP communication. `calc_pipeline_bubble(PP, microbatches)` for PP overhead. `Nodes.DGX_H100.intra_node_bw` (900 GB/s) for TP bandwidth.

---

## Lab V2-06: When Failure is Routine

**Story arc**: Students confront the reality that at fleet scale, failure is not an exception but a statistical certainty. They briefly recall the exponential reliability collapse (from V2-01), then spend the bulk of the lab on the Young-Daly checkpoint optimization (the U-shaped cost curve), the checkpoint storm problem, serving fault tolerance (where millisecond recovery is required), and the reliability budget trade-off. The Young-Daly sweet spot is one of the best parts in the entire suite -- protect it.

**Time budget**: 56 min (6 + 14 + 12 + 14 + 10 = 56 min)

---

### Part A -- Failure as Routine: A Recall (~6 min)

**Concept**: Rapid recall of the reliability collapse from V2-01 Part A, grounded in this chapter's MTBF equation: MTBF_system = MTBF_component / N. A 10,000-GPU cluster with 50,000-hour GPU MTTF experiences a failure every 5 hours. This is a brief warm-up, NOT a re-teach.

**Prediction**: "A 10,000-GPU cluster uses GPUs with MTBF of 50,000 hours. Approximately how often does the cluster experience a failure?"

| Option | Value |
|--------|-------|
| A | Once a week |
| B | Once a day |
| **C (correct)** | **Every ~5 hours** |
| D | Every ~30 minutes |

**Common wrong answer**: A or B. Even students who saw V2-01 may not remember the exact math.

**Why wrong**: MTBF_cluster = 50,000 / 10,000 = 5 hours. At this rate, a 30-day training run will experience ~144 failures.

**Instrument**:
- Slider: Cluster size (100 to 25,000 GPUs)
- Metric: System MTBF in hours, expected failures per training day/week/month
- Chart: Probability of surviving T hours without failure (exponential decay curve)

**mlsysim grounding**: `calc_mtbf_cluster(GPU_MTTF_HOURS, N)` from `mlsysim.core.formulas`. `GPU_MTTF_HOURS` = 50,000 from `mlsysim.core.defaults`.

**Transition to B**: "Failures every 5 hours mean you must save your progress regularly -- checkpoint. But checkpoint too often and you waste time writing. Checkpoint too rarely and you lose days of work. Where is the sweet spot?"

---

### Part B -- The Young-Daly Sweet Spot (~14 min)

**Concept**: The optimal checkpoint interval tau_opt = sqrt(2 x T_write x MTBF) minimizes total wasted work (checkpoint overhead + rework from failures). The U-shaped cost curve has a clear minimum. This is one of the best pedagogical moments in the entire lab suite.

**Prediction**: "A 16,000-GPU cluster has MTBF of ~3 hours. Checkpoint writes take 2 minutes. What is the optimal checkpoint interval?"

| Option | Value |
|--------|-------|
| A | Every 2 minutes -- match the write time |
| B | Every 10 minutes -- frequent saves |
| **C (correct)** | **Every ~27 minutes -- the square-root law** |
| D | Every 90 minutes -- halfway to MTBF |

**Common wrong answer**: B. Students either checkpoint too aggressively (afraid of failures) or aim for the midpoint of MTBF.

**Why wrong**: tau_opt = sqrt(2 x 120s x 10,800s) = sqrt(2,592,000) = 1,610s = ~27 minutes. The square root law means the optimal interval is geometrically between write time and MTBF, not linearly.

**Instrument**:
- Slider: Cluster size (determines MTBF via MTBF = GPU_MTTF / N)
- Slider: Checkpoint write time (10s to 5 min)
- Slider: Checkpoint interval (draggable, 1 min to 3 hours)
- Chart: U-shaped waste curve with three visible components:
  - Checkpoint overhead (decreasing hyperbola in blue)
  - Expected rework (increasing line in red)
  - Total waste (U-curve in black, minimum marked)
- Metric: Optimal interval, total waste %, dollar cost of waste per day
- Annotation: Young-Daly formula on chart

**mlsysim grounding**: `calc_young_daly_interval(checkpoint_cost_s, mtbf_s)` from `mlsysim.core.formulas`. `calc_mtbf_cluster(GPU_MTTF_HOURS, N)` for MTBF derivation. GPU cost from `Hardware.H100.unit_cost` for dollar cost of waste.

**Transition to C**: "The Young-Daly formula gives the optimal interval. But it assumes checkpointing is free once you start -- it is not. Writing a 175B checkpoint to storage takes time, bandwidth, and money, and at scale, all GPUs write simultaneously."

---

### Part C -- The Checkpoint Storm (~12 min)

**Concept**: When thousands of GPUs write checkpoints simultaneously, storage saturates. The "stop-the-world" cost scales with model size and cluster size. If storage bandwidth is too low, checkpoint write time exceeds the optimal interval, creating a pathological state where the system spends more time checkpointing than computing.

**Prediction**: "A 175B model checkpoints on a 1,000-GPU cluster with NFS storage (1 GB/s aggregate write). How long does one checkpoint take?"

| Option | Value |
|--------|-------|
| A | ~10 seconds -- fast with modern storage |
| B | ~2 minutes -- manageable |
| **C (correct)** | **~41 minutes -- longer than the Young-Daly optimal interval** |
| D | ~5 minutes -- within budget |

**Common wrong answer**: A or B. Students underestimate checkpoint size.

**Why wrong**: 175B params x 14 bytes (weights + Adam states) = 2.45 TB per checkpoint. At 1 GB/s NFS: 2,450 seconds = ~41 minutes. With MTBF of ~5 hours, the optimal interval is ~27 minutes. The checkpoint takes LONGER than the optimal interval -- the system is in a pathological state.

**Instrument**:
- Slider: Model size (1B to 175B)
- Select: Storage type (NFS 1 GB/s / Parallel FS 10 GB/s / NVMe RAID 100 GB/s)
- Slider: GPU count
- Metrics: Checkpoint size (TB), write time, dollar cost per checkpoint, daily checkpoint cost
- Failure state: Banner when write time > Young-Daly optimal interval

**mlsysim grounding**: `calc_checkpoint_size(n_params, bytes_per_param=14)` from `mlsysim.core.formulas`. Storage bandwidth from `CHECKPOINT_WRITE_BW_GBS` (100 GB/s default) in `mlsysim.core.defaults`, with overrides for NFS/parallel FS.

**Transition to D**: "Training fault tolerance is about saving progress and restarting. Serving fault tolerance is fundamentally different: failures must be invisible to users, with millisecond recovery. The strategies are different too."

---

### Part D -- Graceful Degradation in Serving (~14 min)

**Concept**: Serving fault tolerance differs from training: failures must be invisible to users with millisecond-scale recovery. Strategies include model fallback (smaller model), feature fallback (drop expensive features), and load shedding. For LLM serving, simple request redirection fails because KV cache state is lost on the failed replica.

**Prediction**: "An LLM serving replica fails mid-generation. You redirect the in-progress request to another replica. What happens?"

| Option | Value |
|--------|-------|
| A | Seamless recovery -- the user notices nothing |
| B | Brief pause (~100 ms) while the new replica catches up |
| **C (correct)** | **The request must restart from scratch -- KV cache is lost, doubling latency** |
| D | The request fails with an error |

**Common wrong answer**: A or B. Students from web-service backgrounds expect stateless redirection to work.

**Why wrong**: LLM generation is stateful -- the KV cache stores all context computed so far. When a replica fails, this state is lost. The new replica must re-process the entire prompt (prefill) before resuming generation. For a 4K-token context, this adds seconds of latency.

**Instrument**:
- Configure: Replica count (2 to 16), failure rate, SLO budget (P99 latency)
- Failure injection: Toggle to kill a random replica
- Strategy selector: Redirect / Fallback model (smaller) / Load shed
- Live dashboard: P99 latency, accuracy/quality, request success rate
- Metric: KV cache reconstruction time for redirect strategy

**mlsysim grounding**: `calc_availability_stacked(single_availability, n_replicas)` from `mlsysim.core.formulas`. Serving latency decomposition from `Engine.solve()` for LLM decode latency. KV cache size from `calc_kv_cache_size()`.

**Transition to E**: "You have two domains of fault tolerance: training (checkpoint frequency) and serving (replica count). Both cost GPUs. With a fixed GPU budget, how should you allocate between productive compute and fault tolerance overhead?"

---

### Part E -- The Reliability Budget (~10 min)

**Concept**: Fault tolerance investment has diminishing returns. The economic framework balances the cost of redundancy against the cost of downtime, with different optimal points for training and serving. Larger clusters require proportionally more fault tolerance investment, creating a "reliability tax."

**Prediction**: "You have a 1,024-GPU budget. How many GPUs should be dedicated to fault tolerance overhead (spare capacity, checkpointing bandwidth, replicas)?"

| Option | Value |
|--------|-------|
| A | ~2% (20 GPUs) -- minimize overhead |
| **B (correct)** | **~10-15% (100-150 GPUs) -- the diminishing-returns sweet spot** |
| C | ~30% (300 GPUs) -- safety first |
| D | ~50% (512 GPUs) -- maximum reliability |

**Common wrong answer**: A. Students want to maximize productive compute.

**Why wrong**: At 2% overhead, the cost of failures (recomputation, downtime) far exceeds the savings. At 30%, most additional reliability provides negligible benefit. The knee is ~10-15% where marginal reliability gain equals marginal compute cost.

**Instrument**:
- Slider: Total GPU budget (256 to 8,192)
- Slider: Fault tolerance allocation (0% to 50%)
- Chart: Pareto curve (effective throughput vs. reliability)
- Metric: Productive GPUs, fault tolerance GPUs, effective TFLOPS, expected uptime %
- Optimal point marker

**mlsysim grounding**: `OVERHEAD_CHECKPOINT` (0.03), `OVERHEAD_FAILURE_RECOVERY` (0.10), `OVERHEAD_MAINTENANCE` (0.05) from `mlsysim.core.defaults`. `calc_effective_flops(peak_flops, mfu, scaling_eff, goodput_ratio)` for net throughput.

---

## Lab V2-07: The Scheduling Trap

**Story arc**: Students discover that scheduling GPUs is fundamentally harder than scheduling CPUs because of heavy-tailed job distributions, multi-dimensional packing constraints, topology sensitivity, and the impossible trade-off between utilization, fairness, and latency. Over four parts (reduced from five -- deadlock simulation dropped as too complex and OS-specific), they encounter the queuing wall, the allocation problem, topology-aware placement, and the utilization paradox.

**Time budget**: 48 min (10 + 14 + 12 + 12 = 48 min)

---

### Part A -- The Queuing Wall (~10 min)

**Concept**: ML workloads have heavy-tailed duration distributions (coefficient of variation C_s = 3-5) that make queue wait times explode at utilizations where web servers feel responsive. At 80% utilization, ML queue wait is 5x worse than uniform workloads. This is the strongest part of the lab.

**Prediction**: "Your GPU cluster runs at 80% utilization. Web service engineers say 80% is comfortable. What is the average queue wait time for an ML job?"

| Option | Value |
|--------|-------|
| A | ~5 minutes -- similar to web service queuing |
| **B (correct)** | **~25 minutes -- 5x worse than uniform workloads** |
| C | ~1 hour -- significant delay |
| D | ~2 minutes -- GPUs are fast |

**Common wrong answer**: A. Students from web-service backgrounds assume 80% utilization is normal.

**Why wrong**: The Pollaczek-Khinchine formula: W_q = (rho / (1-rho)) x ((1 + C_s^2) / (2 x mu)). For C_s = 3 (ML heavy tail), the (1 + C_s^2)/2 factor = 5, making wait times 5x worse than uniform (C_s = 1). The heavy tail means rare but massive training jobs block hundreds of short experiments.

**Instrument**:
- Slider: Cluster utilization (0% to 99%)
- Toggle: Workload type (Uniform C_s=1 vs. ML C_s=3 vs. Research C_s=5)
- Animation: Queue depth showing jobs arriving and being served, with heavy-tail jobs visually large
- Chart: Wait time vs. utilization for each workload type (diverging curves)
- Metric: Average wait, P99 wait, queue depth

**mlsysim grounding**: `calc_queue_latency_mmc()` from `mlsysim.core.formulas` for baseline. Heavy-tail correction via C_s coefficient. Arrival rate and service rate calibrated from `AVERAGE_RESEARCHER_JOB_DAYS` (2.0 days) and `TARGET_CLUSTER_UTILIZATION` (0.80) in `mlsysim.core.defaults`.

**Transition to B**: "The heavy tail makes queuing painful. But even when a job reaches the front of the queue, it might not run. GPU, CPU, memory, and topology constraints create a multi-dimensional packing problem where the cluster has free GPUs but cannot schedule any pending job."

---

### Part B -- The Allocation Problem (~14 min)

**Concept**: This part merges the fragmentation and gang scheduling problems. Multi-dimensional bin packing with GPU, CPU, memory, and topology constraints creates fragmentation: 30% of GPUs can be free but unusable because they are scattered across nodes. Gang scheduling (all-or-nothing allocation) prevents deadlock but increases fragmentation by requiring contiguous blocks. The combined effect: effective capacity is far less than physical capacity.

**Prediction**: "Your 256-GPU cluster shows 30% free capacity (77 GPUs idle). A researcher submits a 64-GPU training job. Can it be scheduled?"

| Option | Value |
|--------|-------|
| A | Yes immediately -- 77 > 64 idle GPUs |
| **B (correct)** | **No -- the 77 idle GPUs are scattered across 12 nodes in fragments of 1-4 GPUs each, and the job requires 8 contiguous 8-GPU nodes** |
| C | Yes, but with 50% reduced performance due to fragmentation |
| D | Yes, after a brief 5-minute repack |

**Common wrong answer**: A. Students see 77 > 64 and assume scheduling is trivial.

**Why wrong**: Gang scheduling requires all 64 GPUs allocated simultaneously. Topology-aware placement requires them in contiguous nodes (8 per node). With fragments of 1-4 GPUs scattered across nodes, no contiguous block of 64 exists. This is the fragmentation tax: physical capacity != effective capacity.

**Instrument**:
- Cluster heatmap: 32 nodes x 8 GPUs, showing occupied (blue) and free (gray) GPUs
- Job queue: Jobs of varying sizes (1, 2, 4, 8, 64 GPUs) with arrival times
- Scheduling heuristic toggle: First-fit / Best-fit / First-fit-decreasing
- Metric: Effective capacity, fragmentation ratio (stranded GPUs / total), largest contiguous block
- Toggle: Gang scheduling on/off (shows deadlock when off, fragmentation when on)

**mlsysim grounding**: Cluster topology from `Clusters.Research_256` (32 nodes x 8 GPUs). `Nodes.DGX_H100.accelerators_per_node` (8) defines the packing unit.

**Transition to C**: "Even when a contiguous block exists, where you place the job matters enormously. Random placement across racks can degrade throughput by 30-50% compared to topology-aware placement. Let us see why."

---

### Part C -- Topology-Aware Placement (~12 min)

**Concept**: Random GPU placement across a datacenter can degrade training throughput by 30-50% compared to topology-aware placement. The NVLink-to-InfiniBand bandwidth cliff (18x from V2-03) compounds at every communication step. Placement alone, with zero code changes, can match the impact of an algorithmic optimization.

**Prediction**: "You place a 64-GPU training job randomly across the cluster vs. topology-optimally (all within 8 adjacent nodes on the same rack). What speedup does optimal placement achieve?"

| Option | Value |
|--------|-------|
| A | ~1.1x -- placement barely matters |
| B | ~1.5x -- moderate improvement |
| C | ~2x -- significant |
| **D (correct)** | **~3-5x -- placement matches the impact of a major algorithmic optimization** |

**Common wrong answer**: A. Students assume that with InfiniBand everywhere, placement is irrelevant.

**Why wrong**: Random placement crosses rack boundaries, adding hops and hitting the spine oversubscription ratio. With 2:1 oversubscription at spine level, cross-rack AllReduce takes 2x longer than intra-rack. With 3-hop paths vs. 1-hop, latency triples. For TP-heavy workloads, the NVLink/IB cliff (18x) between intra-node and inter-node further amplifies the difference.

**Instrument**:
- Topology visualization: Nodes within racks, racks connected by spine switches
- Placement toggle: Random / Rack-aware / Topology-optimal
- Slider: Job size (8 to 256 GPUs)
- Metric: AllReduce latency, training throughput (samples/sec)
- Congestion heatmap: Shows traffic hot spots for each placement strategy

**mlsysim grounding**: Bandwidth hierarchy from `Nodes.DGX_H100.intra_node_bw` (900 GB/s NVLink), `Fabrics.InfiniBand_NDR.bandwidth` (50 GB/s IB NDR). Cross-rack penalty modeled via `NetworkFabric.oversubscription_ratio`. `calc_hierarchical_allreduce_time()` for communication time at different topological placements.

**Transition to D**: "You now understand that queuing, fragmentation, and placement all constrain scheduling. But here is the trap: optimizing one metric (utilization, fairness, or latency) necessarily hurts the others. You cannot make all stakeholders happy simultaneously."

---

### Part D -- The Utilization Paradox (~12 min)

**Concept**: Maximizing GPU utilization, fairness, job latency, and cost efficiency simultaneously is impossible. Every scheduling policy represents a trade-off point. The conflict between throughput (favor large jobs) and latency (favor small jobs) is the central tension. This is the synthesis of the lab.

**Prediction**: "You operate a shared GPU cluster. Can you achieve >90% utilization AND keep average wait time under 10 minutes AND ensure fair access across 5 research teams?"

| Option | Value |
|--------|-------|
| A | Yes -- a good scheduler can do all three |
| **B (correct)** | **No -- these goals are fundamentally in conflict; improving one degrades another** |
| C | Yes, but only with preemption |
| D | Yes, but only at 50% utilization |

**Common wrong answer**: A. Students believe scheduling is a solved problem from operating systems courses.

**Why wrong**: High utilization requires keeping GPUs busy, which means running large jobs that block queues. Low latency requires running small jobs first, which fragments the cluster and reduces large-job throughput. Fairness requires equal access, which may starve the most productive teams. Preemption helps but adds recomputation cost (lost work since last checkpoint).

**Instrument**:
- Sliders: Priority weight for throughput vs. fairness vs. latency (3 sliders summing to 100%)
- Job queue: Mixed workload (one 512-GPU month-long run + hundreds of 8-GPU 1-hour experiments)
- Live dashboard with 4 metrics: Cluster utilization, average wait time, max wait time, Jain's fairness index
- Color coding: Green when metric meets target, red when violated
- Key insight: It is impossible to turn all 4 metrics green simultaneously

**mlsysim grounding**: Queuing model from `calc_queue_latency_mmc()`. GPU costs from `Hardware.H100.unit_cost` for dollar cost of preemption. `AVERAGE_RESEARCHER_JOB_DAYS` and `TARGET_CLUSTER_UTILIZATION` from defaults for calibrating arrival rates.

---

# Cross-Lab Reference Map

| New Number | Old Number(s) | Title | Key mlsysim Functions |
|------------|--------------|-------|----------------------|
| V2-01 | V2-01 | The Scale Illusion | `calc_failure_probability`, `calc_ring_allreduce_time`, `Engine.solve` |
| V2-02 | V2-02 | The Compute Infrastructure Wall | `Engine.solve`, `Hardware.*`, `Nodes.DGX_H100`, `calc_fleet_tco` |
| V2-03 | V2-03 + V2-06 | Communication at Scale | `calc_ring_allreduce_time`, `calc_tree_allreduce_time`, `calc_hierarchical_allreduce_time`, `Fabrics.InfiniBand_NDR` |
| V2-04 | V2-04 | The Data Pipeline Wall | `calc_young_daly_interval`, `calc_checkpoint_size`, `calc_mtbf_cluster` |
| V2-05 | V2-05 | The Parallelism Puzzle | `Engine.solve(zero_stage=3, dp_size=64)`, `calc_pipeline_bubble`, `calc_ring_allreduce_time` |
| V2-06 | V2-07 | When Failure is Routine | `calc_young_daly_interval`, `calc_mtbf_cluster`, `calc_checkpoint_size`, `calc_kv_cache_size`, `calc_availability_stacked` |
| V2-07 | V2-08 | The Scheduling Trap | `calc_queue_latency_mmc`, `Clusters.Research_256`, `Fabrics.*` |

# Dropped Content (with rationale)

| Content | Was in | Why dropped |
|---------|--------|------------|
| RDMA deep-dive (Go-Back-N, GPUDirect) | Old V2-03 Part B | Too specialized; protocol internals are not durable knowledge |
| Rail-optimized topology | Old V2-03 Part D | Niche; applies only to specific vendor configurations |
| Deadlock simulation | Old V2-08 Part C | Too complex for 12 min; too OS-specific; merged into Part B |
| Redundant alpha-beta intro | Old V2-06 Part A | Merged with V2-03 Part A (single treatment) |
| Separate bandwidth hierarchy | Old V2-06 Part C | Covered by V2-03 Part C (topology + hierarchy in one part) |

# Dependency Chain

```
V2-01 (Scale Illusion)
  |-- V2-02 (Compute Infrastructure) [builds on reliability + efficiency concepts]
  |     |-- V2-03 (Communication at Scale) [builds on bandwidth staircase]
  |           |-- V2-05 (Parallelism Puzzle) [builds on AllReduce + hierarchy]
  |-- V2-04 (Data Pipeline Wall) [independent, references V2-01 MTBF as given]
  |     |-- V2-06 (Fault Tolerance) [forward-referenced by V2-04 Part E]
  |-- V2-07 (Scheduling Trap) [references V2-03 bandwidth for topology placement]
```

# Round 2: Distributed AI Infrastructure 🚀

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Single_Node_Physics.md">🧱 Round 1</a> ·
  <a href="02_Distributed_Infrastructure.md">🚀 Round 2</a> ·
  <a href="03_Production_Serving.md">⚡ Round 3</a> ·
  <a href="04_Operations_and_Economics.md">💼 Round 4</a> ·
  <a href="05_Visual_Architecture_Debugging.md">🖼️ Round 5</a> ·
  <a href="06_Advanced_Systems.md">⚙️ Round 6</a>
</div>

---

The domain of the AI Infrastructure Engineer. This round tests your understanding of what happens when a model exceeds the capacity of a single node: 3D parallelism, network topologies, and fault tolerance.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/cloud/02_Distributed_Infrastructure.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 🔀 Parallelism & Memory Sharding

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The OOM Error</b> · <code>parallelism</code> <code>memory</code></summary>

- **Interviewer:** "We are training a 30B parameter model using standard Data Parallelism on 80GB GPUs. The model weights are 60GB, but the system OOMs instantly on step 1. Why?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "60 GB fits in 80 GB, so it should work. Maybe the batch size is too large." This ignores the elephant in the room.

  **Realistic Solution:** You forgot to account for the Optimizer State. An optimizer like Adam requires 8 bytes per parameter (for the first and second moments) plus 4 bytes for a master FP32 copy of the weights. That adds 12 bytes per parameter on top of the FP16 weights. You must use ZeRO (Zero Redundancy Optimizer) or FSDP to shard these states across the workers instead of replicating them.

  > **Napkin Math:** 30B params × 2 bytes (FP16 weights) = 60 GB. But Adam needs: 30B × 4 bytes (FP32 master) + 30B × 4 bytes (moment 1) + 30B × 4 bytes (moment 2) = 360 GB. Plus 60 GB gradients. Total: **480 GB per GPU** — 6× what you have.

  > **Key Equation:** $\text{Memory}_{Adam} = \text{Params} \times (2 + 4 + 4 + 4 + 2) = 16\ \text{bytes/param}$

  📖 **Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Pipeline Bubble</b> · <code>parallelism</code></summary>

- **Interviewer:** "We implemented Pipeline Parallelism across 8 GPUs. However, our profiler shows the GPUs are only active 50% of the time, sitting idle while waiting for the previous GPU to finish its layer. How do we increase utilization without changing the hardware?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We should switch to Data Parallelism." DP won't work if the model doesn't fit on one GPU — that's why you used PP in the first place.

  **Realistic Solution:** You need to implement microbatching. By splitting the global batch into smaller microbatches, GPU 1 can process microbatch 2 while GPU 2 processes microbatch 1. This overlaps computation and reduces the "Pipeline Bubble" fraction.

  > **Napkin Math:** Bubble fraction = $(P-1)/(M+P-1)$ where $P$ = pipeline stages, $M$ = microbatches. With $P=8$ and $M=1$: bubble = 87.5%. With $M=32$: bubble = 17.9%. With $M=64$: bubble = 9.8%. You need $M \gg P$ to keep GPUs busy.

  > **Key Equation:** $\text{Bubble Fraction} = \frac{P - 1}{M + P - 1}$

  📖 **Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Amdahl Ceiling</b> · <code>parallelism</code> <code>roofline</code></summary>

- **Interviewer:** "We upgraded our CPUs to H100 GPUs, giving us a 500x speedup in raw matrix math. However, our end-to-end training throughput only increased by 20x. Where did the other 480x of our hardware investment go?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The PCIe bus is bottlenecking the data transfer." PCIe can be a factor, but the issue is more fundamental.

  **Realistic Solution:** The Acceleration Wall (Amdahl's Law). Hardware acceleration only speeds up the parallelizable fraction ($p$) of the workload. If data loading, KV-cache updates, or Python overhead take even 5% of the step time ($p=0.95$), your maximum theoretical speedup is capped at $1/(1-0.95) = 20\times$. The serial bottlenecks will always cap the parallel gains.

  > **Key Equation:** $\text{Speedup}_{\max} = \frac{1}{(1 - p) + \frac{p}{S}}$ where $p$ = parallelizable fraction, $S$ = speedup of parallel part

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> Dimensioning the 3D Cube</b> · <code>parallelism</code> <code>network</code></summary>

- **Interviewer:** "We have 1,024 GPUs. How do you allocate the dimensions for Data ($D$), Tensor ($T$), and Pipeline ($P$) parallelism for a 175B model?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Split evenly: $D=T=P$ or just use Data Parallelism for everything." Even splits ignore physical constraints; pure DP won't fit the model.

  **Realistic Solution:** You solve a physical constraint satisfaction problem. $T$ is strictly bounded by the NVLink domain (usually $T=8$ per node). $P$ is bounded by the number of transformer layers and the microbatch count required to hide the bubble (e.g., $P=16$). Data parallelism ($D$) gets the remainder. Total GPUs = $D \times T \times P$, so $D = 1024 / (8 \times 16) = 8$.

  > **Napkin Math:** 175B params × 2 bytes = 350 GB weights. One H100 has 80 GB. Minimum TP to fit weights: $350/80 \approx 5$, round up to $T=8$ (NVLink domain). With 96 transformer layers and $P=16$ stages: 6 layers per stage, bubble = $(16-1)/(M+16-1)$. Need $M \geq 45$ to keep bubble under 25%. $D = 1024/(8 \times 16) = 8$ data-parallel replicas.

  > **Key Equation:** $\text{Total GPUs} = D \times T \times P$

  📖 **Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)

  </details>

</details>

---

### 🌐 Network Topology & Collectives

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Cross-Rack Stall</b> · <code>network</code> <code>parallelism</code></summary>

- **Interviewer:** "We tried to scale our 70B model training by spreading Tensor Parallelism (TP) across two server racks connected by 100 Gbps Ethernet. Training speed immediately dropped to zero. What did we misunderstand about network topology?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "100 Gbps should be enough bandwidth for 8 GPUs." This confuses inter-node bandwidth with intra-node bandwidth.

  **Realistic Solution:** You failed the "Jeff Dean Test." Tensor Parallelism requires an AllReduce operation on the activations of *every single layer* during the forward and backward pass. This requires the massive bandwidth of intra-node interconnects like NVLink (900 GB/s). Standard Ethernet/InfiniBand between racks (~12.5 GB/s for 100 GbE) will instantly bottleneck the GPUs. For cross-rack scaling, you must use Pipeline or Data parallelism.

  > **Napkin Math:** TP AllReduce per layer for 70B model ≈ 2 × hidden_size × batch × bytes ≈ hundreds of MB per layer, 80 layers per step. NVLink: 900 GB/s → microseconds. 100 GbE: 12.5 GB/s → milliseconds per layer × 80 layers = seconds of pure network wait per step.

  📖 **Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Oversubscription Choke</b> · <code>network</code></summary>

- **Interviewer:** "We placed half our GPUs in Rack A and half in Rack B. The intra-rack AllReduce is incredibly fast, but the global AllReduce crawls, even though we bought 400 Gbps InfiniBand. Where is the bottleneck?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The InfiniBand NICs must be misconfigured." Configuration is fine — the problem is the switch topology.

  **Realistic Solution:** Oversubscribed spine switches. A true Fat-Tree (Clos) topology guarantees non-blocking, full bisection bandwidth across the entire cluster. However, if your data center uplinks are $3:1$ oversubscribed (e.g., 3 downlinks for every 1 uplink to the spine), cross-rack traffic will instantly choke during global gradient synchronization.

  > **Napkin Math:** 8 nodes per rack, each with 400 Gbps NIC = 3.2 Tbps aggregate demand per rack. If the spine uplink is only 1.6 Tbps (2:1 oversubscription), half the gradient data is queued, doubling AllReduce time. At 3:1 oversubscription, it triples.

  📖 **Deep Dive:** [Volume II: Network Fabrics](https://mlsysbook.ai/vol2/network_fabrics.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Ring vs Tree Dilemma</b> · <code>network</code> <code>collectives</code></summary>

- **Interviewer:** "For our 10B parameter model, Ring AllReduce utilizes our network perfectly. However, when we switch to a 100M parameter model, it is terribly slow despite moving far less data. Why does the 'best' algorithm fail here?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Ring AllReduce is always optimal — something else must be wrong." Ring is bandwidth-optimal but not latency-optimal.

  **Realistic Solution:** Ring AllReduce is bandwidth-optimal but latency-bound for small payloads. It requires $2(N-1)$ steps around the ring. For huge models, the bandwidth saturation hides the latency. For small models, the network transfer happens instantly, but the latency of hopping through $N$ nodes dominates. You must switch to a Tree reduction ($O(\log N)$ latency) for small messages.

  > **Napkin Math:** 64 nodes, 5 μs per hop. Ring: $2 \times 63$ hops × 5 μs = 630 μs of pure latency. Tree: $2 \times \log_2(64)$ = 12 hops × 5 μs = 60 μs. For a 100M param model (200 MB), the actual data transfer at 50 GB/s takes only 4 ms — but Ring adds 630 μs of latency overhead (16% tax) vs Tree's 60 μs (1.5% tax).

  📖 **Deep Dive:** [Volume II: Collective Communication](https://mlsysbook.ai/vol2/collective_communication.html)

  </details>

</details>

---

### 🛡️ Fault Tolerance & Reliability

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Straggler Problem</b> · <code>fault-tolerance</code></summary>

- **Interviewer:** "99 of our 100 nodes finish their backward pass in 500ms. Node 42 takes 800ms. What is the total step time for the cluster?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "About 500ms on average" or "505ms accounting for the outlier." Averaging is wrong here.

  **Realistic Solution:** 800ms. Synchronous SGD requires a global barrier (AllReduce) to synchronize gradients before the optimizer step. The entire cluster moves at the speed of the slowest node (the straggler). You must implement robust observability to detect if Node 42 is thermal throttling, experiencing a slow PCIe lane, or if the data shard lengths are unbalanced.

  > **Napkin Math:** 100 nodes × 500ms = 50 seconds of useful compute per step. Actual wall time = 800ms. Cluster efficiency = $500/800 = 62.5\%$. That one straggler is wasting 37.5% of your entire fleet's compute budget every single step.

  📖 **Deep Dive:** [Volume II: Fleet Orchestration](https://mlsysbook.ai/vol2/fleet_orchestration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The MTBF Crisis</b> · <code>fault-tolerance</code></summary>

- **Interviewer:** "We are scaling our training from 1,000 to 10,000 GPUs. Our current strategy pauses training for 5 minutes every hour to save a checkpoint. Is this viable at 10k scale?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "5 minutes per hour is only 8% overhead — that's fine at any scale." This ignores how failure rates scale with node count.

  **Realistic Solution:** No. As node count ($N$) increases, the Mean Time Between Failures ($MTBF$) decreases: $MTBF_{cluster} = MTBF_{node} / N$. At 10,000 GPUs, failures happen constantly. You must use the **Young-Daly equation** to balance the checkpoint overhead against the cost of lost work, which usually demands asynchronous, in-memory checkpointing to avoid stalling the training loop.

  > **Napkin Math:** If $MTBF_{node}$ = 1000 hours, then $MTBF_{cluster}$ at 1,000 GPUs = 1 hour. At 10,000 GPUs = 6 minutes. Your 1-hour checkpoint interval means you lose an average of 3 minutes of work per failure at 1k GPUs, but at 10k GPUs you fail every 6 minutes — you'll never complete an hour of training between checkpoints.

  > **Key Equation:** $\tau_{opt} = \sqrt{2 \cdot T_{write} \cdot MTBF_{cluster}}$ (Young-Daly optimal checkpoint interval)

  📖 **Deep Dive:** [Volume II: Fault Tolerance](https://mlsysbook.ai/vol2/fault_tolerance.html)

  </details>

</details>

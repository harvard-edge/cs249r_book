# Round 2: Distributed AI Infrastructure 🚀

<div align="center">
  <a href="README.md">🏠 Hub Home</a> |
  <a href="00_The_Architects_Rubric.md">📋 The Rubric</a> |
  <a href="01_Single_Node_Physics.md">🧱 Round 1</a> |
  <a href="02_Distributed_Infrastructure.md">🚀 Round 2</a> |
  <a href="03_Production_Serving.md">⚡ Round 3</a> |
  <a href="04_Operations_and_Economics.md">💼 Round 4</a> |
  <a href="05_Visual_Architecture_Debugging.md">🖼️ Round 5</a>
</div>

---


The domain of the AI Infrastructure Engineer. This round tests your understanding of what happens when a model exceeds the capacity of a single node: 3D parallelism, network topologies, and fault tolerance.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/02_Distributed_Infrastructure.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>[LEVEL]: [Your Question Title Here]</b></summary>

**Interviewer:** [The scenario or crisis]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Cross-Rack Stall</b></summary>

**Interviewer:** "We tried to scale our 70B model training by spreading Tensor Parallelism (TP) across two server racks connected by 100 Gbps Ethernet. Training speed immediately dropped to zero. What did we misunderstand about network topology?"
**Realistic Solution:** You failed the "Jeff Dean Test." Tensor Parallelism requires an AllReduce operation on the activations of *every single layer* during the forward and backward pass. This requires the massive bandwidth of intra-node interconnects like NVLink (900 GB/s). Standard Ethernet/InfiniBand between racks (~50 GB/s) will instantly bottleneck the GPUs. For cross-rack scaling, you must use Pipeline or Data parallelism.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Amdahl Ceiling</b></summary>

**Interviewer:** "We upgraded our CPUs to H100 GPUs, giving us a 500x speedup in raw matrix math. However, our end-to-end training throughput only increased by 20x. Where did the other 480x of our hardware investment go?"
**Realistic Solution:** The Acceleration Wall (Amdahl's Law). Hardware acceleration only speeds up the parallelizable fraction ($p$) of the workload. If data loading, KV-cache updates, or Python overhead take even 5% of the step time ($p=0.95$), your maximum theoretical speedup is capped at $1/(1-0.95) = 20x$. The serial bottlenecks will always cap the parallel gains.
**📖 Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The OOM Error</b></summary>

**Interviewer:** "We are training a 30B parameter model using standard Data Parallelism on 80GB GPUs. The model weights are 60GB, but the system OOMs instantly on step 1. Why?"
**Realistic Solution:** You forgot to account for the Optimizer State. An optimizer like Adam requires 8 bytes per parameter (for the first and second moments) plus 4 bytes for a master FP32 copy of the weights. That adds 12 bytes per parameter on top of the FP16 weights. You must use ZeRO (Zero Redundancy Optimizer) or FSDP to shard these states across the workers instead of replicating them.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Pipeline Bubble</b></summary>

**Interviewer:** "We implemented Pipeline Parallelism across 8 GPUs. However, our profiler shows the GPUs are only active 50% of the time, sitting idle while waiting for the previous GPU to finish its layer. How do we increase utilization without changing the hardware?"
**Realistic Solution:** You need to implement microbatching. By splitting the global batch into smaller microbatches, GPU 1 can process microbatch 2 while GPU 2 processes microbatch 1. This overlaps computation and reduces the "Pipeline Bubble" fraction, which is calculated as $(P-1)/M$, where $P$ is the number of pipeline stages and $M$ is the number of microbatches.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The MTBF Crisis</b></summary>

**Interviewer:** "We are scaling our training from 1,000 to 10,000 GPUs. Our current strategy pauses training for 5 minutes every hour to save a checkpoint. Is this viable at 10k scale?"
**Realistic Solution:** No. As node count ($N$) increases, the Mean Time Between Failures ($MTBF$) decreases exponentially ($MTBF_{cluster} = MTBF_{node} / N$). At 10,000 GPUs, failures happen constantly. You must use the **Young-Daly equation** ($\tau_{opt} = \sqrt{2 \cdot T_{write} \cdot MTBF}$) to balance the checkpoint overhead against the cost of lost work, which usually demands asynchronous, in-memory checkpointing to avoid stalling the training loop.
**📖 Deep Dive:** [Volume II: Fault Tolerance](https://mlsysbook.ai/vol2/fault_tolerance.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Oversubscription Choke</b></summary>

**Interviewer:** "We placed half our GPUs in Rack A and half in Rack B. The intra-rack AllReduce is incredibly fast, but the global AllReduce crawls, even though we bought 400 Gbps InfiniBand. Where is the bottleneck?"
**Realistic Solution:** Oversubscribed spine switches. A true Fat-Tree (Clos) topology guarantees non-blocking, full bisection bandwidth across the entire cluster. However, if your data center uplinks are $3:1$ oversubscribed (e.g., 3 downlinks for every 1 uplink to the spine), cross-rack traffic will instantly choke during global gradient synchronization.
**📖 Deep Dive:** [Volume II: Network Fabrics](https://mlsysbook.ai/vol2/network_fabrics.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Ring vs Tree Dilemma</b></summary>

**Interviewer:** "For our 10B parameter model, Ring AllReduce utilizes our network perfectly. However, when we switch to a 100M parameter model, it is terribly slow despite moving far less data. Why does the 'best' algorithm fail here?"
**Realistic Solution:** Ring AllReduce is bandwidth-optimal but latency-bound for small payloads. It requires $2(N-1)$ steps around the ring. For huge models, the bandwidth saturation hides the latency. For small models, the network transfer happens instantly, but the latency of hopping through $N$ nodes dominates. You must switch to a Tree reduction ($O(\log N)$ latency) for small messages.
**📖 Deep Dive:** [Volume II: Collective Communication](https://mlsysbook.ai/vol2/collective_communication.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> Dimensioning the 3D Cube</b></summary>

**Interviewer:** "We have 1,024 GPUs. How do you allocate the dimensions for Data ($D$), Tensor ($T$), and Pipeline ($P$) parallelism for a 175B model?"
**Realistic Solution:** You solve a physical constraint satisfaction problem. $T$ is strictly bounded by the NVLink domain (usually $T=8$ per node). $P$ is bounded by the number of transformer layers and the microbatch count required to hide the bubble (e.g., $P=16$). Data parallelism ($D$) gets the remainder. Total GPUs = $D \times T \times P$, so $D = 1024 / (8 \times 16) = 8$.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Straggler Problem</b></summary>

**Interviewer:** "99 of our 100 nodes finish their backward pass in 500ms. Node 42 takes 800ms. What is the total step time for the cluster?"
**Realistic Solution:** 800ms. Synchronous SGD requires a global barrier (AllReduce) to synchronize gradients before the optimizer step. The entire cluster moves at the speed of the slowest node (the straggler). You must implement robust observability to detect if Node 42 is thermal throttling, experiencing a slow PCIe lane, or if the data shard lengths are unbalanced.
**📖 Deep Dive:** [Volume II: Fleet Orchestration](https://mlsysbook.ai/vol2/fleet_orchestration.html)
</details>

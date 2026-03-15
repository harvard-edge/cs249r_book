# Round 2: Distributed AI Infrastructure 🚀

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
<summary><b>🟡 LEVEL 2: The Cross-Rack Stall (Tensor Parallelism vs. Topology)</b></summary>

**Interviewer:** "We tried to scale our 70B model training by spreading Tensor Parallelism (TP) across two server racks connected by 100 Gbps Ethernet. Training speed immediately dropped to zero. What did we misunderstand about the network topology?"
**Realistic Solution:** You failed the "Jeff Dean Test." Tensor Parallelism requires an AllReduce operation on the activations of *every single layer* during the forward and backward pass. This requires the massive bandwidth of intra-node interconnects like NVLink (900 GB/s). Standard Ethernet/InfiniBand between racks will instantly bottleneck. For cross-rack scaling, you must use Pipeline or Data parallelism.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The Amdahl Ceiling (Communication-Computation Ratio)</b></summary>

**Interviewer:** "We doubled our cluster size from 512 to 1,024 GPUs using Data Parallelism, but our training throughput only increased by 10%. Where did the other 90% of our hardware investment go?"
**Realistic Solution:** It went to Communication Overhead (Amdahl's Law). The Iron Law of Scale dictates that $T_{step} = (T_{compute}/N) + T_{comm}(N) - T_{overlap}$. As you add nodes ($N$), computation time drops, but the time to synchronize gradients (AllReduce) grows. If the time to "talk" exceeds the time to "compute," adding more GPUs actively degrades scaling efficiency.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>🟢 LEVEL 1: The Out-Of-Memory Error (ZeRO and FSDP)</b></summary>

**Interviewer:** "We are training a 30B parameter model using standard Data Parallelism on 80GB GPUs. The model weights are 60GB, but the system OOMs instantly on step 1. Why, and how do we fix it?"
**Realistic Solution:** You forgot the Optimizer State (Adam requires 12 bytes per parameter) and Gradients. The total memory required is roughly $20\times$ the parameter count. To fix this, you must use ZeRO (Zero Redundancy Optimizer) or FSDP to shard the optimizer states and gradients across the data parallel workers instead of replicating them.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: The Pipeline Bubble (Microbatching)</b></summary>

**Interviewer:** "We implemented Pipeline Parallelism across 8 GPUs. However, our profiler shows the GPUs are only active 50% of the time, waiting for the previous GPU to finish its layer. How do we increase utilization without changing the hardware?"
**Realistic Solution:** You need to implement microbatching. By splitting the global batch into smaller microbatches, GPU 1 can process microbatch 2 while GPU 2 processes microbatch 1. This overlaps computation and reduces the "Pipeline Bubble" fraction, which is calculated as $(P-1)/M$, where P is stages and M is microbatches.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The MTBF Crisis (Optimal Checkpointing)</b></summary>

**Interviewer:** "We are scaling to a 10,000 GPU cluster. Our current checkpointing strategy pauses training for 5 minutes every hour to save to network storage. Is this viable?"
**Realistic Solution:** No. As node count ($N$) increases, the Mean Time Between Failures ($MTBF$) of the cluster decreases exponentially ($MTBF_{cluster} = MTBF_{node} / N$). At 10,000 GPUs, failures happen constantly. You must use the Young-Daly equation to balance the checkpoint overhead against the cost of lost work, and likely implement asynchronous, in-memory checkpointing to avoid stalling the training loop.
**📖 Deep Dive:** [Volume II: Fault Tolerance](https://mlsysbook.ai/vol2/fault_tolerance.html)
</details>

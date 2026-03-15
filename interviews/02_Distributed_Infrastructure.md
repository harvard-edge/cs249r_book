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
<summary><b>🟡 LEVEL 2: The Cross-Rack Stall (Topology vs Parallelism)</b></summary>

**Interviewer:** "We tried to scale our 70B model training by spreading Tensor Parallelism (TP) across two server racks connected by 100 Gbps Ethernet. Training speed immediately dropped to zero. What did we misunderstand about network topology?"
**Realistic Solution:** You failed the "Jeff Dean Test." Tensor Parallelism requires an AllReduce operation on the activations of *every single layer* during the forward and backward pass. This requires the massive bandwidth of intra-node interconnects like NVLink (900 GB/s). Standard Ethernet/InfiniBand between racks (~50 GB/s) will instantly bottleneck the GPUs. For cross-rack scaling, you must use Pipeline or Data parallelism.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The Amdahl Ceiling (H100 Parallelism)</b></summary>

**Interviewer:** "We upgraded our CPUs to H100 GPUs, giving us a 500x speedup in raw matrix math. However, our end-to-end training throughput only increased by 20x. Where did the other 480x of our hardware investment go?"
**Realistic Solution:** The Acceleration Wall (Amdahl's Law). Hardware acceleration only speeds up the parallelizable fraction ($p$) of the workload. If data loading, KV-cache updates, or Python overhead take even 5% of the step time ($p=0.95$), your maximum theoretical speedup is capped at $1/(1-0.95) = 20x$. The serial bottlenecks will always cap the parallel gains.
**📖 Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b>🟢 LEVEL 1: The OOM Error (Optimizer State Accounting)</b></summary>

**Interviewer:** "We are training a 30B parameter model using standard Data Parallelism on 80GB GPUs. The model weights are 60GB, but the system OOMs instantly on step 1. Why?"
**Realistic Solution:** You forgot to account for the Optimizer State. An optimizer like Adam requires 8 bytes per parameter (for the first and second moments) plus 4 bytes for a master FP32 copy of the weights. That adds 12 bytes per parameter on top of the FP16 weights. You must use ZeRO (Zero Redundancy Optimizer) or FSDP to shard these states across the workers instead of replicating them.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: The Pipeline Bubble (Microbatching)</b></summary>

**Interviewer:** "We implemented Pipeline Parallelism across 8 GPUs. However, our profiler shows the GPUs are only active 50% of the time, sitting idle while waiting for the previous GPU to finish its layer. How do we increase utilization without changing the hardware?"
**Realistic Solution:** You need to implement microbatching. By splitting the global batch into smaller microbatches, GPU 1 can process microbatch 2 while GPU 2 processes microbatch 1. This overlaps computation and reduces the "Pipeline Bubble" fraction, which is calculated as $(P-1)/M$, where $P$ is the number of pipeline stages and $M$ is the number of microbatches.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The MTBF Crisis (Young-Daly Checkpointing)</b></summary>

**Interviewer:** "We are scaling our training from 1,000 to 10,000 GPUs. Our current strategy pauses training for 5 minutes every hour to save a checkpoint. Is this viable at 10k scale?"
**Realistic Solution:** No. As node count ($N$) increases, the Mean Time Between Failures ($MTBF$) decreases exponentially ($MTBF_{cluster} = MTBF_{node} / N$). At 10,000 GPUs, failures happen constantly. You must use the **Young-Daly equation** ($\tau_{opt} = \sqrt{2 \cdot T_{write} \cdot MTBF}$) to balance the checkpoint overhead against the cost of lost work, which usually demands asynchronous, in-memory checkpointing to avoid stalling the training loop.
**📖 Deep Dive:** [Volume II: Fault Tolerance](https://mlsysbook.ai/vol2/fault_tolerance.html)
</details>

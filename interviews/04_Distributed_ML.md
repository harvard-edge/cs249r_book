# Distributed ML & Scaling 🚀

Physics of fleets, communication, and fault tolerance at scale.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/04_Distributed_ML.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>[LEVEL]: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>🟢 LEVEL 1: What are the three axes of the 3D Parallelism Cube?</b></summary>

**Answer:** Data, Tensor, and Pipeline Parallelism.
**Realistic Solution:** **Data Parallelism** unrolls the batch loop (replicating the model across devices). **Tensor Parallelism** vectorizes the inner matrix math of a single layer across devices. **Pipeline Parallelism** stages the sequential layers across different nodes. Production runs of models like GPT-4 require a hybrid of all three.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: Why can't you use Tensor Parallelism across different server racks?</b></summary>

**Answer:** The "Jeff Dean Test" for bandwidth constraints.
**Realistic Solution:** Tensor Parallelism requires an AllReduce operation on the activations of *every single layer*. This volume demands the massive bandwidth of intra-node interconnects like NVLink (900 GB/s). If you attempt this across server racks connected by standard InfiniBand/Ethernet (~50 GB/s), the GPUs will stall waiting for data. For cross-rack scaling, you must switch to Pipeline or Data parallelism.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: How does the Communication-Computation Ratio cap cluster performance?</b></summary>

**Answer:** Amdahl's Law at the cluster level.
**Realistic Solution:** The Iron Law of Scale dictates that $T_{step} = (T_{compute}/N) + T_{comm}(N) - T_{overlap}$. As you add nodes ($N$), computation time drops, but communication time (synchronization) grows. If the time to "talk" exceeds the time to "compute", adding more H100s actively degrades throughput. A senior engineer must optimize for communication intensity, not just arithmetic intensity.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

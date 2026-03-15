# Round 2: Distributed AI Infrastructure 🚀

The domain of the AI Infrastructure Engineer. This round tests your understanding of what happens when a model exceeds the capacity of a single node: 3D parallelism, network topologies, and fault tolerance.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/02_Distributed_Infrastructure.md)** (Edit in Browser)

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
<summary><b>🔴 LEVEL 3: How do we size checkpoint intervals for a 10,000 GPU cluster?</b></summary>

**Answer:** The Young-Daly Equation (Balancing MTBF and Checkpoint Cost).
**Realistic Solution:** As node count ($N$) increases, the Mean Time Between Failures ($MTBF$) of the cluster decreases exponentially ($MTBF_{cluster} = MTBF_{node} / N$). A 10,000 GPU cluster will experience failures every few hours. You must balance the time wasted saving frequent checkpoints against the time lost re-calculating work after a failure.
**📖 Deep Dive:** [Volume II: Fault Tolerance](https://mlsysbook.ai/vol2/fault_tolerance.html)
</details>

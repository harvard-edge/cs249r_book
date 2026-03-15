# Distributed ML & Scaling 🚀

Physics of fleets, communication, and fault tolerance at scale.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/04_Distributed_ML.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>SCALE: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>AMDAHL: Why does adding more GPUs sometimes decrease throughput?</b></summary>

**Answer:** Communication Overhead.
**Realistic Solution:** In distributed training, you must synchronize gradients via AllReduce. As you add nodes, the computation time per node decreases, but the communication time (synchronization) increases. If the time to "talk" exceeds the time to "compute," the scaling efficiency drops.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

<details>
<summary><b>FRAGILITY: How do we size checkpoint intervals for large clusters?</b></summary>

**Answer:** The Young-Daly Equation.
**Realistic Solution:** As node count increases, the Mean Time Between Failures (MTBF) of the cluster decreases exponentially. You must balance the time wasted saving checkpoints against the time lost re-calculating work after a failure.
**📖 Deep Dive:** [Volume II: Fault Tolerance](https://mlsysbook.ai/vol2/fault_tolerance.html)
</details>

<details>
<summary><b>LOCALITY: Why use Fat-Tree networks in AI datacenters?</b></summary>

**Answer:** To maximize Bisection Bandwidth.
**Realistic Solution:** AllReduce operations require massive bandwidth between all pairs of nodes. A Fat-Tree topology ensures that any group of nodes can communicate at full wire speed without being throttled by network hierarchy bottlenecks.
**📖 Deep Dive:** [Volume II: Network Fabrics](https://mlsysbook.ai/vol2/network_fabrics.html)
</details>

# Distributed ML & Scaling 🚀

Physics of fleets, communication, and fault tolerance at scale.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/04_Distributed_ML.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>SCALE: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Why:** [Explain the physics/logic here]
**Deep Dive:** [Optional: Maintainers will add link to Chapter/Wall]
</details>
```

---

<details>
<summary><b>AMDAHL: Why does adding more GPUs sometimes make training slower?</b></summary>

**Answer:** Communication Overhead (**Wall 14: Communication**).
**Why:** In distributed training, you must synchronize gradients via AllReduce. As you add nodes, the computation time per node decreases, but the communication time (synchronization) often increases. If the time to "talk" exceeds the time to "compute," the scaling efficiency ($\eta$) drops, and total throughput may actually decrease.
**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html) (**Wall 14: Communication**)
</details>

<details>
<summary><b>FRAGILITY: How do we size the checkpoint interval for a 10,000 GPU cluster?</b></summary>

**Answer:** The Young-Daly Equation.
**Why:** As the number of nodes ($N$) increases, the Mean Time Between Failures ($MTBF$) of the cluster decreases exponentially ($M_{cluster} = M_{node} / N$). This is **Wall 15: Fragility**. You must balance the time wasted saving checkpoints against the time lost re-calculating work after a failure.
**📖 Deep Dive:** [Volume II: Fault Tolerance](https://mlsysbook.ai/vol2/fault_tolerance.html) (**Wall 15: Fragility**)
</details>

<details>
<summary><b>LOCALITY: Why is Fat-Tree topology common in AI datacenters?</b></summary>

**Answer:** To maximize Bisection Bandwidth (**Wall 10: Locality**).
**Why:** AllReduce operations require massive bandwidth between all pairs of nodes. A Fat-Tree topology ensures that any group of nodes can communicate at full wire speed without being throttled by a "skinny" uplink higher in the network hierarchy.
**📖 Deep Dive:** [Volume II: Network Fabrics](https://mlsysbook.ai/vol2/network_fabrics.html) (**Wall 10: Locality**)
</details>

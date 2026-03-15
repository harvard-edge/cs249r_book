# Distributed ML & Scaling 🚀

Physics of fleets, communication, and fault tolerance at scale.

---

<details>
<summary><b>COMMUNICATION: Why does adding more GPUs sometimes decrease throughput?</b></summary>

**Answer:** Communication overhead.
**Explanation:** synchronization overhead (AllReduce) grows with the number of nodes. If communication time exceeds computation time, total throughput drops.
</details>

<details>
<summary><b>RELIABILITY: How do we size checkpoint intervals for large clusters?</b></summary>

**Answer:** Young-Daly Equation.
**Explanation:** As node count increases, Mean Time Between Failures ($MTBF$) decreases. Checkpoint intervals must balance the cost of saving against the cost of re-calculating work after a crash.
</details>

<details>
<summary><b>TOPOLOGY: Why use Fat-Tree networks in AI datacenters?</b></summary>

**Answer:** To maximize Bisection Bandwidth.
**Explanation:** Distributed training requires massive node-to-node communication. Fat-Tree ensures any node pair can communicate at wire speed.
</details>

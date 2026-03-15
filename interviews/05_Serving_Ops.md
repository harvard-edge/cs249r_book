# Model Serving & MLOps 💼

The economics and reliability of production AI systems.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/05_Serving_Ops.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>SERVING: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b>TAIL LATENCY: Why do LLMs feel slow even when average throughput is high?</b></summary>

**Answer:** Non-linear queueing delays at high utilization.
**Realistic Solution:** Request latency follows an Erlang-C distribution. As GPU utilization approaches 100%, the probability of requests "piling up" increases exponentially. A system at 90% utilization might have an average latency of 50ms but a P99 latency of 5 seconds. 
**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
</details>

<details>
<summary><b>TCO: Why is electricity cost as critical as hardware cost?</b></summary>

**Answer:** 3-year Total Cost of Ownership (TCO).
**Realistic Solution:** Training a frontier model can cost $100M in hardware (CapEx), but the energy and cooling bills (OpEx) over 3 years can match or exceed that cost. System efficiency isn't just about speed; it's about financial viability.
**📖 Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)
</details>

<details>
<summary><b>DRIFT: Why do accurate models fail in production?</b></summary>

**Answer:** Statistical distribution shift.
**Realistic Solution:** Unlike traditional software, ML systems fail silently when the live data distribution shifts away from the training distribution. Continuous monitoring is required to identify when the "Silicon Contract" has been broken.
**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ops.html)
</details>

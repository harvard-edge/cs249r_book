# Model Serving & MLOps 💼

The economics and reliability of production AI systems.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/05_Serving_Ops.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>SERVING: [Your Question Title Here]</b></summary>

**Answer:** [Direct answer here]
**Why:** [Explain the physics/logic here]
**Deep Dive:** [Optional: Maintainers will add link to Chapter/Wall]
</details>
```

---

<details>
<summary><b>TAIL LATENCY: Why do LLMs feel slow even when average throughput is high?</b></summary>

**Answer:** Non-linear queueing delays (**Wall 7: Tail Latency**).
**Why:** Request latency follows an Erlang-C distribution. As GPU utilization approaches 100%, the probability of requests "piling up" increases exponentially. A system at 90% utilization might have an average latency of 50ms but a P99 latency of 5 seconds. Managing the "Tail at Scale" is critical for user experience.
**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html) (**Wall 7: Tail Latency**)
</details>

<details>
<summary><b>TCO: Why is the price of electricity as important as the price of H100s?</b></summary>

**Answer:** The Total Cost of Ownership ($TCO$) duality (**Wall 17: Capital**).
**Why:** Training a frontier model can cost $100M in hardware (CapEx), but the energy and cooling bills (OpEx) over 3 years can match or exceed that cost. Efficiency ($\eta$) isn't just about speed; it's about the financial viability of the system.
**📖 Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html) (**Wall 17: Capital**)
</details>

<details>
<summary><b>DRIFT: Why do accurate models fail in production?</b></summary>

**Answer:** Statistical Drift (The Silent Degradation Wall).
**Why:** Unlike traditional software, ML systems fail silently when the live data distribution shifts away from the training distribution. You must monitor features and predictions continuously to identify when the "Silicon Contract" has been broken by the real world.
**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ops.html) (**Wall 19: Drift**)
</details>

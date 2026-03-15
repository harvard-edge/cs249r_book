# Model Serving & MLOps 💼

The economics and reliability of production AI systems.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/05_Serving_Ops.md)** (Edit in Browser)

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
<summary><b>🟢 LEVEL 1: What is the 'Serving Inversion' compared to Training?</b></summary>

**Answer:** The shift from maximizing throughput to minimizing latency.
**Realistic Solution:** Training maximizes throughput ($T$) by using massive batches to keep the GPUs at 100% utilization. Serving minimizes latency ($L_{lat}$) because a slow response is a broken product. The batch-heavy architectures that saturate accelerators during training are fundamentally ill-suited for the bursty, latency-critical reality of production traffic.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: Why do production serving clusters rarely run at 90% GPU utilization?</b></summary>

**Answer:** The Tail Latency Explosion (Queueing Theory).
**Realistic Solution:** As system utilization ($\rho$) passes the "Knee" at ~70%, request queue lengths grow exponentially, not linearly (per Erlang-C / Little's Law). A system at 90% load might have an average latency of 50ms but a P99 latency of 5 seconds. To maintain SLAs, engineers must deliberately over-provision hardware to maintain 40-60% headroom, drastically increasing the dollar cost per query.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: Why can an ML system be perfectly available but perfectly wrong?</b></summary>

**Answer:** The Operational Mismatch (Silent Degradation).
**Realistic Solution:** Traditional DevOps monitors deterministic health (uptime, 200 OK HTTP codes). MLOps must monitor statistical health. A model experiencing Data Drift will continue serving predictions with full confidence, triggering no infrastructure alerts, while its accuracy silently drops from 95% to 80%. You must engineer telemetry (KL Divergence, PSI) to detect distribution shifts before they hit business metrics.
**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

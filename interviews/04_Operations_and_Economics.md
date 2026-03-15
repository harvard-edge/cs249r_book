# Round 4: ML Operations & Economics 💼

The domain of the ML Leadership and Responsible Engineer. This round tests your ability to maintain system health over time: managing data drift, technical debt, and the Total Cost of Ownership (TCO).

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/04_Operations_and_Economics.md)** (Edit in Browser)

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
<summary><b>🟢 LEVEL 1: Explain the '95% Problem' in production ML systems.</b></summary>

**Answer:** The ML model code is only a tiny fraction of the overall system.
**Realistic Solution:** Based on Sculley et al. (2015), the actual ML modeling code is ~5% of the total codebase. The other 95% is "Glue Code"—data ingestion, monitoring, feature extraction, and resource management. Optimizing the model architecture alone ignores the largest sources of technical debt and system friction.
**📖 Deep Dive:** [Volume I: Introduction](https://mlsysbook.ai/vol1/introduction.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: Why can an ML system be perfectly available but perfectly wrong?</b></summary>

**Answer:** The Operational Mismatch (Silent Degradation).
**Realistic Solution:** Traditional DevOps monitors deterministic health (uptime, 200 OK HTTP codes). MLOps must monitor statistical health. A model experiencing Data Drift will continue serving predictions with full confidence, triggering no infrastructure alerts, while its accuracy silently drops from 95% to 80%. You must engineer telemetry (e.g., KL Divergence, PSI) to detect distribution shifts before they hit business metrics.
**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: Why is electricity cost considered a hard constraint for AI infrastructure?</b></summary>

**Answer:** The Total Cost of Ownership ($TCO$) duality.
**Realistic Solution:** Training a frontier model can cost $100M in hardware (CapEx), but the energy and cooling bills (OpEx) over a 3-year lifecycle can match or exceed that initial cost. Furthermore, deploying models at scale must account for the carbon intensity of the grid. System efficiency isn't just about speed ($T$); it is about the financial viability and environmental sustainability of the deployment.
**📖 Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)
</details>

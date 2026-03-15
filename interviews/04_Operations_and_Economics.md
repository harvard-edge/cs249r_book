# Round 4: ML Operations & Economics 💼

The domain of the ML Leadership and Responsible Engineer. This round tests your ability to maintain system health over time: managing data drift, technical debt, and the Total Cost of Ownership (TCO).

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/04_Operations_and_Economics.md)** (Edit in Browser)

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
<summary><b>🟢 LEVEL 1: The '95% Problem' (Technical Debt)</b></summary>

**Interviewer:** "We hired an elite team of researchers to optimize our model architecture, but our deployment velocity is still terrible. Based on the Google 'Hidden Technical Debt' paper, where should we be looking instead of the ML code?"
**Realistic Solution:** The ML model code is only a tiny fraction (~5%) of a production system. The other 95% is "Glue Code"—data ingestion, monitoring, feature extraction, and resource management. If you only optimize the math, the system's velocity will remain bound by the manual overhead of the surrounding infrastructure.
**📖 Deep Dive:** [Volume I: Introduction](https://mlsysbook.ai/vol1/introduction.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: The Silent Failure (Data Drift)</b></summary>

**Interviewer:** "Our DevOps dashboard shows 99.99% uptime and 50ms latency. The HTTP error rate is zero. But the business team is furious because our recommendations are completely wrong. How can the system be perfectly healthy but perfectly wrong?"
**Realistic Solution:** The Operational Mismatch. Traditional software fails loudly (crashes). ML systems fail *silently*. A model experiencing Data Drift (e.g., user behavior changed due to a holiday) will continue serving predictions with full confidence. You must engineer statistical telemetry (e.g., KL Divergence, PSI) to monitor input distributions, not just server health.
**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The Energy Economics (TCO)</b></summary>

**Interviewer:** "We are purchasing a $100M cluster of H100 GPUs. The CFO wants to know the Total Cost of Ownership (TCO) over a 3-year lifecycle. Why is the $100M figure severely underestimating the budget?"
**Realistic Solution:** You forgot the OpEx (Operating Expenses), specifically power and cooling. Over a 3-year lifespan, the electricity required to run a massive cluster at high utilization—plus the power to cool it—frequently matches or exceeds the initial CapEx (hardware cost). System efficiency ($\eta$) isn't just about training speed; it's the primary lever for financial viability.
**📖 Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: The Training-Serving Skew (Feature Stores)</b></summary>

**Interviewer:** "Our model achieves 95% accuracy in the offline test set, but drops to 70% in production. The model weights are identical. Where is the bug most likely located?"
**Realistic Solution:** Training-Serving Skew. The python code used by the Data Science team to compute features in a batch notebook is different from the Java/C++ code the engineering team uses to compute features in real-time. This causes the model to see different data distributions in production. The architectural fix is implementing a **Feature Store** to guarantee identical feature computation across both environments.
**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The Bias Amplifier (Responsible Engineering)</b></summary>

**Interviewer:** "Our training data consists of 1 million images from North America and 10,000 images from Southeast Asia. If we evaluate our model on a global test set and achieve 95% overall accuracy, why might the model still be considered structurally unsafe for deployment?"
**Realistic Solution:** Global accuracy masks subgroup failure. Because the model optimizes for average loss, it will achieve 95% by being highly accurate on the majority class (North America) while potentially failing completely on the minority class (Southeast Asia). In safety-critical domains (like self-driving or medical diagnosis), you must evaluate and threshold metrics on a *per-subgroup* basis, not just the global aggregate.
**📖 Deep Dive:** [Volume I: Responsible Engineering](https://mlsysbook.ai/vol1/responsible_engr.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The Retraining Math (Cost-Aware Automation)</b></summary>

**Interviewer:** "Our model degrades by 1% accuracy every week. Retraining the model costs us $50,000 in GPU time. A 1% accuracy drop costs the business $100,000 a week. Exactly how often should we trigger a retraining pipeline?"
**Realistic Solution:** Cost-aware automation. You do not retrain based on a calendar; you retrain when the cumulative cost of performance degradation intersects with the fixed cost of retraining. This is the Retraining Staleness Model. You must formulate the threshold mathematically so the MLOps pipeline triggers training autonomously when the financial math dictates it.
**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: The Deployment Risk (Shadow vs Canary)</b></summary>

**Interviewer:** "We want to test a brand new recommendation model, but if it performs poorly, it will severely impact our daily revenue. A standard A/B test (sending 10% of users to the new model) is deemed too risky. How do we test it on production traffic without any business risk?"
**Realistic Solution:** Shadow Deployment. You send real production requests to the new model asynchronously. The system records its predictions, but actually serves the *old* model's predictions back to the user. This allows you to evaluate the new model's latency and accuracy on real-world distributions with exactly zero impact on the user experience.
**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: The Trust Boundary (Prompt Injection)</b></summary>

**Interviewer:** "We deployed a customer service LLM. Within a day, users figured out how to trick it into ignoring its system instructions and issuing $500 refunds. Why did standard software security fail here?"
**Realistic Solution:** Adversarial Prompt Injection. In traditional software, code and data are stored in separate memory spaces (preventing SQL injection). In LLMs, the "system instructions" and the "untrusted user data" are concatenated into the exact same context window. The model cannot inherently distinguish between the two, meaning security must be implemented at the orchestration layer (via input/output guardrails).
**📖 Deep Dive:** [Volume II: Robust AI](https://mlsysbook.ai/vol2/robust_ai.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The Privacy Audit (Differential Privacy)</b></summary>

**Interviewer:** "We trained a diagnostic model on sensitive medical records. A security audit shows that by querying the model millions of times, an attacker can determine if a specific patient was part of the training set. How do we prevent this?"
**Realistic Solution:** Membership Inference Attack. The model has memorized the training data. To fix this, you must implement Differentially Private SGD (DP-SGD). By clipping the gradients of individual examples and injecting statistical noise during the backward pass, you can provide mathematical guarantees (measured by an epsilon bound $\epsilon$) that the model's output cannot reveal the presence or absence of any single data point.
**📖 Deep Dive:** [Volume II: Security and Privacy](https://mlsysbook.ai/vol2/security_privacy.html)
</details>

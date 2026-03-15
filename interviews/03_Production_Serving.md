# Round 3: Production ML Systems Design ⚡

The domain of the MLOps and Deployment Engineer. This round tests your ability to survive unpredictable user traffic: latency constraints, continuous batching, and KV-cache management.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/03_Production_Serving.md)** (Edit in Browser)

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

**Answer:** The shift from maximizing throughput ($T$) to minimizing latency ($L_{lat}$).
**Realistic Solution:** Training maximizes throughput by using massive batches to keep the GPUs at 100% utilization. Serving minimizes latency because a slow response is a broken product. The batch-heavy architectures that saturate accelerators during training are fundamentally ill-suited for the bursty, latency-critical reality of production traffic.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: Why do production serving clusters rarely run at 90% GPU utilization?</b></summary>

**Answer:** The Tail Latency Explosion (Queueing Theory).
**Realistic Solution:** As system utilization ($ho$) passes the "Knee" at ~70%, request queue lengths grow exponentially, not linearly (per Erlang-C / Little's Law). A system at 90% load might have an average latency of 50ms but a P99 latency of 5 seconds. To maintain SLAs, engineers must deliberately over-provision hardware to maintain 40-60% headroom, drastically increasing the dollar cost per query.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: Why is PagedAttention required for scaling LLM inference?</b></summary>

**Answer:** To eliminate KV-Cache memory fragmentation.
**Realistic Solution:** Standard attention allocates contiguous VRAM for the maximum possible sequence length. Because sequence lengths are unpredictable, this wastes 60-80% of memory to fragmentation. PagedAttention maps virtual KV-cache blocks to non-contiguous physical blocks (like an OS virtual memory page table). This near-zero fragmentation allows the system to fit $2	imes$ to $3	imes$ more concurrent requests in the same HBM, drastically improving throughput.
**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

# Round 3: Production ML Systems Design ⚡

The domain of the MLOps and Deployment Engineer. This round tests your ability to survive unpredictable user traffic: latency constraints, continuous batching, and KV-cache management.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/03_Production_Serving.md)** (Edit in Browser)

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
<summary><b>🟢 LEVEL 1: The Serving Inversion (Throughput vs. Latency)</b></summary>

**Interviewer:** "We took our highly-optimized training architecture (large batches, deep pipelines) and deployed it directly to serving. Now, user requests are timing out. What fundamental priority did we fail to invert?"
**Realistic Solution:** The shift from maximizing throughput ($T$) to minimizing latency ($L_{lat}$). Training maximizes throughput by using massive batches to keep the GPUs at 100% utilization. Serving minimizes latency because a slow response is a broken product. The batch-heavy architectures that saturate accelerators during training are fundamentally ill-suited for the bursty, latency-critical reality of production traffic.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: The Black Friday Collapse (Queueing Theory)</b></summary>

**Interviewer:** "Our serving cluster handles 1,000 QPS at 50ms latency. On Black Friday, traffic spiked 10x to 10,000 QPS. The system didn't just slow down 10x; it completely collapsed, with latencies hitting 10 seconds. Why did the system fail non-linearly?"
**Realistic Solution:** The Tail Latency Explosion. As system utilization ($\rho$) passes the "Knee" at ~70%, request queue lengths grow exponentially, not linearly (per Erlang-C / Little's Law). The system spends more time managing context switches than doing useful work. You must run clusters with 40-60% headroom or implement aggressive load-shedding to survive traffic spikes.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The Fragmentation Crisis (PagedAttention)</b></summary>

**Interviewer:** "We are serving a chatbot. Even though we have 40GB of free VRAM, our inference server refuses to accept new concurrent requests, citing an 'Out of Memory' error. What is consuming our VRAM invisibly, and how do we fix it?"
**Realistic Solution:** KV-Cache memory fragmentation. Standard attention allocates contiguous VRAM for the *maximum possible* sequence length of every request. Because actual sequence lengths are unpredictable, this wastes 60-80% of memory. You must implement PagedAttention (like vLLM), which maps virtual KV-cache blocks to non-contiguous physical blocks, allowing near-zero fragmentation and 2-3x higher batch sizes.
**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b>🟡 LEVEL 2: The LLM Metrics (TTFT vs TPOT)</b></summary>

**Interviewer:** "Our users are complaining the LLM feels 'laggy', but our monitoring shows 'Average Latency' is well under our 2-second SLO. What specific generation metrics are we failing to monitor?"
**Realistic Solution:** You are measuring the total request time, which masks the user experience. You must split monitoring into Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT). TTFT measures the compute-bound prefill phase (when the user is waiting for the bot to start typing). TPOT measures the memory-bandwidth-bound decode phase (the reading speed). A fast TPOT cannot save a slow TTFT.
**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
</details>

<details>
<summary><b>🔴 LEVEL 3: The Batching Dilemma (Continuous Batching)</b></summary>

**Interviewer:** "We use static batching for our LLM API. If a request generating 5 tokens is batched with a request generating 500 tokens, the 5-token request sits in VRAM until the 500-token request finishes. How do we fix this idle capacity?"
**Realistic Solution:** Implement Continuous (or In-Flight) Batching. Instead of waiting for all requests in a batch to finish, continuous batching operates at the iteration level. As soon as the 5-token request finishes and emits its `<EOS>` token, it is immediately evicted from the batch, and a new request from the queue is slotted in for the very next forward pass.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

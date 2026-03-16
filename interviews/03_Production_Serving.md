# Round 3: Production ML Systems Design ⚡

<div align="center">
  <a href="README.md">🏠 Hub Home</a> |
  <a href="00_The_Architects_Rubric.md">📋 The Rubric</a> |
  <a href="01_Single_Node_Physics.md">🧱 Round 1</a> |
  <a href="02_Distributed_Infrastructure.md">🚀 Round 2</a> |
  <a href="03_Production_Serving.md">⚡ Round 3</a> |
  <a href="04_Operations_and_Economics.md">💼 Round 4</a> |
  <a href="05_Visual_Architecture_Debugging.md">🖼️ Round 5</a>
</div>

---


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
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Serving Inversion</b></summary>

**Interviewer:** "We took our highly-optimized training architecture (large batches, deep pipelines) and deployed it directly to serving. Now, user requests are timing out. What fundamental priority did we fail to invert?"
**Realistic Solution:** The shift from maximizing throughput ($T$) to minimizing latency ($L_{lat}$). Training maximizes throughput by using massive batches to keep the GPUs at 100% utilization. Serving minimizes latency because a slow response is a broken product. The batch-heavy architectures that saturate accelerators during training are fundamentally ill-suited for the bursty, latency-critical reality of production traffic.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Black Friday Collapse</b></summary>

**Interviewer:** "Our serving cluster handles 1,000 QPS at 50ms latency. On Black Friday, traffic spiked 10x to 10,000 QPS. The system didn't just slow down 10x; it completely collapsed, with latencies hitting 10 seconds. Why did the system fail non-linearly?"
**Realistic Solution:** The Tail Latency Explosion. As system utilization ($\rho$) passes the "Knee" at ~70%, request queue lengths grow exponentially, not linearly (per Erlang-C / Little's Law). The system spends more time managing context switches than doing useful work. You must run clusters with 40-60% headroom or implement aggressive load-shedding to survive traffic spikes.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Fragmentation Crisis</b></summary>

**Interviewer:** "We are serving a chatbot. Even though we have 40GB of free VRAM, our inference server refuses to accept new concurrent requests, citing an 'Out of Memory' error. What is consuming our VRAM invisibly, and how do we fix it?"
**Realistic Solution:** KV-Cache memory fragmentation. Standard attention allocates contiguous VRAM for the *maximum possible* sequence length of every request. Because actual sequence lengths are unpredictable, this wastes 60-80% of memory. You must implement PagedAttention (like vLLM), which maps virtual KV-cache blocks to non-contiguous physical blocks, allowing near-zero fragmentation and 2-3x higher batch sizes.
**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The LLM Metrics</b></summary>

**Interviewer:** "Our users are complaining the LLM feels 'laggy', but our monitoring shows 'Average Latency' is well under our 2-second SLO. What specific generation metrics are we failing to monitor?"
**Realistic Solution:** You are measuring the total request time, which masks the user experience. You must split monitoring into Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT). TTFT measures the compute-bound prefill phase (when the user is waiting for the bot to start typing). TPOT measures the memory-bandwidth-bound decode phase (the reading speed). A fast TPOT cannot save a slow TTFT.
**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Batching Dilemma</b></summary>

**Interviewer:** "We use static batching for our LLM API. If a request generating 5 tokens is batched with a request generating 500 tokens, the 5-token request sits in VRAM until the 500-token request finishes. How do we fix this idle capacity?"
**Realistic Solution:** Implement Continuous (or In-Flight) Batching. Instead of waiting for all requests in a batch to finish, continuous batching operates at the iteration level. As soon as the 5-token request finishes and emits its `<EOS>` token, it is immediately evicted from the batch, and a new request from the queue is slotted in for the very next forward pass.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Disaggregated Serving Architecture</b></summary>

**Interviewer:** "In our LLM deployment, users sending very long prompts are causing massive latency spikes for other users who are currently in the middle of generating tokens. How do we isolate these workloads?"
**Realistic Solution:** Disaggregated Serving. The prompt phase (Prefill) is heavily compute-bound and monopolizes the GPU ALUs, starving the token generation phase (Decode), which is memory-bandwidth bound. You must split Prefill and Decode onto entirely separate GPU clusters, computing the KV-Cache on the Prefill nodes and transmitting it over the network to the Decode nodes.
**📖 Deep Dive:** [Volume II: Inference at Scale](https://mlsysbook.ai/vol2/inference.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Decoding Bottleneck</b></summary>

**Interviewer:** "We are heavily memory-bandwidth bound during LLM decoding. How can we generate tokens faster without changing the model weights, quantizing, or losing exact mathematical accuracy?"
**Realistic Solution:** Speculative Decoding. You use a tiny, fast "draft" model to guess the next $K$ tokens. You then pass these $K$ tokens to your massive target model in a *single forward pass*. The large model verifies the guesses in parallel (trading spare ALU compute capacity to save memory fetches). All correct tokens are accepted, maintaining identical output distributions but yielding 2-3x speedups.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Pre-computation Trade-off</b></summary>

**Interviewer:** "We are deploying a photo classification model. Running it in real-time on user upload costs us $10,000 a month in GPU compute. How do we reduce costs without losing model accuracy?"
**Realistic Solution:** Shift from Dynamic (Real-time) to Static (Batch) inference. Instead of keeping a GPU idling to wait for user uploads, run the model periodically (e.g., overnight) on cheap spot instances where you can use massive batch sizes to hit 100% GPU utilization. Store the results in a fast key-value cache (Redis) for millisecond serving to the user.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Serverless Freeze</b></summary>

**Interviewer:** "Our serverless inference endpoint scales down to 0 replicas to save money. However, the first user request after scaling back up takes 30 seconds to process. What is causing this delay?"
**Realistic Solution:** Cold Start Latency. The system must provision the container and load the Python runtime, but most importantly, it must transfer the massive model weights (gigabytes of data) from network storage into the GPU's HBM *before* the first forward pass can execute. You must use persistent warm replicas or optimized weight loading strategies to fix this.
**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

# Round 3: Production ML Systems Design ⚡

<div align="center">
  <a href="README.md">🏠 Home</a> ·
  <a href="00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Single_Node_Physics.md">🧱 Round 1</a> ·
  <a href="02_Distributed_Infrastructure.md">🚀 Round 2</a> ·
  <a href="03_Production_Serving.md">⚡ Round 3</a> ·
  <a href="04_Operations_and_Economics.md">💼 Round 4</a> ·
  <a href="05_Visual_Architecture_Debugging.md">🖼️ Round 5</a>
</div>

---

The domain of the MLOps and Deployment Engineer. This round tests your ability to survive unpredictable user traffic: latency constraints, continuous batching, and KV-cache management.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/03_Production_Serving.md)** (Edit in Browser) — see [README](README.md#question-format) for the template.

---

### ⏱️ Latency & Throughput

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Serving Inversion</b> · <code>latency</code></summary>

**Interviewer:** "We took our highly-optimized training architecture (large batches, deep pipelines) and deployed it directly to serving. Now, user requests are timing out. What fundamental priority did we fail to invert?"

**Common Mistake:** "We need to scale up the serving cluster." More hardware won't fix a design that optimizes for the wrong metric.

**Realistic Solution:** The shift from maximizing throughput ($T$) to minimizing latency ($L_{lat}$). Training maximizes throughput by using massive batches to keep the GPUs at 100% utilization. Serving minimizes latency because a slow response is a broken product. The batch-heavy architectures that saturate accelerators during training are fundamentally ill-suited for the bursty, latency-critical reality of production traffic.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The LLM Metrics</b> · <code>latency</code> <code>serving</code></summary>

**Interviewer:** "Our users are complaining the LLM feels 'laggy', but our monitoring shows 'Average Latency' is well under our 2-second SLO. What specific generation metrics are we failing to monitor?"

**Common Mistake:** "We should look at P99 latency instead of average." P99 helps, but you're still measuring the wrong thing.

**Realistic Solution:** You are measuring the total request time, which masks the user experience. You must split monitoring into Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT). TTFT measures the compute-bound prefill phase (when the user is waiting for the bot to start typing). TPOT measures the memory-bandwidth-bound decode phase (the reading speed). A fast TPOT cannot save a slow TTFT.

> **Napkin Math:** If TTFT = 3 seconds and TPOT = 20ms, a 200-token response takes $3 + (200 \times 0.02) = 7$ seconds total. Average latency looks fine across short and long requests, but users experience a 3-second blank screen before any text appears — that *feels* broken regardless of total latency.

**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Black Friday Collapse</b> · <code>latency</code> <code>queueing</code></summary>

**Interviewer:** "Our serving cluster handles 1,000 QPS at 50ms latency. On Black Friday, traffic spiked 10x to 10,000 QPS. The system didn't just slow down 10x; it completely collapsed, with latencies hitting 10 seconds. Why did the system fail non-linearly?"

**Common Mistake:** "The system should degrade linearly — 10x traffic means 10x latency." Queueing systems don't work that way.

**Realistic Solution:** The Tail Latency Explosion. As system utilization ($\rho$) passes the "Knee" at ~70%, request queue lengths grow exponentially, not linearly (per Erlang-C / Little's Law). The system spends more time managing context switches than doing useful work. You must run clusters with 40-60% headroom or implement aggressive load-shedding to survive traffic spikes.

> **Napkin Math:** At $\rho = 0.5$ (50% util): avg queue length ≈ 1. At $\rho = 0.9$ (90% util): avg queue length ≈ 9. At $\rho = 0.99$: avg queue length ≈ 99. The relationship is $L_q \approx \rho/(1-\rho)$ — it's a hyperbola that goes to infinity as utilization approaches 100%.

> **Key Equation:** $L_q = \frac{\rho^2}{1 - \rho}$ (M/M/1 queue length) and $W = \frac{L_q}{\lambda}$ (Little's Law)

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

---

### 🗂️ KV-Cache & Memory Management

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Fragmentation Crisis</b> · <code>kv-cache</code> <code>memory</code></summary>

**Interviewer:** "We are serving a chatbot. Even though we have 40GB of free VRAM, our inference server refuses to accept new concurrent requests, citing an 'Out of Memory' error. What is consuming our VRAM invisibly, and how do we fix it?"

**Common Mistake:** "There must be a memory leak in the serving framework." It's not a leak — it's by design.

**Realistic Solution:** KV-Cache memory fragmentation. Standard attention allocates contiguous VRAM for the *maximum possible* sequence length of every request. Because actual sequence lengths are unpredictable, this wastes 60-80% of memory. You must implement PagedAttention (like vLLM), which maps virtual KV-cache blocks to non-contiguous physical blocks, allowing near-zero fragmentation and 2-3x higher batch sizes.

> **Napkin Math:** Max sequence = 8192 tokens. Average actual sequence = 500 tokens. Waste per request = $(8192 - 500)/8192 = 93.9\%$. With 40 GB free and each max-length reservation taking ~13 GB, you can only serve 3 concurrent requests. With PagedAttention, you allocate only what's used: 3 requests × 500 tokens = 1500 tokens worth of cache — leaving room for 20+ more concurrent requests.

**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🔄 Batching & Scheduling

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Pre-computation Trade-off</b> · <code>serving</code></summary>

**Interviewer:** "We are deploying a photo classification model. Running it in real-time on user upload costs us $10,000 a month in GPU compute. How do we reduce costs without losing model accuracy?"

**Common Mistake:** "Quantize the model to INT8 to halve the compute cost." Quantization helps, but there's a 10x cheaper architectural change.

**Realistic Solution:** Shift from Dynamic (Real-time) to Static (Batch) inference. Instead of keeping a GPU idling to wait for user uploads, run the model periodically (e.g., overnight) on cheap spot instances where you can use massive batch sizes to hit 100% GPU utilization. Store the results in a fast key-value cache (Redis) for millisecond serving to the user.

> **Napkin Math:** Real-time: 1 GPU reserved 24/7 at $2/hr = $1,440/month at ~10% average utilization. Batch: same workload in 2 hours overnight on spot instances at $0.70/hr = $1.40/day = $42/month at 100% utilization. That's a **34× cost reduction**.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Batching Dilemma</b> · <code>serving</code> <code>batching</code></summary>

**Interviewer:** "We use static batching for our LLM API. If a request generating 5 tokens is batched with a request generating 500 tokens, the 5-token request sits in VRAM until the 500-token request finishes. How do we fix this idle capacity?"

**Common Mistake:** "Set a maximum generation length to keep batch members similar." This caps functionality and still wastes compute on padding.

**Realistic Solution:** Implement Continuous (or In-Flight) Batching. Instead of waiting for all requests in a batch to finish, continuous batching operates at the iteration level. As soon as the 5-token request finishes and emits its `<EOS>` token, it is immediately evicted from the batch, and a new request from the queue is slotted in for the very next forward pass.

> **Napkin Math:** Static batch of 8 requests: shortest = 5 tokens, longest = 500 tokens. Wasted compute = 7 requests × (500 - avg_length) × cost_per_token. With continuous batching, the 5-token request is done in 5 iterations and replaced — the slot processes ~100 different short requests in the time the long request takes, increasing effective throughput by 2-4×.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

---

### 🏗️ Serving Architecture

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Serverless Freeze</b> · <code>serving</code></summary>

**Interviewer:** "Our serverless inference endpoint scales down to 0 replicas to save money. However, the first user request after scaling back up takes 30 seconds to process. What is causing this delay?"

**Common Mistake:** "The model needs to warm up its caches." There's no cache warm-up — the delay is purely about data movement.

**Realistic Solution:** Cold Start Latency. The system must provision the container and load the Python runtime, but most importantly, it must transfer the massive model weights (gigabytes of data) from network storage into the GPU's HBM *before* the first forward pass can execute. You must use persistent warm replicas or optimized weight loading strategies to fix this.

> **Napkin Math:** A 70B model in FP16 = 140 GB. Network storage throughput (EBS/S3) ≈ 5-10 GB/s. Load time = $140/7.5 \approx 19$ seconds just for weight transfer. Add container startup (~3s) and Python/CUDA init (~5s) = ~27 seconds total cold start.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Disaggregated Serving Architecture</b> · <code>serving</code> <code>kv-cache</code></summary>

**Interviewer:** "In our LLM deployment, users sending very long prompts are causing massive latency spikes for other users who are currently in the middle of generating tokens. How do we isolate these workloads?"

**Common Mistake:** "Rate-limit long prompts" or "Add more GPUs to the pool." Neither addresses the fundamental resource contention.

**Realistic Solution:** Disaggregated Serving. The prompt phase (Prefill) is heavily compute-bound and monopolizes the GPU ALUs, starving the token generation phase (Decode), which is memory-bandwidth bound. You must split Prefill and Decode onto entirely separate GPU clusters, computing the KV-Cache on the Prefill nodes and transmitting it over the network to the Decode nodes.

> **Napkin Math:** Prefill for a 10k-token prompt on a 70B model: ~2 seconds of pure compute, consuming 100% of GPU ALUs. During those 2 seconds, every concurrent decode request (which needs ~5ms per token) is blocked. With 50 concurrent users, that's 50 × 2s = 100 user-seconds of stalled generation from a single long prompt.

**📖 Deep Dive:** [Volume II: Inference at Scale](https://mlsysbook.ai/vol2/inference.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Decoding Bottleneck</b> · <code>serving</code> <code>roofline</code></summary>

**Interviewer:** "We are heavily memory-bandwidth bound during LLM decoding. How can we generate tokens faster without changing the model weights, quantizing, or losing exact mathematical accuracy?"

**Common Mistake:** "Use a faster GPU" or "Increase the batch size." A faster GPU won't help if you're bandwidth-bound (Round 1: Roofline Shift), and larger batches increase throughput but not per-request latency.

**Realistic Solution:** Speculative Decoding. You use a tiny, fast "draft" model to guess the next $K$ tokens. You then pass these $K$ tokens to your massive target model in a *single forward pass*. The large model verifies the guesses in parallel (trading spare ALU compute capacity to save memory fetches). All correct tokens are accepted, maintaining identical output distributions but yielding 2-3x speedups.

> **Napkin Math:** Normal decode: 1 token per forward pass, each pass loads all 140 GB of weights from HBM. 100 tokens = 100 weight loads = $100 \times 140\text{GB} / 3.35\text{TB/s} = 4.2$ seconds. Speculative decode with $K=5$ and 80% acceptance: ~100 tokens in ~25 forward passes = $25 \times 140\text{GB} / 3.35\text{TB/s} = 1.04$ seconds. **4× speedup**.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

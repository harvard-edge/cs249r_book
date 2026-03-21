# The Serving Stack

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <b>☁️ Cloud</b> · <a href="../edge/README.md">🤖 Edge</a> · <a href="../mobile/README.md">📱 Mobile</a> · <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

*How you serve models to real users*

Latency budgets, batching strategies, KV-cache management, autoscaling, and speculative decoding — surviving real user traffic at scale.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/cloud/03_serving_stack.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---


### ⏱️ Latency & Throughput


#### 🟢 L3 — Recall & Define

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Serving Inversion</b> · <code>latency</code></summary>

- **Interviewer:** "We took our highly-optimized training architecture (large batches, deep pipelines) and deployed it directly to serving. Now, user requests are timing out. What fundamental priority did we fail to invert?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We need to scale up the serving cluster." More hardware won't fix a design that optimizes for the wrong metric.

  **Realistic Solution:** The shift from maximizing throughput ($T$) to minimizing latency ($L_{lat}$). Training maximizes throughput by using massive batches to keep the GPUs at 100% utilization. Serving minimizes latency because a slow response is a broken product. The batch-heavy architectures that saturate accelerators during training are fundamentally ill-suited for the bursty, latency-critical reality of production traffic.

  > **Napkin Math:** Training batch=256 on H100: 312 TFLOPS × 85% MFU = 265 TFLOPS effective, throughput = 10,000 tokens/sec. Serving batch=1 on same H100: arithmetic intensity drops below ridge point (295 Ops/Byte), becomes memory-bandwidth-bound, effective throughput = 800 tokens/sec but latency = 25ms. The same GPU is 12× more efficient at training but 12× worse at serving — because the optimization target flipped from throughput to latency.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>


#### 🔵 L4 — Apply & Identify

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The LLM Metrics</b> · <code>latency</code> <code>serving</code></summary>

- **Interviewer:** "Our users are complaining the LLM feels 'laggy', but our monitoring shows 'Average Latency' is well under our 2-second SLO. What specific generation metrics are we failing to monitor?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We should look at P99 latency instead of average." P99 helps, but you're still measuring the wrong thing.

  **Realistic Solution:** You are measuring the total request time, which masks the user experience. You must split monitoring into Time-To-First-Token (TTFT) and Time-Per-Output-Token (TPOT). TTFT measures the compute-bound prefill phase (when the user is waiting for the bot to start typing). TPOT measures the memory-bandwidth-bound decode phase (the reading speed). A fast TPOT cannot save a slow TTFT.

  > **Napkin Math:** If TTFT = 3 seconds and TPOT = 20ms, a 200-token response takes $3 + (200 \times 0.02) = 7$ seconds total. Average latency looks fine across short and long requests, but users experience a 3-second blank screen before any text appears — that *feels* broken regardless of total latency.

  📖 **Deep Dive:** [Benchmarking](https://harvard-edge.github.io/cs249r_book_dev/contents/benchmarking/benchmarking.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Shadow GPU Budget</b> · <code>mlops</code> <code>serving</code></summary>

- **Interviewer:** "We want to shadow-test a new 70B LLM before replacing our production 13B model. The plan: mirror all production traffic to the new model, compare outputs, then swap. The infrastructure team comes back and says the shadow deployment will cost more than the production deployment itself. Why is shadow-testing an LLM so much more expensive than shadow-testing a traditional ML model, and how do you make it feasible?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Shadow deployment is free — we're just logging predictions." This is true for a 100ms classification model. It's catastrophically wrong for autoregressive LLMs.

  **Realistic Solution:** Shadow-testing a 70B LLM is expensive because you must run full autoregressive decoding for every request — not just a single forward pass. A classification model shadow costs one forward pass (~10ms, ~0.1 GPU-second). An LLM shadow costs prefill + N decode steps (~2-5 seconds of GPU time per request). At 1,000 QPS production traffic, the shadow needs 2,000-5,000 GPU-seconds per second of wall time — more GPUs than production itself. The fix: sample shadow traffic (10-20% of requests), use speculative decoding with the production 13B as the draft model, or run shadow only on prefill (compare logits at the first token without full generation).

  > **Napkin Math:** Production 13B at 1,000 QPS: ~50 A100s (continuous batching, 20 requests/GPU). Shadow 70B at 1,000 QPS: needs ~200 A100s (5× more params, 4× less efficient batching). Shadow at 10% sampling: 20 A100s — now feasible. Prefill-only shadow (no decode): 5 A100s — cheap enough to run 100%.

  📖 **Deep Dive:** [ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Guardrail Latency Tax</b> · <code>security</code> <code>latency</code></summary>

- **Interviewer:** "After a prompt injection incident, your team adds a safety classifier that screens every LLM input and output. The classifier is a fine-tuned BERT model that takes 15ms per call. Your LLM's TTFT was 200ms and TPOT was 25ms. Users now complain the system feels noticeably slower. The PM says '15ms is nothing.' Why is the PM wrong, and how do you fix it without removing the guardrail?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "15ms is only 7.5% of the 200ms TTFT — users won't notice." This ignores where the 15ms hits in the pipeline.

  **Realistic Solution:** The guardrail runs twice per request (input screening + output screening) and the output screen runs on *every generated token* in streaming mode, not just once. Input screen adds 15ms to TTFT (200ms → 215ms — barely noticeable). But output screening adds 15ms per token chunk, turning TPOT from 25ms to 40ms — a 60% increase in perceived generation speed. For a 200-token response, total time goes from $200 + (200 \times 25) = 5.2s$ to $215 + (200 \times 40) = 8.2s$ — a 58% regression. The fix: batch output screening (check every 10 tokens instead of every token), run the classifier on a dedicated GPU so it doesn't contend with the LLM's KV-cache memory, or use a smaller distilled classifier (2ms instead of 15ms).

  > **Napkin Math:** Per-token screening at 25ms TPOT + 15ms guardrail = 40ms effective TPOT. At 200 tokens: 8.0s decode vs 5.0s without guardrail. Batch-10 screening: 25ms TPOT + 1.5ms amortized = 26.5ms effective TPOT. At 200 tokens: 5.3s decode — nearly invisible overhead.

  📖 **Deep Dive:** [Security and Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Feature Store Consistency Trap</b> · <code>feature-store</code> <code>serving</code></summary>

- **Interviewer:** "Your team builds a feature store to eliminate training-serving skew. The store materializes 2,000 features into an online Redis cluster and an offline Parquet lake. After launch, you discover that 15% of features have different values between the online and offline paths for the same entity at the same timestamp. Your feature store was supposed to solve this exact problem. What went wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "There's a bug in the ETL pipeline" or "Redis and Parquet use different serialization." Both assume a simple data bug — the real issue is architectural: time semantics.

  **Realistic Solution:** The online path computes features at *request time* using the latest state (point-in-time lookup), while the offline path computes features in a batch job that runs hours later using event-time windows. The divergence comes from late-arriving data. A user's "last-7-day purchase count" computed online at 2:00 PM sees transactions up to 1:59 PM. The offline batch job running at midnight replays the same window but now includes transactions that arrived late (payment processor delays, retry queues). The fix is a **dual-write architecture**: compute features once in a streaming engine (Flink/Spark Streaming), write simultaneously to both online (Redis) and offline (Parquet) stores from the same computation. This guarantees bitwise consistency. Additionally, implement a **feature validation pipeline** that samples 1% of online-served features, replays them offline, and alerts on divergence >0.1%.

  > **Napkin Math:** 2,000 features × 10M entities = 20B feature values. Redis cluster for online serving: 20B × 8 bytes avg = 160 GB → 3 nodes of r6g.4xlarge (128 GB each) at $1.60/hr = $3,456/month. Parquet offline: 160 GB compressed (~40 GB) on S3 = $0.92/month. The cost asymmetry is 3,750:1, which is why teams are tempted to compute them differently — and that's where skew creeps in. Streaming dual-write adds ~$800/month in Flink compute but eliminates the 15% skew that was causing a 3% model accuracy drop worth $200k/month in revenue.

  📖 **Deep Dive:** [Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>


#### 🟡 L5 — Analyze & Predict

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Black Friday Collapse</b> · <code>latency</code> <code>queueing</code></summary>

- **Interviewer:** "Our GPU serving cluster handles 1,000 QPS at 50ms latency. On Black Friday, traffic spiked 10x to 10,000 QPS. The system didn't just slow down 10x; it completely collapsed, with latencies hitting 10 seconds and GPUs throwing Out Of Memory errors despite appearing to have free VRAM. Why did the system fail non-linearly, and how does continuous batching on GPUs shift this queueing knee point differently than traditional CPU-based web serving?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The system should degrade linearly — 10x traffic means 10x latency." Queueing systems don't work that way, and GPU memory doesn't degrade gracefully.

  **Realistic Solution:** The collapse is caused by the interaction of queueing theory and GPU memory fragmentation. In a CPU web server, as utilization ($\rho$) passes the "Knee" at ~70%, request queue lengths grow exponentially (per Little's Law). But in an LLM serving cluster, a traffic spike causes the batch scheduler to accept more concurrent requests. Without PagedAttention, each new request statically allocates maximum potential KV-cache in HBM. The VRAM rapidly fragments. The non-linear collapse happens because the GPU spends cycles swapping KV-cache blocks to CPU RAM or simply OOMs, rather than just queueing. Continuous batching shifts this knee point significantly higher (to $\rho \approx 0.85$) because it dynamically slots new requests into the batch at the token level, absorbing the burst into the GPU's compute capacity without requiring new static memory reservations.

  > **Napkin Math:** At $\rho = 0.5$ (50% util): avg requests in system ≈ 1. At $\rho = 0.9$ (90% util): avg requests in system ≈ 9. The relationship is $L = \rho/(1-\rho)$. On a CPU, this just means waiting. On an 80GB H100, if 9 concurrent requests each statically reserve 8GB for a max 8K context window, you need 72GB. You hit the memory wall before you hit the compute wall. Continuous batching + PagedAttention changes the denominator: instead of allocating 8GB per request upfront, it allocates 16MB blocks dynamically. This allows the GPU to sustain the queueing spike by keeping memory utilization proportional to actual tokens generated, not max potential tokens.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Input Chunking Pipeline Bubble</b> · <code>latency</code> <code>serving</code></summary>

- **Interviewer:** "Users submit 100,000-token documents to your summarization model. To avoid OOMing during the prefill phase, you implement 'Chunked Prefill' (breaking the document into 4,000-token blocks). The prefill now succeeds. But users notice that the Time-To-First-Token (TTFT) actually *increased* compared to when you ran the 100k prompt all at once on a bigger GPU. Why does chunking the prompt slow down the time to first token?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Chunking adds Python looping overhead." Python overhead is microseconds; the slowdown is seconds.

  **Realistic Solution:** You destroyed the **Parallel Math Advantage (Arithmetic Intensity)**.

  Attention and feed-forward layers are matrix multiplications.
  When you process 100,000 tokens in a single massive batch, you formulate a massive matrix multiplication `[100k, hidden_dim] * [hidden_dim, hidden_dim]`. GPUs are incredibly efficient at this. The arithmetic intensity is astronomical, pushing the GPU to its absolute theoretical peak TFLOPS (e.g., hitting 900+ TFLOPS on an H100).

  When you chunk the prompt into 25 sequential blocks of 4,000 tokens, you force the GPU to do 25 separate, smaller matrix multiplications.
  1. The arithmetic intensity drops.
  2. The GPU must load the *entire* 140 GB model weights from HBM to the SRAM 25 separate times (once for each chunk).

  Instead of loading the weights once and doing massive math, you load the weights 25 times and do small math. You shifted the workload from compute-bound back toward memory-bandwidth-bound, ruining efficiency.

  **The Fix:** Chunked prefill is a memory-saving compromise, not a speedup. To fix the speed, you must use Tensor Parallelism across multiple GPUs to keep the sequence contiguous, or interleave the chunks of different users (continuous batching) to restore the arithmetic intensity.

  > **Napkin Math:** Single 100k pass: Read 140 GB weights once. Compute 140 TFLOPs. Time = 140T / 900TFLOPS = 0.15s.
  > Chunked 25 passes: Read 140 GB weights 25 times = 3,500 GB of memory traffic. At 3.3 TB/s bandwidth, just *loading* the weights takes 1.06 seconds. TTFT increases by nearly 7x.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>


#### 🔴 L6+ — Synthesize & Derive

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Continuous Batching Stall</b> · <code>queueing-theory</code></summary>

- **Interviewer:** "You implement continuous batching (iteration-level scheduling) for an LLM endpoint to improve throughput. Under low load, TTFT (Time To First Token) is 100ms. Under high load, your throughput triples, but users start complaining that TTFT spikes to over 4 seconds, even though token generation speed (TPOT) remains fast. What scheduling flaw caused this?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming that because continuous batching optimizes VRAM and throughput, it automatically guarantees low latency for incoming requests."

  **Realistic Solution:** You starved the prefill phase. In LLM serving, a request has two phases: Prefill (processing the prompt, compute-bound) and Decode (generating tokens, memory-bound). With continuous batching, an incoming request is injected into the very next iteration of the running batch. However, the prefill phase for a new prompt requires a massive matrix multiplication that consumes almost all the GPU's compute. If you inject a new 2,000-token prompt into a running batch, the GPU must pause decoding for everyone else to crunch that prefill. To prevent this, schedulers often limit the prefill chunk size or delay new prefills. If delayed too aggressively under high load, incoming requests wait in the queue for seconds, destroying TTFT.

  > **Napkin Math:** A 2000-token prefill on a 70B model requires `2 * 70B * 2000 = 280 TFLOPs`. An A100 (FP16) does ~312 TFLOPs. That prefill takes almost `1 full second` of dedicated compute. If 4 users send prompts simultaneously, the 4th user sits in the queue for 3 seconds before their prefill even begins, destroying TTFT, while the active decoders are completely stalled waiting for the prefills to finish.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>


### 🔄 Batching & Scheduling


#### 🔴 L6+ — Synthesize & Derive

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Underutilized Accelerator</b> · <code>continuous-batching</code></summary>

- **Interviewer:** "You're optimizing an LLM inference service running on A100 GPUs. You've already implemented dynamic batching, where requests are grouped together until the GPU queue is full or a timeout is reached. However, under bursty traffic with highly variable input sequence lengths, you still observe significant GPU underutilization and high tail latencies. What is the fundamental limitation of standard dynamic batching in this scenario, and what advanced technique would you propose to maximize GPU throughput and reduce tail latency?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We need to use larger batch sizes." While larger batches increase throughput for homogeneous requests, standard dynamic batching still struggles with variable lengths and bursty traffic, leading to idle GPU cycles and increased tail latency as the system waits for full batches.

  **Realistic Solution:** The fundamental limitation of standard dynamic batching is that it processes requests sequentially within a batch and waits for *all* sequences in a batch to complete before moving to the next. For LLMs, sequences have variable lengths and generate tokens one by one. If one sequence in a batch finishes early, its allocated GPU memory and compute resources remain idle while other, longer sequences in the same batch continue decoding. This leads to **memory fragmentation** and **GPU underutilization** for the entire batch.

  The advanced technique to address this is **Continuous Batching** (or **Iteration-level Batching**), pioneered by systems like vLLM:
    1.  **Token-level Scheduling:** Instead of batching entire requests, continuous batching batches *individual tokens* or *decode steps*. The scheduler dynamically adds new requests to the GPU's processing queue as soon as capacity (memory, compute) becomes available, without waiting for the current batch to fully complete.
    2.  **PagedAttention Integration:** Coupled with PagedAttention (as discussed in the OOMing Generator question), continuous batching efficiently manages KV cache memory. PagedAttention allows non-contiguous memory allocation for KV caches, enabling the GPU to interleave processing of different sequences and reuse memory more effectively.
    3.  **Speculative Decoding:** For very high throughput, combine continuous batching with speculative decoding, where a smaller, faster draft model predicts several future tokens, and the larger model verifies them in parallel.
    4.  **Request Coalescing:** Group multiple small, concurrent requests into a single larger request before processing, if their latency budgets allow.

  This approach maximizes GPU utilization by keeping the GPU busy with useful work, significantly reducing the "bubble" time where the GPU is idle waiting for a batch to fill or for long sequences to finish.

  > **Napkin Math:** Consider a batch of 4 requests, with sequence lengths `[100, 200, 300, 400]` tokens.
  > - **Standard Dynamic Batching:** The GPU waits for the 400-token sequence to complete before starting the next batch. The resources for the 100, 200, 300-token sequences are idle after they complete.
  > - **Continuous Batching:** As soon as the 100-token sequence completes, its resources are freed, and a new incoming request (if available) can immediately start processing on those freed resources, interleaving with the remaining 200, 300, 400-token sequences. This can lead to 2-4x higher throughput and lower tail latency compared to standard dynamic batching.

  > **Key Equation:** `GPU_Utilization = (TotalActiveComputeTime / TotalTime) * 100%`

  📖 **Deep Dive:** [Volume I: Continuous Batching for LLM Inference](https://mlsysbook.ai/vol1/llms/continuous-batching)

  </details>

</details>


### 🗂️ KV-Cache & Memory Management


#### 🟡 L5 — Analyze & Predict

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The OOMing Generator</b> · <code>kv-cache-management</code></summary>

- **Interviewer:** "You're responsible for deploying a large language model (LLM) on a GPU cluster for real-time inference. Under high concurrent user requests, you observe frequent Out-Of-Memory (OOM) errors on the GPUs, despite the total memory of the model weights fitting comfortably. What is the most likely cause of these OOMs, and what advanced technique would you implement to mitigate this?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model weights are too large, we need model parallelism." While true for extremely large models, the prompt states weights fit comfortably. The OOMs under *concurrent requests* point to a per-request memory overhead that scales with the number of users, rather than just the model size itself.

  **Realistic Solution:** The most likely cause is the **Key-Value (KV) Cache** for attention mechanisms. During auto-regressive decoding, each generated token requires re-computing attention over all previous tokens. To optimize this, the Keys and Values of the attention mechanism for previous tokens are cached in GPU memory. This KV cache grows with the sequence length for *each concurrent request*. Under high concurrency, the sum of all individual KV caches quickly exhausts GPU memory.

  The advanced technique to mitigate this is **PagedAttention** (as implemented in vLLM) or similar KV-cache management strategies:
    1.  **PagedAttention:** Analogous to virtual memory in operating systems, PagedAttention breaks the KV cache into fixed-size "blocks." These blocks are non-contiguous in physical GPU memory but are logically contiguous for a sequence. This allows for:
        *   **Efficient Memory Sharing:** Different sequences can share KV cache blocks if they have common prefixes (e.g., in a batch or beam search).
        *   **Reduced Fragmentation:** Memory is allocated in fixed-size pages, reducing fragmentation compared to variable-sized contiguous allocations.
        *   **Dynamic Allocation:** Only allocate KV cache blocks as needed, rather than pre-allocating for the maximum possible sequence length.
    2.  **KV Cache Compression:** Techniques like quantization or pruning applied to the KV cache itself.
    3.  **Offloading:** Moving less frequently accessed parts of the KV cache to CPU memory, though this introduces latency.

  > **Napkin Math:** For a 7B LLM (e.g., Llama-2 with 32 layers, 32 heads, 128 head dim, FP16), the KV cache size per token is `32 layers * 2 (K+V) * 32 heads * 128 dim * 2 bytes/FP16 = 524 KB`.
  > If each request has an average sequence length of 1024 tokens, one request consumes `524 KB/token * 1024 tokens ≈ 0.5 GB`.
  > With 16 concurrent requests, this is `16 * 0.5 GB = 8 GB` of KV cache, quickly saturating a 16GB GPU *on top of model weights*. PagedAttention can reduce this by 2-4x for typical workloads.

  > **Key Equation:** `KV_Cache_Memory_Per_Request = SequenceLength * NumLayers * 2 * NumHeads * HeadDim * BytesPerFloat`

  📖 **Deep Dive:** [Volume I: KV-Cache Optimization for LLMs](https://mlsysbook.ai/vol1/llms/kv-cache-optimization)

  </details>

</details>


### 🏗️ Serving Architecture


#### 🟢 L3 — Recall & Define

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Serverless Freeze</b> · <code>serving</code></summary>

- **Interviewer:** "Our serverless inference endpoint scales down to 0 replicas to save money. However, the first user request after scaling back up takes 30 seconds to process. What is causing this delay?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model needs to warm up its caches." There's no cache warm-up — the delay is purely about data movement.

  **Realistic Solution:** Cold Start Latency. The system must provision the container and load the Python runtime, but most importantly, it must transfer the massive model weights (gigabytes of data) from network storage into the GPU's HBM *before* the first forward pass can execute. You must use persistent warm replicas or optimized weight loading strategies to fix this.

  > **Napkin Math:** A 70B model in FP16 = 140 GB. Network storage throughput (EBS/S3) ≈ 5-10 GB/s. Load time = $140/7.5 \approx 19$ seconds just for weight transfer. Add container startup (~3s) and Python/CUDA init (~5s) = ~27 seconds total cold start.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> The Streaming vs Batch Inference</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "Our chatbot returns the full response only after all tokens are generated. Users say it feels 'broken' — they stare at a blank screen for 8 seconds. A colleague suggests streaming tokens as they're generated. Walk me through the latency math and when streaming actually helps vs. hurts."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Streaming always improves the user experience." Streaming improves *perceived* latency for long responses but adds per-token SSE/WebSocket overhead and complicates error handling. For short responses (<20 tokens), the overhead can make streaming slower than batch.

  **Realistic Solution:** Streaming decouples perceived latency from total latency. The user sees the first token after TTFT (prefill time), then each subsequent token arrives at TPOT intervals. The *perceived* wait is TTFT (~200ms), not the total generation time. But streaming adds: (1) SSE/WebSocket framing overhead (~0.5ms/token), (2) client-side rendering cost, and (3) inability to retry or filter the full response before sending. For short responses, the total time with streaming overhead can exceed the non-streaming total time.

  > **Napkin Math:** 200-token response on H100 serving a 13B model:
  > - **Non-streaming:** TTFT = 150ms, TPOT = 30ms/token. User waits: 150 + (200 × 30) = **6,150ms** before seeing anything.
  > - **Streaming:** User sees first token at 150ms. Full response arrives at same 6,150ms total, but perceived wait = **150ms**. That's a **41× improvement** in perceived responsiveness.
  > - **Short response (10 tokens):** Non-streaming total = 150 + (10 × 30) = 450ms. Streaming: first token at 150ms, but SSE overhead adds ~5ms total. Streaming barely helps — 150ms vs 450ms perceived, but the user reads the full 10-token response in <1 second either way.
  > - **Break-even:** Streaming helps when total generation time > ~1 second (roughly >30 tokens at 30ms/token).

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> The Embedding Model Serving</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "We're deploying a 335M parameter embedding model (like BGE-large) for our search engine. Unlike LLMs, there's no autoregressive decode — it's a single forward pass. The team sized the GPU as if it were an LLM workload and provisioned an H100. Is that the right hardware? What's the optimal batch size?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Embedding models are small, so any GPU works — just use the cheapest one." Embedding models are compute-bound (not memory-bandwidth bound like LLM decode), so the optimization strategy is completely different. An H100 is overkill for a single replica but the wrong framing — you should be asking about throughput per dollar.

  **Realistic Solution:** Embedding inference is a single forward pass with no autoregressive loop. The workload is compute-bound at moderate batch sizes because the full sequence is processed in parallel (like LLM prefill, not decode). This means: (1) batch size directly increases throughput with near-linear scaling until you hit compute saturation, (2) an A10G ($0.75/hr) at high batch sizes can match H100 throughput-per-dollar because you're not paying for unused HBM bandwidth, and (3) CPU inference (with ONNX Runtime) is viable for low-QPS workloads.

  > **Napkin Math:** BGE-large (335M params, FP16 = 670 MB):
  > - **H100 (990 TFLOPS, $3.50/hr):** At BS=1, 512-token input: 2 × 335M × 512 = 343 GFLOPs. Latency = 343G / 990T = **0.35ms**. Throughput = 2,857 embeddings/sec. Cost = $3.50 / (2,857 × 3,600) = **$0.00000034/embedding**.
  > - **A10G (125 TFLOPS, $0.75/hr):** Same workload: latency = 343G / 125T = **2.7ms** at BS=1. At BS=64: 64 × 343G = 22 TFLOPs, latency = 22T / 125T = **176ms** for 64 embeddings = 2.75ms/embedding. Throughput = 364 embeddings/sec. Cost = $0.75 / (364 × 3,600) = **$0.00000057/embedding**.
  > - **H100 is 1.7× cheaper per embedding** but costs 4.7× more per hour. **A10G wins if you need <400 embeddings/sec.** H100 wins at >1,000 embeddings/sec.
  > - **Optimal batch size on A10G:** Throughput plateaus around BS=128–256 where compute utilization hits ~80%. Beyond that, latency increases linearly with no throughput gain.

  📖 **Deep Dive:** [Benchmarking](https://harvard-edge.github.io/cs249r_book_dev/contents/benchmarking/benchmarking.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> The Warm-up Request Strategy</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "We just deployed a new model version. The first inference request takes 12 seconds, but subsequent requests take 200ms. Our health check passed (model loaded successfully), so why is the first real request 60× slower? And how many warm-up requests do we need before routing production traffic?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model needs to 'warm up' its internal caches like a CPU cache." GPU inference doesn't have the same cache-warming semantics as CPU workloads. The slowdown is from one-time initialization costs that happen on the first forward pass, not cache misses.

  **Realistic Solution:** The first-request penalty comes from three sources: (1) **CUDA kernel JIT compilation:** PyTorch/TensorRT compiles and caches GPU kernels on first use. Different input shapes trigger different kernels. (2) **CUDA context and memory pool initialization:** The first `cudaMalloc` initializes the memory allocator, and the first kernel launch initializes the CUDA context on each GPU. (3) **cuBLAS handle and workspace allocation:** The first GEMM operation allocates cuBLAS workspace buffers. After the first request, all of these are cached. You need warm-up requests that cover all expected input shapes (different sequence lengths, batch sizes) to trigger compilation of all kernel variants.

  > **Napkin Math:** First-request breakdown on H100 with a 13B model:
  > - CUDA context init: **~2 seconds** (one-time per GPU).
  > - cuBLAS workspace allocation: **~500ms**.
  > - Kernel JIT compilation (first unique shape): **~3–8 seconds** depending on model complexity and whether using torch.compile or TensorRT.
  > - Actual inference: **200ms**.
  > - Total first request: **~6–11 seconds**.
  >
  > **Warm-up strategy:** Send 3–5 requests with representative input shapes:
  > - Short input (128 tokens) — triggers short-sequence kernels.
  > - Medium input (1024 tokens) — triggers medium-sequence kernels.
  > - Long input (4096 tokens) — triggers long-sequence kernels.
  > - Batch of 8 short inputs — triggers batched kernels.
  > Total warm-up time: ~15–30 seconds. After this, all kernel variants are compiled and cached. **Add warm-up to the readiness probe, not the liveness probe** — the pod is alive but not ready until warm-up completes.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Tokenizer Mismatch</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "Your team fine-tuned a 7B model and deployed it to production. The weights are identical, but the serving cluster is suddenly OOMing under load, and the KV-cache is filling up 30% faster than expected for the exact same text inputs. Why does a tokenizer version mismatch cause a massive GPU memory leak, and how does it destroy your serving economics?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The serving framework must be allocating memory differently" or "There's a memory leak in the Python code." Both ignore the relationship between text, tokens, and hardware memory.

  **Realistic Solution:** The serving pipeline is using an older or different tokenizer than training. If the training pipeline used `tokenizer_v2` (which maps "machine learning" → tokens [4521, 3892], 2 tokens) but the serving pipeline loads `tokenizer_v1` (which maps "machine learning" → [4521, 12, 3892, 44], 4 tokens), the exact same input text produces a longer sequence of token IDs. Because KV-cache memory scales linearly with sequence length, a less efficient tokenizer artificially inflates the memory footprint of every request. The model isn't just producing worse answers; it's physically consuming more VRAM per word of text, causing the batch scheduler to OOM or reject requests much earlier than expected.

  > **Napkin Math:** Typical tokenizer vocabulary: 32,000 tokens. If `v1` is a naive BPE and `v2` is highly optimized for your domain, `v1` might average 1.5 tokens per word while `v2` averages 1.1 tokens per word. For a 1,000-word prompt: `v2` = 1,100 tokens. `v1` = 1,500 tokens. That's a 36% increase in sequence length. For a 7B model (KV-cache = ~1MB per token), the prompt takes 1.1GB with the correct tokenizer, but 1.5GB with the mismatched one. Across a batch of 64 requests, you're wasting 25GB of HBM just storing the extra intermediate states of inefficiently tokenized text. This drops your maximum concurrent requests by 30%, forcing you to scale out the GPU cluster to handle the same QPS.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Cold Start Penalty</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "Your Kubernetes cluster auto-scales model serving pods based on QPS. When traffic spikes, new pods are created, but the first requests to each new pod take 45 seconds instead of the normal 40 ms. Users see timeouts. The pod's readiness probe passes after 30 seconds. What's taking so long, and how do you get that 45 seconds down to under 5?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model is being downloaded from S3 — use a faster network" or "Pre-pull the container image." Both address parts of the problem but miss the dominant cost.

  **Realistic Solution:** The 45-second cold start has four sequential phases: (1) **Container pull** (~5 s if image is cached on the node, ~30 s if not — the container image with CUDA runtime and framework is 15–25 GB); (2) **Model weight loading** (~10–15 s — reading 14 GB of FP16 weights from network storage into CPU RAM); (3) **GPU transfer** (~3–5 s — copying 14 GB from CPU RAM to GPU VRAM over PCIe Gen4 at ~25 GB/s effective); (4) **CUDA/cuDNN warmup** (~5–10 s — first inference triggers JIT compilation of CUDA kernels and cuDNN autotuning). The readiness probe passes after phase 3 (model is on GPU), but phase 4 means the first real request still pays the warmup penalty. Fixes: (1) pre-cache container images on all nodes using a DaemonSet; (2) store model weights on local NVMe (7 GB/s read) instead of network storage (1 GB/s); (3) use CUDA graph capture during startup to pre-compile kernels; (4) send synthetic warmup requests before marking the pod as ready. Combined: cold start drops from 45 s to ~4 s (0 s image pull + 2 s NVMe load + 0.6 s GPU transfer + 1.5 s warmup).

  > **Napkin Math:** **Current breakdown:** Image pull (cached): **5 s**. Weight load from NFS (14 GB at 1 GB/s): **14 s**. CPU→GPU transfer (14 GB at 25 GB/s PCIe): **0.56 s**. CUDA warmup (kernel JIT + cuDNN autotune): **8 s**. Readiness probe delay: **2 s**. Total: **~30 s** to ready + **15 s** first-request warmup = **45 s** effective. **Optimized:** Image pre-cached: **0 s**. Weight load from local NVMe (14 GB at 7 GB/s): **2 s**. GPU transfer: **0.56 s**. Pre-compiled CUDA graphs (loaded from cache): **1.5 s**. Warmup requests (3 synthetic): **0.12 s**. Total: **~4.2 s**. At 100 scale-up events/day, saving 41 s each: **4,100 s** of reduced user-facing latency. More importantly: no more timeouts during traffic spikes.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Normalization Mismatch</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "Your image classification model achieves 91% accuracy in the training notebook but only 72% when deployed to a Flask serving endpoint on the same A100 GPU. Same model weights, same test images (you verified by saving the raw bytes). The training team uses PyTorch, the serving team uses ONNX Runtime. What's the most likely preprocessing bug?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "ONNX Runtime must have a conversion bug" or "The model needs to be re-exported." ONNX conversion for standard architectures is well-tested and unlikely to cause a 19-point drop. The model is fine — the *input* is wrong.

  **Realistic Solution:** The serving pipeline applies different input normalization than training. The most common variant: the training pipeline normalizes images to [0, 1] range and then applies ImageNet mean/std normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). The serving pipeline either: (1) forgets the mean/std normalization entirely (passes [0, 1] values directly); (2) applies normalization but in the wrong channel order (RGB vs BGR — OpenCV loads BGR by default, PIL loads RGB); or (3) normalizes to [-1, 1] instead of using ImageNet statistics (a common TensorFlow convention applied to a PyTorch-trained model). Any of these produces inputs that are statistically different from what the model saw during training. The model still gets *some* predictions right (72%) because the spatial structure of the image is preserved — the model can still detect edges and shapes — but the magnitude and distribution of activations are wrong. Fix: extract the exact preprocessing pipeline from the training code (including library-specific defaults) and replicate it byte-for-byte in the serving code. Add an assertion that compares the preprocessed tensor's mean and std against expected values.

  > **Napkin Math:** ImageNet normalization: pixel value 128 (mid-gray) → $(128/255 - 0.485) / 0.229 = (0.502 - 0.485) / 0.229 = 0.074$. Without normalization: the model receives 0.502 instead of 0.074 — a **6.8× magnitude error**. With BGR instead of RGB: the red channel's mean (0.485) is applied to the blue channel (which has mean 0.406). Error per pixel: $(0.406 - 0.485) / 0.229 = -0.345$ systematic bias in the blue channel. Over a 224×224 image: every one of the 50,176 pixels has a ~0.3 bias — the model sees a systematically tinted image. For a model trained on correctly normalized inputs, this is equivalent to adding a colored filter to every image. The 72% accuracy (vs 91%) means the model is robust enough to classify ~79% of "easy" images despite the bias, but fails on the ~21% that require precise color or intensity discrimination.

  📖 **Deep Dive:** [Volume I: Model Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>


#### 🔵 L4 — Apply & Identify

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Pre-computation Trade-off</b> · <code>serving</code></summary>

- **Interviewer:** "We are deploying a photo classification model. Running it in real-time on user upload costs us $10,000 a month in GPU compute. How do we reduce costs without losing model accuracy?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Quantize the model to INT8 to halve the compute cost." Quantization helps, but there's a 10x cheaper architectural change.

  **Realistic Solution:** Shift from Dynamic (Real-time) to Static (Batch) inference. The reason batch inference is dramatically cheaper isn't just about amortizing GPU idle time — it's about arithmetic intensity. At batch=1, each forward pass loads the full model weights from HBM but performs very little compute per byte moved, placing the workload deep in the memory-bound regime of the Roofline. At batch=256, the same weight load is reused across 256 inputs, pushing arithmetic intensity above the ridge point into the compute-bound regime where the GPU's ALUs are fully saturated. Run the model periodically (e.g., overnight) on cheap spot instances with large batches, and store results in a fast key-value cache (Redis) for millisecond serving.

  > **Napkin Math:** Batch=1 inference: arithmetic intensity ~1 Op/Byte (memory-bound, 5% GPU utilization). Batch=256 inference: arithmetic intensity ~200 Ops/Byte (compute-bound, 70% GPU utilization). Same model, same GPU — **14× better utilization** just from batching. Real-time: 1 GPU reserved 24/7 at $2/hr = $1,440/month at ~10% average utilization. Batch: same workload in 2 hours overnight on spot instances at $0.70/hr = $1.40/day = $42/month at 70%+ utilization. That's a **34× cost reduction**.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Token Budget Economics</b> · <code>serving</code> <code>economics</code></summary>

- **Interviewer:** "Your company is choosing between H100, A100, and TPU v5e for serving a 70B parameter LLM. The CFO wants a cost-per-million-tokens comparison across these accelerators at realistic batch sizes. Walk me through the napkin math and tell me which hardware wins — and when."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Pick the GPU with the highest FLOPS — H100 always wins." Raw FLOPS is irrelevant for decode-phase serving, which is memory-bandwidth bound. The cheapest $/token depends on batch size, model size, and whether you're prefill-bound or decode-bound.

  **Realistic Solution:** LLM decode is memory-bandwidth bound: each token requires loading all model weights from HBM once. The metric that matters is $/GB/s of memory bandwidth, not $/TFLOP. At small batch sizes, the accelerator with the best bandwidth-per-dollar wins. At large batch sizes, the workload shifts compute-bound and FLOPS/$ matters more. TPU v5e wins on raw $/token at high batch sizes due to lower per-chip cost; H100 wins on latency-sensitive low-batch workloads due to 3.35 TB/s HBM3 bandwidth.

  > **Napkin Math:** 70B model in FP16 = 140 GB weights. Each decode token loads 140 GB.
  > - **H100 (80 GB HBM3, 3.35 TB/s):** Decode token = 140 GB / 3.35 TB/s = 42 ms at BS=1. On-demand ~$3.50/hr → $3.50 / (3600s / 0.042s) = $0.000041/token → **$41/M tokens** at BS=1. At BS=32, throughput scales ~25×, cost drops to **~$1.60/M tokens**.
  > - **A100 (80 GB HBM2e, 2.0 TB/s):** Decode token = 140 GB / 2.0 TB/s = 70 ms at BS=1. On-demand ~$2.20/hr → **$43/M tokens** at BS=1. At BS=32, cost drops to **~$2.10/M tokens** (less compute headroom than H100).
  > - **TPU v5e (16 GB HBM, $1.20/chip/hr):** Requires 16-way tensor parallelism to fit 140 GB. Aggregate bandwidth = 16 × 819 GB/s = 13.1 TB/s. Decode = 140 GB / 13.1 TB/s = 10.7 ms at BS=1. Cost = 16 × $1.20 = $19.20/hr → **$57/M tokens** at BS=1. At BS=64, throughput scales, cost drops to **~$2.50/M tokens**.
  >
  > H100 wins across the board at these prices ($3.50/hr) because it delivers more bandwidth-per-dollar (957 GB/s per $) and more compute-per-dollar (282 TFLOPS/$) than both the A100 and TPU v5e.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Inference Server Autoscaling</b> · <code>serving</code> <code>economics</code></summary>

- **Interviewer:** "Our LLM endpoint on Kubernetes has a scale-to-zero policy to save money overnight. But the first morning request takes 45 seconds. The SRE team proposes keeping 1 warm replica 24/7. The finance team says that wastes $2,500/month. Who's right? Show me the break-even math."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Always keep warm replicas — cold starts are unacceptable." This ignores workloads with genuinely sparse traffic where paying for 24/7 idle GPUs is pure waste.

  **Realistic Solution:** The answer depends on the traffic pattern. For bursty workloads with predictable schedules (e.g., business-hours traffic), use scheduled scaling — spin up replicas 15 minutes before peak and scale to zero overnight. For unpredictable but sparse traffic, use a tiered approach: keep a small quantized model (7B on CPU or A10G) as a "warm fallback" while the full model cold-starts, then route to the full model once it's ready.

  > **Napkin Math:** 70B model on 2× H100:
  > - **Cold start breakdown:** Container pull (5s) + weight download from S3 (140 GB / 10 GB/s = 14s) + CUDA context init (3s) + model load to GPU (140 GB / 64 GB/s PCIe = 2.2s) + warmup inference (2s) = **~26 seconds**.
  > - **1 warm replica 24/7:** 2 × $3.50/hr × 730 hr = **$5,110/month**.
  > - **Scale-to-zero, 50 cold starts/day:** 50 × 26s = 1,300s of user-facing delay/day. If each delayed request costs $0.50 in user churn: 50 × $0.50 × 30 = **$750/month in churn cost**.
  > - **Break-even:** Warm replica costs $5,110 vs cold-start churn costs $750. Scale-to-zero wins **unless** cold-start churn exceeds ~$170/day (340 impacted requests/day).
  > - **Hybrid:** Keep 1 A10G ($548/month) running a 7B fallback model. Cold-start the H100s on first request. Users get a fast (if lower-quality) response in 200ms while the full model loads. Total cost: $548/month + near-zero churn.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Triton Inference Server Ensemble</b> · <code>serving</code> <code>deployment</code></summary>

- **Interviewer:** "We're deploying a RAG pipeline: text preprocessing (tokenization + embedding lookup) → retrieval model → LLM generation → JSON postprocessing. An engineer wants to chain these as a Triton Inference Server ensemble. Another says 'just put it all in one Python process.' Compare the latency and throughput of both approaches."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The ensemble adds network hops between stages, so the monolithic approach is always faster." Triton ensembles use shared-memory IPC, not network calls, and the pipeline parallelism enables higher throughput.

  **Realistic Solution:** A Triton ensemble chains models as a DAG within a single server process, passing tensors via shared memory (zero-copy on GPU). The latency for a single request is slightly higher than monolithic (~1–2ms overhead per stage for scheduling), but throughput is dramatically higher because stages run as a pipeline: while the LLM generates tokens for request N, the preprocessor handles request N+1. The monolithic approach serializes everything, leaving the GPU idle during CPU-bound pre/postprocessing.

  > **Napkin Math:** RAG pipeline stages:
  > - Preprocessing (CPU): 5ms
  > - Embedding retrieval (GPU): 3ms
  > - LLM generation (GPU): 200ms
  > - Postprocessing (CPU): 2ms
  >
  > **Monolithic (serial):** Total = 5 + 3 + 200 + 2 = **210ms/request**. Throughput = 1000/210 = **4.8 req/s**. GPU utilization = 203/210 = 97% (looks good for single request, but CPU stages block the pipeline).
  >
  > **Triton ensemble (pipelined):** Single-request latency = 210 + 3ms scheduling overhead = **213ms**. But pipeline throughput: bottleneck = LLM at 200ms → **5.0 req/s per pipeline**. With 4 LLM instances: CPU stages overlap with GPU, throughput = **~19 req/s**. The monolithic approach can't overlap CPU and GPU stages, capping at 4.8 req/s regardless of GPU count.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Structured Output Constraint</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "Our API must return valid JSON for downstream systems. We're using constrained decoding (grammar-guided generation) to guarantee valid JSON output. The team reports that constrained decoding adds 15ms per token on top of the normal 30ms TPOT. For a 500-token JSON response, that's 7.5 extra seconds. Is this overhead acceptable, and can we reduce it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Constrained decoding is always worth the overhead because it guarantees valid output." For long structured outputs, the cumulative overhead can exceed the cost of simply retrying on parse failure.

  **Realistic Solution:** Constrained decoding works by masking invalid tokens at each step according to a grammar (e.g., JSON CFG). The overhead comes from: (1) computing the valid token mask by traversing the grammar state machine against the full vocabulary (~32K–128K tokens), and (2) applying the mask before sampling. The 15ms/token overhead is typical for naive implementations. Optimizations include: pre-computing grammar state transitions into a lookup table, using a compressed vocabulary trie, or using speculative grammar checking (validate only the top-K tokens instead of the full vocabulary). Alternatively, for simple schemas, use structured output mode with retry: generate unconstrained, parse, retry on failure — cheaper if failure rate is <10%.

  > **Napkin Math:** 500-token JSON response on H100:
  > - **Unconstrained:** 500 × 30ms = **15 seconds**.
  > - **Constrained (naive):** 500 × (30 + 15)ms = **22.5 seconds**. 50% overhead.
  > - **Constrained (optimized, pre-computed masks):** 500 × (30 + 2)ms = **16 seconds**. 7% overhead.
  > - **Retry strategy:** If 5% of unconstrained outputs are invalid JSON, expected cost = 15s × 1.05 = **15.75 seconds** average. Cheaper than even optimized constrained decoding, but with 5% of requests taking 30s (2 attempts).
  > - **Break-even:** Constrained decoding wins when JSON failure rate > overhead%. With optimized masks (7% overhead), constrained wins if failure rate > 7%. With naive masks (50% overhead), retry wins unless failure rate > 50%.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The A/B Testing at Scale</b> · <code>serving</code> <code>mlops</code></summary>

- **Interviewer:** "We want to A/B test our current 13B model against a new 70B model in shadow mode. The product team says 'just mirror 100% of traffic to both models for a week to gather data.' Why is this statistically unnecessary but infrastructure-ruinous, and how does the GPU memory asymmetry between these models dictate your A/B testing architecture?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just spin up enough 70B instances to handle the mirrored traffic." This ignores the massive non-linear cost of serving a 70B model and the fact that statistical significance requires a fraction of that traffic.

  **Realistic Solution:** The infrastructure cost of an A/B test is dominated by hardware provisioning, not the statistical sample size. A 13B model fits on a single A100 (40GB) with room for KV-cache. A 70B model requires tensor parallelism across at least 4× A100s (80GB) or 2× H100s just to fit the weights, plus massive KV-cache overhead. Mirroring 100% of traffic means provisioning a 70B cluster that is 5-8× larger and more expensive than your production 13B cluster, just for a test. Instead, use an asymmetric traffic split (e.g., 95/5) or sample-based shadow routing. You only need to route enough traffic to the 70B model to reach statistical significance, which is typically a few thousand requests.

  > **Napkin Math:**
  > - **Memory Asymmetry:** 13B FP16 = 26GB weights. Fits on 1× A100 (40GB) with 14GB for KV-cache. 70B FP16 = 140GB weights. Requires 2× H100 (80GB) or 4× A100 (40GB). The 70B model requires 4× the GPUs per replica.
  > - **Traffic Mirroring Cost:** If production is 1,000 QPS running on 50× A100s, mirroring 100% to 70B would require ~200-300 GPUs due to lower throughput and higher memory footprint. Cost: ~$500k/week.
  > - **Statistical Reality:** To detect a 3% quality improvement with 80% power, you need ~3,200 samples per arm. At 1,000 QPS, you collect 3,200 samples in 3.2 seconds.
  > - **Optimized Architecture:** Route 99.9% of traffic to 13B, and 0.1% (1 QPS) to a single minimal 70B deployment (2× H100s). You gather the required 3,200 samples in under an hour, costing ~$10 in GPU compute instead of $500k.

  📖 **Deep Dive:** [Benchmarking](https://harvard-edge.github.io/cs249r_book_dev/contents/benchmarking/benchmarking.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Rate Limiting Design</b> · <code>serving</code> <code>economics</code></summary>

- **Interviewer:** "Our LLM API rate-limits users to 100 requests per minute. A power user sends 100 requests, each with a 50,000-token prompt and requesting 4,000 output tokens. A casual user sends 100 requests with 100-token prompts and 50 output tokens. Both hit the same rate limit, but one costs 1,000× more GPU time. How do you design a fair rate limiter?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Request-based rate limiting is fine — all requests are equal." This is the fundamental error. A request with 50K input tokens consumes ~500× more prefill compute and ~80× more decode compute than a 100-token request. Request-count rate limiting lets heavy users subsidize their costs with the same limits as light users.

  **Realistic Solution:** Implement token-based rate limiting with separate budgets for input tokens (prefill cost) and output tokens (decode cost). Each user gets a token budget per time window (e.g., 1M input tokens + 100K output tokens per minute). This aligns rate limiting with actual resource consumption. Weight input and output tokens differently because prefill is compute-bound (cheaper per token at high batch sizes) while decode is memory-bandwidth-bound (expensive per token regardless of batch size).

  > **Napkin Math:** 70B model on H100:
  > - **Prefill cost per token:** 2 × 70B = 140 GFLOPs per token. At 990 TFLOPS: 0.14ms/token.
  > - **Decode cost per token:** Load 140 GB weights from HBM. At 3.35 TB/s: 42ms/token (at BS=1).
  > - **Power user (50K input + 4K output per request, 100 requests):**
  >   - Prefill: 100 × 50,000 × 0.14ms = **700 seconds** of compute.
  >   - Decode: 100 × 4,000 × 42ms = **16,800 seconds** of compute (at BS=1; batching helps).
  > - **Casual user (100 input + 50 output per request, 100 requests):**
  >   - Prefill: 100 × 100 × 0.14ms = **1.4 seconds**.
  >   - Decode: 100 × 50 × 42ms = **210 seconds**.
  > - **Cost ratio:** Power user consumes (700 + 16,800) / (1.4 + 210) = **83× more GPU time** for the same number of requests.
  > - **Token-based limit:** Give both users 500K input tokens/min + 50K output tokens/min. Power user can send ~10 requests. Casual user can send all 100. Fair.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Structured Output Parsing Tax</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "You need your LLM to output strict JSON. You implement a constrained decoding wrapper (like Guidance or Outlines) that forces the model to only select tokens valid under your JSON schema. The outputs are now perfect JSON, but the Time-Per-Output-Token (TPOT) doubled. What is the CPU doing that takes so much time?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model has to think harder to generate JSON." The model is doing the exact same neural network math. The delay is outside the model.

  **Realistic Solution:** You are paying the **Vocabulary Masking Overhead**.

  Constrained decoding works by intercepting the logits (the probabilities for all 32,000+ words in the vocabulary) right before sampling. The CPU must evaluate the current state of the JSON string against a complex Regular Expression or Context-Free Grammar.

  For every single token generated, the CPU must loop through all 32,000 vocabulary words, check if adding that word violates the JSON schema, and if so, force its probability to zero (masking).

  If this masking logic is written in pure Python or uses inefficient regex state machines, checking 32,000 strings takes 20-30 milliseconds. If the GPU only takes 15ms to do the forward pass, you have effectively bottlenecked your trillion-dollar GPU behind a slow Python text-parsing script.

  **The Fix:**
  1. Use pre-compiled FSMs (Finite State Machines) written in C++/Rust (like the latest versions of Outlines).
  2. Push the masking logic down to the GPU as a custom CUDA kernel so the logits never have to be copied back to the CPU for evaluation.

  > **Napkin Math:** GPU forward pass: 15ms.
  > Logit transfer to CPU (32k floats): 1ms.
  > Python Regex over 32k strings: 25ms.
  > Total TPOT = 41ms. The CPU parsing text took nearly double the time it took the GPU to run 70 billion parameters.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Scheduler</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "We serve a 13B model with static batching: we collect 32 requests, run them together, and return all results when the longest sequence finishes. Average latency is 4 seconds, but P99 is 12 seconds. The team wants to try vLLM's continuous batching. Explain the systems mechanism and quantify the improvement."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Continuous batching just means we use a bigger batch size, so throughput goes up." This misses the fundamental scheduling innovation — it's not about batch *size*, it's about batch *dynamics*.

  **Realistic Solution:** In static batching, all requests in a batch must wait for the longest sequence to finish generating. If 31 requests finish in 2 seconds but one request generates a 500-token response taking 10 seconds, all 31 completed requests sit in memory waiting — wasting both GPU cycles (padding with no-ops) and user time. Continuous batching (also called iteration-level scheduling) operates at the granularity of individual decode steps. After each token generation step, the scheduler can: (1) evict completed requests immediately (freeing their KV-cache memory); (2) admit new requests from the queue into the freed slots. The GPU is never idle and never wasting cycles on completed requests. The result: short requests return immediately upon completion (latency improvement), and the freed memory slots are instantly reused for new requests (throughput improvement).

  > **Napkin Math:** **Static batching (batch=32):** Sequence lengths vary: 20% generate 50 tokens (~1 s), 60% generate 200 tokens (~4 s), 20% generate 500 tokens (~10 s). All 32 requests wait for the slowest: **10 s** per batch. Throughput: 32 requests / 10 s = **3.2 req/s**. GPU utilization: after 1 s, 6 requests are done but still occupying slots → 19% waste. After 4 s, 26 requests done → 81% waste. Average utilization: ~**40%**. **Continuous batching:** Short requests (50 tokens) complete in **1 s** and exit immediately. Their slots are filled by new requests from the queue. Medium requests complete in **4 s**. Long requests complete in **10 s**. Average latency: $0.2 \times 1 + 0.6 \times 4 + 0.2 \times 10 = $ **4.6 s** (vs 10 s static). But crucially, the GPU processes new requests in freed slots: effective throughput ≈ 32 slots × continuous fill ≈ **7-8 req/s** — a **2-2.5× throughput improvement**. P99 drops from 12 s to ~10 s (the inherent generation time), and P50 drops from 4 s to ~2 s.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The GC Pause Latency Spike</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "Your Python-based inference server on an A100 serves a 7B model with P50 latency of 40 ms. But every ~30 seconds, a burst of requests hits P99 latency of 800 ms — 20× the normal. The GPU utilization trace shows brief drops to 0% that correlate exactly with the spikes. The model and batch size haven't changed. What's stealing 760 ms from your GPU?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The GPU is overheating and throttling" or "There's network congestion from other services." Thermal throttling is gradual (not a sharp 0% drop), and network issues wouldn't zero out GPU utilization.

  **Realistic Solution:** This is a CPython garbage collector (GC) stop-the-world pause. Python's cyclic GC (triggered by `gc.collect()` or automatically when the generation-2 threshold is reached) freezes *all* Python threads while it traces object references. During this pause, no new CUDA kernels can be launched because the Python threads that submit them are frozen. The GPU drains its kernel queue in ~1–2 ms and then sits idle until the GC completes. With a large Python heap (serving frameworks maintain request queues, tokenizer caches, and KV-cache metadata as Python objects), a gen-2 collection can take 500–800 ms. The ~30-second interval matches the default gen-2 threshold: after 10 gen-1 collections (each triggered by ~700 allocations), a gen-2 sweep runs. Fix: (1) disable automatic GC (`gc.disable()`) and run `gc.collect()` explicitly during low-traffic windows; (2) reduce Python-side object churn by pre-allocating buffers; (3) move the hot path to C++/Rust (like vLLM's C++ scheduler) so the GC has fewer objects to trace.

  > **Napkin Math:** Python heap for a serving process: ~2–4 GB of Python objects (request metadata, tokenizer vocab, batch scheduler state). Gen-2 GC traces all objects: at ~5M objects, tracing at ~8M objects/s takes **625 ms**. During this pause: GPU kernel queue depth ≈ 20 kernels × 0.05 ms each = drains in **1 ms**. GPU sits idle for remaining **624 ms**. Requests arriving during the pause queue up: at 100 req/s, ~62 requests are delayed. Each sees an additional 624 ms latency → P99 spike. After disabling auto-GC and running manual collection every 60 s during a scheduled 50 ms micro-pause (by reducing heap to <500K objects via C++ offload): GC time drops to **12 ms**, P99 drops to **52 ms**.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The TensorRT Incompatibility</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "Friday evening, the ops team updates the NVIDIA driver from 535.104 to 545.23 across your serving fleet for a security patch. Monday morning, all TensorRT inference engines fail to load with `INVALID_CONFIG` errors. The model files haven't changed. Rolling back the driver fixes it. How can a driver update break a model file, and what's the correct deployment practice?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "TensorRT models are hardware-independent — a driver update shouldn't affect them" or "Just rebuild the TensorRT engines on the new driver." The first is wrong; the second is the fix but misses *why* and how to prevent recurrence.

  **Realistic Solution:** TensorRT engines are *not* portable across driver versions. When you build a TensorRT engine, the builder selects specific kernel implementations (called "tactics") from cuDNN and cuBLAS based on the current GPU architecture *and* the installed library versions. These tactics are compiled into the serialized engine file as binary GPU code (SASS). A driver update changes the cuDNN/cuBLAS libraries bundled with the driver, which may deprecate or rename tactics that the engine references. When the engine tries to load a tactic that no longer exists in the new driver's library, it fails with `INVALID_CONFIG`. This is by design — TensorRT trades portability for performance by baking in hardware-specific optimizations. The correct practice: (1) pin the driver version in your container image, not the host; (2) store TensorRT engines with metadata (driver version, CUDA version, GPU architecture) and rebuild automatically when any component changes; (3) maintain a CI pipeline that rebuilds engines on driver update and validates accuracy before fleet rollout.

  > **Napkin Math:** TensorRT engine build time for a 13B model: **15–45 minutes** (profiling thousands of tactic combinations). Engine file size: **~28 GB** (2× the FP16 weights due to embedded tactic metadata and workspace allocations). Rebuilding across a 100-GPU fleet: if engines are built per-GPU, that's 100 × 30 min = **50 GPU-hours** of downtime. With a centralized build (one engine per GPU architecture, copied to all): 30 min build + 5 min distribution = **35 minutes** total downtime. The Friday driver update without engine rebuild caused **~60 hours** of serving downtime (Friday 6 PM to Monday 8 AM). At $0.50/request and 1000 req/s: lost revenue = $0.50 × 1000 × 60 × 3600 = **$108M** — a catastrophic incident from a "routine" update. Prevention: driver updates go through a staging environment that automatically rebuilds TensorRT engines and runs accuracy validation before promoting to production.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The BatchNorm Drift</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "Your image classification model (ResNet-50 on A100) was deployed 6 months ago with 94% accuracy. Without any model updates, accuracy has gradually degraded to 87%. The model weights are identical to deployment day — you verified the checkpoint hash. The training team says the model is fine. What's silently changing?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model must have been accidentally updated" or "This is just random variance in the test set." The checkpoint hash verification rules out the first, and a 7-point drop over 6 months is a systematic trend, not noise.

  **Realistic Solution:** This is data distribution drift interacting with Batch Normalization's frozen statistics. During training, BatchNorm layers compute running mean ($\mu$) and running variance ($\sigma^2$) from the training data. At inference time, these frozen statistics are used to normalize inputs: $\hat{x} = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$. If the production data distribution shifts over time (lighting conditions change seasonally, camera firmware updates alter preprocessing, new product categories are added), the true mean and variance diverge from the frozen statistics. The normalization now *distorts* the data instead of normalizing it — pushing activations into regions the downstream layers never saw during training. The gradual 7-point drop over 6 months matches a slow seasonal drift (e.g., winter lighting vs summer lighting for a retail image classifier). Fix: (1) implement periodic BatchNorm recalibration — run a forward pass over recent production data with BatchNorm in training mode to update running statistics, then freeze again; (2) replace BatchNorm with LayerNorm or GroupNorm, which compute statistics per-sample and are immune to distribution drift; (3) deploy a data drift monitor that tracks input feature statistics and alerts when they diverge from training distribution.

  > **Napkin Math:** ResNet-50 has 53 BatchNorm layers. Training data mean pixel value: $\mu_{\text{train}} = 0.485$ (ImageNet). After 6 months of drift (new camera firmware increases brightness): $\mu_{\text{prod}} = 0.52$. Shift = $0.035$. BatchNorm normalizes: $\hat{x} = (0.52 - 0.485) / 0.229 = 0.153$ — the network sees a constant bias of 0.153 added to every activation. Through 53 layers with ReLU, this bias compounds: early layers clip negative activations that should have been positive (or vice versa), changing ~**3–8% of activation signs** per layer. By the final layer, the cumulative effect shifts the feature representation enough to flip predictions for **7% of samples** — matching the 94% → 87% accuracy drop. BatchNorm recalibration: forward pass of 10,000 production images through ResNet-50 on A100: $10000 \times 8\text{ GFLOPs} = 80\text{ TFLOPs}$. At 989 TFLOPS: **0.08 seconds**. Running this weekly costs essentially nothing and prevents the drift.

  📖 **Deep Dive:** [Volume I: Model Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The One-Replica Meltdown</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "You have 8 replicas of a 7B model behind an L7 load balancer, each on its own A100. Monitoring shows total QPS is normal, but P99 latency spiked from 80 ms to 2.4 seconds. Seven replicas report healthy latency. One replica's average latency is 18 seconds. The load balancer is configured for round-robin. Why is one slow replica destroying the P99 for the entire fleet?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Round-robin distributes traffic evenly, so the slow replica only affects 1/8 of requests — P99 should only increase slightly." This misunderstands how percentiles work with heterogeneous backends.

  **Realistic Solution:** Round-robin sends exactly 1/8 of all requests to the slow replica. P99 means 1% of requests exceed the threshold. If 12.5% of requests (those hitting the slow replica) have 18-second latency, then *all* of those requests are in the worst 12.5% — far exceeding the 1% P99 threshold. The P99 is entirely determined by the slow replica. Worse, the slow replica creates *cascading* damage: (1) clients waiting 18 seconds for the slow replica hold open HTTP connections, consuming connection pool slots on the load balancer; (2) if clients have a 5-second timeout and retry, the retried request has a 1/8 chance of hitting the slow replica again, amplifying load; (3) the slow replica's request queue grows, making each subsequent request even slower (queuing theory: latency → ∞ as utilization → 1). Fix: (1) switch from round-robin to least-connections or latency-weighted routing — the slow replica naturally receives fewer requests; (2) implement health-check-based ejection: if a replica's P50 exceeds 2× the fleet median, remove it from the pool; (3) add client-side hedging: send the request to 2 replicas simultaneously and take the first response.

  > **Napkin Math:** 8 replicas, round-robin, total 1000 QPS → 125 QPS per replica. 7 healthy replicas: latency ~40 ms (P50), ~80 ms (P99). 1 slow replica: latency ~18 s. **P99 calculation:** Sort all 1000 requests by latency. The slowest 1% = 10 requests. The slow replica handles 125 requests — *all* 125 are slower than any healthy replica's request. P99 = the 10th-slowest request, which is from the slow replica: **~18 s**. Even P87.5 is terrible: the 125th-slowest request is still from the slow replica. **With least-connections routing:** The slow replica accumulates connections (each held for 18 s). At steady state: slow replica has $125 \times 18 = 2250$ open connections. Healthy replicas have $125 \times 0.04 = 5$ each. Least-connections routes new requests to healthy replicas (5 connections) instead of the slow one (2250). The slow replica effectively drains to ~1 QPS. P99 drops back to **~80 ms** because <0.1% of traffic hits the slow replica.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The CPU Preprocessing Bottleneck</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "Your LLM serving endpoint on an H100 should deliver 40 tokens/s for a 13B model (bandwidth-bound: 26 GB weights / 3.35 TB/s ≈ 7.8 ms/token → 128 tokens/s theoretical). But you're measuring only 15 tokens/s. `nvidia-smi` shows GPU utilization flickering between 0% and 95% in a sawtooth pattern. CPU utilization is pegged at 100% on one core. Where's the bottleneck?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model is too large for the GPU — we need tensor parallelism" or "The batch size is too small." The model fits fine (26 GB on 80 GB), and the sawtooth GPU pattern reveals the real issue.

  **Realistic Solution:** The sawtooth pattern (0% → 95% → 0% → 95%) means the GPU is alternating between waiting for input and processing it. The CPU is the bottleneck — specifically, the tokenizer and/or detokenizer running in Python on a single core. The Python GIL ensures only one thread executes Python bytecode at a time. The serving pipeline is: (1) CPU tokenizes input (Python, single-threaded) → (2) GPU runs model forward pass → (3) CPU detokenizes output token (Python, single-threaded) → (4) CPU samples next token → repeat. If tokenization + detokenization + sampling takes 50 ms on CPU, but the GPU forward pass takes only 7.8 ms, the GPU is idle for $50 / (50 + 7.8) = 87\%$ of the time. Effective throughput: $1000 / (50 + 7.8) = 17.3$ tokens/s — matching the observed 15 tokens/s (with overhead). Fix: (1) use a Rust/C++ tokenizer (HuggingFace `tokenizers` library) that's 10–50× faster than pure Python; (2) batch tokenization — tokenize multiple requests simultaneously; (3) move sampling to GPU (`torch.multinomial` on CUDA); (4) use a C++ serving runtime (TensorRT-LLM, vLLM's C++ backend) that eliminates the Python hot path.

  > **Napkin Math:** **CPU tokenizer (Python `transformers`):** Tokenizing 512 tokens: ~**25 ms** (pure Python, regex-heavy). Detokenizing 1 token: ~**0.5 ms**. Sampling (Python `torch.multinomial` on CPU): ~**2 ms** (includes GPU→CPU transfer of logits). Total CPU per token: **~27.5 ms**. GPU forward pass: **7.8 ms**. Pipeline: $27.5 + 7.8 = 35.3$ ms/token → **28.3 tokens/s** theoretical, ~**15 tokens/s** with Python overhead and GIL contention. **After optimization:** Rust tokenizer: **0.5 ms** (50× faster). GPU-side sampling: **0.1 ms** (no transfer). Detokenize: **0.05 ms** (Rust). Total CPU per token: **0.65 ms**. Pipeline: $0.65 + 7.8 = 8.45$ ms/token → **118 tokens/s** — a **7.9× improvement**, now approaching the GPU's theoretical bandwidth limit.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The GIL Bottleneck</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "Your inference server runs 4 model replicas on 4 A100 GPUs within a single Python process using threads. Each replica should handle 100 req/s (400 req/s total). But total throughput plateaus at 110 req/s regardless of how many replicas you add. `nvidia-smi` shows all 4 GPUs at ~25% utilization. Adding a 5th replica doesn't increase throughput at all. What's the ceiling?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The network or storage is the bottleneck" or "We need to increase the batch size." Network and storage are fine (the server isn't even close to saturating them), and batch size doesn't help when the GPUs are underutilized.

  **Realistic Solution:** The Python Global Interpreter Lock (GIL) is the ceiling. CPython allows only one thread to execute Python bytecode at a time. While CUDA kernel launches release the GIL (allowing GPU work to proceed in parallel), the Python-side preprocessing (tokenization, tensor construction, result postprocessing) for each request requires holding the GIL. With 4 threads competing for the GIL, each thread gets ~25% of the CPU time for its Python work — explaining the ~25% GPU utilization (each GPU is starved of new work 75% of the time). The total throughput is bounded by how fast a single CPU core can execute the Python preprocessing pipeline: if preprocessing takes 9 ms per request, the GIL-limited throughput is $1000 / 9 \approx 111$ req/s — matching the observed 110 req/s. Adding more replicas just adds more threads competing for the same GIL. Fix: (1) use multiprocessing instead of threading — each process has its own GIL; (2) use a C++ serving runtime (Triton Inference Server, TensorRT-LLM) that bypasses the GIL entirely; (3) minimize Python-side work by moving tokenization and postprocessing to compiled extensions.

  > **Napkin Math:** Per-request Python work: tokenization (3 ms) + tensor creation (2 ms) + result decoding (1 ms) + HTTP response (1 ms) = **7 ms** of GIL-holding time. GPU forward pass: **10 ms** (releases GIL). **Single thread:** 1 request every $7 + 10 = 17$ ms → **59 req/s**. GPU utilization: $10/17 = 59\%$. **4 threads (GIL-limited):** GIL serializes the 7 ms Python portions. Total GIL demand: $4 \times 7 = 28$ ms per 17 ms window — GIL is **oversubscribed by 1.65×**. Effective throughput: $1000 / 7 = $ **143 req/s** theoretical GIL limit, but with GIL acquisition overhead (~2 ms per context switch × 4 threads): effective = **~110 req/s**. Each GPU gets $110/4 = 27.5$ req/s × 10 ms = 275 ms of work per second → **27.5% utilization**. **Multiprocessing fix (4 processes):** Each process has its own GIL. Per-process throughput: 59 req/s. Total: $4 \times 59 = $ **236 req/s** — a **2.1× improvement**. GPU utilization rises to 59% each.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>


#### 🟡 L5 — Analyze & Predict

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Multi-Tenant GPU Sharing Problem</b> · <code>serving</code> <code>economics</code></summary>

- **Interviewer:** "You manage an inference platform serving 7 different small models (each <10GB). Your finance team wants to consolidate them onto a single A100-80GB using MIG (Multi-Instance GPU) to save costs. Each MIG slice gets 1/7th of the GPU: ~11 GB memory, ~45 TFLOPS. After deployment, two of the seven models have 3x worse tail latency than on dedicated GPUs. What went wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "MIG provides perfect isolation, so each model should perform as if it has a dedicated 1/7th GPU." MIG isolates compute and memory capacity, but not all resources.

  **Realistic Solution:** MIG does not isolate the shared L2 cache or the memory bandwidth. The A100's 40 MB L2 cache is shared across all MIG instances. Two of the seven models are memory-bandwidth-bound (low arithmetic intensity), and they're now competing for the same 2 TB/s of HBM bandwidth. With 7 tenants, each effectively gets ~285 GB/s — far less than the 2 TB/s they had on a dedicated GPU. The compute-bound models are fine because their bottleneck (Tensor Cores) is properly partitioned.

  > **Napkin Math:** Dedicated A100 per model: $2,000 \text{ GB/s}$ bandwidth, $312 \text{ TFLOPS}$. Cost: 7 × \$2.50/hr = \$17.50/hr. MIG 1g.10gb slice: $2,000 / 7 \approx 285 \text{ GB/s}$ effective bandwidth (shared), $312/7 \approx 45 \text{ TFLOPS}$ (isolated). Cost: 1 × \$2.50/hr = \$2.50/hr. For a memory-bound model with $I = 5$ Ops/Byte: dedicated throughput = $\min(312\text{T}, 2000 \times 5) = 312\text{ TFLOPS}$. MIG throughput = $\min(45\text{T}, 285 \times 5) = 45\text{ TFLOPS}$ — a 7× slowdown, not the expected 7× from compute partitioning alone. Effective cost: \$2.50/hr but 7× slower = \$2.50 × 7 = \$17.50/hr equivalent. No savings for bandwidth-bound models.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Request Routing Strategy</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "Our LLM API serves both autocomplete requests (~10 output tokens, 50ms SLA) and document summarization requests (~10,000 output tokens, no strict SLA). Both hit the same pool of 8× H100 GPUs. Autocomplete users are furious — P99 latency is 4 seconds instead of 50ms. The summarization users are fine. What's happening and how do you fix it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Add more GPUs to handle the load." The problem isn't capacity — it's mixing workloads with incompatible SLA profiles in the same scheduling queue.

  **Realistic Solution:** Head-of-line blocking from mixed request lengths. With continuous batching, a 10,000-token summarization request occupies a batch slot for ~400 seconds (at 25 tokens/sec). During that time, the scheduler must run prefill for incoming autocomplete requests, but the long requests consume KV-cache memory, reducing the maximum batch size. Worse, if the batch is full of long-running decodes, new short requests queue behind prefill scheduling delays. The fix is workload-aware routing: separate GPU pools (or priority lanes) for latency-sensitive short requests vs throughput-oriented long requests. Short-request GPUs run with small max-sequence-length and aggressive preemption; long-request GPUs optimize for throughput with large batches.

  > **Napkin Math:** 8× H100 shared pool, continuous batching, max batch = 32.
  > - Long requests: 10,000 tokens × 40ms/token = 400s each. 20 concurrent long requests fill 20/32 batch slots.
  > - Short requests: 10 tokens × 40ms/token = 0.4s each. Only 12 batch slots remain.
  > - But each long request's KV-cache: 10,000 tokens × 131 KB = 1.3 GB. 20 requests = 26 GB of KV-cache, leaving only 54 GB for weights + short request KV-cache on an 80 GB GPU.
  > - With dedicated pools: 2 GPUs for short requests (batch=32, all slots turn over in <1s, P99 TTFT < 50ms) and 6 GPUs for long requests (optimized for throughput). Short-request P99 drops from 4s to **40ms**.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Output Token Length Prediction</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "Our vLLM scheduler pre-allocates KV-cache memory for each request based on `max_tokens`. Most requests use only 10% of their allocation, wasting 90% of VRAM. But if we reduce `max_tokens`, long requests get preempted mid-generation. How do you solve this without wasting memory or killing long requests?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Set `max_tokens` to the average output length." This causes half of all requests to be preempted, which is worse than over-allocation because preemption means re-computing the entire KV-cache from scratch.

  **Realistic Solution:** Predict the output length per-request using a lightweight classifier (or heuristic based on prompt structure), then allocate KV-cache accordingly with a safety margin. Requests that exceed their prediction are handled via PagedAttention's dynamic allocation — pages are added on-demand rather than pre-allocated. The predictor doesn't need to be perfect; even a coarse bucketing (short: <50 tokens, medium: 50–500, long: >500) reduces waste by 60–70%.

  > **Napkin Math:** 70B model on H100-80GB, 50 concurrent requests:
  > - **Over-allocation (max_tokens=4096):** KV-cache per request = 4,096 × 640 KB = 2.6 GB. 50 requests = 130 GB → exceeds 80 GB VRAM. Can only serve **30 concurrent requests**.
  > - **Tight allocation (max_tokens=200):** KV-cache per request = 200 × 640 KB = 128 MB. 50 requests = 6.4 GB. Fits easily, but 15% of requests exceed 200 tokens and get preempted.
  > - **Preemption cost:** Re-prefill a 2,000-token prompt = 283ms of wasted compute per preemption. At 150 preemptions/minute: 150 × 283ms = **42.5 GPU-seconds/minute wasted**.
  > - **With prediction + PagedAttention:** Allocate predicted length + 20% buffer. Average allocation = 300 tokens × 640 KB = 192 MB. 50 requests = 9.6 GB. Requests that exceed prediction grow dynamically via page allocation. Preemption rate drops from 15% to <1%. **Net: 3× more concurrent requests than over-allocation, <1% preemption rate.**

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The SLA Violation Cascade</b> · <code>serving</code> <code>monitoring</code></summary>

- **Interviewer:** "Our LLM API has a 2-second P99 SLA. Last Tuesday, a single user sent a 32,000-token prompt that took 10 seconds to prefill. In the next 30 seconds, 100 requests violated their SLA. Explain the cascade mechanism and design a system that prevents one bad request from taking down the fleet."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Rate-limit users who send long prompts." Rate limiting prevents abuse but doesn't prevent the cascade once a long request is admitted — the damage happens during prefill, before you can react.

  **Realistic Solution:** The cascade works through three mechanisms: (1) **Prefill starvation:** The 10-second prefill monopolizes GPU compute, blocking all decode iterations for concurrent requests. (2) **Queue buildup:** During the 10-second stall, new requests accumulate in the queue at the arrival rate. (3) **Batch bloat:** When the prefill finishes, the scheduler tries to drain the queue by admitting many requests at once, causing KV-cache memory pressure and further slowdowns. The fix requires admission control at multiple levels: max prompt length enforcement, chunked prefill (split long prefills into 512-token chunks interleaved with decode iterations), and request-level timeout with graceful preemption.

  > **Napkin Math:** Normal operation: 100 QPS, 200ms TTFT, 30ms TPOT, 100-token avg response = 3.2s avg latency. P99 < 2s for short requests.
  > - **The cascade:** 32K-token prefill on 70B model: 2 × 70B × 32,000 = 4,480 TFLOPs. On H100 (990 TFLOPS): **4.5 seconds** of pure compute. During this time, 100 QPS × 4.5s = **450 requests queue up**.
  > - **Drain phase:** 450 queued requests all start prefill. Even with continuous batching, prefilling 450 requests (avg 500 tokens each) takes: 450 × 500 × 140 GFLOPS / 990 TFLOPS = **~32 seconds** to clear the backlog. Every request admitted during the drain phase also violates SLA.
  > - **With chunked prefill (512-token chunks):** The 32K prompt is split into 63 chunks. Each chunk takes 72ms. Between chunks, the scheduler runs 1 decode iteration for all active requests (~5ms). Total prefill time increases to 63 × (72 + 5) = **4.9s**, but no decode request is stalled for more than 72ms. **Zero cascade.**

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Serverless Inference Trade-off</b> · <code>serving</code> <code>economics</code></summary>

- **Interviewer:** "We're choosing between AWS SageMaker Serverless (pay-per-invocation, scales to zero) and a dedicated `ml.g5.2xlarge` instance (1× A10G, $1.52/hr always-on) for serving a 7B model. Our traffic is bursty: 500 QPS during business hours (10 hrs/day), near-zero overnight. Find the break-even point and tell me which option wins."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Serverless always wins for bursty traffic because you don't pay for idle time." Serverless has per-invocation costs that scale linearly with QPS, while dedicated instances have fixed costs that become cheaper per-request at high utilization.

  **Realistic Solution:** The break-even depends on three factors: (1) the per-invocation cost of serverless (compute time × price per GB-second), (2) the cold-start penalty and its frequency, and (3) the utilization-adjusted cost of dedicated instances. For bursty workloads, the optimal strategy is often a hybrid: dedicated instances sized for the base load with serverless overflow for spikes.

  > **Napkin Math:**
  > - **Dedicated (ml.g5.2xlarge, 1× A10G):** $1.52/hr × 24 hr × 30 days = **$1,094/month**. Throughput: ~50 tokens/sec for 7B model. At 500 QPS with 100-token responses: need 500 × 100 / 50 = 1,000 instances during peak. That's $1,094 × 1,000 = **$1.09M/month**. Clearly need fewer instances with batching — at BS=32, throughput ≈ 800 tok/s per instance, need ~63 instances = **$68,922/month**.
  > - **Serverless (SageMaker Serverless):** ~$0.0001 per invocation + $0.00003/sec compute. Per request (200ms inference): $0.0001 + 0.2 × $0.00003 = **$0.000106/request**. At 500 QPS × 10 hr/day × 30 days: 500 × 36,000 × 30 = 540M requests. Cost = 540M × $0.000106 = **$57,240/month**. Plus cold starts: ~5% of requests hit cold start (3s penalty), adding latency but no extra cost.
  > - **Break-even:** Serverless < dedicated when: $0.000106 × QPS × 3,600 < $1.52/hr per instance. Per instance: QPS < 1.52 / (0.000106 × 3,600) = **~4 QPS per instance**. Above 4 QPS sustained per instance, dedicated wins.
  > - **Hybrid:** 40 dedicated instances for base load ($43,776/month) + serverless for overflow spikes. Saves ~20% vs pure dedicated during off-peak.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Batching Starvation</b> · <code>serving</code> <code>scheduling</code></summary>

- **Interviewer:** "You implement Continuous Batching (iteration-level scheduling) for your LLM server. It successfully groups users together to maximize GPU utilization. However, you notice that users asking for very short, 10-token answers are experiencing incredibly long wait times, even though their requests require very little compute. How did your batching algorithm starve the easy requests?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The GPU prioritizes larger matrix math." The GPU doesn't prioritize anything; the software scheduler does.

  **Realistic Solution:** You caused **Head-of-Line Blocking via First-Come-First-Served (FCFS) Scheduling**.

  If a user asks for a 2,000-token story, their request enters the batch. The batch size is capped by the GPU's KV-cache capacity (e.g., max 32 concurrent sequences).
  Because generating 2,000 tokens takes a long time, that user occupies 1 of the 32 slots for several seconds.

  If 32 users ask for long stories, the batch becomes completely full. When User 33 arrives and asks a simple question requiring a 10-token response, the Continuous Batching scheduler cannot let them in because there is no KV-cache memory available.
  User 33 must wait entirely for one of the long stories to finish generating all 2,000 tokens before a slot frees up.

  **The Fix:** You must implement **Preemption and Fair-Share Scheduling**. The scheduler should detect long-running requests, temporarily evict/preempt them (saving their KV-cache to CPU RAM or recomputing it later), inject the short requests into the batch to clear them out quickly, and then resume the long requests.

  > **Napkin Math:** 32 users request 2,000 tokens at 20ms/token = 40 seconds to finish. User 33 requests 10 tokens (0.2 seconds of work). User 33 waits 40 seconds to get 0.2 seconds of compute. This ruins P99 latency metrics.

  📖 **Deep Dive:** [Volume I: Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The LLM Canary Trap</b> · <code>canary-deployment</code> <code>llm-serving</code></summary>

- **Interviewer:** "You're deploying a fine-tuned 70B LLM to replace your production 13B model. You set up a standard canary: route 5% of traffic to the new model, monitor error rate and latency for 30 minutes, then promote. After full rollout, users report the new model occasionally hallucinates wildly long, repetitive outputs. Your canary detected no latency or error spikes. Why do standard infra metrics miss LLM quality regressions, and how could you have used the GPU's KV-cache memory profile as a hardware-level canary signal?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The canary window was too short — extend it to 24 hours." More time helps, but the fundamental problem is that LLM failures are not captured by traditional canary metrics.

  **Realistic Solution:** Traditional canary metrics (latency, error rate, throughput) measure *infrastructure health*, not *output quality*. A model that suddenly starts outputting repetitive garbage ("I am an AI, I am an AI...") will still return HTTP 200s and keep the GPU busy. However, LLM quality regressions often manifest as changes in generation length (verbosity or truncation). You can use the GPU's KV-cache memory allocation as a proxy metric for output behavior. If the new model is hallucinating longer responses, the average KV-cache blocks allocated per request will spike. If it's failing to reason and returning short "I don't know" answers, the KV-cache allocation will drop. Monitoring the physical memory footprint provides a real-time, hardware-level signal of behavioral changes before users complain.

  > **Napkin Math:** Baseline model generates 200 tokens/request average. Canary model has a repetition bug and generates 800 tokens/request. Latency might look okay if the batch scheduler just drops throughput to compensate. But KV-cache memory tells the truth: Baseline uses 200 tokens × 1MB/token (for a large batch) = 200MB per request. Canary uses 800MB per request. By monitoring `vllm:gpu_cache_usage_perc`, you would see the canary instances hitting 90% KV-cache saturation 4× faster than the baseline instances, instantly flagging a behavioral shift.

  📖 **Deep Dive:** [ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Multi-Model Serving Platform</b> · <code>serving</code> <code>economics</code></summary>

- **Interviewer:** "Your company runs 12 different LLMs in production (ranging from 1B to 70B parameters) serving 50,000 QPS total across all models. Currently each model has its own dedicated GPU pool. The CFO says GPU costs are \$2M/month. Design a multi-model serving platform that cuts costs by at least 40% without violating per-model latency SLAs."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just put all models on the same GPUs and multiplex." Naive co-location causes memory fragmentation and SLA violations because large models evict small ones from VRAM.

  **Realistic Solution:** A multi-model platform needs three architectural layers: (1) **Model tiering** — group models by size and latency SLA. Tier 1 (1B–7B, <100ms): always resident in VRAM, share GPUs via MPS or time-slicing. Tier 2 (13B–30B, <500ms): resident on dedicated GPUs with overflow to CPU offload. Tier 3 (70B, <2s): tensor-parallel across multi-GPU, dedicated but right-sized. (2) **Predictive autoscaling** — use traffic forecasting (not reactive scaling) to pre-warm models 10 minutes before demand spikes. (3) **KV-cache sharing** — for models serving similar prompts (e.g., system prompts), cache the KV-state of common prefixes across requests.

  > **Napkin Math:** Current: 12 models × dedicated pools. Typical utilization: 30% (peak-provisioned). Monthly cost: \$2M. Optimized: Tier 1 (8 small models): consolidate from 40 GPUs to 15 (shared, 80% util). Savings: 25 GPUs × \$2/hr × 720 = \$36k/mo. Tier 2 (3 mid models): right-size from 60 GPUs to 35 with autoscaling. Savings: \$36k/mo. Tier 3 (1 large model): reduce from 80 GPUs to 50 with continuous batching + PagedAttention. Savings: \$43k/mo. Prefix caching saves ~20% of KV-cache memory across all tiers → another 15% GPU reduction: \$60k/mo. **Total savings: \$175k/mo (8.75%)** — wait, that's not 40%. The real lever: **spot instances for Tier 1** (stateless, fast restart) saves 65% on those GPUs: \$200k/mo. Plus **quantizing Tier 2 to INT8** halves their GPU count: \$180k/mo. **Revised total: \$850k/mo savings (42.5%).**

  📖 **Deep Dive:** [ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Speculative Decoding Speedup</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "Our 70B model serves chat completions with a P50 time-to-first-token of 200 ms and a P50 inter-token latency of 45 ms on H100. Product wants 2× faster decoding without changing the model. Someone suggests speculative decoding with a 1B draft model. Walk me through the systems math — when does this help, and when does it backfire?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The draft model is 70× smaller, so it generates tokens 70× faster, and we just verify them in parallel — easy 5-10× speedup." This ignores the acceptance rate, the memory overhead of running two models, and the verification cost.

  **Realistic Solution:** Speculative decoding works in three steps: (1) the draft model generates $K$ candidate tokens autoregressively (cheap, ~1 ms/token for a 1B model); (2) the target 70B model verifies all $K$ tokens in a single forward pass (parallel verification — same cost as generating 1 token, since decode is memory-bandwidth-bound and the extra compute for $K$ tokens is negligible); (3) the target accepts the first $n \leq K$ tokens where the draft model's distribution matches, and resamples the $(n+1)$-th token. The speedup depends critically on the **acceptance rate** $\alpha$ — the probability the draft model's token matches the target's. If $\alpha = 0.8$ and $K = 5$, the expected accepted tokens per verification step is $\sum_{i=0}^{K} \alpha^i \approx 1/(1-\alpha) = 5$ tokens per target forward pass (in the limit). But you also pay for the draft model's memory and compute. If the draft model's memory displaces KV-cache space, your maximum batch size drops, reducing throughput even as per-request latency improves.

  > **Napkin Math:** **Without speculation:** 70B decode = 45 ms/token (bandwidth-bound: 140 GB weights / 3.35 TB/s ≈ 42 ms + overhead). **With speculation (K=5, α=0.8):** Draft generates 5 tokens: 5 × 1 ms = **5 ms**. Target verifies: **45 ms** (one forward pass). Expected accepted tokens: $1/(1-0.8) = 5$ tokens. Effective per-token latency: $(5 + 45) / 5 = $ **10 ms/token** — a **4.5× speedup**. Memory cost: 1B draft model = 2 GB + its KV-cache ≈ 2.5 GB. On 80 GB H100 with 70B model (140 GB sharded TP=2 → 70 GB/GPU), free memory drops from 10 GB to 7.5 GB — **max batch drops from ~9 to ~7 requests**. **When it backfires:** If $\alpha$ drops to 0.4 (e.g., code generation where the draft model is weak), expected accepted = $1/0.6 \approx 1.67$ tokens per step. Effective latency: $(5 + 45) / 1.67 = $ **30 ms/token** — only 1.5× speedup, and you've lost 25% of your batch capacity for it.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The KV-Cache OOM Attack</b> · <code>serving</code> <code>incident-response</code></summary>

- **Interviewer:** "Your LLM serving endpoint supports 32k context. On Monday, your cluster starts OOM-killing pods every few minutes. The traffic volume hasn't increased, but you notice a handful of users sending prompts that are exactly 32,000 tokens — the maximum. Before Monday, average prompt length was 500 tokens. How does a 64× increase in prompt length from a few users crash the entire cluster, and what's your emergency response?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just reject prompts over a certain length" or "Add more GPUs to handle the load." Rejecting long prompts breaks legitimate use cases, and adding GPUs doesn't help if the scheduling algorithm is the problem.

  **Realistic Solution:** The KV-cache memory for a single request scales linearly with sequence length. A 32k-token request consumes 64× the KV-cache of a 500-token request. If the scheduler admits requests based on *count* (e.g., "batch up to 32 requests") rather than *memory*, a few 32k requests can exhaust the entire KV-cache budget that normally serves hundreds of short requests. This is effectively a resource exhaustion attack — intentional or not. The cluster crashes because: (1) the scheduler admits 32 requests including several 32k-token ones; (2) KV-cache allocation exceeds VRAM; (3) the CUDA OOM kills the serving process; (4) Kubernetes restarts the pod, which immediately admits the same queued requests and crashes again (crash loop). Emergency response: (1) implement admission control based on *memory budget*, not request count — estimate KV-cache cost before admitting: $\text{KV\_bytes} = 2 \times L \times H \times d_h \times S \times 2$; (2) set per-user rate limits on total token throughput, not just request count; (3) implement preemption — if a new request would cause OOM, preempt (pause and swap to CPU) the lowest-priority in-flight request.

  > **Napkin Math:** 7B model on A100 80 GB. Weights = 14 GB. Free for KV-cache = 66 GB. KV-cache per token per request (32 layers, 32 heads, d=128, FP16): $2 \times 32 \times 32 \times 128 \times 2 = $ **524 KB/token**. At 500-token avg: **262 MB/request**. Batch of 32 normal requests: $32 \times 262\text{ MB} = $ **8.4 GB** — fits easily. At 32k tokens: **16.8 GB/request**. Just 4 adversarial requests: $4 \times 16.8 = $ **67.2 GB** — exceeds the 66 GB budget → **OOM**. Those 4 requests consume more memory than 256 normal requests. Memory-aware scheduling: admit requests until $\sum_i \text{KV}(S_i) \leq 66\text{ GB}$. This limits concurrent 32k requests to 3, while allowing 250+ concurrent 500-token requests. PagedAttention (vLLM) helps by allocating KV-cache in 4 KB pages on-demand rather than pre-allocating for max length, reducing waste from requests that don't use their full context window.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>


#### 🔴 L6+ — Synthesize & Derive

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Batching Dilemma</b> · <code>serving</code> <code>batching</code></summary>

- **Interviewer:** "We use static batching for our LLM API. If a request generating 5 tokens is batched with a request generating 500 tokens, the 5-token request sits in VRAM until the 500-token request finishes. How do we fix this idle capacity?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Set a maximum generation length to keep batch members similar." This caps functionality and still wastes compute on padding.

  **Realistic Solution:** Implement Continuous (or In-Flight) Batching. Instead of waiting for all requests in a batch to finish, continuous batching operates at the iteration level. As soon as the 5-token request finishes and emits its `<EOS>` token, it is immediately evicted from the batch, and a new request from the queue is slotted in for the very next forward pass.

  > **Napkin Math:** Static batch of 8 requests: shortest = 5 tokens, longest = 500 tokens. Wasted compute = 7 requests × (500 - avg_length) × cost_per_token. With continuous batching, the 5-token request is done in 5 iterations and replaced — the slot processes ~100 different short requests in the time the long request takes, increasing effective throughput by 2-4×.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Speculative Decoding Accept Rate Crash</b> · <code>serving</code> <code>algorithms</code></summary>

- **Interviewer:** "You implement Speculative Decoding to speed up an LLM. You use a 1B draft model to guess 5 tokens, and a 70B target model to verify them in a single pass. On standard English text, it yields a 2.5x speedup. When you deploy it to your coding assistant API, the overall throughput actually *drops* by 15% compared to not using Speculative Decoding at all. Why did the optimization become a penalty?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Code is harder to generate." It is, but that alone doesn't explain why a speedup technique becomes an active *penalty*.

  **Realistic Solution:** You suffered a **Catastrophic Draft Acceptance Rate Drop**.

  Speculative Decoding works by gambling: the draft model is fast, and the target model verifies its guesses in parallel. If the target model agrees with the draft model, you get 5 tokens for the price of 1.

  However, the target model's verification pass requires reading the KV-cache and processing all 5 draft tokens. If the draft model guesses *wrong* on the very first token, the target model rejects the entire draft sequence. You spent compute running the draft model, and you spent compute running the target model to verify a 5-token sequence, but you only got 1 valid token out of it.

  Your 1B draft model was trained on general English, so it guessed English well. But on complex, highly structured programming languages (Python, C++), the 1B model was completely lost. Its guess-acceptance rate dropped from 70% to near 0%.

  Because you constantly paid the overhead of running the draft model and evaluating longer target sequences without getting any parallel tokens accepted, the net throughput was worse than just running the 70B model autoregressively.

  **The Fix:** Speculative Decoding is entirely reliant on the Draft/Target alignment. You must either fine-tune the 1B draft model specifically on your domain (code), use a smaller draft model to reduce overhead, or dynamically disable Speculative Decoding if the rolling acceptance rate drops below the break-even mathematical threshold.

  > **Napkin Math:** Break-even Acceptance Rate. If draft takes 5ms, and target takes 20ms per pass.
  > Standard (1 token): 20ms.
  > Speculative (5 tokens, 0% accept): 5ms (draft) + 20ms (eval) = 25ms to get 1 token. (25% slower).
  > Speculative (5 tokens, 100% accept): 5ms + 20ms = 25ms to get 5 tokens = 5ms per token. (400% faster).

  📖 **Deep Dive:** [Volume I: Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>


### 🚀 Advanced Inference


#### 🔵 L4 — Apply & Identify

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The KV-Cache Context Explosion</b> · <code>attention-mechanisms</code></summary>

- **Interviewer:** "You are serving a Llama-3 8B model. At a batch size of 1 with a 1,000-token prompt, inference requires roughly 16GB of VRAM. A customer wants you to increase the context window to 128,000 tokens for document summarization. Your PM assumes it will just take a bit more memory and asks you to deploy it on a 24GB RTX 4090. Why is this physically impossible?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Forgetting that the KV-Cache grows linearly with sequence length, and assuming model weights are the only significant memory consumer."

  **Realistic Solution:** You hit the KV-Cache memory wall. While the model weights for an 8B model (at 16-bit precision) consume about 16GB of VRAM regardless of the prompt size, the Key-Value (KV) cache does not. To avoid recomputing attention for past tokens, autoregressive generation stores the K and V vectors for every single token in the context window. At 128,000 tokens, the physical memory required just to store the state of the conversation dwarfs the size of the model itself.

  > **Napkin Math:** For Llama-3 8B (32 layers, 8 KV heads, 128 head dim, 2 bytes per param):
  > `Memory per token = 2 (K&V) * 32 layers * 8 heads * 128 dim * 2 bytes = ~131 KB per token`.
  > `1,000 tokens * 131 KB = ~130 MB` (Fits easily in a 4090).
  > `128,000 tokens * 131 KB = ~16.7 GB`.
  > `16GB (weights) + 16.7GB (KV Cache) = 32.7 GB`. It physically cannot fit into a 24GB GPU.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Inference Cost Attribution Puzzle</b> · <code>cost-attribution</code> <code>inference</code></summary>

- **Interviewer:** "Your platform serves 5 different LLMs on a shared pool of 64 A100 GPUs using vLLM with continuous batching. The finance team wants to charge each product team for their inference costs. Your colleague proposes: 'Divide total GPU cost by total requests, charge per request.' The search team (short prompts, 50-token outputs) and the content team (long prompts, 2,000-token outputs) both object. Design a fair cost attribution model."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Charge per request" or "Charge per token." Per-request ignores the 40× difference in compute between a 50-token and 2,000-token generation. Per-token ignores that prefill and decode have different costs.

  **Realistic Solution:** Fair cost attribution must reflect actual GPU resource consumption, which has two distinct phases: (1) **Prefill cost** — proportional to input tokens (compute-bound, scales as O(n²) for attention), and (2) **Decode cost** — proportional to output tokens (memory-bandwidth-bound, scales linearly but occupies KV-cache memory for the entire generation duration). The correct model charges: `cost = α × input_tokens + β × output_tokens + γ × generation_time_ms`. The α term captures prefill compute, β captures decode compute, and γ captures KV-cache memory occupancy (a long-running generation blocks memory that could serve other requests). Calibrate α, β, γ by profiling each model: measure GPU-seconds consumed per input token (prefill) and per output token (decode) at representative batch sizes. For continuous batching, the γ term is critical — a request generating 2,000 tokens at 40ms/token occupies KV-cache for 80 seconds, blocking ~2 GB of GPU memory that could serve 20 short requests.

  > **Napkin Math:** 64 A100s at $2/hr = $128/hr total. Search team: 10,000 req/hr × 500 input + 50 output tokens. Content team: 1,000 req/hr × 2,000 input + 2,000 output tokens. Per-request billing: search pays 10/11 × $128 = $116/hr, content pays $12/hr. Per-token billing: search = 5.5M tokens, content = 4M tokens → search pays $74, content pays $54. Per-GPU-second billing (actual resource use): search prefill ≈ 0.5s/req × 10k = 5,000 GPU-sec, content prefill ≈ 4s/req × 1k = 4,000 GPU-sec, content decode ≈ 80s/req × 1k = 80,000 GPU-sec. Content uses 94% of decode resources → fair charge ≈ $108/hr. The naive per-request model undercharges content by 9×.

  📖 **Deep Dive:** [ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>


#### 🟡 L5 — Analyze & Predict

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-gold?style=flat-square" alt="Level 3" align="center"> The Speculative Memory Trade-off</b> · <code>speculative-decoding</code></summary>

- **Interviewer:** "You implement speculative decoding for a 70B parameter LLM using a 1.5B draft model. On a single A100 (80GB), generation speed increases by 2.5x. You deploy this to production, but under heavy concurrent load (batch size 64), the speculative decoding actually makes generation *slower* than standard autoregressive decoding. Why?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming speculative decoding always provides a speedup because it reduces the number of forward passes of the large model."

  **Realistic Solution:** Speculative decoding trades spare compute (ALU cycles) for memory bandwidth. At batch size 1, LLM inference is severely memory-bandwidth bound, so the A100's ALUs are mostly idle. Using those idle ALUs to run a small draft model and verify multiple tokens in one memory sweep of the 70B model is a massive win. However, at batch size 64, the arithmetic intensity of the workload increases significantly. The system becomes compute-bound. Now, running the draft model steals precious ALU cycles away from the main model's generation, and the overhead of verifying rejected tokens actually reduces overall throughput.

  > **Napkin Math:** A 70B model at BS=1 has an arithmetic intensity of `~2 FLOPs/byte`. The A100 is memory bound up to `156 FLOPs/byte`. But at BS=64, intensity jumps to `128 FLOPs/byte`, nearing the compute roofline. The draft model requires `1.5B * 2 = 3 GFLOPs` per token. At BS=64, that's 192 GFLOPs of pure overhead per speculative step that is now eating into a fully saturated compute pipeline.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Serverless Freeze</b> · <code>serverless-inference</code></summary>

- **Interviewer:** "Your team is exploring serverless functions (e.g., AWS Lambda, Google Cloud Functions) for an on-demand, low-volume ML inference endpoint to minimize operational overhead and cost. However, initial tests reveal significant 'cold start' latencies, sometimes exceeding 5 seconds, which is unacceptable for the user experience. What are the main contributors to serverless cold starts for ML workloads, and what strategies would you employ to bring them within an acceptable range?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Serverless is just too slow for ML, we need dedicated instances." While dedicated instances avoid cold starts, this misses the point of optimizing serverless, which offers significant cost and operational benefits for bursty or low-volume workloads if cold starts can be managed.

  **Realistic Solution:** Serverless cold starts for ML workloads are primarily due to:
    1.  **Container Image Download:** Large model artifacts and dependencies increase the time to download the container image to a new execution environment.
    2.  **Runtime Initialization:** Spinning up the language runtime (Python, Java), loading frameworks (TensorFlow, PyTorch), and initializing the serving environment.
    3.  **Model Loading:** Loading the model weights from disk (or S3) into GPU/CPU memory. This is often the largest contributor for ML.

  Strategies to mitigate cold starts:
    1.  **Provisioned Concurrency:** Pre-warm a specified number of execution environments, ensuring they are always ready. This is the most direct solution but incurs continuous cost.
    2.  **Container Image Optimization:**
        *   **Multi-stage Builds:** Create lean production images with only necessary runtime dependencies.
        *   **Smaller Base Images:** Use `alpine` or `distroless` for minimal footprint.
        *   **Dependency Pruning:** Remove development tools, cached packages.
    3.  **Lazy Loading of Model Weights:** Load model weights only when the first inference request arrives, or use a custom runtime that initializes the model outside the main handler.
    4.  **Model Splitting/Distillation:** Reduce the model size itself through techniques like knowledge distillation or pruning, leading to faster download and load times.
    5.  **Pre-computation/Caching:** For inputs with high locality, pre-compute embeddings or intermediate features and cache them in a low-latency store (e.g., Redis).
    6.  **"Keep-alive" Pings:** For non-critical workloads, periodically invoke the function to keep instances warm (though this is a hack and adds cost).

  > **Napkin Math:** A typical serverless cold start:
  > - Image download (500MB on 100Mbps): `~4 seconds`
  > - Runtime init (Python, TF/PyTorch): `~1-2 seconds`
  > - Model load (200MB model into memory): `~0.5-1 second`
  > Total: `~5.5-7 seconds`.
  > With provisioned concurrency, this can drop to `~<100ms` for subsequent requests, at the cost of continuous billing for the pre-warmed instances.

  > **Key Equation:** `ColdStartTime = ImageDownloadTime + RuntimeInitTime + ModelLoadTime`

  📖 **Deep Dive:** [Volume I: Serverless Inference Challenges and Solutions](https://mlsysbook.ai/vol1/mlops/serverless-inference)

  </details>

</details>

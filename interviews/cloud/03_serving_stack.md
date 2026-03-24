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


### Latency & Throughput


#### 🟢 L1/L2


<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Per-Token KV-Cache Cost</b> · <code>kv-cache-cost</code></summary>

- **Interviewer:** "You are calculating the memory requirements for serving a large language model. The model has 40 layers, 64 attention heads per layer, and a head dimension of 128. If you are using FP16 precision, calculate the amount of memory the KV-cache consumes for *each single token* added to the sequence."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common errors are forgetting to account for both the Key (K) and the Value (V) tensors (an off-by-2x error) or using the wrong precision (e.g., 4 bytes for FP32 instead of 2 for FP16). Another mistake is calculating for just one layer instead of all of them.

  **Realistic Solution:** For each token in the input sequence, every attention layer must store a Key and a Value vector. The size of these vectors is determined by the number of heads and the head dimension. The total memory per token is the sum of the sizes of these K and V vectors across all layers.

The calculation is: `2 (for K and V) × num_layers × num_heads × head_dim × bytes_per_element`.

  > **Napkin Math:** 1. **Identify dimensions:** 40 layers, 64 heads, 128 head dimension.
2. **Identify precision:** FP16 = 2 bytes.
3. **Apply the formula:** 2 (K and V) × 40 layers × 64 heads × 128 dim × 2 bytes/element.
4. **Calculate:** 2 × 40 × 64 × 128 × 2 = 1,310,720 bytes.
5. **Convert to a readable unit:** 1,310,720 bytes / 1,000,000 ≈ 1.31 MB.

  > **Key Equation:** $\text{KV Cache (token)} = 2 \times N_{layers} \times N_{heads} \times d_{head} \times \text{sizeof(dtype)}$

  > **Options:**
  > [ ] About 655 KB
  > [ ] About 2.62 MB
  > [ ] About 32.7 KB
  > [x] About 1.31 MB

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Continuous Batching Target</b> · <code>llm-inference-latency</code></summary>

- **Interviewer:** "A user of your LLM API complains about the long initial wait before their generated text starts appearing. A teammate suggests implementing continuous batching to specifically reduce this initial wait time. Which of the following metrics is continuous batching designed to primarily improve?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often believe that any batching technique will speed up all aspects of inference. They incorrectly assume continuous batching will reduce the 'Time To First Token' (TTFT) for a new request, when its main benefit lies in improving overall system efficiency rather than the latency of a single, isolated request.

  **Realistic Solution:** Continuous batching primarily improves overall throughput, which is measured in tokens per second across all concurrent users. This has the effect of reducing the average 'Time Per Output Token' (TPOT). It achieves this by eliminating head-of-line blocking and ensuring the GPU is always busy decoding *some* token for *some* user, rather than waiting for a full batch to complete. The initial wait (TTFT) is dominated by the prefill stage for the new prompt, which is not the primary target of this optimization.

  > **Napkin Math:** Imagine a GPU that can process one prompt prefill in 500ms or one token decoding in 50ms.
- Without batching, TTFT is 500ms. TPOT is 50ms.
- With continuous batching, a new request still takes 500ms to prefill, so TTFT remains ~500ms.
- However, by packing tokens from many users together, the GPU can generate a token every 50ms for the *system*, drastically increasing total tokens/second (throughput) and lowering the average TPOT for all users.

  > **Options:**
  > [ ] Time To First Token (TTFT)
  > [x] System Throughput / Average Time Per Output Token (TPOT)
  > [ ] Model Loading Time
  > [ ] GPU Idle Time

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The KV-Cache Memory Hog</b> · <code>kv-cache-sizing</code></summary>

- **Interviewer:** "An LLM with 80 layers is processing a single inference request with a context (sequence length) of 100,000 tokens. Each token's state is represented by Key and Value vectors totaling 32KB per layer. What is the approximate VRAM consumed by the KV cache for this single request?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is an order-of-magnitude error, either by calculating the memory for a single layer and forgetting to multiply by the number of layers, or by misplacing the decimal when converting from MB to GB. Another error is to calculate the memory for just one token instead of the full sequence.

  **Realistic Solution:** The KV-cache stores the Key and Value vectors for every token in the sequence at every layer of the model. This allows the model to attend to previous tokens without re-computing their states.
The total size is the memory per token per layer, multiplied by the number of layers and the number of tokens.

  > **Napkin Math:** 1. Memory per token: `32 KB/layer × 80 layers = 2,560 KB` or `~2.56 MB`
2. Total KV Cache Size: `2.56 MB/token × 100,000 tokens = 256,000 MB`
3. Convert to GB: `256,000 MB / 1000 ≈ 256 GB`.
This demonstrates how quickly the KV cache can become the dominant memory consumer in long-context scenarios, often dwarfing the model weights themselves.

  > **Key Equation:** V_{cache} \approx S \times L \times (H_{keys} + H_{values})

  > **Options:**
  > [ ] ~2.56 GB
  > [ ] ~32 GB
  > [x] ~256 GB
  > [ ] ~2.56 MB

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The KV-Cache Memory Bomb</b> · <code>kv-cache</code></summary>

- **Interviewer:** "You are planning the deployment for a 70-billion parameter LLM. A product requirement is to support very long context windows up to 128,000 tokens. For a single user request with a full context window, what is the approximate VRAM required just to store the KV-cache?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to only consider the model's weight memory (~140GB for a 70B model in FP16), completely forgetting that the KV-cache's memory footprint scales linearly with the sequence length and can easily surpass the model weight memory.

  **Realistic Solution:** The KV-cache for a single 128k-token request will require approximately 335 GB of VRAM. This is why serving models with long context windows is extremely challenging and expensive.

  > **Napkin Math:** KV Cache Size ≈ (Batch Size × Sequence Length × 2 × Num Layers × Hidden Dim) × Bytes/Value.
For Llama 70B (80 layers, 8192 hidden dim) and a 128k sequence length in FP16:
1 × 128,000 × 2 × 80 × 8192 × 2 bytes ≈ 3.35 × 10¹¹ bytes ≈ 335 GB.

  > **Key Equation:** $\text{KV Cache Memory} \approx B \times S \times 2 \times L \times D_{model} \times \text{bytes}$

  > **Options:**
  > [ ] ~140 GB
  > [ ] ~2.6 MB
  > [x] ~335 GB
  > [ ] ~1.1 TB

  📖 **Deep Dive:** [Scaling Rules for Cloud/LLM](https://github.com/mlsysbook/interviews/blob/main/ironlaw.qmd#4-scaling-rules-arithmetic--hardware-independent)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The VRAM Cost of Context</b> · <code>kv-cache-vram</code></summary>

- **Interviewer:** "You are deploying a large language model for a chatbot service. The model architecture is similar to Llama-2 70B, which has a hidden dimension of 8192 (`d_model`) and 80 layers. The product team wants to guarantee support for a 4096-token context window. Explain how you would calculate the VRAM required *just for the KV-cache* for a single user request, assuming the cache is stored in FP16 precision."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to only account for model weights and forget the KV-cache, which scales linearly with the context length and can easily dominate VRAM usage for long sequences. Another frequent error is to miscalculate the cache size, often by forgetting one of the key factors: the `x2` for storing both Keys and Values, or the bytes per parameter (2 for FP16).

  **Realistic Solution:** The KV-cache stores the Key and Value vectors for every token in the context, for every layer. For a single sequence, its size can be estimated by multiplying the sequence length by the size of the K and V vectors required for each token across all layers. A simple and safe estimation uses the model's hidden dimension (`d_model`) to represent the combined vector size per layer.

  > **Napkin Math:** The calculation is as follows:
1.  **Formula:** `Total Cache = Seq. Length × Num Layers × Hidden Dimension × 2 (for K/V) × Bytes per Value`
2.  **Parameters:**
    - Sequence Length: 4,096 tokens
    - Num Layers: 80
    - Hidden Dimension: 8,192
    - Bytes per Value: 2 (for FP16)
3.  **Calculation:** `Cache = 4096 × 80 × 8192 × 2 (K/V) × 2 (bytes)`
    - `Cache = 4096 × 80 × 8192 × 4`
    - `Cache = 10,737,418,240` bytes
4.  **Conversion:** `10,737,418,240 bytes / (1024^3) = 10 GB`

  > **Key Equation:** $\text{KV Cache (Bytes)} = S \times L \times D_{model} \times 2 \times B_{val}$

  > **Options:**
  > [ ] 1.25 GB
  > [ ] 5 GB
  > [x] 10 GB
  > [ ] 80 GB

  📖 **Deep Dive:** [Cloud / LLM Scaling Rules](https://github.com/mlsysbook/mlsysbook/blob/main/interviews/NUMBERS.md#4-scaling-rules-arithmetic--hardware-independent)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Deadline</b> · <code>continuous-batching-latency</code></summary>

- **Interviewer:** "You're designing an interactive LLM-based chatbot service running on H100 GPUs. The system uses continuous batching, where new user requests can be added to the processing queue between token generation steps. Your measurement team has determined that a single token generation step for the live batch takes a constant 150ms. Your service level objective (SLO) for user experience is a Time-To-First-Token (TTFT) of 500ms.

Explain the worst-case TTFT a new user might experience. Does your current system design meet the SLO?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to assume the TTFT is simply the processing time of a single step (150ms). This ignores the 'waiting in line' problem, where a request might arrive just after a batch has been kicked off, forcing it to wait for the entire next processing cycle before it can even begin.

  **Realistic Solution:** The worst-case scenario happens when a user's request arrives immediately after the server has started an iteration. The request must wait for that 150ms iteration to complete before it can be included in the next batch. Then, its own batch iteration takes another 150ms to generate the first token.

Therefore, the worst-case TTFT is the sum of the maximum wait time and the processing time. This is 150ms (wait) + 150ms (process) = 300ms.

Since 300ms is less than the 500ms SLO, the system currently meets its latency requirements. The binding constraint on scaling this single GPU is not the TTFT, but likely the total number of concurrent users that can fit into HBM before causing an out-of-memory error.

  > **Napkin Math:** 1. **Define Iteration Time**: $T_{\text{iteration}} = 150\text{ms}$
2. **Define Wait Time**: A new request's worst-case wait time, $T_{\text{wait}}$, is one full iteration, as it just missed the bus. $T_{\text{wait}} = T_{\text{iteration}} = 150\text{ms}$.
3. **Define Processing Time**: The time to process the request's first token is one iteration. $T_{\text{process}} = T_{\text{iteration}} = 150\text{ms}$.
4. **Calculate Worst-Case TTFT**: $TTFT_{\text{worst}} = T_{\text{wait}} + T_{\text{process}} = 150\text{ms} + 150\text{ms} = 300\text{ms}$.
5. **Compare to SLO**: $300\text{ms} < 500\text{ms}$. The system meets the SLO.

  > **Key Equation:** $TTFT_{\text{worst}} = T_{\text{wait}} + T_{\text{process}} \approx 2 \times T_{\text{iteration}}$

  > **Options:**
  > [ ] The worst-case TTFT is 150ms, which meets the SLO. (Forgets wait time)
  > [ ] The worst-case TTFT is 650ms (500ms SLO + 150ms process), which violates the SLO. (Incorrectly adds SLO to calculation)
  > [x] The worst-case TTFT is 300ms, which meets the SLO. (Correctly accounts for wait time + process time)
  > [ ] It's impossible to know without the number of users in the batch. (Incorrectly assumes iteration time depends on batch size)

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Throughput Saturation Fallacy</b> · <code>throughput-queueing</code></summary>

- **Interviewer:** "You are capacity planning for an LLM service that uses H100 GPUs. When fully saturated, your monitoring shows a single H100 can produce a total of 3,000 tokens per second using continuous batching. You need to calculate the total time required to serve a workload of 5,000 total tokens.

Compare Scenario A (100 concurrent users, each requesting 50 tokens) with Scenario B (1 user requesting 5,000 tokens). Assuming the GPU operates at its peak saturated throughput, calculate the total time to complete the entire workload in each scenario."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often conflate per-user latency with total system throughput. They might incorrectly assume that handling 100 concurrent users (Scenario A) must be inherently slower due to 'overhead' or 'context switching', even when the total amount of computational work (5,000 tokens) is identical to the single-user scenario and the system is specified to be operating at a constant peak throughput.

  **Realistic Solution:** From a total throughput perspective, the scenarios are identical. The GPU's job is to generate 5,000 tokens, and it does so at a rate of 3,000 tokens per second. The distribution of those tokens among users doesn't change the total amount of work.

The calculation is the same for both scenarios: Total Time = Total Tokens / Token Rate. This results in 5,000 tokens / 3,000 tokens/sec ≈ 1.67 seconds. While the user-perceived latency would be drastically different between the scenarios (especially TTFT), the time for the GPU to complete the *entire batch of work* is the same.

  > **Napkin Math:** 1. **Define Total Workload (Scenario A)**: $100 \text{ users} \times 50 \frac{\text{tokens}}{\text{user}} = 5,000 \text{ tokens}$.
2. **Define Total Workload (Scenario B)**: $1 \text{ user} \times 5,000 \frac{\text{tokens}}{\text{user}} = 5,000 \text{ tokens}$.
3. **Define System Throughput**: $R = 3,000 \frac{\text{tokens}}{\text{second}}$.
4. **Calculate Time to Complete**: $T_{\text{total}} = \frac{\text{Total Tokens}}{R}$.
5. **Result**: $T_{\text{total}} = \frac{5,000}{3,000} \approx 1.67 \text{ seconds for both scenarios}$.

  > **Key Equation:** $T_{\text{total}} = \frac{\sum(\text{Tokens per request})}{\text{System Throughput}}$

  > **Options:**
  > [ ] Scenario A is slower due to multi-user overhead. (Misinterprets 'saturated throughput')
  > [ ] Scenario B is slower because long generations are inefficient. (Confuses sequence length with system throughput)
  > [x] Both scenarios take ~1.67 seconds to complete. (Correctly calculates total work / rate)
  > [ ] Both scenarios take ~0.06 seconds to complete. (Unit error, likely 3000 / 5000 / 100 or similar)

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Network Tax: NVLink vs. InfiniBand</b> · <code>interconnect-latency-comparison</code></summary>

- **Interviewer:** "You are optimizing a large-scale LLM training job. To make informed decisions about your communication strategy, you need to know your hardware's limits. Roughly how much slower is a cross-rack data transfer using InfiniBand NDR compared to a GPU-to-GPU transfer within the same server using NVLink 4.0?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often conflate different types of interconnects, assuming all 'fast networks' are roughly equivalent. They might underestimate the significant latency penalty of leaving the server chassis and traversing the data center fabric, assuming it's a minor difference (e.g., 2x) rather than an order-of-magnitude jump.

  **Realistic Solution:** A cross-rack InfiniBand NDR transfer is approximately 10 times slower than an intra-server NVLink 4.0 transfer. NVLink latency is ~500 ns, while crossing racks over InfiniBand takes ~5,000 ns (5 µs). This 'network tax' is a fundamental constraint in distributed systems design and dictates choices like communication-avoiding algorithms and network topology-aware scheduling.

  > **Napkin Math:** Using the 'human time' analogy where 1ns is 1 second:
- An NVLink 4.0 transfer takes ~500 seconds, or about **8 minutes**.
- An InfiniBand NDR transfer takes ~5,000 seconds, or about **1.4 hours**.

The ratio is `1.4 hours / 8 minutes ≈ 84 minutes / 8 minutes ≈ 10.5×`.

  > **Options:**
  > [ ] They are roughly the same latency (~1x)
  > [ ] About 2x slower
  > [x] About 10x slower
  > [ ] About 100x slower

  📖 **Deep Dive:** [Distributed Systems](https://mlsysbook.ai/vol2/distributed_training.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Speed of Light Constraint in RAG</b> · <code>latency-rag-network</code></summary>

- **Interviewer:** "You're designing a Retrieval-Augmented Generation (RAG) system. The core LLM is in a datacenter in Virginia, but to satisfy data residency requirements, the vector database for European users is hosted in a datacenter in Ireland. Ignoring all other sources of latency (compute, disk I/O, queuing), what is the approximate, physics-imposed round-trip time (RTT) for a single document retrieval across the Atlantic?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse inter-rack latency (microseconds) with cross-continental latency (tens of milliseconds). They dramatically underestimate the non-optimizable delay imposed by the speed of light in fiber, assuming it's a few milliseconds at most.

  **Realistic Solution:** The absolute minimum round-trip latency is dictated by the speed of light through fiber optic cables. A trip across the US is roughly 40ms, and across the Atlantic is similar or slightly longer. The closest order-of-magnitude answer is ~40ms. This is a hard physical floor on performance for any synchronous, cross-continental operation.

  > **Napkin Math:** From the 'ML Latency Hierarchy' numbers, a cross-country fiber round trip in the US is ~40,000,000 ns (40 ms).

To put this in perspective: If a single clock cycle on the CPU (L1 cache access) took 1 second, then this 40ms network round trip would take 1.2 years. This is a fundamental, non-negotiable cost of distance.

  > **Options:**
  > [ ] ~5 µs
  > [ ] ~1 ms
  > [x] ~40 ms
  > [ ] ~500 ms

  📖 **Deep Dive:** [Distributed Systems](https://mlsysbook.ai/cloud/02_distributed_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Blue/Green Memory Budget</b> · <code>model-serving-rollout</code></summary>

- **Interviewer:** "You are deploying an updated 7-billion parameter LLM to your company's RAG-based chatbot service. The model is served in FP16 precision. Your Kubernetes cluster uses a blue/green deployment strategy, where the new model pod must be healthy and running before traffic is switched over from the old one. To ensure a smooth rollout without evicting other pods, you need to calculate the peak GPU memory required on a single multi-model node during this transition. Explain how you would calculate the total GPU memory required to hold both the old and the new models simultaneously during the update."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to calculate the memory for only a single model instance. This forgets that during a blue/green deployment, the node must have enough capacity to hold both the old and new model containers in memory at the same time before the old one is terminated. Another common error is using the memory footprint for a different precision, like INT8 (1 byte/param) or FP32 (4 bytes/param), instead of the specified FP16.

  **Realistic Solution:** To calculate the peak memory, you need to account for both models co-existing on the same GPU. For a 7B parameter model in FP16 (which uses 2 bytes per parameter), a single instance requires 14 GB of memory. Since a blue/green deployment requires both the old and new versions to be resident before traffic switchover, the total required memory is double that of a single instance.

  > **Napkin Math:** 1. **Memory for one model:** 7 Billion parameters × 2 bytes/parameter (for FP16) = 14 GB.
2. **Peak memory for Blue/Green:** 2 models (old + new) × 14 GB/model = 28 GB.

  > **Key Equation:** $\text{Peak Memory} = (\text{Parameters} \times \text{Bytes per Parameter}) \times N_{\text{concurrent_models}}$

  > **Options:**
  > [ ] 14 GB. The new model replaces the old one, so you only need space for one.
  > [ ] 7 GB. A 7B model requires 7GB of memory.
  > [x] 28 GB. Both the old and new 14 GB models must be in memory at the same time.
  > [ ] 56 GB. An FP16 model uses 4 bytes/param, and you need two of them.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Geographic Skew Tax</b> · <code>data-latency</code></summary>

- **Interviewer:** "You're debugging a production model and suspect training-serving skew. Your offline training pipeline pulls raw logs from a cheap archival storage system located in a different country, while your online serving system uses a feature store running on local NVMe SSDs. To quantify the data access bottleneck, state the approximate latency difference: how much slower is a single read from the cross-country archive compared to the local SSD?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate the speed-of-light penalty for wide-area networks (WANs). They might intuitively guess it's 10-50x slower, failing to realize that a cross-country round trip is hundreds of times slower than even 'slow' local storage like SSDs.

  **Realistic Solution:** A read from the cross-country archive is approximately 400 times slower. A cross-country fiber round-trip takes about 40ms, while a local NVMe SSD read is about 100µs. The difference is a factor of 400, highlighting a massive potential source of training-serving skew if data preprocessing differs.

  > **Napkin Math:** Cross-country Fiber RTT ≈ 40,000,000 ns
NVMe SSD Read ≈ 100,000 ns

Ratio = 40,000,000 ns / 100,000 ns = 400x

In human-scaled time: If an SSD read took 1 day, waiting for the cross-country data would take 1.2 years.

  > **Options:**
  > [ ] ~4x slower
  > [ ] ~40x slower
  > [x] ~400x slower
  > [ ] ~4,000x slower

  📖 **Deep Dive:** [Distributed Systems](https://mlsysbook.ai/vol2/distributed_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Llama 3 KV Cache Footprint</b> · <code>kv-cache-vram</code></summary>

- **Interviewer:** "You are deploying a Llama 3 8B model for inference, which uses Grouped-Query Attention (GQA). A request comes in with a sequence length of 8,192 tokens. Explain how you would calculate the VRAM required just for the FP16 KV-cache for this single request, and compare the potential options."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to confuse the number of query heads (32 for Llama 3 8B) with the much smaller number of key/value heads (8 for Llama 3 8B). In GQA, the number of KV heads is the one that determines the cache size. Other frequent errors include forgetting to account for both the Key and the Value tensors (a 2x factor) or using the wrong number of bytes for FP16 precision (it's 2, not 4 like FP32).

  **Realistic Solution:** The KV-cache stores the key and value vectors for every token in the context window, across all attention layers. The total size is the product of the sequence length, number of layers, 2 (for K and V), number of KV heads, the dimension of each head, and the number of bytes per value. For a model with Grouped-Query Attention (GQA) like Llama 3 8B, it's critical to use the number of KV heads, which is intentionally smaller than the number of query heads to save memory.

  > **Napkin Math:** Llama 3 8B specs: 32 layers, 8 KV heads, 128 head dimension.
- Precision: FP16 = 2 bytes per value.
- Sequence Length: 8,192 tokens.

- Total Bytes = `seq_len` × `layers` × 2 (for K and V) × `kv_heads` × `head_dim` × `bytes_per_value`
- Total Bytes = 8,192 × 32 × 2 × 8 × 128 × 2
- Total Bytes = 1,073,741,824 bytes

- Convert to GiB: 1,073,741,824 / (1024^3) = 1 GiB.

  > **Key Equation:** $\text{KV Cache Size} = S \times L \times 2 \times H_{kv} \times D_h \times B_p$

  > **Options:**
  > [ ] 4 GiB
  > [ ] 0.5 GiB
  > [x] 1 GiB
  > [ ] 2 GiB

  📖 **Deep Dive:** [The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html#kv-cache)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The True Cost of Batching on TTFT</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You're designing an LLM serving system on a single H100 GPU. The product manager insists on a 'real-time' user experience, which they've defined with a strict Service Level Objective (SLO): Time-To-First-Token (TTFT) must be under 250ms. Your system improves GPU utilization by queueing incoming requests and forming batches. A new batch is processed every 100ms. Given that the prompt processing (prefill) for a single request takes about 15ms on this hardware, explain the maximum number of requests you can group into a single batch without violating the TTFT SLO."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate the max batch size by only considering the processing time (`250ms / 15ms ≈ 16 requests`). They forget to account for the maximum time a request might have to wait in the queue *before* processing begins, which is a critical component of the end-to-end latency.

  **Realistic Solution:** The total latency experienced by a user is the sum of the time spent waiting for a batch to be formed (queue time) and the time it takes to process the batch. In the worst-case scenario, a request arrives just after a batch has been dispatched, meaning it must wait the full 100ms for the next batching window. This leaves `250ms (SLO) - 100ms (Max Queue Time) = 150ms` for the actual batch processing. To find the max batch size, we divide the remaining time by the per-request processing time: `150ms / 15ms/request = 10 requests`.

  > **Napkin Math:** 1. **Identify the Total Latency Budget:** SLO for TTFT = 250ms.
2. **Identify Worst-Case Queue Time:** The system forms a batch every 100ms. A request could arrive 1ms after a batch starts, so it waits ~100ms. `T_queue_max = 100ms`.
3. **Calculate the Remaining Time for Processing:** `T_processing_budget = TTFT_SLO - T_queue_max = 250ms - 100ms = 150ms`.
4. **Calculate Max Batch Size:** The entire batch prefill must complete within this budget. `Max_Batch_Size = T_processing_budget / T_prefill_per_request = 150ms / 15ms = 10 requests`.

  > **Key Equation:** $\text{TTFT} = T_{\text{queue}} + (N_{\text{batch}} \times T_{\text{prefill}})$

  > **Options:**
  > [x] 10 requests. The worst-case queueing time must be subtracted from the SLO before calculating batch capacity.
  > [ ] 16 requests. The SLO of 250ms can be divided directly by the 15ms per-request time.
  > [ ] 6 requests. This assumes the queue time (100ms) and processing time (150ms) are independent.
  > [ ] It's unlimited. The H100 is fast enough that prefill time is negligible.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The TPOT Memory Wall</b> · <code>llm-serving-throughput</code></summary>

- **Interviewer:** "Your team is serving a 70B parameter LLM on a single H100 GPU. A user complains that while the first token appears almost instantly, the subsequent text generates very slowly, making the chatbot feel sluggish. They expect a comfortable reading speed of at least 20 tokens per second. Interpret this complaint: what is the primary hardware bottleneck for Time Per Output Token (TPOT), and can the H100 GPU theoretically meet this user's expectation? Use the provided hardware constants."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to assume that token generation is compute-bound. An engineer might incorrectly try to calculate the FLOPs required per token and divide by the H100's TFLOPS rating. This ignores the fact that autoregressive decoding is a memory-bound operation, as the entire model's weights must be read from HBM for each token generated.

  **Realistic Solution:** The user's complaint correctly distinguishes between TTFT (fast) and TPOT (slow). The bottleneck for TPOT in autoregressive models is memory bandwidth, as generating each token requires reading all model parameters from High Bandwidth Memory (HBM).

A 70B parameter model using FP16 precision requires `70 * 10^9 params * 2 bytes/param = 140 GB` of memory. The H100 has an HBM3 memory bandwidth of 3.35 TB/s. The time to read the entire model is `140 GB / 3350 GB/s ≈ 0.0418 seconds`, or 41.8ms per token. This translates to a theoretical maximum generation speed of `1 / 0.0418s ≈ 23.9 tokens/second`.

Therefore, the hardware is theoretically capable of meeting the user's 20 token/sec expectation. The perceived slowness is likely due to software overhead, network latency, or inefficient kernel implementation, not a fundamental hardware limitation.

  > **Napkin Math:** 1. **Identify the Bottleneck:** Autoregressive token generation (TPOT) is memory-bandwidth bound.
2. **Calculate Model Size in Memory:** For a 70B model in FP16: `70B params * 2 bytes/param = 140 GB`.
3. **Look up Hardware Spec:** H100 HBM3 Memory Bandwidth = 3.35 TB/s (or 3350 GB/s).
4. **Calculate Time per Token:** `Time = Total Data / Bandwidth = 140 GB / 3350 GB/s ≈ 0.0418 s/token` (or 41.8 ms/token).
5. **Calculate Tokens per Second:** `TPOT = 1 / Time_per_token = 1 / 0.0418 s ≈ 23.9 tokens/sec`.
6. **Compare to Requirement:** `23.9 tokens/sec (theoretical max) > 20 tokens/sec (user expectation)`. The hardware is sufficient.

  > **Key Equation:** $\text{TPOT (tokens/sec)} = \frac{\text{Memory Bandwidth (Bytes/sec)}}{\text{Model Size (Bytes)}}$

  > **Options:**
  > [ ] No, the H100 is compute-bound during generation and cannot meet this demand.
  > [ ] No, the H100's memory bandwidth is only 3.35 GB/s, making this impossible.
  > [x] Yes, the H100's memory bandwidth supports a theoretical speed of ~24 tokens/sec, so the issue is likely in the software.
  > [ ] Yes, because the KV-cache makes all subsequent token generation instantaneous.

  📖 **Deep Dive:** [ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The RAG Retrieval Step-Cost</b> · <code>rag-latency</code></summary>

- **Interviewer:** "You're debugging a slow Retrieval-Augmented Generation (RAG) pipeline. The system takes a user query, retrieves relevant documents from a vector index, and then feeds them to an LLM. The vector index is too large for memory and is stored on a datacenter-grade NVMe SSD. What is the approximate latency you should expect for a single, random read from this SSD to retrieve a document chunk?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate I/O latency, thinking of it in the single-digit microsecond range. They confuse the speed of storage I/O with much faster network or memory access, failing to realize that reading from flash is orders of magnitude slower than HBM or even DRAM.

  **Realistic Solution:** The expected latency is around 100,000 nanoseconds (100 µs). This is a fundamental physical limitation of current flash storage technology in datacenters. While a full RAG pipeline has many components (embedding, LLM inference), isolating the retrieval step shows that I/O can be a significant bottleneck compared to on-chip operations.

  > **Napkin Math:** Based on the standard ML Latency Hierarchy, an L1 cache access is ~1 ns. A read from an NVMe SSD is ~100,000 ns. If you scale that to human time where 1 ns is 1 second, the L1 access takes 1 second, but the SSD read takes over a day (1.1 days). This is the 'cost' of going to storage.

  > **Options:**
  > [ ] ~300 ns
  > [ ] ~5,000 ns (5 µs)
  > [x] ~100,000 ns (100 µs)
  > [ ] ~40,000,000 ns (40 ms)

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The FP16 Memory Footprint</b> · <code>model-serving-memory</code></summary>

- **Interviewer:** "Your team is upgrading a production language model from 7B to 13B parameters as part of a new rollout. Your serving stack uses NVIDIA H100 GPUs and the model is served in FP16 precision. To inform the container orchestration plan, calculate the memory required for the new 13B model's weights and explain the impact on a single GPU's memory capacity."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the memory requirements for *training* versus *inference*. During training with the Adam optimizer, memory usage is ~16 bytes per parameter. For inference, it's only the model weights, which is much lower. Another common error is using the wrong number of bytes for the given precision (e.g., 4 for FP32 or 1 for INT8 instead of 2 for FP16).

  **Realistic Solution:** The new 13B model will require 26 GB of HBM to store the model weights. This is a significant increase from the 14 GB required by the 7B model, but it still fits comfortably within the 80 GB of HBM available on a single H100 GPU. This calculation is critical for capacity planning in your orchestration system (like Kubernetes) to ensure pods are scheduled on nodes with sufficient available GPU memory.

  > **Napkin Math:** 1. Identify the number of parameters: 13 Billion
2. Identify the bytes per parameter for the precision: FP16 uses 2 bytes.
3. Calculate the total memory: 13,000,000,000 parameters × 2 bytes/parameter = 26,000,000,000 bytes.
4. Convert bytes to gigabytes: 26,000,000,000 bytes / (1024^3 bytes/GB) ≈ 24.2 GB. In industry napkin math, we often approximate by dividing by 10^9, giving 26 GB.

  > **Key Equation:** $\text{Inference Memory (Bytes)} = \text{Parameters} \times \text{Bytes per Parameter}$

  > **Options:**
  > [ ] 13 GB. (Error: Assumes 1 byte per parameter, as with INT8 quantization)
  > [ ] 208 GB. (Error: Confuses inference memory with training memory using the Adam optimizer, which needs ~16 bytes/param)
  > [x] 26 GB. (Correct: 13 billion parameters × 2 bytes for FP16 precision)
  > [ ] 52 GB. (Error: Assumes 4 bytes per parameter, as with FP32 precision)

  📖 **Deep Dive:** [The Iron Law of ML Systems](ironlaw.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Skew from the Disk</b> · <code>training-serving-skew</code></summary>

- **Interviewer:** "A new fraud detection model performs perfectly during offline training, where feature data is loaded from large files that are effectively cached in memory. In the live production environment, its performance drops significantly. You discover that for each inference request, one critical feature must be fetched from a database running on a server with NVMe SSDs. Based on the fundamental physics of a computer, what is a primary suspect for this drop in performance?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often attribute all training-serving skew to software bugs in the feature transformation code. They forget that the *physical reality* of data retrieval is often a major culprit. The difference between having data 'ready' in memory (training) versus fetching it from a much slower medium like an SSD (serving) can introduce significant latency. This delay can cause the system to use a default or stale value for the feature, a pattern the model was never trained on, leading to skew.

  **Realistic Solution:** Accessing data from an NVMe SSD is orders of magnitude slower than accessing data already in HBM or DRAM. According to the standard latency hierarchy, an HBM memory access is around 300 ns, while a random read from an NVMe SSD is about 100,000 ns (100 µs). This massive latency gap (over 300x) means the feature data may not arrive in time for the inference deadline, forcing the system to impute a value. The model's performance degrades because it is seeing a data distribution (with imputed values) that it was never exposed to in training.

  > **Napkin Math:** Using the human-scaled latency numbers: If one memory access from HBM felt like 5 minutes, a single read from the NVMe SSD would feel like 1.1 days. The system is waiting an eternity for the feature at inference time compared to having it instantly available during training.

  > **Options:**
  > [ ] The NVMe SSD read is roughly 10x slower than the memory access, which is a minor source of error.
  > [ ] A subtle floating-point precision difference between the Python training and C++ serving environments.
  > [x] The NVMe SSD read is over 300x slower than HBM memory access, likely causing data to be unavailable at inference time.
  > [ ] The CPU clock speed is dynamically throttled lower during inference, affecting numerical stability.

  📖 **Deep Dive:** [ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The KV-Cache VRAM Budget</b> · <code>kv-cache-vram</code></summary>

- **Interviewer:** "You are scoping the GPU memory requirements for serving a Llama-3 8B model. Your service needs to handle a context length of 8,192 tokens. The model is running in FP16 precision.

Given the model's architecture:
- 32 transformer layers
- 8 Key/Value heads (due to Grouped-Query Attention)
- 128 dimensions per head

Calculate the VRAM consumed *just by the KV-cache* for a single sequence at full context length. Explain your reasoning."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to confuse the memory needed for the KV-cache (which is dynamic and scales with sequence length) with the static memory needed for the model's weights. Another frequent error is using the number of *query* heads instead of the smaller number of *Key/Value* heads for a model with Grouped-Query Attention (GQA), leading to a significant overestimation of the memory requirement.

  **Realistic Solution:** The correct approach is to calculate the total storage needed for the Key and Value tensors across all layers for the entire sequence length. For each token in the sequence, for each layer, we must store a Key vector and a Value vector.

With Grouped-Query Attention (GQA), multiple query heads share a single key/value head, so we use the `n_kv_heads` in our calculation, not the total number of attention heads.

The total size is the product of these dimensions, multiplied by 2 for FP16's byte size.

  > **Napkin Math:** 1. **Identify dimensions:**
   - Sequence Length (S): 8,192 tokens
   - Layers (L): 32
   - K/V Heads (H_kv): 8
   - Head Dimension (D_h): 128
   - Bytes per element (B): 2 (for FP16)

2. **Calculate Cache Size:**
   - `Total Size = 2 (for K & V) * S * L * H_kv * D_h * B`
   - `Total Size = 2 * 8192 * 32 * 8 * 128 * 2` bytes
   - `Total Size = 1,073,741,824` bytes

3. **Convert to Gigabytes:**
   - 1 GB = 1024 * 1024 * 1024 = 2^30 bytes
   - `Total Size = 1,073,741,824 / 2^30` GB
   - `Total Size = 1 GB`

  > **Key Equation:** $\text{KV Cache Size} = 2 \times S \times L \times H_{kv} \times D_h \times \text{bytes}_{elem}$

  > **Options:**
  > [ ] 16 GB
  > [ ] 4 GB
  > [x] 1 GB
  > [ ] 512 MB

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Static Batching Penalty</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "An LLM inference service uses static batching to serve a 7B parameter model. The static batching configuration has a timeout of 100ms and a batch size of 8. A user sends a single request to an idle server. Explain the minimum time the user must wait for the *first* token, considering only the delay introduced by the batching strategy itself."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often forget the 'timeout' component of static batching. They might assume that if a batch isn't full, the request waits indefinitely, or conversely, that a single request is processed immediately. They might also incorrectly calculate the per-token processing time instead of identifying the queueing delay, which is the dominant factor for TTFT in this scenario.

  **Realistic Solution:** The minimum wait time is determined by the static batching timeout. When a single request arrives, the server doesn't have a full batch (1 of 8). It will therefore wait, hoping more requests arrive to fill the batch. If no other requests come, it will wait for the entire 100ms timeout period before dispatching the incomplete batch to the GPU. Therefore, the batching strategy itself introduces a minimum of 100ms of latency before the model even begins processing the request (prefill).

  > **Napkin Math:** 1. Request arrives at T=0 into an empty queue.
2. Server checks batch size: current_batch=1, target_batch=8.
3. The batch is not full, so the server waits.
4. The static batching timeout is 100ms.
5. The server will wait until either the batch is full or the timeout is reached.
6. In this case, no other requests arrive, so the server dispatches the batch at T=100ms.
7. Minimum batching delay = 100ms.

  > **Key Equation:** $\text{TTFT} \ge T_{\text{wait}} + T_{\text{prefill}} + T_{\text{decode_first}}$

  > **Options:**
  > [ ] 0ms, because the server is idle and can process the request immediately.
  > [ ] ~5ms, the approximate time to generate a single token on an H100.
  > [x] 100ms, because the server waits for the batching timeout to expire.
  > [ ] 800ms, calculated by multiplying the batch size by the timeout.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> Continuous Batching and TPOT</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "An inference service for a 70B model runs on an H100 GPU using continuous batching. The system's measured time to generate one token for an entire batch (the 'token step') is ~20ms. The service has a strict Time Per Output Token (TPOT) SLO of 33ms to ensure a real-time conversational experience. If a new user request joins a batch that already contains 4 other active users, what is the effective TPOT for that new user?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common misconception is to believe that the time per token step scales linearly with the number of sequences in the batch. Engineers might multiply the base step time (20ms) by the number of users (5), incorrectly concluding the TPOT is 100ms. This fails to recognize that continuous batching systems process one token for *all* sequences in the batch in parallel during a single forward pass.

  **Realistic Solution:** The effective TPOT for any user in a continuous batch is simply the time it takes the system to complete a single token generation step for the entire batch. The power of this technique is that the step time is largely independent of the number of sequences (up to a point). Since the token step time is given as 20ms, the TPOT for the new user—and all other users in the batch—is 20ms. This is well within the 33ms SLO.

  > **Napkin Math:** 1. System uses continuous batching.
2. Time per token generation step for the entire batch = 20ms.
3. This step generates one token for *every* sequence in the batch simultaneously.
4. Therefore, the TPOT experienced by any single user is equal to the batch step time.
5. Effective TPOT = 20ms.
6. Compare to SLO: 20ms < 33ms. The system meets the deadline.

  > **Key Equation:** $\text{TPOT}_{\text{effective}} = T_{\text{step}}$

  > **Options:**
  > [ ] 100ms, because the 20ms step time is multiplied by the 5 users in the batch.
  > [ ] 4ms, because the 20ms step time is divided by the 5 users in the batch.
  > [x] 20ms, because one token is generated for all users in a single step.
  > [ ] 33ms, because the system will throttle to match the SLO exactly.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The 4x Data Cost Bug</b> · <code>training-serving-skew</code></summary>

- **Interviewer:** "You manage the data ingest pipeline for a fleet of 100 autonomous vehicles. An engineer, intending to improve data fidelity for the next training run, just pushed an OTA update. The update changes the format of uploaded camera images from a compressed JPEG (averaging 0.5 bytes/pixel) to uncompressed FP16 (2 bytes/pixel).

Each vehicle has 8 cameras, captures at 1920x1080 resolution, and runs at 10 FPS. For retraining purposes, all sensor data is uploaded to your cloud bucket while the vehicle is active, which is about 4 hours per day.

First, explain the category of problem this change introduces. Second, calculate the new total data volume the fleet will upload per day."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Candidates often focus only on the engineer's positive intent (improving data fidelity) and miss the immediate, massive negative impact on operational costs and system stability. They may also miscalculate the data increase, confusing byte-per-pixel values (e.g., using 1 for INT8 or 4 for FP32) or forgetting to account for all variables like the number of cameras or vehicles in the fleet.

  **Realistic Solution:** This is a classic example of a data pipeline issue causing a severe form of training-serving skew, not at the model-prediction level, but at the system's operational level. The change from a highly efficient compressed format to a raw, high-precision format introduces a 4x increase in data volume.

This explodes data transfer and storage costs overnight. It can saturate the vehicles' cellular uplinks, leading to data loss, and overwhelm the cloud ingest pipeline, potentially causing cascading failures. The operational cost increase likely far outweighs any marginal model quality benefit, making it a critical system design failure.

The correct calculation for the new daily data volume is approximately 460 TB.

  > **Napkin Math:** We calculate the total data volume step-by-step:

1.  **Pixels per frame:** 1920 × 1080 ≈ 2.1 million pixels
2.  **Data per frame (FP16):** 2.1M pixels × 2 bytes/pixel = 4.2 MB
3.  **Data per second (per camera):** 4.2 MB/frame × 10 FPS = 42 MB/s
4.  **Data per second (per vehicle):** 42 MB/s × 8 cameras = 336 MB/s
5.  **Data per day (per vehicle):** 336 MB/s × (4 hours × 3600 s/hr) = ~4,838 GB/day ≈ 4.8 TB/day
6.  **Total Fleet Data per Day:** 4.8 TB/vehicle × 100 vehicles ≈ 480 TB/day.

Rounding to two significant figures, we get ~480 TB.

  > **Key Equation:** $\text{Data Volume} = N_{\text{vehicles}} \times N_{\text{cameras}} \times (H \times W \times B_{\text{pixel}}) \times \text{FPS} \times T_{\text{active}}$

  > **Options:**
  > [ ] ~240 TB per day
  > [ ] ~48 TB per day
  > [x] ~480 TB per day
  > [ ] ~4.8 TB per day

  📖 **Deep Dive:** [Numbers Every ML Systems Engineer Should Know](NUMBERS.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The KV-Cache VRAM Budget</b> · <code>kv-cache-vram-accounting</code></summary>

- **Interviewer:** "You're deploying a Llama-70B model for a high-throughput inference service. The model uses FP16 precision. To correctly provision your H100 GPUs, you need to calculate the memory budget. Explain how much VRAM will be consumed *solely by the KV-cache* if you need to support a maximum context length of 128,000 tokens."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often forget one of two key multipliers in the KV-cache formula: either they forget the factor of 2 for the Key and Value tensors, underestimating the cache size by half; or they confuse the dynamic KV-cache size, which scales with sequence length, with the static model weight size.

  **Realistic Solution:** The KV-cache stores the Key and Value vectors for every token in the sequence at each decoder layer. For a model like Llama-70B, which has 80 layers and a hidden dimension of 8192, the calculation for a 128k token sequence in FP16 is straightforward. The total size is approximately 335.5 GB, which immediately shows that the cache for a single user's long context request will not fit on a single 80 GB H100 GPU, highlighting the critical need for tensor parallelism even for inference.

  > **Napkin Math:** Total VRAM = Sequence Length × Num Layers × 2 (K/V) × Hidden Dimension × Bytes per Element

- Sequence Length (S): 128,000 tokens
- Number of Layers (L): 80 (for Llama-70B)
- Hidden Dimension (H): 8192 (for Llama-70B)
- Bytes per Element (B_fp16): 2

Calculation:
`VRAM = 128000 × 80 × 2 × 8192 × 2`
`VRAM = 335,544,320,000 bytes`
`VRAM ≈ 335.5 GB`

  > **Key Equation:** $\text{VRAM}_{KV} = S \times L \times 2 \times H_{D} \times \text{bytes_per_element}$

  > **Options:**
  > [ ] ~168 GB
  > [ ] 140 GB
  > [x] ~335 GB
  > [ ] ~671 GB

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Queue</b> · <code>continuous-batching-queueing</code></summary>

- **Interviewer:** "Your team runs an LLM inference service on a single H100 GPU using continuous batching. The GPU can sustain a total throughput of 800 tokens/second across all concurrent requests. The service receives an average of 5 requests per second (RPS), and each request requires the model to generate 128 tokens. Explain if the system is stable and calculate the average time a request spends in the system (from arrival to completion)."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Confusing the total token throughput of the system with the per-request service rate. Engineers might incorrectly calculate the service time for a single request in isolation, ignoring that the 800 tokens/second is a shared resource. Another common mistake is to see that the 'required throughput' (5 RPS * 128 tokens = 640 tokens/s) is less than the 'max throughput' (800 tokens/s) and assume latency is low, without quantifying the impact of queueing at high utilization.

  **Realistic Solution:** The system is stable because the required token throughput (640 tokens/s) is less than the system's maximum capacity (800 tokens/s). However, with a system utilization of 80%, we can expect significant queueing delays. We can model this as an M/M/1 queue. The average time a request spends in the system is not just its service time, but the service time divided by one minus the utilization, which accounts for the time spent waiting in the queue.

  > **Napkin Math:** 1. **Calculate Required Throughput (λ_tokens):**
   - Arrival Rate (λ_requests) = 5 RPS
   - Tokens per Request = 128
   - Required Throughput = 5 requests/s * 128 tokens/request = 640 tokens/s

2. **Calculate System Utilization (ρ):**
   - System Capacity (μ_tokens) = 800 tokens/s
   - Utilization (ρ) = Required Throughput / System Capacity = 640 / 800 = 0.8

3. **Calculate Average Service Time per Request (W_s):**
   - A single request needs 128 tokens from a system that provides 800 tokens/s.
   - Service Time (W_s) = 128 tokens / 800 tokens/s = 0.16 s = 160 ms

4. **Calculate Average Time in System (W):**
   - Using the M/M/1 queue formula, W = W_s / (1 - ρ)
   - W = 160 ms / (1 - 0.8) = 160 ms / 0.2 = 800 ms

  > **Key Equation:** W = \frac{W_s}{1 - \rho}

  > **Options:**
  > [ ] 160 ms. The system is under capacity, so latency is simply the time it takes to service one request.
  > [ ] 200 ms. The latency is the service time plus a 20% overhead for being busy.
  > [x] 800 ms. The system is 80% utilized, leading to significant queueing delay which quintuples the average latency.
  > [ ] The system is unstable and will crash, because the required token rate is too close to the maximum.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Blue/Green Memory Squeeze</b> · <code>serving-rollout-capacity</code></summary>

- **Interviewer:** "You are managing a fleet of H100 GPUs, each with 80 GB of HBM memory. Your current service runs a 15-billion parameter model served in FP16 precision. The team wants to upgrade to a 30-billion parameter model using a blue/green deployment strategy on each node. This means that for a brief period, both the old and new models must be loaded into a single GPU's memory simultaneously before traffic is switched. Can a single H100 support this deployment strategy? Explain your calculation."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate the memory needed for the new model and see if it fits, completely forgetting that a blue/green deployment requires holding *both* the old and new models in memory concurrently. This leads to an incorrect assessment that the deployment is possible, which would cause out-of-memory (OOM) errors in production.

  **Realistic Solution:** No, a single H100 cannot support this strategy. The total memory required during the transition exceeds the GPU's capacity.

1.  **Old Model Memory:** The 15B parameter model at FP16 precision requires `15e9 * 2 bytes = 30 GB`.
2.  **New Model Memory:** The 30B parameter model at FP16 precision requires `30e9 * 2 bytes = 60 GB`.
3.  **Total Required Memory:** For a blue/green deployment, the total memory is the sum of both models: `30 GB + 60 GB = 90 GB`.
4.  **Conclusion:** The required 90 GB is greater than the H100's 80 GB capacity, so the deployment will fail.

  > **Napkin Math:** Old Model: 15B params × 2 bytes/param (FP16) = 30 GB
New Model: 30B params × 2 bytes/param (FP16) = 60 GB
Blue/Green Total: 30 GB (old) + 60 GB (new) = 90 GB

Capacity Check: 90 GB (required) > 80 GB (H100 capacity) -> Fails.

  > **Key Equation:** $\text{Total Memory} = (P_{old} \times \text{bytes/param}) + (P_{new} \times \text{bytes/param})$

  > **Options:**
  > [ ] Yes, the new 60 GB model fits within the 80 GB capacity.
  > [ ] Yes, the total memory needed is only 45 GB (15 GB + 30 GB).
  > [x] No, the combined 90 GB footprint exceeds the 80 GB capacity.
  > [ ] Yes, there is 50 GB free, and the new model is only 30 GB larger.

  📖 **Deep Dive:** [Inference and Serving](https://mlsysbook.ai/cloud/03_inference_and_serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The True Cost of an A/B Test</b> · <code>economics-serving-cost</code></summary>

- **Interviewer:** "You are designing an A/B experiment to evaluate a new, larger recommendation model. The current production model (Group A) is 1 billion parameters. The new experimental model (Group B) is 7 billion parameters, which you hypothesize will improve user engagement significantly. To serve these models for inference, they must be loaded into H100 GPU memory. Compare the memory footprint required to serve one instance of the experimental model versus one instance of the production model, assuming both are using FP16 precision."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate the memory for the new model in isolation, but forget to compare it to the baseline. They might also confuse the bytes-per-parameter for different precisions, using 4 for FP32 or 1 for INT8 instead of the 2 bytes required for FP16. The most critical mistake is failing to connect this memory increase to the direct economic impact on serving cost—fewer model replicas can be packed onto a single expensive GPU, increasing the cost-per-query.

  **Realistic Solution:** The analysis requires calculating the memory for each model and then finding the difference. The economic implication is key.

1.  **Production Model (1B):** `1,000,000,000 parameters × 2 bytes/parameter (FP16) = 2 GB`.
2.  **Experimental Model (7B):** `7,000,000,000 parameters × 2 bytes/parameter (FP16) = 14 GB`.

The experimental model requires `14 GB - 2 GB = 12 GB` of additional memory per instance. This means an H100 GPU with 80 GB of HBM can host `floor(80/2) = 40` replicas of the old model, but only `floor(80/14) = 5` replicas of the new one. This is an 8x reduction in server density, meaning the A/B test isn't just measuring user engagement; it's testing if the engagement lift is worth an ~8x increase in serving hardware cost.

  > **Napkin Math:** 1. **Identify Memory Formula:** `Inference Memory = Parameters × Bytes_per_Parameter`
2. **Identify Bytes for FP16:** From scaling rules, FP16 uses 2 bytes.
3. **Calculate Production Memory:** `1B params × 2 bytes/param = 2 GB`
4. **Calculate Experimental Memory:** `7B params × 2 bytes/param = 14 GB`
5. **Calculate Difference:** `14 GB (Experimental) - 2 GB (Production) = 12 GB`

  > **Key Equation:** $\text{Inference Memory (GB)} = \frac{\text{Parameters} \times \text{Bytes per Parameter}}{10^9}$

  > **Options:**
  > [ ] 24 GB. The experiment will have a higher memory cost due to using 4 bytes per parameter.
  > [ ] 6 GB. The memory increase is manageable as it only requires 1 byte per parameter.
  > [x] 12 GB. The experimental model requires 7x more memory, significantly increasing the serving cost per user in the A/B test.
  > [ ] 14 GB. The experimental model needs 14 GB, which is the primary cost driver.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The KV-Cache VRAM Budget</b> · <code>kv-cache-vram-accounting</code></summary>

- **Interviewer:** "You are tasked with serving a Llama 70B model on H100 GPUs. A single user request arrives with a large context window of 128,000 tokens. Explain how to calculate the VRAM required *just for the KV-cache* for this single user, assuming FP16 precision. Can this request be handled by a single H100?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to only consider the memory for model weights (70B params × 2 bytes/param ≈ 140 GB) and ignore the dynamic memory consumption of the KV-cache. The KV-cache size is not fixed; it scales linearly with the sequence length and can easily surpass the weight memory for large contexts, becoming the primary memory bottleneck.

  **Realistic Solution:** The request cannot be handled by a single H100 GPU because the KV-cache alone requires over four times the available VRAM. The calculation shows that for long sequences, the KV-cache, not the model weights, is the dominant factor in memory consumption. This necessitates multi-GPU inference (e.g., using Tensor Parallelism) not just to hold the weights, but also to shard the distributed KV-cache across multiple accelerators.

  > **Napkin Math:** # 1. Get specs from the problem and constants
Model: Llama 70B (80 layers, 8192 hidden dimension)
Sequence Length (S): 128,000 tokens
Precision: FP16 (2 bytes per element)
GPU VRAM: An H100 has 80 GB HBM3

# 2. Use the standard formula for total KV-cache size
KV Cache Size = S × Num_Layers × Hidden_Dim × 2 (for K and V) × Bytes_per_element
KV Cache Size = 128,000 × 80 × 8192 × 2 (K,V) × 2 bytes
KV Cache Size = 335,544,320,000 bytes

# 3. Convert bytes to Gigabytes (using power-of-10 for simplicity, matching the reference)
KV Cache Size ≈ 335 GB

# 4. Compare to GPU memory
Required VRAM (≈335 GB) > Available VRAM (80 GB)

# 5. Conclusion
The KV-cache is more than 4x the size of the GPU's memory. It will not fit.

  > **Key Equation:** $\text{KV Cache Size} = S \times L \times D_{hidden} \times 2 \times \text{sizeof(FP16)}$

  > **Options:**
  > [ ] ~140 GB. It will not fit, as this is larger than the H100's 80GB VRAM.
  > [ ] ~84 GB. It will just fit, but leaves no room for the model weights.
  > [x] ~335 GB. It will not fit, as this is over 4x the H100's 80GB VRAM.
  > [ ] ~4.2 GB. It fits easily with room for weights and activations.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Dilemma</b> · <code>continuous-batching-tradeoffs</code></summary>

- **Interviewer:** "Your team is serving a 7B parameter LLM on an H100 GPU using continuous batching for a chat application. The goal is to maximize token throughput. The current system waits up to 20ms to collect incoming requests before dispatching a batch for processing. A junior engineer proposes increasing this wait time to 100ms, arguing that bigger batches lead to higher throughput. Explain the impact of this change on both user-perceived latency (Time to First Token) and overall system throughput (Tokens per second)."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The common mistake is to focus solely on the increased computational efficiency of larger batches (higher throughput) while ignoring the 'wait tax' this imposes on every single user, which directly increases the Time-To-First-Token (TTFT). Engineers often forget that user-perceived latency starts the moment the request is sent, not when the GPU starts processing it.

  **Realistic Solution:** Increasing the batching window from 20ms to 100ms will likely increase the system's maximum theoretical throughput (tokens/second) because the GPU spends more time in efficient, parallel computation and less time on launch overhead. However, it will worsen the user-perceived latency (TTFT) for *every* user. A user whose request arrives at the beginning of the 100ms window now has to wait an additional 80ms before their request even *starts* processing, directly adding to their TTFT. This creates a classic trade-off: higher system efficiency vs. worse individual user experience.

  > **Napkin Math:** Let's compare the average wait time.

*   **Inference Time per Token (H100):** A 7B model requires `~2 * 7B = 14 GFLOPs` per token. An H100 provides `~989 TFLOPS`.
    `T_inference = 14e9 FLOPs / 989e12 FLOPS/sec ≈ 0.014 ms`. This raw compute time is negligible compared to the batching window.
*   **Scenario A (20ms window):**
    *   Assuming requests arrive uniformly, `Average Wait Time ≈ 20ms / 2 = 10ms`.
    *   `User TTFT ≈ 10ms (wait) + T_inference ≈ 10.014ms`.
*   **Scenario B (100ms window):**
    *   `Average Wait Time ≈ 100ms / 2 = 50ms`.
    *   `User TTFT ≈ 50ms (wait) + T_inference ≈ 50.014ms`.

The change adds roughly 40ms to the average user's wait time for their first token, a ~5x increase in perceived latency.

  > **Key Equation:** $\text{TTFT} = \text{T}_{\text{queue\_wait}} + \text{T}_{\text{inference}}$

  > **Options:**
  > [ ] It improves both throughput and user latency because the GPU is more efficient.
  > [ ] It has no significant effect, as inference time is measured in microseconds and dominates latency.
  > [x] It worsens average user latency by roughly 5x, but increases the system's maximum token throughput.
  > [ ] It only improves throughput if the request arrival rate is very high; otherwise, it has no effect.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Little's Law Bottleneck</b> · <code>queueing-theory-and-slos</code></summary>

- **Interviewer:** "You're operating an LLM inference service on a single H100 GPU. The service has a strict P99 SLO that the Time-To-First-Token (TTFT) must be under 100ms. From profiling, you know that the GPU can process a single request and generate the first token in 10ms (this is the service time, T_service). Using basic queueing theory, explain what happens to the user wait time as the average request arrival rate approaches the system's maximum service rate."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming that as long as the arrival rate is less than the service rate (e.g., 99 requests/sec when the service rate is 100 requests/sec), the system is fine. This ignores the non-linear, exponential explosion in queue wait times as system utilization approaches 100%. Engineers often think linearly and don't account for the 'tail at scale' effects on latency caused by queueing.

  **Realistic Solution:** The maximum service rate (μ) is 1 request / 10ms = 100 requests/sec. As the arrival rate (λ) gets closer and closer to 100 req/s, the system utilization (ρ = λ / μ) approaches 1. According to queueing theory, the expected wait time in the queue grows exponentially as utilization nears 100%. A small increase in arrival rate from 90 req/s (ρ=0.9) to 95 req/s (ρ=0.95) will cause a much larger increase in average wait time than an increase from 10 req/s to 15 req/s. To maintain the 100ms P99 SLO, the system must be provisioned to operate at a utilization significantly below 100% (typically 70-80%) to absorb bursts and keep queue times from exploding and violating the latency budget.

  > **Napkin Math:** We can model the wait time using the M/M/1 queue approximation.

*   **Service Rate (μ):** `1 request / 10ms = 100 requests/sec`.
*   **Latency Budget for waiting:** `SLO (100ms) - T_service (10ms) = 90ms`.
*   **System Utilization (ρ):** `λ / μ`, where λ is arrival rate.
*   **Wait Time Formula:** `W ≈ T_service / (1 - ρ)`.

*   **At 50% utilization (λ=50 req/s):**
    `Wait Time ≈ 10ms / (1 - 0.5) = 20ms`. Total TTFT = 10ms + 20ms = 30ms. (SLO met).
*   **At 90% utilization (λ=90 req/s):**
    `Wait Time ≈ 10ms / (1 - 0.9) = 100ms`. Total TTFT = 10ms + 100ms = 110ms. (SLO VIOLATED).

The system cannot even run at 90% utilization without violating the 100ms SLO, demonstrating the non-linear penalty.

  > **Key Equation:** $\text{System Utilization } (\rho) = \frac{\text{Arrival Rate } (\lambda)}{\text{Service Rate } (\mu)}$

  > **Options:**
  > [ ] User wait time increases linearly with the arrival rate.
  > [ ] User wait time remains near zero as long as the arrival rate is below the service rate.
  > [x] User wait time grows exponentially as the arrival rate approaches the service rate, quickly exceeding the latency budget.
  > [ ] The system can handle an arrival rate of 99 requests/sec without violating the 100ms SLO.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Interconnect Latency Ladder</b> · <code>interconnect-latency</code></summary>

- **Interviewer:** "You're profiling a distributed training job and see latency spikes from three different sources: GPU-to-GPU data transfers within a single server, GPU-to-CPU memory copies, and server-to-server communication across the cluster network. Which of these communication links typically has the highest latency?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often conflate bandwidth with latency, assuming high-bandwidth interconnects must have low latency. A common error is to think the general-purpose PCIe bus is the main bottleneck, without appreciating the immense latency cost of going 'off-box' to another server, even over a fast network like InfiniBand.

  **Realistic Solution:** A server-to-server InfiniBand NDR transfer has the highest latency, around 5,000 ns (5 µs). This is because the signal must travel meters of fiber optic cable and pass through multiple network switches and NICs. In contrast, on-server transfers like NVLink (~500 ns) and PCIe Gen5 (~1,000 ns) are an order of magnitude faster because the physical distance is just centimeters of copper on a PCB.

  > **Napkin Math:** Using the '1 ns = 1 second' human-scale analogy from the playbook: A fast NVLink transfer takes about 8 minutes. A PCIe transfer takes about 16 minutes. A cross-rack InfiniBand transfer takes about 1.4 hours. The physics of distance makes the server-to-server trip dramatically slower.

  > **Options:**
  > [ ] NVLink 4.0 Transfer (GPU-GPU, within server)
  > [ ] PCIe Gen5 Transfer (CPU-GPU, within server)
  > [x] InfiniBand NDR Transfer (server-to-server)
  > [ ] HBM3 Memory Access

  📖 **Deep Dive:** [Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The KV-Cache Memory Bomb</b> · <code>kv-cache-vram</code></summary>

- **Interviewer:** "You are scoping hardware for a new service that will run a Llama-2 70B model with a 128k token context window. The model has 80 layers and a hidden dimension of 8192. Your colleague claims a single H100 with 80GB of VRAM is sufficient, since the FP16 weights (~140GB) can be quantized to 4-bits to fit (~35GB). Explain how much VRAM the KV-cache *alone* will consume for a single user at full context length, assuming it's stored in FP16."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus exclusively on the memory required for model weights, which is static. They forget or massively underestimate the KV-cache, which is a dynamic activation that scales linearly with sequence length (`S`) and can easily consume more memory than the weights themselves in long-context scenarios. Another common error is assuming that quantizing the weights will solve all memory issues; the KV-cache is typically stored in higher precision (like FP16) to maintain quality, and its size is unaffected by weight quantization.

  **Realistic Solution:** The colleague's analysis is incorrect because it ignores the massive memory footprint of the KV-cache at long sequence lengths. Each token in the sequence requires storing a Key and a Value vector for each of the 80 layers. The calculation shows this requires 320 GB, which is 4x the available VRAM on a single H100. This makes the proposal infeasible without using multiple GPUs for tensor/pipeline parallelism or specialized memory-saving techniques like activation offloading.

  > **Napkin Math:** The KV-cache stores a Key and a Value vector for each token at each layer.
1. **Formula:** `Total Cache = seq_len × num_layers × 2 (for K/V) × hidden_dim × bytes_per_element`
2. **Plug in values:** `Total Cache = 131,072 × 80 × 2 × 8192 × 2 bytes`
3. **Calculate:** `Total Cache = 343,597,383,680 bytes`
4. **Convert to GB:** `343,597,383,680 bytes / (1024 * 1024 * 1024) = 320 GB`

  > **Key Equation:** $\text{Cache Size} = S \times L \times 2 \times D_{hidden} \times \text{sizeof(dtype)}$

  > **Options:**
  > [ ] Roughly 40 GB. It should fit if weights are quantized.
  > [ ] Roughly 2.6 MB. It's negligible compared to the weights.
  > [x] Roughly 320 GB. It will not fit on a single H100.
  > [ ] Roughly 140 GB, about the same as the model's weights.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Trade-off</b> · <code>continuous-batching-latency</code></summary>

- **Interviewer:** "You're running an LLM inference service on a single H100 GPU. Your system uses a static batching policy with a batch size of 8. The GPU takes 200ms to process one generation step for a full batch. New user requests are arriving at a steady rate of 30 per second. From a user's perspective, what is the approximate Time to First Token (TTFT) if their request arrives right after a batch has been dispatched for processing?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often forget to account for queueing delay. They calculate the model's processing time for the batch but neglect the time the user's request spends waiting for the currently executing batch to finish. This leads to underestimating user-perceived latency by at least 2x.

  **Realistic Solution:** The total Time to First Token is the sum of the time spent waiting in the queue and the actual processing time.
1. **Wait Time:** Since the user's request arrives just after a batch has started, it must wait for that in-flight batch to complete its first generation step. This wait is 200ms.
2. **Processing Time:** After waiting, the request is included in the *next* batch. The processing time for this new batch to generate its first token is also 200ms.
Therefore, the total TTFT is the sum of these two periods.

  > **Napkin Math:** TTFT = Wait Time + Processing Time
Wait Time = Time for current batch to complete one step = 200 ms
Processing Time = Time for the user's batch to complete one step = 200 ms

Total TTFT = 200 ms + 200 ms = 400 ms

  > **Key Equation:** $\text{TTFT} = T_{\text{wait}} + T_{\text{process}}$

  > **Options:**
  > [ ] 200 ms
  > [ ] 10 ms
  > [x] 400 ms
  > [ ] 33 ms

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The KV Cache Memory Trap</b> · <code>llm-serving-memory</code></summary>

- **Interviewer:** "You are serving a 7B parameter LLM on a single H100 GPU with 80 GB of HBM3 memory. A user makes a request with a very long context window of 128k tokens. Identify which component consumes the most memory and is the primary cause of an out-of-memory (OOM) error in this scenario."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often anchor on the model's parameter count (e.g., 'a 7B model is small') as the main driver of memory usage. They forget or underestimate that the KV cache size scales linearly with sequence length and can dwarf the memory required for the model weights, especially with modern long-context windows.

  **Realistic Solution:** The KV Cache is the largest memory consumer. While the model weights are fixed, the KV cache must store the key and value vectors for every token in the input sequence. For a 7B model, the weights are ~14 GB, but the KV cache for a 128k token sequence is ~62.5 GB. Combined, they approach the GPU's capacity limit.

  > **Napkin Math:** 1. **Model Weights Memory:** 7B params × 2 bytes/param (for FP16) = **14 GB**.
2. **KV Cache Memory:** The cache for a single token in a 7B model (like Llama-7B with 32 layers, 32 heads, 128 head_dim) is `2 × 32 × 32 × 128 × 2 bytes` = 524,288 bytes ≈ 0.5 MB. For a 128k sequence: 128,000 tokens × 0.5 MB/token = 64,000 MB ≈ **~62.5 GB**.
3. **Total Memory:** 14 GB (weights) + 62.5 GB (KV Cache) = **76.5 GB**. This leaves only ~3.5 GB for CUDA context overhead, activation buffers, and any batching, making the system extremely memory-constrained and prone to OOM under realistic serving conditions.

  > **Key Equation:** $\text{Memory}_{\text{KV Cache}} = 2 \times N_{\text{layers}} \times L_{\text{sequence}} \times d_{\text{model}}$

  > **Options:**
  > [ ] Model parameters (weights)
  > [ ] Optimizer state (e.g., Adam)
  > [x] The KV Cache
  > [ ] Intermediate activations for the final token

  📖 **Deep Dive:** [Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Blue-Green Memory Tax</b> · <code>model-serving-rollout</code></summary>

- **Interviewer:** "You are an ML Systems Engineer responsible for a fleet of servers running a 7B parameter RAG chatbot. The current version, `v1`, is loaded in FP16 precision. For a safe rollout of the new `v2` model, which includes new guardrail features, your team decides on a blue-green deployment strategy. During the transition, some servers in the fleet must have *both* `v1` and `v2` loaded into memory simultaneously to allow for instant rollbacks. Explain how you would calculate the additional inference memory required per server to support this strategy, and what is that value?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate the total memory for both models (28 GB) instead of the *additional* memory required, failing to read the question carefully. Another common error is confusing FP16 (2 bytes) with INT8 (1 byte), leading to an answer of 7 GB. A less common but critical error is confusing inference memory with training memory, which uses 16 bytes/param for the Adam optimizer, resulting in a wildly incorrect estimate (112 GB).

  **Realistic Solution:** The core task is to calculate the memory footprint of a single 7B model, as this represents the 'additional' load on a server that is already running the `v1` model. A 7 billion parameter model loaded in FP16 (half-precision) requires 2 bytes for every parameter. The calculation is a direct application of the scaling rule for inference memory.

  > **Napkin Math:** Parameters per model: 7 billion
Precision: FP16, which is 2 bytes per parameter.
Calculation: `7,000,000,000 params × 2 bytes/param = 14,000,000,000 bytes`
Conversion to GB: `14,000,000,000 bytes / 10^9 bytes/GB ≈ 14 GB`.
Since `v1` is already loaded, adding `v2` requires an additional 14 GB of HBM per server.

  > **Key Equation:** $\text{Inference Memory} = \text{Parameters} \times \text{Bytes per Parameter}$

  > **Options:**
  > [ ] 7 GB
  > [ ] 28 GB
  > [x] 14 GB
  > [ ] 112 GB

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Runaway KV-Cache</b> · <code>kv-cache-vram</code></summary>

- **Interviewer:** "You are serving a 70B parameter LLM, quantized to INT4, on a single H100 GPU with 80 GB of VRAM. The model weights occupy 35 GB. A user running a summarization task on a long document (64,000 tokens) gets a CUDA out-of-memory error. The model has 80 layers, 64 attention heads, and a head dimension of 128. Calculate the FP16 KV-cache size for this request and explain why it causes the OOM."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus only on the static memory cost of the model weights (~35 GB in this case). They forget that the KV-cache is dynamic and its size scales linearly with the input sequence length. For long-context applications, the KV-cache memory can easily grow to be much larger than the model weight memory.

  **Realistic Solution:** The out-of-memory error is caused by the KV-cache. While the 35 GB of weights fit, the memory required for the KV-cache grows with each token in the sequence. For a 64,000 token sequence, the cache requires approximately 169 GB of VRAM. The total required memory is the sum of the weights and the KV-cache (35 GB + 169 GB = 204 GB), which far exceeds the H100's 80 GB capacity.

  > **Napkin Math:** 1.  **Identify the formula:** The memory for the KV-cache is calculated as: `sequence_length × (2 for K/V) × num_layers × num_heads × head_dim × bytes_per_element`.
2.  **Plug in the values:**
    -   `sequence_length`: 64,000 tokens
    -   `num_layers`: 80
    -   `num_heads`: 64
    -   `head_dim`: 128
    -   `bytes_per_element`: 2 (for FP16 precision)
3.  **Calculate total bytes:** `64,000 × 2 × 80 × 64 × 128 × 2 = 168,884,986,000` bytes.
4.  **Convert to GB:** `168,884,986,000 bytes / (1000^3) ≈ 168.9 GB`.

  > **Key Equation:** $\text{KV Cache (Bytes)} = \text{seq_len} \times 2 \times N_{layers} \times N_{heads} \times d_{head} \times \text{bytes/elem}$

  > **Options:**
  > [ ] ~84.5 GB. This exceeds the available memory.
  > [ ] ~2.6 MB. This is negligible and shouldn't cause an OOM.
  > [x] ~169 GB. The cache size far exceeds the remaining VRAM.
  > [ ] 35 GB. The memory is determined by the weights, so the error must be from fragmentation.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Static Batching Deadline</b> · <code>serving-batching-latency</code></summary>

- **Interviewer:** "You're designing a real-time transcription service using an LLM on a single H100 GPU. The service receives a steady stream of requests at 50 per second. Your hard P99 latency deadline is 500ms. After benchmarking, you find the GPU processing time for a batch is well-approximated by the formula: `T_process = 40ms + (10ms * batch_size)`. Your team proposes using a simple static batching strategy with a batch size of 8. Explain the components of worst-case latency for this system and calculate it. Does this design meet the deadline?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus only on the processing time gain from batching and forget to account for the 'wait time' penalty. For static batching, the first request in a batch must wait for the entire batch to assemble before processing begins. This wait time is often the largest component of end-to-end latency in latency-sensitive systems and cannot be ignored.

  **Realistic Solution:** The total latency for a request is the time it spends waiting for the batch to fill plus the time it takes the GPU to process the full batch.
1.  **Wait Time:** With an arrival rate of 50 req/sec, a new request arrives every `1 / 50 = 20ms`. In the worst case, a request is the very first to arrive for a new batch. It must wait for 7 more requests to arrive before the batch of 8 is full and can be sent for processing. The worst-case wait time is `7 * 20ms = 140ms`.
2.  **Processing Time:** Using the provided formula for a batch size of 8: `T_process = 40ms + (10ms * 8) = 120ms`.
3.  **Total Latency:** The worst-case latency is the sum of the wait and processing time: `140ms (wait) + 120ms (process) = 260ms`.
This total latency of 260ms is well within the 500ms P99 deadline, so the design is valid.

  > **Napkin Math:** Arrival Interval = 1 / 50 req/sec = 20 ms/req
Worst-Case Wait Time = (Batch Size - 1) * Arrival Interval = (8 - 1) * 20ms = 140ms
Processing Time = 40ms + (10ms * 8) = 120ms
Total Worst-Case Latency = Wait Time + Processing Time = 140ms + 120ms = 260ms
Check Deadline: 260ms < 500ms → Pass

  > **Key Equation:** T_{\text{worst_latency}} = (B-1) \times \frac{1}{\lambda} + T_{\text{process}}(B)

  > **Options:**
  > [ ] 120ms. It meets the deadline with significant headroom.
  > [ ] 540ms. It fails the deadline.
  > [x] 260ms. It meets the deadline.
  > [ ] 190ms. It meets the deadline.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The RAG Index Rollout</b> · <code>model-serving</code></summary>

- **Interviewer:** "You're an ML Systems Engineer managing a fleet-wide assistant for an autonomous vehicle company. The assistant uses a RAG model served from a Kubernetes cluster to answer passenger questions. The knowledge base is a 10 GB vector index file that is updated daily. Your serving cluster has 100 pods, and you perform a rolling update to deploy the new index. If all 100 pods attempt to download the 10 GB file simultaneously from your artifact storage, how long would the download phase take? Assume the pods fully saturate a shared 400 Gbps InfiniBand network link to the storage."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to confuse bits (b) and bytes (B), or to calculate the download time for only a single pod instead of the entire cluster. Engineers often see a 'Gbps' network figure and a 'GB' file size and forget to perform the 8x conversion from bytes to bits, leading to an answer that is 8x too fast. Another frequent error is to forget that all 100 pods are pulling data, so the total data transferred is 100 times the single file size.

  **Realistic Solution:** The correct approach is to first calculate the total amount of data that needs to be transferred for all pods. Then, convert this total data size from Gigabytes (GB) to Gigabits (Gb) to match the units of the network bandwidth. Finally, divide the total data in bits by the network bandwidth in bits per second to find the total time.

  > **Napkin Math:** 1. **Calculate Total Data:** 100 pods × 10 GB/pod = 1,000 GB
2. **Convert Data to Gigabits:** 1,000 GB × 8 bits/byte = 8,000 Gb
3. **Identify Network Bandwidth:** 400 Gbps
4. **Calculate Time:** Time = Total Data / Bandwidth = 8,000 Gb / 400 Gbps = 20 seconds.

  > **Key Equation:** $\text{Time} = \frac{\text{Total Data Size (bits)}}{\text{Bandwidth (bits/sec)}} = \frac{\text{Pods} \times \text{File Size (bytes)} \times 8}{\text{Bandwidth (bps)}}$

  > **Options:**
  > [ ] 2.5 seconds
  > [ ] 0.2 seconds
  > [x] 20 seconds
  > [ ] 0.025 seconds

  📖 **Deep Dive:** [Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Head-of-Line Blocking Problem</b> · <code>continuous-batching</code></summary>

- **Interviewer:** "Your team is serving a large language model. You observe that overall throughput is low and, more importantly, users with short, quick queries are experiencing high latency. You suspect this is because their requests get stuck in batches with users who are generating very long sequences. What is the primary latency-related problem that continuous batching (also known as in-flight batching) is designed to solve compared to traditional static batching?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The common mistake is to think that batching is only about maximizing throughput. Engineers often fail to recognize that with static batching, all requests in a batch are tied to the completion time of the single longest request. This creates a "head-of-line blocking" scenario where short, fast requests are unnecessarily delayed, leading to high average and tail latency, even if the GPU itself is busy.

  **Realistic Solution:** The correct answer is that continuous batching solves the "head-of-line blocking" problem. In static batching, the entire batch of requests is processed as a single unit; it is not finished until the longest sequence in the batch has generated its final token. This means a request that only needs one token can be stuck waiting for another request that needs a thousand tokens. Continuous batching decouples the requests, allowing the server to evict finished sequences from the batch and add new ones in a continuous, iterative process. This dramatically reduces the average latency for all requests and improves overall GPU utilization by not wasting compute on padded, completed sequences.

  > **Napkin Math:** Let's model two requests arriving in the same static batch:
- Request A: Needs to generate 100 tokens.
- Request B: Needs to generate just 1 token.
- Assume the per-token generation time (TPOT) is ~50ms.

**Static Batching:**
Request B is finished after the first step, but it is not evicted. It remains in the batch, occupying GPU memory and compute resources, until Request A is also finished.
- Latency for Request B = 100 tokens (longest sequence) * 50 ms/token = 5,000 ms (5 seconds).

**Continuous Batching:**
Request B is evicted from the batch after the first generation step is complete.
- Latency for Request B = 1 token * 50 ms/token = 50 ms.

By preventing head-of-line blocking, continuous batching provides a 100x latency reduction for the shorter request in this scenario.

  > **Key Equation:** L = W / \lambda

  > **Options:**
  > [ ] It primarily increases the maximum theoretical throughput (tokens/sec) of the GPU.
  > [x] It solves head-of-line blocking, where short requests are stuck waiting for the longest request in a batch to complete.
  > [ ] It reduces the VRAM required for the KV cache by using a different compression algorithm.
  > [ ] It strictly processes requests in a first-in, first-out (FIFO) order to ensure fairness.

  📖 **Deep Dive:** [Cloud Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The KV-Cache VRAM Budget</b> · <code>kv-cache-vram</code></summary>

- **Interviewer:** "You are deploying a Llama-2-70B model for inference on an H100 GPU. The model is running in FP16 precision. A user sends a request with a sequence length of 4,096 tokens. Given that the Llama-2-70B architecture has 80 layers and a hidden dimension of 8,192, calculate the memory required just for the KV-cache for this single user."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often forget one of the '2s' in the KV-cache formula. The cache stores both a Key tensor and a Value tensor for every token at every layer, so you must multiply by 2. Another common error is using the wrong data type size, for instance using 4 bytes for FP32 instead of 2 bytes for FP16.

  **Realistic Solution:** The KV-cache stores the Key and Value state for every token in the context window, for every layer of the model. For FP16 precision, each number requires 2 bytes. The total memory is the product of all these dimensions: sequence length × layers × hidden dimension × 2 (for K and V) × 2 (for FP16 bytes).

  > **Napkin Math:** Sequence Length: 4,096 tokens
Layers: 80
Hidden Dimension: 8,192
Precision: FP16 (2 bytes/value)
Tensors per token/layer: 2 (Key and Value)

Total Bytes = 4096 × 80 × 8192 × 2 (K/V) × 2 (bytes)
= 5,368,709,120 bytes

To convert bytes to gigabytes (GB), we divide by 10^9:
5,368,709,120 / 1,000,000,000 = 5.368 GB

This is approximately 5.4 GB.

  > **Key Equation:** $\text{KV-Cache Memory} = S \times L \times D_{\text{hidden}} \times 2 \times \text{sizeof(FP16)}$

  > **Options:**
  > [ ] ~2.7 GB
  > [ ] ~10.8 GB
  > [x] ~5.4 GB
  > [ ] ~2.6 MB

  📖 **Deep Dive:** [Inference and Serving](https://mlsysbook.ai/cloud/03_inference_and_serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Real-Time Batching Dilemma</b> · <code>real-time-batching</code></summary>

- **Interviewer:** "You are designing a real-time transcription service. Users send a continuous stream of audio chunks, and your service has a strict P99 latency SLA of 500ms to return the transcript for each chunk. Your model inference for a single request takes 150ms on the cloud GPU. To improve throughput, you introduce batching, which adds a fixed overhead of 50ms per batch for padding and dispatch. Explain the trade-off here and calculate the maximum batch size you can use without violating the SLA."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus solely on maximizing throughput (requests per second) by using large batches, forgetting that for synchronous, real-time services, the latency experienced by every user in the batch increases with batch size. They fail to account for the fact that all requests must wait for the entire batch to be processed.

  **Realistic Solution:** The core trade-off is between throughput and latency. Larger batches increase the number of requests processed per second by the GPU, but they also increase the end-to-end latency for every request in that batch because the user must wait for the batch to fill and for all items to be processed. The maximum batch size is limited by the latency SLA. We must calculate the total time it takes to process one full batch and ensure it's less than or equal to the 500ms deadline, as this represents the worst-case latency for a request in that batch.

  > **Napkin Math:** 1. **Define the total latency equation:** The total time (`T_total`) is the sum of the fixed batch overhead and the per-request inference time multiplied by the batch size (`B`).
2. **Set up the inequality:** `T_total = T_overhead + (T_inference * B) <= T_SLA`
3. **Plug in the numbers:** `50ms + (150ms * B) <= 500ms`
4. **Solve for B:** `150ms * B <= 450ms`
5. **Calculate the maximum batch size:** `B <= 450 / 150`, so `B <= 3`. The maximum allowed batch size is 3.

  > **Key Equation:** $\text{T}_{total} = \text{T}_{overhead} + (\text{T}_{inference} \times \text{BatchSize})$

  > **Options:**
  > [ ] 2
  > [ ] 4
  > [x] 3
  > [ ] 8

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Interconnect Latency Ladder</b> · <code>nvlink-vs-infiniband-latency</code></summary>

- **Interviewer:** "You're debugging a distributed training job and notice that nodes are waiting on each other. You suspect network latency. To build your intuition, your tech lead asks a quick question: 'Roughly how much slower is a cross-rack InfiniBand NDR transfer compared to a local, on-node NVLink 4.0 transfer between two GPUs?'"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate the latency penalty for going 'off-node'. They might mentally group all high-speed interconnects together, assuming their latencies are roughly comparable (e.g., within 2-3x of each other). They forget that crossing from an on-node electrical interconnect (NVLink) to an optical, cross-rack network (InfiniBand) involves significantly more protocol overhead and physical distance, pushing the latency up by an order of magnitude.

  **Realistic Solution:** A cross-rack InfiniBand transfer is approximately 10 times slower than an on-node NVLink transfer. NVLink 4.0 latency is around 500 nanoseconds, whereas a trip across the datacenter rack fabric via InfiniBand NDR is about 5,000 nanoseconds (5 microseconds). This 10x gap is a fundamental reality of datacenter topology: on-node communication is always significantly faster than node-to-node communication.

  > **Napkin Math:** We can pull the numbers directly from the ML Latency Hierarchy:
- NVLink 4.0 Transfer Latency: ~500 ns
- InfiniBand NDR Transfer Latency: ~5,000 ns

Ratio = InfiniBand Latency / NVLink Latency
Ratio = 5,000 ns / 500 ns = 10x slower.

Scaled to human time: An NVLink transfer is like waiting 8 minutes, while an InfiniBand transfer is like waiting 1.4 hours.

  > **Options:**
  > [ ] ~2x slower
  > [ ] ~100x slower
  > [x] ~10x slower
  > [ ] They have roughly the same latency

  📖 **Deep Dive:** [ML Systems Latency Hierarchy](https://mlsysbook.ai/NUMBERS.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The FP16 Memory Footprint</b> · <code>model-serving</code></summary>

- **Interviewer:** "You're deploying a 70-billion parameter LLM for a new service. Forgetting the KV cache and activation memory for a moment, roughly how much GPU memory is required just to load the model weights for inference using standard FP16 precision?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the memory requirements for different precisions. A common error is to assume 1 byte per parameter (like INT8), which would incorrectly halve the requirement, or 4 bytes per parameter (like FP32), which would double it. Another mistake is to forget the rule of thumb entirely and simply guess.

  **Realistic Solution:** The standard rule of thumb for inference memory at FP16 precision is 2 bytes per parameter. Therefore, a 70B parameter model requires approximately 140 GB of memory. This is a foundational calculation for capacity planning, as it tells you that the model cannot fit onto a single 80 GB H100 GPU and will require at least two, necessitating tensor parallelism.

  > **Napkin Math:** 70 Billion Parameters × 2 Bytes/Parameter (for FP16) = 140 Billion Bytes = 140 GB.

  > **Key Equation:** $\text{Inference Memory (GB)} \approx \frac{\text{Parameters} \times 2}{10^9}$

  > **Options:**
  > [ ] 70 GB
  > [x] 140 GB
  > [ ] 280 GB
  > [ ] 14 GB

  📖 **Deep Dive:** [Cloud / LLM Scaling Rules](https://github.com/mlsysbook/mlsysbook/blob/main/interviews/NUMBERS.md#4-scaling-rules-arithmetic--hardware-independent)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Batching Tipping Point</b> · <code>gpu-roofline-batching</code></summary>

- **Interviewer:** "An engineering team is optimizing a BERT-Large (340M parameters) model for FP16 inference on an H100 GPU. For a single input sequence, the forward pass requires approximately 25 GFLOPs and reads 750 MB of data (weights, KV cache, activations). The H100 has a peak FP16 performance of 989 TFLOPS and 3.35 TB/s of HBM3 memory bandwidth. First, calculate the Arithmetic Intensity (AI) for a single inference (batch size 1). Then, explain what happens to the workload's character (memory-bound vs. compute-bound) as the team increases the batch size."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often mistakenly assume that all memory access scales linearly with batch size. They forget that the model's weights (which form a large part of the memory footprint) are read only once per kernel launch, regardless of the batch size. This leads to the incorrect conclusion that Arithmetic Intensity remains constant and that the workload cannot become compute-bound through batching.

  **Realistic Solution:** The workload shifts from being memory-bound to compute-bound. At batch size 1, the Arithmetic Intensity is low, bottlenecked by the time it takes to read data from HBM. By increasing the batch size, the total FLOPs increase linearly, but the total memory read does not. The large weight matrix is read once, and only the per-input activation data scales with the batch. This reuse of weights drastically increases the ratio of compute to memory access (the AI), pushing the workload over the GPU's ridge point and into the compute-bound regime. By becoming compute-bound, the GPU can utilize its Tensor Cores more effectively, leading to higher achieved TOPS and better overall energy efficiency (TOPS/W).

  > **Napkin Math:** 1. **Calculate H100 Ridge Point:** The ridge point is the AI needed to be compute-bound.
   Ridge Point = Peak FLOPs / Memory Bandwidth = (989 * 10^12 Ops/sec) / (3.35 * 10^12 Bytes/sec) ≈ 295 Ops/Byte.

2. **Calculate AI for Batch Size 1:**
   AI = 25 GFLOPs / 750 MB = (25 * 10^9) / (750 * 10^6) ≈ 33.3 Ops/Byte.
   Since 33.3 < 295, the workload is heavily **memory-bound**.

3. **Calculate AI for Batch Size 64:**
   - First, separate weight memory from activation memory. Weights = 340M params * 2 bytes/param = 680 MB.
   - Activation memory per item = 750 MB (total) - 680 MB (weights) = 70 MB.
   - Total Compute (Batch 64) = 25 GFLOPs * 64 = 1,600 GFLOPs.
   - Total Memory (Batch 64) = 680 MB (weights) + (70 MB/item * 64 items) = 680 + 4480 = 5,160 MB.
   - New AI = 1,600 GFLOPs / 5,160 MB = (1600 * 10^9) / (5160 * 10^6) ≈ 310 Ops/Byte.

4. **Conclusion:** Since 310 > 295, the workload has crossed the ridge point and is now **compute-bound**.

  > **Key Equation:** $\text{Arithmetic Intensity} = \frac{\text{Total Operations (FLOPs)}}{\text{Total Memory Moved (Bytes)}}$

  > **Options:**
  > [ ] The workload remains memory-bound because Arithmetic Intensity is constant; both compute and memory scale linearly.
  > [ ] The workload is compute-bound at batch 1 and just becomes more compute-bound at batch 64.
  > [x] The workload shifts from memory-bound to compute-bound as the AI increases from ~33 to ~310 Ops/Byte.
  > [ ] The workload becomes compute-bound, but its AI decreases because the memory grows faster than the compute.

  📖 **Deep Dive:** [Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The KV-Cache VRAM Budget</b> · <code>kv-cache-vram</code></summary>

- **Interviewer:** "You are scoping the VRAM requirements for serving a Llama-2 70B model with a 128,000 token context window. For a single user request (batch size of 1), calculate the approximate VRAM needed *just for the FP16 KV-cache*. You can assume the model has 80 layers, a head dimension of 128, and uses Grouped-Query Attention (GQA) with 8 KV-heads."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often miscalculate KV-cache size by using the wrong number of heads (query heads vs. KV heads in GQA), forgetting to account for both Keys and Values (a 2x factor), or using the wrong precision (FP16 is 2 bytes, not 1). A more fundamental error is confusing the cache size, which scales with sequence length, with the model's parameter size, which is fixed.

  **Realistic Solution:** The correct calculation accounts for sequence length, the number of layers, the number of KV heads (not query heads), the dimension of each head, and the data type. For a GQA model, the smaller number of KV heads is the key to making long contexts manageable. The cache stores a Key and a Value vector for each token, at each layer, for each KV head, requiring 2 bytes per element for FP16 precision.

  > **Napkin Math:** 1.  **Identify variables:**
    *   Sequence Length (S): 128,000 tokens
    *   Number of Layers (L): 80
    *   Number of KV-Heads (H_kv): 8
    *   Head Dimension (D): 128
    *   Bytes per element: 2 (for FP16)
    *   K and V pair: 2

2.  **Calculate total elements:**
    *   `Elements = S × L × H_kv × D × 2` (for K and V)
    *   `Elements = 128,000 × 80 × 8 × 128 × 2 = 20,971,520,000`

3.  **Calculate total bytes:**
    *   `Bytes = Elements × 2` (for FP16)
    *   `Bytes = 20,971,520,000 × 2 = 41,943,040,000`

4.  **Convert to Gigabytes:**
    *   `GB = Bytes / (1024^3)`
    *   `GB = 41,943,040,000 / 1,073,741,824 ≈ 39.06 GB`

  > **Key Equation:** $$\text{VRAM}_{\text{cache}} = S \times L \times H_{kv} \times D \times 2 \times \text{sizeof(FP16)}$$

  > **Options:**
  > [ ] ~20 GB
  > [ ] ~140 GB
  > [x] ~39 GB
  > [ ] ~313 GB

  📖 **Deep Dive:** [Model Training](https://mlsysbook.ai/vol1/training.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Chatbot's Response Time</b> · <code>llm-inference-latency</code></summary>

- **Interviewer:** "You are operating a chatbot service backed by a 7B parameter LLM. The system has two main latency phases: a 'prefill' phase to process the user's prompt, which takes a fixed 150ms, and a 'decode' phase where it generates tokens one by one. The time per output token (TPOT) is 30ms. First, explain the difference between Time To First Token (TTFT) and Time Per Output Token (TPOT). Then, calculate the total time a user has to wait from sending their request to receiving the *20th* output token."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to only calculate the decoding time for all tokens (`20 * 30ms = 600ms`), completely ignoring the significant, fixed cost of the prefill phase. Another error is to correctly calculate TTFT but then fail to apply it properly for the full sequence, leading to an off-by-one error.

  **Realistic Solution:** TTFT (Time To First Token) is the total latency to generate the very first token, including both prompt processing (prefill) and the first decode step. TPOT (Time Per Output Token), also called inter-token latency, is the time to generate each subsequent token. The total time for the Nth token is the prefill time plus N decode steps.

Here, the TTFT = 150ms (prefill) + 30ms (decode) = 180ms.

The total time to receive the 20th token is the fixed prefill cost plus the cost of decoding all 20 tokens.

  > **Napkin Math:** Total Latency = Prefill Latency + (Number of Tokens × TPOT)

Total Latency = 150ms + (20 × 30ms)
Total Latency = 150ms + 600ms
Total Latency = 750ms

  > **Key Equation:** $$ T_{total}(n) = T_{prefill} + n \times T_{decode} $$

  > **Options:**
  > [ ] 600ms
  > [ ] 180ms
  > [x] 750ms
  > [ ] 780ms

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Interconnect Latency Ladder</b> · <code>nvlink-vs-infiniband-latency</code></summary>

- **Interviewer:** "Identify the approximate latency of a local GPU-to-GPU transfer using NVLink 4.0 versus a cross-rack, server-to-server transfer using InfiniBand NDR."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate the 'local vs. remote' latency gap, even for high-performance networks. They might treat all 'high-speed interconnects' as roughly equivalent, failing to appreciate that crossing a physical server boundary, even with RDMA and InfiniBand, imposes a significant and predictable latency penalty compared to on-node communication over a specialized bus like NVLink.

  **Realistic Solution:** An NVLink 4.0 transfer has a latency of approximately 500 ns, while a cross-rack InfiniBand NDR transfer is about 5,000 ns (5 µs). Therefore, the local NVLink transfer is roughly 10 times faster than the cross-rack InfiniBand transfer. This order-of-magnitude difference is fundamental to understanding network topology and optimizing distributed workloads.

  > **Napkin Math:** Using the 'Human Time Scale' where 1 ns = 1 second:
- **NVLink 4.0 Transfer @ ~500 ns:** Becomes ~8 minutes.
- **InfiniBand NDR Transfer @ ~5,000 ns:** Becomes ~1.4 hours.

The cross-rack transfer takes an order of magnitude longer, a critical factor when designing communication patterns for distributed training.

  > **Options:**
  > [ ] They have roughly the same latency (~500 ns).
  > [x] NVLink is ~10x faster (~500 ns vs ~5,000 ns).
  > [ ] InfiniBand is ~10x faster (~500 ns vs ~5,000 ns).
  > [ ] NVLink is ~100x faster (~50 ns vs ~5,000 ns).

  📖 **Deep Dive:** [Distributed Systems](https://mlsysbook.ai/cloud/02_distributed_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Iceberg of Inference Cost</b> · <code>model-serving-economics</code></summary>

- **Interviewer:** "You are deploying a new RAG-based customer support chatbot that uses a 7B parameter model for generation. The service must be available 24/7. From a Total Cost of Ownership (TCO) perspective, identify the dominant, long-term cost for this system."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus heavily on the one-time, upfront cost of model training, which is computationally intensive and memorable, while underestimating that the continuous, operational cost of running inference 24/7 quickly eclipses it. They see training as the 'big expense' and forget the system is 'always on'.

  **Realistic Solution:** The dominant long-term cost is running the GPU for inference 24/7. While training is a significant one-time cost and vector database storage is relatively cheap, the operational expense of keeping a GPU active for inference accumulates to become the largest part of the TCO over the system's lifecycle.

  > **Napkin Math:** A 7B parameter model requires approximately 14 GB of VRAM in FP16 (7B params × 2 bytes/param). This fits on a single cloud GPU. Assuming a conservative cost of ~$2/hour for that GPU, the annual inference cost is $2/hr × 24 hr/day × 365 days/yr = $17,520/year. This recurring operational expense quickly surpasses the one-time training cost (often in the tens of thousands for a model this size) and dwarfs the storage cost for embeddings (typically a few hundred dollars per year).

  > **Key Equation:** $\text{TCO} \approx \text{Cost}_{\text{train}} + N_{\text{years}} \times (\text{Cost}_{\text{inference_annual}} + \text{Cost}_{\text{storage_annual}})$

  > **Options:**
  > [ ] The one-time cost of training the 7B model.
  > [ ] The annual cost of storing the vector database embeddings.
  > [x] The annual cost of running the GPU for 24/7 inference.
  > [ ] The network bandwidth costs for handling user queries.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The KV-Cache Memory Bomb</b> · <code>kv-cache-memory</code></summary>

- **Interviewer:** "You are deploying a Llama 2 70B model for an application that requires a 32,768 token context window. The model will run in FP16 precision. Explain how you would calculate the memory required just for the KV-cache, and then calculate its approximate size."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often forget one of the '2's in the formula: either forgetting to account for both the Key (K) and Value (V) matrices, or using the wrong number of bytes for the given precision (e.g., 1 for FP16 instead of 2). This leads to an answer that's off by a factor of 2 or more.

  **Realistic Solution:** The KV-cache stores the Key and Value tensors for every token in the context window for every decoder layer. To calculate its size, you multiply these dimensions together.

For Llama 2 70B, we have 80 layers and a hidden dimension of 8192. At FP16 precision (2 bytes/element), the calculation for a 32,768 token sequence is straightforward and reveals that the cache is a massive 80 GB, often exceeding the memory required for the model weights themselves and filling an entire H100 GPU.

  > **Napkin Math:** 1. **Identify parameters:**
   - Layers: 80
   - Hidden Dimension: 8192
   - Sequence Length: 32,768 tokens
   - Precision: FP16 (2 bytes)
   - K and V Tensors: 2

2. **Apply the formula:**
   `Total Bytes = 2 (for K/V) * layers * hidden_dim * sequence_length * bytes_per_element`
   `Total Bytes = 2 * 80 * 8192 * 32768 * 2`

3. **Calculate in GB:**
   - `Total Bytes = 85,899,345,920 bytes`
   - `GB = Total Bytes / (1024^3)`
   - `GB = 85,899,345,920 / 1,073,741,824 = 80 GB`

  > **Key Equation:** $\text{KV Cache Size} = 2 \times (\text{num\_layers} \times \text{hidden\_dim}) \times \text{sequence\_length} \times \text{bytes\_per\_element}$

  > **Options:**
  > [ ] ~40 GB
  > [ ] ~160 GB
  > [x] ~80 GB
  > [ ] ~8 GB

  📖 **Deep Dive:** [Cloud Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Static Batching Latency Tax</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are designing an LLM serving system on a single H100 GPU. The team has decided to use a simple, time-based static batching strategy to improve throughput. The system collects all incoming requests for a fixed window of **50ms** and then dispatches them as a single batch.

The GPU has a fixed per-batch processing overhead of **20ms** (e.g., for kernel launches and memory setup) before token generation can begin. After this overhead, generating the first token for the entire batch takes **5ms**.

Explain the Time-To-First-Token (TTFT) for a user whose request arrives at the very beginning of a batching window. Calculate its value."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often conflate system throughput with user-perceived latency. They might only calculate the GPU processing time (25ms) and forget that for a user who arrives early, the largest component of their wait time is the time spent in the queue waiting for the batch window to close. The total latency is the sum of queueing time and processing time.

  **Realistic Solution:** The total Time-To-First-Token (TTFT) is the sum of three distinct periods: the time waiting for the batch window to fill, the fixed overhead to process the batch, and the time to compute the first token.

For a request arriving at the beginning of the window, it experiences the worst-case queuing delay. It must wait the full duration for the window to close before its request is even considered for processing. Therefore, the total latency is the batch window time + the batch overhead + the token generation time.

  > **Napkin Math:** # 1. Time spent waiting for the batch window to close (worst case)
Wait Time = 50ms

# 2. Fixed overhead to launch the batch on the GPU
Batch Overhead = 20ms

# 3. Time to generate the first token for the batch
Generation Time = 5ms

# 4. Total TTFT is the sum of these components
Total TTFT = Wait Time + Batch Overhead + Generation Time
Total TTFT = 50ms + 20ms + 5ms = 75ms

  > **Key Equation:** $\text{TTFT} = T_{\text{wait}} + T_{\text{overhead}} + T_{\text{generate}}$

  > **Options:**
  > [ ] 25ms
  > [ ] 55ms
  > [x] 75ms
  > [ ] 50ms

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The RAG Retrieval Bottleneck</b> · <code>rag-latency-bottleneck</code></summary>

- **Interviewer:** "You are debugging a new RAG-powered chatbot in a cloud environment. The application retrieves context documents from a vector database running on a server with NVMe SSDs in the same datacenter rack before passing the context to an LLM. A user complains about high response latency. Identify the most likely source of this latency from the options below."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers, especially those new to systems design, often assume the LLM inference itself is the bottleneck due to its massive parameter count. They underestimate the profound latency cost of I/O operations, even with fast storage like NVMe SSDs, compared to the speed of on-chip and HBM memory access.

  **Realistic Solution:** The retrieval step involving the NVMe SSD read is the dominant source of latency. The ML Latency Hierarchy shows that while HBM access for the model is in the hundreds of nanoseconds, an NVMe read is in the hundreds of *microseconds*—a nearly 1000x difference that overshadows the rest of the pipeline.

  > **Napkin Math:** Let's compare the numbers from the hierarchy. An HBM3 memory access for the LLM is ~300 ns. A cross-rack InfiniBand hop to the database server is ~5,000 ns. A single NVMe SSD read is ~100,000 ns. Therefore, the SSD read is **~20x slower** than the network transfer and **~333x slower** than a memory access for the model's weights. The I/O for retrieval is the clear bottleneck.

  > **Key Equation:** $T_{total} \approx T_{retrieval} + T_{inference}$

  > **Options:**
  > [x] Reading documents from the NVMe SSD vector database
  > [ ] LLM forward pass memory access to HBM
  > [ ] Network transfer to the database server via InfiniBand
  > [ ] L2 cache misses on the GPU during the forward pass

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Batching Window Dilemma</b> · <code>llm-serving-queueing-theory</code></summary>

- **Interviewer:** "You are managing an LLM inference service that uses continuous batching on H100 GPUs. Your team has a strict P99 Time-To-First-Token (TTFT) SLO of 200ms. Based on production metrics, you know that the round-trip network latency adds a fixed 30ms, and the GPU inference time to generate the first token for any batch is a constant 50ms. To maximize throughput, you want to wait as long as possible to batch incoming requests. Explain the tradeoff here and calculate the absolute maximum time a request can wait in the queue before being processed, without violating the SLO."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to focus only on one part of the latency equation, such as the inference time or the queue time, while forgetting to account for other fixed costs like network latency. This leads to an overestimation of the available budget for batching, causing SLO violations in production. The total latency experienced by the user is the sum of all parts: `Network + Queue + Compute`.

  **Realistic Solution:** The core tradeoff is between throughput and latency. Longer batching windows allow for larger batches, which increases GPU utilization and overall throughput. However, this added queue time directly increases the latency perceived by the end-user. To solve this, we must sum all latency components and ensure they are less than or equal to the SLO.

`Total Latency = Network Latency + Queue Time + Inference Time`

The maximum allowed queue time is the SLO minus the fixed costs of network and inference.

  > **Napkin Math:** 1. **Total Latency Budget (SLO):** 200 ms
2. **Subtract Fixed Network Latency:** 200 ms - 30 ms = 170 ms
3. **Subtract Fixed Inference Time:** 170 ms - 50 ms = 120 ms

**Result:** The maximum time a request can wait in the batching queue is 120 ms.

  > **Key Equation:** $\text{Latency}_{\text{Total}} = \text{Latency}_{\text{Network}} + \text{Time}_{\text{Queue}} + \text{Time}_{\text{Inference}}$

  > **Options:**
  > [ ] 150 ms
  > [ ] 170 ms
  > [x] 120 ms
  > [ ] 200 ms

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Interconnect Latency Ladder</b> · <code>interconnect-latency</code></summary>

- **Interviewer:** "When moving data between GPUs in a large training cluster, what is the correct rank-ordering of common interconnects from *lowest* latency to *highest* latency for a small data transfer?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Confusing bandwidth with latency, or not knowing the physical hierarchy. Many engineers know InfiniBand is 'fast' (high bandwidth), but they forget the significant latency overhead for cross-server communication compared to on-board links like NVLink. Another common error is mixing up the roles of PCIe and NVLink.

  **Realistic Solution:** The correct order is NVLink < PCIe < InfiniBand. NVLink is a specialized, extremely low-latency link for GPUs on the same server board (~500 ns). PCIe is a general-purpose bus connecting the GPU to the motherboard, with higher latency (~1,000 ns). InfiniBand is a high-speed network for connecting different servers, but it has the highest latency of the three for a single transaction due to the physics of crossing racks (~5,000 ns).

  > **Napkin Math:** Using the '1ns = 1 second' human scale from the playbook:
- An NVLink 4.0 transfer takes ~500 ns, which is like **8 minutes**.
- A PCIe Gen5 transfer takes ~1,000 ns, which is like **16 minutes**.
- An InfiniBand NDR transfer takes ~5,000 ns, which is like **1.4 hours**.

This clearly shows the ~2x step from NVLink to PCIe, and the ~5x step from PCIe to InfiniBand.

  > **Options:**
  > [ ] InfiniBand < NVLink < PCIe
  > [ ] PCIe < NVLink < InfiniBand
  > [x] NVLink < PCIe < InfiniBand
  > [ ] All are roughly the same (~5,000 ns)

  📖 **Deep Dive:** [Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The RAG Latency Trap</b> · <code>rag-latency-serving</code></summary>

- **Interviewer:** "You're designing a Retrieval-Augmented Generation (RAG) system for a real-time chatbot. To ground the LLM's response, you need to fetch a document chunk. Roughly how much slower is retrieving that chunk from a local NVMe SSD compared to fetching it from an in-memory cache stored in HBM?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate the storage hierarchy gap, thinking SSDs are 'fast' and maybe only 10-50x slower than HBM. They fail to internalize that for a GPU, any I/O operation that leaves the memory subsystem is an eternity, making disk-based lookups totally infeasible for low-latency serving without aggressive caching.

  **Realistic Solution:** An NVMe SSD read is approximately 333 times slower than an HBM access. This huge performance gap is fundamental to ML systems design. For a P99 latency target of 100ms, you cannot afford to go to disk. The RAG context must be served from an in-memory system (like Redis or directly in HBM/DRAM) to meet the budget.

  > **Napkin Math:** This is a direct lookup from the ML Latency Hierarchy table:
- NVMe SSD Read: ~100,000 ns
- HBM3 Memory Access: ~300 ns
- Ratio: 100,000 ns / 300 ns ≈ 333×

On a human time scale, if an HBM access took 5 minutes, the SSD read would take over 27 hours.

  > **Key Equation:** $\text{Latency Ratio} = \frac{\text{Latency}_{\text{NVMe SSD}}}{\text{Latency}_{\text{HBM}}}$

  > **Options:**
  > [ ] ~17x slower
  > [x] ~333x slower
  > [ ] ~100,000x slower
  > [ ] ~5,000x slower

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Rollout Memory Budget</b> · <code>model-serving-rollout</code></summary>

- **Interviewer:** "You are a Staff ML Systems Engineer planning a service update. You need to roll out a new 13B parameter LLM to replace an existing 7B parameter model used in a RAG-based customer support chatbot. The entire fleet of 500 serving instances will be updated. Both models are served in FP16 precision. Calculate the *total additional* HBM memory capacity required across the entire fleet to accommodate the larger model."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to calculate the total memory required for the new fleet (13B params × 2 bytes/param × 500 instances = 13 TB) instead of the *additional* or *delta* memory needed. This leads to requesting far more capacity than necessary for the upgrade. Another error is to confuse model parameters with memory footprint, forgetting to multiply by the bytes-per-parameter for the given precision.

  **Realistic Solution:** The correct approach is to calculate the memory increase per instance caused by the model upgrade, and then multiply that delta by the total number of instances in the serving fleet. This gives the precise additional capacity needed for the rollout.

  > **Napkin Math:** 1. **Calculate parameter increase:** 13B params - 7B params = 6B additional parameters.
2. **Convert increase to memory per instance (FP16):** 6B params × 2 bytes/param = 12 GB of additional HBM per instance.
3. **Calculate total additional memory for the fleet:** 12 GB/instance × 500 instances = 6,000 GB.
4. **Convert to Terabytes:** 6,000 GB = 6 TB.

  > **Key Equation:** $\text{Total Additional Memory} = (P_{new} - P_{old}) \times \text{Bytes per Parameter} \times N_{instances}$

  > **Options:**
  > [ ] 3 TB
  > [ ] 13 TB
  > [x] 6 TB
  > [ ] 9.5 TB

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Static Batching Waiting Game</b> · <code>inference-queueing-theory</code></summary>

- **Interviewer:** "You are managing an LLM inference service on a single H100 GPU. New requests arrive at a steady rate, one every 150ms. To optimize throughput, you've configured a static batching policy with a batch size of 8. Once a batch of 8 is full, it takes the H100 800ms to process it. Calculate the average time a request spends waiting in the queue before its batch even begins processing."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the *average* wait time with the *maximum* wait time (the time the very first request in the batch has to wait). Another common error is to focus on the GPU processing time (800ms) instead of the queueing delay caused by waiting for the batch to fill.

  **Realistic Solution:** The correct answer is 525ms. The core insight is that the waiting time is determined by how long it takes for a batch to assemble, not the GPU processing time. The first request waits for 7 more to arrive, the second for 6, and so on, until the last request waits for 0. The average of these waiting times is the answer.

This demonstrates a fundamental trade-off in batching systems: increasing batch size improves GPU utilization and throughput, but at the cost of increased latency for individual requests due to longer queue wait times. This is why more advanced techniques like continuous batching are used to mitigate this problem.

  > **Napkin Math:** 1. **Arrival Rate:** A new request arrives every `T_arrival = 150ms`.
2. **Batch Size:** `N = 8` requests.
3. **Waiting Times:**
    - Request 1 waits for 7 more requests: `7 * 150ms = 1050ms`
    - Request 2 waits for 6 more: `6 * 150ms = 900ms`
    - ...
    - Request 8 waits for 0 more: `0 * 150ms = 0ms`
4.  **Sum of wait times:** This is an arithmetic series: `150ms * (7 + 6 + 5 + 4 + 3 + 2 + 1 + 0) = 150ms * 28 = 4200ms`.
5.  **Average Wait Time:** `Total Wait / N = 4200ms / 8 = 525ms`.

  > **Key Equation:** $\text{W}_{\text{avg}} = \frac{\text{T}_{\text{arrival}} \times (\text{N} - 1)}{2}$

  > **Options:**
  > [ ] 100ms
  > [ ] 1050ms
  > [x] 525ms
  > [ ] 800ms

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The RAG Retrieval Tax</b> · <code>rag-latency-guardrails</code></summary>

- **Interviewer:** "A user query to a production RAG system must first retrieve context from a vector database before hitting the LLM. This retrieval happens over the data center network. What is the approximate latency of a single cross-rack InfiniBand network hop to the vector database server?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Confusing latency scales. Engineers often mix up on-chip latency (nanoseconds), data center network latency (microseconds), and wide-area network latency (milliseconds). A cross-rack hop is incredibly fast, but it's still orders of magnitude slower than accessing local memory.

  **Realistic Solution:** The latency for a cross-rack InfiniBand NDR hop is approximately 5,000 nanoseconds (5 microseconds). This is a fundamental number in distributed systems design, representing the irreducible 'tax' for any remote procedure call within the data center, even before the remote server does any work.

  > **Napkin Math:** From the 'ML Latency Hierarchy' table:
- HBM3 Memory Access: ~300 ns
- **InfiniBand NDR (cross-rack): ~5,000 ns (5 µs)**
- NVMe SSD Read: ~100,000 ns (100 µs)
- Cross-country Fiber: ~40,000,000 ns (40 ms)

The 5 µs cost is for the network transit alone. To put this in human terms: if an L1 cache read (1 ns) was one second, this network hop would take about 1.4 hours.

  > **Options:**
  > [ ] ~300 ns
  > [x] ~5,000 ns (5 µs)
  > [ ] ~100,000 ns (100 µs)
  > [ ] ~40,000,000 ns (40 ms)

  📖 **Deep Dive:** [ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Concurrent User Limit</b> · <code>llm-serving-throughput</code></summary>

- **Interviewer:** "Your team is deploying a new 13B parameter LLM for a real-time coding assistant on a single NVIDIA H100 GPU. The service level objective (SLO) requires an average generation speed of at least 64 tokens per second for each user to feel interactive. Assuming the serving stack is well-optimized with continuous batching and achieves 50% of the H100's peak FP16 compute throughput, explain how you would calculate the maximum number of concurrent users the server can support."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Candidates often calculate throughput using the GPU's theoretical peak TFLOPS, ignoring realistic utilization (MFU - Model FLOPs Utilization). Peak numbers are unattainable in practice due to memory bandwidth bottlenecks and other system overheads. A second common mistake is to ignore the impact of batching; without an efficient strategy like continuous batching, the GPU would be severely underutilized and latency-bound, supporting far fewer users than this calculation implies.

  **Realistic Solution:** The correct approach is to first determine the GPU's effective throughput in FLOPs by applying the utilization factor to its peak performance. Second, calculate the FLOPs required to generate a single token for the 13B model. Dividing the GPU's effective throughput by the per-token cost gives the total number of tokens the system can generate per second. Finally, dividing this total token capacity by the per-user requirement yields the maximum number of concurrent users.

  > **Napkin Math:** 1. **H100 Effective Compute:** An H100 provides 989 TFLOPS (FP16). At a realistic 50% utilization, the effective throughput is `989 TFLOPS * 0.5 = 494.5` TFLOPS.
2. **FLOPs per Token:** For a 13B model, each generated token requires approximately `2 * 13B = 26` GFLOPs, based on the `2 * P` rule of thumb for inference compute.
3. **Total Tokens per Second:** Divide the effective compute by the per-token cost: `(494.5 * 10^12 FLOPs/sec) / (26 * 10^9 FLOPs/token) ≈ 19,019` tokens/sec.
4. **Max Concurrent Users:** Divide the total token capacity by the per-user requirement: `19,019 tokens/sec / 64 tokens/sec/user ≈ 297` users.

  > **Key Equation:** $\text{Max Users} = \frac{\text{GPU FLOPS} \times \eta_{\text{util}}}{\left( 2 \times \text{Params} \right) \times \text{Tokens/sec/user}}$

  > **Options:**
  > [ ] About 60 users
  > [x] About 300 users
  > [ ] About 600 users
  > [ ] About 19,000 users

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The On-Node Interconnect Tax: NVLink vs. PCIe</b> · <code>nvlink-vs-pcie-latency</code></summary>

- **Interviewer:** "You're profiling a model training job on a multi-GPU server and notice significant data transfer overhead. To build a mental model of the costs, you compare the latencies of the main on-node interconnects. Roughly how much slower is a data transfer over a standard PCIe Gen5 bus compared to a direct GPU-to-GPU transfer using NVLink 4.0?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse latency with bandwidth. While NVLink's *bandwidth* can be an order of magnitude higher than PCIe, its *latency* advantage for a single transfer is much smaller. It's also common to lump all 'fast' on-board interconnects together, forgetting that the physical path (direct GPU-to-GPU vs. via CPU/motherboard) has a real, measurable latency cost.

  **Realistic Solution:** A PCIe Gen5 transfer is approximately 2x slower than an NVLink 4.0 transfer. The latency for a direct GPU-to-GPU transfer over NVLink is ~500 ns, while traversing the PCIe bus to another device takes around 1,000 ns (1 µs). While both are extremely fast, this 2x difference is critical for high-frequency communication patterns like those in model-parallel or data-parallel training.

  > **Napkin Math:** Using the 'human time' analogy where 1 ns is 1 second:
- An NVLink 4.0 transfer takes ~500 seconds, or about 8 minutes.
- A PCIe Gen5 transfer takes ~1,000 seconds, or about 16 minutes.
The difference is noticeable and significant at scale, but it is not an order of magnitude.

  > **Options:**
  > [ ] About the same latency
  > [ ] ~10x slower
  > [x] ~2x slower
  > [ ] ~18x slower

  📖 **Deep Dive:** [Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The FP16 Inference Memory Rule</b> · <code>model-serving-footprint</code></summary>

- **Interviewer:** "You're scoping the infrastructure for a new RAG-based chatbot that uses a Llama-3-8B model. As a first-pass capacity check before worrying about container orchestration or KV cache, what is the absolute minimum memory (RAM or HBM) you must provision on a server just to load the model's weights in its standard FP16 precision?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to confuse the memory requirements for inference with those for training. Training requires storing gradients and optimizer states (like Adam), which dramatically increases the footprint to ~16 bytes per parameter. Another frequent error is to assume 1 byte per parameter, which is only correct for INT8-quantized models, not the standard FP16.

  **Realistic Solution:** The standard rule of thumb for loading a model's weights for inference using 16-bit floating-point precision (FP16 or BFloat16) is 2 bytes per parameter. Therefore, for an 8 billion parameter model, the calculation is straightforward: 8 billion parameters multiplied by 2 bytes per parameter equals 16 billion bytes, or 16 GB. This is the non-negotiable floor for the hardware before considering KV cache, activations, or OS overhead.

  > **Napkin Math:** 8 Billion Parameters × 2 bytes/parameter = 16 GB

  > **Key Equation:** $\text{Inference Memory (FP16)} = \text{Parameters} \times 2 \text{ bytes}$

  > **Options:**
  > [ ] 8 GB
  > [ ] 128 GB
  > [x] 16 GB
  > [ ] 2 GB

  📖 **Deep Dive:** [NUMBERS.md](https://github.com/mlsysbook/mlsysbook/blob/main/playbook/NUMBERS.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Deadline</b> · <code>continuous-batching-latency</code></summary>

- **Interviewer:** "You are operating an LLM inference service using an engine that implements continuous batching on H100 GPUs. Your key customer has a strict P99 Time-To-First-Token (TTFT) service level objective (SLO) of 150ms. You've profiled the system and found that the prefill computation for a new request takes 40ms, and the iteration time to generate one token for a full batch (the TPOT) is 10ms. A new request arrives at the worst possible moment (just after an iteration has started). Calculate its TTFT and explain if it meets the SLO."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the components of latency. A common mistake is to believe TTFT is solely the prefill time, ignoring the queuing delay inherent in any batching system. Another mistake is to model the delay as waiting for an entire sequence to finish (as in static batching), rather than just one iteration, which is the key advantage of continuous batching.

  **Realistic Solution:** The total Time-To-First-Token (TTFT) is the sum of the queuing delay and the prefill computation time. In a continuous batching system, the worst-case queuing delay occurs when a request arrives just after a batch processing iteration has begun. The request must wait for that single iteration to complete. Therefore, the worst-case TTFT is the iteration time plus the prefill time. In this case, 10ms + 40ms = 50ms. This easily meets the customer's 150ms SLO.

  > **Napkin Math:** `Worst-Case TTFT = Wait Time + Prefill Time`
`Wait Time (worst case) = Iteration Time (TPOT)`
`Worst-Case TTFT = 10ms (TPOT) + 40ms (Prefill) = 50ms`
`50ms < 150ms (SLO)`

  > **Key Equation:** $\text{TTFT} = T_{\text{queue}} + T_{\text{prefill}}$

  > **Options:**
  > [ ] 40ms. The TTFT is determined solely by the prefill computation time.
  > [ ] 10ms. The TTFT is equivalent to the Time Per Output Token (TPOT).
  > [x] 50ms. The TTFT is the prefill time plus the worst-case wait for the next iteration cycle.
  > [ ] Up to several seconds. The request must wait for the longest sequence in the current batch to complete.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Basic Inference Memory Footprint</b> · <code>model-serving-memory</code></summary>

- **Interviewer:** "You're scoping the GPU requirements for a new RAG-based chatbot service. The core of the service uses a 7-billion parameter Llama model for generation. Before even considering the KV cache or the retrieval index, what is the minimum memory required just to load the model's weights for inference using FP16 precision?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the memory requirements for *training* with those for *inference*. Training memory is dominated by optimizer states (like Adam), which can require 16 bytes or more per parameter. Another common mistake is to assume a more aggressive quantization (like INT8) by default, which would halve the requirement.

  **Realistic Solution:** The correct answer is approximately 14 GB. A 7-billion parameter model has 7e9 parameters. In FP16 (half-precision floating point), each parameter requires 2 bytes of storage. Therefore, the total memory for the weights is 7 billion × 2 bytes, which is 14 billion bytes or 14 GB.

  > **Napkin Math:** 7B params × 2 bytes/param (for FP16) = 14 GB

  > **Key Equation:** $\text{Inference Memory (FP16)} = \text{Parameters} \times 2 \text{ bytes}$

  > **Options:**
  > [ ] ~7 GB
  > [ ] ~112 GB
  > [x] ~14 GB
  > [ ] ~2 GB

  📖 **Deep Dive:** [Cloud/LLM Scaling Rules](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/fuse.py)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The RAG Retrieval Tax</b> · <code>rag-retrieval-latency</code></summary>

- **Interviewer:** "Your team is deploying a new Retrieval-Augmented Generation (RAG) model. During the rollout, you notice P99 latency is much higher than expected. A profiler shows that the 'retrieval' step, which fetches context from an NVMe SSD-backed vector database, is the bottleneck. To build intuition for the performance review, roughly how much slower is a single read from that NVMe SSD compared to the model accessing its own parameters in HBM3 memory?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the SSD-vs-HBM gap with the much larger SSD-vs-L1 cache gap (~100,000x) or vastly underestimate it, thinking modern SSDs are only about 10-50x slower than GPU memory. This misses the fundamental latency cost of going from on-package HBM to an off-chip peripheral over PCIe.

  **Realistic Solution:** An NVMe SSD read is approximately 333x slower than an HBM3 access. A typical HBM3 access is around 300 ns, while a read from an NVMe SSD is about 100,000 ns (100 µs). This three-orders-of-magnitude gap is a foundational constraint in designing performant RAG systems, necessitating aggressive caching and optimization of the retrieval step.

  > **Napkin Math:** Latency Ratio = (NVMe SSD Read Latency) / (HBM3 Memory Access Latency) = 100,000 ns / 300 ns ≈ 333x. If an HBM access were scaled to 5 minutes, waiting for the SSD read would be like waiting 1.1 days.

  > **Key Equation:** $\text{Latency Ratio} = \frac{\text{Latency}_{\text{Storage}}}{\text{Latency}_{\text{GPU Memory}}}$

  > **Options:**
  > [ ] ~30x slower
  > [x] ~300x slower
  > [ ] ~3,000x slower
  > [ ] ~30,000x slower

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The RAG Update Bottleneck</b> · <code>model-serving-rollout</code></summary>

- **Interviewer:** "You are an engineer at an automotive company responsible for a customer support chatbot. The bot uses a RAG model with its knowledge base stored in a 200 GB vector index. To add information about a new car model, you must roll out an updated index to the serving cluster. Assuming you have to ship the entire 200 GB file, how long would it take to transfer the index to a single serving pod over the datacenter's 400 Gbps InfiniBand network?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often overestimate datacenter transfer times because they mentally anchor on their own home internet speeds (Mbps) or older datacenter specs (e.g., 10 GbE). They fail to internalize the massive bandwidth of modern interconnects like InfiniBand, leading them to incorrectly identify network transfer as the primary bottleneck in an update process. The real bottlenecks have shifted to index loading, memory mapping, and cache warming.

  **Realistic Solution:** The transfer time is surprisingly short. A 400 Gbps InfiniBand network provides approximately 50 GB/s of bandwidth. Transferring a 200 GB index takes only 4 seconds. This calculation reveals that for most RAG updates, the network transfer itself is not the dominant latency. The real engineering challenge is what happens *after* the transfer: safely loading the new index into the serving process without dropping requests, handling the memory pressure, and warming up the index.

  > **Napkin Math:** 1. **Convert Bandwidth Units:** The network speed is 400 Gigabits per second (Gbps). To work with file sizes in Gigabytes (GB), convert bits to bytes. There are 8 bits in a byte. So, `400 Gbps / 8 = 50 GB/s`.
2. **Identify Data Size:** The index file is 200 GB.
3. **Calculate Time:** Time = Total Data / Bandwidth = `200 GB / 50 GB/s` = 4 seconds.
4. **Conclusion:** The network transfer is trivial. The system's ability to hot-swap a 200 GB file in memory is the real engineering problem.

  > **Key Equation:** $\text{Time} = \frac{\text{Data Size (GB)}}{\text{Bandwidth (GB/s)}}$

  > **Options:**
  > [ ] 32 seconds
  > [ ] 2.7 minutes
  > [x] 4 seconds
  > [ ] 0.5 seconds

  📖 **Deep Dive:** [Cloud: The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Real-Time Batching Deadline</b> · <code>real-time-inference-batching</code></summary>

- **Interviewer:** "You're architecting an LLM inference service for a 7-billion parameter model running on a single NVIDIA H100. The service has a strict P99 latency SLO of 100ms to generate 50 tokens per request. Assuming the H100 achieves 50% of its peak FP16 throughput and ignoring all other sources of latency (e.g., network, queueing), explain how you would calculate the maximum batch size the system can handle in a single, continuous generation without violating the SLO."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the compute formulas for training (`~6PD`) and inference (`~2PT`), leading to a significant underestimation of capacity. A second common error is forgetting to apply a realistic utilization factor; using peak theoretical FLOPS will vastly overestimate the batch size the system can actually handle. A third mistake is to incorrectly analyze the bottleneck, calculating a memory-bound limit when the question is explicitly about a compute-bound deadline.

  **Realistic Solution:** The correct approach is to determine the total computational budget within the 100ms latency window and divide that by the computational cost of a single request. This tells you how many requests can be processed in parallel (i.e., the batch size) while meeting the deadline. The problem is compute-bound by definition due to the latency SLO.

  > **Napkin Math:** 1. **Calculate Effective H100 Throughput:** The H100's peak FP16 throughput is 989 TFLOPS. For napkin math, we can round this to 1,000 TFLOPS/s. At 50% utilization, the effective throughput is `1,000 TFLOPS/s * 0.5 = 500 TFLOPS/s`.

2. **Calculate FLOPs per Request:** The rule of thumb for inference compute is `2 * Parameters * Tokens`. For a 7B model generating 50 tokens, this is `2 * 7e9 * 50 = 700e9` FLOPs, or `700 GFLOPs`.

3. **Calculate FLOPs Available in Budget:** The latency budget is 100ms, or 0.1 seconds. The total compute available in this window is `500 TFLOPS/s * 0.1s = 50 TFLOPs`.

4. **Calculate Max Batch Size:** Divide the available FLOPs budget by the FLOPs required per request: `Max Batch Size = 50 TFLOPs / 700 GFLOPs = 50,000 GFLOPs / 700 GFLOPs ≈ 71.4`.

5. **Conclusion:** Since the batch size must be an integer, the maximum possible batch size is 71.

  > **Key Equation:** $\text{Max Batch Size} = \lfloor \frac{(\text{Peak FLOPs} \times \eta_{\text{util}}) \times T_{\text{budget}}}{\text{2} \times P \times D_{\text{tokens}}} \rfloor$

  > **Options:**
  > [ ] 142
  > [ ] 5
  > [x] 71
  > [ ] 23

  📖 **Deep Dive:** [Cloud Serving Stacks](https://mlsysbook.ai/vol2/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Blue-Green Memory Tax</b> · <code>model-serving-rollout</code></summary>

- **Interviewer:** "Your team is updating the LLM for an in-car voice assistant. You are moving from a 7B parameter model to a 13B parameter model. Your Kubernetes cluster uses a blue-green deployment strategy, meaning for a short time during the rollout, both the old (blue) and new (green) models must be loaded in memory on the same nodes before traffic is switched. Explain how much total HBM memory will be consumed *just for the model weights* on a node running both models during this transition period, assuming the models are loaded in FP16 precision."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often only plan for the final state (the new, larger model), forgetting that during a blue-green transition, resources must accommodate both versions simultaneously. This leads to insufficient memory provisioning, causing nodes to crash during the rollout. Another common error is using the wrong number of bytes for the given precision (e.g., calculating for INT8 instead of FP16).

  **Realistic Solution:** The correct approach is to calculate the memory required for each model individually and then sum them up. In a blue-green deployment, both models co-exist in the cluster's memory to allow for instant traffic switching and rollback. An FP16 parameter requires 2 bytes of storage.

- The old 7B model requires 7 billion × 2 bytes = 14 GB.
- The new 13B model requires 13 billion × 2 bytes = 26 GB.

During the transition, a node must hold both, so the total memory required is 14 GB + 26 GB = 40 GB.

  > **Napkin Math:** 1. **Old Model (Blue) Memory:**
   7 Billion parameters × 2 bytes/parameter (FP16) = 14 GB

2. **New Model (Green) Memory:**
   13 Billion parameters × 2 bytes/parameter (FP16) = 26 GB

3. **Total Transitional Memory:**
   14 GB (Blue) + 26 GB (Green) = 40 GB

  > **Key Equation:** $\text{Total Memory} = (P_{\text{old}} \times \frac{\text{bytes}}{\text{param}}) + (P_{\text{new}} \times \frac{\text{bytes}}{\text{param}})$

  > **Options:**
  > [ ] 26 GB
  > [ ] 20 GB
  > [x] 40 GB
  > [ ] 80 GB

  📖 **Deep Dive:** [ML Operations](https://mlsysbook.ai/vol1/ops.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Batching Window Trap</b> · <code>serving-latency-tradeoffs</code></summary>

- **Interviewer:** "You are designing the serving infrastructure for a 7B parameter LLM on a single H100 GPU. The product team has a strict requirement: Time To First Token (TTFT) must be under 200ms to feel 'real-time'. You implement a simple timed (or 'slotted') batching strategy: every 150ms, the server groups any waiting requests into a batch and sends it to the GPU. Your profiling shows that the pre-fill stage for a typical batch of this size takes 80ms on the H100.

Explain why this serving strategy is at risk of violating the 200ms TTFT SLA. Calculate the theoretical worst-case TTFT a user might experience, ignoring network latency."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to only consider the GPU compute time (80ms) and conclude that the system is well within the 200ms deadline. This completely ignores the latency introduced by the batching schedule itself. Engineers often forget that total latency is not just processing time, but also the time spent waiting in a queue ('head-of-line blocking').

  **Realistic Solution:** The total latency a user experiences is the sum of the time they spend waiting in the queue and the actual GPU processing time for their request.

In a timed batching system, the worst-case queuing delay occurs when a user's request arrives *just after* a batch has been dispatched. They must then wait for the entire duration of the next batching window to elapse before their request is even sent to the GPU.

Therefore, the worst-case TTFT is the maximum queuing delay plus the pre-fill processing time. This is 150ms + 80ms = 230ms, which violates the 200ms SLA. This is the fundamental problem that more advanced strategies like continuous batching are designed to solve.

  > **Napkin Math:** 1. **Identify Maximum Queuing Delay:** The longest a request can wait before being processed is the full duration of the batching window.
   - `T_queue_max = 150 ms`

2. **Identify Processing Time:** The time for the GPU to compute the pre-fill for the batch is given.
   - `T_prefill = 80 ms`

3. **Calculate Worst-Case TTFT:** The total time is the sum of the waiting time and the processing time.
   - `TTFT_worst_case = T_queue_max + T_prefill`
   - `TTFT_worst_case = 150 ms + 80 ms = 230 ms`

4. **Compare to SLA:**
   - `230 ms > 200 ms` (SLA Violated)

  > **Key Equation:** $\text{TTFT}_{\text{worst}} = T_{\text{batch\_window}} + T_{\text{prefill}}$

  > **Options:**
  > [ ] 80 ms. The TTFT is simply the pre-fill computation time.
  > [ ] 150 ms. The TTFT is determined by the batching window, as it's the longest delay.
  > [x] 230 ms. The worst case is the full batch window delay plus the pre-fill time.
  > [ ] 70 ms. The available time is the batch window minus the compute time.

  📖 **Deep Dive:** [The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Cold Start Penalty</b> · <code>model-serving-latency</code></summary>

- **Interviewer:** "During a blue-green deployment of a new foundational model, a request is routed to a new server instance. The model weights are not yet loaded in the GPU's HBM and must be read from the local NVMe SSD. Roughly how much slower is this initial read from the SSD compared to a subsequent read from HBM?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate the massive latency gap between fast storage and high-bandwidth memory. While NVMe is fast compared to spinning disks, it is orders of magnitude slower than HBM. A common mistake is to think the difference is small, like 10-20x, failing to grasp the physics of on-chip vs. off-chip vs. storage access.

  **Realistic Solution:** An NVMe SSD read takes approximately 100,000 ns, whereas an HBM3 memory access is around 300 ns. This results in a latency difference of about 333x. This 'cold start' penalty is why techniques like pre-warming caches and keeping models resident in memory are critical for low-latency serving systems.

  > **Napkin Math:** HBM3 Access Latency: ~300 ns. NVMe SSD Read Latency: ~100,000 ns. Ratio: 100,000 ns / 300 ns ≈ 333x. Using the human-scale analogy: if an HBM read took 5 minutes, the 'cold' read from NVMe would take ~1.1 days.

  > **Options:**
  > [ ] ~30x slower
  > [ ] ~3x slower
  > [x] ~330x slower
  > [ ] ~3,300x slower

  📖 **Deep Dive:** [The ML Latency Hierarchy](https://mlsysbook.ai/numbers.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Canary Memory Footprint</b> · <code>model-serving-memory</code></summary>

- **Interviewer:** "You are an ML Systems Engineer responsible for a cloud-based route-planning service for autonomous vehicles. The current service uses a 7-billion parameter model running in FP16 precision. You are planning a canary rollout of a new, much larger 70-billion parameter model, also in FP16. Your deployment pods are configured with a single H100 GPU. Explain the memory implications of this upgrade. Specifically, calculate the inference memory required for the new 70B model and contrast it with the available HBM on the GPU."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate the memory requirement (140 GB) in isolation but fail to connect it to the physical hardware constraints (an H100 has 80 GB HBM). They might assume this is a simple software problem solved by increasing a Kubernetes memory request, without realizing that a model of this size cannot be served on a single GPU and requires re-architecting the deployment for multi-GPU inference (e.g., with tensor parallelism).

  **Realistic Solution:** The new 70B model's memory requirement exceeds the capacity of a single H100 GPU, forcing a change in the serving architecture.

**Calculation:**
A model's inference memory footprint in FP16 is the number of parameters multiplied by 2 bytes per parameter.
- **New Model Memory:** 70 billion parameters × 2 bytes/parameter = 140 GB.
- **GPU Capacity:** A single H100 GPU has 80 GB of HBM3 memory.

**Conclusion:**
The 140 GB required by the model is significantly greater than the 80 GB available on the GPU. A simple pod update is not feasible. The serving strategy must be re-designed to use tensor parallelism, splitting the model across at least two H100 GPUs to fit into memory.

  > **Napkin Math:** 1. **Calculate Required Memory:**
   - Model Parameters: 70B
   - Precision: FP16 (2 bytes/parameter)
   - Memory = 70,000,000,000 params * 2 bytes/param = 140,000,000,000 bytes
   - Memory ≈ 140 GB

2. **Compare with Hardware:**
   - H100 HBM3 Memory: 80 GB

3. **Find the Gap:**
   - Memory Deficit = 140 GB (Required) - 80 GB (Available) = 60 GB

4. **Determine Minimum GPU Count:**
   - GPUs = ceil(Required Memory / GPU Memory) = ceil(140 / 80) = ceil(1.75) = 2 GPUs

  > **Key Equation:** $\text{Inference Memory (Bytes)} = \text{Parameters} \times \text{Bytes per Parameter}$

  > **Options:**
  > [ ] The model requires 70 GB, so it will fit on the 80 GB H100.
  > [ ] The model requires 140 GB; you just need to update the pod's memory request in Kubernetes.
  > [x] The model requires 140 GB, which exceeds the H100's 80 GB. A multi-GPU strategy is now required.
  > [ ] The model requires over 1.1 TB to store optimizer states, making it impossible to serve.

  📖 **Deep Dive:** [Cloud: Inference and Serving](https://mlsysbook.ai/cloud/03_inference_and_serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Static Batching Throughput Trap</b> · <code>llm-serving-latency-throughput</code></summary>

- **Interviewer:** "You are operating a user-facing chatbot on an H100 GPU. Your service has a strict P99 Time-To-First-Token (TTFT) SLA of 200ms. Your inference engine uses a simple static batching strategy. The time to run the prefill (prompt processing) stage for a batch of size `B` is dominated by memory operations and can be modeled as `T_prefill(B) = 50ms + 15ms * B`. After the prefill stage, the system generates one token for *every* request in the batch, and this decode step takes 10ms.

Calculate the maximum batch size `B` you can use without violating the 200ms TTFT SLA, and explain what the resulting total system throughput is in tokens per second when running continuously at this batch size."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to confuse the latency constraint (TTFT) with the throughput rate (decode time). Engineers often incorrectly use the 10ms per-token decode time to calculate the batch size limit (e.g., `10ms * B <= 200ms`), leading to a batch size that would actually take `50ms + 15ms * 20 = 350ms` to start, grossly violating the SLA. Another mistake is correctly calculating the batch size but then miscalculating the system throughput by not accounting for the number of parallel requests in the batch.

  **Realistic Solution:** The correct approach is to recognize that for a static batch, the time to the first token is determined entirely by the prefill stage for the whole batch. The SLA applies directly to this prefill time. Once the maximum batch size is found, the system throughput is the number of tokens generated per decode step (which is equal to the batch size) divided by the time for that step.

1.  **Find Max Batch Size:** Set the prefill time formula to be less than or equal to the SLA: `50ms + 15ms * B ≤ 200ms`.
2.  **Solve for B:** `15ms * B ≤ 150ms`, which simplifies to `B ≤ 10`.
3.  **Find System Throughput:** With a batch size of 10, the system generates 10 tokens every 10ms. This is equivalent to 1 token per millisecond, or 1000 tokens per second.

  > **Napkin Math:** SLA_TTFT = 200 ms
T_prefill(B) = 50ms + 15ms * B
T_decode_step = 10 ms

# Constraint for TTFT
T_prefill(B) ≤ SLA_TTFT
50 + 15 * B ≤ 200
15 * B ≤ 150
B_max = 10

# Throughput Calculation
Tokens_per_step = B_max = 10 tokens
Time_per_step = T_decode_step = 10 ms

System_Throughput = Tokens_per_step / Time_per_step
System_Throughput = 10 tokens / 10 ms = 1 token/ms
System_Throughput = 1000 tokens/sec

  > **Key Equation:** T_{\text{prefill}}(B) \le \text{SLA}_{\text{TTFT}}

  > **Options:**
  > [ ] Max Batch: 20, Throughput: 2000 tokens/sec
  > [ ] Max Batch: 13, Throughput: 1300 tokens/sec
  > [x] Max Batch: 10, Throughput: 1000 tokens/sec
  > [ ] Max Batch: 10, Throughput: 100 tokens/sec

  📖 **Deep Dive:** [Cloud: The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The On-Node Interconnect Ladder</b> · <code>interconnect-latency-topology</code></summary>

- **Interviewer:** "When analyzing a performance trace on a modern multi-GPU server, you notice two distinct latency buckets for data transfers between components: one taking approximately 500 nanoseconds and another taking approximately 1,000 nanoseconds (1 microsecond). Which interconnects most likely correspond to these two latencies?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to assume all on-server interconnects are roughly equivalent in performance or to confuse their roles. Some engineers might incorrectly guess PCIe is faster due to its ubiquity, or misremember the magnitude of the difference, thinking it's 10x or more. They are fundamentally different: NVLink is a specialized GPU-to-GPU interconnect, while PCIe is a general-purpose bus.

  **Realistic Solution:** The ~500 ns latency corresponds to an NVLink 4.0 transfer. This is the specialized, high-bandwidth, low-latency interconnect for direct GPU-to-GPU communication within a server, like on an NVIDIA HGX baseboard. The ~1,000 ns (1 µs) latency corresponds to a transfer over the PCIe Gen5 bus, which connects GPUs to the CPU or to other peripherals. The 2x latency penalty for using PCIe is a critical factor in system topology design.

  > **Napkin Math:** The numbers are recalled directly from the 'ML Latency Hierarchy'. NVLink 4.0 latency is ~500 ns, while PCIe Gen5 is ~1,000 ns. This represents a 2x latency difference. Using the human-scale analogy where 1 ns = 1 second: an L1 cache access is 1 second, an NVLink transfer takes ~8 minutes, and a PCIe transfer takes ~16 minutes. A trip across the country via InfiniBand would take 1.4 hours in comparison.

  > **Options:**
  > [ ] The ~500ns transfer is via PCIe Gen5; the ~1,000ns transfer is via NVLink 4.0.
  > [ ] The ~500ns transfer is via L2 Cache; the ~1,000ns transfer is via HBM3 memory access.
  > [x] The ~500ns transfer is via NVLink 4.0; the ~1,000ns transfer is via PCIe Gen5.
  > [ ] The ~500ns transfer is via InfiniBand NDR; the ~1,000ns transfer is via NVMe SSD read.

  📖 **Deep Dive:** [Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The RAG Retrieval Tax</b> · <code>rag-latency</code></summary>

- **Interviewer:** "You're designing a Retrieval-Augmented Generation (RAG) system for an automotive recall analysis chatbot. The vector database is stored on local NVMe SSDs. To meet your overall latency budget, you must first know the baseline cost of the retrieval step. What is the approximate latency for a single random read from an NVMe SSD?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse storage latency with memory or network latency. A common error is to assume SSD reads are in the nanosecond range (like HBM/DRAM) or the multi-millisecond range (like a cross-continent network call). The correct answer is in the distinct microsecond range.

  **Realistic Solution:** The latency is approximately 100 microseconds (100,000 nanoseconds). This is a fundamental number for system design involving fast storage. While orders of magnitude slower than memory access (~300 ns for HBM), it is significantly faster than traditional hard drives or long-distance network calls (~40 ms). In a RAG system, this retrieval latency is often a significant portion of the total time-to-first-token.

  > **Napkin Math:** If an L1 cache read on the CPU took 1 second, a single read from the NVMe SSD would take approximately 1.1 days (100,000 ns vs 1 ns). This illustrates why minimizing I/O, even to fast SSDs, is a critical design constraint for low-latency applications.

  > **Options:**
  > [ ] ~300 ns (HBM Memory Access)
  > [ ] ~5 µs (Cross-Rack InfiniBand)
  > [x] ~100 µs (NVMe SSD Read)
  > [ ] ~40 ms (Cross-Country Fiber)

  📖 **Deep Dive:** [The ML Latency Hierarchy](https://mlsysbook.ai/ironlaw.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Trade-off</b> · <code>continuous-batching-throughput</code></summary>

- **Interviewer:** "You are designing a serving system for a 70B parameter LLM on an H100 GPU. You are comparing two batching strategies: static batching and continuous batching. Your service has a strict P99 Time-To-First-Token (TTFT) SLO of 150ms.

Profiling gives you this data:
- Prompt processing (prefill) for any request takes a fixed 50ms.
- With **static batching**, the system waits to accumulate a batch of 8 requests. The total execution time for a batch-of-8 is 200ms.
- With **continuous batching**, the scheduler runs an iteration (generating one token for all active requests) every 25ms.

Explain the effective Time-Per-Output-Token (TPOT) for a user under the continuous batching setup, and compare the viability of the two systems."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to miscalculate the TTFT for the continuous batching system by either ignoring the scheduler wait time or incorrectly adding the prefill time sequentially. Another error is to apply the latency numbers from the static batching system to the continuous batching system, failing to recognize they are different operational models. Finally, candidates often confuse TTFT (latency to the *first* token) with TPOT (latency between *subsequent* tokens).

  **Realistic Solution:** First, we evaluate the static batching system. Even assuming zero wait time for the batch to fill, the TTFT is the sum of prefill and the batch execution time. This is `50ms + 200ms = 250ms`, which immediately violates the 150ms SLO. Therefore, static batching is not a viable strategy here.

Next, we evaluate the continuous batching system. A new request's TTFT is the sum of its prefill time, the wait time for the next scheduler cycle to begin, and the time for that single cycle to execute. In the worst case, a request finishes its 50ms prefill just as a new 25ms cycle begins, forcing it to wait for the next one.

So, `Worst-Case TTFT = Prefill Time + Max Cycle Wait Time + Cycle Execution Time = 50ms + 25ms + 25ms = 100ms`.
This is well within the 150ms SLO, making continuous batching viable.

The effective Time-Per-Output-Token (TPOT) for any given user in this system is the time it takes for the scheduler to complete one cycle and generate the next token for everyone. Therefore, the TPOT is equal to the iteration time.

  > **Napkin Math:** ## Static Batching Check:
- `T_prefill = 50ms`
- `T_exec_static_batch = 200ms`
- `TTFT_static = T_prefill + T_exec_static_batch = 50ms + 200ms = 250ms`
- `250ms (TTFT) > 150ms (SLO)` -> **FAILS**

## Continuous Batching Check:
- `T_prefill = 50ms`
- `T_iteration = 25ms`
- `TTFT_continuous (worst case) = T_prefill + T_wait_cycle + T_iteration = 50ms + 25ms + 25ms = 100ms`
- `100ms (TTFT) < 150ms (SLO)` -> **PASSES**

## Effective TPOT Calculation:
- The time between subsequent tokens for a user is one full scheduler cycle.
- `TPOT_continuous = T_iteration = 25ms`

  > **Key Equation:** $\text{TTFT}_{continuous} = T_{prefill} + T_{wait\_cycle} + T_{iteration}$

  > **Options:**
  > [ ] 250ms. The system is non-viable as it violates the SLO.
  > [ ] 100ms. This is the worst-case time to first token.
  > [x] 25ms. The system meets the SLO and the TPOT is the iteration time.
  > [ ] 50ms. This is the sum of the cycle wait and execution time.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Interconnect Latency Ladder</b> · <code>interconnect-latency</code></summary>

- **Interviewer:** "You're diagnosing a training job slowdown. To start, you want to sanity check your understanding of the system's latency characteristics. Which of these three operations has the highest latency (is the slowest)?

1. An NVLink 4.0 transfer between two GPUs on the same motherboard.
2. A PCIe Gen5 transfer between a GPU and the host CPU.
3. An InfiniBand NDR network hop to a GPU in an adjacent rack."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse bandwidth with latency. InfiniBand has enormous bandwidth (400 Gbps), so it's easy to assume it's the 'fastest'. However, latency is dominated by the physical distance and number of hops. A network trip, even over fast fiber, is vastly slower than on-board communication.

  **Realistic Solution:** The InfiniBand NDR hop has the highest latency. Communication that leaves the server and traverses the network (inter-node) is always significantly slower than communication within the server (intra-node).

- **NVLink** is fastest (~500 ns) because it's a direct, on-board link between GPUs.
- **PCIe** is next (~1,000 ns) as it connects the GPU to the CPU across the motherboard.
- **InfiniBand** is slowest (~5,000 ns) because the signal must leave the server, travel over cables to a network switch, and then to the destination server.

  > **Napkin Math:** Using the 'human time' analogy where 1ns = 1 second:
- **NVLink Transfer (~500 ns):** Takes about 8 minutes.
- **PCIe Transfer (~1,000 ns):** Takes about 16 minutes.
- **InfiniBand Hop (~5,000 ns):** Takes about 1.4 hours.

The network hop is an order of magnitude slower than staying on the server.

  > **Options:**
  > [ ] NVLink 4.0 Transfer
  > [ ] PCIe Gen5 Transfer
  > [x] InfiniBand NDR Hop
  > [ ] They are all roughly the same latency

  📖 **Deep Dive:** [Distributed Systems](https://mlsysbook.ai/cloud/02_distributed_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Inference Memory Footprint</b> · <code>model-serving-memory</code></summary>

- **Interviewer:** "You're preparing to deploy a standard 7 Billion parameter Large Language Model for an RAG application. For basic inference using half-precision (FP16), how much GPU memory should you budget just to load the model weights?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the memory requirements for *training* with those for *inference*. Training requires storing gradients and optimizer states (like Adam), which can be 8-10x more memory than just the model weights. Another common error is using the wrong number of bytes for the given precision (e.g., 4 for FP32 or 1 for INT8 instead of 2 for FP16).

  **Realistic Solution:** For FP16 (or BF16) inference, each parameter requires 2 bytes of memory. Therefore, a 7 Billion parameter model requires 14 GB of GPU memory just to load the weights. This is a critical baseline for selecting a cloud GPU instance (e.g., an NVIDIA A10G with 24GB would suffice, but a T4 with 16GB would be very tight). This static cost doesn't include the dynamic memory needed for the KV-cache, which grows with the sequence length and batch size.

  > **Napkin Math:** 7 Billion parameters × 2 bytes/parameter = 14 Billion bytes = 14 GB.

  > **Key Equation:** $\text{Memory (GB)} = \frac{\text{Parameters} \times \text{Bytes per Parameter}}{10^9}$

  > **Options:**
  > [ ] 7 GB
  > [ ] 28 GB
  > [ ] 112 GB
  > [x] 14 GB

  📖 **Deep Dive:** [Cloud / LLM Scaling Rules](https://mlsysbook.ai/vol2/NUMBERS.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Real-Time Batching Dilemma</b> · <code>llm-inference-ttft</code></summary>

- **Interviewer:** "You are deploying a Llama 70B model on a single H100 GPU for a real-time chatbot. The product team has a strict requirement that the time-to-first-token (TTFT) must be under 300ms. Given the hardware specs, explain how batch size impacts TTFT and calculate the maximum theoretical batch size you can use while staying within this latency budget. For this calculation, you can ignore all overheads like network latency and CUDA kernel launch times, focusing only on the raw prefill computation."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often make unit errors or misidentify the bottleneck. A common error is mixing up TFLOPS and GFLOPS, leading to a 1000x error in the calculation. Another is incorrectly assuming the prefill stage is memory-bound and basing the calculation on HBM bandwidth, rather than being compute-bound. Finally, a simple slip is forgetting the `2 * P` rule of thumb for inference FLOPs, leading to an off-by-2x error.

  **Realistic Solution:** The correct approach is to determine if the operation is compute-bound or memory-bound. The prefill step for a large language model is a large matrix multiplication, which is compute-bound. Therefore, we should use the GPU's computational throughput (FLOPS), not its memory bandwidth.

First, we calculate the number of operations required for the prefill of a single item in the batch. Then, we find the time this takes on the H100. Finally, we can determine how many items can fit into our 300ms latency budget.

The calculation shows that a surprisingly large batch can theoretically be processed. This highlights that for pure on-chip compute, modern accelerators are incredibly fast, and real-world latency is often dominated by other factors (memory access, software overhead, networking) that were explicitly excluded by the question.

  > **Napkin Math:** 1. **Calculate FLOPs for Prefill (per item):** The rule of thumb for inference is `2 * Parameters` FLOPs per token generated. For a 70B model, this is:
   `Compute_per_item = 2 * 70B params = 140 GFLOPs`

2. **Find GPU Throughput:** From the provided table, an NVIDIA H100 provides `989 TFLOPS` for FP16 compute.

3. **Calculate Time per Item:** Find the time to process one item by dividing the required FLOPs by the GPU's FLOPs.
   `Time_per_item = (140 * 10^9 FLOPs) / (989 * 10^12 FLOPs/sec) ≈ 0.000141 seconds ≈ 0.14 ms`

4. **Calculate Max Batch Size:** Divide the total latency budget by the time it takes to process a single item in the batch.
   `Max_Batch_Size = Latency_Budget / Time_per_item = 300 ms / 0.14 ms ≈ 2142`

Therefore, the theoretical maximum batch size is approximately 2142.

  > **Key Equation:** $\text{Time}_{\text{prefill}} = \frac{2 \times \text{Parameters} \times \text{Batch Size}}{\text{GPU FLOPS}}$

  > **Options:**
  > [ ] ~2. This calculation incorrectly mixes up GFLOPS and TFLOPS, leading to a 1000x error and drastically underestimating the GPU's capability.
  > [ ] ~7. This calculation incorrectly uses memory bandwidth to load the 140GB model weights into memory, confusing a compute-bound problem with a memory-bound one.
  > [x] ~2120. This correctly identifies the problem as compute-bound and divides the latency budget by the time required to perform the 140 GFLOPs of computation for each request on a 989 TFLOP/s H100.
  > [ ] ~4240. This calculation makes an off-by-two error, using `1 * Parameters` instead of the correct `2 * Parameters` rule of thumb for inference compute.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The 7B Model Memory Footprint</b> · <code>model-serving-memory</code></summary>

- **Interviewer:** "You're tasked with deploying a 7-billion parameter language model for a new Retrieval-Augmented Generation (RAG) feature. Before considering orchestration or the KV cache, what is the absolute minimum GPU memory required just to load the model's weights for inference using half-precision (FP16)?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the memory requirements for inference with those for training, which are much larger (e.g., 16 bytes/param with the Adam optimizer). Another common mistake is using the wrong number of bytes for a given precision, such as assuming 1 byte/param (for INT8) or 4 bytes/param (for FP32) when FP16 is specified.

  **Realistic Solution:** The standard rule of thumb for FP16 inference is that each parameter requires 2 bytes of memory. Therefore, a 7-billion parameter model requires 14 billion bytes, or 14 GB of GPU memory, just to hold the weights. This is a baseline before accounting for activations or the KV cache.

  > **Napkin Math:** 7 Billion Parameters × 2 bytes/parameter (FP16) = 14 Billion Bytes = 14 GB.

  > **Key Equation:** $\text{Inference Memory} \approx \text{Parameters} \times \text{Bytes per Parameter}$

  > **Options:**
  > [ ] 7 GB
  > [x] 14 GB
  > [ ] 28 GB
  > [ ] 112 GB

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Static Batching Trap</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are optimizing an LLM inference service that uses a static batching strategy on a single H100 GPU. The server collects incoming requests for a fixed window of 40ms. At the end of the window, it processes the entire batch, which takes 10ms for the prefill stage. A user's request arrives at the worst possible moment (just after a batch was sent for processing). What is the user's Time-To-First-Token (TTFT)?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus solely on the GPU processing time (10ms), a metric related to inference speed, and forget that in a queued system, the total user-perceived latency is dominated by wait time (head-of-line blocking). The worst-case TTFT is the sum of the maximum wait time and the processing time.

  **Realistic Solution:** The worst possible moment for a request to arrive is nano-seconds after a batch has been finalized and sent to the GPU. The new request has 'missed the bus.' It must wait the *entire* next batching window (40ms) to be collected. After this wait, its batch is processed, which takes 10ms. Therefore, the total time from request arrival to the first token being ready is the sum of the wait and processing times. This scenario highlights the core latency problem that continuous batching was invented to solve: eliminating the forced idle time imposed by static batch windows.

  > **Napkin Math:** Worst-Case TTFT = Maximum Batch Wait Time + Batch Processing Time
Worst-Case TTFT = 40ms + 10ms = 50ms

  > **Key Equation:** $T_{\text{TTFT, worst}} = T_{\text{wait\_max}} + T_{\text{process}}$

  > **Options:**
  > [ ] 10ms
  > [ ] 40ms
  > [x] 50ms
  > [ ] 30ms

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The On-Node vs. Off-Node Divide</b> · <code>interconnect-latency</code></summary>

- **Interviewer:** "Your model training requires frequent peer-to-peer transfers between two H100 GPUs. For minimizing latency on a small data payload, which is faster: a transfer between GPUs on the same server connected by NVLink 4.0, or a transfer between GPUs on different servers connected by an InfiniBand NDR link? By roughly what factor?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often conflate bandwidth and latency. They might know that both NVLink and InfiniBand are 'fast' high-bandwidth interconnects and assume their latencies are comparable. They forget that crossing the server boundary (off-node) incurs significant protocol and distance overhead compared to staying on the same motherboard (on-node).

  **Realistic Solution:** NVLink is roughly 10 times faster in terms of latency. NVLink is a direct, on-node (intra-server) interconnect between GPUs, resulting in extremely low latency. InfiniBand is a high-performance, off-node (inter-server) network. While it has massive bandwidth, a transfer must go through the network interface card (NIC), across cables, to another NIC, which adds significant latency compared to the short, on-board path of NVLink.

  > **Napkin Math:** From the 'ML Latency Hierarchy' table:
- NVLink 4.0 Transfer Latency: ~500 ns
- InfiniBand NDR Transfer Latency: ~5,000 ns (5 µs)

Ratio = InfiniBand Latency / NVLink Latency
Ratio = 5,000 ns / 500 ns = 10x

If an L1 cache access took 1 second, an NVLink transfer would take 8 minutes, while an InfiniBand transfer would take 1.4 hours.

  > **Options:**
  > [ ] They are roughly the same speed.
  > [ ] InfiniBand is ~2x faster.
  > [x] NVLink is ~10x faster.
  > [ ] NVLink is ~100x faster.

  📖 **Deep Dive:** [Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The FP16 Inference Memory Footprint</b> · <code>model-serving</code></summary>

- **Interviewer:** "You are deploying a 7B parameter LLM for a RAG application. To serve this model for inference, approximately how much GPU memory is required to hold just the model weights in standard half-precision (FP16)?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is confusing inference memory with training memory. During training with an optimizer like Adam, memory requirements are much higher (~16 bytes per parameter) to store gradients and optimizer state. Another frequent error is using the wrong precision; for example, calculating for full precision (FP32, 4 bytes/param) or 8-bit quantization (INT8, 1 byte/param) instead of the specified FP16.

  **Realistic Solution:** In FP16 (half-precision), each parameter requires 2 bytes of storage. Therefore, a 7 billion parameter model will require 7 billion × 2 bytes = 14 billion bytes, or 14 GB of GPU memory just for the weights. This is a baseline; additional memory is always needed for the KV cache, activations, and the serving framework's overhead.

  > **Napkin Math:** 7B params × 2 bytes/param (FP16) = 14 GB

  > **Key Equation:** $\text{Inference Memory (FP16)} = \text{Parameters} \times 2 \text{ bytes}$

  > **Options:**
  > [ ] 7 GB
  > [x] 14 GB
  > [ ] 28 GB
  > [ ] 112 GB

  📖 **Deep Dive:** [Cloud: The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Static Batching Latency Tax</b> · <code>serving-latency-batching</code></summary>

- **Interviewer:** "You're operating an LLM inference service for a real-time code assistant. The service uses a simple static batching strategy. The server is configured to wait up to **100ms** to form a batch of 4 requests before sending them to the GPU. The actual GPU computation for a full or partial batch takes **300ms**.

A user's request arrives at your service, and it happens to be the very first one to arrive when the server is idle. Calculate the total latency this specific user will experience, from the moment their request hits the server until the computation is finished and the first token is ready."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus on throughput and only consider the raw GPU compute time (300ms), completely forgetting the latency cost imposed by the batching window. They calculate the best-case or average-case scenario, not the worst-case latency for a specific user, which is critical for interactive applications with real-time deadlines.

  **Realistic Solution:** The total latency is the sum of the maximum time the server waits to form a batch and the time it takes to compute the batch.

1.  The user's request is the first in the queue, so the server starts its 100ms waiting window.
2.  Since no other requests arrive, the server waits for the full `100ms` timeout.
3.  After the timeout, the server sends the single-request batch to the GPU.
4.  The GPU computation takes `300ms`.

Therefore, the user experiences the waiting delay *plus* the compute delay.

  > **Napkin Math:** Total Latency = Batching Wait Time + GPU Compute Time
Total Latency = 100ms + 300ms
Total Latency = 400ms

  > **Key Equation:** $$ T_{\text{latency}} = T_{\text{wait}} + T_{\text{compute}} $$

  > **Options:**
  > [ ] 300ms
  > [ ] 100ms
  > [x] 400ms
  > [ ] 75ms

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The On-Node vs. Off-Node Latency Chasm</b> · <code>interconnect-latency</code></summary>

- **Interviewer:** "State the approximate latency difference between a GPU-to-GPU data transfer using on-node NVLink 4.0 versus a cross-rack InfiniBand NDR transfer."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate the 'physics tax' of leaving the server chassis. They might assume high-performance networks like InfiniBand have latencies comparable to on-node interconnects, confusing high bandwidth with low latency. In reality, the physical distance and protocol overhead create a significant latency gap.

  **Realistic Solution:** An InfiniBand transfer is roughly 10 times slower (higher latency) than an NVLink transfer. NVLink is a direct, on-board connection between GPUs with a latency of ~500 ns, while InfiniBand is a switched network fabric that must traverse NICs, cables, and switches to reach another node, resulting in a latency of ~5,000 ns.

  > **Napkin Math:** Using the 'human time' analogy where 1 ns is 1 second:
- NVLink 4.0 Transfer (~500 ns) → 8 minutes
- InfiniBand NDR Transfer (~5,000 ns) → 1.4 hours

The cross-rack transfer takes over an hour longer in human-scaled time, making the ~10x difference clear.

  > **Options:**
  > [ ] InfiniBand is about the same latency as NVLink.
  > [ ] InfiniBand is about 100x slower than NVLink.
  > [x] InfiniBand is about 10x slower than NVLink.
  > [ ] InfiniBand is about 2x faster than NVLink.

  📖 **Deep Dive:** [Distributed Systems](https://mlsysbook.ai/cloud/distributed-systems)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The 7B Model Memory Footprint</b> · <code>model-serving-memory</code></summary>

- **Interviewer:** "You've been tasked with deploying a 7-billion parameter Large Language Model for a new RAG-based chatbot. Before you even think about container orchestration or KV cache, what is the *absolute minimum* memory (RAM or HBM) required to simply load the model weights for inference using a standard FP16 format?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the memory requirements for *training* with *inference*. During training with optimizers like Adam, memory usage is much higher (around 16 bytes per parameter) to store gradients and optimizer states. For simple inference, you only need to store the model weights themselves. Another common error is mixing up FP16 (2 bytes) with INT8 (1 byte) or FP32 (4 bytes).

  **Realistic Solution:** A 7-billion parameter model requires approximately 14 GB of memory for inference in FP16. Each parameter in FP16 (half-precision floating-point) format takes up 2 bytes of memory.

  > **Napkin Math:** The calculation is a direct application of the scaling rule for inference memory:
`Memory = 7 billion parameters × 2 bytes/parameter`
`Memory = 14,000,000,000 bytes`
`Memory = 14 GB`

  > **Key Equation:** $\text{Inference Memory (FP16)} = \text{Parameters} \times 2 \text{ bytes}$

  > **Options:**
  > [ ] 112 GB
  > [ ] 7 GB
  > [x] 14 GB
  > [ ] 28 GB

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Canary Rollout Memory Budget</b> · <code>model-serving-memory</code></summary>

- **Interviewer:** "An autonomous vehicle company uses a central, cloud-hosted RAG model to provide real-time, complex query support to its fleet. The current production model has 7B parameters. A new, more accurate 13B parameter model is ready for a canary rollout. As the Staff ML Systems Engineer, you need to provision the canary cluster. Calculate the minimum VRAM required for a single server instance to simply load this new 13B model for inference, assuming it's served in FP16 precision."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the memory requirements for *training* with those for *inference*. Training with an optimizer like Adam requires storing parameters, gradients, and optimizer states, leading to a much larger footprint (~16 bytes per parameter). Inference, in its simplest form, only requires the model weights.

  **Realistic Solution:** For inference in half-precision (FP16), each parameter requires 16 bits, which is 2 bytes. To calculate the total memory for the model weights, you multiply the number of parameters by the size of each parameter in bytes. This gives you the minimum VRAM required to load the model, not including the KV cache or activation memory.

  > **Napkin Math:** 1. **Parameters:** 13 Billion
2. **Precision:** FP16 = 16 bits = 2 bytes per parameter
3. **Calculation:** 13 Billion parameters × 2 bytes/parameter = 26 Billion bytes
4. **Conversion:** 26 Billion bytes = 26 GB

The server must have at least 26 GB of VRAM.

  > **Key Equation:** $\text{Inference Memory (FP16)} = \text{Parameters} \times 2 \text{ bytes}$

  > **Options:**
  > [ ] 13 GB
  > [ ] 52 GB
  > [x] 26 GB
  > [ ] 208 GB

  📖 **Deep Dive:** [Inference and Serving](https://mlsysbook.ai/cloud/03_inference_and_serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Throughput Limit</b> · <code>continuous-batching-throughput</code></summary>

- **Interviewer:** "You are tasked with estimating the peak capacity of an LLM inference service running on a single H100 GPU. The service runs a 70B parameter model and uses continuous batching. At steady state, telemetry shows that there are, on average, 32 requests being actively processed concurrently. The average request requires a 20ms prefill/prompt processing phase and then generates 128 new tokens.

Given this, explain how you would calculate the maximum sustainable query rate in Queries Per Second (QPS)."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often misapply throughput formulas. A common mistake is to simply invert the per-token generation time, which calculates the *token* throughput, not the *request* throughput. Another is to calculate the throughput for a single, isolated request, ignoring the massive parallelism gains from batching. A third mistake is to apply static batching logic, where the throughput is the batch size divided by the time it takes for the slowest member of the batch to complete, which underestimates the efficiency of continuous batching.

  **Realistic Solution:** The correct approach is to use Little's Law, which states that the average number of items in a system (L) is equal to their average arrival rate (λ) multiplied by the average time they spend in the system (W).

Here, L is the average number of concurrent requests (32), and W is the total time a request is processed. We need to calculate W and then solve for λ (the QPS).

1.  **Calculate per-token compute:** A 70B model needs `~2 FLOPs/param * 70B params = 140 GFLOPs` per token.
2.  **Calculate batch compute:** For a batch of 32, one token-generation step requires `32 * 140 GFLOPs = 4.48 TFLOPs`.
3.  **Calculate Time Per Output Token (TPOT):** An H100 provides 989 TFLOPS of FP16 compute. The time for one step is `4.48 TFLOPs / 989 TFLOPS ≈ 4.53 ms`.
4.  **Calculate total decode time:** For 128 tokens, this is `128 tokens * 4.53 ms/token ≈ 580 ms`.
5.  **Calculate total request time (W):** `W = Prefill Time + Decode Time = 20 ms + 580 ms = 600 ms = 0.6 s`.
6.  **Apply Little's Law (λ = L/W):** `λ = 32 requests / 0.6 s ≈ 53.3 QPS`.

The system can sustain approximately 53 queries per second.

  > **Napkin Math:** 1. **Find total time per request (W):**
   - Per-token compute: `70B params * 2 FLOPs/param = 140 GFLOPs`
   - Batch-step compute: `140 GFLOPs/req * 32 reqs = 4.48 TFLOPs`
   - Batch-step time (TPOT): `4.48 TFLOPs / 989 TFLOPS (H100) ≈ 4.5 ms`
   - Decode time: `128 tokens * 4.5 ms/token = 576 ms`
   - Total time W: `20ms (prefill) + 576ms (decode) = 596ms ≈ 0.6s`

2. **Apply Little's Law (λ = L/W):**
   - `λ = 32 requests / 0.6s ≈ 53.3 QPS`

  > **Key Equation:** $\text{Little's Law: } L = \lambda W \implies \lambda = L / W$

  > **Options:**
  > [ ] ~220 QPS
  > [ ] ~1.7 QPS
  > [x] ~53 QPS
  > [ ] ~56 QPS

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Blue-Green Memory Tax</b> · <code>model-serving-rollout</code></summary>

- **Interviewer:** "You are the ML Systems Engineer for a popular RAG-based customer support bot. The bot's generator is a 7B parameter LLM. Your team is rolling out an updated version of this model using a blue-green deployment strategy to ensure zero downtime and instant rollback capabilities. To do this, you must run both the old (blue) and new (green) model containers simultaneously in your Kubernetes cluster during the transition. Explain the memory implications of this strategy and calculate the minimum total memory required just for these two generator models during the deployment overlap, assuming they are both served in FP16 precision."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to calculate the memory requirement for only a single model, forgetting that a blue-green deployment by definition duplicates the entire service, doubling the peak resource requirement during the switchover. Another frequent error is using the wrong number of bytes per parameter, such as 4 for FP32 or 1 for INT8, instead of the correct 2 for FP16.

  **Realistic Solution:** A blue-green deployment requires running two full, independent copies of the service concurrently to allow for a seamless traffic switch. Therefore, we must calculate the memory for one model and then double it.

An FP16 parameter requires 2 bytes of storage. For a 7B model, the memory for the parameters is 7 billion parameters × 2 bytes/parameter, which equals 14 GB. Since we have two models (the blue and green deployments) running at the same time, the total required memory is 14 GB × 2 = 28 GB. The cluster must have at least 28 GB of available memory to handle the deployment without killing other pods.

  > **Napkin Math:** 1. **Parameters per model:** 7 Billion
2. **Precision:** FP16 (requires 2 bytes per parameter)
3. **Memory per model:** 7B params × 2 bytes/param = 14 GB
4. **Deployment Strategy:** Blue-Green (requires 2 models running concurrently)
5. **Total Required Memory:** 14 GB/model × 2 models = 28 GB

  > **Key Equation:** $\text{Total Memory} = (\text{Parameters} \times \text{Bytes per Parameter}) \times 2$

  > **Options:**
  > [ ] 14 GB (Forgets to double for blue-green deployment)
  > [ ] 56 GB (Incorrectly uses 4 bytes/param for FP32 precision)
  > [x] 28 GB
  > [ ] 7 GB (Incorrectly uses 1 byte/param and forgets to double)

  📖 **Deep Dive:** [Production Ops](https://mlsysbook.ai/vol2/cloud/production_ops.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Static Batching Timeout Trap</b> · <code>continuous-batching-latency</code></summary>

- **Interviewer:** "You are optimizing an LLM chatbot service running on a single H100 GPU. The service uses a **static batching** strategy: it waits for up to 8 requests or a **200ms timeout** before processing a batch. The model's prefill stage takes **50ms**, and each decoding step (TPOT) takes **5ms**. A user sends a request when the server is completely idle. Explain the components of the Time-To-First-Token (TTFT) and calculate the best-case TTFT this user will experience."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The common mistake is to only consider the raw computation time (50ms for prefill) and ignore the queuing delay imposed by the batching strategy. In a static batching system, especially when traffic is low or bursty, the timeout window often dominates the end-to-end latency, leading to poor user experience despite having a powerful GPU.

  **Realistic Solution:** The total Time-To-First-Token is the sum of the time the request waits in the queue plus the time it takes to process the prompt (prefill). In this static batching scenario, even though the server is idle, the new request must wait for the 200ms timeout to expire. Only then will the batch of size 1 be sent to the GPU for processing. Therefore, the TTFT is the queuing delay (timeout) plus the prefill time.

With continuous batching, the server would process the request immediately, making the TTFT just the prefill time (50ms). This illustrates the primary advantage of continuous batching for reducing idle-server latency.

  > **Napkin Math:** `TTFT = T_queue + T_prefill`

- **T_queue (Static Batching):** The request arrives at an idle server but must wait for the batching timeout. `T_queue = 200ms`.
- **T_prefill:** The time to process the prompt. `T_prefill = 50ms`.

- **Total TTFT:** `200ms (timeout) + 50ms (prefill) = 250ms`.

  > **Key Equation:** $TTFT = T_{queue} + T_{prefill}$

  > **Options:**
  > [ ] 5ms
  > [ ] 50ms
  > [x] 250ms
  > [ ] 255ms

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The FP16 Memory Footprint</b> · <code>model-serving</code></summary>

- **Interviewer:** "You're part of the cloud infrastructure team for an autonomous vehicle company. You need to deploy a 7-billion parameter LLM for a new log analysis service. For basic capacity planning, what is the approximate GPU memory required just to load the model weights for inference using FP16 precision?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is confusing inference memory with training memory. Training with an optimizer like Adam requires storing gradients and optimizer states, leading to a much larger footprint (around 16 bytes per parameter). Another error is using the wrong byte size, for instance, assuming 4 bytes (FP32) instead of 2 bytes for FP16.

  **Realistic Solution:** A model's memory footprint for inference is determined by the number of parameters and the precision used. For FP16 (half-precision floating point), each parameter requires 2 bytes. Therefore, a 7-billion parameter model needs approximately 14 GB of memory just for the weights.

  > **Napkin Math:** 7 billion parameters × 2 bytes/parameter = 14 GB

  > **Key Equation:** $\text{Inference Memory} = \text{Parameters} \times \text{bytes_per_parameter}$

  > **Options:**
  > [ ] 112 GB
  > [ ] 28 GB
  > [x] 14 GB
  > [ ] 1.4 GB

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Blue/Green Capacity Trap</b> · <code>model-serving-rollout</code></summary>

- **Interviewer:** "You are managing a fleet of inference servers for a production RAG chatbot. The current 'blue' deployment uses a 7B parameter model, with each replica running on a dedicated H100 GPU. To handle the production traffic load, this 'blue' deployment is scaled to 10 replicas (10 GPUs).

You are tasked with rolling out a new, larger 13B parameter model via a 'green' deployment. Explain how a blue/green deployment strategy impacts your cluster's capacity, and calculate the total number of H100 GPUs required *at the peak of the deployment* to ensure the update happens with zero downtime."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate the peak resource requirements of a blue/green deployment. A common mistake is to only calculate the resources for the new 'green' deployment, assuming resources from the 'blue' deployment can be immediately reused. This leads to resource starvation, failing the new deployment or causing downtime as the container orchestrator cannot schedule the new pods.

  **Realistic Solution:** A blue/green deployment requires running two full production stacks in parallel: the existing 'blue' version and the new 'green' version. The 'green' deployment must be fully scaled, deployed, and healthy before traffic is switched over. Therefore, you must have enough capacity for both fleets to exist simultaneously.

Since the 'green' deployment must be ready to handle the full production load, it needs the same number of replicas as the 'blue' deployment. The total number of GPUs required at peak is the sum of GPUs for both deployments.

  > **Napkin Math:** 1. **Blue Deployment Capacity:** The problem states the current deployment requires 10 H100 GPUs.
2. **Green Deployment Capacity:** The new 13B model must also handle the full production load, so it requires the same number of replicas: 10. We verify the model fits on a single GPU: 13B params × 2 bytes/param (for FP16) = 26 GB. This is well within the 80 GB HBM of an H100. So, the green deployment also requires 10 H100 GPUs.
3. **Peak Capacity:** During the transition, both deployments are live.
4. **Total GPUs:** 10 GPUs (Blue) + 10 GPUs (Green) = 20 H100 GPUs.

  > **Key Equation:** $\text{Capacity}_{\text{peak}} = \text{Capacity}_{\text{blue}} + \text{Capacity}_{\text{green}}$

  > **Options:**
  > [ ] 10 GPUs. The new model replaces the old one, so the same number of GPUs is sufficient.
  > [ ] 11 GPUs. You just need one extra GPU to start the rollout, and the orchestrator will handle the rest.
  > [x] 20 GPUs. The entire new 'green' deployment must run in parallel with the old 'blue' deployment before traffic is switched.
  > [ ] 19 GPUs. The new model is roughly twice as large (13B/7B), so you need about twice the GPUs, but you can reuse one from the old fleet.

  📖 **Deep Dive:** [Production Ops](cloud/04_production_ops.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Real-Time Dilemma: TTFT vs. Throughput</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are designing the serving infrastructure for a real-time AI code assistant. The absolute strictest user-facing SLO is a P99 Time-To-First-Token (TTFT) of less than 200ms. The system uses H100 GPUs. Your team is debating two strategies for processing the initial user prompts (the 'prefill' stage):

1.  **Static Batching:** The system collects incoming requests for a fixed window of 80ms and then processes them together. This batched prefill computation takes 40ms to complete.
2.  **Continuous Batching:** The system processes requests as they arrive, adding them to the current batch on the GPU whenever possible.

First, explain the fundamental trade-off between static and continuous batching for this real-time application. Second, calculate the worst-case TTFT a user could experience with the **static batching** strategy."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to focus only on the GPU compute time and ignore the queueing delay. Engineers often answer '40ms', completely forgetting that a request's latency includes the time it spends waiting to be processed. Another error is calculating an *average* case (e.g., assuming a 40ms average wait) instead of the *worst case*, which is what a P99 SLO requires.

  **Realistic Solution:** The fundamental trade-off is between system throughput and individual request latency.

*   **Static Batching** prioritizes throughput. By waiting to group requests, it ensures the GPU processes a large, efficient batch, maximizing FLOP/s utilization. However, it introduces significant 'head-of-line blocking,' where early-arriving requests are forced to wait for the batch window to close, hurting latency (TTFT).
*   **Continuous Batching** (or in-flight batching) prioritizes latency. It dynamically adds new requests to the batch being processed by the GPU. This drastically reduces queueing time and improves TTFT, making the system feel more responsive. The cost can be slightly lower overall throughput if batches are less optimally packed.

To calculate the worst-case TTFT for the static batching system, we must consider a request that arrives *just after* a batching window has closed. This request experiences the maximum possible wait time before it can even be considered for processing.

`Worst-Case TTFT = Max Batching Window Wait Time + Batch Prefill Compute Time`

The request waits the full 80ms for the *next* batch to be collected. Then, its prefill is computed as part of that batch, which takes another 40ms.

`Worst-Case TTFT = 80ms + 40ms = 120ms`.

This is below the 200ms P99 SLO, but it highlights the significant latency cost of static batching.

  > **Napkin Math:** `Max_Wait_Time = 80ms` (A request arrives at t=0.01ms, right after the prior batch collection ended. It must wait for the entire next 80ms collection window).

`Prefill_Compute_Time = 40ms` (This is the given time to process the batch on the GPU).

`Worst_Case_TTFT = Max_Wait_Time + Prefill_Compute_Time`

`Worst_Case_TTFT = 80ms + 40ms = 120ms`

  > **Key Equation:** $TTFT_{worst} = T_{wait\_max} + T_{compute}$

  > **Options:**
  > [ ] 40ms
  > [ ] 80ms
  > [x] 120ms
  > [ ] 200ms

  📖 **Deep Dive:** [The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The On-Node vs. Cross-Node Latency Jump</b> · <code>nvlink-vs-infiniband-latency</code></summary>

- **Interviewer:** "You're debugging a distributed training job and notice high communication overhead. To start, you want to sanity check your understanding of the network latencies. Roughly how much slower is a cross-rack InfiniBand NDR transfer compared to a transfer between two GPUs on the same H100 server using NVLink 4.0?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate the 'datacenter tax.' They assume that high-performance networks like InfiniBand are nearly as fast as on-server interconnects like NVLink, perhaps only 2-3x slower. They forget that leaving the server motherboard, traversing optical cables, and hitting a switch adds a significant and discrete latency penalty, even with RDMA.

  **Realistic Solution:** A cross-rack InfiniBand transfer is about 10 times slower than an on-node NVLink transfer. NVLink is a highly optimized, short-distance electrical link between GPUs on the same physical board (~500 ns). InfiniBand has to send the signal out of the server, often over optical cables to a rack switch (and potentially another switch for cross-rack communication), which introduces significant latency from signal conversion and distance, resulting in a latency of ~5,000 ns (5 µs).

  > **Napkin Math:** From the 'ML Latency Hierarchy' numbers:
- NVLink 4.0 Transfer Latency: ~500 ns
- InfiniBand NDR (cross-rack) Latency: ~5,000 ns

Ratio = Cross-rack Latency / On-node Latency
Ratio = 5,000 ns / 500 ns = 10x slower.

Using the human-time analogy: An NVLink transfer is like waiting 8 minutes, while a cross-rack InfiniBand transfer is like waiting 1.4 hours.

  > **Key Equation:** $\text{LatencyRatio} = \frac{\text{Latency}_{\text{cross-rack}}}{\text{Latency}_{\text{on-node}}}$

  > **Options:**
  > [ ] Roughly the same, the main difference is bandwidth
  > [ ] ~2x slower
  > [x] ~10x slower
  > [ ] ~100x slower

  📖 **Deep Dive:** [Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The RAG Latency Trap</b> · <code>rag-latency-bottleneck</code></summary>

- **Interviewer:** "A user query to a chatbot triggers a Retrieval-Augmented Generation (RAG) pipeline. The first step is a lookup in a vector database index stored on a local NVMe SSD. The second step is feeding the retrieved context to an LLM for response generation. Which of these two stages is the dominant source of latency?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often anchor on the outdated intuition that I/O (like a database lookup) is always the slowest part of any operation. This was true for spinning disks, but the immense speed of modern NVMe SSDs combined with the massive memory-bandwidth requirements of LLM inference has completely flipped this dynamic.

  **Realistic Solution:** The LLM generation stage is, by far, the dominant source of latency. A lookup from a local NVMe SSD is an extremely fast operation, typically taking around 100 microseconds. In contrast, generating a response from a multi-billion parameter LLM is a memory-bandwidth-bound process that takes hundreds of milliseconds, even on a top-tier GPU. The generation phase is several orders of magnitude slower than the retrieval phase.

  > **Napkin Math:** From the ML Latency Hierarchy, an NVMe SSD read is ~100,000 ns (100 µs). A conservative estimate for generating a ~50 token response from a 7B model on an H100 is ~100ms, or 100,000,000 ns. The ratio is 100,000,000 ns / 100,000 ns = 1,000x. If the SSD lookup took 1 second, the LLM generation would take over 16 minutes.

  > **Options:**
  > [ ] The vector database lookup, because disk I/O is fundamentally slower than on-chip computation.
  > [x] The LLM generation, because it is an intensely memory-bandwidth-bound operation.
  > [ ] They are roughly equal, with compute and I/O taking about the same amount of time.
  > [ ] The network transfer between the database and the LLM server.

  📖 **Deep Dive:** [Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Static Batching Penalty</b> · <code>inference-serving-latency</code></summary>

- **Interviewer:** "You are managing a real-time translation service running on a single GPU. The service has a strict Time-To-First-Token (TTFT) deadline of 200ms. Your system uses a static batching strategy, where it collects incoming requests for a fixed 100ms window before processing the batch. The actual GPU processing for the batch takes 50ms. If a request arrives at the very beginning of the batching window, calculate its total TTFT and determine if it meets the deadline."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus solely on the GPU processing time (50ms) and forget to include the artificial latency introduced by the batching window. They see the low processing time and assume the system is well within its SLA, ignoring that the end-user experiences the full end-to-end latency, which includes time spent waiting in a queue.

  **Realistic Solution:** The total Time-To-First-Token is the sum of the time spent waiting for the batch window to close and the subsequent GPU processing time. For a request that arrives at the beginning of the window, it must wait the full 100ms for the window to elapse. After that, the batch is sent for processing, which takes another 50ms. Therefore, the total TTFT is 150ms. Since 150ms is less than the 200ms deadline, the system meets its SLA for this 'worst-case' request.

  > **Napkin Math:** Total Latency = Batch Window Wait Time + GPU Processing Time

TTFT = 100ms (waiting for the static batch window to close) + 50ms (processing the batch)
TTFT = 150ms

Comparison: 150ms (Actual) < 200ms (SLA Requirement)

Conclusion: The system meets the deadline.

  > **Key Equation:** $\text{TTFT} = T_{\text{wait_max}} + T_{\text{process}}$

  > **Options:**
  > [ ] 50ms. It easily meets the deadline.
  > [ ] 100ms. It meets the deadline.
  > [x] 150ms. It meets the deadline.
  > [ ] 250ms. It fails to meet the deadline.

  📖 **Deep Dive:** [The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Interconnect Latency Ladder</b> · <code>interconnect-latency</code></summary>

- **Interviewer:** "An ML training job needs to transfer a small control message between two H100 GPUs. Roughly how much slower is this transfer if the GPUs are in different racks, connected by an InfiniBand NDR switch, compared to if they are in the same physical server connected by NVLink?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often underestimate the 'tax' of leaving the server chassis. They might assume modern networks like InfiniBand are nearly as fast as on-board interconnects, confusing high throughput (GB/s) with low latency (ns). While both are fast, the physical distance and protocol overhead of going cross-rack adds a significant, order-of-magnitude latency penalty compared to the tightly-coupled, on-board NVLink.

  **Realistic Solution:** A cross-rack InfiniBand transfer is approximately 10 times slower than an intra-node NVLink transfer. NVLink latency is around 500 ns, as it's a direct GPU-to-GPU interconnect on the same server board. InfiniBand NDR, while extremely fast for a network, involves traversing network interface cards (NICs), cables, and a switch, resulting in a latency of about 5,000 ns (5 µs).

  > **Napkin Math:** Using the human-scale analogy where 1 ns is 1 second: An NVLink transfer feels like waiting 8 minutes. The same transfer over InfiniBand would feel like waiting 1.4 hours. The ratio is 5000 ns / 500 ns = 10x.

  > **Key Equation:** $\text{Latency Ratio} = \frac{\text{Latency}_{\text{cross-rack}}}{\text{Latency}_{\text{intra-node}}}$

  > **Options:**
  > [ ] Roughly the same speed
  > [ ] ~2x slower
  > [x] ~10x slower
  > [ ] ~100x slower

  📖 **Deep Dive:** [The ML Latency Hierarchy](ironlaw.qmd#1-the-ml-latency-hierarchy-2025-update)
  </details>
</details>

































<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Real-Time Translation Bottleneck</b> · <code>continuous-batching-latency</code></summary>

- **Interviewer:** "You are designing an LLM serving system for a real-time translation application on a single H100 GPU. The service has a P99 Time-To-First-Token (TTFT) SLO of 500ms. The model is a 40B parameter model, quantized to INT8 (40 GB total size). Your system uses continuous batching. To understand the system's limits, first calculate the baseline Time Per Output Token (TPOT) for a single, isolated user. Assume this is dominated by reading the model weights from High Bandwidth Memory. Given the H100's HBM bandwidth of 3.35 TB/s, what is the theoretical minimum TPOT?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often assume that a GPU's performance is defined purely by its TFLOPS. They calculate the time based on compute operations and get a sub-millisecond result, ignoring that the arithmetic intensity of LLM inference is often too low to saturate the compute units. The real bottleneck is the time it takes to shuttle the model's massive weights from HBM to the cores for every single token generated.

  **Realistic Solution:** The process is memory-bound. We can prove this by comparing the time for memory access vs. compute. The time to read the 40GB of weights is the dominant factor. The theoretical minimum time per output token (TPOT) is determined by dividing the model size by the memory bandwidth.

A ~12ms TPOT for a single user is very fast and by itself does not pose a risk to a 500ms TTFT SLO. However, this is the best-case scenario. Under load, queueing delays (time spent waiting for the GPU) become the main contributor to TTFT. If 20 requests arrive at once in a simple FIFO queue, the 20th user could wait `19 * 12ms = 228ms` before their request even starts processing, significantly eating into the 500ms budget. This highlights why managing queue depth and scheduling (as done in systems like vLLM) is critical.

  > **Napkin Math:** ### Parameters
- **Model Size:** 40B params × 1 byte/INT8 = 40 GB
- **H100 HBM Bandwidth:** 3.35 TB/s = 3350 GB/s

### Calculation
- **Formula:** `Time = Total Data / Bandwidth`
- **Substitution:** `Time = 40 GB / 3350 GB/s`
- **Result:** `Time ≈ 0.0119 seconds ≈ 12 ms`

### Conclusion
The theoretical minimum TPOT is approximately 12 ms, dictated entirely by memory bandwidth.

  > **Key Equation:** $\text{Latency} = \frac{\text{Model Size (Bytes)}}{\text{Memory Bandwidth (Bytes/sec)}}$

  > **Options:**
  > [ ] Less than 1 ms. The operation is compute-bound, limited by the H100's PetaFLOP-scale compute.
  > [ ] Approximately 95 ms. The 3.35 TB/s bandwidth is measured in Terabits, not TeraBytes, reducing effective bandwidth.
  > [x] Approximately 12 ms. The operation is memory-bound by the time it takes to read 40 GB of weights over the 3.35 TB/s HBM interface.
  > [ ] Around 300 ns. This is the fundamental latency of a single HBM3 memory access.

  📖 **Deep Dive:** [Inference and Serving](https://mlsysbook.ai/cloud/03_inference_and_serving.html)
  </details>
</details>





























































#### 🟢 L3
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

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The P99 Latency Spike</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are the on-call engineer for a new LLM-based chatbot service deployed on H100s. The service uses a simple static batching strategy with a batch size of 32 and a fixed 100ms timeout to collect requests. During a load test, you observe that while average Time to First Token (TTFT) is acceptable at ~120ms, the P99 TTFT spikes to over 500ms, violating your SLO. `nvidia-smi` shows the GPU is consistently busy. Diagnose the most likely cause of this high tail latency."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the hardware. Engineers often assume latency spikes are due to the GPU being underpowered, the model being too large, or network issues. They might suggest scaling up to more powerful GPUs (e.g., B200s) without addressing the underlying scheduling problem.

  **Realistic Solution:** The most likely cause is head-of-line blocking exacerbated by the static batching policy. A user request that arrives just after a batch has been dispatched to the GPU must wait for the *entire* 100ms timeout window to elapse, and *then* wait for the next full batch to be processed. Under load, most requests experience some queueing, but the unlucky 'tail' requests that just miss the window experience the maximum possible delay. This creates a bimodal latency distribution, where some users get fast responses and others get very slow ones, dragging up the P99. The solution is to implement continuous batching, which processes requests as they arrive and adds new requests to the running batch dynamically, eliminating the fixed timeout and drastically reducing queue times.

  > **Napkin Math:** Let's analyze the worst-case scenario (the P99 tail).
1. **Request Arrival:** A user request arrives at time `t = 1ms`, just after a full batch was sent for processing at `t = 0ms`.
2. **Queue Wait:** This request must wait in the queue. The server will wait up to `99ms` more for other requests to arrive to fill the next batch (the `100ms` static timeout).
3. **Processing Delay:** Assume a forward pass for a full batch of 32 takes `~50ms` on an H100.
4. **Total TTFT:** The TTFT for this unlucky request is `Wait Time + Processing Time`. So, `TTFT ≈ 99ms (queueing) + 50ms (batch processing) = 149ms`.
However, under load, the queue isn't empty. Little's Law tells us the number of users in the system `L` is arrival rate `λ` times wait time `W`. As `λ` (requests/sec) increases, the wait time `W` increases non-linearly, especially with fixed batch windows. The P99 latency spike to >500ms indicates severe queueing delay where requests are waiting for multiple batch cycles.

  > **Key Equation:** $\text{Little's Law: } L = \lambda W$

  > **Options:**
  > [ ] The H100's memory bandwidth is insufficient, causing delays when loading model weights for each batch.
  > [ ] The model is too large, and the raw computation time for the forward pass exceeds the latency budget.
  > [x] The static batching timeout is causing head-of-line blocking, leading to severe queueing delays for requests that just miss the batch window.
  > [ ] The InfiniBand network connecting the GPU servers is saturated, causing high latency for internode communication.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Throughput Trap</b> · <code>ttft-vs-tpot</code></summary>

- **Interviewer:** "Your team has optimized an LLM inference server for maximum throughput, achieving 5,000 tokens/sec aggregate on a single H100. The deployment uses large static batches to maximize GPU utilization. However, user feedback is poor, with complaints that the chatbot feels 'laggy' and 'unresponsive' on the first response. You profile the system and find that while the per-user Time Per Output Token (TPOT) is reasonable (~12ms at batch=64, since 64 tokens are generated in parallel each step), the Time To First Token (TTFT) can be very high due to queuing. Apply your knowledge of serving latency to explain this phenomenon."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Focusing only on throughput as the primary success metric. Engineers often equate high tokens/second with a good user experience. They fail to distinguish between the system's overall processing rate (TPOT-driven throughput) and the perceived responsiveness for an individual user (TTFT).

  **Realistic Solution:** This is a classic trade-off between throughput and latency. Large static batches are excellent for throughput because they maximize the arithmetic intensity of the computation, keeping the GPU cores busy. However, this comes at the cost of high TTFT. A request arriving at the server may have to wait a significant amount of time for the batch to fill before it is even processed. While the subsequent tokens (TPOT) are generated quickly as part of the efficient batch, the initial wait time dominates the user's perception of latency. The system is optimized for batch processing, not for single-user responsiveness. To fix this, one would need to switch to a system like continuous batching that can start processing requests immediately, improving TTFT at the potential cost of slightly lower overall throughput.

  > **Napkin Math:** 1. **Batching Policy:** Assume the server waits up to 80ms to form a large batch (e.g., 64 requests).
2. **Request Arrival:** A user sends a request at `t=1ms`. The server just started a new batching window.
3. **Wait Time:** The request sits in the queue for the full `79ms` waiting for the batch window to close.
4. **Processing Time:** Let's say processing this large batch takes `120ms`.
5. **TTFT vs TPOT:** The TTFT for this user is `79ms (wait) + 120ms (processing) = 199ms`. This is perceived as lag. However, once the batch starts, the server generates tokens for all 64 users in parallel. If it generates 2048 tokens in that 120ms across all users, the *average* TPOT is `120ms / (2048 tokens / 64 users) = 3.75ms/token`, which is very fast, but this speed is not what the user experiences initially.

  > **Key Equation:** $\text{TTFT} = T_{\text{queue}} + T_{\text{process_first_token}}$

  > **Options:**
  > [ ] The server has high TTFT because the H100 has low clock speed, making the first forward pass slow.
  > [ ] The high TTFT is due to the latency of loading the model from NVMe SSD into HBM for every new user.
  > [x] The system's large static batching policy creates high queueing delays, leading to high TTFT, even though the per-token generation speed (TPOT) is fast.
  > [ ] The Python Global Interpreter Lock (GIL) is preventing true parallelism, causing the first token to be delayed.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Memory Wall of Long Contexts</b> · <code>kv-cache-memory</code></summary>

- **Interviewer:** "You are running a multi-tenant LLM serving system on H100 GPUs (80 GB HBM3). The system uses continuous batching and performs well with short user prompts. However, when a few users start engaging in very long conversations (e.g., summarizing a document with 16k tokens), you receive out-of-memory (OOM) errors and the entire batch fails. Your manager asks why the system, which can handle 64 concurrent users with short prompts, is failing with just a few users with long contexts. Using the scaling rules, diagnose the problem."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming memory usage is only for model weights. Many engineers calculate the memory required for the model parameters (e.g., Llama-70B FP16 ≈ 140GB) and forget that the KV cache for the context window is often a much larger and more dynamic consumer of memory during inference.

  **Realistic Solution:** The problem is the massive memory consumption of the KV cache for long sequences. In a transformer, the key-value (KV) states for every token in the context window must be stored in GPU memory to generate the next token. This memory usage scales linearly with sequence length and batch size. While continuous batching is efficient, it doesn't eliminate the fundamental memory requirement of the KV cache. A few users with very long contexts can quickly exhaust the 80 GB of HBM on an H100, causing an OOM that kills the entire batch of users currently being processed. The system needs KV cache compression (like quantization), paged attention mechanisms, or a stricter limit on the maximum sequence length per user to manage this.

  > **Napkin Math:** Let's use the provided formula for KV cache size per token, simplified as `Bytes ≈ 2 * layers * d_model * 2`. For a Llama-70B class model, let's approximate this as `2 Bytes/param/token` is too simplistic, let's use the formula from the table: `2 × layers × heads × head_dim × 2 bytes` which is `2 * n_layers * d_model * 2 bytes`.
For Llama-70B: 80 layers, head_dim 128, 64 heads -> `d_model = 8192`.
1. **KV Cache per User (16k context):** `Memory = sequence_length × 2 × num_layers × d_model × 2 bytes` (the first 2 is for K and V, the last is for FP16). So, `16,384 tokens * 2 * 80 layers * 8192 * 2 bytes/float ≈ 42.9 GB`.
2. **System Failure:** Just TWO users with a 16k context would require `2 * 42.9 GB = 85.8 GB` for their KV caches alone.
3. **Conclusion:** This exceeds the H100's 80 GB HBM, before even accounting for model weights (~140GB, so this requires 2 GPUs) or activations for other users in the batch. This calculation clearly demonstrates how a few long-context users can create an OOM condition.

  > **Key Equation:** $\text{KV Cache} \approx B \times S \times 2 \times L \times D_{\text{model}} \times \text{bytes/param}$

  > **Options:**
  > [ ] The OOM is caused by the model weights being loaded multiple times for each user.
  > [x] The KV cache for users with long sequence lengths is consuming all available HBM3 memory.
  > [ ] The PCIe Gen5 bus is too slow to transfer the long prompts, causing a memory backlog.
  > [ ] The continuous batching algorithm has a memory leak that gets worse with longer sequences.

  📖 **Deep Dive:** [Training](https://mlsysbook.ai/vol1/training.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Continuous Batching Paradox</b> · <code>continuous-batching-tradeoffs</code></summary>

- **Interviewer:** "Your team is deciding whether to switch from static batching to continuous batching for your LLM service. A junior engineer argues against it, stating: 'Continuous batching is less efficient. By processing requests one-by-one as they arrive instead of in large, optimized batches, we will lower our GPU utilization and reduce our total throughput (tokens/sec).' You are the tech lead. Solve this dispute by demonstrating how continuous batching can actually *increase* system throughput."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Misunderstanding how GPU time is spent. The misconception is that continuous batching means processing one request at a time (batch size of 1). It's also easy to think that if you aren't running max-size batches, you are wasting GPU potential. This ignores the time the GPU spends *idle*.

  **Realistic Solution:** The junior engineer's concern is valid but misses a key point: GPU idle time. With static batching, the GPU is often idle, waiting for a batch to fill. This is especially true with a wide distribution of request arrival times. Continuous batching (or 'dynamic batching') eliminates this idle time. When a sequence in the current batch finishes, a new waiting sequence is immediately swapped in. This keeps the total number of users being processed by the GPU (the 'batch size') consistently high, leading to higher overall utilization. While a single 'iteration' of the batch may not be at the *maximum theoretical* size, the GPU is performing useful work more of the time. This reduction in idle time often outweighs the slightly lower efficiency of not *always* running a full batch, leading to higher effective throughput.

  > **Napkin Math:** Let's compare two scenarios over a 200ms period.
**Scenario A: Static Batching (Max batch 32, 100ms timeout)**
- `t=0-100ms`: Server waits. Let's say 10 requests arrive. The GPU is 100% idle.
- `t=100-150ms`: GPU processes the batch of 10. Let's say this takes 25ms (since it's not a full batch).
- `t=150-200ms`: GPU is idle again.
- **Result:** In 200ms, the GPU was busy for 25ms. **GPU Utilization: 12.5%**.

**Scenario B: Continuous Batching**
- `t=0ms`: 2 requests are waiting. Batch starts immediately.
- `t=0-50ms`: GPU processes the batch of 2. During this time, 8 more requests arrive.
- `t=50ms`: The first 2 requests finish. They are replaced by 2 of the waiting requests. The batch continues with 8 users.
- `t=50-200ms`: The GPU is continuously processing a batch, swapping users in as others finish. The batch size might fluctuate but it never drops to zero.
- **Result:** The GPU is busy for nearly the entire 200ms. **GPU Utilization: ~90-100%**. Higher utilization directly translates to more tokens processed over time.

  > **Key Equation:** $\eta_{\text{effective}} = \frac{T_{\text{compute}}}{T_{\text{compute}} + T_{\text{idle}}}$

  > **Options:**
  > [ ] Continuous batching uses speculative decoding to predict future tokens, which is faster.
  > [x] Continuous batching reduces GPU idle time by immediately swapping in new requests when old ones finish, leading to higher overall utilization and throughput.
  > [ ] Continuous batching works by quantizing the model to INT8 on the fly, making each computation faster.
  > [ ] Continuous batching uses more HBM bandwidth, which the H100 has in excess, allowing it to process more tokens.

  📖 **Deep Dive:** [Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Interactive API Latency Spike</b> · <code>ttft-vs-tpot</code></summary>

- **Interviewer:** "You are managing an interactive LLM API using a 70B model on H100 GPUs. Users are complaining about poor Time To First Token (TTFT), with P99 latency exceeding your 500ms SLA. The system uses static batching with a fixed timeout of 100ms to form batches. Monitoring shows that during peak load, GPU utilization is spiky (oscillating between 90% and 20%) and the request queue is consistently long. Diagnose the most likely cause for the high TTFT."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus only on throughput (TPOT) and assume high batching is always better. They might blame the model's inference speed or the network, ignoring that for interactive use-cases, TTFT is dominated by scheduling and queueing delays, a phenomenon known as head-of-line blocking.

  **Realistic Solution:** The most likely cause is head-of-line blocking induced by the static batching strategy. A new, short-sequence request arriving in the queue is forced to wait up to the full 100ms timeout for a batch to form. If it gets batched with a long-running request from another user, it must also wait for that entire generation to complete. This inflates the wait time (W) for every request. The spiky GPU utilization confirms this: the GPU processes a batch quickly, then waits idly while the next batch is slowly formed by the timeout mechanism. Switching to a continuous batching scheduler (like vLLM's) would solve this by processing requests on a token-by-token basis, eliminating idle time and head-of-line blocking.

  > **Napkin Math:** Let's use Little's Law. If the arrival rate (λ) is high, the queue length (L) grows. The total time a request spends (W) is the sum of wait time in the queue (W_q) and service time (W_s). With static batching, W_q includes up to the 100ms timeout *plus* the service time of the *longest* request in the batch. If a request for 1 token gets batched with a request for 500 tokens, its perceived latency isn't just its own generation time; it's the time for all 500 tokens. Continuous batching decouples requests, so a new request can start processing on the next GPU iteration, dramatically reducing W_q and thus improving TTFT.

  > **Key Equation:** W = W_q + W_s \quad (\text{Total Latency = Wait Time + Service Time})

  > **Options:**
  > [ ] The H100 GPU is compute-bound and cannot generate tokens fast enough for the user load.
  > [ ] The network latency between the API server and the inference server is the primary contributor to the 500ms delay.
  > [x] Head-of-line blocking from the static batching timeout is causing excessive queueing delays for new requests.
  > [ ] The HBM3 memory bandwidth on the H100 is insufficient, causing a bottleneck when loading model weights for each batch.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Code-Gen Throughput Ceiling</b> · <code>continuous-batching-throughput</code></summary>

- **Interviewer:** "Your team runs a code-generation service using a 34B parameter model. The primary goal is maximizing overall throughput (Tokens Per Second, or TPOT) as generations are often long. The service uses static batching with a batch size of 8. You observe that GPU utilization averages only 60%. You propose a switch to continuous batching. To justify the engineering effort, you need to demonstrate the potential throughput gain. Given a representative batch of 8 requests with the following output token lengths: [20, 50, 80, 100, 150, 200, 300, 1000], solve for the approximate throughput improvement."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common misconception is that total computation is just the sum of tokens, so batching strategy shouldn't affect theoretical max throughput. This ignores the massive waste from padding and idle time in static batching, where all requests in a batch are only as fast as the slowest member.

  **Realistic Solution:** The throughput improvement comes from eliminating the computational waste of static batching. In static batching, every request in the batch must wait for the longest request to finish. The total computational work is proportional to the size of the padded tensor. In continuous batching, as soon as a request finishes, its slot can be used for a new request, maximizing GPU utilization.

For the static batch, the total 'token-iterations' the GPU must perform is `1000 tokens * 8 requests = 8000`.
For continuous batching, the GPU only performs the necessary work, which is the sum of all tokens: `20+50+80+100+150+200+300+1000 = 1900` token-iterations.

The static method forces the GPU to do `8000 / 1900 ≈ 4.2×` more work for the same output. Therefore, switching to continuous batching could lead to a ~4.2x improvement in throughput by reclaiming the wasted compute cycles.

  > **Napkin Math:** 1. **Static Batching Cost:** The batch's execution time is dictated by the longest sequence. All 8 requests are padded to a length of 1000. Total GPU work is `8 requests * 1000 tokens/request = 8000` effective token generations.
2. **Continuous Batching Cost:** The GPU only works on the actual tokens needed for each request. Total GPU work is the sum of all tokens: `20 + 50 + 80 + 100 + 150 + 200 + 300 + 1000 = 1900` token generations.
3. **Improvement Ratio:** The ratio of work is `8000 / 1900 ≈ 4.21`. By eliminating the waste, continuous batching offers a theoretical throughput speedup of over 4x for this specific batch.

  > **Key Equation:** \text{Throughput Gain} = \frac{\text{Work}_{\text{static}}}{\text{Work}_{\text{continuous}}} = \frac{N \times L_{\max}}{\sum_{i=1}^{N} L_i}

  > **Options:**
  > [ ] There will be no improvement, as the total number of tokens to generate is the same regardless of batching strategy.
  > [ ] The improvement will be marginal (~10-20%) because the overhead of managing dynamic requests will negate most of the gains.
  > [x] A speedup of approximately 4.2x is possible by eliminating the idle GPU time and computation on padded tokens.
  > [ ] The system will likely crash due to KV cache memory fragmentation when handling so many variable-length sequences.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Unstable Translation Queue</b> · <code>queueing-theory-deadlines</code></summary>

- **Interviewer:** "You are designing a real-time translation service with a hard deadline: Time To First Token (TTFT) must be under 200ms. The system uses a 7B parameter model on an H100, which takes 20ms for prefill and 8ms per generated token. At peak, the service receives requests at an average rate (λ) of 10 requests/second. The average request needs 15 tokens. The current design uses a simple FIFO queue feeding a single inference worker. Use queueing theory to diagnose if this system can meet its deadline."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate the average service time and, if it's less than the deadline, assume the system is fine. This completely ignores queueing delay, which becomes non-linear and explosive as system utilization approaches 100%. They fail to check the fundamental stability condition: arrival rate must be less than service rate.

  **Realistic Solution:** The system is unstable and cannot meet its deadline. The critical issue is that the arrival rate (λ) is higher than the service rate (μ), which guarantees that the queue will grow infinitely long over time, causing unbounded latency.

First, we calculate the average service time for one request: `20ms (prefill) + 15 tokens * 8ms/token = 20 + 120 = 140ms`.
This means the system can process `1 / 0.140s ≈ 7.14` requests per second. This is the service rate (μ).

The arrival rate (λ) is 10 requests/second. Since `λ (10) > μ (7.14)`, the system utilization `ρ = λ/μ > 1`. An M/M/1 queue is only stable if ρ < 1. Because the queue is unstable, wait times will grow without bound, and the 200ms deadline will be missed consistently.

  > **Napkin Math:** 1. **Calculate Service Time (W_s):**
   `W_s = T_prefill + (N_tokens * T_per_token)`
   `W_s = 20ms + (15 * 8ms) = 140ms`

2. **Calculate Service Rate (μ):**
   `μ = 1 / W_s = 1 / 0.140s ≈ 7.14` requests/sec

3. **Calculate System Utilization (ρ):**
   `ρ = λ / μ = 10 req/s / 7.14 req/s ≈ 1.4`

4. **Diagnose Stability:**
   Since `ρ > 1`, the queue is unstable. The number of requests waiting will grow to infinity, and latency will become effectively infinite, making the 200ms deadline impossible to meet.

  > **Key Equation:** \rho = \frac{\lambda}{\mu} < 1 \quad (\text{Stability Condition})

  > **Options:**
  > [ ] The system is stable, as the average service time of 140ms is less than the 200ms deadline.
  > [x] The system is unstable because the arrival rate is greater than the service rate, leading to an infinitely growing queue.
  > [ ] The system can be fixed by increasing the batch size, which will increase the service rate μ.
  > [ ] The model is the bottleneck; switching to a smaller model would make the system stable.

  📖 **Deep Dive:** [ML Operations](https://mlsysbook.ai/vol1/ops.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Continuous Batching OOM</b> · <code>kv-cache-memory</code></summary>

- **Interviewer:** "Your team is serving a 70B parameter LLM using INT8 quantization on H100s (80GB HBM). You've implemented continuous batching with a paged KV cache, which has boosted throughput. However, under spiky traffic with long user contexts, the server is crashing with out-of-memory (OOM) errors. You need to apply napkin math to demonstrate to your team where the memory pressure is coming from. Assume the Llama-70B architecture (80 layers, 8 KV heads, 128-dim head) and that the KV cache is stored in FP16."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Many engineers underestimate the memory footprint of the KV cache, especially with long contexts. They focus on the model weights' size, which is static, and forget that the KV cache grows dynamically with every user and every generated token. They might incorrectly blame memory leaks or fragmentation without calculating the cache's fundamental memory requirement.

  **Realistic Solution:** The primary source of memory pressure is the KV cache. While the model weights have a large but fixed size, the KV cache's memory footprint scales linearly with the sequence length and the number of concurrent requests. With only 10GB of free memory after loading the weights, a few users with long contexts can easily exhaust all available HBM.

Calculation shows that a single request with a sequence length of 8192 tokens consumes ~2.68 GB of KV cache. With only 10GB of memory available for the cache, the server can only handle 3-4 such users simultaneously before OOMing. The 'spiky traffic with long contexts' is the exact workload that would trigger this memory exhaustion.

  > **Napkin Math:** 1. **Calculate Weight Memory:**
   Model weights occupy `70B parameters * 1 byte/param (INT8) = 70 GB`.

2. **Calculate Available Memory for KV Cache:**
   `Available Memory = Total HBM - Weight Memory = 80 GB - 70 GB = 10 GB`.

3. **Calculate KV Cache Size per Token:**
   Formula: `2 (K/V) * num_layers * num_kv_heads * head_dim * bytes_per_element`
   `Size per token = 2 * 80 layers * 8 heads * 128 dim * 2 bytes (FP16) = 327,680 bytes/token ≈ 0.328 MB/token`.

4. **Calculate KV Cache for a Single Long Request:**
   For a context length of 8192 tokens: `8192 tokens * 0.328 MB/token ≈ 2687 MB ≈ 2.68 GB`.

5. **Calculate Maximum Concurrent Users:**
   `Max Users = Available Memory / Memory_per_User = 10 GB / 2.68 GB/user ≈ 3.7 users`.

This shows the system cannot support more than 3 long-context users before running out of memory.

  > **Key Equation:** M_{KV} = N_{req} \times L_{seq} \times 2 \times N_{layers} \times D_{model}

  > **Options:**
  > [ ] The server has a memory leak, as the number of active requests is low when the OOMs occur.
  > [ ] The memory is becoming heavily fragmented by the paged KV cache allocator, leaving no large blocks free.
  > [x] The KV cache for just a few users with long sequences is consuming the entire 10GB of available HBM.
  > [ ] The intermediate activations from the model's forward pass are causing the memory to run out.

  📖 **Deep Dive:** [Training](https://mlsysbook.ai/vol1/training.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Interactive Playground Dilemma</b> · <code>ttft-memory-bound</code></summary>

- **Interviewer:** "You are the tech lead for a new user-facing LLM playground running on H100s. The product manager wants to guarantee a Time-To-First-Token (TTFT) of under 200ms for a 70B parameter model to ensure a snappy user experience. The model is fully pre-loaded into HBM. For this analysis, you can ignore network latency between the user and the datacenter. Can you apply your systems knowledge to tell the PM if this is a feasible promise?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often calculate TTFT using the model's FLOPs and the GPU's TFLOPS. This gives a deceptively small number (sub-millisecond) and ignores the fact that a single-token forward pass (prefill) for a batch size of 1 is entirely constrained by memory bandwidth, not compute.

  **Realistic Solution:** Yes, this is a feasible promise. For a pre-loaded model, the primary latency for the first token is the time it takes to complete a single forward pass. With a batch size of 1, this operation is memory-bound, not compute-bound. A simple way to estimate this is to calculate the time required to read the entire model's parameters from HBM. The H100's memory bandwidth is the limiting factor.

  > **Napkin Math:** 1. **Model Size:** A 70B parameter model using FP16 precision requires 70B × 2 bytes/param = 140 GB of memory.
2. **Hardware Spec:** An NVIDIA H100 has an HBM3 memory bandwidth of 3.35 TB/s.
3. **Calculate Time:** Time = Total Size / Bandwidth = 140 GB / 3.35 TB/s ≈ 41.8 ms.
4. **Conclusion:** This ~42ms represents a theoretical lower bound for the forward pass. Even with real-world overheads from kernel launches and non-sequential memory access, the actual TTFT will be comfortably under the 200ms target.

  > **Key Equation:** T_{\text{forward_pass}} \approx \frac{\text{Model Size}}{\text{Memory Bandwidth}}

  > **Options:**
  > [ ] No, it's impossible. Loading the 140GB model from the NVMe SSD before every request will take several seconds.
  > [x] Yes, it's feasible. The bottleneck is memory bandwidth, not compute. The time to stream the ~140GB model through the H100's 3.35 TB/s HBM is roughly 42ms, leaving ample room in the 200ms budget.
  > [ ] No, it's not feasible. A 70B model requires ~140 TFLOPs per token, and the H100 can't compute this in under 200ms.
  > [ ] Yes, it's trivial. An H100 performs ~1 PetaFLOP, so the compute time is microseconds. The latency will be dominated by the ~5µs InfiniBand transfer, so it will be almost instant.

  📖 **Deep Dive:** [Single Machine Performance](https://mlsysbook.ai/cloud/01_single_machine.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The High-Throughput API Crisis</b> · <code>continuous-batching-padding</code></summary>

- **Interviewer:** "You're running a high-throughput LLM inference API on H100s. During a traffic spike with varied sequence lengths, you diagnose a major problem: P99 latency skyrockets and Time Per Output Token (TPOT) is terrible, yet `nvidia-smi` shows GPU utilization is only hovering around 60-70%. What is the most likely cause of this poor performance despite the GPU not being fully utilized?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Seeing GPU utilization below 100% and assuming the bottleneck must be upstream (e.g., CPU, network). While this can be true, in this scenario, the moderate utilization combined with high latency points towards *inefficient* use of the GPU, not starvation.

  **Realistic Solution:** The system is using static batching. With varied sequence lengths, shorter sequences in a batch are padded to the length of the longest one. The GPU wastes a significant number of cycles performing useless computations on these padding tokens. This means the GPU is busy, but not on productive work, leading to low *effective* throughput. Shorter jobs get stuck waiting for long jobs to finish, causing tail latency to explode. The solution is to implement continuous batching, which processes and evicts sequences from the batch as they complete, maximizing useful computation.

  > **Napkin Math:** 1. **Scenario:** A batch contains two requests. Req A: 1000 tokens. Req B: 100 tokens.
2. **Static Batching:** Both sequences are padded to 1000 tokens. The total work for the GPU is `2 * 1000 = 2000` tokens.
3. **Waste Calculation:** The work for Req B only needed 100 tokens. The wasted work is `1000 - 100 = 900` tokens. The waste percentage is `900 / 2000 = 45%`.
4. **Impact:** The GPU spends nearly half its time on padding. Req B, which should have finished quickly, is now tied to the completion time of the much longer Req A, drastically increasing its perceived latency.

  > **Key Equation:** \text{Waste %} = \frac{\sum (L_{\text{max}} - L_i)}{N \times L_{\text{max}}}

  > **Options:**
  > [ ] The model's KV cache is too large, saturating HBM memory bandwidth.
  > [x] The system is using static batching, causing massive internal fragmentation and wasted computation on padding tokens, which lowers useful throughput.
  > [ ] The CPU is bottlenecked on tokenizing incoming requests, which is starving the GPU and preventing it from reaching 100% utilization.
  > [ ] The network is the bottleneck; requests are arriving too slowly to form large enough batches, leaving the GPU idle between batches.

  📖 **Deep Dive:** [Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Real-Time Ad Bidding SLA</b> · <code>queueing-theory-tail-latency</code></summary>

- **Interviewer:** "You are designing a cloud service for real-time ad bidding which must respond to 99.9% of requests within a 50ms SLA. Your inference model, running on a dedicated H100, has an average processing time of 30ms. However, due to feature variability, you observe that 5% of requests are 'complex' and take 80ms to process. Using queueing theory, determine if the current system can meet this strict SLA."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Looking only at the average processing time (30ms) and concluding that since it's well under the SLA deadline (50ms), the system is healthy. This completely ignores the cascading failure effect of tail latency in any system with queues and strict deadlines.

  **Realistic Solution:** No, the system cannot meet the SLA. While the average time is acceptable, the P95 latency of 80ms guarantees failure. Any request that takes 80ms will immediately miss its own 50ms deadline. More importantly, it occupies the server for an extra 30ms, forcing the next request to wait in a queue. This initial wait time is then added to the next request's processing time, likely causing it to miss its deadline as well. This creates a cascade of failures. Since 5% of requests trigger this failure mode, it's impossible to achieve a 99.9% success rate.

  > **Napkin Math:** 1. **Deadline:** 50ms.
2. **Slow Request:** An 80ms request arrives. It misses the deadline by 30ms.
3. **Queue Forms:** While the server is busy, other requests arrive. Let's say a 'fast' 30ms request arrives just after the slow one started. It must wait 80ms for the server to become free.
4. **Cascading Failure:** The total time for the 'fast' request becomes `80ms (wait) + 30ms (process) = 110ms`. It also misses the 50ms deadline catastrophically.
5. **Conclusion:** The system has a 5% chance of entering a state that causes multiple, consecutive deadline misses. This far exceeds the 0.1% failure rate allowed by the SLA.

  > **Key Equation:** T_{\text{total}} = T_{\text{wait}} + T_{\text{process}}

  > **Options:**
  > [ ] Yes. The average processing time of 30ms leaves a 20ms buffer, which is more than enough to absorb occasional slow requests.
  > [x] No. The 80ms P95 latency exceeds the 50ms SLA, which will cause a request queue to form and trigger cascading deadline misses, making the 99.9% SLA impossible.
  > [ ] Yes, but only if you upgrade to a B200 GPU to reduce the average processing time.
  > [ ] It's impossible to say without knowing the network RTT from the ad exchange.

  📖 **Deep Dive:** [Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Continuous Batching Trade-off</b> · <code>latency-throughput-tradeoff</code></summary>

- **Interviewer:** "Your team is tuning a continuous batching server (like vLLM or TGI) for a production LLM. An engineer proposes dramatically increasing the `max_batch_prefill_tokens` parameter, which governs how many prompt tokens from new requests are collected before being processed in one large batch. How does increasing this value from a small number (e.g., 2048) to a very large one (e.g., 16384) affect the system's average Time-To-First-Token (TTFT) and its overall throughput?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming that any change that increases throughput must be beneficial for all latency metrics. Engineers often miss the 'batch formation time'—the waiting period for a batch to fill up—which is a key component of TTFT.

  **Realistic Solution:** This change creates a classic latency-throughput trade-off. Increasing `max_batch_prefill_tokens` will almost certainly **increase overall system throughput** but will also **worsen (increase) the average TTFT**. The higher throughput is achieved because processing one massive prefill batch is far more GPU-efficient (higher arithmetic intensity, less kernel launch overhead) than processing many small ones. However, individual requests now have to wait longer in a queue until this very large batch is assembled, and this waiting time is a direct component of TTFT.

  > **Napkin Math:** 1. **Low `max_tokens` (2048):** Assume average prompt is 200 tokens. The server waits for ~10 requests to form a batch. If requests arrive every 50ms, the 10th request waits `9 * 50ms = 450ms` to be batched.
2. **High `max_tokens` (16384):** The server now waits for `16384 / 200 ≈ 82` requests. The 82nd request could wait up to `81 * 50ms = 4050ms` (4 seconds) just to be batched.
3. **Trade-off:** The 4-second wait time is a huge regression for TTFT. However, running one prefill on 16k tokens is much more efficient per-token than running 8 separate prefills on 2k tokens, thus increasing the total number of requests processed per minute (throughput).

  > **Key Equation:** TTFT = T_{\text{queue_wait}} + T_{\text{prefill}} + T_{\text{decode_1}}

  > **Options:**
  > [ ] It will improve TTFT but worsen throughput.
  > [ ] It will improve both TTFT and throughput by making the GPU more efficient.
  > [ ] It will worsen both TTFT and throughput due to increased overhead.
  > [x] It will worsen TTFT due to longer batch formation times, but improve overall throughput due to more efficient GPU utilization during prefill.

  📖 **Deep Dive:** [Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Throughput Trap</b> · <code>static-batching-vs-latency</code></summary>

- **Interviewer:** "You are leading the ML inference team for a new generative AI chatbot. Your dashboard shows that your H100 GPUs are only at 50% utilization. Your manager wants to increase throughput by doubling the static batch size from 32 to 64 to get utilization to 100%. However, shortly after deploying this change, you receive alerts that the P99 Time-To-First-Token (TTFT) has violated its 100ms SLA. Diagnose the most likely cause for this regression."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Believing that higher GPU utilization is always better. Engineers often focus on maximizing hardware throughput (tokens/sec) without considering the 'queuing delay' cost that large static batches impose on individual requests.

  **Realistic Solution:** The increased batch size directly increases the time a new request has to wait before it can even begin processing. This is due to head-of-line blocking. With static batching, all requests in a batch must wait for the *entire batch* to be assembled before computation starts. Doubling the batch size likely doubled the average time requests spend waiting in the queue, pushing the P99 latency beyond the SLA.

  > **Napkin Math:** Let's assume an average request arrival rate (λ) of 100 requests/sec.
- **Scenario A (Batch Size 32):** To fill a batch, we need to wait for 32 requests. The time to fill the batch window is 32 req / 100 req/s = 320ms. A request arriving at the start of this window waits ~320ms for the batch to fill. The average wait time is half of this, ~160ms. This queuing delay alone violates the 100ms SLA.
- **Scenario B (Batch Size 64):** The time to fill the batch window is now 64 req / 100 req/s = 640ms. The average wait time balloons to ~320ms. The change from 50% to 100% utilization came at the cost of a 2x increase in queuing delay, making the latency violation even worse.

  > **Key Equation:** L = \lambda W

  > **Options:**
  > [ ] The model's architecture is inefficient, causing kernel launch overhead to dominate at larger batch sizes.
  > [ ] The GPU's memory bandwidth (HBM) is saturated, as the larger batch requires moving more data for activations and weights.
  > [x] Increasing the static batch size introduced significant queuing delay (head-of-line blocking), causing new requests to wait much longer before processing begins.
  > [ ] The network is the bottleneck; doubling the batch size saturated the NIC's ability to receive requests and transmit tokens.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Continuous Batching Hiccup</b> · <code>continuous-batching-tail-latency</code></summary>

- **Interviewer:** "Your team runs a multi-tenant LLM inference service using continuous batching. You notice that P99 end-to-end latency is spiking, even at moderate load. The profiler shows that some requests with very short prompts and short generation lengths are getting stuck behind requests that generate very long sequences (e.g., summarizing a document). What is the most likely design flaw in your continuous batching scheduler?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the hardware (e.g., 'we need more HBM for the KV cache') or the model. While these can be factors, a common and subtle issue is a naive scheduler implementation in continuous batching that doesn't account for heterogeneous workloads.

  **Realistic Solution:** The scheduler is likely using a simple First-In, First-Out (FIFO) policy to add new requests to the running batch. When a long-running request (e.g., generating 4000 tokens) is active in a batch, it occupies a slot for a long time. Any new, short requests (e.g., generate 20 tokens) that arrive get added but cannot complete and be evicted until the long request is finished, as the batch iteration proceeds in lock-step. This is a form of head-of-line blocking *within* the batching mechanism itself. A better scheduler would allow finished sequences to be evicted and replaced, or even preempt long-running requests.

  > **Napkin Math:** Assume one H100 (989 TFLOPS), a 70B model (~140 GFLOPs/token), and a batch size of 8.
- **Request A (long):** Generate 4000 tokens. Compute time ≈ (4000 tokens * 140e9 FLOPs/token) / 989e12 FLOPs/sec ≈ 566ms.
- **Request B (short):** Arrives 10ms later. Generate 20 tokens. Ideal compute time ≈ (20 tokens * 140e9 FLOPs/token) / 989e12 FLOPs/sec ≈ 2.8ms.
- **Outcome:** With a naive FIFO batch scheduler, Request B is added to the batch but is 'stuck'. Its actual perceived latency becomes the remaining time for Request A, which is ~556ms, instead of its ideal 2.8ms. This single event dramatically inflates P99 latency.

  > **Key Equation:** \text{SRPT (Shortest Remaining Processing Time is optimal for mean response time)}

  > **Options:**
  > [ ] The KV cache is fragmented, forcing expensive memory reallocation mid-generation.
  > [x] The scheduler lacks preemption or a fair-queuing policy, causing short requests to be blocked by long-running requests already in the batch.
  > [ ] The Python GIL is causing contention in the request handling server, preventing the scheduler from adding new requests fast enough.
  > [ ] The system is thrashing HBM because the combined KV cache of all requests exceeds the 80 GB HBM capacity.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Live Caption Deadline</b> · <code>queueing-theory-real-time</code></summary>

- **Interviewer:** "You are designing a service that provides live audio transcriptions for video calls. The system has a hard real-time deadline: 99% of 10-second audio chunks must be transcribed and returned in under 500ms. Your model, running on a single H100, can process one 10s chunk in 250ms (service time). The service receives an average of 3 requests per second (arrival rate), following a Poisson process. Using M/M/c queuing theory, solve for the minimum number of H100 servers (c) needed to meet the P99 500ms deadline."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Only looking at the average utilization. Average utilization (ρ = λ / (cμ)) might look safe, but queuing theory shows that tail latency can be very high even at 50-70% utilization due to the random nature of request arrivals. Engineers often under-provision because they don't account for this variance.

  **Realistic Solution:** This is a capacity planning problem that can be modeled with M/M/c queues. We need to find the number of servers 'c' that keeps the tail of the response time distribution within our SLA.
1. **Define Rates:** Arrival Rate λ = 3 req/s. Service Rate per server μ = 1 / 0.250s = 4 req/s.
2. **Analyze c=1 Server:** The load ρ = λ / μ = 3/4 = 75%. For an M/M/1 queue, the average time a request spends in the system (wait + service) is W = 1 / (μ - λ) = 1 / (4-3) = 1 second (1000ms). Since the *average* response time already fails the 500ms P99 deadline, one server is not enough.
3. **Analyze c=2 Servers:** The total system service rate is cμ = 2 * 4 = 8 req/s. The load ρ = λ / (cμ) = 3 / 8 = 0.375. At this much lower load, the probability of a request having to wait in the queue drops dramatically. The response time will be dominated by the 250ms service time, and the long tail of queuing delay will be short enough to meet the 500ms P99 SLA.

  > **Napkin Math:** 1. **Establish Rates:** Arrival Rate (λ) = 3 req/s. Service Time = 250ms, so Service Rate per server (μ) = 1/0.25s = 4 req/s.
2. **Test One Server (c=1):** Load (ρ) = λ / μ = 3/4 = 0.75. This system is 75% utilized. Average response time W = 1 / (μ - λ) = 1 / (4 - 3) = 1s or 1000ms. The average response time is already 2x the P99 deadline. This is clearly insufficient.
3. **Test Two Servers (c=2):** Load (ρ) = λ / (c*μ) = 3 / (2 * 4) = 3/8 = 0.375. The system is only 37.5% utilized. At this low load, queues are short and infrequent. The response time for most requests will be very close to the service time of 250ms. The P99 response time will comfortably be under the 500ms deadline.

  > **Key Equation:** \text{For M/M/1 queue: } W = \frac{1}{\mu - \lambda}

  > **Options:**
  > [ ] One server is sufficient, as the average service time (250ms) is less than the deadline (500ms) and utilization is under 100%.
  > [x] Two servers are needed to keep system utilization low, ensuring that variance in arrival times doesn't create a long queue that violates the P99 deadline.
  > [ ] Three servers are needed to handle potential traffic spikes, matching one server per average incoming request per second.
  > [ ] The problem is unsolvable without knowing the exact distribution of arrivals; the M/M/c model is purely theoretical.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The TPOT Trade-off</b> · <code>tpot-economics-continuous-batching</code></summary>

- **Interviewer:** "You're optimizing an LLM inference service that handles requests with highly variable output lengths. You need to decide whether to stick with a simple static batching implementation (batch size 64) or invest in building a continuous batching system. For a representative workload, the static batch must pad all 64 requests to the longest sequence in the batch (512 tokens), while the average sequence length is only 100 tokens. Using a 70B parameter model on an H100, calculate the wasted computation in the static batching case and determine the approximate throughput improvement (in tokens/sec) of moving to continuous batching."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Underestimating the cost of padding. Engineers often look at batch size and model FLOPs but forget that in static batching, every token—real or padding—consumes compute. The waste can be enormous with variable-length sequences.

  **Realistic Solution:** The core of the problem is calculating the total FLOPs used vs. the useful FLOPs. The ratio reveals the efficiency and the potential for improvement.
1. **Static Batching:** All 64 requests are padded to 512 tokens, meaning the GPU processes 64 * 512 = 32,768 tokens. However, only 64 * 100 = 6,400 tokens were 'useful'. The computational waste is (32,768 - 6,400) / 32,768 ≈ 80.5%.
2. **Continuous Batching:** This method avoids padding, so the GPU only processes the 6,400 useful tokens to achieve the same work.
3. **Throughput Gain:** Since continuous batching does the same useful work with ~1/5th of the computation, it results in an approximately 5x increase in useful throughput (TPOT).

  > **Napkin Math:** 1. **Establish Specs:** Model: 70B params. Inference FLOPs: ~2 FLOPs/param/token. Total FLOPs per token: 2 * 70e9 = 140 GFLOPs. GPU: H100 @ 989 TFLOPS.
2. **Static Batch Analysis:**
   - Tokens processed per batch = 64 requests * 512 tokens/req = 32,768 tokens.
   - Useful tokens per batch = 64 requests * 100 tokens/req = 6,400 tokens.
   - Efficiency = 6,400 / 32,768 = 19.5%.
   - Time per batch = (32,768 tokens * 140e9 FLOPs/token) / 989e12 FLOPs/sec ≈ 4.63 ms.
   - Useful Throughput (TPOT) = 6,400 tokens / 4.63 ms ≈ 1.38 M tokens/sec.
3. **Continuous Batch Analysis:**
   - Time to process the same useful work = (6,400 tokens * 140e9 FLOPs/token) / 989e12 FLOPs/sec ≈ 0.91 ms.
   - Useful Throughput (TPOT) = 6,400 tokens / 0.91 ms ≈ 7.03 M tokens/sec.
4. **Improvement:** 7.03 / 1.38 ≈ 5.1x.

  > **Key Equation:** \text{Efficiency} = \frac{\sum_{i=1}^{N} \text{actual_len}_i}{N \times \max(\text{actual_len})}

  > **Options:**
  > [ ] The improvement is negligible (~10%) because kernel launch overhead dominates for short sequences.
  > [ ] The improvement is around 2x, as continuous batching allows better memory management of the KV cache.
  > [x] The improvement is approximately 5x, as static batching wastes over 80% of the computation on padding tokens.
  > [ ] Static batching is actually faster because it uses contiguous memory access, whereas continuous batching causes fragmentation.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Algorithmic Trading Deadline</b> · <code>static-vs-continuous-batching</code></summary>

- **Interviewer:** "You are an ML Systems Engineer at a hedge fund. A critical service uses a 7B parameter LLM running on an H100 GPU to summarize breaking news articles for an automated trading algorithm. The algorithm requires the summary within 500ms of the article's arrival to be effective. The service receives a steady 10 requests per second (λ=10). Your system currently uses static batching with a fixed batch size of 8. You observe that a significant percentage of trading opportunities are being missed. Monitoring shows P99 TTFT is ~1.2 seconds. The measured service time for a full batch of 8 (average prefill and generation) is 750ms. What is the most likely bottleneck causing the missed deadlines?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often blame a single component, like 'the network is slow' or 'the GPU isn't fast enough'. They fail to see that the system's *scheduling policy* (static batching) is the source of the inefficiency, creating artificial delays (head-of-line blocking) even when the hardware is powerful.

  **Realistic Solution:** The primary bottleneck is head-of-line blocking caused by static batching. The system is operating at very high utilization, which magnifies queuing delays. A request may have to wait for two reasons: 1) waiting for 7 other requests to arrive just to form a batch, and 2) waiting for the *entire* current batch to finish processing. With a service rate close to the arrival rate, this queue wait time explodes, pushing P99 latency far beyond the 500ms deadline. Continuous batching would resolve this by processing requests as they arrive and returning them as they complete, eliminating the need to wait for a full batch to form and finish.

  > **Napkin Math:** 1. **Calculate Service Rate (μ):** The system processes a batch of 8 requests in 750ms (0.75s). Therefore, the service rate is μ = 8 requests / 0.75s = 10.67 requests/sec.
2. **Calculate Utilization (ρ):** The arrival rate is given as λ = 10 requests/sec. Utilization is ρ = λ / μ = 10 / 10.67 ≈ 0.937, or 93.7%.
3. **Interpret Result:** A system operating at ~94% utilization is highly loaded. According to queueing theory, latency increases exponentially as utilization approaches 100%. A request that arrives just after a batch has started must wait the full 750ms for that batch to complete, plus any time spent waiting for other requests ahead of it in the queue. This massive queuing delay is the direct cause of the 1.2s P99 latency, which is far above the 500ms SLA.

  > **Key Equation:** $$ W = \frac{1}{\mu - \lambda} $$

  > **Options:**
  > [ ] The H100 GPU is too slow and needs to be upgraded to a B200.
  > [ ] Network latency between the application and the inference server is too high.
  > [x] Head-of-line blocking from static batching is causing extreme queuing delays at high utilization.
  > [ ] The 7B model is too large; it should be quantized to INT8 or replaced with a smaller model.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Throughput Collapse</b> · <code>queueing-theory-inference</code></summary>

- **Interviewer:** "You run a popular AI chatbot service using a 13B model on H100 GPUs. During peak hours, the arrival rate (λ) of new conversations hits 15 requests/second per server. Your server uses a simple FIFO queue and static batching with a batch size of 16. The measured service time for one full static batch (including prefill and generating an average of 100 tokens for all 16 users) is 1.2 seconds. Users are complaining of extreme delays, with some waiting over 10 seconds for a response. Using queueing theory, what is the fundamental state of this system?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to think about system load linearly, assuming that a 10% overload might cause a 10% increase in latency. Engineers often forget that once arrival rate exceeds service rate, the system becomes unstable and queue length (and thus latency) will grow without bound.

  **Realistic Solution:** The system is unstable because the arrival rate exceeds the service rate (λ > μ), leading to a utilization greater than 100%. In this state, the queue of waiting requests will grow infinitely over time, causing latency to skyrocket. This is known as throughput collapse. No amount of waiting will fix this; the system cannot keep up with the demand. The only solutions are to decrease the arrival rate per server (e.g., by adding more servers) or, more effectively, increase the service rate with optimizations like continuous batching.

  > **Napkin Math:** 1. **Define Arrival Rate (λ):** The problem states λ = 15 requests/sec.
2. **Calculate Service Rate (μ):** The system services a batch of 16 requests in 1.2 seconds. The service rate is therefore μ = 16 requests / 1.2 s = 13.33 requests/sec.
3. **Calculate Utilization (ρ):** Utilization is the ratio of arrival rate to service rate: ρ = λ / μ = 15 / 13.33 ≈ 1.125.
4. **Interpret Result:** A utilization of 1.125 means the system is 112.5% utilized. Since ρ > 1, the queue is unstable and will grow infinitely, leading to unbounded latency.

  > **Key Equation:** $$\rho = \frac{\lambda}{\mu}$$

  > **Options:**
  > [ ] The system is memory-bound due to the KV cache size for 16 users.
  > [ ] The system is operating in a stable but highly-loaded state (ρ ≈ 99%).
  > [ ] The network is saturated from streaming back responses for 16 users simultaneously.
  > [x] The system is unstable (ρ > 1) because the arrival rate is higher than the service rate.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Prefill-Decode Collision</b> · <code>continuous-batching-scheduling</code></summary>

- **Interviewer:** "Your team has successfully implemented continuous batching for your LLM inference service on H100s. Average latency has dropped dramatically. However, you notice a troubling pattern in your P99 TTFT metrics: sometimes, a very short user prompt (e.g., 'Hello') gets a TTFT of over 800ms. The system is not overloaded, and this happens when the short prompt arrives just after a request with a very long prompt (e.g., a 32k token document) begins processing. Why would a short prompt be forced to wait so long in a continuous batching system?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Thinking that 'continuous batching' magically allows all operations to be perfectly interleaved. Engineers often underestimate the duration and monolithic nature of the initial prefill/prompt processing stage for very long sequences, which can introduce a transient, but significant, head-of-line blocking effect even in advanced schedulers.

  **Realistic Solution:** The problem is the non-preemptive nature of the prompt processing (prefill) stage. When the 32k token document begins processing, the GPU is occupied with a single, large compute kernel to calculate its initial KV cache. This operation is compute-intensive and cannot be interrupted. The short prompt, despite being in a 'continuous' system, must wait in a queue for this long-running prefill to complete before its own (very fast) prefill can start. The subsequent decode steps are short and iterative, allowing for easy interleaving of different requests, but the initial prefill of a very long context is a major source of P99 latency.

  > **Napkin Math:** 1. **Estimate Prefill Compute:** For a large model, the compute for prefill is roughly proportional to the number of parameters and the sequence length. A simplified estimate is ~2 × P × N tokens. For a 13B model and 32k tokens: 2 × 13e9 × 32768 ≈ 8.5e14 FLOPs = 850 TFLOPs.
2. **Estimate Prefill Time on H100:** An H100 provides 989 TFLOPS (FP16). The time is Compute / Speed: T_prefill = 850 TFLOPs / 989 TFLOPS ≈ 0.86 seconds.
3. **Interpret Result:** The short prompt arrives and finds the GPU busy with an 860ms monolithic prefill task. It must wait for this entire duration before it can even begin its own processing. This wait time is the dominant factor in its TTFT, easily explaining the >800ms observation.

  > **Key Equation:** $$T_{\text{TTFT}} \approx T_{\text{queue_wait_for_long_prefill}} + T_{\text{own_prefill}}$$

  > **Options:**
  > [ ] The KV cache ran out of memory, forcing an eviction/recomputation cycle.
  > [ ] The continuous batching scheduler's overhead becomes too high with long sequences.
  > [x] The long-context prefill is a monolithic, non-preemptive operation that blocks new requests.
  > [ ] The inter-token latency (TPOT) for the long-context request slowed down the whole system.

  📖 **Deep Dive:** [Training](https://mlsysbook.ai/vol1/training.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Static Batching TTFT Penalty</b> · <code>ttft-static-batching</code></summary>

- **Interviewer:** "You are an ML Systems Engineer at a startup building a chatbot. The service runs a 7B parameter LLM on H100 GPUs. To handle a few users who submit long articles, your team configured the inference server for static batching with a fixed context length of 4096 tokens. However, the vast majority of users send short prompts (e.g., ~100 tokens). Users are complaining that the chatbot takes seconds to start typing its first word. Your PM argues the H100 is overkill if it's this slow. Using napkin math, diagnose the most likely cause of the high Time-To-First-Token (TTFT)."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the model size (7B is relatively small for an H100), the network latency, or suggesting a bigger batch size. These mistakes fail to identify that the core problem is a massive amount of wasted computation caused by the batching strategy itself. Increasing the batch size would only worsen the latency for individual requests.

  **Realistic Solution:** The primary cause of high TTFT is the use of static batching with a fixed, large sequence length. Every short prompt is padded to 4096 tokens, forcing the GPU to perform a massive prefill calculation on thousands of useless padding tokens. The time taken for this padded prefill directly translates to TTFT for every single request, regardless of its actual length. The system is wasting most of its computational power.

  > **Napkin Math:** The prefill phase, where the prompt is processed, is compute-bound.
1. **Required FLOPs for a padded prompt:** `C ≈ 2 * P * S`, where P=7B params and S=4096 tokens. `C ≈ 2 * 7e9 * 4096 ≈ 57.3e12 FLOPs` or 57.3 TFLOPs.
2. **Time to compute on H100:** An H100 provides ~989 TFLOPS (FP16). `T_prefill ≈ 57.3 TFLOPs / 989 TFLOPS ≈ 58ms`.
3. **Required FLOPs for an actual prompt:** For a 100-token prompt, `C ≈ 2 * 7e9 * 100 ≈ 1.4 TFLOPs`.
4. **Waste Ratio:** The system spends ~58ms on prefill compute when it should only take `1.4 / 989 ≈ 1.4ms`. The wasted computation is `(4096 - 100) / 4096 ≈ 97.5%`. This ~58ms delay, plus network and queuing time, is the direct cause of the slow TTFT.

  > **Key Equation:** $T_{\text{prefill}} \approx \frac{2 \times \text{Parameters} \times \text{SequenceLength}}{R_{\text{peak_flops}}}$

  > **Options:**
  > [ ] The network latency between the web server and the H100 is too high.
  > [ ] The 7B parameter model is too large, causing slow weight loading from VRAM.
  > [x] Static batching forces every short prompt to pay the full computational cost of the 4096-token context window.
  > [ ] The batch size is too small, which underutilizes the H100's Tensor Cores.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Throughput Collapse</b> · <code>queueing-theory-littles-law</code></summary>

- **Interviewer:** "You are managing an LLM serving endpoint that uses a 13B model on an H100 for document summarization. Your goal is to maximize throughput (summaries per minute). You observe that as you increase the server's batch size, throughput increases, but at a batch size of 64, P99 latency skyrockets and the effective throughput starts to *decrease*. An engineer suggests the GPU is running out of VRAM. Using queueing theory, demonstrate a more likely cause for this throughput collapse."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Immediately assuming a hardware limit like VRAM capacity (OOM) or memory bandwidth has been hit. While these are physical limits, a throughput collapse often happens before a hard OOM due to queueing dynamics. Blaming VRAM doesn't explain why a *lower* batch size worked better.

  **Realistic Solution:** This is a classic case of head-of-line blocking explained by queueing theory. Throughput is the number of requests completed over time. As the batch size increases, the time to process one batch (`T_batch`) also increases. A new request arriving just after a large batch has started must wait in the queue for the entire `T_batch` to finish before it can even be processed. When the request arrival rate (`λ`) approaches the batch service rate (`1/T_batch`), the queue length and wait times grow exponentially. The P99 latency skyrockets because some requests get very unlucky, and the 'goodput' (useful work) of the system collapses as it spends more time managing a massive queue.

  > **Napkin Math:** Let's apply Little's Law: `L = λ * W`, where `L` is the number of requests in the system, `λ` is the arrival rate, and `W` is the average time a request spends in the system (wait time + service time).
1. **Scenario A (Batch Size 32):** Let's say `T_batch` = 1 second. Service rate = 32 req/sec. If `λ` = 20 req/sec, the system is stable.
2. **Scenario B (Batch Size 64):** The batch is twice as big. Let's say `T_batch` = 1.8 seconds (due to efficiencies). Service rate = 64/1.8 ≈ 35.5 req/sec. The throughput looks higher.
3. **The Trap:** The wait time (`W`) is now dominated by `T_batch`. A request might wait up to 1.8 seconds before its batch even starts. If the arrival rate `λ` fluctuates to 40 req/sec, it now exceeds the batch service rate (1 batch per 1.8s). The queue (`L`) will grow infinitely because `λ > 1/T_batch`. The system becomes unstable, P99 latency explodes, and many requests may time out, causing effective throughput to drop.

  > **Key Equation:** $L = \lambda W \quad (\text{Little's Law})$

  > **Options:**
  > [ ] The GPU has run out of VRAM to store the KV caches for a batch of 64.
  > [x] The system is experiencing head-of-line blocking, where queue wait times explode as arrival rate nears the batch service time.
  > [ ] The H100's memory bandwidth (3.35 TB/s) is saturated, making larger batches less efficient.
  > [ ] The CPU is bottlenecking on preparing batches, starving the GPU.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The 60% Utilization Mystery</b> · <code>continuous-batching-utilization</code></summary>

- **Interviewer:** "Your team has successfully migrated an LLM inference service from static batching to a modern continuous batching framework. Under heavy, sustained load, you see a 2.5x increase in throughput. However, when you check `nvidia-smi`, you're puzzled to find that GPU utilization never exceeds ~60-70%, even though the request queue is full. An executive asks why you aren't 'fully using' the expensive H100s. What is the most likely reason for this phenomenon?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming that the GPU is bottlenecked by a hardware limitation like memory bandwidth or PCIe transfer speeds. While these can be factors, they don't explain why a system under full load with dynamic request patterns would be unable to keep the execution units busy 100% of the time. Another common mistake is to blame the Python GIL.

  **Realistic Solution:** The system is bottlenecked by the CPU-bound scheduling logic of the continuous batching engine itself. GPU utilization measures the percentage of time a kernel is active. In a continuous batching loop, there are two phases: 1) The GPU executes a forward pass on the current batch of requests. 2) The CPU-bound scheduler runs to check for finished sequences, free their KV cache blocks, check the queue for new requests, and allocate blocks for them before launching the next GPU kernel. If this scheduling phase on the CPU takes a non-trivial amount of time relative to the GPU execution phase, the GPU will sit idle during that period, leading to an average utilization below 100%.

  > **Napkin Math:** This is a conceptual test of systems thinking. Let's model the duty cycle of the GPU.
1. `T_gpu`: Time for the GPU to execute one forward pass on the dynamic batch. Let's say this is `3ms`.
2. `T_schedule`: Time for the CPU scheduler to manage memory, swap requests, and prepare the next batch. This can be complex. Let's say this takes `2ms`.
3. **Total Cycle Time:** The time for one full loop (GPU compute + CPU schedule) is `T_gpu + T_schedule = 3ms + 2ms = 5ms`.
4. **GPU Utilization:** The fraction of time the GPU was active during this cycle is `η_gpu = T_gpu / (T_gpu + T_schedule) = 3ms / 5ms = 0.60` or 60%.
This demonstrates how even under saturating load, the serial dependency on the CPU scheduler can cap the observable GPU utilization.

  > **Key Equation:** $\eta_{\text{GPU}} = \frac{T_{\text{GPU\_active}}}{T_{\text{GPU\_active}} + T_{\text{CPU\_schedule}}}$

  > **Options:**
  > [ ] The GPU is memory-bandwidth bound, so compute cores are idle waiting for HBM.
  > [ ] The Python GIL in the server is preventing the scheduler from running in parallel with GPU execution.
  > [ ] The request arrival rate is not high enough to fully saturate the server's capacity.
  > [x] The GPU is frequently idle, waiting for the CPU-bound scheduler to manage requests and memory between batches.

  📖 **Deep Dive:** [Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Reinforcement Learning Latency Stall</b> · <code>pcie-latency-overhead</code></summary>

- **Interviewer:** "You are designing a large-scale RL system. A CPU-based simulator generates 20,000 small observation tensors per second, which must be sent to a single GPU for a policy update. Your profiling shows that while the total data volume is very low (a few MB/s), end-to-end latency is unexpectedly high and GPU utilization is poor. A junior engineer suggests the GPU is too slow. Diagnose the most likely bottleneck by applying your knowledge of bus protocols."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to be 'bandwidth-blind'. Because the total size of the data is small, engineers often assume data movement cannot be the problem. They fail to account for the high *per-transaction* latency cost of a bus like PCIe. Every individual transfer, no matter how small, pays a fixed latency tax. When you have tens of thousands of tiny transfers, that tax adds up and dominates the timeline.

  **Realistic Solution:** The bottleneck is the cumulative latency overhead from thousands of individual PCIe transfers. While PCIe Gen5 has high bandwidth, each individual transfer has a base latency of ~1,000 nanoseconds (1 microsecond). When the CPU sends 20,000 separate tensors, it's initiating 20,000 separate transfers. The total time wasted just on the latency of these transfers is 20,000 * 1µs = 20ms. This means 20ms of every second is pure communication overhead before any useful data is even moved or computed. The correct solution is to batch the small tensors on the CPU into a single, larger tensor before sending it to the GPU. This amortizes the fixed per-transfer cost over a much larger data payload, paying the 1µs tax only once per batch instead of 20,000 times.

  > **Napkin Math:** 1. **Identify Per-Transfer Latency:** From the ML Latency Hierarchy, a single PCIe Gen5 transfer has a latency of ~1,000 ns (1 µs).
2. **Identify Transfer Frequency:** The system is sending 20,000 tensors per second, presumably as 20,000 separate transfers.
3. **Calculate Total Latency Overhead:** Total Overhead = Number of Transfers × Per-Transfer Latency.
   Total Overhead = 20,000 transfers/sec × 1 µs/transfer = 20,000 µs/sec.
4. **Convert to Milliseconds:** 20,000 µs = 20 ms. This means 20ms of every second (or 2% of total wall-clock time) is being consumed by latency overhead alone, starving the GPU.

  > **Key Equation:** $\text{Total Latency Overhead} = \sum_{i=1}^{N} \text{Latency}_i$

  > **Options:**
  > [ ] The GPU's HBM3 memory bandwidth is insufficient to handle the stream of tensors.
  > [ ] The CPU-to-GPU connection should be using NVLink for lower latency.
  > [x] The cumulative per-transaction latency of 20,000 individual PCIe transfers is creating a bottleneck.
  > [ ] The CPU is not powerful enough to generate 20,000 observations per second.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Translation API Latency Spike</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are an ML Systems Engineer for a new LLM-powered translation API running on H100 GPUs. A major client reports that while overall translation speed (TPOT) is good, the 'time to see the first translated word' (Time To First Token - TTFT) is unacceptably slow, with P99 TTFT spiking over 1.5 seconds and violating your 500ms SLA.

Your dashboard for their traffic shows:
- **Arrival Rate (λ):** 10 requests/sec
- **Batching Strategy:** Static
- **Static Batch Size:** 32
- **Batching Timeout:** 1000ms
- **GPU Utilization:** ~40%

Based on this data, what is the most likely cause of the high P99 TTFT?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the model architecture or the hardware itself. Engineers often see low GPU utilization and assume the model is too small or the GPU is oversized. They focus on the model's processing time per token, forgetting that in many real-world serving scenarios, the queuing and batching policy is the dominant factor in end-user latency.

  **Realistic Solution:** The primary cause is the 1000ms batching timeout. With an arrival rate of 10 requests/sec, the system never gathers enough requests to fill a batch of 32 within the timeout window. Instead, every batch is formed only after the 1-second timeout is hit. A request arriving at the beginning of a batching window must wait the full second for the timeout, plus a small processing time, before its first token is generated. This queuing delay is the direct cause of the high P99 TTFT.

The correct solution is to implement a more advanced scheduler, like continuous batching. This decouples batch formation from GPU execution, allowing new requests to be added to a running batch, which drastically reduces the initial wait time and improves TTFT without sacrificing throughput.

  > **Napkin Math:** 1. **Calculate time to fill a static batch:**
   - `Time to Fill = Batch Size / Arrival Rate`
   - `32 requests / 10 requests/sec = 3.2 seconds`

2. **Compare fill time to timeout:**
   - `Time to Fill (3.2s) > Batch Timeout (1.0s)`
   - This confirms the batch is *always* formed by the timeout, not by filling up.

3. **Estimate P99 TTFT:**
   - The worst-case wait time for a request is the full duration of the batching window.
   - `P99 Wait Time ≈ Batch Timeout`
   - `P99 TTFT = P99 Wait Time + Processing Time`
   - `P99 TTFT ≈ 1000ms + T_process`
   - Even with a near-instant processing time, the latency is already 2x the SLA. The reported 1.5s P99 TTFT is fully explained by this 1s wait time plus ~500ms for scheduling and processing overhead.

  > **Key Equation:** $\text{TTFT} = T_{\text{wait_queue}} + T_{\text{process}} + T_{\text{network}}$

  > **Options:**
  > [ ] The H100's compute is underutilized; switching to a smaller, cheaper GPU would be more efficient.
  > [ ] The network latency between the load balancer and the inference servers must be spiking to over 1 second.
  > [x] The 1000ms batching timeout is too high for the low arrival rate, causing requests to wait too long in the queue before processing.
  > [ ] The model is too large, causing slow 'cold starts' during token generation, which increases the time to the first token.

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The P99 Latency Explosion</b> · <code>llm-serving-queueing</code></summary>

- **Interviewer:** "You are the tech lead for a new AI code completion service running on H100 GPUs. The business requires a P99 Time-to-First-Token (TTFT) of less than 250ms to feel 'real-time'. Your service receives requests at a steady rate of 20 RPS. Your team implements a standard static batching strategy with a maximum batch size of 16. Monitoring shows that while average TTFT is a healthy 110ms, the P99 TTFT is spiking to over 900ms, violating the SLO. Your GPU utilization is high but not at 100% saturation. Applying queueing theory, diagnose the most likely cause of this high tail latency."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often blame the hardware (e.g., 'the GPU isn't fast enough') or the model itself. While a slow model increases service time, the massive gap between average and P99 latency points to a systemic issue in the serving *strategy*, not just the raw processing speed. Increasing the batch size would be a common but incorrect reaction, as it would actually make the P99 latency even worse.

  **Realistic Solution:** The primary cause is **Head-of-Line Blocking** inherent to static batching. In this system, an unlucky request arriving just after a batch has been dispatched must wait for the *entire next batch* to be collected before it can be processed. This collection time, not the GPU processing time, dominates the end-to-end latency for these tail-end requests. The solution is to move to a continuous batching (also called dynamic batching or iteration-level batching) scheduler. This allows new requests to be added to the currently running batch on-the-fly, decoupling the queue wait time from the batch formation time and dramatically reducing P99 latency.

  > **Napkin Math:** 1. **Characterize the workload:** The arrival rate (λ) is 20 RPS, so the average time between requests is 1 / 20s = 50ms.
2. **Calculate batch formation time:** With a max batch size of 16, the time to collect a full batch is 16 requests * 50ms/request = 800ms.
3. **Estimate service time:** Let's assume the H100 takes 100ms to process a full batch of 16 requests for the first token (`T_service`).
4. **Analyze the worst-case scenario (P99):** An unlucky request arrives right after a batch has been dispatched. It must wait in the queue for the next full batch to be collected. This wait time (`T_queue`) is approximately the batch formation time: ~800ms.
5. **Calculate total latency:** The total TTFT for this request is the sum of its wait time and the service time for its batch: `T_total = T_queue + T_service` ≈ 800ms + 100ms = 900ms. This matches the observed P99 latency and confirms head-of-line blocking as the bottleneck.

  > **Key Equation:** T_{latency} = T_{queue} + T_{service}

  > **Options:**
  > [ ] The batch size is too small, resulting in inefficient, low-throughput GPU kernels.
  > [ ] The H100 GPU is not powerful enough to handle 20 RPS, causing a persistent backlog of requests.
  > [x] Head-of-line blocking from the static batching strategy is creating extreme queueing delays for some requests.
  > [ ] The network connection to the NVMe drives used for swapping KV-cache is saturated.

  📖 **Deep Dive:** [Cloud: The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Latency Crisis</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are the lead ML Systems Engineer for a new AI chatbot startup. The service is built on a fleet of H100 GPUs and uses a simple static batching strategy with a batch size of 32. During a load test, you observe that GPU utilization is consistently high at ~95%, but user complaints about slow initial response times are flooding in. Monitoring reveals that the P99 Time to First Token (TTFT) is over 4 seconds, far exceeding the 500ms target. Your request workload is a mix: 90% are short Q&A prompts requiring ~50 new tokens, and 10% are long-form generation prompts requiring ~800 new tokens. What is the most likely cause of high TTFT despite the high GPU utilization?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often see high GPU utilization and assume the GPU itself is the bottleneck, leading them to request more powerful hardware. They fail to see that high utilization can be 'bad' utilization, where hardware is busy but not making progress on the most critical user-facing metrics due to inefficient scheduling.

  **Realistic Solution:** The root cause is head-of-line blocking due to the static batching strategy. In static batching, the entire batch of 32 requests must wait for the longest request to complete before the next batch can be processed. When a long request (800 tokens) enters a batch, it forces all other requests, including the short Q&A ones, to wait for its entire generation, catastrophically increasing their latency. The GPU remains 95% utilized because it's busy processing the long request, but this work isn't helping the 31 other users in the batch who are waiting. The correct solution is to implement continuous batching (or 'in-flight' batching), where the server iterates on a token-by-token basis across all active requests in the batch. Once a request finishes, its slot is immediately filled by a new request from the queue, eliminating head-of-line blocking and dramatically reducing average TTFT.

  > **Napkin Math:** Let's model one 'unlucky' static batch. It contains one long request (800 tokens) and 31 short requests (50 tokens).

1. **Estimate Per-Token Time:** A 70B model on an H100 might have a Time Per Output Token (TPOT) of around 5ms for a medium-sized batch.
2. **Calculate Batch Processing Time:** The entire batch's duration is dictated by the longest request. Time = 800 tokens * 5 ms/token = 4000 ms (4 seconds).
3. **Diagnose User Pain:** The 31 users who submitted short requests should have been served in 50 tokens * 5 ms/token = 250 ms. Instead, they are forced to wait the full 4 seconds for the long request to finish. Their perceived latency (TTFT) is ~16x higher than it should be.
4. **Conclusion:** The system is queueing requests and processing them in static batches, causing short jobs to be blocked by long ones. The GPU is busy (high utilization), but the queue of waiting users ($L$) grows because the average time in the system ($W$) is dominated by the slow minority of requests, per Little's Law.

  > **Key Equation:** $L = \lambda W$

  > **Options:**
  > [ ] The H100 GPU is the bottleneck; its compute capacity is insufficient to handle the request volume, requiring an upgrade to B200s.
  > [ ] The network connection to the GPU servers has excessive latency, adding seconds to the initial token response time.
  > [x] The static batching strategy is causing head-of-line blocking, where short requests are stuck waiting for long requests in the same batch to complete.
  > [ ] The model's KV-cache is too large, causing excessive memory swapping to HBM which increases latency for all requests.

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot's Awkward Silence</b> · <code>llm-inference-latency</code></summary>

- **Interviewer:** "You are the Staff ML Engineer for a popular AI code assistant. Users are complaining that the service 'thinks for too long' before starting to generate a suggestion. Your dashboard shows conflicting signals: GPU utilization is high (>90%), and overall token throughput (TPOT) is excellent. However, the P99 Time-To-First-Token (TTFT) is over 800ms, violating the product team's 100ms real-time deadline. The service runs a 70B parameter LLM on H100 GPUs. The serving engine uses static batching, waiting up to 500ms to collect a full batch of requests before running inference. Based on this data, diagnose the most likely cause of the high TTFT."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus exclusively on throughput (TPOT) and GPU utilization. They see high utilization and assume the system is optimized. However, they fail to see that the batching strategy itself can introduce artificial latency that harms the user experience, even when the hardware is running at full speed. They misdiagnose the problem as a hardware capacity issue (needing more GPUs) or a model issue (prefill is too slow) instead of a queueing logic issue.

  **Realistic Solution:** The root cause is the long, fixed 500ms timeout for static batching. This strategy prioritizes throughput by creating large batches, but it does so at the direct expense of per-request latency. A request arriving just after a batch has been dispatched must wait for nearly the full 500ms timeout, which dominates the TTFT. The correct solution is to switch to a continuous batching (or 'in-flight batching') scheduler. This allows the server to add new requests to the current batch on the fly, starting the prefill computation immediately without waiting for a timeout. This minimizes queue time and dramatically reduces TTFT, satisfying the real-time deadline while dynamically creating efficient batches.

  > **Napkin Math:** Let's analyze the timeline for a single request. The P99 TTFT is ~800ms, but the serving engine has a static batching window of 500ms.

1.  **Queue Wait Time:** With static batching, if a request arrives at an unlucky moment (just after a batch started), it could wait for almost the entire 500ms window. Assuming arrivals are random, the average wait time is `500ms / 2 = 250ms`. The P99 wait time will approach the full 500ms.
2.  **Prefill Compute Time:** A 70B model requires `~2 * 70B = 140` GFLOPs per token for inference. Let's assume an average prompt of 100 tokens. Total FLOPs = `100 * 140e9 = 1.4e13` FLOPs. An H100 provides ~989 TFLOPS (FP16). Compute time = `1.4e13 FLOPs / 989e12 FLOPs/s ≈ 14ms`.
3.  **Total TTFT:** The user-perceived latency is `Wait Time + Compute Time`. For an unlucky user (P99), this is approximately `500ms + 14ms ≈ 514ms`. This calculation alone shows the batching window is the dominant factor and easily explains why the P99 TTFT is over 500ms. The extra delay to 800ms likely comes from system overhead, network latency, and queueing delays under high load (Little's Law).

The key insight is that the `14ms` compute time is negligible compared to the `500ms` artificial wait time.

  > **Key Equation:** $\text{TTFT} = T_{\text{queue}} + T_{\text{prefill}} + T_{\text{network}}$

  > **Options:**
  > [ ] The 70B model's prefill computation is too slow for the H100 GPU, creating a compute bottleneck.
  > [ ] The system needs more H100 GPUs to handle the request volume and reduce queueing delays.
  > [x] The static batching window (500ms) forces requests to wait artificially, which is the primary contributor to TTFT.
  > [ ] Network latency between the user and the datacenter is the most likely cause for the >800ms delay.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Unstable Chatbot Queue</b> · <code>llm-serving-queueing</code></summary>

- **Interviewer:** "You are the ML systems engineer for a new AI chatbot service running on H100 GPUs. Users are complaining that the bot feels sluggish, and your metrics confirm that the average Time-To-First-Token (TTFT) is over 800ms, violating your 500ms P99 SLO. Your service receives approximately 10 requests per second (RPS). The team is using a static batching strategy with a fixed timeout of 400ms to group requests. Your model's prefill stage for a full batch takes about 450ms. Using these numbers, diagnose the most likely root cause of the excessive TTFT."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often fixate on a single component, like the GPU speed or model size, without analyzing the system's dynamics. They might say 'the model is too slow' or 'the GPU isn't powerful enough.' While these are contributing factors, they miss the actual failure mode, which is the interaction between the workload (arrival rate) and the serving strategy (batching timeout + processing time), leading to an unstable system.

  **Realistic Solution:** The system is in an unstable queueing state. The service rate is lower than the arrival rate, causing the request queue to grow indefinitely, which results in unbounded wait times. This is a classic queueing theory failure. The static batching timeout adds directly to the service time for a batch, creating a cycle that is too long to handle the incoming request rate. The solution is to abandon static batching in favor of a continuous batching system (e.g., vLLM's PagedAttention) which decouples batch formation from a fixed time window, allowing the system to process requests as soon as the hardware is ready, maximizing utilization and stabilizing the queue.

  > **Napkin Math:** 1. **Analyze Arrival vs. Service Rate:** The core of the problem lies in whether the system can serve requests faster than they arrive.
2. **Calculate Time Between Arrivals:** With an arrival rate (λ) of 10 RPS, a new request arrives every `1 / 10 = 0.1` seconds or 100ms.
3. **Calculate Total Service Time per Batch:** The server waits a fixed `T_wait = 400ms` to form a batch. Then, it takes `T_process = 450ms` to process it. The total time occupied by one batch cycle is `T_cycle = T_wait + T_process = 400ms + 450ms = 850ms`.
4. **Calculate Max Service Rate (μ):** During the 400ms wait, `400ms / 100ms_per_request = 4` requests will have arrived to form the batch. The system spends 850ms to serve these 4 requests. Therefore, the maximum request service rate is `4 requests / 0.850 seconds ≈ 4.7 RPS`.
5. **Diagnose Instability:** The arrival rate `λ = 10 RPS` is more than double the maximum service rate `μ = 4.7 RPS`. Since λ > μ, the request queue is unstable and will grow infinitely, causing TTFT to skyrocket. The system cannot keep up.

  > **Key Equation:** $\rho = \frac{\lambda}{\mu} > 1 \implies \text{Unstable Queue}$

  > **Options:**
  > [ ] The model's 450ms processing time is too slow for the H100 GPU, indicating a compute bottleneck.
  > [ ] The 400ms batching timeout is too short, preventing the formation of larger, more efficient batches.
  > [x] The arrival rate (10 RPS) exceeds the system's maximum service rate (~4.7 RPS), causing an unstable and ever-growing request queue.
  > [ ] Network latency between the user and the datacenter is the primary contributor to the 800ms+ TTFT.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Timeout Crisis</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are an MLE on a Generative AI platform team, responsible for a real-time chatbot service built on a 13B parameter LLM. Your service has a strict P99 Time-To-First-Token (TTFT) SLO of 500ms. You observe that while average TTFT is a healthy 250ms, the P99 latency is hitting 800ms, causing user-facing timeouts. The service currently uses a static batching policy on H100 GPUs, where the server waits to collect a full batch of 32 requests before running inference. Monitoring shows a single forward pass for this full batch takes about 450ms, and GPU utilization is high. Given this data, diagnose the primary cause of the SLO violation."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often misdiagnose queuing problems as compute problems. They see high latency and immediately think 'the GPU is too slow' or 'the model is too big'. This leads them to propose expensive hardware upgrades (e.g., more GPUs) or model changes (quantization) when the root cause is an inefficient scheduling algorithm. The key indicator of a queuing bottleneck is a large gap between average and tail (P99) latency, which points to some requests waiting much longer than others.

  **Realistic Solution:** The root cause is **head-of-line blocking** from the static batching policy. Requests that arrive just after a 450ms batch-processing cycle has begun are forced to wait in a queue for that entire cycle to complete before their own batch can even start forming. This waiting time is the direct cause of the high P99 latency. The correct solution is to replace static batching with **continuous batching** (or dynamic batching). This approach allows the server to iterate on all currently active sequences in a single step, immediately adding new requests to the batch as they arrive. This eliminates the queue, drastically reducing the wait time for all users and bringing the P99 TTFT much closer to the average TTFT.

  > **Napkin Math:** The P99 latency is a direct result of queuing delay. In the worst-case scenario for static batching, a user's request arrives just after a full batch of 32 has started processing.
1. **Wait Time:** The request must wait for the current batch to finish. `Wait Time ≈ Batch Processing Time = 450ms`.
2. **Queuing Effect:** This 450ms wait is the primary contributor to the tail latency. The observed 800ms P99 is composed of this waiting period plus the time to form and process the next batch. The average user doesn't experience this wait, hence the low average TTFT of 250ms.
3. **Conclusion:** The large delta between P99 (800ms) and average (250ms) is characteristic of a queuing system with head-of-line blocking, not a raw compute bottleneck. Continuous batching directly targets this by removing the waiting period, making TTFT for a new request largely independent of other requests being processed.

  > **Key Equation:** $$ W_q = L_q / \lambda $$ (Little's Law for the queue, where high P99 latency implies a long queue wait time $W_q$)

  > **Options:**
  > [ ] The H100 GPU is not powerful enough. We should upgrade to B200s to reduce the batch processing time from 450ms.
  > [x] The static batching policy is causing head-of-line blocking. We should implement continuous batching to eliminate queuing delay.
  > [ ] The 13B model is too large. We should quantize the model to INT8 to decrease the per-batch inference time.
  > [ ] The issue is inefficient token generation. We should implement speculative decoding to improve Time Per Output Token (TPOT).

  📖 **Deep Dive:** [Cloud: Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot's Silent Wait</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are an ML Systems Engineer optimizing a real-time LLM chatbot service built on an H100 GPU. The service has a strict P99 Time-To-First-Token (TTFT) SLO of 500ms. Under a moderate, spiky load averaging 20 RPS, you observe that the P99 TTFT is spiking to ~800ms, yet `nvidia-smi` shows the GPU is only 60% utilized on average. Your serving stack uses a static batching strategy with a fixed batching timeout of 200ms. Given this data, use the provided metrics to diagnose the most likely cause of the SLO miss."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the GPU hardware or the model itself. Engineers often default to thinking 'high latency means the GPU is too slow' or 'the model is too big.' This leads them to suggest expensive hardware upgrades or premature optimization (like quantization) while ignoring the clear signal from the low GPU utilization metric that the bottleneck is elsewhere. They fail to see latency as a system-wide property, much of which can be pure wait time.

  **Realistic Solution:** The primary bottleneck is queueing delay introduced by the static batching scheduler. The fixed 200ms timeout forces requests that arrive early in the window to sit idle, accumulating latency before computation even begins. The 60% average GPU utilization is the key signal: the GPU is spending a significant fraction of its time waiting for the next batch window to close. The P99 latency spike to 800ms can be explained by a cascade of these delays under load. The correct architectural solution is to replace the static batcher with a continuous batching scheduler (as used in systems like vLLM or TGI), which dispatches a new batch as soon as the GPU is free, minimizing idle time and drastically reducing queueing latency.

  > **Napkin Math:** We can diagnose this by working backwards from the 60% utilization.

1.  **Define the system cycle:** The total time for one cycle is `T_cycle = T_compute + T_wait`. The serving stack uses a fixed wait time (`T_wait = 200ms`).
2.  **Calculate compute time from utilization:** GPU utilization is the ratio of compute time to total cycle time: `Utilization = T_compute / (T_compute + T_wait)`.
3.  **Solve for T_compute:**
    `0.60 = T_compute / (T_compute + 200ms)`
    `0.60 * (T_compute + 200ms) = T_compute`
    `0.60 * T_compute + 120ms = T_compute`
    `120ms = 0.40 * T_compute`
    `T_compute = 120ms / 0.40 = 300ms`.
    This means an average batch takes 300ms to process.
4.  **Model the P99 Latency Event:** The worst-case latency happens when a request arrives just after a batch has been dispatched, and the queue is already backed up from a load spike. The request experiences a cascade of waits:
    - **Wait 1 (Prior Batch):** The GPU is busy processing the previous batch (`~300ms`).
    - **Wait 2 (Batching Window):** The request must wait for its own batching window to close (`~200ms`).
    - **Wait 3 (Own Batch Execution):** The request is finally processed as part of the next batch (`~300ms`).
    - **Total P99 TTFT:** `300ms + 200ms + 300ms = 800ms`.
This matches the observed symptom exactly. The 200ms static timeout is a direct and significant contributor to the SLO miss.

  > **Key Equation:** L = \lambda W

  > **Options:**
  > [ ] The H100 GPU is not powerful enough for this load. We should upgrade to a B200 to reduce the per-batch compute time.
  > [ ] The model's prefill computation is the bottleneck. We should apply INT8 quantization to reduce the TFLOPs required.
  > [x] The fixed 200ms batching timeout is causing excessive queueing delay; requests wait idly instead of being processed.
  > [ ] The bottleneck is network I/O from fetching user data for each request, causing the serving process to block before batching.

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Overloaded Translator</b> · <code>llm-serving-bottlenecks</code></summary>

- **Interviewer:** "You are an ML Systems Engineer on a team launching a new real-time translation service using a Llama-3-8B model on H100 GPUs. The service has a P99 Time-to-First-Token (TTFT) target of < 300ms. During a load test with a static batching implementation, you observe the following metrics:

- Arrival Rate (λ): 12 requests/sec
- Static Batch Size: 32
- Measured P99 TPOT (Time Per Output Token): 50ms
- Measured P99 Prefill Latency: 40ms
- Average Tokens per Request: 80
- GPU Utilization: ~65%
- Observed P99 TTFT: ~1200ms (and climbing)

Diagnose the primary cause of the high and climbing Time-to-First-Token."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often see sub-100% GPU utilization and immediately rule out the GPU as a bottleneck. They may misattribute the latency to a single component like prefill or suggest a simple parameter tweak like reducing the batch size, without calculating the system's maximum theoretical throughput. This fails to identify that the scheduling method itself can starve the hardware and create an overload condition.

  **Realistic Solution:** The system is overloaded because the incoming request rate (λ = 12 RPS) exceeds the system's maximum possible throughput (μ). The reason for the low throughput is the inefficiency of static batching. With static batching, the entire batch is blocked until the longest sequence (in this case, 80 tokens) is complete. This leads to significant idle time on the GPU as shorter sequences finish early but their batch slots cannot be reused, hence the ~65% utilization. Because the system is unstable (λ > μ), the request queue grows indefinitely, causing the wait time (and thus TTFT) to climb continuously. The solution is to move to a more advanced scheduling strategy like continuous batching, which processes requests on a token-by-token basis and can immediately evict finished sequences to onboard new ones, dramatically improving GPU utilization and overall throughput.

  > **Napkin Math:** 1. **Calculate time to process one static batch:** The batch isn't finished until the last token for the longest request is generated.
   - `T_batch = T_prefill + (Avg_Tokens * T_per_token)`
   - `T_batch = 40ms + (80 tokens * 50ms/token) = 40ms + 4000ms = 4040ms ≈ 4s`

2. **Calculate maximum system throughput (μ):** This is the number of requests processed per unit time.
   - `μ = Batch_Size / T_batch`
   - `μ = 32 requests / 4s = 8 requests/sec`

3. **Compare arrival rate (λ) to throughput (μ):**
   - Arrival Rate `λ = 12 RPS`
   - Max Throughput `μ = 8 RPS`

4. **Conclusion:** Since λ > μ, the system is in an overload state. The request queue will grow infinitely, and observed latency will continue to increase as long as this condition holds.

  > **Key Equation:** $\text{Little's Law: } L = \lambda W \text{ (In an overloaded system where } \lambda > \mu \text{, W } \to \infty)$

  > **Options:**
  > [ ] The static batch size is too large, increasing per-batch latency. Reducing it to 8 would lower TTFT.
  > [ ] The H100 GPU is underpowered for this model. GPU utilization would be 100% if it were the bottleneck.
  > [x] The system is overloaded because its maximum throughput is lower than the arrival rate, causing the request queue to grow. Static batching is artificially depressing throughput.
  > [ ] The 40ms prefill latency is the primary bottleneck. Optimizing the data input path is the highest priority.

  📖 **Deep Dive:** [Cloud: The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Lag Spike</b> · <code>llm-inference-serving</code></summary>

- **Interviewer:** "You are the Staff ML Systems Engineer for a popular AI chatbot service running a 70B parameter model on H100 GPUs. Users are complaining that the service "feels laggy to start," though the token generation speed is fast once it begins.

Your dashboard shows:
- Peak arrival rate (λ): 20 QPS
- P99 Time-To-First-Token (TTFT): >1200ms
- P50 TTFT: 150ms
- Average Time Per Output Token (TPOT): 30ms/token
- GPU utilization: 95%
- The server uses static batching with a max batch size of 64.

Given this data, what is the primary cause of the high P99 TTFT?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often misdiagnose this as a hardware limitation, suggesting a GPU upgrade. However, with 95% utilization, the GPU is already saturated. The problem isn't the speed of computation itself, but the waiting time *before* computation begins. Another mistake is focusing only on the prefill stage, ignoring the massive gap between P50 and P99 TTFT which clearly points to a systemic queuing problem.

  **Realistic Solution:** The primary cause is head-of-line blocking from the static batching strategy in an overloaded system. A new request that arrives just after a large batch has started must wait in a queue for the *entire* batch of 64 requests to complete their generation. This queue time is the dominant factor in the P99 TTFT. The huge difference between the P50 and P99 TTFT is the classic signature of a long-tail latency problem caused by queuing. The correct solution is to replace static batching with continuous batching (or a similar technique like PagedAttention), which allows new requests to be added to the batch as soon as *any* existing request finishes, drastically reducing queue times.

  > **Napkin Math:** 1.  **Calculate System Service Rate (μ):**
    - The server processes requests in static batches of 64.
    - Let's assume an average generation length of 100 tokens per request.
    - Total generation time for one request = 100 tokens/req * 30 ms/token = 3000 ms = 3 seconds.
    - With static batching, the batch isn't finished until the last request is done. So, a full batch takes ~3 seconds to process (ignoring prefill for this calculation).
    - Service Rate (μ) = 64 requests / 3 seconds ≈ 21.3 req/s.
2.  **Calculate Traffic Intensity (ρ):**
    - Arrival Rate (λ) = 20 req/s.
    - Traffic Intensity (ρ) = λ / μ = 20 / 21.3 ≈ 0.94.
3.  **Diagnose the Queue:**
    - Even with ρ < 1, queueing delay grows exponentially as ρ approaches 1. At 94% intensity, wait times become highly variable and significant.
    - The P99 TTFT is `Wait_Time_P99 + Prefill_Time`. The P50 TTFT is low (~150ms) because many requests arrive when the queue is short and their wait time is near zero. The P99 TTFT is high (>1200ms) because those requests get stuck waiting for a full 3-second batch to complete before they can even start. This queue wait time (~1000ms+) dominates the metric.

  > **Key Equation:** $\rho = \frac{\lambda}{\mu}$  (Traffic Intensity)

  > **Options:**
  > [ ] The model's prefill computation is too slow. It needs to be optimized with kernel fusion.
  > [ ] The H100 GPUs are too slow and cannot handle the TPOT demand.
  > [x] The system is experiencing severe head-of-line blocking due to its static batching policy, causing long queue delays for incoming requests.
  > [ ] The network connection to the server is the bottleneck, delaying the arrival of user prompts.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot SLO Catastrophe</b> · <code>continuous-batching-queueing</code></summary>

- **Interviewer:** "You are an ML Systems Engineer for a new LLM chat service running on H100s. Your P99 TTFT (Time To First Token) SLO is 500ms. Users are reporting slow initial responses, and your SLO is being breached. You check your dashboards and see the following:

- GPU utilization is consistently high, around 90%.
- The average request arrival rate (λ) is 20 req/s.
- The average number of requests in the system (in queue + processing, L) is 100.
- The service uses static batching with a fixed batch size of 32.

Using this data, diagnose the most likely cause of the TTFT SLO breach."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Seeing high GPU utilization and assuming the system is efficient. High utilization can be 'bad' if the work being done is blocking higher-priority or faster tasks. Another common mistake is to blame the hardware's raw performance instead of the scheduling and batching logic that feeds it.

  **Realistic Solution:** The core problem is head-of-line blocking caused by the static batching strategy. While the GPU is busy, it's working on a batch that may contain a single, very long generation request. This forces dozens of other short, quick requests to wait in the queue until the *entire* batch completes. This unnecessarily inflates the TTFT for the queued requests, causing the SLO breach.

The correct solution is to implement continuous batching. This allows the server to iterate on the batch, swapping out completed requests with new ones from the queue without waiting for the slowest request to finish. This maintains high GPU utilization while drastically reducing queueing time for new requests and improving TTFT.

  > **Napkin Math:** We can apply Little's Law to find the average time a request spends in the system (W), which includes both wait time in the queue and processing time.

1.  **Identify variables from the prompt:**
    -   Arrival Rate (λ) = 20 req/s
    -   Average requests in system (L) = 100 requests

2.  **Apply Little's Law:** Calculate the average time a request spends in the system, `W`.
    -   `W = L / λ`

3.  **Calculate:**
    -   `W = 100 requests / 20 req/s = 5 seconds`

4.  **Conclusion:** The average request takes **5 seconds** from arrival to completion. Given a P99 TTFT SLO of 500ms (0.5 seconds), a 5-second average system time is a definitive indicator of a massive queueing delay. The wait time in the queue is the dominant factor blowing the SLO, a classic symptom of head-of-line blocking in static batching systems.

  > **Key Equation:** W = L / \lambda \quad (\text{Little's Law})

  > **Options:**
  > [ ] The H100s are underpowered for this workload; the high utilization proves they can't keep up with the request volume.
  > [ ] The InfiniBand network latency is adding too much overhead, causing requests to miss their deadline.
  > [x] Head-of-line blocking from static batching is causing massive queueing delays, and the average request wait time is 10x the SLO.
  > [ ] GPU utilization is too high, leading to thermal throttling. We should reduce the batch size to give the GPU recovery time.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Latency Crisis</b> · <code>llm-inference-serving</code></summary>

- **Interviewer:** "You are the ML systems engineer for a real-time chatbot service running on an H100 GPU. The service has a strict P99 Time-To-First-Token (TTFT) SLO of 150ms. The current system uses static batching with a fixed 100ms timeout to group incoming requests. The model's prefill (the forward pass to generate the first token) takes 40ms. Monitoring shows that average GPU utilization is only 40%, yet the P99 TTFT is hovering at 240ms, clearly violating the SLO. Your task is to diagnose the bottleneck and apply the most effective solution."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often fixate on GPU utilization or raw throughput (tokens/sec), assuming higher is always better. They may incorrectly attribute the high latency to the GPU being 'too slow' or an external factor like the network, because they see low average utilization. They fail to recognize that the batching strategy itself is adding a large, fixed latency penalty that is fundamentally incompatible with a low-latency, real-time service.

  **Realistic Solution:** The root cause is the static batching timeout. This strategy forces requests to wait up to 100ms for a batch to form, even if the GPU is idle. This fixed wait time creates a high latency floor that makes achieving a 150ms P99 SLO impossible. The low GPU utilization is a symptom of this problem: the GPU is often idle, waiting for the next batch to 'bake' for 100ms.

The correct solution is to switch to a continuous batching (also known as in-flight batching) scheduler. This approach processes requests from a queue as soon as the GPU is free, dynamically creating batches. It eliminates the fixed batching wait time, dramatically reducing TTFT for requests that arrive when the server is idle or lightly loaded. The latency becomes a function of queue depth and inference time, which, given the low utilization, will be much closer to the 40ms base inference time for most requests, easily meeting the 150ms SLO.

  > **Napkin Math:** Let's decompose the latency for a request. Total Latency `W = T_q + T_b + T_i`.
1.  **Inference Time (`T_i`):** Given as 40ms. This is the base processing time on the GPU.
2.  **Batching Wait Time (`T_b`):** With static batching, a request can wait up to the full timeout duration. So, `T_b` can be up to 100ms. This is a fixed, artificial delay.
3.  **Queue Wait Time (`T_q`):** This is the time spent waiting for a *previous* batch to finish processing. The 40% GPU utilization implies the GPU is idle 60% of the time, so on average, `T_q` should be low. However, P99 latency captures worst-case scenarios where requests bunch up.

A request that arrives at an idle server just after a batch has been sent must wait the full 100ms for its batch to form, plus the 40ms for inference. This single-request journey already takes 140ms. Any additional queueing due to request 'burstiness' will push the P99 latency far beyond this, explaining the observed 240ms. The 100ms static timeout is the dominant factor and makes the 150ms SLO mathematically impossible to meet reliably.

  > **Key Equation:** $W = T_q + T_b + T_i$

(Total Latency = Queue Wait Time + Batching Wait Time + Inference Time)

  > **Options:**
  > [ ] Increase the static batching timeout to 200ms. This will capture more requests per batch, increasing GPU utilization and overall throughput.
  > [ ] The H100 is not powerful enough. Upgrade to a B200 to reduce the 40ms inference time.
  > [ ] The problem is likely CPU preprocessing or network latency. The 40% GPU utilization proves the inference server itself is not the bottleneck.
  > [x] Replace the static batching scheduler with a continuous batching implementation. This eliminates the fixed batching timeout, directly reducing the primary contributor to P99 latency.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Translation API's Latency Crisis</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are an ML Systems Engineer responsible for a real-time translation API running on H100 GPUs. The service has a strict P99 Time-To-First-Token (TTFT) SLA of 250ms. Users are complaining about slow initial responses. Your dashboard shows the following stable metrics:

- **Workload:** An LLM handling translation requests.
- **Arrival Rate (λ):** 150 requests/second.
- **Batching Strategy:** Static batching with a fixed timeout of 200ms.
- **GPU Prefill Time:** The model takes 150ms to process a batch and generate the first token for all requests in it.

Using these numbers, diagnose the primary cause of the 250ms SLA violation."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus exclusively on the model's execution time (service time), assuming a faster GPU or a smaller model is the only solution. They forget that in a queued system, the time a request spends waiting *before* execution is often the dominant factor in end-to-end latency.

  **Realistic Solution:** The root cause is head-of-line blocking induced by the static batching strategy. A request's total time-to-first-token is its wait time in the queue plus the GPU's processing time. In the worst-case scenario, a request arrives just after a batch has been dispatched. It must wait the full 200ms timeout period before it can even be considered for processing. This wait time, when added to the 150ms prefill time, results in a P99 TTFT of 350ms, which violates the 250ms SLA. The high TPOT (throughput) is achieved at the expense of TTFT. The correct solution is to implement continuous batching (or 'dynamic batching'), which decouples batching from request arrival and can add incoming requests to an in-flight batch, dramatically reducing queueing delay.

  > **Napkin Math:** The total latency for the first token is the sum of the time spent waiting in the queue and the time spent in service (prefill).

1.  **Calculate worst-case wait time ($T_{wait}$):** With a static 200ms timeout, a request that arrives just after a batch has started must wait for the entire timeout duration for the next batch to be formed. Therefore, the maximum queueing time is 200ms.

2.  **Identify service time ($T_{service}$):** The problem states the GPU prefill time for a batch is 150ms.

3.  **Calculate worst-case TTFT:**
    $T_{TTFT} = T_{wait} + T_{service}$
    $T_{TTFT} = 200\text{ms} + 150\text{ms} = 350\text{ms}$

4.  **Compare to SLA:** The calculated worst-case TTFT of 350ms is significantly higher than the 250ms SLA, confirming the diagnosis. This long tail latency is caused directly by the queuing delay from the static batching timeout.

  > **Key Equation:** $T_{TTFT} = T_{wait} + T_{service}$

  > **Options:**
  > [ ] The H100's prefill time of 150ms is too slow for this workload. The model needs to be optimized or run on newer hardware like a B200.
  > [x] The system is experiencing head-of-line blocking. A request can wait in the queue for up to 200ms before processing even begins, pushing the total TTFT to 350ms.
  > [ ] The request arrival rate (150 req/s) is too high, overwhelming the system. The service needs more GPU replicas to handle the load.
  > [ ] Network latency between the user and the datacenter is the likely cause, adding 100-200ms of un-accounted-for delay to every request.

  📖 **Deep Dive:** [Serving Stack](https://mlsysbook.ai/vol2/serving)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Latency Crisis</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are the lead ML Systems Engineer at a fast-growing startup deploying a new chatbot powered by a 7B parameter LLM on H100 GPUs. User satisfaction is plummeting due to slow response times. Your service level agreement (SLA) requires a P99 Time-To-First-Token (TTFT) of less than 200ms.

Your current system uses a simple static batching strategy: it waits up to 100ms to collect a full batch of 32 requests before running inference. Your observability platform shows a steady ingress of 20 requests per second (RPS). You profile a single request and find that the prefill (prompt processing) takes 150ms on the H100.

Diagnose the most likely cause for violating the TTFT SLA and determine how to solve it."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often conflate throughput (Tokens-Per-Output-Time, TPOT) with latency (TTFT). They might try to increase the batch size to improve overall system efficiency, but this action actually makes the TTFT worse for any individual user by increasing the time they must wait in a queue before their prompt is even processed.

  **Realistic Solution:** The root cause is the static batching policy. A request that arrives just after a batch has been dispatched must wait the entire 100ms timeout window before it can even be considered for processing. This waiting time is pure queuing delay. The worst-case TTFT is at least this queuing delay plus the prefill time. Switching to continuous batching (or iterative batching) resolves this. Continuous batching systems process requests in micro-batches, adding new requests to the queue on-the-fly without a fixed waiting window. A new request only has to wait for the current micro-batch step to finish before its prompt processing begins, dramatically reducing the queuing delay and thus the TTFT.

  > **Napkin Math:** Under static batching, the worst-case scenario for a user is arriving right after a batch starts. Their total TTFT is the sum of the time they wait for a new batch to form and the time it takes to process that batch's prefill.

1.  **Queuing Delay**: The server waits up to **100ms** to form a batch. This is the guaranteed minimum wait time for the unluckiest user.
2.  **Prefill Time**: The time to process the prompt for the whole batch is **150ms**.
3.  **Worst-Case TTFT (Static)**: `Queuing Delay + Prefill Time` = `100ms + 150ms` = **250ms**.

This 250ms violates the 200ms P99 SLA. With continuous batching, the 100ms artificial queuing delay is eliminated. A new request only waits for the current iteration (a few milliseconds) to complete before it's added, making its TTFT dominated by the ~150ms prefill time, thus comfortably meeting the SLA.

  > **Key Equation:** $$W_q = \text{Time in Queue}$$

  > **Options:**
  > [ ] The H100's prefill time is the bottleneck. We should use a smaller model or upgrade to B200s to reduce the 150ms processing time.
  > [ ] The arrival rate of 20 RPS is too high for the system to handle, causing a queue backup. We need to add more H100 replicas to handle the load.
  > [x] The static batching window is the bottleneck. The 100ms timeout adds unacceptable queuing delay, and we must switch to continuous batching.
  > [ ] System throughput is too low. We should increase the static batch size from 32 to 64 to improve the H100's utilization and TPOT.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Latency Spike</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are an MLE on the AI Chatbot team, responsible for a real-time LLM inference service using a 7B parameter model on H100 GPUs. Users are complaining about 'laggy' responses. Your monitoring dashboard shows that while average Time-To-First-Token (TTFT) is acceptable at ~30ms, the P99 TTFT is over 200ms. Furthermore, `nvidia-smi` shows GPU utilization is spiky and averages only 40%, well below your 90% target.

You're using a simple static batching strategy with a fixed timeout of 50ms to collect incoming requests before sending them to the GPU. Given this information, diagnose the most likely cause of both the high P99 latency and the low GPU utilization."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus only on the raw model execution time (prefill and decode) and forget the queueing and scheduling time, which often dominates the end-to-end latency in a live service. They mistakenly blame the hardware's speed (e.g., 'we need faster GPUs') instead of the system's scheduling logic, which is the actual source of the bottleneck.

  **Realistic Solution:** The root cause is head-of-line blocking introduced by the static batching strategy. This approach forces requests that arrive early in the 50ms window to wait unnecessarily, directly causing high tail latency (P99). This fixed wait also leads to inefficient, small batches when the request rate is low or bursty, causing the low GPU utilization. The combination of high latency and low utilization is a classic symptom of a poorly designed scheduling/batching system that fails to adapt to a variable workload.

The correct solution is to switch to a continuous batching (or 'in-flight batching') system. This decouples the request arrival from the GPU processing loop. The GPU can always process a full, optimally-sized batch from the requests currently in the queue, maximizing utilization. New requests are added to the queue and will be included in the *next* iteration, minimizing their individual wait time. This simultaneously increases throughput and dramatically reduces P99 latency.

  > **Napkin Math:** Let's analyze the latency components. A 7B model has ~14 GB of weights (7B params × 2 bytes/param for FP16). The theoretical Time Per Output Token (TPOT), which is memory-bound, is the time to read these weights.
1. **TPOT (Decode Bound):** 14 GB / 3.35 TB/s (H100 HBM3 bandwidth) ≈ 4.2 ms. This is very fast and is not the source of the 200ms latency.
2. **Static Batching Wait Time:** A request arrives at t=1ms into a 50ms batching window. It must wait 49ms for the window to close. If the system is under load and a queue has formed, it might have to wait for *several* 50ms cycles.
3. **Queuing Theory (Little's Law):** The system's low utilization means its effective throughput (requests/sec) is low. As the arrival rate (λ) exceeds this low throughput, the queue length (L) grows. Per Little's Law ($L = \lambda W$), the wait time (W) for each request in the queue must increase proportionally. This explains why P99 latency can explode to 200ms+, as some requests are stuck waiting in a long queue created by the inefficient batching.

  > **Key Equation:** $L = \lambda W \quad \text{(Little's Law)}$

  > **Options:**
  > [ ] The H100's memory bandwidth is insufficient for the 7B model, making the memory-bound decode step (TPOT) the bottleneck.
  > [ ] The prefill computation for processing the input prompt is too slow, making the service compute-bound on the GPU.
  > [x] The static batching timeout creates head-of-line blocking and inefficient small batches, leading to high queueing delay and low GPU utilization.
  > [ ] Network latency for incoming requests is highly variable, and the serving system has no control over this external factor.

  📖 **Deep Dive:** [Cloud: The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Latency Crisis</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are an ML Systems Engineer at a chatbot startup, tasked with optimizing your LLM serving stack. Your service runs on an H100 GPU (989 TFLOPS FP16) and you estimate your model achieves 50% MFU. The model requires 14 GFLOPs per token generated.

Your current deployment uses a **Static Batching** policy with a batch size of 32 and a fixed timeout of 100ms. Monitoring shows your P99 Time-To-First-Token (TTFT) is approximately 105ms and overall GPU utilization is disappointingly low.

To improve the user experience, your team wants to switch to **Continuous Batching**. After the switch, you anticipate requests will be processed in iterations with an average batch size of 16.

**Question:** Demonstrate the primary cause of the high TTFT in the original system and solve for the new, expected P99 TTFT after switching to Continuous Batching."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Mistaking the GPU hardware or model architecture as the bottleneck when the serving *policy* is the root cause. Engineers often focus on per-iteration compute time (`T_service`) and forget that for interactive applications, the time spent waiting in a queue (`T_queue`) due to inefficient batching is usually the dominant factor in user-perceived latency.

  **Realistic Solution:** The correct answer is that the P99 TTFT will drop dramatically to the low single-digit milliseconds. The original 105ms TTFT is almost entirely composed of the 100ms static batching timeout (`T_queue`), which is incurred by any request that doesn't arrive in a full batch. The low GPU utilization confirms the GPU is sitting idle waiting for batches to fill.

By switching to Continuous Batching, this artificial waiting period is eliminated. A new request only waits for the current (and very short) generation step to complete before being added to the batch. The new latency is therefore dominated by the time for a single forward pass, which is on the order of a millisecond.

  > **Napkin Math:** 1.  **Diagnose Original State:** The observed P99 TTFT is ~105ms and the static batching timeout is 100ms. This structure (`TTFT ≈ Timeout + T_compute`) strongly implies `T_queue ≈ 100ms` is the bottleneck.

2.  **Verify Compute Time:** Calculate the effective FLOPS of the system. `Effective FLOPS = 989 TFLOPS × 50% MFU = 494.5 TFLOPS`.

3.  **Calculate Original Service Time:** For the static batch of 32, the time to compute one token for the entire batch is: `T_service_static = (Batch Size × FLOPs/token) / Effective FLOPS = (32 × 14×10^9) / (494.5×10^12) ≈ 0.9ms`.

4.  **Confirm Diagnosis:** The total time `T_queue + T_service = 100ms + 0.9ms ≈ 100.9ms`, which matches the observed P99 of ~105ms. The diagnosis is correct: the system is queue-bound.

5.  **Calculate New Service Time:** For continuous batching with an average batch size of 16: `T_service_continuous = (16 × 14×10^9) / (494.5×10^12) ≈ 0.45ms`.

6.  **Calculate New TTFT:** With continuous batching, the queuing delay is the time waiting for the current iteration to finish. In the P99 case, a user might wait for one full iteration. So, `New P99 TTFT ≈ T_service_continuous + T_service_continuous ≈ 0.45ms + 0.45ms = 0.9ms`. The result is in the low single-digit millisecond range.

  > **Key Equation:** TT_{total} = T_{queue} + T_{service}

  > **Options:**
  > [ ] Slightly worse, ~110ms. The smaller batch size has lower arithmetic intensity, reducing MFU and making each step slower, which dominates any queueing gains.
  > [ ] ~53ms. The average batch size is halved (32 → 16), so the system's throughput is halved, and thus latency must also be halved.
  > [ ] ~105ms. The GPU is the fundamental bottleneck. Serving policy doesn't change the time it takes to compute a token, so the TTFT will remain the same.
  > [x] ~1-3ms. The 100ms static batching timeout (T_queue) is eliminated. The new latency is simply the compute time of one or two generation steps, which is on the order of milliseconds.

  📖 **Deep Dive:** [Cloud: The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.md)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The P99 Latency Explosion</b> · <code>llm-serving-queueing</code></summary>

- **Interviewer:** "You are the systems engineer for a new LLM-powered chatbot. The service runs on a single H100 GPU and has a strict P99 Time-To-First-Token (TTFT) deadline of 150ms. During load testing, you observe a steady arrival rate of 100 requests per second (RPS). Your serving system uses static batching, where it waits up to 40ms to collect a batch of requests before sending them to the GPU. Your profiling shows that the prefill computation for any given batch takes a fixed 30ms. While the *average* TTFT is well within limits, you see the P99 TTFT spiking above 200ms, violating your SLO. Your manager asks you to diagnose the cause of this high tail latency."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often misdiagnose this as a hardware limitation, suggesting a faster GPU is needed. They focus only on the raw 30ms service time, failing to see that the system's *queueing dynamics*, not the hardware speed, are the source of the tail latency. Another common mistake is to suggest increasing the batching window to improve 'efficiency', which actually makes the problem worse.

  **Realistic Solution:** The correct diagnosis applies queueing theory. The system is not just processing requests; it's processing *batches*. The static batching policy creates a bottleneck where requests wait for a batch to form, and then the batches themselves wait for the GPU. The P99 latency explosion is a classic symptom of a queueing system approaching high utilization.

With continuous batching, the rigid 'batch formation' step is eliminated. As soon as the GPU finishes an iteration, it can pull new requests from the queue. This breaks the head-of-line blocking imposed by static batches and dramatically reduces the time requests spend waiting, directly addressing the P99 latency problem by keeping the GPU constantly fed without artificial delays.

  > **Napkin Math:** 1.  **Model the System:** This is an M/M/1 queue where the 'customers' are batches and the 'server' is the GPU.
2.  **Calculate Batch Arrival Rate (λ_batch):** A new batch is formed every 40ms. So, the rate at which batches arrive at the GPU is `1 / 40ms = 25 batches/sec`.
3.  **Identify Service Time (T_service):** The GPU takes 30ms to process one batch. `T_service = 30ms`.
4.  **Calculate System Utilization (ρ):** Utilization is the product of arrival rate and service time. `ρ = λ_batch * T_service = 25 batches/sec * 0.030 sec/batch = 0.75`. A utilization of 75% is high and prone to queueing.
5.  **Calculate Average Batch Wait Time (W_q):** Using the Pollaczek-Khinchine formula for M/G/1 queues, the average time a batch waits in the queue is `W_q = (ρ * T_service) / (1 - ρ) = (0.75 * 30ms) / (1 - 0.75) = 22.5ms / 0.25 = 90ms`.
6.  **Estimate P99 TTFT:** The total time for a request includes the maximum time it waits for a batch to form (40ms), plus the P99 time for the batch to get through the queue and be processed. In a high-utilization queue, P99 wait time can be many multiples of the average. A reasonable estimate is that P99 queue wait is > 3-4x the average wait. Total P99 TTFT ≈ `40ms (batching) + (4 * 90ms) (queueing) + 30ms (processing) ≈ 430ms`. This demonstrates why the 150ms SLO is being violated.

  > **Key Equation:** $\text{System Utilization } \rho = \lambda \times T_{\text{service}}$

  > **Options:**
  > [ ] The H100 GPU is too slow. The 30ms service time is the bottleneck, and upgrading to a faster GPU like the B200 is the only solution.
  > [ ] The 40ms batching window is inefficient. We should increase it to 80ms to form larger batches, which will improve throughput and lower latency.
  > [x] The system is acting as a high-utilization queue for *batches*, not requests. The combination of batch formation delay and queueing delay for the GPU resource is causing an exponential increase in P99 tail latency.
  > [ ] The incoming 100 RPS is saturating the server's network card before requests can even be batched, leading to packet loss and high latency. The GPU is not the problem.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Lag Catastrophe</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are the tech lead for an LLM-powered chatbot service running a 70B parameter model on H100 GPUs. The service has a strict P99 Time-To-First-Token (TTFT) SLO of 250ms. Your current system uses static batching with a timeout of 150ms to form batches. During peak load, you observe that while average TTFT is acceptable, P99 TTFT is spiking to over 800ms, and `nvidia-smi` shows GPU utilization is only 40%. Your PM is asking why users are complaining about lag when the expensive GPUs seem idle. Based on this data, diagnose the most likely cause of the P99 latency violation."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often misdiagnose the symptom of low GPU utilization. They assume the bottleneck is data input (e.g., network, PCIe) or that the hardware is simply too slow, leading them to propose expensive hardware changes. The real issue is often the scheduling algorithm causing the GPU to sit idle, waiting for a poorly constructed batch.

  **Realistic Solution:** The primary cause is head-of-line blocking, a classic problem with static batching. The 40% GPU utilization isn't a sign of a slow data pipe, but rather the GPU being starved for work. A new, short user request gets stuck waiting for two things: 1) the current, long-running batch to completely finish generating all its tokens, and 2) its own batch to fill up or hit the 150ms timeout. This combination creates a massive P99 latency tail. Switching to a continuous batching scheduler (like those in vLLM or TGI) would solve this. It decouples prefill from decode, allowing new requests to be processed and added to the KV cache while older requests are decoding. This eliminates head-of-line blocking, dramatically improves GPU utilization, and shrinks the P99 TTFT.

  > **Napkin Math:** Let's model the P99 worst-case scenario for a user request under the static batching system:
1.  **`T_wait_previous`**: A request arrives just after a long-running batch was dispatched. This batch's processing time becomes the user's primary wait time. Given the observed 800ms P99, let's assume this worst-case processing time is ~600ms.
2.  **`T_wait_own_batch`**: After the previous batch finishes, our user's request must wait in a new batching window for other requests to arrive. In the worst case, it waits for the full timeout: 150ms.
3.  **`T_prefill`**: Once the new batch is dispatched, the prompt must be processed to generate the initial KV cache state. This is a large, parallel operation that might take ~50ms.

**Worst-Case TTFT (Static Batching) ≈ `T_wait_previous` + `T_wait_own_batch` + `T_prefill` ≈ 600ms + 150ms + 50ms = 800ms.**

With **continuous batching**, the `T_wait_previous` is eliminated. A new request only waits for the next scheduling step (e.g., 5ms) to begin prefill, which runs concurrently with the decoding of other requests.

**New TTFT (Continuous Batching) ≈ `T_scheduling_interval` + `T_prefill` ≈ 5ms + 50ms = 55ms.** This is well within the 250ms SLO.

  > **Key Equation:** $\text{TTFT}_{static} \approx T_{wait\_previous} + T_{wait\_own} + T_{prefill}$

  > **Options:**
  > [ ] The H100's memory bandwidth is insufficient, causing a bottleneck when loading model weights for each batch. We should use tensor parallelism to split the model across multiple GPUs.
  > [ ] The static batching timeout is too short. We should increase it to 300ms to create larger, more efficient batches, which will increase the 40% GPU utilization.
  > [x] The system is experiencing head-of-line blocking due to static batching, where new requests are stuck waiting for long-running batches to complete. Switching to continuous batching would solve this.
  > [ ] The PCIe bus is saturated, preventing the CPU from feeding data to the H100 fast enough, which explains the low 40% utilization.

  📖 **Deep Dive:** [Cloud: Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot's Unresponsive Start</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are the ML Systems Engineer for a popular AI chatbot service running on H100s. The service has a strict P99 Time-To-First-Token (TTFT) SLO of 250ms. Users are complaining that the bot feels 'laggy' to start its response, even though the subsequent tokens appear quickly. Your dashboard shows high GPU utilization (~90%) and excellent throughput (tokens per second), but the P99 TTFT has spiked to ~500ms. Your current serving stack uses a static batching strategy with a batch size of 32 and a fixed 400ms timeout window. Given these symptoms, diagnose the most likely cause of the P99 TTFT violation."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often fixate on raw throughput (tokens/sec) and high hardware utilization as universal signs of health. They misattribute latency issues to the model being too slow or the hardware being too weak, failing to see that the queuing strategy itself is the bottleneck. They assume that if the GPU is busy, it's doing useful work, when in reality, the *wrong work* (processing requests that have already waited too long) is being done.

  **Realistic Solution:** The most likely cause is the static batching policy. The fixed 400ms timeout window means that a request arriving at the beginning of the window must wait the full 400ms before processing even begins, immediately violating the 250ms SLO. This queuing delay is the dominant factor in P99 TTFT. While good for throughput, static batching is poor for latency-sensitive applications. The solution is to switch to a continuous (or dynamic) batching scheduler. Continuous batching decouples the batching from time, iterating on the batch in-flight and adding new requests as soonp as they arrive. This minimizes the wait time (T_wait) and allows the system to start processing requests almost immediately, dramatically improving TTFT for interactive workloads.

  > **Napkin Math:** Let's analyze the worst-case latency for a request under the static batching policy.

1.  **Parameters:**
    *   Batching Timeout (`T_wait_max`): 400ms
    *   SLO (TTFT): 250ms

2.  **Worst-Case Wait Time:** The very first request to arrive in an empty queue (at time `t=0`) must wait until the 400ms timeout window expires before the batch is sent to the GPU. Therefore, its wait time is at least 400ms.

3.  **Processing Time (`T_process`):** Let's calculate how long the GPU takes to generate the first token for the batch. We'll use a Llama-70B model.
    *   Compute per token: `2 * 70B Params = 140 GFLOPs`
    *   Total compute for a batch of 32: `32 * 140 GFLOPs = 4.48 TFLOPs`
    *   Achieved H100 performance (assuming ~40% of peak): `~400 TFLOPS`
    *   `T_process` = `4.48 TFLOPs / 400 TFLOPS ≈ 11.2 ms`

4.  **Total TTFT (Worst Case):**
    *   `TTFT = T_wait + T_process`
    *   `TTFT = 400ms + 11.2ms = 411.2ms`

This calculation demonstrates that the fixed waiting period from the static batching policy single-handedly causes the 250ms SLO to be violated before the request even reaches the GPU. With continuous batching, `T_wait` approaches 0, making the TTFT dominated by the ~11ms processing time.

  > **Key Equation:** T_{\text{TTFT}} = T_{\text{wait}} + T_{\text{process}}

  > **Options:**
  > [ ] The Llama-70B model's per-token processing time is too high for the H100 GPU, and a smaller model is needed to meet the SLO.
  > [ ] The network connection between the load balancer and the inference servers is adding significant latency.
  > [x] The static batching window forces early-arriving requests to wait for the timeout, causing high queuing delay that violates the TTFT SLO.
  > [ ] The H100's memory bandwidth is saturated, causing delays in loading model weights for each batch.

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Translation Service Traffic Jam</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are the ML Systems Engineer for a new real-time translation API that uses a 13B parameter LLM hosted on H100 GPUs. Users are complaining about unpredictable freezes, where the service feels responsive one moment and hangs the next. Your dashboard shows that average Time To First Token (TTFT) is acceptable at ~110ms, but the P99 TTFT is spiking to over 500ms during peak traffic. The service uses static batching with a fixed timeout of 100ms to collect requests. GPU utilization is consistently high (90%+). Based on this data, diagnose the most likely cause of the P99 latency spikes."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often misdiagnose this as a raw throughput problem and try to solve it by either adding more hardware (which doesn't fix the scheduling issue) or by shrinking the batch size (which destroys throughput and makes the queue even longer). They fail to see that the core issue is head-of-line blocking caused by the static batching window.

  **Realistic Solution:** The most likely cause is head-of-line blocking due to the static 100ms batching timeout. A request arriving at the beginning of a window must wait the full 100ms for the batch to be dispatched, even if the GPU is free. Under load, this wait time cascades, causing a long queue to form and leading to high P99 latency. The correct solution is to replace static batching with continuous batching (also known as in-flight batching). This allows the system to add new requests to the currently running batch, decoupling the arrival of requests from the batch dispatch schedule and minimizing idle time and queue length.

  > **Napkin Math:** Let's analyze the worst-case scenario for a single request with static batching:
1. A user request arrives at T=1ms, just after the previous batch was dispatched.
2. The system must wait for the static window to close. Wait_batch = 100ms.
3. The model's prefill (first token computation) on an H100 for a batch is very fast, let's say T_service = 10ms.
4. Total TTFT = Wait_batch + T_service = 100ms + 10ms = 110ms. This matches the *average* case.
5. During a traffic spike, the queue builds up. A request might have to wait for the *current* batch to finish *and* wait for its *own* batching window: W_total = W_queue + W_batch + T_service. If the queue is long, W_queue can easily be several hundred milliseconds, explaining the 500ms+ P99 latency.

  > **Key Equation:** W_{total} = W_{queue} + W_{batch} + T_{service}

  > **Options:**
  > [ ] The model is too large for the H100, causing compute delays. We should use a smaller model or upgrade to B200s.
  > [x] The static batching timeout is causing head-of-line blocking. We should switch to a continuous batching strategy.
  > [ ] Network latency between the user and the server is fluctuating, causing the P99 spikes.
  > [ ] The batch size is too large. We should reduce the batch size to process individual requests faster.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Chatbot Latency Spike</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are an ML Systems Engineer diagnosing a performance issue in a new AI chatbot. The service uses a Llama 70B model served on H100 GPUs and has a strict P99 Time-To-First-Token (TTFT) SLA of 300ms. Your team chose a static batching strategy, collecting requests for up to 50ms before dispatching a batch to the GPU. During load testing with spiky user traffic, your dashboard shows that while average TTFT is acceptable (~200ms), the P99 TTFT balloons to over 800ms. During these spikes, GPU utilization is pegged at 100% and the incoming request queue grows rapidly. Based on this data, diagnose the most likely cause of the SLA violation."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often blame the most expensive component first. They'll claim the H100 GPU is too slow, the model is too big, or the network is the bottleneck. This ignores the fact that the system's queueing strategy, not just the raw hardware speed, governs tail latency.

  **Realistic Solution:** The root cause is head-of-line blocking induced by the static batching policy. Static batching optimizes for throughput by waiting to form a full batch, but this imposes a waiting 'tax' on every request. Under spiky load, a large number of requests can arrive just after a batch has been dispatched, forcing them to wait for the entire duration of the current batch's processing *plus* their own batch formation time. This delay cascades, causing the request queue to grow and P99 latency to explode. The correct solution is to switch to a continuous batching (or 'in-flight' batching) scheduler, which adds incoming requests to the currently running batch, minimizing queue time and decoupling TTFT from batch formation.

  > **Napkin Math:** Let's model the queue during a spike. Assume the Llama 70B prefill (to generate the first token) for a full batch takes ~200ms on an H100. The static batch timeout is 50ms.

1.  **A spike of 100 requests arrives at T=0.**
2.  **Batch 1 (requests #1-32):** The server waits 50ms for the batch to form. It starts processing at T=50ms. It finishes at T=50ms (wait) + 200ms (compute) = 250ms.
3.  **Batch 2 (requests #33-64):** This batch can only begin after Batch 1 is complete. It starts its own 50ms wait at T=250ms, beginning processing at T=300ms. It finishes at T=300ms + 200ms = 500ms.
4.  **Batch 3 (requests #65-96):** This batch can only begin after Batch 2 is complete. It starts waiting at T=500ms and begins processing at T=550ms.

- The user who sent request #65, despite arriving near T=0, doesn't even *begin* processing until T=550ms. Their TTFT is at least 550ms (queueing) + 200ms (compute) = 750ms. This is why the P99 latency is so high.

  > **Key Equation:** L = \lambda W

  > **Options:**
  > [ ] The H100's memory bandwidth is insufficient to handle the KV cache for a large batch, causing contention and delaying token generation.
  > [x] The static batching policy creates head-of-line blocking, causing extreme queueing delays under spiky traffic patterns.
  > [ ] The network connection between the load balancer and the inference server is saturated, preventing requests from reaching the server in time.
  > [ ] The Llama 70B model is too compute-intensive for the H100, and the ~200ms prefill time for a batch is the fundamental bottleneck.

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>













































#### 🔵 L4
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
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Shadow GPU Budget</b> · <code>deployment</code> <code>serving</code></summary>

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

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The P99 Latency Anomaly</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You're analyzing the performance of a new LLM serving system for a 70B parameter model on H100s. The team is using a basic static batching strategy. Your dashboards show that as request volume increases, the average latency per output token (TPOT) remains stable, but the P99 time-to-first-token (TTFT) has exploded, violating your SLO. Differentiate the system dynamics causing the stable TPOT from those causing the high P99 TTFT. Analyze the root cause of the TTFT explosion, referencing the interaction between your batching strategy and queueing dynamics."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Attributing the high P99 TTFT to a simple hardware bottleneck like being 'compute-bound' or 'memory-bound'. While the system is using hardware, the problem is a logical one introduced by the scheduling strategy. Another common mistake is confusing TTFT and TPOT, assuming that if one is good, the other must be.

  **Realistic Solution:** The root cause is head-of-line blocking inherent to static batching. TPOT is stable because once a batch starts generating, it's an efficient, parallel operation limited by the H100's memory bandwidth, and this throughput is consistent batch-over-batch. However, P99 TTFT represents the worst-case user experience. This user likely arrives just after a large batch has begun processing. They must first wait in the queue for that *entire* batch (prefill and all subsequent token generation) to complete. Then, they must wait for their *own* batch to fill and for its expensive, compute-heavy prefill operation to finish. This double-waiting period (waiting for the previous batch + waiting for your own prefill) is what causes the P99 TTFT to explode, even while the per-token generation speed (TPOT) of a running batch looks good. The system isn't necessarily overloaded; its scheduling is just inefficient for latency-sensitive users.

  > **Napkin Math:** Let's model the P99 TTFT. Assume a batch size of 8 and a batch processing time `T_batch` of 800ms (e.g., 200ms prefill + 600ms for token generation).
1. A user arrives just after a batch begins. They wait `T_queue` ≈ 800ms for the current batch to finish.
2. Their request then enters a new batch. Let's assume the batch fills quickly. This new batch must perform its own prefill. `T_prefill` ≈ 200ms.
3. The user's P99 TTFT is the sum of these waiting periods: `P99 TTFT ≈ T_queue + T_prefill`
4. `P99 TTFT ≈ 800ms + 200ms = 1000ms`.
This is a full second, likely violating a typical <500ms SLO. Meanwhile, the TPOT is determined by the generation phase, which might be `600ms / (8 users * 100 tokens/user)` = 0.75ms/token, which appears extremely fast and masks the severe entry latency.

  > **Key Equation:** $\text{TTFT}_{p99} \approx \text{T}_{\text{process}}(\text{Batch}_{n-1}) + \text{T}_{\text{prefill}}(\text{Batch}_n)$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Plateau</b> · <code>continuous-batching-bottlenecks</code></summary>

- **Interviewer:** "A team migrates their Llama-70B service from static batching to a continuous batching system (like PagedAttention) on H100 GPUs. They expected a 2-3x throughput increase based on papers, but they only observe a modest ~30% gain at their target latency. Profiling shows high KV cache utilization, but the GPU is not at 100% utilization. Analyze this outcome. What practical system constraint is preventing them from reaching the theoretical throughput gains? Differentiate the bottleneck solved by continuous batching from the new bottleneck it has exposed."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming continuous batching is a 'free' throughput win. Engineers often believe it solves all scaling issues, without realizing it trades one bottleneck (memory fragmentation) for another (often prefill compute or memory bandwidth). They might incorrectly diagnose it as a network issue or a bug in the implementation.

  **Realistic Solution:** Continuous batching brilliantly solves the problem of memory fragmentation by allowing requests of different lengths to be packed together efficiently. This is why KV cache utilization is high. However, it does not eliminate the fundamental costs of inference. The modest throughput gain and sub-100% GPU utilization suggest the system is now bottlenecked by the *prefill phase* for new, incoming requests. While the GPU is efficiently decoding tokens for the existing batch (a memory-bandwidth bound task), it must periodically switch to the compute-heavy prefill operation for new requests. If new requests arrive frequently, the system spends a significant fraction of its time in this compute-bound prefill state, which has a lower arithmetic intensity than the decode phase and can't always saturate the H100's massive compute resources. The 'plateau' occurs because they have traded a memory capacity/fragmentation bottleneck for a prefill compute bottleneck.

  > **Napkin Math:** 1. **Decode (Memory-Bound):** To generate one token for the batch, the GPU must stream the 70B model weights from HBM. Time is `Weight Size / BW`. `(70B params * 2 bytes/param) / 3.35 TB/s` = `140 GB / 3350 GB/s` ≈ 42ms. This is the theoretical time per token for the *entire batch*.
2. **Prefill (Compute-Bound):** A new request with 1024 prompt tokens arrives. Compute is `2 * Params * Tokens`. `2 * 70e9 * 1024` ≈ 1.4×10¹⁴ FLOPs ≈ 143 TFLOPs. On an H100 with ~989 TFLOPS, this takes `143 TFLOPs / 989 TFLOPS` ≈ 145ms.
3. **Analysis:** The system juggles these two states. If a new request arrives every 200ms, the system spends `145ms / 200ms` = ~73% of its time in the prefill state. The massive throughput gains of continuous batching are primarily in the decode phase, where many users' token generations are parallelized. If the system is constantly bogged down with expensive prefills, it can't spend enough time in the highly efficient decode state, thus limiting the overall throughput gain.

  > **Key Equation:** $T_\text{total} = \alpha \cdot T_\text{prefill} + (1-\alpha) \cdot T_\text{decode}$

  📖 **Deep Dive:** [Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The SLO vs. Throughput Squeeze</b> · <code>serving-queue-slos</code></summary>

- **Interviewer:** "You are designing an LLM serving system for a 13B model on H100 GPUs. Your workload consists of long conversations (average 2048 prompt tokens, 2048 completion tokens). The business requires a P99 TTFT of less than 400ms. Using a simple static batching strategy, analyze the trade-off between throughput (batch size) and this strict TTFT SLO. Calculate the maximum batch size you can support, and explain the implications for your system's throughput."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Focusing only on the inference time for a single request. This ignores the queuing dynamics that dominate P99 latency. Engineers often calculate the best-case TTFT (an empty queue) and assume they have plenty of headroom, failing to account for the 'wait for previous batch' time in a production system under load.

  **Realistic Solution:** The P99 TTFT is dominated by the time a request spends waiting for the previous batch to complete, plus the time for its own batch's prefill. To meet a strict 400ms SLO with long generations, the total processing time of a batch must be kept extremely short. This forces the system into using a very small batch size, which in turn cripples throughput.

The core tension is that `P99 TTFT ≈ T_process(Batch_N-1) + T_prefill(Batch_N)`. Since `T_process` includes a long, multi-second decoding phase, this value will almost always be larger than the SLO. The only way to satisfy the equation is to make `T_process` incredibly small. With long generation lengths, this can only be achieved by reducing the batch size to its absolute minimum (i.e., 1), which makes the GPU highly inefficient.

  > **Napkin Math:** 1. **Define the P99 TTFT equation:** `P99 TTFT ≈ T_process(B) + T_prefill(B)`. We need this to be `< 400ms`.
2. **Estimate `T_prefill(B)`:** For a 13B model and 2k tokens, compute is `2 * 13B * 2048` ≈ 53 TFLOPs. On an H100 (989 TFLOPS), this is `53 / 989` ≈ 54ms. For a small batch `B`, let's round this up to `T_prefill(B)` ≈ 60ms.
3. **Estimate `T_process(B)`:** `T_process(B) = T_prefill(B) + T_decode(B)`. `T_decode(B)` is the time to generate `B * 2048` tokens. H100s can generate ~3000 tokens/sec on 13B models. Let's assume this is for an optimal batch size. The throughput scales with batch size. Let's model token time as `T_token_time(B)`. A simple model could be `T_decode(B) = (B * 2048) / (1500 * B)` = ~1.36s. So `T_process(B) ≈ 60ms + 1360ms = 1420ms`.
4. **Calculate P99 TTFT:** `P99 TTFT ≈ 1420ms + 60ms = 1480ms`.
5. **Analysis:** This is nearly 4x the 400ms SLO. To meet the SLO, `T_process(B) + 60ms < 400ms`, meaning `T_process(B) < 340ms`. With a decode time over 1 second, this is impossible for any batch size `B >= 1`. This demonstrates that for this workload and SLO, static batching is fundamentally unworkable. The system cannot provide both long generation and low TTFT without a more advanced scheduling strategy like continuous batching.

  > **Key Equation:** $T_\text{process}(\text{Batch}) = T_\text{prefill}(\text{Batch}) + N_\text{tokens} \times T_\text{per_token}(\text{Batch})$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The ROI of Heterogeneity</b> · <code>continuous-batching-economics</code></summary>

- **Interviewer:** "Your company runs many LLM services on H100s with a bimodal workload: 95% of requests are short (100 prompt/100 completion tokens), but 5% are long-context 'power user' requests (4k prompt/4k completion). All services use static batching, causing inefficient partitioning of the expensive H100 fleet. A platform team proposes a 3-month project to build a central continuous batching service. Compare the performance of static vs. continuous batching for this specific workload. Calculate the potential throughput gain to build a business case for the project."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Averaging the workload. Engineers might calculate the average sequence length and assume the system behaves according to the average. This completely misses the point that in static batching, the *maximum* length in the batch dictates performance, and the tail wags the dog. The cost of heterogeneity is the core issue.

  **Realistic Solution:** The business case is extremely strong. Static batching is pathologically bad for heterogeneous workloads. When a single long-context (8k total tokens) request enters a batch, all other short-context requests (200 tokens) in that batch are padded to the 8k length. This creates massive memory and compute waste. The KV cache allocation is based on the 8k length for all requests, meaning a batch with just one power user can fit almost no one else. In practice, this leads to de-facto batch size 1 for all power users, leaving the GPU idle most of the time.

Continuous batching (with PagedAttention) solves this directly by eliminating internal fragmentation. Memory is allocated proportional to the *actual* length of each request. This allows the system to pack one or more power users alongside hundreds of short-running users, dramatically increasing hardware utilization and overall throughput. The project is not just an optimization; it's a fundamental enabler for efficiently serving the mixed workload on the same hardware fleet.

  > **Napkin Math:** 1. **Static Batching Waste:** Consider a batch of 8 containing one 8k-token power user and seven 200-token normal users. Due to padding, the system effectively processes eight 8k-token requests.
   - Useful compute FLOPs are proportional to `(1 * 8000) + (7 * 200) = 9400` tokens.
   - Wasted compute FLOPs are proportional to `(7 * 8000) - (7 * 200) = 47600` tokens.
   - **Waste Ratio:** `Wasted / Useful = 47600 / 9400` ≈ 5x. Over 80% of the compute is spent on padding.
2. **Continuous Batching Gain:** It eliminates this waste. The potential throughput gain is proportional to this waste elimination. A conservative estimate would be a 2-3x increase in effective throughput across the fleet.
3. **ROI Calculation:** If the current fleet is 100 H100s, a 2x throughput gain means you can serve the same traffic with only 50 H100s.
   - Hardware saved: 50 H100s.
   - H100 cost: ~$30,000.
   - CapEx savings: `50 * $30,000 = $1,500,000`.
   A 3-month engineering project is easily justified by millions of dollars in hardware savings, not to mention power and operational cost reductions.

  > **Key Equation:** $\text{Waste} \propto \sum_{i \in \text{batch}} (L_{\text{max}} - L_i)$

  📖 **Deep Dive:** [ML Operations](https://mlsysbook.ai/vol1/ops.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The SLO Squeeze: Interactive vs. Batch Throughput</b> · <code>queueing-theory-slo</code></summary>

- **Interviewer:** "You run a unified inference service for a 70B parameter model on a cluster of 8 H100 GPUs. Your traffic is mixed: 80% is low-latency interactive chat with a P99 Time-to-First-Token (TTFT) SLO of 500ms, and 20% is high-throughput batch summarization jobs. Your current system uses a single FIFO queue with continuous batching. You're meeting your batch throughput goals, but violating the chat TTFT SLO due to head-of-line blocking from large summarization requests. Analyze the fundamental trade-off between two alternative designs:

1.  **Static Partitioning:** Dedicate 6 GPUs to a low-latency chat queue and 2 GPUs to a batch queue.
2.  **Priority Scheduling:** Keep the unified 8-GPU cluster, but implement a priority-based scheduler that can preempt batch requests to service interactive ones.

Distinguish the failure modes and resource utilization implications of each approach."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming that simply splitting the cluster is optimal. This ignores the risk of idle resources and mis-provisioning. For example, if chat traffic lulls, the 6 GPUs in that partition sit idle, while the batch queue may be overloaded. Conversely, a priority scheduler is more complex to implement and can lead to starvation for low-priority tasks if not designed carefully.

  **Realistic Solution:** The core of the analysis is the trade-off between isolation and efficiency.

**1. Static Partitioning:** This design provides strong isolation. The chat service's latency is protected from the batch service. However, it's statically provisioned and inefficient. If chat traffic is low, its 6 dedicated GPUs are underutilized. If batch traffic spikes, its 2 GPUs might be insufficient, leading to a large backlog, even while chat GPUs are idle. It trades overall cluster utilization for latency predictability.

**2. Priority Scheduling:** This design is more efficient, as all 8 GPUs are part of a single fungible pool. It can absorb bursts in either traffic type by dynamically allocating resources. The key challenge is the implementation complexity of preemption. A naive priority queue could starve the batch jobs. A well-designed system would need preemption (pausing a batch forward pass, servicing a chat request, then resuming) and potentially aging or other mechanisms to guarantee some forward progress for the batch queue. It trades implementation complexity for higher hardware utilization and dynamic load handling.

The superior engineering solution is typically Priority Scheduling, as hardware is expensive and maximizing utilization is key, but it requires a much more sophisticated software layer to manage the queueing and preemption logic without violating either SLO or starving tasks.

  > **Napkin Math:** Let's model the head-of-line blocking.

**Assumptions:**
- An H100 with continuous batching can achieve a throughput of ~300 tokens/sec for a 70B model.
- An interactive request is small (e.g., generates 50 tokens).
- A batch summarization request is large (e.g., processes a 2000 token prompt and generates 500 tokens).
- Time per output token (TPOT) is ~3ms.
- Let's say a batch summarization forward pass (prefill+decode) for one iteration takes 50ms on the batch.

**FIFO Scenario:** An interactive request arrives right after a batch job starts processing. The batch job might hold the batch for, say, 10 iterations to generate 500 tokens. The wait time for the interactive request is `10 iterations * 50ms/iteration = 500ms`. This *alone* blows the 500ms P99 TTFT SLO, before even considering other queueing delays. This is classic head-of-line blocking.

**Partitioning Scenario:** The 6-GPU chat cluster is isolated. Its queue length is determined only by chat arrivals (λ_chat). Using Little's Law, `L = λW`, the wait time `W` is kept low. The 2-GPU batch cluster handles its own queue. Utilization on the chat cluster might only be 50% on average, meaning 3 GPUs worth of capacity is wasted.

**Priority Scenario:** The batch job is running. The interactive request arrives, and the scheduler preempts. It pauses the batch job's state, runs the interactive request (e.g., TTFT of 40ms), and then resumes the batch job. The interactive user is happy. The batch job's total time is extended by only 40ms, which is negligible for a long-running job. The cluster utilization remains high.

  > **Key Equation:** $\text{Little's Law: } L = \lambda W \text{ (Average items in system = Arrival Rate × Average Time in System)}$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Tail Latency Paradox</b> · <code>continuous-batching-latency</code></summary>

- **Interviewer:** "Your LLM inference service uses a state-of-the-art continuous batching scheduler on H100s. Your average Time-to-First-Token (TTFT) is excellent, around 150ms. However, your P99 TTFT metric is failing its 500ms SLO, with frequent spikes into the 1.5-2 second range. Your manager suspects a hardware or network issue is causing intermittent stalls. You, however, notice in the logs that these latency spikes correlate with the arrival of requests with very long prompts (e.g., 4000+ tokens for summarization) mixed in with normal, short-prompt chat requests. Differentiate the underlying mechanism of how a continuous batching scheduler's prefill phase can create this tail latency from a potential hardware fault. Examine the relationship between the prefill operation and the decode operation in the batching loop."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the hardware (like the manager) or assuming continuous batching is 'magic'. A common misconception is that continuous batching completely isolates requests. While it's a huge improvement over static batching, the core scheduling loop is still a shared resource, and a single monolithic operation can become a bottleneck.

  **Realistic Solution:** This is not a hardware fault, but a fundamental characteristic of how Transformer inference and batching schedulers work. The process is two-phased:

1.  **Prefill/Prompt Processing:** The system takes the initial prompt tokens and processes them in a single, large forward pass to generate the KV cache for the first output token. This operation's duration is *proportional to the prompt length*.
2.  **Decode/Generation:** The system takes the KV cache and, one token at a time, generates the output sequence. Each step is very fast and its duration is constant.

A continuous batching scheduler typically cannot add *new* requests to the batch while a prefill operation is running. If a request with a 4000-token prompt arrives, the system might spend hundreds of milliseconds just on its prefill step. Any other (short-prompt) requests that arrive during this window must wait until the prefill is complete and the next scheduling iteration begins. This wait time is what creates the P9ax latency spike. It's a form of head-of-line blocking *at the iteration level*. A hardware fault would likely manifest as random, uncorrelated latency spikes or crashes, not ones that correlate perfectly with long-prompt requests.

  > **Napkin Math:** **Assumptions:**
- H100 GPU serving a 70B model.
- Per-token decode time: ~3ms.
- For simplicity, assume prefill time per token is also ~3ms (in reality it's less efficient, but let's use this as a lower bound).

**Normal Request:**
- Prompt: 20 tokens. Prefill time: `20 tokens * 3ms/token = 60ms`.
- Any request arriving during this 60ms window has to wait. This is acceptable.

**Long-Prompt Request:**
- Prompt: 4000 tokens. Prefill time: `4000 tokens * 3ms/token = 12,000ms = 12 seconds.` (This is too high, let's use a more realistic number based on compute).

**Let's re-calculate prefill time using FLOPs:**
- A 70B model needs `~2 * 70B = 140 GFLOPs` per token.
- An H100 provides `~989 TFLOPS` of FP16 compute.
- Time per token (compute): `140e9 / 989e12 = ~0.14ms`. This is compute-bound time, not wall time. Wall time is dominated by memory access.
- A more realistic wall-time measurement for a well-optimized system is ~0.5ms per prefill token for large prompts.

**Revised Napkin Math:**
- Long-Prompt (4000 tokens) Prefill Time: `4000 tokens * 0.5ms/token = 2000ms = 2 seconds`.

During these 2 seconds, the scheduler is locked processing the long prompt. No new requests can be added to the batch. A normal chat request that arrives at the beginning of this window will have its TTFT be `2 seconds (wait time) + 60ms (its own prefill) = 2.06 seconds`. This perfectly explains why the P99 latency is spiking to over 2 seconds while the average (dominated by non-blocked requests) remains low.

  > **Key Equation:** $T_{\text{TTFT}} = T_{\text{wait_queue}} + T_{\text{prefill}}(N_{\text{prompt_tokens}})$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Throttling Dilemma: Per-User vs. Global Queueing</b> · <code>queueing-rate-limiting</code></summary>

- **Interviewer:** "You are designing the queueing and rate-limiting architecture for a new multi-tenant LLM API. The system must provide fair service and prevent a single 'power user' from degrading performance for others. You need to analyze the latency implications of two different designs under a 'thundering herd' scenario where one user submits a large burst of requests.

1.  **Global FIFO Queue:** A single queue for all users' requests feeding the GPU batcher.
2.  **Per-User Queues:** Each user/API key gets their own logical queue. A scheduler (e.g., weighted fair queueing) pulls from these user queues to build a batch.

Examine how these two systems handle a burst of 100 requests from a single power user. Specifically, calculate the expected wait time for a normal, low-volume user who submits a request immediately after the burst."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Only considering the total throughput. A global queue might seem simpler and achieve the same theoretical maximum throughput. The mistake is ignoring the variance in wait time and the lack of fairness. A single user can easily saturate the system's capacity, leading to extreme tail latency for everyone else. Fairness is not just a 'nice-to-have'; it's a critical part of the SLO for a multi-tenant service.

  **Realistic Solution:** The two designs have fundamentally different fairness and latency profiles.

**1. Global FIFO Queue:** This design is simple but unfair. When the power user submits 100 requests, they are all enqueued ahead of the normal user's request. The normal user must wait for all 100 of those requests to be processed before their own request even enters the batcher. The power user monopolizes the service capacity, leading to high latency for all other tenants. The system has no concept of 'fairness'.

**2. Per-User Queues:** This design isolates user impacts. The power user's 100 requests fill their *own* queue. The normal user's single request goes into their separate queue. The scheduler, designed for fairness, will not exclusively service the power user. Using a round-robin or weighted fair queueing policy, it will pick one request from the power user's queue, then one from the normal user's queue, and so on. The normal user's request gets processed almost immediately, experiencing minimal wait time. The power user's requests are processed over time, but they don't block other users. This architecture provides strong performance isolation between tenants, which is essential for a multi-tenant service.

  > **Napkin Math:** **Assumptions:**
- The system can process a batch of 10 requests every 100ms. So, the service rate (`μ`) is 100 requests/second.
- Arrival rate (`λ`) is low, say 20 requests/second, so the system is normally not overloaded (`ρ = λ/μ = 0.2`).

**Scenario:** A power user dumps 100 requests at t=0. A normal user sends 1 request at t=0.1s.

**Global FIFO Queue Analysis:**
- At t=0, the queue length `L` jumps to 100.
- At t=0.1s, the normal user's request arrives and is placed at position 101 in the queue.
- The system processes 10 requests every 100ms. To process the 100 requests ahead of the normal user, it will take `100 requests / (10 requests / 0.1s) = 1 second`.
- The **wait time for the normal user is at least 1 second**. This is a massive latency spike caused by another user.

**Per-User Queue Analysis:**
- At t=0, User_Power's queue length becomes 100.
- At t=0.1s, User_Normal's queue length becomes 1.
- The fair scheduler's turn is at t=0.1s. It sees two non-empty queues. It builds a batch. Let's say it pulls 9 requests from User_Power and 1 from User_Normal (or just round-robins 1 from each).
- The normal user's request is included in the very next batch, which is processed at t=0.1s.
- The **wait time for the normal user is effectively near-zero** (just the scheduler iteration time, maybe a few milliseconds). The 100-request burst from the power user is handled without impacting the normal user's latency.

  > **Key Equation:** $\text{Fairness: WaitTime}(\text{User}_A) \stackrel{?}{\approx} \text{WaitTime}(\text{User}_A | \text{Load}(\text{User}_B))$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The P99 Latency Volcano</b> · <code>queueing-theory-latency</code></summary>

- **Interviewer:** "You are operating a fleet of H100 GPUs serving a 70B parameter LLM using a static batching strategy with a batch size of 8. As your request volume approaches 50% of the theoretical maximum throughput, you notice that average GPU utilization is stable around 60%, but the P99 latency for end-users is exploding, leading to frequent timeouts. A team member suggests adding more GPUs to handle the load. Differentiate the impact of adding more hardware versus redesigning the batching strategy. Analyze the root cause of the latency spike."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often assume that latency problems at moderate utilization are due to insufficient hardware capacity. They would recommend adding more GPUs (scaling out). While this reduces the arrival rate (λ) per machine, it doesn't solve the core issue of service time variance, known as head-of-line blocking.

  **Realistic Solution:** The root cause is head-of-line blocking exacerbated by high service time variance in a static batching system. An LLM request for a long generation (e.g., 2048 tokens) paired in a static batch with a request for a short generation (e.g., 32 tokens) forces the short request to wait for the entire long generation to complete. This disproportionately affects tail latency (P99). Adding more GPUs is a brute-force solution that reduces the probability of a short request getting stuck, but it's inefficient. The correct solution is to switch to a continuous batching (or 'dynamic batching') scheduler. This strategy decouples batching from request arrival. The server can continuously add new requests to a running batch as soon as space is available (i.e., another request in the batch finishes), dramatically reducing queueing delay for short requests and improving overall GPU utilization.

  > **Napkin Math:** Let's analyze two requests arriving close together: R1 (long) needs 2048 tokens, R2 (short) needs 32 tokens.

- **Hardware:** H100 provides ~989 TFLOPS (FP16).
- **Compute per token (70B model):** `C ≈ 2 * P = 2 * 70e9 = 140 GFLOPs`.
- **Time per token (unbatched):** `T_token = 140e9 FLOPs / 989e12 FLOPS ≈ 0.14 ms`.
- **Static Batching Scenario:** If R1 and R2 are batched together, the batch isn't done until the longest request is done.
  - `T_R1 = 2048 tokens * 0.14 ms/token ≈ 287 ms`.
  - R2 is finished in `32 * 0.14 ≈ 4.5 ms`, but it is trapped in the batch. Its effective latency becomes `287 ms`, an inflation of `287 / 4.5 ≈ 64×`.
- **Continuous Batching Scenario:** R2 can be processed and leave the batch as soon as it's done. Its latency is its own processing time plus a very small queueing delay. R1's presence doesn't block it for the full duration. GPU utilization increases because as soon as R2's slots in the batch are free, a new request can be swapped in.

  > **Key Equation:** $W_q \approx \frac{\lambda E[S^2]}{2(1-\rho)}$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Two-Tier Traffic Jam</b> · <code>tpot-vs-ttft-batching</code></summary>

- **Interviewer:** "You manage a single GPU serving cluster for a 13B parameter model on H100s. The service has two use cases: interactive chat, where users demand low Time-To-First-Token (TTFT), and offline document summarization, where total job time (which depends on Throughput of Output Tokens, or TPOT) is the key metric. You are using continuous batching with a single configuration. Chat users complain of high latency, while the summarization workload is inefficient. Differentiate between a system optimized for TTFT and one for TPOT, and analyze why a single configuration fails for both. Propose an alternative architecture."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often conflate latency and throughput, attempting to find a single 'optimal' batch size. The common mistake is failing to recognize that TTFT and TPOT are fundamentally different metrics driven by different system dynamics. TTFT is sensitive to queueing delays, while TPOT is sensitive to computational efficiency. A compromise configuration is mediocre at both.

  **Realistic Solution:** A single configuration fails because the two workloads are at opposite ends of the performance spectrum defined by the Roofline model.
- **TTFT Optimization (Chat):** To minimize TTFT, you must minimize the time a request waits in a queue before processing begins. This requires using a very small batch size and a short batching timeout. This keeps the system highly responsive. However, small batches have low arithmetic intensity, making them memory-bandwidth bound. The GPU is 'starved' for computation, leading to low efficiency and poor TPOT.
- **TPOT Optimization (Summarization):** To maximize TPOT, you need to make the GPU as efficient as possible. This requires assembling large batches to increase arithmetic intensity, pushing the workload over the 'ridge point' of the Roofline model into the compute-bound regime. This maximizes FLOPs utilization but introduces long queueing delays, which is poison for TTFT.

**Proposed Architecture:** Split the cluster into two pools or implement a two-level scheduler:
1.  **Real-time Pool (Chat):** A dedicated set of GPUs configured for small batches (e.g., max_batch_size=4) and short timeouts (e.g., 5ms). This pool will provide excellent TTFT.
2.  **Batch Pool (Summarization):** Another set of GPUs configured for large batches (e.g., max_batch_size=128) and longer timeouts (e.g., 50ms). This pool will deliver maximum TPOT.

  > **Napkin Math:** Let's compare a small batch vs. a large batch for a 13B model on an H100.

- **Hardware:** H100: 989 TFLOPS peak compute, 3.35 TB/s memory bandwidth. Ridge Point `I_c = 989e12 / 3.35e12 ≈ 295` FLOPs/Byte.
- **Compute per token (13B model):** `2 * 13e9 = 26 GFLOPs`.
- **Case 1 (Chat, Batch Size 1):**
  - Memory traffic is dominated by loading the `13e9 * 2 = 26 GB` of weights.
  - Arithmetic Intensity `I = 26e9 FLOPs / 26e9 Bytes = 1.0` FLOP/Byte.
  - Since `1.0 < 295`, we are deeply memory-bound.
  - Achieved Performance: `Perf = I * BW = 1.0 * 3.35 TB/s = 3.35 TFLOPS`.
  - TPOT: `3.35e12 FLOPS / (26e9 FLOPs/token) ≈ 128` tokens/sec. TTFT is excellent due to low queueing delay.
- **Case 2 (Summarization, Batch Size 128):**
  - Weight memory access is amortized: `26 GB` for `128` tokens. Activation memory becomes significant but let's assume total traffic gives an AI > 295.
  - The system is now compute-bound.
  - Achieved Performance: `Perf = 989 TFLOPS` (Peak).
  - TPOT: `989e12 FLOPS / (26e9 FLOPs/token) ≈ 38,000` tokens/sec.
- **Analysis:** The large batch yields `38000 / 128 ≈ 300×` higher throughput. A chat request stuck in the queue for a large batch to form would have unacceptable TTFT, while running summarization jobs with a tiny batch size is incredibly inefficient.

  > **Key Equation:** $\text{Performance} = \min(\text{Peak Compute}, \text{Arithmetic Intensity} \times \text{Memory Bandwidth})$

  📖 **Deep Dive:** [Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Priority Queue Impasse</b> · <code>ttft-tpot-queueing</code></summary>

- **Interviewer:** "You're running a multi-tenant LLM service on H100s, using a continuous batching inference server. You have two user classes: 'interactive' users who need low Time-To-First-Token (TTFT) for a chatbot, and 'batch' users who submit long document summarization jobs and care about overall Time-Per-Output-Token (TPOT). Interactive users complain of multi-second wait times, even though `nvidia-smi` shows 100% GPU utilization. Analyze the conflict between these workloads within a single FIFO continuous batching queue and propose a system-level solution."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the hardware ('the H100 is too slow') or suggesting static partitioning of GPUs. This is a suboptimal solution that misses the core issue: the hardware is fully utilized, but for the wrong priority of work at the wrong time, leading to poor interactive latency.

  **Realistic Solution:** The problem is the FIFO queueing discipline interacting with workloads of vastly different lengths. The continuous batching server is maximizing throughput, but it isn't prioritizing latency-sensitive work. A long batch job (e.g., summarizing a 50k token document) can occupy the GPU for seconds. While it's running, incoming interactive requests (which might only need to generate 50 tokens) are queued up, waiting. The GPU is busy (high TPOT for the batch job), but TTFT for new requests is terrible because their wait time is dominated by the completion of the long job ahead of them. The correct solution is to implement request preemption or a priority-based scheduling policy within the inference server. High-priority interactive requests must be able to interrupt (preempt) and pause a low-priority batch job, get serviced, and then allow the batch job to resume. This ensures low TTFT for interactive users while still using idle cycles to process long batch jobs.

  > **Napkin Math:** Let's model a 70B parameter model on an H100.
- **Compute per token:** From the rules, this is `2 * 70B = 140 GFLOPs`.
- **H100 FP16 throughput:** 989 TFLOPS.
- **Ideal time per token (single request):** `140 GFLOPs / 989 TFLOPS ≈ 0.14 ms/token`.
- **Batch Job:** A user wants to generate a 20k token summary. The total generation time will be `20,000 tokens * 0.14 ms/token = 2800 ms = 2.8 seconds`.
- **The Conflict:** If an interactive chat request arrives 10ms after this batch job starts, it's stuck in the FIFO queue. Its wait time will be nearly the full 2.8 seconds before its first token can even be computed. This is an unacceptable TTFT. A preemption system would pause the 2.8s job, service the interactive request (e.g., 50 tokens * 0.14ms/token ≈ 7ms), and then resume, keeping TTFT low.

  > **Key Equation:** L = \lambda W

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The KV Cache Thrashing Cascade</b> · <code>continuous-batching-kv-cache</code></summary>

- **Interviewer:** "Your multi-tenant LLM serving cluster (H100s, 80GB HBM) uses continuous batching. GPU compute utilization is consistently high, but P99 latency is spiking and overall requests-per-second is 30% below your benchmark. The KV cache is allocated 60 GB of HBM, and monitoring shows this buffer is always near 100% full. Differentiate between a system that is compute-bound versus one that is memory-capacity-bound in this specific context, and analyze the cascading failure mode that is likely occurring."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming that a full KV cache is a sign of an efficient system ('we are using all our memory!'). This fails to distinguish between useful, warm cache occupancy and destructive cache thrashing where the cost of recomputing evicted state outweighs the benefit of accepting a new request.

  **Realistic Solution:** The system is memory-capacity-bound, not compute-bound, and is suffering from KV cache thrashing. A perpetually full KV cache means the scheduler has no room for new requests. To accept a new request, it must evict the KV cache of a sequence that is currently paused (e.g., waiting for a user's next turn in a chat). When that user's request returns, its entire prompt history must be re-processed from scratch to regenerate the evicted KV cache state. This re-computation, or 'recompute tax,' consumes valuable GPU cycles that could have been used for generating new tokens for other users. This directly explains both the reduced system throughput (as cycles are wasted) and the P99 latency spikes experienced by unlucky users whose caches were evicted. The solution involves either reducing the maximum sequence length, using a more memory-efficient model, or implementing more sophisticated eviction policies (like Least Recently Used) instead of what is likely a random or FIFO eviction.

  > **Napkin Math:** Let's model a 70B model like Llama 70B (80 layers, 8192 model dimension).
- **KV Cache per token:** The formula is `2 × layers × d_model × 2 bytes`. So, `2 * 80 * 8192 * 2 bytes ≈ 2.62 MB/token`.
- **User Context:** A user with a 16k token context history consumes `16,384 tokens * 2.62 MB/token ≈ 42.9 GB` of KV cache.
- **Capacity Wall:** The 60 GB HBM allocation can't even hold two of these users (`42.9 GB * 2 = 85.8 GB`). To serve a second user, the first must be evicted.
- **Recompute Tax:** The cost to regenerate the 16k token cache is `16,384 tokens * 140 GFLOPs/token = 2,293 TFLOPs`.
- **Time Wasted:** Recomputing on an H100 takes `2,293 TFLOPs / 989 TFLOPS ≈ 2.3 seconds`. This 2.3 seconds of pure re-computation work is added directly to the user's TTFT and prevents the GPU from doing useful new work for other clients, killing throughput.

  > **Key Equation:** T_{\text{recompute}} = (C_{\text{token}} \times S_{\text{prefix}}) / \Theta_{\text{GPU}}

  📖 **Deep Dive:** [Training](https://mlsysbook.ai/vol1/training.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Deadline-Missing Detector</b> · <code>queueing-theory-realtime</code></summary>

- **Interviewer:** "You are designing a real-time LLM-based fraud detection system. Each transaction must be classified in under 100ms. Your service runs on a single H100 GPU, and instrumenting the model shows that the average processing time (TPOT) is 50ms per request. The system uses a standard FIFO queue. During peak market hours, the arrival rate hits 18 requests per second. You observe that a significant fraction of requests miss their 100ms deadline. Examine the system using queueing theory. Why is a system with 90% average utilization failing so badly, and what queueing strategy would you recommend?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Only looking at the average service time (`50ms < 100ms`) and average utilization, then concluding the system should be fine. This reasoning ignores the non-linear nature of queueing delays, where the time spent *waiting* for service begins to dominate the total latency as utilization approaches 100%.

  **Realistic Solution:** This is a classic queueing theory problem that trips up many engineers. As system utilization (ρ) gets high, the average waiting time in the queue (Wq) grows exponentially, not linearly. At 90% utilization, the average wait time is already massive and, due to statistical variance in arrival and service times (the 'M's in M/M/1), a large fraction of individual requests will experience waits far longer than the average, thus exceeding the deadline. The solution is to abandon the simple FIFO queue. Potential strategies include:
1. **Load Shedding:** If a request has already been in the queue for >50ms, it's guaranteed to miss the 100ms deadline. Drop it immediately and return an error or fallback to a simpler, faster rule-based system. This preserves resources for requests that still have a chance.
2. **Autoscaling:** Add more GPUs (servers) to handle the peak load. This changes the system from an M/M/1 to an M/M/c queue, which has vastly better waiting time characteristics for the same level of overall utilization.

  > **Napkin Math:** Let's model the system as an M/M/1 queue.
- **Service rate (μ):** The system can process 1 request every 50ms, so `1 / 0.050s = 20 requests/sec`.
- **Arrival rate (λ):** During peak, this is `18 requests/sec`.
- **System Utilization (ρ):** `ρ = λ / μ = 18 / 20 = 0.9` (or 90%).
- **Average wait time in queue (Wq):** For an M/M/1 queue, the formula is `Wq = ρ / (μ * (1-ρ))`.
- `Wq = 0.9 / (20 * (1 - 0.9)) = 0.9 / (20 * 0.1) = 0.9 / 2 = 0.45 seconds = 450 ms`.
- **Total average response time (W):** `W = Wq + (1/μ) = 450ms + 50ms = 500ms`.
- This is 5 times the required 100ms deadline. The math proves that the system is fundamentally unstable under these conditions, despite the average service time looking safe.

  > **Key Equation:** W_q = \frac{\rho}{\mu(1-\rho)}

  📖 **Deep Dive:** [ML Operations](https://mlsysbook.ai/vol1/ops.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Cannibalistic Batching Strategy</b> · <code>autoscaling-batching-tradeoff</code></summary>

- **Interviewer:** "Your team runs a serverless LLM API that autoscales based on pending requests. The current batching strategy waits up to 400ms to form a large batch to maximize H100 utilization and TPOT, which is key to keeping costs low. However, users are complaining about high and unpredictable TTFT. You observe that during spiky traffic, the system scales up aggressively but the new instances show low initial utilization. Analyze the fundamental tension between the batching strategy's goal (high TPOT) and the user-perceived latency (TTFT). Why is the batching strategy cannibalizing its own efficiency?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Suggesting a simple tweak like 'just shorten the batching timeout.' This is a temporary fix that doesn't address the core, systemic conflict. A shorter timeout will improve TTFT but decrease batch size, cratering throughput, increasing costs per token, and potentially causing the autoscaler to trigger even more aggressively, leading to a fleet of under-utilized, expensive GPUs.

  **Realistic Solution:** The system is working at cross-purposes. The batching algorithm's long timeout artificially inflates the queue waiting time to build large batches for high TPOT. This long wait time is precisely what users perceive as bad TTFT. Worse, it's also what triggers the autoscaler. The autoscaler sees a long queue and spins up a new, expensive H100 instance, which *itself* then sits mostly idle, also waiting up to 400ms to form *its own* large batch. The batching strategy is actively creating the latency that leads to inefficient scaling. The solution is dynamic or adaptive batching. When the request queue is short, the system should use a very short timeout and small batch size to prioritize TTFT. When the queue grows long (indicating high sustained load), the timeout and target batch size should increase to shift priority to maximizing TPOT and system throughput. This aligns the batching strategy with the current system state, providing low latency at low load and high throughput at high load.

  > **Napkin Math:** Let's compare two static strategies for a 13B model on an H100. Assume inference time per token is 0.026ms and there's a fixed overhead of 5ms per batch.
- **Strategy A (High TPOT):** Wait up to 400ms. Forms a batch of 32 requests, each generating 100 tokens.
  - Total tokens: `32 * 100 = 3200`.
  - Batch processing time: `5ms + (3200 * 0.026ms) = 5ms + 83.2ms = 88.2ms`.
  - Throughput: `3200 tokens / 88.2ms ≈ 36,281 tokens/sec`.
  - TTFT for last request in batch: `400ms (wait) + 88.2ms (process) = 488.2ms` (Poor).
- **Strategy B (Low TTFT):** Wait up to 20ms. Forms a batch of 4 requests, each generating 100 tokens.
  - Total tokens: `4 * 100 = 400`.
  - Batch processing time: `5ms + (400 * 0.026ms) = 5ms + 10.4ms = 15.4ms`.
  - Throughput: `400 tokens / 15.4ms ≈ 25,974 tokens/sec` (28% lower throughput).
  - TTFT for last request: `20ms (wait) + 15.4ms (process) = 35.4ms` (Excellent).

The analysis shows that naively prioritizing TTFT costs ~28% in throughput, which directly translates to higher operational costs. The correct analysis is that the system must dynamically switch between these strategies based on load.

  > **Key Equation:** \Theta_{\text{eff}} = \frac{N_{\text{tokens}}}{T_{\text{wait}} + T_{\text{overhead}} + N_{\text{tokens}} \times T_{\text{token}}}

  📖 **Deep Dive:** [Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Head-of-Line Blocking Crisis</b> · <code>continuous-batching-vs-static</code></summary>

- **Interviewer:** "You are leading the ML Systems team for a new AI chatbot service using a 70B parameter model on H100 GPUs. Users are reporting highly variable response times. Your dashboard shows that while peak throughput is high, average GPU utilization is only 40% and P99 TTFT (Time To First Token) is poor. The current serving stack uses static batching with a fixed batch size of 8. Your junior engineers suggest increasing the batch size to improve GPU utilization. Analyze the fundamental flaw in the current system and distinguish why continuous batching is a better approach for this interactive workload."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Believing that higher batch size is always better for utilization. While true for offline jobs, for interactive serving, large static batches introduce head-of-line blocking. A short request can get stuck waiting for a very long request in the same batch to complete, killing average latency and user experience even if the GPU is technically busy.

  **Realistic Solution:** The root cause is head-of-line blocking inherent to static batching. The entire batch of 8 requests is bottlenecked by the single longest-running request in the batch. If one user asks for a 2000-token summary while another asks a 10-token question, the quick question is forced to wait for the long summary to finish decoding. This leads to low effective GPU utilization (as slots in the batch sit idle after finishing early) and terrible P99 TTFT.

Continuous batching (or iteration-level batching) solves this. The server operates in fine-grained steps (iterations). At each step, it generates one token for all requests currently in the batch. Once a request is complete, its slot is immediately freed and can be filled by a new request from the queue. This decouples request completion times, eliminates head-of-line blocking, and dramatically increases GPU utilization and overall throughput for interactive workloads with variable sequence lengths.

  > **Napkin Math:** Let's model one static batch. Assume a 70B model on an H100 has a per-token generation time (TPOT) of ~25ms for a full batch. The batch contains requests with output lengths of [20, 30, 40, 50, 60, 70, 80, 2000] tokens.

1.  **Static Batch Time:** The batch isn't finished until the longest request completes. Total time = `2000 tokens * 25 ms/token = 50,000 ms = 50 seconds`. All 7 other requests, including the one that needed only 20 tokens (which should have taken `20 * 25ms = 0.5s`), are stuck waiting for the full 50 seconds.

2.  **Effective Utilization:** During the first 0.5s, all 8 slots are active. After that, the first request's slot is idle but still occupies the batch. As more requests finish, the number of active slots dwindles, tanking utilization. The average utilization over the 50s window is extremely low.

3.  **Continuous Batching:** The 20-token request finishes in ~0.5s and its result is returned. Its slot is immediately filled by a new user. The system stays packed, GPU utilization remains high, and average user wait time plummets.

  > **Key Equation:** W_{total} = W_{queue} + T_{service}

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The SLA Violation Cascade</b> · <code>queueing-theory-sla</code></summary>

- **Interviewer:** "You're designing a real-time transcription service with a strict P99 TTFT SLA of 300ms. On a single H100, the model's prompt processing and first token generation takes 200ms. During steady state, the service receives 30 requests per second. Your junior engineer provisions 5 H100s, arguing the total capacity (5 servers * 5 req/s/server = 25 req/s) is 'close enough' to the 30 req/s arrival rate. During a live test, the P99 latency spirals to over 1 second. Analyze this failure using queueing theory and determine the minimum number of GPUs required to reliably meet the SLA. Examine what happens to the queue if traffic spikes by just 20%."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Sizing a system based on average arrival rate and service rate, ignoring the distribution of arrivals and the non-linear impact of system utilization on queueing delay. Engineers often forget that as utilization approaches 100%, wait times don't just increase linearly, they increase exponentially, causing a cascade of SLA violations. 'Close enough' to 100% utilization is a recipe for failure.

  **Realistic Solution:** The analysis was flawed because it neglected queueing dynamics. A system's stability requires total service rate to be strictly greater than arrival rate. At 30 req/s arrival and 25 req/s total service rate, the system is unstable ($\\rho > 1$). The request queue will grow indefinitely, and latency will spiral to infinity.

To meet a P99 SLA, utilization must be kept low enough to absorb natural burstiness in traffic. A good rule of thumb for strict latency targets is to keep utilization at or below 50-70%.

1.  **Calculate Minimum Stable GPUs:** Each H100 has a service rate $\\mu = 1 / 0.2s = 5$ req/s. For an arrival rate $\\lambda = 30$ req/s, we need $N > \\lambda / \\mu \implies N > 30 / 5 \implies N > 6$. So, at least 7 GPUs are needed just for stability.

2.  **Calculate GPUs for SLA:** Let's target 60% utilization to keep tail latency low. $\\rho = \\lambda / (N * \\mu) \implies 0.6 = 30 / (N * 5) \implies N = 30 / (0.6 * 5) = 30 / 3 = 10$. We need 10 GPUs. With 10 GPUs, the system is well-provisioned, queueing delays are minimal, and the P99 latency will be very close to the 200ms service time, comfortably meeting the 300ms SLA.

3.  **Spike Analysis:** A 20% spike means $\\lambda_{new} = 30 * 1.2 = 36$ req/s. With our 10 GPUs, the new utilization is $\\rho_{new} = 36 / (10 * 5) = 36/50 = 72%$. The system remains stable and will likely still meet the SLA, demonstrating the importance of maintaining a capacity buffer.

  > **Napkin Math:** Let's formalize the analysis:
- Service Rate per GPU ($\mu$): $1 / 200\text{ms} = 5$ requests/sec.
- Arrival Rate ($\lambda$): $30$ requests/sec.

**Initial Flawed Design (5 GPUs):**
- Total Service Rate: $5 \text{ GPUs} \times 5 \text{ req/s/GPU} = 25$ req/s.
- Utilization ($\rho$): $\lambda / (N\mu) = 30 / 25 = 1.2$. Since $\rho > 1$, the queue is unstable and will grow infinitely.

**Correct Design for SLA (10 GPUs):**
- Total Service Rate: $10 \text{ GPUs} \times 5 \text{ req/s/GPU} = 50$ req/s.
- Utilization ($\rho$): $30 / 50 = 0.6$ (60%).
- At 60% utilization, queueing theory for M/M/c queues suggests P99 latency is roughly $1.5\times-2\times$ the service time. P99 Latency $\approx 1.8 \times 200\text{ms} = 360\text{ms}$. This is slightly over, so we might need 11-12 GPUs for a hard SLA, but 10 is the right analytical starting point. Let's re-calculate with 12 GPUs: $\rho = 30 / (12*5) = 50%$. At 50% utilization, queue delay is minimal, and P99 latency will be very close to the 200ms service time.

  > **Key Equation:** $\rho = \frac{\lambda}{N \cdot \mu} < 1$

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Prefill vs. Decode Dilemma</b> · <code>ttft-tpot-tradeoff</code></summary>

- **Interviewer:** "You are optimizing an LLM serving system for a code completion assistant on H100s. The user experience is sensitive to both TTFT (time to first suggestion) and TPOT (throughput of the generated code block). Your profiler shows that prompt processing (prefill) is compute-bound, while token generation (decode) is memory-bound. To improve TPOT, a team member suggests doubling the batch size. Examine the relationship between batching, prefill, and decode. Differentiate the impact of increasing the batch size on TTFT versus TPOT and explain why the suggestion might harm the user experience."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Conflating throughput (tokens/sec) with latency (ms/token). Engineers often assume that techniques to improve aggregate throughput, like larger batches, will improve per-user perceived latency. For LLM serving, the opposite is often true. The decode step is memory-bandwidth bound, so a larger batch doesn't make token generation much faster but significantly increases the queueing and prefill time, hurting TTFT.

  **Realistic Solution:** The suggestion is based on a misunderstanding of LLM inference bottlenecks. We need to analyze the two phases separately:

1.  **Prefill Phase (TTFT):** This phase processes the user's entire prompt at once. The computation is highly parallelizable across tokens and resembles a large matrix multiplication. It is compute-bound. Doubling the batch size means more total computation, but more importantly, it increases the time a request has to wait in the queue for a batch to fill. This *increases* the average time until the prefill phase even starts for a given user, directly harming TTFT.

2.  **Decode Phase (TPOT):** This phase generates one token at a time. The computation for a single token is small, but it requires loading the entire model's weights from HBM. This phase is almost always memory-bandwidth bound. Since the H100's HBM bandwidth (3.35 TB/s) is a fixed resource, and each forward pass needs to read the weights, the time per decoding step is largely constant regardless of batch size (beyond a small batch that saturates the memory bus). Doubling the batch size from 16 to 32 does not halve the time per token; it barely changes it. The system generates 32 tokens in roughly the same time it generates 16.

**Conclusion:** Doubling the batch size will increase the aggregate system throughput (total tokens per second across all users) but at the cost of higher per-user TTFT due to increased queueing delays. For an interactive code assistant where initial responsiveness is key, this is a bad trade-off. The better approach is to use continuous batching to keep batch sizes dynamically adjusted, minimizing queue times and thus optimizing for TTFT while still packing the GPU efficiently for good TPOT.

  > **Napkin Math:** Let's model a 70B model on an H100.
- **Model Weights Size:** $70\text{B params} \times 2 \text{ bytes/param} = 140$ GB.

- **Decode (TPOT):** The bottleneck is reading the 140 GB of weights from HBM.
  - H100 HBM3 Bandwidth: 3.35 TB/s.
  - Time to read weights (theoretical lower bound on TPOT): $140 \text{ GB} / 3.35 \text{ TB/s} \approx 41.8$ ms.
  - This ~42ms latency per token is the physical limit. Whether the batch size is 8 or 32, you still pay this memory access cost for every single generation step. Batching helps amortize kernel launch overhead but doesn't change this fundamental memory wall. So, TPOT is largely insensitive to batch size once the pipe is full.

- **Prefill (TTFT):** Assume a 2048 token prompt.
  - Compute FLOPs: $\approx 2 \times 70\text{B params} \times 2048 \text{ tokens} \approx 286 \times 10^{15}$ FLOPs = 286 PFLOPs.
  - H100 (realistic): ~500 TFLOPS.
  - Prefill time: $286 \text{ PFLOPs} / 500 \text{ TFLOPS} \approx 572$ ms.
  - If we use static batching, the TTFT is $T_{queue} + 572\text{ms}$. Increasing the batch size directly increases the average $T_{queue}$. For example, if requests arrive at 10/sec and batch size is 32, the batch fill time alone is $3.2$s. This is a catastrophic hit to TTFT.

  > **Key Equation:** T_{decode} \approx \frac{\text{Model Size (Bytes)}}{\text{Memory Bandwidth (Bytes/s)}}

  📖 **Deep Dive:** [Volume II: Inference](https://mlsysbook.ai/vol2/inference.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The P99 Latency Explosion</b> · <code>llm-serving-queueing-theory</code></summary>

- **Interviewer:** "You are managing an LLM serving endpoint for a real-time translation feature with a strict 100ms P99 latency SLA, running on an H100 GPU. Your service uses dynamic batching with the maximum batching window set to 10ms. During a load test, you observe that while the mean latency is excellent (~50ms), the P99 latency frequently spikes to over 500ms, violating the SLA. Your profiler confirms that the model's forward pass is computationally fast enough for the traffic volume. Analyze the interaction between your batching strategy and queueing dynamics to explain this P99 explosion."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the model, the hardware, or network I/O. A common guess is that some individual requests are just 'unlucky' or 'pathologically slow'. This ignores the systemic nature of queue buildup.

  **Realistic Solution:** The P99 latency explosion is a classic symptom of a queueing bottleneck caused by an inefficient service rate, even if the hardware is powerful. The short 10ms batching window forces the server to process many small, inefficient batches during traffic bursts. While this keeps latency low for requests in those small batches (contributing to a good mean), it drastically lowers the maximum system throughput (the service rate, μ). When the arrival rate (λ) from a traffic spike exceeds this low service rate, the request queue grows unboundedly. The unlucky requests at the tail end (P99) are the ones that get stuck in this massive queue, leading to extremely high end-to-end latency.

  > **Napkin Math:** Let's analyze the service rate (μ). Assume an H100 has a fixed overhead of 5ms per batch (kernel launch, data movement) and a per-request compute time of 2ms.
- **Scenario 1: Small Batches (e.g., avg. batch size of 4):**
  - Time per batch = 5ms (overhead) + 4 * 2ms (compute) = 13ms.
  - Service Rate (μ) = 4 requests / 13ms = ~307 req/s.
- **Scenario 2: Large Batches (e.g., avg. batch size of 32):**
  - Time per batch = 5ms (overhead) + 32 * 2ms (compute) = 69ms.
  - Service Rate (μ) = 32 requests / 69ms = ~463 req/s.
If a traffic spike hits 400 req/s, the small-batch strategy (μ=307 req/s) cannot keep up (ρ = λ/μ > 1), and its queue will grow infinitely. The large-batch strategy (μ=463 req/s) can handle the load (ρ < 1). The P99 spike is the physical manifestation of an unstable queue.

  > **Key Equation:** $$W_q \approx \frac{\rho}{1-\rho} \times \frac{C_a^2 + C_s^2}{2\lambda}$$

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Continuous Batching Paradox</b> · <code>continuous-batching-vs-static</code></summary>

- **Interviewer:** "You're designing the serving stack for an interactive coding assistant on H100s. Two strategies are proposed:
1. **Static Batching:** Pad all requests in a batch to the length of the longest prompt.
2. **Continuous Batching (Orca-style):** Use a memory-efficient KV cache (like PagedAttention) that decouples request lifetimes.
You receive two requests simultaneously: Request A (prompt: 1000 tokens) and Request B (prompt: 50 tokens). Differentiate how these two strategies handle the prefill stage and impact Time To First Token (TTFT). Calculate the wasted KV cache memory for Request B under static batching, assuming a Llama-7B-scale model."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming that since both requests are in the same batch, their TTFT will be identical. This overlooks the massive compute and memory waste from padding in static batching, which directly impacts throughput and queue times, thereby delaying the start of the next batch and hurting every request's TTFT.

  **Realistic Solution:** Static batching forces Request B to be padded to 1000 tokens. This wastes both computation (the GPU processes 950 padding tokens for B) and memory (the KV cache must be allocated for 1000 tokens for B). This bloat reduces the overall system throughput, meaning fewer batches can be processed per second, which increases queue times for everyone. Continuous batching eliminates this waste. It allocates memory only for the tokens present (1000 for A, 50 for B). This dramatically increases memory efficiency, allowing more requests to be processed concurrently and increasing the service rate. The result is higher throughput, shorter queue times, and therefore a better TTFT for all incoming requests.

  > **Napkin Math:** The key is wasted memory. From the `NUMBERS.md` guide, the KV cache for a Llama-like model is dominated by the parameters for the Key and Value projections per token, per layer.
- **KV Cache per Token Formula:** `2 * num_layers * d_model * 2 bytes` (for FP16)
- **Llama-7B Specs:** 32 layers, hidden size (d_model) of 4096.
- **Calculation:** `2 * 32 * 4096 * 2 bytes = 524,288 bytes ≈ 0.5 MB per token`.
- **Static Batching Waste:** Request B's prompt is 50 tokens but is padded to 1000.
- **Wasted Memory for Req B:** `(1000 tokens - 50 tokens) * 0.5 MB/token = 950 * 0.5 MB = 475 MB`.
This 475 MB of HBM is completely wasted and cannot be used to serve another request, directly reducing the system's capacity.

  > **Key Equation:** $$\text{Memory}_{\text{waste}} = \sum_{i \in \text{batch}} (L_{\max} - L_i) \times \text{Mem}_{\text{token}}$$

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Coding Assistant's Latency Crisis</b> · <code>inference-serving-tradeoffs</code></summary>

- **Interviewer:** "You are the lead engineer for a new AI code completion service running on H100 GPUs. The product requirement is a P99 Time-To-First-Token (TTFT) of less than 500ms to feel instantaneous. Your dashboards show excellent aggregate throughput, processing thousands of requests per minute. However, user feedback is poor, and your detailed metrics confirm the P99 TTFT is hovering around 1500ms, badly missing the SLO.

Your team's current serving configuration uses static batching with a maximum batch size of 64 and a batching timeout window of 1000ms to maximize GPU utilization.

Analyze the relationship between the batching strategy and the TTFT SLO violation. Differentiate between the latency experienced by a request arriving early in the batch window versus one arriving late, and use this to explain why high throughput and high tail latency can coexist."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus exclusively on GPU compute time or throughput, assuming that if the GPU is busy, the system is efficient. They mistake high throughput for low latency. A common incorrect diagnosis would be to blame the model size ('the model is too slow') or the hardware ('we need faster GPUs'), ignoring the fact that the GPU is often waiting due to a system-level bottleneck: queueing delay.

  **Realistic Solution:** The root cause is the static batching strategy with its long 1000ms timeout. This creates a significant queueing delay, also known as head-of-line blocking. A request that arrives at the beginning of a batching window must wait up to 1000ms for the batch to fill or for the timeout to expire before it even begins processing. This waiting time is the dominant contributor to TTFT, especially for the tail (P99).

While this strategy maximizes throughput by creating large, full batches for the GPU, it does so at the direct expense of latency. The correct solution is to switch to a more dynamic serving strategy like **continuous batching** (or 'in-flight batching'). Continuous batching decouples request arrival from batch execution. It maintains a running batch on the GPU and dynamically swaps in new requests as old ones finish, drastically reducing queueing time to nearly zero and thus optimizing for TTFT.

  > **Napkin Math:** Let's analyze the waiting time (`W`) from the static batching window.

*   **Timeout Window:** `T_window = 1000ms`
*   **SLO:** P99 TTFT < 500ms
*   **Model Prefill Time (Assumption):** `T_prefill` ≈ 200ms

1.  **Worst-Case Wait (Early Arrival):** A request arrives at `t=1ms`, just after a batch has been dispatched. It must wait for the entire window to expire before processing begins.
    *   `W_worst` ≈ `T_window` = 1000ms
    *   `TTFT_worst` = `W_worst` + `T_prefill` = 1000ms + 200ms = 1200ms

2.  **Best-Case Wait (Late Arrival):** A request arrives at `t=999ms`, just as the window is closing.
    *   `W_best` ≈ 0ms
    *   `TTFT_best` = `W_best` + `T_prefill` = 0ms + 200ms = 200ms

The waiting time is uniformly distributed between 0ms and 1000ms. This means a large percentage of requests will have a high waiting time, pushing the P99 latency far above the 500ms SLO. The 1200ms worst-case TTFT is consistent with the observed 1500ms P99, which also includes other system overheads.

With **continuous batching**, the wait time `W` is reduced to the scheduler delay, typically <10ms, making the TTFT ≈ 10ms + 200ms = 210ms, comfortably meeting the SLO.

  > **Key Equation:** L = \lambda W \quad \text{(Little's Law)}

  > **Options:**
  > [ ] The model's prefill computation is too slow. We should quantize the model to INT8 to speed up the initial processing time.
  > [ ] The H100's memory bandwidth is insufficient for this batch size. We should reduce the max batch size to 16 to reduce memory pressure.
  > [x] The static 1000ms batching window introduces excessive queueing delay (head-of-line blocking), which is the primary cause. We should switch to continuous batching.
  > [ ] The P99 latency is caused by a few 'noisy neighbor' requests with extremely long contexts. We should implement a strict context length limit.

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Tyranny of Throughput</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are the tech lead for a new real-time AI coding assistant served on H100s. The service has a strict P99 Time-To-First-Token (TTFT) SLO of 200ms. Your dashboards show fantastic system health: `nvidia-smi` reports 95% utilization and your Tokens-Per-Second (TPS) throughput metrics are exceeding targets. However, user feedback is overwhelmingly negative, with frequent complaints of "lag" and "unresponsiveness." Your team is using a classic static batching strategy with a batching timeout of 150ms to maximize hardware efficiency.

Differentiate the system's high hardware utilization from the poor user-perceived latency. Analyze the fundamental trade-off your current static batching approach is making, and compare it to a modern alternative like continuous batching. Use napkin math to examine why your P99 TTFT SLO is being violated despite the high throughput."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Confusing throughput with latency. Engineers often believe that high GPU utilization and high tokens/second automatically translate to a good user experience. They might suggest optimizing the model itself or adding more hardware, failing to see that the serving *strategy* is the root cause of the user-facing latency by introducing artificial queueing delays.

  **Realistic Solution:** The core problem is the conflict between the goals of maximizing hardware utilization and minimizing user latency, a conflict that static batching handles poorly for real-time services. The 150ms batching timeout is an artificial queueing delay imposed on every single request to build a larger, more efficient batch. While this maximizes the H100's throughput (many tokens generated per second *across all users*), it front-loads a massive latency penalty on each user's TTFT.

The P99 latency is experienced by users whose requests arrive just after a batch has been dispatched, forcing them to wait the *entire* 150ms timeout period before their processing even begins.

Continuous batching (used in systems like vLLM or Orca) resolves this. It decouples batching from queueing. New requests are immediately added to a dynamic batch, and their prefill computation is scheduled in the very next forward pass. The wait time is reduced from a fixed timeout to the sub-millisecond duration of a single model iteration, drastically improving TTFT while still allowing the system to maintain high utilization by packing subsequent generation steps together.

  > **Napkin Math:** Let's analyze the P99 TTFT for the static batching system.

**Key Equation:** Total Latency = Queue Time + Compute Time

1.  **Parameters:**
    *   Static Batching Timeout (`T_queue_max`): 150 ms
    *   Model Prefill Time on H100 (`T_prefill`): Let's estimate a fast 30 ms for a typical prompt.

2.  **Worst-Case Scenario (P99):**
    *   A user's request arrives 1 microsecond after the previous batch was sent to the GPU.
    *   This request must wait the full `T_queue_max` before the server gives up waiting for more requests and dispatches the new, partially-filled batch.
    *   Worst-Case Queue Time = 150 ms.

3.  **P99 TTFT Calculation:**
    *   `P99 TTFT` = `Worst-Case T_queue` + `T_prefill`
    *   `P99 TTFT` = 150 ms + 30 ms = 180 ms

4.  **Analysis vs. SLO:**
    *   The calculated P99 TTFT of 180 ms is already dangerously close to the 200ms SLO, leaving almost no budget for network latency or any other system variance. This is why the SLO is being violated.

5.  **Comparison with Continuous Batching:**
    *   The 150ms `T_queue_max` is eliminated. A new request only waits for the current GPU forward pass to complete before being scheduled.
    *   `T_queue_continuous` ≈ `T_iteration` ≈ 5 ms (a generous estimate for one step).
    *   `TTFT_continuous` = 5 ms + 30 ms = 35 ms. This is a ~5x improvement and comfortably meets the SLO.

  > **Key Equation:** L = \lambda W \quad \text{(Little's Law)}

  📖 **Deep Dive:** [Cloud: The Serving Stack](https://mlsysbook.ai/vol2/cloud/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Code Assistant's Latency Crisis</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are leading the ML Systems team for a new AI code assistant. The product requirement is a P99 Time-To-First-Token (TTFT) of under 200ms to feel 'real-time'. The service runs a 13B parameter model on H100 GPUs.

Your profiler shows that an isolated pre-fill for a typical 2048-token prompt takes about 55ms. The per-token decode step takes about 8ms. However, when you deploy a standard continuous batching server (like vLLM) and run a load test at 100 requests per second (RPS), you observe that while the average throughput (tokens/sec) is excellent, the P99 TTFT balloons to over 800ms, completely missing the product requirement.

**Differentiate** the performance characteristics of the pre-fill and decode phases of LLM inference, and **analyze** the system-level queueing dynamics that explain why a seemingly fast system collapses under this load."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often assume that if the individual components are fast (55ms pre-fill, 8ms decode), the system should also be fast. They might incorrectly blame network latency or suggest simply using more GPUs. Another common mistake is to believe all phases of LLM inference are compute-bound on modern accelerators, failing to distinguish the critical difference between pre-fill and decode.

  **Realistic Solution:** The root cause is a queueing theory concept: the system's traffic intensity is greater than 1, leading to an unstable queue.

1.  **Differentiate Pre-fill vs. Decode:** Pre-fill processes the user's entire prompt at once. It's a series of large matrix multiplications, making it parallelizable across the token dimension and thus **compute-bound** on an H100. Decode, however, generates one token at a time. For each token, the entire model's weights must be read from HBM. This makes the decode step **memory-bandwidth-bound**.

2.  **Analyze Queueing Dynamics:** A standard continuous batching server processes work in iterations. When a new request arrives, it must wait for the current iteration to finish before it can be processed. The critical insight is what happens in the *next* iteration. The server batches the compute-bound pre-fill of the new request with the memory-bound decode of all existing requests. The iteration time is determined by the longest of these two operations — which is the pre-fill (`~55ms`).

3.  **Unstable Queue (`ρ > 1`):** At 100 RPS, new requests arrive every 10ms (`T_arrival`). However, every time a new request is introduced, it forces the service to spend `~55ms` on its pre-fill (`T_service`). Since the time to service a new arrival (`55ms`) is much longer than the time between arrivals (`10ms`), the arrival rate outstrips the service rate. The traffic intensity, `ρ = T_service / T_arrival`, is `55/10 = 5.5`. A `ρ > 1` means the queue will grow infinitely, and P99 latency will explode as requests suffer extreme waiting times. The system is fundamentally unstable at this load.

  > **Napkin Math:** **1. Characterize the Workload:**
- Hardware: NVIDIA H100
- Pre-fill Time (compute-bound for 2k tokens): `T_prefill ≈ 55 ms`
- Decode Time (memory-bound for 1 token): `T_decode ≈ 8 ms` (Calculated as `13B params * 2 bytes/param / 3.35 TB/s HBM BW`)

**2. Characterize the Arrivals:**
- Arrival Rate `λ = 100 RPS`
- Inter-Arrival Time `T_arrival = 1 / λ = 1 / 100 = 10 ms`

**3. Analyze the Service Rate & Stability:**
- A continuous batcher mixes pre-fill and decode. When a new request arrives, the *entire batch* is stalled for the duration of the new request's pre-fill, because it's the longest operation in the iteration.
- Effective Service Time for a new request `T_service ≈ T_prefill = 55 ms`.
- Calculate Traffic Intensity `ρ` (rho):
  `ρ = T_service / T_arrival = 55 ms / 10 ms = 5.5`

**4. Conclusion:**
- Since `ρ = 5.5`, which is much greater than 1, the queue is unstable. The server is receiving new work 5.5 times faster than it can process the pre-fill stage for that new work. This guarantees that the wait queue will grow without bound, leading to catastrophic P99 latency.

  > **Key Equation:** $\rho = \frac{\lambda}{\mu} = \lambda \times T_{\text{service}} > 1$

  📖 **Deep Dive:** [Cloud: The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Real-Time Voice Assistant Stutter</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are the lead ML Systems Engineer for a new real-time voice assistant powered by a 7B parameter LLM, hosted on H100 GPUs. The product mandate is a P99 Time-To-First-Token (TTFT) of under 500ms to feel responsive. Your operations team reports that GPU utilization is stuck at a mediocre 40%, and they are proposing to increase the static batch size from 8 to 32 to improve throughput and cost-efficiency. However, user feedback indicates the assistant already feels sluggish to start talking. Your Grafana dashboard shows the arrival rate (λ) is about 20 requests/second.

Analyze the fundamental systems conflict at play here. Differentiate the performance characteristics of static batching versus continuous batching in this scenario, and use napkin math to examine why simply increasing the static batch size is likely to make the TTFT problem worse, not better, despite the low GPU utilization."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common L3-level mistake is to focus only on the compute time and suggest simply using a smaller static batch size. While this reduces the compute portion of latency for a single batch, it craters throughput, causing the request queue to grow infinitely as the arrival rate exceeds the service rate (λ > μ). This leads to catastrophic P99 latency due to unbounded queueing delays, even though individual requests are processed 'fast'. The L4 insight is to change the batching *algorithm* itself to decouple queueing from batch formation.

  **Realistic Solution:** The core conflict is head-of-line blocking imposed by static batching. To achieve high GPU utilization, the system must wait for a large batch to assemble, but this waiting time directly contributes to TTFT, violating the SLO.

1.  **Static Batching:** Forces a brutal trade-off. A small batch size (e.g., 1) minimizes wait time but underutilizes the GPU, leading to low throughput and an unstable queue if λ is high. A large batch size (e.g., 32) improves GPU efficiency but introduces a massive batch-formation delay that makes meeting a tight TTFT SLO impossible.

2.  **Continuous Batching (In-flight Batching):** This is the correct solution. It decouples the request queue from the batching process. The GPU processes a continuous 'megabatch' of requests. In each forward pass, it adds any newly arrived requests to the batch, processes one step for all requests, returns tokens for those in the generation phase, and removes requests that have completed. New requests only have to wait for the next forward pass (a few milliseconds) to be incorporated, virtually eliminating head-of-line blocking. This allows the system to achieve both high GPU utilization (by maintaining a large, dynamic batch) and extremely low TTFT.

  > **Napkin Math:** Let's analyze the latency for a user under static batching.

**System Parameters:**
- Arrival Rate (λ): 20 requests/second
- TTFT SLO: < 500ms
- GPU: H100
- Let's assume the time for one forward pass (token generation) on a large batch is ~5ms.

**Analysis of Static Batching (proposed size N=32):**
- The server must wait for 32 requests to arrive before starting a batch.
- Time to fill one batch = Batch Size / Arrival Rate = 32 req / 20 req/s = **1.6 seconds**.
- For a user who is the first to arrive in a new batch, their minimum wait time is 1.6 seconds before their request is even sent to the GPU.
- Total TTFT for this user is `Wait Time + GPU Prefill Time`.
- `TTFT > 1.6s`. This catastrophically violates the 500ms SLO.

**Analysis of Continuous Batching:**
- With continuous batching, a new request doesn't wait for a batch to fill. It waits for the *next iteration* of the scheduler.
- Scheduler iteration time ≈ Time for one forward pass ≈ 5ms.
- Average Wait Time in queue ≈ 5ms / 2 = **2.5ms**.
- Let's assume the GPU prefill computation for the new request (when added to the megabatch) takes ~40ms.
- Total TTFT ≈ `Avg. Wait Time + GPU Prefill Time` = `2.5ms + 40ms` = **~42.5ms**.
- This is well within the 500ms SLO, while the GPU remains highly utilized by servicing a large, constantly changing set of users.

  > **Key Equation:** L = \lambda W

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>
























#### 🟡 L5

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

  > **Napkin Math:** Single 100k pass: Total compute = `2 × 70e9 × 100,000` = 14 PFLOPs. At ~989 TFLOPS (peak, because 100k-token matmuls have astronomical arithmetic intensity), time ≈ 14.15s. Weights (140 GB) are streamed from HBM once per layer.
  > Chunked 25 passes: Same 14 PFLOPs total, but split into 25 chunks of 560 TFLOPs each. Each chunk must reload all 140 GB of model weights from HBM — 25 reloads × 140 GB = 3,500 GB of memory traffic. At 3.35 TB/s, that's 1.04s of pure weight streaming overhead (vs. 0.042s for one pass). Worse, the smaller 4k-token matmuls achieve lower effective TFLOPS (~600 vs ~989) due to reduced arithmetic intensity, stretching each chunk to ~0.93s. Total ≈ 25 × 0.93s ≈ 23s vs. 14.15s — a ~1.6× TTFT regression. The GPU shifts from a fully compute-saturated regime to a partially memory-bandwidth-limited one.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Batching Paradox</b> · <code>continuous-batching-queueing-theory</code></summary>

- **Interviewer:** "Your team implements a state-of-the-art continuous batching engine, like vLLM, for your LLM serving stack to increase GPU utilization. As expected, aggregate throughput (tokens/sec/GPU) skyrockets. However, you receive an alert that P99 Time-To-First-Token (TTFT) has degraded by 500%, violating your interactive chat SLO. Your dashboard shows GPU utilization is pegged at 100%. Evaluate this situation: why would an optimization that improves overall throughput cause a severe latency regression for newly arriving requests? Justify your reasoning with queueing theory."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The system is just overloaded and needs more GPUs. Or, the Python overhead of batching is too high. These are symptoms or minor factors, not the root cause. The core issue is a policy choice within the batcher itself that creates a specific type of queuing delay.

  **Realistic Solution:** The root cause is that the continuous batcher, in its aggressive pursuit of maximizing throughput, has created a head-of-line blocking scenario at the token level. To keep the GPU fed, the scheduler prioritizes filling every available slot in the next forward pass. Under high load, these slots are overwhelmingly filled by extending existing, long-running generation requests (optimizing for TPOT) rather than admitting new, incoming requests (which need a fast TTFT). A newly arrived request must wait for the current, large, and relatively slow forward pass to complete before it can be scheduled. The larger the batch configured to maximize throughput, the longer this 'batching tax' delay becomes for every new request, causing P99 TTFT to explode. The system is optimized for global throughput at the expense of fairness and latency for individual new tasks.

  > **Napkin Math:** Let's assess the 'batching tax' on TTFT. We'll use a 70B model on an H100.

1.  **Parameters:**
    *   Model Compute per Token: ~2 FLOPs * 70B Params = 140 GFLOPs/token
    *   GPU Peak Compute: ~989 TFLOPS (H100 FP16)

2.  **Scenario A (Low Load / Small Batch):** The batcher forms a small batch of 8 requests.
    *   Compute per forward pass: 8 requests * 140 GFLOPs/token/req = 1.12 TFLOPs.
    *   Time per forward pass: 1.12 TFLOPs / 989 TFLOPS ≈ **1.1 ms**.
    *   A new request waits at most 1.1 ms before its first token is processed. This is excellent TTFT.

3.  **Scenario B (High Load / Throughput-Optimized Batch):** The batcher is configured to maximize utilization and forms a large batch of 128 requests.
    *   Compute per forward pass: 128 requests * 140 GFLOPs/token/req = 17.92 TFLOPs.
    *   Time per forward pass: 17.92 TFLOPs / 989 TFLOPS ≈ **18.1 ms**.
    *   A new request arriving just as this batch is submitted must wait the full 18.1 ms before it can even be considered for the *next* batch. This wait time is a direct addition to its TTFT, representing a >16x increase in base latency before processing even begins.

  > **Key Equation:** L = \lambda W

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The SLO-Violating Deadline Scheduler</b> · <code>real-time-scheduling-queueing-theory</code></summary>

- **Interviewer:** "You run a multi-tenant LLM inference service. To ensure fairness, you design a strict deadline-based scheduler. Requests specify a deadline (e.g., 'generate 50 tokens within 2 seconds'), and the scheduler preemptively drops any request about to miss its deadline. After launch, your high-priority customers with short requests are happy. However, customers submitting long-running analysis jobs (e.g., 4000 tokens with a 60-second deadline) complain their jobs are almost always dropped, despite overall GPU utilization being only 50%. Assess this scheduler design. Why is it failing long jobs, and what underlying statistical phenomenon is being ignored?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The deadlines are too tight for the long jobs. Or, we just need more GPUs. The first is a configuration issue, not a design flaw. The second is incorrect because the system is underutilized on average.

  **Realistic Solution:** The scheduler's design is flawed because it fails to account for service time variance, a key concept in queueing theory. By always prioritizing requests with the nearest deadlines (the short jobs), it creates a system where long-running jobs are perpetually starved of compute. The arrival of many short, high-priority requests constantly interrupts the processing of the long jobs. Each interruption pushes the long job further down the queue, increasing its total time in the system until its generous deadline is finally breached. The core problem is that 50% *average* utilization is masking short, intense bursts of 100% utilization dedicated to the high-priority traffic. This high variance in service times (e.g., 20ms vs. 2000ms) leads to disastrously high wait times for the longer jobs, a predictable outcome in M/G/1 queues (queues with general service time distributions).

  > **Napkin Math:** Let's model the queueing behavior.

1.  **Parameters:**
    *   System Capacity: Can generate 2,000 tokens/sec.
    *   Traffic Mix: 95% 'short' jobs (40 tokens, 1s deadline), 5% 'long' jobs (3000 tokens, 60s deadline).

2.  **Service Time Calculation:**
    *   Short Job Service Time: 40 tokens / 2000 tokens/s = **20 ms**.
    *   Long Job Service Time: 3000 tokens / 2000 tokens/s = **1,500 ms**.

3.  **Queue Dynamics Analysis:**
    *   A 'long' job arrives. It needs 1.5s of compute. Its deadline is a far-off 60s.
    *   While it's waiting, a stream of 'short' jobs arrives. Since their deadlines are much closer, the scheduler always processes them first.
    *   To get its required 1.5s of execution time, the long job needs to find a 1.5s continuous gap where no high-priority jobs arrive. In a high-traffic system, this is statistically improbable.
    *   Instead, it might get 5ms of compute here, 10ms there, constantly being preempted. Its total time spent waiting in the queue skyrockets. After 60 seconds of wall-clock time, it has only accumulated a fraction of its required 1.5s of service, the deadline is breached, and the scheduler drops it. The system isn't failing due to a lack of *average* capacity, but a lack of *contiguous, non-preempted* capacity for low-priority tasks.

  > **Key Equation:** W_{M/G/1} = \frac{\lambda E[S^2]}{2(1-\rho)}

  📖 **Deep Dive:** [ML Operations](https://mlsysbook.ai/vol1/ops.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Speculative Decoding Memory Bomb</b> · <code>speculative-decoding-kv-cache-memory</code></summary>

- **Interviewer:** "To improve Time Per Output Token (TPOT), your team implements speculative decoding for a 70B model on H100s. You use a small draft model to generate 4 candidate tokens, and the 70B orchestrator model verifies them in a single, parallel forward pass. In staging, TPOT improves by 2.5x. You deploy to production. Shortly after, the service suffers a cascading failure from OOM errors, even though the number of concurrent users did not increase. Your observability tools show that peak GPU memory usage *per request* has inexplicably quadrupled. Critique this implementation. Why did an optimization designed to reduce latency cause a catastrophic memory failure?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The draft model itself uses too much memory. A 7B model uses ~14GB, which is significant, but it's a fixed cost and doesn't explain the per-request memory explosion. Another mistake is to blame weight loading overhead, which affects latency, not memory capacity.

  **Realistic Solution:** The catastrophic failure stems from a misunderstanding of the memory access pattern during the verification step. The 4x increase in memory per request is not a coincidence; it's a direct result of the speculation length (k=4). To verify 4 candidate tokens in parallel, the attention mechanism's internal data structures grow proportionally. The query tensor becomes `[batch, k, dim]` instead of `[batch, 1, dim]`, and the intermediate attention score matrices become `k` times larger. These temporary activation tensors, which must exist in HBM during the forward pass, are the source of the memory explosion. The system was provisioned for the memory profile of standard decoding, where activations grow linearly with sequence length. It was not provisioned for the multiplicative peak memory required by the parallel verification step of speculative decoding, causing any GPU serving a moderate number of users to instantly OOM.

  > **Napkin Math:** Let's analyze the peak activation memory during one forward pass.

1.  **Parameters:**
    *   Speculation length (k): 4.
    *   Let's assume for a long-sequence request, the model weights take 50GB and the KV cache + other activations take 25GB of HBM on an 80GB H100, leaving a 5GB buffer.

2.  **Scenario A (Standard Decoding):**
    *   To generate one token, the model computes attention for a single new query vector.
    *   Peak Memory Usage: 50GB (weights) + 25GB (activations) = **75 GB**.
    *   The system is stable.

3.  **Scenario B (Speculative Decoding):**
    *   To verify `k=4` tokens, the model computes attention for 4 query vectors in parallel against the existing KV cache.
    *   Key intermediate data structures, like the query-key attention matrix, are now `k` times larger than in standard decoding. This bloats the activation memory footprint.
    *   Peak Activation Memory (approximate): 25GB * k = 25GB * 4 = **100 GB**.
    *   Peak Total Memory: 50GB (weights) + 100GB (peak activations) = **150 GB**.
    *   This massively exceeds the H100's 80GB HBM capacity, causing an immediate OOM error for any request being speculatively decoded. The system fails not because average memory grew, but because the instantaneous peak memory during the verification step became insurmountably large.

  > **Key Equation:** \text{Memory}_{\text{peak}} \propto \text{Memory}_{\text{params}} + N_{\text{users}} \times (k \times \text{Memory}_{\text{activations}})

  📖 **Deep Dive:** [Training](https://mlsysbook.ai/vol1/training.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Batching Paradox</b> · <code>continuous-batching-latency</code></summary>

- **Interviewer:** "Your team is serving a 70B parameter summarization model on H100s. To improve GPU utilization, you replace static batching with a continuous batching strategy (like vLLM's). Average throughput triples, and utilization is stable at 80%. However, you receive an alert that P99 Time-To-First-Token (TTFT) has increased by 500ms, violating your 250ms SLA for interactive (short prompt) users. Your manager asks you to assess the situation. Why would a technique designed to improve performance catastrophically degrade P99 TTFT for your most sensitive users?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming Python overhead, network latency, or other microsecond-scale factors. These are orders of magnitude too small to explain a 500ms regression. Another common mistake is suggesting turning off continuous batching, which throws away the massive throughput gains.

  **Realistic Solution:** This is a classic 'Head-of-Line Blocking' problem introduced by a naive FIFO continuous batcher. Continuous batching combines requests that arrive close together into a single dynamic batch. If a short, fast interactive query (e.g., 50 tokens) arrives right after a long, slow summarization query (e.g., 4000 tokens), the batcher groups them. The GPU processes the entire batch in parallel, but the total time is dictated by the longest job. The short query is 'stuck' waiting for the long query to finish its prefill, even though its own compute cost is trivial. This makes the system 'fair' from a FIFO perspective but disastrous for latency-sensitive applications. The fix is not to remove continuous batching, but to evolve it: implement priority queuing where high-priority (short) requests can form their own batches, preempting or bypassing the queue of long-running, low-priority jobs.

  > **Napkin Math:** Let's model the prefill stage for a 70B model on an H100.
- **Long Summarization (4k tokens):** Prefill compute is roughly `2 * 70B params * 4000 tokens = 5.6e14 FLOPs`. On an H100 at its peak 989 TFLOPS, this takes `560 TFLOPS / 989 TFLOPS ≈ 566 ms`.
- **Short Interactive (50 tokens):** Prefill compute is `2 * 70B params * 50 tokens = 7e12 FLOPs`. This takes `7 TFLOPS / 989 TFLOPS ≈ 7 ms`.

**Scenario:** The 7ms query gets batched with the 566ms query. The entire batch takes ~566ms to process. The TTFT for the short query is no longer 7ms, but `(Batch Processing Time) + (scheduling overhead)`, which is now dominated by the 566ms job. Its perceived latency just increased by over 80x.

  > **Key Equation:** T_{\text{wait}} = T_{\text{batch_processing}} - T_{\text{own_processing}}

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Throughput-Optimized Cascade Failure</b> · <code>queueing-theory-tpot-ttft</code></summary>

- **Interviewer:** "You are tasked with optimizing a Llama-70B serving cluster for maximum cost-efficiency. You profile the system and find that a large static batch size of 64 maximizes throughput (TPOT) and achieves 95% GPU utilization. You deploy this configuration. The system runs smoothly during low-traffic periods. However, during the daily traffic peak, the entire service collapses: request queues grow infinitely, all requests time out, and the system requires a full restart. Your VP wants a post-mortem. Evaluate the design decision. Why did optimizing for a peak hardware metric (TPOT) lead to a catastrophic system-level failure?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Claiming the GPUs are underpowered or that there was a bug in the code. This was not a hardware capacity or a logic error; it was a fundamental queuing theory failure. Suggesting a smaller static batch size is a partial fix, but doesn't address the core issue with the static batching paradigm itself under fluctuating load.

  **Realistic Solution:** The team optimized for the wrong metric. While high TPOT and GPU utilization are good for offline batch processing, they are often toxic for an online, interactive service. A large static batch size fundamentally imposes a high Time-To-First-Token (TTFT) because every request must wait for the batch to fill. This initial wait time is the 'W' in Little's Law ($L = \lambda W$). By fixing the batch size at 64, you guaranteed a multi-second wait time (`W = 64 / \lambda`) even before the request hits the GPU. When real users experience high TTFT, they often retry or abandon the session, which ironically increases the arrival rate ($\lambda$). This creates a vicious cycle: higher arrival rate -> longer time to fill the *next* batch (if the server is saturated) -> even higher TTFT -> more retries -> runaway queue length (L). The system didn't just get slow; it entered a positive feedback loop and became unstable. The correct approach is to prioritize P99 TTFT as the primary SLO and use a dynamic or continuous batching strategy that minimizes wait time, even at the cost of slightly lower peak GPU utilization.

  > **Napkin Math:** Let's apply Little's Law ($L = \lambda W$).
- **Configuration:** Static batch size of 64.
- **Low Load:** Arrival rate $\lambda = 5$ requests/sec. The time to fill the batch is `W = 64 / 5 = 12.8` seconds. This is the *minimum* TTFT. The average queue length `L = 5 * 12.8 = 64` requests.
- **Peak Load:** Arrival rate spikes to $\lambda = 10$ requests/sec. Time to fill is `W = 64 / 10 = 6.4` seconds. The queue length is `L = 10 * 6.4 = 64` requests.

The key insight is that even in the best case, users are waiting 6-13 seconds for their first token. In a real-world chat application, a 2-second TTFT is considered slow. A 13-second TTFT will trigger a massive wave of user retries, causing the real $\lambda$ to surge far beyond the organic rate. If the service time per batch is greater than `1/$\lambda$`, the queue length `L` will grow infinitely, leading to cascade failure.

  > **Key Equation:** L = \lambda W \quad (\text{Little's Law})

  📖 **Deep Dive:** [ML Operations](https://mlsysbook.ai/vol1/ops.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Batching Paradox</b> · <code>continuous-batching-latency-throughput</code></summary>

- **Interviewer:** "Your team just deployed a new LLM inference server using continuous batching to improve GPU utilization for a 7B parameter model on an H100. As expected, average throughput (tokens/sec) has increased by 50% and GPU utilization is stable at 95%. However, you receive an alert that P99 Time-To-First-Token (TTFT) has increased from 150ms to 400ms, violating your 250ms SLA. Your on-call playbook suggests increasing the maximum batch size to improve throughput further. Justify why the playbook is wrong and predict why this 'more efficient' batching scheme has degraded your P99 latency."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The overhead of the continuous batching scheduler is too high. Python is slow. This answer is too generic and misses the specific scheduling dynamics of batching systems.

  **Realistic Solution:** The playbook is wrong because increasing batch size will likely make P99 latency even worse. The core issue is **Head-of-Line Blocking within a batch**. Continuous batching combines requests to fill the GPU. If a request that will generate many tokens (a 'long' request) is batched with requests that will generate few tokens ('short' requests), the short requests are now trapped. They cannot complete until the long request also completes its generation steps. While the GPU is always busy (high throughput), the variance of completion times for short requests skyrockets because they are periodically held hostage by long ones. This dramatically lengthens the tail of the latency distribution (P99), even if the average latency is acceptable.

  > **Napkin Math:** Let's model the trap. An H100 can run a 7B model generation step in roughly 15ms for a large batch. A 'short' request needs 5 tokens (e.g., a chat response). A 'long' request needs 500 tokens (e.g., summarization).
- **Without batching:** The short request would take ~5 steps * 15ms/step = 75ms.
- **With continuous batching:** A batch forms containing one 'long' and multiple 'short' requests. The short requests are stuck in the batch for all 500 generation steps of the long request. Their completion time is now tied to the long request: 500 steps * 15ms/step = 7,500ms.
Even if the batcher can add/remove requests, the initial cohort of short requests is still delayed significantly as long as they share iteration steps with the long job. The system sacrifices predictable low latency for high aggregate throughput.

  > **Key Equation:** $\text{Wait}_{P99} \gg \text{Wait}_{P50} \text{ when } \sigma^2_{\text{service_time}} \text{ is high}$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The In-Flight Priority Queue Failure</b> · <code>queueing-theory-preemption-sla</code></summary>

- **Interviewer:** "You run a multi-tenant LLM API on a cluster of H100s. 'Premium' users have a strict P99 TTFT SLA of 200ms. To enforce this, you implemented a priority queue, ensuring premium requests always go to the front of the line. However, you are still paying SLA violation penalties. Digging into the logs, you confirm a premium request arrived, was immediately placed at the front of the queue, but its TTFT was still 500ms. The model, a 13B parameter LLM, has not changed. The system is busy but not overloaded. Evaluate the design of your priority queue. Why is it insufficient for guaranteeing the SLA, and what physical constraint of the hardware is it failing to account for?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The network latency is too high, or the server is overloaded. This ignores the explicit evidence that the queueing worked and the system isn't overloaded. It's an easy but incorrect excuse.

  **Realistic Solution:** The priority queue is insufficient because it can only reorder the *waiting room* of requests; it cannot preempt a **batch already in-flight on the GPU**. LLM inference is not a continuously divisible process; it proceeds in discrete, atomic generation steps (iterations). Once the GPU begins a generation step for a batch, that computation must run to completion, which can take 50-100ms for a large batch. Your premium request is experiencing 'service time' blocking. It gets to the front of the line, but has to wait for the *current* batch of lower-priority users to finish its ongoing, non-interruptible computation step. The wait time is dominated not by queueing, but by waiting for the GPU to become available again. The priority queue design correctly addresses queueing delay but fails to account for the physical, non-preemptible service time of the hardware.

  > **Napkin Math:** Let's assume your system uses a continuous batcher that starts a new iteration every 50ms on a 13B model. The compute time for this batch iteration is 40ms. The GPU is therefore busy for 40ms out of every 50ms cycle.
A premium request arrives at a random time. Its wait time is composed of two parts:
1.  **Queue Wait:** Your priority queue makes this ~0ms.
2.  **In-Flight Batch Wait:** The request must wait for the current 40ms computation to finish. The arrival is random, so the average wait is half the service time: 40ms / 2 = 20ms. But in the worst case (P99), it arrives just as a batch starts, so it waits the full 40ms.
3.  **Batching Delay:** It must also wait for the *next* batch to form, which could be up to 50ms.
Worst-case TTFT = (Queue Wait) + (In-Flight Wait) + (Batching Delay) + (Own Compute) = 0ms + 40ms + 50ms + 40ms = 130ms. This is within the 200ms SLA, but what if the service time varies? If a batch has a very long sequence length and the iteration takes 150ms? The P99 wait time approaches the P99 service time of existing batches. Worst-case TTFT could be 0 + 150ms + 50ms + 150ms = 350ms, clearly violating the 200ms SLA. The system needs true preemption, not just priority queueing.

  > **Key Equation:** $W_{p99} = W_{\text{queue}} + W_{\text{service}} \approx W_{\text{service-p99}}$

  📖 **Deep Dive:** [ML Operations](https://mlsysbook.ai/vol1/ops.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Batching Paradox</b> · <code>continuous-batching-queueing-theory</code></summary>

- **Interviewer:** "Your team just deployed a new LLM serving system using continuous batching on H100s. The goal is to serve two traffic types: real-time, low-latency chatbot queries (short prompts) and large, offline document summarization jobs (long prompts). After launch, you see average throughput has increased significantly, but you get an alert that P99 latency for the chatbot has degraded by 500%. Your manager is confused: 'I thought continuous batching was supposed to *solve* Head-of-Line blocking. How can throughput be up, but latency for our most important users be so much worse?' Justify the root cause of this non-linear system degradation."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the KV cache size. While KV cache is a constraint, it doesn't explain why short, low-memory-footprint requests are the ones that suffer. Another common but incorrect answer is 'Python overhead' or 'network latency', which are orders of magnitude too small to cause a 500% degradation.

  **Realistic Solution:** The system is experiencing scheduler-induced Head-of-Line Blocking, a more subtle variant that continuous batching alone doesn't solve. The default scheduler is likely FIFO ('first-in, first-out') to maximize hardware utilization. When a burst of long summarization jobs arrives, they fill up the batch slots. Even with continuous batching, a new, short chatbot query is blocked until one of the long jobs finishes its *entire*, multi-second prefill and generation cycle. The system's throughput is high because the GPU is always busy with the long jobs, but the queue time for short, latency-sensitive requests explodes. The fix is to implement a priority-based scheduler that reserves a portion of the batch capacity for high-priority (e.g., short, interactive) requests, or to create separate queues and even separate GPU pools for the different traffic classes.

  > **Napkin Math:** Let's model the system. A 70B model on an H100. A long prompt is 8k tokens, a short one is 50. Prefill for an 8k prompt is compute-heavy: `2 * 70B * 8000 tokens ≈ 1.12 PetaFLOPs`. On an H100 (989 TFLOPS), this prefill alone takes `1.12e15 / 989e12 ≈ 1.13 seconds`. If 16 such jobs arrive and fill a batch, they will occupy the GPU for over a second just for prefill. A new short chatbot query arriving during this time must wait in the queue for this entire duration, plus any decode time. Its TTFT is now >1 second, a 500%+ degradation from a baseline of <200ms. The GPU is 100% utilized, so throughput metrics look great, but the user experience for the chatbot is destroyed.

  > **Key Equation:** L = \lambda W

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The TTFT vs. TPOT Tug-of-War</b> · <code>ttft-tpot-tradeoff-roofline</code></summary>

- **Interviewer:** "You are designing the inference engine for a new AI code completion product. The product manager's primary requirement is 'instantaneous suggestions that stream out quickly.' You are serving a 70B parameter model on H100s. After initial tests, you report that you can achieve a very low Time-To-First-Token (TTFT) with a batch size of 1, but the Time-Per-Output-Token (TPOT) is poor, making the code stream slowly. Conversely, with a large batch size, TPOT is excellent, but requests get stuck in a queue, making TTFT unacceptable. The PM pushes back, 'This is a flagship product, we can't compromise. Justify with physics why we can't have both, and propose a system design that can satisfy this seemingly impossible requirement.'"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Suggesting a hardware-only solution like 'use a B200'. While faster hardware improves all metrics, it does not change the fundamental, underlying physics of the trade-off. Another is simply 'use a smaller model,' which evades the core systems challenge of optimizing for conflicting requirements with the given model.

  **Realistic Solution:** The trade-off is rooted in the Roofline Model and the changing arithmetic intensity of the workload. TTFT is dominated by the prefill step, which for a single request (batch size 1) is latency-bound and has minimal queue wait. However, the subsequent decode step (TPOT) becomes memory-bandwidth-bound at batch size 1, as the GPU spends most of its time waiting to read the 140GB of model weights from HBM, leading to poor utilization and slow token generation. Conversely, a large batch size makes the decode step compute-bound, maximizing GPU utilization and leading to excellent TPOT. However, this large batch introduces queueing delays, which kills TTFT.

A viable system design is a multi-stage or hybrid strategy:
1. **Speculative Execution:** Immediately respond with a suggestion from a much smaller, distilled model that can run at batch size 1 with minimal latency. This provides the 'instant' TTFT.
2. **Quality Path:** Simultaneously, the full prompt is sent to the large 70B model running with an optimized large batch for high TPOT. Once this model begins generating, its high-quality output replaces the speculative one. This gives the 'streams out quickly' experience without compromising on the initial perceived latency.

  > **Napkin Math:** Let's analyze the decode step for a 70B model (140GB weights) on an H100 (3.35 TB/s HBM bandwidth). For each token at batch size 1, the GPU does `2 * 70B = 140 GFLOPs` of compute but must read `140 GB` of weights. The arithmetic intensity is `140e9 FLOPs / 140e9 Bytes = 1 FLOP/Byte`. This is far below the H100's ridge point (~295 Ops/Byte), making it severely memory-bound. The time is dominated by the weight read: `140 GB / 3.35 TB/s ≈ 42 ms` per token. With a large batch size of 64, the math becomes `64 * 140 GFLOPs` for the same `140 GB` weight read. The intensity is now `64 FLOPs/Byte`, pushing the operation towards being compute-bound and achieving much better hardware efficiency and a lower per-token generation time (better TPOT).

  > **Key Equation:** I = \frac{\text{FLOPs}}{\text{Bytes}}

  📖 **Deep Dive:** [Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Real-Time Queue Collapse</b> · <code>queueing-theory-deadlines-load-shedding</code></summary>

- **Interviewer:** "You run an LLM-based system for a hedge fund that analyzes market news in real time. Each request has a hard 150ms P99 deadline from the moment it's sent; if a response takes longer, it's useless. The service runs on H100s with continuous batching. During a market volatility event, the request rate triples. The on-call engineer reports that the rate of dropped requests (those missing the deadline) has skyrocketed. More alarmingly, the absolute number of *successful* requests served *within* the deadline has fallen below the pre-surge baseline. The system hasn't crashed, but its useful throughput has collapsed. Explain this phenomenon and critique the existing system's lack of a critical feature."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The brute-force answer: 'We need more GPUs.' While adding capacity would help, it doesn't explain why the system's useful throughput *decreased*. A well-designed system should plateau, not collapse. Another incorrect diagnosis is 'network saturation,' as the core failure mechanism is within the serving logic itself.

  **Realistic Solution:** The system is experiencing a 'death spiral' caused by processing stale requests. When the arrival rate exceeds the system's capacity, a queue forms. Without deadline awareness, the server continues to pull from the head of the queue in a FIFO manner. A request might wait 120ms in the queue, then get processed by the GPU for 50ms. The total time is 170ms, which misses the 150ms deadline. The critical failure is that the system spent 50ms of valuable H100 time on a request that was already doomed. This wasted work prevents the GPU from serving a newer request that *could* have met the deadline. As the queue grows, the system spends more and more of its capacity on expired work, causing the useful throughput to plummet. The critical missing feature is **deadline-aware load shedding**. The system must:
1. **Check at Admission:** When a request arrives, estimate the current wait time. If `estimated_wait + processing_time > deadline`, reject it immediately (e.g., HTTP 503).
2. **Cancel in Queue:** Actively prune requests from the waiting queue if their time-in-system has already exceeded the deadline. This stops the death spiral by ensuring GPU cycles are only spent on viable work.

  > **Napkin Math:** Let's assume the H100 can process a batch of 32 requests with an average per-request processing time of 50ms. The system's maximum capacity, μ, is `32 requests / 50ms = 640 req/s`. The baseline arrival rate, λ, is `300 req/s`. Utilization ρ is `300/640 ≈ 47%`, and queueing is minimal. During the surge, λ triples to `900 req/s`. Now ρ is `900/640 ≈ 140%`. The system is overloaded and the queue grows. After just one second of the surge, there are `900 - 640 = 260` requests backlogged. A new request arriving at t=1.0s will have to wait for those 260 requests to be processed. The wait time will be at least `260 requests / 640 req/s ≈ 400ms`. This is far beyond the 150ms deadline. The server, however, will dutifully process these 260 stale requests before even starting the new one, guaranteeing that all of them miss their deadline and waste capacity.

  > **Key Equation:** \rho = \frac{\lambda}{\mu} > 1

  📖 **Deep Dive:** [ML Operations](https://mlsysbook.ai/vol1/ops.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Batching Paradox</b> · <code>continuous-batching-queueing</code></summary>

- **Interviewer:** "Your team is responsible for a real-time translation service for a major international conference. The SLA is strict: translations for a speaker's sentence (avg. 50 tokens) must appear on screen in under 500ms (P99). The service runs on H100s using a continuous batching server. To maximize GPU utilization and overall system throughput, a junior engineer tunes the server's `max_batched_tokens` setting to 8192. `nvidia-smi` now shows a beautiful, sustained 95% utilization. However, user complaints flood in about the translation lagging far behind the speaker. You receive an alert: P99 latency has spiked to over 1.5 seconds. The junior engineer is confused, 'But the GPU is more efficient than ever!' Assess the situation. Why did increasing batch size and GPU utilization lead to a catastrophic failure to meet the real-time deadline?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Blaming the model architecture, network latency, or the overhead of the Python batching loop. These are components of latency, but they don't explain why a throughput-optimization *caused* the latency spike. The core issue is a misunderstanding of the trade-off between aggregate throughput and per-request latency.

  **Realistic Solution:** This is a classic queueing theory failure. By setting `max_batched_tokens` too high for a low-latency workload, the continuous batching engine waits too long to accumulate a 'full' batch from incoming requests before launching a forward pass. This is Head-of-Line Blocking within the dynamic batch. While this is great for amortizing kernel launch overhead and maximizing the arithmetic intensity of the computation (high throughput), it introduces a fatal queueing delay for the individual requests. For a real-time service with a strict P99 deadline, per-request latency is the primary metric, not aggregate system throughput. The system was correctly optimized for the given metric (throughput), but that metric was wrong for the user-facing SLA. The fix is to reduce `max_batched_tokens` to a level where the queueing delay is a small fraction of the overall latency budget.

  > **Napkin Math:** Let's model the queueing delay. Assume peak traffic is 100 requests/second, and each request averages 50 tokens. The token arrival rate is `100 RPS * 50 tokens/req = 5,000 tokens/second`.

1.  **Calculate Queueing Delay:** The time the server must wait to form a batch is the batch size divided by the arrival rate.
    `T_queue = max_batched_tokens / token_arrival_rate`

2.  **Evaluate the 'Optimized' Setting:** With `max_batched_tokens = 8192`:
    `T_queue = 8192 tokens / 5000 tokens/s = 1.64 seconds`
    The very first request in a batch has to wait 1.64 seconds *before the GPU even starts processing*. This alone violates the 500ms SLA catastrophically.

3.  **Evaluate a 'Corrected' Setting:** Let's budget 100ms for queueing.
    `max_batched_tokens = T_queue * token_arrival_rate = 0.1s * 5000 tokens/s = 500 tokens`
    A much smaller max batch size of ~500-1024 tokens would keep queueing delay manageable while still providing a large enough batch to achieve reasonable GPU utilization.

  > **Key Equation:** $\text{T}_{\text{total}} = \underbrace{ \frac{\text{N}_{\text{batch_tokens}}}{\lambda_{\text{tokens}}} }_{\text{Queue Delay}} + \underbrace{ \text{T}_{\text{prefill}} + \text{N}_{\text{gen}} \times \text{T}_{\text{decode}} }_{\text{GPU Processing}} $

  📖 **Deep Dive:** [Inference and Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The SLO Violation Cascade</b> · <code>serving-latency-queueing</code></summary>

- **Interviewer:** "Your team runs a multi-tenant LLM serving platform on H100 GPUs. It must handle two distinct workloads:
1. **Interactive Chat:** High-priority requests with a strict P99 Time-To-First-Token (TTFT) SLO of 150ms.
2. **Batch Summarization:** Low-priority jobs that process 100k+ token documents, where overall throughput (TPOT) is the goal.

To maximize hardware utilization, your team implements a state-of-the-art continuous batching engine. The scheduler is designed to be work-conserving and will immediately begin processing a high-priority chat request as soon as it arrives, even if a low-priority batch job is in the middle of its generation sequence. To manage the large batch jobs, they are processed using chunked prefill, with a chunk size of 4,096 tokens.

During a sudden surge in summarization jobs, the on-call engineer gets an alert: P99 TTFT for the interactive chat workload has spiked to over 600ms, catastrophically violating the 150ms SLO. `nvidia-smi` shows 100% GPU utilization. The on-call playbook suggests scaling up more H100 replicas. Evaluate this situation. Why is the playbook's recommendation likely wrong, and what is the underlying physical reason for the TTFT degradation despite the high-priority scheduling?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common incorrect diagnosis is simply 'the GPU is overloaded, so we need more of them.' This mistakes high utilization for productive work. Engineers often assume 100% utilization means the system is at its global throughput limit, failing to see that a single low-priority task can create a 'head-of-line blocking' scenario that starves high-priority tasks of their latency budget, even if the scheduler is 'prioritizing' them in software.

  **Realistic Solution:** The playbook is wrong because simply adding more replicas will not solve the core problem; it will just create more GPUs that exhibit the same failure mode. The root cause is **Head-of-Line Blocking at the CUDA kernel level**.

The continuous batching scheduler operates in software, but a CUDA kernel, once launched, is generally non-preemptible. When a high-priority chat request arrives, the scheduler might have *just* launched a 4,096-token prefill kernel for a low-priority batch job. The OS and the CUDA runtime cannot interrupt this kernel. The 'high-priority' chat request is forced to wait in a queue for the entire duration of the batch job's prefill chunk.

This creates a latency cascade. The maximum wait time for a chat request is no longer just its own processing time, but is now bounded by the execution time of the largest, non-preemptible work unit in the system—the batch prefill. The system's P99 TTFT is now directly proportional to the duration of a low-priority chunk computation, which is orders of magnitude larger than the chat SLO. The correct solution is to either physically isolate the workloads onto different GPUs (sacrificing utilization) or, more practically, to drastically reduce the maximum prefill chunk size for batch jobs, creating a trade-off between batch throughput and guaranteeing a ceiling on high-priority wait times.

  > **Napkin Math:** Let's assess the duration of the blocking low-priority kernel. We are evaluating a 70B parameter model on an H100.

1.  **Calculate Prefill Compute:** The compute required for a transformer prefill is approximately $C \approx 2 \times \text{Parameters} \times \text{Sequence Length}$.
    - Parameters ($P$): 70B
    - Chunk Size ($S$): 4,096 tokens
    - $C \approx 2 \times (70 \times 10^9) \times 4096 \approx 5.73 \times 10^{14}$ FLOPs, or 573 TFLOPs.

2.  **Check for Bottleneck (Arithmetic Intensity):** We compare the compute-to-memory ratio against the H100's ridge point.
    - Model Weights (FP16): $70 \times 10^9 \times 2 \text{ bytes} = 140$ GB.
    - Arithmetic Intensity (AI) = FLOPs / Bytes_Read = $(573 \times 10^{12}) / (140 \times 10^9) \approx 4092$ FLOPs/Byte.
    - From our constants, the H100 Ridge Point is ~295 Ops/Byte.
    - Since AI (4092) > Ridge Point (295), the operation is **compute-bound**, not memory-bound. The GPU will be running near its peak compute speed.

3.  **Calculate Kernel Execution Time:** We divide the required FLOPs by the H100's peak performance.
    - H100 FP16 Peak Performance: 989 TFLOPS.
    - Time = Total FLOPs / Peak TFLOPS = $573 / 989 \approx 0.58$ seconds.

4.  **Evaluate SLO Impact:** The execution time is ~580ms. If a high-priority chat request gets unlucky and arrives just as this kernel is launched, it must wait **at least 580ms** before its own prefill can even begin. This single, non-preemptible operation completely shatters the 150ms P99 TTFT SLO. Adding more GPUs just increases the probability that another GPU will also enter this state.

  > **Key Equation:** $W_{\text{max, high-pri}} \approx T_{\text{service, low-pri}} + T_{\text{service, high-pri}}$

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Batching Death Spiral</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "Your team runs a real-time translation service using a 70B parameter model on H100 GPUs. The business requires a strict P99 Time-To-First-Token (TTFT) of under 500ms. To improve cost-efficiency, you replace the simple static batching server with a state-of-the-art continuous batching engine. Under normal load, the results are fantastic: GPU utilization jumps from 40% to 85%, and P99 TTFT sits comfortably at ~250ms. However, during a traffic spike, P99 TTFT doesn't just degrade—it explodes to over 8 seconds. Your dashboard shows the GPU is pegged at 99% utilization, which the team celebrates as a sign of peak efficiency. Evaluate this situation. Why did an optimization designed for high throughput result in a catastrophic latency failure, and what fundamental law of systems have your team forgotten?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to focus only on the GPU's execution time. Engineers might say 'the batch became too large, increasing per-token latency' or 'the KV cache evicted and had to be recomputed'. While these can be factors, they don't explain a jump from milliseconds to multiple seconds. The core error is viewing the system as just a GPU, rather than as a complete queueing system, and misinterpreting high utilization as a universal good.

  **Realistic Solution:** This is a classic 'Queueing Death Spiral'. The team forgot that any service with a finite capacity is governed by Queueing Theory. The continuous batching engine, while efficient, has a maximum service rate (μ), measured in requests per second. Under normal load, the arrival rate (λ) is well below μ, so the queue of incoming requests is short, and wait times are minimal.

During the spike, λ approached and then exceeded μ. According to the formula for wait time in a queue, which is proportional to `1 / (1 - ρ)` where ρ is utilization (λ/μ), the time a request spends waiting *before it even gets to the GPU* explodes non-linearly as utilization approaches 100%. The 8-second TTFT isn't the model's inference time; it's ~50ms of inference time preceded by over 7.9 seconds of waiting in the request queue.

The team celebrated 99% utilization, but for a latency-sensitive service, this is a red alert. It means the system has no spare capacity to absorb even the slightest variance in traffic, guaranteeing that a long queue will form. The fix is not to tweak the model, but to treat the system as a whole: implement aggressive load shedding, provision more replicas to increase the total system capacity (μ), or shape traffic at the load balancer *before* it can form a catastrophically long queue at the server.

  > **Napkin Math:** Let's model the system. A single H100 serving a 70B model might have a maximum sustainable service rate (μ) of around 15 requests/second.

**1. Normal Load:**
- Arrival Rate (λ): Let's say 9 requests/second.
- Utilization (ρ): `ρ = λ / μ = 9 / 15 = 0.6` (60% utilization).
- Average Wait Time in Queue (W_q): Using the M/M/1 queue approximation, `W_q ≈ (1/μ) * (ρ / (1-ρ)) = (1/15) * (0.6 / 0.4) = 0.066s * 1.5 = 100ms`.
- Total TTFT ≈ Wait Time + Prefill Time. Let's assume prefill is ~150ms. `TTFT ≈ 100ms + 150ms = 250ms`. This matches the observation.

**2. Peak Load:**
- The arrival rate (λ) spikes to 14.5 requests/second.
- New Utilization (ρ): `ρ = 14.5 / 15 = 0.967` (96.7% utilization).
- New Average Wait Time (W_q): `W_q ≈ (1/15) * (0.967 / (1-0.967)) = 0.066s * (0.967 / 0.033) ≈ 0.066s * 29.3 ≈ 1.93 seconds`.
- New TTFT ≈ `1930ms + 150ms = 2.08 seconds`. The wait time has already grown 20x while traffic only went up by ~60%.

If λ ≥ μ (e.g., 16 req/sec), the queue grows infinitely long in theory. An 8-second TTFT indicates a request waited ~7.85s in the queue, meaning the system was running at or above its absolute capacity for a sustained period.

  > **Key Equation:** $$ W_q \approx \frac{\rho}{\mu(1-\rho)} \quad \text{where} \quad \rho = \frac{\lambda}{\mu} $$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The High-Priority Queue Stall</b> · <code>ttft-tpot-continuous-batching-queueing-theory</code></summary>

- **Interviewer:** "Your team runs a real-time LLM-based agent for a financial services client. The agent must respond to urgent market events with a strict P99 Time-To-First-Token (TTFT) SLA of 100ms. To maximize hardware utilization, your team implemented a state-of-the-art continuous batching inference server. The server's dashboard looks great: GPU utilization is consistently high, and the aggregate throughput (tokens per second) is near the hardware's peak. However, the client is reporting that during periods of high background traffic, the agent is missing its 100ms deadline for critical alerts. You dig into the logs and confirm: a high-priority request that arrived when the system was under load had a TTFT of nearly 150ms. Evaluate the current system design. Why does an optimization designed to increase throughput lead to a catastrophic failure in latency, and what do you predict is the primary cause of the excessive delay?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to blame the model size or the hardware ('the H100 is too slow for the prefill'). This focuses only on the execution time of the priority request itself, ignoring the time it spent waiting. Another incorrect path is to say 'Python overhead' or 'network latency', which are orders of magnitude too small to account for a ~50ms regression.

  **Realistic Solution:** The continuous batching server, in its quest to maximize throughput, has introduced a classic queueing theory problem: **Head-of-Line Blocking**. The server assembles a large batch of requests to maximize arithmetic intensity and saturate the GPU. When the high-priority request arrives, the server has likely just started processing a large, already-formed batch (e.g., 32 or 64 other requests). The new request must wait in the queue for the *entire ongoing batch* to complete at least one forward pass (the generation of one token). This wait time, not the execution of the request itself, is the dominant factor in the total TTFT and the cause of the SLA violation. The system is optimized for average throughput (TPOT) at the expense of worst-case latency (P99 TTFT). A real-time system requires priority-based preemption or dynamic batch splitting to pause or finish the low-priority work and immediately service the high-priority request.

  > **Napkin Math:** Let's assume a 70B model on an H100 and a continuous batching server that has formed a batch of 32 requests.
1. **Compute per Token:** A 70B model requires `~2 * 70B = 140 GFLOPs` per token generated.
2. **Batch Compute Step:** For the whole batch of 32, one token generation step requires `32 requests * 140 GFLOPs/request = 4.48 TFLOPs`.
3. **Wait Time (Head-of-Line Blocking):** An H100 (989 TFLOPS) processes this batch step in `4.48 TFLOPs / 989 TFLOPS ≈ 4.5 ms`. This is the time our high-priority request must wait if it arrives just as a batch is dispatched.
4. **Priority Request Prefill:** Assume the high-priority alert has a 1,000 token prompt. The prefill compute is `2 * 70B * 1000 = 140 TFLOPs`.
5. **Prefill Execution Time:** The H100 executes this prefill in `140 TFLOPs / 989 TFLOPS ≈ 141.5 ms`.
6. **Total TTFT:** The total time to first token for the priority request is `Wait Time + Prefill Time = 4.5 ms + 141.5 ms = 146 ms`.
This 146ms violates the 100ms SLA. The system failed not because the prefill was slow, but because the 4.5ms of head-of-line blocking pushed the total time over the budget.

  > **Key Equation:** $W_q = L_q / \lambda$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>
















#### 🔴 L6+

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Continuous Batching Stall</b> · <code>queueing</code></summary>

- **Interviewer:** "You implement continuous batching (iteration-level scheduling) for an LLM endpoint to improve throughput. Under low load, TTFT (Time To First Token) is 100ms. Under high load, your throughput triples, but users start complaining that TTFT spikes to over 4 seconds, even though token generation speed (TPOT) remains fast. What scheduling flaw caused this?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming that because continuous batching optimizes VRAM and throughput, it automatically guarantees low latency for incoming requests."

  **Realistic Solution:** You starved the prefill phase. In LLM serving, a request has two phases: Prefill (processing the prompt, compute-bound) and Decode (generating tokens, memory-bound). With continuous batching, an incoming request is injected into the very next iteration of the running batch. However, the prefill phase for a new prompt requires a massive matrix multiplication that consumes almost all the GPU's compute. If you inject a new 2,000-token prompt into a running batch, the GPU must pause decoding for everyone else to crunch that prefill. To prevent this, schedulers often limit the prefill chunk size or delay new prefills. If delayed too aggressively under high load, incoming requests wait in the queue for seconds, destroying TTFT.

  > **Napkin Math:** A 2000-token prefill on a 70B model requires `2 * 70B * 2000 = 280 TFLOPs`. An A100 (FP16) does ~312 TFLOPs. That prefill takes almost `1 full second` of dedicated compute. If 4 users send prompts simultaneously, the 4th user sits in the queue for 3 seconds before their prefill even begins, destroying TTFT, while the active decoders are completely stalled waiting for the prefills to finish.

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Multi-Tenant SLO Crisis</b> · <code>serving-architecture-queueing-theory</code></summary>

- **Interviewer:** "You are the Principal Engineer for the AI Serving Platform at a major cloud provider. Your platform serves two models on the same H100 GPU cluster:

1.  **"Swift"**: A 7B parameter model for a real-time coding assistant (SLO: P99 TTFT < 250ms).
2.  **"Deep"**: A 70B parameter model for offline code repository analysis (SLO: Throughput-oriented, no strict latency).

Currently, you meet the "Swift" model's SLO by dedicating a silo of GPUs to it, running a simple batching server with a tiny batch timeout. This results in an average GPU utilization of only 15%, and your CFO is demanding you improve hardware efficiency by co-locating both services on all GPUs. Your initial attempt to mix workloads on a standard batching server caused the "Swift" model's P99 TTFT to skyrocket to over 2 seconds, a major SLO violation.

Propose a new serving architecture that allows both 'Swift' and 'Deep' workloads to share the same H100 GPUs. Your design must (1) provably protect the 'Swift' model's 250ms P99 TTFT SLO and (2) significantly improve overall cluster utilization. Justify your core architectural decisions with napkin math."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common but insufficient proposal is to simply add a priority queue. While this ensures 'Swift' requests are picked before 'Deep' requests from the waiting queue, it doesn't solve the fundamental problem: **Head-of-Line Blocking**. If a 'Deep' model prefill has already started, the non-preemptive nature of standard batching means the 'Swift' request must wait for the entire multi-hundred-millisecond 'Deep' computation to finish, violating the SLO.

  **Realistic Solution:** The correct architecture replaces the standard batching server with one that supports **fine-grained, preemptive scheduling**, like the mechanisms found in modern engines such as vLLM or Orca. This is often called 'continuous batching'.

This design decouples the monolithic 'prefill' operation into a series of smaller, schedulable chunks (e.g., at the transformer layer or operator level). When a high-priority 'Swift' request arrives, the scheduler doesn't wait for the entire 'Deep' prefill to complete. Instead, it waits for the current, very short, computational chunk of the 'Deep' request to finish. It then **preempts** the 'Deep' request, serves the entire 'Swift' request (which is computationally very small), and then resumes the 'Deep' request. This prevents head-of-line blocking from the long-running task and guarantees low latency for the short, high-priority task, allowing both to safely co-exist and drive up utilization.

  > **Napkin Math:** The justification is proven by comparing the worst-case queue time for the 'Swift' model under both architectures.

**1. Architecture Failure (Standard Batching):**
- A 'Deep' request (70B model, 4k prompt) begins its prefill computation.
- Compute required: `C ≈ 2 × P × D = 2 × 70B × 4096 ≈ 5.7e14 FLOPs`.
- Time on an H100: `T_deep = 5.7e14 FLOPs / 989 TFLOPS ≈ 580 ms`.
- A 'Swift' request arriving just after this starts must wait for the entire 'Deep' prefill.
- Worst-case `T_queue` for Swift is `~580 ms`.
- `P99 TTFT (Swift) = T_queue + T_prefill (Swift) ≈ 580ms + (a few ms)`. This massively violates the 250ms SLO.

**2. Proposed Architecture (Preemptive Continuous Batching):**
- The 'Deep' model's 580ms prefill is broken into schedulable chunks. A Llama-70B model has 80 layers, so we can preempt at layer boundaries.
- Duration of one chunk: `T_chunk = T_deep / num_layers = 580 ms / 80 ≈ 7.25 ms`.
- A 'Swift' request's maximum wait time is now the time to finish one chunk, not the whole sequence.
- `T_queue_max (Swift) ≈ 7.25 ms`.
- Now, let's calculate the 'Swift' model's own service time (7B model, 128 token prompt).
- Compute required: `C ≈ 2 × 7B × 128 ≈ 1.8e12 FLOPs`.
- Time on H100: `T_prefill(Swift) = 1.8e12 FLOPs / 989 TFLOPS ≈ 1.8 ms`.
- New protected P99 TTFT: `P99 TTFT (Swift) = T_queue_max + T_prefill(Swift) = 7.25 ms + 1.8 ms = 9.05 ms`.
- This is well within the 250ms SLO, allowing the 'Deep' model to use all spare cycles and dramatically increasing cluster utilization.

  > **Key Equation:** $$T_{\text{TTFT}} = T_{\text{queue}} + T_{\text{service}}$$

  📖 **Deep Dive:** [Cloud: Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Phoenix False Positive</b> · <code>training-serving-skew</code></summary>

- **Interviewer:** "You are the Principal Engineer for a new 'Vigilant Driver' feature that uses in-car cameras to detect driver drowsiness. After a successful offline evaluation, you roll it out to 1% of the fleet. Within hours, alerts spike for drivers in sunny cities like Phoenix. The model is flagging them as drowsy, but dashcam footage shows their eyes are open. Your constraints: the model was trained on data from the overcast Bay Area; the in-car camera has a fixed auto-exposure; the edge device only does a simple resize and greyscale conversion before running inference; and to save on cellular costs, high-resolution video is only uploaded at the end of a driver's 8-hour shift. Propose how you would triage this crisis, and then design the long-term architectural changes to the data and training pipeline to prevent this class of 'environmental skew' failure."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common reaction is to state the obvious: 'We need to retrain the model on data from Phoenix.' This is reactive, not systemic. It doesn't explain *how* to identify the root cause without waiting a full day for data, nor does it create a system that *anticipates* this problem in the future. Another mistake is to assume it's a model bug without first looking for a data-driven explanation for the geographic correlation.

  **Realistic Solution:** The root cause is a severe case of training-serving skew driven by an unmodeled environmental variable: ambient brightness. The model, trained on overcast data, has incorrectly learned that 'squinting due to bright sun' is equivalent to 'eyes closing from drowsiness.'

**Triage (Immediate):** The end-of-day upload policy means we're flying blind. We can't wait 20+ hours for video. The key is to derive a proxy for 'brightness' from the metadata we *can* get quickly. My first step is to ask the on-call team to immediately start logging the *average pixel intensity* of the greyscale frames being processed on the edge device, alongside the model's score. This is a tiny payload. By plotting `avg_pixel_intensity` vs. `drowsiness_score`, we can quickly confirm the hypothesis that high brightness is causing false positives. This proves it's a data problem, not a code bug.

**Architectural Fix (Long-Term):** The system must be redesigned to be environment-aware.
1.  **Sensor Pipeline:** The edge device's preprocessing must be upgraded from a simple converter to a feature extractor. It must now compute and stream a 'metadata packet' for every frame, including `avg_pixel_intensity` and maybe an `image_histogram_percentiles`. This is a few dozen bytes, cheap to transmit in near real-time.
2.  **Feature Store:** These new environmental features must be ingested into the feature store as first-class citizens, time-synced with the inference results.
3.  **Model & Training Pipeline:** The model architecture must change. We will move to a multi-modal input, feeding it both the image embedding AND the environmental feature vector. This allows the model to learn the crucial difference between `(squint, high_brightness)` and `(squint, low_brightness)`.
4.  **Proactive Monitoring:** We will build a data quality monitoring system that perpetually tracks the statistical distribution of these environmental features across the fleet. If we launch in a new city and its `avg_pixel_intensity` distribution is vastly different from our training set's, the system automatically creates an active learning task to label a small sample of this new data, allowing us to proactively retrain *before* a false positive crisis occurs.

  > **Napkin Math:** The 'end-of-shift' upload policy is a critical failure point that prevents effective triage. Let's quantify why.
- **Data per Shift:** An 8-hour shift at 10 FPS with 640x480 greyscale frames (307.2 KB/frame) generates:
  `8 hr × 3600 s/hr × 10 FPS × 0.3072 MB/frame ≈ 88.5 GB`
- **Upload Time:** A typical vehicle's 4G connection averages ~10 Mbps (1.25 MB/s).
  `88,500 MB / 1.25 MB/s = 70,800 seconds`
- **The Blind Spot:** `70,800 s / 3600 s/hr ≈ 19.6 hours`.
It takes nearly 20 hours to upload 8 hours of data. This cost-saving measure makes rapid-response debugging impossible and proves why a new real-time metadata-first telemetry strategy is essential for operational stability.

  > **Key Equation:** $$ P(\text{Drowsy} | \text{Image}, \text{Brightness}) \neq P(\text{Drowsy} | \text{Image}) $$

  📖 **Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The AI Analyst's Deadline</b> · <code>serving-queueing-theory-sla</code></summary>

- **Interviewer:** "You are the Principal Engineer at a new FinTech startup building a unified 'AI Analyst' API on a fixed cluster of H100 GPUs. The API must serve two workloads with conflicting demands:

1.  **Real-time News Analysis:** Analyzes a stream of incoming news articles (~2,000 tokens each) and must return a structured JSON object within a **P99 deadline of 500ms**. The arrival rate is bursty, averaging 10 req/s but peaking at 50 req/s during market events.
2.  **Quarterly Report Generation:** Takes a large context of financial filings (~100,000 tokens) and generates a 4,000-token summary. This is a throughput-sensitive, non-interactive job that can take minutes.

Your CFO has denied your request for more GPUs. Design the serving architecture for this API. What are your first three architectural decisions to ensure the real-time news analysis meets its 500ms P99 deadline, without completely starving the long-running report jobs? Justify your decisions with napkin math."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Using a single, simple FIFO queue for all requests. This is the most common and fatal mistake. While simple to implement, it leads to **Head-of-Line Blocking**. A single, long-running 100k-token report generation job will enter the batch, and its massive prefill computation will occupy the GPU for seconds. All subsequent real-time news requests that arrive during this time will be stuck in the queue, immediately violating their 500ms deadline.

  **Realistic Solution:** The core of the problem is isolating the latency-sensitive workload from the throughput-oriented workload on the same hardware. My first three decisions would be:

1.  **Implement Dual-Queue Prioritization:** Immediately split the incoming requests into two separate queues: a **High-Priority** queue for the real-time news analysis and a **Low-Priority** queue for the batch report generation. The scheduler MUST be configured to always drain the High-Priority queue before even considering a job from the Low-Priority queue. This is the foundational step to prevent Head-of-Line blocking.

2.  **Adopt a Preemptive, Iteration-Level Scheduler:** A simple priority queue isn't sufficient. If a low-priority report job is already in its decoding phase when a high-priority news request arrives, we can't afford to wait for the entire 4,000-token generation to complete. The scheduler must operate at the level of individual forward passes (iterations). It must support preemption: pause the decoding of the low-priority job, use the GPU to run the full prefill-and-decode of the newly-arrived high-priority job, and only then resume the low-priority job. This maximizes GPU utilization while strictly adhering to latency SLAs.

3.  **Establish System Capacity and Admission Control:** We must use queueing theory to determine if the 500ms SLA is even physically possible during peak load, and if not, how to gracefully degrade. We need to calculate the maximum service rate (μ) for our H100 cluster on the high-priority workload. If the arrival rate (λ) exceeds this, queueing delay will approach infinity. Therefore, we need an admission controller that sheds load (e.g., returns HTTP 503) when the system is near saturation to protect the latency of the requests we do accept.

  > **Napkin Math:** Let's calculate the capacity needed to meet the real-time SLA. We'll model a 70B Llama-class model on H100s.

1.  **Calculate Service Time per Request:** The main latency driver for the news analysis is the prefill of the 2,000-token prompt.
    -   `Compute FLOPs ≈ 2 × Parameters × Prompt Length`
    -   `Compute FLOPs ≈ 2 × 70×10⁹ × 2000 = 2.8 × 10¹⁴ FLOPs`
    -   An H100 provides 989 TFLOPS (FP16), so let's say ~1 PFLOP/s for simplicity.
    -   `Time_compute ≈ 2.8×10¹⁴ FLOPs / 1×10¹⁵ FLOPs/s = 280 ms`
    -   Adding ~70ms for system overheads (kernel launch, memory access, etc.), we get a service time (`t_s`) of **~350ms per request**.

2.  **Apply Queueing Theory (M/M/c):** Now we can determine how many GPUs (`c`) we need.
    -   Service Time (`t_s`) = 0.35s
    -   Service Rate per GPU (`μ`) = `1 / t_s = 1 / 0.35 ≈ 2.85 req/s`
    -   Peak Arrival Rate (`λ`) = 50 req/s
    -   The system utilization (`ρ`) must be less than 1 to have a stable queue: `ρ = λ / (c × μ) < 1`
    -   `c > λ / μ  => c > 50 / 2.85 ≈ 17.5`

3.  **Architectural Decision:** We need a minimum of **18 H100 GPUs** dedicated to the high-priority queue just to handle the peak traffic without queueing delays exploding. My admission controller will monitor the queue depth on this 18-GPU sub-cluster. If `λ` spikes further or if a GPU fails, it will start rejecting requests to maintain the 500ms P99 for accepted traffic. The report generation jobs can then run opportunistically on any of these 18 GPUs during non-peak times.

  > **Key Equation:** $\rho = \frac{\lambda}{c \cdot \mu}$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Silent Utilization Collapse</b> · <code>arithmetic-intensity-batching</code></summary>

- **Interviewer:** "You are the Principal Engineer for the AI Platform serving a 70B parameter LLM on a fleet of H100s. For months, the service was cost-effective, handling long document summarization tasks. To meet latency SLOs, you use a serving system that processes incoming requests in statically-sized batches.

A recent product change has shifted user traffic from long documents to short, conversational queries. Your total daily token count remains the same, but your cloud bill has tripled. Your dashboards show that GPU power draw is consistently high, yet your effective TFLOPS metric has cratered to less than 25% of its previous value.

Formulate a hypothesis for the root cause of this simultaneous drop in efficiency and explosion in cost. Use quantitative reasoning based on hardware physics to prove your case. Propose a concrete architectural change to the serving stack to restore efficiency, and discuss the engineering trade-offs of your proposal."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Attributing the problem to 'Python overhead' or 'network latency' between services. These are minor effects. The core mistake is failing to connect the change in workload (request pattern) to the fundamental physics of the GPU's memory hierarchy and its impact on hardware utilization.

  **Realistic Solution:** The root cause is a collapse in arithmetic intensity due to the shift in workload, which turns a compute-bound system into a memory-bound one. The autoregressive decoding phase of LLMs, where tokens are generated one by one, has intrinsically low arithmetic intensity. When serving long documents, the high-AI 'prefill' phase amortized the low-AI decoding phase. With short conversational queries, the workload is now dominated by the memory-bound decoding phase. The static batching system is unable to pack enough requests together to raise the batch dimension ('B') high enough to overcome this. The GPU spends almost all its time waiting for data from HBM (memory-bound), achieving only a fraction of its peak performance, even though it's drawing full power. Since we pay for the GPU-hour, this low utilization translates directly into wasted money.

**Architectural Change:** Replace the static batching system with **continuous (or in-flight) batching**. Instead of waiting to form a full static batch, a continuous batcher maintains a running batch of requests. As soon as one request in the batch finishes generating its token, a new waiting request is immediately slotted in. This strategy's primary goal is to maximize the batch dimension 'B' during the memory-bound decoding phase. This increases arithmetic intensity, pushes the operation closer to the compute-bound region, and dramatically improves GPU utilization. Technologies like PagedAttention are key enablers for this by managing the KV cache efficiently across a dynamic set of requests.

**Trade-offs:** The primary trade-off is increased system complexity. The scheduler for a continuous batching system is significantly more complex than a static one. It may also introduce slightly higher latency for some individual requests (P50) in order to wait for an opportunity to form a larger, more efficient batch, but it almost always improves overall throughput and P99 latency by preventing head-of-line blocking and maximizing hardware efficiency.

  > **Napkin Math:** The diagnosis is proven by analyzing the Arithmetic Intensity (AI) of the decoding phase, which dominates conversational workloads.

1.  **H100 Hardware Specs:**
    *   Peak Compute (FP16): 989 TFLOPS
    *   Memory Bandwidth: 3.35 TB/s
    *   Ridge Point: `989 TFLOPS / 3.35 TB/s ≈ 295 Ops/Byte`.

2.  **The Physics of Decoding:**
    *   In each step of autoregressive decoding, we feed the model a batch of `B` tokens (one for each sequence in the flight) to predict the *next* token for each. The effective sequence length is `S=1`.
    *   The Arithmetic Intensity of a decode step is therefore: `AI = (FLOPs / Bytes) ≈ (B × S × 2P) / (Memory Traffic)`. For a single step, the dominant memory traffic is reading the weights (`2P`), so AI simplifies to `B × S`. With `S=1`, the `AI ≈ B`.

3.  **Scenario A: Efficient, Packed Batch (Old System w/ Favorable Traffic):**
    *   Let's assume the old system could effectively batch `B=64` requests together during decoding.
    *   `AI ≈ 64`. This is well below the H100's ridge point of 295. The operation is **memory-bound**.
    *   Max Theoretical Throughput = `AI × Bandwidth = 64 Ops/Byte × 3.35 TB/s = 214.4 TFLOPS`.
    *   Utilization = `214.4 / 989 = 21.7%`. (This was the baseline 'good' state for decoding).

4.  **Scenario B: Inefficient, Small Batches (New Conversational Traffic):**
    *   With many short, fast requests, the static batcher fails to form large batches, leading to an average effective batch size of, say, `B=8`.
    *   `AI ≈ 8`.
    *   Max Theoretical Throughput = `AI × Bandwidth = 8 Ops/Byte × 3.35 TB/s = 26.8 TFLOPS`.
    *   Utilization = `26.8 / 989 = 2.7%`.

**Conclusion:** The shift in traffic pattern caused the effective batch size during the dominant decoding phase to plummet. This dropped the AI by a factor of 8 (from 64 to 8), which in turn cut the achievable throughput and utilization by a factor of 8 (from 21.7% to 2.7%). Since the cost per hour of the H100 is fixed, an 8x drop in efficiency leads to an ~8x increase in cost per token processed, explaining the tripling (or worse) of the cloud bill.

  > **Key Equation:** $\text{Throughput}_{\text{Mem-Bound}} = \text{Arithmetic Intensity} \times \text{Memory Bandwidth}$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Earnings Call Meltdown</b> · <code>ttft-tpot-continuous-batching-queueing</code></summary>

- **Interviewer:** "You are the Staff ML Systems Engineer at a fintech company that provides real-time transcription for Wall Street earnings calls. The product promise is a live, streaming transcript with a P99 Time-To-First-Token (TTFT) under 500ms.

The current system uses a 70B parameter model on a single H100 GPU. Audio is chunked into 2-second segments. The serving stack is a simple Python Flask server that takes a request, runs a single prefill and decode loop, and returns the result. During testing with a single user, the median TTFT is excellent, around 250ms.

Today is a major tech company's earnings call. 20 high-value clients connect simultaneously. Within a minute, your P99 TTFT alert fires: latency has skyrocketed to over 30 seconds. The live transcript is uselessly delayed. Your VP asks for an architectural proposal to fix this class of failure permanently. What do you propose, and why did the old system collapse?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to blame the model or the GPU speed. Junior engineers might say 'We need a faster GPU like a B200' or 'We need to distill the model to be smaller.' While these might help, they don't address the fundamental architectural flaw. The system isn't slow because the GPU is slow; it's slow because it's idle, waiting for a serialized, request-level lock to be released. This is a systems and queueing failure, not a raw compute problem.

  **Realistic Solution:** The current architecture of processing one request at a time is a classic Head-of-Line Blocking problem. The system's service rate is fixed by the time it takes to process one full request. When the arrival rate of requests exceeds this service rate, a queue builds up infinitely (in theory). A new request might have to wait for 19 other requests to complete their *entire* generation sequence, which can take many seconds each.

The correct solution is to replace the request-level batching with **Continuous Batching** (also known as dynamic batching or iteration-level batching). Instead of a queue of requests, we maintain a queue of sequences. On every *iteration* of the model's forward pass, the server gathers all sequences that are ready for a new token, batches them into a single large tensor, and executes one forward pass. New requests arriving can be added to the running batch in the *next* iteration, which is typically just a few milliseconds away. This decouples TTFT from the number of active users. TTFT is now bounded by the time for a single forward pass plus the batching overhead, while overall throughput (TPOT) is maximized by filling the GPU with parallel computation.

  > **Napkin Math:** Let's analyze the failure and the solution. A 70B model requires ~140GB of weights. Let's assume each user generates 100 tokens from a 100-token prompt.

**Old System (Request-Response):**
1. **Prefill Time:** Compute for one user's 100-token prompt is `2 * 70B * 100 = 14 TFLOPs`. On an H100 (989 TFLOPS), this is `14e12 / 989e12 ≈ 14ms`.
2. **Decode Time:** Generating 100 tokens sequentially takes `100 * (2 * 70B / 989e12) ≈ 100 * 0.14ms = 14ms`. Total processing time per request is `14ms + 14ms = 28ms` of pure compute. *However*, each request has scheduling overhead, data transfer, etc. Let's be generous and say total service time `T_service` is ~100ms per request.
3. **Queueing Collapse (Little's Law):** The system can serve `1 / 0.1s = 10` requests per second. With 20 clients, requests arrive at `20 users / 2s audio chunk = 10 req/s`. The arrival rate `λ` (10 req/s) is equal to the service rate `μ` (10 req/s). The system utilization `ρ = λ/μ = 1`. In queueing theory, as utilization approaches 1, the wait time `W` approaches infinity. The 20th user has to wait for the 19 others to finish, so their wait time is `19 * 100ms = 1.9s`, plus their own service time. This is the meltdown.

**New System (Continuous Batching):**
1. **Batching:** Now, we batch the 20 users together. Let's say at a given moment, all 20 need a token.
2. **Batch Decode Time:** We perform one forward pass for a batch of 20. The compute is `2 * 70B * 20 = 2.8 TFLOPs`. Time on H100 is `2.8e12 / 989e12 ≈ 2.8ms`. Let's add overhead and call it a 5ms iteration time.
3. **TTFT for a NEW user:** A new 21st user arrives. They don't wait for 20 sequences to complete. They wait for the *current 5ms iteration* to finish, then get added to the next batch. Their TTFT is now dominated by the iteration time (`~5ms`) plus prefill for their sequence, which can also be batched. P99 TTFT stays low and is independent of the number of users.
4. **TPOT:** The old system had a throughput of 10 users/sec * 100 tokens/user = 1000 tokens/sec. The new system, in one 5ms iteration, generates 20 tokens. Its throughput is `20 tokens / 0.005s = 4000 tokens/sec`. We get 4x the throughput while simultaneously fixing TTFT.

  > **Key Equation:** $W = \frac{\rho}{\lambda(1-\rho)}$

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Real-Time Translation Stalemate</b> · <code>ttft-tpot-continuous-batching-queueing-theory</code></summary>

- **Interviewer:** "You are the tech lead architecting a new real-time translation service. The service powers a live conversation, translating speech-to-text-to-speech using a 70B parameter model. The business has a non-negotiable product requirement: the P99 Time-To-First-Token (TTFT) for the LLM generation step must be under 400ms to feel 'instantaneous' to the user. During load testing on H100s, you observe that as the number of concurrent users crosses about 16 per GPU, the P99 TTFT skyrockets to over 5 seconds, even though average GPU utilization is only 60%. Your initial attempt to fix this by adding more H100s to a simple round-robin load balancing pool does not meaningfully decrease the P99 latency. Propose a new serving architecture from first principles. What are your first three design decisions to address this P99 latency explosion, and why are they the correct levers to pull? Justify your proposal with napkin math."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common L4/L5 answer is to blame raw capacity: 'We just need to autoscale and add more GPUs.' This fails because the problem isn't a lack of aggregate compute, but a systemic scheduling and resource allocation failure known as Head-of-Line (HOL) blocking. Another mistake is to immediately suggest a smaller model, which compromises on quality before exhausting systems-level optimizations.

  **Realistic Solution:** The core of the problem is Head-of-Line (HOL) blocking within a naive static batching framework. An L6+ answer must identify this and propose an architecture that eliminates it.

1.  **Replace Static Batching with Continuous Batching:** This is the most critical change. Instead of forming a static batch and waiting for all requests in it to complete before starting the next, a continuous batcher (e.g., vLLM, Orca) decouples requests. The server processes the shared prefix of a batch, and then iterates one decode step at a time, continuously admitting new requests from a queue as old ones complete and free up resources. This prevents a single long-running request (e.g., one with a long prompt or requiring a long generation) from blocking the entire batch of otherwise short requests.

2.  **Implement a Priority Queue with Deadline-Aware Scheduling:** A simple FIFO queue is not sufficient for a strict P99 SLO. The system must be able to prioritize requests. An Earliest Deadline First (EDF) scheduler, for example, would prioritize requests that are closest to violating their 400ms SLO budget. When deciding which new request to admit into the continuous batch, the scheduler would pick the one most at risk, rather than the one that arrived first. This directly manages the P99 tail.

3.  **Decouple Prefill and Decode Stages:** Prefill (processing the prompt) is compute-bound and happens once, while decode (generating tokens) is memory-bandwidth-bound and iterative. A large prefill operation can starve the decode loop for many active requests. A sophisticated architecture separates these. One pool of GPUs could be dedicated to prefill, and once a request's KV cache is computed, it's passed to a separate pool of GPUs optimized for decode. This is an advanced technique, but for a P99-critical service, it isolates the performance characteristics of the two phases, preventing prefill-induced latency spikes in the decode loop.

  > **Napkin Math:** Let's quantify why static batching fails. Assume a batch of 16 requests on an H100 (989 TFLOPS, 3.35 TB/s memory BW) with a 70B model (140GB weights).

**Scenario: 1 'long' request, 15 'short' requests.**
-   Short Request: 100 token prompt, 10 token generation.
-   Long Request: 1000 token prompt, 512 token generation.

**Static Batching Failure (Head-of-Line Blocking):**
The batch processing time is dictated by the slowest request.
1.  **Prefill:** The whole batch is padded to the longest prompt (1000 tokens). The compute time is thus determined by the 1000-token prompt.
    -   Prefill FLOPs: `2 * 70e9 params * 1000 tokens = 1.4e14 FLOPs ≈ 140 TFLOPs`
    -   Prefill Time: `140 TFLOPs / 989 TFLOPS ≈ 141 ms`
2.  **Decode:** The entire batch must wait as the server generates 512 tokens for the long request. The decode step is memory-bound.
    -   Time per token (for the whole batch): `140GB weights / 3.35 TB/s ≈ 41.8 ms`
    -   Total Decode Time for Long Request: `512 tokens * 41.8 ms/token ≈ 21.4 seconds`
3.  **Result:** The 15 short-request users get their first token only after the long request's prefill is done and one decode step completes. Their TTFT is `141ms (prefill) + 41.8ms (decode step) = 182.8ms`. This is okay. However, if the system waits for the *entire batch* to finish, the next batch is blocked for over 21 seconds. Any request arriving behind this batch sees its queue time explode, thus destroying the P99.

**Continuous Batching Success:**
The server runs a shared decode step for all active requests.
1.  **TTFT for Short Requests:** A short request gets added to the queue. It waits for a prefill slot. Prefill takes `(2*70B*100)/989e12 ≈ 14ms`. It then enters the decode loop.
2.  **Shared Decode Step:** In the decode loop, let's say 16 requests are active. The server batches the next-token generation for all 16. This batched operation is still memory-bound and takes `~42ms`. But in that `42ms`, it has produced a token for *all 16 users*.
3.  **Result:** The TTFT for a new short request is `Wait_Time_in_Queue + Prefill_Time + Time_for_First_Decode_Step`. Even at high load, the time for the first decode step is `~42ms`. The total time is `Queue_Time + 14ms + 42ms`. The system is now stable as long as `Queue_Time` is managed below `400 - 14 - 42 ≈ 344ms`. The long request takes a long time to finish, but it does so *in parallel* with all other requests, rather than blocking them.

  > **Key Equation:** $$W_q = L_q / \lambda$$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The 8 Petabyte Skew Problem</b> · <code>training-serving-skew</code></summary>

- **Interviewer:** "You are hired as the founding Principal Engineer for the ML Platform team at a self-driving car company that is scaling its fleet from 100 to 1,000 vehicles. Each vehicle generates ~8 TB of raw sensor data daily. The current 'log-file-to-training' pipeline is brittle, and worse, models retrained on new data are underperforming in the field, indicating significant training-serving skew between the C++ in-vehicle stack and the Python cloud training stack.

Your mandate is to design the next-generation 'Sensor-to-Model' data platform to solve this problem permanently, reduce data-to-model latency from weeks to days, and manage the exploding storage costs. Your primary constraints are the fleet size (1,000 cars), the data generation rate (8 TB/day/car), and a 100 Gbps network link to the cloud.

Propose a high-level architecture. What are your first three major architectural decisions, and how do you justify them with quantitative reasoning?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** 1. 'Let's just buy a feature store': This fails to recognize that standard cloud feature stores are built for request/response serving and don't solve the core problem of ensuring bit-for-bit consistency between a real-time C++ embedded system and a Python cloud environment. 2. 'We need a bigger network pipe': This ignores the physics of the data volume; the scale of the problem (petabytes/day) cannot be solved with bandwidth alone. It shows a failure to do the initial napkin math. 3. 'Let's write more unit tests': While essential, testing alone cannot guarantee that two independently evolving codebases (C++ and Python) for feature extraction will remain perfectly aligned over time, especially with compiler differences and floating-point nuances.

  **Realistic Solution:** The correct approach involves tackling the root causes: the dual-codebase problem and the unmanageable data volume. A robust architecture would be built on these three decisions:

1.  **Unify the Feature Definition:** The highest priority is to eliminate the source of skew. Create a single, hermetically sealed library (e.g., in C++ with Python bindings) that contains the *exact* code for all feature extraction. This same library is compiled into the car's real-time system AND used by the cloud-based data processing pipeline. This is the only way to guarantee bit-for-bit feature identity.

2.  **Implement Data Triage at the Edge:** It is physically impossible to upload all the data. The architecture must include a 'smart logger' on each vehicle. This system runs lightweight models to identify 'interesting' events (e.g., perception failures, near-misses, disengagements) and only uploads those high-value scenarios, along with a statistical sampling of 'normal' driving. This curates the data at the source, turning an intractable firehose into a manageable stream.

3.  **Architect a Two-Tier Data Store:** Instead of a raw data lake, build a pipeline that processes the triaged data *once* using the unified library and stores the output in a structured format.
    *   **Tier 1 (Offline Features):** Store the extracted features (not raw sensor data) in a low-cost, queryable format like Parquet in an object store (e.g., S3). This becomes the single source of truth for all model training and validation. Storing features is orders of magnitude cheaper than storing raw LIDAR/camera feeds.
    *   **Tier 2 (Raw Scenarios):** For the small subset of triaged events, store the *raw* sensor data for a limited time (e.g., 90 days) to enable debugging and development of new features. After that, it can be archived to deep storage or discarded.

  > **Napkin Math:** The justification is grounded in the physics of the data volume and network constraints.

**1. Data Upload Feasibility Check:**
   *   Total daily data: 1,000 cars × 8 TB/car/day = 8,000 TB/day = 8 PB/day.
   *   Network capacity: 100 Gbps = 12.5 GB/s.
   *   Seconds in a day: 24 * 3600 = 86,400 s.
   *   Total upload capacity per day: 12.5 GB/s × 86,400 s/day = 1,080,000 GB/day = 1.08 PB/day.
   *   **Conclusion:** The fleet generates 8 PB of data daily, but the network can only upload ~1 PB. We can only upload ~13% of the data. This proves that a 'smart triage' system isn't optional; it's a physical requirement.

**2. Storage Cost Analysis (Features vs. Raw):**
   *   Assume features are ~100x smaller than raw sensor data (a conservative estimate).
   *   Cost to store 1 month of raw data (30 days * 8 PB/day * 13% upload rate) = ~31.2 PB.
   *   At S3 standard pricing (~$23/TB/month), this is 31,200 TB * $23/TB ≈ $717,600 per month.
   *   Cost to store 1 month of features instead: $717,600 / 100 = $7,176 per month.
   *   **Conclusion:** The >$700k/month savings in storage costs easily justifies the engineering investment and compute cost of building a unified feature extraction pipeline.

  > **Key Equation:** $\text{Upload Feasibility Ratio} = \frac{\text{Network Capacity (Bytes/Day)}}{\text{Fleet Data Generation (Bytes/Day)}} < 1

  📖 **Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The SLA Collision</b> · <code>serving-architecture</code></summary>

- **Interviewer:** "You are the tech lead for the ML Serving Platform at a major cloud provider. Your team manages a shared H100 GPU cluster that serves two main services:

1.  **An interactive developer chatbot (Service A):** This service is latency-sensitive and has a strict P99 Time-To-First-Token (TTFT) SLA of 250ms. Requests are frequent, with short input/output sequences.
2.  **A batch code summarizer (Service B):** This service runs on large pull requests, values throughput over latency, and has no strict SLA. Requests are infrequent but involve very long input sequences and are computationally expensive.

The current system uses a simple FIFO queue and static batching. You've discovered that Service A is missing its TTFT SLA over 30% of the time because its requests frequently get stuck in the queue behind long-running summarization jobs from Service B.

The CFO has denied your request for more GPUs, citing budget constraints. To make matters worse, the product team wants to add a *third* real-time, low-latency service to the same cluster next quarter.

**Propose a new scheduling and batching architecture** that allows both current services (and a future one) to meet their SLAs on the existing hardware. Describe your first three architectural decisions, justify them with napkin math, and explain how they resolve the contention."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Proposing a simple priority queue or OS-style preemption. A priority queue alone doesn't solve the core problem: a low-priority, long-running batch will still block the GPU for its entire duration once it starts. Full, stateful preemption (saving the KV cache of a running batch to HBM or DRAM, swapping in a new request, then restoring) is often proposed but is prohibitively expensive. The context-switching cost of moving gigabytes of KV cache state would be many milliseconds, violating the very SLA you're trying to meet.

  **Realistic Solution:** The correct architectural solution involves moving away from static, monolithic batches towards a more dynamic, continuous system. The key decisions are:

1.  **Replace FIFO with a Priority Queue:** This is the first step. Interactive Service A requests must always be prioritized over batch Service B requests.

2.  **Abolish Static Batching; Implement Continuous Batching:** This is the core of the solution. Instead of creating a fixed batch and running it to completion, the server maintains a *running* batch. New, high-priority requests can be added to this batch at almost any time. This is the foundation of systems like Orca and vLLM. It maximizes GPU utilization by dynamically filling any unused slots in a batch.

3.  **Implement Iteration-Level Preemption:** This is the mechanism that makes the priority queue effective. You don't preempt in the middle of a forward pass (which is an indivisible unit of work on the GPU). Instead, after each *single token generation step* (one iteration), the scheduler checks the high-priority queue. If a new request from Service A has arrived, it is immediately scheduled for prefill in the very next iteration. The existing work from Service B is temporarily paused (its KV cache remains on the GPU) and is resumed once the high-priority work is done. Because a single token generation is very fast, the high-priority request gets serviced almost instantly.

  > **Napkin Math:** Let's model the two services on an H100 (989 TFLOPS) with a 13B model (26 GFLOPs/token).

- **Service B (Summarizer):** Prefill for a 4096-token input takes `(2 * 13B * 4096) / 989 TFLOPS ≈ 108ms`.
- **Service A (Chatbot):** Prefill for a 512-token input takes `(2 * 13B * 512) / 989 TFLOPS ≈ 13.5ms`.
- **Single Token Decode Time:** A single token generation (decode) step is very fast, let's estimate `~3ms` including overhead.

**Old FIFO Architecture:** If a Service A request arrives 1ms after a Service B job begins its prefill, it must wait. The total TTFT for Service A becomes `Wait Time (108ms) + Service A Prefill (13.5ms) = 121.5ms`. This seems to meet the 250ms SLA. But what if Service A arrives while Service B is in its long *decode* phase? If Service B has to generate 1024 tokens, that takes `1024 * 3ms = 3.072s`. The chatbot request waits over 3 seconds, catastrophically missing its SLA.

**New Architecture (Iteration-Level Preemption):** Service B is in its long decode phase. A high-priority Service A request arrives. The server finishes the current `3ms` decode step for Service B. On the next step, it pauses Service B and schedules the prefill for Service A. The total TTFT for the chatbot is now: `Max wait for current iteration to finish (3ms) + Service A Prefill (13.5ms) = 16.5ms`. This is well within the 250ms SLA, and the architecture successfully isolates the latency-sensitive service from the throughput-oriented one.

  > **Key Equation:** $$W_q = \frac{\rho}{1-\rho} \cdot \frac{C_a^2 + C_s^2}{2} \cdot E[s]$$

(This is the Pollaczek-Khinchine formula for an M/G/1 queue). It shows that wait time ($W_q$) depends heavily on the service time distribution ($C_s^2$, the squared coefficient of variation). In the old FIFO system, the service time distribution is enormous because of the mix of tiny chatbot requests and huge summarizer jobs. The new architecture effectively creates a high-priority queue that only sees small, uniform service times, dramatically reducing its wait time.

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The 'Laggy' Code Assistant: A Batching Design Challenge</b> · <code>continuous-batching-queueing-theory</code></summary>

- **Interviewer:** "You are the tech lead for 'StaffML Code,' a new real-time AI code assistant service. The product manager has defined two strict, user-facing SLOs:

1.  **P95 Time-To-First-Token (TTFT): < 250ms** (The user must see the first character of the completion almost instantly).
2.  **P50 Throughput (TPOT): > 50 tokens/sec** (The code must generate quickly once it starts).

The backend serves a 13B parameter model on H100 GPUs. Your initial prototype uses a simple static batching server: it waits for up to 100ms to collect a batch of incoming requests, processes them together, and then sends the responses.

During the internal alpha, developers report that the assistant often feels 'laggy,' with the cursor blinking for a long time before generation begins. Your metrics confirm the problem: P95 TTFT is spiking to over 2 seconds, and TPOT is highly variable, clearly violating the SLOs.

Your simple batching strategy is failing. Propose a new architecture for the request scheduler and batching engine to meet these conflicting SLOs. Describe the core algorithm, explain how it resolves the TTFT/TPOT tension, and justify your design with quantitative reasoning."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to propose a naive fix like 'just decrease the static batching window.' While this reduces the maximum queuing delay, it shrinks the batch size, which lowers arithmetic intensity and kills throughput (TPOT). This makes the system more memory-bound and inefficient. Another common mistake is to suggest 'adding more GPUs,' which increases cost but doesn't fix the underlying scheduling flaw causing head-of-line blocking.

  **Realistic Solution:** The correct approach is to design and implement a **continuous batching** (or 'dynamic batching') engine. This architecture decouples request arrival from batch execution.

**Core Design:**
1.  **Remove Static Windows:** Abolish the fixed 100ms waiting period. Requests that arrive are immediately placed into a 'pending' queue.
2.  **Iteration-Level Scheduling:** The core of the engine is a loop that runs for every single forward pass of the model.
3.  **Dynamic Batch Formation:** On each iteration (e.g., every ~12ms), the scheduler:
    a. Gathers the token IDs for all requests already 'in-flight' (i.e., currently generating).
    b. Inspects the 'pending' queue. If there is capacity in the GPU (e.g., `current_batch_size + new_requests <= max_batch_size`), it pulls new requests into the 'in-flight' set for prefill.
    c. Concatenates the single tokens from in-flight requests and the prompt tokens from new requests into a single, large batch.
    d. Executes exactly one forward pass on this dynamically created batch.
    e. De-multiplexes the resulting logits, sending the correct token back to each individual request stream.
    f. Removes any requests that have completed (emitted an EOS token) from the in-flight set.

This design resolves the TTFT/TPOT tension. TTFT is minimized because new requests join a batch on the very next forward pass instead of waiting in a queue. TPOT is maximized because the GPU is always processing a large, combined batch from many users, which maximizes hardware utilization and arithmetic intensity.

  > **Napkin Math:** Let's analyze why static batching fails and continuous batching succeeds for a 13B model on an H100.

**Hardware & Model Assumptions:**
- H100 HBM3 Bandwidth: 3.35 TB/s
- 13B Model Weights (FP16): 26 GB
- Time to load weights from HBM (memory-bound portion): `26 GB / 3.35 TB/s ≈ 7.8ms`
- Let's estimate a single forward pass (decode step) for a full batch takes `T_step = 12ms` (dominated by memory access).
- Let's assume an average user generation length of 200 tokens.

**Analysis of Static Batching (100ms window):**
1.  **Service Time:** A batch of, say, 32 requests needs to generate 200 tokens each. The total service time is `T_service ≈ 200 tokens * 12ms/token = 2400ms` (2.4 seconds).
2.  **Worst-Case TTFT:** A new user request (Request #33) that arrives just after a batch has started processing must wait for that *entire* batch to finish.
    `P95 Wait Time ≈ T_service + T_window = 2400ms + 100ms = 2500ms` (2.5s).
3.  This catastrophically fails the `< 250ms` TTFT SLO due to head-of-line blocking.

**Analysis of Continuous Batching:**
1.  **Wait Time:** A new request only needs to wait for the *current* forward pass to complete before it can be added to the next batch.
    `P95 Wait Time ≈ T_step = 12ms`.
2.  **TTFT:** The total time to first token is this minimal wait plus its own prefill time (which is also one forward pass).
    `P95 TTFT ≈ T_wait + T_prefill ≈ 12ms + 12ms = 24ms`.
3.  This easily passes the `< 250ms` TTFT SLO.
4.  **TPOT:** Each user in the batch gets one new token every `T_step` (12ms).
    `Per-user TPOT = 1 token / 12ms = 1 / 0.012s ≈ 83 tokens/sec`.
5.  This meets the `> 50 tokens/sec` TPOT SLO by ensuring the GPU is always fed a large batch, maximizing utilization.

  > **Key Equation:** $L = \lambda W$

  📖 **Deep Dive:** [The Serving Stack](https://mlsysbook.ai/cloud/03_serving_stack.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Copilot Latency Paradox</b> · <code>llm-serving-latency</code></summary>

- **Interviewer:** "You are the new Staff ML Systems Engineer at a startup building a real-time AI coding assistant to compete with GitHub Copilot. The key feature is streaming multi-line code suggestions. The product team has a strict user experience mandate: when a developer pauses typing for 500ms, a suggestion must *begin* to appear (Time-To-First-Token) within a P99 deadline of 200ms, and then stream at 15 tokens/second.

The backend runs a 70B parameter code generation model on a large cluster of H100 GPUs using a naive static batching strategy. During peak hours, users report that suggestions take seconds to appear, completely missing the 200ms deadline. Simultaneously, the CFO is questioning the massive cloud bill, as your own metrics show the H100 cluster has a shockingly low average utilization of only 35%.

Design the next-generation serving architecture to solve this paradox. What are your first three architectural changes, and how do you justify them to both the product and finance teams using first-principles and quantitative reasoning?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to think linearly: 'high latency means we need more GPUs' or 'low utilization means we can't afford this model.' This fails to identify the core systems-level problem. The paradox of high latency coexisting with low utilization is a classic sign of a queueing bottleneck caused by an inefficient scheduling or batching strategy, specifically Head-of-Line (HOL) blocking. Simply adding more machines will not solve HOL blocking and will only worsen the cost issue.

  **Realistic Solution:** The problem is Head-of-Line (HOL) blocking from static batching. A single long-running request (e.g., generating a full function) forces all other quick, interactive requests in the same batch to wait, killing TTFT. The low utilization comes from the GPU being idle while waiting for a full static batch to assemble.

My first three architectural changes would be:

1.  **Implement Continuous Batching:** Replace the static batching scheduler with an in-flight, continuous batching system (e.g., vLLM, Orca). New requests are added to the running batch as soon as space is available from completed sequences. This decouples request arrival from model execution, eliminating HOL blocking and dramatically improving both TTFT for short requests and overall GPU utilization.

2.  **Introduce Priority Queueing and Cancellation:** Not all requests are equal. I would implement a priority queue that schedules requests with tight deadlines (like interactive suggestions) ahead of background or less time-sensitive requests. Furthermore, if a user starts typing again, a cancellation token should be sent to immediately stop the in-flight generation for that user, freeing up GPU resources for the new, more relevant request. This prevents wasting compute on suggestions that will never be seen.

3.  **Deploy PagedAttention for KV Cache:** The 70B model's KV cache is a massive memory consumer. PagedAttention acts like virtual memory for the KV cache, solving internal fragmentation. This allows the system to pack significantly more sequences into a single batch, increasing the number of concurrent users a single GPU can serve and further boosting throughput. It's the key enabler that makes continuous batching maximally effective.

  > **Napkin Math:** Here's how I'd justify this to the CFO and Product Manager:

**1. The 'Before' Scenario (Static Batching):**
- Let's assume a static batch size of 16 and a realistic H100 step time of 50ms for the batch.
- A single 'long' request needs 500 tokens. It will block the entire batch for `500 tokens * 50ms/step = 25 seconds`.
- A new interactive request that gets stuck behind this batch has a TTFT of up to 25 seconds, catastrophically missing the 200ms SLO.
- Why the low utilization? The server waits (e.g., 200ms) to form a full batch. If the arrival rate is 40 requests/sec, a batch of 16 should form in `16/40 = 400ms`. But if one is a long request, the throughput of that GPU becomes `16 reqs / 25s = 0.64 reqs/sec`. The GPU is busy, but the *system throughput* is terrible. The average 35% utilization comes from the GPU being idle waiting for batches to form, especially during non-peak times.

**2. The 'After' Scenario (Continuous Batching):**
- The 'long' request no longer blocks anyone. The 15 'short' requests (needing ~10 tokens) in the batch finish in `10 tokens * 50ms/step = 500ms`.
- As soon as a short request finishes, its slot is filled by a new request from the priority queue. A new interactive request just has to wait for the next step (~50ms) to get processed. The TTFT is now dominated by queue wait time, which we've just made drastically shorter.
- **The Financial Case:** By eliminating HOL blocking and idle batch-formation time, we can keep the GPU fed with work constantly. Utilization can realistically increase from 35% to >85%. This is a `85/35 ≈ 2.4x` improvement in efficiency. We can now handle the same user load with `1 / 2.4 = 41%` of the original H100 cluster size. I can tell the CFO we can meet our SLOs while cutting our GPU bill by over 50%.

  > **Key Equation:** L = \lambda W

  📖 **Deep Dive:** [Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/vol1/serving/serving.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Conversational AI Traffic Jam</b> · <code>llm-serving-latency-throughput</code></summary>

- **Interviewer:** "You are the lead ML Systems Engineer designing the cloud backend for a new in-car voice assistant. The product requirement is a "natural, conversational feel," which the product team has translated to a P99 Time-To-First-Token (TTFT) of 500ms. The system must support 10,000 concurrent users with a 70B parameter model. Traffic is highly variable: 80% are short commands ('next song'), while 20% are long, generative queries ('summarize my last meeting'). Propose a serving architecture that meets the latency budget while minimizing cost. What is your core architectural choice for request handling and batching, and how do you justify it against the alternatives with quantitative analysis?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A typical answer is to propose static batching and simply over-provision with more GPUs until the latency target is met. This approach mistakes a throughput problem for a latency problem. It leads to an incredibly expensive and inefficient system because GPUs will sit idle most of the time just to ensure queues are always empty. This solution fails to identify the core issue: Head-of-Line blocking, which makes meeting a P99 latency target under load mathematically impossible for a static system.

  **Realistic Solution:** The correct architectural choice is to build the system around **continuous batching** (also known as in-flight batching), likely using a framework like vLLM with PagedAttention. The justification is that it is the only way to solve the Head-of-Line (HoL) blocking problem that makes static batching non-viable for this workload.

With static batching, a short, latency-sensitive request (e.g., 'play music') can get stuck in a queue behind a long-running, generative request ('summarize this article'). The short request's time-to-first-token is now dependent on the completion time of the long request, which can be tens of seconds, catastrophically failing the 500ms P99 budget.

Continuous batching decouples requests from batches. The system maintains a dynamic, in-flight batch on the GPU. As new requests arrive, their prefill computation is interleaved with the single-token decoding steps of the requests already in the batch. Once a request is complete, it is evicted, freeing up its resources. This architecture allows the system to run at very high utilization (maximizing throughput and minimizing cost) while still providing low latency for new arrivals, as they don't have to wait for long-running jobs to finish. The scheduler can even be designed to prioritize requests with short prompts to further protect the P99 TTFT.

  > **Napkin Math:** The justification is in the numbers:

1.  **Model & Hardware Physics:** We use a 70B model on an H100. The per-token generation step (decode) is memory-bound. The time is dominated by reading the 140 GB of FP16 weights from HBM.
    - Time per generated token = 140 GB / 3.35 TB/s (H100 HBM bandwidth) ≈ **42 ms**.

2.  **Static Batching Failure Analysis:** Consider a worst-case P99 scenario. A batch is formed that includes a long-generation request that needs to output 500 tokens. The decode phase for this batch will take at least 500 tokens * 42 ms/token = **21 seconds**. A new, short-prompt request that arrives just as this batch begins is now blocked. Its wait time is at least 21 seconds. Its TTFT will be >21,000 ms, which is 42x higher than the 500ms budget. This proves that a static system cannot meet the P99 requirement as soon as utilization becomes non-trivial.

3.  **Continuous Batching Success Analysis:** With continuous batching, the 21-second generation does not block new requests. A new short-prompt request arrives. It waits in a queue only until the scheduler can interleave its prefill phase. The prefill for a short 20-token prompt is compute-bound and very fast: (2 * 70B * 20) FLOPs / 989 TFLOPS ≈ **2.8 ms**. The total TTFT is now `queue_wait_time + prefill_time`. With an appropriately provisioned cluster, queueing theory (Little's Law) shows that wait times can be kept in the low milliseconds, easily meeting the 500ms P99 target. This allows the system to run near peak throughput (cost-efficient) while behaving like a low-latency system for new arrivals (good user experience).

  > **Key Equation:** $$ W_q = \frac{\lambda E[S^2]}{2(1 - \rho)} $$

  📖 **Deep Dive:** [ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Bi-Modal GPU Dilemma</b> · <code>unified-serving-architecture</code></summary>

- **Interviewer:** "You are the founding ML Systems Engineer at a startup that just secured a single, precious Blackwell B200 GPU. To maximize ROI, leadership wants to launch two products simultaneously:

1.  **Premium API:** A long-form summarization service for enterprise clients, processing up to 100,000-token legal documents. SLA is P99 latency < 5 seconds and requires the highest possible accuracy.
2.  **Internal Chatbot:** A free, conversational assistant for employees. It must feel instantaneous, with a P50 Time-To-First-Token (TTFT) goal of < 250ms. Accuracy can be slightly compromised for speed.

Your constraint is that both workloads must be served by this single B200 GPU. Propose a unified serving architecture that can meet these conflicting requirements. Describe the key model optimizations, memory management strategies, and scheduling logic you would formulate. Justify your architectural decisions with quantitative napkin math."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common proposal is to use time-slicing or a standard resource scheduler like CUDA MPS to rapidly swap two different models on the GPU. This approach fundamentally misunderstands the memory physics of modern LLMs. The 70B model's context (weights + KV cache) will occupy most of the B200's 192GB HBM. Evicting and reloading a ~170GB+ context to serve a chatbot query would take seconds in memory transfer time alone, catastrophically violating both SLAs. The problem isn't about scheduling compute; it's about managing a persistent state in memory.

  **Realistic Solution:** The correct approach is a unified, symbiotic architecture where the models assist each other, rather than competing for resources.

1.  **Two-Model Strategy:** The core of the design is to use two models. The Premium API uses a large, high-accuracy model (e.g., a 70B parameter model). For the chatbot, we create a much smaller model (e.g., a 7B parameter model) via **knowledge distillation** from the 70B model. This smaller model is optimized for speed.

2.  **Synergistic Co-residency with Speculative Decoding:** The 70B model's weights (140GB) remain resident in HBM at all times. The 7B model's weights (~7-14GB) are small enough to also co-reside in HBM. For the chatbot's low-latency requirement, we use **speculative decoding**. The small 7B model generates a block of `k` draft tokens very quickly. The large 70B model, which is already in memory, then validates all `k` tokens in a single, parallel forward pass. This allows the system to generate multiple tokens for the wall-clock cost of a single large-model pass, dramatically improving TTFT and throughput.

3.  **Long-Context Optimization:** For the Premium API's 100k token context, the primary bottleneck is the attention mechanism. The solution is to use **FlashAttention**, which avoids explicitly materializing the N×N attention matrix in HBM, reducing memory usage from quadratic to linear and optimizing HBM access patterns. The candidate must also recognize that the KV cache is the next bottleneck and that modern models like Llama-70B use Grouped-Query Attention (GQA) to make the cache manageable.

4.  **Unified Scheduler:** A custom continuous batching scheduler (like those in vLLm or TGI) is required. It must be able to batch requests for both the 70B model (API) and the 7B/70B speculative pair (chatbot), while managing the shared KV cache space in HBM using a technique like PagedAttention.

  > **Napkin Math:** **1. Fitting the 70B Model on the B200:**
   - **Weights (FP16):** 70B params × 2 bytes/param = **140 GB**.
   - **KV Cache (100k seq, Llama-70B w/ GQA):** Llama-70B has 80 layers and 8 KV heads (GQA ratio of 8). Head dim is 128.
     `Cache = 2 × layers × (kv_heads × head_dim) × seq_len × 2 bytes`
     `Cache = 2 × 80 × (8 × 128) × 100,000 × 2 bytes = 3.27e10 bytes ≈ 33 GB`.
   - **Total Footprint:** `140 GB (weights) + 33 GB (KV) ≈ 173 GB`. This fits within the B200's 192GB HBM, leaving ~19GB for the smaller model and workspace.

**2. Verifying the Chatbot SLA (< 250ms TTFT) with Speculative Decoding:**
   - Assume a `k=4` speculation from the 7B model, verified by the 70B model.
   - **7B Draft Latency:** A 7B forward pass is mostly memory-bound. On a B200 (8 TB/s HBM), reading 14GB of weights takes `14GB / 8000GB/s ≈ 1.75ms`. Add kernel overheads, let's say one pass is **~5ms**. Four draft tokens: `4 × 5ms = 20ms`.
   - **70B Verification Latency:** This pass is more compute-bound. `Compute = 2 × 70B params = 140 TFLOPs`. On a B200 (2250 TFLOPS FP16, assume 50% utilization): `140 TFLOPs / (2250 TFLOPS/s × 0.5) ≈ 124ms`.
   - **Total Time for 4 Tokens:** `20ms (draft) + 124ms (verify) = 144ms`.
   - This is well under the 250ms SLA and is for *four* tokens, not one. The effective per-token latency is `144ms / 4 = 36ms`, a ~3.4x speedup over a naive 124ms per token.

  > **Key Equation:** $\text{Speculative Speedup} \approx \frac{k \times T_{large}}{T_{draft} + T_{verify}}$

  📖 **Deep Dive:** [ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Two-Tier SLA Conundrum</b> · <code>serving-architecture-sla</code></summary>

- **Interviewer:** "You are the founding ML Systems Engineer at a well-funded startup. Your first product is an API powered by a 70B parameter LLM, running on a fixed cluster of 16 NVIDIA H100s. The product has two tiers with conflicting requirements:

1.  **Premium Tier:** An interactive chat service for live users. This tier is sold with a strict P99 Time-To-First-Token (TTFT) SLA of 500ms.
2.  **Standard Tier:** An asynchronous batch summarization service for long documents (up to 100,000 tokens). This tier is price-sensitive, so maximizing throughput (and thus minimizing cost-per-job) is the key business goal.

Your CEO is asking for the high-level system architecture. Propose a design for the GPU serving stack that can simultaneously meet the Premium SLA and maximize Standard tier throughput. What are your first three architectural decisions, and how do you justify them with quantitative reasoning?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Proposing a single, unified serving cluster for both traffic types. A less experienced engineer might assume that a modern continuous batching server (like vLLM or TGI) can handle both workloads by itself. This fails to account for 'Head-of-Line Blocking', where a single large, long-running batch job (Standard Tier) can occupy a GPU for seconds, causing all subsequent interactive requests (Premium Tier) to miss their strict latency deadlines.

  **Realistic Solution:** The core challenge is that the two tiers have fundamentally incompatible optimization targets: one is latency-bound, the other is throughput-bound. A robust architecture must create isolation between them.

1.  **Architectural Decision 1: Partition the Cluster.** The GPU cluster should be partitioned, either physically or logically. For example, dedicate 4 of the 16 H100s exclusively to the Premium tier and the remaining 12 to the Standard tier. This provides hard isolation, guaranteeing that a surge in Standard jobs cannot impact Premium TTFT. The partition size itself is a parameter to be tuned based on expected traffic mix.

2.  **Architectural Decision 2: Tier-Specific Batching Policies.** Each pool must be configured with a different batching strategy.
    *   **Premium Pool:** Use continuous batching with a very small maximum token count per batch (e.g., 8,192) and a very short max wait time (e.g., 10-20ms). This policy prioritizes starting a small batch *immediately* to meet the TTFT SLA, even if the GPU is not fully utilized.
    *   **Standard Pool:** Use continuous batching with the largest possible batch size the H100's memory can handle (e.g., 250,000+ tokens) and a longer wait time (e.g., 100ms). This policy prioritizes packing as many requests as possible into each batch to maximize arithmetic intensity and overall throughput.

3.  **Architectural Decision 3: Implement a Priority Queue with Preemption.** A single request router receives all traffic. It pushes Premium requests into a high-priority queue and Standard requests into a low-priority queue. The scheduler for the GPU pools always serves the high-priority queue first. For a more advanced system, implement preemption: if a Standard-tier job is running on a GPU that is needed for a new Premium request (e.g., in a dynamic partitioning scheme), the system can pause or kill the Standard job, service the Premium request, and then resume the Standard job. This maximizes resource usage while maintaining the SLA.

  > **Napkin Math:** Justification for partitioning is critical. A unified queue is quantitatively non-viable.

1.  **Define the SLA-Breaker:** A single Standard Tier job processing a 100k token prompt.
    *   **Compute Required:** The prefill phase dominates. Using the rule-of-thumb `C ≈ 2 * P * D`, where P=70B params and D=100k tokens: `2 * 70e9 * 100e3 = 1.4e16` FLOPs, or 14 PetaFLOPs.
    *   **Time on H100:** An H100 provides 989 TFLOPS (FP16). The time to process this single job is `14e15 FLOPs / 989e12 FLOPs/sec ≈ 14.15` seconds.

2.  **Calculate Queuing Impact:** Assume a Premium chat request (e.g., 512 tokens) arrives 1ms after this Standard job begins in a unified queue.
    *   The Premium request must wait for the Standard job to complete its 14.15-second prefill.
    *   **Resulting TTFT for Premium User:** `14.15 seconds` (wait time) `+` `~70ms` (own prefill time) `≈ 14.22 seconds`.
    *   This is `14220ms`, which is **28.4x** greater than the 500ms SLA. This single calculation proves that a unified queue architecture is fundamentally broken.

3.  **Capacity Planning for the Premium Pool:** How many requests can one H100 in the Premium pool handle?
    *   Let's assume the average Premium request is 512 tokens. Prefill time is `(2 * 70e9 * 512) / 989e12 ≈ 72ms`.
    *   Let's budget for a total processing time of 150ms per request inside the batch to be safe (includes decoding, overhead).
    *   Using Little's Law ($L = \lambda W$), if we want to keep the average wait time (W) below, say, 250ms to comfortably meet a 500ms P99, we can calculate the supportable arrival rate ($\lambda$).
    *   The total time a request spends is Wait Time + Service Time. `500ms > W + 150ms`, so `W < 350ms`.
    *   If a continuous batcher can process a batch every ~150ms on average, it can handle `1000ms / 150ms ≈ 6.6` batches/sec. If each batch contains ~3-4 users, one H100 can handle `~20-25` requests/sec. This justifies dedicating a specific number of GPUs to meet the target QPS for the Premium tier.

  > **Key Equation:** $L = \lambda W$

  📖 **Deep Dive:** [ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Blackwell Disappointment</b> · <code>serving-optimization-strategy</code></summary>

- **Interviewer:** "You are the Staff ML Systems Engineer leading the serving infrastructure team at a major AI provider. Your company just deployed its first large cluster of B200 GPUs, intending to replace an existing A100-based stack. The primary goal is a 3x reduction in cost-per-million-tokens for your flagship multi-tenant LLM product, which serves thousands of customer fine-tuned models ranging from 7B to 70B parameters.

However, the initial migration benchmarks for the interactive chatbot workloads are disastrous. While raw throughput for large, offline batch jobs is phenomenal, the end-to-end P99 Time-To-First-Token (TTFT) has actually *regressed by 20%* compared to the mature A100 stack. The cost-per-request is only marginally better, nowhere near the 3x target.

Your VP of Engineering is concerned. 'I thought newer hardware was always faster. What did we miss, and what's your 3-point plan to get us back on track?'

Design a recovery plan to present to your VP. Your proposal must diagnose the likely root cause of the latency regression and formulate a concrete 3-point strategy to fully exploit the B200's capabilities and meet the original business goals. Your plan must integrate at least three distinct optimization techniques from the following list: pruning, distillation, operator fusion, FlashAttention, speculative decoding."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Focusing on a single, isolated fix, like 'we just need to upgrade our CUDA driver' or 'we need to use FP4'. This ignores the systemic nature of the problem. Another common mistake is assuming the B200 is simply a 'faster A100' and failing to recognize that its architectural changes (dramatically more compute vs. memory bandwidth) shift the bottleneck landscape, invalidating previous system optimizations. Blaming the hardware ('B200s are bad for latency') is the most junior-level mistake.

  **Realistic Solution:** The core insight is that simply swapping hardware is not enough; the entire software and serving strategy must be co-designed with the hardware's characteristics. The latency regression suggests the B200's massive compute potential is being starved. This is likely because the existing software stack, highly optimized for A100, is falling back to generic, un-optimized kernels on the new architecture. For latency-sensitive workloads with small batches, fixed overheads (like kernel launches) and non-fused memory-bound operations now dominate the execution time because the pure compute part has shrunk so dramatically.

A strategic plan involves a top-to-bottom re-architecture of the serving stack:

1.  **Deploy a Tiered Model Strategy with Distillation and Speculative Decoding:** The most powerful lever for latency is to do less work. I propose a system to automatically distill customers' large 70B models into highly specialized 7B 'student' models. For interactive requests, we use the 7B model to generate a draft of 4-5 tokens, then use the large 70B model to validate them in a single, parallel forward pass. This amortizes the cost of a single 70B forward pass over several generated tokens, dramatically cutting latency and improving throughput.

2.  **Launch a B200-Native Kernel Program (Operator Fusion & New Data Types):** The existing library of fused operators is obsolete. I will form a dedicated team to develop a new set of fused kernels specifically for the B200, with a focus on attention and MLP layers. Critically, this effort will natively support the new FP4 and FP6 data types. This is not just about quantization for memory savings; it's about redesigning the kernels to maximize data fetched per memory operation, increasing arithmetic intensity to keep the B200's tensor cores fed.

3.  **Implement Hardware-Aware Dynamic Routing with FlashAttention Upgrade:** All requests are not equal. I will architect a new scheduler that is aware of the live performance characteristics of different hardware/model combinations. Interactive chats for 70B models will be routed exclusively to B200s running our new speculative decoding stack. Large, offline summarization jobs will be batched heavily and can be routed to either A100s or B200s based on capacity. As a foundational step, we will mandate an immediate upgrade to the latest version of FlashAttention that includes B200-optimized kernels to fix the most immediate performance regression.

  > **Napkin Math:** The core of the proposal is the latency improvement from Speculative Decoding. Let's quantify the expected speedup.

**Assumptions:**
- A single forward pass for a 70B model on a B200 takes ~50ms.
- A 7B 'student' model is ~10x smaller and we'll assume ~10x faster, so its forward pass takes ~5ms.
- For a given prompt, the student and master models have a high degree of agreement.

**Baseline (Standard Autoregressive Generation):**
To generate 4 new tokens, the 70B model must run a forward pass for each token sequentially.
- Total Time = 4 tokens * 50ms/token = **200ms**

**Proposed (Speculative Decoding with γ=4):**
1.  The 7B student model autoregressively generates a 'draft' of 4 tokens. Time = 4 tokens * 5ms/token = 20ms.
2.  The 70B master model takes the original prompt + 4 draft tokens and runs a *single* forward pass to validate all 4 tokens in parallel. Time = 50ms.
3.  Assume 3 of the 4 tokens are accepted by the validator. We have successfully generated 3 tokens.
- Total Time = 20ms (draft) + 50ms (validation) = **70ms**

**Result:**
We generated 3 tokens in 70ms. To generate 3 tokens, the baseline method would have taken 3 * 50ms = 150ms. The speedup is 150ms / 70ms ≈ **2.14x**. This directly attacks the latency regression, improves user experience, and increases the serving capacity of each B200, contributing directly to the 3x cost reduction goal.

  > **Key Equation:** $\text{Speculative Speedup} \approx \frac{\gamma \cdot T_{\text{large}}}{T_{\text{small}} \cdot \gamma + T_{\text{large}}}$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>











<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The Interactive Coding Assistant SLO Catastrophe</b> · <code>llm-serving-architecture-queueing</code></summary>

- **Interviewer:** "You are the Staff ML Systems Engineer designing the serving infrastructure for a new AI coding assistant. The service uses a 70B parameter model and has strict, product-driven SLOs: P99 Time-To-First-Token (TTFT) must be under 500ms, and sustained Time-Per-Output-Token (TPOT) must be under 50ms. Your initial deployment uses 8 server nodes, each with two H100 GPUs (16 GPUs total) using tensor parallelism (TP=2). The system uses static batching. Under light load, performance is acceptable. However, as you approach the target peak load of 200 concurrent users, the P99 TTFT skyrockets to over 10 seconds, and the service is unusable. You correctly identify Head-of-Line blocking as the culprit and propose migrating to a continuous batching engine. The Head of Infrastructure is skeptical about the migration cost and asks you to prove it's necessary and sufficient.

First, formulate a quantitative argument for your leadership team explaining exactly why the static batching system cannot meet the P99 TTFT SLO under this load. Second, using the principles of continuous batching, calculate the maximum number of concurrent users your 16-GPU cluster can *actually* support while satisfying the SLOs, assuming an average user context of 1024 prompt tokens and 512 generated tokens. Is the initial provisioning of 16 H100s sufficient to meet the 200-user goal?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often focus only on the aggregate throughput of the system (e.g., total tokens/sec across all users), see high GPU utilization, and assume the system is efficient. They neglect the queueing delay experienced by new requests (the 'waiting in line' cost), which dominates the user-perceived latency (P99 TTFT) in a static batching system. They mistake high system throughput for low user latency.

  **Realistic Solution:** The correct answer has two parts: proving static batching is inadequate and then calculating the true capacity under a continuous batching regime.

**Part 1: The Case Against Static Batching**
The key issue is Head-of-Line (HOL) blocking. With static batching, a new request (user #9) that arrives just after a batch of 8 has started must wait for that *entire batch* to finish its full generation before its own processing can even begin. A typical generation of 512 tokens for a batch of 8 on H100s takes several seconds. This wait time is added directly to the new user's TTFT. For example, if a batch takes 5 seconds to complete, user #9's TTFT will be at least 5 seconds, making the <500ms P99 SLO impossible to meet under any meaningful load.

**Part 2: Continuous Batching Capacity Planning**
The primary constraint for continuous batching is total available HBM for the KV Cache. We must calculate this to find the true user capacity.
1.  **Memory Budget per Server:** Each server has 2 H100s with 80GB HBM each, for 160GB total. The 70B model's weights in FP16 are 140GB. This leaves `160GB - 140GB = 20GB` for the KV Cache and other overhead.
2.  **KV Cache per User:** The memory required for a single token in the KV cache for a 70B model is `2 (K/V) * 80 layers * 8192 hidden_dim * 2 bytes/fp16 ≈ 2.62 MB`. The total sequence length per user is `1024 prompt + 512 completion = 1536 tokens`. So, each user requires `1536 tokens * 2.62 MB/token ≈ 4024 MB`, or ~4 GB of KV Cache.
3.  **Max Users per Server:** With 20GB of available HBM for the cache, each 2-GPU server can handle `20 GB / 4 GB_per_user = 5` concurrent users.
4.  **Total Cluster Capacity:** The entire cluster has 8 servers. So, the maximum capacity is `8 servers * 5 users/server = 40` concurrent users.

**Conclusion:** The current 16-GPU cluster can only support 40 concurrent users, not the 200 required. The initial hardware provisioning is insufficient by a factor of 5x (`200 / 40`). The migration to continuous batching is necessary to eliminate HOL blocking, but to meet the business goals, a significant hardware expansion to ~80 H100s (`16 GPUs * 5`) is also required.

  > **Napkin Math:** ### Part 1: Static Batching - Why TTFT Fails
- A new request (user #9) arrives when a batch of 8 is running.
- Time per decode step for a batch is dominated by reading weights from HBM: `140 GB / (3.35 TB/s * 2 GPUs) ≈ 21ms`.
- Time to generate 512 tokens for the current batch: `512 tokens * 21ms/token ≈ 10.75 seconds`.
- User #9's TTFT = `Wait Time + Prefill Time`. Their wait time is at least 10.75 seconds.
- **Result:** P99 TTFT will be > 10 seconds, catastrophically missing the 500ms SLO.

### Part 2: Continuous Batching - True Capacity Calculation
- **Memory per Server:** `(2 * 80GB HBM) - (70B * 2 bytes/param) = 160GB - 140GB = 20GB` available for KV Cache.
- **KV Cache per Token:** For a 70B model (e.g., Llama-70B: 80 layers, 8192 hidden dim): `2 * 80 * 8192 * 2 bytes ≈ 2.62 MB`.
- **KV Cache per User:** `(1024 prompt + 512 completion) * 2.62 MB/token = 1536 * 2.62 MB ≈ 4024 MB ≈ 4.0 GB`.
- **Max Users per Server:** `20 GB available / 4.0 GB_per_user = 5` users.
- **Max Cluster Capacity:** `8 servers * 5 users/server = 40` concurrent users.
- **Hardware Shortfall:** `200 required_users / 40 supported_users = 5x`.
- **Recommendation:** Expand the cluster by 5x to 80 H100s to meet the 200 concurrent user goal.

  > **Key Equation:** $$N_{\text{max\_users}} = \frac{M_{\text{HBM}} - M_{\text{weights}}}{L_{\text{seq}} \times (2 \cdot N_{\text{layers}} \cdot d_{\text{model}} \cdot \text{sizeof(dtype)})}$$

  📖 **Deep Dive:** [Model Serving](https://mlsysbook.ai/vol1/serving.html)
  </details>
</details>







---


### Batching & Scheduling


#### 🔴 L6+

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Underutilized Accelerator</b> · <code>batching</code></summary>

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


---


### KV-Cache & Memory Management


#### 🟡 L5

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The OOMing Generator</b> · <code>kv-cache</code></summary>

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

  > **Napkin Math:** For a 7B LLM (e.g., Llama-2-7B with 32 layers, 32 KV heads, 128 head dim, FP16), the KV cache size per token is `32 layers × 2 (K+V) × 32 heads × 128 dim × 2 bytes/FP16 = 524 KB/token`.
  > If each request has an average sequence length of 1024 tokens, one request consumes `524 KB/token × 1024 tokens ≈ 0.5 GB`.
  > With 16 concurrent requests, this is `16 × 0.5 GB = 8 GB` of KV cache, quickly saturating a 16 GB GPU *on top of model weights*. PagedAttention can reduce this by 2-4× for typical workloads.
  > **Note on GQA:** Larger models like Llama-2-70B use Grouped Query Attention (GQA) with only 8 KV heads instead of the full 64 attention heads, reducing KV cache by 8× compared to standard multi-head attention. This is critical for making 70B-class models servable — without GQA, the KV cache alone would consume ~4 GB/request at 4k context in FP16.

  > **Key Equation:** `KV_Cache_Memory_Per_Request = SequenceLength * NumLayers * 2 * NumHeads * HeadDim * BytesPerFloat`

  📖 **Deep Dive:** [Volume I: KV-Cache Optimization for LLMs](https://mlsysbook.ai/vol1/llms/kv-cache-optimization)

  </details>

</details>


---


### Serving Architecture


#### 🟢 L1/L2

#### 🟢 L3
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


#### 🔵 L4
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
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The A/B Testing at Scale</b> · <code>serving</code> <code>deployment</code></summary>

- **Interviewer:** "We want to A/B test our current 13B model against a new 70B model in shadow mode. The product team says 'just mirror 100% of traffic to both models for a week to gather data.' Why is this statistically unnecessary but infrastructure-ruinous, and how does the GPU memory asymmetry between these models dictate your A/B testing architecture?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just spin up enough 70B instances to handle the mirrored traffic." This ignores the massive non-linear cost of serving a 70B model and the fact that statistical significance requires a fraction of that traffic.

  **Realistic Solution:** The infrastructure cost of an A/B test is dominated by hardware provisioning, not the statistical sample size. A 13B model fits on a single A100 (40GB) with room for KV-cache. A 70B model requires tensor parallelism across at least 4× A100s (80GB) or 2× H100s just to fit the weights, plus massive KV-cache overhead. Mirroring 100% of traffic means provisioning a 70B cluster that is 5-8× larger and more expensive than your production 13B cluster, just for a test. Instead, use an asymmetric traffic split (e.g., 95/5) or sample-based shadow routing. You only need to route enough traffic to the 70B model to reach statistical significance, which is typically a few thousand requests.

  > **Napkin Math:**
  > - **Memory Asymmetry:** 13B FP16 = 26GB weights. Fits on 1× A100 (40GB) with 14GB for KV-cache. 70B FP16 = 140GB weights. Requires 2× H100 (80GB) or 4× A100 (40GB). The 70B model requires 4× the GPUs per replica.
  > - **Traffic Mirroring Cost:** If production is 1,000 QPS running on 50× A100s, mirroring 100% to 70B would require ~200-300 H100 GPUs due to lower throughput and higher memory footprint. At ~$4/hr per H100, 250 GPUs × $4/hr × 168 hrs/week ≈ **$168k/week**.
  > - **Statistical Reality:** To detect a 3% quality improvement with 80% power, you need ~3,200 samples per arm. At 1,000 QPS, you collect 3,200 samples in 3.2 seconds.
  > - **Optimized Architecture:** Route 99.9% of traffic to 13B, and 0.1% (1 QPS) to a single minimal 70B deployment (2× H100s). You gather the required 3,200 samples in under an hour, costing ~$10 in GPU compute instead of $168k.

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


#### 🟡 L5

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
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Batching Starvation</b> · <code>serving</code> <code>real-time</code></summary>

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
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The LLM Canary Trap</b> · <code>rollout</code> <code>llm-serving</code></summary>

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

  > **Napkin Math:** Current: 12 models × dedicated pools. Typical utilization: 30% (peak-provisioned). Monthly cost: \$2M across ~700 GPUs at ~\$4/hr average (mix of A100 and H100 instances). Optimized: Tier 1 (8 small models, 1B–7B): consolidate from 200 GPUs to 75 (shared via MPS, 80% util). Savings: 125 GPUs × \$4/hr × 720 = \$360k/mo. Tier 2 (3 mid models, 13B–30B): right-size from 200 GPUs to 120 with predictive autoscaling. Savings: 80 GPUs × \$4/hr × 720 = \$230k/mo. Tier 3 (1 large model, 70B): reduce from 300 GPUs to 200 with continuous batching + PagedAttention. Savings: 100 GPUs × \$4/hr × 720 = \$288k/mo. **Total savings: \$878k/mo (43.9%)** — exceeds the 40% target. The additional levers (spot instances for stateless Tier 1, INT8 quantization for Tier 2, prefix caching across all tiers) provide further headroom and redundancy against traffic spikes.

  📖 **Deep Dive:** [ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Speculative Decoding Speedup</b> · <code>serving</code> <code>latency</code></summary>

- **Interviewer:** "Our 70B model serves chat completions with a P50 time-to-first-token of 200 ms and a P50 inter-token latency of 45 ms on H100. Product wants 2× faster decoding without changing the model. Someone suggests speculative decoding with a 1B draft model. Walk me through the systems math — when does this help, and when does it backfire?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The draft model is 70× smaller, so it generates tokens 70× faster, and we just verify them in parallel — easy 5-10× speedup." This ignores the acceptance rate, the memory overhead of running two models, and the verification cost.

  **Realistic Solution:** Speculative decoding works in three steps: (1) the draft model generates $K$ candidate tokens autoregressively (cheap, ~1 ms/token for a 1B model); (2) the target 70B model verifies all $K$ tokens in a single forward pass (parallel verification — same cost as generating 1 token, since decode is memory-bandwidth-bound and the extra compute for $K$ tokens is negligible); (3) the target accepts the first $n \leq K$ tokens where the draft model's distribution matches, and resamples the $(n+1)$-th token. The speedup depends critically on the **acceptance rate** $\alpha$ — the probability the draft model's token matches the target's. If $\alpha = 0.8$ and $K = 5$, the expected accepted tokens per verification step is $(1 - \alpha^{K+1})/(1-\alpha) = (1 - 0.8^6)/0.2 \approx 3.69$ tokens per target forward pass (the $1/(1-\alpha) = 5$ formula only applies in the limit as $K \to \infty$). But you also pay for the draft model's memory and compute. If the draft model's memory displaces KV-cache space, your maximum batch size drops, reducing throughput even as per-request latency improves.

  > **Napkin Math:** **Without speculation:** 70B decode = 45 ms/token (bandwidth-bound: 140 GB weights / 3.35 TB/s ≈ 42 ms + overhead). **With speculation (K=5, α=0.8):** Draft generates 5 tokens: 5 × 1 ms = **5 ms**. Target verifies: **45 ms** (one forward pass). Expected accepted tokens: $(1 - 0.8^6)/0.2 \approx 3.69$ tokens. Effective per-token latency: $(5 + 45) / 3.69 = $ **13.6 ms/token** — a **3.3× speedup**. Memory cost: 1B draft model = 2 GB + its KV-cache ≈ 2.5 GB. On 80 GB H100 with 70B model (140 GB sharded TP=2 → 70 GB/GPU), free memory drops from 10 GB to 7.5 GB — **max batch drops from ~9 to ~7 requests**. **When it backfires:** If $\alpha$ drops to 0.4 (e.g., code generation where the draft model is weak), expected accepted = $(1 - 0.4^6)/0.6 \approx 1.66$ tokens per step. Effective latency: $(5 + 45) / 1.66 = $ **30 ms/token** — only 1.5× speedup, and you've lost 25% of your batch capacity for it.

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


#### 🔴 L6+

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
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Speculative Decoding Accept Rate Crash</b> · <code>serving</code> <code>roofline</code></summary>

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


---


### Advanced Inference


#### 🔵 L4
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The KV-Cache Context Explosion</b> · <code>attention</code></summary>

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
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Inference Cost Attribution Puzzle</b> · <code>economics</code> <code>serving</code></summary>

- **Interviewer:** "Your platform serves 5 different LLMs on a shared pool of 64 A100 GPUs using vLLM with continuous batching. The finance team wants to charge each product team for their inference costs. Your colleague proposes: 'Divide total GPU cost by total requests, charge per request.' The search team (short prompts, 50-token outputs) and the content team (long prompts, 2,000-token outputs) both object. Design a fair cost attribution model."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Charge per request" or "Charge per token." Per-request ignores the 40× difference in compute between a 50-token and 2,000-token generation. Per-token ignores that prefill and decode have different costs.

  **Realistic Solution:** Fair cost attribution must reflect actual GPU resource consumption, which has two distinct phases: (1) **Prefill cost** — proportional to input tokens (compute-bound, scales as O(n²) for attention), and (2) **Decode cost** — proportional to output tokens (memory-bandwidth-bound, scales linearly but occupies KV-cache memory for the entire generation duration). The correct model charges: `cost = α × input_tokens + β × output_tokens + γ × generation_time_ms`. The α term captures prefill compute, β captures decode compute, and γ captures KV-cache memory occupancy (a long-running generation blocks memory that could serve other requests). Calibrate α, β, γ by profiling each model: measure GPU-seconds consumed per input token (prefill) and per output token (decode) at representative batch sizes. For continuous batching, the γ term is critical — a request generating 2,000 tokens at 40ms/token occupies KV-cache for 80 seconds, blocking ~2 GB of GPU memory that could serve 20 short requests.

  > **Napkin Math:** 64 A100s at $2/hr = $128/hr total. Search team: 10,000 req/hr × 500 input + 50 output tokens. Content team: 1,000 req/hr × 2,000 input + 2,000 output tokens. Per-request billing: search pays 10/11 × $128 = $116/hr, content pays $12/hr. Per-token billing: search = 5.5M tokens, content = 4M tokens → search pays $74, content pays $54. **Per-GPU-second billing (actual resource use):** search prefill ≈ 0.5s/req × 10k = 5,000 GPU-sec; search decode ≈ 2s/req × 10k = 20,000 GPU-sec; content prefill ≈ 4s/req × 1k = 4,000 GPU-sec; content decode ≈ 80s/req × 1k = 80,000 GPU-sec. Total = 109,000 GPU-sec. Content share = 84,000/109,000 = **77%** → fair charge ≈ $98/hr. Search share = 25,000/109,000 = 23% → $30/hr. The naive per-request model undercharges content by ~8× ($12 vs $98).

  📖 **Deep Dive:** [ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>


#### 🟡 L5

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
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Serverless Freeze</b> · <code>serving</code></summary>

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

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Foundation-brightgreen?style=flat-square" alt="Level 2" align="center"> Handling KV Cache Fragmentation</b> · <code>kv-cache</code></summary>

- **Interviewer:** "When serving a large language model with multiple concurrent requests, we notice that memory utilization is poor despite having free memory available, leading to out-of-memory errors for new requests. How does PagedAttention specifically solve this issue in the KV cache?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Thinking that PagedAttention compresses the KV cache to save memory rather than managing fragmentation.
  **Realistic Solution:** PagedAttention partitions the KV cache into fixed-size blocks (pages) that can be non-contiguous in memory, similar to virtual memory in OS, virtually eliminating external fragmentation.

  > **Napkin Math:**
  > - 13B model, 40 layers, 40 heads, head_dim=128, FP16. KV cache per token = `num_layers × 2 (K+V) × num_kv_heads × head_dim × bytes_per_element` = `40 × 2 × 40 × 128 × 2B` = 819 KB/token.
  > - Without paging: pre-allocate max_seq_len=2048 per request = 819 KB × 2048 = 1.6 GB. 50 concurrent requests = 80 GB (saturates an H100).
  > - With PagedAttention at avg 512 tokens/request: only allocate pages for actual tokens used. 819 KB × 512 = 0.4 GB/request × 50 = 20 GB. Memory waste drops from ~60% to <5%, freeing 60 GB for additional concurrent requests.

  > **Options:**
  > [ ] It compresses the key and value tensors using quantization before storing them in the cache.
  > [x] It partitions the KV cache into non-contiguous fixed-size blocks, eliminating external fragmentation and allowing dynamic memory allocation per token.
  > [ ] It proactively evicts the least recently used KV cache tensors to free up contiguous memory blocks for new requests.
  > [ ] It offloads the KV cache to CPU RAM when GPU memory is fragmented and prefetches it when needed.
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> Advantages of Continuous Batching</b> · <code>batching</code></summary>

- **Interviewer:** "In traditional static batching, the inference engine waits for all sequences in a batch to finish generating before starting a new batch. How does continuous batching (also known as in-flight batching) improve throughput compared to static batching?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming continuous batching increases the generation speed of individual tokens rather than the overall system throughput.
  **Realistic Solution:** Continuous batching inserts new requests into the batch as soon as a sequence finishes, instead of waiting for the longest sequence in the batch to complete, thereby reducing idle compute resources.

  > **Napkin Math:**
  > - Static batch of 8 requests: lengths [50, 100, 200, 500, 50, 80, 120, 500] tokens. All wait for longest (500 tokens).
  > - GPU idle time for shortest request: (500−50)/500 = 90% wasted. Average waste across batch: ~55%.
  > - => Continuous batching fills vacated slots immediately. Throughput improvement: typically 2-3× over static batching.

  > **Options:**
  > [ ] It increases the clock speed of the GPU dynamically to generate tokens faster for shorter sequences.
  > [x] It allows new requests to join the active batch as soon as other sequences complete, preventing GPU underutilization caused by waiting for the longest sequence.
  > [ ] It batches all requests that have the exact same prompt length together, ensuring synchronized generation steps.
  > [ ] It caches the output tokens of the continuous stream to reuse them across different users in real-time.
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Foundation-brightgreen?style=flat-square" alt="Level 2" align="center"> Mechanism of Speculative Decoding</b> · <code>speculative-decoding</code></summary>

- **Interviewer:** "Speculative decoding aims to speed up LLM inference without changing the output distribution. It uses a smaller 'draft' model and a larger 'target' model. What is the fundamental mechanism that allows it to achieve speedup?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Believing that the target model is only used when the draft model is uncertain, rather than verifying every token in parallel.
  **Realistic Solution:** The smaller draft model sequentially generates multiple candidate tokens quickly. The larger target model then verifies all these candidate tokens in a single parallel forward pass, accepting the valid ones and correcting the first mistake.

  > **Napkin Math:**
  > - Draft model (1B) generates 5 tokens: 5 × 2ms = 10ms. Target model (70B) verifies all 5 in one forward pass: 30ms.
  > - Total for 5 tokens: 40ms. Without speculation: 5 × 30ms = 150ms.
  > - => Speedup: 150/40 = 3.75× (assuming ~80% acceptance rate, effective speedup ≈ 2.5-3×).

  > **Options:**
  > [ ] The target model is used only for the first few tokens, and the draft model completes the rest of the sequence to save time.
  > [x] The draft model generates multiple tokens sequentially, which the target model then verifies in a single parallel forward pass, accepting correct tokens and correcting the first divergence.
  > [ ] The draft model and target model generate tokens in parallel, and a majority vote decides which token to output.
  > [ ] The draft model continuously fine-tunes the target model during inference to make it generate tokens faster.
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> Applying Little's Law to Inference Servers</b> · <code>queueing</code></summary>

- **Interviewer:** "According to Little's Law ($L = \lambda W$), if our LLM inference server has an average arrival rate of 10 requests per second ($\lambda$) and each request takes an average of 5 seconds to process ($W$), what does the system need to support in terms of concurrent requests ($L$) to remain stable?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Confusing the arrival rate with throughput or thinking the server only needs to handle the arrival rate concurrently.
  **Realistic Solution:** Little's Law states that the average number of items in the system ($L$) equals the arrival rate multiplied by the average time spent in the system. $10 	ext{ req/s} 	imes 5 	ext{ s} = 50$ concurrent requests.

  > **Napkin Math:**
  > - L = λ × W = 10 req/s × 5s = 50 concurrent requests.
  > - If each request needs ~1.5 GB KV cache memory: 50 × 1.5 GB = 75 GB. Requires at least one 80 GB H100.
  > - => Under-provisioning (e.g., max 30 concurrent) causes queue buildup: queue length grows at 10 − 30/5 = 4 req/s.

  > **Options:**
  > [ ] The server needs to support 5 concurrent requests at any given time.
  > [ ] The server needs to support 15 concurrent requests to have a buffer for peak loads.
  > [x] The server needs to support an average of 50 concurrent requests in the system.
  > [ ] The server needs to support 2 concurrent requests, as 10 divided by 5 is 2.
  </details>
</details>

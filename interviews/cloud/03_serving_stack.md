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

- **Interviewer:** "Your team has optimized an LLM inference server for maximum throughput, achieving 5,000 tokens/sec on a single H100. The deployment uses large static batches to maximize GPU utilization. However, user feedback is poor, with complaints that the chatbot feels 'laggy' and 'unresponsive' on the first response. You profile the system and find that while Time Per Output Token (TPOT) is extremely low (~5ms), the Time To First Token (TTFT) can be very high. Apply your knowledge of serving latency to explain this phenomenon."

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
2. **Prefill (Compute-Bound):** A new request with 1024 prompt tokens arrives. Compute is `2 * Params * Tokens`. `2 * 70B * 1024` ≈ 1.4x10¹⁷ FLOPs. On an H100 with ~989 TFLOPS, this takes `140 PFLOPs / 989 TFLOPS` ≈ 141ms.
3. **Analysis:** The system juggles these two states. If a new request arrives every 200ms, the system spends `141ms / 200ms` = ~70% of its time in the prefill state. The massive throughput gains of continuous batching are primarily in the decode phase, where many users' token generations are parallelized. If the system is constantly bogged down with expensive prefills, it can't spend enough time in the highly efficient decode state, thus limiting the overall throughput gain.

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

  > **Napkin Math:** Single 100k pass: Read 140 GB weights once. Compute 140 TFLOPs. Time = 140T / 900TFLOPS = 0.15s.
  > Chunked 25 passes: Read 140 GB weights 25 times = 3,500 GB of memory traffic. At 3.3 TB/s bandwidth, just *loading* the weights takes 1.06 seconds. TTFT increases by nearly 7x.

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

  > **Napkin Math:** For a 7B LLM (e.g., Llama-2 with 32 layers, 32 heads, 128 head dim, FP16), the KV cache size per token is `32 layers * 2 (K+V) * 32 heads * 128 dim * 2 bytes/FP16 = 524 KB`.
  > If each request has an average sequence length of 1024 tokens, one request consumes `524 KB/token * 1024 tokens ≈ 0.5 GB`.
  > With 16 concurrent requests, this is `16 * 0.5 GB = 8 GB` of KV cache, quickly saturating a 16GB GPU *on top of model weights*. PagedAttention can reduce this by 2-4x for typical workloads.

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

  > **Napkin Math:** 64 A100s at $2/hr = $128/hr total. Search team: 10,000 req/hr × 500 input + 50 output tokens. Content team: 1,000 req/hr × 2,000 input + 2,000 output tokens. Per-request billing: search pays 10/11 × $128 = $116/hr, content pays $12/hr. Per-token billing: search = 5.5M tokens, content = 4M tokens → search pays $74, content pays $54. Per-GPU-second billing (actual resource use): search prefill ≈ 0.5s/req × 10k = 5,000 GPU-sec, content prefill ≈ 4s/req × 1k = 4,000 GPU-sec, content decode ≈ 80s/req × 1k = 80,000 GPU-sec. Content uses 94% of decode resources → fair charge ≈ $108/hr. The naive per-request model undercharges content by 9×.

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

  > **Options:**
  > [ ] The server needs to support 5 concurrent requests at any given time.
  > [ ] The server needs to support 15 concurrent requests to have a buffer for peak loads.
  > [x] The server needs to support an average of 50 concurrent requests in the system.
  > [ ] The server needs to support 2 concurrent requests, as 10 divided by 5 is 2.
  </details>
</details>

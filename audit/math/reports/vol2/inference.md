# Math Audit Report: `book/quarto/contents/vol2/inference/inference.qmd`

## Checked scope

Audited inference-serving math, including serving cost equations, batching/queueing formulas, throughput and latency examples, KV-cache memory accounting, model-sharding communication estimates, MoE capacity claims, unit conversions, scaling claims, and prose-equation consistency using direct reasoning only. No Gemini assistance was used. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 968-995 and 1078-1088: the GPT-3 batching table omits batch-formation delay and therefore reports impossible low total latencies.**  
  The text defines response time as queue wait plus batch accumulation plus compute (lines 952-960), then later substitutes `B/(2 lambda)` as the batch-formation term (line 985). For `lambda = 100 req/s`, that term alone is `40 ms` at `B=8`, `80 ms` at `B=16`, and `160 ms` at `B=32`. The table's `E[T]` values are much smaller than `E[W] + B/(2 lambda) + S(B)`: for `B=32`, the listed total is `91.5 ms`, but `8.9 + 160 + 66 = 234.9 ms`. Also, the fill-case batch delay on lines 968-970 is the time to fill a whole batch from empty, not the average per-request wait; it should be about `(B_max - 1)/(2 lambda)` for a full batch under Poisson arrivals.
  - Proposed correction: Recompute `E[T]` with a consistent model. If using per-request batch accumulation, use `(B-1)/(2 lambda)` or `B/(2 lambda)` consistently and update the table and Little's Law verification. With the current parameters, `B=16` likely has lower mean latency than `B=32` once batch accumulation is included.

- **Lines 1072-1074 and 1080-1088: the "square-root law" conclusion contradicts the table produced by the full model.**  
  The approximation gives `B ≈ 6.3`, and for the stated service model `B=8` has utilization `100 * 0.054 / 8 = 67.5%`, not `>80%`. The prose says the approximation's small batch creates high utilization and the true latency optimum lies higher, but the table's own utilization values fall as batch size grows. Once the missing batch-formation delay is included, larger batches do not automatically reduce total latency.
  - Proposed correction: Replace the claim that the full model pushes the optimum toward `B=32` with a recalculated sweep that includes queue wait, batch accumulation, compute time, and memory limits. If the intended lesson is throughput rather than latency minimization, label the table as throughput-oriented capacity planning.

- **Lines 1792-1813: the RecSys embedding lookup example is off by 1000x.**  
  The calculation `10M QPS x 5000 lookups / 1000 shards` equals `50,000,000 lookups/sec per shard`, not `50 billion lookups/sec` (line 1805). With a 1 ms accumulation window, each shard receives about `50,000` lookups per batch, not `50M` (line 1811). The request count per global 1 ms batch, `10,000`, is correct.
  - Proposed correction: Change line 1805 to `50 million lookups/sec per shard` and line 1811 to `50K lookups per shard per batch`. Reassess the prose about single-threaded processing and memory-bandwidth utilization under the corrected rate.

- **Lines 2129-2151: the Llama-3-70B 128K KV-cache capacity estimator uses MHA head counts, making the max-batch conclusion wrong for Llama 3.**  
  The example names Llama-3-70B but uses `n_heads = 64` KV heads. That is an MHA baseline. The same chapter's GQA calculation uses 8 KV heads (lines 2531-2535). At 128K context, 64 KV heads give about `343 GB/request`, so `floor(480/340)=1` is arithmetically consistent for MHA. For 8 KV heads, the cache is about `42.9 GB/request`, so the same `480 GB` KV budget would support about `11` concurrent requests before other overheads.
  - Proposed correction: Either rename the example as a hypothetical 70B MHA model, or change `n_heads` to `n_kv_heads = 8` for Llama-3-70B and update the max batch from `1` to about `11`.

- **Lines 3150-3163 and 3636-3642: the tensor-parallel AllReduce latency is internally inconsistent by about 15x.**  
  Line 3163 says an 8 MB activation AllReduce over 900 GB/s NVLink takes `~0.3 ms`. The ring estimate from line 3607 is `2*(7/8)*8 MB / 900 GB/s ≈ 0.016 ms`, matching the later table's `0.02 ms` on line 3640. If the AllReduce is `0.02 ms` rather than `0.3 ms`, the 8-way TP per-layer time is about `1.5 + 0.02 + 2.25 + 0.02 = 3.79 ms`, not `4.35 ms`, and the realized speedup is about `7.9x`, not `6.9x`.
  - Proposed correction: Use one communication model throughout. Either change the per-layer table to `~0.02 ms` AllReduce and recompute speedup, or explain what extra latency/implementation overhead makes the measured AllReduce `0.3 ms` despite the bandwidth estimate.

- **Line 5194: the quantization + PagedAttention max-batch example has a MB/GB unit error.**  
  It says FP16 KV cache costs `2.5 MB per 1K tokens per sequence`, then concludes `45 GB` supports `~18` concurrent 1K-token sequences. `45 GB / 2.5 MB ≈ 18,000`, not `18`. The `~18` result corresponds to about `2.5 GB` per 1K tokens, which matches a 70B MHA-style KV cache, not MB.
  - Proposed correction: Change `2.5 MB` to `2.5 GB` if using the MHA baseline, or keep MB and update the concurrency to thousands. If using Llama-style GQA, use about `0.33 GB` per 1K tokens for FP16 KV and recompute the supported batch.

### Medium Severity

- **Lines 2024 and 2116-2123: the KV-cache equation defines `H` as hidden dimension, while later examples alternate between hidden dimension and `n_heads x d_head`.**  
  The equation `2 x L x H x S x B x s_elem` is correct if `H` is the total KV hidden width per layer. The capacity estimator instead expands `H` into `n_heads x d_head` (lines 2131-2142). This is fine algebraically, but only if `n_heads` means KV heads, not query heads. The chapter later intentionally distinguishes MHA and GQA, so the notation can easily lead to 8x errors.
  - Proposed correction: Define the equation as `2 x n_layers x n_kv_heads x d_head x S x B x s_elem`, and state that for MHA `n_kv_heads = n_query_heads`, while for GQA it is smaller.

- **Lines 2148-2149: the available-memory calculation subtracts only one copy of weights from aggregate HBM without stating the sharding assumption.**  
  `640 GB - 140 GB - 20 GB = 480 GB` assumes the 140 GB model weights are sharded once across the 8 GPUs. That is plausible for tensor/pipeline parallelism, but it is not true if each replica or pipeline stage holds duplicated weights. The max-batch result is sensitive to this assumption.
  - Proposed correction: Add "assuming the FP16 weights are sharded once across the 8-GPU node" before computing available KV memory.

- **Lines 2513-2517 and 2565-2579: the KV-cache examples mix full-model and per-GPU cache sizes without always labeling the distinction.**  
  The 4K MHA cache calculation gives about `10.7 GB` per request for the whole model. In the precision-dividend example, it is divided by 4 GPUs for per-GPU capacity planning. That division is correct only when KV heads/layers are sharded across those GPUs in the same proportion. The prose jumps from full-model cache to per-GPU cache, which can confuse whether a batch of 8 exceeds a single GPU or a 4-GPU shard group.
  - Proposed correction: Label each cache size as "per request across the full model" or "per request per GPU under 4-way tensor parallelism."

- **Lines 2871-2872: the prefill/decode complexity table overstates decode as `O(1)` per token.**  
  Decode avoids quadratic recomputation, but the attention operation for a new token still attends over the existing sequence, so KV-cache attention work and KV reads scale as `O(context_length)` per generated token. The weight-read term can dominate for small or moderate context, but the complexity is not generally `O(1)`.
  - Proposed correction: Change decode computation to `O(context_length)` for attention plus `O(model size)` weight reads, and explain that in common decode regimes the model-weight bandwidth term dominates.

- **Lines 2978 and 3094: minimum sharding for a 140 GB model on 80 GB GPUs ignores non-weight memory.**  
  Two 80 GB GPUs provide 160 GB raw HBM, leaving only about 20 GB total for KV cache, activations, runtime workspace, fragmentation, and system overhead. Calling 2-way sharding the "minimum" is true for weights only, but not for a usable serving configuration.
  - Proposed correction: Say "minimum for weights only is 2-way; practical serving usually needs more devices or quantization to leave KV-cache headroom."

- **Lines 3194, 3206-3207, and 4021: pipeline-parallel throughput scaling is presented as `p x` without the fill/drain qualification.**  
  In steady state with many independent requests and balanced stages, throughput can approach `p` times a single full-model replica's throughput. For finite batches or imbalanced stages, throughput is limited by the slowest stage and bubble overhead. The text mentions fill but the table states `p x` as if exact.
  - Proposed correction: Change the table to "up to `p x` in steady state with balanced stages" and mention bubble/imbalance losses.

- **Lines 3312-3316: the MoE hardware takeaway uses an unclear comparison baseline.**  
  The example says the FP8 671B MoE requires two 8-H100 nodes and concludes it needs `2x` more hardware for memory capacity. That is `2x` relative to one 8-H100 node, but much more than a 37B dense active-parameter model and less than a dense 671B model under the same precision. The comparison baseline changes between total parameters, active parameters, and node count.
  - Proposed correction: State the baseline explicitly, e.g. "2 nodes rather than 1 node for memory, while per-token bandwidth resembles a 37B dense model."

- **Lines 3607-3619: communication equations mix latency symbols and simplified bandwidth models.**  
  The point-to-point equation uses `L`, but the explanation says `alpha` is latency (line 3615). The AllToAll formula assumes serialized sends of `M/N` to each peer; on real fabrics, concurrent links and collective implementation can change the constant substantially.
  - Proposed correction: Use one latency symbol consistently and label the AllToAll expression as a simple serialized upper-bound/estimate unless the intended topology is specified.

### Low Severity

- **Line 199: the 1 GB serialization example is underspecified and conflicts with the zero-copy footnote's "microseconds" phrasing.**  
  `1 GB / 100 ms` implies an effective serialization bandwidth of only `10 GB/s`, which may be reasonable for some CPU serialization/copy paths but not for zero-copy tensor handoff. The footnote says zero-copy drops per-hop overhead to microseconds, so the two statements need clearer scope.
  - Proposed correction: Specify whether the 100 ms includes CPU encode/decode, memory copies, or network transfer, and distinguish it from zero-copy metadata serialization.

- **Lines 781-782: the batching throughput curve is concave, not super-linear in batch size.**  
  For `X(B)=B/(a+bB)`, throughput increases with batch size but with diminishing marginal gains (`X''(B)<0`). It can improve dramatically from tiny batches, but it is not super-linear as a function of `B`.
  - Proposed correction: Replace "super-linear throughput gains" with "large but diminishing throughput gains."

- **Lines 852 and 1149: queue growth language is imprecise.**  
  Little's Law does not imply queues "build up exponentially" after capacity; an unstable queue grows without bound over time. The M/M/1 curve `rho/(1-rho)` diverges hyperbolically as utilization approaches 1.
  - Proposed correction: Use "queues grow without bound when arrivals exceed service capacity" and reserve exponential language for tail distributions only when derived.

- **Line 2611: the draft-model timing claim for speculative decoding is too broad.**  
  Generating `K` draft tokens in the time of one or two target decode steps depends on draft/target size ratio, hardware placement, KV-cache state, and batching. The later model assumes a 20x faster draft and `K=5`, giving `0.25` target-step equivalents, not one or two.
  - Proposed correction: Qualify the sentence as a typical design target and align it with the later normalized timing assumption.

- **Lines 2878 and 5139: 4-bit/FP4 decode speedup claims assume pure memory-bandwidth limitation and no overhead.**  
  Halving or quartering weight bytes can double or quadruple the bandwidth roofline, but realized throughput can be lower due to dequantization, KV-cache traffic, attention work, communication, and scheduler overhead.
  - Proposed correction: Say "up to" 2x or 4x under a weight-bandwidth-bound decode roofline, then note common overheads.

- **Line 5619: the final hardware check may fail if the H100 memory constant is exactly 80 GB.**  
  The prose expects an 80 GB H100, but the guard uses `h100_mem_gb_val > 80`. If the constant is exactly 80, the check rejects a valid value.
  - Proposed correction: Use `>= 80` or check against a narrower expected range.

## Verified Correct

- **Lines 242-247:** The simple serving-cost example is arithmetically consistent after rounding: `1,000,000 x 50 x 365 = 18.25B` requests, about `$18.25M` at `$0.001/request`, roughly `9x` a `$2M` training cost.
- **Lines 253-255:** The total-cost equation is dimensionally coherent if `C_serving` is cost/query and `T_deployment x Q_rate` yields total queries.
- **Lines 299-305 and 338-358:** The DLRM serving-cost multiplier calculation is internally consistent: training cost is `$12K`, lifetime traffic is about `6.31e11` queries, serving cost at `$10 per million` is about `$6.31M`, and the ratio is about `526x`.
- **Lines 770-777 and 817-824:** The batching efficiency knee arithmetic is correct for the stated approximation: `40/0.5 = 80` for LLM decode and `2/1 = 2` for the vision example.
- **Lines 1270-1301:** The ResNet dynamic batching example's utilization arithmetic is consistent: no batching gives `rho = 2.5`, option B gives `500 x 0.0016 = 0.8`, and option C gives `500 x 0.0012 = 0.6`.
- **Lines 1335-1339 and 1572-1576:** The static batching waste examples are arithmetically correct: `(100-10)/(100 x 8)=11.25%`, and `1 - 125/200 = 37.5%`.
- **Lines 1611-1623:** The continuous-batching table is consistent with `speedup = L_max / mean = 1/(1 - waste)` after rounding.
- **Lines 2531-2535:** The GQA KV-cache calculation is correct: reducing 64 KV heads to 8 KV heads reduces the 4K FP16 cache from about `10.7 GB` to about `1.34 GB`.
- **Lines 2673-2750:** The speculative decoding expected-token and speedup arithmetic matches the stated model: for `K=5`, `alpha=0.9`, and draft cost `0.25` target steps, speedup is about `3.7x`; lower acceptance rates produce the listed smaller gains after rounding.
- **Lines 2972-2978 and 3090-3094:** The FP16 weight-memory calculation for a 70B model is correct: `70B x 2 bytes = 140 GB`, exceeding a single 80 GB GPU.
- **Lines 3274-3276:** The MoE dense-vs-active bandwidth arithmetic is consistent: `400B x 2 = 800 GB`, `37B x 2 = 74 GB`, and the ratio is about `10.8x`.
- **Lines 3840-3864:** The heterogeneous GPU weighted-routing example is arithmetically consistent: total capacity is `22,000 QPS`, weighted per-server loads are about `682 QPS` for H100 and `409 QPS` for A100, both about `68%` utilization.

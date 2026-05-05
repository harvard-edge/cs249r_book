# Math Audit: vol1/model_serving/model_serving.qmd

Audited source: `book/quarto/contents/vol1/model_serving/model_serving.qmd`

Scope: serving throughput, latency, batching, queueing, cache, memory, unit conversions, scaling claims, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### High: adaptive batching pseudocode mixes seconds and milliseconds

- Lines: 2948-2969
- Issue: `adaptive_batching_window(queue_depth, arrival_rate, slo_ms)` computes `max_wait = slo_ms * 0.3`, which is in milliseconds, but `estimated_wait = requests_needed / arrival_rate`, which is in seconds if `arrival_rate` is requests/second. The returned `min(estimated_wait, max_wait)` compares unlike units.
- Why it matters: At `queue_depth = 0`, `arrival_rate = 1000 QPS`, `target_batch_size = 16`, the estimated wait is `16 / 1000 = 0.016 s = 16 ms`. The code compares `0.016` to `15` and returns `0.016`, which would be interpreted as 0.016 ms if the function contract is milliseconds. That is a 1000x error in the batching window.
- Proposed correction: Convert the estimate to milliseconds:
  `estimated_wait_ms = requests_needed / arrival_rate * 1000`, then return `min(estimated_wait_ms, max_wait_ms)`. Alternatively rename the API to `slo_s` and keep all values in seconds.

### High: Llama-3-8B KV-cache capacity ignores GQA even though Llama-3-8B uses it

- Lines: 4048, 4108-4123, 4154-4158
- Issue: The case study is explicitly for Llama-3-8B, but the KV-cache estimate uses `0.5 MB/token` from standard multi-head attention. The prose notes that Grouped Query Attention can reduce KV memory by up to 4x, but treats that as an aside rather than applying it to Llama-3-8B.
- Reasoning: The stated formula is `32 layers x 4096 dim x 2 vectors x 2 bytes = 524,288 bytes ~= 0.5 MB/token` for full KV heads. Llama-3-8B uses GQA, with fewer KV heads than query heads, so the per-token KV cache is about one quarter of that, roughly `0.125 MB/token` before implementation overhead. With 72 GB available, capacity is roughly `72,000 / 0.125 = 576,000 tokens`, not `144,000 tokens`. For `1,256 tokens/request`, the concurrency estimate becomes roughly `458 requests`, not `114`.
- Why it matters: The case study’s central claim is that memory capacity bounds throughput and cost. Using the wrong attention layout underestimates concurrency by about 4x and therefore distorts the unit economics.
- Proposed correction: Either change the model to a standard-MHA 8B model, or apply GQA in the calculation:
  `kv_per_token_mb_value = 0.125` for Llama-3-8B-style GQA, then recompute `kv_capacity_tokens`, `concurrent_batch`, tokens/hour, and cost per million tokens. Also update line 4156 to say the estimate already includes GQA.

### Medium: N+2 redundancy GPU count is overstated

- Lines: 2087-2094
- Issue: The text says N+2 redundancy would require "eleven or twelve GPUs." That does not follow from the preceding capacity numbers.
- Reasoning: The safe-utilization requirement is `rho <= 0.72`, peak traffic is `5000 QPS`, and each V100 provides `1143 img/s`. To survive two failures:
  `n - 2 >= 5000 / (1143 * 0.72) = 6.08`, so `n >= 9`. Nine GPUs leave seven active after two failures, with utilization `5000 / (7 * 1143) = 62.5%`, below the 72% threshold. Even if preserving the already-added 30% variance headroom after two failures is desired, ten GPUs leave eight active and utilization `54.7%`.
- Proposed correction: Replace "eleven or twelve GPUs" with "nine GPUs under the same safe-utilization threshold, or about ten GPUs if the 30 percent headroom must remain after two simultaneous failures."

### Medium: carbon-cost example contradicts its own 10x utilization claim

- Lines: 3396-3429, 3463
- Issue: The comments/docstring claim poor utilization causes 10x higher energy per token, but the implemented calculation gives only about 3x relative to the full-utilization example and omits host overhead in the low-utilization denominator.
- Reasoning: Full-utilization energy is `(700 W + 300 W) / (114 * 7.5 tok/s) = 1000 / 855 = 1.17 J/token`. The low-utilization code computes `300 W / 85.5 tok/s = 3.51 J/token`, a 3.0x increase. If the same total platform power were used at 10% throughput, it would be `1000 / 85.5 = 11.7 J/token`, which is 10x the full-utilization value.
- Proposed correction: Decide which model is intended:
  - For a 10x claim, compute `cc_low_util_joules_value = cc_total_power_w_value / cc_low_util_tokens_sec_value`.
  - For idle-GPU-only power, change the comment and prose to "about 3x" and clarify that host power is excluded.

### Medium: "time in system" is mislabeled as wait time in the M/M/1 notation discussion

- Lines: 1842-1844, 1948-1950, 1968, 1986-1990
- Issue: The notation alert says `$T_{\text{lat}}$ denotes wait time (time in system per request)`, conflating queue wait with response time. Later equations use `W = 1 / (mu - lambda)`, which is M/M/1 mean response time in system, not queue-only wait. Queue-only wait is `W_q = rho / (mu - lambda)`.
- Why it matters: The chapter later distinguishes `L_lat,wait` from compute/service time. Calling time-in-system "wait time" can lead readers to double-count service time or use the wrong M/M/1 formula.
- Proposed correction: Change line 1844 to: "`T_{\text{lat}}` denotes response time or time in system; queue-only waiting time is `W_q`." Keep line 1968's "wait + service" phrasing.

### Medium: batching-window guideline uses average wait but calls it a maximum window

- Lines: 2999, 3038-3041, 3056-3058
- Issue: The prose says allocating 20-30 percent of the SLO to batching wait time "bounds the maximum window at `T_max = 0.3 x L_lat,SLO`." But earlier the chapter correctly states average wait is `T/2`, while p99/worst-case wait is approximately `T`.
- Reasoning: If the budget is for average wait, then `T_max = 2 * 0.3 * SLO`. If the budget is for p99/worst-case wait, then `T_max = 0.3 * SLO`. The current text does not specify which, and the worked example uses `15 ms` as both "batching budget" and "maximum window."
- Proposed correction: Make the percentile explicit:
  - "Allocate 30 percent of the p99 SLO to worst-case batching wait, so `T_max = 0.3 L_SLO`"; or
  - "Allocate 30 percent to average wait, so `T_max = 2 * 0.3 L_SLO`, subject to p99 validation."

### Low: Llama-3-8B model weight size is likely low for 4-bit 8B weights

- Lines: 4063-4074, 4108, 4147
- Issue: The case study uses `3.5 GB` for Llama-3-8B 4-bit weights. A direct parameter count estimate is `8B parameters x 0.5 bytes = 4.0 GB` before quantization metadata/scales. Some packed artifacts may differ, but 3.5 GB is low for a literal 8B model.
- Proposed correction: Use approximately `4.0-4.5 GB` for 4-bit 8B weights, or state that `3.5 GB` is an effective post-compression artifact size and adjust the token-time example accordingly. On H100 at 3.35 TB/s, `4.0 GB / 3.35 TB/s ~= 1.2 ms/token` theoretical; rounded to 1 ms remains fine.

### Low: table values for Poisson batch variability are numerically consistent

- Lines: 2640-2653
- Check: For a 10 ms window, means are `lambda T = 0.5, 2, 5, 10`; standard deviations are `sqrt(mean)`; `P(batch=0)=e^-mean`; and the listed `P(batch >= 2 x mean)` values are consistent when the threshold is interpreted as integer counts `>=1, >=4, >=10, >=20`.
- Proposed correction: No numeric correction required. To avoid ambiguity, add a note that the tail probability uses the nearest integer event threshold, e.g. `P(K >= ceil(2 lambda T))`.

### Low: capacity-planning p99 adjustment should avoid saying M/D/1 p99 is "roughly half"

- Lines: 1956-1958, 1986-1990, 2071-2072
- Issue: The text correctly says M/M/1 is conservative for deterministic inference, but "p99 values are roughly half" is too broad. The "half" relationship is exact only for the mean queue wait under M/D/1 versus M/M/1 with the same utilization and deterministic service-time assumptions; percentile response times do not scale by a universal 0.5 factor.
- Proposed correction: Replace "p99 values are roughly half" with "mean queueing delay is roughly half; percentile response times are also lower but should be computed or simulated for the target service distribution."

## Overall Assessment

Most arithmetic examples in the latency budget, Little's Law, M/M/1 utilization table, batching throughput tables, model-swap transfer time, runtime speedups, and precision speedups are internally consistent. The main corrections needed are not basic arithmetic slips; they are unit discipline and consistency between the chosen model assumptions and the prose that interprets them.

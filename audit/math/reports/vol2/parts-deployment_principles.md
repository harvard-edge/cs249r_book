# Math Audit Report: `book/quarto/contents/vol2/parts/deployment_principles.qmd`

## Checked scope

Audited `book/quarto/contents/vol2/parts/deployment_principles.qmd` for deployment-at-scale math, throughput/latency/cost equations, numeric examples, unit conversions, scaling claims, asymptotic claims, and prose-equation consistency using direct reasoning only. No Gemini assistance was used.

The file contains one displayed asymptotic expression and several quantitative or quasi-quantitative claims: thousands of servers, billions of edge devices, decode-phase memory-bandwidth limits, batch-size throughput scaling, inference OpEx exceeding training CapEx by `100x-1000x`, and Power-of-Two-Choices tail/load scaling.

## Findings

### Medium: Part numbering is internally inconsistent

- **Lines:** 1, 26, 28
- **Current text:** The title says `# Part III: Deployment at Scale`, but the roadmap heading says `Part VII Roadmap`, and line 28 says "Part VII takes the trained model from the cluster to the world."
- **Issue:** The file identifies the same part as both Part III and Part VII. This is a prose/count consistency defect. A reader cannot tell whether the numbering is local to Volume 2 or global across the full book.
- **Proposed correction:** Use one numbering scheme consistently. If this is the third part of Volume 2, change lines 26 and 28 to "Part III." If this is globally Part VII across both volumes, change line 1 to `# Part VII: Deployment at Scale`.

### Medium: Decode is stated as strictly memory-bandwidth bound without scope

- **Lines:** 7-10
- **Current text:** "In generative models, the 'Decode' phase is strictly memory-bandwidth bound because the entire model weight set must be loaded for every single token generated." The implication adds: "Throughput scales with batch size (sharing weight loads across multiple requests), not compute power."
- **Issue:** The direction is often right for dense transformer decode at small or moderate batch sizes: per generated token, the model performs a relatively small amount of work per parameter loaded, so memory bandwidth can dominate. But "strictly" and "entire model weight set" are too broad as a mathematical invariant. The active weights can be less than the full parameter set for sparse/MoE models; weights may be reused across a decode batch; and at sufficiently large batch, long-context attention, quantization/dequantization overhead, speculative decoding, or hardware balance can shift the bottleneck toward compute, interconnect, or KV-cache bandwidth. Throughput also cannot scale with batch size indefinitely because latency budgets and resource ceilings cap usable batch size.
- **Proposed correction:** Scope the claim. For example: "For dense autoregressive transformers during decode, especially at low-to-moderate batch sizes, token generation is often memory-bandwidth limited because active layer weights must be streamed for each decode step. Batching improves arithmetic intensity by sharing weight reads across requests, until latency, compute, KV-cache, or interconnect limits dominate."

### Medium: The `100x-1000x` serving-cost dominance claim is not generally derivable

- **Lines:** 13-16
- **Current text:** "Over a successful model's lifetime, Inference OpEx exceeds Training CapEx by 100$\times$-1000$\times$."
- **Issue:** This is a quantitative economic claim stated as an invariant, but no assumptions are given for request volume, tokens per request, model size, hardware amortization, energy price, utilization, training run count, model lifetime, or whether "training CapEx" means hardware purchase cost, allocated depreciation, or one-time training compute spend. The comparison also mixes OpEx and CapEx categories, which can be valid in a total-cost model but needs a time horizon and accounting convention. Some high-traffic products may satisfy the claim; low-traffic, short-lived, internal, or frequently retrained models may not.
- **Proposed correction:** Recast as workload-dependent and define the accounting basis. For example: "For high-traffic production models, cumulative inference serving cost can exceed one-time training cost over the model lifetime, sometimes by orders of magnitude. The ratio depends on traffic, tokens/request, model size, utilization, hardware amortization, energy, and deployment lifetime."

### High: The P2C asymptotic expression is mathematically imprecise and mismatched to tail latency

- **Lines:** 19-23
- **Current text/equation:** "When load balancing, querying just two random replicas and selecting the least-loaded one exponentially reduces tail latency compared to random selection." The displayed expression is `$$ O(\log n) \to O(\log \log n) $$`.
- **Issue:** The classic Power-of-Two-Choices result is about maximum load in a balls-into-bins model, not directly tail latency. With `n` balls and `n` bins, one random choice gives maximum load about `\Theta(\log n / \log \log n)` with high probability, while two choices gives about `\Theta(\log \log n)` up to constants. The displayed `O(\log n) -> O(\log \log n)` omits the `\log \log n` denominator for the random-choice baseline and uses only big-O, which weakens the comparison and can make a loose upper bound look like the theorem. Tail latency may improve as a consequence of lower queue imbalance, but that requires a queueing/load model not shown by this expression.
- **Proposed correction:** Tie the equation to maximum load, then state latency as an implication. For example: "In the standard `n` balls into `n` bins model, random placement has maximum load `\Theta(\log n / \log \log n)`, while two choices reduces this to `\Theta(\log \log n)` with high probability. Lower queue imbalance can reduce tail latency in serving systems when queue length is the dominant latency contributor."

### Low: "Exponentially reduces" is not the right description for the shown asymptotic change

- **Lines:** 20-21
- **Current text/equation:** "exponentially reduces tail latency" followed by `O(\log n) \to O(\log \log n)`.
- **Issue:** Even accepting the displayed expression, the change from `\log n` to `\log \log n` is a logarithmic-to-iterated-logarithmic improvement, not an exponential reduction of the quantity being plotted. The phrase may be trying to express that the number of bins supported at a given load imbalance grows exponentially, but that is the inverse interpretation and is not what the sentence says.
- **Proposed correction:** Replace "exponentially reduces tail latency" with a mathematically precise phrase such as "dramatically reduces maximum load imbalance" or "improves the maximum-load scaling from logarithmic-like to iterated-logarithmic-like under the standard balls-into-bins model."

### Low: "Thousands of servers and billions of edge devices" is plausible but unqualified

- **Lines:** 3-5
- **Current text:** The Global Inference Fleet is described as "the thousands of geographically distributed servers and billions of edge devices that serve model outputs to users in real-time."
- **Issue:** This is not an arithmetic error, but it is a scale claim presented as if it applies to the fleet built by Parts I and II. "Billions of edge devices" is plausible for global consumer deployment, but not for every production inference fleet. The text later discusses edge deployment as an option, so the opening sentence may overstate the default deployment scale.
- **Proposed correction:** Qualify the range. For example: "the geographically distributed servers, and in some products billions of edge devices, that serve model outputs to users in real time."

## No-Issue Checks

- **Lines 3-5:** The qualitative trade-off between datacenter throughput, user latency budgets, and edge memory/power constraints is directionally consistent. No equation or unit conversion is attached.
- **Lines 10, 30-32:** The importance of continuous batching, PagedAttention, KV-cache management, sharding, and compression is consistent with inference-serving throughput and memory-pressure concerns, apart from the over-broad bottleneck wording noted above.
- **Lines 30-33:** The roadmap lists four chapters and then provides exactly four numbered items, so the item count is internally consistent.
- **Line 14:** The `100x-1000x` notation is syntactically clear as a multiplicative range; the issue is the unsupported universality of the economic claim, not the range formatting itself.

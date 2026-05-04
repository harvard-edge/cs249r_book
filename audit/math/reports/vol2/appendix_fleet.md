# Math Audit: `book/quarto/contents/vol2/backmatter/appendix_fleet.qmd`

Scope: fleet constants, reliability/scale/throughput/capacity equations, numeric examples, unit conversions, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. Overhead budget fractions render as 100x too small

- **Lines:** 72-77, 522-529, 675-676
- **Severity:** High
- **Issue:** The overhead constants are stored as fractions (`0.05`, `0.03`, `0.10`, `0.05`), but the overhead table and compound-overhead prose append `percent` or `%` directly to those raw values. This renders as `0.05%`, `0.03%`, `0.10%`, and `0.05%`, while the caption and goodput calculation clearly intend 5%, 3%, 10%, and 5%.
- **Explanation:** A fraction of `0.05` equals 5 percent, not 0.05 percent. The stated goodput calculation uses `(1 - 0.05)`, so the display should show `5%` for pipeline bubbles, not `0.05%`.
- **Proposed correction:** Format these values with `fmt_percent(...)` or multiply by 100 before appending `%`. For example, render pipeline bubbles as `~5%`, checkpointing as `~3%`, failure recovery as `~10%`, and maintenance as `~5%`.

### 2. 7B AllReduce example is communication-bound, contradicting the prose

- **Lines:** 696, 708-717, 754, 760
- **Severity:** High
- **Issue:** The code comments say the 7B data-parallel example should have `rho` around `0.1`, and the prose says communication can be partially overlapped. But the specified numbers produce `rho > 11`, meaning communication dominates computation.
- **Explanation:** The gradient is `7e9 * 2 = 14 GB`. Ring AllReduce over 256 GPUs at 50 GB/s takes approximately `2*(255/256)*14/50 = 0.558 s` in the bandwidth term, plus about `2*255*5 us = 2.55 ms` latency. Total communication is about `560 ms`. Dividing by `50 ms` compute gives `rho ~= 11.2`, so communication is about `1,120%` as long as computation, not a small overlappable cost.
- **Proposed correction:** Either change the prose to state that this naive full-gradient, single-link IB NDR data-parallel setup is communication-bound, or change the example assumptions. To make `rho ~= 0.1` with 50 ms compute, communication must be about 5 ms, requiring a much smaller effective message, much higher effective bandwidth, hierarchical/sharded communication, or a much larger compute interval.

### 3. Goodput is computed additively but described as multiplicative

- **Lines:** 111, 529, 791-805, 811-817
- **Severity:** Medium
- **Issue:** `goodput_ratio` is computed as `1 - (0.05 + 0.03 + 0.10 + 0.05) = 0.77`, but the table caption says the compound effect is multiplicative and writes `(1 - 0.05)(1 - 0.03)(1 - 0.10)(1 - 0.05) ≈ 77%`.
- **Explanation:** The displayed product is `0.95 * 0.97 * 0.90 * 0.95 = 0.7878`, or about `79%`, not `77%`. The effective FLOPS calculation also uses the additive `0.77`, yielding `0.50 * 0.50 * 0.77 = 19.25%` of peak. With the multiplicative overhead model, it would be about `19.7%`.
- **Proposed correction:** Choose one model. If overheads are additive budget shares, change the caption to `1 - (0.05 + 0.03 + 0.10 + 0.05) = 77%`. If the intended model is multiplicative survival through independent loss factors, compute `goodput_ratio` as the product and update the effective FLOPS output accordingly.

### 4. Ring AllReduce byte-volume prose confuses per-GPU traffic with total traffic

- **Lines:** 338-340
- **Severity:** Medium
- **Issue:** The text says Ring AllReduce transfers `2(N-1)/N * M` bytes and that "the total bytes transferred is nearly independent of the number of GPUs."
- **Explanation:** `2(N-1)/N * M` is the per-participant bandwidth term used for the ring time model under ideal concurrent links. Cluster-wide aggregate traffic is closer to `2(N-1)M`, which grows linearly with `N`. The current wording can make readers think the whole network moves only about `2M` bytes regardless of cluster size.
- **Proposed correction:** Replace "total bytes transferred" with "bytes transferred per participant in the time model" or "per-link/per-GPU bandwidth volume." Note separately that aggregate network traffic still grows with participant count.

### 5. Weak-scaling explanation overstates quadratic compute scaling

- **Lines:** 766-768
- **Severity:** Medium
- **Issue:** The prose says that as the model grows, computation per step grows quadratically while communication grows linearly, so `rho` improves for larger models.
- **Explanation:** For transformer training at fixed tokens per step, FLOPs scale roughly linearly with parameter count, and gradient communication also scales linearly with parameter count. `rho` improves only when the computation per synchronization grows faster than communicated bytes, such as from larger global batch, gradient accumulation, longer sequences, or changed model/data scaling assumptions. It is not automatic from parameter count alone.
- **Proposed correction:** Qualify the claim: "When weak scaling also increases tokens per synchronization point, sequence length, or accumulation work, compute per synchronization can grow faster than gradient bytes, improving `rho`." Avoid stating quadratic scaling as a general fleet law.

### 6. PUE reduction percentage is inverted

- **Lines:** 852-861
- **Severity:** Medium
- **Issue:** The caption says the gap between legacy PUE `1.58` and liquid-cooled PUE `1.06` represents a `49 percent reduction in total facility power`.
- **Explanation:** For the same IT load, total facility power drops from `1.58` units to `1.06` units. The reduction relative to legacy is `(1.58 - 1.06) / 1.58 = 32.9%`, not 49%. The ratio `1.58 / 1.06 = 1.49` means legacy uses about 49% more total power than liquid-cooled, not that liquid cooling reduces total power by 49%.
- **Proposed correction:** Change to "a 33 percent reduction in total facility power relative to legacy" or "legacy requires about 49 percent more facility power than liquid-cooled."

### 7. "Failure is a certainty within any run longer than a few hours" overstates the 8,192-GPU probability

- **Lines:** 96-107, 470-479, 875-876, 893
- **Severity:** Low
- **Issue:** The MTBF table arithmetic is internally consistent, but the prose says that at 8,192 GPUs and above, failure is a certainty within any training run longer than a few hours.
- **Explanation:** With a 50,000-hour per-GPU MTTF, the 8,192-GPU MTBF is `50,000 / 8,192 = 6.10 hours`. The chance of at least one failure in 3 hours is `1 - exp(-3/6.10) = 38.8%`; in 6 hours it is `62.6%`; in 24 hours it is `98.0%`. Failure is highly likely over day-scale runs, but not certain after only "a few hours" at the 8K scale.
- **Proposed correction:** Say "highly likely over day-scale training windows" for 8,192 GPUs, and reserve "effectively certain within a few hours" for larger clusters such as 100,000 GPUs, whose GPU-only MTBF is about 30 minutes.

### 8. Checkpoint-size table includes gradients, which may not be checkpointed

- **Lines:** 147-151, 245, 483, 509-516
- **Severity:** Low
- **Issue:** The appendix labels the 16 bytes/parameter calculation as checkpoint size and includes `2B for gradients`. Gradients are usually transient step state, not persisted in a standard optimizer checkpoint.
- **Explanation:** The table arithmetic is correct for 16 bytes/parameter: 7B gives `112 GB`, 70B gives `1,120 GB`, 175B gives `2,800 GB`, and 1T gives `16 TB`. But if the intended quantity is persisted mixed-precision Adam checkpoint state, a common accounting is 14 bytes/parameter: FP16/BF16 weights, FP32 master weights, FP32 momentum, and FP32 variance. That would make the table about 12.5% smaller.
- **Proposed correction:** Either rename the quantity to "training state footprint" and keep 16 bytes/parameter, or remove gradients from checkpoint accounting and use 14 bytes/parameter for persisted checkpoint sizes.

## Checks That Look Consistent

- Lines 87-88, 386-397, and 891-892: NVLink H100 bandwidth of 900 GB/s divided by IB NDR bandwidth of 50 GB/s gives the stated 18x hierarchy gap.
- Lines 96-107 and 470-477: GPU-only MTBF values follow `50,000 hours / N`, and the 24-hour failure probabilities follow `1 - exp(-24 / MTBF)`.
- Lines 509-516: The listed checkpoint write times are consistent with the table's 16 bytes/parameter assumption and 100 GB/s aggregate storage bandwidth.
- Lines 622-673 and 867-868: The Amdahl example is arithmetically consistent for a 10% serial fraction; the asymptotic speedup limit is 10x.
- Lines 731-736 and 756-762: The tensor-parallel NVLink transfer is consistent: `16 MB / 900 GB/s = 0.0178 ms`, giving `rho ~= 0.018` for a 1 ms layer.
- Lines 827-828 and 544-546: `10,000 * 700 W = 7 MW`; multiplying by PUE `1.4` gives `9.8 MW`.
- Lines 835-844: The rack power-density ratio is consistent: `70 kW / 12 kW = 5.83x`, rendered as about `5.8x`.
- Lines 852-861: The absolute PUE savings example is correct: a 10 MW IT load requires 15.8 MW at PUE 1.58 and 10.6 MW at PUE 1.06, for a 5.2 MW difference.

# Math Audit Report: `book/quarto/contents/vol1/training/training.qmd`

## Checked scope

Audited the chapter for equations, derivations, numeric examples, unit conversions, complexity/scaling claims, tensor-shape claims, optimization math, and prose-equation consistency using direct reasoning only. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 213, 237, and 4391: FlashAttention is described as reducing FLOP count, but the chapter later correctly says it does not.**  
  The iron-law table says FlashAttention affects `Total Operations ↓` because it "reduces FLOP count" (line 213), and the historical summary says FlashAttention reduces `O` (line 237). But the detailed FlashAttention section states that standard attention and FlashAttention both require `O(n^2 d)` FLOPs and that FlashAttention adds roughly 20 percent more FLOPs in backward recomputation (line 4391). The correct improvement is reduced HBM traffic and peak activation memory, not reduced mathematical work.
  - Proposed correction: Change the FlashAttention row to `Utilization ↑ / Memory traffic ↓` and change the mechanism to "same asymptotic FLOPs, much lower HBM IO; may add recomputation FLOPs." On line 237, replace "FlashAttention reduces `O` while improving `eta`" with "FlashAttention reduces memory traffic while improving `eta`."

- **Lines 967, 971-978, and 2798-2803: Adam memory multipliers mix optimizer-state-only, model-state, and inference-baseline definitions.**  
  Line 967 says "optimizer state" for a 7B model consumes 84 GB, but Adam's two FP32 moment tensors alone are `7B x 2 x 4 bytes = 56 GB`. The 84 GB total is FP16 weights (14 GB) + FP16 gradients (14 GB) + FP32 Adam moments (56 GB), i.e. training model state before activations. Similarly, lines 971-978 list SGD as `1x`, momentum as `2x`, and Adam as `3x`, which is only true if the baseline is FP32 parameter storage and the multiplier counts parameters plus optimizer auxiliary state, not gradients. Lines 2798-2803 use the 84 GB decomposition correctly but call the optimizer line 56 GB, exposing the inconsistency.
  - Proposed correction: Use three explicit categories: weights, gradients, and optimizer state. Say "Adam optimizer state alone is 56 GB; total FP16 training state before activations is 84 GB." In the optimizer table, label the multiplier as "parameters + optimizer auxiliary state, excluding gradients" or revise the rows to include gradients explicitly.

- **Lines 2757-2762 and 2800: activation-memory example reports 2 GB but the stated formula gives about 0.54 GB.**  
  The callout says activations use `Batch x SeqLen x Hidden x Layers x Bytes` and gives batch 1, sequence 2048, hidden 4096, 32 layers, FP16. That arithmetic is `1 x 2048 x 4096 x 32 x 2 = 536,870,912 bytes`, about 0.54 GB decimal or 0.5 GiB, not 2 GB. A 2 GB value may be reasonable if the estimate includes multiple stored tensors per transformer block, but that multiplier is not in the displayed formula.
  - Proposed correction: Either change the displayed activation value to about 0.5 GB for the simplified formula, or include an explicit multiplier of about 4 for stored attention/MLP/intermediate tensors: `Batch x SeqLen x Hidden x Layers x Bytes x activation_factor ≈ 2 GB`.

### Medium Severity

- **Lines 500-525: the GPT-2 attention computation callout says "single attention head" but uses all heads and full-layer totals.**  
  The prose introduces the example "For a single attention head" (line 500), but the formulas and code use `heads = 25` (lines 425, 512-514), full hidden dimension projections, full attention score cost across all heads, FFN cost, and all 48 layers for the per-step total (lines 523-525). The arithmetic is broadly consistent for a full transformer layer, but the text's scope is wrong.
  - Proposed correction: Replace "For a single attention head" with "For one GPT-2 XL transformer layer across all 25 heads." Also change "petaFLOPS total training computation" on line 525 to "petaFLOPs" or "quadrillions of FLOPs" to avoid using the rate unit for total work.

- **Line 3214: the MFU example's 1.2 ms step time is off by three orders of magnitude for the stated workload.**  
  The arithmetic shown is `7B x 6 / (312e12 x 1.2e-3) = 0.112`, so the displayed 11 percent follows from the formula. But a 1.2 ms full training step for a 7B transformer is not plausible: even the simplified `6N` forward-pass count gives 42 billion FLOPs, and 1.2 ms implies 35 TFLOP/s of useful work before backward/optimizer costs. If the intended example is a full training step, the numerator should include forward+backward work and the step time should be in seconds, not milliseconds.
  - Proposed correction: Use a plausible seconds-scale step time or reframe the example as per-token/per-microbatch throughput. For a simple forward-only MFU illustration, `T_step = 1.2 s` gives about `0.00011` and is not useful; for 11 percent MFU with `42e9` useful FLOPs on a 312 TFLOP/s A100, `T_step` must be about 1.2 ms, so the example needs a much smaller workload or a different label.

- **Lines 4719-4727: the checkpointing optimum mixes a memory model with recomputation claims that do not follow from the table.**  
  The expression `kA + (L/k)A` is minimized at `k = sqrt(L)`, and for `L = 48` the memory estimate `7A + 48/7 A ≈ 14A` is arithmetically correct. But the recompute-cost row says `k checkpoints` costs `(L-k)` forward ops (line 4727), which for `k = 7` is 41 extra layer forwards, much larger than "approximately 33 percent additional forward time" unless a different scheduling/checkpointing algorithm is assumed. The table and prose conflate a simple segmented checkpoint model with a specific empirical overhead.
  - Proposed correction: Either keep the `sqrt(L)` memory derivation and state the recompute cost separately as implementation-dependent, or revise the table to a consistent checkpointing algorithm whose recompute overhead supports the 33 percent claim.

- **Lines 4108-4112: BF16 minimum value is stated as `10^-45`, but BF16's normal range tracks FP32's normal exponent range, about `10^-38`.**  
  BF16 uses FP32's 8-bit exponent, so its minimum positive normal value is about `1.18 x 10^-38`, matching the table on line 3786. The `10^-45` figure is the FP32 subnormal floor, not BF16's practical normal threshold. The prose later contrasts FP16 subnormal flushing with BF16 and should avoid implying BF16 exactly reaches FP32 subnormal precision.
  - Proposed correction: Replace "matches FP32 at approximately `10^-45`" with "has the same normal exponent range as FP32, down to about `1.18 x 10^-38`." If subnormals are discussed, distinguish normal and subnormal behavior explicitly.

- **Lines 4381-4387: the FlashAttention IO reduction factor should be `n/d`, not `n`, under the formulas shown.**  
  The text states standard IO is `O(n d + n^2)` and FlashAttention IO is `O(n d)`. For `n >> d`, the ratio is approximately `n^2 / (n d) = n/d`, not `n`. With `n = 4096` and `d = 64`, the ratio is `4096/64 = 64x`, so the numeric result is correct, but the prose reason is wrong.
  - Proposed correction: Change "reduces memory traffic by a factor of `n`" to "by a factor of approximately `n/d` under this simplified accounting; for `4096/64`, that is `64x`."

### Low Severity

- **Lines 2665 and 420-481: operation/FLOP terminology is inconsistent for multiply-accumulate counts.**  
  Line 2665 says a `512 x 1024` layer with batch 64 executes 33 million floating-point operations. The calculation `64 x 512 x 1024 = 33.6M` counts MACs; if one multiply-add is counted as two FLOPs, it is about 67M FLOPs. The GPT-2 FLOP code correctly multiplies MACs by 2 (lines 439-448), so this local example is inconsistent with the chapter's later FLOP convention.
  - Proposed correction: Change "floating-point operations" to "MACs" for the 33M example, or double the number to about 67M FLOPs.

- **Lines 4096-4101: theoretical Tensor Core speedup is presented as directly translating to wall-clock speedup.**  
  The A100 FP16/FP32 peak ratio can be 16x for dense Tensor Core GEMMs, but line 4101 says this "directly" translates to wall-clock speedups. The chapter's own mixed-precision benchmark gives about 2.4x (lines 3980-4028), because memory, non-GEMM operations, kernel overhead, and utilization limit end-to-end gains.
  - Proposed correction: Replace "directly translating to wall-clock speedups" with "setting an upper bound; end-to-end speedups are typically much lower, often 2-3x for full training workloads."

- **Lines 4925 and 4999-5010: the walkthrough switches hardware labels from V100 to A100.**  
  The section opens with "single 32 GB V100 GPU" (line 4925), but the callout title and prose use a 40 GB A100 (lines 4999-5010). The memory result fitting at about 32 GB is used against the A100's 40 GB capacity, while the opening implies the stricter V100 target.
  - Proposed correction: Use one hardware target throughout. If the intended target is A100, change line 4925 to "single 40 GB A100." If it is V100, change the callout title/prose and compare against 32 GB.

- **Line 6132: the wall-clock example says nearly 3,000 years, but `1e24 / 1e15 = 1e9` days is about 2.7 million years.**  
  At `1e15 FLOPs/day`, a `1e24` FLOP run takes `1e9` days. Dividing by 365 gives roughly 2.74 million years, not 3,000 years. A 3,000-year result would require about `1e18 FLOPs/day`.
  - Proposed correction: Either change the duration to "nearly 3 million years" or change the sustained rate to `1e18 FLOPs/day`.

## Verified Correct

- Lines 138-140: the 7B FP16 training-state calculation is internally correct: 14 GB weights + 14 GB gradients + 56 GB Adam moments = 84 GB, or 6x the FP16 inference weight memory.
- Lines 439-458 and 523-525: the GPT-2 FLOP code consistently counts multiply-adds as two FLOPs and uses a forward+backward multiplier of 3; the per-step and 50K-step totals follow from those assumptions.
- Lines 2686-2699: the wave-quantization table is arithmetically consistent: batch 33 uses 2 warps with `33/64 ≈ 52%` utilization and about `2x` relative time; batch 65 uses 3 warps with `65/96 ≈ 68%` utilization and about `1.5x` relative time relative to batch 64.
- Lines 4773-4794 and 4809-4848: the gradient accumulation cost example is internally consistent: `16 x 4 x 8 = 512` global effective batch, 32 naive GPUs vs. 8 accumulated GPUs, and 75 percent hourly cost reduction.
- Lines 5057-5147: the GPT-2 optimization summary table totals are internally consistent: baseline memory `6 + 6 + 0 + 12 + 65 = 89 GB`, optimized memory `3 + 3 + 6 + 12 + 8 = 32 GB`, and electricity costs at USD 0.10/kWh match the listed energy values.

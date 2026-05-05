# Math Audit Report: `book/quarto/contents/vol2/distributed_training/distributed_training.qmd`

## Checked scope

Audited the distributed training chapter for equations, parallelism/scaling formulas, communication and compute overlap calculations, numeric examples, unit conversions, complexity claims, and prose-equation consistency using direct reasoning only. No Gemini assistance was used. No source `.qmd` files were modified.

## Findings

### High Severity

- **Lines 745-979: the GPT-2 scaling example computes throughput speedup but reports fixed-step training time, making the main conclusion numerically wrong.**  
  The code defines throughput speedup as `N * T_1 / T_N` (lines 789-790 and 854), but then computes training hours as `25 / (speedup/N)` (lines 793 and 856). That divides by parallel efficiency, not speedup. For the 8-GPU case, `T_N = 1806.7 ms`, so throughput speedup is about `7.97x`; fixed-sample training time should be `25 / 7.97 ≈ 3.1 h`, not `25.1 h`. For the 32-GPU case, speedup is about `8.72x`; fixed-sample training time should be `25 / 8.72 ≈ 2.9 h`, not `91.8 h`. If the intent is fixed 50K optimizer steps at larger global batch, then the "speedup (throughput)" label is wrong and 32 GPUs are actually slower per step, not faster.
  - Proposed correction: Decide whether the comparison is fixed samples or fixed optimizer steps. For fixed samples, set `training_hours_* = training_hours_1gpu / speedup_*`. For fixed steps, report step-time slowdown separately and do not call it throughput speedup or compare as a fixed-sample run.

- **Lines 963-979: the gradient-accumulation cost comparison inherits the bad 32-GPU time and also uses an inconsistent denominator in the displayed overhead formula.**  
  The code computes overhead as `comm_8gpu_ms / (4 * 1800 ms) ≈ 0.093%` (lines 884-895), but the prose displays `5 ms / (4 x 180 ms)` (line 969), which would be about `0.69%` if using 5 ms, or about `0.93%` if using the actual 6.7 ms communication. The cost comparison also says the 32-GPU run costs USD 3,021 (line 963), while the code's 32-GPU hours from the flawed formula times the 8-GPU hourly rate gives about USD 11,745; using fixed-sample time and a 4x larger 32-GPU hourly rate would give about USD 1,470. The reported savings therefore do not correspond to a coherent cost model.
  - Proposed correction: Use the actual computed denominator `4 x 1800 ms`, or change the code to `180 ms` if that is intended. Then recompute both 8-GPU and 32-GPU costs under one explicit pricing model, e.g. USD 128/h for 8 GPUs and USD 512/h for 32 GPUs.

- **Lines 2323, 2331, and 2422-2449: the RLHF memory budget switches value-model assumptions and the KV-cache arithmetic is off by 4x.**  
  Line 2331 computes the KV cache as `2 x 80 x 8192 x 1024 x 256 x 2` bytes. That equals about `687 GB`, not `172 GB`; `172 GB` corresponds to batch 64, not 256. Separately, line 2323's `2,406 GB` total assumes a 70B value model: `1120 + 140 + 26 + 1120`. But the worked PPO/DPO example says the value model is 13B (line 2422), and the code computes value memory as `13B x 16 = 208 GB` (lines 2375-2393), making the four-model total `1120 + 140 + 26 + 208 = 1494 GB`, not the table's `~2,406 GB` (line 2443).
  - Proposed correction: Either keep a 70B value model consistently, or change the PPO total-memory row to about `1,494 GB` for a 13B value model. Recompute the KV cache as about `687 GB` for batch 256, or change the batch to 64 where `172 GB` is correct.

- **Lines 1211, 1215, and 1235: the distributed-SGD convergence theorem and explanation are internally inconsistent.**  
  The displayed bound has an optimization term proportional to `1/M` and a variance term proportional to `eta/(Nb)` (line 1208). From that displayed expression, choosing `eta = O(sqrt(Nb/M))` makes the variance term `O(1/sqrt(NbM))`, but the first term remains `O(1/M)` and the expression is not the usual SGD trade-off form with a `1/(eta M)` term. The prose also says "`N` workers with batch size `B` each behave equivalently to a single worker with batch size `Nb`" (line 1215), but by the chapter's notation each worker has local batch `b`; if each worker had batch `B`, the combined batch would be `NB`.
  - Proposed correction: Use a standard bound with the learning-rate dependence explicit, such as `O(1/(eta M)) + O(eta sigma^2/(Nb))`, then state that optimizing `eta` gives `O(1/sqrt(NbM))`. Change line 1215 to "N workers with local batch size b behave equivalently to a single worker with batch size Nb."

- **Lines 1271-1277 and 2520: the explanation for the linear learning-rate scaling rule is mathematically wrong for averaged gradients.**  
  With the chapter's gradient definition as a mean over the batch (lines 517-533), increasing batch size from `B` to `kB` does not reduce the expected gradient magnitude; it reduces variance. Multiplying the learning rate by `k` therefore does not "maintain the same expected update magnitude" (line 1277); it makes the expected update `k` times larger. The usual linear-scaling intuition is that one large-batch step with learning rate `k eta` approximates `k` consecutive small-batch steps of size `eta` when parameters do not move much during those `k` steps.
  - Proposed correction: Replace the magnitude-preservation explanation with the multi-small-step approximation, and keep warmup as the practical guardrail.

- **Lines 2548-2609: the Young-Daly example reports the optimal overhead as the waste from a 15-minute checkpoint interval, while the code compares 30 minutes to optimal.**  
  `T_opt = sqrt(2 x 5 x 240) ≈ 49 minutes`, so the interval value is correct. But `loss_pct_str` is `total_overhead = C/T_opt + T_opt/(2 MTBF) ≈ 20.4%`, the estimated overhead at the optimal interval, not the waste caused by checkpointing every 15 minutes. The `daily_savings` variable compares a 30-minute interval against optimal (lines 2593-2598), while the prose says "every 15 minutes" (line 2609). For 15 minutes, the overhead is `5/15 + 15/(2*240) ≈ 36.5%`, so the excess over optimal is about `16.1 percentage points`, not 20.4%.
  - Proposed correction: Add separate variables for `overhead_15min`, `excess_15min`, and `savings_15min`, or change the prose to say the comparison is against a 30-minute interval. Label `loss_pct_str` as the optimal expected overhead if it is retained.

### Medium Severity

- **Lines 521-533: local-gradient averaging is equivalent to a combined-batch gradient only when all local batches have equal size.**  
  The chapter defines `g_global = (1/N) sum_k g_k` (line 528) and then equates it to the gradient over the union of all local batches (lines 531-533). That equality requires `|B_k|` to be identical for all workers. If local batches differ, the correct combined-batch gradient is `sum_k (|B_k| / |B_total|) g_k`.
  - Proposed correction: Either state the equal-local-batch assumption explicitly, or replace the global average with the weighted average.

- **Lines 636 and 651: AllReduce communication formulas use ambiguous or inconsistent symbols.**  
  Line 636 cites a bound `2N(D-1)/D` without defining whether `N` is message size or worker count. Elsewhere the chapter uses `N` for devices and `M` for model/message size, and line 651 correctly uses `2(N-1)/N x 3 GB / 25 GB/s`. The formula on line 636 can be read as multiplying by the device count, which would be wrong for per-worker ring AllReduce volume.
  - Proposed correction: Use `2(D-1)/D x M` bytes per worker for ring AllReduce, where `D` is device count and `M` is message size. Reserve `N` for worker count only if used consistently.

- **Lines 695 and 1467: staleness learning-rate scaling uses two incompatible formulas.**  
  Line 695 gives `eta' = eta / sqrt(tau)`, which is undefined at `tau = 0` and differs from the later worked example `eta' = eta_BSP / sqrt(1 + 4)` (line 1467). The latter avoids division by zero and matches the usual heuristic form used in the example.
  - Proposed correction: Change line 695 to `eta' = eta / sqrt(1 + tau)` if this heuristic is intended.

- **Lines 996-1002, 1031-1033, and 1595-1601: model-state accounting alternates between optimizer-state-only and full training-state definitions.**  
  The ZeRO example's 12 bytes of "Optimizer State" includes FP32 master weights plus Adam moments (line 998), not just optimizer moments. Later, the GPT-3 memory-wall calculation includes FP16 parameters plus 12 bytes of Adam/master state but omits FP16 gradients (lines 1595-1598). Including gradients adds `175B x 2 bytes = 350 GB`, raising per-GPU static memory under 64-way sharding by about `5.5 GB` and the stated total from `88 GB` to about `94 GB` before activations. The conclusion still holds, but the memory budget is undercounted.
  - Proposed correction: Use explicit categories: FP16 weights, FP16 gradients, FP32 master weights, Adam first moment, Adam second moment. Then recompute the memory-wall example with or without gradients, labeling the choice.

- **Lines 1037-1046: FSDP communication volume bullets imply 4M, but the summary says 3M.**  
  The bullets say the forward pass performs AllGather for parameters `M bytes x 2` and the backward pass performs ReduceScatter for gradients `M bytes x 2` (lines 1041-1042), which totals `4M` under that wording. Line 1046 then states total FSDP volume is approximately `3M` versus `2M` for DDP. The comparison may be defensible under a particular per-parameter accounting, but the displayed bullets do not produce it.
  - Proposed correction: Define `M` as total parameter bytes and list the components that sum to `3M`, or change the total to match the bullets.

- **Lines 1516, 1595-1601, and 1685: GPT-3 parameter memory is inconsistently doubled.**  
  A 175B FP16 parameter set is `175B x 2 = 350 GB`. Line 1516 says the model "requires 700 GB / 64 = 11 GB of parameters per GPU," which is 4 bytes per parameter, not FP16 parameters. Line 1685 correctly says FP16 parameters are 350 GB and `350/8 ≈ 44 GB`, but then says this leaves room for "optimizer state" on A100s; optimizer state for that shard is far larger than the remaining 36 GB unless it is also sharded/offloaded.
  - Proposed correction: Use `350 GB` for FP16 parameters, `700 GB` only for FP32 parameters or FP16 parameters plus FP16 gradients, and state what happens to optimizer state in the 8-way partition.

- **Lines 1721-1723, 1750-1772, 1780, 1792, 1978, and 2053: pipeline bubble formulas alternate between `(p-1)/m` and `(p-1)/(m+p-1)`.**  
  For `p=8, m=32`, `(p-1)/m = 21.875%`, matching the 22 percent claim on line 1721. But the notebook uses `(p-1)/(m+p-1) = 7/39 ≈ 17.9%` (lines 1750-1770). For `p=16, m=16`, line 1723's 48 percent comes from `15/31`, not `15/16`. The chapter should not present both formulas as the same bubble fraction.
  - Proposed correction: Pick the schedule model and use one formula. If using the common fill/drain fraction, use `(p-1)/(m+p-1)` and update the `p=8,m=32` value to about 18 percent. If using the approximation `(p-1)/m`, label it as an approximation valid for `m >> p`.

- **Lines 1804-1808, 1840, and 1856-1861: tensor-parallel communication frequency and volume are inconsistent.**  
  The definition says tensor parallelism requires one AllReduce per transformer block (line 1804), while the quantitative description says Megatron-style tensor parallelism has two AllReduce operations per transformer block (lines 1806 and 1856). The volume formula on line 1859, `2 x B x S x H x sizeof(dtype)`, matches two activation-sized AllReduces per layer; line 1840's "single operation per transformer block" contradicts that.
  - Proposed correction: Say a column/row pair needs one AllReduce, and a transformer block typically has two such pairs, one after attention output and one after MLP output.

- **Lines 1806 and 1858-1861: the tensor-parallel activation-size formula is missing sequence length in one place and overgeneralized in another.**  
  Line 1806 describes each AllReduce as transferring `hidden_dim x batch x 2 bytes`, missing the sequence dimension. Line 1859 includes `S` and computes `2 x 4 x 2048 x 4096 x 2 ≈ 134 MB`, which is arithmetically correct for two activation-sized FP16 transfers. These two descriptions should agree.
  - Proposed correction: Change line 1806 to `B x S x H x sizeof(dtype)` per AllReduce, or `2 x B x S x H x sizeof(dtype)` for two AllReduces per block.

- **Lines 1863-1867: tensor-parallel communication volume does not grow linearly with tensor-parallel degree under ring AllReduce.**  
  For an activation AllReduce of fixed tensor size, per-worker ring volume is `2(t-1)/t x message_size`, which approaches `2 x message_size`; it does not grow linearly without bound. What grows with `t` is the number of latency steps, while compute per GPU shrinks.
  - Proposed correction: Replace "Communication volume grows linearly with tensor parallel degree" with "Ring latency steps grow with tensor parallel degree, while per-GPU compute shrinks and per-worker bandwidth volume approaches a constant multiple of the activation size."

- **Lines 1955-1957: MoE capacity-factor formula omits batch size and top-k routing.**  
  The example correctly computes one-way dispatch volume as `B x S x H x 2 ≈ 67 MB` for `B=4, S=2048, H=4096` (line 1955). But the capacity cap is stated as `C x S/E` tokens per expert (line 1957), omitting the batch dimension and the number of selected experts. For top-k routing, the expected assignments per expert scale with `B x S x k / E`.
  - Proposed correction: Use `C x B x S x k / E` token slots per expert, or explicitly state that the simplified formula assumes `B=1` and top-1 routing.

- **Lines 2033-2049: hybrid 3D communication prose treats per-replica and per-shard traffic as if each device moved the full 350 GB gradient.**  
  In a TP=8, PP=16, DP=128 setup, each data-parallel group synchronizes corresponding parameter shards, not a full 350 GB model on every GPU. The aggregate model gradient is 350 GB per replica, but per-GPU DP AllReduce payload is closer to the local shard size, roughly `350 GB / (TP x PP)` before optimizer/state details. Saying "all 128 replicas must synchronize their gradients, requiring the summation of 350 GB ... across the entire cluster" (line 2035) blurs aggregate and per-link volume.
  - Proposed correction: Distinguish global gradient size from per-GPU/per-DP-group payload. For TP=8, PP=16, note that each shard participates in a DP AllReduce over 128 replicas for about `350/128 ≈ 2.7 GB` of FP16 gradients per rank, while the aggregate synchronized gradient across all shards is 350 GB.

- **Line 2045: the hybrid memory estimate `~15 GB per GPU` is too low under the chapter's own memory accounting.**  
  The same chapter computes FP16 parameters plus 12 bytes/parameter master/Adam state as `2450 GB` for 175B parameters (lines 1595-1598). With `t=8, p=16`, that static state divided over `128` GPUs is `2450/128 ≈ 19.1 GB` before gradients. Including FP16 gradients gives `2800/128 ≈ 21.9 GB`. The stated `15 GB` does not follow from either accounting.
  - Proposed correction: Change the static footprint to about `19 GB` excluding gradients or about `22 GB` including gradients, and state which model states are sharded by which dimension.

- **Lines 2536-2538: the model-size scaling claim overstates compute growth relative to communication.**  
  The text says forward/backward computation grows "superlinearly due to larger matrix operations" and later says computation scales as `O(n^2)` to `O(n^3)` while communication scales as `O(n)`. For transformer training at fixed tokens, FLOPs are commonly proportional to parameter count to first order; superlinear behavior depends on how hidden size, depth, sequence length, and vocabulary are co-scaled. As written, the claim presents an architecture-dependent scaling path as general.
  - Proposed correction: Qualify the claim: "depending on how width, depth, and sequence length scale, compute per step can grow faster than some communication terms; for fixed architecture family and token count, both parameter-gradient communication and dense-layer compute often scale roughly with parameter count."

### Low Severity

- **Lines 159-160 and 689, 723: "99.9 percent per-node reliability" and "1 percent straggler probability" examples are directionally valid but underspecified.**  
  A 100-node cluster with 99.9 percent reliability has about `1 - 0.999^100 ≈ 9.5%` failure probability over the reliability interval, but the text does not define the interval. Similarly, at 1000 GPUs and independent 1 percent straggler probability, the expected straggler count is 10, but the probability of at least one straggler is essentially 100 percent.
  - Proposed correction: State the time interval for reliability and distinguish expected straggler count from probability of any straggler.

- **Line 2006: FP8 energy-efficiency wording mixes throughput and energy-per-operation.**  
  If FP8 has half the energy per operation, that alone is a 2x operations-per-joule improvement. A 4x improvement requires an additional assumption, such as doubled throughput at the same or lower power, and should be stated explicitly.
  - Proposed correction: Change to "approximately 2x lower energy per operation; when combined with higher throughput under fixed power envelopes, system-level throughput per watt can improve further."

- **Lines 2057-2059: Blackwell tensor-parallel claims are too absolute for a math statement.**  
  Saying `t=16` or `t=32` across multiple nodes has "negligible latency overhead" depends on NVLink Switch topology, message size, collective implementation, and placement. The qualitative direction is plausible, but the term "negligible" is not derived.
  - Proposed correction: Rephrase as "lower overhead than Hopper-era inter-node fabrics for supported NVLink Switch domains" unless a concrete latency/bandwidth calculation is supplied.

## Verified Correct

- **Lines 134-135 and 1165-1177:** The distributed step-time law and scaling-efficiency equation are algebraically consistent for a strong-scaling interpretation where `T_compute` is the single-device compute time for fixed work and overlap is capped by communication/compute availability.
- **Lines 581 and 583:** The dataset split examples are arithmetically correct: `100,000 / 4 = 25,000` and `1.2M / 32 = 37,500`.
- **Lines 650-651:** The 3 GB over 25 GB/s transfer examples are arithmetically consistent: naive transfer is `120 ms`, and ring AllReduce across 128 nodes is about `2 x 127/128 x 120 ms ≈ 238 ms`, reasonably rounded to `240 ms`.
- **Lines 1001-1015:** The 7B ZeRO-3 memory arithmetic is internally consistent under the stated 16 bytes/parameter accounting: `7B x 16 = 112 GB`, and `112/64 = 1.75 GB`.
- **Lines 1321-1327:** The critical batch-size expression is dimensionally coherent: gradient-variance trace divided by squared gradient norm yields a sample-count scale.
- **Lines 1452, 1462, 1471, and 1486-1488:** The 8-vs-64 worker convergence speedups and listed cost calculations are arithmetically consistent with their stated iteration counts, overhead factors, per-iteration times, and USD 3/GPU-hour assumption.
- **Lines 1687 and 1955:** The activation-transfer examples are arithmetically correct: `4 x 2048 x 12288 x 2 ≈ 201 MB`, and `4 x 2048 x 4096 x 2 ≈ 67 MB` one-way MoE dispatch.
- **Lines 2031 and 2083-2096:** The 3D parallelism GPU-count product is correct: `TP 8 x PP 16 x DP 128 = 16,384`.

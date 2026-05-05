# Math Audit Report: `book/quarto/contents/vol2/network_fabrics/network_fabrics.qmd`

## Checked scope

Audited networking bandwidth, latency, bisection bandwidth, topology scaling, collective communication cost models, unit conversions, power/cost examples, and prose-equation consistency using direct reasoning only. No Gemini or external model verification was used. Source `.qmd` files were not modified.

## Findings

### High Severity

- **Lines 737, 747, and 782-801: the AllReduce bottleneck notebook uses a different compute workload than the prose, which invalidates the displayed comparison.**  
  The code computes `T_compute` from `flops_per_sample_base = 5e13`, while the prose says each GPU processes `2 x 10^18` FLOPs. These differ by a factor of 40,000. With H100 throughput around 1 PFLOP/s and 50 percent utilization, `5e13` FLOPs gives about 0.1 s, but `2e18` FLOPs gives about 4,000 s if interpreted per GPU as written. If the intended prose value was `2 x 10^14` FLOPs, the time would be about 400 ms. The later claim that 70B communication "dominates computation by nearly 3x" also does not follow from the code: the ring term for 70B gradients is about `2 * 280 GB / 50 GB/s = 11.2 s` if using decimal GB per GPU, or about 5.6 s under the code's `2(p-1)/p * m/beta` calculation with `m = 280 GB`; compared with the code's ~0.1 s compute time, that is tens of times larger, not nearly 3x.
  - Proposed correction: Make the prose and code use the same FLOP count. If the goal is a ~100 ms compute iteration, change the prose to `5 x 10^13 FLOPs`. If the goal is a "nearly 3x" comparison for the 70B case, choose and state a compute time around 1.9 s for the code's ~5.6 s communication time, or revise the conclusion to match the rendered values.

- **Lines 813-815, 927-961, and 1124-1162: bisection bandwidth is alternately computed as half-cluster cut bandwidth and full aggregate injection bandwidth.**  
  Line 815 correctly defines a 1,024-GPU bisection as `512 x 50 GB/s = 25.6 TB/s`. The later notebook computes `bw_1_1 = n_gpus * beta_gbs = 1024 x 50 = 51.2 TB/s` and labels it bisection bandwidth. That is aggregate injection bandwidth, not the capacity crossing an equal bisection under the chapter's own definition. The later 128-node example repeats the same pattern with `128 x 50 = 6.4 TB/s`; if cutting 128 equal-bandwidth nodes into halves, the one-direction bisection is `64 x 50 = 3.2 TB/s`.
  - Proposed correction: Use `N/2 x beta` for one-direction bisection bandwidth, or explicitly say the notebook uses full aggregate injection bandwidth rather than bisection bandwidth. If time is computed from bisection, use data crossing the cut consistently, e.g. `512 x 100 GB / 25.6 TB/s = 2 s` for the 1,024-GPU example, preserving the same 4x oversubscription slowdown without mislabeling the bandwidth.

### Medium Severity

- **Lines 859-861, 907, 978, and 980: radix-64 fat-tree host and switch-count claims use inconsistent topology formulas.**  
  Lines 859 and 907 say a two-tier radix-64 pod supports 1,024 GPUs, while footnote line 978 says radix-64 allows a two-tier fat-tree to support up to 2,048 GPUs. Under the usual two-tier leaf-spine construction with 64-port switches split 32 down / 32 up, 32 leaf switches and 32 spine switches support `32 leaves x 32 hosts/leaf = 1,024` hosts. Line 980 says a non-blocking `k=64` tree reaching `k^3/4 = 65,536` hosts requires 4,096 switches. In the standard k-ary fat-tree count, there are `k` pods, each with `k/2` edge and `k/2` aggregation switches, plus `(k/2)^2` core switches: `64 * 64 + 32^2 = 4,096 + 1,024 = 5,120` switches. The 4,096 count omits core switches.
  - Proposed correction: Use 1,024 hosts for a two-tier radix-64 non-blocking pod unless a different oversubscribed or port-allocation convention is intended. For a standard three-tier `k=64` fat-tree, change the switch count to 5,120, or specify that 4,096 excludes the core tier.

- **Lines 927-963 and 1124-1169: the economic waste examples disagree.**  
  The first oversubscription notebook states that a 4:1 network wastes approximately `$75 Million` for a `$300M` cluster with 30 percent communication. The later notebook computes relative throughput as `1 / (0.70 + 0.30 * 4) = 1/1.9 = 0.526`, so waste is `(1 - 0.526) * 300M = 142M`. Both use the same 4:1 slowdown and 30 percent communication premise, so `$75M` and `$142M` cannot both represent the same model.
  - Proposed correction: If the cost model is throughput loss from Amdahl-style slowdown, use about `$142M`. If `$75M` is intended, state a simpler model such as `25 percent of cluster value` and explain why it differs from the Amdahl calculation.

- **Lines 988, 996-998, 1039-1049, and 1179: the rail-optimized prose swaps tensor-parallel and data-parallel beneficiaries.**  
  The figure caption says rail wiring ensures **data-parallel** AllReduce traffic between corresponding GPUs avoids bisection competition. Line 996 says the same wiring benefits tensor-parallel communication. Line 998 then says data parallelism creates the stratified GPU0-to-GPU0 pattern because large models partition matrix multiplications across GPUs, but matrix partitioning is tensor parallelism, while data-parallel gradient synchronization is the replica-wise AllReduce. The checkpoint at line 1179 correctly separates tensor parallelism within groups of 8 GPUs from data parallelism across all nodes.
  - Proposed correction: Split the claims by communication pattern. Say rail optimization primarily aligns corresponding-GPU data-parallel collectives across nodes, while tensor-parallel activation exchange is usually most bandwidth-sensitive inside the NVLink/NVSwitch domain or within explicitly placed TP groups. If a rail fabric is intended for inter-node tensor parallelism, state the TP group layout and avoid calling it data parallelism.

- **Lines 1389-1425: the optical power summary confuses pluggable consumption with CPO savings.**  
  The code asserts `savings_kw == 1.28`. With 128 ports, that implies a per-switch saving of 1.28 kW. The notebook also renders CPO power as 1.28 kW, so pluggable power must be 2.56 kW if CPO halves it. Line 1425 says "pluggable optics consume 1.28 Megawatts" for 1,000 switches, but that is the CPO total or the savings total, not the pluggable total.
  - Proposed correction: Change line 1425 to "pluggable optics consume 2.56 MW; CPO consumes 1.28 MW, saving 1.28 MW, enough to power about 1,800 700-W GPUs." Keep the per-switch savings line unchanged if the constants are intended.

- **Line 1465: wasted GPU-hours use a 10 percent slowdown, not the stated 2-3 percent iteration slowdown.**  
  A 30-day run on 1,000 GPUs has `30 x 24 x 1,000 = 720,000` GPU-hours. A 2-3 percent slowdown wastes about `14,400-21,600` GPU-hours. The stated `over 72,000` GPU-hours corresponds to a 10 percent slowdown. The `$200,000` cost also matches roughly `72,000 x $2.78/GPU-hour`, not a 2-3 percent drag.
  - Proposed correction: Either change the iteration slowdown to 10 percent, or change the waste to about `14,000-22,000 GPU-hours` and the cost to roughly `$40,000-$60,000` at the implied cloud price.

### Low Severity

- **Lines 273, 299-314, and 339: the Gantt utilization values are slightly inconsistent.**  
  The TikZ figure uses 100 ms forward + 100 ms backward + 1 ms AllReduce for intra-node, which gives `200/201 = 99.5%`, matching line 303. It uses 100 ms + 100 ms + 30 ms for inter-node, which gives `200/230 = 87.0%`, matching line 314. The figure alt text says 87.8 percent, and line 339 says the utilization gap is 12.5 percentage points. The gap from 99.5 to 87.0 is 12.5 points; the gap from 99.5 to 87.8 is 11.7 points.
  - Proposed correction: Change the alt text to 87.0 percent, or adjust the inter-node AllReduce duration so the displayed utilization is 87.8 percent.

- **Line 827: a 2:1 oversubscribed spine with 30 percent communication costs about 23 percent throughput, not 15 percent.**  
  If communication time doubles and the original step is 30 percent communication, the new normalized step time is `0.70 + 2 x 0.30 = 1.30`. Throughput becomes `1/1.30 = 76.9%`, a 23.1 percent throughput loss. A 15 percent throughput loss would correspond to a smaller communication fraction or partial overlap.
  - Proposed correction: Change "roughly 15 percent" to "roughly 23 percent" under the simple exposed-communication model, or state the overlap/hidden-communication assumption that reduces the loss to 15 percent.

- **Line 1211: PFC pause propagation times mix microsecond link response with millisecond cascade timing.**  
  The sentence says PAUSE must arrive within `1-5 us`, then says propagation across `3-5` hops occurs within `10-50 ms`. Pure hop-by-hop signaling over 3-5 switch-to-switch RTTs would be on the order of microseconds, not milliseconds. A millisecond-scale fabric freeze can still occur because buffers fill and queues drain over time, but that is a congestion-dynamics statement, not direct PAUSE-frame propagation.
  - Proposed correction: Write "PAUSE signaling propagates hop-by-hop on microsecond RTTs; the resulting congestion cascade can freeze large portions of the fabric within 10-50 ms."

- **Line 1506: the degraded-link count should be described as an expectation, not a guarantee.**  
  `0.1 percent x 3,000 links = 3` degraded links is the expected value. It does not guarantee exactly or at least 3 degraded links at any moment without a probability model.
  - Proposed correction: Change "guarantees that approximately 3 links are degraded" to "implies an expected roughly 3 degraded links."

## Verified Correct

- Lines 171-189: `400 Gbps = 50 GB/s`, and the H100 NVLink-to-NDR ratio of about `900/50 = 18x` is internally consistent.
- Lines 221-246 and 269: the bandwidth hierarchy ratios are consistent with the data arrays: HBM grows `8000/900 = 8.9x`, NVLink grows `1800/300 = 6x`, InfiniBand grows `100/24 = 4.2x`, and NVLink/IB ratios span roughly 12x to 25x.
- Lines 393, 401, and 412: FEC latency arithmetic is broadly consistent when round trip or encode/decode-at-each-port assumptions are stated. Six one-way hop events at 100-200 ns gives 600-1,200 ns.
- Lines 417 and 433-435: `20,000 x $500 = $10M` and `20,000 x 10 W = 200 kW`; `3,000 x 25 W = 75 kW`.
- Lines 576-603: the NDR alpha-beta crossover is correct: `1.5 us x 50 GB/s = 75 KB`; `4 KB / 50 GB/s = 0.08 us`; `100 MB / 50 GB/s = 2 ms`.
- Lines 1449 and 1455: `400 Gbps / 8 VFs = 50 Gbps` per VF, and `100 GB / 50 GB/s = 2 s`.
- Lines 1575-1604: the small-message HDR/NDR fallacy arithmetic is correct under decimal GB/s: a 10 KB message takes about 1.91 us on 200 Gbps and 1.70 us on 400 Gbps, an improvement of about 10.7 percent.

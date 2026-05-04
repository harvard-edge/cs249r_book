# Math Audit Report: `book/quarto/contents/vol2/compute_infrastructure/compute_infrastructure.qmd`

## Checked scope

Audited the assigned chapter for hardware throughput, latency, bandwidth, memory capacity, power, cooling, TCO, scaling, reliability, and prose-equation consistency. I used direct arithmetic and dimensional checks only. I did not edit the source `.qmd`.

## Findings

### High severity

1. **Line 621: HBM cost for 1,000 accelerators is overstated by roughly 20--60x.**  
   The text says HBM alone in 1,000 accelerators might represent `$50--80 million`. But the immediately preceding cost model says HBM costs `$10--15/GB` (line 613). At 80 GB per H100, that is `$800--1,200` per accelerator, or `$0.8--1.2 million` for 1,000 accelerators. Even at 192 GB per accelerator, it is `$1.92--2.88 million`.  
   **Proposed correction:** Change `$50--80 million` to a value consistent with the stated `$10--15/GB`, or explicitly state a much higher all-in HBM/package cost model if that is intended.

2. **Line 640: INT4 175B weights do not fit in a single 80 GB H100.**  
   A 175B-parameter model at INT4 requires `175e9 * 0.5 bytes = 87.5 GB`. That exceeds 80 GB before metadata, scales, KV cache, and runtime buffers.  
   **Proposed correction:** Say INT4 fits in two H100s, or in one accelerator with at least about 96 GB usable HBM, such as a 192 GB B200-class device.

3. **Lines 713--765 and 1146--1419: FP16/BF16/FP8 throughput labels are inconsistent.**  
   Several plots and tables label H100 `1,979 TFLOPS` as FP16/BF16, while the code uses `H100_FLOPS_FP8_TENSOR`. Elsewhere the chapter correctly distinguishes lower BF16/FP16 ceilings from higher FP8 ceilings. This corrupts ridge-point, efficiency, and "equivalent GPUs saved" claims.  
   **Proposed correction:** Pick one precision per calculation. If using H100 FP8 peak, label it FP8 and use a ridge point of about `1979 / 3.35 = 591 FLOP/byte`. If discussing BF16/FP16 training, use the appropriate BF16/FP16 peak and recompute all efficiency and roofline values.

4. **Line 1090: Achievable throughput units are off by 10^12.**  
   `RooflineInferenceMath.inf_throughput_math` renders `3.35 x 10^12`, and the prose appends `TFLOPS`. `3.35 x 10^12 FLOP/s` is `3.35 TFLOPS`, not `3.35 x 10^12 TFLOPS`.  
   **Proposed correction:** Render this as `3.35 TFLOPS` or `3.35 x 10^12 FLOP/s`.

5. **Lines 1529, 1541, 1559, 1870: Adam training-state memory is undercounted.**  
   For 175B parameters, FP16 weights are 350 GB and FP16 gradients are another 350 GB. Adam has two FP32 tensors, so optimizer state is `175e9 * 2 * 4 bytes = 1,400 GB`, not 700 GB. The full weights + gradients + Adam state total is about 2.1 TB before activations, not 1.4 TB.  
   **Proposed correction:** Replace `350 + 350 + 700 = 1,400 GB` with `350 + 350 + 1,400 = 2,100 GB`, and update derived ZeRO Stage 3 and TP/DP memory numbers.

6. **Line 1866: Table caption says `1,760B-Parameter` but the table uses the 175B running example.**  
   The component totals in the table are for 175B parameters: `175B * 2 bytes = 350 GB`. A 1,760B-parameter model would have 3.52 TB of FP16 weights.  
   **Proposed correction:** Change the caption to `175B-Parameter Frontier Model`.

7. **Lines 1795 and 1805: Tensor-parallel AllReduce sizes and times are internally inconsistent.**  
   Line 1795 computes approximately `2 * batch_size * hidden * 2 bytes`; with batch size 4 and hidden 12,288 this is about 197 KB, not 200 MB. But if "4 sequences at 2,048 tokens" is intended, the tensor is `4 * 2048 * 12288 * 2 = 201 MB`, matching line 1711. The war story then mixes 4 ms PCIe, 0.2 ms NVLink, and 200 KB/200 MB-sized transfers.  
   **Proposed correction:** Include sequence length in the formula if the intended tensor is the full activation tensor, and recompute transfer times. `201 MB / 900 GB/s` is about 0.22 ms; `201 MB / 64 GB/s` is about 3.1 ms before collective factors.

8. **Line 2023: 1,024 GPUs cannot require 128 racks under the chapter's own rack model.**  
   The chapter defines a rack as four 8-GPU nodes, i.e. 32 GPUs/rack. Therefore 1,024 GPUs require `1024 / 32 = 32 racks`, not 128. At 27 kW/rack, IT power is about `32 * 27 kW = 0.86 MW`, not approaching 5 MW.  
   **Proposed correction:** Change to 32 racks and about 0.9--1.1 MW depending on whether PUE and additional overhead are included.

9. **Lines 2364--2453: The training-time callout computes 0.1 weeks, not 2--4 weeks or a 15--25x gap.**  
   The code uses 8,192 H100s, 300B tokens, 45% MFU, and sequential overhead multipliers. Direct arithmetic gives about 12.0 hours physics limit and about 17.6 hours after the listed multipliers, which is `0.105 weeks`, not 2--4 weeks. The listed multipliers produce only a 1.47x increase, not 15--25x.  
   **Proposed correction:** Either change the prose to "~18 hours under these assumptions" or add explicit operational margin multipliers large enough to justify 2--4 weeks.

10. **Lines 2461 and 2711: Pod aggregate throughput is off by orders of magnitude.**  
    Line 2461 says 10,000 H100s deliver `3,120 ExaFLOP/s BF16`; if using 312 TFLOPS BF16 per H100, the aggregate is `10,000 * 312 TFLOPS = 3.12 EFLOP/s`, not 3,120 EFLOP/s. Line 2711 says 8,960 TPU v5p chips at 459 TFLOPS provide 459 PFLOPS, but the product is about `4,113 PFLOPS = 4.1 EFLOP/s`.  
    **Proposed correction:** Use `3.12 EFLOP/s` for the H100 example and `~4.1 EFLOP/s` for the TPU v5p pod.

11. **Lines 2789, 2807, 2811: TCO scale examples conflict by factors of 6--8.**  
    Line 2789 says a 10,000-GPU H100 cluster costs `$3.5B` in hardware, but the later 10,000-GPU callout uses 1,250 nodes at `$350,000` each, or `$437.5M`. Line 2811 says an 8,000-GPU cloud line grows at about `$3M/month`, but `8000 * $4/hr * 24 * 30 * 0.8 = $18.4M/month`. The same figure caption says on-prem CapEx starts near `$58M`, but 1,000 nodes at `$350,000` is `$350M` before network and facility.  
    **Proposed correction:** Decide whether `$350,000` is per node or per GPU. If per node, revise the `$3.5B`, `$3M/month`, and `$58M` statements. If per GPU, revise all node-based TCO examples.

12. **Line 2811: Build-vs-buy break-even conflicts with the executable TCO cell.**  
    The figure caption says break-even requires 55--65% utilization. The code below it computes roughly 42% using its own assumptions: `$350k / 3 / (8 * $4 * 8760 - electricity)` equals about 42%.  
    **Proposed correction:** Either update the caption to match the code, or include omitted facility/network/staffing costs in the formula and recompute the displayed `TcoScenario.breakeven_util`.

### Medium severity

13. **Line 648: B200 HBM bandwidth is inconsistent within the same section.**  
    Line 642 uses `4.8 TB/s` for B200 token latency, while line 648 claims 16 stacks at 600 GB/s each and "exceeding 8 TB/s." Those cannot both be the same B200 configuration.  
    **Proposed correction:** Use one B200 memory-bandwidth value consistently, or distinguish B200 from GB200/B200 variants.

14. **Line 701: "500x faster" is precision-dependent but the precision is not stated.**  
    For the 70B H100 example, compute time at 1,979 TFLOPS is about `0.071 ms`, and memory time is about `41.8 ms`; the ratio is about 590x. If using 989 TFLOPS, it is about 295x. The prose is correct only if it uses the same FP8 peak assumed in nearby code, but that precision is not stated.  
    **Proposed correction:** Tie the ratio to the same peak precision used in the calculation, or state "hundreds of times faster."

15. **Line 1213: TDP throttling example mixes precision baselines.**  
    The example says cooling from 700 W to 500 W drops `R_peak` from 312 to 220 TFLOPS. That is a linear 500/700 scaling of a 312 TFLOPS baseline, but much of the chapter uses 1,979 TFLOPS for H100 peak.  
    **Proposed correction:** State that this example uses a BF16/FP16 baseline, or recompute using the same precision used in nearby roofline examples.

16. **Line 1285: Space-heater comparison does not match the rack-power calculation.**  
    `RackPowerScenario` gives about 27 kW for four H100 nodes. That is about 18 standard 1.5 kW space heaters, not 33. It is 33 only if assuming about 800 W per heater.  
    **Proposed correction:** Change to "about 18 household space heaters" or specify the wattage used.

17. **Line 1435: Per-GPU memory-subsystem power allocation is not reconciled with 700 W TDP.**  
    The text assigns 400--500 W to memory and 150--250 W to Tensor Cores, leaving only 0--150 W for clocking, I/O, leakage, control, and cooling-adjacent components. That can exceed the 700 W envelope at the high end before "the rest" is counted.  
    **Proposed correction:** Present these as illustrative ranges that sum within 700 W, or avoid attributing mutually additive component powers.

18. **Line 1451: Computation-energy formula treats FLOPs and MACs inconsistently.**  
    The text says a token requires about 350 billion MACs for the 175B model, but standard dense inference is about 2 FLOPs per parameter, i.e. 350 billion FLOPs or 175 billion MACs if one MAC is counted as two FLOPs.  
    **Proposed correction:** Use either `175e9 MACs` or `350e9 FLOPs` consistently, and update the compute-energy ratio accordingly.

19. **Line 2094: "300 kW saves enough to power 10 additional GPU nodes" is low under the chapter's node model.**  
    A DGX H100 node is about `8 * 700 W * 1.2 = 6.72 kW` before PUE. Saving 300 kW powers about 44 such nodes before PUE, or about 40 at PUE 1.1.  
    **Proposed correction:** Change "10 additional GPU nodes" to "about 40 additional 8-GPU nodes" under the stated assumptions, or clarify a much higher per-node power basis.

20. **Lines 2123--2124: Rack power table mixes IT load and PUE overhead without clear denominator.**  
    The table includes cooling overhead as part of "rack total"; PUE is normally facility power divided by IT power, not a rack component. The 2.7 kW cooling value is approximately 10% of a 26.8 kW IT subtotal, but including it in the denominator makes the percentages ambiguous.  
    **Proposed correction:** Split IT rack power from facility overhead, or label the table as facility-equivalent power for one rack.

21. **Line 2185: PUE savings calculation is directionally right but should name the baseline.**  
    For 7 MW IT load, PUE 1.5 wastes 3.5 MW and PUE 1.1 wastes 0.7 MW, so the difference is 2.8 MW. The text is correct if "wasted power" means non-IT overhead, but readers may interpret total facility draw.  
    **Proposed correction:** Say "reduces non-IT overhead by 2.8 MW."

22. **Line 2807: Cloud per-run cost applies utilization in a confusing way.**  
    For 1,000 GPUs at `$4/hr`, a 2-week run costs `1000 * 4 * 24 * 14 = $1.344M`; a 4-week run costs `$2.688M`. The text's `$1.075M` equals the 2-week cost multiplied by 0.8, but cloud billing usually charges allocated GPU-hours, not "useful" GPU-hours.  
    **Proposed correction:** Either remove the 0.8 multiplier for per-run cloud cost, or explain that the run consumes only 80% of the nominal two-week allocation.

23. **Lines 2979--3000: Power-efficiency table and refresh math inherit the FP8/FP16 label error.**  
    The table says H100 FP16 is 1,979 TFLOPS and computes 2.83 TFLOPS/W from `H100_FLOPS_FP8_TENSOR`. If comparing FP16 across generations, H100 should use the H100 FP16/BF16 value, not FP8. If comparing peak tensor at each generation, the column label should not say FP16.  
    **Proposed correction:** Rename the column to the precision actually used, or recompute with consistent FP16/BF16 peaks.

24. **Line 2994: Fixed-power refresh speedup should be efficiency ratio, not same-GPU-count throughput ratio.**  
    Replacing 1,000 V100s with 1,000 H100s gives 15.8x throughput for 2.3x power. Under a fixed 300 kW power budget, only about 428 H100s fit, and the throughput gain is about 6.8x, matching the TFLOPS/W ratio.  
    **Proposed correction:** Change `15.8x` to about `6.8x` for the fixed-power scenario, or state that 15.8x assumes increasing power from 300 kW to 700 kW.

25. **Line 3073: The 1,024-GPU planning example is internally consistent for 4 ideal compute days, but conflicts with earlier 8,192-GPU/12-hour and 2--4-week statements.**  
    The calculation `3.15e23 / 9.12e17 = 345,000 s` is about 4 days and is correct. But it should not be used interchangeably with the earlier 8,192-GPU example unless the change in cluster size is made explicit.  
    **Proposed correction:** Explicitly distinguish the 1,024-GPU example from the 8,192-GPU pod example and align the operational overhead factors.

## Notes

Many broad qualitative claims are directionally sound: HBM bandwidth dominates batch-1 decode, PUE materially affects facility power, and interconnect hierarchy constrains parallelism. The main audit risk is not the qualitative systems argument but precision mixing and inconsistent scenario sizing. A small set of shared scenario constants for "175B on 1,024 GPUs" vs. "175B on 8,192 GPUs" would remove many downstream contradictions.

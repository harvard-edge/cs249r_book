# Math Audit: `book/quarto/contents/vol2/data_storage/data_storage.qmd`

Scope: storage throughput, latency, capacity, reliability, data-rate calculations, equations, unit conversions, scaling claims, and prose-equation consistency. Direct reasoning only; no Gemini.

## Findings

### 1. Checkpoint-size baseline is internally inconsistent

- **Lines:** 89-90, 114-116, 146, 1570, 1600, 1692-1694
- **Severity:** High
- **Issue:** The calculation cell computes a full checkpoint as FP16 weights plus two FP32 Adam states: `2 + 2*4 = 10` bytes/parameter. For 175B parameters, that is `175e9 * 10 / 1e9 = 1,750 GB`, not `1,050 GB`. Several prose/table entries still describe a `1,050 GB` checkpoint and `700 GB` optimizer state.
- **Explanation:** `1,050 GB` corresponds to 6 bytes/parameter, e.g. FP16 weights plus two FP16 states, not two FP32 Adam states. The table also lists optimizer state as `1,400 GB`, which conflicts with the `1,050 GB` checkpoint row.
- **Proposed correction:** Choose one checkpoint model and make all references match. If keeping FP16 weights plus two FP32 Adam states, use `1,750 GB` per full checkpoint and `7.6 PB` for 4,320 checkpoints. If the desired checkpoint is `1,050 GB`, change the formula to 6 bytes/parameter and describe the optimizer/checkpoint precision accordingly.

### 2. Introductory checkpoint interval conflicts with the computed total

- **Lines:** 116, 146, 1600, 1694
- **Severity:** High
- **Issue:** Line 146 says checkpoints are generated every 30 minutes, but the checkpoint-total calculation uses `4320` checkpoints, which is a 10-minute interval over 30 days (`30*24*6 = 4320`).
- **Explanation:** A 30-minute interval over 30 days gives `30*24*2 = 1440` checkpoints. With the current formula's 1,750 GB checkpoint, that is `2.52 PB`; with a 1,050 GB checkpoint, it is `1.51 PB`. The later sections use a 10-minute interval.
- **Proposed correction:** Change line 146 to "every 10 minutes" if the 4,320-checkpoint total is intended, or change `ckpt_fleet_total_pb_val` to multiply by `1440` and update the later 10-minute references.

### 3. HBM-to-object-storage bandwidth gap is off by orders of magnitude

- **Lines:** 433-442, 1752, 1760
- **Severity:** High
- **Issue:** The table gives HBM bandwidth as about `3.35 TB/s` and object storage as `100 GB/s aggregate`. That ratio is `3,350 GB/s / 100 GB/s = 33.5x`, not `300,000x`.
- **Explanation:** Even comparing HBM to the archive row (`1 GB/s`) gives about `3,350x`, still far below `300,000x`.
- **Proposed correction:** Replace `300,000x` with the ratio implied by the table, or redefine the comparison to a much slower per-request/per-client archive/object-storage path and state that denominator explicitly.

### 4. HBM-vs-DDR5 speedup contradicts the chapter's own DRAM bandwidth

- **Lines:** 435-436, 464, 496
- **Severity:** High
- **Issue:** Line 496 says HBM is roughly `500x` faster than DDR5 system DRAM, but the table and bandwidth-cliff prose use HBM `3.35 TB/s` and host DRAM `200 GB/s`, a ratio of about `16.75x`.
- **Explanation:** The chapter already states the adjacent HBM-to-host-DRAM cliff correctly as roughly `17x` on line 464.
- **Proposed correction:** Change line 496 to "roughly 17x faster than the host DRAM bandwidth used here" or qualify the `500x` as a different comparison with explicit units.

### 5. NVMe RAID-to-HBM ratio is computed as if it were a single drive

- **Lines:** 142, 659, 1641
- **Severity:** Medium
- **Issue:** A single 7 GB/s NVMe drive is about `3,350/7 = 479x` slower than HBM, but a 4-drive 14 GB/s RAID-0 is about `3,350/14 = 239x` slower, and a 25 GB/s high-end local NVMe configuration is about `134x` slower. Line 659 calls "the fastest NVMe RAID configuration" roughly `500x` slower.
- **Explanation:** The `~500x` number is appropriate for one 7 GB/s drive, not for the RAID examples used nearby.
- **Proposed correction:** Use `~500x` only for a single drive; use `~240x` for a 14 GB/s RAID-0 and `~130x` for 25 GB/s.

### 6. Cost-gradient variable is reused for the wrong tiers

- **Lines:** 1439-1456, 1464-1466, 1767
- **Severity:** High
- **Issue:** `EconRatios.tier_cost_ratio_str` is computed as NVMe/S3 (`0.10/0.02 = 5x`) but the prose says it is the cost difference between HBM and archive storage.
- **Explanation:** Using the table values, HBM/archive is `$15 / $0.004 = 3,750x`, not `5x`. The rendered text would materially understate the hierarchy's cost gradient.
- **Proposed correction:** Either rename the rendered prose to "5x cost difference between local NVMe and S3" or compute a separate `hbm_archive_ratio = 15 / 0.004 = 3750`.

### 7. HBM equivalent cost for 100 TB is off by 10x

- **Lines:** 1459-1465
- **Severity:** Medium
- **Issue:** `100 TB = 100,000 GB` under the chapter's decimal convention. At `$15/GB`, the cost is `$1,500,000`, not `$15,000,000`.
- **Explanation:** `$15,000,000 / $15/GB = 1,000,000 GB = 1,000 TB`, so the prose has a factor-of-10 mismatch.
- **Proposed correction:** Change the HBM equivalent to `$1,500,000` for 100 TB, or change the dataset size to 1 PB if `$15,000,000` is intended.

### 8. Idle accelerator daily loss is inconsistent

- **Lines:** 1470, 1478
- **Severity:** Medium
- **Issue:** Line 1470 correctly computes the utilization drop from 90% to 70% on a 1,000-GPU cluster at `$2/GPU-hour` as `$9,600/day`: `1000*2*24*0.20 = 9600`. Line 1478 says the same scenario costs `$13,000/day`.
- **Explanation:** The two lines use the same inputs but different results.
- **Proposed correction:** Change line 1478 to `$9,600/day`, or state different assumptions if `$13,000/day` is intended.

### 9. Egress-tax break-even point is too low

- **Lines:** 1487-1502
- **Severity:** Medium
- **Issue:** The notebook says local NVMe caching breaks even at 2 epochs. With the displayed annual numbers, streaming costs `13,800 + 18,000*epochs` for 4 runs/year, while staging costs `91,800`. Break-even is `(91,800 - 13,800) / 18,000 = 4.33` epochs per run.
- **Explanation:** At 2 epochs, streaming would cost `$49,800/year`, less than the `$91,800/year` staging option.
- **Proposed correction:** Change the break-even statement to "about 5 epochs per run" under the displayed assumptions, or revise the assumptions so the 2-epoch claim is true.

### 10. Inference loading text labels RAID bandwidth as a single-drive rate

- **Lines:** 1558-1560
- **Severity:** Low
- **Issue:** Line 1558 says loading from a "single high-performance NVMe drive at 14 GB/s" takes 25 seconds. Elsewhere, 14 GB/s is the aggregate of four 3.5 GB/s drives, not a single drive.
- **Explanation:** `350 GB / 14 GB/s = 25 s` is correct arithmetic, but the hardware description is not.
- **Proposed correction:** Say "from a 4-drive local NVMe RAID at 14 GB/s" or use a single-drive 7 GB/s rate and update the time to 50 seconds.

### 11. CXL example overstates what a single node can hold

- **Lines:** 1546
- **Severity:** Low
- **Issue:** The prose says expanding a node to "over 4 TB" of CXL memory is large enough to hold the entire 3 TB dataset "without touching NVMe." That is true for the compressed 3 TB dataset but not for the chapter's decoded/preprocessed `~15 TB` dataset footprint.
- **Explanation:** Earlier lines distinguish the 3 TB compressed dataset from roughly 15 TB including preprocessed variants/decoded data.
- **Proposed correction:** Qualify this as "the 3 TB compressed dataset" or avoid implying it covers all preprocessed/decoded forms.

### 12. Checkpoint overhead conclusion overstates the pause

- **Lines:** 1586-1590, 1629
- **Severity:** Low
- **Issue:** The checkpoint storm notebook computes the critical training pause as `0.29 s` every `600 s`, i.e. `0.048%`. Line 1629 says the pipeline pause is "under two seconds" and "less than 0.3 percent" while also saying the PFS copy happens in the background.
- **Explanation:** If the background PFS copy truly overlaps and does not block training, the exposed pause is the local write time (`~0.29 s`, or `~0.05%`), not the local write plus async copy.
- **Proposed correction:** Keep the durable-copy time as "just over one second in the background," but state the exposed training pause as `~0.3 s` and `~0.05%`.

## Checks That Look Consistent

- Text bandwidth example: `2048 * 4096 tokens/GPU * 4 B/token / 0.2 s = 167.8 MB/s`, reasonably rounded to `160 MB/s` (lines 474-476).
- Image bandwidth example: `2048 * 256 * 150 KB / 0.2 s = 393 GB/s` (lines 480-482).
- Data stall example: without pipelining `250/450 = 55.6%`; with overlap `(250-200)/250 = 20%` (lines 1041-1049).
- CPU bypass example: `64,000 * 120 us = 7.68 CPU-s/s`; `64,000 * 30 us = 1.92 CPU-s/s` (lines 1389-1393).
- Synthetic storage amplification formula is internally consistent: `1 TB * (1 + overhead) * verification_passes`, with the guard expecting `4.2x` (lines 1710-1723).

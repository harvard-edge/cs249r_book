# Corpus audit findings — 2026-05-03

**Source:** `interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json`
**Generated:** 2026-05-03T22:38:47+00:00
**Audit model:** `gemini-3.1-pro-preview`
**Triggered by:** CORPUS_HARDENING_PLAN.md Phase 4 finalization

---

## Executive summary

- **9446** questions audited
- **0** errored (no Gemini response — should be retried)

| gate | pass | fail | other |
|---|---:|---:|---:|
| format_compliance | 8403 | 1043 | 0 |
| level_fit | 7649 | 1795 | 2 |
| coherence | 8865 | 581 | 0 |
| math_correct | 8986 | 444 | 16 |
| title_quality | 9192 good | 216 placeholder + 30 malformed | 0 error |

---

## Per-track failure rates

| track | total | format | level_fit | coherence | math |
|---|---:|---:|---:|---:|---:|
| cloud | 4028 | 421 (10.5%) | 720 (17.9%) | 182 (4.5%) | 160 (4.0%) |
| edge | 2079 | 278 (13.4%) | 401 (19.3%) | 143 (6.9%) | 96 (4.6%) |
| global | 313 | 28 (8.9%) | 35 (11.2%) | 24 (7.7%) | 21 (6.7%) |
| mobile | 1824 | 153 (8.4%) | 382 (20.9%) | 148 (8.1%) | 117 (6.4%) |
| tinyml | 1202 | 163 (13.6%) | 257 (21.4%) | 84 (7.0%) | 50 (4.2%) |

---

## Coherence failure-mode breakdown

When coherence=fail, Gemini classifies the failure into one of four modes. The distribution tells us where to focus targeted fixes.

| failure_mode | count |
|---|---:|
| mismatch | 299 |
| arithmetic | 132 |
| physical_absurdity | 126 |
| vendor_fabrication | 24 |

---

## Priority lists for human review

### Math errors — 444 questions

**Highest priority.** Each requires per-instance human review (per CORPUS_HARDENING_PLAN.md §10 Q2). When fixing, rewrite BOTH napkin_math AND realistic_solution as a unit.

| qid | track | first error |
|---|---|---|
| `cloud-0027` | cloud | Claims the 200ms wait is purely in the queue, but 200ms is the total time in... |
| `cloud-0125` | cloud | Scenario claims P99 TTFT is 2.5 seconds, but napkin_math calculates 2.15 seconds |
| `cloud-0126` | cloud | A service rate of 25 req/s implies a 40ms mean service time, but the... |
| `cloud-0145` | cloud | Calculates prefill compute time by multiplying the number of input tokens by... |
| `cloud-0178` | cloud | Claims prefilling 1000 tokens for a batch is 140 TFLOPs, but for a batch of 32,... |
| `cloud-0230` | cloud | Claims latency causes a 20% throughput drop, but then claims VRAM overhead... |
| `cloud-0454` | cloud | Calculates 11.6s total step time which contradicts the 11.3s stated in the... |
| `cloud-0480` | cloud | Claims 2 * (N-1)/N * 140 GB is 280 GB, which is inaccurate for both N=8 (245... |
| `cloud-0491` | cloud | Calculates Centralized 3-year TCO by adding only one year of compute cost... |
| `cloud-0536` | cloud | Calculates '$764k/year' savings but cost assumptions are entirely missing from... |
| `cloud-0555` | cloud | Calculates FLOPs per block as 32.7 TFLOPs, but 4 * (1M/32)^2 * 64 * 128 is... |
| `cloud-0582` | cloud | claims 3x H100s frees GPUs and doubles fleet capacity but 5/3 = 1.66x capacity... |
| `cloud-0583` | cloud | claims INT8 Quantization size is 105 GB but conclusion relies on... |
| `cloud-0595` | cloud | Total cold start calculated as ~23s does not match the scenario's 45s. |
| `cloud-0614` | cloud | claims 33 concurrent * 640 MiB is 21.1 GiB but 33 * 640 MiB = 21120 MiB =... |
| `cloud-0621` | cloud | Claims 0.96 sec/token is 39x slower, but baseline for 140 GB in HBM is 0.041... |
| `cloud-0625` | cloud | Claims effective capacity is 21.6GB, but the scenario explicitly states it... |
| `cloud-0647` | cloud | Claims wasting 6.6 bits 'Exceeds FP8 E4M3 4-bit exponent range'. A 100x ratio... |
| `cloud-0663` | cloud | Claims pure LLM prefill takes 7.75 seconds in the solution but correctly... |
| `cloud-0671` | cloud | The scenario claims 4% utilization, but the math calculates 0.33% utilization,... |
| `cloud-0709` | cloud | Calculates 3.67x speedup relative to 8 nodes, contradicting the 5.2x speedup... |
| `cloud-0741` | cloud | Calculates for 512 GPUs instead of the 256 GPUs specified in the scenario. |
| `cloud-0745` | cloud | Claims 256 * $3.50 * (5*40/3600) * 24 = $995, but it actually equals $1194.66. |
| `cloud-0813` | cloud | 500 * 292 is 146,000, not 146,200. |
| `cloud-0866` | cloud | Claims inter-node AllGather only transfers 5.38GB total instead of the 344GB... |

_…and 419 more._

### Vendor fabrication — 24 questions

| qid | track | rationale |
|---|---|---|
| `cloud-4266` | cloud | The scenario names 'MI300X APU', but MI300X is a discrete GPU; the APU variant is MI300A. |
| `edge-0099` | edge | The i.MX 8M Plus has a Vivante NPU, not an ARM Ethos-U65 (which is in the i.MX 93). |
| `edge-2391` | edge | Jetson AGX Orin does not support Multi-Instance GPU (MIG); MIG is a data center GPU feature. |
| `mobile-0077` | mobile | The scenario incorrectly attributes an Adreno-class GPU to the Tensor G3 SoC, which actually uses... |
| `mobile-0122` | mobile | The solution suggests bypassing NNAPI with Qualcomm's QNN SDK on a Pixel 8, but the Pixel 8 uses... |
| `mobile-1669` | mobile | nsys is an NVIDIA profiling tool and cannot be used to profile a Samsung Exynos NPU. |
| `mobile-1675` | mobile | The solution misattributes nsys, which is an NVIDIA profiling tool, to the Samsung Exynos NPU. |
| `tinyml-0991` | tinyml | The Ethos-U55 NPU only scales up to a maximum of 256 MACs/cycle; a 512 MACs/cycle variant does not... |
| `tinyml-1096` | tinyml | Corstone-300 is the reference design for Cortex-M55, not Cortex-M7. |
| `tinyml-1097` | tinyml | Corstone-300 is the reference design for Cortex-M55, not Cortex-M7. |
| `tinyml-1098` | tinyml | Corstone-300 is the reference design for Cortex-M55, not Cortex-M7. |
| `tinyml-1099` | tinyml | Corstone-300 is the reference design for Cortex-M55, not Cortex-M7. |
| `tinyml-1102` | tinyml | Corstone-300 is the reference design for Cortex-M55, not Cortex-M7. |
| `tinyml-1107` | tinyml | Corstone-300 is the reference design for Cortex-M55, not Cortex-M7. |
| `tinyml-1156` | tinyml | The ARM Corstone-300 subsystem features a Cortex-M55, not a Cortex-M7. |
| `tinyml-1158` | tinyml | The ARM Corstone-300 platform is built around the Cortex-M55, not the Cortex-M7. |
| `tinyml-1159` | tinyml | The ARM Corstone-300 subsystem features a Cortex-M55, not a Cortex-M7. |
| `tinyml-1171` | tinyml | The ARM Corstone-300 subsystem specifically features a Cortex-M55, not a Cortex-M7. |
| `tinyml-1180` | tinyml | The Corstone-300 reference design is based on the Cortex-M55, not the Cortex-M7. |
| `tinyml-1187` | tinyml | The Corstone-300 reference design is based on the Cortex-M55, not the Cortex-M7. |
| `tinyml-1437` | tinyml | The Ethos-U55 NPU only exists in 32, 64, 128, and 256 MAC/cycle configurations; a 200 MAC/cycle... |
| `tinyml-1438` | tinyml | The STM32F4 series is based on the Cortex-M4F core which inherently includes a hardware FPU;... |
| `tinyml-1442` | tinyml | The STM32F4 series is based on the Cortex-M4F core which inherently includes a hardware FPU;... |
| `tinyml-1444` | tinyml | The STM32F4 series is based on the Cortex-M4F core which inherently includes a hardware FPU;... |

### Physical absurdity — 126 questions

| qid | track | rationale |
|---|---|---|
| `cloud-0176` | cloud | Claiming a single 14-second non-preemptible CUDA kernel blocks execution is a physical absurdity;... |
| `cloud-0180` | cloud | The math divides the heavy prefill compute of a 70B model by a single H100's peak compute,... |
| `cloud-0304` | cloud | Calculating GPU rack capacity without accounting for host server power consumption is physically... |
| `cloud-0375` | cloud | Applying a facility-level PUE overhead directly to reduce the IT power capacity of a standard... |
| `cloud-0851` | cloud | An NVIDIA A100 GPU achieves a ResNet-50 image throughput of >10,000 images/sec, making 120ms for a... |
| `cloud-0925` | cloud | NVIDIA T4 hardware natively supports 1-bit Tensor Core operations (520 TOPS), so claiming it lacks... |
| `cloud-1041` | cloud | Calculating latency by multiplying the total number of cache misses serially by 85ns ignores memory... |
| `cloud-1046` | cloud | Calculating a 500ms stall from 5 million cache misses by multiplying serially by 100ns ignores... |
| `cloud-1201` | cloud | Table-wise sharding places the entire table on one GPU. A 400GB table cannot physically fit on a... |
| `cloud-1206` | cloud | Streaming 250 MB of raw user history per real-time request is physically absurd and would instantly... |
| `cloud-1226` | cloud | Processing 10,000 loan approvals per second implies over 300 billion applications per year, which... |
| `cloud-1260` | cloud | A 1050ms latency increase for a 16MB all_gather and a 68 GFLOP computation on A100 GPUs is... |
| `cloud-1262` | cloud | Claiming a 68.7 GFLOP computation is a severe computational bottleneck on an A100 GPU (which has... |
| `cloud-1313` | cloud | The napkin math calculates 99GB of VRAM per GPU, which physically exceeds the 80GB capacity of the... |
| `cloud-1485` | cloud | A NUMA cross-node memory access penalty is typically on the order of ~150-200 nanoseconds, making... |
| `cloud-1574` | cloud | The hardware ridge point for an A100 is >150 FLOPs/byte, but the napkin math claims it is ~10... |
| `cloud-2174` | cloud | Register bandwidth of ~20 TB/s 'per SM' is physically absurd for an H100 (which has ~2-3 TB/s per... |
| `cloud-2235` | cloud | A buildup of 23.04 GB (or 461 GB) cannot cause a 'no space left' error on a 3.84 TB NVMe drive. |
| `cloud-2448` | cloud | A single H100 GPU has 80GB of HBM, which physically cannot fit a 140GB (70B parameter) model. |
| `cloud-2895` | cloud | MobileNetV3 throughput on an A100 is severely underestimated; it should be well over 30,000 QPS,... |
| `cloud-3079` | cloud | A single H100 takes 12.5s to decode 300 tokens for LLaMA-70B, physically violating the <2s SLO... |
| `cloud-3091` | cloud | The scenario claims 12 TFLOPS throughput, but the memory bandwidth roofline strictly limits this... |
| `cloud-3149` | cloud | Claiming 800 KB accesses per head for a block_size of 1 is physically absurd, as a single token's... |
| `cloud-3171` | cloud | A 405B model (810GB in FP16) cannot fit on 8x 80GB GPUs (640GB capacity). |
| `cloud-3278` | cloud | A 1 Trillion parameter model requires terabytes of memory just for weights and cannot be trained or... |

_…and 101 more._

### Scenario/solution mismatch — 299 questions

| qid | track | rationale |
|---|---|---|
| `cloud-0230` | cloud | The napkin_math claims latency alone causes a 20% throughput drop, yet the scenario states the... |
| `cloud-0423` | cloud | The scenario states GPU utilization fluctuates violently between 0% and 100%, but the napkin math... |
| `cloud-0454` | cloud | Numbers contradict; scenario states 11.3s step time but math derives 11.6s. |
| `cloud-0582` | cloud | Scenario mandates a 2x throughput increase but solution achieves 1.66x while claiming 2x. |
| `cloud-0595` | cloud | Numbers contradict: Scenario claims a 45 second wait, but math calculates ~23 seconds. |
| `cloud-0625` | cloud | The napkin_math claims static allocation leads to ~21.6GB effective capacity, which contradicts the... |
| `cloud-0629` | cloud | The solution blames OOM on dynamic activation/gradient memory for hot keys, but the napkin_math... |
| `cloud-0663` | cloud | Numbers contradict across the realistic_solution (which claims 7.75s) and napkin_math (which... |
| `cloud-0671` | cloud | Numbers contradict: The scenario claims GPU utilization is at 4%, but the napkin math correctly... |
| `cloud-0709` | cloud | Scenario claims 5.2x speedup while napkin_math evaluates to a 3.67x speedup relative to 8 nodes. |
| `cloud-0741` | cloud | The scenario and question specify 256 GPUs, but the napkin_math calculates for 512 GPUs. |
| `cloud-0885` | cloud | The scenario specifies a sequence length of 2048, but the solution and napkin math calculate the KV... |
| `cloud-1088` | cloud | Scenario claims utilization drops to 42%, but napkin math calculates current utilization as 29.4%. |
| `cloud-1090` | cloud | Napkin math assumes 5,000 legitimate refund examples which is never stated in the scenario. |
| `cloud-1111` | cloud | The napkin_math calculates storage strictly in base-10 (43.2M * 2KB = 86.4 GB) while the... |
| `cloud-1241` | cloud | The realistic_solution completely fails to answer the quantitative question regarding the total... |
| `cloud-1255` | cloud | The realistic_solution states the frequency of failures but omits a direct answer to the 'how many'... |
| `cloud-1592` | cloud | The scenario and question state a 4.5-minute delay, but the napkin math calculates 283 seconds,... |
| `cloud-1643` | cloud | The scenario specifies 4,000 images/sec total, but the napkin math treats it as 4,000 per GPU... |
| `cloud-1657` | cloud | The math shows 25.92GB of VRAM growth in 48 hours, which contradicts the scenario's claim that it... |
| `cloud-1661` | cloud | The conclusion incorrectly claims the 1.6ms of calculated PCIe transfer overhead dominates the 45ms... |
| `cloud-1748` | cloud | The napkin_math calculation proves a 1.14x slowdown (219ms vs 192ms) but the conclusion and... |
| `cloud-1844` | cloud | The scenario states the node is pulling 1.5 GB/s and hitting a 5,000 GET/sec limit, but the napkin... |
| `cloud-2010` | cloud | The question asks for an exact memory traffic calculation, but the scenario omits the tensor... |
| `cloud-2037` | cloud | The scenario states a 20-minute outage and a 3-minute spike, but the napkin math improperly uses... |

_…and 274 more._

### Level inflation — 1795 questions

Per CORPUS_HARDENING_PLAN.md §10 Q3, default disposition is **relabel down** to the actual cognitive level. Rewriting the question up is a separate authoring task, not a Phase-5 concern.

| qid | track | claimed level | rationale |
|---|---|---|---|
| `cloud-0002` | cloud | ? | Labeled as L1 recall, but the question asks for a multi-step calculation which... |
| `cloud-0016` | cloud | ? | Labeled as L1 recall, but identifying the primary suspect for a performance... |
| `cloud-0062` | cloud | ? | Verb mismatch: Calculating a math constraint is L3 apply, not L2 understand. |
| `cloud-0075` | cloud | ? | Verb mismatch: Executing a calculation is L3 apply, not L2 understand. |
| `cloud-0093` | cloud | ? | Level inflation: Claimed L3 but the question demands conceptual recall and... |
| `cloud-0099` | cloud | ? | Level inflation: L3+ stamped on a simple multiplication/division problem with... |
| `cloud-0109` | cloud | ? | Level inflation: L3+ stamped on a simple multiplication/division problem with... |
| `cloud-0168` | cloud | ? | Verb mismatch: Root-causing a latency degradation is L4 analyze, but level is... |
| `cloud-0171` | cloud | ? | The question asks for the root cause of an SLA violation (L4), but the level is... |
| `cloud-0177` | cloud | ? | The question asks for root-cause diagnosis of a system failure, which is L4... |
| `cloud-0179` | cloud | ? | Verb mismatch: question's actual verb is 'What scheduling flaw' (L4 analyze)... |
| `cloud-0195` | cloud | ? | Level inflation: the question asks to recall an existing limitation and name a... |
| `cloud-0229` | cloud | ? | Claimed L6+ (create) but the question simply asks to recall a standard industry... |
| `cloud-0230` | cloud | ? | Level inflation: labeled L6+ Mastery/Create, but diagnosing a performance... |
| `cloud-0259` | cloud | ? | Verb mismatch: calculating or estimating the Chinchilla-optimal training... |
| `cloud-0279` | cloud | ? | Verb mismatch: determining bounding by calculating and comparing AI requires... |
| `cloud-0296` | cloud | ? | Verb mismatch: determining if workload is compute or memory bound by... |
| `cloud-0325` | cloud | ? | Verb mismatch: question requires root-cause analysis to determine bottleneck... |
| `cloud-0329` | cloud | ? | Verb mismatch: question asks to compare and choose between architectural... |
| `cloud-0349` | cloud | ? | Verb mismatch: question requires root-cause analysis to determine if memory or... |
| `cloud-0374` | cloud | ? | The question requires analyzing constraints and evaluating trade-offs between... |
| `cloud-0384` | cloud | ? | The question requires decomposition and root-cause reasoning to explain why a... |
| `cloud-0409` | cloud | ? | Verb mismatch: calculating and interpreting arithmetic intensity to determine... |
| `cloud-0421` | cloud | ? | Verb mismatch: L2 stamped but the actual question asks to compute a calculation... |
| `cloud-0441` | cloud | ? | Identifying a root cause (dispatch-bound stall) requires diagnosis/analysis... |

_…and 1770 more._

### Placeholder titles — 216 questions

Phase 7 will batch these for Gemini-proposed replacements (~1 call per 5 placeholders). All require human review of the proposed title.

qids (first 25): `cloud-2510`, `cloud-2512`, `cloud-2513`, `cloud-2514`, `cloud-2515`, `cloud-2516`, `cloud-2518`, `cloud-2519`, `cloud-2520`, `cloud-2522`, `cloud-2523`, `cloud-2524`, `cloud-2525`, `cloud-2526`, `cloud-2528`, `cloud-2529`, `cloud-2530`, `cloud-2531`, `cloud-2532`, `cloud-2533`, `cloud-2534`, `cloud-2535`, `cloud-2536`, `cloud-2537`, `cloud-2538`

_…and 191 more._


---

## Format-compliance: regex vs. Gemini

Of the rows where both verdicts are present, **236** disagree. Regex is the source of truth (it's mechanical); disagreements indicate Gemini missed a marker the regex caught (or vice versa).

| qid | gemini | regex | regex_issues |
|---|---|---|---|
| `cloud-0295` | fail | pass |  |
| `cloud-0494` | fail | pass |  |
| `cloud-0557` | fail | pass |  |
| `cloud-0752` | pass | fail | napkin_math missing ['**Assumptions', '**Calculations:**', '**Conclusion'] |
| `cloud-0799` | pass | fail | napkin_math missing ['**Assumptions', '**Calculations:**', '**Conclusion'] |
| `cloud-0802` | fail | pass |  |
| `cloud-0966` | pass | fail | napkin_math missing ['**Assumptions', '**Calculations:**', '**Conclusion'] |
| `cloud-1017` | fail | pass |  |
| `cloud-1097` | fail | pass |  |
| `cloud-1271` | fail | pass |  |
| `cloud-1383` | fail | pass |  |
| `cloud-1748` | fail | pass |  |
| `cloud-1764` | fail | pass |  |
| `cloud-1767` | fail | pass |  |
| `cloud-1771` | fail | pass |  |
| `cloud-1780` | fail | pass |  |
| `cloud-1781` | fail | pass |  |
| `cloud-1782` | fail | pass |  |
| `cloud-1784` | fail | pass |  |
| `cloud-2235` | fail | pass |  |

_…and 216 more._


---

## Recommendations

1. **Resume to clear errored batches.** 0 rows show `format_compliance: error` — these batches didn't get a Gemini response. Re-running `audit_corpus_batched.py --output <same-dir>` retries them.

2. **Run `--propose-fixes` on the format-fail subset.** 1070 rows fail format. Most are mechanical marker additions — `apply_corrections.py --auto-accept-format` will auto-apply low-risk fixes.

3. **Per-instance review for math errors.** 444 rows have math errors. Each needs a human to verify the proposed fix. Use `apply_corrections.py --filter-gate math_correct`.

4. **Per-instance review for coherence failures** (581 rows). Categorize by failure_mode; vendor-fabrication failures often need a question rewrite, physical-absurdity failures often need a number adjustment.

5. **Relabel down for level-fit failures** (1795 rows). Default disposition is L_claimed → L_actual. `apply_corrections.py --filter-gate level_fit` walks them.

6. **Bulk-fix placeholder titles** (216 rows) via Phase 7's title-only --propose-fixes pass.

Once the corpus is clean, Phase 6 lifts the format gate into `vault check --strict`'s structural tier and tightens the LinkML schema with pattern constraints — making the new state of cleanliness load-bearing.

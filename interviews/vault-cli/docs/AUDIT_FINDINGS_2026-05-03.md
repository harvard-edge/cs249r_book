# Corpus audit findings — 2026-05-03

**Source:** `interviews/vault/_pipeline/runs/full-corpus-20260503-merged/01_audit.json`
**Generated:** 2026-05-03T18:26:21+00:00
**Audit model:** `gemini-3.1-pro-preview`
**Triggered by:** CORPUS_HARDENING_PLAN.md Phase 4 finalization

---

## Executive summary

- **9446** questions audited
- **20** errored (no Gemini response — should be retried)

| gate | pass | fail | other |
|---|---:|---:|---:|
| format_compliance | 8411 | 1015 | 20 |
| level_fit | 7655 | 1768 | 23 |
| coherence | 8802 | 624 | 20 |
| math_correct | 8978 | 430 | 38 |
| title_quality | 9185 good | 223 placeholder + 18 malformed | 20 error |

---

## Per-track failure rates

| track | total | format | level_fit | coherence | math |
|---|---:|---:|---:|---:|---:|
| cloud | 4028 | 398 (9.9%) | 692 (17.2%) | 228 (5.7%) | 155 (3.8%) |
| edge | 2079 | 276 (13.3%) | 401 (19.3%) | 142 (6.8%) | 87 (4.2%) |
| global | 313 | 25 (8.0%) | 36 (11.5%) | 22 (7.0%) | 21 (6.7%) |
| mobile | 1824 | 153 (8.4%) | 382 (20.9%) | 148 (8.1%) | 117 (6.4%) |
| tinyml | 1202 | 163 (13.6%) | 257 (21.4%) | 84 (7.0%) | 50 (4.2%) |

---

## Coherence failure-mode breakdown

When coherence=fail, Gemini classifies the failure into one of four modes. The distribution tells us where to focus targeted fixes.

| failure_mode | count |
|---|---:|
| mismatch | 298 |
| arithmetic | 164 |
| physical_absurdity | 138 |
| vendor_fabrication | 24 |

---

## Priority lists for human review

### Math errors — 430 questions

**Highest priority.** Each requires per-instance human review (per CORPUS_HARDENING_PLAN.md §10 Q2). When fixing, rewrite BOTH napkin_math AND realistic_solution as a unit.

| qid | track | first error |
|---|---|---|
| `cloud-0009` | cloud | The calculation stops at 14GB for one model and fails to calculate the peak... |
| `cloud-0027` | cloud | Claims the 200ms wait is purely in the queue, but 200ms is the total time in... |
| `cloud-0126` | cloud | Claims a 100ms service time in the final addition but earlier specifies a... |
| `cloud-0145` | cloud | Calculates prefill compute time by multiplying the number of input tokens by... |
| `cloud-0209` | cloud | Claims ~200 H100s at $4/hr costs ~$168k/week, but 200 * 4 * 168 = 134,400 |
| `cloud-0230` | cloud | Concludes a 15% overall throughput drop despite calculating a latency penalty... |
| `cloud-0247` | cloud | Claims 32 * 256 * 1024 * 1024 * 2 bytes is 17.1 GB, but it is actually 17.18 GB... |
| `cloud-0454` | cloud | Calculates 11.6s total step time which contradicts the 11.3s stated in the... |
| `cloud-0468` | cloud | Claims transfer volume is 2 * 140 GB = 280 GB, but for 16 nodes it should be 2... |
| `cloud-0480` | cloud | Claims 2 * (N-1)/N * 140 GB is 280 GB, which is inaccurate for both N=8 (245... |
| `cloud-0491` | cloud | Claims centralized 3-year compute is $500k, but scenario states $500k... |
| `cloud-0536` | cloud | Calculates '$764k/year' savings but cost assumptions are entirely missing from... |
| `cloud-0537` | cloud | Calculates '$31k per incident' savings using component values ($15k GPU waste,... |
| `cloud-0555` | cloud | Calculates FLOPs per block as 32.7 TFLOPs, but 4 * (1000000/32)^2 * 64 * 128... |
| `cloud-0582` | cloud | claims 3x H100s frees GPUs and doubles fleet capacity but 5/3 = 1.66x capacity... |
| `cloud-0583` | cloud | claims INT8 Quantization size is 105 GB but conclusion relies on... |
| `cloud-0614` | cloud | claims 33 concurrent * 640 MiB is 21.1 GiB but 33 * 640 MiB = 21120 MiB =... |
| `cloud-0621` | cloud | Claims 0.96 sec/token is 39x slower, but baseline for 140 GB in HBM is 0.041... |
| `cloud-0625` | cloud | The math calculates an effective capacity of 21.6GB, which contradicts the... |
| `cloud-0647` | cloud | Claims wasting 6.6 bits 'Exceeds FP8 E4M3 4-bit exponent range'. A 100x ratio... |
| `cloud-0663` | cloud | claims 1,097 TFLOPs / 156 TFLOPS = ~7.0 seconds but relies on a 7.8s delay... |
| `cloud-0709` | cloud | Claims ~3.67x speedup relative to 8 nodes, but 64-node throughput relative to 8... |
| `cloud-0741` | cloud | Uses 28 GB for bandwidth calculation despite stating the model has 140 GB of... |
| `cloud-0745` | cloud | Claims 256 * $3.50 * (200 / 3600) * 24 = ~$995, but it actually equals ~$1,195. |
| `cloud-0813` | cloud | 500 * 292 = 146,000, not 146,200 |

_…and 405 more._

### Vendor fabrication — 24 questions

| qid | track | rationale |
|---|---|---|
| `cloud-0560` | cloud | The scenario uses a clearly fabricated hardware accelerator name ('Prometheus-1'). |
| `edge-1988` | edge | The scenario invents a fabricated hardware accelerator named 'EdgeCompute X'. |
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

### Physical absurdity — 138 questions

| qid | track | rationale |
|---|---|---|
| `cloud-0172` | cloud | A 70B parameter model in BF16/FP16 requires 140GB of memory, which physically cannot fit on a... |
| `cloud-0176` | cloud | Dividing the compute of a 70B model by a single H100's peak TFLOPS implies running it on one GPU,... |
| `cloud-0177` | cloud | The math explicitly claims a 'Single GPU' is serving a 70B (140GB) model decode, which physically... |
| `cloud-0178` | cloud | The calculations state a 70B model with 140GB of weights is running on 'an H100', but a single H100... |
| `cloud-0179` | cloud | The calculations divide the 70B model's compute by a single A100's peak TFLOPS, but a 70B model... |
| `cloud-0180` | cloud | The math divides the heavy prefill compute of a 70B model by a single H100's peak compute,... |
| `cloud-0304` | cloud | Calculating GPU rack capacity without accounting for host server power consumption is physically... |
| `cloud-0375` | cloud | Applying a facility-level PUE overhead directly to reduce the IT power capacity of a standard... |
| `cloud-0851` | cloud | An NVIDIA A100 GPU achieves a ResNet-50 image throughput of >10,000 images/sec, making 120ms for a... |
| `cloud-0925` | cloud | NVIDIA T4 hardware natively supports 1-bit Tensor Core operations (520 TOPS), so claiming it lacks... |
| `cloud-1041` | cloud | Calculating latency by multiplying the total number of cache misses serially by 85ns ignores memory... |
| `cloud-1042` | cloud | Assuming 234,000 cache line fetches execute purely sequentially with a 300ns stall each ignores... |
| `cloud-1046` | cloud | Calculating a 500ms stall from 5 million cache misses by multiplying serially by 100ns ignores... |
| `cloud-1201` | cloud | Table-wise sharding places the entire table on one GPU. A 400GB table cannot physically fit on a... |
| `cloud-1206` | cloud | Streaming 250 MB of raw user history per real-time request is physically absurd and would instantly... |
| `cloud-1226` | cloud | Processing 10,000 loan approvals per second implies over 300 billion applications per year, which... |
| `cloud-1260` | cloud | A 1050ms latency increase for a 16MB all_gather and a 68 GFLOP computation on A100 GPUs is... |
| `cloud-1262` | cloud | Claiming a 68.7 GFLOP computation is a severe computational bottleneck on an A100 GPU (which has... |
| `cloud-1313` | cloud | The napkin math calculates 99GB of VRAM per GPU, which physically exceeds the 80GB capacity of the... |
| `cloud-1377` | cloud | An ESP32 microcontroller lacks the compute power to run a motion-detection CNN at 10 FPS. |
| `cloud-1485` | cloud | A NUMA cross-node memory access penalty is typically on the order of ~150-200 nanoseconds, making... |
| `cloud-1574` | cloud | The hardware ridge point for an A100 is >150 FLOPs/byte, but the napkin math claims it is ~10... |
| `cloud-2171` | cloud | Claims Q, K, and V are read 'once each' for N=8192, implying 6MB of data fits simultaneously into... |
| `cloud-2174` | cloud | Register bandwidth of ~20 TB/s 'per SM' is physically absurd for an H100 (which has ~2-3 TB/s per... |
| `cloud-2443` | cloud | An A10G GPU has only 24GB of VRAM, making it physically impossible to load and serve a 28GB (14B... |

_…and 113 more._

### Scenario/solution mismatch — 298 questions

| qid | track | rationale |
|---|---|---|
| `cloud-0125` | cloud | The scenario and question state a P99 TTFT of 2.5 seconds, but the napkin math calculates it to be... |
| `cloud-0168` | cloud | The scenario states TTFT increased by 500ms, but the napkin math calculates a queue delay of over... |
| `cloud-0171` | cloud | The scenario claims a 500ms TTFT, but the solution and math indicate the request only waits for a... |
| `cloud-0423` | cloud | The scenario states GPU utilization fluctuates violently between 0% and 100%, but the napkin math... |
| `cloud-0441` | cloud | The scenario explicitly claims that GPU utilization is less than 15%, but the unfused napkin math... |
| `cloud-0454` | cloud | Numbers contradict; scenario states 11.3s step time but math derives 11.6s. |
| `cloud-0582` | cloud | Scenario mandates a 2x throughput increase but solution achieves 1.66x while claiming 2x. |
| `cloud-0629` | cloud | Scenario states GPUs normally sit at 40% memory utilization, but napkin math claims the 93.75 GB... |
| `cloud-0716` | cloud | The solution incorrectly attributes a sudden 2-second stall after 100 perfect steps to the regular... |
| `cloud-0741` | cloud | The napkin math analyzes a 512-GPU ring AllReduce for a 28 GB model, contradicting the 256-GPU, 70B... |
| `cloud-0885` | cloud | The scenario specifies a sequence length of 2048, but the solution and napkin math calculate the KV... |
| `cloud-0909` | cloud | The 40GB GPU VRAM limit is heavily relied upon in the solution and math but is never stated in the... |
| `cloud-0983` | cloud | The scenario claims 1.2 GB gradients take 240 ms on a 10 GB/s network, contradicting mathematical... |
| `cloud-1088` | cloud | Scenario claims utilization drops to 42%, but napkin math calculates current utilization as 29.4%. |
| `cloud-1090` | cloud | Napkin math assumes 5,000 legitimate refund examples which is never stated in the scenario. |
| `cloud-1111` | cloud | The napkin_math calculates storage strictly in base-10 (43.2M * 2KB = 86.4 GB) while the... |
| `cloud-1147` | cloud | The scenario claims GPU utilization is roughly 16%, but the napkin math calculates it as exactly... |
| `cloud-1235` | cloud | The realistic_solution fails to identify the hardware and data conditions that negate power... |
| `cloud-1241` | cloud | The realistic_solution completely fails to answer the quantitative question regarding the total... |
| `cloud-1242` | cloud | The realistic_solution fails to state the numeric speedup and memory costs requested by the prompt. |
| `cloud-1255` | cloud | The realistic_solution states the frequency of failures but omits a direct answer to the 'how many'... |
| `cloud-1592` | cloud | The scenario and question state a 4.5-minute delay, but the napkin math calculates 283 seconds,... |
| `cloud-1643` | cloud | The scenario specifies 4,000 images/sec total, but the napkin math treats it as 4,000 per GPU... |
| `cloud-1657` | cloud | The math shows 25.92GB of VRAM growth in 48 hours, which contradicts the scenario's claim that it... |
| `cloud-1661` | cloud | The conclusion incorrectly claims the 1.6ms of calculated PCIe transfer overhead dominates the 45ms... |

_…and 273 more._

### Level inflation — 1768 questions

Per CORPUS_HARDENING_PLAN.md §10 Q3, default disposition is **relabel down** to the actual cognitive level. Rewriting the question up is a separate authoring task, not a Phase-5 concern.

| qid | track | claimed level | rationale |
|---|---|---|---|
| `cloud-0002` | cloud | ? | Labeled as L1 recall, but the question asks for a multi-step calculation which... |
| `cloud-0016` | cloud | ? | Labeled as L1 recall, but identifying the primary suspect for a performance... |
| `cloud-0099` | cloud | ? | Level inflation: L3+ stamped on a simple multiplication/division problem with... |
| `cloud-0109` | cloud | ? | Level inflation: L3+ stamped on a simple multiplication/division problem with... |
| `cloud-0195` | cloud | ? | Level inflation: the question asks to recall an existing limitation and name a... |
| `cloud-0229` | cloud | ? | Claimed L6+ (create) but the question simply asks to recall a standard industry... |
| `cloud-0230` | cloud | ? | Claimed L6+ (create) but the question asks to root-cause a performance penalty,... |
| `cloud-0259` | cloud | ? | Claimed L1 (remember) but the question requires applying multi-step formulas to... |
| `cloud-0279` | cloud | ? | Verb mismatch: determining bounding by calculating and comparing AI requires... |
| `cloud-0295` | cloud | ? | Verb mismatch: determining the bottleneck requires multi-step analysis (L4),... |
| `cloud-0296` | cloud | ? | Verb mismatch: determining if workload is compute or memory bound by... |
| `cloud-0325` | cloud | ? | Verb mismatch: question requires root-cause analysis to determine bottleneck... |
| `cloud-0329` | cloud | ? | Verb mismatch: question asks to compare and choose between architectural... |
| `cloud-0340` | cloud | ? | Verb mismatch: question asks to evaluate which scenario is more cost-effective... |
| `cloud-0349` | cloud | ? | Verb mismatch: question requires root-cause analysis to determine if memory or... |
| `cloud-0356` | cloud | ? | Verb mismatch: question requires root-cause analysis to determine if memory or... |
| `cloud-0374` | cloud | ? | The question requires analyzing constraints and evaluating trade-offs between... |
| `cloud-0384` | cloud | ? | The question requires decomposition and root-cause reasoning to explain why a... |
| `cloud-0409` | cloud | ? | The question requires decomposing a matrix multiplication to calculate AI and... |
| `cloud-0454` | cloud | ? | Verb mismatch; question asks 'Why' (L2 understand / L4 analyze) but level is L3... |
| `cloud-0456` | cloud | ? | Verb mismatch; question asks 'Why' and 'what approach' (L2/L5) but level is L3... |
| `cloud-0457` | cloud | ? | Verb mismatch; question asks 'Why' (L2/L4) but level is L3 (apply). |
| `cloud-0458` | cloud | ? | Verb mismatch; question asks 'What is the most likely cause' (L4) but level is... |
| `cloud-0461` | cloud | ? | Verb mismatch; question asks 'What is the most likely physical bottleneck' (L4)... |
| `cloud-0463` | cloud | ? | Verb mismatch; question asks 'What is the most likely cause' (L4) but level is... |

_…and 1743 more._

### Placeholder titles — 223 questions

Phase 7 will batch these for Gemini-proposed replacements (~1 call per 5 placeholders). All require human review of the proposed title.

qids (first 25): `cloud-2510`, `cloud-2512`, `cloud-2513`, `cloud-2514`, `cloud-2515`, `cloud-2516`, `cloud-2517`, `cloud-2518`, `cloud-2519`, `cloud-2520`, `cloud-2521`, `cloud-2522`, `cloud-2523`, `cloud-2524`, `cloud-2525`, `cloud-2526`, `cloud-2527`, `cloud-2528`, `cloud-2529`, `cloud-2530`, `cloud-2531`, `cloud-2532`, `cloud-2533`, `cloud-2534`, `cloud-2535`

_…and 198 more._


---

## Format-compliance: regex vs. Gemini

Of the rows where both verdicts are present, **218** disagree. Regex is the source of truth (it's mechanical); disagreements indicate Gemini missed a marker the regex caught (or vice versa).

| qid | gemini | regex | regex_issues |
|---|---|---|---|
| `cloud-0752` | pass | fail | napkin_math missing ['**Assumptions', '**Calculations:**', '**Conclusion'] |
| `cloud-0966` | pass | fail | napkin_math missing ['**Assumptions', '**Calculations:**', '**Conclusion'] |
| `cloud-1764` | fail | pass |  |
| `cloud-1767` | fail | pass |  |
| `cloud-1771` | fail | pass |  |
| `cloud-1780` | fail | pass |  |
| `cloud-1781` | fail | pass |  |
| `cloud-1782` | fail | pass |  |
| `cloud-1784` | fail | pass |  |
| `cloud-2278` | fail | pass |  |
| `cloud-2279` | fail | pass |  |
| `cloud-2282` | fail | pass |  |
| `cloud-2283` | fail | pass |  |
| `cloud-2285` | fail | pass |  |
| `cloud-2290` | fail | pass |  |
| `cloud-2294` | fail | pass |  |
| `cloud-2296` | fail | pass |  |
| `cloud-2297` | fail | pass |  |
| `cloud-2302` | fail | pass |  |
| `cloud-2308` | fail | pass |  |

_…and 198 more._


---

## Recommendations

1. **Resume to clear errored batches.** 20 rows show `format_compliance: error` — these batches didn't get a Gemini response. Re-running `audit_corpus_batched.py --output <same-dir>` retries them.

2. **Run `--propose-fixes` on the format-fail subset.** 1038 rows fail format. Most are mechanical marker additions — `apply_corrections.py --auto-accept-format` will auto-apply low-risk fixes.

3. **Per-instance review for math errors.** 430 rows have math errors. Each needs a human to verify the proposed fix. Use `apply_corrections.py --filter-gate math_correct`.

4. **Per-instance review for coherence failures** (624 rows). Categorize by failure_mode; vendor-fabrication failures often need a question rewrite, physical-absurdity failures often need a number adjustment.

5. **Relabel down for level-fit failures** (1768 rows). Default disposition is L_claimed → L_actual. `apply_corrections.py --filter-gate level_fit` walks them.

6. **Bulk-fix placeholder titles** (223 rows) via Phase 7's title-only --propose-fixes pass.

Once the corpus is clean, Phase 6 lifts the format gate into `vault check --strict`'s structural tier and tightens the LinkML schema with pattern constraints — making the new state of cleanliness load-bearing.

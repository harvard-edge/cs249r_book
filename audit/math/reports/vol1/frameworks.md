# Math Audit Report: `book/quarto/contents/vol1/frameworks/frameworks.qmd`

## Checked scope

Audited `book/quarto/contents/vol1/frameworks/frameworks.qmd` for equations, generated numeric examples, unit conversions, complexity/scaling claims, performance claims, and prose-equation consistency. Direct reasoning only; no Gemini or external verification used.

Checked items include:

- Lines 49-107, 247-300: A100 compute/bandwidth constants and memory-wall reasoning.
- Lines 854-930, 1449-1547, 1828-1888, 1966-2118: dispatch tax, fusion speedup, compilation continuum, breakeven, and overhead-ratio equations.
- Lines 2351-2520, 2760-2815, 3008-3084, 4841-4876: GPT-3, ResNet-50, 1B-model, and 7B-model memory examples.
- Lines 3266-3370, 3491-3551: bandwidth hierarchy, transfer-time examples, and dataloader throughput.
- Lines 3848-3897, 4359-4378, 4535-4746, 4780-4942: ResNet FLOPs, framework efficiency table, training-step roofline table, fallacies, and summary claims.

## Findings

### 1. Training-step roofline table misclassifies small matmuls as compute-bound

- **Lines:** 4635-4675, 4682-4690
- **Severity:** Medium
- **Issue:** The table computes arithmetic intensities of about `12.8M FLOPs / 0.94 MB = 13.6 FLOPs/byte` for `x @ W1` and `164K FLOPs / 44 KB = 3.7 FLOPs/byte` for `h @ W2`. Those values are above 1.0, but the chapter's own A100 roofline point is about `312 TFLOPS / 2 TB/s = 156 FLOPs/byte` (lines 255-290). On an A100, these matmuls are still below the compute-bound threshold. The caption's phrase "compute bound on most hardware" is therefore not supported by the chapter's roofline model.
- **Proposed correction:** Change the caption to say the matmuls have higher arithmetic intensity than element-wise operations but remain memory/overhead-sensitive at these small dimensions on high-end GPUs. Reserve "compute-bound" for large GEMMs whose arithmetic intensity exceeds the target hardware ridge point.

### 2. Adam optimizer-state multiplier is wrong for FP16 weights

- **Lines:** 3076-3078
- **Severity:** Medium
- **Issue:** The administrative-tax callout says Adam optimizer states are `$2 \times \text{weights}$` for momentum and velocity in FP32. With FP16 weights, the weights use 2 bytes/parameter, while two FP32 Adam buffers use `2 * 4 = 8` bytes/parameter. The optimizer state is therefore `8 / 2 = 4x` the FP16 weight memory, not `2x`. The generated 1B-parameter numbers are internally consistent with `2 GB` weights and `8 GB` optimizer state.
- **Proposed correction:** Replace the parenthetical with "`4 \times` FP16 weight memory, because Adam stores two FP32 buffers" or define the comparison as "`2 \times` FP32 parameter memory."

### 3. A100 remaining-memory example mixes GB and GiB

- **Lines:** 4857-4871, 4876
- **Severity:** Medium
- **Issue:** The code converts `A100_MEM_CAPACITY` with `.m_as(GiB)` but computes model weights in decimal `GB`, then subtracts the two raw numbers. If the A100 capacity is intended to be 80 GiB, `14 GB` is about `13.0 GiB`, leaving about `67 GiB` or `71.9 GB` depending on display unit. If the prose intends decimal "A100-80 GB", then using `GiB` in the calculation is inconsistent even though the displayed `80 - 14 = 66` arithmetic is simple.
- **Proposed correction:** Use one unit convention throughout. For decimal prose, convert capacity with `GB` and report `80 GB - 14 GB = 66 GB`. For binary capacity, label it `80 GiB` and subtract `14 GB = 13.0 GiB`, leaving about `67 GiB`.

### 4. 7B optimizer-state source comments claim an impossible total

- **Lines:** 2770-2775, 2803-2815
- **Severity:** Low
- **Issue:** The comments say `14 GB weights + 56 GB Adam state = ~98 GB total`, but the code correctly computes `14 + 56 = 70 GB`. Including FP16 gradients would give `84 GB`; including an FP32 master copy would give a different total; none gives 98 from the listed terms.
- **Proposed correction:** Change the comments to "~70 GB total (14 GB weights + 56 GB optimizer state)" if the intended total excludes gradients, or explicitly add any extra terms and update the code/prose accordingly.

### 5. Dataloader throughput silently assumes raw 8-bit images while GPU consumption wording suggests tensors

- **Lines:** 3498-3528, 3545-3551
- **Severity:** Low
- **Issue:** The throughput calculation for 1,000 images/s uses `224 * 224 * 3 * 1 byte`, yielding about `150 MB/s`. Later, the batch-transfer calculation uses FP32 and gives about `64 * 224 * 224 * 3 * 4 = 38.5 MB` per batch. If the pipeline must sustain FP32 tensors at 1,000 images/s, the required throughput is about `602 MB/s`; for FP16 tensors it is about `301 MB/s`. The rendered prose says "accelerator's consumption rate," which reads like post-preprocessing tensor bandwidth, not compressed/raw image bytes.
- **Proposed correction:** Qualify the 150 MB/s number as raw uint8 image ingress before preprocessing, or multiply by `BYTES_FP32`/`BYTES_FP16` when describing tensor throughput to the accelerator.

### 6. Summary memory-gap claim disagrees with the generated fallacy calculation

- **Lines:** 4807-4818, 4827, 4942
- **Severity:** Low
- **Issue:** The fallacy calculation uses decimal units: `220 MB * 1000 / 32 KB = 6,875x`. The summary reports a `7,040x` memory gap, which corresponds to binary `1 MB = 1024 KB`. Both conventions are defensible, but using both in the same chapter creates a prose-equation inconsistency.
- **Proposed correction:** Use `6,875x` everywhere for decimal MB/KB, or switch the calculation to binary units and label the operands as MiB/KiB if the intended summary number is `7,040x`.

### 7. Dispatch-overhead examples use inconsistent per-operation overheads

- **Lines:** 854, 877-895, 4564-4568, 4716-4725
- **Severity:** Low
- **Issue:** The chapter alternates among ~10 microseconds per Python dispatch (line 854 and `DispatchTax`), ~1 microsecond Python dispatch plus ~5 microseconds kernel launch in the training-step trace, and 5 microseconds per operation in the MNIST overhead calculation. These can be different components, but the prose does not clearly separate Python dispatcher overhead from CUDA kernel-launch overhead and combined per-op overhead.
- **Proposed correction:** Define the components once, for example `T_op_overhead = T_python_dispatch + T_kernel_launch`, and label each later example as Python-only, launch-only, or combined overhead.

## Verified calculations and consistency checks

- Lines 99-104: Sparse A100 throughput as `2 * dense` is internally consistent with the code's intended structured-sparsity example.
- Lines 283-290 and 300: `312 TFLOPS / 2.0 TB/s = 156 FLOPs/byte`, supporting the memory-wall discussion and the "less than one percent" ReLU utilization claim.
- Lines 889-895: Dispatch tax values are correct: `10 / (10 + 1) = 90.9%` and `10 / (10 + 100) = 9.1%`.
- Lines 1468-1492: Fusion example is arithmetically correct: two eager launches vs one fused launch gives `30 us / 15 us = 2x`, and memory traffic factor `4 / 2 = 2x`.
- Lines 1836-1848: The compilation-benefit equation is dimensionless, and the `> 1` decision rule is coherent.
- Lines 1884-1888: ResNet-50 breakeven is correct: `30 / (1/1450 - 1/2150) = 133,607` images, which rounds to about `134,000`.
- Lines 1968-1970: The dispatch-overhead equation is dimensionless and correctly separates overhead from compute plus memory time.
- Lines 2014-2024 and 2058-2060: Small-model overhead example is correct: `6 * 5 us = 30 us`, `30 / 2.6 = 11.5`, and `30 / 32.6 = 92%`.
- Lines 2374-2395: GPT-3 FP16 weights are correct: `175B * 2 bytes = 350 GB`.
- Lines 2507-2538: ResNet-50 FP32 weights are about `25.6M * 4 bytes = 102.4 MB`; two FP32 Adam buffers are about `204.8 MB`.
- Lines 2803-2810: 7B FP16 weights and Adam state are correctly computed as `14 GB` and `56 GB`; the issue is only the nearby comments noted above.
- Lines 3024-3044 and 3076-3084: 1B FP16 weights `2 GB`, gradients `2 GB`, Adam state `8 GB`, activations about `6.7 GB`, and administrative tax about `17 GB` are internally consistent.
- Lines 3298-3320 and 3352-3369: 4 MB transfer times are correct at the stated bandwidths: `4 MB / 32 GB/s = 0.125 ms`, `4 MB / 300 GB/s = 0.013 ms`, and `4 MB / 2 TB/s = 0.002 ms`.
- Lines 3523-3529 and 3551: FP32 batch transfer is correct: `64 * 224 * 224 * 3 * 4 bytes = 38.5 MB`, and over PCIe 4.0 x16 at `32 GB/s` this is about `1.2 ms`.
- Lines 3866-3889: ResNet-50 `8.2 GFLOPs` and the five element-wise operations on a 100 MB tensor moving `5 * 2 * 100 MB = 1 GB` are consistent.
- Lines 4362-4376 and 4823: PyTorch vs TensorRT latency ratio is `52 / 3 = 17.3`, so the rounded `17x` claim is correct; utilization ratio is `88 / 32 = 2.75x`.
- Lines 4716-4746: MNIST overhead example is correct under its assumptions: `40M / 312 TFLOPS = 0.13 us`, `5 MB / 2 TB/s = 2.5 us`, and `6 * 5 us = 30 us`.
- Lines 4903-4922: Compilation-overhead example is correct: eager time `10000 / 1450 = 6.9 s`; compiled time `10000 / 2150 + 10 * 30 = 304.7 s`.

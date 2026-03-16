# Round 2: Edge Advanced — Memory, Architecture & Deployment 🏭

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Edge_Systems.md">🤖 Edge Round 1</a> ·
  <a href="02_Edge_Advanced.md">🏭 Edge Round 2</a>
</div>

---

This round expands the Edge track into memory management under shared DRAM, architecture selection for real-time vision, model optimization ladders, deployment and OTA, and security against physical-world adversarial attacks.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/edge/02_Edge_Advanced.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 📐 Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Bandwidth-Bound Orin</b> · <code>roofline</code></summary>

**Interviewer:** "Your Jetson Orin NX profiler reports 70 TOPS out of a rated 100 TOPS INT8. The team celebrates — 70% utilization sounds great. But your YOLOv8-M model still runs at 15 FPS instead of the expected 45 FPS. What is the profiler actually telling you?"

**Common Mistake:** "We're at 70% compute utilization, so the remaining 30% is overhead we can optimize away." This confuses which resource is being utilized.

**Realistic Solution:** The 70% figure is memory bandwidth utilization, not compute utilization. The Orin NX has 102.4 GB/s LPDDR5 bandwidth and a ridge point of ~976 Ops/Byte (100 TOPS / 102.4 GB/s). YOLOv8-M has an arithmetic intensity of roughly 200 Ops/Byte — well below the ridge. The model is memory-bandwidth bound: the GPU's INT8 cores are starved, waiting for data from DRAM. Buying a faster accelerator won't help. You need to reduce memory traffic: fuse layers to keep activations in on-chip SRAM, use depth-wise convolutions that have higher arithmetic intensity, or reduce input resolution.

> **Napkin Math:** YOLOv8-M: ~39 GFLOPs, ~80 MB memory traffic per inference. Arithmetic intensity = 39G / 80M ≈ 488 Ops/Byte. Bandwidth ceiling = 102.4 GB/s × 488 = 50 TOPS attainable. At 70% bandwidth utilization: 0.7 × 50 = 35 TOPS effective. Per-frame time = 39G / 35T = 1.1ms compute, but memory stalls stretch it to ~22ms → 45 FPS theoretical, 15 FPS actual due to activation spills and strided access patterns.

> **Key Equation:** $\text{Attainable TOPS} = \min(\text{Peak TOPS},\ \text{BW} \times \text{Arithmetic Intensity})$

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Dataflow vs GPU Roofline</b> · <code>roofline</code></summary>

**Interviewer:** "Your team is choosing between a Hailo-8 (26 TOPS, 2.5W, dataflow architecture) and a Jetson Orin NX (100 TOPS, 15W, GPU architecture) for a drone running YOLOv8-S. The Orin has 4× the TOPS. But in your benchmarks, the Hailo runs the model at 28 FPS while the Orin only hits 35 FPS. Why is the 4× TOPS advantage only yielding a 1.25× speedup?"

**Common Mistake:** "The Orin's drivers aren't optimized yet" or "The Hailo benchmark is wrong." Both dodge the architectural explanation.

**Realistic Solution:** The Hailo-8 is a dataflow architecture — it maps the entire model graph onto a spatial pipeline of physical compute units. Activations flow between stages through on-chip buffers without ever touching external DRAM. The Orin NX is a GPU — it executes layers sequentially, reading weights and activations from LPDDR5 between each layer. For YOLOv8-S (arithmetic intensity ~516 Ops/Byte), the Orin is memory-bandwidth bound: its 102.4 GB/s LPDDR5 limits effective throughput to ~50 TOPS. The Hailo eliminates the DRAM bottleneck entirely, so its 26 TOPS are nearly fully utilized. Effective throughput: Hailo ~24 TOPS (92%) vs Orin ~35 TOPS (35%). The dataflow architecture changes the shape of the roofline itself — there is no memory wall.

> **Napkin Math:** Hailo-8: 26 TOPS peak, ~92% utilization (no DRAM stalls) = 24 TOPS effective. Per-frame: 28.4 GFLOPs / 24 TOPS = 1.18ms → ~28 FPS with overhead. Orin NX: 100 TOPS peak, bandwidth-limited to ~50 TOPS, 35% utilization = 35 TOPS effective. Per-frame: 28.4G / 35T = 0.81ms compute + ~28ms memory stalls → ~35 FPS. Power: Hailo at 2.5W = 11.2 FPS/W. Orin at 15W = 2.3 FPS/W. The Hailo is **4.9× more power-efficient** for this workload.

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

---

### 🧠 Memory Systems

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Shared DRAM Budget</b> · <code>memory</code></summary>

**Interviewer:** "Your edge box has 8 GB of LPDDR5 DRAM. Your ML model needs 1.5 GB for weights and activations. Your colleague says 'we have 6.5 GB of headroom — plenty.' Why is this dangerously optimistic?"

**Common Mistake:** "8 GB minus 1.5 GB leaves 6.5 GB free." This treats the edge device like a dedicated ML accelerator with nothing else running.

**Realistic Solution:** Unlike a data center GPU with dedicated HBM, edge DRAM is shared with everything: the Linux kernel and drivers (~500 MB), camera ISP and sensor pipelines (~1 GB for 4K video), display compositor (~300 MB), networking stack (~200 MB), system services and logging (~500 MB). Realistic free memory: ~5 GB. But that's the static picture. Under load, the camera ISP can burst to 2 GB for multi-frame HDR processing, and the Linux page cache will aggressively claim free memory. Your ML process can be OOM-killed at any time if the kernel needs memory for a higher-priority subsystem. You must use `mlockall()` to pin your model's pages in RAM, set cgroup memory limits to protect your allocation, and budget for worst-case concurrent memory usage — not average.

> **Napkin Math:** 8 GB total. Kernel + drivers: 500 MB. Camera ISP (4K, burst): 2 GB. Display: 300 MB. Network + services: 700 MB. Available for ML: 8 - 3.5 = **4.5 GB worst case**. Your 1.5 GB model fits, but only with 3 GB headroom — not 6.5 GB. If another process spikes, you're at risk.

**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The eMMC Cold Start</b> · <code>memory</code></summary>

**Interviewer:** "Your edge device loads a 200 MB model from eMMC flash into DRAM at boot. The eMMC spec says 300 MB/s sequential read, so you expect a 0.67-second load time. In practice, first inference takes over 3 seconds. Where did the other 2.3 seconds go?"

**Common Mistake:** "The eMMC is slower than spec." The raw bandwidth is fine — the overhead is elsewhere.

**Realistic Solution:** The 300 MB/s spec is for large sequential reads. Model loading involves: (1) filesystem overhead — ext4 metadata lookups, inode traversal, and block allocation table reads add ~200ms, (2) the model file is fragmented on a well-used eMMC — random 4K reads drop to ~20 MB/s, adding ~500ms, (3) TensorRT engine deserialization — parsing the serialized engine, allocating CUDA memory, and building the execution context takes ~1.5s for a 200 MB engine, (4) CUDA context initialization on first use adds ~300ms. The fix: (a) store the model on a dedicated, defragmented partition, (b) use `mmap()` to map the weight file so pages load on demand during the first inference rather than all at once, (c) keep a lightweight fallback model (~20 MB, loads in <200ms) that serves immediately while the full model loads in a background thread.

> **Napkin Math:** Sequential read: 200 MB / 300 MB/s = 0.67s. Filesystem overhead: +0.2s. Fragmentation penalty: +0.5s. TensorRT deserialization: +1.5s. CUDA init: +0.3s. Total: **3.17s**. With mmap + background load: fallback model ready in 0.2s, full model ready in ~3s but user never waits.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Multi-Model Memory Sharing</b> · <code>memory</code></summary>

**Interviewer:** "Your autonomous vehicle runs four models concurrently: detection (YOLOv8-L, 80 MB weights), tracking (DeepSORT, 30 MB), depth estimation (MiDaS, 200 MB), and path planning (custom, 50 MB). Total: 360 MB of weights plus ~400 MB of activation buffers. Your Jetson AGX Orin has 32 GB DRAM, but after the OS and sensor pipelines, only 4 GB is available for ML. How do you fit 760 MB of ML workload into 4 GB with room for growth?"

**Common Mistake:** "4 GB is way more than 760 MB — there's no problem." This ignores peak memory, concurrent allocations, and fragmentation.

**Realistic Solution:** The 760 MB is the *minimum* — it ignores peak activation memory during concurrent execution. When detection and depth estimation run simultaneously, their activation buffers peak at ~600 MB combined (not 400 MB, because peaks overlap). Plus TensorRT workspace memory (~200 MB per engine). Real peak: ~1.2 GB. The strategy: (1) **Shared tensor allocator** — use a single CUDA memory pool (like TensorRT's `IGpuAllocator`) that reuses activation buffers between sequential stages. Detection's output buffer becomes tracking's input buffer without a copy. (2) **Temporal multiplexing** — depth estimation doesn't need to run every frame. Run it at 10 FPS (every 3rd frame) and reuse its memory during off-frames for other models. (3) **Backbone sharing** — if detection and depth share a ResNet backbone, load the backbone weights once and run both heads on the shared features. This saves 60+ MB of duplicated weights. (4) **Memory-mapped weights** — mmap the planning model's weights so they're paged in only when the planning module runs (every 100ms), freeing physical RAM between invocations.

> **Napkin Math:** Naive: 360 MB weights + 600 MB peak activations + 400 MB TensorRT workspace = 1.36 GB. With shared allocator: activations drop to ~350 MB (sequential reuse). With backbone sharing: weights drop to ~280 MB. With temporal multiplexing: peak concurrent memory = ~800 MB. Headroom in 4 GB: 3.2 GB free for sensor buffers and growth.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🔢 Numerical Representation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The PTQ vs QAT Question</b> · <code>quantization</code></summary>

**Interviewer:** "Your manager says 'just quantize the model to INT8 — it takes five minutes with TensorRT.' You push back and say you need two weeks for quantization-aware training. Justify the two weeks."

**Common Mistake:** "QAT is always better than PTQ." This is too vague — you need to explain *when* PTQ fails and *why* QAT fixes it.

**Realistic Solution:** Post-training quantization (PTQ) collects activation statistics from a small calibration dataset (~1000 images) and sets quantization ranges based on observed min/max values. It works well when: activation distributions are symmetric and well-behaved, the model is large (redundancy absorbs quantization noise), and the task is tolerant of small errors (classification). PTQ *fails* when: (1) the model has outlier channels — a few activations with 10× the magnitude of others force the quantization range to be wide, crushing precision for the majority, (2) the task is precision-sensitive — depth estimation, segmentation boundaries, or small-object detection where a 1% pixel shift matters, (3) the model is already small — a MobileNet has less redundancy to absorb quantization error than a ResNet-50. QAT inserts fake-quantization nodes during training, so the model's gradients learn to compensate for quantization error. The two weeks cover: fine-tuning with quantization noise (~3 days), hyperparameter search for learning rate and quantization scheme (~5 days), validation across the full test distribution including edge cases (~4 days).

> **Napkin Math:** PTQ: 1000 calibration images × 5ms each = 5 seconds + TensorRT build = 5 minutes total. Accuracy: typically -0.5% to -3% for classification, -2% to -15% for detection on hard classes. QAT: 50 epochs × 10,000 images × 20ms/image = ~2.8 hours training + validation = 2 weeks of engineering time. Accuracy: typically -0.1% to -0.5% — recovering most of the PTQ loss.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Night Scene Calibration Failure</b> · <code>quantization</code></summary>

**Interviewer:** "You INT8-quantized your pedestrian detection model using PTQ. Daytime accuracy drops 2% — acceptable. But your test suite reveals nighttime recall drops 15%. The safety team blocks deployment. Your calibration dataset had 1000 images. What went wrong?"

**Common Mistake:** "INT8 is too aggressive for night scenes — use FP16 at night." This doubles your compute cost and doesn't explain the root cause.

**Realistic Solution:** Your 1000 calibration images were predominantly daytime scenes (reflecting the training distribution). PTQ set the quantization ranges based on daytime activation statistics — bright, high-contrast features with large activation magnitudes. Nighttime pedestrians produce activations in the low-magnitude tail of the distribution. With per-tensor quantization, the step size is set by the max activation value (dominated by daytime). Nighttime activations, being 10-50× smaller, get crushed into just a few quantization bins, destroying the signal. Fixes: (1) curate a calibration dataset that represents the full operating domain — 50% day, 30% night, 20% twilight/rain, (2) switch from per-tensor to per-channel quantization, which sets independent ranges per output channel and is more robust to distribution variation, (3) use percentile calibration (99.99th percentile instead of min/max) to clip outliers and give more precision to the common range.

> **Napkin Math:** Per-tensor INT8 range set by daytime max activation = 10.0. Step size = 10.0 / 127 = 0.079. Nighttime pedestrian activations in range [0.01, 0.05]: only (0.05 - 0.01) / 0.079 = **0.5 bins** — the entire signal collapses to a single quantized value. Per-channel quantization: if the relevant channel's max = 0.1, step size = 0.1/127 = 0.0008 → nighttime range spans 50 bins. Signal preserved.

> **Key Equation:** $\text{Step Size} = \frac{x_{\max} - x_{\min}}{2^b - 1}$

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Mixed-Precision Perception Stack</b> · <code>quantization</code></summary>

**Interviewer:** "You're architecting the inference pipeline for an autonomous vehicle with a Jetson AGX Orin. The stack has three models: object detection, monocular depth estimation, and motion planning. Design the precision strategy for each model and justify your choices from first principles."

**Common Mistake:** "Run everything in INT8 for maximum speed" or "Run everything in FP32 for maximum safety." Both ignore the error-tolerance profile of each component.

**Realistic Solution:** Each model has a different error-tolerance profile that dictates its precision:

**Detection (INT8):** Object detection is inherently robust to small numerical perturbations — a bounding box shifted by 1 pixel or a confidence score off by 0.01 doesn't change the detection outcome. INT8 quantization typically costs <1% mAP on well-calibrated models. The 2× speedup and 4× memory reduction over FP32 are critical for meeting the 33ms frame budget.

**Depth Estimation (FP16):** Monocular depth maps are sensitive to numerical precision because small activation differences map to large depth differences at range. A 1% error in a feature map can translate to a 2-meter depth error at 100m — the difference between "safe to proceed" and "collision imminent." FP16 preserves sufficient precision while still running on Tensor Cores (2× speedup over FP32).

**Motion Planning (FP32):** The planning module must be bit-exact with the simulation environment used for safety validation. Any numerical divergence between the deployed model and the simulator invalidates the safety case. FP32 guarantees reproducibility. The planning model is small (~50 MB) and runs at 10 Hz, so the compute cost is negligible.

> **Napkin Math:** Detection INT8: 28 GFLOPs / 100 TOPS = 0.28ms. Depth FP16: 40 GFLOPs / 137.5 TFLOPS FP16 = 0.29ms. Planning FP32: 2 GFLOPs / 68.75 TFLOPS FP32 = 0.03ms. Total compute: 0.6ms. Memory: detection 11 MB (INT8) + depth 80 MB (FP16) + planning 200 MB (FP32) = 291 MB. All fit comfortably in the Orin's 32/64 GB DRAM with the pipeline completing well within the 33ms budget even with memory stalls.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 🏗️ Architecture → System Cost

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The YOLO vs ViT Question</b> · <code>architecture</code></summary>

**Interviewer:** "A researcher on your team wants to replace your YOLOv8-S detector with a ViT-B/16 vision transformer because it scores 2% higher mAP on COCO. You're deploying on a Jetson Orin NX at 30 FPS. Why do you push back?"

**Common Mistake:** "ViT has more parameters, so it's slower." ViT-B actually has fewer FLOPs than some YOLO variants — the issue is deeper than parameter count.

**Realistic Solution:** ViT-B/16 has ~17.6 GFLOPs vs YOLOv8-S's ~28.4 GFLOPs — fewer FLOPs, yet it runs *slower* on edge GPUs. The reason: attention's memory access pattern. Self-attention computes Q×K^T (an N×N matrix for N=196 patches at 224×224), which has low arithmetic intensity — it's a series of small matrix multiplies with large intermediate tensors that must be written to and read from DRAM. Convolutional layers in YOLO have regular, predictable memory access patterns that map efficiently to GPU SRAM tiling and TensorRT optimization. Additionally, ViT's dynamic shapes (variable sequence length) prevent many TensorRT optimizations (layer fusion, kernel auto-tuning) that assume fixed tensor dimensions. On the Orin NX: YOLOv8-S runs at ~45 FPS with TensorRT INT8. ViT-B runs at ~15 FPS because TensorRT can't fuse the attention layers effectively.

> **Napkin Math:** YOLOv8-S: 28.4 GFLOPs, ~55 MB memory traffic → arithmetic intensity ≈ 516 Ops/Byte. ViT-B: 17.6 GFLOPs, ~120 MB memory traffic (attention matrices) → arithmetic intensity ≈ 147 Ops/Byte. On Orin NX (ridge ~976 Ops/Byte): both are memory-bound, but ViT is 3.5× more memory-hungry per FLOP. The 2% mAP gain costs a 3× FPS penalty.

**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Tracker Addition Budget</b> · <code>architecture</code></summary>

**Interviewer:** "Your perception stack runs YOLOv8-S detection at 20ms per frame on a Jetson Orin NX, leaving 13ms of headroom in your 33ms budget. Your team wants to add a Transformer-based tracker (ByteTrack with a ReID model, ~15 GFLOPs). The ReID model alone takes 12ms. Your colleague says '20 + 12 = 32ms — we fit with 1ms to spare.' Why is this estimate dangerously wrong?"

**Common Mistake:** "The math works: 32ms < 33ms." This assumes zero overhead and perfect sequential execution.

**Realistic Solution:** The 20ms + 12ms = 32ms estimate ignores: (1) **GPU memory contention** — both models compete for the same LPDDR5 bandwidth. When running concurrently, memory bandwidth is split, increasing both models' latency by 15-30%. (2) **CUDA context switching** — swapping between two TensorRT engines incurs ~1-2ms of overhead per switch. (3) **NMS and post-processing** — detection NMS takes 2-4ms on dense scenes (not included in the 20ms). (4) **Data transfer** — copying detection crops to the ReID model's input tensor takes ~0.5ms. Realistic total: 20ms (detect) + 3ms (NMS) + 0.5ms (copy) + 12ms (ReID) + 2ms (context switch) + 15% bandwidth penalty = **~43ms** → 23 FPS, missing the 30 FPS deadline. Fix: use a lightweight tracker (DeepSORT with a MobileNet ReID: 2 GFLOPs, ~3ms) or run the ReID model at half rate (every other frame, using motion prediction to interpolate).

> **Napkin Math:** Optimistic: 20 + 12 = 32ms. Realistic: 20 + 3 (NMS) + 0.5 (copy) + 14 (ReID with BW contention) + 2 (context switch) = 39.5ms → 25 FPS. With lightweight ReID (3ms): 20 + 3 + 0.5 + 3 + 1 = 27.5ms → 36 FPS ✓. Half-rate heavy ReID: alternating 20ms and 39.5ms frames → average 29.75ms → 33 FPS ✓.

**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
</details>

---

### ⏱️ Latency & Throughput

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Adaptive Quality Ladder</b> · <code>latency</code></summary>

**Interviewer:** "Your 30 FPS edge perception pipeline runs fine on open roads but drops to 22 FPS in dense urban intersections (many pedestrians, vehicles, signs). You can't miss frames — the safety system requires continuous perception. Design a system that maintains 30 FPS under all conditions without changing hardware."

**Common Mistake:** "Just use a faster model." You can't swap models at runtime without a pre-planned strategy, and a single model can't be optimal for all scene complexities.

**Realistic Solution:** Design an **adaptive quality ladder** — a pre-defined set of operating points with known latency, each trading quality for speed:

**Rung 0 (Nominal):** Full resolution (640×640), full model, all post-processing. Latency: 25ms. Used when scene complexity is low.

**Rung 1 (Medium):** Reduced resolution (480×480), reducing FLOPs by ~44%. Latency: 17ms. Accuracy drops ~2% mAP but still above safety threshold.

**Rung 2 (Fast):** Reduced resolution (320×320) + raised confidence threshold (0.5 → 0.7), reducing NMS work. Latency: 11ms. Misses small/distant objects but detects all nearby obstacles.

**Rung 3 (Emergency):** Switch to a pre-compiled lightweight model (YOLOv8-N, 6.3 GFLOPs). Latency: 6ms. Lowest accuracy but guaranteed to meet any deadline.

The controller monitors per-frame inference time with an exponential moving average. When the EMA exceeds 28ms (85% of budget), it steps down one rung. When the EMA drops below 20ms for 10 consecutive frames, it steps back up. The key: every rung must be pre-validated for safety — you must prove that Rung 2's reduced accuracy still detects all obstacles within the braking distance at the current speed.

> **Napkin Math:** Dense intersection: 50 objects → NMS takes 8ms instead of 2ms. Rung 0: 25 + 6 = 31ms → misses deadline. Step to Rung 1: 17 + 4 = 21ms → safe. If NMS still spikes: Rung 2: 11 + 2 = 13ms → ample headroom. Total frames delivered over 10 seconds at Rung 1: 300 (vs 220 if we stayed at Rung 0 and dropped frames).

**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The WCET Analysis</b> · <code>latency</code></summary>

**Interviewer:** "You're certifying a perception pipeline for an autonomous vehicle under ISO 26262 ASIL-B. Industry practice requires end-to-end inference latency under 50ms for safety-critical decisions, and your safety case must *guarantee* the pipeline completes within 100ms under all operating conditions (including the full sensor-to-actuator path). Your average-case inference latency is 45ms. How do you construct the worst-case execution time (WCET) argument?"

**Common Mistake:** "Our P99 latency is 80ms, which is under 100ms." P99 means 1 in 100 frames can exceed 80ms — at 30 FPS, that's a missed deadline every 3.3 seconds. Safety certification requires *guarantees*, not statistics.

**Realistic Solution:** WCET analysis for safety-critical systems requires eliminating all sources of non-determinism:

(1) **No dynamic memory allocation** — all buffers pre-allocated at init. `malloc` during inference can trigger page faults with unbounded latency.

(2) **Dedicated GPU partition** — use the Orin's MIG (Multi-Instance GPU) or DLA to isolate your workload. Other processes sharing the GPU can preempt your kernels with unbounded delay.

(3) **Measured WCET with margin** — run the pipeline on 100,000+ worst-case inputs (maximum object count, lowest visibility, highest resolution). Take the observed maximum and multiply by 1.5× (the safety margin accounts for untested corner cases). If measured worst case is 65ms: WCET claim = 97.5ms.

(4) **Watchdog timer** — a hardware watchdog triggers at 95ms. If the pipeline hasn't produced output, the system switches to the fallback path: a rule-based emergency controller that uses raw ultrasonic/radar data to brake. The neural network is advisory; the fallback is the safety guarantee.

(5) **Thermal derating** — WCET must be measured at the worst-case operating temperature (e.g., 80°C junction on Orin), where DVFS has already throttled the SoC to its lowest P-state. Use the vendor's thermal design guide (NVIDIA publishes detailed thermal resistance data) to determine the junction temperature at your worst-case ambient.

> **Napkin Math:** Average: 45ms. P99: 80ms. Measured worst case (100K frames, 85°C): 65ms. WCET claim = 65 × 1.5 = 97.5ms < 100ms budget ✓. Watchdog at 95ms gives 5ms for fallback activation. Fallback (ultrasonic braking): deterministic, <1ms response. Total system WCET: 100ms guaranteed.

**📖 Deep Dive:** [Volume II: Robust AI](https://mlsysbook.ai/vol2/robust_ai.html)
</details>

---

### ⚡ Power & Thermal

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Field Thermal Surprise</b> · <code>thermal</code></summary>

**Interviewer:** "Your edge AI box runs perfectly in the lab — 30 FPS, no issues. You deploy it in a sealed IP67 enclosure on a factory floor. Within 20 minutes, FPS drops to 18. The hardware is identical. What happened?"

**Common Mistake:** "The device is defective" or "The factory has electromagnetic interference." Neither explains a gradual performance degradation.

**Realistic Solution:** The lab is air-conditioned at 22°C. The factory floor is 35°C, and the sealed IP67 enclosure traps heat — internal temperature reaches 50°C+. Your device's thermal solution (small heatsink + fan) was designed assuming 25°C ambient with free airflow. In the sealed enclosure, there's no airflow, and the ambient temperature is 15-25°C higher than the design point. The SoC hits its thermal throttling threshold (typically 80°C on Jetson Orin, per NVIDIA's thermal design guide) much faster, triggering DVFS to a lower P-state. Fix: (1) derate the power budget for worst-case ambient — design for 50°C, not 25°C, (2) use a fanless design with a large aluminum heatsink that conducts heat to the enclosure walls, (3) add thermal interface material (TIM) between the SoC and the enclosure, turning the entire enclosure into a heat sink, (4) reduce the power mode from the start (e.g., 15W instead of 25W) to stay below the thermal ceiling indefinitely.

> **Napkin Math:** Thermal resistance of small heatsink: ~5°C/W. At 25W TDP: ΔT = 125°C. Junction temp = 22°C (lab) + 125°C = 147°C → throttles immediately to ~15W → ΔT = 75°C → 97°C junction → stable at ~20 FPS. In field (50°C ambient): 50 + 75 = 125°C → still throttles. Need larger heatsink (~2°C/W): 50 + 50 = 100°C → borderline. At 15W mode: 50 + 30 = 80°C → stable, no throttling.

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Duty Cycling Power Budget</b> · <code>thermal</code></summary>

**Interviewer:** "Your edge device has a 30W steady-state thermal budget, but your perception model requires 45W to run at full speed (30 FPS). You can't upgrade the cooling. How do you run a 45W workload on a 30W thermal budget?"

**Common Mistake:** "Reduce the model size until it fits in 30W." This sacrifices accuracy. There's a way to keep the full model.

**Realistic Solution:** Duty cycling. Run inference at full power (45W) for a burst period, then idle (5W) to let the thermal mass absorb and dissipate the heat. The average power must stay at or below 30W. The math: if you run for $t_{on}$ seconds at 45W and idle for $t_{off}$ seconds at 5W, the average power is $(45 \times t_{on} + 5 \times t_{off}) / (t_{on} + t_{off}) \leq 30$. Solving: $t_{on}/t_{off} \leq 25/15 = 5/3$. So for every 5 seconds of inference, you idle for 3 seconds. Effective duty cycle: 62.5%. If the full model runs at 30 FPS, your effective rate is 30 × 0.625 = **18.75 FPS average**. The trade-off is explicit: thermal budget directly constrains sustained throughput. During the off period, the system can use the last detection result or run a lightweight tracker to interpolate.

> **Napkin Math:** 45W on, 5W off. Target: ≤30W average. $t_{on} = 5s$, $t_{off} = 3s$. Average = (45×5 + 5×3) / 8 = 240/8 = **30W** ✓. Effective FPS = 30 × (5/8) = 18.75 FPS. With 2s on / 1.5s off: (45×2 + 5×1.5) / 3.5 = 97.5/3.5 = **27.9W** ✓. Effective FPS = 30 × (2/3.5) = 17.1 FPS.

> **Key Equation:** $P_{\text{avg}} = \frac{P_{\text{active}} \times t_{\text{on}} + P_{\text{idle}} \times t_{\text{off}}}{t_{\text{on}} + t_{\text{off}}} \leq P_{\text{thermal}}$

**📖 Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)
</details>

---

### 🔧 Model Optimization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Optimization Ladder</b> · <code>optimization</code></summary>

**Interviewer:** "Your YOLOv8-S runs at 15 FPS on a Jetson Orin NX. You need 30 FPS. Your team immediately starts designing a custom smaller architecture. Why is this the wrong first step?"

**Common Mistake:** "We need a smaller model — let's try YOLOv8-N or train a custom architecture." Architecture changes are the most expensive optimization and should be the last resort.

**Realistic Solution:** Follow the optimization ladder — a prioritized sequence from cheapest to most expensive:

**Step 1: TensorRT with FP16** (effort: minutes). Export your PyTorch model to ONNX, compile with TensorRT. Layer fusion, kernel auto-tuning, and FP16 Tensor Core utilization typically give 1.5-2× speedup for free. 15 FPS → 22-30 FPS.

**Step 2: INT8 quantization** (effort: hours). Calibrate with 1000 representative images. Another 1.5-2× on top of FP16. 22 FPS → 33-44 FPS.

**Step 3: Input resolution reduction** (effort: minutes). Drop from 640×640 to 512×512 or 480×480. FLOPs scale quadratically with resolution: (480/640)² = 0.56× FLOPs. Accuracy drops ~1-2% mAP.

**Step 4: Structured pruning** (effort: days). Remove channels with lowest L1-norm. 20-40% channel reduction for 1-2% mAP loss. Requires fine-tuning.

**Step 5: Architecture change** (effort: weeks-months). Only if steps 1-4 are insufficient. Design or adopt a smaller architecture (YOLOv8-N, EfficientDet-Lite).

Most teams jump straight to Step 5, leaving 3-4× of free performance on the table from Steps 1-2.

> **Napkin Math:** Baseline (PyTorch FP32): 15 FPS. After Step 1 (TensorRT FP16): 15 × 1.8 = 27 FPS. After Step 2 (INT8): 27 × 1.7 = 46 FPS. Already 1.5× over target — no need for Steps 3-5. Engineering time: 2 hours vs 2 months for a custom architecture.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Pruning Paradox</b> · <code>optimization</code></summary>

**Interviewer:** "You structured-pruned 40% of channels from your detection model. FLOPs dropped 40%. But when you benchmark on the Hailo-8, latency only dropped 10%. On the Jetson Orin, it dropped 35%. Why does the same pruning give wildly different speedups on different hardware?"

**Common Mistake:** "The Hailo's compiler isn't optimized for pruned models." The compiler is fine — the issue is architectural.

**Realistic Solution:** Edge accelerators have fundamentally different execution models. The **Jetson Orin (GPU)** executes layers as CUDA kernels with configurable thread blocks. Fewer channels = fewer threads = proportional speedup (with some overhead). The **Hailo-8 (dataflow)** maps the model onto a fixed spatial pipeline at compile time. Each layer is assigned physical compute units based on its original shape. When you prune a layer from 64 to 38 channels, the Hailo's compiler must still allocate compute units in multiples of its native SIMD width (typically 8 or 16). A 38-channel layer executes as if it had 48 channels (rounded up to the next multiple of 16), wasting 21% of the allocated compute. Across many layers, these rounding penalties accumulate, eroding the theoretical 40% FLOP reduction to a 10% latency reduction. Fix: use **hardware-aware pruning** that constrains channel counts to multiples of the target hardware's native width. Prune from 64 to 32 channels (50% reduction, but hardware-aligned) instead of 64 to 38 (41% reduction, but misaligned).

> **Napkin Math:** 10 layers, each pruned from 64 to 38 channels. Hailo SIMD width = 16. Effective channels per layer: 48 (rounded up). Effective FLOP reduction: (64-48)/64 = 25%, not 40%. With hardware-aware pruning to 32 channels: effective reduction = 50%, and every channel is utilized. Latency improvement: Hailo ~45% (vs 10% naive), Orin ~48% (vs 35% naive).

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 🚀 Deployment

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The A/B Partition Scheme</b> · <code>deployment</code></summary>

**Interviewer:** "You manage 10,000 edge cameras deployed across a city. You need to update the detection model on all of them. Your colleague says 'just SSH in and copy the new model file.' What are the four ways this can catastrophically fail, and what is the correct deployment architecture?"

**Common Mistake:** "OTA updates are just file copies — what could go wrong?" Everything.

**Realistic Solution:** Four failure modes of naive OTA: (1) **Power loss during write** — the camera loses power mid-copy. The old model is partially overwritten, the new model is incomplete. The device boots to a corrupted model and is bricked. No one can physically access a camera on a pole 30 feet up. (2) **Storage exhaustion** — the new model must coexist with the old model during the copy. If storage is 80% full with video buffer, there's no room for both. (3) **Runtime incompatibility** — the new model was compiled for TensorRT 8.6 but the device runs 8.5. Inference crashes or produces garbage. (4) **Fleet-wide failure** — pushing to all 10,000 devices simultaneously means a bad model bricks the entire fleet at once.

The correct architecture is **A/B partitioning**: the device has two model slots (A and B). The running system uses slot A. The new model is written to slot B while A continues serving. After the write completes, the device runs a validation check — inference on a test image with a known-good output hash. Only if validation passes does the bootloader atomically switch the active pointer to B. If validation fails, or if the device fails to boot from B (watchdog timeout), it automatically reverts to A. Roll out in waves: 1% → 10% → 50% → 100%, with automatic rollback if >5% of any wave reports failure.

> **Napkin Math:** Model: 12 MB. A/B slots: 24 MB reserved. OTA over LTE (5 Mbps): 12 MB × 8 / 5 Mbps = 19.2 seconds per device. Wave 1 (100 devices): 20s transfer + 10 min validation. Wave 2 (900): 20s + 10 min. Wave 3 (4000): 20s + 10 min. Wave 4 (5000): 20s. Total safe rollout: ~40 minutes. Naive push to all 10,000: 20 seconds — but one bad model = 10,000 bricked cameras.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

---

### 🔒 Security

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Adversarial Patch Attack</b> · <code>security</code></summary>

**Interviewer:** "Your autonomous vehicle's camera-based detection system correctly identifies stop signs 99.9% of the time. A security researcher demonstrates that a carefully designed sticker (an adversarial patch) placed on a stop sign causes your model to classify it as a speed limit sign with 95% confidence. Your model is state-of-the-art. How do you defend against this?"

**Common Mistake:** "Retrain the model with adversarial examples" or "Add input preprocessing to detect patches." Adversarial training helps but is an arms race — new patches can always be designed. Input preprocessing is easily circumvented.

**Realistic Solution:** Defense in depth — no single layer is sufficient:

(1) **Multi-sensor fusion** — LiDAR and radar see a physical object at the stop sign's location regardless of the visual patch. If camera says "speed limit" but LiDAR says "vertical planar object at expected stop sign height," the fusion layer flags a conflict.

(2) **Temporal consistency** — a real stop sign doesn't change classification frame-to-frame. If the sign is "stop" for 28 frames, "speed limit" for 2 frames, then "stop" again, the temporal filter rejects the transient misclassification.

(3) **HD map priors** — the map database says there's a stop sign at this GPS coordinate. If the model disagrees with the map, trust the map for safety-critical decisions and flag the discrepancy for review.

(4) **Ensemble disagreement** — run two architecturally different models (e.g., CNN and ViT). Adversarial patches are typically crafted for a specific architecture. If the two models disagree, escalate to the safety system.

(5) **Behavioral safety** — regardless of classification, if the vehicle is approaching an intersection, reduce speed. The sign classification informs behavior but doesn't override geometric safety rules.

> **Napkin Math:** Single-model vulnerability: 1 adversarial patch defeats 1 model. With 2 independent models: attacker must defeat both simultaneously — success rate drops from ~95% to ~5% (assuming independent failure). With map prior: attacker must also spoof GPS or compromise the map database. With temporal filter (majority vote over 30 frames): attacker must sustain misclassification for >15 consecutive frames — much harder with a physical patch that only works at specific viewing angles.

**📖 Deep Dive:** [Volume II: Robust AI](https://mlsysbook.ai/vol2/robust_ai.html)
</details>

# Round 2: Constraints & Trade-offs ⚖️

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Edge_Systems.md">🤖 Round 1</a> ·
  <a href="02_Edge_Constraints.md">⚖️ Round 2</a> ·
  <a href="03_Edge_Ops_and_Deployment.md">🚀 Round 3</a> ·
  <a href="04_Edge_Visual_Debugging.md">🖼️ Round 4</a> ·
  <a href="05_Edge_Advanced.md">🔬 Round 5</a>
</div>

---

Every edge system is a negotiation between competing constraints: memory budgets, thermal envelopes, quantization accuracy, architecture choices, latency deadlines, and power caps. This round tests whether you can reason about these trade-offs quantitatively — not just name them, but calculate the consequences.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/edge/02_Edge_Constraints.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 📐 Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Bandwidth-Bound Orin</b> · <code>roofline</code></summary>

- **Interviewer:** "Your Jetson Orin NX profiler reports 70 TOPS out of a rated 100 TOPS INT8. The team celebrates — 70% utilization sounds great. But your YOLOv8-M model still runs at 15 FPS instead of the expected 45 FPS. What is the profiler actually telling you?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We're at 70% compute utilization, so the remaining 30% is overhead we can optimize away." This confuses which resource is being utilized.

  **Realistic Solution:** The 70% figure is memory bandwidth utilization, not compute utilization. The Orin NX has 102.4 GB/s LPDDR5 bandwidth and a ridge point of ~976 Ops/Byte (100 TOPS / 102.4 GB/s). YOLOv8-M has an arithmetic intensity of roughly 200 Ops/Byte — well below the ridge. The model is memory-bandwidth bound: the GPU's INT8 cores are starved, waiting for data from DRAM. Buying a faster accelerator won't help. You need to reduce memory traffic: fuse layers to keep activations in on-chip SRAM, use depth-wise convolutions that have higher arithmetic intensity, or reduce input resolution.

  > **Napkin Math:** YOLOv8-M: ~39 GFLOPs, ~80 MB memory traffic per inference. Arithmetic intensity = 39G / 80M ≈ 488 Ops/Byte. Bandwidth ceiling = 102.4 GB/s × 488 = 50 TOPS attainable. At 70% bandwidth utilization: 0.7 × 50 = 35 TOPS effective. Per-frame time = 39G / 35T = 1.1ms compute, but memory stalls stretch it to ~22ms → 45 FPS theoretical, 15 FPS actual due to activation spills and strided access patterns.

  > **Key Equation:** $\text{Attainable TOPS} = \min(\text{Peak TOPS},\ \text{BW} \times \text{Arithmetic Intensity})$

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Dataflow vs GPU Roofline</b> · <code>roofline</code></summary>

- **Interviewer:** "Your team is choosing between a Hailo-8 (26 TOPS, 2.5W, dataflow architecture) and a Jetson Orin NX (100 TOPS, 15W, GPU architecture) for a drone running YOLOv8-S. The Orin has 4× the TOPS. But in your benchmarks, the Hailo runs the model at 28 FPS while the Orin only hits 35 FPS. Why is the 4× TOPS advantage only yielding a 1.25× speedup?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The Orin's drivers aren't optimized yet" or "The Hailo benchmark is wrong." Both dodge the architectural explanation.

  **Realistic Solution:** The Hailo-8 is a dataflow architecture — it maps the entire model graph onto a spatial pipeline of physical compute units. Activations flow between stages through on-chip buffers without ever touching external DRAM. The Orin NX is a GPU — it executes layers sequentially, reading weights and activations from LPDDR5 between each layer. For YOLOv8-S (arithmetic intensity ~516 Ops/Byte), the Orin is memory-bandwidth bound: its 102.4 GB/s LPDDR5 limits effective throughput to ~50 TOPS. The Hailo eliminates the DRAM bottleneck entirely, so its 26 TOPS are nearly fully utilized. Effective throughput: Hailo ~24 TOPS (92%) vs Orin ~35 TOPS (35%). The dataflow architecture changes the shape of the roofline itself — there is no memory wall.

  > **Napkin Math:** Hailo-8: 26 TOPS peak, ~92% utilization (no DRAM stalls) = 24 TOPS effective. Per-frame: 28.4 GFLOPs / 24 TOPS = 1.18ms → ~28 FPS with overhead. Orin NX: 100 TOPS peak, bandwidth-limited to ~50 TOPS, 35% utilization = 35 TOPS effective. Per-frame: 28.4G / 35T = 0.81ms compute + ~28ms memory stalls → ~35 FPS. Power: Hailo at 2.5W = 11.2 FPS/W. Orin at 15W = 2.3 FPS/W. The Hailo is **4.9× more power-efficient** for this workload.

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The DLA vs GPU Partition</b> · <code>roofline</code></summary>

- **Interviewer:** "The Jetson Orin has both a GPU and two DLA (Deep Learning Accelerator) engines. Your colleague runs the entire YOLOv8-S model on the GPU and gets 35 FPS. You suggest splitting the model: backbone on the DLA, detection head on the GPU. Your colleague says 'that's more complex for no benefit — the GPU is faster.' Is your colleague right?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The GPU is always faster because it has more TOPS." This ignores that the DLA and GPU can run in parallel.

  **Realistic Solution:** Your colleague is right that the GPU alone is faster for a *single model*. But the DLA and GPU can execute concurrently — they are independent compute engines, though they share the same LPDDR5 memory bus and MMU. If you run the backbone on the DLA (15ms) while the GPU runs a tracking model (8ms), both execute in parallel. The detection head then runs on the GPU (5ms) after the DLA finishes. The caveat: because DLA and GPU share memory bandwidth, running both simultaneously increases individual latencies by 10-20% due to DRAM contention. Adjusted: DLA backbone ~17ms (parallel with GPU tracker ~9ms) + GPU head 5ms = 22ms. Still much better than sequential GPU: 28 + 8 = 36ms. The DLA/GPU split gives **~1.6× higher pipeline throughput** by exploiting hardware parallelism, even accounting for bandwidth contention. The benefit is largest when at least one workload is compute-bound rather than memory-bound.

  > **Napkin Math:** GPU-only pipeline: detection 28ms + tracking 8ms = 36ms → 27 FPS. DLA+GPU pipeline (with ~15% bandwidth contention): DLA backbone 17ms (parallel with GPU tracker 9ms) + GPU head 5ms = 22ms → 45 FPS. Speedup: 1.6×. Power: DLA at 5W + GPU at 15W = 20W vs GPU-only at 15W for 36ms. Energy per frame: DLA+GPU = 20W × 22ms = 0.44J. GPU-only = 15W × 36ms = 0.54J. The split is **19% more energy-efficient**.

  > **Key Equation:** $t_{\text{pipeline}} = \max(t_{\text{DLA}} \times (1 + \text{contention}), t_{\text{GPU\_parallel}} \times (1 + \text{contention})) + t_{\text{GPU\_sequential}}$

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)

  </details>

</details>

---

### 🧠 Memory Systems

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Shared DRAM Budget</b> · <code>memory</code></summary>

- **Interviewer:** "Your edge box has 8 GB of LPDDR5 DRAM. Your ML model needs 1.5 GB for weights and activations. Your colleague says 'we have 6.5 GB of headroom — plenty.' Why is this dangerously optimistic?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "8 GB minus 1.5 GB leaves 6.5 GB free." This treats the edge device like a dedicated ML accelerator with nothing else running.

  **Realistic Solution:** Unlike a data center GPU with dedicated HBM, edge DRAM is shared with everything: the Linux kernel and drivers (~500 MB), camera ISP and sensor pipelines (~1 GB for 4K video), display compositor (~300 MB), networking stack (~200 MB), system services and logging (~500 MB). Realistic free memory: ~5 GB. But that's the static picture. Under load, the camera ISP can burst to 2 GB for multi-frame HDR processing, and the Linux page cache will aggressively claim free memory. Your ML process can be OOM-killed at any time if the kernel needs memory for a higher-priority subsystem. You must use `mlockall()` to pin your model's pages in RAM, set cgroup memory limits to protect your allocation, and budget for worst-case concurrent memory usage — not average.

  > **Napkin Math:** 8 GB total. Kernel + drivers: 500 MB. Camera ISP (4K, burst): 2 GB. Display: 300 MB. Network + services: 700 MB. Available for ML: 8 - 3.5 = **4.5 GB worst case**. Your 1.5 GB model fits, but only with 3 GB headroom — not 6.5 GB. If another process spikes, you're at risk.

  📖 **Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The eMMC Cold Start</b> · <code>memory</code></summary>

- **Interviewer:** "Your edge device loads a 200 MB model from eMMC flash into DRAM at boot. The eMMC spec says 300 MB/s sequential read, so you expect a 0.67-second load time. In practice, first inference takes over 3 seconds. Where did the other 2.3 seconds go?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The eMMC is slower than spec." The raw bandwidth is fine — the overhead is elsewhere.

  **Realistic Solution:** The 300 MB/s spec is for large sequential reads. Model loading involves: (1) filesystem overhead — ext4 metadata lookups, inode traversal, and block allocation table reads add ~200ms, (2) the model file is fragmented on a well-used eMMC — random 4K reads drop to ~20 MB/s, adding ~500ms, (3) TensorRT engine deserialization — parsing the serialized engine, allocating CUDA memory, and building the execution context takes ~1.5s for a 200 MB engine, (4) CUDA context initialization on first use adds ~300ms. The fix: (a) store the model on a dedicated, defragmented partition, (b) use `mmap()` to map the weight file so pages load on demand during the first inference rather than all at once, (c) keep a lightweight fallback model (~20 MB, loads in <200ms) that serves immediately while the full model loads in a background thread.

  > **Napkin Math:** Sequential read: 200 MB / 300 MB/s = 0.67s. Filesystem overhead: +0.2s. Fragmentation penalty: +0.5s. TensorRT deserialization: +1.5s. CUDA init: +0.3s. Total: **3.17s**. With mmap + background load: fallback model ready in 0.2s, full model ready in ~3s but user never waits.

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Multi-Model Memory Sharing</b> · <code>memory</code></summary>

- **Interviewer:** "Your autonomous vehicle runs four models concurrently: detection (YOLOv8-L, 80 MB weights), tracking (DeepSORT, 30 MB), depth estimation (MiDaS, 200 MB), and path planning (custom, 50 MB). Total: 360 MB of weights plus ~400 MB of activation buffers. Your Jetson AGX Orin has 32 GB DRAM, but after the OS and sensor pipelines, only 4 GB is available for ML. How do you fit 760 MB of ML workload into 4 GB with room for growth?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "4 GB is way more than 760 MB — there's no problem." This ignores peak memory, concurrent allocations, and fragmentation.

  **Realistic Solution:** The 760 MB is the *minimum* — it ignores peak activation memory during concurrent execution. When detection and depth estimation run simultaneously, their activation buffers peak at ~600 MB combined (not 400 MB, because peaks overlap). Plus TensorRT workspace memory (~200 MB per engine). Real peak: ~1.2 GB. The strategy: (1) **Shared tensor allocator** — use a single CUDA memory pool (like TensorRT's `IGpuAllocator`) that reuses activation buffers between sequential stages. Detection's output buffer becomes tracking's input buffer without a copy. (2) **Temporal multiplexing** — depth estimation doesn't need to run every frame. Run it at 10 FPS (every 3rd frame) and reuse its memory during off-frames for other models. (3) **Backbone sharing** — if detection and depth share a ResNet backbone, load the backbone weights once and run both heads on the shared features. This saves 60+ MB of duplicated weights. (4) **Memory-mapped weights** — mmap the planning model's weights so they're paged in only when the planning module runs (every 100ms), freeing physical RAM between invocations.

  > **Napkin Math:** Naive: 360 MB weights + 600 MB peak activations + 400 MB TensorRT workspace = 1.36 GB. With shared allocator: activations drop to ~350 MB (sequential reuse). With backbone sharing: weights drop to ~280 MB. With temporal multiplexing: peak concurrent memory = ~800 MB. Headroom in 4 GB: 3.2 GB free for sensor buffers and growth.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)

  </details>

</details>

---

### 🔢 Numerical Representation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The PTQ vs QAT Question</b> · <code>quantization</code></summary>

- **Interviewer:** "Your manager says 'just quantize the model to INT8 — it takes five minutes with TensorRT.' You push back and say you need two weeks for quantization-aware training. Justify the two weeks."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "QAT is always better than PTQ." This is too vague — you need to explain *when* PTQ fails and *why* QAT fixes it.

  **Realistic Solution:** Post-training quantization (PTQ) collects activation statistics from a small calibration dataset (~1000 images) and sets quantization ranges based on observed min/max values. It works well when: activation distributions are symmetric and well-behaved, the model is large (redundancy absorbs quantization noise), and the task is tolerant of small errors (classification). PTQ *fails* when: (1) the model has outlier channels — a few activations with 10× the magnitude of others force the quantization range to be wide, crushing precision for the majority, (2) the task is precision-sensitive — depth estimation, segmentation boundaries, or small-object detection where a 1% pixel shift matters, (3) the model is already small — a MobileNet has less redundancy to absorb quantization error than a ResNet-50. QAT inserts fake-quantization nodes during training, so the model's gradients learn to compensate for quantization error. The two weeks cover: fine-tuning with quantization noise (~3 days), hyperparameter search for learning rate and quantization scheme (~5 days), validation across the full test distribution including edge cases (~4 days).

  > **Napkin Math:** PTQ: 1000 calibration images × 5ms each = 5 seconds + TensorRT build = 5 minutes total. Accuracy: typically -0.5% to -3% for classification, -2% to -15% for detection on hard classes. QAT: 50 epochs × 10,000 images × 20ms/image = ~2.8 hours training + validation = 2 weeks of engineering time. Accuracy: typically -0.1% to -0.5% — recovering most of the PTQ loss.

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Night Scene Calibration Failure</b> · <code>quantization</code></summary>

- **Interviewer:** "You INT8-quantized your pedestrian detection model using PTQ. Daytime accuracy drops 2% — acceptable. But your test suite reveals nighttime recall drops 15%. The safety team blocks deployment. Your calibration dataset had 1000 images. What went wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "INT8 is too aggressive for night scenes — use FP16 at night." This doubles your compute cost and doesn't explain the root cause.

  **Realistic Solution:** Your 1000 calibration images were predominantly daytime scenes (reflecting the training distribution). PTQ set the quantization ranges based on daytime activation statistics — bright, high-contrast features with large activation magnitudes. Nighttime pedestrians produce activations in the low-magnitude tail of the distribution. With per-tensor quantization, the step size is set by the max activation value (dominated by daytime). Nighttime activations, being 10-50× smaller, get crushed into just a few quantization bins, destroying the signal. Fixes: (1) curate a calibration dataset that represents the full operating domain — 50% day, 30% night, 20% twilight/rain, (2) switch from per-tensor to per-channel quantization, which sets independent ranges per output channel and is more robust to distribution variation, (3) use percentile calibration (99.99th percentile instead of min/max) to clip outliers and give more precision to the common range.

  > **Napkin Math:** Per-tensor INT8 range set by daytime max activation = 10.0. Step size = 10.0 / 127 = 0.079. Nighttime pedestrian activations in range [0.01, 0.05]: only (0.05 - 0.01) / 0.079 = **0.5 bins** — the entire signal collapses to a single quantized value. Per-channel quantization: if the relevant channel's max = 0.1, step size = 0.1/127 = 0.0008 → nighttime range spans 50 bins. Signal preserved.

  > **Key Equation:** $\text{Step Size} = \frac{x_{\max} - x_{\min}}{2^b - 1}$

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Mixed-Precision Perception Stack</b> · <code>quantization</code></summary>

- **Interviewer:** "You're architecting the inference pipeline for an autonomous vehicle with a Jetson AGX Orin. The stack has three models: object detection, monocular depth estimation, and motion planning. Design the precision strategy for each model and justify your choices from first principles."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run everything in INT8 for maximum speed" or "Run everything in FP32 for maximum safety." Both ignore the error-tolerance profile of each component.

  **Realistic Solution:** Each model has a different error-tolerance profile that dictates its precision:

  **Detection (INT8):** Object detection is inherently robust to small numerical perturbations — a bounding box shifted by 1 pixel or a confidence score off by 0.01 doesn't change the detection outcome. INT8 quantization typically costs <1% mAP on well-calibrated models. The 2× speedup and 4× memory reduction over FP32 are critical for meeting the 33ms frame budget.

  **Depth Estimation (FP16):** Monocular depth maps are sensitive to numerical precision because small activation differences map to large depth differences at range. A 1% error in a feature map can translate to a 2-meter depth error at 100m — the difference between "safe to proceed" and "collision imminent." FP16 preserves sufficient precision while still running on Tensor Cores (2× speedup over FP32).

  **Motion Planning (FP32):** The planning module must be bit-exact with the simulation environment used for safety validation. Any numerical divergence between the deployed model and the simulator invalidates the safety case. FP32 guarantees reproducibility. The planning model is small (~50 MB) and runs at 10 Hz, so the compute cost is negligible.

  > **Napkin Math:** Detection INT8: 28 GFLOPs / 100 TOPS = 0.28ms. Depth FP16: 40 GFLOPs / 137.5 TFLOPS FP16 = 0.29ms. Planning FP32: 2 GFLOPs / 68.75 TFLOPS FP32 = 0.03ms. Total compute: 0.6ms. Memory: detection 11 MB (INT8) + depth 80 MB (FP16) + planning 200 MB (FP32) = 291 MB. All fit comfortably in the Orin's 32/64 GB DRAM with the pipeline completing well within the 33ms budget even with memory stalls.

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

---

### 🏗️ Architecture → System Cost

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The YOLO vs ViT Question</b> · <code>architecture</code></summary>

- **Interviewer:** "A researcher on your team wants to replace your YOLOv8-S detector with a ViT-B/16 vision transformer because it scores 2% higher mAP on COCO. You're deploying on a Jetson Orin NX at 30 FPS. Why do you push back?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "ViT has more parameters, so it's slower." ViT-B actually has fewer FLOPs than some YOLO variants — the issue is deeper than parameter count.

  **Realistic Solution:** ViT-B/16 has ~17.6 GFLOPs vs YOLOv8-S's ~28.4 GFLOPs — fewer FLOPs, yet it runs *slower* on edge GPUs. The reason: attention's memory access pattern. Self-attention computes Q×K^T (an N×N matrix for N=196 patches at 224×224), which has low arithmetic intensity — it's a series of small matrix multiplies with large intermediate tensors that must be written to and read from DRAM. Convolutional layers in YOLO have regular, predictable memory access patterns that map efficiently to GPU SRAM tiling and TensorRT optimization. Additionally, ViT's dynamic shapes (variable sequence length) prevent many TensorRT optimizations (layer fusion, kernel auto-tuning) that assume fixed tensor dimensions. On the Orin NX: YOLOv8-S runs at ~45 FPS with TensorRT INT8. ViT-B runs at ~15 FPS because TensorRT can't fuse the attention layers effectively.

  > **Napkin Math:** YOLOv8-S: 28.4 GFLOPs, ~55 MB memory traffic → arithmetic intensity ≈ 516 Ops/Byte. ViT-B: 17.6 GFLOPs, ~120 MB memory traffic (attention matrices) → arithmetic intensity ≈ 147 Ops/Byte. On Orin NX (ridge ~976 Ops/Byte): both are memory-bound, but ViT is 3.5× more memory-hungry per FLOP. The 2% mAP gain costs a 3× FPS penalty.

  📖 **Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Tracker Addition Budget</b> · <code>architecture</code></summary>

- **Interviewer:** "Your perception stack runs YOLOv8-S detection at 20ms per frame on a Jetson Orin NX, leaving 13ms of headroom in your 33ms budget. Your team wants to add a Transformer-based tracker (ByteTrack with a ReID model, ~15 GFLOPs). The ReID model alone takes 12ms. Your colleague says '20 + 12 = 32ms — we fit with 1ms to spare.' Why is this estimate dangerously wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The math works: 32ms < 33ms." This assumes zero overhead and perfect sequential execution.

  **Realistic Solution:** The 20ms + 12ms = 32ms estimate ignores: (1) **GPU memory contention** — both models compete for the same LPDDR5 bandwidth. When running concurrently, memory bandwidth is split, increasing both models' latency by 15-30%. (2) **CUDA context switching** — swapping between two TensorRT engines incurs ~1-2ms of overhead per switch. (3) **NMS and post-processing** — detection NMS takes 2-4ms on dense scenes (not included in the 20ms). (4) **Data transfer** — copying detection crops to the ReID model's input tensor takes ~0.5ms. Realistic total: 20ms (detect) + 3ms (NMS) + 0.5ms (copy) + 12ms (ReID) + 2ms (context switch) + 15% bandwidth penalty = **~43ms** → 23 FPS, missing the 30 FPS deadline. Fix: use a lightweight tracker (DeepSORT with a MobileNet ReID: 2 GFLOPs, ~3ms) or run the ReID model at half rate (every other frame, using motion prediction to interpolate).

  > **Napkin Math:** Optimistic: 20 + 12 = 32ms. Realistic: 20 + 3 (NMS) + 0.5 (copy) + 14 (ReID with BW contention) + 2 (context switch) = 39.5ms → 25 FPS. With lightweight ReID (3ms): 20 + 3 + 0.5 + 3 + 1 = 27.5ms → 36 FPS ✓. Half-rate heavy ReID: alternating 20ms and 39.5ms frames → average 29.75ms → 33 FPS ✓.

  📖 **Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Resolution-Accuracy Pareto</b> · <code>architecture</code></summary>

- **Interviewer:** "Your edge camera system needs to detect people at distances from 5m to 100m. Your 4K camera (3840×2160) feeds a YOLOv8-M detector, but running at full 4K resolution takes 85ms — far over your 33ms budget. Your colleague says 'just resize to 640×640.' What critical information does resizing destroy, and how do you design a system that meets the deadline while preserving long-range detection?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Resize to 640×640 and accept the accuracy loss." This treats resolution as a single knob when it's actually a spatial information budget.

  **Realistic Solution:** Resizing 4K to 640×640 is a 6× downscale. A person at 100m occupies roughly 20 pixels tall in 4K. After resizing: 20/6 ≈ 3 pixels — below the minimum receptive field for any detector. You've made long-range detection physically impossible. The fix is a **multi-scale tiling strategy**: (1) run a lightweight classifier on the full 4K frame at low resolution to identify regions of interest (ROIs), (2) crop and run the full detector only on the ROIs at native resolution. For a scene with 3 ROIs: 3 × 640×640 crops × 20ms each = 60ms sequential, but you can batch them: 3 crops in one batch = ~25ms on TensorRT. Alternative: use a fixed tiling scheme — divide 4K into 6 overlapping 1280×720 tiles, run detection on each, merge with NMS. At 12ms per tile (smaller than full 4K): 6 × 12ms = 72ms sequential, but with 2-tile batching on the GPU: 3 batches × 15ms = 45ms. Still over budget. The real solution: run tiles on the DLA and GPU in parallel — 3 tiles on DLA (45ms) + 3 tiles on GPU (45ms) = 45ms total with full parallelism. Add a temporal skip: only re-tile every 3rd frame, using tracking to interpolate. Effective rate: 45ms / 3 = 15ms amortized → 66 FPS equivalent.

  > **Napkin Math:** Person at 100m in 4K: ~20 pixels tall. After 6× resize: ~3 pixels → undetectable. With tiling (6 × 1280×720): person stays at ~10 pixels in each tile → detectable. Compute: 6 tiles × 12ms = 72ms sequential. With DLA+GPU parallel: 36ms. With temporal skip (every 3rd frame): 12ms amortized. Accuracy: ~95% of full-4K detection at 1/7th the compute.

  📖 **Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)

  </details>

</details>

---

### 🧠 Edge LLM & Generative AI

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Edge LLM Memory Wall</b> · <code>memory</code> <code>kv-cache</code></summary>

- **Interviewer:** "You're deploying a 3B parameter small language model (Phi-3-mini) on a Jetson Orin NX with 8 GB DRAM. The model weights in INT4 are 1.5 GB. Your colleague says 'plenty of room — we have 6.5 GB free.' The first few conversation turns work fine, but at turn 15 the system crashes with an OOM error. What's consuming the memory?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model is leaking memory" or "Activations are too large." The weights are static and activations are small for a single-token decode step. The growing cost is something else.

  **Realistic Solution:** The KV-cache. During autoregressive generation, the model stores the key and value tensors for every token generated so far, across every layer. For Phi-3-mini (32 layers, 32 heads, head_dim=96): KV-cache per token = 2 (K+V) × 32 layers × 32 heads × 96 dim × 2 bytes (FP16) = 393 KB per token. At turn 15 of a conversation with ~2000 tokens of context: 2000 × 393 KB = **786 MB**. Add the model weights (1.5 GB), OS and sensor overhead (~3.5 GB on the 8 GB Orin NX), and you're at 5.8 GB — dangerously close to the 8 GB limit. By turn 20 (~3000 tokens): KV-cache = 1.15 GB, total = 6.2 GB. A spike in OS memory usage (camera ISP burst) pushes you over the edge.

  Fixes: (1) **KV-cache quantization** — quantize the KV-cache to INT4 (4× reduction): 786 MB → 196 MB. Accuracy impact is minimal for most conversational tasks. (2) **Sliding window attention** — only keep the last N tokens in the KV-cache (e.g., N=1024). Older context is summarized into a compressed representation. (3) **Context length limit** — hard-cap at 2048 tokens and force conversation summarization before hitting the limit. (4) **Memory-mapped KV-cache** — spill older KV entries to NVMe SSD and page them back on demand, trading latency for memory.

  > **Napkin Math:** Phi-3-mini KV-cache: 393 KB/token. At 2048 tokens: 786 MB. At 4096 tokens: 1.57 GB. With INT4 KV quantization: 196 MB at 2048 tokens. Memory budget on 8 GB Orin NX: 8 GB - 3.5 GB (OS) - 1.5 GB (weights) = 3 GB for KV-cache. Max tokens without quantization: 3 GB / 393 KB = ~7,800 tokens. With INT4 KV: ~31,000 tokens. KV quantization is mandatory for edge LLMs.

  > **Key Equation:** $\text{KV-cache (bytes)} = 2 \times L \times H \times d_h \times n_{\text{tokens}} \times \text{bytes per element}$

  📖 **Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)

  </details>

</details>

---

### 📷 Sensor & ISP Pipeline

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The First-Frame Latency</b> · <code>latency</code> <code>sensor-fusion</code></summary>

- **Interviewer:** "Your smart doorbell camera triggers on motion and runs person detection. The ML model takes 30ms. But users complain that the system misses the first 500ms of action — by the time the first detection fires, the person is already at the door. The model is fast enough. Where is the 470ms going?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model needs to be faster." At 30ms, the model is not the bottleneck. The delay is upstream.

  **Realistic Solution:** The 500ms delay is the **ISP (Image Signal Processor) convergence time**. When the camera wakes from low-power sleep mode, the ISP must:

  (1) **Sensor power-up and clock stabilization:** ~50ms for the CMOS sensor to stabilize its PLL and begin outputting valid pixel data.

  (2) **Auto-Exposure (AE) convergence:** The ISP adjusts exposure time and gain based on scene brightness. Starting from default settings, AE typically needs 3-5 frames to converge. At 30 FPS: 5 × 33ms = **165ms**. During convergence, frames are either too dark or too bright for reliable detection.

  (3) **Auto-White Balance (AWB) convergence:** Color correction needs 2-3 frames after AE stabilizes: ~100ms.

  (4) **Auto-Focus (AF):** If the lens has AF, it needs 100-200ms to find focus. Many edge cameras use fixed-focus to avoid this.

  (5) **First valid frame to ML pipeline:** After ISP convergence, the first usable frame enters the detection pipeline: 30ms inference + 10ms post-processing = 40ms.

  Total: 50 + 165 + 100 + 40 = **355ms** minimum. With AF: 555ms.

  Fixes: (1) **Circular buffer with always-on sensor** — keep the camera running at low resolution (320×240, minimal power) continuously. On motion trigger, the last 1 second of low-res frames is already in the buffer. Run detection on the buffered frames immediately while the ISP converges to full resolution. (2) **Pre-configured ISP** — store the last-known AE/AWB settings and apply them on wake-up. Convergence drops from 5 frames to 1-2 frames. (3) **Low-power motion detection** — use a separate low-power PIR sensor or always-on low-res camera for motion detection, keeping the main camera in a warm standby (ISP powered, sensor streaming but not processing) that reduces wake time to ~50ms.

  > **Napkin Math:** Full cold start: 50ms (sensor) + 165ms (AE) + 100ms (AWB) + 40ms (inference) = 355ms. With pre-configured ISP: 50ms + 33ms (1 frame AE) + 33ms (1 frame AWB) + 40ms = 156ms. With warm standby: 0ms (sensor) + 33ms (AE fine-tune) + 40ms (inference) = 73ms. With circular buffer: first detection available from buffered frame in 40ms, full-res detection in 156ms.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)

  </details>

</details>

---

### ⏱️ Latency & Throughput

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Adaptive Quality Ladder</b> · <code>latency</code></summary>

- **Interviewer:** "Your 30 FPS edge perception pipeline runs fine on open roads but drops to 22 FPS in dense urban intersections (many pedestrians, vehicles, signs). You can't miss frames — the safety system requires continuous perception. Design a system that maintains 30 FPS under all conditions without changing hardware."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just use a faster model." You can't swap models at runtime without a pre-planned strategy, and a single model can't be optimal for all scene complexities.

  **Realistic Solution:** Design an **adaptive quality ladder** — a pre-defined set of operating points with known latency, each trading quality for speed:

  **Rung 0 (Nominal):** Full resolution (640×640), full model, all post-processing. Latency: 25ms. Used when scene complexity is low.

  **Rung 1 (Medium):** Reduced resolution (480×480), reducing FLOPs by ~44%. Latency: 17ms. Accuracy drops ~2% mAP but still above safety threshold.

  **Rung 2 (Fast):** Reduced resolution (320×320) + raised confidence threshold (0.5 → 0.7), reducing NMS work. Latency: 11ms. Misses small/distant objects but detects all nearby obstacles.

  **Rung 3 (Emergency):** Switch to a pre-compiled lightweight model (YOLOv8-N, 6.3 GFLOPs). Latency: 6ms. Lowest accuracy but guaranteed to meet any deadline.

  The controller monitors per-frame inference time with an exponential moving average. When the EMA exceeds 28ms (85% of budget), it steps down one rung. When the EMA drops below 20ms for 10 consecutive frames, it steps back up. The key: every rung must be pre-validated for safety — you must prove that Rung 2's reduced accuracy still detects all obstacles within the braking distance at the current speed.

  > **Napkin Math:** Dense intersection: 50 objects → NMS takes 8ms instead of 2ms. Rung 0: 25 + 6 = 31ms → misses deadline. Step to Rung 1: 17 + 4 = 21ms → safe. If NMS still spikes: Rung 2: 11 + 2 = 13ms → ample headroom. Total frames delivered over 10 seconds at Rung 1: 300 (vs 220 if we stayed at Rung 0 and dropped frames).

  📖 **Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The WCET Analysis</b> · <code>latency</code></summary>

- **Interviewer:** "You're certifying a perception pipeline for an autonomous vehicle under ISO 26262 ASIL-B. Industry practice requires end-to-end inference latency under 50ms for safety-critical decisions, and your safety case must *guarantee* the pipeline completes within 100ms under all operating conditions (including the full sensor-to-actuator path). Your average-case inference latency is 45ms. How do you construct the worst-case execution time (WCET) argument?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Our P99 latency is 80ms, which is under 100ms." P99 means 1 in 100 frames can exceed 80ms — at 30 FPS, that's a missed deadline every 3.3 seconds. Safety certification requires *guarantees*, not statistics.

  **Realistic Solution:** WCET analysis for safety-critical systems requires eliminating all sources of non-determinism:

  (1) **No dynamic memory allocation** — all buffers pre-allocated at init. `malloc` during inference can trigger page faults with unbounded latency.

  (2) **Dedicated GPU partition** — use the Orin's MIG (Multi-Instance GPU) or DLA to isolate your workload. Other processes sharing the GPU can preempt your kernels with unbounded delay.

  (3) **Measured WCET with margin** — run the pipeline on 100,000+ worst-case inputs (maximum object count, lowest visibility, highest resolution). Take the observed maximum and multiply by 1.5× (the safety margin accounts for untested corner cases). If measured worst case is 65ms: WCET claim = 97.5ms.

  (4) **Watchdog timer** — a hardware watchdog triggers at 95ms. If the pipeline hasn't produced output, the system switches to the fallback path: a rule-based emergency controller that uses raw ultrasonic/radar data to brake. The neural network is advisory; the fallback is the safety guarantee.

  (5) **Thermal derating** — WCET must be measured at the worst-case operating temperature (e.g., 80°C junction on Orin), where DVFS has already throttled the SoC to its lowest P-state. Use the vendor's thermal design guide (NVIDIA publishes detailed thermal resistance data) to determine the junction temperature at your worst-case ambient.

  > **Napkin Math:** Average: 45ms. P99: 80ms. Measured worst case (100K frames, 85°C): 65ms. WCET claim = 65 × 1.5 = 97.5ms < 100ms budget ✓. Watchdog at 95ms gives 5ms for fallback activation. Fallback (ultrasonic braking): deterministic, <1ms response. Total system WCET: 100ms guaranteed.

  📖 **Deep Dive:** [Volume II: Robust AI](https://mlsysbook.ai/vol2/robust_ai.html)

  </details>

</details>

---

### ⚡ Power & Thermal

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Field Thermal Surprise</b> · <code>thermal</code></summary>

- **Interviewer:** "Your edge AI box runs perfectly in the lab — 30 FPS, no issues. You deploy it in a sealed IP67 enclosure on a factory floor. Within 20 minutes, FPS drops to 18. The hardware is identical. What happened?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The device is defective" or "The factory has electromagnetic interference." Neither explains a gradual performance degradation.

  **Realistic Solution:** The lab is air-conditioned at 22°C. The factory floor is 35°C, and the sealed IP67 enclosure traps heat — internal temperature reaches 50°C+. Your device's thermal solution (small heatsink + fan) was designed assuming 25°C ambient with free airflow. In the sealed enclosure, there's no airflow, and the ambient temperature is 15-25°C higher than the design point. The SoC hits its thermal throttling threshold (typically 80°C on Jetson Orin, per NVIDIA's thermal design guide) much faster, triggering DVFS to a lower P-state. Fix: (1) derate the power budget for worst-case ambient — design for 50°C, not 25°C, (2) use a fanless design with a large aluminum heatsink that conducts heat to the enclosure walls, (3) add thermal interface material (TIM) between the SoC and the enclosure, turning the entire enclosure into a heat sink, (4) reduce the power mode from the start (e.g., 15W instead of 25W) to stay below the thermal ceiling indefinitely.

  > **Napkin Math:** Thermal resistance of small heatsink: ~5°C/W. At 25W TDP: ΔT = 125°C. Junction temp = 22°C (lab) + 125°C = 147°C → throttles immediately to ~15W → ΔT = 75°C → 97°C junction → stable at ~20 FPS. In field (50°C ambient): 50 + 75 = 125°C → still throttles. Need larger heatsink (~2°C/W): 50 + 50 = 100°C → borderline. At 15W mode: 50 + 30 = 80°C → stable, no throttling.

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Duty Cycling Power Budget</b> · <code>thermal</code></summary>

- **Interviewer:** "Your edge device has a 30W steady-state thermal budget, but your perception model requires 45W to run at full speed (30 FPS). You can't upgrade the cooling. How do you run a 45W workload on a 30W thermal budget?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Reduce the model size until it fits in 30W." This sacrifices accuracy. There's a way to keep the full model.

  **Realistic Solution:** Duty cycling. Run inference at full power (45W) for a burst period, then idle (5W) to let the thermal mass absorb and dissipate the heat. The average power must stay at or below 30W. The math: if you run for $t_{on}$ seconds at 45W and idle for $t_{off}$ seconds at 5W, the average power is $(45 \times t_{on} + 5 \times t_{off}) / (t_{on} + t_{off}) \leq 30$. Solving: $t_{on}/t_{off} \leq 25/15 = 5/3$. So for every 5 seconds of inference, you idle for 3 seconds. Effective duty cycle: 62.5%. If the full model runs at 30 FPS, your effective rate is 30 × 0.625 = **18.75 FPS average**. The trade-off is explicit: thermal budget directly constrains sustained throughput. During the off period, the system can use the last detection result or run a lightweight tracker to interpolate.

  > **Napkin Math:** 45W on, 5W off. Target: ≤30W average. $t_{on} = 5s$, $t_{off} = 3s$. Average = (45×5 + 5×3) / 8 = 240/8 = **30W** ✓. Effective FPS = 30 × (5/8) = 18.75 FPS. With 2s on / 1.5s off: (45×2 + 5×1.5) / 3.5 = 97.5/3.5 = **27.9W** ✓. Effective FPS = 30 × (2/3.5) = 17.1 FPS.

  > **Key Equation:** $P_{\text{avg}} = \frac{P_{\text{active}} \times t_{\text{on}} + P_{\text{idle}} \times t_{\text{off}}}{t_{\text{on}} + t_{\text{off}}} \leq P_{\text{thermal}}$

  📖 **Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Solar-Powered Edge Budget</b> · <code>thermal</code> <code>power</code></summary>

- **Interviewer:** "You're designing an edge AI wildlife monitoring station powered by a 20W solar panel and a 100Wh battery. The station runs a bird species classifier on a Google Coral (4 TOPS, 2W) with a camera. The system must operate 24/7 in a remote forest with 5 hours of usable sunlight per day. Design the power budget and determine the maximum inference rate you can sustain."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "20W panel × 5 hours = 100Wh, which exactly matches the battery — we can run continuously." This ignores system overhead, solar conversion losses, and seasonal variation.

  **Realistic Solution:** Start from the energy budget. Solar input: 20W × 5h × 0.8 (panel efficiency loss from angle, clouds, dirt) = **80Wh/day**. System consumers: (1) Coral TPU during inference: 2W, (2) camera module: 0.5W, (3) Raspberry Pi host: 3W idle / 5W active, (4) cellular modem (burst uploads): 3W for 10 min/day = 0.5Wh, (5) system overhead (voltage regulator, watchdog): 0.5W constant = 12Wh/day. Fixed daily cost: 12Wh (overhead) + 0.5Wh (modem) = 12.5Wh. Remaining for compute: 80 - 12.5 = **67.5Wh**. Active power (camera + Pi + Coral): 0.5 + 5 + 2 = 7.5W. Idle power (Pi sleep + watchdog): 0.5 + 0.5 = 1W. Maximum active hours: solve $7.5 \times t_{active} + 1 \times (24 - t_{active}) = 67.5$. $6.5 \times t_{active} = 43.5$. $t_{active} = 6.7$ hours/day. At 10 inferences/second during active hours: 6.7h × 3600s × 10 = **241,200 inferences/day**. But birds are most active at dawn and dusk (~4 hours). Schedule active periods to match: 4h active (dawn/dusk) + 2.7h active (midday sampling) = 6.7h. Reserve 20% battery for cloudy days: effective active time = 5.4h → **194,000 inferences/day**.

  > **Napkin Math:** Solar: 80Wh/day. Fixed overhead: 12.5Wh. Compute budget: 67.5Wh. Active power: 7.5W. Max active time: 6.7h. At 10 inf/s: 241K inferences/day. With 20% reserve: 194K/day. Battery can sustain ~13h of active operation without sun (100Wh / 7.5W), providing 1.5 cloudy days of buffer.

  📖 **Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)

  </details>

</details>

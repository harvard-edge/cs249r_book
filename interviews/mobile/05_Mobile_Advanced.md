# Round 5: Advanced Mobile Systems 🔬

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Mobile_Systems.md">📱 Round 1</a> ·
  <a href="02_Mobile_Constraints.md">⚖️ Round 2</a> ·
  <a href="03_Mobile_Ops_and_Deployment.md">🚀 Round 3</a> ·
  <a href="04_Mobile_Visual_Debugging.md">🖼️ Round 4</a> ·
  <a href="05_Mobile_Advanced.md">🔬 Round 5</a>
</div>

---

This round covers the hardest problems in mobile ML systems: on-device LLM architecture, cross-platform deployment strategy, federated learning at scale, hardware-aware neural architecture search, on-device personalization, real-time video pipelines, multi-modal sensor fusion, device fragmentation, and testing without a device farm. These are the questions that separate senior engineers from architects.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/mobile/05_Mobile_Advanced.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 🤖 On-Device LLM Systems

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The On-Device LLM System Design</b> · <code>architecture</code> <code>memory</code></summary>

**Interviewer:** "Design the complete system architecture for running a 3B parameter LLM on a phone with 8 GB RAM. Cover memory management, inference pipeline, context handling, and user experience. The model must generate tokens at ≥20 tokens/second with 2048 context length."

**Common Mistake:** "Quantize to INT4 and load the whole model into RAM." INT4 gets the weights to 1.5 GB, but you haven't addressed the KV-cache, the inference pipeline, or what happens when the user switches to another app.

**Realistic Solution:** Design a **four-layer architecture** from storage to silicon:

**(1) Storage layer — model format and loading.** Store INT4 weights (1.5 GB) on flash as a memory-mapped file. The OS loads pages on demand — only the layers currently executing are in physical RAM. Use group quantization (128 values per group) with FP16 scale factors for acceptable quality. Store the model in a single contiguous file optimized for sequential access (layer 0 weights first, then layer 1, etc.) to minimize random flash reads.

**(2) Compute layer — inference pipeline.** Prefill phase (processing the prompt): batch all input tokens, run through all 32 transformer layers. This is compute-bound — use the NPU for attention projections and FFN layers. On Apple A17 Pro: 35 TOPS INT8 ≈ 70 TOPS INT4. Prefill 512 tokens: ~6 GFLOPs per token × 512 = 3.07 TFLOPs / 70 TOPS = 44ms. Decode phase (generating tokens): memory-bandwidth bound. Each token requires loading all 1.5 GB of weights from memory. LPDDR5x at 77 GB/s: 1.5 GB / 77 GB/s = 19.5ms per token = 51 tokens/second. Exceeds the 20 tok/s target.

**(3) Memory layer — KV-cache management.** At 2048 context: KV-cache = 2 × 32 layers × 32 heads × 128 dim × 2048 tokens × 2 bytes (FP16) = 1.07 GB. Too large. Solutions: (a) INT8 KV-cache (268 MB) — 0.5% quality loss, acceptable. (b) Sliding window attention (1024 window) — halves KV-cache to 134 MB. (c) GQA (grouped-query attention, 8 KV heads instead of 32) — KV-cache = 268 MB / 4 = 67 MB. Combine INT8 + GQA: **67 MB KV-cache**. Total resident memory: 67 MB (KV) + ~200 MB (active weight pages) + 100 MB (activations) + 50 MB (runtime) = **417 MB**.

**(4) Lifecycle layer — app integration.** When the user backgrounds the app: save the KV-cache to flash (67 MB, ~30ms write). When foregrounded: reload from flash. If the app is jetsammed: the KV-cache file persists — reload it and resume the conversation without re-prefilling the entire context. Implement a "conversation compaction" feature: when context approaches 2048 tokens, summarize the oldest 1024 tokens into 128 tokens using the model itself, freeing KV-cache space.

> **Napkin Math:** Weights (INT4): 1.5 GB on flash, ~200 MB resident. KV-cache (INT8 + GQA): 67 MB. Activations: 100 MB. Runtime: 50 MB. Total resident: **417 MB** (fits in 5 GB available). Prefill 512 tokens: 44ms. Decode: 19.5ms/token → 51 tok/s ✓. Battery per 100-token response: 2s × 3W = 6J = 0.01% of 12.8 Wh battery. App size: 1.5 GB model downloaded separately, not in bundle.

**📖 Deep Dive:** [Volume II: Edge Intelligence](https://mlsysbook.ai/vol2/edge_intelligence.html)
</details>

---

### 🌐 Cross-Platform Deployment

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Cross-Platform ML Runtime Decision</b> · <code>deployment</code> <code>frameworks</code></summary>

**Interviewer:** "Your team needs to deploy the same image classification model on iOS (Apple Neural Engine), Android flagships (Qualcomm Hexagon NPU), and Android budget phones (MediaTek APU). You're debating between Core ML + TFLite (native per-platform), ONNX Runtime (single runtime, multiple backends), and a custom solution. Walk through the trade-offs."

**Common Mistake:** "Use ONNX Runtime everywhere — write once, run anywhere." Cross-platform runtimes sacrifice per-platform optimization for portability. The performance gap can be 2-5×.

**Realistic Solution:** Each approach has a different cost-performance frontier:

**Option A: Native per-platform (Core ML + TFLite)**
- *Performance:* Best. Core ML is co-designed with the Apple Neural Engine — Apple optimizes the compiler for each chip generation. TFLite's NNAPI/QNN delegates are tuned for Qualcomm and MediaTek NPUs. Expect 100% of hardware capability.
- *Engineering cost:* High. Maintain two model conversion pipelines (PyTorch → Core ML, PyTorch → TFLite). Two sets of pre/post-processing code. Two testing matrices. Two deployment pipelines.
- *Risk:* Low. Apple and Google maintain these runtimes as first-party products.

**Option B: ONNX Runtime (cross-platform)**
- *Performance:* 60-80% of native. ONNX Runtime supports CoreML EP (execution provider) on iOS and NNAPI/QNN EP on Android. But the EP layer adds abstraction overhead, and some optimizations (Apple's layer fusion, Qualcomm's graph transformations) are unavailable through the generic ONNX interface.
- *Engineering cost:* Low. One model format (ONNX). One pre/post-processing pipeline. One testing framework.
- *Risk:* Medium. ONNX EP support lags behind native runtime features by 6-12 months. New hardware features (Apple's stateful prediction, Qualcomm's micro-NPU) may not be exposed.

**Option C: Hybrid (Core ML on iOS, ONNX Runtime on Android)**
- *Performance:* Best on iOS (native), good on Android (80% of native).
- *Engineering cost:* Medium. Native pipeline for iOS (where 60% of revenue comes from), cross-platform for the fragmented Android ecosystem.
- *Risk:* Low-medium. Pragmatic trade-off.

**Recommendation:** Option C for most teams. iOS users are typically higher-value (willing to pay for premium features), so invest in native Core ML for the best experience. Android's fragmentation (1000+ devices) makes native optimization per-chipset impractical — ONNX Runtime's "good enough" performance across all devices is the right trade-off.

> **Napkin Math:** MobileNetV3-Large inference: Core ML on A17 Pro: 1.8ms. TFLite on Snapdragon 8 Gen 3: 2.1ms. ONNX Runtime (CoreML EP) on A17 Pro: 2.9ms (61% slower). ONNX Runtime (QNN EP) on Snapdragon 8 Gen 3: 3.4ms (62% slower). ONNX Runtime (NNAPI) on MediaTek Dimensity 9300: 5.2ms. Engineering cost: Native dual-platform: ~3 engineer-months. ONNX single-platform: ~1 engineer-month. Hybrid: ~2 engineer-months.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🔐 Federated Learning at Scale

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Federated Learning System for a Social Media App</b> · <code>privacy</code> <code>distributed</code></summary>

**Interviewer:** "Design a federated learning system for a social media app with 500 million daily active users. The model personalizes the content feed. You must improve the model without collecting user interaction data on your servers. Cover device selection, communication efficiency, convergence, and privacy guarantees."

**Common Mistake:** "Run FedAvg across all 500M devices." Naive FedAvg at this scale would require petabytes of gradient uploads per round and wouldn't converge due to extreme data heterogeneity (non-IID user behavior).

**Realistic Solution:** Design a **hierarchical federated system** with four key subsystems:

**(1) Device selection and eligibility.** Not all devices can participate. Eligibility criteria: charging, on WiFi, idle (screen off for >5 minutes), sufficient RAM (>2 GB free), OS version supports on-device training (iOS 16+ with Core ML Training, Android 12+ with TFLite training API). At any given moment, ~5% of DAU meets all criteria = 25M eligible devices. Per round, sample 10,000 devices (stratified by region, device tier, and user activity level to ensure representativeness).

**(2) Communication efficiency.** A feed ranking model has ~50M parameters. FP32 gradients: 200 MB per device per round. At 10,000 devices: 2 TB per round. Unacceptable. Compression stack: (a) Top-k sparsification (k=1%) — send only the 1% largest gradient values + their indices. 200 MB → 4 MB. (b) INT8 quantization of gradient values: 4 MB → 2 MB. (c) Error feedback — accumulate the unsent 99% of gradients locally and add them to the next round's update. This preserves convergence despite aggressive sparsification. Total upload per device: **2 MB**. Per round: 20 GB. Manageable.

**(3) Convergence under non-IID data.** User behavior is extremely heterogeneous — a teenager's feed interactions look nothing like a retiree's. Standard FedAvg diverges under this heterogeneity. Solutions: (a) **FedProx** — add a proximal term that penalizes local models from drifting too far from the global model. (b) **Clustered FL** — group users into behavioral clusters (identified by embedding similarity, computed on-device). Run separate federated models per cluster. 10-20 clusters typically suffice. (c) **Personalization layers** — the base model (shared backbone) is trained federally. The final 2-3 layers are trained locally and never uploaded. This gives each user a personalized model while the shared layers capture global patterns.

**(4) Privacy guarantees.** Differential privacy with (ε=8, δ=10⁻¹⁰) per round. Gradient clipping norm C=1.0. Gaussian noise σ = C × √(2 ln(1.25/δ)) / ε ≈ 0.7. With 10,000 devices per round, the noise in the aggregate is σ/√10,000 = 0.007 — negligible impact on model quality. Secure aggregation ensures the server never sees individual gradients — only the encrypted sum. Annual privacy budget: 50 rounds × ε=8 = ε=400 (with advanced composition, actual ε ≈ 120 using Rényi DP accounting).

> **Napkin Math:** 500M DAU. 25M eligible per moment. 10,000 sampled per round. Upload: 2 MB × 10,000 = 20 GB/round. Server aggregation: ~5 minutes on 8 GPUs. Rounds per day: 4 (every 6 hours). Daily bandwidth: 80 GB. Monthly: 2.4 TB. CDN cost: ~$200/month. On-device training time: ~3 minutes per round (50 local steps on 50M-param model). Battery cost per round: 3 min × 2W = 0.1 Wh = 0.8% of a 12.8 Wh battery. Convergence: ~200 rounds (50 days) to match centralized training quality within 1%.

**📖 Deep Dive:** [Volume II: Security & Privacy](https://mlsysbook.ai/vol2/security_privacy.html)
</details>

---

### 🧬 Neural Architecture Search

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Hardware-Aware NAS for Mobile</b> · <code>architecture</code> <code>optimization</code></summary>

**Interviewer:** "You need a model that runs in <5ms on the Apple Neural Engine AND <8ms on the Qualcomm Hexagon NPU. Standard NAS finds architectures that minimize FLOPs, but a low-FLOP model isn't necessarily fast on either accelerator. How do you run NAS that optimizes for actual on-device latency?"

**Common Mistake:** "Minimize FLOPs — lower FLOPs always means lower latency." A depthwise separable convolution has fewer FLOPs than a standard convolution but can be *slower* on NPUs that are optimized for dense matrix operations. FLOPs ≠ latency.

**Realistic Solution:** Build a **hardware-aware NAS** system with latency predictors for each target platform:

**(1) Latency lookup tables (LUTs).** For each target hardware, profile every candidate operation (3×3 conv, 5×5 depthwise conv, SE block, etc.) at every possible input resolution and channel count. On the Apple Neural Engine: profile using Core ML's `MLModel.predict()` with `MLPredictionOptions` timing. On Hexagon: profile using the QNN Profiler. Store results in a lookup table: `LUT[op_type][input_channels][output_channels][resolution] → latency_ms`. Building the LUT takes ~2 days per hardware target (automated), but it's a one-time cost.

**(2) Differentiable latency objective.** During NAS, the search objective becomes: minimize $\mathcal{L}_{task} + \lambda_1 \cdot \text{LAT}_{ANE}(arch) + \lambda_2 \cdot \text{LAT}_{Hexagon}(arch)$, where LAT is computed by summing LUT entries for the candidate architecture. This is differentiable (the LUT is interpolated as a continuous function), so gradient-based NAS (DARTS, ProxylessNAS) works.

**(3) Hardware-specific constraints.** The Neural Engine prefers: channel counts that are multiples of 32 (its SIMD width), depthwise convolutions (native support), and models with <500 layers (compiler limitation). The Hexagon NPU prefers: channel counts that are multiples of 16, avoids dynamic shapes, and has a 64 MB on-chip SRAM that benefits from activation reuse. Encode these as hard constraints in the search space — don't waste search time on architectures that violate hardware requirements.

**(4) Multi-objective Pareto search.** The result is a Pareto frontier of architectures trading off accuracy vs ANE latency vs Hexagon latency. Select the architecture that meets both latency targets with maximum accuracy. Typical result: a model with non-uniform channel widths (wider in early layers where both NPUs are efficient, narrower in later layers) and mixed operation types (standard conv in bottleneck layers, depthwise in expansion layers).

> **Napkin Math:** Search space: 10²⁰ possible architectures (20 layers × 6 op choices × 5 channel widths). LUT size: 6 ops × 10 channel configs × 8 resolutions = 480 entries per hardware. Profiling: 480 × 100 runs × 10ms = 8 minutes per hardware. NAS search: 500 GPU-hours on a cloud cluster (ProxylessNAS). Found architecture: 3.8ms on ANE (under 5ms ✓), 7.2ms on Hexagon (under 8ms ✓), 76.3% ImageNet top-1. Comparable FLOPs-optimized NAS result: 4.1ms ANE, 11.3ms Hexagon (fails Hexagon target). The hardware-aware search found a 37% faster Hexagon architecture at equal accuracy.

**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)
</details>

---

### 🎯 On-Device Personalization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The On-Device Fine-Tuning Pipeline</b> · <code>training</code> <code>privacy</code></summary>

**Interviewer:** "Your photo app has a 'search by description' feature powered by a CLIP-like model. Users in Japan complain it doesn't understand Japanese food categories. Users in Brazil say it misclassifies local bird species. You can't collect user photos on your servers. Design an on-device personalization system that adapts the model to each user's photo library."

**Common Mistake:** "Fine-tune the entire model on-device." A CLIP model has 150M+ parameters. Fine-tuning all of them on a phone would take hours and drain the battery.

**Realistic Solution:** Use **parameter-efficient fine-tuning (PEFT)** on-device:

**(1) Freeze the backbone, train adapters.** Add LoRA (Low-Rank Adaptation) layers to the model's attention projections. LoRA adds two small matrices (rank 4-8) per attention layer. For a 150M-param model with 12 attention layers: trainable parameters = 12 × 2 × (768 × 4 + 4 × 768) = 73,728 parameters. That's 0.05% of the full model. Training 73K parameters is feasible on-device.

**(2) Training data from the user's library.** Use the user's existing photo-label associations as training signal: album names ("Tokyo Trip 2025"), faces (contacts), locations (geo-tags), and user corrections (when the user searches for "ramen" and taps a photo the model ranked low). Generate (image, text) training pairs automatically. No manual labeling needed.

**(3) Training schedule.** Fine-tune during charging + idle, using Core ML's `MLUpdateTask` (iOS) or TFLite's training API (Android). 50 gradient steps on 200 training pairs: ~90 seconds on the A17 Pro Neural Engine. Power: 2W × 90s = 180J = 0.4% battery. Run weekly or when the user adds >50 new photos.

**(4) Quality safeguard.** After fine-tuning, run the adapted model on a held-out validation set (20 photos the user has previously searched for successfully). If search recall drops >10% vs the base model, discard the adaptation and keep the base model. This prevents catastrophic forgetting.

**(5) Privacy guarantee.** The LoRA weights (73K × 4 bytes = 294 KB) stay on-device. They encode user-specific knowledge but are meaningless without the base model and the user's photo library. Even if exfiltrated, they reveal minimal information about the user's data.

> **Napkin Math:** Base model: 150M params, 300 MB (FP16). LoRA adapters: 73K params, 294 KB. Training: 200 pairs × 50 steps = 10,000 forward+backward passes. Time: 90 seconds on NPU. Memory during training: 300 MB (frozen weights, mmap'd) + 294 KB (trainable) + 50 MB (activations + gradients for trainable params only) = ~350 MB. Fits on any phone with 4+ GB RAM. Personalization improvement: +12% recall on user-specific categories (measured in internal testing with Japanese and Brazilian photo libraries).

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 📹 Real-Time Video Processing

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The 60 FPS Camera ML Pipeline</b> · <code>latency</code> <code>architecture</code></summary>

**Interviewer:** "Design a camera pipeline that runs portrait segmentation at 60 FPS with ML-powered depth-of-field blur. The segmentation model takes 25ms per frame. The camera outputs at 60 FPS (16.67ms per frame). The math doesn't work — 25ms > 16.67ms. How do you deliver 60 FPS?"

**Common Mistake:** "Use a faster model." Even a 15ms model leaves only 1.67ms for everything else (rendering, UI, camera capture). One GC pause and you drop frames.

**Realistic Solution:** Design a **decoupled multi-rate pipeline** where the camera, ML, and rendering run at independent rates:

**(1) Camera capture: 60 FPS.** The camera ISP produces frames every 16.67ms into a triple buffer. This never stalls — the ISP runs on dedicated hardware independent of the CPU/NPU.

**(2) ML inference: 24 FPS (every ~42ms).** The NPU processes every 2.5th camera frame. A dedicated ML thread picks the latest frame from the triple buffer, runs segmentation (25ms), and publishes the mask to a shared atomic buffer. The NPU runs at ~96% utilization (25ms compute / 42ms period × 2 for double-buffering overhead).

**(3) Rendering: 60 FPS.** The GPU composites each camera frame with the most recent segmentation mask. Between mask updates (42ms gap = 2-3 frames), the GPU warps the previous mask using optical flow estimated from the camera's gyroscope data (available at 200 Hz, <0.5ms to compute affine transform). The warped mask tracks head movement smoothly.

**(4) Synchronization.** Use lock-free single-producer/single-consumer queues between pipeline stages. The camera thread never blocks on the ML thread. The render thread never blocks on either. Each stage reads the latest available data — if the ML thread is slow, the render thread uses a slightly older (but warped) mask. No frame drops.

**(5) Latency budget per render frame:** Camera readout: 1ms (hardware). Gyro-based mask warp: 0.5ms (GPU). Depth-of-field blur: 4ms (GPU compute shader). Composite + display: 2ms (GPU). Total: **7.5ms** — well within the 16.67ms budget. The 25ms ML inference happens in parallel on the NPU and never touches the render budget.

> **Napkin Math:** Camera: 60 FPS (16.67ms period). ML: 25ms inference → 40 FPS max on NPU, but we target 24 FPS (every 2.5 frames) for thermal headroom. Render: 7.5ms per frame → 133 FPS theoretical, 60 FPS actual (VSync locked). Mask staleness: worst case 42ms (2.5 frames). At arm's length, a head moves ~3 pixels in 42ms — the gyro warp corrects this to <0.5 pixel error. Imperceptible to the user. Power: NPU at 24 FPS × 25ms = 60% duty cycle × 2W = 1.2W. GPU render: 60 FPS × 7.5ms = 45% duty cycle × 1.5W = 0.675W. Total ML+render: 1.875W.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

---

### 🔀 Multi-Modal On-Device AI

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Multi-Modal Sensor Fusion System</b> · <code>architecture</code> <code>sensor-fusion</code></summary>

**Interviewer:** "Design an on-device system for a fitness app that fuses camera (pose estimation), microphone (rep counting from audio cues), and accelerometer (motion tracking) to provide real-time exercise form feedback. The three sensors run at different rates, produce different data types, and compete for compute resources. How do you architect the fusion?"

**Common Mistake:** "Run three separate models and combine their outputs." Three independent models triple the compute and memory cost, and you lose cross-modal correlations (e.g., the sound of a weight hitting the floor confirms the accelerometer's "rep complete" signal).

**Realistic Solution:** Design an **asynchronous multi-rate fusion architecture**:

**(1) Sensor layer — independent capture at native rates.**
- Camera: 30 FPS, 720p (reduced from 1080p to save bandwidth). Produces 720×1280×3 = 2.76 MB per frame.
- Microphone: 16 kHz, mono. Produces 32 KB per second.
- Accelerometer: 100 Hz, 3-axis. Produces 1.2 KB per second.

**(2) Per-modal feature extraction — runs at each sensor's native rate.**
- Camera → Pose model (MoveNet Lightning, 3.6 MB, 8ms on NPU): extracts 17 keypoints per frame. Runs at 30 FPS on the NPU.
- Microphone → Audio classifier (YAMNet-like, 1.2 MB, 2ms on CPU): extracts a 512-dim audio embedding per 960ms window. Runs at ~1 Hz.
- Accelerometer → Signal processor (FFT + peak detection, no ML, 0.1ms on CPU): extracts motion features (cadence, amplitude, jerk) per 1-second window. Runs at 1 Hz.

**(3) Fusion model — runs at the slowest modality rate (1 Hz).**
A lightweight transformer (0.8 MB, 3ms on NPU) takes as input: the latest 30 pose keypoint sequences (1 second of poses), the latest audio embedding, and the latest motion feature vector. It outputs: exercise type, rep count, form score (0-100), and specific form corrections ("keep your back straight," "lower the weight slower").

**(4) Temporal alignment.** The three modalities are asynchronous. Use timestamps to align: each feature extraction module tags its output with the sensor timestamp. The fusion model's input buffer stores the latest features from each modality. At each 1 Hz fusion tick, it reads the most recent features regardless of when they were produced. Staleness is bounded: camera features are at most 33ms old (1 frame), audio/accel features are at most 1 second old.

**(5) Resource scheduling.** The NPU is shared between pose estimation (8ms × 30 FPS = 240ms/s = 24% duty) and fusion (3ms × 1 Hz = 0.3%). Total NPU utilization: 24.3%. The CPU handles audio + accel processing: 2ms × 1 Hz + 0.1ms × 1 Hz = 2.1ms/s = 0.2%. Plenty of headroom for the UI and other app logic.

> **Napkin Math:** Memory: Pose model (3.6 MB) + Audio model (1.2 MB) + Fusion model (0.8 MB) + buffers (30 frames × 17 keypoints × 3 coords × 4 bytes = 6 KB + 2 KB audio + 0.1 KB accel) = **5.6 MB total ML footprint**. Latency: form feedback updates at 1 Hz (1 second). Pose overlay updates at 30 FPS (33ms). Power: NPU at 24.3% duty × 2W = 0.486W. CPU at 0.2% = negligible. Camera: ~300 mW. Total: ~800 mW. Battery for 1-hour workout: 0.8 Wh / 12.8 Wh = 6.25%.

**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

---

### 📱 Device Fragmentation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The 1000-Device Android Fragmentation Problem</b> · <code>deployment</code> <code>optimization</code></summary>

**Interviewer:** "Your app runs on 1000+ Android device models. The top 10 devices account for 30% of users; the remaining 970 devices account for 70%. You have NPU support on Qualcomm (Hexagon), Samsung (Exynos NPU), MediaTek (APU), and Google (Tensor TPU) — each with different operator support, quantization formats, and performance characteristics. How do you ship one model that works well across all of them?"

**Common Mistake:** "Test on the top 10 devices and hope the rest work." The long tail of 970 devices will produce crashes, silent failures, and performance regressions that you'll never catch in testing.

**Realistic Solution:** Build a **tiered execution strategy** with runtime capability detection:

**(1) Capability fingerprinting at first launch.** Run a 2-second benchmark that tests: (a) NNAPI availability and supported op set, (b) GPU delegate availability (OpenCL/Vulkan), (c) CPU NEON/dotprod instruction support, (d) available RAM, (e) sustained thermal performance (short burst test). Hash the results into a "device capability fingerprint" and map it to a pre-defined execution tier.

**(2) Four execution tiers:**
- **Tier S (Flagship NPU):** Snapdragon 8 Gen 2+, Tensor G3+, Exynos 2400+. Full INT4 quantized model, NPU-delegated. Latency: 3-5ms.
- **Tier A (Mid-range NPU):** Snapdragon 7 Gen 1+, Dimensity 8000+. INT8 quantized model, NPU-delegated with CPU fallback for unsupported ops. Latency: 8-15ms.
- **Tier B (GPU compute):** Devices with capable GPUs but weak/no NPU. INT8 model, GPU delegate (OpenCL). Latency: 15-30ms.
- **Tier C (CPU only):** Budget devices, old devices, or devices where all delegates fail. INT8 model, XNNPACK CPU delegate with NEON optimization. Latency: 30-80ms.

**(3) Graceful fallback chain.** At inference time: try NPU delegate → if it fails (unsupported op, driver crash), fall back to GPU delegate → if it fails, fall back to CPU. Log the fallback event for analytics. The model always runs — the question is how fast.

**(4) Remote configuration.** Maintain a server-side mapping of `device_model → recommended_tier` based on aggregated telemetry from your user base. When a new device model appears, it starts at Tier C (safest) and is automatically promoted as telemetry confirms stability at higher tiers.

**(5) Automated device testing.** Use Firebase Test Lab or AWS Device Farm to run your model on 50+ physical devices nightly. Test: (a) inference produces valid output, (b) latency is within tier expectations, (c) no crashes after 1000 consecutive inferences, (d) memory doesn't grow over time (leak detection).

> **Napkin Math:** User distribution: Tier S (15%), Tier A (30%), Tier B (25%), Tier C (30%). Weighted average latency: 0.15×4 + 0.30×12 + 0.25×22 + 0.30×50 = 0.6 + 3.6 + 5.5 + 15.0 = **24.7ms average**. If you only optimized for Tier S: 4ms for 15% of users, crashes for 30% (Tier C devices can't run INT4). Capability fingerprint size: ~200 bytes. Benchmark time: 2 seconds (first launch only). Firebase Test Lab: 50 devices × $5/device/month = $250/month.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🧪 Mobile ML Testing

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Device-Free ML Testing Strategy</b> · <code>testing</code> <code>deployment</code></summary>

**Interviewer:** "Your team ships an ML-powered camera feature to 200 million users across iOS and Android. You don't have a device farm — you have 5 test phones on your desk. How do you test the ML pipeline across devices without a device farm?"

**Common Mistake:** "Test on our 5 phones and rely on crash reports from production." This means your 200 million users are your test farm. You'll discover bugs from 1-star reviews.

**Realistic Solution:** Build a **multi-layer testing pyramid** that catches ML failures before they reach users:

**(1) Unit tests (no device needed).** Test the model in isolation on your CI server (x86 CPU). Run inference on a golden test set (1000 images with known outputs). Assert: output tensor shape is correct, confidence scores sum to 1.0, known inputs produce expected outputs within tolerance (cosine similarity >0.999 vs reference). This catches model conversion errors, quantization bugs, and preprocessing mismatches. Run on every PR.

**(2) Integration tests (emulator/simulator).** Run the full inference pipeline (preprocessing → model → postprocessing) on iOS Simulator and Android Emulator. These don't have NPU/GPU delegates, so inference runs on CPU — but they catch: memory leaks (run 10,000 consecutive inferences, assert RSS doesn't grow), threading bugs (run inference from multiple threads simultaneously), lifecycle bugs (simulate backgrounding/foregrounding during inference), and API compatibility (test against multiple OS versions).

**(3) Cloud device testing (no physical farm needed).** Use Firebase Test Lab (Android, 100+ real devices) or AWS Device Farm (iOS + Android). Run a 5-minute smoke test on 30 representative devices: 10 flagships, 10 mid-range, 10 budget. Assert: inference completes without crash, latency is within tier expectations, output matches reference within tolerance. Cost: ~$150/test run. Run on every release candidate.

**(4) Canary release (production testing).** Release to 1% of users with enhanced telemetry: per-inference latency, confidence distribution, crash rate, and delegate fallback rate. Monitor for 48 hours. Alert thresholds: crash rate >0.1%, median latency >2× baseline, confidence distribution KL divergence >0.1 vs previous version. If any threshold is breached, halt rollout automatically.

**(5) Synthetic stress testing.** Generate adversarial inputs programmatically: all-black images, all-white images, 1×1 pixel images, maximum-resolution images, images with extreme aspect ratios, corrupted JPEG headers. Assert the model returns gracefully (empty output or error code) instead of crashing or hanging. Run in CI.

> **Napkin Math:** Testing pyramid coverage: Unit tests catch ~60% of bugs (model conversion, preprocessing). Integration tests catch ~20% (memory, threading, lifecycle). Cloud device tests catch ~15% (hardware-specific failures, delegate bugs). Canary catches ~4% (real-world edge cases). Remaining ~1%: discovered in production. Cost: Unit + integration: $0 (CI time). Cloud devices: $150 × 4 releases/month = $600/month. Canary: $0 (uses production infrastructure). Total: **$600/month** to test across 100+ devices without owning any.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

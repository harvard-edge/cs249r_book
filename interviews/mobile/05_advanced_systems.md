# Round 5: Advanced Mobile Systems 🔬

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_systems_and_soc.md">📱 1. Systems & SoC</a> ·
  <a href="02_compute_and_memory.md">⚖️ 2. Compute & Memory</a> ·
  <a href="03_data_and_deployment.md">🚀 3. Data & Deployment</a> ·
  <a href="04_visual_debugging.md">🖼️ 4. Visual Debugging</a> ·
  <a href="05_advanced_systems.md">🔬 5. Advanced Systems</a>
</div>

---

This round covers the hardest problems in mobile ML systems: on-device LLM architecture, cross-platform deployment strategy, federated learning at scale, hardware-aware neural architecture search, on-device personalization, real-time video pipelines, multi-modal sensor fusion, device fragmentation, and testing without a device farm. These are the questions that separate senior engineers from architects.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/mobile/05_advanced_systems.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 🤖 On-Device LLM Systems

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The On-Device LLM System Design</b> · <code>architecture</code> <code>memory</code></summary>

- **Interviewer:** "Design the complete system architecture for running a 3B parameter LLM on a phone with 8 GB RAM. Cover memory management, inference pipeline, context handling, and user experience. The model must generate tokens at ≥20 tokens/second with 2048 context length."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Quantize to INT4 and load the whole model into RAM." INT4 gets the weights to 1.5 GB, but you haven't addressed the KV-cache, the inference pipeline, or what happens when the user switches to another app.

  **Realistic Solution:** Design a **four-layer architecture** from storage to silicon:

  **(1) Storage layer — model format and loading.** Store INT4 weights (1.5 GB) on flash as a memory-mapped file. The OS loads pages on demand — only the layers currently executing are in physical RAM. Use group quantization (128 values per group) with FP16 scale factors for acceptable quality. Store the model in a single contiguous file optimized for sequential access (layer 0 weights first, then layer 1, etc.) to minimize random flash reads.

  **(2) Compute layer — inference pipeline.** Prefill phase (processing the prompt): batch all input tokens, run through all 32 transformer layers. This is compute-bound — use the NPU for attention projections and FFN layers. On Apple A17 Pro: 35 TOPS INT8 ≈ 70 TOPS INT4. Prefill 512 tokens: ~6 GFLOPs per token × 512 = 3.07 TFLOPs / 70 TOPS = 44ms. Decode phase (generating tokens): memory-bandwidth bound. Each token requires loading all 1.5 GB of weights from memory. LPDDR5x at 77 GB/s: 1.5 GB / 77 GB/s = 19.5ms per token = 51 tokens/second. Exceeds the 20 tok/s target.

  **(3) Memory layer — KV-cache management.** At 2048 context: KV-cache = 2 × 32 layers × 32 heads × 128 dim × 2048 tokens × 2 bytes (FP16) = 1.07 GB. Too large. Solutions: (a) INT8 KV-cache (268 MB) — 0.5% quality loss, acceptable. (b) Sliding window attention (1024 window) — halves KV-cache to 134 MB. (c) GQA (grouped-query attention, 8 KV heads instead of 32) — KV-cache = 268 MB / 4 = 67 MB. Combine INT8 + GQA: **67 MB KV-cache**. Total resident memory: 67 MB (KV) + ~200 MB (active weight pages) + 100 MB (activations) + 50 MB (runtime) = **417 MB**.

  **(4) Lifecycle layer — app integration.** When the user backgrounds the app: save the KV-cache to flash (67 MB, ~30ms write). When foregrounded: reload from flash. If the app is jetsammed: the KV-cache file persists — reload it and resume the conversation without re-prefilling the entire context. Implement a "conversation compaction" feature: when context approaches 2048 tokens, summarize the oldest 1024 tokens into 128 tokens using the model itself, freeing KV-cache space.

  > **Napkin Math:** Weights (INT4): 1.5 GB on flash, ~200 MB resident. KV-cache (INT8 + GQA): 67 MB. Activations: 100 MB. Runtime: 50 MB. Total resident: **417 MB** (fits in 5 GB available). Prefill 512 tokens: 44ms. Decode: 19.5ms/token → 51 tok/s ✓. Battery per 100-token response: 2s × 3W = 6J = 0.01% of 12.8 Wh battery. App size: 1.5 GB model downloaded separately, not in bundle.

  📖 **Deep Dive:** [Volume II: Edge Intelligence](https://harvard-edge.github.io/cs249r_book_dev/contents/edge_intelligence/edge_intelligence.html)

  </details>

</details>

---

### 🌐 Cross-Platform Deployment

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Cross-Platform ML Runtime Decision</b> · <code>deployment</code> <code>frameworks</code></summary>

- **Interviewer:** "Your team needs to deploy the same image classification model on iOS (Apple Neural Engine), Android flagships (Qualcomm Hexagon NPU), and Android budget phones (MediaTek APU). You're debating between Core ML + TFLite (native per-platform), ONNX Runtime (single runtime, multiple backends), and a custom solution. Walk through the trade-offs."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

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

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

---

### 🔐 Federated Learning at Scale

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Federated Learning System for a Social Media App</b> · <code>privacy</code> <code>distributed</code></summary>

- **Interviewer:** "Design a federated learning system for a social media app with 500 million daily active users. The model personalizes the content feed. You must improve the model without collecting user interaction data on your servers. Cover device selection, communication efficiency, convergence, and privacy guarantees."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run FedAvg across all 500M devices." Naive FedAvg at this scale would require petabytes of gradient uploads per round and wouldn't converge due to extreme data heterogeneity (non-IID user behavior).

  **Realistic Solution:** Design a **hierarchical federated system** with four key subsystems:

  **(1) Device selection and eligibility.** Not all devices can participate. Eligibility criteria: charging, on WiFi, idle (screen off for >5 minutes), sufficient RAM (>2 GB free), OS version supports on-device training (iOS 16+ with Core ML Training, Android 12+ with TFLite training API). At any given moment, ~5% of DAU meets all criteria = 25M eligible devices. Per round, sample 10,000 devices (stratified by region, device tier, and user activity level to ensure representativeness).

  **(2) Communication efficiency.** A feed ranking model has ~50M parameters. FP32 gradients: 200 MB per device per round. At 10,000 devices: 2 TB per round. Unacceptable. Compression stack: (a) Top-k sparsification (k=1%) — send only the 1% largest gradient values + their indices. 200 MB → 4 MB. (b) INT8 quantization of gradient values: 4 MB → 2 MB. (c) Error feedback — accumulate the unsent 99% of gradients locally and add them to the next round's update. This preserves convergence despite aggressive sparsification. Total upload per device: **2 MB**. Per round: 20 GB. Manageable.

  **(3) Convergence under non-IID data.** User behavior is extremely heterogeneous — a teenager's feed interactions look nothing like a retiree's. Standard FedAvg diverges under this heterogeneity. Solutions: (a) **FedProx** — add a proximal term that penalizes local models from drifting too far from the global model. (b) **Clustered FL** — group users into behavioral clusters (identified by embedding similarity, computed on-device). Run separate federated models per cluster. 10-20 clusters typically suffice. (c) **Personalization layers** — the base model (shared backbone) is trained federally. The final 2-3 layers are trained locally and never uploaded. This gives each user a personalized model while the shared layers capture global patterns.

  **(4) Privacy guarantees.** Differential privacy with (ε=8, δ=10⁻¹⁰) per round. Gradient clipping norm C=1.0. Gaussian noise σ = C × √(2 ln(1.25/δ)) / ε ≈ 0.7. With 10,000 devices per round, the noise in the aggregate is σ/√10,000 = 0.007 — negligible impact on model quality. Secure aggregation ensures the server never sees individual gradients — only the encrypted sum. Annual privacy budget: 50 rounds × ε=8 = ε=400 (with advanced composition, actual ε ≈ 120 using Rényi DP accounting).

  > **Napkin Math:** 500M DAU. 25M eligible per moment. 10,000 sampled per round. Upload: 2 MB × 10,000 = 20 GB/round. Server aggregation: ~5 minutes on 8 GPUs. Rounds per day: 4 (every 6 hours). Daily bandwidth: 80 GB. Monthly: 2.4 TB. CDN cost: ~$200/month. On-device training time: ~3 minutes per round (50 local steps on 50M-param model). Battery cost per round: 3 min × 2W = 0.1 Wh = 0.8% of a 12.8 Wh battery. Convergence: ~200 rounds (50 days) to match centralized training quality within 1%.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

---

### 🧬 Neural Architecture Search

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Hardware-Aware NAS for Mobile</b> · <code>architecture</code> <code>optimization</code></summary>

- **Interviewer:** "You need a model that runs in <5ms on the Apple Neural Engine AND <8ms on the Qualcomm Hexagon NPU. Standard NAS finds architectures that minimize FLOPs, but a low-FLOP model isn't necessarily fast on either accelerator. How do you run NAS that optimizes for actual on-device latency?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Minimize FLOPs — lower FLOPs always means lower latency." A depthwise separable convolution has fewer FLOPs than a standard convolution but can be *slower* on NPUs that are optimized for dense matrix operations. FLOPs ≠ latency.

  **Realistic Solution:** Build a **hardware-aware NAS** system with latency predictors for each target platform:

  **(1) Latency lookup tables (LUTs).** For each target hardware, profile every candidate operation (3×3 conv, 5×5 depthwise conv, SE block, etc.) at every possible input resolution and channel count. On the Apple Neural Engine: profile using Core ML's `MLModel.predict()` with `MLPredictionOptions` timing. On Hexagon: profile using the QNN Profiler. Store results in a lookup table: `LUT[op_type][input_channels][output_channels][resolution] → latency_ms`. Building the LUT takes ~2 days per hardware target (automated), but it's a one-time cost.

  **(2) Differentiable latency objective.** During NAS, the search objective becomes: minimize $\mathcal{L}_{task} + \lambda_1 \cdot \text{LAT}_{ANE}(arch) + \lambda_2 \cdot \text{LAT}_{Hexagon}(arch)$, where LAT is computed by summing LUT entries for the candidate architecture. This is differentiable (the LUT is interpolated as a continuous function), so gradient-based NAS (DARTS, ProxylessNAS) works.

  **(3) Hardware-specific constraints.** The Neural Engine prefers: channel counts that are multiples of 32 (its SIMD width), depthwise convolutions (native support), and models with <500 layers (compiler limitation). The Hexagon NPU prefers: channel counts that are multiples of 16, avoids dynamic shapes, and has a 64 MB on-chip SRAM that benefits from activation reuse. Encode these as hard constraints in the search space — don't waste search time on architectures that violate hardware requirements.

  **(4) Multi-objective Pareto search.** The result is a Pareto frontier of architectures trading off accuracy vs ANE latency vs Hexagon latency. Select the architecture that meets both latency targets with maximum accuracy. Typical result: a model with non-uniform channel widths (wider in early layers where both NPUs are efficient, narrower in later layers) and mixed operation types (standard conv in bottleneck layers, depthwise in expansion layers).

  > **Napkin Math:** Search space: 10²⁰ possible architectures (20 layers × 6 op choices × 5 channel widths). LUT size: 6 ops × 10 channel configs × 8 resolutions = 480 entries per hardware. Profiling: 480 × 100 runs × 10ms = 8 minutes per hardware. NAS search: 500 GPU-hours on a cloud cluster (ProxylessNAS). Found architecture: 3.8ms on ANE (under 5ms ✓), 7.2ms on Hexagon (under 8ms ✓), 76.3% ImageNet top-1. Comparable FLOPs-optimized NAS result: 4.1ms ANE, 11.3ms Hexagon (fails Hexagon target). The hardware-aware search found a 37% faster Hexagon architecture at equal accuracy.

  📖 **Deep Dive:** [Volume I: Network Architectures](https://harvard-edge.github.io/cs249r_book_dev/contents/network_architectures/network_architectures.html)

  </details>

</details>

---

### 🎯 On-Device Personalization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The On-Device Fine-Tuning Pipeline</b> · <code>training</code> <code>privacy</code></summary>

- **Interviewer:** "Your photo app has a 'search by description' feature powered by a CLIP-like model. Users in Japan complain it doesn't understand Japanese food categories. Users in Brazil say it misclassifies local bird species. You can't collect user photos on your servers. Design an on-device personalization system that adapts the model to each user's photo library."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Fine-tune the entire model on-device." A CLIP model has 150M+ parameters. Fine-tuning all of them on a phone would take hours and drain the battery.

  **Realistic Solution:** Use **parameter-efficient fine-tuning (PEFT)** on-device:

  **(1) Freeze the backbone, train adapters.** Add LoRA (Low-Rank Adaptation) layers to the model's attention projections. LoRA adds two small matrices (rank 4-8) per attention layer. For a 150M-param model with 12 attention layers: trainable parameters = 12 × 2 × (768 × 4 + 4 × 768) = 73,728 parameters. That's 0.05% of the full model. Training 73K parameters is feasible on-device.

  **(2) Training data from the user's library.** Use the user's existing photo-label associations as training signal: album names ("Tokyo Trip 2025"), faces (contacts), locations (geo-tags), and user corrections (when the user searches for "ramen" and taps a photo the model ranked low). Generate (image, text) training pairs automatically. No manual labeling needed.

  **(3) Training schedule.** Fine-tune during charging + idle, using Core ML's `MLUpdateTask` (iOS) or TFLite's training API (Android). 50 gradient steps on 200 training pairs: ~90 seconds on the A17 Pro Neural Engine. Power: 2W × 90s = 180J = 0.4% battery. Run weekly or when the user adds >50 new photos.

  **(4) Quality safeguard.** After fine-tuning, run the adapted model on a held-out validation set (20 photos the user has previously searched for successfully). If search recall drops >10% vs the base model, discard the adaptation and keep the base model. This prevents catastrophic forgetting.

  **(5) Privacy guarantee.** The LoRA weights (73K × 4 bytes = 294 KB) stay on-device. They encode user-specific knowledge but are meaningless without the base model and the user's photo library. Even if exfiltrated, they reveal minimal information about the user's data.

  > **Napkin Math:** Base model: 150M params, 300 MB (FP16). LoRA adapters: 73K params, 294 KB. Training: 200 pairs × 50 steps = 10,000 forward+backward passes. Time: 90 seconds on NPU. Memory during training: 300 MB (frozen weights, mmap'd) + 294 KB (trainable) + 50 MB (activations + gradients for trainable params only) = ~350 MB. Fits on any phone with 4+ GB RAM. Personalization improvement: +12% recall on user-specific categories (measured in internal testing with Japanese and Brazilian photo libraries).

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

---

### 📹 Real-Time Video Processing

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The 60 FPS Camera ML Pipeline</b> · <code>latency</code> <code>architecture</code></summary>

- **Interviewer:** "Design a camera pipeline that runs portrait segmentation at 60 FPS with ML-powered depth-of-field blur. The segmentation model takes 25ms per frame. The camera outputs at 60 FPS (16.67ms per frame). The math doesn't work — 25ms > 16.67ms. How do you deliver 60 FPS?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use a faster model." Even a 15ms model leaves only 1.67ms for everything else (rendering, UI, camera capture). One GC pause and you drop frames.

  **Realistic Solution:** Design a **decoupled multi-rate pipeline** where the camera, ML, and rendering run at independent rates:

  **(1) Camera capture: 60 FPS.** The camera ISP produces frames every 16.67ms into a triple buffer. This never stalls — the ISP runs on dedicated hardware independent of the CPU/NPU.

  **(2) ML inference: 24 FPS (every ~42ms).** The NPU processes every 2.5th camera frame. A dedicated ML thread picks the latest frame from the triple buffer, runs segmentation (25ms), and publishes the mask to a shared atomic buffer. The NPU runs at ~96% utilization (25ms compute / 42ms period × 2 for double-buffering overhead).

  **(3) Rendering: 60 FPS.** The GPU composites each camera frame with the most recent segmentation mask. Between mask updates (42ms gap = 2-3 frames), the GPU warps the previous mask using optical flow estimated from the camera's gyroscope data (available at 200 Hz, <0.5ms to compute affine transform). The warped mask tracks head movement smoothly.

  **(4) Synchronization.** Use lock-free single-producer/single-consumer queues between pipeline stages. The camera thread never blocks on the ML thread. The render thread never blocks on either. Each stage reads the latest available data — if the ML thread is slow, the render thread uses a slightly older (but warped) mask. No frame drops.

  **(5) Latency budget per render frame:** Camera readout: 1ms (hardware). Gyro-based mask warp: 0.5ms (GPU). Depth-of-field blur: 4ms (GPU compute shader). Composite + display: 2ms (GPU). Total: **7.5ms** — well within the 16.67ms budget. The 25ms ML inference happens in parallel on the NPU and never touches the render budget.

  > **Napkin Math:** Camera: 60 FPS (16.67ms period). ML: 25ms inference → 40 FPS max on NPU, but we target 24 FPS (every 2.5 frames) for thermal headroom. Render: 7.5ms per frame → 133 FPS theoretical, 60 FPS actual (VSync locked). Mask staleness: worst case 42ms (2.5 frames). At arm's length, a head moves ~3 pixels in 42ms — the gyro warp corrects this to <0.5 pixel error. Imperceptible to the user. Power: NPU at 24 FPS × 25ms = 60% duty cycle × 2W = 1.2W. GPU render: 60 FPS × 7.5ms = 45% duty cycle × 1.5W = 0.675W. Total ML+render: 1.875W.

  📖 **Deep Dive:** [Volume I: Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

---

### 🔀 Multi-Modal On-Device AI

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Multi-Modal Sensor Fusion System</b> · <code>architecture</code> <code>sensor-fusion</code></summary>

- **Interviewer:** "Design an on-device system for a fitness app that fuses camera (pose estimation), microphone (rep counting from audio cues), and accelerometer (motion tracking) to provide real-time exercise form feedback. The three sensors run at different rates, produce different data types, and compete for compute resources. How do you architect the fusion?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

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

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

---

### 📱 Device Fragmentation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The 1000-Device Android Fragmentation Problem</b> · <code>deployment</code> <code>optimization</code></summary>

- **Interviewer:** "Your app runs on 1000+ Android device models. The top 10 devices account for 30% of users; the remaining 970 devices account for 70%. You have NPU support on Qualcomm (Hexagon), Samsung (Exynos NPU), MediaTek (APU), and Google (Tensor TPU) — each with different operator support, quantization formats, and performance characteristics. How do you ship one model that works well across all of them?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

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

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

---

### 🧪 Mobile ML Testing

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Device-Free ML Testing Strategy</b> · <code>testing</code> <code>deployment</code></summary>

- **Interviewer:** "Your team ships an ML-powered camera feature to 200 million users across iOS and Android. You don't have a device farm — you have 5 test phones on your desk. How do you test the ML pipeline across devices without a device farm?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Test on our 5 phones and rely on crash reports from production." This means your 200 million users are your test farm. You'll discover bugs from 1-star reviews.

  **Realistic Solution:** Build a **multi-layer testing pyramid** that catches ML failures before they reach users:

  **(1) Unit tests (no device needed).** Test the model in isolation on your CI server (x86 CPU). Run inference on a golden test set (1000 images with known outputs). Assert: output tensor shape is correct, confidence scores sum to 1.0, known inputs produce expected outputs within tolerance (cosine similarity >0.999 vs reference). This catches model conversion errors, quantization bugs, and preprocessing mismatches. Run on every PR.

  **(2) Integration tests (emulator/simulator).** Run the full inference pipeline (preprocessing → model → postprocessing) on iOS Simulator and Android Emulator. These don't have NPU/GPU delegates, so inference runs on CPU — but they catch: memory leaks (run 10,000 consecutive inferences, assert RSS doesn't grow), threading bugs (run inference from multiple threads simultaneously), lifecycle bugs (simulate backgrounding/foregrounding during inference), and API compatibility (test against multiple OS versions).

  **(3) Cloud device testing (no physical farm needed).** Use Firebase Test Lab (Android, 100+ real devices) or AWS Device Farm (iOS + Android). Run a 5-minute smoke test on 30 representative devices: 10 flagships, 10 mid-range, 10 budget. Assert: inference completes without crash, latency is within tier expectations, output matches reference within tolerance. Cost: ~$150/test run. Run on every release candidate.

  **(4) Canary release (production testing).** Release to 1% of users with enhanced telemetry: per-inference latency, confidence distribution, crash rate, and delegate fallback rate. Monitor for 48 hours. Alert thresholds: crash rate >0.1%, median latency >2× baseline, confidence distribution KL divergence >0.1 vs previous version. If any threshold is breached, halt rollout automatically.

  **(5) Synthetic stress testing.** Generate adversarial inputs programmatically: all-black images, all-white images, 1×1 pixel images, maximum-resolution images, images with extreme aspect ratios, corrupted JPEG headers. Assert the model returns gracefully (empty output or error code) instead of crashing or hanging. Run in CI.

  > **Napkin Math:** Testing pyramid coverage: Unit tests catch ~60% of bugs (model conversion, preprocessing). Integration tests catch ~20% (memory, threading, lifecycle). Cloud device tests catch ~15% (hardware-specific failures, delegate bugs). Canary catches ~4% (real-world edge cases). Remaining ~1%: discovered in production. Cost: Unit + integration: $0 (CI time). Cloud devices: $150 × 4 releases/month = $600/month. Canary: $0 (uses production infrastructure). Total: **$600/month** to test across 100+ devices without owning any.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>


### 🕸️ Model Architecture -> System Cost

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Dilated Convolution Penalty</b> · <code>npu-architecture</code></summary>

- **Interviewer:** "You port an image segmentation model (DeepLabV3) to an Android phone. It relies heavily on Atrous (Dilated) Convolutions to expand the receptive field without adding parameters. On the CPU, it takes 200ms. On the NPU, it takes 800ms. Why is the dedicated AI hardware 4x slower than the general-purpose CPU?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming that because an NPU accelerates 'convolutions', it accelerates *all* types of convolutions equally well."

  **Realistic Solution:** You hit a hardware memory-access pattern mismatch. NPUs are highly specialized to read contiguous blocks of data (e.g., standard 3x3 patches) into their systolic arrays. Dilated convolutions, by definition, skip pixels (e.g., reading pixel 1, then pixel 4, then pixel 7). The NPU's DMA engine cannot fetch this non-contiguous data efficiently. It must issue multiple separate, tiny memory read requests, destroying memory bandwidth. Furthermore, many NPU compilers do not natively support dilated math, causing them to secretly 'im2col' the image (copying and expanding the image to make the convolution look standard), which completely exhausts the NPU's tiny local SRAM.

  > **Napkin Math:** A standard 3x3 convolution reads 9 contiguous pixels (or nearby depending on channel layout). A 3x3 convolution with a dilation rate of 4 reads 9 pixels spread across a 9x9 grid. The NPU must fetch `9` separate cache lines from system memory instead of `1` or `2`, plummeting effective memory bandwidth by 80% and causing the ALUs to starve.

  📖 **Deep Dive:** [Volume I: Network Architectures](https://harvard-edge.github.io/cs249r_book_dev/contents/network_architectures/network_architectures.html)

  </details>

</details>


### 🧠 Model Architecture -> System Cost

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-gold?style=flat-square" alt="Level 3" align="center"> The Depthwise Memory Bound</b> · <code>compute-intensity</code></summary>

- **Interviewer:** "To optimize your mobile vision model, you replace all standard 3x3 convolutions with Depthwise Separable Convolutions (like in MobileNet). The total FLOPs (compute operations) decrease by 8x. However, the actual latency on the mobile GPU only improves by 1.5x. Why didn't the 8x reduction in math translate to an 8x reduction in time?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming latency scales linearly with FLOPs, and treating FLOP reduction as the ultimate metric for model optimization."

  **Realistic Solution:** You drastically reduced the arithmetic intensity (FLOPs/byte) and made the layer strictly memory-bound. A standard convolution reuses the same input activation patch across many output channels, heavily leveraging the L1 cache. A depthwise convolution applies one spatial filter to one input channel, meaning it reads a patch of memory, does very little math, and immediately discards it. The mobile GPU spends all its time waiting for the memory bus to fetch the next channel, leaving its ALUs largely idle.

  > **Napkin Math:** Standard 3x3 Conv (64 in, 64 out): Reads `64` values, does `3x3x64x64 = 36,864` MACs. Arithmetic Intensity = High.
  > Depthwise 3x3 Conv (64 in): Reads `64` values, does `3x3x64 = 576` MACs. You reduced the math by `64x`, but you still have to move the exact same amount of input activation data across the memory bus. You hit the 'Memory Wall'.

  📖 **Deep Dive:** [Volume I: Network Architectures](https://harvard-edge.github.io/cs249r_book_dev/contents/network_architectures/network_architectures.html)

  </details>

</details>

---

### 🆕 Advanced Topics

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Snapdragon 8 Elite NPU Scheduling</b> · <code>architecture</code> <code>heterogeneous-compute</code></summary>

- **Interviewer:** "You're running a real-time object detection model on a Snapdragon 8 Elite phone. The Hexagon NPU shares a 6 MB system-level cache (SLC) with the Kryo CPU cores. During inference, the CPU is simultaneously decoding a 4K H.265 video stream for the camera preview. Your NPU inference latency spikes from 5ms to 18ms every few hundred milliseconds. What's happening, and how do you fix it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The NPU is throttling due to heat from the CPU video decode." Thermal throttling takes tens of seconds to onset, not hundreds of milliseconds. The spikes are too fast and too periodic for thermal effects — this is a cache-level problem, not a power problem.

  **Realistic Solution:** The CPU's H.265 decoder is evicting the NPU's weight tiles from the shared 6 MB SLC. The Hexagon NPU stages model weights into the SLC to avoid round-trips to LPDDR5 DRAM. When the CPU decoder flushes the SLC with video reference frames (each I-frame decode touches ~12 MB of reference data, cycling through the 6 MB SLC twice), the NPU's next layer must re-fetch weights from DRAM at ~50 GB/s instead of reading from the SLC at ~200 GB/s — a 4× bandwidth penalty.

  **Fix with a three-part strategy:**

  **(1) SLC partitioning.** Qualcomm's Hexagon SDK exposes SLC partitioning via `HTP_PERF_INFRASTRUCTURE_SLC_PARTITION`. Reserve 2 MB of the SLC exclusively for the NPU. The CPU gets the remaining 4 MB. The NPU's working set per layer is ~1.5 MB (weight tile for a single convolution layer), so 2 MB is sufficient. The CPU decoder's throughput drops ~10% from the reduced cache, but video decode has margin.

  **(2) Inference-decode scheduling.** Align NPU inference with the video decode's frame boundaries. H.265 at 30 FPS produces one frame every 33ms. The CPU decoder bursts for ~8ms then idles for ~25ms. Schedule NPU inference during the idle window. Use `QNN_SIGNAL` to synchronize: the decoder signals completion, the NPU starts inference within the 25ms quiet window.

  **(3) Weight prefetching.** Use the Hexagon DMA engine to prefetch the next layer's weights into the SLC while the current layer is computing. This hides the DRAM latency behind compute. With prefetching, even if the SLC is partially polluted, the NPU pipeline doesn't stall.

  > **Napkin Math:** SLC bandwidth: ~200 GB/s. LPDDR5 bandwidth: ~51 GB/s. Weight tile per layer: 1.5 MB. SLC hit: 1.5 MB / 200 GB/s = 7.5 μs. DRAM miss: 1.5 MB / 51 GB/s = 29 μs. Per-layer penalty on cache miss: 21.5 μs. A 40-layer model accumulates 40 × 21.5 μs = 860 μs ≈ 0.86ms extra per full cache flush. But during an I-frame decode, the SLC thrashes repeatedly across multiple layers, causing 8-12 layers to miss simultaneously → 8 × 21.5 μs × multiple eviction rounds ≈ 10-13ms spike. Matches the observed 5ms → 18ms jump. After SLC partitioning: NPU partition never evicted → stable 5ms.

  📖 **Deep Dive:** [Volume I: Accelerators](https://harvard-edge.github.io/cs249r_book_dev/contents/accelerators/accelerators.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The MediaTek Dimensity APU Architecture</b> · <code>architecture</code> <code>npu-delegation</code></summary>

- **Interviewer:** "You've optimized a speech enhancement model for Qualcomm's Hexagon NPU using QNN. Now you need to ship the same model on MediaTek Dimensity 9300 phones. Your engineer says 'just switch the TFLite delegate from QNN to NeuroPilot.' The model runs but is 4× slower than on Hexagon. What went wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "MediaTek's APU is just slower hardware — buy a faster phone." The Dimensity 9300 APU is rated at 45 TOPS INT8, comparable to Hexagon's 48 TOPS. Raw throughput isn't the problem — the delegation strategy is.

  **Realistic Solution:** Qualcomm's Hexagon and MediaTek's APU have fundamentally different architectures that require different model optimizations:

  **(1) The architecture mismatch.** Hexagon is a single monolithic DSP with a unified vector pipeline — it processes entire layers as single dispatches. MediaTek's APU has separate integer processing units (IPUs) and floating-point processing units (FPUs) connected by an on-chip interconnect. When your model mixes INT8 convolutions with FP16 layer norms (common in speech models), Hexagon handles both in its unified pipeline. On the APU, each transition between INT8 and FP16 requires a data transfer between the IPU and FPU — adding ~0.3ms per transition.

  **(2) Count the transitions.** A typical speech enhancement model (like DCCRN) has 16 encoder layers, each with: INT8 conv → FP16 layer norm → INT8 conv → FP16 LSTM. That's 4 IPU↔FPU transitions per layer × 16 layers = 64 transitions × 0.3ms = 19.2ms of pure data-movement overhead on top of the actual compute.

  **(3) The fix: quantization-aware architecture adaptation.** Convert layer norms to INT8 using quantized layer norm (available in MediaTek's NeuroPilot SDK 7.0+). Replace FP16 LSTMs with INT8 GRUs (fewer gates, fully quantizable). This eliminates IPU↔FPU transitions entirely. The entire model runs on the IPU.

  **(4) Operator fusion differences.** Hexagon's compiler fuses Conv+ReLU+BatchNorm into a single kernel automatically. NeuroPilot requires explicit fusion annotations in the model graph via `mtk_converter` tool. Without these annotations, each operator dispatches separately, adding kernel launch overhead.

  > **Napkin Math:** Hexagon: 48 TOPS INT8, unified pipeline. Speech model (DCCRN): 2.1 GMACs → 2.1 GMACs / 48 TOPS = 0.044ms compute + 0.5ms overhead = 0.55ms total. MediaTek APU: 45 TOPS INT8, split IPU/FPU. Same model: 0.047ms compute + 19.2ms IPU↔FPU transitions + 1.2ms unfused kernel launches = 20.4ms. After fixing (all-INT8 + fusion annotations): 0.047ms compute + 0ms transitions + 0.5ms overhead = 0.55ms. Performance parity restored.

  📖 **Deep Dive:** [Volume I: Accelerators](https://harvard-edge.github.io/cs249r_book_dev/contents/accelerators/accelerators.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Samsung Exynos NPU Fragmentation</b> · <code>architecture</code> <code>fragmentation</code></summary>

- **Interviewer:** "Your app ships a face mesh model to Samsung Galaxy phones. You test on the Galaxy S24 (Exynos 2400) and it works great at 8ms. Then bug reports flood in: Galaxy S22 (Exynos 2200) users get garbled outputs, and Galaxy S21 (Exynos 2100) users see 45ms latency. Same model binary, same TFLite version. How do you ship one model that works across three Exynos generations?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "It's a TFLite bug — file an issue and wait for a patch." The root cause isn't the runtime; it's that Samsung changes NPU architectures across generations more aggressively than Qualcomm, and each generation has different operator support, quantization behavior, and compiler quirks.

  **Realistic Solution:** Understand the generational differences and build around them:

  **(1) Diagnose the per-generation failures.**
  - Exynos 2400 (S24): Samsung's latest NPU with full INT8 support, dynamic shape handling, and a mature compiler. Your model runs natively. 8ms.
  - Exynos 2200 (S22): The NPU uses an AMD RDNA2-derived architecture (Samsung's Xclipse GPU). The NNAPI driver has a known bug with grouped convolutions — it silently produces incorrect output instead of falling back to CPU. This causes garbled face meshes.
  - Exynos 2100 (S21): Older NPU with limited operator coverage. The `HARD_SWISH` activation isn't supported, so the entire model falls back to CPU via NNAPI. CPU inference: 45ms.

  **(2) Build a per-device operator audit.** At app startup, run a micro-benchmark: inference on a known input with a known expected output. Compare the result (cosine similarity > 0.995). If the output is wrong (Exynos 2200 case), blacklist the NPU delegate for this device and fall back to GPU. If the output is correct but slow (Exynos 2100 case), check latency against a threshold.

  **(3) Ship three model variants, select at runtime.**
  - **Model A (flagship):** Full model, INT8, NPU-targeted. For Exynos 2400+.
  - **Model B (safe):** Replace grouped convolutions with standard convolutions (slightly larger, avoids the 2200 bug). INT8, NPU-targeted. For Exynos 2200.
  - **Model C (compatible):** Replace `HARD_SWISH` with `RELU6` (supported everywhere). INT8, GPU-delegated. For Exynos 2100 and older.

  **(4) Delivery via Android App Bundles.** Use Play Feature Delivery to ship only the relevant model variant per device. The APK doesn't bloat — each user downloads only their model (~4 MB each). Use `<dist:device-feature>` targeting in the manifest to match SoC generation.

  > **Napkin Math:** Model A: 4.2 MB, 8ms on Exynos 2400 NPU. Model B: 4.8 MB (grouped→standard conv adds 14% weights), 9ms on Exynos 2200 NPU. Model C: 4.0 MB, 12ms on Exynos 2100 GPU (Mali), 22ms on CPU fallback. Total storage on Play Store: 13 MB (all variants). Per-user download: 4-5 MB (one variant). Micro-benchmark at startup: 50ms (5 inferences × 10ms). Samsung Galaxy market share by generation: S24 (35%), S23 (30%), S22 (20%), S21+ older (15%). Weighted average latency: 0.35×8 + 0.30×8.5 + 0.20×9 + 0.15×12 = **8.95ms**.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Google Tensor G4 TPU Trade-off</b> · <code>architecture</code> <code>power-thermal</code></summary>

- **Interviewer:** "Google claims the Tensor G4 in the Pixel 9 Pro is purpose-built for AI workloads. Your team is choosing between the Pixel 9 Pro (Tensor G4) and the Galaxy S24 Ultra (Snapdragon 8 Gen 3) as the reference device for your on-device ML product. The Tensor G4 TPU is rated at 27 TOPS; the Hexagon NPU at 48 TOPS. Your PM says 'go with Qualcomm, it has more TOPS.' Is that the right call?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "More TOPS = faster inference. Snapdragon wins." TOPS measures peak theoretical throughput under ideal conditions — perfectly parallelizable, perfectly memory-resident workloads. Real models rarely hit peak TOPS. The metric that matters is TOPS actually achieved on your specific model, and TOPS per watt for sustained workloads.

  **Realistic Solution:** Compare the architectures on what actually matters for your workload:

  **(1) Peak TOPS vs sustained TOPS.** The Snapdragon 8 Gen 3 Hexagon NPU hits 48 TOPS for ~10 seconds before thermal throttling reduces it to ~30 TOPS sustained. The Tensor G4 TPU sustains 27 TOPS continuously because Google designed the thermal envelope around sustained AI workloads, not burst benchmarks. For a continuous inference workload (e.g., real-time video processing), the Tensor G4 delivers more consistent performance.

  **(2) TOPS/W efficiency.** Tensor G4 TPU: 27 TOPS at ~3.5W = 7.7 TOPS/W. Snapdragon 8 Gen 3 Hexagon: 48 TOPS at ~8W peak = 6.0 TOPS/W (burst), 30 TOPS at ~5W = 6.0 TOPS/W (sustained). The Tensor G4 is 28% more power-efficient. For a battery-constrained mobile product, this translates directly to longer battery life or more inferences per charge.

  **(3) Workload-specific performance.** Google's TPU is optimized for transformer-based models (attention, layer norms, softmax) because Google designs the hardware for its own ML models (Gemini Nano, speech recognition, computational photography). Qualcomm's Hexagon is more general-purpose and excels at CNNs and traditional vision models. If your product uses a transformer: Tensor G4 likely wins despite lower TOPS. If your product uses a CNN: Hexagon likely wins on raw throughput.

  **(4) Software stack maturity.** Tensor G4 + AI Edge (Google's ML runtime) is a vertically integrated stack — Google controls the hardware, compiler, runtime, and models. Hexagon + QNN is also vertically integrated within Qualcomm's ecosystem, but you're a third-party developer relying on Qualcomm's compiler optimizations. Google's stack tends to optimize faster for new model architectures (they ship the model and the hardware together).

  > **Napkin Math:** Workload: 7B-param LLM (Gemini Nano-class), INT4, decode phase. Weights: 3.5 GB. Tokens/sec = memory bandwidth / weight size per token. Tensor G4 LPDDR5X: 51.2 GB/s → 51.2 / 3.5 = 14.6 tok/s. Snapdragon 8 Gen 3 LPDDR5X: 77 GB/s → 77 / 3.5 = 22 tok/s. Qualcomm wins on bandwidth-bound LLM decode. But for compute-bound prefill (512 tokens): Tensor G4 TPU at 27 TOPS sustained: 512 × 7 GFLOPs / 27 TOPS = 132ms. Hexagon at 30 TOPS sustained: 512 × 7 GFLOPs / 30 TOPS = 119ms. Closer than TOPS suggests. Energy per 100-token response: Tensor G4: 7s × 3.5W = 24.5J. Snapdragon: 4.5s × 8W = 36J. Tensor G4 uses 32% less energy despite being slower.

  📖 **Deep Dive:** [Volume I: Accelerators](https://harvard-edge.github.io/cs249r_book_dev/contents/accelerators/accelerators.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The On-Device LLM Memory Wall</b> · <code>serving</code> <code>memory-hierarchy</code></summary>

- **Interviewer:** "Your product team wants to run a 3B parameter LLM on-device for a smart assistant feature. The target phone has 8 GB LPDDR5 RAM. The OS reserves 3 GB, background apps consume 2 GB, and your app's non-ML code uses 500 MB. Walk me through whether this model fits, and what breaks first — memory, latency, or battery."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Quantize to INT4, that's 1.5 GB, and we have 2.5 GB free. Ship it." This accounts for weights only. The KV-cache, activations, and runtime overhead are invisible killers that push you over the memory cliff — and on mobile, exceeding available memory doesn't cause swapping, it causes the OS to kill your app.

  **Realistic Solution:** Build a complete memory budget, then engineer around the gaps:

  **(1) Memory accounting.** Available RAM: 8 GB - 3 GB (OS) - 2 GB (background) - 0.5 GB (app) = **2.5 GB for ML**. Model weights (INT4, group quantization with FP16 scales): 3B params × 0.5 bytes + 3B/128 groups × 2 bytes scales = 1.5 GB + 47 MB = **1.55 GB**. KV-cache at 2048 context (FP16, 32 layers, 32 heads, 128 dim): 2 × 32 × 32 × 128 × 2048 × 2 bytes = **1.07 GB**. Activations per token: ~100 MB. Runtime overhead (memory allocator, graph executor): ~50 MB. **Total: 2.77 GB. Exceeds budget by 270 MB.**

  **(2) What breaks.** On iOS, Jetsam kills your app when memory pressure exceeds the per-app limit (~2.8 GB on iPhone 15 with 6 GB RAM, ~4 GB on iPhone 15 Pro with 8 GB). On Android, the LMK (Low Memory Killer) terminates your process. You don't get a warning — the app just disappears mid-sentence. Users see a crash.

  **(3) Fitting the model.** Attack the KV-cache first — it's the largest variable component. (a) INT8 KV-cache: 1.07 GB → 535 MB. Quality loss: <0.5% on benchmarks. (b) Grouped-Query Attention (GQA) with 8 KV-heads instead of 32: 535 MB / 4 = **134 MB**. (c) Sliding window attention (1024 tokens): 134 MB / 2 = **67 MB**. Use memory-mapped weights (mmap): the OS pages in only the layers currently executing. Resident weight footprint drops from 1.55 GB to ~200-300 MB (the active layer set). New total: 300 MB (resident weights) + 67 MB (KV) + 100 MB (activations) + 50 MB (runtime) = **517 MB**. Fits with 2 GB headroom.

  **(4) What breaks next: latency.** Decode is memory-bandwidth bound. Each token loads all 1.55 GB of weights from DRAM (even with mmap, the working set cycles through all layers). LPDDR5 at 51.2 GB/s: 1.55 GB / 51.2 GB/s = 30.3ms per token = **33 tokens/second**. Acceptable for a chat assistant (human reading speed is ~4 tokens/second).

  **(5) What breaks last: battery.** LPDDR5 power at full bandwidth: ~3W. SoC compute: ~2W. Total: ~5W during decode. A 100-token response takes 3 seconds → 15J → 0.03% of a 15 Wh battery. Sustainable for occasional queries. For continuous conversation (e.g., 30 minutes), that's 5W × 1800s = 9 kJ = 2.5 Wh = 17% battery. Significant but acceptable.

  > **Napkin Math:** Available: 2.5 GB. Naive budget: 2.77 GB (over by 270 MB → app killed). Optimized budget: 517 MB (under by 2 GB → safe). Decode latency: 30.3ms/token → 33 tok/s. Prefill 512 tokens at 35 TOPS INT4: 512 × 6 GFLOPs / 70 TOPS = 44ms. Time-to-first-token: 44ms. Battery per 100-token response: 3s × 5W = 15J = 0.03%. Battery for 30-min conversation: 17%.

  📖 **Deep Dive:** [Volume I: Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The CoreML vs TFLite Performance Gap</b> · <code>frameworks</code> <code>npu-delegation</code></summary>

- **Interviewer:** "You benchmark the same MobileNetV3-Large model on an iPhone 15 Pro using Core ML and on a Pixel 8 Pro using TFLite. Core ML reports 1.8ms; TFLite reports 5.2ms. Your manager concludes 'Apple hardware is 3× faster.' Then you run TFLite on the iPhone and Core ML on the Pixel (hypothetically, via ONNX Runtime with each EP). Now TFLite on iPhone gets 4.1ms and the CoreML EP on Pixel gets 6.8ms. What's actually happening?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Apple's Neural Engine is just faster silicon." The A17 Pro ANE is rated at 35 TOPS; the Tensor G3 TPU is rated at 22 TOPS. That's a 1.6× hardware gap, not 3×. The extra performance comes from the software stack, not the silicon.

  **Realistic Solution:** The performance gap is dominated by **compiler optimization depth**, not hardware speed:

  **(1) Core ML's unfair advantage on Apple silicon.** Core ML is co-designed with the ANE. Apple's compiler performs: (a) Layer fusion — Conv+BN+ReLU becomes a single hardware dispatch. MobileNetV3's 15 inverted residual blocks each have 3 fusible sequences = 45 fusions. Each fusion saves ~0.02ms of dispatch overhead = 0.9ms saved. (b) Weight layout transformation — weights are pre-arranged in the ANE's native tiled format at compile time, not at runtime. (c) Memory planning — the compiler pre-allocates a single contiguous activation buffer and reuses it across layers (in-place operation). Zero malloc during inference.

  **(2) TFLite's generality tax on Pixel.** TFLite targets dozens of hardware backends through NNAPI or vendor-specific delegates. The QNN delegate for Hexagon (or Google's AI Edge delegate for Tensor) adds an abstraction layer. Graph partitioning splits the model between NPU-supported and CPU-fallback ops. Each partition boundary requires a memory copy between NPU and CPU address spaces. MobileNetV3 with `HARD_SWISH` activations: if the NPU doesn't support `HARD_SWISH` natively, each occurrence (15 in MobileNetV3) creates a partition boundary → 15 round-trips × 0.15ms = 2.25ms of pure overhead.

  **(3) The cross-framework test proves it.** TFLite on iPhone (4.1ms vs Core ML's 1.8ms): same hardware, 2.3× slower — the gap is purely software. The ANE hardware is identical; TFLite simply can't access Apple's proprietary compiler optimizations. CoreML EP on Pixel (6.8ms vs TFLite's 5.2ms): ONNX Runtime's CoreML execution provider doesn't exist on Android — this would actually use a generic backend, proving the framework matters more than the hardware.

  > **Napkin Math:** A17 Pro ANE: 35 TOPS. Tensor G3 TPU: 22 TOPS. Hardware gap: 1.6×. Observed gap (Core ML vs TFLite): 1.8ms vs 5.2ms = 2.9×. Software contribution: 2.9 / 1.6 = 1.8× from compiler optimization. Breakdown of TFLite's 5.2ms on Pixel: ~2.1ms compute (hardware) + 2.25ms partition boundary overhead + 0.85ms dispatch/scheduling. Breakdown of Core ML's 1.8ms on iPhone: ~1.3ms compute (hardware) + 0ms partition overhead (full graph on ANE) + 0.5ms dispatch. If TFLite eliminated partition overhead on Pixel: 2.1 + 0.85 = 2.95ms → gap narrows to 1.6× (matching the hardware ratio).

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> The App Store Binary Size Limit</b> · <code>deployment</code> <code>model-compression</code></summary>

- **Interviewer:** "Apple's App Store enforces a 200 MB limit for downloads over cellular. Your iOS app is 60 MB without the ML model. Your product requires an on-device image generation model (Stable Diffusion-like) with INT8 weights totaling 800 MB. How do you ship this to users without exceeding the cellular download limit?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Compress the model to fit in 140 MB (200 - 60)." You cannot compress an 800 MB model to 140 MB without catastrophic quality loss. INT8 is already quantized — further compression (INT4) would halve it to 400 MB, still 3× over budget. And you'd lose significant image quality.

  **Realistic Solution:** Use Apple's **On-Demand Resources (ODR)** and a staged download architecture:

  **(1) Separate the model from the app binary.** The 60 MB app ships through the App Store normally (under the 200 MB cellular limit). The 800 MB model is tagged as an On-Demand Resource with `NSBundleResourceRequest`. It downloads in the background over WiFi after the user first opens the app. The user sees a progress bar: "Downloading AI model (800 MB) — requires WiFi."

  **(2) Chunk the model for progressive availability.** Split the 800 MB model into functional chunks: (a) Text encoder: 120 MB — downloads first, enables text understanding immediately. (b) UNet (denoising): 520 MB — the core generation engine. (c) VAE decoder: 80 MB — converts latent space to pixels. (d) Safety classifier: 30 MB. Download priority: safety classifier first (30 MB, enables content filtering), then UNet + VAE (600 MB, enables generation), then text encoder refinement. The user can start generating with a simpler text pipeline while the full encoder downloads.

  **(3) On-device storage management.** ODR files are purgeable — iOS can delete them under storage pressure. Protect against this: (a) Pin the model using `beginAccessingResources()` and never call `endAccessingResources()` while the app is active. (b) Store a backup copy in the app's Documents directory (not purgeable) after first download. (c) Check model integrity at launch with a SHA-256 checksum (fast: 800 MB at 2 GB/s = 400ms).

  **(4) Android equivalent.** Use Play Asset Delivery with "fast-follow" delivery mode. The model downloads immediately after app install. Play's asset packs support up to 2 GB per pack. Same chunking strategy applies.

  > **Napkin Math:** App binary: 60 MB (under 200 MB cellular limit ✓). Model: 800 MB via ODR (WiFi only). Total on-device footprint: 860 MB. Download time on 50 Mbps WiFi: 800 MB / 6.25 MB/s = 128 seconds ≈ 2 minutes. If you tried to fit everything in 200 MB: 200 - 60 = 140 MB for model. 800 MB → 140 MB requires 5.7× compression. INT8 → INT2 would give 4× (200 MB) — still over, and INT2 quality is unusable for image generation. Alternative: INT4 (400 MB) + pruning 50% (200 MB) + gzip (140 MB). Quality loss: ~40% FID degradation. Not viable.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Background Inference Power Budget</b> · <code>power-thermal</code> <code>battery</code></summary>

- **Interviewer:** "Your health app runs a heart rhythm classification model on Apple Watch sensor data forwarded to the paired iPhone. iOS gives background apps approximately 30 seconds of CPU time per background fetch cycle, and the system schedules these opportunistically (roughly every 15-30 minutes). The model takes 12ms per inference on the ANE. How many inferences can you run per background cycle, and what's the energy cost of running on ANE vs CPU?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "30 seconds / 12ms = 2,500 inferences on ANE. Easy." The 30-second budget is wall-clock time, not compute time. You must account for: model loading, ANE warm-up, data preprocessing, and the fact that iOS may terminate your background task early if the system is under resource pressure.

  **Realistic Solution:** Build a realistic background inference pipeline with defensive time management:

  **(1) Time budget breakdown.** 30 seconds wall-clock. Model load from flash (if not cached): ~800ms for a 4 MB CoreML model. ANE warm-up (first inference is 3-5× slower due to compilation): ~60ms. Data fetch from HealthKit: ~200ms for 30 minutes of heart rate data. Preprocessing (resampling, windowing): ~100ms. Safety margin (iOS can reclaim early): reserve 5 seconds. **Effective inference window: 30 - 0.8 - 0.06 - 0.2 - 0.1 - 5 = 23.84 seconds.**

  **(2) Inferences per cycle.** Each inference classifies a 30-second ECG window. 30 minutes of data = 60 windows. At 12ms per inference: 60 × 12ms = 720ms. Well within the 23.84s budget. You can process all accumulated data with 23.12 seconds to spare. Use the remaining time for: result persistence to Core Data (50ms), complication update on Apple Watch (100ms), and optional: batch-process older unanalyzed data.

  **(3) ANE vs CPU energy comparison.** ANE: 12ms × 60 inferences = 720ms active. ANE power: ~1W during inference. Energy: 1W × 0.72s = **0.72J**. CPU (A16, performance core): same model runs in ~85ms per inference. 85ms × 60 = 5.1s active. CPU power: ~3.5W during inference. Energy: 3.5W × 5.1s = **17.85J**. The ANE uses **24.8× less energy** for the same workload.

  **(4) Battery impact.** iPhone 15 battery: 12.8 Wh = 46,080J. ANE path: 0.72J per cycle × 48 cycles/day (every 30 min) = 34.6J/day = **0.075% battery/day**. CPU path: 17.85J × 48 = 856.8J/day = **1.86% battery/day**. The ANE path is invisible to the user. The CPU path would show up in Settings → Battery as a noticeable consumer.

  > **Napkin Math:** Background budget: 30s wall-clock → 23.84s effective. Data: 60 windows × 12ms = 720ms on ANE. Utilization: 720ms / 23,840ms = 3% of budget. Energy: ANE = 0.72J vs CPU = 17.85J (24.8× difference). Daily battery: ANE = 0.075% vs CPU = 1.86%. Annual energy: ANE = 12.6 kJ vs CPU = 312.7 kJ. If 10M users run this app: ANE saves 300 MJ/year of collective battery energy.

  📖 **Deep Dive:** [Volume I: Accelerators](https://harvard-edge.github.io/cs249r_book_dev/contents/accelerators/accelerators.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 6+" align="center"> The Federated Learning Communication Cost</b> · <code>training</code> <code>battery</code></summary>

- **Interviewer:** "You're building federated learning for a keyboard prediction model (30M parameters, FP32) across 100 million Android devices. Each training round selects 5,000 devices. Walk me through the communication cost, cellular data usage, and battery drain per training round per device. At what point does the user notice?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Upload the full gradient — 30M × 4 bytes = 120 MB per device. It's just one upload." 120 MB over cellular costs the user real money in many markets. In India, 1 GB of mobile data costs ~$0.17. You're asking users to spend $0.02 per training round without their knowledge. At 4 rounds/day, that's $2.40/month — more than many users' entire data plans.

  **Realistic Solution:** Engineer the communication stack to be invisible to the user:

  **(1) Baseline cost.** Model: 30M params × 4 bytes (FP32 gradients) = **120 MB upload per device per round**. Server receives: 5,000 × 120 MB = 600 GB per round. At 4 rounds/day: 2.4 TB/day server ingress. Completely impractical.

  **(2) Compression stack — reduce 120 MB to < 1 MB.**
  - Top-k sparsification (k=0.1%): send only the 30,000 largest gradient values + 30,000 INT32 indices. 30K × 4 bytes (values) + 30K × 4 bytes (indices) = 240 KB. Convergence preserved via error feedback (accumulate unsent gradients locally).
  - Stochastic quantization of gradient values: FP32 → INT8. 240 KB → 90 KB (values shrink 4×, indices unchanged).
  - Entropy coding (Huffman on the sparse indices): 90 KB → ~60 KB.
  - Add metadata (model version, device fingerprint, round ID): +2 KB.
  - **Total upload: ~62 KB per device per round.**

  **(3) Download cost.** The aggregated model update (server → device): 30M × 4 bytes = 120 MB if sent naively. But send only the delta from the previous round: typically 5-10% of weights change significantly → 6-12 MB. With INT8 quantization + entropy coding: **~2 MB download per round**.

  **(4) Cellular data budget.** Upload: 62 KB × 4 rounds/day = 248 KB/day = **7.4 MB/month**. Download: 2 MB × 4 rounds/day = 8 MB/day = **240 MB/month**. Total: **247.4 MB/month**. In the US (average plan: 15 GB), this is 1.6% of the data plan. In India (average plan: 1.5 GB), this is 16.5% — still too high. Solution: only train on WiFi in markets with expensive cellular data. Use Android's `ConnectivityManager` to check `NetworkCapabilities.NET_CAPABILITY_NOT_METERED`.

  **(5) Battery drain per round.** Upload 62 KB over WiFi: ~0.01J (radio power: 0.5W × 0.02s). Download 2 MB: ~0.1J. On-device training (50 local SGD steps on 30M params): ~45 seconds on CPU at 2W = 90J. Total per round: **~90J**. Battery: 90J / 46,080J (typical 12.8 Wh phone) = **0.2% per round**. At 4 rounds/day: 0.8%/day. Users won't notice if training runs during charging.

  **(6) When the user notices.** Battery: >2%/day of unexplained drain triggers user investigation. Data: >500 MB/month triggers carrier warnings in many markets. Thermal: if training runs while the user is actively using the phone, the SoC heats up noticeably. **Constraint: train only when charging + WiFi + idle + cool.**

  > **Napkin Math:** Naive upload: 120 MB → compressed: 62 KB (1,935× reduction). Monthly data: 247 MB (WiFi-only in emerging markets). Battery per round: 90J = 0.2%. Daily: 0.8%. Training dominates communication by 1000×: 90J training vs 0.11J communication. Server cost: 5,000 × 62 KB = 310 MB ingress per round. 4 rounds/day = 1.24 GB/day. At $0.01/GB: $4.50/month server bandwidth. Convergence: ~200 rounds with error feedback ≈ 50 days to match centralized baseline within 2%.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Neural Engine Quantization Cliff</b> · <code>quantization</code> <code>architecture</code></summary>

- **Interviewer:** "You're deploying a 1B parameter language model on-device. Apple's ANE supports INT8 and FP16 but not INT4. Qualcomm's Hexagon NPU supports INT4 natively. Your model at FP16 is 2 GB, at INT8 is 1 GB, and at INT4 is 500 MB. The phone has 6 GB RAM with 2 GB available. Walk me through the quantization decision for each platform and where the 'cliff' is."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use INT4 everywhere for smallest size. On Apple, just dequantize INT4 to INT8 at runtime." Runtime dequantization INT4→INT8 on the ANE doesn't work the way you'd hope. The ANE can't execute INT4 operations — the CoreML compiler will route INT4-dequantized layers to the CPU or GPU instead of the ANE, destroying your latency.

  **Realistic Solution:** Each platform has a different optimal quantization point, and the "cliff" is where you fall off the NPU:

  **(1) Apple ANE — the INT8 ceiling.** The ANE's compute units are wired for INT8 and FP16 multiply-accumulate. INT4 weights must be dequantized to INT8 before the ANE can process them. CoreML's `MLModelConfiguration` with `computeUnits = .cpuAndNeuralEngine` will silently route INT4 layers to the CPU. Result: your "INT4 model" runs the dequantization on CPU (slow), then the INT8 matmul on ANE, then transfers back. The CPU↔ANE data transfer overhead per layer (~0.1ms) × 24 transformer layers = 2.4ms overhead. Your 1B model at INT8 on ANE: 1 GB weights, 15ms inference. Same model with INT4 weights + runtime dequant: 500 MB weights but 15ms + 2.4ms dequant + CPU overhead = **22ms**. You saved memory but lost 47% latency.

  **(2) Qualcomm Hexagon — INT4 native.** The Hexagon v75+ NPU has native INT4 MAC units. INT4 weights are processed directly — no dequantization step. 1B model at INT4: 500 MB weights, memory-bandwidth bound decode. LPDDR5 at 51 GB/s: 500 MB / 51 GB/s = 9.8ms per token. At INT8: 1 GB / 51 GB/s = 19.6ms per token. INT4 is genuinely 2× faster on Hexagon because it halves the memory traffic.

  **(3) The quantization cliff.** On Apple: INT8 → INT4 saves 500 MB RAM but costs 7ms latency (47% regression). The cliff is at INT4 — you fall off the ANE. On Qualcomm: INT8 → INT4 saves 500 MB RAM AND gains 2× speed. No cliff — INT4 is the sweet spot. On both: INT8 → FP16 doubles memory (1 GB → 2 GB) for marginal quality improvement (~0.3% perplexity). Not worth it on a memory-constrained phone.

  **(4) The right strategy.** Ship two model variants: INT8 for Apple devices (stays on ANE, 1 GB), INT4 for Qualcomm devices (native INT4, 500 MB, faster). Use platform detection at runtime. The Apple variant uses 500 MB more RAM but runs at full ANE speed. The Qualcomm variant is smaller AND faster — a rare win-win.

  > **Napkin Math:** Apple ANE (1B model): FP16 = 2 GB, 30ms (fits but slow). INT8 = 1 GB, 15ms (sweet spot ✓). INT4 = 500 MB, 22ms (memory savings but latency cliff ✗). Qualcomm Hexagon (1B model): FP16 = 2 GB, 39ms. INT8 = 1 GB, 19.6ms. INT4 = 500 MB, 9.8ms (sweet spot ✓). Memory budget with 2 GB available: FP16 doesn't fit on either (2 GB weights + KV-cache + activations > 2 GB). INT8 fits on both (1 GB + ~300 MB overhead = 1.3 GB). INT4 fits comfortably on Qualcomm (500 MB + 300 MB = 800 MB). Accuracy: INT8 = 0.5% perplexity increase vs FP16. INT4 (GPTQ) = 1.2% increase. Both acceptable for on-device assistant use cases.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Camera Pipeline Memory Contention</b> · <code>memory-hierarchy</code> <code>sensor-pipeline</code></summary>

- **Interviewer:** "Your app runs a semantic segmentation model on live camera frames for an AR furniture placement feature. The phone's ISP (Image Signal Processor) is simultaneously processing raw Bayer sensor data into RGB frames. Both the ISP and your ML model compete for LPDDR5 DRAM bandwidth. On a Snapdragon 8 Gen 3 with 51.2 GB/s total DRAM bandwidth, your ML inference latency is stable at 8ms when the camera is paused but spikes to 14ms during live preview. Why?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The ISP and ML model are on different hardware blocks, so they don't interfere." They are on different compute blocks, but they share the same DRAM bus. Memory bandwidth is a shared, finite resource — and on mobile SoCs, it's the most contested resource in the system.

  **Realistic Solution:** Map out the bandwidth consumers and engineer around the contention:

  **(1) ISP bandwidth consumption.** The camera sensor outputs 48 MP raw Bayer frames at 30 FPS. Each raw frame: 48M pixels × 10 bits = 60 MB. ISP reads: 60 MB per frame (raw input). ISP writes: 12 MP RGB output (after pixel binning) = 12M × 3 bytes = 36 MB per frame. ISP also reads reference frames for temporal noise reduction (TNR): 2 previous frames × 36 MB = 72 MB. ISP total bandwidth: (60 + 36 + 72) MB × 30 FPS = **5.04 GB/s**.

  **(2) ML model bandwidth consumption.** Segmentation model (DeepLabV3-MobileNet, INT8): 6.2M parameters = 6.2 MB weights. Input: 512×512×3 = 786 KB. Activations (peak): ~15 MB. Total memory traffic per inference: ~45 MB (weights + activations + intermediate buffers, accounting for read/write). At 30 FPS: 45 MB × 30 = **1.35 GB/s**.

  **(3) Other bandwidth consumers.** Display refresh (1080p @ 120 Hz): 1080 × 2400 × 4 bytes × 120 = **1.24 GB/s**. GPU (UI rendering): ~**1.5 GB/s**. CPU (app logic, OS): ~**0.8 GB/s**. Total system bandwidth demand: 5.04 + 1.35 + 1.24 + 1.5 + 0.8 = **9.93 GB/s**.

  **(4) Why 51.2 GB/s isn't enough.** The 51.2 GB/s is theoretical peak. Effective bandwidth with bank conflicts, refresh cycles, and arbitration overhead: ~35 GB/s usable. At 9.93 GB/s demand, utilization is 28% — should be fine. But bandwidth isn't uniform across time. The ISP bursts at the start of each frame: it reads the entire 60 MB raw frame in ~2ms (30 GB/s burst). During that 2ms window, the ISP consumes 59% of usable bandwidth. If your ML inference's memory-intensive layers (early convolutions with large activation maps) overlap with the ISP burst, the NPU's memory requests queue behind the ISP's, adding 4-6ms of stall time.

  **(5) The fix: temporal scheduling.** Use the camera's frame timestamp callback to schedule ML inference in the ISP's idle window. ISP bursts for ~2ms at the start of each 33ms frame period, then idles for ~31ms. Start ML inference 3ms after the frame timestamp (ISP burst complete). Your 8ms inference fits in the 31ms quiet window with 23ms margin. Alternatively, reduce ISP bandwidth: disable TNR (saves 72 MB × 30 = 2.16 GB/s) if the scene is well-lit, or reduce raw capture to 12 MP (saves 75% of ISP read bandwidth).

  > **Napkin Math:** Total DRAM: 51.2 GB/s theoretical, ~35 GB/s effective. ISP burst: 30 GB/s for 2ms per frame. ML steady-state: 1.35 GB/s. During ISP burst overlap: ML bandwidth drops from 1.35 GB/s to ~0.6 GB/s (ISP has higher QoS priority). ML inference stretches: 8ms × (1.35/0.6) = 18ms worst case. Observed: 14ms (partial overlap — not all ML layers hit the burst window). After temporal scheduling: ML avoids ISP burst entirely → stable 8ms.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Thermal Throttling Prediction</b> · <code>power-thermal</code> <code>latency</code></summary>

- **Interviewer:** "Your video editing app runs a style transfer model at 30 FPS. For the first 45 seconds, inference takes a steady 12ms per frame on the Snapdragon 8 Gen 3 NPU. Then latency gradually climbs to 24ms over the next 30 seconds and stabilizes there. The user complains about 'stuttering after a minute.' You can't make the model faster. How do you deliver a smooth user experience despite thermal throttling?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Add a bigger heat sink or tell users to take breaks." You can't change the phone's thermal design, and telling users to stop using the feature is not a product solution. The mistake is treating thermal throttling as a hardware problem when it's a software scheduling problem.

  **Realistic Solution:** Build a **thermal-aware inference scheduler** that predicts throttling and adapts before the user notices:

  **(1) Model the thermal behavior.** Mobile SoC thermal throttling follows a first-order exponential: $T(t) = T_{ambient} + P \cdot R_{thermal} \cdot (1 - e^{-t/\tau})$, where $\tau$ is the thermal time constant (~30-60 seconds for a phone SoC) and $R_{thermal}$ is the thermal resistance. At steady-state power P = 4W (NPU at full utilization) and $R_{thermal}$ ≈ 12°C/W: junction temperature rises 48°C above ambient. Throttling threshold: typically 85°C. At 25°C ambient: you hit 85°C at $t = -\tau \cdot \ln(1 - 60/48)$... but 60 > 48, so you always throttle at steady state. Time to first throttle (at 73°C, where the governor starts reducing frequency): $t = -45 \cdot \ln(1 - 48/48 \cdot (73-25)/(85-25))$ ≈ 45 seconds. Matches the observed behavior.

  **(2) Proactive quality scaling.** Instead of running at full quality until throttling forces degradation (jarring stutter), smoothly reduce quality before the thermal limit:
  - **0-30 seconds:** Full model, 12ms, 30 FPS. Thermal headroom is ample.
  - **30-45 seconds:** Reduce model input resolution from 1080p to 720p. Inference drops to 7ms. Power drops from 4W to 2.5W. Thermal rise slows. The quality reduction is subtle on a phone screen.
  - **45-90 seconds:** If temperature is still rising, switch to a lightweight model variant (half the channels). Inference: 4ms at 720p. Power: 1.5W. Thermal equilibrium reached below throttle threshold.
  - **90+ seconds:** Sustained indefinitely at reduced quality without throttling.

  **(3) Thermal monitoring API.** On Android: read `/sys/class/thermal/thermal_zone*/temp` or use `PowerManager.getThermalStatus()` (API 29+). On iOS: `ProcessInfo.thermalState` reports `.nominal`, `.fair`, `.serious`, `.critical`. Trigger quality transitions at `.fair` (before `.serious` causes system-level throttling).

  **(4) Smooth transitions.** Cross-fade between quality levels over 500ms (15 frames). The user perceives a gradual softening, not a jarring switch. A/B testing shows users prefer consistent 720p over alternating 1080p/stuttering.

  > **Napkin Math:** Thermal time constant τ ≈ 45s. Full power: 4W → throttle at 45s. Half power: 2W → thermal equilibrium at 49°C above ambient = 74°C (below 85°C threshold) → never throttles. Strategy: burst at 4W for 30s (high quality), then sustain at 2W indefinitely (medium quality). Effective user experience: 30s of premium quality + unlimited medium quality vs. 45s of premium + degraded stuttering quality. Latency profile: Option A (no thermal management): 12ms for 45s → 24ms forever (drops to 15 FPS, visible stutter). Option B (thermal-aware): 12ms for 30s → 7ms forever at 720p (locked 30 FPS, smooth). User satisfaction: Option B scores 4.2/5 vs Option A at 2.8/5 in internal testing.

  📖 **Deep Dive:** [Volume I: Accelerators](https://harvard-edge.github.io/cs249r_book_dev/contents/accelerators/accelerators.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Cross-Platform Model Optimization</b> · <code>deployment</code> <code>compiler-runtime</code></summary>

- **Interviewer:** "Your company ships a speech recognition model to iOS (Core ML + ANE), Android Qualcomm (QNN + Hexagon), and Android Samsung (NNAPI + Exynos NPU). Each platform has different quantization support, operator coverage, and compiler optimizations. How many optimized model binaries do you actually need, and what does the build pipeline look like?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Three binaries — one per platform." Three is the minimum if every platform had identical operator support and quantization. In reality, you need more because of operator coverage gaps and quantization format differences across NPU generations.

  **Realistic Solution:** You need **5-7 optimized binaries** from a single PyTorch source model, managed by an automated build pipeline:

  **(1) Inventory the platform constraints.**
  - **iOS (Core ML + ANE):** Supports INT8 (per-channel), FP16. No INT4. Fuses Conv+BN+ReLU natively. Requires `coremltools` conversion. ANE has a 500-layer limit and doesn't support dynamic shapes.
  - **Qualcomm flagship (QNN + Hexagon v75):** Supports INT4, INT8, INT16, FP16. Fuses Conv+ReLU but not BN (must be folded manually). Requires QNN SDK conversion. Supports dynamic batch but not dynamic sequence length.
  - **Qualcomm mid-range (QNN + Hexagon v73):** INT8 and FP16 only. Older compiler with fewer fusion patterns. Some operators (e.g., `GELU`) fall back to CPU.
  - **Samsung flagship (NNAPI + Exynos 2400):** INT8, FP16. NNAPI 1.3 operator set — missing `HARD_SWISH`, `GELU`. Grouped convolutions have known accuracy bugs.
  - **Samsung mid-range (NNAPI + Exynos 1380):** INT8 only. NNAPI 1.2 — even smaller operator set. No attention operators.
  - **CPU fallback (XNNPACK):** INT8, FP32. Runs everywhere. Slowest but most compatible.

  **(2) The binary matrix.**
  | Binary | Target | Quantization | Special Handling |
  |--------|--------|-------------|-----------------|
  | 1 | iOS ANE | INT8 per-channel | CoreML format, static shapes |
  | 2 | Hexagon v75 | INT4 (weights) + INT8 (activations) | QNN format, GELU native |
  | 3 | Hexagon v73 | INT8 | QNN format, GELU→RELU approx |
  | 4 | Exynos 2400 | INT8 | NNAPI, HARD_SWISH→RELU6, no grouped conv |
  | 5 | Exynos mid-range | INT8 | NNAPI 1.2, simplified attention |
  | 6 | CPU fallback | INT8 | XNNPACK, universal compatibility |

  **(3) Automated build pipeline.** Source: PyTorch model (FP32). Step 1: Export to ONNX (canonical intermediate format). Step 2: Platform-specific conversion scripts (one per binary). Step 3: Accuracy validation — run each binary on a golden test set, assert output cosine similarity > 0.995 vs FP32 reference. Step 4: Latency benchmarking on target hardware (CI farm or cloud device lab). Step 5: Package into platform-specific delivery (iOS App Bundle, Android App Bundle with feature splits).

  **(4) Runtime selection.** On Android, detect the SoC at startup using `Build.SOC_MODEL` or `Build.HARDWARE`. Map to the appropriate binary. Fallback chain: platform-specific NPU binary → CPU binary. On iOS: single binary (Apple controls the hardware).

  > **Napkin Math:** Build pipeline: 1 source model → 6 binaries. Conversion time: ~5 minutes per binary (automated). Validation: 6 × 1000 test samples × 50ms = 5 minutes. Total CI time: ~35 minutes per model update. Storage: 6 binaries × ~15 MB average = 90 MB total. Per-user download: 15 MB (one binary via conditional delivery). Engineering cost to set up pipeline: ~2 engineer-weeks. Ongoing maintenance: ~2 engineer-days per quarter (when new SoCs launch). Performance range: 4ms (iOS ANE) to 45ms (CPU fallback). 11× spread across the same model.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The On-Device Personalization Privacy</b> · <code>privacy</code> <code>training</code></summary>

- **Interviewer:** "Your photo search app needs to learn each user's personal categories — 'my dog,' 'my office,' 'my kids' soccer games.' You'll fine-tune a CLIP-like vision encoder (150M params, FP16 = 300 MB) on-device using LoRA. The adapted weights never leave the device. Walk me through the LoRA adapter sizing, training memory, time to convergence, and what happens when the user gets a new phone."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "LoRA is tiny, so training is free." LoRA reduces trainable parameters dramatically, but training still requires: forward pass through the full frozen model (300 MB resident), backward pass through LoRA layers (storing activations for all layers), and an optimizer state (Adam stores 2× the trainable parameters). The memory cost is dominated by the frozen model and activations, not the LoRA weights.

  **Realistic Solution:** Design the full on-device training pipeline with realistic resource accounting:

  **(1) LoRA adapter sizing.** CLIP vision encoder: 12 transformer layers, each with Q/K/V/O projections (768 × 768). LoRA at rank 4: each adapter = 2 matrices of (768 × 4) and (4 × 768). Per projection: 768 × 4 + 4 × 768 = 6,144 parameters. 4 projections × 12 layers = 48 adapters × 6,144 = **294,912 trainable parameters**. Storage: 295K × 2 bytes (FP16) = **590 KB**. Negligible.

  **(2) Training memory budget.** Frozen model weights (mmap'd, ~200 MB resident): 200 MB. LoRA trainable weights: 590 KB. Activations for backward pass (stored per layer): 12 layers × batch_size × seq_len × hidden_dim × 2 bytes. For batch=1, 50 image patches (ViT): 12 × 1 × 50 × 768 × 2 = 921 KB. Adam optimizer state (m and v for each trainable param): 295K × 2 × 2 bytes = 1.18 MB. Gradients: 590 KB. **Total training memory: ~203 MB**. Fits on any phone with 4+ GB RAM.

  **(3) Training data and convergence.** Source: user's photo library with implicit labels (album names, faces, locations, user search corrections). Typical user: 5,000 photos, 20-50 personal categories. Training set: ~500 (image, text) pairs generated from metadata. Training: 50 epochs × 500 pairs / batch_size 8 = 3,125 gradient steps. Per step on ANE: forward (12ms) + backward (18ms) = 30ms. Total training time: 3,125 × 30ms = **93.75 seconds ≈ 1.5 minutes**. Power: 2W × 94s = 188J = 0.4% of a 12.8 Wh battery. Schedule during overnight charging.

  **(4) New phone migration.** The LoRA adapter (590 KB) is backed up to iCloud/Google Drive as part of app data backup. On the new phone: download the base model (same version from the app), apply the 590 KB adapter. Instant personalization without retraining. If the base model version changes (app update with new model): the old LoRA adapter is incompatible. Trigger a retraining cycle on the new phone — 1.5 minutes, invisible to the user during first overnight charge.

  **(5) Privacy analysis.** The 590 KB adapter encodes correlations between visual features and text labels. Without the base model and the user's photos, the adapter is meaningless — it's a set of low-rank perturbations to attention projections. Even with the base model, reconstructing training images from LoRA weights is computationally infeasible (the adapter compresses 500 images into 295K parameters — a 10,000× information bottleneck). Differential privacy isn't needed because the weights never leave the device.

  > **Napkin Math:** LoRA params: 295K (0.2% of 150M). Adapter size: 590 KB. Training memory: 203 MB. Training time: 94 seconds. Battery: 0.4%. Migration: 590 KB backup. Personalization gain: +15% recall on user-specific categories (internal benchmark). Retraining frequency: weekly or on 50+ new photos. Annual training cost per user: 52 weeks × 188J = 9.8 kJ = 0.27% of annual phone energy budget (assuming 10 Wh/day usage).

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The WebGPU ML Inference</b> · <code>frameworks</code> <code>deployment</code></summary>

- **Interviewer:** "Your team debates shipping an ML feature as a native mobile SDK (Core ML / TFLite) vs running it in the browser via WebGPU. The PM argues 'WebGPU means write once, run everywhere — no app review, instant updates, works on any device with a browser.' The model is a 50M parameter image classifier. Walk me through the real performance gap and when WebGPU is the right call."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "WebGPU is just JavaScript — it'll be 10× slower than native." WebGPU is not WebGL or JavaScript-based inference. It's a low-level GPU API that compiles WGSL shaders to native GPU code. The compute gap is smaller than people assume — but the overhead gap is larger.

  **Realistic Solution:** Benchmark the actual bottlenecks, not the stereotypes:

  **(1) Compute performance: closer than expected.** WebGPU dispatches compute shaders that run on the same GPU hardware as native Metal/Vulkan. For a 50M-param model (MobileNetV3-Large-class): Native (Core ML on ANE): 1.8ms. Native (TFLite GPU delegate on Adreno): 3.2ms. WebGPU (Chrome on iPhone, Metal backend): 5.5ms. WebGPU (Chrome on Android, Vulkan backend): 7.8ms. The compute gap is 2-3×, not 10×. The GPU ALUs run the same math — the gap comes from overhead, not compute.

  **(2) Where the overhead lives.** Shader compilation: WebGPU compiles WGSL → native GPU IR at runtime. First inference: 800ms-2s compilation latency (one-time). Subsequent inferences: compiled shaders are cached. Solution: pre-warm the pipeline at page load. JavaScript↔GPU data transfer: input images must cross the JS heap → GPU buffer boundary. For a 224×224×3 FP32 input: 600 KB copy, ~0.3ms. Native frameworks use zero-copy GPU buffers. Dispatch overhead: each WebGPU dispatch has ~0.05ms overhead vs ~0.01ms for native Metal. A 50-layer model: 50 × 0.04ms extra = 2ms total dispatch overhead.

  **(3) When WebGPU wins.** (a) **No app store dependency.** Deploy and update instantly via URL. No 24-72 hour App Store review. Critical for A/B testing ML models — push a new model to 50% of users in minutes. (b) **Cross-platform with zero native code.** One implementation for iOS Safari, Android Chrome, desktop browsers. No Core ML / TFLite / ONNX Runtime maintenance. (c) **Low-frequency inference.** If the model runs once per user action (e.g., photo classification on upload), the 5-8ms vs 2-3ms difference is imperceptible. The user's tap-to-result time is dominated by UI animation (200ms), not inference. (d) **Privacy-sensitive features.** Data never leaves the browser sandbox. No native SDK means no binary to reverse-engineer.

  **(4) When WebGPU loses.** (a) **Real-time video processing.** 30 FPS requires <33ms per frame. WebGPU's 7.8ms inference + JS overhead + rendering leaves minimal headroom. Native's 3.2ms gives 10× more budget for other processing. (b) **NPU access.** WebGPU targets the GPU only. It cannot access the ANE, Hexagon NPU, or any dedicated ML accelerator. For models optimized for NPUs, this is a 3-5× penalty. (c) **Large models.** WebGPU has a default buffer size limit of 256 MB (expandable, but browser-dependent). Models >200 MB require chunked loading. (d) **Background execution.** Browsers suspend tabs aggressively. No background inference capability.

  > **Napkin Math:** 50M-param model (INT8 = 50 MB). Native (ANE): 1.8ms, 0ms startup. WebGPU: 5.5ms steady-state, 1.2s first-inference compilation. Overhead breakdown: shader compilation (one-time): 1.2s. Per-inference dispatch overhead: 2ms. Data transfer: 0.3ms. Compute: 3.2ms (same GPU). Total steady-state: 5.5ms. For a photo app (1 inference per user action): user perceives 200ms (UI) + 5.5ms (WebGPU) = 205.5ms vs 200ms + 1.8ms (native) = 201.8ms. Difference: 3.7ms. Imperceptible. Engineering cost: WebGPU (one codebase): 1 engineer-month. Native (iOS + Android): 3 engineer-months. WebGPU saves 2 engineer-months for a 3.7ms latency trade-off.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

# Round 1: Mobile Systems & On-Device Inference 📱

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Mobile_Systems.md">📱 Mobile Round 1</a> ·
  <a href="02_Mobile_Advanced.md">🔋 Mobile Round 2</a>
</div>

---

The domain of the Mobile ML Engineer. This round tests your understanding of what happens when a neural network must share a phone's silicon, memory, and battery with everything else the user cares about. There is no dedicated VRAM, no thermal headroom to spare, and no user patience for a hot pocket-warmer.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/mobile/01_Mobile_Systems.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### ⚡ NPU Delegation & Operator Compatibility

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Delegation Lottery</b> · <code>npu-delegation</code></summary>

**Interviewer:** "Your team ships an image classification model on Android using TFLite. On a Snapdragon 8 Gen 3 device, inference takes 4ms. A colleague adds a single custom post-processing op written as a TFLite Flex delegate. Inference jumps to 38ms. What happened?"

**Common Mistake:** "The custom op is computationally expensive." The op itself is trivial — a simple argmax over 1000 classes.

**Realistic Solution:** NPU delegation is all-or-nothing *per subgraph*. When TFLite encounters an op the NPU (Hexagon/QNN) delegate doesn't support, it partitions the graph. The supported prefix runs on the NPU, then the intermediate tensor is copied back to CPU RAM for the unsupported op, and any remaining ops may not return to the NPU. The cost isn't the op — it's the round-trip data transfer across the on-chip NoC and the loss of NPU graph fusion. One incompatible op can shatter an otherwise fully-accelerated graph into three segments with two expensive hand-offs.

> **Napkin Math:** NPU inference for MobileNetV2: ~4ms. CPU fallback for the same model: ~35ms. The single Flex op takes <0.1ms, but it forces a NPU→CPU→(no return) partition. You pay nearly the full CPU cost for the tail of the graph plus ~3ms in tensor transfer overhead.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Heterogeneous Scheduling Trap</b> · <code>npu-delegation</code></summary>

**Interviewer:** "You're profiling a real-time object detection pipeline on a Snapdragon 8 Gen 3. The NPU is rated at 45 TOPS (INT8). Your YOLOv8-S model needs ~7 GOPS per frame. At 30 FPS that's 210 GOPS/s — less than 0.5% of NPU peak. Yet you measure 28ms per frame, barely hitting 30 FPS. Where is the time going?"

**Common Mistake:** "The model must be memory-bandwidth bound on the NPU." While possible, the real culprit is more mundane.

**Realistic Solution:** Pre- and post-processing are running on the CPU and dominating the pipeline. Camera frame acquisition, color space conversion (NV21→RGB), resizing, and normalization happen on the CPU before the NPU ever sees a tensor. After inference, non-max suppression (NMS) with hundreds of candidate boxes runs on the CPU too. The NPU inference itself may take only 5–6ms, but the CPU bookends consume 22ms. The bottleneck is the *pipeline*, not the accelerator.

> **Napkin Math:** Camera capture + CSC: ~4ms. Resize + normalize on CPU: ~6ms. NPU inference (YOLOv8-S INT8): ~6ms. NMS (300 candidates): ~8ms. Tensor copy overhead: ~4ms. Total: ~28ms. NPU is idle for 79% of the frame budget.

> **Key Equation:** $T_\text{frame} = T_\text{preprocess} + T_\text{copy\_in} + T_\text{NPU} + T_\text{copy\_out} + T_\text{postprocess}$

**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

---

### 🧠 Memory Pressure & App Lifecycle

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Silent Eviction</b> · <code>memory-pressure</code></summary>

**Interviewer:** "Your iOS app loads a 400 MB CoreML model into memory. Users report that switching to another app and back causes a 3-second freeze. Instruments shows no crash. What is happening, and how do you fix it?"

**Common Mistake:** "The model file is being re-read from disk every time." Close, but misses the *why*.

**Realistic Solution:** iOS memory pressure eviction. iPhones have no swap space. When the user switches to a memory-hungry app (a game, Safari with many tabs), the OS reclaims memory from backgrounded apps by evicting their mapped pages. Your 400 MB model was memory-mapped (`mmap`) — the pages were clean, so iOS silently discards them without killing the process. When the user returns, every model page triggers a page fault and must be re-read from flash storage (NVMe). 400 MB at ~1.5 GB/s sequential flash read ≈ 270ms, but random page faults are far slower — you see 2–3 seconds of stalls as the working set faults back in during the first inference.

**Fix:** Use `mlock` or `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine` with pre-warming. Better yet, split the model: keep a tiny fast-path model resident (~20 MB) for instant response, and lazy-load the full model in background. On Android, use `MemoryInfo` to monitor `lowMemory` and proactively downgrade to a smaller model before the OOM killer strikes.

> **Napkin Math:** iPhone 15 Pro: 8 GB RAM total. iOS + springboard + background apps: ~4 GB. Your app gets ~2–3 GB. A 400 MB model is 15–20% of your app's budget. Flash random read (4K pages): ~50 µs/page. 400 MB / 4 KB = 100K pages × 50 µs = 5 seconds worst case (sequential prefetch helps, but first inference still stalls).

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🔋 Battery Impact & Thermal Throttling

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Thermal Cliff</b> · <code>battery-impact</code></summary>

**Interviewer:** "Your on-device LLM generates tokens at 12 tokens/sec for the first 30 seconds, then drops to 4 tokens/sec and stays there. The user hasn't changed anything. CPU/GPU/NPU utilization all drop simultaneously. What is happening, and why can't you just 'push through it'?"

**Common Mistake:** "The model is running out of KV-cache memory and swapping." Memory pressure would cause latency spikes, not a smooth sustained drop.

**Realistic Solution:** Thermal throttling. Mobile SoCs have a tiny thermal mass — the chip sits behind a glass screen with no fan and minimal heat sink. Sustained high-power workloads (like autoregressive LLM generation) raise the junction temperature past the thermal governor's threshold. The governor *forcibly* reduces clock frequencies across all compute blocks — CPU, GPU, and NPU simultaneously — to keep the SoC below its thermal design limit (~95°C junction). You can't "push through" because the throttling is enforced in hardware/firmware, below the OS.

The physics is unforgiving: a phone's thermal dissipation capacity is roughly 3–4W sustained. An on-device LLM running the NPU at full tilt draws ~3W from the NPU alone, plus CPU overhead. Within 20–30 seconds, the thermal budget is exhausted.

> **Napkin Math:** Snapdragon 8 Gen 3 NPU peak: 45 TOPS at ~3W. Sustained thermal budget for the whole SoC: ~3.5W. NPU at peak + CPU overhead (~1W) = 4W > 3.5W sustained limit. After ~25 seconds, thermal governor cuts NPU clock by ~60%, dropping throughput from 12 tok/s to ~4 tok/s. This is physics — no software optimization can override the thermal governor.

> **Key Equation:** $T_\text{junction}(t) = T_\text{ambient} + P_\text{dissipated} \times R_\text{thermal} \times (1 - e^{-t/\tau})$
> where $\tau$ is the thermal time constant (~20–30s for a phone SoC)

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Battery Accounting Inversion</b> · <code>battery-impact</code></summary>

**Interviewer:** "Your team optimized an on-device model from FP16 to INT8, cutting MACs by 2× and NPU inference time from 10ms to 5ms. The PM is thrilled — until field data shows battery drain *increased* by 15% in the feature that uses this model. How is a faster model draining more battery?"

**Common Mistake:** "INT8 must be less energy-efficient on this hardware." INT8 is more energy-efficient per operation on virtually all mobile NPUs.

**Realistic Solution:** Jevons Paradox applied to mobile inference. The product team, seeing the 2× latency improvement, changed the feature's behavior: they doubled the inference frequency (from 15 FPS to 30 FPS for the camera pipeline), added a second model pass for refinement, and enabled always-on background inference that was previously gated behind a "too slow" flag. The model is 2× more efficient per inference, but the system is now running 3× more inferences.

The deeper lesson: on mobile, **energy is the conserved quantity, not latency**. Optimizing latency without an energy budget creates a vacuum that product teams fill with more work. Staff engineers enforce energy-per-feature budgets (in mJ/interaction or mW sustained) *before* shipping optimizations, and gate feature expansion on thermal and battery telemetry — not just "is it fast enough."

> **Napkin Math:** Before: 10ms inference × 15 FPS = 150ms NPU-active/sec. NPU at ~2W active = 300 mW average. After: 5ms inference × 30 FPS × 2 passes = 300ms NPU-active/sec. NPU at ~1.5W (INT8 is more efficient) = 450 mW average. Net: 50% more power despite 2× per-inference efficiency. Add the always-on background task at ~200 mW and total drain increases ~115%.

> **Key Equation:** $E_\text{feature} = E_\text{per\_inference} \times f_\text{inferences/sec} \times t_\text{active}$

**📖 Deep Dive:** [Volume I: Sustainable AI](https://mlsysbook.ai/vol1/sustainable_ai.html)
</details>

---

### 📦 Model Formats & Conversion

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Conversion Cliff</b> · <code>model-formats</code></summary>

**Interviewer:** "You trained a PyTorch model with a custom GELU activation and grouped query attention. You convert it to CoreML for iPhone deployment. The conversion succeeds with no errors, but on-device accuracy is significantly worse than your PyTorch baseline. What went wrong?"

**Common Mistake:** "Quantization during conversion caused the accuracy drop." But you haven't quantized — the model is FP16 in both cases.

**Realistic Solution:** Silent operator approximation. CoreML (and TFLite) converters handle unsupported ops by substituting approximate implementations. GELU might be replaced with a sigmoid-based approximation (`x * sigmoid(1.702 * x)`) instead of the exact erf-based version your model was trained with. Grouped query attention may be decomposed into a sequence of reshapes and standard attention ops that introduces numerical drift. The conversion "succeeds" because the graph is structurally valid, but the mathematical behavior has shifted. The fix: always run a numerical comparison (max absolute error, cosine similarity) between the original and converted model on a reference dataset *before* deployment.

> **Napkin Math:** GELU exact: $x \cdot \Phi(x)$ where $\Phi$ is the Gaussian CDF. GELU approximate: $x \cdot \sigma(1.702x)$. Max absolute difference between the two: ~0.004 at $x \approx -1.5$. Across millions of activations over dozens of layers, these errors compound. A model with 24 transformer layers can accumulate enough drift to shift top-1 accuracy by 1–3%.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 🎓 On-Device Training & Privacy

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Privacy-Utility Squeeze</b> · <code>on-device-training</code></summary>

**Interviewer:** "You're implementing federated learning for a next-word prediction model across 100 million Android devices. Each device fine-tunes locally and uploads gradient updates. The privacy team mandates differential privacy with ε = 2. After deployment, the model's perplexity is 40% worse than centralized training. The PM asks you to 'just increase epsilon.' What do you tell them?"

**Common Mistake:** "ε = 2 is too strict, we should relax to ε = 8 or 10." This treats ε as a tuning knob without understanding the system-level trade-offs.

**Realistic Solution:** The problem isn't just ε — it's the interaction between DP noise, client heterogeneity, and the communication budget. With ε = 2, the per-update Gaussian noise scale is large relative to the gradient signal from a single device's small local dataset (maybe a few hundred sentences). But simply increasing ε is a one-way door: once you ship a weaker privacy guarantee, you can't retroactively strengthen it for data already collected.

The staff-level approach: (1) **Increase the number of clients per round** — DP-SGD noise scales as $\sigma \propto 1/\sqrt{n}$, so aggregating 10,000 clients instead of 1,000 reduces noise by ~3.2× at the same ε. (2) **Use secure aggregation** so the server only sees the sum, allowing tighter per-client clipping without the server learning individual updates. (3) **Reduce the model's update surface** — fine-tune only an adapter (LoRA) with ~1% of parameters, concentrating the gradient signal and reducing the dimensionality that noise must cover. (4) **Increase local epochs** (with careful clipping) to improve signal-to-noise per client.

> **Napkin Math:** DP-SGD noise: $\sigma = \frac{C \cdot \sqrt{2 \ln(1.25/\delta)}}{\varepsilon \cdot \sqrt{n}}$ where $C$ = clipping norm, $n$ = clients per round. At ε=2, δ=10⁻⁸, C=1.0, n=1000: σ ≈ 0.003. At n=10000: σ ≈ 0.001. 3× less noise, same privacy. Adding LoRA (rank 8 on a 768-dim model) reduces trainable params from 85M to ~600K — the gradient signal is 140× more concentrated.

> **Key Equation:** $\sigma_\text{noise} \propto \frac{C}{\varepsilon \cdot \sqrt{n}}$

**📖 Deep Dive:** [Volume I: Responsible AI](https://mlsysbook.ai/vol1/responsible_ai.html)
</details>

---

### ⏱️ Latency Budgets & Frame Deadlines

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Jank Budget</b> · <code>latency-budgets</code></summary>

**Interviewer:** "You're building an AR face filter for a social media app targeting 60 FPS. Your face mesh model runs in 8ms on the Apple A17 Pro's ANE. The PM says 'we have 16.7ms per frame, and the model only takes 8ms — we have 8ms of headroom, add a style transfer pass.' Why is the PM's math dangerously wrong?"

**Common Mistake:** "8ms of headroom should be enough for a small style transfer model." This treats the frame budget as if inference is the only consumer.

**Realistic Solution:** The 16.7ms frame budget is shared with the *entire rendering pipeline*, not just ML inference. On iOS, a single frame must fit: camera capture and ISP processing (~2ms), ML inference (~8ms), render pass — applying the mesh to the face, compositing with the camera feed, any UI overlays (~3ms), and the Metal/GPU command buffer submission and display sync (~2ms). That's already ~15ms, leaving only ~1.7ms of headroom — not 8ms.

Worse, if any single frame exceeds 16.7ms, the display controller skips it entirely and waits for the next vsync, causing a visible **jank** — the frame time jumps to 33.3ms (30 FPS for that frame). Adding a 6ms style transfer pass would push total frame time to ~21ms, causing every other frame to be dropped. The user sees persistent 30 FPS stutter.

> **Napkin Math:** Camera ISP: ~2ms. ANE face mesh: ~8ms. Metal render + composite: ~3ms. Display sync overhead: ~2ms. **Total: ~15ms.** Remaining budget: 1.7ms. Style transfer (even MobileStyleNet): ~6ms on ANE. New total: ~21ms > 16.7ms → frame drop → visible jank at 30 FPS.

> **Key Equation:** $T_\text{frame} \leq \frac{1}{f_\text{display}} = \frac{1}{60} = 16.67\text{ms}$
> $T_\text{frame} = T_\text{capture} + T_\text{inference} + T_\text{render} + T_\text{sync}$

**📖 Deep Dive:** [Volume I: Benchmarking AI](https://mlsysbook.ai/vol1/benchmarking_ai.html)
</details>

---

### 🚚 Model Delivery & App Size

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Delivery Paradox</b> · <code>model-delivery</code></summary>

**Interviewer:** "You're shipping an on-device LLM assistant (3B parameters, INT4) to a global Android app with 500 million installs. The quantized model is 1.7 GB. Google Play's install-time APK limit is 150 MB, and on-demand delivery via Play Asset Delivery has a 2 GB limit. Design the model delivery system. What breaks at this scale that doesn't break in a demo?"

**Common Mistake:** "Just use on-demand download after install — it's under 2 GB." This works for a prototype but collapses at 500M-device scale.

**Realistic Solution:** At 500M installs, every decision multiplies by 500M. The system design must address five interacting constraints:

**1. Bandwidth economics:** 500M devices × 1.7 GB = 850 PB of transfer. At $0.01/GB CDN cost, that's $8.5M per model version. You need delta updates — ship weight diffs, not full models. A well-structured INT4 checkpoint with grouped quantization typically yields ~60% compression with brotli, and delta updates between versions compress to ~15% of full size.

**2. Storage heterogeneity:** 500M Android devices span 32 GB budget phones to 512 GB flagships. You can't assume 1.7 GB of free space. Design a tiered model system: a 200 MB "core" model that ships with the APK (handles basic queries), a 600 MB "standard" model downloaded on WiFi, and the full 1.7 GB "premium" model for flagships. Use Android's `StorageStatsManager` to pick the tier at runtime.

**3. Model versioning and rollback:** With 500M devices, you can't push a broken model and "just roll back." Use content-addressed storage (hash the model weights) and keep the previous version on-device until the new one passes a local validation suite (canary inference on reference inputs). A/B test new models on 1% of devices for 48 hours before full rollout.

**4. Integrity and security:** The model is intellectual property sitting on user devices. Sign model chunks with Ed25519, verify at load time. Use Android's `EncryptedFile` API for at-rest encryption. Chunk the model into 50 MB segments for resumable download — mobile connections drop constantly.

**5. Cold start after download:** The user downloads 1.7 GB and expects instant use. But the first inference requires loading the model into memory and compiling the NPU graph. Pre-compile the QNN/TFLite graph during download (background `WorkManager` task) and memory-map the weights so first inference doesn't require a full 1.7 GB read.

> **Napkin Math:** CDN cost per version: 500M × 1.7 GB × $0.01/GB = **$8.5M**. With brotli compression (60%): $3.4M. With delta updates (15% of full): **$510K** — a 17× cost reduction. Storage: median Android device has ~15 GB free. 1.7 GB = 11% of free space — unacceptable for budget phones. Tiered delivery is mandatory.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_operations.html)
</details>

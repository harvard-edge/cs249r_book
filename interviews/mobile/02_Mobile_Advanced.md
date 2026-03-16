# Round 2: Mobile Advanced — Architecture, Optimization & Privacy 🔋

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Mobile_Systems.md">📱 Mobile Round 1</a> ·
  <a href="02_Mobile_Advanced.md">🔋 Mobile Round 2</a>
</div>

---

This round expands the Mobile track into compute analysis across heterogeneous SoCs, memory management under app lifecycle pressure, numerical precision during format conversion, architecture selection for on-device models, model optimization for NPU delegation, deployment through app stores, monitoring without ground truth, and privacy-preserving on-device learning.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/mobile/02_Mobile_Advanced.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### ⚡ Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Budget Phone Mystery</b> · <code>compute</code></summary>

**Interviewer:** "Your image classification model runs in 4ms on a Pixel 8 Pro (Tensor G3). On a budget phone with a MediaTek Dimensity 700 — which claims 'AI accelerator support' on the spec sheet — the same model takes 40ms. Both phones advertise NPU capability. Why is there a 10× performance gap?"

**Common Mistake:** "The budget phone's NPU has fewer TOPS." The spec sheets may show similar peak numbers — the issue is deeper than headline TOPS.

**Realistic Solution:** Five factors create the gap:

(1) **Operator coverage** — the Tensor G3's NPU supports 95%+ of TFLite operators natively. The Dimensity 700's NPU supports ~60%. Every unsupported op falls back to the CPU, and each fallback incurs a 1-3ms data transfer penalty across the on-chip NoC.

(2) **Memory bandwidth** — the Pixel 8 Pro has LPDDR5x at 51.2 GB/s. The budget phone has LPDDR4x at 17 GB/s. For memory-bound models, this alone accounts for a 3× difference.

(3) **Driver maturity** — Google optimizes the Tensor G3's NPU driver for its own TFLite runtime. Third-party SoCs often have less-optimized delegate implementations with higher overhead per inference call.

(4) **Thermal throttling** — the budget phone has a smaller thermal budget and cheaper cooling. After 10 seconds of continuous inference, it throttles from peak to ~40% of rated performance. The Pixel 8 Pro sustains performance longer with its vapor chamber cooling.

(5) **Shared bus contention** — the budget phone's NPU shares its memory bus with the camera ISP and display controller. During camera preview, available bandwidth drops further.

> **Napkin Math:** Pixel 8 Pro: 95% NPU delegation, 51.2 GB/s bandwidth, no throttling → 4ms. Budget phone: 60% NPU delegation (40% CPU fallback adds ~15ms), 17 GB/s bandwidth (3× slower memory access adds ~8ms), bus contention adds ~5ms, thermal throttle after 10s adds ~12ms. Total: ~40ms. The "NPU" badge on the spec sheet is marketing, not a performance guarantee.

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Heterogeneous Execution Strategy</b> · <code>compute</code></summary>

**Interviewer:** "You're deploying a multi-modal model on a Snapdragon 8 Gen 3 that has a Hexagon NPU (45 TOPS), an Adreno GPU (4.6 TFLOPS FP16), and Kryo CPU cores. The model has Conv2D layers, a custom attention mechanism, GELU activations, and dynamic control flow. Design the execution strategy that minimizes total latency."

**Common Mistake:** "Run everything on the NPU — it has the most TOPS." The NPU can't handle all operator types, and forcing unsupported ops onto it causes catastrophic graph partitioning.

**Realistic Solution:** Partition the model by operator affinity. Note: even when you specify a single compute unit, runtimes like CoreML and TFLite may silently re-partition — CoreML models often execute with "Mixed (Float16, Float32, Int32)" precision even when FP16 is requested, because the Neural Engine lacks native support for certain ops. Always profile with the vendor's tools (Xcode GPU Report, Snapdragon Profiler) to see the *actual* execution plan.

**NPU (Hexagon):** Standard Conv2D, depthwise Conv2D, MatMul, ReLU, average/max pooling, concatenation. These are the NPU's sweet spot — fixed-function datapaths optimized for these exact operations. Run the entire convolutional backbone and linear projections here.

**GPU (Adreno):** Custom attention (Q×K^T softmax, with dynamic shapes), GELU activation (not natively supported on most NPUs), and any ops with non-standard tensor layouts. The GPU is flexible enough to handle these via compute shaders, at moderate power cost.

**CPU (Kryo):** Dynamic control flow (if/else branching based on intermediate results), non-tensor operations (tokenization, beam search), and pre/post-processing (image resize, NMS, text decoding).

The critical optimization is **minimizing partition boundaries**. Each NPU→GPU or GPU→CPU handoff costs 0.5-2ms for data transfer across the on-chip NoC. Group all NPU ops contiguously, all GPU ops contiguously. If a single unsupported op sits between two NPU-compatible sections, consider replacing it with a supported approximation (e.g., GELU ≈ x × σ(1.702x) using sigmoid, which the NPU supports) to avoid splitting the graph.

> **Napkin Math:** Model: 50 layers. 40 layers NPU-compatible, 8 GPU (attention + GELU), 2 CPU (control flow). Naive partitioning: 40 NPU layers (8ms) + 2 handoffs (3ms) + 8 GPU layers (6ms) + 1 handoff (1.5ms) + 2 CPU layers (1ms) = 19.5ms. Optimized (replace GELU with sigmoid approximation, fuse attention into fewer GPU calls): 45 NPU layers (9ms) + 1 handoff (1.5ms) + 5 GPU layers (4ms) + 1 handoff (1.5ms) + 2 CPU (1ms) = 17ms. Saving 3 handoffs = 2.5ms = 13% latency reduction.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🧠 Memory Systems

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Background Kill</b> · <code>memory</code></summary>

**Interviewer:** "Your iOS app loads a 400 MB Core ML model into memory. Users report that when they switch to the camera app and come back, the app takes 8 seconds to respond. Your app didn't crash — what happened?"

**Common Mistake:** "The model needs to warm up its caches" or "iOS paused our app." Neither explains the 8-second delay.

**Realistic Solution:** iOS killed your app's process while it was in the background to reclaim memory for the camera app. When the user returns, iOS relaunches your app from scratch — it must reload the 400 MB model from flash storage into RAM. This is a **cold start**, not a resume. On an iPhone 15 (UFS 3.1, ~2 GB/s sequential read): 400 MB / 2 GB/s = 0.2s for raw I/O, but Core ML model compilation (optimizing the graph for the Neural Engine) adds 5-7 seconds. Total: ~8 seconds.

Fixes: (1) Use a smaller model (<100 MB) that loads in <2 seconds. (2) Split into a lightweight always-resident model (~30 MB) for immediate response and a heavy model that loads in the background. (3) Use memory-mapped weights (`mmap`) so the OS can evict and reload pages transparently without killing the process — the model stays "loaded" in virtual address space even if physical pages are reclaimed. (4) Save model compilation artifacts to disk so recompilation isn't needed on relaunch.

> **Napkin Math:** iPhone 15: 6 GB RAM. iOS + system services: ~2.5 GB. Camera app: ~1.5 GB. Available for your app: ~2 GB. Your model: 400 MB weights + ~200 MB activations + ~100 MB runtime = 700 MB. Fits when your app is foreground. But camera app launch pushes total to 2.5 + 1.5 + 0.7 = 4.7 GB → iOS jetsams your app. With mmap: OS evicts weight pages (backed by file, no write-back needed) → your process survives, pages reload on demand during next inference.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Memory-Mapped Weight Strategy</b> · <code>memory</code></summary>

**Interviewer:** "Your team's mobile ML app has a 3-second cold start because it loads 300 MB of model weights into a malloc'd buffer at launch. The PM wants sub-500ms startup. You can't make the model smaller. How do you eliminate the cold start without changing the model?"

**Common Mistake:** "Load the model in a background thread and show a loading screen." This hides the delay but doesn't eliminate it — the user still waits.

**Realistic Solution:** Replace `malloc` + `read` with `mmap()`. Memory-mapping the weight file maps it directly into the process's virtual address space without copying any data into physical RAM. The OS loads pages on demand — only the weight pages needed for the *currently executing layer* are faulted into RAM. First inference is slightly slower (page faults add ~50μs per 4 KB page), but the app is responsive immediately because no upfront loading is needed.

Key benefits: (1) **Zero startup cost** — the mmap call returns instantly. (2) **Graceful under memory pressure** — the OS can evict weight pages at any time (they're backed by the file, so no dirty-page write-back). When needed again, they're silently reloaded. Your process is never killed for memory. (3) **Shared across processes** — if two instances of your model run (e.g., in an app extension), the OS shares the same physical pages.

Trade-off: random access patterns cause excessive page faults. You must ensure the model executes layers sequentially (not randomly accessing distant weights), and the weight file should be stored on fast flash (UFS 4.0: 4.2 GB/s) with minimal fragmentation.

> **Napkin Math:** malloc + read: 300 MB / 2 GB/s (UFS 3.1) = 150ms I/O + 100ms allocation + 200ms framework init = 450ms minimum. With Core ML compilation: +2-5s. mmap: 0ms upfront. First inference: ~200 layers × ~1.5 MB weights per layer × page fault overhead ≈ 50ms extra on first run. Second inference: all pages cached, no overhead. Startup: **<50ms** vs 3 seconds.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The On-Device LLM Memory Architecture</b> · <code>memory</code></summary>

**Interviewer:** "Your PM wants to run a 3B parameter LLM on a phone with 8 GB RAM. In FP16, the weights alone are 6 GB. The OS uses 3 GB. You have 5 GB available. Design a memory architecture that makes this work."

**Common Mistake:** "It doesn't fit — tell the PM it's impossible" or "Just quantize to INT4." INT4 (1.5 GB) is one solution, but the PM also wants FP16 quality for premium users. You need a general architecture.

**Realistic Solution:** Design a **paged weight streaming** system — the LLM equivalent of virtual memory:

(1) **Quantize to INT4 as the default** — 3B × 0.5 bytes = 1.5 GB. Fits entirely in RAM with 3.5 GB headroom. This serves 90% of users.

(2) **For FP16 quality, stream weights from flash** — partition the 6 GB of FP16 weights into 256 MB chunks (one chunk per ~4 transformer blocks). Allocate a 1 GB weight buffer in RAM. At any time, only the currently-executing blocks' weights are resident. As the model advances through layers, prefetch the next chunk from flash while the current chunk executes. UFS 4.0 reads at 4.2 GB/s → 256 MB loads in 61ms. If one transformer block takes ~15ms to execute and you have 4 blocks per chunk (60ms compute), the prefetch completes before the next chunk is needed — zero stall.

(3) **KV-cache budget** — at 2048 context length: 3B model with 32 layers, 32 heads, 128 dim per head. KV-cache = 2 × 32 × 32 × 128 × 2048 × 2 bytes = 1.07 GB in FP16, or 268 MB in INT8 (quantized KV-cache). Use INT8 KV-cache to fit within budget.

(4) **Total memory** — INT4 path: 1.5 GB weights + 268 MB KV-cache + 100 MB activations + 50 MB runtime = 1.92 GB. FP16 streaming path: 1 GB weight buffer + 268 MB KV-cache + 100 MB activations + 50 MB runtime = 1.42 GB resident (6 GB on flash).

> **Napkin Math:** FP16 weights: 3B × 2 = 6 GB (doesn't fit in 5 GB available). INT4 weights: 3B × 0.5 = 1.5 GB ✓. FP16 streaming: 1 GB buffer, 256 MB chunks, UFS 4.0 at 4.2 GB/s → 61ms per chunk load. 4 blocks × 15ms = 60ms compute per chunk. Prefetch hides latency: 61ms load overlaps with 60ms compute → ~1ms stall per chunk. 32 layers / 4 per chunk = 8 chunks per forward pass. Total stall: ~8ms per token. Token latency: 60ms compute + 8ms stall = **68ms/token** (vs 60ms if all weights were resident). Acceptable.

**📖 Deep Dive:** [Volume II: Edge Intelligence](https://mlsysbook.ai/vol2/edge_intelligence.html)
</details>

---

### 🔢 Numerical Representation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Conversion Precision Loss</b> · <code>precision</code></summary>

**Interviewer:** "You convert a PyTorch model from FP32 to Core ML FP16 for the Apple Neural Engine. Overall accuracy drops 0.3% — acceptable. But one specific layer's output diverges by 12% from the FP32 reference. Which layer type is most likely the culprit, and why?"

**Common Mistake:** "The convolution layers are losing precision." Standard Conv2D layers are highly robust to FP16 — their outputs are sums of many small products, which average out rounding errors.

**Realistic Solution:** The most likely culprits are **BatchNorm** and **activation functions with wide dynamic range**. BatchNorm computes $\hat{x} = (x - \mu) / \sqrt{\sigma^2 + \epsilon}$. In CoreML's FP16 conversion, a documented bug causes the epsilon parameter to remain FP32 while mean values are cast to FP16, creating type mismatches that produce large errors (Apple coremltools issues #2470, #2625). Even without the bug, dividing by a small variance in FP16 amplifies rounding errors.

Activation functions are another real-world trap: **Mish** (x × tanh(softplus(x))) and **hard-swish** in MobileNetV3 produce mean absolute errors exceeding 1.0 in intermediate layers when run in FP16 on the Neural Engine (coremltools issue #2359). The chained nonlinearities (exp, tanh, multiply) compound FP16 rounding at each step.

Other culprits: (1) **Softmax** — exp() amplifies small input differences. (2) **Large logits** — values exceeding FP16 max (65504) overflow to infinity. (3) **Residual connections** — adding a large tensor to a small one causes catastrophic cancellation.

Fix: use mixed precision — keep BatchNorm, problematic activations (Mish, hard-swish), softmax, and the final projection in FP32 while running everything else in FP16. Core ML supports per-layer precision specification via `compute_units` and typed execution.

> **Napkin Math:** LayerNorm with σ² = 1e-4. FP32 precision: ~7 decimal digits → division accurate to 0.0001%. FP16 precision: ~3 decimal digits → division accurate to 0.1%. Relative error amplification: 0.1% / 0.0001% = 1000×. If the layer output range is [0, 1], a 0.1% error = 0.001 absolute. After 10 subsequent layers each amplifying by 1.1×: 0.001 × 1.1¹⁰ = 0.0026 → 12% divergence on a feature with range [0, 0.02] is entirely plausible.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Mixed-Precision Deployment Plan</b> · <code>precision</code></summary>

**Interviewer:** "You're deploying a 100-layer model on the Apple Neural Engine via Core ML. Running everything in FP16 gives a 2% accuracy drop — unacceptable for your medical imaging app. Running everything in FP32 means the Neural Engine can't be used (it only supports FP16). Design a mixed-precision strategy that gets Neural Engine speed with FP32 accuracy."

**Common Mistake:** "Run the whole model in FP32 on the GPU." This works but is 3-5× slower than the Neural Engine. You're leaving performance on the table.

**Realistic Solution:** Profile each layer's precision sensitivity by running the model in FP32 and comparing each layer's output to its FP16 equivalent using cosine similarity. Layers with cosine similarity > 0.999 are FP16-safe. Layers below that threshold need FP32.

Typical sensitivity profile: **FP16-safe** (95% of layers): Conv2D, depthwise Conv2D, ReLU, average pooling, concatenation, linear projections. These layers' outputs are sums of many products — rounding errors average out. **FP32-required** (5% of layers): LayerNorm (division by small variance), softmax (exponential amplification), the final classification head (small differences in logits change the predicted class), and any layer immediately after a residual addition with large magnitude difference.

Execution plan: the FP16-safe layers run on the Neural Engine. At each FP32-required layer, data transfers to the CPU/GPU for FP32 computation, then returns to the Neural Engine. Each transfer costs ~1-2ms. With 5 FP32 layers: 5 × 2 round-trips × 1.5ms = 15ms overhead. Total: 20ms (Neural Engine) + 15ms (transfers) + 3ms (FP32 compute) = 38ms. Compare to: all-FP16 Neural Engine = 20ms (but 2% accuracy loss), all-FP32 GPU = 100ms. The mixed approach gives 98% of FP32 accuracy at 38% of FP32 latency.

> **Napkin Math:** 100 layers. 95 on Neural Engine FP16: 95 × 0.2ms = 19ms. 5 on CPU FP32: 5 × 0.6ms = 3ms. 10 data transfers (5 round-trips): 10 × 1.5ms = 15ms. Total: **37ms**. All-FP16: 20ms (fast but inaccurate). All-FP32 GPU: 100ms (accurate but slow). Mixed: 37ms — 1.85× slower than all-FP16, but 2.7× faster than all-FP32, with full accuracy.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 🏗️ Architecture → System Cost

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Squeeze-and-Excitation Question</b> · <code>architecture</code></summary>

**Interviewer:** "MobileNetV3 adds squeeze-and-excitation (SE) blocks that increase FLOPs by 2%. Your colleague says 'that's wasted compute — remove them to speed up inference.' Why is your colleague wrong?"

**Common Mistake:** "2% more FLOPs = 2% slower inference." This treats all FLOPs as equal, ignoring the accuracy-per-FLOP trade-off.

**Realistic Solution:** SE blocks are a channel attention mechanism: global average pool → small FC layer → ReLU → small FC layer → sigmoid → channel-wise multiply. They learn *which channels matter* for each input, effectively giving the network input-dependent feature selection. The 2% FLOP increase buys a 2-3% accuracy improvement. This means you can use a *smaller* base model (e.g., MobileNetV3-Small instead of MobileNetV2-1.0) and still hit the same accuracy target — saving 30%+ FLOPs overall. The SE block's operations (global pool, small FC) are also extremely NPU-friendly — they map to a few MAC operations on the Neural Engine with near-zero overhead.

The deeper insight: on mobile, the goal is maximum accuracy per milliwatt, not minimum FLOPs. A 2% FLOP increase that enables a 30% smaller base model is a massive win for battery life.

> **Napkin Math:** MobileNetV2-1.0: 300 MFLOPs, 72% top-1 ImageNet. MobileNetV3-Small with SE: 56 MFLOPs, 67.4% top-1. MobileNetV3-Large with SE: 219 MFLOPs, 75.2% top-1. To match MobileNetV2's 72% accuracy: MobileNetV3 needs ~150 MFLOPs (with SE) vs 300 MFLOPs (without SE). The 2% FLOP overhead of SE enables a 50% total FLOP reduction at equal accuracy.

**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The On-Device LLM Feasibility Check</b> · <code>architecture</code></summary>

**Interviewer:** "Your PM saw Apple's on-device LLM demo and wants you to ship a 3B parameter chatbot in your app by next quarter. Walk through the feasibility analysis — memory, compute, latency, and battery — and tell the PM what's actually possible."

**Common Mistake:** "3B parameters is too big for a phone — it's impossible." It's not impossible, but the constraints are severe and the PM needs to understand the trade-offs.

**Realistic Solution:** Walk through each constraint:

**Memory:** 3B params in FP16 = 6 GB. iPhone 15 Pro has 8 GB RAM, ~5 GB available. Doesn't fit. INT4 quantization: 3B × 0.5 bytes = 1.5 GB weights. Plus KV-cache for 2048 context: ~270 MB (INT8). Plus activations: ~100 MB. Plus runtime: ~50 MB. Total: **1.92 GB**. Fits with 3 GB headroom.

**Compute:** Autoregressive decoding: ~2 × 3B = 6 GFLOPs per token. Apple A17 Pro Neural Engine: ~35 TOPS. But LLM decoding is memory-bandwidth bound (loading all weights per token). At INT4: 1.5 GB weights / 77 GB/s (LPDDR5x) = **19.5ms per token** = ~51 tokens/second. Feels responsive.

**Latency:** Prefill (processing the prompt): 512 input tokens × 6 GFLOPs = 3.07 TFLOPs. At 35 TOPS: ~88ms. Acceptable. Decode: 19.5ms/token. 100-token response: ~2 seconds. Acceptable.

**Battery:** INT4 inference at ~3W (NPU). 100-token response: 2 seconds × 3W = 6 joules. iPhone 15 Pro battery: 17.3 Wh = 62,280 J. Each response costs 6/62,280 = 0.01% battery. 100 conversations/day = 1% battery. Acceptable.

**Verdict:** Feasible with INT4 quantization. Quality will be noticeably worse than cloud GPT-4, but usable for simple tasks. Ship it as a "fast local mode" with cloud fallback for complex queries.

> **Napkin Math:** Memory: 1.92 GB (INT4) ✓. Decode: 19.5ms/token → 51 tok/s ✓. Prefill: 88ms for 512 tokens ✓. Battery: 1% per 100 conversations ✓. App size: 1.5 GB model (download on WiFi, not in app bundle). Feasible with caveats.

**📖 Deep Dive:** [Volume II: Edge Intelligence](https://mlsysbook.ai/vol2/edge_intelligence.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Streaming ASR Trade-off</b> · <code>architecture</code></summary>

**Interviewer:** "You're building live captions for a video calling app. Your team is debating between Whisper-small (244M params, non-streaming) and a custom RNN-T model (30M params, streaming). The PM wants 'the best accuracy.' Why might the smaller model be the right choice?"

**Common Mistake:** "Whisper is more accurate on benchmarks, so use Whisper." Benchmark accuracy doesn't account for the system-level constraints of real-time captioning.

**Realistic Solution:** Whisper is a sequence-to-sequence model that processes audio in 30-second chunks. For live captions, this means: (1) **30-second latency** — the user speaks, and captions appear 30 seconds later. Unusable for live conversation. (2) **Memory:** 244M params × 2 bytes (FP16) = 488 MB always resident, plus attention KV-cache for 30 seconds of audio. (3) **Compute burst:** processing 30 seconds of audio at once requires a large compute burst, causing thermal spikes.

The RNN-T model processes 80ms audio frames incrementally: (1) **200ms latency** — captions appear within 200ms of speech, feeling real-time. (2) **Memory:** 30M × 2 = 60 MB weights, plus minimal hidden state (~1 MB). (3) **Steady compute:** small, constant inference every 80ms — no thermal spikes, predictable power draw.

Accuracy comparison: Whisper-small WER ~8% on LibriSpeech. RNN-T WER ~12%. But for live captions, the 4% WER gap is invisible to users because: (a) captions are read in real-time where context helps comprehension, (b) 30-second delayed captions are functionally useless regardless of accuracy. The streaming model wins on the metric that matters: **usable accuracy at acceptable latency**.

> **Napkin Math:** Whisper: 488 MB memory, 30s latency, 8% WER. Burst power: ~5W for 3 seconds every 30 seconds. RNN-T: 60 MB memory (8× less), 200ms latency (150× less), 12% WER. Steady power: ~0.5W continuous. Battery for 1-hour call: Whisper ~5W × 0.1 duty = 0.5W avg → 0.5 Wh. RNN-T: 0.5W × 1.0 duty = 0.5W avg → 0.5 Wh. Similar battery, but RNN-T frees 428 MB of RAM for other apps.

**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)
</details>

---

### ⏱️ Latency & Throughput

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Jank Explanation</b> · <code>latency</code></summary>

**Interviewer:** "Your camera app runs an ML-powered background blur filter. The model takes 20ms per frame. Your colleague says '20ms is fast — that's 50 FPS!' But users complain the app feels 'laggy and stuttery.' What is happening?"

**Common Mistake:** "20ms inference is fast enough for 60 FPS." This confuses model inference time with the total frame budget.

**Realistic Solution:** At 60 FPS, each frame has a **16.67ms total budget** — not just for ML inference, but for everything: touch input handling (~1ms), view layout (~2ms), rendering (~4ms), and ML inference. If inference takes 20ms, it exceeds the entire frame budget by itself. The UI thread blocks, the frame misses its VSync deadline, and the display shows the previous frame again. This visible stutter is called **jank**. Users perceive jank as "laggy" even though the model is objectively fast.

Fix: **never run inference on the UI thread**. Use a dedicated background thread for ML inference. The camera preview renders at 60 FPS using the GPU. The ML model runs asynchronously on the NPU, processing every 2nd or 3rd frame. The most recent segmentation mask is composited onto the live preview. The user sees smooth 60 FPS video with a background blur mask that updates at 30 FPS — imperceptible to the human eye, but the UI never stutters.

> **Napkin Math:** Frame budget at 60 FPS: 16.67ms. UI overhead (input + layout + render): ~7ms. Available for ML: 16.67 - 7 = 9.67ms. Model takes 20ms → exceeds budget by 10.33ms → **jank on every frame**. With async inference on background thread: UI thread always completes in 7ms (no jank). ML runs at 1000/20 = 50 FPS on its own thread. Mask updates at 50 FPS, display at 60 FPS. Zero jank.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Async Camera Pipeline</b> · <code>latency</code></summary>

**Interviewer:** "Design an inference pipeline for a camera app that maintains 60 FPS preview while running a 50ms portrait segmentation model. The model cannot run every frame. How do you architect this so the user sees smooth video with accurate segmentation?"

**Common Mistake:** "Run the model every 3rd frame and hold the mask for the other 2 frames." This works but produces visible mask "popping" — the segmentation boundary jumps every 3 frames instead of moving smoothly.

**Realistic Solution:** Design a **triple-buffer async pipeline** with temporal interpolation:

**Buffer A (Display):** The frame currently being shown to the user. Always available, never blocked.

**Buffer B (Processing):** The frame currently being processed by the ML model on the NPU. Takes 50ms.

**Buffer C (Queued):** The most recent camera frame, waiting to be processed when the NPU finishes Buffer B.

The camera produces frames at 60 FPS (every 16.67ms). Every frame goes to Buffer C (overwriting the previous queued frame). When the NPU finishes Buffer B, it immediately starts on Buffer C. The display thread composites Buffer A's camera frame with the most recent completed segmentation mask.

**Temporal interpolation:** Between mask updates (every ~50ms = every 3rd frame), use optical flow or simple affine transform to warp the previous mask to match the current frame's motion. This makes the mask boundary move smoothly at 60 FPS even though the model only runs at 20 FPS. Cost: ~2ms per frame for the warp on the GPU.

**Result:** 60 FPS smooth preview, 20 FPS mask updates, smooth mask boundaries via interpolation, zero jank. The user perceives real-time segmentation.

> **Napkin Math:** Camera: 60 FPS → frame every 16.67ms. Model: 50ms → 20 FPS. Mask interpolation: 2ms per frame on GPU. Display thread: 7ms (UI) + 2ms (interpolation) + 1ms (composite) = 10ms < 16.67ms budget ✓. NPU utilization: 50ms/50ms = 100% (always processing). Perceived quality: smooth 60 FPS with mask updating every 50ms (3 frames). Without interpolation: mask "jumps" every 3 frames. With interpolation: mask moves smoothly every frame.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

---

### ⚡ Power & Thermal

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Battery Blame Game</b> · <code>power</code></summary>

**Interviewer:** "Your ML-powered fitness app drains 1% battery per minute during a workout. The PM says 'the model is too expensive — optimize it.' You profile the power breakdown and find the model is not the main culprit. What is?"

**Common Mistake:** "The ML model must be the power hog — it's the most computationally intensive component." This assumes compute = power, ignoring the rest of the system.

**Realistic Solution:** Profile the full system power breakdown during a workout session: (1) **GPS radio:** continuously active for route tracking — ~200 mW. (2) **Screen:** always on showing the workout dashboard at high brightness (outdoor use) — ~800 mW. (3) **Heart rate sensor:** continuous optical sensing — ~100 mW. (4) **Cellular/WiFi radio:** uploading telemetry every 5 seconds — ~500 mW average. (5) **ML inference:** pose estimation model running every 500ms at 5ms per inference — duty cycle of 1%. NPU power during inference: ~2W. Average: 2W × 0.01 = **20 mW**.

Total: 200 + 800 + 100 + 500 + 20 = **1620 mW**. The ML model accounts for 20/1620 = **1.2% of total power**. Optimizing the model to zero would save 1.2% of battery drain. The real levers: dim the screen (save 400 mW), reduce GPS polling rate (save 100 mW), batch telemetry uploads (save 300 mW).

> **Napkin Math:** iPhone 15 battery: 3349 mAh × 3.83V = 12.8 Wh. At 1620 mW total draw: 12.8 Wh / 1.62W = 7.9 hours. 1% per minute = 100 minutes to drain → 1.62W × (100/60) = 2.7W total (our estimate is conservative). Even doubling model efficiency saves: 20 mW / 2700 mW = 0.7% of battery life. Dimming the screen saves 15%.

**📖 Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The NPU Efficiency Advantage</b> · <code>power</code></summary>

**Interviewer:** "Your model runs at 2 TOPS on both the Snapdragon 8 Gen 3's Hexagon NPU and its Adreno GPU. Same model, same throughput. But the NPU uses 0.5W while the GPU uses 1.5W. Why does the NPU achieve 3× better TOPS/W for the same workload?"

**Common Mistake:** "The NPU has a newer manufacturing process." Both are on the same 4nm die — the difference is architectural.

**Realistic Solution:** The NPU and GPU achieve the same TOPS but through fundamentally different architectures:

**GPU (Adreno):** A general-purpose SIMD processor. Every MAC operation requires: (1) instruction fetch and decode, (2) register file read (source operands), (3) ALU execution, (4) register file write (result), (5) thread scheduling and warp management. The control logic (instruction decoder, scheduler, branch predictor) consumes power even though it does no useful math. For a simple multiply-accumulate, ~60% of energy goes to data movement and control, ~40% to the actual computation.

**NPU (Hexagon):** A fixed-function accelerator with hardwired datapaths for MAC operations. There's no instruction fetch per operation — the dataflow is configured once at model load time. Data moves through a spatial pipeline of MAC units with minimal control overhead. ~85% of energy goes to computation, ~15% to data movement. No branch prediction, no thread scheduling, no instruction cache.

The result: for the specific operations the NPU supports (Conv2D, MatMul, pooling), it achieves 3-5× better energy efficiency than the GPU. The GPU's flexibility is its strength for general compute but its weakness for the narrow, repetitive operations that dominate neural network inference.

> **Napkin Math:** GPU at 2 TOPS, 1.5W: 1.33 TOPS/W. Energy per MAC: 1.5W / (2 × 10¹² ops/s) = 0.75 pJ/op. NPU at 2 TOPS, 0.5W: 4.0 TOPS/W. Energy per MAC: 0.5W / (2 × 10¹²) = 0.25 pJ/op. The NPU is 3× more efficient per operation. Over a 1-hour session at continuous inference: GPU = 1.5 Wh, NPU = 0.5 Wh. On a 12.8 Wh battery: GPU uses 11.7% battery, NPU uses 3.9%.

> **Key Equation:** $\text{Energy per op} = \frac{P_{\text{total}}}{\text{TOPS} \times 10^{12}}\ \text{(joules/op)}$

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

---

### 🔧 Model Optimization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Single-Op Delegation Fix</b> · <code>optimization</code></summary>

**Interviewer:** "Your TFLite model runs in 4ms when fully delegated to the Snapdragon NPU. After adding a single custom GELU activation layer, inference jumps to 38ms. The GELU itself takes <0.1ms on CPU. Why did one tiny op cause a 9.5× slowdown, and how do you fix it?"

**Common Mistake:** "The GELU op is slow on CPU." It's not — the op itself is trivial. The cost is in what it does to the execution graph.

**Realistic Solution:** NPU delegation is all-or-nothing per subgraph. When TFLite encounters an op the NPU delegate doesn't support, it **partitions the graph** at that point. The supported prefix runs on the NPU, then the intermediate tensor is copied back to CPU RAM for the unsupported op, and the remaining ops may or may not return to the NPU. One incompatible op in the middle of the graph shatters it into three segments with two expensive data transfers across the on-chip NoC.

Fixes, in order of preference: (1) **Replace with a supported approximation** — GELU(x) ≈ x × σ(1.702x). Sigmoid is NPU-supported. The approximation error is <0.01 for all practical input ranges. The entire graph stays on the NPU: 4ms. (2) **Move the unsupported op to the end** — restructure the model so GELU is the last operation. Only the tail falls back to CPU, and there's only one NPU→CPU transfer instead of two. (3) **Custom NPU kernel** — if the SoC vendor provides a custom op API (Qualcomm QNN SDK), implement GELU as a native NPU op. High effort but zero overhead. (4) **Use a different runtime** — ONNX Runtime's QNN execution provider may support GELU natively where TFLite's delegate doesn't.

> **Napkin Math:** Fully delegated: 4ms (all on NPU). With GELU partition: 2ms (NPU prefix) + 1.5ms (NPU→CPU transfer) + 0.1ms (GELU on CPU) + 1.5ms (CPU→NPU transfer) + 2ms (NPU suffix) + overhead = 7.1ms minimum. But the partition also breaks NPU graph optimizations (layer fusion, buffer reuse), inflating the NPU segments from 2+2ms to 15+15ms. Total: ~38ms. With sigmoid approximation: 4ms (no partition). **9.5× speedup from changing one activation function.**

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🚀 Deployment

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The App Size Limit</b> · <code>deployment</code></summary>

**Interviewer:** "Your model is 500 MB (FP32 weights). The App Store allows up to 200 MB for cellular downloads — anything larger requires WiFi. Your PM wants the app to work immediately after download, even on cellular. What are your options?"

**Common Mistake:** "Compress the model file with zip." Model weights don't compress well — they're essentially random floating-point numbers. Zip might save 5-10%.

**Realistic Solution:** Four strategies, from simplest to most complex:

(1) **Quantize to INT8** — 500 MB FP32 → 125 MB INT8. Fits under 200 MB. Accuracy loss: typically <1% for classification, <2% for detection. This is the first thing to try.

(2) **Ship a small model in the bundle, download the full model on WiFi** — include a MobileNet-V3-Small (~8 MB) in the app bundle for immediate functionality. On first WiFi connection, download the full 500 MB model in the background. The user gets instant (lower quality) results that upgrade transparently.

(3) **On-demand model download** — use Apple's On-Demand Resources (ODR) or Android's Play Asset Delivery (PAD) to stream model chunks after install. The model is not in the initial download. First inference triggers a download of the required model shard.

(4) **Knowledge distillation** — train a smaller student model (~50 MB) that mimics the 500 MB teacher. Ship the student in the bundle. This requires ML engineering effort but produces a permanently smaller model.

> **Napkin Math:** FP32: 500 MB (over limit). INT8: 125 MB ✓ (under 200 MB). INT4: 62.5 MB ✓. Distilled student: ~50 MB ✓. Bundle + background download: 8 MB initial + 500 MB on WiFi. Cellular download at 10 Mbps: 200 MB = 160 seconds. 500 MB = 400 seconds (requires WiFi). User experience: INT8 is the best trade-off — immediate, no WiFi dependency, minimal accuracy loss.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 📊 Monitoring & Reliability

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Silent Accuracy Degradation</b> · <code>monitoring</code></summary>

**Interviewer:** "Your on-device image classification model was deployed 6 months ago. Users haven't complained, but your A/B test shows the new model version improves engagement by 15%. This suggests the old model has silently degraded. You have no server-side ground truth labels for on-device predictions. How do you detect model degradation without ground truth?"

**Common Mistake:** "Monitor accuracy using a held-out test set." You don't have labels for on-device predictions — there's no test set.

**Realistic Solution:** Four proxy signals that detect degradation without ground truth:

(1) **Confidence distribution shift** — track the distribution of the model's softmax confidence scores over time. A healthy model has a bimodal distribution (high confidence for easy inputs, low for hard). If the distribution shifts toward uniform confidence (the model becomes "confused"), it's seeing out-of-distribution data. Use KL divergence between the current week's confidence distribution and the baseline.

(2) **User implicit feedback** — track behavioral proxies: if the model powers a photo search feature, monitor search refinement rate (user searches again immediately = bad result), feature abandonment rate, and time-to-action after a prediction. A 15% engagement improvement from the new model implies the old model's predictions were increasingly irrelevant.

(3) **Lightweight anomaly detector** — deploy a small autoencoder (~1 MB) alongside the main model. Train it on the same distribution as the main model. If reconstruction error exceeds a threshold, the input is OOD. Track the OOD rate over time — a rising rate indicates distribution drift.

(4) **Federated evaluation** — periodically sample a small subset of users, send them a labeled evaluation batch (e.g., 100 images with known labels), and compare the model's predictions. This gives you direct accuracy measurement without collecting user data. Privacy-preserving: the evaluation data comes from you, not from users.

> **Napkin Math:** Confidence monitoring: ~0.1 KB per inference (just the max softmax score). 1000 inferences/day × 30 days = 30,000 data points. KL divergence computation: trivial. Storage: 30 KB/month. Anomaly detector: 1 MB model, 0.5ms per inference. Federated evaluation: 100 labeled images × 4 evaluations/year = 400 labeled predictions per user per year. With 10,000 users: 4M labeled predictions — statistically powerful.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

---

### 🔒 Security & Privacy

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Federated Keyboard</b> · <code>privacy</code></summary>

**Interviewer:** "You're building a next-word prediction model for a mobile keyboard. The model must improve from user typing patterns, but you cannot collect user keystrokes on your servers — that's a privacy violation and a regulatory risk. How do you train the model?"

**Common Mistake:** "Anonymize the data before uploading." Anonymization is insufficient — keystroke patterns can re-identify users, and regulators (GDPR, CCPA) consider this personal data regardless of anonymization.

**Realistic Solution:** **Federated Learning with Differential Privacy (DP-FedAvg):**

(1) **Local training:** Each phone fine-tunes a local copy of the model on the user's recent typing data. Training happens on-device during charging (to avoid battery drain). The phone computes a gradient update (the difference between the local model and the global model).

(2) **Gradient clipping:** Before sending anything, clip the gradient to a maximum L2 norm $C$. This bounds the influence of any single user's data on the global model.

(3) **Noise injection:** Add calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ to the clipped gradient. This provides differential privacy — mathematically guaranteeing that the server cannot determine whether any specific user participated in training.

(4) **Secure aggregation:** The server collects noised gradients from thousands of phones and averages them. Individual gradients are encrypted so the server only sees the aggregate. The noise cancels out in the average (law of large numbers), but protects individual contributions.

(5) **Privacy budget:** The privacy guarantee is measured by $\epsilon$ (epsilon). Lower $\epsilon$ = stronger privacy but more noise = slower learning. Typical production values: $\epsilon = 8$ per training round, with a total budget of $\epsilon = 100$ per year. At these levels, accuracy loss vs non-private training is ~2%.

> **Napkin Math:** 10,000 phones per round. Each sends a 5 MB gradient update (clipped + noised). Server bandwidth: 50 GB per round. Noise per phone: σ = 1.0, C = 1.0. After averaging 10,000 updates: effective noise = σ/√10,000 = 0.01 — negligible. Privacy: ε = 2 per round (strong). 50 rounds/year: total ε = 100 (within budget). Accuracy: ~2% worse than centralized training, but zero user data leaves the device.

> **Key Equation:** $\tilde{g}_{\text{user}} = \text{clip}(g, C) + \mathcal{N}(0, \sigma^2 C^2 I)$

**📖 Deep Dive:** [Volume II: Security & Privacy](https://mlsysbook.ai/vol2/security_privacy.html)
</details>

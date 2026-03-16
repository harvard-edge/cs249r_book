# Round 2: TinyML Advanced — Latency, Deployment & Reliability 🔩

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_TinyML_Systems.md">🔬 TinyML Round 1</a> ·
  <a href="02_TinyML_Advanced.md">🔩 TinyML Round 2</a>
</div>

---

This round expands the TinyML track into compute analysis on Cortex-M, latency budgets for real-time sensor pipelines, model optimization under extreme memory constraints, firmware deployment and OTA updates, on-device monitoring without connectivity, and security against physical access threats.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/tinyml/02_TinyML_Advanced.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 📐 Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The MAC Budget</b> · <code>compute</code></summary>

**Interviewer:** "Your Cortex-M4 runs at 168 MHz with no FPU. Your keyword spotting model requires 5 million INT8 multiply-accumulate (MAC) operations per inference. Can you run inference in under 100ms?"

**Common Mistake:** "168 MHz means 168 million operations per second, so 5M MACs takes 30ms — easy." This assumes one MAC per clock cycle, which is wrong for a scalar processor without SIMD.

**Realistic Solution:** A Cortex-M4 without SIMD extensions executes one 32-bit instruction per cycle, but an INT8 MAC requires multiple instructions: load two 8-bit operands (1-2 cycles), multiply (1 cycle), accumulate (1 cycle), store (1 cycle). Effective throughput: ~1 MAC per 3-4 cycles. At 168 MHz: 168M / 4 = **42 million MACs/second**. Time for 5M MACs: 5M / 42M = **119ms** — over budget.

With CMSIS-NN and the Cortex-M4's DSP extension (SIMD): the `SMLAD` instruction performs two 16-bit MACs per cycle. Packing two INT8 values into 16-bit lanes: effective throughput = 2 MACs per cycle = **336 million MACs/second**. Time: 5M / 336M = **14.9ms** — well within budget. The lesson: on MCUs, SIMD is not optional — it's the difference between feasible and infeasible.

> **Napkin Math:** Without SIMD: 168 MHz / 4 cycles per MAC = 42 MMAC/s. 5M MACs / 42M = 119ms ✗. With CMSIS-NN SIMD: 168 MHz × 2 MACs/cycle = 336 MMAC/s. 5M / 336M = 14.9ms ✓. SIMD gives **8× speedup** for INT8 workloads on Cortex-M4.

> **Key Equation:** $t_{\text{inference}} = \frac{\text{Total MACs}}{\text{Clock} \times \text{MACs per cycle}}$

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The MCU Throughput Ceiling</b> · <code>compute</code></summary>

**Interviewer:** "Your anomaly detection model needs 10 million MACs per inference. You need 10 inferences per second on a Cortex-M4 at 168 MHz with CMSIS-NN SIMD. Does the math work? And what happens to the rest of the system?"

**Common Mistake:** "336 MMAC/s with SIMD. 10M × 10 = 100 MMAC/s needed. That's only 30% utilization — plenty of headroom." This ignores the non-MAC overhead.

**Realistic Solution:** The 336 MMAC/s is the peak throughput for pure MAC operations. Real inference includes: (1) **Memory loads** — fetching weights and activations from SRAM takes cycles. If data isn't in the CPU's small cache (typically 4-16 KB on M4), every load stalls the pipeline. (2) **Activation functions** — ReLU is cheap (1 cycle), but sigmoid/tanh require lookup tables or polynomial approximation (10-20 cycles each). (3) **Requantization** — between layers, INT32 accumulator results must be rescaled back to INT8 (shift, round, clamp: ~5 cycles per element). (4) **Loop overhead** — index computation, bounds checking, pointer arithmetic.

Realistic throughput: ~40-60% of peak = 134-202 MMAC/s. At 150 MMAC/s effective: 100M MACs/s needed / 150 MMAC/s = **67% CPU utilization for ML alone**. That leaves 33% for: sensor data acquisition (ADC reads, DMA), communication (UART/SPI to radio), and application logic (thresholds, state machines). If the sensor pipeline needs 20% and comms need 10%, you're at 97% — no headroom for interrupt latency spikes. You either need a faster MCU (Cortex-M7 at 480 MHz) or a smaller model.

> **Napkin Math:** Peak: 336 MMAC/s. Effective (60% utilization): 202 MMAC/s. ML workload: 100 MMAC/s = 50% of effective throughput. Sensor acquisition (16 kHz ADC, FFT): ~30 MMAC/s = 15%. UART comms: ~10 MMAC/s = 5%. Total: 70% utilization. Headroom: 30% — tight but feasible. At 80% effective utilization threshold (reliability margin): 100/0.8 = 125 MMAC/s needed → 62% of effective capacity. Feasible with careful scheduling.

**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The MCU Roofline</b> · <code>compute</code></summary>

**Interviewer:** "Construct a roofline model for a Cortex-M7 with 512 KB SRAM, a 64-bit AXI bus at 240 MHz, and SIMD capable of 4 INT8 MACs per cycle at 480 MHz. Where is the ridge point, and what does it tell you about which models are feasible?"

**Common Mistake:** "The roofline is the same concept as for GPUs — just smaller numbers." The concept is the same, but the implications are radically different because the memory hierarchy is flat (no HBM, no L2 cache in most M7 variants).

**Realistic Solution:** Build the roofline:

**Peak compute:** 480 MHz × 4 MACs/cycle = **1.92 GMAC/s** (1.92 GOPS INT8).

**Peak memory bandwidth:** 64-bit AXI bus at 240 MHz = 64/8 × 240M = **1.92 GB/s** from SRAM. (In practice, bus contention with DMA and peripherals reduces this to ~1.2-1.5 GB/s.)

**Ridge point:** 1.92 GOPS / 1.92 GB/s = **1.0 Ops/Byte** (or ~1.6 Ops/Byte with realistic bandwidth).

This is extraordinarily low compared to GPUs (ridge ~300 Ops/Byte). It means: almost every neural network layer is **compute-bound** on an MCU, not memory-bound. A standard 3×3 Conv2D has arithmetic intensity of ~18 Ops/Byte (9 MACs × 2 ops / 1 byte per weight) — well above the ridge. This is the opposite of GPUs, where most workloads are memory-bound.

Implication: on MCUs, reducing FLOPs directly reduces latency (unlike GPUs where reducing FLOPs often doesn't help because you're memory-bound). Depthwise separable convolutions (which reduce FLOPs 8-9×) give nearly proportional speedups on MCUs — a property that doesn't hold on GPUs.

> **Napkin Math:** Ridge point: 1.0-1.6 Ops/Byte. Conv2D 3×3 intensity: ~18 Ops/Byte → compute-bound ✓. Depthwise Conv2D 3×3 intensity: ~9 Ops/Byte → compute-bound ✓. Fully connected layer (1024→1024): 2 Ops/Byte → compute-bound ✓. Pointwise Conv2D (1×1): ~2 Ops/Byte → compute-bound ✓. On an MCU, everything is compute-bound. On a GPU (ridge ~300), everything except large MatMuls is memory-bound. This is why model optimization strategies differ radically between the two platforms.

> **Key Equation:** $\text{Ridge Point}_{\text{MCU}} = \frac{\text{Peak GOPS}}{\text{Bus Bandwidth (GB/s)}}$

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

---

### 🧠 Memory Systems

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Tensor Arena Overflow</b> · <code>memory</code></summary>

**Interviewer:** "Your model's tensor arena needs 210 KB for peak activation memory, but your Cortex-M4 only has 200 KB of SRAM available (after firmware and stack). The model fits in flash. You can't change the model architecture. How do you fit it?"

**Common Mistake:** "Reduce the input resolution." The question says you can't change the model — input resolution is part of the model's expected input shape.

**Realistic Solution:** The 210 KB peak occurs at one specific point in the model — typically where two large activation tensors coexist (e.g., the input and output of the largest layer). TFLite Micro's memory planner already optimizes tensor lifetimes, but you can push further:

(1) **In-place operations** — for layers where the output has the same shape as the input (ReLU, batch norm), configure the runtime to write the output directly into the input buffer. Saves one tensor's worth of memory at that layer.

(2) **Operator reordering** — if the memory planner's default order creates a peak at layer 5, check if reordering independent branches (in a multi-branch architecture) shifts the peak. Sometimes processing branch A before branch B reduces the maximum number of simultaneously live tensors.

(3) **Tensor splitting** — split the largest layer's computation into tiles. Process the input in 4 spatial tiles, each requiring 1/4 of the activation memory. Peak drops from 210 KB to ~60 KB for that layer. Trade-off: 4× more kernel invocations (overhead ~5-10%).

(4) **Scratch buffer sharing** — some operators (depthwise conv, transpose) use temporary scratch buffers. Ensure these share the same memory region and are never live simultaneously.

> **Napkin Math:** Peak at layer 5: input tensor (100 KB) + output tensor (110 KB) = 210 KB. With in-place ReLU after layer 5: output overwrites input → peak = 110 KB. With 2× tiling: input tile (50 KB) + output tile (55 KB) = 105 KB. With both: 55 KB peak at layer 5. New global peak shifts to layer 8: 180 KB. Fits in 200 KB with 20 KB headroom.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🔢 Numerical Representation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Zero-Point Question</b> · <code>quantization</code></summary>

**Interviewer:** "A junior engineer says 'to quantize a model to INT8, just round each FP32 weight to the nearest integer.' Why is this wrong, and what is a zero-point?"

**Common Mistake:** "Rounding FP32 to INT8 is quantization." This destroys the model because FP32 weights are typically in the range [-0.1, 0.1] — rounding them all to 0 gives you a zero model.

**Realistic Solution:** Quantization is an affine mapping from the float range to the integer range, not simple rounding. The formula is: $x_q = \text{round}(x_f / s) + z$, where $s$ is the scale (step size) and $z$ is the zero-point (the integer value that represents float 0.0).

Why the zero-point matters: if your weights range from [-0.1, 0.1], the scale is $s = 0.2 / 255 = 0.000784$. The zero-point $z = \text{round}(0 / s) + 128 = 128$ (mapping float 0.0 to the middle of the INT8 range). Now -0.1 maps to 0, 0.0 maps to 128, and 0.1 maps to 255 — the full INT8 range is utilized.

Without a zero-point (symmetric quantization around 0): INT8 range [-128, 127] maps to [-0.1, 0.1]. This works for weights (which are roughly symmetric) but wastes range for activations after ReLU, which are always ≥ 0. An asymmetric scheme with zero-point maps [0, max_activation] to [0, 255], using the full unsigned INT8 range.

> **Napkin Math:** Weight range [-0.1, 0.1]. Naive rounding: all weights → 0. Model is dead. Proper quantization: scale = 0.2/255 = 0.000784. Weight 0.05 → round(0.05/0.000784) + 128 = 192. Weight -0.03 → round(-0.03/0.000784) + 128 = 90. Dequantized: (192-128) × 0.000784 = 0.0502, (90-128) × 0.000784 = -0.0298. Error: <0.001 — acceptable.

> **Key Equation:** $x_q = \text{round}\left(\frac{x_f}{s}\right) + z, \quad x_f = s \times (x_q - z)$

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Per-Channel Trade-off</b> · <code>quantization</code></summary>

**Interviewer:** "Your per-tensor quantized keyword spotting model loses 5% accuracy going from FP32 to INT8. Switching to per-channel quantization recovers 3%. Your colleague says 'always use per-channel — it's strictly better.' Why might you still choose per-tensor on an MCU?"

**Common Mistake:** "Per-channel is always better because it has more quantization parameters." More parameters means more accuracy, but ignores the system cost.

**Realistic Solution:** Per-channel quantization assigns a separate scale and zero-point to each output channel of a convolution. For a layer with 64 output channels, that's 64 scales and 64 zero-points instead of 1 each. The accuracy benefit comes from each channel having its own optimal range.

The system costs on an MCU: (1) **Code size** — per-channel requantization requires a loop over channels with per-channel scale/zero-point lookups. Per-tensor uses a single multiply-shift. The per-channel kernel is ~2× larger in flash. On a 512 KB flash MCU, this matters. (2) **Inference speed** — per-channel requantization between layers requires loading the scale array from memory for each output element. On a Cortex-M4 without cache, this adds ~15-20% overhead per layer due to extra memory loads. (3) **CMSIS-NN support** — CMSIS-NN's optimized kernels support per-channel natively, but older versions or custom runtimes may not, forcing a fallback to unoptimized C code (5-10× slower).

Decision framework: use per-channel when accuracy is critical (medical, safety) and the MCU has sufficient flash and CMSIS-NN support. Use per-tensor when flash is tight, latency is critical, or the accuracy loss is acceptable for the application (e.g., keyword spotting where 90% vs 93% accuracy doesn't change user experience).

> **Napkin Math:** Per-tensor: 1 scale + 1 zero-point = 8 bytes per layer. 20 layers = 160 bytes. Requantization: 1 multiply + 1 shift per element. Per-channel: 64 scales + 64 zero-points = 512 bytes per layer. 20 layers = 10 KB. Requantization: 1 multiply + 1 shift + 1 load per element per channel. Overhead: ~18% more cycles. On a 100ms inference budget: per-tensor = 85ms, per-channel = 100ms. Accuracy: per-tensor = 87%, per-channel = 90%. If the deadline is 100ms: per-channel barely fits. If 90ms: per-tensor only.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 🏗️ Architecture → System Cost

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Depthwise Separable FLOP Ratio</b> · <code>architecture</code></summary>

**Interviewer:** "Your colleague says 'depthwise separable convolutions are just a trick to reduce parameters.' Calculate the exact FLOP ratio between a standard 3×3 Conv2D and a depthwise separable Conv2D for 64 input channels and 64 output channels on a 32×32 feature map. Why does this ratio matter more on an MCU than on a GPU?"

**Common Mistake:** "Depthwise separable is about 3× cheaper." The actual ratio is much larger, and the reason it matters more on MCUs is the key insight.

**Realistic Solution:**

**Standard Conv2D:** Each output pixel requires $K^2 \times C_{in}$ MACs. Total: $K^2 \times C_{in} \times C_{out} \times H \times W = 3^2 \times 64 \times 64 \times 32 \times 32 = 37,748,736$ MACs.

**Depthwise separable:** Depthwise (one filter per channel): $K^2 \times C_{in} \times H \times W = 9 \times 64 \times 1024 = 589,824$ MACs. Pointwise (1×1 conv to mix channels): $C_{in} \times C_{out} \times H \times W = 64 \times 64 \times 1024 = 4,194,304$ MACs. Total: **4,784,128** MACs.

**Ratio:** 37.7M / 4.8M = **7.9×** fewer MACs. Not 3× — nearly 8×.

Why this matters more on MCUs: on a GPU, most convolutions are memory-bound (arithmetic intensity below the ridge point). Reducing FLOPs by 8× doesn't give 8× speedup because you're still waiting for memory. On an MCU, convolutions are compute-bound (ridge point ~1 Ops/Byte). Reducing FLOPs by 8× gives nearly **8× speedup**. This is why MobileNet architectures (built on depthwise separable convolutions) are disproportionately effective on MCUs compared to GPUs.

> **Napkin Math:** Standard: 37.7M MACs. Depthwise separable: 4.8M MACs. Ratio: 7.9×. On Cortex-M4 (336 MMAC/s SIMD): Standard = 112ms. Depthwise separable = 14ms. On GPU (memory-bound): Standard = 2ms. Depthwise separable = 1.5ms (only 1.3× speedup due to memory bottleneck). The MCU gets 8× speedup; the GPU gets 1.3×.

> **Key Equation:** $\text{FLOP Ratio} = \frac{K^2 \times C_{in} \times C_{out}}{K^2 \times C_{in} + C_{in} \times C_{out}} = \frac{K^2 \times C_{out}}{K^2 + C_{out}}$

**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Operator Support Gap</b> · <code>architecture</code></summary>

**Interviewer:** "Your person detection model uses MobileNetV2 as the backbone. TFLite Micro on your Cortex-M4 supports 85% of the operators. The remaining 15% include Resize Bilinear (in the feature pyramid), Pad (before some convolutions), and a custom Swish activation. What is your strategy?"

**Common Mistake:** "Implement the missing operators in C." You can, but custom operator implementations are unoptimized and can be 10-50× slower than CMSIS-NN kernels, potentially blowing your latency budget.

**Realistic Solution:** Address each unsupported operator with the least-effort fix:

(1) **Swish activation** (x × sigmoid(x)): Replace with ReLU6 during training. ReLU6 is universally supported and the accuracy difference on person detection is typically <0.5%. Requires retraining (~1 day).

(2) **Resize Bilinear**: Replace with nearest-neighbor resize (supported) during model export. For feature pyramid networks, the quality difference is minimal because the subsequent convolution layers smooth out the interpolation artifacts. Zero retraining needed — just change the export config.

(3) **Pad**: Fuse padding into the subsequent convolution by using the `padding='same'` option in the convolution layer itself. Most frameworks can do this automatically during export. If not, implement a simple zero-pad kernel (~20 lines of C, runs in microseconds).

(4) **Validation**: After all substitutions, re-evaluate on the test set. If accuracy drops >1%, selectively retrain with the substituted operators (quantization-aware fine-tuning for 10 epochs).

The general principle: never implement a complex custom operator on an MCU if you can replace it with a supported approximation. The supported operators have CMSIS-NN SIMD-optimized kernels; custom ops run on scalar C code at 5-10× lower throughput.

> **Napkin Math:** Original model: 85% supported (CMSIS-NN, ~15ms) + 15% unsupported (scalar C, ~45ms) = 60ms total. After substitutions: 100% supported = 18ms total (15ms + 3ms for the slightly different ops). Speedup: 3.3×. Accuracy change: -0.3% (Swish→ReLU6) + 0% (resize) + 0% (pad fusion) = -0.3% total. Acceptable.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The MCU NAS Search Space</b> · <code>architecture</code></summary>

**Interviewer:** "You're designing a Neural Architecture Search (NAS) pipeline to find the optimal model for a Cortex-M4 with 256 KB SRAM and 1 MB flash. What constraints must your search space encode that a standard NAS for desktop/cloud would ignore?"

**Common Mistake:** "Just add a FLOP constraint to the search." FLOPs alone don't capture the MCU's constraints — memory is the binding constraint, not compute.

**Realistic Solution:** MCU-aware NAS must encode at least five constraints that standard NAS ignores:

(1) **Peak SRAM constraint** — the tensor arena (activations + scratch buffers) must fit in SRAM at every point during inference. This is not just total activation size — it's the *maximum simultaneously live* activation memory, which depends on the operator execution order. The search must simulate the memory planner for each candidate architecture.

(2) **Flash constraint** — all weights + the TFLite Micro runtime + application code must fit in flash. Weights in INT8: model_params × 1 byte. Runtime: ~50-100 KB. Application: ~20-50 KB. Available for weights: flash - 150 KB.

(3) **Operator support constraint** — every operator in the candidate architecture must be supported by the target runtime's kernel library (TFLite Micro, STM32Cube.AI, or microTVM). Architectures with unsupported ops (e.g., certain attention mechanisms, group convolutions with non-standard groups) are infeasible regardless of accuracy. Different runtimes support different operator sets — STM32Cube.AI supports more operators on STM32 targets but is vendor-locked.

(4) **Latency constraint** — measured on the actual target MCU, not estimated from FLOPs. CMSIS-NN kernel performance varies non-linearly with tensor shapes (some shapes align with SIMD width, others don't). The search must include a hardware-in-the-loop latency measurement or an accurate latency lookup table.

(5) **Channel width alignment** — channels should be multiples of 4 (for CMSIS-NN's SIMD packing of 4 INT8 values into 32-bit registers). Non-aligned channel counts waste SIMD lanes. The search space should only include channel widths in {4, 8, 16, 32, 64, ...}.

MCUNet (MIT) demonstrated this approach: the search found architectures that fit in 256 KB SRAM and achieved 70.7% ImageNet top-1 accuracy — impossible with standard MobileNet architectures that exceed the SRAM budget.

> **Napkin Math:** Search space: 5 blocks × 4 kernel sizes × 6 channel widths × 3 expansion ratios = 360 candidate ops per block. Total architectures: 360⁵ ≈ 6 × 10¹² — must use efficient search (e.g., one-shot NAS). Constraint evaluation per candidate: SRAM check (~1ms simulation), flash check (parameter count), latency check (lookup table, ~0.1ms). Full search: ~10,000 candidates evaluated × 1.1ms = 11 seconds. Training the supernet: ~24 GPU-hours. Total: 1-2 days vs months for brute-force search.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### ⏱️ Latency & Throughput

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Keyword Spotting Latency Budget</b> · <code>latency</code></summary>

**Interviewer:** "Your keyword spotting system must respond within 500ms of the user saying the wake word. Budget the time across the full pipeline: audio capture, feature extraction, inference, and action."

**Common Mistake:** "The model inference is the bottleneck — optimize the model." On MCUs, the non-ML pipeline stages often dominate.

**Realistic Solution:** Break down the 500ms budget:

(1) **Audio capture window:** The model needs ~1 second of audio context to detect a keyword. But you don't wait for the full second — you use a sliding window. The wake word might start at any point in the buffer. Worst case: the word starts right after the last inference → you must wait for the next window. At 20ms hop size: worst-case delay = **20ms**.

(2) **Feature extraction (Mel spectrogram):** 1 second of audio at 16 kHz = 16,000 samples. FFT (512-point) with 20ms hop = 50 frames. Each FFT: ~5,000 operations. 40 Mel filter banks per frame. Total: ~500K operations. On Cortex-M4 at 168 MHz: **~3ms**.

(3) **Model inference:** Typical keyword spotting model (DS-CNN): ~500K MACs. With CMSIS-NN SIMD: **~1.5ms**.

(4) **Post-processing:** Smoothing (average over 3 consecutive predictions to reduce false positives): **<0.1ms**.

(5) **Action (GPIO toggle, wake main processor):** **<0.1ms**.

**Total: ~25ms worst case.** The 500ms budget is generous — the real constraint is power, not latency. The system spends 475ms sleeping between inference cycles, and the duty cycle determines battery life.

> **Napkin Math:** Audio capture: 20ms (hop size). Feature extraction: 3ms. Inference: 1.5ms. Post-processing + action: 0.2ms. Total: **24.7ms**. Budget: 500ms. Utilization: 24.7/500 = 5%. The MCU sleeps 95% of the time. At 5mW active, 10μW sleep: average power = 0.05 × 5 + 0.95 × 0.01 = **0.26 mW**. On a 225 mAh coin cell (3.3V = 742 mWh): 742/0.26 = **2,854 hours ≈ 119 days**.

**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Sensor Pipeline Without Drops</b> · <code>latency</code></summary>

**Interviewer:** "Your vibration sensor samples at 1000 Hz (1 sample per ms). Your anomaly detection model processes windows of 256 samples and takes 50ms to run inference. How do you ensure no samples are dropped?"

**Common Mistake:** "Process every 256 samples: 256ms of data, 50ms inference, repeat." This leaves a 50ms gap where samples are lost.

**Realistic Solution:** Use a **double-buffer (ping-pong) scheme** with DMA:

Buffer A and Buffer B each hold 256 samples. The ADC writes to Buffer A via DMA (hardware-driven, zero CPU involvement). When Buffer A is full (after 256ms), the DMA automatically switches to Buffer B and triggers an interrupt. The CPU processes Buffer A (50ms inference) while the DMA fills Buffer B. When Buffer B is full (256ms later), the DMA switches back to A, and the CPU processes B.

The key constraint: inference (50ms) must complete before the other buffer fills (256ms). Since 50ms < 256ms, there's 206ms of slack — the CPU can sleep or handle other tasks.

For overlapping windows (common in audio/vibration): use a circular buffer with a stride. The model processes samples [0:255], then [128:383], then [256:511], etc. (50% overlap, stride=128). Now the deadline is tighter: 128ms between inferences. Still feasible: 50ms < 128ms.

> **Napkin Math:** Sensor: 1000 Hz. Window: 256 samples = 256ms. Inference: 50ms. Ping-pong: 50ms processing / 256ms window = 19.5% CPU utilization. With 50% overlap (stride=128): 128ms between windows. 50ms / 128ms = 39% utilization. With 75% overlap (stride=64): 64ms between windows. 50ms / 64ms = 78% utilization — tight but feasible. With 87.5% overlap (stride=32): 32ms < 50ms → **drops samples**. Maximum overlap: stride ≥ 50 samples (50ms at 1 kHz).

> **Key Equation:** $\text{Max Overlap} = 1 - \frac{t_{\text{inference}}}{t_{\text{window}}} = 1 - \frac{50}{256} = 80.5\%$

**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Sub-Millisecond Fault Detector</b> · <code>latency</code></summary>

**Interviewer:** "You're designing a vibration anomaly detector for a high-speed motor (10,000 RPM). A bearing fault must be detected within 1ms of onset to trigger an emergency shutdown before catastrophic failure. Your model takes 0.5ms to run. Design the interrupt-driven inference pipeline."

**Common Mistake:** "Sample at 1 kHz and run inference every 1ms." At 10,000 RPM, the motor completes one revolution every 6ms. A 1 kHz sample rate gives only 6 samples per revolution — far too few to detect a bearing fault signature (typically 5-20 kHz vibration).

**Realistic Solution:** Design an interrupt-driven pipeline with zero buffering delay:

(1) **Sampling:** ADC at 50 kHz (50 samples per revolution at 10,000 RPM). DMA fills a 50-sample circular buffer (1ms of data).

(2) **Feature extraction:** Instead of FFT (too slow for 1ms budget), use a **matched filter** — a pre-computed template of the fault vibration signature. Cross-correlation of 50 samples with the template: 50 MACs. On Cortex-M7 with SIMD: **<1μs**.

(3) **Detection logic:** Compare the correlation output against a threshold. If exceeded for 2 consecutive windows (to reject noise), trigger the fault interrupt. Logic: **<0.1μs**.

(4) **Emergency shutdown:** GPIO pin drives the motor controller's emergency stop input. Hardware response: **<10μs**.

**Total pipeline: <1.1μs per window.** But the 0.5ms ML model is for a more sophisticated classifier that runs in parallel on a lower-priority interrupt. The matched filter provides the hard real-time guarantee (<1ms); the ML model provides classification (bearing fault vs imbalance vs misalignment) for the maintenance log, with a 1ms latency that's acceptable for logging but not for shutdown.

This is a **dual-path architecture**: a fast, simple detector for safety (hard real-time) and a slower, accurate classifier for diagnostics (soft real-time).

> **Napkin Math:** Fast path: 50-sample matched filter at 50 kHz. Correlation: 50 MACs / 1.92 GMAC/s = 26ns. Threshold + logic: 100ns. GPIO: 10μs. Total: **~10μs** — 100× under the 1ms deadline. Slow path (ML model): 500μs inference, runs every 1ms window. Provides fault type classification for maintenance scheduling. If the fast path triggers shutdown, the slow path result is logged but not used for the safety decision.

**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
</details>

---

### ⚡ Power & Thermal

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Coin Cell Budget</b> · <code>power</code></summary>

**Interviewer:** "Your device runs on a CR2032 coin cell battery (225 mAh, 3.0V). Inference costs 50 mW for 10ms. The device must last 1 year. How many inferences per day can you afford?"

**Common Mistake:** "225 mAh × 3V = 675 mWh. At 50 mW × 10ms = 0.5 mJ per inference. 675 mWh / 0.5 mJ = 4.86 billion inferences." This ignores the quiescent power draw and battery self-discharge.

**Realistic Solution:** Account for the full power budget:

(1) **Battery capacity:** 225 mAh × 3.0V = 675 mWh. But CR2032 self-discharge rate is ~1% per year → effective capacity over 1 year: ~668 mWh. Also, capacity drops at low temperatures and high discharge rates. Derate to **600 mWh** for reliability.

(2) **Quiescent power:** The MCU in deep sleep + RTC + voltage regulator: ~5 μW. Over 1 year: 5 μW × 8760 hours = **43.8 mWh**. This is non-negotiable — it's always on.

(3) **Available for inference:** 600 - 43.8 = **556.2 mWh**.

(4) **Energy per inference:** 50 mW × 10ms = 0.5 mJ = 0.000139 mWh. But also account for wake-up energy (transitioning from deep sleep to active): ~0.1 mJ. And sensor read (ADC + amplifier): ~0.2 mJ. Total per inference cycle: **0.8 mJ = 0.000222 mWh**.

(5) **Total inferences:** 556.2 mWh / 0.000222 mWh = **2,505,405 inferences per year**.

(6) **Per day:** 2,505,405 / 365 = **6,864 inferences per day** = ~286 per hour = ~4.8 per minute.

> **Napkin Math:** Battery: 600 mWh effective. Quiescent: 44 mWh/year. Available: 556 mWh. Per inference: 0.8 mJ (inference + wake + sensor). Inferences/year: 556 / 0.000222 = 2.5M. Per day: **6,864**. If you need 1 inference per second (86,400/day): 86,400 × 0.000222 = 19.2 mWh/day × 365 = 7,008 mWh/year. Battery only has 600 mWh. Need 11.7× more battery or 11.7× less energy per inference.

> **Key Equation:** $N_{\text{inferences}} = \frac{E_{\text{battery}} - E_{\text{quiescent}} \times t}{E_{\text{inference}} + E_{\text{wake}} + E_{\text{sensor}}}$

**📖 Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Solar Harvesting Budget</b> · <code>power</code></summary>

**Interviewer:** "You're designing a solar-powered wildlife monitor that runs image classification on a Cortex-M7. The device has a small solar panel (50mm × 50mm), a 0.47F supercapacitor, and no battery. Design the power budget: when can you run inference, and how do you handle cloudy days?"

**Common Mistake:** "Solar panel generates power during the day, so run inference during the day." This ignores the mismatch between solar harvest rate and inference energy cost.

**Realistic Solution:** Build the energy budget from physics:

**Solar harvest:** A 50×50mm panel at ~20% efficiency in direct sunlight (~1000 W/m²): 0.0025 m² × 1000 × 0.20 = **0.5W peak**. Average over a sunny day (6 effective sun-hours): 0.5W × 6h = **3 Wh/day**. Cloudy day: ~0.5 Wh/day (10× less).

**Supercapacitor storage:** 0.47F × (3.3V)² / 2 = **2.56 J = 0.71 mWh**. This is a tiny buffer — it can power ~14 inferences (at 0.05 mWh each) before it's empty.

**Inference cost:** Cortex-M7 at 100 mW active × 50ms = 5 mJ per inference. Camera capture: 200 mW × 100ms = 20 mJ. Total per inference cycle: **25 mJ = 0.00694 mWh**.

**Quiescent:** MCU deep sleep + voltage regulator: 10 μW. Over 24 hours: 0.24 mWh.

**Sunny day budget:** 3000 mWh - 0.24 mWh = 2999.76 mWh for inference. 2999.76 / 0.00694 = **432,244 inferences/day** — more than enough.

**Cloudy day budget:** 500 mWh available. 500 / 0.00694 = **72,046 inferences/day**.

**The real constraint:** The supercapacitor can only buffer 0.71 mWh. If a cloud passes and solar drops to zero, the device has energy for 0.71 / 0.00694 = **102 inferences** before shutdown. At 1 inference/second: **102 seconds** of operation.

**Design:** Use an energy-aware scheduler. Monitor supercapacitor voltage. When voltage > 3.0V (well-charged): run inference at full rate. When voltage drops below 2.5V: reduce to 1 inference per 10 seconds. Below 2.0V: sleep until voltage recovers. On cloudy days, the device operates intermittently — capturing images when energy is available and sleeping when it's not. Store images in flash during sleep for batch processing when the sun returns.

> **Napkin Math:** Sunny: 0.5W harvest, 0.125W average consumption (1 inf/s × 25mJ + 10μW quiescent) → net positive, supercap stays full. Cloudy: 0.05W harvest, 0.125W consumption → net negative, supercap drains in 0.71mWh / (125-50)mW = 9.5 seconds. Must reduce to 0.05W consumption: 1 inference every 0.5s × 25mJ = 50mW → balanced at 0.05W harvest. Effective rate on cloudy days: **2 inferences/second** (vs 40/s on sunny days).

**📖 Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)
</details>

---

### 🔧 Model Optimization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The SRAM Overflow Options</b> · <code>optimization</code></summary>

**Interviewer:** "Your model fits in flash (weights: 180 KB, flash: 1 MB) but the tensor arena needs 240 KB and your SRAM is only 200 KB. Walk through the optimization options in order of engineering effort."

**Common Mistake:** "Use a smaller model." That's the nuclear option — there are several less destructive approaches.

**Realistic Solution:** Optimization ladder for SRAM overflow:

**Step 1: Optimize the memory planner** (effort: minutes). TFLite Micro's default memory planner may not find the optimal tensor layout. Try different planner strategies (e.g., greedy by size vs greedy by lifetime). Sometimes reordering operator execution reduces peak memory by 10-20%. Check if any tensors have unnecessarily long lifetimes.

**Step 2: In-place operations** (effort: hours). Configure ReLU, batch norm, and element-wise add to operate in-place (output overwrites input). This eliminates duplicate buffers at those layers. Typical savings: 10-15% of peak memory.

**Step 3: Reduce input resolution** (effort: minutes). If the model accepts 96×96 input, try 80×80 or 64×64. Activation memory scales quadratically with spatial dimensions: (64/96)² = 0.44× → 44% reduction. Accuracy impact: typically 1-3% for classification.

**Step 4: Tiled execution** (effort: days). Split the largest layer's spatial computation into tiles. Process the feature map in 4 quadrants, each requiring 1/4 of the activation memory. Requires custom kernel modifications.

**Step 5: Architecture change** (effort: weeks). Replace the bottleneck layer with a more memory-efficient alternative (e.g., reduce the expansion ratio in an inverted residual block from 6 to 4). Requires retraining.

> **Napkin Math:** Baseline: 240 KB peak. Step 1 (planner): -10% → 216 KB (still over). Step 2 (in-place): -12% → 190 KB ✓ (fits in 200 KB with 10 KB headroom). If Step 2 isn't enough: Step 3 (96→80 resolution): 240 × (80/96)² = 167 KB ✓. Total engineering time for Steps 1+2: ~4 hours. For Step 3: 5 minutes + accuracy validation.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🚀 Deployment

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The FOTA Update Risk</b> · <code>deployment</code></summary>

**Interviewer:** "You have 10,000 sensor nodes deployed in a warehouse, each running a vibration anomaly detection model on a Cortex-M4. You need to update the model. The nodes communicate via LoRaWAN (250 bytes/second effective throughput). How do you update them, and what happens if the update fails?"

**Common Mistake:** "Send the new firmware over LoRaWAN and flash it." At 250 bytes/second, a 200 KB model takes 200,000/250 = 800 seconds = 13 minutes per device. With 10,000 devices sharing the LoRa channel: years.

**Realistic Solution:** FOTA (Firmware Over-The-Air) for constrained networks requires a different approach:

(1) **Delta updates** — don't send the full model. Compute a binary diff between the old and new model weights. If only 10% of weights changed (fine-tuning), the delta is ~20 KB instead of 200 KB. Transfer time: 20,000/250 = 80 seconds per device.

(2) **Multicast** — LoRaWAN Class C supports multicast. Send the update once, all 10,000 devices receive it simultaneously. Transfer time: 80 seconds total (not per device).

(3) **A/B flash partitioning** — the MCU's 1 MB flash is split: 500 KB for the running firmware (slot A), 500 KB for the update (slot B). The new model is written to slot B while slot A continues running. After verification (CRC check + test inference on a known input), the bootloader atomically swaps the active slot pointer.

(4) **Failure recovery** — if the CRC check fails, the device stays on slot A and reports the failure. If the device boots from slot B and the watchdog timer fires (model crashes), the bootloader automatically reverts to slot A. The device is never bricked.

(5) **Staged rollout** — update 100 devices first (1% of fleet). Monitor their anomaly detection accuracy for 24 hours. If no degradation, update the remaining 9,900.

> **Napkin Math:** Full model: 200 KB. Delta: 20 KB. LoRaWAN multicast: 20 KB / 250 B/s = 80 seconds. Verification: CRC (1ms) + test inference (50ms) = 51ms. Swap: atomic pointer write (1ms). Total per device: 80s transfer + 0.05s verify + 0.001s swap. Fleet of 10,000 via multicast: **80 seconds** + staged validation (24 hours for safety). Without delta/multicast: 200 KB × 10,000 / 250 B/s = 8,000,000 seconds = **92.6 days**.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

---

### 📊 Monitoring & Reliability

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Offline Drift Detector</b> · <code>monitoring</code></summary>

**Interviewer:** "Your deployed anomaly detector starts producing false positives after 3 months. The device has no cloud connection — it operates fully offline. How do you detect and handle model drift on a device with 256 KB SRAM and no internet?"

**Common Mistake:** "Upload data to the cloud for analysis." There's no cloud connection. You must handle this entirely on-device.

**Realistic Solution:** On-device drift detection with minimal resources:

(1) **Running statistics** — maintain exponential moving averages of the model's input feature statistics (mean and variance of each input channel). Storage: 2 floats × N channels × 4 bytes = ~64 bytes for a 8-channel sensor. When the running mean drifts beyond 3σ of the baseline (computed at deployment), flag a drift event. This detects covariate shift (e.g., sensor degradation, environmental change).

(2) **Prediction distribution monitoring** — track the distribution of the model's output confidence scores. A healthy model produces mostly high-confidence predictions (normal) with occasional low-confidence ones (anomalies). If the ratio of low-confidence predictions exceeds a threshold (e.g., >30% of predictions in the last hour), the model is likely seeing OOD data.

(3) **Self-calibration** — store a small set of "golden" reference inputs in flash (10 known-normal vibration signatures, ~5 KB). Periodically (once per hour), run inference on these references. If the model's predictions on known-normal inputs start drifting (confidence drops below 0.95), the model or the sensor has degraded.

(4) **Graceful response** — when drift is detected: (a) increase the anomaly threshold to reduce false positives (accepting more false negatives), (b) activate an LED indicator for maintenance personnel, (c) log the drift event with timestamp to flash for later retrieval, (d) if drift exceeds a critical threshold, fall back to a simple threshold-based detector (no ML) until the device is serviced.

> **Napkin Math:** Running statistics: 64 bytes RAM. Golden references: 5 KB flash. Hourly self-test: 10 inferences × 50ms = 500ms per hour = 0.014% CPU overhead. Drift detection latency: 1 hour (self-test interval). Storage for drift log: 20 bytes per event × 100 events = 2 KB flash. Total resource cost: 64 bytes RAM + 7 KB flash — negligible on a 256 KB SRAM / 1 MB flash device.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

---

### 🔒 Security

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Flash Extraction Attack</b> · <code>security</code></summary>

**Interviewer:** "An attacker has physical access to your deployed MCU. They want to extract your proprietary model weights from flash memory. How do they do it, and what can you do to prevent it?"

**Common Mistake:** "Flash memory is internal to the MCU — it can't be read externally." Many MCUs have debug interfaces (JTAG/SWD) that provide full memory access by default.

**Realistic Solution:** Attack vectors for model extraction:

(1) **Debug interface (JTAG/SWD)** — most Cortex-M MCUs ship with JTAG/SWD enabled by default. An attacker connects a $20 debug probe, halts the CPU, and dumps the entire flash contents in seconds. **Defense:** set the Read-Out Protection (RDP) fuse to Level 2 (permanent, irreversible). This disables all debug access. Caution: Level 2 also prevents firmware updates via the debug interface — you must have FOTA capability before enabling it.

(2) **Side-channel analysis** — even with RDP enabled, an attacker can monitor power consumption during inference. Different model weights produce different power traces (data-dependent power consumption in MAC operations). With enough traces (~10,000 inferences), they can statistically recover individual weights. **Defense:** add random dummy operations between layers to decorrelate power traces from weight values. Cost: ~10% performance overhead.

(3) **Fault injection** — voltage glitching or clock glitching can cause the MCU to skip the RDP check, re-enabling debug access. **Defense:** use MCUs with hardware fault detection (e.g., STM32 with anti-tamper pins, voltage monitoring, clock security system). These detect glitches and trigger a secure erase of flash.

(4) **Decapping** — physically removing the MCU's package and reading flash contents with an electron microscope. **Defense:** essentially none at the MCU level. Use MCUs with active mesh shields (e.g., secure elements like ATECC608) that detect physical intrusion and erase keys.

Practical recommendation for most applications: RDP Level 2 + FOTA capability blocks 99% of attackers. Side-channel and decapping attacks require expensive equipment and expertise — only worth defending against for high-value IP.

> **Napkin Math:** Attack cost: JTAG dump with RDP Level 0 (default): $20 probe + 5 minutes. With RDP Level 1: $200 glitching setup + 1 day. With RDP Level 2: $10,000+ side-channel setup + weeks, or $50,000+ decapping. Defense cost: RDP Level 2: 1 line of code (free). Side-channel countermeasures: 10% performance overhead. Anti-tamper hardware: +$2 per unit. Choose defense proportional to model value.

**📖 Deep Dive:** [Volume II: Security & Privacy](https://mlsysbook.ai/vol2/security_privacy.html)
</details>

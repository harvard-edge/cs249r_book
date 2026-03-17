# Round 2: Constraints & Trade-offs ⚖️

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_TinyML_Systems.md">🔬 Round 1</a> ·
  <a href="02_TinyML_Constraints.md">⚖️ Round 2</a> ·
  <a href="03_TinyML_Ops_and_Deployment.md">🚀 Round 3</a> ·
  <a href="04_TinyML_Visual_Debugging.md">🖼️ Round 4</a> ·
  <a href="05_TinyML_Advanced.md">🔬 Round 5</a>
</div>

---

Every TinyML system is a negotiation between competing constraints: SRAM budgets, flash capacity, quantization fidelity, compute throughput, latency deadlines, and power caps — all measured in kilobytes, microseconds, and milliwatts. This round tests whether you can reason about these trade-offs quantitatively on microcontrollers where every byte and every clock cycle is a design decision.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/tinyml/02_TinyML_Constraints.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 📐 Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The MAC Budget</b> · <code>compute</code></summary>

- **Interviewer:** "Your Cortex-M4 runs at 168 MHz with no FPU. Your keyword spotting model requires 5 million INT8 multiply-accumulate (MAC) operations per inference. Can you run inference in under 100ms?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "168 MHz means 168 million operations per second, so 5M MACs takes 30ms — easy." This assumes one MAC per clock cycle, which is wrong for a scalar processor without SIMD.

  **Realistic Solution:** A Cortex-M4 without SIMD extensions executes one 32-bit instruction per cycle, but an INT8 MAC requires multiple instructions: load two 8-bit operands (1-2 cycles), multiply (1 cycle), accumulate (1 cycle), store (1 cycle). Effective throughput: ~1 MAC per 3-4 cycles. At 168 MHz: 168M / 4 = **42 million MACs/second**. Time for 5M MACs: 5M / 42M = **119ms** — over budget.

  With CMSIS-NN and the Cortex-M4's DSP extension (SIMD): the `SMLAD` instruction performs two 16-bit MACs per cycle. Packing two INT8 values into 16-bit lanes: effective throughput = 2 MACs per cycle = **336 million MACs/second**. Time: 5M / 336M = **14.9ms** — well within budget. The lesson: on MCUs, SIMD is not optional — it's the difference between feasible and infeasible.

  > **Napkin Math:** Without SIMD: 168 MHz / 4 cycles per MAC = 42 MMAC/s. 5M MACs / 42M = 119ms ✗. With CMSIS-NN SIMD: 168 MHz × 2 MACs/cycle = 336 MMAC/s. 5M / 336M = 14.9ms ✓. SIMD gives **8× speedup** for INT8 workloads on Cortex-M4.

  > **Key Equation:** $t_{\text{inference}} = \frac{\text{Total MACs}}{\text{Clock} \times \text{MACs per cycle}}$

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The MCU Throughput Ceiling</b> · <code>compute</code></summary>

- **Interviewer:** "Your anomaly detection model needs 10 million MACs per inference. You need 10 inferences per second on a Cortex-M4 at 168 MHz with CMSIS-NN SIMD. Does the math work? And what happens to the rest of the system?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "336 MMAC/s with SIMD. 10M × 10 = 100 MMAC/s needed. That's only 30% utilization — plenty of headroom." This ignores the non-MAC overhead.

  **Realistic Solution:** The 336 MMAC/s is the peak throughput for pure MAC operations. Real inference includes: (1) **Memory loads** — fetching weights and activations from SRAM takes cycles. If data isn't in the CPU's small cache (typically 4-16 KB on M4), every load stalls the pipeline. (2) **Activation functions** — ReLU is cheap (1 cycle), but sigmoid/tanh require lookup tables or polynomial approximation (10-20 cycles each). (3) **Requantization** — between layers, INT32 accumulator results must be rescaled back to INT8 (shift, round, clamp: ~5 cycles per element). (4) **Loop overhead** — index computation, bounds checking, pointer arithmetic.

  Realistic throughput: ~40-60% of peak = 134-202 MMAC/s. At 150 MMAC/s effective: 100M MACs/s needed / 150 MMAC/s = **67% CPU utilization for ML alone**. That leaves 33% for: sensor data acquisition (ADC reads, DMA), communication (UART/SPI to radio), and application logic (thresholds, state machines). If the sensor pipeline needs 20% and comms need 10%, you're at 97% — no headroom for interrupt latency spikes. You either need a faster MCU (Cortex-M7 at 480 MHz) or a smaller model.

  > **Napkin Math:** Peak: 336 MMAC/s. Effective (60% utilization): 202 MMAC/s. ML workload: 100 MMAC/s = 50% of effective throughput. Sensor acquisition (16 kHz ADC, FFT): ~30 MMAC/s = 15%. UART comms: ~10 MMAC/s = 5%. Total: 70% utilization. Headroom: 30% — tight but feasible. At 80% effective utilization threshold (reliability margin): 100/0.8 = 125 MMAC/s needed → 62% of effective capacity. Feasible with careful scheduling.

  📖 **Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The MCU Roofline</b> · <code>compute</code></summary>

- **Interviewer:** "Construct a roofline model for a Cortex-M7 with 512 KB SRAM, a 64-bit AXI bus at 240 MHz, and SIMD capable of 4 INT8 MACs per cycle at 480 MHz. Where is the ridge point, and what does it tell you about which models are feasible?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The roofline is the same concept as for GPUs — just smaller numbers." The concept is the same, but the implications are radically different because the memory hierarchy is flat (no HBM, no L2 cache in most M7 variants).

  **Realistic Solution:** Build the roofline:

  **Peak compute:** 480 MHz × 4 MACs/cycle = **1.92 GMAC/s** (1.92 GOPS INT8).

  **Peak memory bandwidth:** 64-bit AXI bus at 240 MHz = 64/8 × 240M = **1.92 GB/s** from SRAM. (In practice, bus contention with DMA and peripherals reduces this to ~1.2-1.5 GB/s.)

  **Ridge point:** 1.92 GOPS / 1.92 GB/s = **1.0 Ops/Byte** (or ~1.6 Ops/Byte with realistic bandwidth).

  This is extraordinarily low compared to GPUs (ridge ~300 Ops/Byte). It means: almost every neural network layer is **compute-bound** on an MCU, not memory-bound. A standard 3×3 Conv2D has arithmetic intensity of ~18 Ops/Byte (9 MACs × 2 ops / 1 byte per weight) — well above the ridge. This is the opposite of GPUs, where most workloads are memory-bound.

  Implication: on MCUs, reducing FLOPs directly reduces latency (unlike GPUs where reducing FLOPs often doesn't help because you're memory-bound). Depthwise separable convolutions (which reduce FLOPs 8-9×) give nearly proportional speedups on MCUs — a property that doesn't hold on GPUs.

  > **Napkin Math:** Ridge point: 1.0-1.6 Ops/Byte. Conv2D 3×3 intensity: ~18 Ops/Byte → compute-bound ✓. Depthwise Conv2D 3×3 intensity: ~9 Ops/Byte → compute-bound ✓. Fully connected layer (1024→1024): 2 Ops/Byte → compute-bound ✓. Pointwise Conv2D (1×1): ~2 Ops/Byte → compute-bound ✓. On an MCU, everything is compute-bound. On a GPU (ridge ~300), everything except large MatMuls is memory-bound. This is why model optimization strategies differ radically between the two platforms.

  > **Key Equation:** $\text{Ridge Point}_{\text{MCU}} = \frac{\text{Peak GOPS}}{\text{Bus Bandwidth (GB/s)}}$

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)

  </details>

</details>

---

### 🧠 Memory Systems

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Tensor Arena Overflow</b> · <code>memory</code></summary>

- **Interviewer:** "Your model's tensor arena needs 210 KB for peak activation memory, but your Cortex-M4 only has 200 KB of SRAM available (after firmware and stack). The model fits in flash. You can't change the model architecture. How do you fit it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Reduce the input resolution." The question says you can't change the model — input resolution is part of the model's expected input shape.

  **Realistic Solution:** The 210 KB peak occurs at one specific point in the model — typically where two large activation tensors coexist (e.g., the input and output of the largest layer). TFLite Micro's memory planner already optimizes tensor lifetimes, but you can push further:

  (1) **In-place operations** — for layers where the output has the same shape as the input (ReLU, batch norm), configure the runtime to write the output directly into the input buffer. Saves one tensor's worth of memory at that layer.

  (2) **Operator reordering** — if the memory planner's default order creates a peak at layer 5, check if reordering independent branches (in a multi-branch architecture) shifts the peak. Sometimes processing branch A before branch B reduces the maximum number of simultaneously live tensors.

  (3) **Tensor splitting** — split the largest layer's computation into tiles. Process the input in 4 spatial tiles, each requiring 1/4 of the activation memory. Peak drops from 210 KB to ~60 KB for that layer. Trade-off: 4× more kernel invocations (overhead ~5-10%).

  (4) **Scratch buffer sharing** — some operators (depthwise conv, transpose) use temporary scratch buffers. Ensure these share the same memory region and are never live simultaneously.

  > **Napkin Math:** Peak at layer 5: input tensor (100 KB) + output tensor (110 KB) = 210 KB. With in-place ReLU after layer 5: output overwrites input → peak = 110 KB. With 2× tiling: input tile (50 KB) + output tile (55 KB) = 105 KB. With both: 55 KB peak at layer 5. New global peak shifts to layer 8: 180 KB. Fits in 200 KB with 20 KB headroom.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)

  </details>

</details>

---

### 🔢 Numerical Representation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Zero-Point Question</b> · <code>quantization</code></summary>

- **Interviewer:** "A junior engineer says 'to quantize a model to INT8, just round each FP32 weight to the nearest integer.' Why is this wrong, and what is a zero-point?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Rounding FP32 to INT8 is quantization." This destroys the model because FP32 weights are typically in the range [-0.1, 0.1] — rounding them all to 0 gives you a zero model.

  **Realistic Solution:** Quantization is an affine mapping from the float range to the integer range, not simple rounding. The formula is: $x_q = \text{round}(x_f / s) + z$, where $s$ is the scale (step size) and $z$ is the zero-point (the integer value that represents float 0.0).

  Why the zero-point matters: if your weights range from [-0.1, 0.1], the scale is $s = 0.2 / 255 = 0.000784$. The zero-point $z = \text{round}(0 / s) + 128 = 128$ (mapping float 0.0 to the middle of the INT8 range). Now -0.1 maps to 0, 0.0 maps to 128, and 0.1 maps to 255 — the full INT8 range is utilized.

  Without a zero-point (symmetric quantization around 0): INT8 range [-128, 127] maps to [-0.1, 0.1]. This works for weights (which are roughly symmetric) but wastes range for activations after ReLU, which are always ≥ 0. An asymmetric scheme with zero-point maps [0, max_activation] to [0, 255], using the full unsigned INT8 range.

  > **Napkin Math:** Weight range [-0.1, 0.1]. Naive rounding: all weights → 0. Model is dead. Proper quantization: scale = 0.2/255 = 0.000784. Weight 0.05 → round(0.05/0.000784) + 128 = 192. Weight -0.03 → round(-0.03/0.000784) + 128 = 90. Dequantized: (192-128) × 0.000784 = 0.0502, (90-128) × 0.000784 = -0.0298. Error: <0.001 — acceptable.

  > **Key Equation:** $x_q = \text{round}\left(\frac{x_f}{s}\right) + z, \quad x_f = s \times (x_q - z)$

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Per-Channel Trade-off</b> · <code>quantization</code></summary>

- **Interviewer:** "Your per-tensor quantized keyword spotting model loses 5% accuracy going from FP32 to INT8. Switching to per-channel quantization recovers 3%. Your colleague says 'always use per-channel — it's strictly better.' Why might you still choose per-tensor on an MCU?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Per-channel is always better because it has more quantization parameters." More parameters means more accuracy, but ignores the system cost.

  **Realistic Solution:** Per-channel quantization assigns a separate scale and zero-point to each output channel of a convolution. For a layer with 64 output channels, that's 64 scales and 64 zero-points instead of 1 each. The accuracy benefit comes from each channel having its own optimal range.

  The system costs on an MCU: (1) **Code size** — per-channel requantization requires a loop over channels with per-channel scale/zero-point lookups. Per-tensor uses a single multiply-shift. The per-channel kernel is ~2× larger in flash. On a 512 KB flash MCU, this matters. (2) **Inference speed** — per-channel requantization between layers requires loading the scale array from memory for each output element. On a Cortex-M4 without cache, this adds ~2-5% overhead per layer when using optimized CMSIS-NN kernels (the scale is loaded once per channel and amortized over the H×W output tile). The overhead increases to ~15-20% only for very small feature maps (4×4 or 1×1) where the per-channel load cost isn't amortized. (3) **CMSIS-NN support** — CMSIS-NN's optimized kernels support per-channel natively, but older versions or custom runtimes may not, forcing a fallback to unoptimized C code (5-10× slower).

  Decision framework: use per-channel when accuracy is critical (medical, safety) and the MCU has sufficient flash and CMSIS-NN support. Use per-tensor when flash is tight, latency is critical, or the accuracy loss is acceptable for the application (e.g., keyword spotting where 90% vs 93% accuracy doesn't change user experience).

  > **Napkin Math:** Per-tensor: 1 scale + 1 zero-point = 8 bytes per layer. 20 layers = 160 bytes. Requantization: 1 multiply + 1 shift per element. Per-channel: 64 scales + 64 zero-points = 512 bytes per layer. 20 layers = 10 KB. Requantization: 1 multiply + 1 shift + 1 load per element per channel. Overhead: ~18% more cycles. On a 100ms inference budget: per-tensor = 85ms, per-channel = 100ms. Accuracy: per-tensor = 87%, per-channel = 90%. If the deadline is 100ms: per-channel barely fits. If 90ms: per-tensor only.

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

---

### 🏗️ Architecture → System Cost

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Depthwise Separable FLOP Ratio</b> · <code>architecture</code></summary>

- **Interviewer:** "Your colleague says 'depthwise separable convolutions are just a trick to reduce parameters.' Calculate the exact FLOP ratio between a standard 3×3 Conv2D and a depthwise separable Conv2D for 64 input channels and 64 output channels on a 32×32 feature map. Why does this ratio matter more on an MCU than on a GPU?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Depthwise separable is about 3× cheaper." The actual ratio is much larger, and the reason it matters more on MCUs is the key insight.

  **Realistic Solution:**

  **Standard Conv2D:** Each output pixel requires $K^2 \times C_{in}$ MACs. Total: $K^2 \times C_{in} \times C_{out} \times H \times W = 3^2 \times 64 \times 64 \times 32 \times 32 = 37,748,736$ MACs.

  **Depthwise separable:** Depthwise (one filter per channel): $K^2 \times C_{in} \times H \times W = 9 \times 64 \times 1024 = 589,824$ MACs. Pointwise (1×1 conv to mix channels): $C_{in} \times C_{out} \times H \times W = 64 \times 64 \times 1024 = 4,194,304$ MACs. Total: **4,784,128** MACs.

  **Ratio:** 37.7M / 4.8M = **7.9×** fewer MACs. Not 3× — nearly 8×.

  Why this matters more on MCUs: on a GPU, most convolutions are memory-bound (arithmetic intensity below the ridge point). Reducing FLOPs by 8× doesn't give 8× speedup because you're still waiting for memory. On an MCU, convolutions are compute-bound (ridge point ~1 Ops/Byte). Reducing FLOPs by 8× gives nearly **8× speedup**. This is why MobileNet architectures (built on depthwise separable convolutions) are disproportionately effective on MCUs compared to GPUs.

  > **Napkin Math:** Standard: 37.7M MACs. Depthwise separable: 4.8M MACs. Ratio: 7.9×. On Cortex-M4 (336 MMAC/s SIMD): Standard = 112ms. Depthwise separable = 14ms. On GPU (memory-bound): Standard = 2ms. Depthwise separable = 1.5ms (only 1.3× speedup due to memory bottleneck). The MCU gets 8× speedup; the GPU gets 1.3×.

  > **Key Equation:** $\text{FLOP Ratio} = \frac{K^2 \times C_{in} \times C_{out}}{K^2 \times C_{in} + C_{in} \times C_{out}} = \frac{K^2 \times C_{out}}{K^2 + C_{out}}$

  📖 **Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Operator Support Gap</b> · <code>architecture</code></summary>

- **Interviewer:** "Your person detection model uses MobileNetV2 as the backbone. TFLite Micro on your Cortex-M4 supports 85% of the operators. The remaining 15% include Resize Bilinear (in the feature pyramid), Pad (before some convolutions), and a custom Swish activation. What is your strategy?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Implement the missing operators in C." You can, but custom operator implementations are unoptimized and can be 10-50× slower than CMSIS-NN kernels, potentially blowing your latency budget.

  **Realistic Solution:** Address each unsupported operator with the least-effort fix:

  (1) **Swish activation** (x × sigmoid(x)): Replace with ReLU6 during training. ReLU6 is universally supported and the accuracy difference on person detection is typically <0.5%. Requires retraining (~1 day).

  (2) **Resize Bilinear**: Replace with nearest-neighbor resize (supported) during model export. For feature pyramid networks, the quality difference is minimal because the subsequent convolution layers smooth out the interpolation artifacts. Zero retraining needed — just change the export config.

  (3) **Pad**: Fuse padding into the subsequent convolution by using the `padding='same'` option in the convolution layer itself. Most frameworks can do this automatically during export. If not, implement a simple zero-pad kernel (~20 lines of C, runs in microseconds).

  (4) **Validation**: After all substitutions, re-evaluate on the test set. If accuracy drops >1%, selectively retrain with the substituted operators (quantization-aware fine-tuning for 10 epochs).

  The general principle: never implement a complex custom operator on an MCU if you can replace it with a supported approximation. The supported operators have CMSIS-NN SIMD-optimized kernels; custom ops run on scalar C code at 5-10× lower throughput.

  > **Napkin Math:** Original model: 85% supported (CMSIS-NN, ~15ms) + 15% unsupported (scalar C, ~45ms) = 60ms total. After substitutions: 100% supported = 18ms total (15ms + 3ms for the slightly different ops). Speedup: 3.3×. Accuracy change: -0.3% (Swish→ReLU6) + 0% (resize) + 0% (pad fusion) = -0.3% total. Acceptable.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The MCU NAS Search Space</b> · <code>architecture</code></summary>

- **Interviewer:** "You're designing a Neural Architecture Search (NAS) pipeline to find the optimal model for a Cortex-M4 with 256 KB SRAM and 1 MB flash. What constraints must your search space encode that a standard NAS for desktop/cloud would ignore?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just add a FLOP constraint to the search." FLOPs alone don't capture the MCU's constraints — memory is the binding constraint, not compute.

  **Realistic Solution:** MCU-aware NAS must encode at least five constraints that standard NAS ignores:

  (1) **Peak SRAM constraint** — the tensor arena (activations + scratch buffers) must fit in SRAM at every point during inference. This is not just total activation size — it's the *maximum simultaneously live* activation memory, which depends on the operator execution order. The search must simulate the memory planner for each candidate architecture.

  (2) **Flash constraint** — all weights + the TFLite Micro runtime + application code must fit in flash. Weights in INT8: model_params × 1 byte. Runtime: ~50-100 KB. Application: ~20-50 KB. Available for weights: flash - 150 KB.

  (3) **Operator support constraint** — every operator in the candidate architecture must be supported by the target runtime's kernel library (TFLite Micro, STM32Cube.AI, or microTVM). Architectures with unsupported ops (e.g., certain attention mechanisms, group convolutions with non-standard groups) are infeasible regardless of accuracy. Different runtimes support different operator sets — STM32Cube.AI supports more operators on STM32 targets but is vendor-locked.

  (4) **Latency constraint** — measured on the actual target MCU, not estimated from FLOPs. CMSIS-NN kernel performance varies non-linearly with tensor shapes (some shapes align with SIMD width, others don't). The search must include a hardware-in-the-loop latency measurement or an accurate latency lookup table.

  (5) **Channel width alignment** — channels should be multiples of 4 (for CMSIS-NN's SIMD packing of 4 INT8 values into 32-bit registers). Non-aligned channel counts waste SIMD lanes. The search space should only include channel widths in {4, 8, 16, 32, 64, ...}.

  MCUNet (MIT) demonstrated this approach: the search found architectures that fit in 256 KB SRAM and achieved 70.7% ImageNet top-1 accuracy — impossible with standard MobileNet architectures that exceed the SRAM budget.

  > **Napkin Math:** Search space: 5 blocks × 4 kernel sizes × 6 channel widths × 3 expansion ratios = 360 candidate ops per block. Total architectures: 360⁵ ≈ 6 × 10¹² — must use efficient search (e.g., one-shot NAS). Constraint evaluation per candidate: SRAM check (~1ms simulation), flash check (parameter count), latency check (lookup table, ~0.1ms). Full search: ~10,000 candidates evaluated × 1.1ms = 11 seconds. Training the supernet: ~24 GPU-hours. Total: 1-2 days vs months for brute-force search.

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

---

### ⏱️ Latency & Throughput

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Keyword Spotting Latency Budget</b> · <code>latency</code></summary>

- **Interviewer:** "Your keyword spotting system must respond within 500ms of the user saying the wake word. Budget the time across the full pipeline: audio capture, feature extraction, inference, and action."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model inference is the bottleneck — optimize the model." On MCUs, the non-ML pipeline stages often dominate.

  **Realistic Solution:** Break down the 500ms budget:

  (1) **Audio capture window:** The model needs ~1 second of audio context to detect a keyword. But you don't wait for the full second — you use a sliding window. The wake word might start at any point in the buffer. Worst case: the word starts right after the last inference → you must wait for the next window. At 20ms hop size: worst-case delay = **20ms**.

  (2) **Feature extraction (Mel spectrogram):** 1 second of audio at 16 kHz = 16,000 samples. FFT (512-point) with 20ms hop = 50 frames. Each FFT: ~5,000 operations. 40 Mel filter banks per frame. Total: ~500K operations. On Cortex-M4 at 168 MHz: **~3ms**.

  (3) **Model inference:** Typical keyword spotting model (DS-CNN): ~500K MACs. With CMSIS-NN SIMD: **~1.5ms**.

  (4) **Post-processing:** Smoothing (average over 3 consecutive predictions to reduce false positives): **<0.1ms**.

  (5) **Action (GPIO toggle, wake main processor):** **<0.1ms**.

  **Total: ~25ms worst case.** The 500ms budget is generous — the real constraint is power, not latency. The system spends 475ms sleeping between inference cycles, and the duty cycle determines battery life.

  > **Napkin Math:** Audio capture: 20ms (hop size). Feature extraction: 3ms. Inference: 1.5ms. Post-processing + action: 0.2ms. Total: **24.7ms**. Budget: 500ms. Utilization: 24.7/500 = 5%. The MCU sleeps 95% of the time. At 5mW active, 10μW sleep: average power = 0.05 × 5 + 0.95 × 0.01 = **0.26 mW**. On a 225 mAh coin cell (3.3V = 742 mWh): 742/0.26 = **2,854 hours ≈ 119 days**.

  📖 **Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Sensor Pipeline Without Drops</b> · <code>latency</code></summary>

- **Interviewer:** "Your vibration sensor samples at 1000 Hz (1 sample per ms). Your anomaly detection model processes windows of 256 samples and takes 50ms to run inference. How do you ensure no samples are dropped?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Process every 256 samples: 256ms of data, 50ms inference, repeat." This leaves a 50ms gap where samples are lost.

  **Realistic Solution:** Use a **double-buffer (ping-pong) scheme** with DMA:

  Buffer A and Buffer B each hold 256 samples. The ADC writes to Buffer A via DMA (hardware-driven, zero CPU involvement). When Buffer A is full (after 256ms), the DMA automatically switches to Buffer B and triggers an interrupt. The CPU processes Buffer A (50ms inference) while the DMA fills Buffer B. When Buffer B is full (256ms later), the DMA switches back to A, and the CPU processes B.

  The key constraint: inference (50ms) must complete before the other buffer fills (256ms). Since 50ms < 256ms, there's 206ms of slack — the CPU can sleep or handle other tasks.

  For overlapping windows (common in audio/vibration): use a circular buffer with a stride. The model processes samples [0:255], then [128:383], then [256:511], etc. (50% overlap, stride=128). Now the deadline is tighter: 128ms between inferences. Still feasible: 50ms < 128ms.

  > **Napkin Math:** Sensor: 1000 Hz. Window: 256 samples = 256ms. Inference: 50ms. Ping-pong: 50ms processing / 256ms window = 19.5% CPU utilization. With 50% overlap (stride=128): 128ms between windows. 50ms / 128ms = 39% utilization. With 75% overlap (stride=64): 64ms between windows. 50ms / 64ms = 78% utilization — tight but feasible. With 87.5% overlap (stride=32): 32ms < 50ms → **drops samples**. Maximum overlap: stride ≥ 50 samples (50ms at 1 kHz).

  > **Key Equation:** $\text{Max Overlap} = 1 - \frac{t_{\text{inference}}}{t_{\text{window}}} = 1 - \frac{50}{256} = 80.5\%$

  📖 **Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Sub-Millisecond Fault Detector</b> · <code>latency</code></summary>

- **Interviewer:** "You're designing a vibration anomaly detector for a high-speed motor (10,000 RPM). A bearing fault must be detected within 1ms of onset to trigger an emergency shutdown before catastrophic failure. Your model takes 0.5ms to run. Design the interrupt-driven inference pipeline."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Sample at 1 kHz and run inference every 1ms." At 10,000 RPM, the motor completes one revolution every 6ms. A 1 kHz sample rate gives only 6 samples per revolution — far too few to detect a bearing fault signature (typically 5-20 kHz vibration).

  **Realistic Solution:** Design an interrupt-driven pipeline with zero buffering delay:

  (1) **Sampling:** ADC at 50 kHz (50 samples per revolution at 10,000 RPM). DMA fills a 50-sample circular buffer (1ms of data).

  (2) **Feature extraction:** Instead of FFT (too slow for 1ms budget), use a **matched filter** — a pre-computed template of the fault vibration signature. Cross-correlation of 50 samples with the template: 50 MACs. On Cortex-M7 with SIMD: **<1μs**.

  (3) **Detection logic:** Compare the correlation output against a threshold. If exceeded for 2 consecutive windows (to reject noise), trigger the fault interrupt. Logic: **<0.1μs**.

  (4) **Emergency shutdown:** GPIO pin drives the motor controller's emergency stop input. Hardware response: **<10μs**.

  **Total pipeline: <1.1μs per window.** But the 0.5ms ML model is for a more sophisticated classifier that runs in parallel on a lower-priority interrupt. The matched filter provides the hard real-time guarantee (<1ms); the ML model provides classification (bearing fault vs imbalance vs misalignment) for the maintenance log, with a 1ms latency that's acceptable for logging but not for shutdown.

  This is a **dual-path architecture**: a fast, simple detector for safety (hard real-time) and a slower, accurate classifier for diagnostics (soft real-time).

  > **Napkin Math:** Fast path: 50-sample matched filter at 50 kHz. Correlation: 50 MACs / 1.92 GMAC/s = 26ns. Threshold + logic: 100ns. GPIO: 10μs. Total: **~10μs** — 100× under the 1ms deadline. Slow path (ML model): 500μs inference, runs every 1ms window. Provides fault type classification for maintenance scheduling. If the fast path triggers shutdown, the slow path result is logged but not used for the safety decision.

  📖 **Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)

  </details>

</details>

---

### ⚡ Power & Thermal

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Coin Cell Budget</b> · <code>power</code></summary>

- **Interviewer:** "Your device runs on a CR2032 coin cell battery (225 mAh, 3.0V). Inference costs 50 mW for 10ms. The device must last 1 year. How many inferences per day can you afford?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

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

  📖 **Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Solar Harvesting Budget</b> · <code>power</code></summary>

- **Interviewer:** "You're designing a solar-powered wildlife monitor that runs image classification on a Cortex-M7. The device has a small solar panel (50mm × 50mm), a 0.47F supercapacitor, and no battery. Design the power budget: when can you run inference, and how do you handle cloudy days?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

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

  📖 **Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)

  </details>

</details>

---

### 💾 Flash vs SRAM Trade-offs

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Flash-SRAM Boundary</b> · <code>memory-layout</code></summary>

- **Interviewer:** "A junior engineer asks: 'Why can't we just run the model entirely from flash? It's 1 MB — way bigger than SRAM.' Explain why model weights live in flash but activations must live in SRAM on a Cortex-M4."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Flash is too slow for inference." Flash read speed on STM32F4 is actually quite fast — up to 168 MHz with the ART accelerator (Adaptive Real-Time memory accelerator). The issue is not read speed.

  **Realistic Solution:** The distinction is about read-only vs read-write access patterns:

  **Weights (Flash):** Model weights are read-only during inference — they never change. Flash supports fast sequential reads (0 wait states with ART accelerator on STM32F4 at 168 MHz). Weights can be executed-in-place (XIP) directly from flash with no copy to SRAM. This is critical: a 200 KB model in flash costs 0 KB of SRAM for weights.

  **Activations (SRAM):** Intermediate tensors are read AND written every layer. Flash memory cannot be written at byte granularity during inference — flash writes require page erases (1-2ms per page) and are limited to ~10,000 write cycles before wear-out. A single inference pass writes millions of bytes to activation buffers. SRAM supports unlimited read/write at full bus speed with zero wear.

  **The trap:** Some MCUs (like ESP32-S3) have external PSRAM (pseudo-static RAM) in addition to internal SRAM. PSRAM is read-write but accessed over SPI/QSPI at ~40 MHz — 4-10× slower than internal SRAM at 240 MHz. Placing activations in PSRAM works but inference slows proportionally. On ESP32-S3: internal SRAM inference = 20ms, PSRAM inference = 80-120ms for the same model.

  > **Napkin Math:** STM32F4: 1 MB flash (weights, read-only, fast), 256 KB SRAM (activations, read-write, fast). A 200 KB INT8 model: weights in flash (0 SRAM cost), activations in SRAM (~150 KB peak). Total SRAM needed: 150 KB + 40 KB firmware + 8 KB stack = 198 KB. Fits in 256 KB. If weights were copied to SRAM: 200 + 150 + 48 = 398 KB — doesn't fit. Flash-resident weights are not an optimization; they're a requirement.

  📖 **Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)

  </details>

</details>

---

### 🔄 DMA & Double-Buffering

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The DMA Pipeline for Sensor Data</b> · <code>sensor-pipeline</code> <code>memory-layout</code></summary>

- **Interviewer:** "Your Cortex-M4 reads accelerometer data over SPI at 1.6 kHz (3 axes × 16-bit = 6 bytes per sample = 9.6 KB/s). Without DMA, your firmware polls the SPI bus in a tight loop, consuming 100% CPU. With DMA, the CPU is free to run inference. But your colleague's DMA setup drops 5% of samples. What went wrong, and how do you fix it with double-buffering?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "DMA handles everything automatically — just point it at a buffer and go." DMA requires careful buffer management to avoid overwriting data the CPU hasn't processed yet.

  **Realistic Solution:** The problem is a single-buffer race condition. The DMA writes sensor data into a buffer while the CPU reads from the same buffer for feature extraction. When the DMA write pointer overtakes the CPU read pointer, unprocessed samples are overwritten — hence the 5% drop rate (it happens during inference when the CPU is busy for 30-50ms).

  **Double-buffering (ping-pong) fix:** Allocate two buffers, A and B, each holding one inference window of data (e.g., 256 samples × 6 bytes = 1,536 bytes each). Configure the DMA in circular mode with half-transfer and transfer-complete interrupts:

  (1) DMA fills Buffer A. When full, it triggers an interrupt and automatically switches to Buffer B.
  (2) The CPU processes Buffer A (feature extraction + inference) while DMA fills Buffer B.
  (3) When Buffer B is full, DMA switches back to A, and the CPU processes B.

  The invariant: the CPU never reads a buffer the DMA is writing to. Zero race conditions, zero dropped samples. Total SRAM cost: 2 × 1,536 = 3,072 bytes — negligible.

  The key timing constraint: CPU processing of one buffer must complete before the DMA finishes filling the other buffer. At 1.6 kHz with 256 samples: buffer fill time = 256 / 1600 = 160ms. If inference takes 50ms + 5ms feature extraction = 55ms: 55ms < 160ms — 105ms of slack for other tasks.

  > **Napkin Math:** Single buffer: 256 samples × 6 bytes = 1.5 KB. DMA fill time: 160ms. CPU inference: 50ms. Overlap window (race condition): 50ms / 160ms = 31% of the time the CPU and DMA compete for the same buffer. Actual drop rate depends on alignment — 5% is typical. Double buffer: 2 × 1.5 KB = 3 KB SRAM. Drop rate: 0%. CPU utilization: 55ms / 160ms = 34%. Sleep time: 66% — available for BLE transmission or other tasks.

  > **Key Equation:** $t_{\text{process}} < t_{\text{fill}} = \frac{N_{\text{samples}}}{f_{\text{sample}}}$

  📖 **Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)

  </details>

</details>

---

### 🔢 Integer-Only Inference Arithmetic

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Requantization Pipeline</b> · <code>quantization</code> <code>integer-inference</code></summary>

- **Interviewer:** "Walk me through the complete integer-only arithmetic for passing data between two consecutive Conv2D layers on a Cortex-M4. Layer 1 outputs INT8 with scale $S_1$ and zero-point $Z_1$. Layer 2 expects INT8 input with scale $S_2$ and zero-point $Z_2$. The scales are different. How do you convert between them without any floating-point operations?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Dequantize to float, then requantize." This requires an FPU, which the Cortex-M4 doesn't have for efficient float operations. Even with a software float library, it's 10-50× slower.

  **Realistic Solution:** The requantization between layers is the critical step that makes integer-only inference possible. The real-valued output of Layer 1 is $x_{\text{real}} = S_1 \times (x_{q1} - Z_1)$. Layer 2 needs $x_{q2} = \text{round}(x_{\text{real}} / S_2) + Z_2$. Substituting:

  $x_{q2} = \text{round}\left(\frac{S_1}{S_2} \times (x_{q1} - Z_1)\right) + Z_2$

  The ratio $M = S_1 / S_2$ is a float — but it's computed once at compile time, not at runtime. The trick: represent $M$ as a fixed-point integer: $M \approx M_0 \times 2^{-n}$, where $M_0$ is an INT32 multiplier and $n$ is a right-shift amount. Both are pre-computed by the quantization tool.

  The runtime computation (pure integer):
  1. Subtract zero-point: $\text{tmp} = x_{q1} - Z_1$ (INT8 subtraction → INT16)
  2. Multiply by fixed-point scale: $\text{tmp2} = M_0 \times \text{tmp}$ (INT32 multiply)
  3. Right-shift to apply the $2^{-n}$: $\text{tmp3} = (\text{tmp2} + (1 \ll (n-1))) \gg n$ (rounding shift)
  4. Add output zero-point: $\text{result} = \text{tmp3} + Z_2$
  5. Clamp to [0, 255]: $x_{q2} = \text{clamp}(\text{result}, 0, 255)$

  Total: 1 subtract + 1 multiply + 1 add + 1 shift + 1 add + 1 clamp = **6 integer operations** per element. On Cortex-M4: ~3-4 cycles per element. For a 48×48×64 feature map (147,456 elements): ~500K cycles = **3ms at 168 MHz**. Negligible compared to the convolution itself.

  > **Napkin Math:** Requantization per element: 6 integer ops, ~4 cycles. Feature map: 48×48×64 = 147,456 elements. Total: 590K cycles / 168 MHz = 3.5ms. Convolution for the same layer: ~5M MACs / 336 MMAC/s = 14.9ms. Requantization overhead: 3.5 / (14.9 + 3.5) = **19%**. Not negligible — this is why CMSIS-NN fuses requantization into the convolution kernel, eliminating the separate pass entirely.

  > **Key Equation:** $x_{q2} = \text{clamp}\left(\left\lfloor M_0 \times (x_{q1} - Z_1) + 2^{n-1} \right\rfloor \gg n + Z_2,\ 0,\ 255\right)$

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

---

### 🔋 Clock Frequency vs Power

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Clock Speed Power Trade-off</b> · <code>power</code> <code>compute</code></summary>

- **Interviewer:** "Your Cortex-M4 can run at either 48 MHz or 168 MHz. At 48 MHz it draws 15 mW; at 168 MHz it draws 50 mW. Your inference takes 60ms at 168 MHz. Should you run at 48 MHz to save power, or at 168 MHz to finish faster and sleep longer?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Lower clock speed = lower power = longer battery life." This ignores the energy-per-task calculation. Power is instantaneous; energy is what drains the battery.

  **Realistic Solution:** Compare the total energy per inference cycle, not the power draw:

  **Option A — 168 MHz:** Inference time = 60ms. Active energy = 50 mW × 60ms = 3.0 mJ. Then sleep for 940ms (assuming 1 inference/second): sleep energy = 0.01 mW × 940ms = 0.0094 mJ. **Total energy per cycle: 3.009 mJ.**

  **Option B — 48 MHz:** Clock is 3.5× slower, so inference takes 60ms × (168/48) = 210ms. Active energy = 15 mW × 210ms = 3.15 mJ. Then sleep for 790ms: sleep energy = 0.01 mW × 790ms = 0.0079 mJ. **Total energy per cycle: 3.158 mJ.**

  The faster clock wins — it uses **5% less energy per inference** despite drawing 3.3× more power. This is the "race to sleep" principle: on MCUs with deep sleep modes, the dominant energy cost is the active period. Finishing faster and sleeping longer almost always wins because sleep power (1-10 µW) is 1000-5000× lower than active power.

  The exception: if your inference is not the bottleneck (e.g., you're waiting for a sensor that delivers data every 500ms regardless), running faster doesn't let you sleep longer — you just idle at active power. In that case, the lower clock saves energy.

  > **Napkin Math:** 168 MHz: 3.0 mJ active + 0.009 mJ sleep = 3.009 mJ/cycle. 48 MHz: 3.15 mJ active + 0.008 mJ sleep = 3.158 mJ/cycle. Difference: 5% more energy at 48 MHz. Over 1 year at 1 inf/s: 168 MHz uses 3.009 × 86400 × 365 = 94.9 J = 26.4 mWh/day. 48 MHz uses 99.6 J = 27.7 mWh/day. On a 675 mWh CR2032: 168 MHz lasts 25.6 days, 48 MHz lasts 24.4 days. The fast clock buys 1.2 extra days.

  > **Key Equation:** $E_{\text{cycle}} = P_{\text{active}} \times t_{\text{inference}} + P_{\text{sleep}} \times (T_{\text{period}} - t_{\text{inference}})$

  📖 **Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)

  </details>

</details>

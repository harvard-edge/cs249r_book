# Round 1: TinyML Systems — Inference at the Edge of Physics 🔬

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_micro_architectures.md">🔬 1. Micro Architectures</a> ·
  <a href="02_compute_and_memory.md">⚖️ 2. Compute & Memory</a> ·
  <a href="03_data_and_deployment.md">🚀 3. Data & Deployment</a> ·
  <a href="04_visual_debugging.md">🖼️ 4. Visual Debugging</a> ·
  <a href="05_advanced_systems.md">🔬 5. Advanced Systems</a>
</div>

---

The domain of the TinyML Systems Engineer. This round tests your understanding of what happens when the entire model, runtime, and inference engine must coexist in kilobytes of SRAM, execute in microseconds, and survive on milliwatts of power. There is no operating system, no virtual memory, no second chance — every byte is a design decision.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/tinyml/01_micro_architectures.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 🧠 Memory Layout & SRAM Partitioning

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Flat Memory Reality</b> · <code>memory-layout</code></summary>

- **Interviewer:** "A junior engineer on your team says they'll use `malloc` to dynamically allocate activation buffers during inference on a Cortex-M4. Why do you stop them?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Malloc is just slow on microcontrollers." Speed isn't the primary concern — the failure mode is far worse.

  **Realistic Solution:** On a bare-metal MCU with 256 KB of SRAM, there is no virtual memory and no MMU. Dynamic allocation causes heap fragmentation in a fixed-size memory pool. After a few hundred inference cycles, `malloc` returns NULL — not because you're out of total memory, but because the free space is scattered into unusable fragments. TFLite Micro solves this by requiring a single, pre-allocated flat tensor arena. All activations, scratch buffers, and intermediate tensors are placed at compile-time offsets within this arena. Zero fragmentation, deterministic memory usage, and you know at build time whether the model fits.

  > **Napkin Math:** A Cortex-M4 with 256 KB SRAM must hold: firmware (~40 KB), stack (~8 KB), tensor arena (remaining ~200 KB). If `malloc` fragments even 10% of the arena into gaps smaller than the largest activation buffer, inference fails — even though 20 KB of total free space remains.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Peak RAM Puzzle</b> · <code>memory-layout</code></summary>

- **Interviewer:** "Your 8-layer convolutional model needs 300 KB of peak activation RAM, but your Cortex-M7 only has 256 KB of SRAM available after firmware and stack. You cannot change the model architecture. How do you make it fit?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Quantize the model to use less memory." Quantizing weights from FP32 to INT8 shrinks the weight storage in Flash, but activations are already INT8 in a quantized pipeline — the peak activation footprint barely changes.

  **Realistic Solution:** Operator scheduling for peak RAM reduction. The key insight: not all layers' activations are alive at the same time. By reordering operator execution (computing and immediately consuming activations before moving to the next layer), you can overwrite dead tensors. Tools like TFLite Micro's memory planner and MCUNet's patch-based inference take this further — instead of computing an entire feature map at once, you compute it in spatial patches, reducing peak RAM from the full feature map size to a single patch's worth.

  > **Napkin Math:** A standard 8-layer CNN with 64 channels at 48×48 spatial resolution: one full activation tensor = $64 \times 48 \times 48 \times 1$ byte (INT8) = 147 KB. Two such tensors alive simultaneously (input + output of a layer) = 294 KB — exceeds 256 KB. With patch-based inference (e.g., 12×48 patches), peak activation drops to $64 \times 12 \times 48 \times 1 = 36$ KB per tensor, and two alive = 72 KB. The model now fits with room to spare.

  > **Key Equation:** $\text{Peak RAM} = \max_{t} \sum_{i \in \text{live}(t)} \text{size}(activation_i)$

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

---

### 🔢 Quantization & Fixed-Point Arithmetic

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Quantization Cliff</b> · <code>quantization</code></summary>

- **Interviewer:** "Your keyword spotting model runs at 92% accuracy in FP32. Post-training quantization to INT8 drops it to 91%. Your manager asks you to push to INT4 to halve the model size again. You try it and accuracy collapses to 74%. What happened, and how do you recover?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "INT4 is just too aggressive — we need to stay at INT8." This gives up without understanding the mechanism or the fix.

  **Realistic Solution:** INT4 has only 16 discrete levels per weight. For layers with wide weight distributions (like the first convolutional layer processing raw spectrograms, or the final classification head), 16 levels cannot capture the dynamic range, causing severe clipping and rounding errors that cascade through the network. The fix is mixed-precision quantization: keep sensitive layers (first and last) at INT8 while quantizing the middle depthwise layers to INT4. This recovers most of the accuracy while still shrinking the model significantly.

  > **Napkin Math:** A 250 KB INT8 keyword spotting model: first conv + last FC ≈ 40 KB (keep at INT8). Remaining layers ≈ 210 KB → 105 KB at INT4. Total = 145 KB — a 42% reduction instead of 50%, but accuracy recovers to ~90% vs the 74% of uniform INT4.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Integer Arithmetic Engine</b> · <code>quantization</code> <code>integer-inference</code></summary>

- **Interviewer:** "Walk me through exactly how a quantized Conv2D layer executes on a Cortex-M4 with no floating-point unit. I want to see the math — from quantized inputs to quantized outputs — with no floats anywhere in the pipeline."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "You just multiply INT8 values and truncate." This ignores the accumulator width, the zero-point offsets, and the requantization step — all of which are essential for correctness.

  **Realistic Solution:** The full integer-only inference pipeline works as follows. Each tensor has a quantization scheme: $x_{real} = S_x (x_{int} - Z_x)$ where $S$ is a float scale and $Z$ is an integer zero-point. For convolution, the real-valued output is $Y = W \cdot X + b$. Substituting the quantization parameters and expanding:

  The core MAC accumulates into a 32-bit integer register: $\text{acc}_{32} = \sum (w_{int8} - Z_w)(x_{int8} - Z_x)$. This stays in pure integer arithmetic on the Cortex-M4's 32-bit ALU. The critical trick is requantization: converting the INT32 accumulator back to INT8 for the next layer. The scale ratio $M = \frac{S_w \cdot S_x}{S_y}$ is pre-computed and represented as a fixed-point multiply: $M \approx M_0 \times 2^{-n}$ where $M_0$ is an INT32 value and $n$ is a right-shift. The final output is: $y_{int8} = \text{clamp}\left(\text{round}\left(\frac{M_0 \cdot \text{acc}_{32}}{2^n}\right) + Z_y,\ 0,\ 255\right)$. No float touched the pipeline.

  > **Napkin Math:** A 3×3 depthwise conv on a 48×48×64 INT8 feature map: MACs = $3 \times 3 \times 48 \times 48 \times 64 = 1,327,104$. Each MAC is an INT8×INT8→INT32 multiply-add — one cycle on Cortex-M4 with CMSIS-NN's SMLAD instruction. At 100 MHz: $1.3M / 100M \approx 13$ ms per layer. The requantization adds ~1 shift + 1 add per output element ($48 \times 48 \times 64 = 147,456$ ops) — negligible overhead.

  > **Key Equation:** $y_{int8} = \text{clamp}\!\left(\text{round}\!\left(M_0 \cdot \sum(w_{int8} - Z_w)(x_{int8} - Z_x) \gg n\right) + Z_y,\ 0,\ 255\right)$

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

---

### 🏗️ Model Architecture for Microcontrollers

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Depthwise Separable Advantage</b> · <code>model-architecture</code></summary>

- **Interviewer:** "You need to deploy a small image classifier on an ESP32-S3 with 512 KB SRAM. A standard Conv2D with 32 input channels, 64 output channels, and a 3×3 kernel works in simulation but exceeds your SRAM budget. Your colleague suggests replacing it with a depthwise separable convolution. How much memory and compute does this actually save, and is there a catch?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Depthwise separable is just a smaller convolution — it's always better." People quote the theoretical reduction ratio without checking whether the activation memory (not just the weight memory) actually fits.

  **Realistic Solution:** A standard 3×3 Conv2D with 32→64 channels has $3 \times 3 \times 32 \times 64 = 18,432$ weight parameters. A depthwise separable replacement splits this into a 3×3 depthwise conv ($3 \times 3 \times 32 = 288$ params) plus a 1×1 pointwise conv ($1 \times 1 \times 32 \times 64 = 2,048$ params) = 2,336 total — an 8× reduction in weights and roughly 8–9× fewer MACs. The catch: on MCUs, the compute savings are real, but the activation memory is unchanged. Both approaches produce a 64-channel output feature map of the same spatial size. The depthwise separable version also creates an intermediate 32-channel feature map between the two stages. If SRAM is the bottleneck (and it usually is), the activation footprint matters more than the weight count.

  > **Napkin Math:** For a 48×48 input: standard conv output activation = $64 \times 48 \times 48 = 147$ KB (INT8). Depthwise separable intermediate = $32 \times 48 \times 48 = 73$ KB, final output = 147 KB. Peak (intermediate + output alive) = 220 KB. Standard conv peak (input + output) = $73 + 147 = 220$ KB. Activation memory is identical — the win is purely in weights (18 KB → 2.3 KB) and compute (18M MACs → 2.2M MACs).

  > **Key Equation:** $\text{Depthwise separable MACs} = K^2 \cdot C_{in} \cdot H \cdot W + C_{in} \cdot C_{out} \cdot H \cdot W$

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

---

### ⚡ Power, Energy & Duty Cycling

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Battery Life Equation</b> · <code>power-energy</code></summary>

- **Interviewer:** "You're deploying a gesture recognition system on a Cortex-M4 powered by a 300 mAh coin cell battery. The MCU draws 50 mW active and 10 µW in deep sleep. Inference takes 30 ms and runs once per second. Your product manager asks: how long will the battery last?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "50 mW continuously from a 300 mAh battery — that's about 20 hours." This treats the MCU as always-on, ignoring the duty cycle entirely.

  **Realistic Solution:** Duty cycling is the key. The MCU is active for 30 ms out of every 1000 ms — a 3% duty cycle. Average power = $(0.03 \times 50\text{ mW}) + (0.97 \times 0.01\text{ mW}) = 1.5\text{ mW} + 0.0097\text{ mW} \approx 1.51\text{ mW}$. A 300 mAh coin cell at 3V provides 900 mWh. Battery life = $900\text{ mWh} / 1.51\text{ mW} \approx 596\text{ hours} \approx 25\text{ days}$. The duty cycle extends battery life from less than a day to nearly a month.

  > **Napkin Math:** Always-on: $900\text{ mWh} / 50\text{ mW} = 18\text{ hours}$. With 3% duty cycle: $900\text{ mWh} / 1.51\text{ mW} = 596\text{ hours}$. That's a 33× improvement — the difference between a disposable prototype and a shippable product.

  > **Key Equation:** $P_{avg} = \delta \cdot P_{active} + (1 - \delta) \cdot P_{sleep}$, where $\delta = t_{inference} / t_{period}$

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Energy Harvesting Wall</b> · <code>power-energy</code></summary>

- **Interviewer:** "You're designing a structural health monitoring sensor powered by a small solar cell that provides an average of 0.5 mW indoors. The Cortex-M4 draws 50 mW active. Your model inference takes 30 ms. What is the maximum inference rate you can sustain indefinitely without a battery?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "0.5 mW / 50 mW = 1% duty cycle, so one inference every 100 seconds." This gets the duty cycle right but forgets that you need a storage capacitor to buffer the energy, and ignores the energy cost of waking up.

  **Realistic Solution:** The energy budget per inference = $P_{active} \times t_{inference} = 50\text{ mW} \times 30\text{ ms} = 1.5\text{ mJ}$. Add wake-up overhead (clock stabilization, sensor warm-up) of ~5 ms at 50 mW = 0.25 mJ. Total per inference ≈ 1.75 mJ. The harvester provides 0.5 mW = 0.5 mJ/s. Maximum sustainable rate = $0.5\text{ mJ/s} / 1.75\text{ mJ} \approx 0.29$ inferences/second, or roughly one inference every 3.5 seconds. But this assumes zero sleep power and 100% harvester efficiency. With realistic 60% DC-DC conversion efficiency and 10 µW sleep current: usable power ≈ 0.3 mW, sustainable rate drops to one inference every ~6 seconds. You also need a capacitor large enough to deliver the 50 mW burst: $C = P \cdot t / (0.5 \cdot V^2) \approx 50\text{mW} \times 35\text{ms} / (0.5 \times 3.3^2) \approx 320\text{ µF}$ minimum.

  > **Napkin Math:** Energy in = 0.5 mW × 6 s = 3 mJ harvested. After 60% conversion = 1.8 mJ available. Energy out = 1.75 mJ per inference. Margin = 0.05 mJ — barely sustainable. Drop below 0.5 mW (cloudy day) and the system dies.

  > **Key Equation:** $f_{max} = \frac{\eta \cdot P_{harvest}}{E_{inference} + E_{wake}}$

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

---

### 🛠️ Compiler, Runtime & CMSIS-NN

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The CMSIS-NN Speedup</b> · <code>compiler-runtime</code></summary>

- **Interviewer:** "Your colleague wrote a naive C implementation of an INT8 matrix multiply for a Cortex-M4. It runs in 45 ms. You replace it with the CMSIS-NN equivalent and it drops to 6 ms — a 7.5× speedup. The clock speed didn't change. Where did the speedup come from?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "CMSIS-NN uses NEON SIMD instructions." Cortex-M4 does not have NEON — that's Cortex-A series. People confuse the two ARM families.

  **Realistic Solution:** CMSIS-NN exploits the Cortex-M4's DSP extension — specifically the SIMD (Single Instruction, Multiple Data) instructions like `SMLAD` (Signed Multiply Accumulate Dual). `SMLAD` performs two 16-bit multiplies and accumulates both results into a 32-bit accumulator in a single cycle. For INT8 data, CMSIS-NN packs two INT8 values into a single 16-bit half-word, then uses `SMLAD` to process two MACs per cycle instead of one. Combined with loop unrolling, data re-ordering for cache-friendly access, and elimination of branch overhead, this yields the 7–8× speedup over naive C. It's the microcontroller equivalent of using Tensor Cores on a GPU — you must use the specialized hardware paths or you leave most of the silicon idle.

  > **Napkin Math:** Naive C: 1 MAC per cycle (load, multiply, accumulate = ~3 cycles with pipeline, but ~1 effective with optimization). CMSIS-NN with `SMLAD`: 2 MACs per cycle. With loop unrolling (4 iterations): pipeline stalls eliminated, achieving ~1.8 MACs/cycle effective. For a 256×256 matrix multiply: $256 \times 256 = 65,536$ MACs. Naive: ~200K cycles. CMSIS-NN: ~36K cycles. At 100 MHz: 2 ms vs 0.36 ms per matmul.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

---

### 🎤 Sensor Pipelines & Real-Time Inference

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Keyword Spotting Pipeline</b> · <code>sensor-pipeline</code> <code>memory-layout</code></summary>

- **Interviewer:** "Design the complete inference pipeline for an always-on keyword spotting system on a Cortex-M4 (100 MHz, 256 KB SRAM, no FPU). The microphone samples at 16 kHz. You need to detect the wake word 'Hey Device' within 500 ms of it being spoken. Walk me through every stage — from raw audio samples to classification output — and show me the memory budget."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Collect 1 second of audio, compute an FFT, run the model." This approach buffers too much audio (32 KB just for raw samples), uses floating-point FFT (no FPU), and doesn't account for the streaming nature of the problem.

  **Realistic Solution:** The pipeline must be streaming and integer-only:

  **Stage 1 — Audio capture with double buffering:** The microphone feeds 16-bit PCM at 16 kHz via I2S/DMA. Use two 30 ms frame buffers (30 ms × 16 kHz × 2 bytes = 960 bytes each, ~2 KB total). While DMA fills one buffer, the CPU processes the other.

  **Stage 2 — Integer Mel spectrogram:** For each 30 ms frame (480 samples), apply a Hanning window (pre-computed INT16 lookup table, 960 bytes). Compute a 512-point fixed-point FFT using CMSIS-DSP's `arm_rfft_q15` — no floats. Apply 40 Mel-scale triangular filter banks (pre-computed INT16 coefficients, ~2 KB). Take log-magnitude using a fixed-point log approximation (lookup table, 256 bytes). Output: 40 INT8 Mel coefficients per frame.

  **Stage 3 — Feature stacking:** Stack the last 49 frames (covering ~1.5 seconds with 20 ms hop) into a 49×40 INT8 spectrogram image = 1,960 bytes. Use a circular buffer — each new frame overwrites the oldest, so no memory copy needed.

  **Stage 4 — Model inference:** Feed the 49×40 spectrogram into a DS-CNN (depthwise separable CNN) quantized to INT8. Model weights in Flash (~80 KB). Tensor arena for activations ~60 KB. Inference time ~20 ms using CMSIS-NN.

  **Memory budget:** Audio double buffer: 2 KB. Window + Mel tables: 3.2 KB. Feature buffer: 2 KB. Tensor arena: 60 KB. Firmware + stack: 40 KB. Model weights (Flash, not SRAM): 0 KB. **Total SRAM: ~107 KB** out of 256 KB available — leaving 149 KB for application logic.

  **Latency budget:** Feature extraction: ~2 ms per frame. Model inference: ~20 ms. Total from last audio frame to classification: ~22 ms. Since the wake word spans ~500 ms of audio and we process frames every 30 ms with 20 ms hop, detection latency after the word ends ≈ 30 ms (one frame delay) + 22 ms (processing) ≈ 52 ms — well within the 500 ms requirement.

  > **Napkin Math:** Raw audio at 16 kHz × 16-bit × 1 second = 32 KB. But streaming with 30 ms frames only needs 960 bytes live — a 34× memory reduction. The 512-point fixed-point FFT on Cortex-M4 takes ~0.3 ms via CMSIS-DSP. 40 Mel filters on 256 frequency bins = 10,240 multiply-accumulates = 0.1 ms. Total feature extraction per frame: ~0.5 ms. The CPU is idle 98% of the time between frames.

  > **Key Equation:** $\text{SRAM}_{total} = \text{buf}_{audio} + \text{buf}_{features} + \text{LUT}_{mel} + \text{arena}_{model} + \text{stack} + \text{firmware}$

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>


### 🧠 Hardware Bus Constraints

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Unaligned Access Fault</b> · <code>memory-alignment</code></summary>

- **Interviewer:** "You port a TFLite Micro vision model from a Cortex-M4 to a cheaper Cortex-M0+. The code compiles perfectly. But when the inference engine loads the model's tensor arena, it immediately crashes with a `HardFault` exception. How does TFLite Micro's packed tensor memory layout cause unaligned access on the Cortex-M0+, and why does the ISA constraint force you to pad your ML tensors?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming all ARM Cortex-M processors handle memory access rules identically, or thinking the model weights are corrupted."

  **Realistic Solution:** You triggered an unaligned memory access fault due to how the ML runtime packs tensors. TFLite Micro allocates all activations and weights contiguously in a single `tensor_arena` byte array to save SRAM. If a 3-channel INT8 image tensor (size $W \times H \times 3$ bytes) is followed immediately by a 32-bit quantization multiplier or a 32-bit bias vector, that 32-bit value may start at an address not divisible by 4.

  The Cortex-M4 hardware supports unaligned memory access—it will automatically fetch a 32-bit word across two bus cycles if the address is unaligned. The simpler, cheaper Cortex-M0+ physically lacks this hardware capability to save silicon area. When the ML kernel attempts to read the 32-bit bias from the unaligned address, the Cortex-M0+ bus matrix cannot physically perform the read and instantly throws a hardware fault. You must configure the ML runtime to align tensors to 4-byte boundaries (padding the arena), trading SRAM efficiency for ISA compatibility.

  > **Napkin Math:** If an INT8 tensor of size 11 bytes starts at `0x20000000`, it ends at `0x2000000A`. The next tensor (a 32-bit float bias) starts at `0x2000000B`. Attempting a 32-bit read at `0x2000000B` fails because `0x2000000B % 4 != 0`. The M4 silently resolves it in 2 cycles. The M0+ hard crashes in 1 cycle. Padding the 11-byte tensor to 12 bytes wastes 1 byte but aligns the next tensor to `0x2000000C`, preventing the crash.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>


### 🔬 Hard Real-Time Constraints

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-gold?style=flat-square" alt="Level 3" align="center"> The Interrupt Jitter Crisis</b> · <code>rtos-scheduling</code></summary>

- **Interviewer:** "You have a vibration-analysis model running on an industrial motor's MCU. It must sample an accelerometer via an ADC every 100 microseconds, buffer it, and run an FFT + NN inference. Occasionally, the system misses an ADC sample entirely, causing the FFT frequencies to skew and the model to output false positives. The CPU utilization is only at 60%. Why did you drop a sample?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Looking at average CPU utilization and ignoring the absolute worst-case execution time (WCET) of critical interrupt service routines."

  **Realistic Solution:** You suffered from interrupt blocking (jitter). In bare-metal or RTOS systems, hardware peripherals (like the ADC) trigger an Interrupt Service Routine (ISR) to tell the CPU to read the data. If the neural network inference code (or a communication task, like sending data over WiFi) disables interrupts temporarily to update a shared memory buffer safely, the CPU goes deaf to the outside world. If the ADC fires its 100µs interrupt while the CPU is locked in a 150µs critical section, the ADC's data register is overwritten by the next reading before the CPU ever sees it.

  > **Napkin Math:** 100µs sampling means a frequency of 10 kHz. If your NN code has a critical section (e.g., `__disable_irq()`) that takes `3000 clock cycles` on a 20 MHz processor, that section takes `150 microseconds`. The 100µs deadline is mathematically impossible to meet during that window, guaranteeing dropped data regardless of overall CPU idle time.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>


### 🧠 Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6_Staff-red?style=flat-square" alt="Level 4" align="center"> The SIMD Lane Starvation</b> · <code>instruction-set</code></summary>

- **Interviewer:** "You are optimizing an audio neural network for an ARM Cortex-M7 (which has DSP extensions). You replace a standard `for` loop with explicit CMSIS-NN SIMD instructions, expecting a 4x speedup since it can process four 8-bit integers per clock cycle. The profiler shows you only got a 1.2x speedup. What pipeline bottleneck did you fail to feed?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming that using a SIMD vector instruction automatically guarantees the math will execute 4x faster, ignoring how the data gets into the vector registers."

  **Realistic Solution:** You starved the SIMD lanes with scalar memory loads. While the CPU can indeed *multiply and accumulate* four 8-bit integers in a single clock cycle, it must first get those integers from SRAM into the CPU registers. If your C-code explicitly loads the vector registers byte-by-byte (`LDRB` instructions), you are spending 4 clock cycles just loading the data before you can perform the 1-cycle SIMD math. The pipeline is entirely bound by the load/store unit.

  > **Napkin Math:** To process 4 items:
  > Scalar approach: 4 loads (4 cycles) + 4 multiplies (4 cycles) = 8 cycles.
  > Naive SIMD approach: 4 scalar loads (4 cycles) + 1 SIMD multiply (1 cycle) = 5 cycles. (Speedup: `8/5 = 1.6x`).
  > Proper SIMD approach: 1 Word-aligned 32-bit load (1 cycle) + 1 SIMD multiply (1 cycle) = 2 cycles. (Speedup: `8/2 = 4x`). By not casting your 8-bit pointers to 32-bit pointers for the memory fetch, you left most of the performance on the table.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

---

### 🆕 Extended Micro-Architectures

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The ISA Tax on Inference</b> · <code>instruction-set</code></summary>

- **Interviewer:** "You benchmark the same INT8 keyword spotting model on two MCUs at the same 80 MHz clock speed: a Cortex-M4 (ARMv7E-M, Thumb-2 ISA) and a RISC-V RV32IMC (ESP32-C3). The M4 completes inference in 18 ms. The RISC-V takes 31 ms. Same clock, same model, same compiler optimization level. Where did the 72% performance gap come from?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "RISC-V is just slower because it's a simpler architecture." RISC-V is not inherently slower — the issue is specific ISA extensions and their impact on ML workloads.

  **Realistic Solution:** The gap comes from three ISA-level differences that compound for ML inference:

  (1) **DSP extensions** — the Cortex-M4's ARMv7E-M ISA includes DSP instructions like `SMLAD` (Signed Multiply Accumulate Dual), which performs two 16×16→32 multiply-accumulates in a single cycle. CMSIS-NN packs two INT8 values into 16-bit half-words and uses `SMLAD` to process 2 MACs/cycle. The RV32IMC lacks a standard DSP extension — each MAC requires a separate `MUL` + `ADD` sequence, achieving ~1 MAC per 3 cycles (multiply latency + accumulate).

  (2) **Saturating arithmetic** — the M4 has `SSAT`/`USAT` instructions for clamping values to INT8 range after accumulation. The RISC-V must emulate this with a branch-compare-move sequence (3-4 instructions). For a quantized model with requantization after every layer, this adds up.

  (3) **Memory access width** — the M4's `LDM` (Load Multiple) instruction can load 4 registers (16 bytes) in a burst from tightly-coupled SRAM. The RV32IMC's compressed instruction set (`C` extension) saves code size but limits load instructions to single-register operations, increasing memory access overhead.

  The RISC-V "P" (packed SIMD) extension addresses this gap, but it's not yet ratified or widely implemented in production silicon. Until then, the ISA tax on ML workloads is real and measurable.

  > **Napkin Math:** Keyword spotting model: ~1.5M MACs. Cortex-M4 with CMSIS-NN: 2 MACs/cycle effective → 750K cycles → 750K / 80M Hz = 9.4 ms compute + ~8.6 ms memory overhead = 18 ms. RISC-V RV32IMC: ~0.33 MACs/cycle → 4.5M cycles → 56 ms compute, but pipelining helps → ~25 ms compute + ~6 ms memory = 31 ms. Ratio: 31/18 = 1.72×. The DSP extension alone accounts for ~6× difference in MAC throughput, partially offset by the RISC-V's simpler pipeline.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Memory-Mapped Sensor Bottleneck</b> · <code>memory-mapped-io</code></summary>

- **Interviewer:** "Your TinyML vibration monitoring system uses a Cortex-M0+ (48 MHz, 32 KB SRAM) with an external accelerometer connected via SPI. The accelerometer produces 3-axis INT16 samples at 3.2 kHz. Your colleague reads each sample by polling the SPI data register in a busy-wait loop inside the main inference function. The system works, but inference latency is 3× worse than expected. Why?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The SPI bus is too slow." SPI at 8 MHz can transfer 6 bytes (3 axes × 2 bytes) in 6 µs — the bus bandwidth is not the bottleneck.

  **Realistic Solution:** The problem is that **memory-mapped peripheral access stalls the CPU pipeline**:

  (1) **Bus contention** — on the Cortex-M0+, the CPU, DMA controller, and peripherals share a single AHB-Lite bus. When the CPU reads the SPI data register (a memory-mapped address in the peripheral region, typically 0x40000000+), the bus arbiter must route the request through the APB bridge. This takes 2-3 wait states per access because the APB bus runs at half the CPU clock (24 MHz). Each 16-bit sample read costs ~6 CPU cycles instead of 1.

  (2) **Polling overhead** — busy-waiting on the SPI status register (checking "is data ready?") burns CPU cycles. At 3.2 kHz sample rate, the CPU polls the status register ~15 times per sample (48 MHz / 3.2 kHz = 15,000 cycles between samples, but each poll takes ~10 cycles including the branch). That's 150 cycles wasted per sample just on polling, plus the 6-cycle read. Over a 512-sample buffer: 512 × 156 = ~80,000 cycles = 1.67 ms of pure I/O overhead.

  (3) **Fix: DMA** — configure the DMA controller to transfer SPI data directly to an SRAM buffer. The DMA runs in the background on the same AHB bus but doesn't stall the CPU. The CPU is free to run inference while the DMA fills the next buffer. Use double-buffering: DMA fills buffer A while the CPU processes buffer B. Interrupt on DMA complete to swap buffers.

  (4) **Result** — CPU I/O overhead drops from 1.67 ms to ~0 ms (one DMA setup + one interrupt = ~50 cycles total). The 3× latency regression disappears.

  > **Napkin Math:** Without DMA: 512 samples × 156 cycles/sample = 79,872 cycles = 1.66 ms at 48 MHz. Inference compute: 0.83 ms. Total: 2.49 ms. Expected inference-only: 0.83 ms. Ratio: 3×. With DMA: DMA setup = 20 cycles. Interrupt handler = 30 cycles. Total I/O overhead: 50 cycles = 1 µs. New total: 0.83 ms + 0.001 ms = 0.831 ms. Speedup: 3×.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The DMA Channel Collision</b> · <code>dma</code></summary>

- **Interviewer:** "Your predictive maintenance system on an STM32F4 uses DMA to stream accelerometer data via SPI. You recently added an ML model whose weights are too large for SRAM, so you store them in external SPI Flash and use a second DMA channel to stream weights into SRAM layer-by-layer during inference. When you run inference, the accelerometer data suddenly has gaps. How does the ML inference's DMA bandwidth requirement conflict with the sensor DMA, and how does the model's layer size determine whether double-buffering can resolve the contention?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The DMA controller can't handle two streams at once." The STM32F4 has two DMA controllers with 8 streams each — it absolutely can handle concurrent transfers.

  **Realistic Solution:** The corruption comes from **DMA stream priority and bus arbitration** driven by the massive asymmetry in payload sizes. The accelerometer DMA requests 2 bytes every 300 µs. The ML inference DMA requests a massive burst of 16 KB (an entire convolutional layer's weights) as fast as the SPI bus allows. If both streams are on the same DMA controller, the ML weight transfer monopolizes the AHB bus matrix. The accelerometer's DMA request is held pending. If the hold exceeds the SPI peripheral's tiny hardware FIFO depth (4 bytes on STM32F4), the FIFO overflows and the sensor sample is permanently lost.

  To fix this, you must configure the DMA stream priorities based on the latency constraints of the workload, not the bandwidth. The accelerometer is low-bandwidth but highly latency-sensitive (hard real-time deadline before FIFO overflow). The ML weight streaming is high-bandwidth but latency-tolerant (the inference just pauses). Set the accelerometer DMA to `Very High` priority and the ML weight DMA to `Low`. When the sensor fires, it preempts the weight transfer for just 2 bus cycles, then the ML transfer resumes. Furthermore, if the model's layer size exceeds the available double-buffer size in SRAM, the CPU will stall waiting for the DMA, so the layer size must be tiled to fit the buffer to keep the CPU fed.

  > **Napkin Math:** SPI accelerometer: 3.2 kHz = 312.5 µs period. FIFO depth: 4 bytes = 2 samples. If DMA is delayed by >625 µs, FIFO overflows. ML weight transfer: 16 KB layer at 20 MHz SPI = 16,384 bytes × 8 bits / 20,000,000 = 6.5 ms. If ML DMA has equal or higher priority, it holds the bus for 6.5 ms. 6.5 ms > 625 µs → guaranteed FIFO overflow and dropped sensor data. Priority inversion fixes this instantly.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Clock Tree Surprise</b> · <code>clock-tree</code></summary>

- **Interviewer:** "You're deploying a gesture recognition model on an STM32L4 (Cortex-M4, up to 80 MHz, 256 KB SRAM). To save power, your firmware engineer configures the MCU to run at 16 MHz from the MSI oscillator instead of 80 MHz from the PLL. Inference time scales from 20 ms to 100 ms — exactly 5×, as expected. But the total energy per inference *increases* by 40% at the lower clock. How is running slower using *more* energy?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Lower clock = lower power = lower energy." This confuses *power* (watts) with *energy* (joules). Power drops at lower clock speeds, but the MCU runs longer, and the energy equation has a non-obvious crossover.

  **Realistic Solution:** The energy paradox comes from **static power domination at low frequencies**:

  (1) **Power model** — MCU power has two components: dynamic power ($P_{dyn} = C \cdot V^2 \cdot f$, proportional to clock frequency) and static power ($P_{static}$, constant leakage current regardless of clock speed). On the STM32L4: at 80 MHz, $P_{dyn}$ ≈ 25 mW, $P_{static}$ ≈ 5 mW, total = 30 mW. At 16 MHz (same voltage — the MSI doesn't trigger a voltage scaling change), $P_{dyn}$ ≈ 5 mW, $P_{static}$ ≈ 5 mW, total = 10 mW.

  (2) **Energy calculation** — Energy = Power × Time. At 80 MHz: 30 mW × 20 ms = 0.6 mJ. At 16 MHz: 10 mW × 100 ms = 1.0 mJ. The 40% energy increase comes from static power accumulating over the 5× longer execution time. The static component contributes 5 mW × 100 ms = 0.5 mJ at 16 MHz vs 5 mW × 20 ms = 0.1 mJ at 80 MHz.

  (3) **The optimal point** — the minimum-energy clock frequency is where $\frac{dE}{df} = 0$. For workloads dominated by compute (like ML inference), the optimal strategy is **race-to-sleep**: run at maximum clock speed to finish as fast as possible, then enter deep sleep (0.01 mW). Energy at 80 MHz + sleep: 0.6 mJ + 0.01 mW × 80 ms = 0.6008 mJ. Energy at 16 MHz + no sleep: 1.0 mJ. Race-to-sleep wins by 40%.

  (4) **Voltage scaling** — the real power savings come from reducing voltage, not just frequency. The STM32L4 supports voltage scaling: at 16 MHz, you can drop from 1.2V to 1.0V, reducing dynamic power by $(1.0/1.2)^2 = 0.69×$. But even with voltage scaling, race-to-sleep at 80 MHz typically wins for bursty ML workloads.

  > **Napkin Math:** Race-to-sleep at 80 MHz: 0.6 mJ compute + 0.8 µJ sleep (80 ms × 0.01 mW) = 0.6008 mJ. Slow at 16 MHz, same voltage: 1.0 mJ. Slow at 16 MHz, voltage-scaled to 1.0V: $P_{dyn}$ = 5 mW × 0.69 = 3.45 mW. Total = 8.45 mW × 100 ms = 0.845 mJ. Still 41% worse than race-to-sleep. The crossover only favors low-frequency operation when the duty cycle is extremely low (<0.1%) and wake-up energy is high.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Instruction Cache Thrashing Loop</b> · <code>instruction-cache</code></summary>

- **Interviewer:** "Your anomaly detection model runs on a Cortex-M7 (216 MHz, 16 KB I-cache, 16 KB D-cache, 512 KB SRAM). Profiling shows that a specific depthwise convolution layer takes 4.2 ms — but your cycle-count estimate based on MAC operations predicts 1.1 ms. The data cache hit rate is 98%. Where are the missing 3.1 ms?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The data cache hit rate is 98%, so the memory system is fine." People check the D-cache and forget the I-cache entirely. On Cortex-M7, the I-cache and D-cache are separate (Harvard architecture).

  **Realistic Solution:** The bottleneck is **instruction cache thrashing**, not data access:

  (1) **I-cache analysis** — the Cortex-M7 has a 16 KB I-cache with 32-byte lines = 512 cache lines. Your depthwise convolution kernel, after loop unrolling and CMSIS-NN inlining, compiles to ~20 KB of machine code. The kernel exceeds the I-cache by 4 KB. As the inner loop executes, it evicts its own earlier instructions. Every iteration of the outer loop re-fetches instructions from Flash.

  (2) **Flash wait states** — the Cortex-M7 at 216 MHz with Flash running at 30 MHz requires 7 wait states per Flash access. An I-cache miss costs 7 cycles per 32-byte line fetch. If the inner loop is 20 KB and the I-cache is 16 KB, 4 KB of instructions are re-fetched every outer loop iteration. At 32 bytes/line: 128 cache line refills × 7 cycles = 896 cycles per outer loop iteration.

  (3) **Quantifying the impact** — the depthwise conv has 64 output channels (outer loop = 64 iterations). I-cache miss penalty: 64 × 896 = 57,344 cycles = 0.27 ms. But the actual miss pattern is worse: the cache replacement policy (pseudo-LRU) causes *conflict misses* — two hot code sections map to the same cache set, evicting each other repeatedly. Effective miss rate: ~15% of instruction fetches. At 216 MHz with ~2M instruction fetches for this layer: 300K misses × 7 cycles = 2.1M cycles = 9.7 ms. Wait — that's too high. The actual penalty is mitigated by the M7's prefetch buffer (4 lines). Effective penalty: ~3.1 ms, matching the observation.

  (4) **Fix** — place the hot convolution kernel in ITCM (Instruction Tightly Coupled Memory). The M7 has 16 KB of ITCM with zero wait states. Copy the 20 KB kernel to ITCM at boot — but it's 20 KB and ITCM is 16 KB. Solution: split the kernel. Place the inner loop (~12 KB) in ITCM. Leave the outer loop setup code in Flash (executed once per layer, cache-friendly). Alternatively, reduce the kernel size by disabling aggressive loop unrolling (`-fno-unroll-loops` for this file) to fit within 16 KB.

  > **Napkin Math:** MAC-only estimate: 1.1 ms (pure compute at 216 MHz). Observed: 4.2 ms. Gap: 3.1 ms. I-cache size: 16 KB. Kernel code size: 20 KB. Overflow: 4 KB. Flash latency: 7 wait states. Inner loop iterations: ~2000. Conflict misses per iteration: ~2 lines. Total miss penalty: 2000 × 2 × 7 = 28,000 cycles = 0.13 ms. But with set-associativity conflicts (4-way): effective misses 10×→ 1.3 ms. Plus prefetch misses on branches: ~1.8 ms. Total: ~3.1 ms. After ITCM placement: 0 wait states → gap eliminated → 1.1 ms.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Branch Prediction Penalty on MCU</b> · <code>branch-prediction</code></summary>

- **Interviewer:** "Your colleague implements a ReLU activation function for an INT8 inference engine on a Cortex-M4 (100 MHz, no branch predictor) using an `if (x < 0) x = 0;` pattern. Profiling shows ReLU takes 15% of total inference time — far more than expected for a simple clamp operation. The model has 500,000 activations per inference. What's going on?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The Cortex-M4 has branch prediction, so branches are cheap." The Cortex-M4 has *no* dynamic branch predictor. It has a simple 1-stage pipeline prefetch that always predicts "not taken." Every taken branch flushes the pipeline.

  **Realistic Solution:** The performance cost comes from **pipeline flushes on taken branches** in a tight loop:

  (1) **Pipeline analysis** — the Cortex-M4 has a 3-stage pipeline (Fetch, Decode, Execute). An `if (x < 0) x = 0;` compiles to: `CMP r0, #0` / `IT LT` / `MOVLT r0, #0`. The IT (If-Then) block on Thumb-2 is predicated execution — it doesn't branch, so no pipeline flush. But many compilers (especially at `-O0` or `-O1`) generate a branch instead: `CMP r0, #0` / `BGE skip` / `MOV r0, #0` / `skip:`. The `BGE` is taken ~50% of the time (assuming roughly symmetric activation distributions). Each taken branch costs 2 pipeline refill cycles.

  (2) **Cost calculation** — 500,000 activations × 50% taken branches × 2 cycles penalty = 500,000 extra cycles. At 100 MHz: 5 ms. If total inference is 33 ms, that's 15% — matching the observation.

  (3) **Fix: branchless ReLU** — replace the conditional with a branchless operation: `x = x & ~(x >> 31)` for signed INT8 (arithmetic right shift fills with sign bit, then AND masks negative values to zero). This compiles to 2 instructions with zero branches: `ASR r1, r0, #31` / `BIC r0, r0, r1`. Constant 2 cycles per activation regardless of value.

  (4) **Even better: SIMD ReLU** — use the Cortex-M4's `USAT` instruction: `USAT r0, #8, r0` clamps to [0, 255] in a single cycle. Or process 4 activations at once using `UQADD8` (unsigned saturating add of 4 packed bytes) with a zero register.

  > **Napkin Math:** Branching ReLU: 500K × (2 cycles base + 1 cycle penalty × 50%) = 500K × 2.5 = 1.25M cycles = 12.5 ms. Branchless ReLU: 500K × 2 cycles = 1M cycles = 10 ms. SIMD ReLU (4 at a time): 125K × 1 cycle = 125K cycles = 1.25 ms. Speedup from branching to SIMD: 10×. Inference time reduction: from 33 ms to 21.5 ms (saving 11.5 ms). The "simple" activation function was the biggest optimization opportunity in the entire model.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Bus Arbitration Starvation</b> · <code>bus-arbitration</code></summary>

- **Interviewer:** "Your industrial IoT sensor runs a vibration anomaly detection model on an STM32H7 (Cortex-M7, 480 MHz, 1 MB SRAM, AXI bus matrix). The system simultaneously: (1) streams accelerometer data via SPI+DMA at 25.6 kHz, (2) runs inference on the buffered data, and (3) transmits results over Ethernet+DMA. In isolation, inference takes 2.8 ms. With all three running, inference balloons to 7.1 ms. CPU utilization is only 45%. Explain the 2.5× slowdown."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The CPU is only at 45% utilization, so there's plenty of headroom." CPU utilization measures instruction execution, not *bus bandwidth*. The CPU can be idle waiting for data while the bus is saturated.

  **Realistic Solution:** The slowdown is caused by **AXI bus matrix contention** between three bus masters:

  (1) **Bus architecture** — the STM32H7's AXI bus matrix connects multiple masters (CPU D-bus, CPU I-bus, DMA1, DMA2, Ethernet DMA) to multiple slaves (SRAM1, SRAM2, SRAM3, Flash, peripherals). The matrix supports concurrent access *only* when masters target different slaves. When two masters target the same SRAM bank, the arbiter serializes access using a round-robin policy.

  (2) **Memory bank conflicts** — your firmware places the accelerometer DMA buffer, the inference tensor arena, and the Ethernet TX buffer all in SRAM1 (the default linker script puts everything in the largest SRAM bank). Now: the SPI DMA writes to SRAM1 (accelerometer data), the CPU reads from SRAM1 (tensor arena), and the Ethernet DMA reads from SRAM1 (TX buffer). Three masters contending for one slave. The round-robin arbiter gives each master 1/3 of the bandwidth.

  (3) **Bandwidth calculation** — SRAM1 bandwidth at 480 MHz with 64-bit AXI bus: 480M × 8 bytes = 3.84 GB/s. CPU inference needs: ~1.5 GB/s (estimated from 2.8 ms inference on compute-bound workload). SPI DMA: 25.6 kHz × 6 bytes = 153.6 KB/s (negligible). Ethernet DMA: ~10 MB/s (100 Mbps link). Total demand: ~1.51 GB/s. Fits easily — so why the slowdown? Because the arbiter doesn't allocate bandwidth proportionally. Each DMA burst (16 beats × 8 bytes = 128 bytes) locks the bus for 16 cycles. During that lock, the CPU stalls. With Ethernet DMA bursting every ~13 µs and SPI DMA bursting every ~39 µs, the CPU sees frequent 16-cycle stalls that fragment its memory access pattern and destroy cache line prefetch efficiency.

  (4) **Fix** — scatter buffers across SRAM banks: tensor arena in SRAM1 (512 KB), accelerometer DMA buffer in SRAM2 (128 KB), Ethernet TX buffer in SRAM3 (64 KB). Now each master targets a different slave — the AXI matrix serves all three concurrently with zero contention. Inference returns to 2.8 ms.

  > **Napkin Math:** SRAM1-only: CPU gets ~1/3 of bus cycles during DMA bursts. Effective CPU bandwidth: 3.84 GB/s × 0.4 (accounting for burst granularity) = 1.54 GB/s. Inference needs 1.5 GB/s → barely fits, but cache miss penalty increases from 1 cycle (no contention) to 3 cycles (arbitration). Inference time: 2.8 ms × 2.5 = 7.0 ms. After bank splitting: CPU has dedicated SRAM1 port. Bandwidth: 3.84 GB/s (full). Inference: 2.8 ms. DMA overhead on CPU: 0 cycles (different bus paths).

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The MCU Model Extraction Attack</b> · <code>security</code></summary>

- **Interviewer:** "Your company deploys a proprietary defect detection model on an STM32F4 MCU inside an industrial inspection camera. A competitor buys your product, connects a JTAG debugger to the exposed debug header, and dumps the entire Flash memory — including your model weights — in under 60 seconds. How do you protect the model on a $3 MCU?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Encrypt the model in Flash and decrypt at runtime." On an MCU with 256 KB SRAM and a 500 KB model in Flash, you can't decrypt the entire model into SRAM — it doesn't fit. Decrypting layer-by-layer adds latency and the decryption key must be stored *somewhere* on the same chip.

  **Realistic Solution:** Defense-in-depth using the MCU's hardware security features:

  (1) **Read-out protection (RDP)** — the STM32F4 has three RDP levels. RDP Level 1: JTAG/SWD can connect but cannot read Flash. RDP Level 2: JTAG/SWD is permanently disabled — the debug port is fused off. Level 2 is irreversible (hardware fuse). This blocks the trivial JTAG dump attack. Cost: $0 (just set an option byte).

  (2) **Proprietary code readout protection (PCROP)** — STM32F4 supports PCROP on specific Flash sectors. Mark the sectors containing model weights as PCROP-protected. Even if an attacker downgrades from RDP Level 2 (impossible, but hypothetically), PCROP sectors return zeros on read. The CPU can *execute* from these sectors but cannot *read* them as data — but model weights are data, not code. Solution: store weights in PCROP sectors and use a small trusted loader that copies weights to SRAM sector-by-sector during inference, erasing each SRAM sector after use.

  (3) **Physical attack mitigation** — a determined attacker can decap the chip and use a focused ion beam (FIB) to read Flash cells directly. Defense: use the STM32's hardware AES-256 engine to encrypt model weights in Flash with a key derived from the device's unique ID (96-bit UID). Each chip has a different key. Decapping one chip doesn't help with another. The AES engine decrypts at hardware speed (~1 cycle/byte at 168 MHz) with negligible latency impact.

  (4) **Accept the economics** — a FIB attack costs $50,000-$100,000 per chip. If your model's competitive advantage is worth less than this, RDP Level 2 + AES encryption is sufficient. If it's worth more, consider a secure element (e.g., ATECC608B, $0.50) to store the decryption key in tamper-resistant silicon.

  > **Napkin Math:** JTAG dump without protection: 60 seconds, $0 cost (just a $20 ST-Link). RDP Level 2: blocks JTAG entirely. Decap + FIB: $50K-$100K, 2-4 weeks. AES decryption overhead: 500 KB model / 168 MB/s = 3 ms (one-time at boot). SRAM budget for layer-by-layer decryption: largest layer = 40 KB. Fits in 256 KB SRAM with room for inference. Secure element cost: $0.50 per unit × 100K units = $50K — same as one FIB attack.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Sleep Mode Wake-Up Cost</b> · <code>low-power</code></summary>

- **Interviewer:** "Your always-on environmental sound classifier runs on an nRF52840 (Cortex-M4F, 64 MHz, 256 KB SRAM). Between inferences (every 2 seconds), you put the MCU in System OFF mode (0.3 µA). But your measured average current is 800 µA — 10× higher than your duty-cycle calculation predicts. The inference itself only takes 25 ms at 3 mA. What's eating the power budget?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The MCU isn't actually entering deep sleep." It is — you've verified with an oscilloscope that current drops to 0.3 µA between inferences.

  **Realistic Solution:** The hidden cost is the **wake-up sequence**, not the sleep or inference:

  (1) **Wake-up from System OFF** — on the nRF52840, System OFF is the deepest sleep mode. Waking from it triggers a full system reset: the CPU reboots, the 64 MHz HFCLK crystal oscillator must stabilize (360 µs typical), the 32 KB RAM contents are lost (must be reloaded from Flash), and the SRAM power domains re-energize. The total wake-up sequence takes ~1.5 ms at full active current (3 mA).

  (2) **Peripheral re-initialization** — after System OFF wake, all peripherals are in their reset state. Your firmware must re-initialize: I2S microphone interface (PDM clock stabilization: 1 ms), DMA channels, GPIO pins, and the ML model (loading the tensor arena from Flash: 60 KB at ~64 MB/s = ~1 ms). Total re-init: ~3.5 ms at 3 mA.

  (3) **Revised duty cycle** — active time is not just 25 ms inference. It's: 1.5 ms wake-up + 3.5 ms re-init + 25 ms inference = 30 ms total active. Duty cycle: 30 ms / 2000 ms = 1.5%. Average current: 1.5% × 3 mA + 98.5% × 0.3 µA = 45 µA + 0.3 µA = 45.3 µA. But measured is 800 µA — still doesn't match.

  (4) **The real culprit: microphone always-on** — the PDM microphone draws 600 µA continuously because it must be powered to detect the wake-up sound event that triggers inference. The MCU sleeps, but the microphone doesn't. Total: 45 µA (MCU average) + 600 µA (mic) + 150 µA (voltage regulator quiescent) = 795 µA ≈ 800 µA.

  (5) **Fix** — use a hardware sound detector (e.g., VM3011, 10 µA) as a wake-up trigger. The VM3011 monitors ambient sound level and asserts a GPIO interrupt when it exceeds a threshold. The PDM microphone only powers on after the wake-up trigger. New budget: 45 µA (MCU) + 10 µA (VM3011) + 150 µA (regulator) = 205 µA.

  > **Napkin Math:** Battery: 230 mAh CR2032 at 3V. At 800 µA: 230 mAh / 0.8 mA = 287 hours = 12 days. At 205 µA (with hardware wake detector): 230 / 0.205 = 1,122 hours = 47 days. At 45 µA (theoretical MCU-only): 230 / 0.045 = 5,111 hours = 213 days. The microphone is 75% of the total power budget. The MCU's deep sleep is irrelevant if the sensor stays on.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Timer-Driven Inference Scheduler</b> · <code>peripheral-timer</code></summary>

- **Interviewer:** "Your air quality monitoring system runs inference every 30 seconds on a Cortex-M0+ (48 MHz, 32 KB SRAM). A junior engineer implements the schedule using `HAL_Delay(30000)` — a busy-wait delay. The system works but draws 12 mA continuously. Redesign the scheduling to draw under 50 µA average."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use `sleep()` instead of `HAL_Delay()`." The standard `sleep()` on bare-metal MCUs is often just another busy-wait, or at best enters a light sleep that still draws milliamps.

  **Realistic Solution:** Use the **RTC (Real-Time Clock) or LPTIM (Low-Power Timer)** to wake the MCU from deep sleep:

  (1) **Architecture** — configure the LPTIM (Low-Power Timer) to count from the 32.768 kHz LSE (Low-Speed External) crystal oscillator. The LPTIM runs independently of the main CPU clock and continues counting in STOP mode (the deepest sleep mode where the CPU clock is halted). Set the LPTIM auto-reload to 30 seconds: counter value = 30 × 32,768 = 983,040. The LPTIM is 16-bit (max 65,535), so use the prescaler: prescaler = 16, counter = 983,040 / 16 = 61,440. On counter match, the LPTIM generates an interrupt that wakes the CPU from STOP mode.

  (2) **Power states** — STOP mode on a typical Cortex-M0+ (e.g., STM32L0): CPU halted, SRAM retained, all clocks stopped except LSE. Current draw: ~1.5 µA. The LPTIM adds ~0.5 µA. Total sleep current: ~2 µA.

  (3) **Wake-up sequence** — LPTIM interrupt fires → CPU wakes from STOP → HSI (High-Speed Internal) oscillator starts (~3 µs) → CPU runs inference at 48 MHz (25 ms at ~8 mA) → CPU re-enters STOP mode. No peripheral re-initialization needed because STOP mode retains all peripheral configurations (unlike STANDBY mode).

  (4) **Average current** — active: 25 ms at 8 mA. Sleep: 29,975 ms at 2 µA. Average: (0.025 × 8 + 29.975 × 0.002) / 30 = (0.2 + 0.06) / 30 = 8.67 mA / 30 = 0.0087 mA = 8.7 µA. Well under the 50 µA target.

  > **Napkin Math:** `HAL_Delay`: 12 mA × 24 hours = 288 mAh/day. CR2477 battery (1000 mAh): 3.5 days. LPTIM + STOP mode: 8.7 µA × 24 hours = 0.21 mAh/day. CR2477: 1000 / 0.21 = 4,762 days = **13 years**. Improvement: 1,380×. The difference between a disposable prototype and a deploy-and-forget sensor.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Hardware MAC Unit Misconception</b> · <code>hardware-mac</code></summary>

- **Interviewer:** "Your team is choosing between two MCUs for an ML product: MCU A (Cortex-M4, 100 MHz, single-cycle 32×32→64 hardware multiplier) and MCU B (Cortex-M33, 100 MHz, single-cycle 32×32→64 hardware multiplier + optional Helium MVE vector extension). Both cost $2.50. Your manager says 'they both have hardware multipliers, so ML performance will be the same.' Is the manager right?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "A hardware multiplier is a hardware multiplier — they're both single-cycle, so performance is identical." This treats the multiplier in isolation and ignores the surrounding datapath.

  **Realistic Solution:** The manager is wrong. The hardware multiplier is necessary but not sufficient for ML performance. The difference is in *how many operations the multiplier feeds per cycle*:

  (1) **Cortex-M4 (DSP extension)** — the `SMLAD` instruction performs 2 × (16-bit multiply + 32-bit accumulate) per cycle. For INT8 inference, CMSIS-NN packs two INT8 values into 16-bit operands. Effective throughput: 2 MACs/cycle. At 100 MHz: 200M MACs/s.

  (2) **Cortex-M33 with Helium MVE** — the M-Profile Vector Extension (MVE/Helium) adds 128-bit vector registers and instructions like `VMLADAVA.S8` (Vector Multiply Accumulate Long Across Vector, signed 8-bit). This processes 16 INT8 multiply-accumulates per instruction, issued over 4 beats (4 cycles for the full vector). Effective throughput: 16 MACs / 4 cycles = 4 MACs/cycle. At 100 MHz: 400M MACs/s.

  (3) **Real-world impact** — for a 1.5M MAC keyword spotting model: M4 = 1.5M / 200M = 7.5 ms. M33 with Helium = 1.5M / 400M = 3.75 ms. The M33 is 2× faster at the same clock speed and price. The multiplier is the same; the vector datapath is the differentiator.

  (4) **Caveat** — Helium requires compiler support (CMSIS-NN v5+ or Arm's Ethos-U NPU compiler). If your inference library doesn't use Helium intrinsics, the M33 falls back to scalar DSP instructions and performs identically to the M4. The hardware is only as good as the software that exploits it.

  > **Napkin Math:** 1.5M MAC model. M4 (2 MACs/cycle, 100 MHz): 7.5 ms. M33 scalar (2 MACs/cycle): 7.5 ms. M33 + Helium (4 MACs/cycle): 3.75 ms. M33 + Helium + loop overhead reduction (vector instructions eliminate loop bookkeeping): ~3.2 ms. Speedup: 2.3×. Energy: M33 Helium uses ~10% more dynamic power per cycle (wider datapath), but finishes 2.3× faster. Energy ratio: 1.1 / 2.3 = 0.48×. Helium uses *half* the energy per inference.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Boot Sequence Race Condition</b> · <code>boot-sequence</code></summary>

- **Interviewer:** "Your safety-critical fall detection wearable uses a Cortex-M4 MCU. The device must be ready to detect falls within 500 ms of power-on (regulatory requirement). Your current boot sequence takes 1.2 seconds. The breakdown: bootloader (50 ms), clock initialization (5 ms), peripheral init (20 ms), model loading from external QSPI Flash to SRAM (800 ms), sensor calibration (325 ms). How do you meet the 500 ms deadline?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use a faster QSPI Flash chip." Even the fastest QSPI Flash at 133 MHz quad-SPI can only transfer at ~50 MB/s. Your 200 KB model loads in 4 ms — the 800 ms is not from raw transfer speed.

  **Realistic Solution:** The 800 ms model load is dominated by **QSPI initialization and XIP (Execute-In-Place) setup**, not data transfer:

  (1) **Diagnose the 800 ms** — QSPI Flash initialization: 50 ms (send reset command, wait for Flash to become ready). QSPI memory-mapped mode setup: 100 ms (configure the QSPI controller for XIP). Model copy from QSPI to SRAM: 200 KB / 50 MB/s = 4 ms. Model validation (CRC check): 10 ms. TFLite Micro interpreter initialization (parsing the FlatBuffer, allocating tensors): 636 ms. The interpreter init is the bottleneck — it walks the entire model graph, resolves operators, and plans memory.

  (2) **Parallel boot** — overlap sensor calibration with model loading. Start the accelerometer calibration (which is just collecting 100 samples at 100 Hz = 1 second, but only 325 ms of CPU time for offset calculation) *before* the model is loaded. Use DMA to collect calibration samples while the CPU initializes the interpreter.

  (3) **Pre-compiled model** — replace TFLite Micro's runtime interpreter with a code-generated inference function (using tools like TVM, TFLM's code generation, or X-CUBE-AI). The model graph is compiled to C code at build time. No FlatBuffer parsing, no operator resolution, no runtime memory planning. Interpreter init drops from 636 ms to ~5 ms (just setting up the pre-allocated tensor arena).

  (4) **XIP for weights** — don't copy model weights from QSPI Flash to SRAM. Use QSPI memory-mapped mode (XIP) to read weights directly from Flash during inference. This eliminates the 4 ms copy and saves 200 KB of SRAM. Inference is ~10% slower (QSPI read latency vs SRAM) but boot is faster.

  (5) **Revised boot timeline** — bootloader: 50 ms. Clock + peripheral init: 25 ms. QSPI init + XIP setup: 150 ms. Code-generated model init: 5 ms. Sensor calibration (overlapped): 0 ms additional (runs during QSPI init via DMA). Total: 230 ms. Under the 500 ms deadline with 270 ms margin.

  > **Napkin Math:** Original: 50 + 5 + 20 + 800 + 325 = 1,200 ms. Optimized: 50 + 5 + 20 + 150 + 5 + 0 (calibration overlapped) = 230 ms. Speedup: 5.2×. The 636 ms interpreter init was 53% of boot time. Code generation eliminated it entirely. If XIP adds 10% inference latency: 25 ms → 27.5 ms per inference. Acceptable for fall detection (needs <100 ms response).

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Voltage Scaling Tightrope</b> · <code>voltage-scaling</code></summary>

- **Interviewer:** "Your battery-powered wildlife monitoring sensor runs a bird call classifier on an STM32L5 (Cortex-M33, 110 MHz max, 256 KB SRAM). To maximize battery life, you reduce the core voltage from 1.2V (Range 1, 110 MHz max) to 0.9V (Range 2, 26 MHz max). Inference time increases from 15 ms to 63 ms — roughly 4.2× slower, matching the clock ratio. But after deploying 500 units in the field, 3% of devices produce incorrect classifications intermittently. The other 97% work perfectly. What's happening?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "3% defect rate means a bad batch of MCUs — replace them." The MCUs are not defective. They're operating at the *margin* of their voltage-frequency specification.

  **Realistic Solution:** The intermittent errors are caused by **voltage-frequency margin violations** due to process variation:

  (1) **Process variation** — semiconductor manufacturing produces chips with slightly different transistor characteristics. The STM32L5 datasheet specifies 0.9V operation up to 26 MHz, but this is the *guaranteed minimum* across all process corners (worst-case slow silicon at maximum temperature). Your 3% of failing devices have "slow" transistors that need slightly more voltage to switch reliably at 26 MHz. At 0.9V, their critical path timing occasionally violates setup time, causing bit flips in the ALU or register file.

  (2) **Why it's intermittent** — the timing violations are temperature-dependent. At 25°C, all devices work. At 45°C (direct sunlight on the sensor enclosure), transistor switching slows further. The 3% of marginal devices cross the failure threshold. The errors manifest as incorrect MAC results deep in the inference pipeline — a single bit flip in a 32-bit accumulator can change the classification output.

  (3) **Why standard testing didn't catch it** — factory testing runs at room temperature for a few seconds. Field deployment sees temperature cycling from -10°C to 50°C over months. The marginal devices only fail at elevated temperatures under sustained compute load (inference heats the die by 5-10°C above ambient).

  (4) **Fix** — add a voltage guard band. Instead of running at exactly 0.9V/26 MHz, run at 0.95V/26 MHz. The 50 mV guard band covers process variation across all corners. Power increase: $(0.95/0.9)^2 = 1.11×$ — only 11% more power. Alternatively, reduce clock to 20 MHz at 0.9V, adding margin by running below the maximum frequency for that voltage.

  (5) **Detection** — add a runtime integrity check: after every inference, re-run the final classification layer (cheap — just a small FC layer) and compare results. If they differ, the device is experiencing bit flips. Flag it for voltage adjustment via OTA config update.

  > **Napkin Math:** At 1.2V/110 MHz: P = C × 1.2² × 110M = 158.4C. Energy per inference: 158.4C × 15 ms. At 0.9V/26 MHz: P = C × 0.9² × 26M = 21.06C. Energy: 21.06C × 63 ms = 1,326.8C. At 1.2V: 158.4C × 15 ms = 2,376C. Ratio: 1,326.8 / 2,376 = 0.56×. Low-voltage operation saves 44% energy per inference. At 0.95V/26 MHz: P = C × 0.95² × 26M = 23.47C. Energy: 23.47C × 63 ms = 1,478.6C. Still 38% energy savings vs 1.2V, but now reliable. The 11% power increase over 0.9V eliminates the 3% field failure rate.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Debug Interface Profiling Trap</b> · <code>debug-interface</code></summary>

- **Interviewer:** "You're profiling your ML inference on a Cortex-M4 using the SWD (Serial Wire Debug) interface and a Segger J-Link. Your profiler reports inference takes 42 ms. You disconnect the debugger, add GPIO toggle pins around the inference call, and measure with an oscilloscope: 28 ms. Why does the debugger add 14 ms to your inference?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The debugger slows down the CPU clock." SWD does not affect the CPU clock. The CPU runs at the same frequency with or without a debugger attached.

  **Realistic Solution:** The 14 ms overhead comes from **debug monitor interactions and DWT sampling**:

  (1) **Periodic halts for sampling** — the J-Link's real-time profiling mode (e.g., Ozone's sampling profiler) works by periodically halting the CPU to read the Program Counter (PC). Each halt takes ~2 µs (SWD clock at 4 MHz, read PC register = 2 SWD frames). At the default 10 kHz sampling rate: 10,000 halts/second × 2 µs = 20 ms/second of halt time. During a 28 ms inference: 0.028 × 20 ms = 0.56 ms. That's only 0.56 ms — not enough to explain 14 ms.

  (2) **The real culprit: ITM trace and SWO** — your profiling configuration enables ITM (Instrumentation Trace Macrocell) output over the SWO (Serial Wire Output) pin. The ITM buffer is 32 bytes. When the buffer fills, the CPU stalls on the next ITM write until the SWO pin drains the buffer. SWO runs at the SWD clock speed (4 MHz), not the CPU clock (100 MHz). If your inference code has CMSIS-NN debug prints or TFLite Micro's `MicroPrintf` calls compiled in (common in debug builds), each printf generates 50-100 bytes of ITM data. At 4 MHz SWO: 100 bytes × 10 bits/byte / 4M bps = 250 µs per printf. With 50 printfs during inference: 50 × 250 µs = 12.5 ms.

  (3) **Plus: Flash breakpoint overhead** — if you have breakpoints set (even disabled ones in some debugger implementations), the debug monitor exception fires on each breakpoint address match. Even a "disabled" breakpoint in some J-Link firmware versions still triggers a comparator match in the FPB (Flash Patch and Breakpoint) unit, adding ~1 µs per match.

  (4) **Fix** — compile with `NDEBUG` defined to strip all `MicroPrintf` calls. Use DWT cycle counter (`DWT->CYCCNT`) for timing instead of ITM trace — it's a hardware counter with zero overhead. Read it before and after inference: `cycles = DWT->CYCCNT_after - DWT->CYCCNT_before`. Convert to time: `cycles / CPU_FREQ`.

  > **Napkin Math:** True inference: 28 ms. ITM overhead: 50 printfs × 250 µs = 12.5 ms. Sampling overhead: 0.56 ms. FPB overhead: ~0.94 ms. Total debugger overhead: 14 ms. Profiled time: 42 ms. Error: 50%. This is why experienced embedded engineers never trust debugger-reported timing for performance-critical code. DWT cycle counter accuracy: ±1 cycle. GPIO toggle + oscilloscope accuracy: ±10 ns. Both are orders of magnitude better than ITM-based profiling.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Heterogeneous MCU Scheduling Problem</b> · <code>heterogeneous-compute</code></summary>

- **Interviewer:** "You're deploying a multi-modal anomaly detection system on an NXP i.MX RT1170 — a heterogeneous MCU with a Cortex-M7 (1 GHz, 2 MB SRAM, I/D caches, FPU) and a Cortex-M4 (400 MHz, 256 KB SRAM, DSP, no cache). Your system runs two models: a vibration FFT + classifier (compute-heavy, 2M MACs) and a temperature trend LSTM (memory-heavy, 500K MACs but 180 KB of state). Which model goes on which core, and how do they communicate?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Put both models on the M7 — it's faster." The M7 is faster per-core, but running both models sequentially on the M7 wastes the M4 entirely and doubles your inference latency.

  **Realistic Solution:** The assignment depends on each model's *bottleneck type*, not just its MAC count:

  (1) **Vibration classifier → Cortex-M7** — this model is compute-bound (2M MACs with small activations). The M7's advantages: 1 GHz clock (2.5× the M4), I-cache (eliminates Flash wait states for the inference loop), D-cache (prefetches weight data), and FPU (if any layers use FP16). With CMSIS-NN on M7: 2M MACs / (2 MACs/cycle × 1 GHz) = 1 ms. On M4: 2M / (2 × 400M) = 2.5 ms. The M7 is 2.5× faster for this workload.

  (2) **Temperature LSTM → Cortex-M4** — this model is memory-bound (180 KB of hidden state that must be read and written every timestep). The M7's D-cache is 32 KB — the 180 KB state thrashes the cache, causing constant evictions and refills. Each cache miss costs ~10 cycles (SRAM access through the bus matrix). The M4 has no cache but has a tightly-coupled SRAM (TCM) with single-cycle access. Place the 180 KB LSTM state in the M4's TCM. Every state access is 1 cycle, guaranteed. The M4 is actually *faster* for this memory-bound workload despite its lower clock: M4 at 400 MHz with 1-cycle access vs M7 at 1 GHz with ~4-cycle average access (cache miss rate × miss penalty).

  (3) **Inter-core communication** — use the i.MX RT1170's shared SRAM (512 KB, accessible by both cores) with a mailbox mechanism. The M7 writes vibration classification results to a shared struct. The M4 writes temperature anomaly results. A hardware semaphore (SEMA4 peripheral) prevents simultaneous access. Message passing overhead: ~100 ns per exchange.

  (4) **Synchronization** — both models run independently at different rates. The vibration model runs at 100 Hz (10 ms period, 1 ms compute — 10% M7 utilization). The LSTM runs at 1 Hz (1 second period, 5 ms compute — 0.5% M4 utilization). A fusion layer on the M7 combines both outputs every second to produce the final anomaly score.

  > **Napkin Math:** Sequential on M7: 1 ms (vibration) + 8 ms (LSTM with cache thrashing) = 9 ms per cycle. Parallel heterogeneous: max(1 ms, 5 ms) = 5 ms per cycle. Speedup: 1.8×. But the real win is utilization: M7 at 10%, M4 at 0.5% — both cores are mostly sleeping. Power: M7 active at 1 GHz = 150 mW. M4 active at 400 MHz = 30 mW. Average power (duty-cycled): M7: 0.1 × 150 = 15 mW. M4: 0.005 × 30 = 0.15 mW. Total: 15.15 mW. vs M7-only: 0.105 × 150 = 15.75 mW. Heterogeneous saves 4% power and halves latency.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

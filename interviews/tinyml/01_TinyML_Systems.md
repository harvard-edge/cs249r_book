# Round 1: TinyML Systems — Inference at the Edge of Physics 🔬

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

The domain of the TinyML Systems Engineer. This round tests your understanding of what happens when the entire model, runtime, and inference engine must coexist in kilobytes of SRAM, execute in microseconds, and survive on milliwatts of power. There is no operating system, no virtual memory, no second chance — every byte is a design decision.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/tinyml/01_TinyML_Systems.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 🧠 Memory Layout & SRAM Partitioning

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Flat Memory Reality</b> · <code>memory-layout</code></summary>

**Interviewer:** "A junior engineer on your team says they'll use `malloc` to dynamically allocate activation buffers during inference on a Cortex-M4. Why do you stop them?"

**Common Mistake:** "Malloc is just slow on microcontrollers." Speed isn't the primary concern — the failure mode is far worse.

**Realistic Solution:** On a bare-metal MCU with 256 KB of SRAM, there is no virtual memory and no MMU. Dynamic allocation causes heap fragmentation in a fixed-size memory pool. After a few hundred inference cycles, `malloc` returns NULL — not because you're out of total memory, but because the free space is scattered into unusable fragments. TFLite Micro solves this by requiring a single, pre-allocated flat tensor arena. All activations, scratch buffers, and intermediate tensors are placed at compile-time offsets within this arena. Zero fragmentation, deterministic memory usage, and you know at build time whether the model fits.

> **Napkin Math:** A Cortex-M4 with 256 KB SRAM must hold: firmware (~40 KB), stack (~8 KB), tensor arena (remaining ~200 KB). If `malloc` fragments even 10% of the arena into gaps smaller than the largest activation buffer, inference fails — even though 20 KB of total free space remains.

**📖 Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Peak RAM Puzzle</b> · <code>memory-layout</code></summary>

**Interviewer:** "Your 8-layer convolutional model needs 300 KB of peak activation RAM, but your Cortex-M7 only has 256 KB of SRAM available after firmware and stack. You cannot change the model architecture. How do you make it fit?"

**Common Mistake:** "Quantize the model to use less memory." Quantizing weights from FP32 to INT8 shrinks the weight storage in Flash, but activations are already INT8 in a quantized pipeline — the peak activation footprint barely changes.

**Realistic Solution:** Operator scheduling for peak RAM reduction. The key insight: not all layers' activations are alive at the same time. By reordering operator execution (computing and immediately consuming activations before moving to the next layer), you can overwrite dead tensors. Tools like TFLite Micro's memory planner and MCUNet's patch-based inference take this further — instead of computing an entire feature map at once, you compute it in spatial patches, reducing peak RAM from the full feature map size to a single patch's worth.

> **Napkin Math:** A standard 8-layer CNN with 64 channels at 48×48 spatial resolution: one full activation tensor = $64 \times 48 \times 48 \times 1$ byte (INT8) = 147 KB. Two such tensors alive simultaneously (input + output of a layer) = 294 KB — exceeds 256 KB. With patch-based inference (e.g., 12×48 patches), peak activation drops to $64 \times 12 \times 48 \times 1 = 36$ KB per tensor, and two alive = 72 KB. The model now fits with room to spare.

> **Key Equation:** $\text{Peak RAM} = \max_{t} \sum_{i \in \text{live}(t)} \text{size}(activation_i)$

**📖 Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)
</details>

---

### 🔢 Quantization & Fixed-Point Arithmetic

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Quantization Cliff</b> · <code>quantization</code></summary>

**Interviewer:** "Your keyword spotting model runs at 92% accuracy in FP32. Post-training quantization to INT8 drops it to 91%. Your manager asks you to push to INT4 to halve the model size again. You try it and accuracy collapses to 74%. What happened, and how do you recover?"

**Common Mistake:** "INT4 is just too aggressive — we need to stay at INT8." This gives up without understanding the mechanism or the fix.

**Realistic Solution:** INT4 has only 16 discrete levels per weight. For layers with wide weight distributions (like the first convolutional layer processing raw spectrograms, or the final classification head), 16 levels cannot capture the dynamic range, causing severe clipping and rounding errors that cascade through the network. The fix is mixed-precision quantization: keep sensitive layers (first and last) at INT8 while quantizing the middle depthwise layers to INT4. This recovers most of the accuracy while still shrinking the model significantly.

> **Napkin Math:** A 250 KB INT8 keyword spotting model: first conv + last FC ≈ 40 KB (keep at INT8). Remaining layers ≈ 210 KB → 105 KB at INT4. Total = 145 KB — a 42% reduction instead of 50%, but accuracy recovers to ~90% vs the 74% of uniform INT4.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Integer Arithmetic Engine</b> · <code>quantization</code> <code>integer-inference</code></summary>

**Interviewer:** "Walk me through exactly how a quantized Conv2D layer executes on a Cortex-M4 with no floating-point unit. I want to see the math — from quantized inputs to quantized outputs — with no floats anywhere in the pipeline."

**Common Mistake:** "You just multiply INT8 values and truncate." This ignores the accumulator width, the zero-point offsets, and the requantization step — all of which are essential for correctness.

**Realistic Solution:** The full integer-only inference pipeline works as follows. Each tensor has a quantization scheme: $x_{real} = S_x (x_{int} - Z_x)$ where $S$ is a float scale and $Z$ is an integer zero-point. For convolution, the real-valued output is $Y = W \cdot X + b$. Substituting the quantization parameters and expanding:

The core MAC accumulates into a 32-bit integer register: $\text{acc}_{32} = \sum (w_{int8} - Z_w)(x_{int8} - Z_x)$. This stays in pure integer arithmetic on the Cortex-M4's 32-bit ALU. The critical trick is requantization: converting the INT32 accumulator back to INT8 for the next layer. The scale ratio $M = \frac{S_w \cdot S_x}{S_y}$ is pre-computed and represented as a fixed-point multiply: $M \approx M_0 \times 2^{-n}$ where $M_0$ is an INT32 value and $n$ is a right-shift. The final output is: $y_{int8} = \text{clamp}\left(\text{round}\left(\frac{M_0 \cdot \text{acc}_{32}}{2^n}\right) + Z_y,\ 0,\ 255\right)$. No float touched the pipeline.

> **Napkin Math:** A 3×3 depthwise conv on a 48×48×64 INT8 feature map: MACs = $3 \times 3 \times 48 \times 48 \times 64 = 1,327,104$. Each MAC is an INT8×INT8→INT32 multiply-add — one cycle on Cortex-M4 with CMSIS-NN's SMLAD instruction. At 100 MHz: $1.3M / 100M \approx 13$ ms per layer. The requantization adds ~1 shift + 1 add per output element ($48 \times 48 \times 64 = 147,456$ ops) — negligible overhead.

> **Key Equation:** $y_{int8} = \text{clamp}\!\left(\text{round}\!\left(M_0 \cdot \sum(w_{int8} - Z_w)(x_{int8} - Z_x) \gg n\right) + Z_y,\ 0,\ 255\right)$

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 🏗️ Model Architecture for Microcontrollers

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Depthwise Separable Advantage</b> · <code>model-architecture</code></summary>

**Interviewer:** "You need to deploy a small image classifier on an ESP32-S3 with 512 KB SRAM. A standard Conv2D with 32 input channels, 64 output channels, and a 3×3 kernel works in simulation but exceeds your SRAM budget. Your colleague suggests replacing it with a depthwise separable convolution. How much memory and compute does this actually save, and is there a catch?"

**Common Mistake:** "Depthwise separable is just a smaller convolution — it's always better." People quote the theoretical reduction ratio without checking whether the activation memory (not just the weight memory) actually fits.

**Realistic Solution:** A standard 3×3 Conv2D with 32→64 channels has $3 \times 3 \times 32 \times 64 = 18,432$ weight parameters. A depthwise separable replacement splits this into a 3×3 depthwise conv ($3 \times 3 \times 32 = 288$ params) plus a 1×1 pointwise conv ($1 \times 1 \times 32 \times 64 = 2,048$ params) = 2,336 total — an 8× reduction in weights and roughly 8–9× fewer MACs. The catch: on MCUs, the compute savings are real, but the activation memory is unchanged. Both approaches produce a 64-channel output feature map of the same spatial size. The depthwise separable version also creates an intermediate 32-channel feature map between the two stages. If SRAM is the bottleneck (and it usually is), the activation footprint matters more than the weight count.

> **Napkin Math:** For a 48×48 input: standard conv output activation = $64 \times 48 \times 48 = 147$ KB (INT8). Depthwise separable intermediate = $32 \times 48 \times 48 = 73$ KB, final output = 147 KB. Peak (intermediate + output alive) = 220 KB. Standard conv peak (input + output) = $73 + 147 = 220$ KB. Activation memory is identical — the win is purely in weights (18 KB → 2.3 KB) and compute (18M MACs → 2.2M MACs).

> **Key Equation:** $\text{Depthwise separable MACs} = K^2 \cdot C_{in} \cdot H \cdot W + C_{in} \cdot C_{out} \cdot H \cdot W$

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### ⚡ Power, Energy & Duty Cycling

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Battery Life Equation</b> · <code>power-energy</code></summary>

**Interviewer:** "You're deploying a gesture recognition system on a Cortex-M4 powered by a 300 mAh coin cell battery. The MCU draws 50 mW active and 10 µW in deep sleep. Inference takes 30 ms and runs once per second. Your product manager asks: how long will the battery last?"

**Common Mistake:** "50 mW continuously from a 300 mAh battery — that's about 20 hours." This treats the MCU as always-on, ignoring the duty cycle entirely.

**Realistic Solution:** Duty cycling is the key. The MCU is active for 30 ms out of every 1000 ms — a 3% duty cycle. Average power = $(0.03 \times 50\text{ mW}) + (0.97 \times 0.01\text{ mW}) = 1.5\text{ mW} + 0.0097\text{ mW} \approx 1.51\text{ mW}$. A 300 mAh coin cell at 3V provides 900 mWh. Battery life = $900\text{ mWh} / 1.51\text{ mW} \approx 596\text{ hours} \approx 25\text{ days}$. The duty cycle extends battery life from less than a day to nearly a month.

> **Napkin Math:** Always-on: $900\text{ mWh} / 50\text{ mW} = 18\text{ hours}$. With 3% duty cycle: $900\text{ mWh} / 1.51\text{ mW} = 596\text{ hours}$. That's a 33× improvement — the difference between a disposable prototype and a shippable product.

> **Key Equation:** $P_{avg} = \delta \cdot P_{active} + (1 - \delta) \cdot P_{sleep}$, where $\delta = t_{inference} / t_{period}$

**📖 Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Energy Harvesting Wall</b> · <code>power-energy</code></summary>

**Interviewer:** "You're designing a structural health monitoring sensor powered by a small solar cell that provides an average of 0.5 mW indoors. The Cortex-M4 draws 50 mW active. Your model inference takes 30 ms. What is the maximum inference rate you can sustain indefinitely without a battery?"

**Common Mistake:** "0.5 mW / 50 mW = 1% duty cycle, so one inference every 100 seconds." This gets the duty cycle right but forgets that you need a storage capacitor to buffer the energy, and ignores the energy cost of waking up.

**Realistic Solution:** The energy budget per inference = $P_{active} \times t_{inference} = 50\text{ mW} \times 30\text{ ms} = 1.5\text{ mJ}$. Add wake-up overhead (clock stabilization, sensor warm-up) of ~5 ms at 50 mW = 0.25 mJ. Total per inference ≈ 1.75 mJ. The harvester provides 0.5 mW = 0.5 mJ/s. Maximum sustainable rate = $0.5\text{ mJ/s} / 1.75\text{ mJ} \approx 0.29$ inferences/second, or roughly one inference every 3.5 seconds. But this assumes zero sleep power and 100% harvester efficiency. With realistic 60% DC-DC conversion efficiency and 10 µW sleep current: usable power ≈ 0.3 mW, sustainable rate drops to one inference every ~6 seconds. You also need a capacitor large enough to deliver the 50 mW burst: $C = P \cdot t / (0.5 \cdot V^2) \approx 50\text{mW} \times 35\text{ms} / (0.5 \times 3.3^2) \approx 320\text{ µF}$ minimum.

> **Napkin Math:** Energy in = 0.5 mW × 6 s = 3 mJ harvested. After 60% conversion = 1.8 mJ available. Energy out = 1.75 mJ per inference. Margin = 0.05 mJ — barely sustainable. Drop below 0.5 mW (cloudy day) and the system dies.

> **Key Equation:** $f_{max} = \frac{\eta \cdot P_{harvest}}{E_{inference} + E_{wake}}$

**📖 Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)
</details>

---

### 🛠️ Compiler, Runtime & CMSIS-NN

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The CMSIS-NN Speedup</b> · <code>compiler-runtime</code></summary>

**Interviewer:** "Your colleague wrote a naive C implementation of an INT8 matrix multiply for a Cortex-M4. It runs in 45 ms. You replace it with the CMSIS-NN equivalent and it drops to 6 ms — a 7.5× speedup. The clock speed didn't change. Where did the speedup come from?"

**Common Mistake:** "CMSIS-NN uses NEON SIMD instructions." Cortex-M4 does not have NEON — that's Cortex-A series. People confuse the two ARM families.

**Realistic Solution:** CMSIS-NN exploits the Cortex-M4's DSP extension — specifically the SIMD (Single Instruction, Multiple Data) instructions like `SMLAD` (Signed Multiply Accumulate Dual). `SMLAD` performs two 16-bit multiplies and accumulates both results into a 32-bit accumulator in a single cycle. For INT8 data, CMSIS-NN packs two INT8 values into a single 16-bit half-word, then uses `SMLAD` to process two MACs per cycle instead of one. Combined with loop unrolling, data re-ordering for cache-friendly access, and elimination of branch overhead, this yields the 7–8× speedup over naive C. It's the microcontroller equivalent of using Tensor Cores on a GPU — you must use the specialized hardware paths or you leave most of the silicon idle.

> **Napkin Math:** Naive C: 1 MAC per cycle (load, multiply, accumulate = ~3 cycles with pipeline, but ~1 effective with optimization). CMSIS-NN with `SMLAD`: 2 MACs per cycle. With loop unrolling (4 iterations): pipeline stalls eliminated, achieving ~1.8 MACs/cycle effective. For a 256×256 matrix multiply: $256 \times 256 = 65,536$ MACs. Naive: ~200K cycles. CMSIS-NN: ~36K cycles. At 100 MHz: 2 ms vs 0.36 ms per matmul.

**📖 Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)
</details>

---

### 🎤 Sensor Pipelines & Real-Time Inference

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Keyword Spotting Pipeline</b> · <code>sensor-pipeline</code> <code>memory-layout</code></summary>

**Interviewer:** "Design the complete inference pipeline for an always-on keyword spotting system on a Cortex-M4 (100 MHz, 256 KB SRAM, no FPU). The microphone samples at 16 kHz. You need to detect the wake word 'Hey Device' within 500 ms of it being spoken. Walk me through every stage — from raw audio samples to classification output — and show me the memory budget."

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

**📖 Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)
</details>

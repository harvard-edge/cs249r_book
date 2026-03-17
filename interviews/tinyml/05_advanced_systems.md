# Round 5: Advanced TinyML Systems 🔬

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

This round is for principal-level TinyML engineers and researchers who design the systems that others deploy. These questions span neural architecture search under MCU constraints, energy harvesting system design, multi-sensor fusion, compiler design, federated learning on constrained fleets, predictive maintenance, always-on detection at sub-milliwatt budgets, hardware-software co-design, and streaming anomaly detection. Each question requires reasoning across the full stack — from silicon to algorithm.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/tinyml/05_TinyML_Advanced.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 🧬 Neural Architecture Search for MCUs

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> MCUNet Search Space Design</b> · <code>architecture</code> <code>nas</code></summary>

- **Interviewer:** "You're building a NAS pipeline to find the best image classification model for a Cortex-M4 (256 KB SRAM, 1 MB flash, 168 MHz). MCUNet showed this is possible, but their search took 24 GPU-hours. Your team has a budget of 4 GPU-hours. How do you design the search space to find a competitive model in 6× less time?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just run MCUNet's search for fewer iterations." Reducing iterations without adapting the search space gives you a random model from a huge space — not a good model from a focused space.

  **Realistic Solution:** The key insight from MCUNet is that the search space itself must be constrained by the hardware. A standard NAS search space (like those used for ImageNet on GPUs) contains architectures that are infeasible on MCUs — they waste search time evaluating models that will never fit. Shrink the search space to only contain feasible architectures:

  **(1) Pre-filter by SRAM:** Before search begins, analytically compute the peak SRAM for each candidate block configuration. A block with expansion ratio 6 and 64 channels at 48×48 resolution creates a peak activation of $6 \times 64 \times 48 \times 48 = 884$ KB — exceeds 256 KB SRAM. Eliminate all such blocks from the search space. This typically removes 60-70% of candidates.

  **(2) Constrain channel widths to SIMD-aligned values:** Only allow channel counts that are multiples of 4 (for CMSIS-NN's `SMLAD` packing). Search space: {4, 8, 16, 24, 32, 48, 64} instead of {1, 2, ..., 128}. This reduces the channel dimension from 128 options to 7.

  **(3) Use a two-stage search:** Stage 1 — search the macro architecture (number of blocks, resolution per block) using a simple latency predictor (lookup table from profiling ~50 configurations on real hardware). This takes minutes, not hours. Stage 2 — search the micro architecture (kernel sizes, expansion ratios) within the feasible macro architectures using one-shot NAS with weight sharing. This takes 3-4 GPU-hours.

  **(4) Hardware-in-the-loop validation:** Instead of training each candidate to convergence, train the supernet once (2 GPU-hours) and evaluate sub-networks by inheriting weights. Validate the top-10 candidates on the actual MCU (flash the model, run inference, measure latency and SRAM). This catches discrepancies between the predictor and real hardware.

  **Reduced search space:** 3 macro configs × 5 blocks × 3 kernel sizes × 4 channel widths × 2 expansion ratios = 3 × 5 × 3 × 4 × 2 = 360 feasible architectures (vs ~6 × 10¹² unconstrained). One-shot evaluation of 360 candidates: ~2 GPU-hours. Total: 4 GPU-hours.

  > **Napkin Math:** Unconstrained search space: ~10¹² architectures. SRAM-filtered: ~10⁴. SIMD-aligned channels: ~10³. Two-stage search: evaluate ~360 candidates. Supernet training: 2 GPU-hours. Sub-network evaluation: 360 × 20 seconds = 2 GPU-hours. Total: 4 GPU-hours. Expected accuracy: within 1-2% of the full MCUNet search (which found 70.7% ImageNet top-1 on Cortex-M4 with 256 KB SRAM).

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

---

### 🔋 Energy Harvesting System Design

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> Solar + Supercapacitor + MCU System Design</b> · <code>power</code> <code>system-design</code></summary>

- **Interviewer:** "Design a battery-free structural health monitoring system powered by a small indoor solar cell (2 cm², ~100 µW average indoors). The system must run vibration anomaly detection on a Cortex-M4 that draws 30 mW active. How do you bridge the 300× gap between harvest rate and compute demand?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use a bigger solar panel." The constraint is physical — the sensor must fit on a machine bearing with limited surface area. You can't make the panel bigger.

  **Realistic Solution:** Bridge the energy gap with a store-and-burst architecture:

  **(1) Energy storage:** Use a supercapacitor (not a battery — batteries degrade in the vibration/temperature environment of industrial machinery). A 100 mF supercapacitor at 3.3V stores $0.5 \times 0.1 \times 3.3^2 = 0.545$ J. This is enough for $0.545 / (0.030 \times 0.060) = 302$ inference cycles (at 30 mW × 60ms each).

  **(2) Energy accumulation:** At 100 µW harvest rate, charging the supercapacitor from 2.0V (minimum operating voltage) to 3.3V takes: energy needed = $0.5 \times 0.1 \times (3.3^2 - 2.0^2) = 0.345$ J. Time = $0.345 / 0.0001 = 3,450$ seconds ≈ **57 minutes**. With 70% DC-DC efficiency: ~82 minutes.

  **(3) Burst operation:** After 82 minutes of charging, the MCU wakes, reads the accelerometer for 1 second (1 kHz × 1s = 1000 samples), runs inference (60ms at 30 mW = 1.8 mJ), transmits the result via BLE (20ms at 40 mW = 0.8 mJ), and sleeps. Total energy per burst: sensor (5 mJ) + inference (1.8 mJ) + BLE (0.8 mJ) + wake overhead (0.5 mJ) = **8.1 mJ**. The supercapacitor has 345 mJ available — enough for 42 bursts before recharging.

  **(4) Adaptive scheduling:** Monitor supercapacitor voltage via ADC. When voltage > 3.0V: run inference every 2 minutes. When 2.5-3.0V: every 10 minutes. Below 2.5V: sleep until voltage recovers. This prevents the system from draining the supercapacitor below the MCU's minimum operating voltage (brownout).

  **(5) Voltage supervisor:** Use a hardware voltage supervisor (e.g., TPS3839) that holds the MCU in reset until the supercapacitor reaches 2.8V. This prevents the MCU from booting into a brownout loop (boot → voltage drops → reset → boot → ...) that wastes all harvested energy.

  > **Napkin Math:** Harvest: 100 µW. Charge time: 82 min. Energy per burst: 8.1 mJ. Bursts per charge: 345 / 8.1 = 42. If we use 1 burst per charge cycle (conservative): 1 inference every 82 minutes. If we use 10 bursts per charge (drain supercap to 2.5V): 10 inferences every 82 minutes, then recharge for ~30 minutes. Effective rate: 10 inferences / 112 minutes = **1 inference every 11 minutes**. For structural health monitoring (detecting bearing degradation over weeks), this is more than sufficient.

  > **Key Equation:** $t_{\text{charge}} = \frac{0.5 \times C \times (V_{max}^2 - V_{min}^2)}{\eta \times P_{\text{harvest}}}$

  📖 **Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)

  </details>

</details>

---

### 🔀 Multi-Sensor Fusion on a Single MCU

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> Fusing Accelerometer + Microphone + Temperature on One MCU</b> · <code>sensor-pipeline</code> <code>memory-layout</code></summary>

- **Interviewer:** "You're building an industrial equipment health monitor on a single Cortex-M4 (256 KB SRAM, 168 MHz). It must fuse data from three sensors: a 3-axis accelerometer (1 kHz), a microphone (16 kHz), and a temperature sensor (1 Hz). Each sensor feeds a different model branch. Design the memory layout, DMA strategy, and inference scheduling."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run three separate models sequentially." Three separate models triple the weight storage and likely exceed flash. And sequential execution may miss real-time deadlines.

  **Realistic Solution:** Design a shared-backbone multi-head architecture with time-multiplexed inference:

  **Memory layout (256 KB SRAM):**

  | Region | Size | Contents |
  |--------|------|----------|
  | Firmware + stack | 48 KB | Application code, interrupt handlers, stack |
  | Accel DMA buffers | 3 KB | 2 × 256 samples × 3 axes × 2 bytes (ping-pong) |
  | Audio DMA buffers | 4 KB | 2 × 1024 samples × 2 bytes (ping-pong) |
  | Feature buffers | 8 KB | Mel spectrogram (2 KB), accel FFT (2 KB), feature vector (4 KB) |
  | Tensor arena | 180 KB | Shared across all model heads |
  | Temp + misc | 13 KB | Temperature history, BLE buffers, logging |

  **DMA strategy:** Three independent DMA channels, each with double-buffering:
  - DMA1: Accelerometer SPI → accel buffer A/B. Interrupt every 256ms.
  - DMA2: Microphone I2S → audio buffer A/B. Interrupt every 64ms.
  - DMA3: Temperature ADC → single variable. Triggered by timer every 1 second.

  **Inference scheduling:** The three sensors produce data at different rates. Use a priority-based scheduler in the main loop:

  1. **Audio (highest priority, 64ms deadline):** Every 64ms, extract Mel features from the audio buffer (2ms). Run the audio head of the model (8ms). Total: 10ms. CPU utilization: 10/64 = 15.6%.

  2. **Accelerometer (medium priority, 256ms deadline):** Every 256ms, compute FFT on the accel buffer (3ms). Run the vibration head (10ms). Total: 13ms. CPU utilization: 13/256 = 5.1%.

  3. **Temperature (lowest priority, 1s deadline):** Every 1 second, append temperature to a 60-sample history buffer. Run the thermal trend head (2ms). CPU utilization: 2/1000 = 0.2%.

  **Total CPU utilization:** 15.6% + 5.1% + 0.2% = **20.9%**. The MCU sleeps ~79% of the time.

  **The shared tensor arena trick:** All three model heads share the same 180 KB tensor arena because they never run simultaneously. The audio head uses ~120 KB peak, the vibration head uses ~80 KB, and the thermal head uses ~10 KB. Since they execute sequentially within each scheduling cycle, the arena is reused — no additional memory needed.

  > **Napkin Math:** Three separate models: 3 × 80 KB weights = 240 KB flash + 3 × 120 KB arenas = 360 KB SRAM. Doesn't fit. Shared backbone + 3 heads: 100 KB weights (shared backbone) + 3 × 10 KB heads = 130 KB flash. Single arena: 180 KB SRAM. Fits with 76 KB headroom. Weight savings: 46%. SRAM savings: 50%.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)

  </details>

</details>

---

### 🛠️ TinyML Compiler Design

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> TFLite Micro vs TVM vs Custom Compiler</b> · <code>compiler-runtime</code> <code>optimization</code></summary>

- **Interviewer:** "Your team is choosing the inference runtime for a new TinyML product line targeting Cortex-M4 and RISC-V MCUs. The candidates are TFLite Micro, Apache TVM (microTVM), and a custom ahead-of-time compiler. Compare them across five dimensions: code size, inference speed, memory efficiency, portability, and engineering effort."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "TFLite Micro is the industry standard — just use it." TFLite Micro is the safest choice but not always the best. Each runtime makes fundamentally different trade-offs.

  **Realistic Solution:**

  **TFLite Micro (Interpreter-based):**
  - **Code size:** ~50-100 KB for the interpreter + kernel library. Only kernels for operators used by your model are linked (selective registration). But the interpreter loop itself adds overhead.
  - **Inference speed:** Baseline. The interpreter dispatches operators at runtime via function pointers — each operator call has ~10-20 cycles of dispatch overhead. For a 20-layer model: ~400 cycles of pure overhead. Negligible for large layers, but for small models with many tiny layers, dispatch overhead can be 5-10% of total inference time.
  - **Memory efficiency:** The tensor arena is statically allocated, but tensor placement is computed at runtime. This means the memory planner runs on the MCU at model load time — consuming stack space and time. Arena size is optimal but load time is ~100ms for complex models.
  - **Portability:** Excellent. Supports Cortex-M, RISC-V, Xtensa (ESP32), and ARC. Adding a new target requires implementing ~20 kernel functions.
  - **Engineering effort:** Low. Well-documented, large community, Google-maintained. Model conversion from TensorFlow is one command.

  **Apache TVM (microTVM, Ahead-of-Time):**
  - **Code size:** ~20-40 KB. TVM compiles the model into standalone C code — no interpreter, no runtime library. Each operator is a specialized C function with the tensor shapes baked in.
  - **Inference speed:** 10-30% faster than TFLite Micro. The ahead-of-time compilation eliminates dispatch overhead and enables cross-operator optimizations (operator fusion, layout transformation). TVM's auto-tuning can search for the best loop tiling and unrolling for each operator on the target hardware.
  - **Memory efficiency:** Superior. TVM's memory planner runs at compile time with global visibility — it can find tensor placements that TFLite Micro's runtime planner misses. Typical SRAM savings: 5-15%.
  - **Portability:** Good but requires more effort per target. Auto-tuning requires running benchmarks on the actual MCU hardware (or an accurate simulator).
  - **Engineering effort:** Medium-high. The TVM stack is complex. Model conversion requires understanding the Relay IR. Auto-tuning requires hardware-in-the-loop infrastructure.

  **Custom ahead-of-time compiler:**
  - **Code size:** Minimal — ~5-10 KB. Every byte of generated code is specific to your model. No generality overhead.
  - **Inference speed:** Potentially the fastest. You can hand-optimize the generated assembly for your specific model and target. But this requires deep expertise in both the model architecture and the target ISA.
  - **Memory efficiency:** Optimal. You control every byte of SRAM allocation.
  - **Portability:** Zero. Every new model or target requires rewriting the compiler.
  - **Engineering effort:** Extreme. 6-12 months of expert engineering for the initial compiler. Each new model architecture requires compiler updates.

  **Decision framework:** Prototype/startup → TFLite Micro (ship fast). Product with volume > 100K units → TVM (performance matters, amortize engineering). Single high-value product with extreme constraints → custom compiler (every byte counts).

  > **Napkin Math:** 20-layer keyword spotting model on Cortex-M4 at 168 MHz. TFLite Micro: 15ms inference, 95 KB code, 60 KB arena. microTVM: 11ms inference (-27%), 35 KB code (-63%), 52 KB arena (-13%). Custom compiler: 9ms inference (-40%), 12 KB code (-87%), 48 KB arena (-20%). Engineering cost: TFLite Micro = 1 week. TVM = 1 month. Custom = 6 months. At 1M units, the per-unit code size savings of TVM (60 KB less flash) could allow a cheaper MCU ($0.50 savings × 1M = $500K).

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)

  </details>

</details>

---

### 📡 Federated Learning on MCU Fleets

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> Federated Learning on Constrained Devices</b> · <code>training</code> <code>deployment</code></summary>

- **Interviewer:** "You manage 10,000 vibration sensors on factory equipment. Each sensor runs anomaly detection on a Cortex-M4. After 6 months, the model drifts because equipment ages and vibration patterns change. You want to update the model using data from the fleet — but you can't upload raw sensor data (proprietary manufacturing data, 100 TB total). Can you do federated learning on MCUs?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run standard federated averaging — each device trains locally and sends gradients." Standard FL requires backpropagation, which needs: (1) storing all activations for the backward pass (2× the forward-pass memory), (2) float32 arithmetic for stable gradients, (3) an optimizer state (momentum, Adam state). None of this fits on a Cortex-M4 with 256 KB SRAM.

  **Realistic Solution:** Standard federated learning is infeasible on MCUs. But there are MCU-compatible alternatives:

  **(1) Federated fine-tuning of the last layer only.** Freeze all layers except the final classification head. The head is a small fully-connected layer (e.g., 64 → 4 classes = 256 weights). Fine-tuning only the head requires: storing 64 activations from the penultimate layer (64 bytes INT8), computing gradients for 256 weights (1 KB in FP16), and running a simple SGD update (no momentum, no Adam). Total memory: ~2 KB. This fits easily on any MCU. Each device fine-tunes its head on local data, then sends the 256 updated weights (512 bytes in FP16) to the server via BLE/LoRa. The server averages the weights from all devices and broadcasts the updated head.

  **(2) Federated feature statistics.** Instead of training on-device, each device computes running statistics of its penultimate-layer activations (mean and covariance per class). These statistics (64-dim mean + 64×64 covariance = ~8 KB per class) are uploaded to the server. The server uses these to retrain the classification head in the cloud, then pushes the updated head to all devices via FOTA. No on-device training required.

  **(3) Federated distillation.** Each device runs inference on its local data and uploads the model's soft predictions (probability distributions, not raw data). The server trains a new model using these soft labels as supervision (knowledge distillation). The new model is pushed to devices via FOTA. Privacy-preserving: soft predictions leak less information than raw data.

  **Communication budget:** 10,000 devices × 512 bytes (head weights) = 5 MB per round. Via BLE to gateways: 5 MB / 60 KB/s per gateway / 5 gateways (2000 devices per gateway) = 17 seconds per round. Via LoRaWAN: 5 MB / 250 B/s multicast = 20,000 seconds ≈ 5.5 hours per round (LoRa is the bottleneck).

  > **Napkin Math:** Full FL (infeasible): backward pass memory = 2× forward = 400 KB. Doesn't fit in 256 KB. Optimizer state (Adam): 2× model size = 400 KB. Doesn't fit. Head-only FL: head weights = 256 × 2 bytes = 512 bytes. Gradient computation: 64 activations × 4 classes × 2 bytes = 512 bytes. SGD update: 512 bytes. Total: ~1.5 KB. Fits in 256 KB with 254.5 KB to spare. Accuracy recovery: head-only fine-tuning typically recovers 60-80% of the drift-induced accuracy loss. Full model retraining (in the cloud, using federated statistics) recovers 90-95%.

  📖 **Deep Dive:** [Volume I: Training](https://mlsysbook.ai/vol1/training.html)

  </details>

</details>

---

### 🏭 Predictive Maintenance System Design

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> Vibration-Based Predictive Maintenance</b> · <code>system-design</code> <code>sensor-pipeline</code></summary>

- **Interviewer:** "Design a vibration-based predictive maintenance system for 500 industrial motors using Cortex-M4 sensors. The system must detect bearing degradation 2 weeks before failure, run for 2 years on a battery, and cost less than $30 per sensor in BOM. Walk through the full system design."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Sample at high frequency, run a big model, transmit everything to the cloud." This blows the power budget, the memory budget, and the BOM budget simultaneously.

  **Realistic Solution:** Design from the constraints inward:

  **BOM budget ($30):** MCU (STM32L4, Cortex-M4, ultra-low-power): $4. Accelerometer (ADXL345, 3-axis, SPI): $3. BLE module (integrated in STM32WB or external nRF52): $5. CR2477 battery (1000 mAh, 3V): $2. PCB + passives + enclosure: $10. Flash (external, 2 MB, for data logging): $2. Antenna + connector: $2. Assembly: $2. **Total: $30.**

  **Power budget (2 years on 1000 mAh):** Battery energy: 1000 mAh × 3V = 3000 mWh. Over 2 years (17,520 hours): average power budget = 3000 / 17520 = **171 µW**. Quiescent (MCU deep sleep + RTC + voltage regulator): 5 µW. Available for sensing + inference + communication: **166 µW**.

  **Sensing strategy:** Bearing degradation manifests as characteristic vibration frequencies (BPFO, BPFI, BSF) that increase in amplitude over weeks. You don't need continuous monitoring — sample once every 10 minutes:

  (1) Wake from deep sleep (10ms, 30 mW = 0.3 mJ).
  (2) Read accelerometer at 3.2 kHz for 1 second (3200 samples × 3 axes × 2 bytes = 19.2 KB). Power: 5 mW × 1s = 5 mJ.
  (3) Compute 1024-point FFT on each axis (CMSIS-DSP, ~2ms, 30 mW = 0.06 mJ). Extract 20 spectral features (peak frequencies, RMS, kurtosis, crest factor).
  (4) Run anomaly detection model (small autoencoder, ~50K MACs, 10ms at 30 mW = 0.3 mJ). The model outputs a health score (0-1).
  (5) Log health score + timestamp to external flash (1ms, 5 mW = 0.005 mJ).
  (6) If health score < 0.7 (degradation detected): transmit via BLE (50ms, 15 mW = 0.75 mJ).
  (7) Sleep for 10 minutes.

  **Energy per cycle:** 0.3 + 5 + 0.06 + 0.3 + 0.005 = 5.665 mJ (normal). With BLE alert: +0.75 = 6.415 mJ. Cycles per day: 144 (every 10 min). Daily energy: 144 × 5.665 = 815.8 mJ = 0.227 mWh. Average power: 0.227 / 24 = **9.4 µW**. Add quiescent 5 µW = **14.4 µW**. Well within the 171 µW budget. Battery life: 3000 / (14.4 × 10⁻³) / 8760 = **23.8 years** theoretical. Derate 50% for temperature and aging: **~12 years**. Far exceeds the 2-year requirement.

  > **Napkin Math:** BOM: $30 ✓. Power: 14.4 µW average (budget: 171 µW) ✓. Memory: 19.2 KB sensor data + 2 KB FFT workspace + 15 KB tensor arena = 36.2 KB SRAM (budget: 256 KB) ✓. Flash: 50 KB model weights + 40 KB firmware = 90 KB (budget: 1 MB) ✓. Detection lead time: bearing degradation increases vibration RMS by 3-6 dB over 2-4 weeks. Sampling every 10 minutes captures the trend with 2-week advance warning. Cost per motor-year of monitoring: $30 / 12 years = $2.50/year. Unplanned motor failure cost: $5,000-50,000. ROI: 2000-20,000×.

  📖 **Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)

  </details>

</details>

---

### 🎤 Always-On Wake Word Detection

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> Sub-Milliwatt Always-On Wake Word Detection</b> · <code>power</code> <code>architecture</code> <code>sensor-pipeline</code></summary>

- **Interviewer:** "Design an always-on wake word detection system that consumes less than 1 mW total — including the microphone, ADC, feature extraction, and neural network inference. The system must detect 'Hey Device' with >95% accuracy and <5% false accept rate. What architectural choices make this possible?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use a Cortex-M4 at low clock speed." Even at the lowest clock speed (e.g., 4 MHz), a Cortex-M4 draws ~1-2 mW in active mode. With the microphone and ADC, you're already over budget before inference starts.

  **Realistic Solution:** Sub-milliwatt always-on detection requires a fundamentally different architecture than running a standard MCU continuously:

  **Tier 1: Analog front-end (~50 µW)**
  Use an analog MEMS microphone with built-in analog comparator (e.g., Vesper VM3011, ~10 µW). The comparator detects sound above a threshold (voice activity detection) without any digital processing. When silence: the entire digital system is powered off. Only the analog comparator draws power. This rejects ~90% of time (silence) at ~10 µW.

  **Tier 2: Ultra-low-power digital feature extraction (~200 µW)**
  When the analog comparator detects sound, it wakes a dedicated ultra-low-power DSP (e.g., Syntiant NDP120, ~150 µW, or a custom ASIC). This DSP runs a fixed-function Mel spectrogram pipeline: 16 kHz ADC → 512-point FFT → 40 Mel filters → log compression. The DSP is hardwired for this pipeline — no instruction fetch, no cache, no branch prediction. Power: ~200 µW including the ADC.

  **Tier 3: Tiny neural network on the DSP (~300 µW)**
  The DSP runs a small keyword spotting model: 2-3 depthwise separable convolutional layers + a fully connected classifier. Model size: ~20 KB weights. ~100K MACs per inference. The DSP executes this at ~500 MMAC/s/W efficiency (vs ~10 MMAC/s/W for a general-purpose Cortex-M4). Power for inference: ~200 µW.

  **Tier 4: Main MCU wake (only on detection)**
  When the DSP detects the wake word with >80% confidence, it wakes the main Cortex-M4 MCU via a GPIO interrupt. The MCU runs a larger, more accurate confirmation model (~500K MACs, 30ms, 30 mW) to verify the detection and reduce false accepts. The MCU is active for <100ms per true detection — negligible average power.

  **Total always-on power budget:**
  - Analog comparator: 10 µW (always on)
  - DSP (active 10% of time — only when sound detected): 200 µW × 0.1 = 20 µW average
  - DSP inference (active 5% of time — only on voice-like sounds): 300 µW × 0.05 = 15 µW average
  - MCU (active 0.01% of time — only on confirmed wake words): 30 mW × 0.0001 = 3 µW average

  **Total: ~48 µW average** — well under the 1 mW budget. On a CR2032 (675 mWh): 675 / 0.048 = 14,063 hours = **1.6 years**.

  > **Napkin Math:** Cortex-M4 always-on approach: 2 mW MCU + 0.5 mW mic + 0.2 mW ADC = 2.7 mW. Battery life: 675 / 2.7 = 250 hours = 10 days. Tiered approach: 48 µW average. Battery life: 14,063 hours = 1.6 years. Improvement: **56×** longer battery life. The tiered architecture is the only way to achieve always-on detection on a coin cell. This is why products like Amazon Echo Buds and Google Pixel Buds use dedicated always-on DSPs — not the main application processor.

  > **Key Equation:** $P_{\text{total}} = P_{\text{analog}} + \sum_{i} P_{\text{tier}_i} \times \delta_i$, where $\delta_i$ is the duty cycle of tier $i$

  📖 **Deep Dive:** [Volume I: TinyML](https://mlsysbook.ai/vol1/tinyml.html)

  </details>

</details>

---

### ⚙️ Hardware-Software Co-Design

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> Co-Designing a TinyML Accelerator</b> · <code>hw-acceleration</code> <code>architecture</code></summary>

- **Interviewer:** "Your company is designing a custom TinyML accelerator to be integrated alongside a Cortex-M4 core on the same die. The accelerator must run INT8 inference 10× faster than CMSIS-NN on the M4 while adding less than 0.5 mm² of silicon area (in 28nm process). What hardware blocks do you include, and what do you leave to the M4?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Build a small GPU with many cores." GPUs are designed for high throughput with high power — the wrong trade-off for TinyML. A tiny GPU would be power-hungry and area-inefficient for the small, sequential workloads typical of MCU models.

  **Realistic Solution:** Design a specialized dataflow accelerator optimized for the specific operations in TinyML models:

  **What to accelerate (high-value operations):**

  (1) **MAC array** — a 16×16 systolic array of INT8 multiply-accumulators. Each PE performs one INT8×INT8→INT32 MAC per cycle. Total: 256 MACs/cycle. At 168 MHz: 43 GMAC/s — vs 0.336 GMAC/s for CMSIS-NN on the M4. Speedup: **128×** for pure MACs. Area: ~0.1 mm² in 28nm (each INT8 MAC is ~200 µm²).

  (2) **On-chip activation SRAM** — 64 KB of tightly-coupled SRAM for activation double-buffering. While the MAC array processes one tile, the DMA loads the next tile from main SRAM. This hides memory latency. Area: ~0.15 mm² in 28nm.

  (3) **Requantization unit** — a hardwired pipeline that performs the INT32→INT8 requantization (multiply by fixed-point scale, shift, clamp) in 1 cycle per element. This is 5× faster than the M4's software requantization. Area: ~0.01 mm².

  (4) **Activation function LUT** — a 256-entry lookup table for ReLU, ReLU6, and sigmoid approximation. Applied to each output element in 1 cycle. Area: negligible.

  **What to leave on the M4 (low-value or irregular operations):**

  (1) **Control flow** — the M4 orchestrates the accelerator: configures DMA, sets up layer parameters, handles interrupts. The accelerator has no instruction fetch — it's purely data-driven.

  (2) **Non-standard operators** — any operator not in the accelerator's fixed-function pipeline (e.g., resize, transpose, custom activations) falls back to CMSIS-NN on the M4. Typically <5% of inference time.

  (3) **Pre/post-processing** — feature extraction (FFT, Mel filters), sensor data handling, BLE communication. These are irregular, branch-heavy workloads that don't benefit from a systolic array.

  **Area budget:** MAC array (0.1) + SRAM (0.15) + requantization (0.01) + control logic (0.05) + DMA engine (0.05) + I/O (0.02) = **0.38 mm²** < 0.5 mm² budget.

  **Performance:** A typical 20-layer TinyML model with 5M MACs: MAC array time = 5M / 43G = 0.12ms. Add DMA overhead (20%) and M4 control (10%): ~0.16ms total. CMSIS-NN on M4: 5M / 0.336G = 14.9ms. **Speedup: 93×** — exceeds the 10× target.

  **Power:** The MAC array at 168 MHz draws ~5 mW (INT8 MACs are extremely energy-efficient). Total accelerator power: ~10 mW. For a 0.16ms inference: energy = 10 mW × 0.16ms = 1.6 µJ per inference. CMSIS-NN: 50 mW × 14.9ms = 745 µJ. **Energy savings: 466×**.

  > **Napkin Math:** Area: 0.38 mm² in 28nm. For context: a Cortex-M4 core is ~0.5-1.0 mm² in 28nm. The accelerator is smaller than the CPU it augments. Speedup: 93× over CMSIS-NN. Energy: 466× less per inference. This is why companies like Syntiant, Eta Compute, and GreenWaves are building dedicated TinyML accelerators — the efficiency gains are orders of magnitude, not percentages.

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)

  </details>

</details>

---

### 📊 Streaming Anomaly Detection

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> Anomaly Detection on Streaming Sensor Data with Limited Memory</b> · <code>memory-layout</code> <code>sensor-pipeline</code></summary>

- **Interviewer:** "Your vibration sensor streams data at 3.2 kHz (3 axes × 16-bit = 19.2 KB/s). You need to detect anomalies in real-time on a Cortex-M4 with 256 KB SRAM. The challenge: your autoencoder model processes 1-second windows (3200 samples × 3 axes × 2 bytes = 19.2 KB per window), but you also need to maintain a running baseline of 'normal' vibration patterns for comparison. How do you fit the streaming pipeline, the model, and the baseline statistics in 256 KB?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Store the last 60 seconds of raw data as the baseline." 60 seconds × 19.2 KB/s = 1.15 MB. Doesn't fit in 256 KB.

  **Realistic Solution:** Use compressed statistical representations instead of raw data for the baseline:

  **Memory budget:**

  | Component | Size | Purpose |
  |-----------|------|---------|
  | Firmware + stack | 48 KB | Application code, ISR handlers |
  | DMA double-buffer | 40 KB | 2 × 19.2 KB for ping-pong sensor capture |
  | FFT workspace | 16 KB | 1024-point complex FFT (in-place) |
  | Feature vector | 1 KB | 128 spectral features (64 per axis pair) |
  | Tensor arena | 80 KB | Autoencoder model activations |
  | Baseline statistics | 8 KB | Running mean + variance of 128 features × 2 (mean, var) × 4 bytes × 8 frequency bands |
  | Model weights (SRAM mirror) | 0 KB | Weights stay in flash (XIP) |
  | Anomaly log | 4 KB | Circular buffer of last 100 anomaly events |
  | Misc (BLE, timers) | 8 KB | Communication buffers |
  | **Headroom** | **51 KB** | Available for future features |

  **Streaming pipeline:**

  (1) **DMA captures** 1 second of data into Buffer A (19.2 KB) while the CPU processes Buffer B.

  (2) **Feature extraction:** Compute 1024-point FFT on each axis (CMSIS-DSP `arm_rfft_q15`, in-place, 2ms per axis). Extract 128 spectral features: power in 8 frequency bands × 3 axes, plus cross-axis correlations, kurtosis, and crest factor. Compress 19.2 KB of raw data into a 1 KB feature vector.

  (3) **Baseline update:** Maintain an exponential moving average (EMA) of the feature vector: $\mu_t = \alpha \times f_t + (1 - \alpha) \times \mu_{t-1}$, with $\alpha = 0.01$ (slow adaptation). Similarly for variance. This captures the "normal" vibration signature in 8 KB — representing thousands of seconds of history in a compressed form.

  (4) **Anomaly scoring:** Two methods, run in parallel:
  - **Statistical:** Diagonal Mahalanobis distance (weighted Z-score per feature) between the current feature vector and the baseline. If $d > 3\sigma$: flag anomaly. Cost: 128 multiplies + 128 adds = 256 operations. Time: <1µs. (A full Mahalanobis with covariance matrix would require O(N²) = 16,384 MACs — use diagonal approximation on MCUs.)
  - **Autoencoder:** Feed the feature vector into a small autoencoder (128→32→128, ~8K parameters). Reconstruction error above a threshold indicates an anomaly the statistical method might miss (novel failure modes). Time: ~5ms.

  (5) **Anomaly logging:** When an anomaly is detected, log the timestamp, anomaly score, and top-5 contributing features to the circular buffer in SRAM. Periodically flush to external flash for long-term storage.

  > **Napkin Math:** Raw baseline (60s): 1.15 MB — doesn't fit. EMA baseline: 8 KB — fits with room to spare. Information loss: the EMA captures the mean and variance of each feature, which is sufficient for detecting gradual degradation (bearing wear) and sudden changes (impact events). It cannot detect subtle temporal patterns (e.g., a specific sequence of vibration events) — that's what the autoencoder is for. Total SRAM: 205 KB used / 256 KB available = 80% utilization. Processing time per window: FFT (6ms) + features (2ms) + baseline update (<0.1ms) + statistical score (<0.1ms) + autoencoder (5ms) = **13.2ms** per 1-second window. CPU utilization: 1.3%.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)

  </details>

</details>

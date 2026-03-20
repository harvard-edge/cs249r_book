# Visual Architecture Debugging

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> · <a href="../edge/README.md">🤖 Edge</a> · <a href="../mobile/README.md">📱 Mobile</a> · <b>🔬 TinyML</b>
</div>

---

*Can you spot the bottleneck in a TinyML system diagram?*

TinyML system architecture diagrams with hidden bottlenecks.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/tinyml/04_visual_debugging.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### In-Place Update with No Rollback

**Common Mistake:** "The CRC check catches corruption." The CRC check runs *after* the write completes. If power is lost during step 4 (writing to the application region), the old firmware is partially overwritten with the new firmware. The device now has neither a valid old firmware nor a valid new firmware. It's bricked.

The fundamental flaw: **writing directly to the active application region**. The 508 KB of free space is wasted — it should be the update staging area.

**The Fix:** A/B partitioning:

(1) **Flash layout:** Bootloader (16 KB) + Boot config (4 KB) + Slot A: application + model (490 KB) + Slot B: staging area (490 KB) + Persistent data (24 KB).

(2) **Update process:** Download new firmware to Slot B (the inactive slot). The active Slot A continues running throughout. After download completes: verify CRC of Slot B. If valid: update boot config to point to Slot B, reboot. If invalid: discard Slot B, request retransmit. Slot A is untouched.

(3) **Power loss scenarios:** Power lost during download → Slot B is partially written, Slot A is untouched → device reboots into Slot A, re-requests update. Power lost during boot config write → boot config has a sequence number; the bootloader picks the slot with the highest valid sequence number. Power lost after reboot into Slot B → if Slot B fails self-test, watchdog fires, bootloader reverts to Slot A.

(4) **Cost:** You lose 490 KB of flash for the staging area. But 490 KB per slot is still enough for most TinyML applications (model + app < 400 KB). The alternative — sending a technician to 200 farms to reflash bricked sensors — costs $50 × 200 = $10,000 per incident.

**📖 Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### Stack Overflow into the Tensor Arena

**Common Mistake:** "The tensor arena is too small — increase it." The arena has 6 KB of headroom beyond the simulation's reported peak. The problem is not the arena size.

The stack grows downward on ARM Cortex-M. The 8 KB stack sits directly above the tensor arena in the memory map. During inference, nested function calls (convolution kernels calling CMSIS-NN helpers calling SIMD intrinsics) plus interrupt handlers (DMA half-transfer, DMA complete, SysTick) can push the stack well beyond 8 KB. A deep interrupt nesting scenario: SysTick fires during a DMA interrupt during a convolution kernel — 3 levels of context saving at ~100 bytes each, plus local variables. The stack overflows downward into the top of the tensor arena, silently corrupting activation data. The model produces garbage outputs or hard-faults — but only when the interrupt timing aligns with deep call stacks, hence the randomness.

**The Fix:** (1) Place a stack canary (known pattern, e.g., 0xDEADBEEF) at the bottom of the stack region. Check it periodically — if corrupted, the stack overflowed. (2) Use the MPU (Memory Protection Unit) to mark the boundary between stack and arena as no-access — a stack overflow triggers a MemManage fault instead of silent corruption. (3) Increase the stack to 16 KB and reduce the arena to 188 KB (still fits the 190 KB peak? No — now you need to optimize the model too). (4) Move the stack to the top of SRAM and the arena to the bottom, so stack overflow hits the end of SRAM and faults immediately instead of corrupting data.

**📖 Deep Dive:** [Volume I: TinyML](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### BLE Advertisement Power is Massively Underestimated

**Common Mistake:** "The model inference (2.5 mJ) dominates the power budget — optimize the model." Inference is only 58% of the per-cycle energy. The real killer is hiding in the BLE advertisement.

The 20ms × 40 mW = 0.8 mJ estimate for BLE assumes a single advertisement packet sent and acknowledged immediately. In reality, BLE advertising involves:

(1) **Three advertisement channels** — BLE advertises on channels 37, 38, and 39 sequentially. Each transmission: ~1ms TX + 1ms RX window = 2ms × 3 channels = 6ms minimum.

(2) **No guaranteed reception** — the gateway may not be listening on the right channel at the right time. The device must advertise repeatedly. Typical BLE advertising interval: 100ms. If the gateway scans every 1 second, the device advertises ~10 times before being heard. Energy: 10 × 6ms × 40 mW = 2.4 mJ — 3× the estimate.

(3) **Connection overhead** — if the device establishes a BLE connection (not just advertising), the connection setup takes 50-200ms at 15-30 mW. Add data exchange: 5-10ms. Total: 200ms × 20 mW = 4 mJ.

(4) **Retransmissions** — in a noisy RF environment (factory floor with WiFi, other BLE devices), packet loss rate can be 10-30%. Each retry adds another advertising cycle.

Realistic BLE energy per wake cycle: **5-10 mJ** — not 0.8 mJ. This makes BLE the dominant power consumer (50-70% of total), not inference. Revised total: 0.6 + 0.3 + 0.125 + 2.5 + 8 = 11.5 mJ per cycle. Average power: 192 µW. Battery life: 675 mWh / 0.192 mW = 3,516 hours = **146 days ≈ 5 months**. With RF retransmissions in a noisy factory: ~3 months. Mystery solved.

**The Fix:** (1) Don't transmit every cycle. Buffer results locally and transmit every 10th cycle (every 10 minutes). BLE energy amortized: 8 mJ / 10 = 0.8 mJ per cycle. (2) Use BLE advertising-only mode (no connection) with encoded data in the advertisement payload — eliminates connection overhead. (3) Use a lower TX power (-20 dBm instead of 0 dBm) if the gateway is nearby — 4× less TX energy. (4) Only transmit when an anomaly is detected — most cycles produce "normal" results that don't need reporting.

**📖 Deep Dive:** [Volume I: TinyML](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### Quantization Range Collapse on Discriminative Features

**Common Mistake:** "The model isn't expressive enough to distinguish left from right swipes." The FP32 model achieves 95% accuracy — it can clearly distinguish them. The problem is introduced by quantization.

The calibration dataset was recorded by a single user in controlled conditions. This user's "swipe left" and "swipe right" gestures have large, distinct accelerometer signatures — peak acceleration of ±8g. The quantization tool sets the activation range to [-8, 8] for the relevant layers.

Real users perform gestures with much more variation. Some users swipe gently (±2g), some swipe at angles (distributing energy across axes). The discriminative features between left and right swipes are subtle differences in the acceleration *profile* — the shape of the curve, not just the peak magnitude. These subtle differences live in the range [-0.5, 0.5] within the activation space.

With per-tensor quantization calibrated on the lab data: step size = 16 / 255 = 0.063. The discriminative features spanning [-0.5, 0.5] are quantized to only 1.0 / 0.063 = **16 bins**. In FP32, these features had thousands of distinct values. In INT8 with this calibration, left and right swipes map to nearly identical quantized activations — the model literally cannot tell them apart.

**The Fix:** (1) **Diverse calibration data** — include gestures from 20+ users with varying speeds, orientations, and strengths. This widens the activation distribution and sets more representative quantization ranges. (2) **Per-channel quantization** — different channels capture different features. The channel that discriminates left/right may have a narrow range [-0.5, 0.5] while other channels span [-8, 8]. Per-channel quantization gives each channel its own range. (3) **Quantization-aware training (QAT)** — the model learns to make discriminative features robust to quantization noise during training. (4) **Confusion matrix testing** — never rely on aggregate accuracy. Test per-class and per-class-pair metrics after quantization.

**📖 Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### The Blocking ISR Tax

**Common Mistake:** "The processor isn't fast enough, increase the clock speed."

The design flaw is executing a slow, blocking I2C read inside an **Interrupt Service Routine (ISR)**.
I2C is a slow protocol (e.g., 100 kHz or 400 kHz). Reading a few registers can easily take 5ms.
When an interrupt fires, it completely halts the main RTOS thread (where your ML model lives).

In a 50ms window, the 10ms timer fires 5 times.
5 interrupts × 5ms per ISR = **25ms of stolen CPU time**.
Your ML model needs 40ms, but it only has 25ms of free time left in the window (50 - 25 = 25ms). The inference cannot complete before the next deadline, leading to dropped frames.

**The Fix:** Never block inside an ISR. The ISR should simply set a flag or trigger a DMA transfer.
1. Use **I2C with DMA** so the hardware fetches the sensor data in the background without CPU intervention.
2. If DMA isn't possible, the ISR should immediately defer the I2C read to a low-priority background thread, allowing the high-priority ML inference to complete uninterrupted.

**📖 Deep Dive:** [Volume I: Benchmarking](https://harvard-edge.github.io/cs249r_book_dev/contents/benchmarking/benchmarking.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### Memory Fragmentation

**Common Mistake:** "TFLite Micro has a memory leak." TFLite Micro's allocator is static and runs once at initialization, so it cannot leak at runtime.

The bottleneck is **Memory Fragmentation**. TFLite Micro uses a greedy memory planner to map tensor lifetimes to specific addresses in the Arena.
Even though your total *peak* usage is 85 KB, those tensors are allocated and freed in different patterns based on the graph topology (e.g., skip connections in a ResNet keep older tensors alive while new ones are allocated).

This leaves "holes" in the memory. If the next layer requires a 30 KB contiguous block for its output, but the free space is fragmented into two 15 KB holes, the allocation fails—even though there is technically enough total free bytes.

**The Fix:**
1. **Increase the Arena Size:** Add a 10-20% safety margin above the theoretical peak to accommodate fragmentation.
2. **In-Place Operations:** Use a framework compiler (like TVM or heavily optimized TFLite) that rewrites the graph to use in-place operations (e.g., Activation functions overwriting their input buffers) to minimize allocation churn.
3. **Graph Reordering:** Change the topology of the network to avoid long-lived skip connections that pin memory blocks and worsen fragmentation.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### Interrupt Priority Inversion

**Common Mistake:** "23ms is too close to the 25.6ms deadline — reduce the model size." The model has 2.6ms of margin, which should be sufficient on a deterministic bare-metal system. The problem is that the margin is being stolen.

The UART debug logging interrupt and the DMA complete interrupt are both at Priority 0 (highest). On Cortex-M, interrupts at the same priority level cannot preempt each other — they execute in arrival order. When UART TX fires during inference (every 100ms, taking 2ms), it doesn't preempt the inference ISR. But the BLE stack at Priority 1 fires every 50ms and takes 8ms. If a BLE event fires just before the DMA complete interrupt, the BLE handler runs for 8ms before the inference ISR can start. Now inference starts 8ms late: 8ms + 23ms = 31ms > 25.6ms deadline. Samples are dropped.

Worse: the UART logging at Priority 0 means that if a UART interrupt fires during inference, it queues and runs immediately after — but if inference is running *as* an ISR at the same priority, the UART blocks until inference completes. The UART TX buffer overflows, triggering an error interrupt that the system doesn't handle, causing a hard fault.

**The Fix:** (1) Never run inference inside an ISR. Use a flag-based approach: DMA ISR sets a flag, main loop polls the flag and runs inference at base level. (2) Set interrupt priorities correctly: DMA complete = Priority 0, BLE = Priority 2, UART = Priority 3. DMA can preempt everything. (3) Remove debug UART logging from production firmware — it's the most common source of timing bugs in embedded systems. (4) Use a deferred processing pattern: ISR copies buffer pointer to a queue, main loop processes the queue.

**📖 Deep Dive:** [Volume I: TinyML](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### Independent Clock Dividers with Rounding Error

**Common Mistake:** "Both peripherals use the same crystal, so they're perfectly synchronized." They use the same crystal as a *reference*, but the SPI and I2S peripherals derive their clocks through independent divider chains with different rounding errors.

The 32 MHz crystal is divided differently for each peripheral:

**SPI clock for accelerometer:** The accelerometer expects a 1 MHz SPI clock. 32 MHz / 32 = 1.000 MHz exactly. The accelerometer's internal ODR (Output Data Rate) is set to 1 kHz, derived from its own internal oscillator — not the SPI clock. The accelerometer's internal oscillator has ±1% tolerance. Actual ODR: 1000 ± 10 Hz.

**I2S clock for microphone:** 16 kHz sample rate requires a bit clock of 16 kHz × 16 bits × 2 channels = 512 kHz. 32 MHz / 62.5 = 512 kHz — but the divider is integer-only. Nearest: 32 MHz / 62 = 516.13 kHz, or 32 MHz / 63 = 507.94 kHz. With divider = 63: actual sample rate = 507.94 / 32 = 15.873 kHz. 1600 samples at 15.873 kHz = 100.8ms — not 100.0ms.

**Combined drift:** Accel buffer fills in 100ms ± 1% = 99-101ms. Audio buffer fills in 100.8ms consistently. After 10 minutes (6,000 cycles): accel is ahead by 6000 × 0.8ms = 4.8 seconds of cumulative drift. The synchronization barrier hides this — it just waits for both — but the *data* is misaligned. The accel window and audio window no longer overlap in time. The model receives accelerometer data from time [0, 100ms] fused with audio from time [0.8ms, 101.6ms]. After an hour, the misalignment grows to seconds.

**The Fix:** (1) **Timestamp each sample** using a single monotonic timer (e.g., the RTC or a dedicated hardware timer). Align data by timestamp, not by buffer-full events. (2) **Resample one stream** to match the other's actual rate. Use linear interpolation to resample the 15.873 kHz audio to exactly 16× the accelerometer's actual rate. (3) **Use a PLL-based I2S clock** (available on nRF5340) to generate an exact 512 kHz bit clock, eliminating the divider rounding error. (4) **Periodic resynchronization** — every N cycles, discard partial buffers and restart both DMA channels simultaneously.

**📖 Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### The Duty Cycle Misses 92% of Events

**Common Mistake:** "The model needs retraining — it's missing detections." The model works fine during the 5-second active window. The problem is that the device is asleep for 55 out of every 60 seconds. A bird call that happens during the 55-second sleep window is never recorded, never classified, and never detected. The ~8% detection rate matches the 8.3% duty cycle almost exactly — the model isn't missing calls, the microphone is.

This is a fundamental design flaw: **uniform duty cycling is wrong for event-driven detection**. The device doesn't know when a bird will call, so sleeping for fixed intervals guarantees missing most events.

**The Fix:** A two-tier wake architecture:

**(1) Always-on analog wake detector:** Use a low-power analog comparator (available on most Cortex-M4 MCUs, ~1-5 µW) connected to the microphone's analog output. Set a threshold just above the ambient noise floor. When a sound exceeds the threshold, the comparator triggers an interrupt that wakes the MCU from deep sleep. The MCU then records and classifies the sound. If it's a bird call: log it. If it's noise (wind, rain): go back to sleep.

**(2) Always-on digital wake detector:** Use a dedicated ultra-low-power audio processor (e.g., Syntiant NDP101, ~140 µW) that runs a tiny neural network (~10 KB) to detect "bird-like" sounds. When triggered, it wakes the main MCU for full classification. False wake rate: ~5% (acceptable — the main MCU goes back to sleep in 100ms on false wakes).

**Power comparison:**
- Fixed duty cycle (8.3%): average power = 0.083 × 50 mW + 0.917 × 0.01 mW = 4.16 mW. Detection rate: 8.3%.
- Analog comparator wake: comparator always on at 5 µW. Assume 20 wake events/hour (5 real + 15 false). Each wake: 2 seconds active at 50 mW = 100 mJ. Average: 20 × 100 mJ / 3600s = 0.56 mW + 0.005 mW = 0.56 mW. Detection rate: ~95% (misses only calls below the comparator threshold).
- Digital wake (NDP101): 0.14 mW always-on. 10 wakes/hour (5 real + 5 false). Average: 10 × 100 mJ / 3600s = 0.28 mW + 0.14 mW = 0.42 mW. Detection rate: ~98%.

The event-driven approach uses **7-10× less power** than fixed duty cycling AND detects **12× more events**.

**📖 Deep Dive:** [Volume I: TinyML](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-gold?style=flat-square" alt="Level 3" align="center"> The Memory-Mapped Weight Corruption</b> · <code>flash-memory</code></summary>

- **Interviewer:** "You deploy a keyword spotting model on an ESP32 microcontroller. The model weights are marked as `const` in C so they are compiled directly into the read-only Flash memory. During a rigorous stress test, a junior engineer accidentally writes a bug in the audio ring-buffer code that causes a buffer overflow. Surprisingly, the bug doesn't cause a HardFault; instead, the neural network permanently forgets how to recognize the wake word until the device is fully rebooted. Why?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming that because a variable is marked `const` in C, the underlying physical memory is fundamentally un-writable by the CPU."

  **Realistic Solution:** Your microcontroller likely uses a unified memory map where Flash memory and SRAM share the same address space. While Flash is technically 'read-only' for standard operations, many MCUs have memory-mapped control registers that, if accidentally hit by an out-of-bounds pointer write, can initiate a Flash erase/write cycle, or alter the cache/MMU (Memory Management Unit) mapping. The buffer overflow from the audio code silently marched a rogue pointer straight into the memory-mapped region of the weights, corrupting the physical data the NPU was reading.

  > **Napkin Math:** If your audio buffer is `int16_t audio[1000]` starting at `0x20000000`, and your weights are mapped at `0x20010000`. A rogue loop like `for(int i=0; i<100000; i++) audio[i] = 0;` will write 0s continuously upward in memory. Once it reaches `0x20010000`, it begins overwriting the physical addresses where the NPU expects to find its weights. The math becomes garbage.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> 🚨 Reveal the Bottleneck</b></summary>

### The SPI Bus Latency Choke

**Common Mistake:** "The model is too big for the CPU." The CPU can handle the math easily; it's starving for data.

The bottleneck is the **SPI bus physics**. Execute-In-Place (XIP) allows the CPU to treat external flash like normal memory, but it masks the severe hardware latency.
While the CPU runs at 240 MHz (one cycle is ~4 ns), a standard SPI flash chip might operate at 40 MHz over a serial bus. Fetching a single 8-bit weight over SPI takes dozens or hundreds of CPU cycles.

During a Convolution layer, the CPU tries to fetch millions of weights. Because they are located across the slow SPI bus, the CPU spends 95% of its time stalled, waiting for the flash chip to return data.

**The Fix:**
1. **Quad-SPI (QSPI) / Octal-SPI:** Upgrade the physical bus to move 4 or 8 bits per clock cycle instead of 1.
2. **Flash Cache:** Ensure the MCU's instruction/data cache for the XIP region is enabled and properly sized so loops reusing weights don't hit the physical flash.
3. **Model Pruning:** Prune the model to < 400 KB so the weights can be copied from Flash into the ultra-fast internal SRAM at boot time, avoiding the SPI bus entirely during inference.

**📖 Deep Dive:** [Volume I: HW Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)
</details>

# Round 3: Operations & Deployment 🚀

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

Deploying a model to a single MCU is an engineering exercise. Deploying it to 10,000 battery-powered sensors in the field — and keeping them running for years — is an operations problem. This round tests firmware updates over constrained links, SRAM overflow triage, bootloader safety, real-time guarantees, security against physical access, fleet management, and energy budgeting for devices that must outlive their batteries.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/tinyml/03_TinyML_Ops_and_Deployment.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

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

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> FOTA for Battery-Powered Sensor Fleets</b> · <code>deployment</code> <code>power</code></summary>

**Interviewer:** "You're rolling out a FOTA update to 10,000 battery-powered vibration sensors. Each sensor has a CR2032 coin cell (225 mAh, 3V) and communicates via BLE 5.3 (nRF5340, 128 MHz). The new model binary is 80 KB. Your field engineer says 'just push the update to all devices tonight.' What could go wrong?"

**Common Mistake:** "BLE is fast enough — 80 KB takes a few seconds." This ignores the energy cost of the radio, which is the dominant power consumer on battery-powered MCUs.

**Realistic Solution:** The radio is the most power-hungry peripheral on a coin-cell device. BLE 5.3 on nRF5340: TX power ~15 mW, RX power ~12 mW. At BLE's practical throughput of ~60 KB/s (with connection overhead): transfer time = 80 KB / 60 KB/s = 1.33 seconds. Energy for transfer: ~15 mW × 1.33s = 20 mJ. But that's just the data transfer. The full FOTA process includes:

(1) **Connection setup:** BLE advertising + connection: ~50ms at 12 mW = 0.6 mJ.
(2) **Data transfer:** 80 KB at 60 KB/s: 1.33s at 15 mW = 20 mJ.
(3) **Flash write:** Writing 80 KB to flash: 80 KB / 4 KB pages = 20 page erases + writes. Each page erase: ~85ms at 10 mW. Total: 1.7s at 10 mW = 17 mJ.
(4) **CRC verification + test inference:** ~100ms at 30 mW = 3 mJ.
(5) **Reboot + bootloader swap:** ~200ms at 30 mW = 6 mJ.

**Total FOTA energy: ~47 mJ.** On a CR2032 (675 mWh = 2,430 J): FOTA consumes 47/2430 = **0.0019%** of battery. Seems negligible — but the risk is different:

**The real danger:** If the update fails mid-transfer (BLE disconnect, interference), the device retries. With 10,000 devices in a warehouse, BLE congestion causes ~5% failure rate per attempt. Each retry costs another 47 mJ. After 10 retries: 470 mJ. If the update loop doesn't have a retry limit, a device in a dead zone burns its battery on futile radio attempts. Worse: if the update bricks the device (bad bootloader), you can't recover a coin-cell sensor in a ceiling — it's physically inaccessible.

**Design:** (1) Maximum 3 FOTA retries, then back off for 24 hours. (2) Stagger updates: 500 devices per hour to avoid BLE congestion. (3) Battery voltage check before FOTA — abort if below 2.5V (insufficient energy to complete safely). (4) A/B partitioning with watchdog rollback — never brick a device.

> **Napkin Math:** FOTA energy: 47 mJ per attempt. Battery: 2,430 J. Max safe retries before 1% battery impact: 2430 × 0.01 / 0.047 = 517 retries. With 3-retry limit: 141 mJ worst case = 0.006% of battery. Fleet update: 10,000 devices × 80 KB multicast over BLE mesh = ~2 minutes for data distribution + 10 minutes for staggered flash writes. Total fleet update: ~15 minutes with <0.01% battery impact per device.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Tensor Arena Exceeds SRAM</b> · <code>deployment</code> <code>memory</code></summary>

**Interviewer:** "Your model fits in flash (weights: 150 KB, flash: 1 MB) but the tensor arena needs 280 KB and your Cortex-M4 has 256 KB SRAM total. After firmware (40 KB) and stack (8 KB), you have 208 KB available. The model was validated in simulation. Why does it fail on hardware, and what are your options?"

**Common Mistake:** "The simulation was wrong." The simulation correctly reports the model's activation memory — the issue is that simulation doesn't account for the MCU's other SRAM consumers.

**Realistic Solution:** The 280 KB tensor arena is the model's peak activation memory as reported by TFLite Micro's allocation analysis. But on real hardware, SRAM is shared:

**SRAM budget:** 256 KB total - 40 KB firmware (.bss + .data sections) - 8 KB stack - 4 KB DMA buffers (sensor, UART) - 2 KB interrupt vector table = **202 KB available** for the tensor arena. The model needs 280 KB. Gap: 78 KB.

**Options in order of effort:**

(1) **Reduce firmware footprint** (effort: hours). Audit .bss and .data sections with `arm-none-eabi-size`. Remove unused libraries, reduce logging buffers, eliminate debug strings. Typical savings: 10-20 KB. New available: ~220 KB. Still short by 60 KB.

(2) **External SRAM/PSRAM** (effort: days). Some Cortex-M4 boards (STM32F4 with FSMC) support external SRAM via the Flexible Static Memory Controller. Add a 512 KB SRAM chip ($1-2). Place the tensor arena in external SRAM. Penalty: external SRAM runs at ~1/3 the speed of internal SRAM (bus wait states). Inference slows by 2-3×. If your latency budget allows this, it's the easiest fix.

(3) **Upgrade to Cortex-M7** (effort: weeks). STM32H7 has 512 KB SRAM. The model fits with 232 KB headroom. Cost: ~$3 more per unit. If you're designing the PCB, this is the right long-term fix. If the PCB is already manufactured: not an option.

(4) **Model surgery** (effort: weeks). Use MCUNet's patch-based inference to reduce peak activation memory. Process the feature map in spatial patches instead of all at once. Can reduce peak SRAM by 3-4× at the cost of ~10% more compute. 280 KB / 3 = 93 KB peak — fits easily.

> **Napkin Math:** Gap: 280 - 202 = 78 KB. Firmware optimization: saves ~15 KB → gap = 63 KB. External SRAM: eliminates gap but 2-3× latency hit. If baseline inference = 30ms → external SRAM = 60-90ms. Cortex-M7 upgrade: $3/unit × 10,000 units = $30,000 NRE. Patch-based inference: 280 KB → ~93 KB peak, +10% compute = 33ms. Best option depends on constraints: if latency-critical → patch-based. If cost-critical → external SRAM. If designing new hardware → M7.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> Bootloader A/B Firmware Partitioning</b> · <code>deployment</code> <code>reliability</code></summary>

**Interviewer:** "Design the flash memory layout for a Cortex-M4 with 1 MB flash that supports A/B firmware partitioning with rollback. The firmware includes a bootloader, application code, and a TFLite Micro model. The device is deployed in a location where physical access costs $500 per visit."

**Common Mistake:** "Split flash 50/50: 512 KB for slot A, 512 KB for slot B." This wastes flash and doesn't account for the bootloader, configuration, or wear leveling.

**Realistic Solution:** Design the flash layout with every sector accounted for:

**Flash map (1 MB = 1024 KB):**

| Region | Size | Address | Purpose |
|--------|------|---------|---------|
| Bootloader | 32 KB | 0x0800_0000 | Immutable first-stage bootloader. Never updated OTA. |
| Boot config | 4 KB | 0x0800_8000 | Active slot pointer, boot count, rollback flag. Wear-leveled. |
| Slot A (firmware + model) | 480 KB | 0x0800_9000 | Application code (~120 KB) + model weights (~350 KB) |
| Slot B (firmware + model) | 480 KB | 0x0808_7000 | Mirror of Slot A for updates |
| Persistent storage | 28 KB | 0x080F_9000 | Calibration data, drift logs, device ID. Survives updates. |

**Boot sequence:**
1. Bootloader reads boot config: which slot is active, boot count, rollback flag.
2. If boot count > 3 (three consecutive failed boots): set rollback flag, switch to other slot, reset boot count.
3. Jump to active slot. Application increments boot count at start, clears it after successful self-test (inference on golden reference input).
4. If self-test fails: reboot (boot count increments → eventually triggers rollback).

**Update sequence:**
1. Download new firmware+model to inactive slot via FOTA.
2. Verify CRC-32 of inactive slot.
3. Write new boot config: set inactive slot as active, reset boot count.
4. Reboot into new firmware.
5. New firmware runs self-test. If pass: clear boot count (update confirmed). If fail: reboot → boot count increments → after 3 failures, bootloader reverts.

**The $500 guarantee:** The device can never be bricked by a bad OTA update. The bootloader is immutable (never updated OTA). The worst case is reverting to the previous working firmware. The only way to brick it is a bootloader bug — which is why the bootloader must be minimal (~2000 lines of C), thoroughly tested, and never updated in the field.

> **Napkin Math:** Flash overhead for A/B: 32 KB (bootloader) + 4 KB (config) + 28 KB (persistent) = 64 KB overhead. Available per slot: (1024 - 64) / 2 = 480 KB. Model budget per slot: 480 - 120 (app code) - 10 (TFLite Micro runtime) = 350 KB for model weights. At INT8: 350K parameters. Sufficient for most TinyML models (keyword spotting: ~80 KB, person detection: ~300 KB). Boot config wear: 4 KB sector, ~100K erase cycles. At 1 update/week: 100,000 / 52 = 1,923 years before wear-out.

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

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> Watchdog Timers and Hard Real-Time Guarantees</b> · <code>reliability</code> <code>latency</code></summary>

**Interviewer:** "Your Cortex-M4 runs inference for a safety-critical vibration monitor. The system must produce a result every 100ms — no exceptions. Your colleague says 'the model runs in 50ms, so we have 50ms of margin.' Why is this insufficient for a hard real-time guarantee, and what hardware mechanism enforces the deadline?"

**Common Mistake:** "50ms margin is plenty — what could go wrong?" On bare-metal MCUs, many things can stall the CPU beyond the expected inference time.

**Realistic Solution:** The 50ms average-case margin hides worst-case scenarios:

(1) **Flash wait states:** If the ART accelerator cache misses (e.g., a branch to a cold code path), flash reads stall for 5-7 wait states per access. A cache-unfriendly model execution path can add 10-20% latency: 50ms → 60ms.

(2) **Interrupt storms:** A burst of sensor interrupts (e.g., UART overflow, DMA error) can preempt inference for milliseconds. If the interrupt handler has a bug (infinite loop, blocking wait), the CPU never returns to inference.

(3) **Flash write during inference:** If a background task writes to flash (logging, config update), the flash controller locks the entire flash bank for 16-100ms during page erase. The CPU stalls completely — it can't even fetch instructions.

**The hardware safety net: the Independent Watchdog Timer (IWDG).** The IWDG is clocked by a separate low-speed oscillator (LSI, ~32 kHz) independent of the main system clock. Configure it with a 100ms timeout. The inference loop must "kick" (reload) the watchdog after each successful inference. If the CPU fails to kick the watchdog within 100ms — for any reason — the IWDG triggers a hardware reset.

**Design pattern:** (1) Kick watchdog at the start of each inference cycle. (2) Run inference. (3) If inference completes within deadline: output result, kick watchdog. (4) If watchdog fires: the reset handler logs the event to a persistent flash region and reboots. The bootloader checks the reset cause register — if IWDG reset, it increments a fault counter. After 3 consecutive IWDG resets, it falls back to a simpler, faster model guaranteed to meet the deadline.

> **Napkin Math:** IWDG timeout: 100ms. Normal inference: 50ms. Worst-case with cache misses: 60ms. Worst-case with interrupt storm (10ms): 70ms. Worst-case with flash write stall: 50 + 100ms = 150ms → **IWDG fires, system resets in <1ms**. Recovery: reboot (20ms) + model load from flash (5ms) + first inference (50ms) = 75ms. Total outage: ~175ms (one missed deadline, then recovered). Without IWDG: CPU hangs indefinitely — the monitor is dead until someone physically resets it.

**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
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

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> Side-Channel Attacks on MCU Inference</b> · <code>security</code> <code>power</code></summary>

**Interviewer:** "Your competitor deploys a proprietary anomaly detection model on a Cortex-M4 with RDP Level 2 enabled. An attacker buys one of their devices for $50. They can't read flash directly. But they connect a $200 current probe to the power rail and record the power trace during 10,000 inference cycles. Can they extract the model?"

**Common Mistake:** "RDP Level 2 makes the device secure — power traces are just noise." Power analysis is a well-established cryptanalysis technique that applies equally to neural network inference.

**Realistic Solution:** Yes, they can extract significant model information through power side-channel analysis. Here's how:

**What the power trace reveals:**

(1) **Model architecture** — each layer type has a distinct power signature. Convolutions show periodic bursts (one per output channel). Fully connected layers show a single long burst. ReLU shows a brief dip (simple comparison). By counting bursts and measuring their duration, the attacker reconstructs the layer sequence, channel counts, and approximate parameter counts.

(2) **Weight values** — the Cortex-M4's `SMLAD` instruction consumes power proportional to the Hamming weight of its operands. An INT8 weight of 127 (0x7F, 7 bits set) draws measurably more power than a weight of 1 (0x01, 1 bit set). With 10,000 traces and statistical techniques (Differential Power Analysis, DPA), the attacker correlates power samples with hypothetical weight values to recover individual weights. Accuracy: ~80-90% of weights recoverable with 10K traces.

(3) **Input data** — the same technique reveals activation values, potentially leaking sensitive input data (e.g., audio from a keyword spotter, vibration signatures from industrial equipment).

**Defenses (in order of effectiveness):**

(1) **Constant-time execution** — ensure all operations take the same number of cycles regardless of data values. Replace data-dependent branches (e.g., `if (activation > 0)` in ReLU) with branchless equivalents (`result = max(0, activation)`). Cost: minimal.

(2) **Random execution order** — shuffle the order of independent operations (e.g., output channels of a convolution) randomly each inference. This decorrelates the power trace from specific weights. Cost: ~5% performance overhead for the shuffling.

(3) **Noise injection** — run dummy computations on random data between layers. This adds noise to the power trace, increasing the number of traces needed from 10K to 100K+. Cost: 10-20% performance overhead.

(4) **Hardware countermeasures** — use MCUs with built-in power noise generators or voltage regulators that mask power consumption patterns (e.g., ARM TrustZone with secure processing environment).

> **Napkin Math:** Attack cost: $50 (device) + $200 (current probe) + $500 (oscilloscope) + 10,000 inferences × 50ms = 500 seconds of recording. Total: $750 + 8 minutes. With noise injection defense: need 100K traces = 83 minutes of recording + more sophisticated analysis. With random execution order + noise: need 1M+ traces = 14 hours + expert-level analysis. Defense cost: constant-time + shuffling = ~5% performance overhead = 2.5ms per inference on a 50ms model. Acceptable for most applications.

**📖 Deep Dive:** [Volume II: Security & Privacy](https://mlsysbook.ai/vol2/security_privacy.html)
</details>

---

### 📡 Fleet Management

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> Fleet Management for Battery-Powered Sensor Networks</b> · <code>deployment</code> <code>monitoring</code> <code>power</code></summary>

**Interviewer:** "You manage a fleet of 10,000 battery-powered vibration sensors deployed across 50 factories. Each sensor runs on a CR2032 coin cell, communicates via BLE to a gateway, and runs inference on a Cortex-M4. The sensors are mounted on motors in hard-to-reach locations — replacing a battery costs $50 in technician time. Design the fleet management system."

**Common Mistake:** "Replace all batteries on a fixed schedule (e.g., every 6 months)." This wastes money replacing batteries that still have 40% charge and misses batteries that die early due to higher-than-expected inference rates.

**Realistic Solution:** Build a fleet management system with three layers:

**Layer 1: Device-level telemetry (on-device, ~100 bytes/day).** Each sensor reports daily via BLE: battery voltage (2 bytes), inference count since last report (4 bytes), drift flag (1 byte), IWDG reset count (1 byte), temperature (2 bytes), firmware version (4 bytes), model version (4 bytes). Total: ~20 bytes per report. The gateway aggregates reports from ~200 sensors and uploads to the cloud via cellular once per day.

**Layer 2: Predictive battery replacement.** Track each sensor's battery voltage curve over time. A CR2032's voltage drops linearly from 3.0V to 2.7V (80% of capacity), then falls off a cliff. Fit a linear regression to each sensor's voltage history and predict when it will hit 2.5V (minimum operating voltage). Schedule replacement 2 weeks before predicted death. This eliminates both premature replacements (waste) and unexpected deaths (downtime).

**Layer 3: Anomaly detection on the fleet itself.** Monitor fleet-wide statistics: if 5% of sensors in Factory #12 suddenly report high inference counts (suggesting a noisy environment triggering false wake-ups), investigate the environment — not the sensors. If a batch of sensors (same manufacturing lot) shows accelerated battery drain, flag a hardware defect.

**Cost analysis:** Fixed 6-month replacement: 10,000 × 2 replacements/year × $50 = **$1,000,000/year**. Predictive replacement: average battery life 14 months (vs 12 months with conservative scheduling). 10,000 × (12/14) replacements/year × $50 = **$428,571/year**. Savings: **$571,429/year** — the fleet management system pays for itself in the first month.

> **Napkin Math:** Telemetry cost: 20 bytes × 10,000 sensors = 200 KB/day. Gateway cellular: 200 KB × 30 days = 6 MB/month. At $0.01/MB: $0.06/month for the entire fleet. Battery prediction accuracy: with 30+ daily voltage readings, linear regression achieves R² > 0.95 for time-to-death prediction. False positive rate (premature replacement): <5%. False negative rate (unexpected death): <2%. Net savings: $571K/year on a 10,000-sensor fleet.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

---

### 🔋 Power Profiling & Energy Budgeting

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> Power Profiling for MCU Inference</b> · <code>power</code> <code>monitoring</code></summary>

**Interviewer:** "Your product manager asks 'how long will the battery last?' You answer '6 months' based on the datasheet's active and sleep current specs. Three months later, devices start dying in the field. Your estimate was 2× too optimistic. What did you miss, and how do you build an accurate power profile?"

**Common Mistake:** "Use the datasheet's typical current values." Datasheets report typical values at 25°C with minimal peripherals active. Real-world power is always higher.

**Realistic Solution:** The datasheet gives you a lower bound, not a prediction. Real power profiling requires measuring the actual device:

**What the datasheet misses:**

(1) **Peripheral power** — the datasheet's "active mode" current assumes the CPU running from flash with no peripherals. In reality: ADC (1-5 mA), SPI to sensor (2-3 mA), BLE radio TX (8-15 mA), voltage regulator quiescent current (5-50 µA), LED indicator (1-20 mA if left on). These add 10-40 mA on top of the CPU's 20-30 mA.

(2) **Transition energy** — waking from deep sleep to active mode takes 5-50ms depending on clock stabilization (HSE startup: 2-10ms) and peripheral initialization. This transition draws near-active current but does no useful work. At 100 wake-ups per second: transition overhead = 100 × 10ms × 30 mA = 30 mA average — as much as the inference itself.

(3) **Temperature effects** — leakage current doubles every 10°C. At 25°C: sleep current = 5 µA. At 55°C (inside an enclosure in summer): sleep current = 20 µA. Over a year: 20 µA × 8760h = 175 mAh — that's 78% of a CR2032.

(4) **Battery derating** — CR2032 capacity drops 20-30% at high discharge rates (>1 mA pulse) and low temperatures (<0°C). Effective capacity: 150-180 mAh, not 225 mAh.

**How to profile correctly:** Use a current measurement tool (Nordic PPK2, Joulescope, or Otii Arc) that captures the full current waveform at µA resolution and µs time resolution. Record a complete duty cycle: sleep → wake → sensor read → inference → BLE transmit → sleep. Integrate the current over time to get energy per cycle. Multiply by cycles per day. Account for temperature range and battery derating.

> **Napkin Math:** Datasheet estimate: 30 mA active × 50ms + 5 µA sleep × 950ms = 1.5 mA + 4.75 µA = 1.505 mA average. Battery life: 225 mAh / 1.505 mA = 149 hours = 6.2 days... wait, that's not 6 months. The engineer probably assumed 1 inference/minute, not 1/second. At 1/min: 30 mA × 50ms / 60s = 25 µA average active + 5 µA sleep = 30 µA. Battery: 225/0.03 = 7,500 hours = 312 days ≈ 10 months. Real-world: add peripherals (+15 µA), transitions (+5 µA), temperature (+10 µA) = 60 µA. Battery (derated to 180 mAh): 180/0.06 = 3,000 hours = 125 days ≈ 4 months. The 2× gap between estimate (10 months) and reality (4 months) is explained by peripherals, transitions, temperature, and battery derating.

**📖 Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)
</details>

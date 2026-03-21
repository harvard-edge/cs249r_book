# The Deployed Device

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> · <a href="../edge/README.md">🤖 Edge</a> · <a href="../mobile/README.md">📱 Mobile</a> · <b>🔬 TinyML</b>
</div>

---

*How you update firmware and keep it alive for years*

FOTA updates, connectivity, monitoring, security, and long-term reliability — operating ML on devices that must run unattended for years.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/tinyml/03_deployed_device.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---


### 🚀 Deployment & Updates


#### 🟢 L3 — Recall & Define

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The FOTA Update Risk</b> · <code>deployment</code></summary>

- **Interviewer:** "You have 10,000 sensor nodes deployed in a warehouse, each running a vibration anomaly detection model on a Cortex-M4. You need to update the model. The nodes communicate via LoRaWAN (250 bytes/second effective throughput). How do you update them, and what happens if the update fails?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Send the new firmware over LoRaWAN and flash it." At 250 bytes/second, a 200 KB model takes 200,000/250 = 800 seconds = 13 minutes per device. With 10,000 devices sharing the LoRa channel: years.

  **Realistic Solution:** FOTA (Firmware Over-The-Air) for constrained networks requires a different approach:

  (1) **Delta updates** — don't send the full model. Compute a binary diff between the old and new model weights. If only 10% of weights changed (fine-tuning), the delta is ~20 KB instead of 200 KB. Transfer time: 20,000/250 = 80 seconds per device.

  (2) **Multicast** — LoRaWAN Class C supports multicast. Send the update once, all 10,000 devices receive it simultaneously. Transfer time: 80 seconds total (not per device).

  (3) **A/B flash partitioning** — the MCU's 1 MB flash is split: 500 KB for the running firmware (slot A), 500 KB for the update (slot B). The new model is written to slot B while slot A continues running. After verification (CRC check + test inference on a known input), the bootloader atomically swaps the active slot pointer.

  (4) **Failure recovery** — if the CRC check fails, the device stays on slot A and reports the failure. If the device boots from slot B and the watchdog timer fires (model crashes), the bootloader automatically reverts to slot A. The device is never bricked.

  (5) **Staged rollout** — update 100 devices first (1% of fleet). Monitor their anomaly detection accuracy for 24 hours. If no degradation, update the remaining 9,900.

  > **Napkin Math:** Full model: 200 KB. Delta: 20 KB. LoRaWAN multicast: 20 KB / 250 B/s = 80 seconds. Verification: CRC (1ms) + test inference (50ms) = 51ms. Swap: atomic pointer write (1ms). Total per device: 80s transfer + 0.05s verify + 0.001s swap. Fleet of 10,000 via multicast: **80 seconds** + staged validation (24 hours for safety). Without delta/multicast: 200 KB × 10,000 / 250 B/s = 8,000,000 seconds = **92.6 days**.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> Model Versioning on MCU</b> · <code>deployment</code></summary>

- **Interviewer:** "You manage a fleet of 500 RP2040-based (Cortex-M0+, 264 KB SRAM, 2 MB flash) environmental sensors. After three OTA updates, your support team can't tell which model version a device is running. A customer reports false alarms, and you need to know if they're on model v1.2 or v1.4. How do you track model versions on a device with no OS and no filesystem?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Store the version in a global variable in firmware." This ties the model version to the firmware version — if you update only the model (not the firmware), the version string is stale. Worse, a global variable lives in SRAM and is lost on reset.

  **Realistic Solution:** Store model metadata in a structured header prepended to the model binary in flash:

  **Model header format (64 bytes, fixed at a known flash address):**

  | Offset | Size | Field | Example |
  |--------|------|-------|---------|
  | 0x00 | 4 B | Magic number | `0x4D4C5359` ("MLSY") |
  | 0x04 | 4 B | Header version | `0x00000001` |
  | 0x08 | 4 B | Model version (semver packed) | `0x00010400` (v1.4.0) |
  | 0x0C | 4 B | Model size (bytes) | `0x0001C000` (114,688) |
  | 0x10 | 4 B | CRC-32 of model weights | `0xA3F7B2C1` |
  | 0x14 | 8 B | Build timestamp (Unix epoch) | `1710000000` |
  | 0x1C | 16 B | Model hash (first 128 bits of SHA-256) | Unique model fingerprint |
  | 0x2C | 4 B | Target hardware ID | `0x00002040` (RP2040) |
  | 0x30 | 16 B | Reserved / padding | `0x00...` |

  **At boot:** firmware reads the magic number at the known flash address. If valid, it parses the header and exposes the model version via a BLE characteristic or UART command. The support team queries any device with `AT+MODELVER` and gets back `v1.4.0, built 2025-03-10, CRC OK`.

  **During OTA:** the new model binary includes its header. After flashing, the bootloader verifies the magic number and CRC before marking the update as valid. If the header is corrupt or the CRC doesn't match, the update is rejected and the device stays on the previous version.

  **Fleet-wide:** the gateway collects model versions during daily telemetry. A dashboard shows: 480 devices on v1.4.0, 15 on v1.2.0 (failed update), 5 offline. The support team immediately knows the customer's device is on v1.2 and pushes a targeted update.

  > **Napkin Math:** Header overhead: 64 bytes per model. On 2 MB flash: 64 / 2,097,152 = 0.003% — negligible. Boot-time header validation: read 64 bytes from flash (64 / 4 bytes per read × 2 cycles = 32 cycles) + CRC check of 114 KB model (114,000 × 3 cycles = 342,000 cycles at 133 MHz = 2.6 ms). Total boot overhead: **< 3 ms**. BLE version query: 20 bytes response, single BLE packet, < 10 ms round-trip.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Model Checksum Paradox</b> · <code>deployment</code> <code>security</code></summary>

- **Interviewer:** "You deploy a `.tflite` model to an MCU via an OTA update. To ensure the file isn't corrupted, you calculate a CRC32 checksum of the file on your server, send it to the device, and the device calculates the CRC32 of the downloaded file. They match. The device reboots. The model fails to load, crashing the device. How can a file have a perfect checksum but still be completely invalid?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "CRC32 has collisions." While CRC32 isn't cryptographically secure, random collisions on OTA updates are vanishingly rare. The problem is what happened *after* the download.

  **Realistic Solution:** You checked the data in transit, but ignored **Flash Write Errors**.

  Your OTA architecture likely did this:
  1. Download chunk from network into RAM.
  2. Update running CRC32 calculation using the RAM buffer.
  3. Write the RAM buffer to Flash memory.
  4. Compare final CRC32. Match!

  The flaw is that you verified the bytes *before* they were written to the physical storage medium. If the Flash memory was worn out, if there was a voltage droop during the write, or if you forgot to erase the Flash sector before writing (Flash can only turn 1s into 0s without an erase), the physical bits on the silicon will be corrupted. Your RAM buffer was perfect, but the persistent storage is garbage.

  **The Fix:** Always calculate the checksum by reading the data **back out of the Flash memory** after the write is complete. This verifies the entire pipeline: network -> RAM -> Flash Controller -> Physical Silicon.

  > **Napkin Math:** Flash memory must be erased to `0xFF` before writing. If you write `0xAA` (10101010) over unerased data like `0x0F` (00001111), the result is the bitwise AND: `0x0A` (00001010). The file on disk is destroyed, even though the byte in RAM you checksummed was a perfect `0xAA`.

  📖 **Deep Dive:** [Volume I: Optimizing AI](https://harvard-edge.github.io/cs249r_book_dev/contents/optimizing_ai/optimizing_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Model Checksum Paradox</b> · <code>deployment</code> <code>security</code></summary>

- **Interviewer:** "You deploy a `.tflite` model to an MCU via an OTA update. To ensure the file isn't corrupted, you calculate a CRC32 checksum of the file on your server, send it to the device, and the device calculates the CRC32 of the downloaded file. They match. The device reboots. The model fails to load, crashing the device. How can a file have a perfect checksum but still be completely invalid?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "CRC32 has collisions." While CRC32 isn't cryptographically secure, random collisions on OTA updates are vanishingly rare. The problem is what happened *after* the download.

  **Realistic Solution:** You checked the data in transit, but ignored **Flash Write Errors**.

  Your OTA architecture likely did this:
  1. Download chunk from network into RAM.
  2. Update running CRC32 calculation using the RAM buffer.
  3. Write the RAM buffer to Flash memory.
  4. Compare final CRC32. Match!

  The flaw is that you verified the bytes *before* they were written to the physical storage medium. If the Flash memory was worn out, if there was a voltage droop during the write, or if you forgot to erase the Flash sector before writing (Flash can only turn 1s into 0s without an erase), the physical bits on the silicon will be corrupted. Your RAM buffer was perfect, but the persistent storage is garbage.

  **The Fix:** Always calculate the checksum by reading the data **back out of the Flash memory** after the write is complete. This verifies the entire pipeline: network -> RAM -> Flash Controller -> Physical Silicon.

  > **Napkin Math:** Flash memory must be erased to `0xFF` before writing. If you write `0xAA` (10101010) over unerased data like `0x0F` (00001111), the result is the bitwise AND: `0x0A` (00001010). The file on disk is destroyed, even though the byte in RAM you checksummed was a perfect `0xAA`.

  📖 **Deep Dive:** [Volume I: Optimizing AI](https://harvard-edge.github.io/cs249r_book_dev/contents/optimizing_ai/optimizing_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> BLE Throughput for Model Update</b> · <code>mlops</code> <code>monitoring</code></summary>

- **Interviewer:** "Your TinyML wearable needs an over-the-air model update via BLE 5.0. The new model is 150 KB (INT8 quantized, stored in external flash). BLE 5.0 supports 2 Mbps PHY with a maximum data throughput of ~1.4 Mbps after protocol overhead. The device has a 100 mAh battery at 3.7V. The BLE radio draws 8 mA during active transmission/reception. Estimate the update time and the battery cost of the update."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "150 KB at 1.4 Mbps = 0.86 seconds. Trivial." This uses the raw PHY throughput and ignores BLE's connection-oriented protocol overhead, which dramatically reduces effective throughput.

  **Realistic Solution:** BLE data transfer is constrained by connection intervals, MTU size, and the number of packets per connection event.

  **(1) BLE throughput reality.** BLE 5.0 at 2M PHY: raw bit rate = 2 Mbps. But BLE transfers data in ATT (Attribute Protocol) notifications within connection events. Maximum ATT payload (with DLE — Data Length Extension): 244 bytes per packet. Packets per connection event: typically 4–6 (limited by connection event length). Connection interval: 7.5ms minimum (iOS enforces 15ms minimum). Effective throughput: 6 packets × 244 bytes / 7.5ms = **195 KB/s** (best case). Realistic (Android, 15ms interval, 4 packets): 4 × 244 / 15ms = **65 KB/s**.

  **(2) Update time.** At 65 KB/s: 150 KB / 65 = **2.3 seconds**. At 195 KB/s (best case): 0.77 seconds. But add protocol overhead: connection setup (200ms), service discovery (500ms), MTU negotiation (100ms), and post-transfer verification (CRC check, 200ms). Total: 2.3 + 1.0 = **3.3 seconds** (realistic). Best case: 1.77 seconds.

  **(3) Battery cost.** BLE radio at 8 mA for 3.3 seconds: 8 mA × 3.3s / 3600 = **0.0073 mAh**. Battery: 100 mAh. Cost: 0.0073% of battery. Negligible — you could do 13,700 model updates on a full charge. The BLE update cost is irrelevant to battery life.

  **(4) The real bottleneck: flash write.** Writing 150 KB to external NOR flash at ~1 MB/s (typical SPI flash write speed with 256-byte page writes): 150 KB / 1 MB/s = **150ms**. But NOR flash requires sector erase before write: 150 KB / 4 KB sectors = 38 sector erases × 50ms each = **1.9 seconds**. Total: BLE transfer (2.3s) + flash erase (1.9s) + flash write (0.15s) + verify (0.2s) = **4.55 seconds**. Flash erase is 42% of the total update time.

  > **Napkin Math:** BLE transfer: 2.3s at 65 KB/s. Flash erase: 1.9s (38 sectors × 50ms). Flash write: 0.15s. Verify: 0.2s. Total: 4.55s. Battery cost: 8 mA × 4.55s = 36.4 mA·ms = 0.01 mAh (0.01% of battery). For a 500 KB model: BLE = 7.7s, flash erase = 6.25s (125 sectors), total = 14.2s. At this size, BLE and flash erase are roughly equal bottlenecks. For models >1 MB: flash erase dominates. Optimization: use a delta update (send only changed weights) — typical delta for a fine-tuned model: 10–20% of weights change → 15–30 KB transfer instead of 150 KB.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>


#### 🔵 L4 — Apply & Identify

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Offline Drift Detector</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your deployed anomaly detector starts producing false positives after 3 months. The device has no cloud connection — it operates fully offline. How do you detect and handle model drift on a device with 256 KB SRAM and no internet?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Upload data to the cloud for analysis." There's no cloud connection. You must handle this entirely on-device.

  **Realistic Solution:** On-device drift detection with minimal resources:

  (1) **Running statistics** — maintain exponential moving averages of the model's input feature statistics (mean and variance of each input channel). Storage: 2 floats × N channels × 4 bytes = ~64 bytes for a 8-channel sensor. When the running mean drifts beyond 3σ of the baseline (computed at deployment), flag a drift event. This detects covariate shift (e.g., sensor degradation, environmental change).

  (2) **Prediction distribution monitoring** — track the distribution of the model's output confidence scores. A healthy model produces mostly high-confidence predictions (normal) with occasional low-confidence ones (anomalies). If the ratio of low-confidence predictions exceeds a threshold (e.g., >30% of predictions in the last hour), the model is likely seeing OOD data.

  (3) **Self-calibration** — store a small set of "golden" reference inputs in flash (10 known-normal vibration signatures, ~5 KB). Periodically (once per hour), run inference on these references. If the model's predictions on known-normal inputs start drifting (confidence drops below 0.95), the model or the sensor has degraded.

  (4) **Graceful response** — when drift is detected: (a) increase the anomaly threshold to reduce false positives (accepting more false negatives), (b) activate an LED indicator for maintenance personnel, (c) log the drift event with timestamp to flash for later retrieval, (d) if drift exceeds a critical threshold, fall back to a simple threshold-based detector (no ML) until the device is serviced.

  > **Napkin Math:** Running statistics: 64 bytes RAM. Golden references: 5 KB flash. Hourly self-test: 10 inferences × 50ms = 500ms per hour = 0.014% CPU overhead. Drift detection latency: 1 hour (self-test interval). Storage for drift log: 20 bytes per event × 100 events = 2 KB flash. Total resource cost: 64 bytes RAM + 7 KB flash — negligible on a 256 KB SRAM / 1 MB flash device.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> FOTA Update Integrity Verification</b> · <code>deployment</code> <code>security</code></summary>

- **Interviewer:** "Your predictive maintenance sensors receive firmware updates over-the-air (FOTA) containing a new TFLite Micro model. The bootloader verifies the binary hash (SHA-256) before swapping partitions. Why is verifying the binary hash insufficient for ML models, and how do you implement functional model attestation (inference on a golden test input) to prove the model's math is intact?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "If the SHA-256 hash matches, the file hasn't been corrupted, so the model is safe to run." This assumes the model was compiled correctly and the runtime on the device is perfectly compatible with the new model's operators.

  **Realistic Solution:** A binary hash only proves the file arrived exactly as it was sent. It does *not* prove that the ML model will actually execute correctly on the device. The new model might use a TFLite operator (e.g., `RESIZE_NEAREST_NEIGHBOR`) that isn't compiled into the device's specific TFLite Micro runtime, causing a hard fault on the first real inference. Or, a quantization bug on the build server might have produced a model that hashes perfectly but outputs garbage predictions.

  To guarantee ML integrity, the bootloader (or a first-boot initialization sequence) must perform **functional model attestation**. The device stores a "golden" test input (e.g., a pre-processed vibration spectrogram) and its expected output tensor in a read-only flash sector. After the SHA-256 check passes, the device loads the new model into the tensor arena, feeds it the golden input, and runs a full forward pass. It then compares the output tensor to the expected reference. If the Mean Squared Error (MSE) is below a strict threshold, the model's *math* is proven intact, and the update is committed. If it crashes or outputs garbage, the device rolls back to the previous partition.

  > **Napkin Math:** SHA-256 on a 200 KB model takes ~15ms on a Cortex-M4. Functional attestation (running one inference) takes ~50ms. The golden input (e.g., 32×32 INT8 spectrogram) takes 1 KB of flash. The expected output (e.g., 4-class probabilities) takes 4 bytes. For a 65ms total boot-time penalty and 1 KB of flash overhead, you completely eliminate the risk of bricking a remote sensor with a mathematically broken model update.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> LoRaWAN Telemetry for ML Metrics</b> · <code>deployment</code> <code>monitoring</code></summary>

- **Interviewer:** "You have 3,000 soil moisture sensors deployed across farmland, each running a crop stress prediction model on an STM32WL (Cortex-M4, 48 MHz, 256 KB flash, 64 KB SRAM) with built-in LoRa radio. You want to monitor model performance remotely — inference confidence, prediction distribution, drift indicators. But LoRaWAN has strict duty cycle limits. Design the telemetry payload and transmission strategy."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Send all inference results over LoRa." At 24 inferences/day (one per hour) with even a minimal 20-byte result, that's 480 bytes/day. LoRaWAN's duty cycle limits make this surprisingly expensive in airtime.

  **Realistic Solution:** LoRaWAN operates under regional duty cycle regulations (EU868: 1% duty cycle, US915: dwell time limits). You must minimize airtime:

  **LoRaWAN constraints (EU868, SF7, 125 kHz BW):**
  - Data rate: ~5.5 kbps (SF7)
  - Max payload per uplink: 222 bytes (DR5)
  - 1% duty cycle on most sub-bands: after transmitting for 1 second, you must wait 99 seconds
  - Airtime for 50-byte payload at SF7: ~72 ms → cooldown: 7.2 seconds

  **Telemetry payload design (compact binary, not JSON):**

  | Field | Size | Encoding | Description |
  |-------|------|----------|-------------|
  | Device ID | 0 B | In LoRaWAN header (DevAddr) | Free — already in the protocol |
  | Timestamp | 2 B | Minutes since midnight | Resets daily, 0-1440 |
  | Battery voltage | 1 B | (V - 2.0) × 100, uint8 | Range 2.0-4.55V, 10 mV resolution |
  | Inference count (24h) | 2 B | uint16 | 0-65535 |
  | Anomaly count (24h) | 2 B | uint16 | Predictions above threshold |
  | Mean confidence (24h) | 1 B | uint8, 0-255 → 0.0-1.0 | Average softmax confidence |
  | Confidence histogram | 4 B | 4 bins × 1 byte (counts) | [0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1.0] |
  | Drift indicator | 1 B | uint8 flags | Bit flags: input drift, output drift, sensor fault |
  | Model version | 2 B | uint16 (major.minor packed) | Which model is running |
  | Temperature | 1 B | int8, °C | -128 to +127°C |
  | **Total** | **16 B** | | |

  **Transmission strategy:** One uplink per day with the 16-byte summary. Airtime at SF7: ~41 ms. Duty cycle consumed: 0.041 / 86,400 = 0.000047% — well within the 1% limit. This leaves 99.99% of the duty cycle budget for emergency alerts (e.g., sudden anomaly spike → immediate uplink).

  **Fleet aggregation:** 3,000 devices × 16 bytes/day = 48 KB/day at the network server. A simple dashboard computes fleet-wide metrics: mean accuracy proxy (confidence distribution), drift prevalence, battery health histogram, model version distribution.

  > **Napkin Math:** Daily telemetry: 16 bytes × 1 uplink = 16 bytes/day. Airtime: 41 ms/day. Duty cycle: 0.000047%. Annual airtime: 41 ms × 365 = 15 seconds/year. Energy per uplink: 40 mA TX × 41 ms × 3.3V = 5.4 mJ. Annual telemetry energy: 5.4 × 365 = 1.97 J. On a 3.6V 19 Ah lithium battery (68,400 J): telemetry = 0.003% of battery — invisible. If you sent raw results (480 bytes/day): airtime = 600 ms/day, duty cycle = 0.0007%, energy = 79 mJ/day = 28.8 J/year = 0.04% of battery. Still manageable, but 15× more expensive for minimal extra insight.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> Inference Result Compression for Upload</b> · <code>deployment</code> <code>data-pipeline</code></summary>

- **Interviewer:** "Your fleet of 1,000 wildlife acoustic monitors runs a bird species classifier on an RP2040 (Cortex-M0+, 133 MHz, 264 KB SRAM, 2 MB flash). Each device classifies 10-second audio clips and detects up to 50 species. The devices upload results daily via cellular (LTE-M Cat-M1, billed at $0.50/MB). You're currently sending raw JSON results and the cellular bill is $3,000/month. Compress the upload payload."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use gzip on the JSON." Gzip on a Cortex-M0+ is expensive — the M0+ has no barrel shifter, making the bit manipulation in DEFLATE very slow. And JSON is the wrong format for constrained uploads in the first place.

  **Realistic Solution:** The problem is the data format, not the compression algorithm. Replace JSON with a compact binary protocol:

  **Current format (JSON):**
  ```
  {"ts":1710000000,"species":"AMRO","conf":0.92,"count":3}
  ```
  ~55 bytes per detection. At 100 detections/day per device: 5.5 KB/day. 1,000 devices: 5.5 MB/day = **165 MB/month = $82.50/month**. Wait — the user said $3,000/month. That means they're sending much more: probably the full 50-class softmax vector per clip.

  **Full softmax upload:** 50 classes × 4 bytes (float32) = 200 bytes per clip. At 6 clips/minute × 60 min × 12 hours of daylight = 4,320 clips/day. 4,320 × 200 = 864 KB/day. 1,000 devices: 864 MB/day = **25.9 GB/month = $12,960/month**. That explains the bill.

  **Optimized binary format:**

  (1) **Top-K only** — instead of 50 softmax values, send only the top-3 species per clip. Encoding: 3 × (1 byte species ID + 1 byte confidence as uint8 0-255) = 6 bytes per clip.

  (2) **Temporal aggregation** — instead of per-clip results, aggregate over 1-hour windows. For each hour, send: {hour_id (1B), top-5 species detected (5 × 2B = 10B), total clip count (2B), anomaly flag (1B)} = 14 bytes per hour. 12 hours/day: 168 bytes/day.

  (3) **Daily summary packet:** 168 bytes payload + 8 bytes header (device ID + date) = **176 bytes/day per device**.

  **Fleet-wide:** 1,000 × 176 = 176 KB/day = **5.28 MB/month = $2.64/month**.

  **Savings:** from $12,960/month to $2.64/month = **99.98% reduction**. Even if the original $3,000 figure was with some optimization already applied, the binary aggregation approach reduces it by 3 orders of magnitude.

  **On the M0+:** No compression algorithm needed. The "compression" is semantic — sending summaries instead of raw data. The aggregation logic (tracking top-K species per hour) requires: 50 species × 2 bytes (count) = 100 bytes of SRAM per hour window. Trivial on 264 KB.

  > **Napkin Math:** Raw softmax: 200 B/clip × 4,320 clips/day = 864 KB/day. Top-3 binary: 6 B/clip × 4,320 = 25.9 KB/day (33× reduction). Hourly aggregation: 168 B/day (5,143× reduction). Fleet monthly cost: raw = $12,960, top-3 = $388, aggregated = $2.64. Annual savings: $155,000. SRAM cost of aggregation: 100 bytes. CPU cost: 50 comparisons per clip to update top-K = 50 × 4,320 = 216,000 ops/day at 133 MHz = 1.6 ms/day. The compression is essentially free.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>


#### 🟡 L5 — Analyze & Predict

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> Bootloader A/B Firmware Partitioning</b> · <code>deployment</code> <code>reliability</code></summary>

- **Interviewer:** "Design the flash memory layout for a Cortex-M4 with 1 MB flash that supports A/B firmware partitioning with rollback. The firmware includes a bootloader, application code, and a TFLite Micro model. The device is deployed in a location where physical access costs $500 per visit."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

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

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> Fleet-Wide Model Update Strategy</b> · <code>deployment</code> <code>mlops</code></summary>

- **Interviewer:** "You manage 100,000 predictive maintenance sensors across 200 factories. The fleet has 5 hardware variants: Cortex-M0+ (nRF52810, 64 KB flash), Cortex-M4 (STM32L4, 1 MB flash), Cortex-M4F (Apollo4, 2 MB flash), Cortex-M33 (nRF5340, 1 MB flash), and ESP32-S3 (8 MB flash). Connectivity is mixed: 40% BLE-only, 35% LoRaWAN, 25% cellular (LTE-M). You need to deploy a retrained anomaly detection model to the entire fleet. Design the update strategy."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Build one model, compile for each target, push to all devices." This ignores that a model fitting in 64 KB flash (nRF52810) is fundamentally different from one using 2 MB flash (Apollo4). You can't deploy the same model to all variants — you need a model family.

  **Realistic Solution:** This is a multi-dimensional logistics problem: model × hardware × connectivity.

  **Step 1: Model family.** Train one base model, then produce 5 target-specific variants:

  | Hardware | Flash budget | Model variant | Size |
  |----------|-------------|---------------|------|
  | nRF52810 (M0+) | 30 KB | INT8, 3-layer, pruned 80% | 28 KB |
  | STM32L4 (M4) | 350 KB | INT8, 8-layer, pruned 50% | 180 KB |
  | Apollo4 (M4F) | 1.5 MB | INT8, 12-layer, full | 420 KB |
  | nRF5340 (M33) | 300 KB | INT8, 8-layer, pruned 60% | 160 KB |
  | ESP32-S3 | 4 MB | INT8, 12-layer, full + ensemble | 800 KB |

  **Step 2: Delta compression.** Compute binary diffs between old and new model for each variant. Typical delta for a retrained model (same architecture, updated weights): 15-25% of full size.

  **Step 3: Connectivity-aware rollout.**

  *Cellular (25K devices):* Push delta updates directly. 25,000 devices × 50 KB avg delta / 50 KB/s LTE-M = 1 second per device. Parallelize across 100 concurrent connections: 25,000 / 100 = 250 batches × 1 s = **4.2 minutes**.

  *BLE (40K devices):* Requires gateway proximity. Each factory has 2-5 BLE gateways. Gateway downloads full delta via Ethernet, then pushes to devices via BLE mesh. 40,000 devices / 200 factories = 200 devices per factory. At 60 KB/s BLE throughput, 50 KB delta: 0.83 s per device × 200 = 166 s per factory. With 3 gateways in parallel: **55 seconds per factory**. All factories in parallel: **55 seconds**.

  *LoRaWAN (35K devices):* The bottleneck. LoRaWAN Class C multicast: 250 B/s effective. 50 KB delta: 200 seconds per multicast group. Devices are grouped by LoRa gateway (typically 500 devices per gateway). 35,000 / 500 = 70 gateways, all multicasting in parallel: **200 seconds = 3.3 minutes**.

  **Step 4: Staged rollout.** Update 1% of each variant first (1,000 devices). Monitor for 24 hours: inference latency, anomaly rate, battery drain, crash rate. If all metrics are within 10% of baseline, proceed with the remaining 99%.

  > **Napkin Math:** Total fleet update time (after staging): max(4.2 min cellular, 55 s BLE, 3.3 min LoRa) = **4.2 minutes** (cellular is the bottleneck due to sequential batching). Add 24-hour staging validation: **24 hours + 4.2 minutes**. Cost: cellular data = 25,000 × 50 KB = 1.25 GB at $0.50/MB = $625. BLE/LoRa: free (local). Total update cost: **$625 + engineering time**. Per device: $0.006. Without delta compression: 25,000 × 250 KB avg = 6.25 GB = $3,125. Delta saves **$2,500 per update cycle**.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> Hardware-in-the-Loop Testing</b> · <code>deployment</code> <code>monitoring</code></summary>

- **Interviewer:** "Your CI pipeline tests ML models in simulation (x86 QEMU), but you've been burned twice by models that pass simulation and fail on real hardware — once due to CMSIS-NN kernel differences, once due to flash timing. You have 5 hardware variants (Cortex-M0+, M4, M4F, M7, M33). Design a hardware-in-the-loop (HIL) CI system. How many test boards do you need, and what's the test time per commit?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Buy one of each board and run tests sequentially." With 5 boards tested sequentially, each taking 2-3 minutes, your CI feedback loop is 10-15 minutes per commit. Developers will skip HIL tests or ignore failures because the feedback is too slow.

  **Realistic Solution:** Design the HIL farm for parallelism and reliability:

  **Test board fleet:**

  | Variant | Board | Unit cost | Quantity | Purpose |
  |---------|-------|-----------|----------|---------|
  | Cortex-M0+ | nRF52810-DK | $40 | 3 | Minimum viable target, tests tight flash/SRAM |
  | Cortex-M4 | STM32L4-Discovery | $20 | 3 | Primary deployment target |
  | Cortex-M4F | Apollo4 Blue EVB | $50 | 2 | FPU-enabled path, large SRAM |
  | Cortex-M7 | STM32H7-Nucleo | $25 | 2 | High-performance target, TCM testing |
  | Cortex-M33 | nRF5340-DK | $45 | 3 | TrustZone + dual-core testing |
  | **Total** | | | **13 boards** | **$435** |

  3 boards per primary target (M0+, M4, M33) for redundancy — if one board fails, tests still run on the other two. 2 boards for secondary targets (M4F, M7).

  **HIL test pipeline per commit:**

  (1) **Flash firmware (parallel across all boards):** SEGGER J-Link connected to each board via USB to a Raspberry Pi 4 test controller. Flash time: ~1 second per board. All 13 boards flash in parallel: **1 second**.

  (2) **Inference accuracy test:** Run inference on 10 golden test inputs. Compare outputs against x86 reference (bit-exact for INT8, within tolerance for FP32). Time per board: 10 inferences × 50 ms (worst case on M0+) = 500 ms. All boards in parallel: **500 ms**.

  (3) **Latency regression test:** Run 100 inferences, measure P50/P99 latency. Compare against baseline. Flag if P99 regresses by > 5%. Time: 100 × 50 ms = 5 seconds on M0+. All boards in parallel: **5 seconds**.

  (4) **Memory high-water-mark test:** Instrument the tensor arena with a canary pattern. After inference, check how much of the arena was touched. Flag if peak SRAM usage increased. Time: **500 ms** (one instrumented inference + canary check).

  (5) **Power measurement (nightly, not per-commit):** Use Nordic PPK2 on one board per variant. Run 1000 inferences, measure energy per inference. Compare against baseline. Time: 50 seconds per board. Run sequentially (one PPK2 per variant): **250 seconds = 4.2 minutes**.

  **Total per-commit HIL time:** 1 + 0.5 + 5 + 0.5 = **7 seconds** (all boards in parallel). Add CI overhead (checkout, build, flash): ~60 seconds. **Total: ~67 seconds per commit.**

  **Infrastructure:**
  - 13 dev boards: $435
  - 1 Raspberry Pi 4 per 5 boards (USB hub): 3 × $75 = $225
  - 3 SEGGER J-Link EDU: 3 × $60 = $180
  - USB hubs, cables, rack: ~$100
  - **Total: ~$940** — less than one engineer-day of debugging a hardware-specific failure.

  > **Napkin Math:** Per-commit HIL: 67 seconds. At 20 commits/day: 22 minutes of total HIL time. Board utilization: 7 s active / 67 s cycle = 10.4%. Boards are idle 90% of the time — plenty of headroom for parallel branches. Cost of one missed hardware bug (field failure on 10,000 devices): $50 per device visit × 10,000 = $500,000. HIL farm cost: $940. ROI: prevents one field failure = **531× return**. Nightly power test: 4.2 minutes × 5 variants = 21 minutes. Catches power regressions before they reach production.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Unsigned Integer Wrap</b> · <code>mlops</code> <code>robustness</code></summary>

- **Interviewer:** "Your predictive maintenance system uses a Cortex-M0+ to monitor motor vibrations. It keeps a running tally of anomalies in a `uint16_t` counter and uploads the total to the cloud every week. After 18 months, the cloud dashboard suddenly reports that the factory had exactly 65,500 *fewer* anomalies this week than last week. The factory hasn't changed. What broke?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The ML model drifted and stopped detecting anomalies." Model drift happens slowly; an instant drop of 65,000 points to a data type failure.

  **Realistic Solution:** You suffered an **Integer Overflow (Wrap-Around)**.

  A `uint16_t` (unsigned 16-bit integer) can only hold a maximum value of `65,535`.
  Because your edge device was keeping a *running total* of anomalies over 18 months, the counter slowly crept up.
  When the counter hit `65,535`, the very next anomaly caused the integer to overflow and wrap back to `0`.

  When the device uploaded `0` to the cloud, the cloud dashboard subtracted last week's value (e.g., `65,500`) from this week's value (`0`), determining that there was a massive negative drop in anomalies.

  **The Fix:**
  1. Use wider data types for absolute accumulators (e.g., `uint32_t` holds up to 4.2 billion).
  2. The edge device should *never* send running totals. It should send the *delta* (number of anomalies since the last upload) and clear the counter to zero.

  > **Napkin Math:** 100 anomalies a day. `65535 / 100 = 655 days`. At exactly 1.8 years (18 months), the integer physically runs out of bits and wraps to zero, corrupting your MLOps dashboard.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>


#### 🔴 L6+ — Synthesize & Derive

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 6+" align="center"> Multi-MCU Distributed Inference System</b> · <code>parallelism</code> <code>serving</code></summary>

- **Interviewer:** "You need to run a model that requires 80 KB of activation memory on a system with three Cortex-M0+ MCUs, each with only 32 KB SRAM. No single MCU can hold the full activation tensor. Design a distributed inference system that splits the model across the three MCUs, connected via SPI at 8 MHz. Specify the partitioning strategy, communication protocol, and the latency overhead of distribution."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Split the model into 3 equal parts and pipeline them." Equal splitting ignores that the activation memory peak varies by layer — the early layers (high resolution, few channels) have different memory profiles than late layers (low resolution, many channels). You need to split at points where the activation memory fits within 32 KB.

  **Realistic Solution:** Use layer-wise profiling to find optimal split points that minimize both per-MCU memory and inter-MCU communication.

  **(1) Layer-by-layer memory profiling.** Profile the model to find the activation memory at each layer boundary. For a typical CNN (e.g., MobileNetV1 at 64×64 input): Layer 0 output: 64×64×32 = 131 KB (too large for any MCU). Layer 3 output (after stride-2): 32×32×64 = 65 KB (still too large). Layer 5 output (after stride-2): 16×16×128 = 32 KB (fits!). Layer 8 output: 8×8×256 = 16 KB (fits). Layer 11 output: 4×4×512 = 8 KB (fits). Final output: 1×1×1000 = 4 KB (fits).

  **(2) Split strategy.** Split at layer boundaries where the output tensor is ≤32 KB: MCU 1: layers 0–5 (input → 16×16×128 output). Peak activation: needs input (64×64×3 = 12 KB) + largest intermediate (64×64×32 = 128 KB). Still too large! The early layers' activations exceed 32 KB.

  **(3) Revised strategy: patch-based inference on MCU 1.** Process the 64×64 input as four 32×32 patches on MCU 1. Each patch: 32×32×3 = 3 KB input. After 2 stride-2 layers: 8×8×128 = 8 KB output per patch. Peak activation per patch: ~20 KB. Fits in 32 KB. MCU 1 processes 4 patches sequentially, producing 4 × 8×8×128 = 32 KB total output (reassembled into 16×16×128). Transfer 32 KB to MCU 2 via SPI.

  **(4) MCU 2: layers 6–8.** Input: 16×16×128 = 32 KB (exactly fits). Output: 8×8×256 = 16 KB. Peak activation during computation: 32 KB input + 16 KB output = 48 KB — doesn't fit! Solution: process in two 16×16×64 channel slices. Each slice: 16 KB input, 8 KB output, 24 KB peak. Fits. Transfer 16 KB to MCU 3.

  **(5) MCU 3: layers 9–end.** Input: 8×8×256 = 16 KB. Output: 1×1×1000 = 4 KB. Peak: ~20 KB. Fits easily.

  **(6) Communication overhead.** SPI at 8 MHz, 8-bit mode = 1 MB/s. MCU 1 → MCU 2: 32 KB / 1 MB/s = **32ms**. MCU 2 → MCU 3: 16 KB / 1 MB/s = **16ms**. Total communication: **48ms**. Compute per MCU: ~5ms each. Total inference: 4×5ms (MCU 1 patches) + 32ms (transfer) + 2×5ms (MCU 2 slices) + 16ms (transfer) + 5ms (MCU 3) = **73ms**.

  > **Napkin Math:** Single MCU (if it had 80 KB SRAM): ~15ms inference. Distributed (3 MCUs): 73ms. Overhead: 4.9× (communication dominates). SPI transfer: 48ms / 73ms = 66% of total time spent on communication. Optimization: use SPI DMA to overlap transfer with computation on the next MCU. MCU 2 starts computing as soon as the first channel slice arrives (16 KB / 1 MB/s = 16ms). While MCU 2 computes slice 1 (5ms), MCU 1 sends slice 2 (16ms). Pipelined: total reduces to ~55ms. Still 3.7× slower than a single MCU, but enables models that physically cannot fit on one device. Alternative: use a single Cortex-M4 with 128 KB SRAM ($2 more) and avoid the distributed complexity entirely. The engineering cost of distributed TinyML rarely justifies the BOM savings.

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>


---


### 🌐 Networking & Connectivity


#### 🔵 L4 — Apply & Identify

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Cellular NAT Timeout</b> · <code>networking</code> <code>deployment</code></summary>

- **Interviewer:** "Your IoT device connects to AWS IoT Core via MQTT over an LTE-M cellular connection. It sends an ML telemetry payload perfectly upon booting. It then sits idle for 30 minutes. When the next anomaly occurs, the `mqtt_publish()` function claims success, but the message never arrives at AWS. The device didn't sleep and the cellular signal is perfect. Why did the network silently swallow the message?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The MQTT broker crashed." AWS IoT Core doesn't crash; the path to it was severed.

  **Realistic Solution:** You hit the **Cellular Carrier NAT (Network Address Translation) Timeout**.

  Cellular networks do not give your device a public IP address. They put you behind a massive Carrier-Grade NAT. The NAT router maintains a state table linking your device's internal IP to the external AWS server's IP.

  To save memory on their expensive routers, cellular carriers aggressively purge idle connections. If no data flows through the TCP socket for a short period (often as little as 2 to 5 minutes), the carrier silently deletes the routing table entry.

  Your device thinks the TCP socket is still perfectly open. AWS thinks the device gracefully disconnected. When your device finally sends data 30 minutes later, the packet hits the carrier's NAT router, the router has no idea where it belongs, and silently drops it.

  **The Fix:** You must configure the MQTT **Keep-Alive Interval** to be strictly shorter than the carrier's NAT timeout (e.g., set MQTT Keep-Alive to 60 or 120 seconds). The device will periodically send a tiny 2-byte PINGREQ packet, which resets the carrier's NAT timer and keeps the TCP tunnel physically open.

  > **Napkin Math:** LTE-M NAT Timeout = ~300 seconds. ML Anomaly Interval = 1800 seconds. 1800 > 300. The TCP connection is guaranteed to be dead every single time the device tries to use it.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)
  </details>
</details>


#### 🟡 L5 — Analyze & Predict

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The OTA Bandwidth Congestion</b> · <code>networking</code> <code>deployment</code></summary>

- **Interviewer:** "You have a fleet of 5,000 smart factory sensors connected via a shared LoRaWAN gateway. You push a 100 KB model update to the fleet simultaneously. The OTA update process stalls, taking days to complete, and normal sensor telemetry stops functioning entirely. What network characteristic of LoRaWAN did you violate?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "100 KB is too large for the gateway." While true, it's not just the size; it's the collision domain and the protocol duty cycle.

  **Realistic Solution:** You violated the **Duty Cycle Limits and the ALOHA MAC protocol**.

  LoRaWAN operates in unlicensed sub-GHz bands (like 868 MHz or 915 MHz). By law in many regions, a device can only transmit for 1% of the time (the duty cycle limit).

  Furthermore, LoRa uses a modified ALOHA protocol. Devices just "shout" their data into the air. If 5,000 devices are all trying to send acknowledgment packets (ACKs) for the OTA chunks they are receiving at the exact same time, the radio waves collide in the air. The gateway receives garbage. The devices wait, timeout, and retry... causing even more collisions. This is a **Broadcast Storm**.

  Your OTA update effectively DDOS'd your own factory network.

  **The Fix:**
  1. Use **Multicast OTA (FUOTA - Firmware Update Over The Air)**. The gateway broadcasts the firmware chunks once, and all 5,000 devices listen simultaneously without sending individual ACKs for every packet. They only request missing packets at the very end.
  2. If Multicast isn't available, you must strictly stagger the updates (e.g., updating only 10 devices an hour) to prevent airwave congestion.

  > **Napkin Math:** In LoRa SF12, a 51-byte payload takes ~2.5 seconds of airtime. A 1% duty cycle means the device must remain completely silent for the next 247 seconds before it can send an ACK. Sending 100 KB point-to-point to 5,000 devices is mathematically impossible under these physics.

  📖 **Deep Dive:** [Volume I: Optimizing AI](https://harvard-edge.github.io/cs249r_book_dev/contents/optimizing_ai/optimizing_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The OTA Bandwidth Congestion</b> · <code>networking</code> <code>deployment</code></summary>

- **Interviewer:** "You have a fleet of 5,000 smart factory sensors connected via a shared LoRaWAN gateway. You push a 100 KB model update to the fleet simultaneously. The OTA update process stalls, taking days to complete, and normal sensor telemetry stops functioning entirely. What network characteristic of LoRaWAN did you violate?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "100 KB is too large for the gateway." While true, it's not just the size; it's the collision domain and the protocol duty cycle.

  **Realistic Solution:** You violated the **Duty Cycle Limits and the ALOHA MAC protocol**.

  LoRaWAN operates in unlicensed sub-GHz bands (like 868 MHz or 915 MHz). By law in many regions, a device can only transmit for 1% of the time (the duty cycle limit).

  Furthermore, LoRa uses a modified ALOHA protocol. Devices just "shout" their data into the air. If 5,000 devices are all trying to send acknowledgment packets (ACKs) for the OTA chunks they are receiving at the exact same time, the radio waves collide in the air. The gateway receives garbage. The devices wait, timeout, and retry... causing even more collisions. This is a **Broadcast Storm**.

  Your OTA update effectively DDOS'd your own factory network.

  **The Fix:**
  1. Use **Multicast OTA (FUOTA - Firmware Update Over The Air)**. The gateway broadcasts the firmware chunks once, and all 5,000 devices listen simultaneously without sending individual ACKs for every packet. They only request missing packets at the very end.
  2. If Multicast isn't available, you must strictly stagger the updates (e.g., updating only 10 devices an hour) to prevent airwave congestion.

  > **Napkin Math:** In LoRa SF12, a 51-byte payload takes ~2.5 seconds of airtime. A 1% duty cycle means the device must remain completely silent for the next 247 seconds before it can send an ACK. Sending 100 KB point-to-point to 5,000 devices is mathematically impossible under these physics.

  📖 **Deep Dive:** [Volume I: Optimizing AI](https://harvard-edge.github.io/cs249r_book_dev/contents/optimizing_ai/optimizing_ai.html)

  </details>

</details>


---


### 🔒 Security & Privacy


#### 🔵 L4 — Apply & Identify

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
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Flash Extraction Attack</b> · <code>security</code></summary>

- **Interviewer:** "An attacker has physical access to your deployed MCU. They want to extract your proprietary model weights from flash memory. How can power side-channel analysis extract model weights by correlating power traces with MAC operations, and why does the model's arithmetic structure make this ML-specific attack possible?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Flash memory is internal to the MCU — it can't be read externally." Or "Just enable Read-Out Protection (RDP)." RDP blocks JTAG debuggers, but side-channel attacks bypass digital locks entirely.

  **Realistic Solution:** The attacker doesn't need to read the flash directly. They connect a high-resolution current probe to the MCU's power rail and record the power consumption while feeding known inputs to the ML model.

  This attack exploits the fundamental arithmetic structure of neural networks. During inference, the MCU executes millions of Multiply-Accumulate (MAC) operations. The power consumed by the ALU during a multiplication $w \times x$ depends on the Hamming weight (number of '1' bits) of the operands. Because the attacker knows the input $x$, they can use statistical methods (like Correlation Power Analysis) across thousands of inference traces to guess the weight $w$. They hypothesize a weight value, simulate the expected power draw for the known inputs, and correlate it with the measured power trace. The value with the highest correlation is the true weight. They repeat this layer by layer.

  **Defense:** (1) **Weight Masking:** XOR the weights with a random mask before storage, and unmask them dynamically during inference using a hardware random number generator (TRNG). (2) **Dummy Operations:** Insert random dummy MAC operations into the inference loop to desynchronize the power trace. (3) **Execution Jitter:** Randomly vary the MCU clock speed or insert random delays between layers to misalign the attacker's traces.

  > **Napkin Math:** A Cortex-M4 executing a `SMLAD` instruction draws ~20 mA. The difference in current between multiplying 0x0000 and 0xFFFF might be just 50 µA. A 1 GS/s oscilloscope can capture this micro-variation. With ~10,000 inference traces (which takes just a few minutes to collect at 10 Hz inference rate), the signal-to-noise ratio is high enough to extract an entire 8-bit weight matrix with >99% accuracy. Defenses like dummy operations add ~10-20% performance overhead but reduce the SNR so severely that the attacker would need millions of traces, making the attack economically unviable.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>


#### 🟡 L5 — Analyze & Predict

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> Secure Boot Chain for ML Models</b> · <code>security</code> <code>deployment</code></summary>

- **Interviewer:** "Your company ships a medical wearable running a cardiac arrhythmia detection model on an STM32U5 (Cortex-M33 with TrustZone, 160 MHz, 2 MB flash, 786 KB SRAM). Regulatory compliance (IEC 62443) requires that only authenticated firmware and models can execute on the device. An attacker who gains physical access must not be able to replace the model with a malicious one. Design the secure boot chain."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Enable Secure Boot and sign the firmware." This protects the firmware but not the model weights, which are stored separately in flash. An attacker could replace the model weights (e.g., a model that always reports "normal rhythm") without touching the signed firmware, and the device would boot successfully with a compromised model.

  **Realistic Solution:** The secure boot chain must cover every executable and data component:

  **Boot chain (4 stages):**

  (1) **ROM bootloader (immutable, in silicon)** — STM32U5's built-in secure boot ROM. Verifies the hash of the first-stage bootloader against a value burned into OTP (One-Time Programmable) fuses. Cannot be modified by software. If verification fails: device halts (no boot).

  (2) **First-stage bootloader (32 KB, in secure flash)** — runs in TrustZone Secure World. Holds the RSA-2048 public key (256 bytes). Verifies the signature of the application firmware: computes SHA-256 hash of the firmware region, then verifies the RSA-2048 signature (stored in the last 256 bytes of the firmware slot) against the public key.

  (3) **Application firmware (verified)** — before loading the model, computes SHA-256 of the model weights region and verifies against a signed model manifest (hash + RSA signature, stored in a protected flash page). This ensures the model hasn't been tampered with independently of the firmware.

  (4) **Runtime integrity** — periodically (every 1000 inferences), re-hash a random 4 KB page of the model weights and compare against the stored hash. This detects runtime flash corruption or fault-injection attacks that modify weights after boot.

  **RSA-2048 verification cost on Cortex-M33 at 160 MHz:**
  RSA-2048 signature verification (modular exponentiation with e=65537) requires ~30 million cycles on a Cortex-M33 without hardware crypto. At 160 MHz: 30M / 160M = **187 ms**. With STM32U5's PKA (Public Key Accelerator): ~5 million cycles = **31 ms**.

  SHA-256 of 500 KB firmware+model: 500,000 × 15 cycles/byte = 7.5M cycles = **47 ms** (software) or **12 ms** (with HASH peripheral).

  **Total secure boot time:** SHA-256 (12 ms) + RSA verify (31 ms) + model hash (8 ms) + model RSA verify (31 ms) = **82 ms** with hardware acceleration, **280 ms** without. Acceptable for a device that boots once and runs for months.

  > **Napkin Math:** Boot time budget: 82 ms (with HW crypto) or 280 ms (SW only). Flash overhead: 256 B RSA signature per firmware slot + 256 B per model slot + 32 B SHA-256 hash per model = 544 bytes. Key storage: 256 B public key in secure OTP. Runtime integrity check: SHA-256 of 4 KB page = 4,096 × 15 / 160M = 0.38 ms every 1000 inferences. At 1 inference/second: 0.38 ms / 1000 s = 0.00004% CPU overhead. Attack cost to bypass: requires extracting the private key (stored only on the signing server, never on the device) or finding a SHA-256 collision (2¹²⁸ operations — infeasible).

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>


---


### 📎 Additional Topics


#### 🟢 L3 — Recall & Define

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The MRAM Wear Illusion</b> · <code>storage</code> <code>mlops</code></summary>

- **Interviewer:** "You switch from traditional SPI Flash to an external MRAM (Magnetoresistive RAM) chip to store telemetry. MRAM is famous for essentially infinite write endurance. You write your ML logs continuously in a tight loop. A year later, the MRAM chip starts returning corrupted bits. If MRAM doesn't wear out like Flash, why did it fail?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "MRAM has a limited cycle count just like Flash." MRAM cycle counts are practically infinite (10^14 cycles), it shouldn't wear out in a year.

  **Realistic Solution:** You fell victim to **Write Endurance vs. Data Retention Trade-offs**.

  While MRAM does not suffer from dielectric breakdown like Flash memory, it is heavily susceptible to thermal energy and continuous magnetic disturbance.

  If you write to the exact same memory cells millions of times a second without pause, the localized heating and continuous magnetic tunneling currents can physically destabilize the magnetic orientation of adjacent cells (similar to Rowhammer in DRAM, but magnetic).

  Furthermore, many cheaper MRAM chips promise "infinite" endurance only if the ambient temperature is tightly controlled. In a hot industrial environment, the magnetic states naturally degrade over time (Data Retention failure), which is accelerated by constant writing.

  **The Fix:** Even with MRAM, you should still implement basic wear-leveling (ring buffers) to distribute the localized thermal and magnetic stress across the entire silicon die, rather than hammering address `0x00` infinitely.

  > **Napkin Math:** 10^14 writes is massive, but at 1 million writes per second (tight C loop), you hit 10^14 in about 3.1 years. If you hammer one address, you can absolutely wear out MRAM.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)
  </details>
</details>


#### 🔵 L4 — Apply & Identify

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Flash Wear-Leveling Blindspot</b> · <code>storage</code> <code>mlops</code></summary>

- **Interviewer:** "Your edge sensors log anomaly data to internal Flash. To prevent wearing out the Flash (which has a 10,000 cycle limit), you write a script to always save logs starting at memory address 0x08000000, and sequentially move forward to 0x08040000 before looping back. After a year, the system crashes because the flash sector at 0x08000000 is physically destroyed. Why didn't your sequential logging work as wear-leveling?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "You didn't make the loop big enough." The size of the loop isn't the primary failure mode; it's how Flash physics requires data to be updated.

  **Realistic Solution:** You ignored **Flash Page Erase Granularity**.

  You can write (program) bits in Flash from 1 to 0 sequentially. But you cannot flip a 0 back to a 1 without **erasing an entire sector/page** at once.
  If your microcontroller's flash sector size is 16 KB, and you write 100 bytes of logs sequentially into that sector, you eventually fill the 16 KB. To write the 16,001st byte, you must erase the *entire* 16 KB sector.

  Your script looped through the memory, but every time it looped back to 0x08000000, it had to issue an Erase command on Sector 0. If you log frequently, Sector 0 absorbs massive amounts of Erase cycles (which is what physically destroys the silicon) while the rest of the memory space might remain lightly used.

  **The Fix:** Never write raw Flash management code yourself. Use a proper **Flash Translation Layer (FTL)** or an embedded filesystem designed for flash (like LittleFS or SPIFFS). These libraries abstract the physical addresses and automatically map logical writes to different physical sectors to ensure perfect, even wear-leveling across the entire chip.

  > **Napkin Math:** If you log 64 bytes a minute, a 16 KB sector fills in 256 minutes (~4.2 hours). You are erasing that sector 5.6 times a day. 5.6 erases * 365 days = 2,044 erase cycles per year. The flash will die in roughly 4.8 years.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Flash Wear-Leveling Blindspot</b> · <code>storage</code> <code>mlops</code></summary>

- **Interviewer:** "Your edge sensors log anomaly data to internal Flash. To prevent wearing out the Flash (which has a 10,000 cycle limit), you write a script to always save logs starting at memory address 0x08000000, and sequentially move forward to 0x08040000 before looping back. After a year, the system crashes because the flash sector at 0x08000000 is physically destroyed. Why didn't your sequential logging work as wear-leveling?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "You didn't make the loop big enough." The size of the loop isn't the primary failure mode; it's how Flash physics requires data to be updated.

  **Realistic Solution:** You ignored **Flash Page Erase Granularity**.

  You can write (program) bits in Flash from 1 to 0 sequentially. But you cannot flip a 0 back to a 1 without **erasing an entire sector/page** at once.
  If your microcontroller's flash sector size is 16 KB, and you write 100 bytes of logs sequentially into that sector, you eventually fill the 16 KB. To write the 16,001st byte, you must erase the *entire* 16 KB sector.

  Your script looped through the memory, but every time it looped back to 0x08000000, it had to issue an Erase command on Sector 0. If you log frequently, Sector 0 absorbs massive amounts of Erase cycles (which is what physically destroys the silicon) while the rest of the memory space might remain lightly used.

  **The Fix:** Never write raw Flash management code yourself. Use a proper **Flash Translation Layer (FTL)** or an embedded filesystem designed for flash (like LittleFS or SPIFFS). These libraries abstract the physical addresses and automatically map logical writes to different physical sectors to ensure perfect, even wear-leveling across the entire chip.

  > **Napkin Math:** If you log 64 bytes a minute, a 16 KB sector fills in 256 minutes (~4.2 hours). You are erasing that sector 5.6 times a day. 5.6 erases * 365 days = 2,044 erase cycles per year. The flash will die in roughly 4.8 years.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>


#### 🟡 L5 — Analyze & Predict

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Continuous Logging Flash Death</b> · <code>storage</code> <code>mlops</code></summary>

- **Interviewer:** "Your smart thermostat logs the room temperature and the ML model's occupancy prediction to internal Flash memory every 5 minutes for user analytics. You are using a standard SPIFFS filesystem. The internal flash has a 10,000 cycle erase limit. A year later, 15% of the devices are permanently bricked. How did logging 20 bytes every 5 minutes destroy the Flash?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "20 bytes * 105,000 logs a year is only 2 MB. It didn't fill up the drive." It didn't fill up the drive, it destroyed the silicon via Write Amplification.

  **Realistic Solution:** You fell victim to **Write Amplification and Erase Granularity**.

  Flash memory cannot overwrite a 0 to a 1 without erasing an entire "Sector" (usually 4 KB or 16 KB).
  When you append 20 bytes to a log file, the filesystem (even flash-aware ones like SPIFFS) eventually fills up a sector. To write the next 20 bytes, it must find a new sector, or erase an old one.

  Because the filesystem also has to update metadata (the file size, the directory index, the wear-leveling headers) for *every single 20-byte write*, it is constantly erasing and rewriting the metadata sectors. You are effectively performing a 4 KB erase cycle just to save 20 bytes of data.

  **The Fix:** Never log high-frequency telemetry synchronously to internal Flash.
  1. Buffer the 20-byte logs in SRAM (RAM has infinite endurance).
  2. Only write to Flash once a day, or when the SRAM buffer hits 4 KB, ensuring you only trigger one Erase cycle per 4 KB of actual payload data.

  > **Napkin Math:** 1 log every 5 mins = 288 logs/day. If each log updates a metadata sector, that sector is erased 288 times a day. 288 * 365 = 105,120 erase cycles per year. The Flash is rated for 10,000 cycles. You physically destroyed the silicon in exactly 34 days.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

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

  📖 **Deep Dive:** [Volume I: Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>


#### 🔴 L6+ — Synthesize & Derive

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 6+" align="center"> TinyML Federated Learning System</b> · <code>training</code> <code>mlops</code></summary>

- **Interviewer:** "You have a fleet of 10,000 smart electricity meters, each with a Cortex-M4F (256 KB SRAM, 1 MB flash, 168 MHz) and a LoRaWAN radio (250 bps effective throughput after duty cycle limits). Each meter runs a load forecasting model (2-layer LSTM, 15 KB weights) that predicts next-hour consumption. After 1 year, the model has drifted because consumer behavior changed (more EVs, more solar panels). Design a federated learning system that retrains the model across the fleet using LoRaWAN's extreme bandwidth constraints."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run standard FedAvg — upload gradients via LoRaWAN." At 250 bps, uploading 15 KB of gradients takes 15,000 × 8 / 250 = 480 seconds = 8 minutes per device. With LoRaWAN duty cycle limits (1% in EU868): actual airtime = 8 min / 0.01 = 800 minutes = **13.3 hours** per device per round. For 10,000 devices: the gateway can handle ~100 devices per hour. One round takes 100 hours = 4.2 days. 20 rounds to convergence: **84 days**. Unacceptable.

  **Realistic Solution:** Extreme compression is mandatory. The communication constraint drives every design decision.

  **(1) On-device training.** The M4F can train a 2-layer LSTM (15 KB weights) using backpropagation through time (BPTT). Training data: last 7 days of hourly readings = 168 samples × 10 features × 4 bytes = 6.7 KB. Forward pass: 50K MACs × 168 time steps = 8.4M MACs → 25ms on M4F with CMSIS-NN. Backward pass: ~3× forward = 75ms. Per epoch: 100ms. 10 epochs: 1 second. Memory: weights (15 KB) + gradients (15 KB) + activations (5 KB) + training data (6.7 KB) + optimizer state (15 KB for Adam) = **57 KB**. Fits in 256 KB SRAM with 199 KB to spare.

  **(2) Extreme gradient compression.** After local training, compute weight delta (new − old). Apply: (a) Top-k sparsification: keep only top 0.1% of deltas = 15 values (indices + values). (b) Quantize delta values to INT8: 15 × (2 bytes index + 1 byte value) = **45 bytes**. (c) LoRaWAN transmission: 45 bytes at 250 bps = 1.44 seconds airtime. With 1% duty cycle: 144 seconds = 2.4 minutes per device. Gateway capacity: ~25 devices per hour. 10,000 devices / 25 = 400 hours per round? No — use multiple gateways (typical LoRaWAN deployment: 1 gateway per 1,000 devices). 10 gateways: 40 hours per round.

  **(3) Aggregation.** The server receives 45-byte sparse deltas from each device. Reconstruct the full gradient by accumulating sparse updates. With 10,000 devices each contributing 15 non-zero deltas: 150,000 updates across 15,000 parameters → each parameter gets ~10 updates on average. Aggregate via weighted mean. Download the updated model: 15 KB, but only send the delta (also sparse): ~45 bytes. Round-trip per device: 90 bytes.

  **(4) Convergence.** With 0.1% sparsification, convergence is slower: ~50 rounds (vs 20 for full gradients). But each round is 40 hours. Total: 50 × 40 = 2,000 hours = **83 days**. Still slow, but this is a background process — the existing model continues serving predictions while retraining happens. After 83 days, the fleet has a model adapted to the new consumption patterns.

  **(5) Practical optimization.** Don't train all 10,000 devices every round. Sample 500 devices (5%) per round. Each round: 500 / 25 per gateway per hour / 10 gateways = 2 hours. 50 rounds × 2 hours = **100 hours = 4.2 days**. Much more practical. The sampling introduces variance but with 500 devices, the gradient estimate is statistically robust.

  > **Napkin Math:** Per-device upload: 45 bytes (0.1% sparse INT8 delta). LoRaWAN airtime: 1.44s (within single-packet limit of 51 bytes for SF7). Duty cycle: 1.44s / 0.01 = 144s between transmissions. Training compute: 1 second on M4F. Training energy: 1s × 30 mW = 30 mJ. LoRa TX energy: 1.44s × 100 mW = 144 mJ. Total per round: 174 mJ. Battery impact (if battery-powered, 3,000 mAh × 3.3V = 9.9 Wh = 35,640 J): 50 rounds × 174 mJ = 8.7 J = 0.024% of battery. Negligible. The extreme compression makes federated learning viable even on LoRaWAN.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

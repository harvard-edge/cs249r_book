# Numbers Every ML Systems Engineer Should Know

<div align="center">
  <a href="README.md">🏠 Playbook Home</a> ·
  <a href="cloud/README.md">☁️ Cloud</a> ·
  <a href="edge/README.md">🤖 Edge</a> ·
  <a href="mobile/README.md">📱 Mobile</a> ·
  <a href="tinyml/README.md">🔬 TinyML</a>
</div>

---

Adapted from the textbook's [Machine Foundations](https://mlsysbook.ai/vol1/) appendix. **Memorize the ratios; they're physics. Use the absolute numbers as sanity checks.** All hardware values sourced from [`mlsysim/core/constants.py`](../mlsysim/core/constants.py), the single source of truth for the book.

---

## 🪜 The Scale Ladder

The defining characteristic of ML Systems Engineering is that the physics change by orders of magnitude depending on where you deploy.

| Dimension | Cloud | Edge | Mobile | TinyML | Cloud:TinyML Ratio |
| --------- | ----- | ---- | ------ | ------ | ------------------ |
| **Compute** | ~1,000 TFLOPS | ~100 TOPS | ~30 TOPS | ~100 MFLOPS | 10,000,000× |
| **Memory** | 80 GB HBM | 8–32 GB DRAM | 6–12 GB shared | 256 KB–2 MB SRAM | 40,000× |
| **Power** | 700 W | 30 W | 5 W | 10 mW | 70,000× |
| **Latency Budget** | 100ms (P99) | 33ms (hard RT) | 16ms (jank) | 1ms (interrupt) | 100× |

---

## 1. The Invariants (Physics — Will Not Change)

<table>
  <thead>
    <tr>
      <th width="35%">Relationship</th>
      <th width="25%">Ratio</th>
      <th width="40%">Why it's stable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>DRAM access vs FP16 compute</b></td>
      <td><b>~580×</b> more energy</td>
      <td>Wire capacitance scales with distance</td>
    </tr>
    <tr>
      <td><b>FP32 vs INT8 energy</b></td>
      <td>~18×</td>
      <td>Bit width determines switching energy</td>
    </tr>
    <tr>
      <td><b>FP32 vs FP16 energy</b></td>
      <td>~3.4×</td>
      <td>Halving bits roughly halves energy</td>
    </tr>
    <tr>
      <td><b>HBM vs L1 cache latency</b></td>
      <td>~300× slower</td>
      <td>On-chip vs off-chip</td>
    </tr>
    <tr>
      <td><b>SSD vs L1 cache latency</b></td>
      <td>~100,000× slower</td>
      <td>Electrical vs flash</td>
    </tr>
    <tr>
      <td><b>Network vs local memory</b></td>
      <td>~17× slower</td>
      <td>Speed of light + switching</td>
    </tr>
    <tr>
      <td><b>Light in fiber</b></td>
      <td>~200 km/ms</td>
      <td>Cross-country US ≈ 40ms RTT</td>
    </tr>
  </tbody>
</table>

---

## 2. Scaling Rules (Arithmetic — Hardware Independent)

These formulas let you estimate memory, compute, and power requirements from basic model specs.

### Cloud / LLM
| What you're estimating | Formula | Example |
|---|---|---|
| **Inference memory (FP16)** | 2 bytes × params | 7B params × 2 bytes = **14 GB** |
| **Inference memory (INT8)** | 1 byte × params | 7B params × 1 byte = **7 GB** |
| **Training memory (Adam)** | **16 bytes × params** | 7B params × 16 bytes = **112 GB** |
| **Inference compute** | ~2 FLOPs × params per token | 7B → **~14 GFLOPs/token** |
| **Training compute** | ~6 FLOPs × params × tokens | 7B on 1T tokens → **4×10²² FLOPs** |
| **KV-cache per token** | 2 × layers × heads × head_dim × 2 bytes | Llama 70B, 128k tokens → **~335 GB** |

### Edge / Vision
| What you're estimating | Formula | Example |
|---|---|---|
| **Activation memory** | $H \times W \times C \times \text{batch} \times \text{bytes}$ | 640×640×32×1×2 = **~26 MB** |
| **FPS budget** | $1000\text{ms} / \text{frame\_deadline\_ms}$ | 33ms deadline = **30 FPS** |
| **Sustained TOPS** | $\text{TOPS/W} \times \text{thermal\_budget\_W}$ | 4.6 TOPS/W × 15W = **69 TOPS** |

### Mobile
| What you're estimating | Formula | Example |
|---|---|---|
| **App memory budget** | $\text{device\_RAM} \times 0.25$ | 8 GB RAM × 0.25 = **2 GB max** |
| **NPU delegation ratio** | $\text{supported\_ops} / \text{total\_ops}$ | 85/100 ops = **85% delegated** |
| **Battery drain** | $P_{\text{inference}} \times \text{duty\_cycle} \times \text{time}$ | 2W × 0.05 × 1 hr = **0.1 Wh** |

### TinyML
| What you're estimating | Formula | Example |
|---|---|---|
| **Tensor arena peak** | $\max(\text{concurrent activation sizes})$ | Layer 3: 40KB + 20KB = **60KB peak** |
| **Flash budget** | $\text{Total} - \text{Bootloader} - \text{RTOS} - \text{OTA}$ | 1MB - 32K - 64K - 450K = **454KB** |
| **Duty cycle energy** | $(P_{\text{active}} t_{\text{active}} + P_{\text{sleep}} t_{\text{sleep}}) / t_{\text{period}}$ | (10mW×1s + 10µW×9s)/10s = **1mW avg** |

---

## 3. Current Hardware Snapshots (2024–2025)

### ☁️ Cloud (Data Center)
| Category | Metric | Value |
|---|---|---|
| **Compute** | H100 FP16 Tensor Core | 989 TFLOPS |
| | B200 FP16 Tensor Core | 2,250 TFLOPS |
| **Memory BW** | H100 HBM3 | 3.35 TB/s |
| | B200 HBM3e | 8.0 TB/s |
| **Interconnect**| NVLink 4.0 (H100) | 900 GB/s |
| | InfiniBand NDR | 400 Gbps (50 GB/s) |
| **Ridge Point** | H100 (FP16) | ~295 Ops/Byte |
| **Power** | H100 TDP | 700 W |
| **Latency** | HBM3 | ~300 ns |

### 🤖 Edge (Autonomous & Industrial)
| Category | Metric | Value |
|---|---|---|
| **Compute** | Jetson AGX Orin (INT8) | 275 TOPS |
| | Hailo-8 (INT8) | 26 TOPS |
| **Memory BW** | Jetson AGX Orin (LPDDR5) | 204.8 GB/s |
| | Hailo-8 (On-chip SRAM) | ~2.5 TB/s |
| **Interconnect**| MIPI CSI-2 (Camera) | ~2.5 GB/s (4-lane) |
| **Ridge Point** | Jetson AGX Orin (INT8) | ~1,342 Ops/Byte |
| **Power** | Jetson AGX Orin | 15W – 60W |
| | Hailo-8 | 2.5W |
| **Latency** | LPDDR5 | ~100 ns |

### 📱 Mobile (Smartphones)
| Category | Metric | Value |
|---|---|---|
| **Compute** | Apple A17 Pro (ANE) | 35 TOPS |
| | Snapdragon 8 Gen 3 (Hexagon) | 45 TOPS |
| **Memory BW** | Apple A17 Pro (LPDDR5) | 51.2 GB/s |
| | Snapdragon 8 Gen 3 (LPDDR5x) | 77 GB/s |
| **Interconnect**| On-chip NoC | ~100 GB/s |
| **Ridge Point** | Apple A17 Pro (INT8) | ~683 Ops/Byte |
| **Power** | Total SoC Active | 3W – 5W |
| **Latency** | UFS 4.0 Flash Read | ~4.2 GB/s |

### 🔬 TinyML (Microcontrollers)
| Category | Metric | Value |
|---|---|---|
| **Compute** | Cortex-M4 (168 MHz) | ~336 MFLOPS |
| | Cortex-M7 (480 MHz) | ~960 MFLOPS |
| **Memory BW** | On-chip SRAM | ~1.2 GB/s |
| **Interconnect**| SPI / I2C | 10 Mbps / 400 Kbps |
| **Ridge Point** | Cortex-M4 | ~0.2 Ops/Byte |
| **Power** | Active (Cortex-M4) | ~10 mW – 50 mW |
| | Sleep (Deep) | ~1 µW – 10 µW |
| **Latency** | Flash Read | ~50 ns |

> **Source:** All values from the textbook's `constants.py`. When hardware generations change, update the constants file and every calculation in the book (and this playbook) updates automatically.

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

## The Invariants (Physics — Will Not Change)

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

## Scaling Rules (Arithmetic — Hardware Independent)

These formulas let you estimate memory, compute, and cache requirements from a model's parameter count. "7B" means a model with 7 billion parameters.

<table>
  <thead>
    <tr>
      <th width="35%">What you're estimating</th>
      <th width="35%">Formula</th>
      <th width="30%">Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Inference memory (FP16)</b></td>
      <td>2 bytes × params</td>
      <td>7B params × 2 bytes = <b>14 GB</b></td>
    </tr>
    <tr>
      <td><b>Inference memory (INT8)</b></td>
      <td>1 byte × params</td>
      <td>7B params × 1 byte = <b>7 GB</b></td>
    </tr>
    <tr>
      <td><b>Training memory (Adam, FP16+FP32)</b></td>
      <td><b>16 bytes × params</b></td>
      <td>7B params × 16 bytes = <b>112 GB</b></td>
    </tr>
    <tr>
      <td><b>Inference compute (transformer)</b></td>
      <td>~2 FLOPs × params per token</td>
      <td>7B → ~<b>14 GFLOPs/token</b></td>
    </tr>
    <tr>
      <td><b>Training compute</b></td>
      <td>~6 FLOPs × params × tokens</td>
      <td>7B on 1T tokens → <b>4×10²² FLOPs</b></td>
    </tr>
    <tr>
      <td><b>KV-cache per token (all layers)</b></td>
      <td>2 × layers × heads × head_dim × 2 bytes</td>
      <td>Llama 70B, 128k tokens → <b>~335 GB</b></td>
    </tr>
  </tbody>
</table>

---

## Current Hardware Snapshot (2024–2025)

<table>
  <thead>
    <tr>
      <th width="20%">Category</th>
      <th width="35%">Metric</th>
      <th width="45%">Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Compute</b></td>
      <td>A100 FP16 Tensor Core</td>
      <td>312 TFLOPS</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 FP16 Tensor Core</td>
      <td>989 TFLOPS</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 FP8 Tensor Core</td>
      <td>1,979 TFLOPS</td>
    </tr>
    <tr>
      <td></td>
      <td>B200 FP16 Tensor Core</td>
      <td>2,250 TFLOPS</td>
    </tr>
    <tr>
      <td><b>Memory BW</b></td>
      <td>A100 HBM2e</td>
      <td>2.0 TB/s</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 HBM3</td>
      <td>3.35 TB/s</td>
    </tr>
    <tr>
      <td></td>
      <td>B200 HBM3e</td>
      <td>8.0 TB/s</td>
    </tr>
    <tr>
      <td><b>Interconnect</b></td>
      <td>NVLink 4.0 (H100)</td>
      <td>900 GB/s</td>
    </tr>
    <tr>
      <td></td>
      <td>NVLink 5.0 (B200)</td>
      <td>1,800 GB/s</td>
    </tr>
    <tr>
      <td></td>
      <td>InfiniBand NDR</td>
      <td>400 Gbps (50 GB/s)</td>
    </tr>
    <tr>
      <td></td>
      <td>PCIe Gen5 x16</td>
      <td>64 GB/s</td>
    </tr>
    <tr>
      <td><b>Roofline Ridge</b></td>
      <td>A100 (FP16)</td>
      <td>~153 Ops/Byte</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 (FP16)</td>
      <td>~295 Ops/Byte</td>
    </tr>
    <tr>
      <td><b>Power</b></td>
      <td>A100 TDP</td>
      <td>400 W</td>
    </tr>
    <tr>
      <td></td>
      <td>H100 TDP</td>
      <td>700 W</td>
    </tr>
    <tr>
      <td></td>
      <td>B200 TDP</td>
      <td>1,000 W</td>
    </tr>
    <tr>
      <td><b>Latency</b></td>
      <td>L1 / Register</td>
      <td>~1 ns</td>
    </tr>
    <tr>
      <td></td>
      <td>L2 Cache</td>
      <td>~4 ns</td>
    </tr>
    <tr>
      <td></td>
      <td>HBM3</td>
      <td>~300 ns</td>
    </tr>
    <tr>
      <td></td>
      <td>PCIe Gen5</td>
      <td>~1 μs</td>
    </tr>
    <tr>
      <td></td>
      <td>InfiniBand</td>
      <td>~5 μs</td>
    </tr>
    <tr>
      <td></td>
      <td>NVMe SSD</td>
      <td>~100 μs</td>
    </tr>
  </tbody>
</table>

> **Source:** All values from the textbook's `constants.py`. When hardware generations change, update the constants file and every calculation in the book (and this playbook) updates automatically.

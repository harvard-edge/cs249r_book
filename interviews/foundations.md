# ML Systems Foundations (L1–L2)

<div align="center">
  <a href="README.md">🏠 Home</a> ·
  <a href="NUMBERS.md">📊 Numbers</a> ·
  <a href="cloud/README.md">☁️ Cloud</a> ·
  <a href="edge/README.md">🤖 Edge</a> ·
  <a href="mobile/README.md">📱 Mobile</a> ·
  <a href="tinyml/README.md">🔬 TinyML</a>
</div>

---

*The Physics Literacy of ML Systems*

Memory ratios, hardware constants, and single-variable napkin math. If you don't know these, you can't design the system.

---

### 🟢 L1 — Foundations (Memorization & Recall)

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The HBM vs L1 Latency Gap</b> · <code>memory-hierarchy</code></summary>

- **Interviewer:** "Roughly how much slower is accessing HBM3 memory compared to an L1 register read?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Assuming the gap is small, like 10×. In reality, crossing the physical distance from the compute core to the HBM stacks is a massive latency event.

  **Realistic Solution:** ~300× slower. L1 registers are ~1ns, while HBM3 access is ~300ns.

  > **Napkin Math:** If an L1 read was 1 second, an HBM read would be 5 minutes.

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The Energy Tax of Data Movement</b> · <code>energy</code></summary>

- **Interviewer:** "Which operation consumes more energy: performing an FP16 multiply-add or reading the operands from DRAM?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Thinking that "Compute" is the expensive part. Modern silicon has made math incredibly cheap; moving bits across wires is the real cost.

  **Realistic Solution:** Reading from DRAM. Accessing DRAM consumes ~580× more energy than a single FP16 compute operation.

  > **Napkin Math:** $\text{Energy}_{\text{DRAM}} \approx 580 \times \text{Energy}_{\text{Compute}}$.

  📖 **Deep Dive:** [Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)
  </details>
</details>

---

### 🔵 L2 — Analytical (Single-Variable Math)

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The FP16 Model Footprint</b> · <code>memory</code></summary>

- **Interviewer:** "A model has 7 billion parameters. How much VRAM does it occupy just to load the weights in FP16 precision?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Forgetting that each parameter in FP16 takes 2 bytes, not 1.

  **Realistic Solution:** 14 GB.

  > **Napkin Math:** 7B parameters × 2 bytes/parameter (FP16) = 14 GB.

  📖 **Deep Dive:** [Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Ridge Point Logic</b> · <code>roofline</code></summary>

- **Interviewer:** "If an accelerator has 1,000 TFLOPS of compute and 2 TB/s of memory bandwidth, what is its ridge point?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Mixing units or assuming the ridge point is a fixed number across all hardware. It is a ratio of compute to bandwidth.

  **Realistic Solution:** 500 Ops/Byte.

  > **Napkin Math:** 1,000 TFLOPS / 2,000 GB/s = 500 FLOPs per Byte.

  📖 **Deep Dive:** [Benchmarking](https://harvard-edge.github.io/cs249r_book_dev/contents/benchmarking/benchmarking.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The FP16 vs INT8 Precision Choice</b> · <code>precision</code></summary>

- **Interviewer:** "During training, we typically use FP16 or BF16. For inference on edge devices, we often use INT8. Why do we move to 8-bit integers for deployment?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Thinking INT8 is "more accurate." It is less accurate but much faster.

  **Realistic Solution:** Throughput and Energy. 8-bit integers occupy half the memory of 16-bit floats, doubling the effective memory bandwidth. Additionally, INT8 math is significantly more energy-efficient and often has higher peak throughput on specialized NPUs.

  > **Napkin Math:** INT8 uses 50% less memory and 2-4x less energy than FP16.

  📖 **Deep Dive:** [Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The FLOPS vs Time Calculation</b> · <code>compute</code></summary>

- **Interviewer:** "An operation requires 10 Teraflops ($10^{13}$ operations). If your GPU has a peak performance of 100 TFLOPS, what is the theoretical minimum time to finish this operation?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Getting the decimal point wrong. $10/100 = 0.1$.

  **Realistic Solution:** 100 milliseconds (0.1 seconds).

  > **Napkin Math:** $\text{Time} = \frac{10 \text{ TFLOPS}}{100 \text{ TFLOPS/sec}} = 0.1 \text{ seconds}$.

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The Battery Drain Math</b> · <code>energy</code></summary>

- **Interviewer:** "A mobile model consumes 2 Watts during inference. If your phone battery has 15 Watt-hours of capacity, how many hours of continuous inference could you theoretically run?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Confusing Watts (Power) with Watt-hours (Energy).

  **Realistic Solution:** 7.5 hours.

  > **Napkin Math:** $15 \text{ Wh} / 2 \text{ W} = 7.5 \text{ hours}$.

  📖 **Deep Dive:** [Edge Intelligence](https://harvard-edge.github.io/cs249r_book_dev/contents/edge_intelligence/edge_intelligence.html)
  </details>
</details>

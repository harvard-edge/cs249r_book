# Mission Plan: lab_11_hw_accel

## 1. Chapter Alignment

- **Chapter:** Hardware Acceleration (`@sec-hardware-acceleration`)
- **Core Invariant:** The **Roofline Model** — attainable performance is $\min(\text{Peak Compute}, \text{Peak Bandwidth} \times \text{Arithmetic Intensity})$. The ridge point ($R_{\text{peak}} / BW$) divides all workloads into compute-bound (above) and memory-bound (below). A 1000x faster processor provides zero speedup for a memory-bound workload because the bottleneck was never computation. DRAM access costs 100x or more the energy of on-chip arithmetic.
- **Central Tension:** Students believe "faster chip = faster model" — that upgrading hardware uniformly accelerates every workload. The chapter's roofline analysis demolishes this: ResNet-50 convolutions sit above the ridge point (compute-bound, benefits from more TFLOPS), while GPT-2 attention layers sit below it (memory-bound, benefits only from more bandwidth). On the same H100, ResNet achieves ~20x system speedup while GPT-2 achieves only ~5x, despite identical hardware. The 500x raw speedup is capped by Amdahl's Law and the memory wall. Students must learn to diagnose *which* ceiling their workload hits before choosing an optimization strategy.
- **Target Duration:** 35–40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict the system speedup of moving a GPT-2 inference workload from CPU to H100 GPU. Most expect 100x–500x (matching the raw hardware advantage). The chapter's Amdahl analysis shows the actual speedup is only ~5x, because GPT-2's serial fraction (20% — KV-cache updates, sampling, Python overhead) caps the theoretical ceiling at 5x regardless of hardware speed. Students place workloads on the Roofline for the first time and discover that the memory-bandwidth slope, not the compute ceiling, determines GPT-2's performance. This is the first lab where students use the RooflineVisualizer instrument.

**Act 2 (Design Challenge, 23 min):** Students must diagnose and optimize two workloads on two devices: a compute-bound workload (ResNet-50 convolution) and a memory-bound workload (GPT-2 attention). On the H100, the challenge is to achieve peak utilization by matching arithmetic intensity to the ridge point. On the Jetson Orin NX, the lower bandwidth makes *every* workload more memory-bound, shifting the ridge point and revealing that the same operation can be compute-bound on one device and memory-bound on another. Students must find the batch size that moves a workload from the bandwidth slope to the compute ceiling, and discover that the energy cost of DRAM access (100x+ vs. on-chip SRAM) makes the memory hierarchy the dominant factor in total system efficiency.

---

## 3. Act 1: The Amdahl Surprise (Calibration — 12 minutes)

### Pedagogical Goal
Students believe that a 500x faster GPU should give a ~500x system speedup. The chapter's Amdahl's Law analysis on H100 shows that GPT-2 token generation, with only 80% parallelizable fraction (p = 0.80), achieves a maximum speedup of 1/(1-0.80) = 5x regardless of hardware speed. Even an infinitely fast accelerator cannot exceed 5x. Meanwhile, ResNet-50 at p = 0.95 achieves ~20x. The gap between 20x and 5x on identical hardware is entirely due to the serial fraction — not the model's size, accuracy, or complexity. Students discover that the "boring" serial bottleneck (data loading, KV-cache, Python overhead) determines more about system performance than the "exciting" accelerator.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "You move GPT-2 token generation from a CPU to an NVIDIA H100 GPU (500x raw speedup for matrix multiply). The workload is 80% parallelizable matrix operations and 20% serial overhead (KV-cache updates, sampling, Python dispatch). What is the total system speedup?"

Options:
- A) ~100x — the GPU accelerates the majority of the workload
- B) ~50x — serial overhead reduces gains but the GPU dominates
- **C) ~5x — Amdahl's Law caps the speedup at $1/(1-p) = 1/0.20 = 5\times$ regardless of GPU speed** (correct)
- D) ~500x — the GPU speedup applies uniformly to the entire pipeline

Common wrong answer: **A** — students apply the 500x speedup to the 80% parallel fraction and expect $0.80 \times 500 + 0.20 \times 1 \approx 400$x, not understanding the harmonic-mean nature of Amdahl's formula.

Why wrong: Amdahl's Law is $1/((1-p) + p/S)$, not $p \times S + (1-p)$. At $p = 0.80$ and $S = 500$: $1/(0.20 + 0.80/500) = 1/(0.20 + 0.0016) = 4.96\times$. The 20% serial fraction completely dominates — the 0.80/500 parallel term is negligible.

### The Instrument: Roofline Visualizer (First Introduction)

This is the **first lab** where students encounter the RooflineVisualizer. The instrument is a **log-log plot** following Williams, Waterman, and Patterson (2009):

- **X-axis (log):** Arithmetic Intensity (FLOP/byte), range 0.1 to 1000
- **Y-axis (log):** Attainable Performance (TFLOPS), range 0.01 to 2000
- **Roofline shape:** Bandwidth slope (diagonal line from origin) meets compute ceiling (horizontal line) at the ridge point
- **Ridge point annotation:** Labeled with exact value ($R_{\text{peak}} / BW$)

**Workload dots:**
- **ResNet-50 Conv layer (BlueLine dot):** AI ~100–200 FLOP/byte — sits above ridge point (compute-bound)
- **GPT-2 Attention layer (RedLine dot):** AI ~5–10 FLOP/byte — sits below ridge point (memory-bound)
- **ReLU/LayerNorm (gray dot):** AI ~0.5 FLOP/byte — deep in memory-bound territory

Controls:
- **Hardware selector** (H100 / Jetson Orin NX): Changes the roofline shape — H100 has higher ceiling and higher ridge point; Jetson has lower ceiling and potentially different ridge point
- **Amdahl's Law panel:** Slider for parallel fraction $p$ (0.50–0.99, default 0.80) and hardware speedup $S$ (10–1000, default 500). Shows computed system speedup with formula.

**Secondary:** Amdahl heatmap matching @fig-iron-law-heatmap — color-coded speedup as function of $p$ and $S$, with student's current selection highlighted.

### The Reveal
After interaction:
> "You predicted [X]x system speedup for GPT-2 on H100. The actual value is **~5x**. Amdahl's Law: $1/(0.20 + 0.80/500) = 4.96\times$. On the Roofline, GPT-2 attention sits at ~5–10 FLOP/byte, well below the H100's ridge point of ~295 FLOP/byte. This means GPT-2 is memory-bound: its performance is determined by HBM bandwidth (3.35 TB/s), not by the 989 TFLOPS compute ceiling. Doubling H100's TFLOPS to 2000 would give zero additional speedup. Doubling bandwidth to 6.7 TB/s would nearly double throughput."

### Reflection (Structured)
Four-option multiple choice:

> "ResNet-50 achieves ~20x system speedup on H100 while GPT-2 achieves only ~5x on the same hardware. Both use identical GPUs. What explains the 4x gap?"

- A) ResNet-50 has fewer parameters and fits better in GPU memory
- B) ResNet-50 uses convolutional operations which are inherently faster than attention
- **C) ResNet-50 has a higher parallelizable fraction (p = 0.95 vs. 0.80) and its convolutions have high arithmetic intensity (compute-bound), while GPT-2's attention is memory-bound with higher serial overhead** (correct)
- D) GPT-2 requires more numerical precision, reducing the effective TFLOPS

### Math Peek (collapsible)
$$\text{Speedup} = \frac{1}{(1-p) + \frac{p}{S}} \qquad \text{(Amdahl's Law, @eq-amdahl)}$$
$$\text{Attainable Performance} = \min(R_{\text{peak}}, \; BW \times AI) \qquad \text{(Roofline, @eq-roofline)}$$
$$\text{Ridge Point} = \frac{R_{\text{peak}}}{BW} \quad \text{(FLOP/byte)}$$

---

## 4. Act 2: The Ridge Point Challenge (Design Challenge — 23 minutes)

### Pedagogical Goal
Students believe that optimization means "make computation faster." The roofline reveals that the correct optimization depends on *which ceiling* the workload hits. For a compute-bound workload (ResNet conv, above the ridge point), the right optimization is more TFLOPS (Tensor Cores, higher precision throughput). For a memory-bound workload (GPT-2 attention, below the ridge point), the right optimization is less data movement (quantization to reduce weight size, operator fusion to reduce memory round-trips, increased batch size to amortize weight loads). Students discover that the *same workload* can change regimes when moving from H100 to Jetson Orin NX, because the Jetson's lower bandwidth shifts the ridge point and makes previously compute-bound operations memory-bound. The energy hierarchy (SRAM: ~5 pJ, DRAM: ~640 pJ per access — a 128x ratio) explains why the memory wall is fundamentally an energy wall.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "A transformer attention layer has arithmetic intensity of 8 FLOP/byte. The H100 has a ridge point of ~295 FLOP/byte. By what factor is this layer underutilizing the H100's compute capability? (Express as actual/peak performance ratio.)"

Students type a percentage (1%–100%). Expected wrong answers: 30–50% (students underestimate the gap). Actual: **~2.7%** — the layer achieves $BW \times AI = 3.35 \text{ TB/s} \times 8 = 26.8 \text{ TFLOPS}$ out of a 989 TFLOPS peak, which is $26.8/989 = 2.7\%$ utilization. The compute units are 97.3% idle, waiting for data.

### The Instrument: Interactive Roofline Designer

**Primary chart:** Full **RooflineVisualizer** with interactive elements:
- **X-axis (log):** Arithmetic Intensity (FLOP/byte), range 0.1 to 1000
- **Y-axis (log):** Attainable Performance (TFLOPS), range 0.01 to 2000
- **Hardware roofline:** Bandwidth slope and compute ceiling for selected device
- **Ridge point:** Vertical dashed line at $R_{\text{peak}} / BW$, labeled with value

**Interactive workload dots (draggable conceptually via sliders):**
- Each dot represents a layer/operation with its current AI
- Moving the **batch size slider** shifts dots rightward (higher AI) because larger batches amortize weight loads
- Moving the **precision selector** shifts both the roofline and the dot: INT8 doubles the compute ceiling but also changes data volume

Controls:
- **Hardware toggle** (H100 / Jetson Orin NX): H100 roofline: 989 TFLOPS FP16, 3.35 TB/s, ridge = ~295 FLOP/byte. Jetson roofline: lower TFLOPS, lower BW, different ridge point.
- **Workload selector** (ResNet-50 Conv / GPT-2 Attention / GPT-2 QKV Projection / LayerNorm): Pre-computed AI values from the chapter
- **Batch size slider** (1 / 4 / 16 / 64 / 256): Increases arithmetic intensity by amortizing weight loads across more inputs
- **Precision selector** (FP32 / FP16 / INT8): Changes both the hardware ceiling and the workload's data volume

**Secondary chart:** **Energy hierarchy bar chart**:
- **X-axis:** Memory level (Register, SRAM L1, SRAM L2, HBM/DRAM)
- **Y-axis:** Energy per access (pJ, log scale)
- **Bars:** ~1 pJ (Register), ~5 pJ (SRAM), ~20 pJ (HBM), ~640 pJ (DRAM)
- **Annotation:** "128x Energy Wall" between SRAM and DRAM
- Connects to the chapter's claim that DRAM read costs 40,000x the energy of an INT8 add

### The Scaling Challenge
**"Find the minimum batch size that moves GPT-2's attention layer from memory-bound to compute-bound on the H100."**

Students must increase batch size until the arithmetic intensity exceeds the ridge point (~295 FLOP/byte). For attention: AI at batch=1 is ~8 FLOP/byte (deep memory-bound). At batch=64, AI rises to ~100 FLOP/byte (still memory-bound). At batch=256+, AI approaches or exceeds the ridge point. The discovery: achieving compute-bound status on modern GPUs requires very large batch sizes for attention — which conflicts with latency requirements in serving, creating a fundamental tension between throughput and latency explored in Lab 13.

**Second challenge:** "Now switch to Jetson Orin NX. At what batch size does the same layer become compute-bound on Jetson?"

The Jetson's lower ridge point means the crossover happens at a smaller batch size — but the absolute throughput is much lower. The same operation is in a different regime on different hardware.

### The Failure State
**Trigger:** `batch_size > device_memory / (model_weights + activation_memory_per_sample * batch_size)`

**Visual:** Memory bar turns RedLine; the workload dot on the roofline gets a red halo.

**Banner:** "OOM — Batch size [X] requires [Y] GB activation memory + [Z] GB weights = [total] GB. Device has only [device_ram] GB. Reduce batch size or quantize to lower precision."

This creates a tension: increasing batch size improves utilization (moves toward compute-bound) but eventually hits the memory wall (OOM). The "sweet spot" is the largest batch that fits — a direct manifestation of the memory-compute trade-off that defines hardware acceleration.

### Structured Reflection
Four-option multiple choice:

> "You observe that a transformer attention layer achieves only 2.7% of the H100's peak compute. The chapter explains this with the Roofline Model. What is the correct optimization strategy?"

- A) Use a faster GPU — the H100 is insufficient for transformer workloads
- B) Reduce the model size — fewer parameters means less computation
- **C) Increase arithmetic intensity by raising batch size, fusing operators to reduce memory round-trips, or quantizing to reduce bytes transferred — because the layer is memory-bound, not compute-bound** (correct)
- D) Switch to CPU — transformers are inherently poorly suited to GPU architecture

### Math Peek
$$\text{Attainable} = \min(R_{\text{peak}}, \; BW \times AI)$$
$$\text{Ridge Point}_{H100} = \frac{989 \text{ TFLOPS}}{3.35 \text{ TB/s}} \approx 295 \text{ FLOP/byte}$$
$$\text{Utilization} = \frac{BW \times AI}{R_{\text{peak}}} = \frac{3.35 \times 8}{989} \approx 2.7\%$$
$$\text{Speedup} = \frac{1}{(1-p) + p/S} \qquad \text{(Amdahl's Law)}$$

---

## 5. Visual Layout Specification

### Act 1: Roofline Introduction + Amdahl
- **Primary:** RooflineVisualizer (log-log). X: arithmetic intensity (FLOP/byte). Y: attainable performance (TFLOPS). Hardware roofline (bandwidth slope + compute ceiling). Workload dots for ResNet Conv (BlueLine, compute-bound), GPT-2 Attention (RedLine, memory-bound), LayerNorm (gray, deep memory-bound). Ridge point as vertical dashed line.
- **Secondary:** Amdahl heatmap. X: accelerator speedup S (log, 1–1000). Y: parallel fraction p (0.80–0.999). Color: system speedup. Student's current selection highlighted. Shows the "Acceleration Wall" where serial fraction dominates.
- **Prediction overlay:** Student's predicted speedup as horizontal dashed line on Amdahl; actual as solid.

### Act 2: Interactive Roofline Designer
- **Primary:** RooflineVisualizer with interactive workload dots. Dots move as batch size and precision change. Hardware roofline changes with device toggle. Ridge point shifts between H100 (~295) and Jetson. Memory-bound region shaded in OrangeLine, compute-bound in BlueLine.
- **Secondary:** Energy hierarchy bar chart (log scale). Register → SRAM → HBM → DRAM. "128x Energy Wall" annotation between SRAM and DRAM.
- **Failure state:** Memory bar turns RedLine on OOM. Workload dot gets red halo when batch size exceeds memory capacity.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---|---|---|---|---|
| **H100 (Compute-bound target)** | NVIDIA H100 SXM | 80 GB HBM3 | 700 W | Peak: 989 TFLOPS FP16, 3.35 TB/s BW. Ridge point ~295 FLOP/byte. ResNet convolutions are compute-bound; transformer attention is memory-bound. High ridge point means most operations underutilize compute. |
| **Jetson Orin NX (Memory-bound target)** | NVIDIA Jetson Orin NX | 16 GB LPDDR5 | 25 W | Lower compute and lower bandwidth shift the ridge point. Operations that were compute-bound on H100 may become memory-bound on Jetson. The 16 GB memory limit caps maximum batch size, creating an earlier OOM boundary. |

The two contexts demonstrate that the roofline is hardware-specific: the same operation (e.g., QKV projection) can be compute-bound on one device and memory-bound on another, requiring different optimization strategies.

---

## 7. Design Ledger Output

```json
{
  "chapter": 11,
  "workload_regime": "memory_bound",
  "optimal_batch_size": 64,
  "achieved_utilization_pct": 35,
  "ridge_point_h100": 295,
  "bottleneck_identified": "memory_bandwidth",
  "optimization_strategy": "increase_batch_size_and_quantize"
}
```

The `workload_regime` and `bottleneck_identified` fields feed forward to:
- **Lab 12 (Performance Benchmarking):** The roofline regime determines which benchmarks are relevant — memory-bound workloads need bandwidth benchmarks, compute-bound need FLOPS benchmarks
- **Lab 13 (Model Serving):** The optimal batch size informs the batching strategy; memory-bound inference requires different serving architecture than compute-bound

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Amdahl ResNet p=0.95 → ~20x speedup | `@sec-hardware-acceleration-ai-hardware-acceleration-fundamentals-9b28`, AmdahlH100 class | "p_resnet = 0.95...speedup_resnet ≈ 20x" |
| Amdahl GPT-2 p=0.80 → ~5x speedup | `@sec-hardware-acceleration-ai-hardware-acceleration-fundamentals-9b28`, AmdahlH100 class | "p_gpt2 = 0.80...speedup_gpt2 ≈ 5x" |
| H100 500x raw speedup over CPU | `@sec-hardware-acceleration-ai-hardware-acceleration-fundamentals-9b28`, AmdahlH100 class | "hw_speedup_factor = 500.0" |
| Amdahl ceiling: $1/(1-p)$ = 5x for GPT-2 | `@sec-hardware-acceleration-ai-hardware-acceleration-fundamentals-9b28`, AmdahlH100 class | "ceiling_gpt2 = 1/(1-p_gpt2) = 5x" |
| Roofline: $\min(R_{\text{peak}}, BW \times AI)$ | `@sec-hardware-acceleration-roofline-model-42ff`, @eq-roofline | "Attainable Performance = min(Peak Compute, Peak Bandwidth x Arithmetic Intensity)" |
| H100: 989 TFLOPS FP16, 3.35 TB/s BW | constants.py | "H100_FLOPS_FP16_TENSOR = 989 * TFLOPs; H100_MEM_BW = 3.35 * TB/second" |
| Ridge point = $R_{\text{peak}} / BW$ | `@sec-hardware-acceleration-hardware-ridge-points-b5b6` | "Ridge Point R = Peak FLOPS / Peak Bandwidth (FLOP/byte)" |
| DRAM 100x+ energy cost vs on-chip | `@sec-hardware-acceleration-understanding-ai-memory-wall-3ea9` | "memory systems dominate accelerator performance: DRAM access has 100x or higher energy cost than on-chip arithmetic" |
| DRAM access ~640 pJ, SRAM ~5 pJ | `@sec-hardware-acceleration-memory-hierarchy-1839`, @tbl-memory-hierarchy and energy hierarchy | "~128x Cost (The Memory Wall)" per the memory cost figure |
| 10% serial → max 10x speedup | `@sec-hardware-acceleration-ai-hardware-acceleration-fundamentals-9b28`, @eq-amdahl | "If data loading takes 10% of the time ($p=0.9$), even an infinite speed accelerator ($S=\infty$) can only achieve a 10x total speedup" |
| Transformer attention: low AI, memory-bound | `@sec-hardware-acceleration-roofline-model-42ff`, arithmetic intensity definition | "attention layers (<10 FLOP/byte) are memory-bound" |
| ResNet convolutions: high AI, compute-bound | `@sec-hardware-acceleration-roofline-model-42ff`, arithmetic intensity definition | "ResNet's convolutions (>50 FLOP/byte) are compute-bound" |
| TPUv1 15–30x faster, 30–80x better perf/watt | `@sec-hardware-acceleration-emergence-domainspecific-architectures-e56e`, TPUv1 callout | "TPUv1 was 15x–30x faster on inference workloads and 30x–80x better performance-per-watt" |

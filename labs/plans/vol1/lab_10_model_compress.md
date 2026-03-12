# Mission Plan: lab_10_model_compress

## 1. Chapter Alignment

- **Chapter:** Model Compression (`@sec-model-compression`)
- **Core Invariant:** The **Quantization Free Lunch** — reducing precision from FP32 to INT8 yields < 1% accuracy loss while providing a 4x memory reduction and up to 20x energy reduction, because neural networks are massively overparameterized in their numerical precision. The "cliff" arrives at 3–4 bits, where accuracy collapses catastrophically.
- **Central Tension:** Students believe that compression always costs accuracy — that making a model smaller makes it worse. The chapter's Pareto frontier reveals three regimes: a "Free Lunch" zone (FP32 to INT8) where compression is essentially free, an "Efficient Trade" zone (INT8 to INT4) where modest accuracy loss buys large efficiency gains, and a "Danger Zone" (below INT4) where the model becomes useless. The surprise is not that compression works but that a vast operating region exists where it costs *nothing* meaningful. The second surprise: unstructured pruning at 50% sparsity gives 0% speedup on standard GPUs because sparse matrix operations cannot exploit the irregular zero patterns.
- **Target Duration:** 35–40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict the accuracy cost of quantizing a 7B LLM from FP16 to INT8. Most expect a significant drop (5–10%) because halving precision sounds dangerous. The chapter data shows < 1% accuracy loss — the "Free Lunch Zone." Students then explore the accuracy-vs-bitwidth curve and discover the quantization cliff: accuracy stays flat from FP32 through INT8, then falls off a cliff at 3–4 bits. The aha moment is that the free lunch is real, and the cliff is sudden, not gradual.

**Act 2 (Design Challenge, 23 min):** Students must deploy a 7B parameter model to two targets: an H100 (accuracy-first, where the constraint is throughput per dollar) and an iPhone 15 Pro (latency-first, where the model must fit in shared memory and meet a 50ms latency budget). They explore the full compression Pareto frontier — quantization, structured pruning, and their interaction — to find configurations that satisfy each target's constraints. The key discovery: on the iPhone, INT4 quantization alone achieves a 4x bandwidth speedup that makes the model viable, while unstructured pruning provides zero speedup because the iPhone's Neural Engine cannot skip irregular zeros.

---

## 3. Act 1: The Free Lunch (Calibration — 12 minutes)

### Pedagogical Goal
Students believe that reducing numerical precision degrades model quality proportionally — that INT8 (half the bits of FP16) means roughly half the accuracy. The chapter's Free Lunch curve demolishes this: CNNs like ResNet-50 lose < 0.1% accuracy going from FP32 to INT8, and even transformer models like BERT lose < 0.5%. The chapter explains why: neural networks are vastly overparameterized in precision — they use 32 bits where 8 bits capture the essential weight distribution. Students discover that the real danger is not gradual degradation but a sudden cliff at 3–4 bits where the model loses its ability to represent critical weight distinctions.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "A 7B parameter LLM is quantized from FP16 (2 bytes per weight, 14 GB) to INT8 (1 byte per weight, 7 GB). What happens to model accuracy on standard benchmarks?"

Options:
- A) Accuracy drops by ~10% — halving precision roughly halves the model's representational capacity
- B) Accuracy drops by ~5% — some information is lost but the model compensates
- **C) Accuracy drops by < 1% — neural networks are overparameterized in precision; INT8 captures the essential weight distribution** (correct)
- D) Accuracy stays exactly the same — quantization is lossless

Common wrong answer: **B** — students expect meaningful but manageable loss.

Why wrong: The chapter's "Physics of Quantization" shows that neural network weight distributions are concentrated in a narrow range. INT8's 256 discrete levels are sufficient to represent these distributions. The accuracy cost is measurable but negligible (< 1%) because the information lost at the least-significant bits is below the noise floor of the training process itself.

### The Instrument: Accuracy-Bitwidth Explorer

A **line chart** replicating @fig-quantization-free-lunch:

- **X-axis:** Precision (FP32, FP16, INT8, INT4, INT3, INT2) — reversed so reduction goes left-to-right
- **Y-axis:** Model accuracy (%)
- **Line 1 (BlueLine):** CNN (ResNet-50): 76.1% at FP32, 76.1% at FP16, 76.0% at INT8, 74.5% at INT4, 55.0% at INT3, 10.0% at INT2
- **Line 2 (RedLine):** Transformer (BERT): 84.0% at FP32, 84.0% at FP16, 83.5% at INT8, 78.0% at INT4, 40.0% at INT3, 10.0% at INT2
- **Shaded regions:** Green = "Free Lunch Zone" (FP32 to INT8), Red = "The Cliff" (INT3 and below)
- **Annotation arrow:** Points to the cliff transition

Controls:
- **Model selector** (ResNet-50 / BERT / 7B LLM): Switches the accuracy curve
- **Context toggle** (H100 Accuracy-first / iPhone 15 Pro Latency-first): H100 shows throughput gain; iPhone shows memory and latency gain

**Secondary metric panel:**
- Memory footprint (GB) at each precision
- Energy per inference (relative to FP32) — 20x reduction at INT8 per chapter
- Tokens/sec for LLM generation (bandwidth-bound)

### The Reveal
After interaction:
> "You predicted [X]% accuracy drop. The actual drop from FP16 to INT8 is **< 1%**. This is the 'Free Lunch Zone' identified in @fig-quantization-free-lunch. The reason: INT8 has 256 discrete levels, more than enough to represent the weight distribution of a trained neural network. The cliff at INT3 (8 levels) is where representational capacity becomes insufficient. The chapter calls INT8 a 'Free Lunch' because you gain 4x memory reduction and up to 20x energy reduction for effectively zero accuracy cost."

### Reflection (Structured)
Four-option multiple choice:

> "The chapter states that moving from FP32 to INT8 saves 4x memory. For a bandwidth-bound LLM generating tokens, what speedup does INT8 quantization provide?"

- A) ~1.5x — quantization helps but does not change the bottleneck
- B) ~2x — half the bits means roughly double the throughput
- **C) ~4x — for bandwidth-bound generation, throughput scales linearly with weight size reduction because the bottleneck is loading weights from memory** (correct)
- D) ~8x — INT8 also enables Tensor Core acceleration

### Math Peek (collapsible)
$$\text{Memory} = \text{Parameters} \times \text{Bytes per Weight}$$
$$\text{Latency}_{\text{bw-bound}} = \frac{\text{Weight Size (bytes)}}{\text{Memory Bandwidth (bytes/s)}}$$
$$\text{INT4 Speedup} = \frac{\text{FP16 Size}}{\text{INT4 Size}} = \frac{2 \text{ B/param}}{0.5 \text{ B/param}} = 4\times$$

---

## 4. Act 2: The Compression Pareto Frontier (Design Challenge — 23 minutes)

### Pedagogical Goal
Students believe that compression is a single-dimensional trade-off: less precision = less accuracy = more speed. The chapter reveals a multi-dimensional Pareto frontier where quantization, pruning, and their interaction create a rich design space. The critical insight is that *which* compression technique helps depends on the hardware: INT8/INT4 quantization provides linear speedup on bandwidth-bound LLM inference because it reduces memory traffic, while unstructured pruning at 50% provides zero speedup on GPUs because sparse matrix operations cannot exploit irregular zero patterns without hardware support. Only structured sparsity (2:4) delivers real speedup on NVIDIA GPUs. Students must navigate this frontier to deploy a 7B LLM to two targets with different binding constraints.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "You prune 50% of a model's weights using unstructured pruning (random zeros scattered throughout the weight matrices). On a standard GPU, what inference speedup do you expect?"

Students type a speedup multiplier (1.0x–5.0x). Expected wrong answers: 1.5x–2.0x (students assume half the weights = roughly double the speed). Actual: **~1.0x (no speedup)** — the chapter explicitly states that "without Specialized Hardware Support, the resulting sparse matrices may actually run *slower* than dense ones due to irregular memory access patterns" and "unstructured pruning therefore primarily benefits model storage rather than inference acceleration."

### The Instrument: Compression Design Space

**Primary chart:** A **Pareto frontier scatter plot**:
- **X-axis:** Model size (GB) — from 14 GB (FP16 baseline) down to 1 GB
- **Y-axis:** Model accuracy (%)
- **Bubble size:** Inference latency (ms per token)
- **Points:** Each point = a (quantization level, pruning ratio, pruning type) configuration
- **Pareto frontier (GreenLine):** Connects dominant configurations
- **Student's current config (OrangeLine marker):** Moves as sliders change
- **Device constraint boxes:** H100 region (accuracy > 98% of baseline) and iPhone region (memory < 8 GB AND latency < 50 ms)

Controls:
- **Quantization level selector** (FP16 / INT8 / INT4): Steps through precision levels
- **Pruning ratio slider** (0%–90%, step 10%, default 0%)
- **Pruning type toggle** (None / Unstructured / Structured 2:4)
- **Context toggle** (H100 Accuracy-first / iPhone 15 Pro Latency-first): Changes the constraint box and speedup calculations

**Secondary chart:** **Speedup comparison bar chart**:
- **X-axis:** Compression technique
- **Y-axis:** Realized speedup (x)
- **Bars:** Quantization speedup (real — proportional to size reduction for BW-bound), Unstructured pruning speedup (near 1.0x on GPU), Structured 2:4 pruning speedup (~2x on Tensor Cores)
- **Annotation:** "The Sparsity Gap" between theoretical and realized speedup for unstructured pruning

### The Scaling Challenge
**"Deploy the 7B LLM to the iPhone 15 Pro: find the compression configuration that fits in 8 GB shared memory, meets a 50 ms per-token latency budget, AND preserves >= 95% of baseline accuracy."**

Students must discover that:
1. FP16 at 14 GB does not fit (OOM)
2. INT8 at 7 GB fits but latency is ~140 ms at 50 GB/s mobile bandwidth — too slow
3. INT4 at 3.5 GB fits and latency is ~70 ms — still over budget
4. INT4 + structured 2:4 pruning at ~1.75 GB fits, latency ~35 ms — meets all constraints
5. Unstructured pruning does not help latency regardless of sparsity level

### The Failure State
**Trigger 1 (OOM):** `model_size > device_ram`

**Visual:** Memory bar turns RedLine.

**Banner:** "OOM — Model requires [X] GB but device has only [Y] GB. Reduce precision or apply pruning to fit."

**Trigger 2 (SLA Violated):** `latency > latency_budget`

**Visual:** Latency bar turns RedLine; timeline extends past budget line.

**Banner:** "SLA VIOLATED — Inference latency of [X] ms exceeds the [Y] ms budget. For bandwidth-bound generation, reduce weight size to decrease memory load time."

**Trigger 3 (Accuracy Collapse):** `accuracy < 0.5 * baseline_accuracy`

**Visual:** Accuracy gauge turns RedLine with skull icon.

**Banner:** "ACCURACY COLLAPSE — Model accuracy has fallen to [X]% (baseline: [Y]%). You have crossed the Quantization Cliff. Increase precision to recover."

### Structured Reflection
Four-option multiple choice:

> "You applied 50% unstructured pruning to a model and observed 0% speedup on the GPU. The chapter explains this by stating that unstructured sparsity 'may actually run slower than dense' operations. What is the root cause?"

- A) The model is too small for the GPU to benefit from any optimization
- B) Pruning removes important weights, causing the model to recompute missing activations
- **C) GPUs execute dense matrix multiplications on regular memory access patterns; scattered zeros create irregular accesses that the hardware cannot skip, providing no bandwidth or compute savings** (correct)
- D) The pruning ratio is too low — 90% would show a speedup

### Math Peek
$$\text{Weight Memory} = N_{\text{params}} \times B_{\text{precision}}$$
$$\text{Token Latency}_{\text{BW-bound}} = \frac{N_{\text{params}} \times B_{\text{precision}}}{\text{BW}_{\text{device}}}$$
$$\text{Structured 2:4 Speedup} \approx 2\times \text{ on Tensor Cores with sparse support}$$

---

## 5. Visual Layout Specification

### Act 1: Accuracy-Bitwidth Explorer
- **Primary:** Line chart with two data series (CNN, Transformer). X: precision (reversed: FP32 to INT2). Y: accuracy (%). Green shaded "Free Lunch Zone" (FP32–INT8). Red shaded "Cliff" (INT3–INT2). Prediction overlay as dashed horizontal line.
- **Secondary:** Metric panel showing memory (GB), energy (relative), and tokens/sec at each precision. Updates dynamically with model selector.

### Act 2: Compression Pareto Frontier
- **Primary:** Scatter plot. X: model size (GB). Y: accuracy (%). Bubble size: latency (ms). Pareto frontier in GreenLine. Device constraint boxes as shaded regions. Student config as OrangeLine dot.
- **Secondary:** Bar chart. X: technique. Y: realized speedup. Shows the "Sparsity Gap" — unstructured pruning bar near 1.0x vs. quantization bar at 4x.
- **Failure states:** Memory bar (RedLine on OOM), Latency bar (RedLine on SLA violation), Accuracy gauge (RedLine on cliff).

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---|---|---|---|---|
| **H100 (Accuracy-first)** | NVIDIA H100 SXM | 80 GB HBM3 | 700 W | Throughput per dollar; 3.35 TB/s bandwidth makes LLM generation bandwidth-bound; quantization provides linear speedup; can afford higher-precision models |
| **iPhone 15 Pro (Latency-first)** | Apple A17 Pro | 8 GB shared LPDDR5 | 3–5 W | Model must fit in ~8 GB shared with OS; ~50 GB/s memory bandwidth; 50 ms per-token latency budget; Neural Engine supports INT8 but not arbitrary sparse patterns |

The two contexts demonstrate that compression priorities depend entirely on the binding constraint: the H100 is throughput-constrained (quantization to INT8 is sufficient), while the iPhone is memory-and-latency-constrained (INT4 plus structured pruning is necessary).

---

## 7. Design Ledger Output

```json
{
  "chapter": 10,
  "quantization_level": "int4",
  "pruning_ratio": 0.5,
  "pruning_type": "structured_2_4",
  "final_model_size_gb": 1.75,
  "accuracy_retention_pct": 96,
  "deployment_target": "iphone15pro"
}
```

The `quantization_level` and `pruning_type` fields feed forward to:
- **Lab 11 (Hardware Acceleration):** The chosen compression configuration determines where the model sits on the Roofline — INT4 models have different arithmetic intensity than FP16 models
- **Lab 13 (Model Serving):** The model size and latency characteristics inform the serving throughput analysis

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Free Lunch Zone: < 1% loss FP32→INT8 | `@sec-model-compression-optimization-framework-9e21`, @fig-quantization-free-lunch | "a 'Free Lunch' plateau where reducing precision from FP32 to INT8 yields < 1% accuracy loss" |
| Quantization cliff at 3–4 bits | `@sec-model-compression-optimization-framework-9e21`, @fig-quantization-free-lunch caption | "This robustness collapses at the 'Quantization Cliff' (typically 3–4 bits)" |
| 7B model = 14 GB FP16, 3.5 GB INT4 | `@sec-model-compression-optimization-framework-9e21`, QuantizationSpeedup class | "7B x 2 bytes = 14 GB (FP16); 7B x 0.5 bytes = 3.5 GB (INT4)" |
| 4x speedup from INT4 (BW-bound) | `@sec-model-compression-optimization-framework-9e21`, QuantizationSpeedup callout | "Quantization is not just about fitting; it is a 4x Linear Speedup because LLM generation is bandwidth-bound" |
| 20x energy reduction INT8 vs FP32 | `@sec-model-compression-optimization-framework-9e21`, CompressionSetup class | "int8_energy_reduction = 20" |
| Unstructured pruning: 0% GPU speedup | `@sec-model-compression-structural-optimization-ee93` | "without Specialized Hardware Support, the resulting sparse matrices may actually run *slower* than dense ones due to irregular memory access patterns" |
| Structured 2:4 sparsity: ~2x speedup | `@sec-model-compression-structural-optimization-ee93`, fn-nm-sparsity-a100 | "N:M structured sparsity...ensuring 2 out of every 4 weights are zero...to align with specialized accelerator capabilities" |
| Smartphone RAM: 8 GB | constants.py | "SMARTPHONE_RAM_GB = 8 * GB" |
| DRAM access: 40,000x energy vs INT8 add | `@sec-model-compression-optimization-framework-9e21`, energy table | "DRAM Read: 64-bit = 40,000x relative energy" |
| MobileNetV3 8 FPS→35 FPS with INT8 | `@sec-model-compression-deployment-scenarios-70c9`, MobileNet callout | "Unoptimized MobileNetV3 (FP32) runs at 8 FPS...INT8: Speed jumps to 35 FPS" |
| 90% pruning: ~10% accuracy drop | `@sec-model-compression-structural-optimization-ee93`, @tbl-optimization-tradeoffs | "Aggressive pruning (e.g., removing 90% of weights) might drop accuracy by 10% to gain another 20% speedup" |

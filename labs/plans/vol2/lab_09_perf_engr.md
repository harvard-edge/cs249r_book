# Mission Plan: lab_09_perf_engr (Volume 2)

## 1. Chapter Alignment

- **Chapter:** Performance Engineering (`@sec-performance-engineering`)
- **Core Invariant:** The Iron Law of ML Performance: `Time = max(Compute/FLOPS, Memory_Access/Bandwidth) + Overhead`. Only the binding constraint matters; optimizing the non-binding term yields zero improvement. The Roofline Model's ridge point `I_ridge = Peak_FLOPS / Peak_BW` determines whether a workload is compute-bound or memory-bound. For H100 at FP16, `I_ridge ~ 295 FLOP/byte`; most LLM inference operations fall below this threshold and are therefore memory-bound.
- **Central Tension:** Students believe that faster GPUs always mean faster inference: "upgrade to H100 and everything gets faster." The chapter's roofline analysis reveals that LLM decode (arithmetic intensity ~1--2 FLOP/byte) is so deeply memory-bound that doubling compute throughput (FLOPS) yields zero speedup. The binding constraint is memory bandwidth, and each GPU generation increases compute faster than bandwidth, pushing the ridge point higher and making *more* workloads memory-bound. An engineer who optimizes compute for a memory-bound workload has wasted their effort entirely.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students place common ML operations on a Roofline plot and predict whether they are compute-bound or memory-bound. Most students expect large matrix multiplications (GEMMs) and LLM decode to both be compute-bound because they involve "heavy math." The roofline reveals that LLM decode at batch size 1 has arithmetic intensity of ~1 FLOP/byte (reading the entire weight matrix for a single token), placing it deep in the memory-bound regime. This calibrates the student's diagnostic instinct: profile before optimizing.

**Act 2 (Design Challenge, 22 min):** Students configure an inference deployment for a 70B LLM, choosing between a batch-inference (throughput-optimized) and streaming-inference (latency-optimized) context. They manipulate batch size, precision, and operator fusion to move their workload on the roofline plot. The challenge reveals the fundamental trade-off: increasing batch size raises arithmetic intensity (moving toward compute-bound), improving throughput but degrading per-request latency. Students must find the batch size that saturates GPU compute without violating a latency SLA, discovering that the "optimal" operating point depends entirely on the application's constraint profile.

---

## 3. Act 1: The Roofline Diagnostic (Calibration -- 12 minutes)

### Pedagogical Goal

Students believe that "AI workloads are compute-bound" because neural networks perform enormous numbers of multiplications. The roofline model reveals that the binding constraint depends on arithmetic intensity (FLOP/byte), not total FLOPS. LLM decode at batch size 1 loads the entire weight matrix (~140 GB for 70B in FP16) to generate a single token, performing only 2 FLOPS per weight loaded. This gives arithmetic intensity of ~1 FLOP/byte, which is 295x below the H100 FP16 ridge point. The GPU's compute cores sit 99.7% idle while waiting for memory. This act forces students to predict the binding constraint for common operations and discover that most inference workloads are memory-bound.

### The Lock (Structured Prediction)

Present a multiple-choice prediction before any instruments unlock:

> "An H100 GPU has 989 TFLOPS of FP16 compute and 3.35 TB/s of memory bandwidth (ridge point = 295 FLOP/byte). LLM autoregressive decode at batch size 1 reads the full 140 GB weight matrix for each token. What percentage of the GPU's compute capacity is utilized during decode?"

Options:
- A) About 80% -- the GPU is doing dense matrix multiplication
- B) About 30% -- some overhead reduces utilization
- **C) About 0.3% -- the GPU is almost entirely idle, waiting for memory** (correct)
- D) About 5% -- memory is a bottleneck but not this severe

Common wrong answer: B. Students know there is "some" memory overhead but dramatically underestimate its severity. At batch=1, arithmetic intensity is ~1 FLOP/byte vs ridge of 295, so utilization is 1/295 ~ 0.3%.

### The Instrument: Interactive Roofline Plot

Controls:
- **Operation selector**: LLM Decode (batch=1) / LLM Decode (batch=64) / LLM Prefill (batch=1) / Large GEMM (4096x4096) / Element-wise (ReLU) / Attention (naive) / Attention (FlashAttention)
- **Hardware selector**: A100 FP16 / H100 FP16 / H100 FP8 / B200 FP16
- **Batch size slider**: 1 / 4 / 16 / 64 / 256 / 1024 (default: 1)
- **Precision toggle**: FP32 / FP16 / FP8 / INT4

Outputs:
- **Primary chart**: Log-log roofline plot. X-axis: arithmetic intensity (FLOP/byte, 0.1 to 10,000). Y-axis: achievable performance (TFLOPS). Diagonal line = bandwidth ceiling. Horizontal line = compute ceiling. Ridge point annotated. Selected operation plotted as a dot. Color: RedLine if memory-bound, GreenLine if compute-bound.
- **Secondary metric cards**: "Arithmetic Intensity", "Achievable TFLOPS", "% of Peak Compute", "Binding Constraint: Compute | Memory"
- **Overlay**: Multiple GPU generation rooflines (V100, A100, H100, B200) when hardware comparison mode is active.

Formulas:
- `AI = total_flops / total_bytes_loaded`
- `achievable = min(peak_flops, bandwidth * AI)`
- `utilization_pct = achievable / peak_flops * 100`
- Ridge point: `I_ridge = peak_flops / bandwidth`
- For LLM decode batch=1: `AI = 2 * params / (2 * params) = 1 FLOP/byte` (2 ops per weight, 2 bytes per FP16 weight)
- For LLM decode batch=B: `AI = 2*B*params / (2*params) = B FLOP/byte`

### The Reveal

After interaction:
> "You predicted [X]% utilization. At batch size 1, LLM decode has arithmetic intensity of ~1 FLOP/byte. On the H100 (ridge point = 295), this means **0.3% compute utilization**. The GPU's 989 TFLOPS of compute sit 99.7% idle while the 3.35 TB/s memory bus is the bottleneck. Increasing batch size to 64 raises arithmetic intensity to ~64 FLOP/byte, improving utilization to ~22%. This is why batching is the single most important inference optimization."

### Reflection (Structured)

Four-option multiple choice:

> "The chapter states that each GPU generation increases compute faster than bandwidth: the ridge point grew from 139 FLOP/byte (V100) to 625 FLOP/byte (B200) in seven years. What is the consequence?"

- A) Newer GPUs always provide faster inference regardless of workload
- **B) More workloads become memory-bound with each generation, making bandwidth optimization increasingly important** (correct)
- C) Older GPUs are better for memory-bound workloads because they have lower ridge points
- D) The trend will reverse as HBM technology catches up to compute scaling

### Math Peek (collapsible)

$$\text{Achievable FLOPS} = \min(P, \; B \times I)$$

$$I_{\text{ridge}} = \frac{P}{B} = \frac{989 \text{ TFLOPS}}{3.35 \text{ TB/s}} \approx 295 \text{ FLOP/byte (H100 FP16)}$$

$$\text{LLM decode, batch=1: } I = \frac{2 \times 70 \times 10^9}{2 \times 70 \times 10^9} = 1 \text{ FLOP/byte}$$

$$\text{Utilization} = \frac{I}{I_{\text{ridge}}} = \frac{1}{295} \approx 0.34\%$$

---

## 4. Act 2: The Efficiency Frontier (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students believe there is a single "best" configuration for inference. The chapter shows that the efficiency frontier is a Pareto curve: latency-optimized configurations (FP16, batch=1, speculative decoding) cost 60x more per token than throughput-optimized configurations (INT4, batch=64, continuous batching). Neither is objectively better; the application's constraint profile determines the optimal operating point. Students must navigate this trade-off space by manipulating batch size, precision, and fusion to satisfy a latency SLA while minimizing cost per token.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "Configuration A: FP16, batch size 1, speculative decoding. Serves ~80 tokens/sec at 20 ms inter-token latency on 8 H100 GPUs. Configuration B: INT4, batch size 64, continuous batching. Serves ~4,000 tokens/sec at 120 ms inter-token latency on 4 H100 GPUs. What is the cost ratio (A/B) per 1,000 output tokens?"

Students type a ratio. Expected wrong answers: 2--5x (students underestimate the compounding of batch size and precision effects). Actual: Configuration A costs ~$0.12/1K tokens; Configuration B costs ~$0.002/1K tokens. Ratio: 60x.

### The Instrument: Inference Configuration Explorer

Controls:
- **Deployment context toggle**: Batch inference (throughput-optimized) / Streaming inference (latency-optimized)
- **Batch size slider**: 1 / 4 / 16 / 64 / 128 / 256 / 512 (default: 1 for streaming, 64 for batch)
- **Precision selector**: FP32 / FP16 / FP8 / INT4 (default: FP16)
- **Operator fusion toggle**: None / Partial (linear+activation) / Full (FlashAttention) (default: None)
- **Speculative decoding toggle**: Off / On (draft model k=4) (default: Off)
- **Latency SLA slider**: 20 / 50 / 100 / 200 / 500 ms (default: 100 ms inter-token)

Outputs:
- **Primary chart**: Roofline plot with the current workload dot. Dot moves as batch size and precision change. Ridge point annotated. Color transitions from Red (memory-bound) to Green (compute-bound) as dot crosses ridge.
- **Secondary chart**: Latency-throughput Pareto curve. X-axis: throughput (tokens/sec). Y-axis: inter-token latency (ms). Current configuration plotted as a dot. SLA threshold as horizontal red line.
- **Tertiary metrics**: "Cost per 1K tokens ($)", "GPU count required", "Arithmetic intensity", "Binding constraint"

Formulas:
- `throughput = batch_size * (1 / decode_time_per_token)`
- `decode_time = weight_bytes / memory_bandwidth` (memory-bound regime)
- `weight_bytes = params * bytes_per_precision`
- Fusion reduces memory access by 10--30x for attention
- Speculative decoding: `effective_AI = k * base_AI` where k = draft token count
- Cost: `$/1K_tokens = (n_gpus * $/gpu-hour) / (throughput * 3600) * 1000`

### The Scaling Challenge

**"Find the minimum batch size that achieves inter-token latency under 100 ms AND cost under $0.01 per 1,000 tokens for a 70B model on H100 GPUs."**

Students discover:
- Batch=1, FP16: latency = 20 ms (passes SLA), cost = $0.12/1K tokens (fails cost target by 12x)
- Batch=64, FP16: latency = 120 ms (fails SLA), cost = $0.004/1K tokens (passes cost target)
- Batch=32, INT4: latency = 80 ms (passes SLA), cost = $0.005/1K tokens (passes cost target)
- Batch=16, FP16 + FlashAttention: latency = 65 ms (passes SLA), cost = $0.008/1K tokens (passes cost target)

The sweet spot requires combining precision reduction AND batching AND fusion. No single optimization suffices.

### The Failure State

**Trigger:** Latency exceeds SLA by more than 2x.

**Visual change:** The latency bar on the Pareto plot turns RedLine. The SLA line is crossed. The cost metric may be green (cheap but slow).

**Banner text:** "SLA VIOLATED -- Inter-token latency is [X] ms, exceeding the [Y] ms SLA by [Z]x. At batch size [B], each token's decode must wait for [B-1] other tokens to process. Reduce batch size, enable speculative decoding, or lower precision to reclaim latency budget."

### Structured Reflection

Four-option multiple choice:

> "An engineer spends a week optimizing the compute kernels for LLM decode (batch size 1, FP16). The optimized kernels achieve 95% of peak FLOPS. What is the expected speedup?"

- A) ~2x -- reducing compute time by half doubles throughput
- B) ~1.5x -- meaningful improvement from better kernels
- **C) ~0x -- the workload is memory-bound at AI = 1 FLOP/byte; compute optimization cannot help** (correct)
- D) ~0.95x -- slight improvement matching the 95% utilization gain

### Math Peek

$$\text{Time} = \max\left(\frac{\text{Compute}}{\text{FLOPS}}, \; \frac{\text{Memory Access}}{\text{Bandwidth}}\right) + \text{Overhead}$$

$$\text{Config A (latency): } \frac{\$0.12}{1\text{K tokens}} \quad \text{vs.} \quad \text{Config B (throughput): } \frac{\$0.002}{1\text{K tokens}}$$

$$\text{Cost ratio} = \frac{0.12}{0.002} = 60\times$$

---

## 5. Visual Layout Specification

### Act 1: Interactive Roofline Plot
- **Primary:** Log-log roofline plot. X-axis: arithmetic intensity (FLOP/byte, 0.1 to 10,000). Y-axis: achievable performance (TFLOPS, 0.1 to 10,000). Diagonal bandwidth ceiling. Horizontal compute ceiling. Ridge point labeled. Operation dot colored Red (memory-bound) or Green (compute-bound).
- **Secondary:** Four metric cards: Arithmetic Intensity, Achievable TFLOPS, % of Peak, Binding Constraint.
- **Failure state:** None (calibration act).

### Act 2: Inference Configuration Explorer
- **Primary:** Roofline plot with movable workload dot. As batch size increases, dot moves right along x-axis. As precision changes, ridge point shifts.
- **Secondary:** Latency-throughput Pareto chart. X-axis: throughput (tokens/sec, log scale). Y-axis: latency (ms, log scale). Current config as dot. SLA line in red. Prior configurations as faded dots (trail).
- **Tertiary:** Three metric cards: cost per 1K tokens, GPU count, binding constraint.
- **Failure state:** SLA violation turns Pareto dot RedLine, banner appears.

---

## 6. Deployment Context Definitions

| Context | Optimization Target | Batch Size | Precision | Key Constraint |
|---|---|---|---|---|
| **Batch inference (throughput-optimized)** | Maximize tokens/sec per GPU, tolerate 200+ ms latency | 64--512 | INT4 or FP8 | Cost per token; memory-bound at low batch, transitions to compute-bound at high batch |
| **Streaming inference (latency-optimized)** | Minimize inter-token latency, sub-50 ms target | 1--16 | FP16 | Inter-token latency SLA; deeply memory-bound; speculative decoding is primary lever |

The two contexts demonstrate that the same model and hardware produce radically different performance profiles depending on the optimization target. The roofline model explains exactly why: batch size is the primary lever for moving along the arithmetic intensity axis.

---

## 7. Design Ledger Output

```json
{
  "chapter": 9,
  "binding_constraint": "memory | compute",
  "batch_size": 32,
  "precision": "int4",
  "operator_fusion": "flash_attention",
  "cost_per_1k_tokens": 0.005,
  "latency_ms": 80,
  "arithmetic_intensity": 32
}
```

- `binding_constraint` feeds forward to **Lab 10 (Distributed Inference)**: determines whether to optimize for bandwidth (KV cache) or compute (batching).
- `batch_size` and `precision` feed forward to **Lab 10 (Distributed Inference)**: sets the starting configuration for serving system design.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Iron Law: Time = max(Compute/FLOPS, Memory/BW) + Overhead | `@eq-iron-law-perf` | "Time = max(Compute/FLOPS, Memory_Access/Bandwidth) + Overhead" |
| H100 FP16: 989 TFLOPS, 3.35 TB/s, ridge = 295 FLOP/byte | `@sec-performance-engineering-roofline` | "I_ridge = 989 TFLOPS / 3.35 TB/s approximately 295 FLOP/byte" |
| A100 FP16 ridge ~153 FLOP/byte | RooflineRidgeCalc LEGO cell | "a100_ridge_str = 153" |
| H100 FP8 ridge ~591 FLOP/byte | RooflineRidgeCalc LEGO cell | "h100_fp8_ridge_str = 591" |
| V100 ridge 139 to B200 ridge 625, 4.5x in 7 years | `@fig-shifting-roofline` | "ridge point has grown from 139 FLOP/byte on the V100 to 625 FLOP/byte on the B200" |
| Config A: FP16, batch=1, ~80 tok/s, $0.12/1K tokens | `@sec-performance-engineering-efficiency-frontier` | "Configuration A...80 tokens/second with 20 ms inter-token latency...approximately $0.12 per 1,000 output tokens" |
| Config B: INT4, batch=64, ~4000 tok/s, $0.002/1K tokens | `@sec-performance-engineering-efficiency-frontier` | "Configuration B...4,000 tokens/second aggregate throughput...approximately $0.002 per 1,000 output tokens" |
| Config B is 60x cheaper than Config A | `@sec-performance-engineering-efficiency-frontier` | "Configuration B achieves 60x lower cost per token than Configuration A" |
| Operator fusion reduces memory access 10--30x for attention | `@sec-performance-engineering-memory-wall` | "operator fusion...effective Memory Access term shrinks dramatically, often by 10--30x for attention computation" |
| H100 sits 95% idle during LLM decode | `@sec-performance-engineering-memory-wall` | "An H100 GPU capable of 989 teraFLOPS often sits 95% idle while generating text" |

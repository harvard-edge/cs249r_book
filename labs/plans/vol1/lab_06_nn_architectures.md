# Mission Plan: lab_06_nn_architectures

## 1. Chapter Alignment

- **Chapter:** Network Architectures (`@sec-network-architectures`)
- **Core Invariant:** **Architecture determines bottleneck regime.** The same hardware hits completely different walls depending on the architecture deployed: ResNet-50 (I ~40 FLOPs/byte) is compute-bound and saturates GPU ALUs, while GPT-2 at inference (I ~0.5 FLOPs/byte) is memory-bandwidth-bound and wastes >98% of peak FLOPS. An 80x arithmetic intensity gap on the same silicon. The architecture is not what the model does but what the hardware *must* do.
- **Central Tension:** Students believe that all "deep learning" workloads stress the same hardware resource and that reducing FLOPs proportionally reduces latency. The chapter demolishes both: a CNN and a Transformer on the same H100 occupy opposite ends of the arithmetic intensity spectrum, and MobileNet with 14x fewer FLOPs than ResNet-50 can run *slower* on datacenter GPUs because its low arithmetic intensity starves compute units. Architecture choice is a hardware contract, not a modeling decision.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students confront the arithmetic intensity gap between architectural families on identical hardware. They predict how much faster ResNet-50 inference runs versus GPT-2 inference on the same H100 GPU, expecting similar performance since "both are neural networks on the same chip." The instrument reveals an 80x arithmetic intensity gap: ResNet-50 at I = 40.2 is compute-bound and achieves high GPU utilization, while GPT-2 at I = 0.5 is memory-bandwidth-bound and leaves >98% of peak FLOPS unused. The reveal forces students to abandon the intuition that "deep learning workload" is a single category and recognize that architecture determines which Iron Law term dominates.

**Act 2 (Design Challenge, 22 min):** Students explore the quadratic attention memory wall by configuring a Transformer deployment to handle a target sequence length within a fixed memory budget. They discover that doubling sequence length quadruples attention memory (O(N^2)), that a 100K-token context window requires ~240 GB of attention memory per layer, and that 32 layers push the total to ~7,680 GB -- far beyond any single GPU. The scaling challenge forces students to find the maximum sequence length that fits on an 80 GB GPU, discovering the hard physical ceiling that no amount of raw FLOPS can overcome.

---

## 3. Act 1: The Arithmetic Intensity Gap (Calibration -- 12 minutes)

### Pedagogical Goal
Students believe that performance on a GPU is primarily a function of model size or FLOPs count. The chapter establishes that arithmetic intensity -- the ratio of operations performed per byte moved -- determines whether a workload is compute-bound or memory-bound. ResNet-50 reuses each weight across spatial dimensions, achieving I ~40 FLOPs/byte; GPT-2 at inference loads the entire model for a single matrix-vector multiply per token, yielding I ~0.5 FLOPs/byte. This 80x gap means the same H100 GPU operates in completely different regimes for these two workloads. Students who predict "roughly similar" performance discover that architecture, not hardware, determines the bottleneck.

### The Lock (Structured Prediction)
Present a multiple-choice prediction before any instruments unlock:

> "ResNet-50 and GPT-2 are both deployed on the same H100 GPU. ResNet-50 performs 4.1 GFLOPs per image inference; GPT-2 performs 3.0 GFLOPs per token inference. Both are 'deep learning' workloads on identical hardware. How different is their arithmetic intensity (FLOPs per byte of data moved)?"

Options:
- A) About the same (~1--2x difference) -- both are neural networks doing matrix math
- B) About 5x difference -- Transformers are a bit less efficient
- **C) About 80x difference -- they occupy opposite ends of the intensity spectrum** <-- correct
- D) About 10x difference -- GPT-2 is larger so it moves more data

**Common wrong answer:** A or B. Students equate "neural network" with a single computational profile. They see similar GFLOPs counts (4.1 vs 3.0) and assume similar hardware behavior. They do not yet understand that GFLOPs-per-inference says nothing about how much data must be moved to perform those operations.

**Why wrong:** ResNet-50 reuses each convolutional filter across thousands of spatial positions (weight sharing), achieving high data reuse. GPT-2 at inference performs only a matrix-vector multiply per layer per token, loading the entire 6 GB weight matrix (FP32) for ~3 GFLOPs of compute. The FLOPs counts are similar but the bytes-moved denominators differ by ~60x.

### The Instrument: Arithmetic Intensity Comparator

A side-by-side comparison panel for three Lighthouse architectures: **ResNet-50**, **GPT-2 (inference)**, and **MobileNetV2**.

For each, show:
- **Arithmetic Intensity (I)**: ResNet-50 ~40.2, GPT-2 ~0.50, MobileNet ~21.4 FLOPs/byte
- **Total FLOPs per inference**: ResNet-50 = 4.1 GFLOPs, GPT-2 = 3.0 GFLOPs/token, MobileNet = 300 MFLOPs
- **Data moved (FP32 weights)**: ResNet-50 = 102 MB, GPT-2 = 6.0 GB, MobileNet = 14 MB
- **Bottleneck regime**: Compute-bound / Memory-bandwidth-bound / Balanced
- **GPU utilization indicator**: A bar showing what fraction of H100 peak FLOPS the workload can utilize at batch=1

Controls:
- **Architecture selector** (radio: ResNet-50 / GPT-2 / MobileNetV2): Switches the primary display
- **Batch size slider** (1, 2, 4, 8, 16, 32, 64): Increasing batch size increases arithmetic intensity for GPT-2 (matrix-vector becomes matrix-matrix) -- students discover that batching is the primary lever for bandwidth-bound workloads

A **horizontal bar chart** showing all three architectures simultaneously:
- X-axis: Arithmetic Intensity (FLOPs/byte), log scale, range 0.1--1000
- Y-axis: Three model bars
- A vertical dashed line at the hardware ridge point (~100 FLOPs/byte for H100) separating "memory-bound" (left) from "compute-bound" (right)
- GPT-2's bar is deep in the memory-bound region; ResNet-50's bar approaches or crosses the ridge point

### The Reveal
After interaction, overlay the student's prediction on the actual intensity gap:

> "You predicted [X] difference. The actual arithmetic intensity ratio is **80x**: ResNet-50 at 40.2 FLOPs/byte vs GPT-2 at 0.50 FLOPs/byte. On the same H100, ResNet-50 utilizes the compute units efficiently because it reuses weights across spatial dimensions. GPT-2 at batch=1 wastes >98% of peak FLOPS because it must load 6 GB of weights to perform just 3 GFLOPs of math per token."

Then surface the MobileNet counterintuition:

> "MobileNet has 14x fewer FLOPs than ResNet-50 but can run *slower* on datacenter GPUs. Its depthwise separable convolutions reduce total operations but lower arithmetic intensity to ~1--10 FLOPs/byte, starving GPU compute units. FLOPs do not equal speed."

### Reflection (Structured)
Four-option multiple choice:

> "GPT-2 performs ~3 GFLOPs per token on an H100 capable of 3,958 TFLOPS (FP8). Why does it achieve <1% utilization at batch=1?"

- A) GPT-2 has too few parameters to keep the GPU busy
- B) The H100 is not optimized for Transformer architectures
- **C) The arithmetic intensity is too low -- the GPU finishes math faster than memory can deliver weights** <-- correct
- D) Autoregressive generation disables GPU parallelism entirely

**Math Peek (collapsible):**
$$I = \frac{\text{FLOPs}}{\text{Bytes Moved}} \qquad I_{\text{ResNet}} = \frac{4.1 \times 10^9}{102 \times 10^6} \approx 40.2 \qquad I_{\text{GPT-2}} = \frac{3.0 \times 10^9}{6.0 \times 10^9} \approx 0.50$$

---

## 4. Act 2: The Quadratic Wall (Design Challenge -- 22 minutes)

### Pedagogical Goal
Students do not viscerally grasp how quadratic scaling transforms a tractable system into an impossible one. The chapter establishes that attention memory scales as O(N^2) with sequence length: doubling the context window quadruples memory. A 100K-token context requires ~240 GB of attention memory per layer; across 32 layers, the total reaches ~7,680 GB -- nearly 100x the capacity of an 80 GB H100. This act forces students to discover the maximum sequence length that fits on a single GPU, then watch how quickly the quadratic wall crushes their configurations as they push toward longer contexts.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "A Transformer with 12 attention heads and 32 layers uses FP16 precision. The H100 GPU has 80 GB of memory. Assuming the entire GPU memory is available for attention matrices, what is the maximum sequence length (in tokens) the model can support?"

Students type a number (tokens). Expected wrong answers: 50,000--500,000 tokens (students linearly extrapolate from "80 GB is a lot of memory"). Actual answer: the maximum is approximately **~4,714 tokens** per layer when computed as sqrt(80e9 / (12 * 2)) considering single-layer constraint, or **~833 tokens** when all 32 layers must fit simultaneously (sqrt(80e9 / (32 * 12 * 2))). The precise answer depends on how memory is shared across layers, which the instrument lets students explore.

### The Instrument: Attention Memory Explorer

An interactive memory calculator for Transformer attention matrices.

**Primary chart: Stacked bar -- Attention Memory vs GPU Capacity**
- X-axis: Component (one bar per layer group or total)
- Y-axis: Memory (GB), range 0--500 (with overflow indicator beyond)
- A red threshold line at the GPU memory budget (80 GB for H100, 2 GB for Edge)
- Bar color: BlueLine when below threshold, RedLine when OOM

Controls:
- **Sequence length slider** (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 100000): The primary knob. Attention memory = N^2 * num_heads * bytes_per_element per layer
- **Number of layers** (1, 4, 8, 16, 32): Total attention memory scales linearly with layers
- **Number of attention heads** (4, 8, 12, 16, 32): Scales linearly
- **Precision toggle** (FP16 = 2 bytes / FP32 = 4 bytes): Halving precision halves memory

**Secondary chart: Quadratic Scaling Curve**
- X-axis: Sequence length (512 -- 100,000), log scale
- Y-axis: Attention memory per layer (MB -- TB), log scale
- A curve showing N^2 growth with current head/precision settings
- Horizontal threshold lines at 80 GB (H100) and 2 GB (Edge GPU)
- The intersection point is annotated: "Maximum feasible sequence length = [N]"

**Live readout panel:**
- Attention memory per layer: N^2 * H * bytes_per_element
- Total attention memory: per_layer * num_layers
- GPU memory remaining: budget - total_attention (or "OOM" in red)
- Scaling factor display: "2x sequence length = 4x memory"

### The Scaling Challenge
**"Find the maximum sequence length where the full 32-layer, 12-head Transformer fits within 80 GB of attention memory (FP16)."**

Students must adjust the sequence length slider to find the boundary. The formula is:

$$\text{Total Attention Memory} = N^2 \times H \times \text{bytes} \times L$$

For L=32, H=12, bytes=2: Total = N^2 * 768. Setting this equal to 80 GB:
N^2 = 80e9 / 768 => N ~= 10,206 tokens.

Students discover that 10K tokens is the hard ceiling for this configuration -- far below the 100K+ context windows they associate with modern LLMs. The instrument then reveals: FlashAttention avoids materializing the full N*N matrix by tiling the computation, processing blocks that fit in SRAM. The compute is the same O(N^2), but the memory footprint drops to O(N).

### The Failure State
**Trigger condition:** `total_attention_memory > gpu_memory_budget`

**Visual change:** The stacked bar chart turns RedLine. The memory readout displays negative remaining memory.

**Banner text:**
> "OOM -- Attention matrix exceeds GPU memory. Required: [X] GB | Available: 80 GB. Reduce sequence length, layers, heads, or switch to FP16."

The failure state is reversible: students pull the sequence length slider back and watch the system recover. The point is to find the exact boundary where the quadratic wall hits.

### Structured Reflection
Four-option multiple choice:

> "At sequence length 512 (BERT's limit), attention memory for one layer with 12 heads at FP16 is ~6 MB. At sequence length 100,000 (modern LLMs), the same layer requires ~240 GB. What scaling law governs this 40,000x increase?"

- A) Linear -- doubling sequence length doubles memory
- B) Log-linear -- memory grows as N log N
- **C) Quadratic -- doubling sequence length quadruples memory, so 200x longer sequences require 40,000x memory** <-- correct
- D) Cubic -- attention involves three matrices (Q, K, V) so memory scales as N^3

**Math Peek:**
$$\text{Attention Memory (per layer)} = N^2 \times H \times b \qquad \text{Total} = N^2 \times H \times b \times L$$
$$\text{At } N = 100{,}000,\ H = 12,\ b = 2,\ L = 32: \quad 10^{10} \times 12 \times 2 \times 32 = 7{,}680 \text{ GB}$$

---

## 5. Visual Layout Specification

### Act 1: Arithmetic Intensity Gap

- **Primary:** Horizontal bar chart comparing arithmetic intensity across three Lighthouse architectures (log-scale x-axis, 0.1--1000 FLOPs/byte). Vertical dashed line at hardware ridge point (~100 FLOPs/byte). GPT-2 bar in RedLine (memory-bound), ResNet-50 bar in BlueLine (compute-bound), MobileNet in OrangeLine (balanced).
  - X-axis: Arithmetic Intensity (FLOPs/byte), log scale
  - Y-axis: Model name (categorical)
  - Data series: One bar per model, colored by bottleneck regime
  - Ridge point line: vertical dashed at ~100, labeled "H100 Ridge Point"

- **Secondary:** GPU utilization gauge per architecture. Three circular gauges (0--100%) showing estimated GPU FLOPS utilization at batch=1. ResNet-50: ~60--70%. MobileNet: ~15--30%. GPT-2: <2%.

- **Prediction overlay:** Student's selected option highlighted, correct answer (80x) revealed with gap annotation.

### Act 2: The Quadratic Wall

- **Primary:** Stacked bar chart -- attention memory total vs GPU memory budget.
  - X-axis: "Attention Memory" (single bar, subdivided by layers if helpful)
  - Y-axis: Memory (GB), 0--500+
  - Threshold line at 80 GB (H100) or 2 GB (Edge), color-coded
  - Bar turns RedLine when OOM triggered

- **Secondary:** Quadratic scaling curve -- sequence length vs attention memory.
  - X-axis: Sequence length (tokens), log scale, 512--100,000
  - Y-axis: Memory per layer (MB to TB), log scale
  - Curve: N^2 * H * b at current settings
  - Horizontal threshold lines at device memory budgets
  - Intersection annotation: "Max N = [value]"

- **Failure state:** Full OOM banner when total memory exceeds GPU budget. Bar chart turns RedLine. Reversible via slider adjustment.

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Key Constraint | Bottleneck Regime |
|---|---|---|---|---|
| **Training Node** | H100 (80 GB HBM3) | 80 GB | High bandwidth (3 TB/s); compute-bound workloads (ResNet) saturate ALUs; bandwidth-bound workloads (GPT-2 inference) waste >98% of peak FLOPS | Architecture determines whether the GPU is compute-starved or bandwidth-starved |
| **Edge Inference** | Mobile GPU (2 GB) | 2 GB | Tiny memory budget; even moderate sequence lengths (2K--4K) cause OOM for attention matrices; forces architectural choices toward CNNs or highly constrained Transformers | The quadratic wall hits orders of magnitude earlier |

The two contexts demonstrate that the same architectural decision (CNN vs Transformer) has different severity depending on the deployment target. On the H100, GPT-2 is bandwidth-bound but feasible; on the Edge GPU, a 4K-token Transformer's attention matrix alone exceeds memory. The architecture-hardware contract is non-negotiable at both scales, but the edge device makes the constraint visceral.

---

## 7. Design Ledger Output

```json
{
  "chapter": 6,
  "ch06": {
    "intensity_gap_prediction_error_x": "<predicted_ratio - 80>",
    "max_seq_len_80gb": "<N tokens found in scaling challenge>",
    "bottleneck_regime_chosen": "compute | bandwidth",
    "architecture_lighthouse": "resnet50 | gpt2 | mobilenet",
    "quadratic_scaling_understood": true
  }
}
```

- **Lab 08 (Training)** reads `bottleneck_regime_chosen` to set the default architecture in the training throughput optimizer -- students who chose a bandwidth-bound architecture see a different MFU profile than those who chose compute-bound.
- **Lab 11 (HW Acceleration)** reads `max_seq_len_80gb` and `architecture_lighthouse` to initialize the Roofline Model visualization with the student's chosen architecture, placing their workload on the roofline plot at the correct arithmetic intensity.

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| ResNet-50 arithmetic intensity ~40.2 FLOPs/byte | `@sec-network-architectures-understanding-arithmetic-intensity-ade5`, `WorkloadSignatures` class (line 260) | `resnet_i = resnet_flops / resnet_bytes` = 4.1e9 / 102e6 ~= 40.2 |
| GPT-2 arithmetic intensity ~0.50 FLOPs/byte | `WorkloadSignatures` class (line 261) | `gpt2_i = gpt2_flops_token / gpt2_bytes` = 3.0e9 / 6.0e9 = 0.50 |
| 80x intensity gap (guard check) | `WorkloadSignatures` class (line 265) | `check(resnet_i > gpt2_i * 50, ...)` -- the code verifies ResNet intensity is >50x GPT-2's |
| ResNet-50: compute-bound, high intensity 50--200+ FLOPs/byte | `@sec-network-architectures-workload-signatures` (line 433) | "High intensity ($\approx 50\text{--}200+$ FLOPs/byte, varying by layer)" |
| GPT-2: bandwidth-bound, ~1 FLOPs/byte | Line 434 | "Low intensity ($\approx 1$ FLOPs/byte). Each token produces only a matrix-vector multiplication" |
| MobileNet: 14x fewer FLOPs, can run slower on GPUs | Key Takeaways (line 4372) | "MobileNet uses 14x fewer FLOPs than ResNet-50 but can run *slower* on datacenter GPUs because its low arithmetic intensity starves compute units" |
| Quadratic attention scaling: doubling N quadruples memory | `TransformerScaling` class (lines 400--408) and `TransformerComplexityAnchor` (line 2615) | `scaling_ratio = (1024 / 512)**2 = 4.0`; "check(scaling_ratio == 4.0)" |
| 100K context window: ~240 GB per layer, ~7,680 GB total | `AttentionMemory` class (lines 2619--2652) | `seq_len=100,000; num_heads=12; bytes_per_element=2; num_layers=32`; single_layer ~240 GB, total ~7,680 GB |
| BERT limited to 512 tokens due to quadratic wall | War Story: "The Quadratic Wall" (line 2699) | "The self-attention mechanism's memory requirement scales quadratically ($O(N^2)$). Doubling the context from 512 to 1024 would quadruple the memory" |
| Attention definition: O(N^2) memory, O(1) depth | `@sec-network-architectures` callout-definition "Attention Mechanisms" (line 2118) | "Attention connects any two tokens in O(1) depth, but the similarity matrix requires O(N^2) memory" |
| FlashAttention: tiling avoids materializing full N*N matrix | Line 2673, line 2921 | "FlashAttention (tiling to avoid materializing the full matrix)"; "tiles computation to avoid materializing the full matrix in HBM" |
| Context explosion: BERT 512 to Gemini 1M+ | `@fig-context-explosion` data (lines 2969--2971) | "(2018, 512, 'BERT'), (2019, 1024, 'GPT-2'), ... (2024.1, 1000000, 'Gemini 1.5')" |
| Architecture determines bottleneck: five Lighthouse models | `@tbl-workload-signatures` and Key Takeaways (line 4370) | "Lighthouse models isolate distinct bottlenecks: ResNet-50 (compute), GPT-2 (bandwidth), DLRM (capacity), MobileNet (latency), KWS (power)" |
| Dense layer intensity ~0.5 FLOPs/byte at batch=1 | `@eq-dense-intensity` (line 1060) | "Intensity ~= 2*M*N (Ops) / 4*M*N (Bytes) = 0.5 FLOPs/byte" |

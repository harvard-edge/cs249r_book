# Mission Plan: lab_07_ml_frameworks

## 1. Chapter Alignment

- **Chapter:** ML Frameworks (`@sec-ml-frameworks`)
- **Core Invariant:** The **Dispatch Tax** — each kernel launch incurs 5–20 μs of CPU-side overhead, independent of the computation performed. For small models with many small operations, this fixed overhead dominates total latency. For large batch, large model inference, it is negligible. The correct execution strategy is therefore workload-dependent: eager mode for development and debugging; compiled graph mode for production throughput.
- **Central Tension:** Students believe that `torch.compile` or graph compilation is always better — "more optimization = faster." The chapter shows the opposite: for a small KWS model with 1,000 tiny kernels, the 5–20 μs dispatch tax per kernel means the GPU is compute-busy for < 1% of wall time, and compilation provides a 1.3–2× throughput gain. But for a model with one giant matrix multiply, the dispatch tax is negligible and compilation provides minimal gain. The break-even depends on arithmetic intensity, not model "complexity."
- **Target Duration:** 35–40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that larger models benefit more from compilation because they have "more to optimize." This act shows the opposite: compilation speedup is highest for models with many small kernels (low arithmetic intensity), where the dispatch tax dominates. A KWS model with 1,000 tiny ops sees >30% speedup from `torch.compile`; a single large matmul sees near-zero speedup because the dispatch tax is already negligible.

**Act 2 (Design Challenge, 22 min):** Students apply kernel fusion to a real pipeline — a LayerNorm + Dropout + ReLU sequence — and discover the 5× speedup from fusing three reads/writes into one. Then they confront the compilation break-even: a 48-second compile time is only justified if the model runs ≥ N iterations. Students compute N for their specific deployment scenario and determine whether compilation is net-positive.

---

## 3. Act 1: The Dispatch Tax Audit (Calibration — 12 minutes)

### Pedagogical Goal
Students believe GPU utilization is limited by compute density (TFLOPS). The chapter's key insight is that for small models, utilization is limited by the *dispatch rate*: at 5–20 μs per kernel launch and 1,000 kernels per forward pass, the GPU is busy for at most 1–5% of wall time on computation — the other 95–99% is framework overhead. A faster GPU would not help; it would simply wait faster. Only kernel fusion or compilation can raise utilization for this class of model.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "A Keyword Spotting model performs 1,000 kernel launches per forward pass. Each kernel computes for an average of 5 μs. Each kernel *launch* costs 10 μs of CPU-side overhead. What fraction of wall-clock time is the GPU actually performing tensor operations?"

Options:
- A) About 90% — the GPU is the bottleneck, not the CPU
- B) About 50% — half compute, half overhead
- **C) About 33% — compute (5 μs) is half the launch overhead (10 μs), so compute = 5/(5+10) = 33%** ← correct
- D) About 5% — the overhead completely dominates

The arithmetic: 1,000 kernels × (5 μs compute + 10 μs launch) = 15,000 μs total; compute = 5,000 μs / 15,000 μs = 33%.

### The Instrument: Dispatch Tax Waterfall

A **latency waterfall** decomposing one forward pass into:
- **Kernel Compute Time** = `kernel_count × compute_per_kernel`
- **Dispatch Overhead** = `kernel_count × launch_latency`
- **Memory Transfer** = computed from arithmetic intensity

Controls:
- **Kernel count slider**: 10 → 10,000 kernels per forward pass. As kernel count rises, the dispatch bar grows proportionally while compute stays near-constant (same total work, more fragmented).
- **Model type selector**:
  - KWS (Keyword Spotting): ~1,000 small kernels, each ~5 μs compute
  - ResNet-50: ~200 medium kernels, each ~50 μs compute
  - GPT-2 Layer: ~20 large kernels, each ~500 μs compute
- **Execution mode toggle**: Eager / Compiled. In compiled mode:
  - Kernel count drops by 30–80% (operator fusion merges adjacent ops)
  - Dispatch overhead drops proportionally
  - Compute time is unchanged or slightly reduced (fusion eliminates intermediate read/writes)

A **GPU Utilization meter** shows: `compute_time / (compute_time + dispatch_overhead + memory_transfer)`. Students observe:
- KWS eager: ~33% utilization
- KWS compiled: ~60–70% utilization (>30% speedup from chapter)
- GPT-2 layer eager: ~90% utilization (dispatch is negligible vs. long matmuls)
- GPT-2 layer compiled: ~92% utilization (minimal gain)

### The Reveal
After interaction:
> "You predicted [X]% GPU utilization. The actual value for KWS eager is **33%**. Your prediction was off by [Y] percentage points. Note: switching from KWS to GPT-2 (20 large kernels instead of 1,000 small ones) raises utilization to 90% without any code changes — the dispatch tax is diluted across longer compute operations."

Surface the key asymmetry:
> "A 2× faster GPU would not fix the 33% KWS utilization. It would complete the 5 μs compute in 2.5 μs and then wait 10 μs for the next launch. Faster hardware amplifies the dispatch tax, not reduces it."

### Reflection (Structured)
Students select the correct statement:

> "A faster GPU sometimes produces *lower* utilization for small models because:"
- A) Faster GPUs have higher power draw, which causes thermal throttling
- B) Faster GPUs require more time to warm up before peak throughput
- **C) Faster compute reduces the compute fraction of each kernel, making the fixed dispatch overhead a larger share of total time** ← correct
- D) Faster GPUs use different memory hierarchies that are incompatible with small models

**Math Peek (collapsible):**
$$\text{GPU Utilization} = \frac{N \cdot t_{compute}}{N \cdot (t_{compute} + t_{launch}) + t_{memory}}$$
$$t_{launch} \in [5, 20] \; \mu\text{s per kernel (CPU-side)}$$

---

## 4. Act 2: The Compilation Break-Even (Design Challenge — 22 minutes)

### Pedagogical Goal
Students believe compilation is a free speedup — "compile once, run fast forever." The chapter shows compilation has a concrete cost: `torch.compile` on ResNet-50 takes ~48 seconds and provides ~48% throughput gain (2,150 vs. 1,450 img/sec). For a web server handling 10,000 requests/day, this break-even is trivially positive. For a CI pipeline running a model once per PR, the 48-second compile time exceeds the total inference cost. Students must compute the break-even point for their deployment and decide whether compilation is net-positive.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "torch.compile on ResNet-50 improves throughput by 48% (from 1,450 to 2,150 images/sec) but requires 48 seconds of one-time compilation. How many inferences must you run before the compilation cost is recovered?"

Students type a number. Expected wrong answers: very large numbers (students expect high overhead). Actual calculation: at 1,450 img/sec baseline, the break-even number of inferences where compilation time is offset by throughput gain is:
$$N_{break-even} = \frac{t_{compile}}{\Delta t_{per\text{-}inference}} = \frac{48s}{1/1450 - 1/2150} \approx 48s \times \frac{1450 \times 2150}{2150-1450} \approx 214,000 \text{ images}$$

### The Instrument: Compilation Trade-off Analyzer

**Two panels:**

**Panel A: Kernel Fusion Explorer**
A concrete LayerNorm + Dropout + ReLU fusion example.

Without fusion: 3 separate kernel launches, each reading/writing to HBM:
- LayerNorm: read input (2 GB/s read), compute, write output to HBM
- Dropout: read LayerNorm output, compute, write output to HBM
- ReLU: read Dropout output, compute, write final output

With fusion: 1 kernel launch, input read once, output written once — 10–20× less HBM traffic.

Controls:
- **Fusion toggle** (Off / On): Shows before/after memory traffic bar and kernel count.
- **HBM bandwidth slider**: 1 / 2 / 3.35 TB/s (A100 HBM3 = 3.35 TB/s). As bandwidth increases, the unfused case improves proportionally; the fused case is already compute-bound and doesn't improve as much.

Output: **Arithmetic intensity meter** (FLOP/byte). Students observe:
- Unfused sequence: ~0.1 FLOP/byte (memory-bound; below roofline ridge point of 156 FLOP/byte)
- Fused sequence: ~0.8 FLOP/byte (still memory-bound, but 8× better)
- Students can confirm: even fused, element-wise ops are memory-bound. Only matmuls clear the ridge point.

**Panel B: Compilation ROI Calculator**
- **Compilation cost slider**: 10 / 48 / 120 / 300 seconds (range of real torch.compile times)
- **Throughput gain slider**: 5% / 30% / 48% / 100% (typical ranges by model type)
- **Inference volume slider**: 1K / 10K / 1M / 100M per day
- **Deployment duration slider**: 1 run / 1 day / 1 week / 1 year

Output: **Break-even visualization** — a timeline showing cumulative time saved (green) vs. compilation overhead (red). The crossover point is labeled "Break-even at [N] inferences." A "Net ROI" badge appears when the timeline goes green.

**Failure state (negative ROI):** When inferences × gain < compilation cost:
> "🟠 **Compilation Not Justified.** At [N] inferences/day, compilation overhead is recovered after [X] days. For a 1-day deployment, eager mode is faster overall."

### The Scaling Challenge
**"Find the minimum production deployment length where torch.compile is net-positive for a KWS model serving 100 requests/hour."**

- KWS compile time: ~10 seconds (small model)
- KWS throughput gain: ~40% (high gain because dispatch-bound)
- 100 requests/hour = ~0.028 req/sec

Students slide the deployment duration until the break-even crossover appears. Key discovery: even for a low-throughput deployment (100 req/hr), the KWS model's compile time of 10 seconds is recovered quickly — but the break-even number varies dramatically by model size and compilation overhead.

### Structured Reflection
Complete the sentence:

> "Kernel fusion provides [5× / 10–20× / 48%] speedup for LayerNorm + Dropout + ReLU because ___."

Dropdown for blank 1: **"5× for wall-clock speedup; 10–20× for HBM traffic reduction"** ← both are chapter-correct; the lab uses 5× wall-clock and 10–20× for HBM.
Dropdown for blank 2:
- **"fusing eliminates intermediate HBM reads and writes between operations, making one pass through memory serve all three computations"** ← correct
- "fusing increases arithmetic intensity above the roofline ridge point"
- "fusing allows the GPU to use Tensor Cores for element-wise operations"
- "fusing reduces the Python interpreter overhead"

Then four-option multiple choice:
> "For a model with one giant matrix multiply (200 ms compute per kernel, 1 kernel total), torch.compile will provide:"
- **A) Near-zero speedup — the dispatch overhead (10 μs) is negligible vs. 200 ms compute** ← correct
- B) ~48% speedup — the same as ResNet-50
- C) >2× speedup — large kernels benefit most from optimization
- D) Negative speedup — compilation makes large kernels slower

**Math Peek:**
$$N_{break-even} = \frac{t_{compile}}{\Delta t_{per\text{-}inference}} \qquad \text{5× fusion speedup: } \frac{3 \text{ kernels} \times t_{HBM}}{1 \text{ kernel} \times t_{HBM}} \approx 3\times \text{ memory, } 5\times \text{ wall-clock}$$

---

## 5. Visual Layout Specification

### Act 1: Dispatch Tax Waterfall
- **Primary:** Horizontal stacked bar (one forward pass = 100%): Kernel Compute / Dispatch Overhead / Memory Transfer. Three segments: BlueLine / OrangeLine / GreenLine.
- **Secondary:** GPU Utilization meter (0–100%, threshold line at 80%). Updates live with kernel count and model type sliders.
- **Prediction overlay:** Student's computed % annotated against actual bar proportions.
- **Model comparison table:** KWS / ResNet-50 / GPT-2 Layer — kernel count, compute per kernel, dispatch per kernel, utilization (eager vs. compiled).

### Act 2: Compilation Break-Even
- **Primary Panel A:** Memory traffic bar (unfused: 3 bars; fused: 1 bar) + Arithmetic Intensity meter.
- **Primary Panel B:** Break-even timeline — X-axis: elapsed time (seconds to days), Y-axis: cumulative time saved. Red line = compilation overhead; Green line = cumulative speedup. Crossover = break-even point.
- **Failure state (Panel B):** OrangeLine banner when break-even > deployment duration.

---

## 6. Deployment Context Definitions

| Context | Device | Inference Volume | Key Constraint |
|---|---|---|---|
| **Production Server** | A100 (312 TFLOPS FP16, 2.0 TB/s HBM) | 10M requests/day | Compilation ROI positive within minutes; fusion provides 10–20× HBM reduction for transformer attention |
| **Edge Inference** | Mobile NPU (10 TOPS INT8, 68 GB/s) | 100 requests/hour | Compilation overhead may exceed deployment lifetime for short sessions; dispatch tax is higher % of budget |

The two contexts reveal that compilation decisions are not universal: on a production server handling 10M requests/day, compile-once-run-always is always net-positive. On an edge device with episodic deployments, eager mode may be more efficient overall — a concrete quantitative decision, not a preference.

---

## 7. Design Ledger Output

```json
{
  "chapter": 7,
  "execution_mode": "eager | compiled",
  "fusion_enabled": true,
  "compilation_roi_positive": true,
  "breakeven_inferences": 214000,
  "kws_utilization_eager_pct": 33,
  "kws_utilization_compiled_pct": 67
}
```

The `execution_mode` and `fusion_enabled` fields feed forward to:
- **Lab 08 (Training):** The framework dispatch overhead becomes part of the MFU pipeline breakdown
- **Lab 11 (HW Acceleration):** The arithmetic intensity values from fusion feed into the roofline analysis

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| 5–20 μs kernel launch overhead | frameworks.qmd, line 307 | "Each kernel launch incurs 5–20 μs of CPU-side overhead" |
| <1% peak compute for ReLU (unfused) | frameworks.qmd, line 301 | "element-wise operations like ReLU achieve less than 1% of peak compute capacity" |
| >30% speedup from torch.compile | frameworks.qmd, line 127 | "forfeiting potential speedups of over 30% that compilers like torch.compile can provide" |
| 1.3–2× throughput gain (torch.compile) | frameworks.qmd, line 1036 | "a permanent 1.3–2× throughput gain on transformer models by reducing kernel launch overhead" |
| 5× speedup (LayerNorm + Dropout + ReLU fusion) | frameworks.qmd, line 305 | "Fusing a sequence of LayerNorm, Dropout, and ReLU into one kernel can yield 5× speedup" |
| 10–20× HBM traffic reduction (FlashAttention) | frameworks.qmd, line 305 | "reducing HBM traffic by 10–20×" |
| 2–4× wall-clock speedup (FlashAttention) | frameworks.qmd, line 305 | "achieving 2–4× wall-clock speedup" |
| 48% speedup on ResNet-50 (torch.compile) | frameworks.qmd, line 1283 | "torch.compile provides ~48% speedup on ResNet-50 (2,150 vs 1,450 img/sec)" |
| 2.0 TB/s A100 bandwidth | frameworks.qmd, line 301 | "A100 GPU with…2.0 TB/s of memory bandwidth" |
| 30–80% utilization range (framework choice) | frameworks.qmd, line 2663 | "whether a training loop achieves 30% or 80% of theoretical hardware throughput" |
| 60× bandwidth gap (PCIe vs HBM) | frameworks.qmd, line 2661 | "bandwidth gap, exceeding 60×, means a single misplaced tensor transfer can erase the entire speedup" |
| 156 ops/byte ridge point (A100 FP32) | frameworks.qmd, implied | 312 TFLOPS ÷ 2.0 TB/s = 156 FLOP/byte |

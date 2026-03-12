# Mission Plan: lab_12_perf_bench

## 1. Chapter Alignment

- **Chapter:** Benchmarking (`@sec-benchmarking`)
- **Core Invariant:** **Amdahl's Law** (Principle 8) caps all optimization gains: `Speedup = 1 / ((1-p) + p/s)`. A 5x inference speedup yields only 1.8x end-to-end improvement when the serial fraction (preprocessing) consumes 44% of total latency. The gap between peak and sustained performance is structurally unavoidable (2--10x).
- **Central Tension:** Students believe that optimizing the model (inference) is the highest-leverage action. The chapter's Amdahl calculation shows that preprocessing consumes 44% of an 18 ms vision pipeline, capping maximum end-to-end speedup at 2.27x regardless of inference optimization. Students who benchmark only the inference component report 5x speedup; those who benchmark end-to-end discover 1.8x. The serial fraction, not the model, is the bottleneck.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that a 5x inference speedup translates linearly to end-to-end performance. The chapter's Amdahl's Law calculation exposes the optimization ceiling: with preprocessing at 8 ms and inference at 10 ms, even infinite inference speed yields at most 2.27x total speedup. Students predict the end-to-end improvement, discover it is 1.8x (not 5x), and learn to identify the serial fraction as the binding constraint before investing optimization effort.

**Act 2 (Design Challenge, 23 min):** Students face a complete benchmarking audit: given a MobileNetV2 inference pipeline on two hardware targets (H100 cloud vs. Jetson Orin NX edge), they must find the configuration that meets a latency SLA while maximizing throughput. The instruments expose the gap between peak TFLOPS and sustained performance, the effect of batch size on throughput-vs-latency, and the Amdahl ceiling that shifts as preprocessing fraction changes across deployment contexts. Students discover that the edge device, despite lower peak TFLOPS, can meet the SLA more efficiently because its serial fraction is smaller (zero-copy inference eliminates data transfer overhead).

---

## 3. Act 1: The Optimization Ceiling (Calibration -- 12 minutes)

### Pedagogical Goal
Students dramatically overestimate the end-to-end benefit of model-only optimization. The chapter demonstrates that vendors report component latency (5--10 ms for model inference) while production latency includes preprocessing, queuing, and postprocessing (50--100 ms total), and that a 3x inference speedup in isolation might yield only 1.3x end-to-end improvement. This act forces students to predict the end-to-end speedup from a known component speedup, then see Amdahl's Law in action.

### The Lock (Structured Prediction)
Present a **multiple-choice prediction** before any instruments unlock:

> "A vision inference pipeline has two stages: preprocessing (JPEG decode + resize + normalize) takes 8 ms and model inference takes 10 ms. Total: 18 ms. You optimize the model with INT8 quantization, achieving a 5x inference speedup (inference drops from 10 ms to 2 ms). What is the end-to-end speedup?"

Options:
- A) About 5x -- the inference speedup propagates directly
- B) About 3x -- some overhead remains, but most of the pipeline benefits
- **C) About 1.8x -- preprocessing is unchanged and now dominates** <-- correct
- D) About 1.2x -- the pipeline barely improves at all

The correct answer is C: total drops from 18 ms to 10 ms (8 + 2), yielding 18/10 = 1.8x. Most students pick A or B because they focus on the optimized component and ignore the unoptimized serial fraction. The insight: you must benchmark the entire pipeline, not just the part you changed.

### The Instrument: Amdahl's Law Visualization

A **stacked bar chart** showing the latency breakdown:

- **X-axis:** Configuration (Before Optimization, After 2x, After 5x, After 10x, After Infinite)
- **Y-axis:** Latency (ms), 0 to 20 ms
- **Stacked segments:** Preprocessing (BlueLine), Inference (GreenLine)
- A **horizontal dashed line** at the original total (18 ms) for reference
- An **Amdahl ceiling line** at 1/f = 2.27x, shown as a red asymptote on a secondary speedup axis

Controls:
- **Inference speedup slider** (1x to 100x, step 1x, default 1x): As students drag, the inference bar shrinks but the preprocessing bar stays constant. The total speedup readout shows diminishing returns.
- **Preprocessing fraction slider** (10% to 90%, step 5%, default 44%): Students adjust the serial fraction to see how Amdahl's ceiling shifts. At 90% preprocessing, even 100x inference speedup yields only 1.1x total.

**Deployment context toggle** (H100 vs. Jetson Orin NX): On H100, preprocessing includes data transfer (PCIe copy to GPU) adding 3 ms. On Jetson (unified memory), zero-copy inference eliminates this overhead, changing the serial fraction.

### The Reveal
After interaction:
> "You predicted [X]x end-to-end speedup. The actual value is **1.8x** (from 18 ms to 10 ms). You were off by [Y]x. Amdahl's Law explains why: preprocessing consumes 44% of the pipeline. The maximum possible speedup, even with infinitely fast inference, is 1/0.44 = 2.27x."

### Reflection (Structured)
Four-option multiple choice:

> "A vendor reports that their new accelerator achieves 10x faster inference on ResNet-50. Based on the chapter's data that production latency includes preprocessing, queuing, and postprocessing (50--100 ms total vs. 5--10 ms model inference), what is the most likely end-to-end speedup?"
- A) 8--10x -- the accelerator dominates total latency
- B) 4--5x -- about half the pipeline benefits
- **C) 1.3--1.5x -- model inference is only 10--20% of total production latency** <-- correct
- D) 0.9--1.0x -- the accelerator provides no benefit at all

**Math Peek (collapsible):**
$$\text{Speedup}_{\text{e2e}} = \frac{1}{(1-p) + \frac{p}{s}}$$
where $p$ is the fraction of total time spent on the optimized component and $s$ is the component speedup. With $p = 0.56$ (inference fraction) and $s = 5$: Speedup = $\frac{1}{0.44 + 0.112} = 1.81\times$

---

## 4. Act 2: The Benchmarking Audit (Design Challenge -- 23 minutes)

### Pedagogical Goal
Students believe that the hardware with higher peak TFLOPS always delivers better sustained performance. The chapter's definition of ML System Benchmarks states that "the same ResNet-50 model can deliver 10x different throughput across hardware stacks" and that "an A100 that achieves 90% utilization on ResNet-50 may achieve only 40% utilization on a recommendation system." Students must audit a pipeline across two deployment contexts, discovering that peak specs predict sustained performance poorly, and that the binding constraint shifts between deployment contexts.

### The Lock (Numeric Prediction)
Before instruments unlock:

> "An H100 advertises 990 TFLOPS FP16. A Jetson Orin NX advertises 100 TOPS INT8. For a MobileNetV2 inference pipeline with a 50 ms P99 latency SLA, which device achieves higher throughput (queries per second) within the SLA? Enter your prediction as a ratio: H100 throughput / Jetson throughput."

Students type a ratio. Expected wrong answers: 5--10x (students scale by peak TFLOPS ratio). Actual: the H100 achieves roughly 2--3x throughput within the SLA because (a) MobileNetV2 is too small to saturate the H100 (low arithmetic intensity yields low utilization), (b) network + PCIe transfer overhead adds latency not present on edge, and (c) the Jetson's unified memory enables zero-copy inference.

### The Instrument: Multi-Dimensional Benchmark Dashboard

**Primary chart -- Throughput vs. Latency Curve (Scatter):**
- **X-axis:** P99 Latency (ms), 0 to 200 ms
- **Y-axis:** Throughput (QPS), 0 to 5,000
- **Data series:** H100 curve (BlueLine), Jetson Orin NX curve (GreenLine)
- **Vertical red line:** SLA threshold (adjustable, default 50 ms)
- **Annotations:** Points to the left of the SLA line are feasible; the highest-throughput feasible point is optimal

Controls:
- **Batch size slider** (1 to 128, step power-of-2, default 1): Increasing batch size moves the operating point right (higher latency) and up (higher throughput). Students see the throughput-latency knee.
- **Precision selector** (FP32 / FP16 / INT8): Each precision level shifts both curves. INT8 delivers ~2x throughput vs. FP16 but may show accuracy drop (displayed as a warning annotation when accuracy drops below threshold).
- **SLA threshold slider** (10 ms to 200 ms, step 10 ms, default 50 ms): The red line moves; feasible configurations change.
- **Deployment context toggle** (H100 vs. Jetson Orin NX): Changes hardware constants, overhead profile, and serial fraction.

**Secondary chart -- Latency Waterfall (Stacked Bar):**
- **Segments:** Data Transfer, Preprocessing, Inference, Postprocessing, Queue Wait
- Shows where time goes for the current configuration
- On H100: Data Transfer is significant (PCIe). On Jetson: Data Transfer is near-zero (unified memory).

**Tertiary chart -- Utilization Gauge:**
- Shows achieved TFLOPS / peak TFLOPS as a percentage
- H100 at batch=1 on MobileNetV2: ~5--15% utilization (vastly underutilized)
- Jetson at batch=1 on MobileNetV2: ~40--60% utilization (well-matched)

### The Scaling Challenge
**"Find the maximum batch size on each device that keeps P99 latency under 50 ms. Compare the resulting throughput."**

Students discover:
- H100: batch=32 stays under 50 ms, achieving ~3,000 QPS, but utilization is still only ~30%
- Jetson: batch=4 stays under 50 ms, achieving ~800 QPS, utilization ~55%
- The H100 wins on absolute throughput but at far lower utilization efficiency
- Switching to INT8 on Jetson allows batch=8 within SLA, reaching ~1,200 QPS

### The Failure State
**Trigger:** `p99_latency > sla_threshold_ms`

**Visual:** The throughput-latency point turns red; the SLA line flashes. Banner appears:
> "**SLA VIOLATED -- P99 exceeds budget.** At batch size [X], P99 latency is [Y] ms (budget: [Z] ms). Reduce batch size or switch to a lower-precision format."

**Secondary failure (utilization):** When utilization < 10%:
> "**HARDWARE UNDERUTILIZED -- Peak TFLOPS wasted.** MobileNetV2 arithmetic intensity is below the ridge point of this device. The workload is memory-bound; adding compute yields zero benefit (Arithmetic Intensity Law)."

### Structured Reflection
Four-option multiple choice:

> "The H100 has 10x the peak TFLOPS of the Jetson Orin NX, but achieves only 3x the throughput on MobileNetV2 within a 50 ms SLA. Why?"
- A) The H100 is defective and not running at full speed
- **B) MobileNetV2 is too small to saturate the H100; low arithmetic intensity means most peak compute sits idle, and network/transfer overhead consumes SLA budget** <-- correct
- C) The Jetson has a better compiler that generates more efficient code
- D) MobileNetV2 is compute-bound and the H100's memory bandwidth is insufficient

**Math Peek:**
$$\text{Amdahl Ceiling} = \frac{1}{f} \quad \text{where } f = \frac{T_{\text{serial}}}{T_{\text{total}}}$$
$$\text{Achieved Throughput} = \frac{R_{\text{peak}} \times \eta}{\text{FLOPs per query}} \quad \text{subject to } P99 < \text{SLA}$$

---

## 5. Visual Layout Specification

### Act 1: Amdahl Ceiling
- **Primary:** Stacked bar chart -- X: Configuration (1x through Infinite), Y: Latency (ms). Segments: Preprocessing (BlueLine), Inference (GreenLine). Dashed line at 18 ms baseline. Asymptotic ceiling annotation.
- **Secondary:** Speedup curve -- X: Inference speedup factor (1x to 100x), Y: End-to-end speedup (1x to 2.5x). Horizontal red dashed line at Amdahl ceiling (1/f). Student's prediction point overlaid.

### Act 2: Benchmarking Audit
- **Primary:** Throughput vs. P99 Latency scatter with SLA threshold line. Two series (H100, Jetson). Turns red beyond SLA.
- **Secondary:** Latency Waterfall (stacked bar) -- segments: Data Transfer, Preprocessing, Inference, Postprocessing, Queue Wait.
- **Tertiary:** Utilization gauge (circular) -- shows percentage of peak TFLOPS achieved. Orange at <30%, green at 30--70%, red at >90% (thermal throttle risk).

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---|---|---|---|---|
| **Cloud (Training Node)** | H100 (80 GB HBM3) | 80 GB | 700 W | Peak TFLOPS is high but small models cannot saturate it; PCIe transfer adds serial overhead |
| **Edge Inference** | Jetson Orin NX (8 GB) | 8 GB | 25 W | Lower peak but unified memory enables zero-copy; workload-hardware match determines real throughput |

The two contexts demonstrate that peak specs predict sustained performance only when the workload saturates the hardware. A small model on a large GPU is like driving a Formula 1 car in a parking lot: the top speed is irrelevant.

---

## 7. Design Ledger Output

```json
{
  "chapter": 12,
  "serial_fraction": 0.44,
  "amdahl_ceiling": 2.27,
  "best_batch_size_h100": 32,
  "best_batch_size_jetson": 4,
  "sla_target_ms": 50,
  "utilization_h100_pct": 30,
  "utilization_jetson_pct": 55
}
```

The `serial_fraction` and `amdahl_ceiling` feed forward to:
- **Lab 13 (Model Serving):** The serial fraction from benchmarking informs the latency budget allocation in the serving pipeline
- **Lab 16 (Conclusion):** The sensitivity analysis uses the serial fraction as a key parameter in the Tornado chart

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Preprocessing = 8 ms, Inference = 10 ms pipeline | @sec-benchmarking, Amdahl callout (line ~2527) | "preprocessing (JPEG decode, resize, normalize) consumes 8 ms and inference consumes 10 ms" |
| 5x inference speedup yields 1.8x end-to-end | @sec-benchmarking, Amdahl callout (line ~2527) | "Optimizing inference by 5x...reduces total latency from 18 ms to only 10 ms, a 1.8x improvement rather than 5x" |
| Amdahl ceiling at 1/f = 2.27x | @sec-benchmarking, Amdahl callout (line ~2529) | "if preprocessing consumes fraction f of total latency, then even infinitely fast inference yields at most 1/f speedup...the maximum achievable speedup is 1/f ≈ 2.27x" |
| Production latency 50--100 ms total vs. 5--10 ms model inference | @sec-benchmarking (line 164) | "Vendors report component latency (5--10 ms for model inference), but production latency includes preprocessing, queuing, and postprocessing (50--100 ms total)" |
| 3x inference speedup yields only 1.3x end-to-end | @sec-benchmarking (line 164) | "A 3x inference speedup in isolation might yield only 1.3x end-to-end improvement" |
| Peak vs. sustained gap is 2--3x | @sec-benchmarking (line 114) | "production Transformer training runs typically sustain 90--155 TFLOPS (30--50% MFU) -- a 2--3x gap" |
| ResNet-50 delivers 10x different throughput across stacks | @sec-benchmarking, ML System Benchmarks definition | "The same ResNet-50 model can deliver 10x different throughput across hardware stacks" |
| A100 90% utilization on ResNet-50, 40% on rec systems | @sec-benchmarking, ML System Benchmarks definition | "An A100 that achieves 90% utilization on ResNet-50 may achieve only 40% utilization on a recommendation system" |
| GPU 300 TFLOPS peak, 30 TFLOPS on memory-bound inference | @sec-benchmarking (line 349) | "A GPU advertising 300 TFLOPS might deliver only 30 TFLOPS on memory-bound transformer inference" |

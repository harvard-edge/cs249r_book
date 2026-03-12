# Mission Plan: lab_02_ml_systems

## 1. Chapter Alignment

- **Chapter:** ML Systems (`@sec-ml-systems`)
- **Core Invariant:** The Iron Law $T = D_{\text{vol}}/BW + O/(R_{\text{peak}} \cdot \eta) + L_{\text{lat}}$ decomposes all ML performance into three terms, and the Bottleneck Principle ($T_{\text{bottleneck}} = \max(\text{memory}, \text{compute}) + L_{\text{lat}}$) dictates that optimizing the non-dominant term yields exactly 0% speedup.
- **Central Tension:** Students believe that buying faster hardware (more TFLOPS) always improves performance. The chapter demonstrates that the same ResNet-50 model is compute-bound during cloud training (high batch, high arithmetic intensity) but memory-bound during single-image inference (batch=1, low arithmetic intensity). Doubling $R_{\text{peak}}$ accelerates the compute-bound case but yields zero speedup in the memory-bound case. Identifying which term dominates is the prerequisite for all optimization.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students predict what happens when a memory-bound workload receives a 2x compute upgrade. The chapter's Bottleneck Principle states that if $D_{\text{vol}}/BW > O/(R_{\text{peak}} \cdot \eta)$, doubling $R_{\text{peak}}$ yields 0% speedup. Students decompose a single ResNet-50 inference into its Iron Law terms using the Latency Waterfall instrument and discover which term dominates. The "zero percent" result is the aha moment.

**Act 2 (Design Challenge, 22 min):** Students must configure a deployment that meets a 100 ms latency SLA by manipulating batch size, precision, and hardware selection. They discover the Ridge Point -- the arithmetic intensity threshold where a workload transitions from memory-bound to compute-bound. On the Cloud context (H100), batch=64 crosses the ridge point. On TinyML (ESP32-S3), the model does not fit in memory at all at FP32, forcing quantization to INT8.

---

## 3. Act 1: The Zero-Percent Speedup (Calibration -- 12 minutes)

### Pedagogical Goal

Students believe that faster processors always help. The chapter's Bottleneck Principle demolishes this: "if your system is Memory Bound ($D_{\text{vol}}/BW > O/(R_{\text{peak}} \cdot \eta)$), buying faster processors ($R_{\text{peak}}$) yields exactly 0% speedup -- just as widening a six-lane highway yields no benefit when all traffic must funnel through a two-lane bridge." This act forces students to predict the speedup before decomposing the workload.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "You are serving ResNet-50 inference at batch=1 on an A100 GPU (312 TFLOPS FP16, 2,039 GB/s memory bandwidth). The system is slow. You upgrade to an H100 (989 TFLOPS FP16, 3,350 GB/s bandwidth). What speedup do you expect for single-image inference?"

Options:
- A) About 3x -- H100 has 3x the TFLOPS
- B) About 1.6x -- proportional to the bandwidth improvement
- **C) About 1.6x -- because batch=1 inference is memory-bound, only the bandwidth improvement matters** <-- correct
- D) About 1x -- the bottleneck is elsewhere entirely (dispatch overhead)

The answer is approximately 1.6x because single-image ResNet-50 inference is memory-bound: the compute term is negligible relative to the data term. The chapter states: "even massive accelerators (A100) are memory-bound at batch=1." The speedup comes entirely from the bandwidth ratio (3,350/2,039 = 1.64x), not the FLOPS ratio (989/312 = 3.17x).

### The Instrument: Latency Waterfall

A **stacked horizontal bar chart** (the `LatencyWaterfall` component) showing three segments for a single inference:

- **Data Term** ($D_{\text{vol}}/BW$): Model weights (98 MB FP32) / Memory BW
- **Compute Term** ($O/(R_{\text{peak}} \cdot \eta)$): 4.1 GFLOPs / (Peak FLOPS x efficiency)
- **Overhead Term** ($L_{\text{lat}}$): Dispatch tax (kernel launch, scheduling)

Controls:
- **Hardware selector** (radio): A100 / H100
- **Batch size** (slider): 1, 2, 4, 8, 16, 32, 64, 128 (powers of 2)
- **Precision** (radio): FP32 / FP16 / INT8
- **Efficiency** ($\eta$): slider, 0.1--0.8, default 0.5, step 0.05

The waterfall updates in real-time. At batch=1, the Data Term dominates (long blue bar). As batch size increases, the Compute Term grows while the Data Term stays roughly constant (weights are loaded once per batch), causing the bottleneck to shift. A **bottleneck indicator** labels the longest segment: "MEMORY-BOUND" or "COMPUTE-BOUND."

### The Reveal

After interaction:
> "You predicted [X]x speedup from A100 to H100. At batch=1: A100 latency = [Y] ms (memory-bound: data term = [Z]%, compute term = [W]%). H100 latency = [V] ms. Actual speedup: **1.64x** -- matching the bandwidth ratio (3,350/2,039), not the FLOPS ratio (989/312 = 3.17x). The 3x FLOPS advantage was invisible because the workload never reached the compute term."

### Reflection (Structured)

Four-option multiple choice:

> "At batch=1, doubling TFLOPS gives 0% speedup. What physical resource should you upgrade instead?"

- A) Storage bandwidth -- the model loads from disk each time
- **B) Memory bandwidth (BW) -- the bottleneck is loading weights from HBM to compute units** <-- correct
- C) Network bandwidth -- the bottleneck is receiving the input image
- D) CPU clock speed -- the Python overhead dominates at batch=1

**Math Peek (collapsible):**
$$T = \frac{D_{\text{vol}}}{BW} + \frac{O}{R_{\text{peak}} \cdot \eta} + L_{\text{lat}}$$
$$\text{At batch=1: } \frac{D_{\text{vol}}}{BW} \gg \frac{O}{R_{\text{peak}} \cdot \eta} \implies T \approx \frac{D_{\text{vol}}}{BW} + L_{\text{lat}}$$
$$\text{Speedup} \approx \frac{BW_{\text{H100}}}{BW_{\text{A100}}} = \frac{3{,}350}{2{,}039} \approx 1.64\times$$

---

## 4. Act 2: The Ridge Point (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students now know that batch=1 is memory-bound. The chapter states: "When does ResNet-50 become compute-bound? Increase batch size until $O/(R_{\text{peak}} \cdot \eta) > D_{\text{vol}}/BW$. On A100, this occurs around batch=64." Students must find this Ridge Point on two different hardware targets and configure a deployment that meets a 100 ms latency SLA. The design space has interacting constraints: increasing batch size improves throughput but increases latency per request; reducing precision shrinks $D_{\text{vol}}$ but may reduce accuracy.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "On an H100 (989 TFLOPS, 3,350 GB/s BW), at what batch size does ResNet-50 inference transition from memory-bound to compute-bound? Enter a number (1--512)."

Expected wrong answers: 2--8 (students underestimate how much batching is needed) or 256+ (students overestimate). Actual: approximately batch=32--64 on H100 (the Ridge Point where arithmetic intensity crosses $R_{\text{peak}}/BW$). The system will overlay the student's estimate on the actual crossover chart.

### The Instrument: Iron Law Decomposition Dashboard

**Primary chart:** A **dual-axis line chart** showing how the three Iron Law terms change with batch size:

- **X-axis:** Batch size (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) -- log scale
- **Y-axis (left):** Latency per batch (ms)
- **Y-axis (right):** Arithmetic Intensity (FLOPs/Byte)
- **Data Term line** (BlueLine): $D_{\text{vol}}/BW$ -- increases slowly (weights loaded once, activations grow with batch)
- **Compute Term line** (OrangeLine): $O/(R_{\text{peak}} \cdot \eta)$ -- increases linearly with batch size
- **Total Latency line** (thick, black): $\max(\text{data}, \text{compute}) + L_{\text{lat}}$
- **Ridge Point annotation** (vertical GreenLine dashed): Where compute term crosses data term
- **SLA line** (RedLine dashed): 100 ms horizontal line

Controls:
- **Hardware selector** (toggle): H100 (Cloud) / ESP32-S3 (TinyML)
- **Precision** (radio): FP32 / FP16 / INT8
- **Efficiency** ($\eta$): slider, 0.1--0.8, default 0.5, step 0.05
- **SLA budget** (slider): 10--500 ms, default 100 ms, step 10 ms

**Secondary chart:** A **memory footprint stacked bar** showing:
- Weights (fixed per precision)
- Activations (grows with batch)
- Total vs. device RAM capacity (horizontal line)

When ESP32-S3 is selected: ResNet-50 at FP32 = 98 MB, ESP32 RAM = 512 KB. The memory bar immediately turns red. Switching to INT8 reduces to ~25 MB -- still far exceeds 512 KB. The model is infeasible on TinyML at any precision.

### The Scaling Challenge

**"Configure the minimum batch size that makes ResNet-50 compute-bound on the H100, while keeping total latency under 100 ms."**

Students must find the sweet spot: batch=32--64 crosses the Ridge Point, but at batch=64 total latency approaches the SLA. Reducing precision to FP16 halves $D_{\text{vol}}$, shifting the Ridge Point to a lower batch size and giving more headroom under the SLA.

On ESP32-S3, the challenge changes to: "Which of the five Lighthouse Models fits on the ESP32?" Answer: only the Keyword Spotter (KWS at ~80 KB INT8). ResNet-50, GPT-2, DLRM, and MobileNet all exceed the 512 KB memory capacity.

### The Failure State

**Trigger condition:** `memory_footprint > device_ram` (OOM) OR `total_latency > sla_budget` (SLA violation)

**Visual change for OOM:** Memory bar chart turns RedLine. Banner:
> "**OOM -- Model does not fit.** ResNet-50 at [precision] requires [X] MB. Device RAM: [Y]. Reduce precision or switch to a smaller model."

**Visual change for SLA violation:** Total latency line crosses the SLA line. The SLA region above the line turns red. Banner:
> "**SLA VIOLATED -- Latency exceeds [X] ms budget.** At batch=[B] on [device]: total latency = [T] ms. Reduce batch size or switch to a faster device."

Both failure states are reversible by adjusting sliders.

### Structured Reflection

Four-option multiple choice:

> "You found the Ridge Point at batch=64 on the H100. A colleague suggests 'just double the batch to 128 for more throughput.' What happens to the per-request latency?"

- A) Latency halves -- more parallelism means faster responses
- B) Latency stays the same -- the GPU handles both batches identically
- **C) Latency approximately doubles -- each request now waits for 128 images to accumulate and process, and the compute term (now dominant) scales linearly with batch size** <-- correct
- D) Latency increases by ~50% -- the overhead term absorbs some of the increase

**Math Peek:**
$$T_{\text{bottleneck}} = \max\left(\frac{D_{\text{vol}}}{BW},\; \frac{O}{R_{\text{peak}} \cdot \eta}\right) + L_{\text{lat}}$$
$$\text{Ridge Point: } \frac{O}{R_{\text{peak}} \cdot \eta} = \frac{D_{\text{vol}}}{BW} \implies AI_{\text{ridge}} = \frac{R_{\text{peak}}}{BW}$$
$$\text{H100: } AI_{\text{ridge}} = \frac{989 \text{ TFLOPS}}{3.35 \text{ TB/s}} \approx 295 \text{ FLOPs/Byte}$$

---

## 5. Visual Layout Specification

### Act 1: Latency Waterfall
- **Primary:** Stacked horizontal bar chart (`LatencyWaterfall`)
  - X-axis: Latency (ms), range 0--50 ms
  - Segments: Data Term (BlueLine), Compute Term (OrangeLine), Overhead (gray)
  - Bottleneck label on the longest segment
- **Annotation:** "MEMORY-BOUND" or "COMPUTE-BOUND" badge

### Act 2: Iron Law Decomposition
- **Primary:** Dual-axis line chart
  - X-axis: Batch size (1--512, log scale)
  - Y-axis (left): Latency (ms), range 0--500 ms
  - Y-axis (right): Arithmetic Intensity (FLOPs/Byte)
  - Series: Data Term (BlueLine), Compute Term (OrangeLine), Total (black), SLA (RedLine dashed)
  - Ridge Point annotation (GreenLine vertical dashed)
  - Failure state: SLA region turns red when total latency exceeds budget
- **Secondary:** Memory footprint stacked bar
  - X-axis: Component (Weights, Activations, Total)
  - Y-axis: Memory (MB or KB depending on context)
  - Device capacity line (horizontal)
  - Failure state: bar turns RedLine when total > capacity

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---|---|---|---|---|
| **Cloud (H100)** | NVIDIA H100 | 80 GB HBM3 | 700 W | Memory bandwidth at batch=1; compute at high batch; SLA is the binding constraint |
| **TinyML (ESP32-S3)** | ESP32-S3 | 512 KB SRAM | 1.2 W | Memory capacity; most models do not fit at all; only KWS-class models are feasible |

The two contexts demonstrate the extreme ends of the deployment spectrum. On H100, the Iron Law analysis is a throughput-latency trade-off (batch size vs. SLA). On ESP32-S3, the Iron Law analysis is a feasibility question (does the model fit at all?). The same equation governs both, but different terms dominate.

---

## 7. Design Ledger Output

```json
{
  "chapter": 2,
  "context": "cloud | tinyml",
  "ridge_point_batch_size": 64,
  "bottleneck_at_batch_1": "memory",
  "precision_chosen": "fp16",
  "sla_budget_ms": 100,
  "sla_met": true,
  "max_feasible_batch": 64
}
```

The `ridge_point_batch_size` and `bottleneck_at_batch_1` fields feed forward to:
- **Lab 05 (NN Compute):** The batch size and precision choice initialize the activation memory visualization baseline
- **Lab 11 (HW Acceleration):** The Ridge Point concept connects to the Roofline Model instrument

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| Iron Law: $T = D_{\text{vol}}/BW + O/(R_{\text{peak}} \cdot \eta) + L_{\text{lat}}$ | `@sec-ml-systems-analyzing-workloads-cbb8`, @eq-iron-law | "This equation decomposes total latency into three terms: data movement, compute, and fixed overhead" |
| Bottleneck Principle: $T = \max(\text{memory}, \text{compute}) + L_{\text{lat}}$ | `@sec-ml-systems-bottleneck-principle-3514`, @eq-bottleneck | "whichever operation is slower determines the system's throughput" |
| Memory-bound at batch=1: 0% speedup from faster FLOPS | `@sec-ml-systems-bottleneck-principle-3514` | "buying faster processors ($R_{\text{peak}}$) yields exactly 0% speedup" |
| ResNet-50 becomes compute-bound at batch=64 on A100 | `@sec-ml-systems-system-balance-hardware-96ab`, ResNet-50 callout | "On A100, this occurs around batch=64" |
| A100: 312 TFLOPS FP16, 2,039 GB/s BW | Hardware Registry `A100_FLOPS_FP16_TENSOR`, `A100_MEM_BW` | From `constants.py`: A100 specs |
| H100: 989 TFLOPS FP16, 3,350 GB/s BW | Hardware Registry `H100_FLOPS_FP16_TENSOR`, `H100_MEM_BW` | From `constants.py`: H100 specs |
| ESP32-S3: 0.0005 TFLOPS, 512 KB SRAM | Hardware Registry `ESP32_S3` | From registry: ESP32-S3 specs |
| ResNet-50: 4.1 GFLOPs, 25.6M params, 98 MB FP32 | `@sec-ml-systems-workload-archetypes-fd10`; `ResnetSetup` LEGO cell | "approximately 4.1 billion floating-point operations using 25.6 million parameters (98 MB at FP32)" |
| Energy of transmission: 1000x more expensive than compute | `@sec-ml-systems-bottleneck-principle-3514`, Energy of Transmission callout | "Transmitting raw data is 1,000x more expensive than processing it locally" |
| Same model: compute-bound in training, memory-bound at inference | `@sec-ml-systems-system-balance-hardware-96ab`, System Balance callout | "The same ResNet-50 model is compute-bound during cloud training...but memory-bound during single-image inference" |
| Mobile thermal throttling: 60 FPS to 15 FPS in 1 minute | `@sec-ml-systems-deployment-spectrum-71be`, Power Wall section | "a mobile model that runs at 60 FPS for 1 minute may throttle to 15 FPS" |

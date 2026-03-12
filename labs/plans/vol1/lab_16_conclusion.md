# Mission Plan: lab_16_conclusion

## 1. Chapter Alignment

- **Chapter:** Conclusion (`@sec-conclusion`)
- **Core Invariant:** The **Conservation of Complexity** -- complexity in an ML system cannot be destroyed, only moved between Data, Algorithm, and Machine. A single engineering decision (e.g., quantization from FP16 to INT8) ripples through the Pareto Frontier, Silicon Contract, Arithmetic Intensity Law, Energy-Movement Invariant, Training-Serving Skew Law, and Latency Budget simultaneously. The chapter formalizes twelve quantitative invariants unified by this meta-principle and demonstrates constraint propagation through the MobileNetV2 Lighthouse Journey.
- **Central Tension:** Students believe that mastering individual components (data pipelines, compression, serving, operations) equals mastering the system. The chapter's opening scenario demolishes this: a team where every sub-team succeeded by its own metric (architecture chose efficient convolutions, compression achieved 4x memory reduction, serving hit P99 < 50 ms) still ships a system that loses 4 percentage points of accuracy within weeks, because no one traced how quantization interacted with a firmware-specific preprocessing path. The Conservation of Complexity guarantees that reducing complexity in one place inflates it in another; the chapter states "Integration complexity exceeds the sum of component complexities because interfaces multiply failure modes." Only a systems-level perspective can diagnose where the compensating cost landed.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students confront the chapter's central quantitative example: serving a single token from Llama-2-70B on an H100. The chapter's Iron Law decomposition computes that $T_{mem}$ is approximately 40x larger than $T_{comp}$, making the system overwhelmingly memory-bound. Students predict the memory-to-compute ratio before seeing the calculation, and most will dramatically underestimate it because they associate "H100" with raw TFLOPS, not bandwidth limits. The reveal forces them to internalize that the Iron Law's Data Movement term, not the Compute term, dominates LLM inference -- and that an engineer who optimizes compute kernels without addressing the memory wall wastes 100% of their effort. This single-invariant diagnosis sets up Act 2, where multiple invariants collide.

**Act 2 (Design Challenge, 23 min):** Students take on the chapter's capstone synthesis problem: configuring an end-to-end MobileNetV2 deployment that must simultaneously satisfy accuracy, latency, memory, power, and drift-resilience constraints across two deployment contexts. The Conservation of Complexity guarantees that every slider improving one metric degrades another. Students discover that the system fails silently (accuracy drops 4 percentage points) when they optimize individual metrics without tracing cross-invariant interactions. A radar chart makes the multi-dimensional trade-off space visible, and the failure state (silent accuracy degradation despite all-green infrastructure metrics) recreates the chapter's opening scenario and central lesson.

---

## 3. Act 1: The Cost of a Token (Calibration -- 12 minutes)

### Pedagogical Goal

Students overestimate the role of compute and underestimate the role of memory bandwidth in LLM inference. The chapter's "Cost of a Token" calculation for Llama-2-70B on H100 shows that $T_{mem} \approx 42$ ms while $T_{comp} \approx 0.14$ ms -- a ratio of approximately 40x. Most students, primed by marketing materials emphasizing TFLOPS, will predict a ratio far below the actual value. This act calibrates their intuition about where inference time actually goes, establishing the diagnostic power of the Arithmetic Intensity Law before Act 2 requires applying multiple invariants simultaneously.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "You are serving Llama-2-70B (70 billion parameters, FP16) on an NVIDIA H100 (3.35 TB/s memory bandwidth, ~990 TFLOPS FP16). For a single token of autoregressive decoding, how much longer does it take to *load the weights from memory* than to *compute on them*?"

Options:
- A) About the same (1--2x) -- the H100 is designed to be balanced for LLM workloads
- B) Memory is ~5x slower -- a moderate imbalance, fixable with kernel optimization
- **C) Memory is ~40x slower -- the system is overwhelmingly memory-bound** <-- correct
- D) Compute is slower -- the 70B parameter count saturates even H100 compute

**Common wrong answer:** B (~5x). Students acknowledge memory matters but vastly underestimate the ratio because they anchor on the H100's headline TFLOPS and assume NVIDIA designed a balanced system for this workload.

**Why wrong:** Autoregressive decoding has arithmetic intensity of approximately 1 FLOP/byte (2 FLOPs per parameter divided by 2 bytes per parameter in FP16). The H100's ridge point requires approximately 295 FLOP/byte to balance compute and memory. The 295:1 gap between required and actual arithmetic intensity means memory dominates by roughly 40x.

### The Instrument: Iron Law Decomposition Bar Chart

A **stacked horizontal bar chart** showing the time breakdown for one token of inference:

- **Y-axis:** Three Iron Law components: Data Movement ($T_{mem}$), Compute ($T_{comp}$), Overhead ($L_{lat}$)
- **X-axis:** Time (ms), range 0--100 ms
- **Bars:** BlueLine for $T_{mem}$, GreenLine for $T_{comp}$, OrangeLine for $L_{lat}$
- **Annotation:** A "40x" ratio label between the $T_{mem}$ and $T_{comp}$ bars, updating live

Controls:
- **Model size selector** (7B / 13B / 70B parameters, default: 70B): Changes $D_{vol}$ and $O$ proportionally. At 7B, the ratio drops to ~40x (still memory-bound). The absolute times shrink but the ratio persists.
- **Precision toggle** (FP32 / FP16 / INT8 / INT4, default: FP16): Each halving of precision halves $D_{vol}$, halving $T_{mem}$. At INT4, Llama-2-70B goes from 42 ms to ~10 ms for $T_{mem}$ while $T_{comp}$ stays at ~0.14 ms -- still memory-bound but the gap narrows from 40x to ~10x.
- **Batch size slider** (1 / 4 / 16 / 64 / 128, default: 1): Increasing batch size increases $T_{comp}$ linearly (more tokens computed per weight load) while $T_{mem}$ stays constant (weights loaded once per batch). At batch=64, the bars equalize. Students see the crossover from memory-bound to compute-bound.

**Secondary:** A **Roofline dot** showing where the current configuration sits relative to the H100's ridge point (~295 FLOP/byte). Memory-bound configurations cluster far left; batch size moves the dot rightward.

### The Reveal

After interaction:
> "You predicted the memory-to-compute ratio was [X]. The actual ratio for Llama-2-70B (FP16, batch=1) on H100 is **approximately 40x**. The system spends ~42 ms loading weights and ~0.14 ms computing on them. The chapter states: 'A systems engineer who optimizes compute kernels ($T_{comp}$) without addressing memory ($T_{mem}$) wastes 100% of their effort.'"
>
> "Notice what *does* help: increasing batch size to 64 amortizes $D_{vol}$ across tokens, raising arithmetic intensity from ~1 to ~64 FLOP/byte and shifting the bottleneck toward compute. Quantization to INT4 halves $D_{vol}$ twice, cutting $T_{mem}$ by 4x. Both strategies attack the dominant Iron Law term."

### Reflection (Structured)

Four-option multiple choice:

> "An engineer proposes optimizing CUDA kernels to achieve 95% compute utilization (up from 70%) for Llama-2-70B single-token inference on H100. What is the maximum end-to-end speedup?"

- A) 1.36x -- the 70%-to-95% utilization improvement maps directly to throughput
- **B) Less than 1.01x -- compute is not the bottleneck; improving it yields negligible end-to-end gain** <-- correct
- C) 1.10x -- some improvement because compute still contributes to total time
- D) 0.95x -- higher utilization increases thermal throttling, slowing the system

**Math Peek (collapsible):**

$$T_{mem} = \frac{D_{vol}}{BW} = \frac{70 \times 10^9 \times 2\;\text{bytes}}{3.35 \times 10^{12}\;\text{B/s}} \approx 41.8\;\text{ms}$$

$$T_{comp} = \frac{O}{R_{peak}} = \frac{70 \times 10^9 \times 2\;\text{FLOPs}}{990 \times 10^{12}\;\text{FLOP/s}} \approx 0.14\;\text{ms}$$

$$\text{Ratio} = \frac{T_{mem}}{T_{comp}} \approx 40\times \qquad \text{(heavily memory-bound)}$$

> **Source:** `ConclusionRoofline` LEGO cell in `conclusion.qmd`. The chapter states: "The memory time $T_{mem}$ is [ratio]$\times$ larger than compute time $T_{comp}$. The system is heavily memory-bound (arithmetic intensity $\approx$ 1). To honor the Silicon Contract, we must either increase Arithmetic Intensity (via batching users to reuse $D_{vol}$) or reduce Data Volume (via quantization to INT4)."

---

## 4. Act 2: The Conservation of Complexity (Design Challenge -- 23 minutes)

### Pedagogical Goal

Students believe they can optimize each component independently and achieve a system that satisfies all constraints simultaneously. The chapter's opening scenario proves otherwise: every sub-team succeeded by its own metric yet the integrated system lost 4 percentage points of accuracy within weeks. The Conservation of Complexity guarantees that reducing complexity in one dimension inflates it in another. Students must navigate a multi-dimensional trade-off space and discover that the only path to a viable system requires tracing how each decision propagates through multiple invariants. The failure state -- silent accuracy degradation with all infrastructure metrics green -- recreates the chapter's central lesson about systems thinking.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "You deploy MobileNetV2 (INT8 quantized) to a fleet of mobile devices. Infrastructure monitoring shows: P99 latency = 45 ms, uptime = 99.9%, error rate = 0.01%. After 3 months, how many percentage points of accuracy do you expect the model to have lost?"

Students enter a number between 0 and 20. Expected wrong answers: 0--1 (students trust green infrastructure dashboards). **Actual: approximately 4 percentage points**, as stated in the chapter's opening scenario: "accuracy has dropped by four percentage points on certain device populations." The gap between prediction and reality is the learning moment: green dashboards do not mean accurate predictions.

### The Instrument: Synthesis Radar with Constraint Satisfaction

**Primary -- Constraint Satisfaction Radar Chart:**

A radar chart with 5 axes representing the multi-dimensional system health:

| Axis | Range | Unit | Governing Invariant |
|:-----|:------|:-----|:--------------------|
| Accuracy | 80--99% | % | Pareto Frontier (Principle 5) |
| P99 Latency | 10--200 ms | ms (inverted: outward = faster) | Latency Budget (Principle 12) |
| Memory Footprint | 1--500 MB | MB (inverted: outward = smaller) | Silicon Contract (Principle 4) |
| Power Draw | 0.5--10 W | W (inverted: outward = less power) | Energy-Movement (Principle 7) |
| Drift Resilience | 0--100% | % (outward = more resilient) | Statistical Drift (Principle 10) |

A dashed pentagon shows the minimum acceptable SLA envelope (accuracy >= 93%, P99 < 50 ms, memory <= device RAM, power < device TDP, drift resilience > 50%). The student's configuration fills the radar polygon. Green when all constraints satisfied; red edges on any violated axis.

Controls:
- **Architecture selector** (ResNet-50 / MobileNetV2 / EfficientNet-B0, default: MobileNetV2)
  - ResNet-50: 96% accuracy, ~100 MB FP16, high latency, high power
  - MobileNetV2: 93% accuracy, ~14 MB FP16, low latency (8--9x fewer FLOPs), low power
  - EfficientNet-B0: 94% accuracy, ~21 MB FP16, moderate latency, moderate power

- **Quantization** (FP32 / FP16 / INT8 / INT4, default: FP16)
  - Each step halves memory. INT8: MobileNetV2 loses <1% accuracy; ResNet-50 loses ~1.5%. INT4: additional 2--5% loss.

- **Monitoring investment** (None / Basic / Full, default: None)
  - None: no drift detection. Drift resilience = 0%. Zero overhead.
  - Basic: PSI-based monitoring. Drift resilience = 60%. Adds ~3 ms latency overhead.
  - Full: PSI + feature store + automated retraining. Drift resilience = 95%. Adds ~8 ms overhead + 5% compute budget.
  - The Conservation of Complexity in action: monitoring investment trades latency and power for drift resilience.

- **Deployment context toggle** (H100 Cloud / Mobile GPU 2 GB)
  - H100: 80 GB RAM, 700 W TDP, 3.35 TB/s bandwidth. Generous on all axes except bandwidth (memory-bound for LLMs).
  - Mobile GPU: 2 GB RAM, 3 W TDP. Tight on every axis.

- **Months deployed slider** (0--12, default: 0)
  - Accuracy decays as: $\text{Acc}(t) \approx \text{Acc}_0 - \lambda \cdot t$
  - At monitoring = None: $\lambda \approx 1.3\%$/month (4 pp loss in 3 months, matching the chapter)
  - At monitoring = Full: $\lambda \approx 0.1\%$/month (retraining catches drift)

**Secondary -- Invariant Activation Panel:**
- A table with 12 rows (one per invariant from `@tbl-twelve-principles`).
- Each row shows: invariant name, status (green check = satisfied, red X = violated, gray = not relevant to current config).
- When a slider changes, newly activated invariants flash briefly.
- At INT8 + monitoring=None + month=3: invariants 5 (Pareto), 4 (Silicon Contract), 6 (Arithmetic Intensity), 7 (Energy-Movement), 10 (Statistical Drift), 11 (Training-Serving Skew), and 12 (Latency Budget) are all active -- at least 7 invariants from one decision chain.

### The Scaling Challenge

**"Configure a deployment that satisfies ALL five constraints simultaneously after 6 months of operation."**

Students discover:
1. MobileNetV2 + INT8 + Mobile GPU hits latency and memory targets easily at month 0.
2. Without monitoring (None), accuracy drops below the 93% threshold by approximately month 4 ($93\% - 1.3\% \times 4 \approx 87.8\%$).
3. Adding Basic monitoring preserves accuracy longer but adds 3 ms latency overhead, eating into the 50 ms budget.
4. Adding Full monitoring adds 8 ms overhead. If MobileNetV2 INT8 latency was 35 ms, it becomes 43 ms -- still within SLA, but with narrow margin.
5. ResNet-50 at FP32 blows the memory and power budgets on Mobile GPU. On H100 it fits, but requires Full monitoring at scale for fairness (aggregate metrics conceal 40x error rate disparities).
6. The sweet spot for Mobile GPU: MobileNetV2 + INT8 + Basic or Full monitoring. For H100: MobileNetV2 or EfficientNet-B0 + INT8 + Full monitoring + batch=4+.

The Conservation of Complexity: quantization reduced model complexity (Algorithm) but increased the need for monitoring complexity (Machine/Data). Students who skip monitoring save latency overhead but lose accuracy. Students who add full monitoring preserve accuracy but tighten the latency budget. No free lunch.

### The Failure State

**Trigger condition:** `months_deployed >= 3 AND monitoring_investment == "None" AND all_infra_axes_green`

**Visual change:** The accuracy axis on the radar chart silently shrinks inward. The other four axes remain green. No alarm fires. The invariant panel shows Statistical Drift (row 10) turning amber, but the overall status bar stays green because infrastructure metrics are all passing. This mimics silent degradation in production.

**Banner text (appears only after student clicks "Diagnose System"):**
> "SILENT DEGRADATION -- Accuracy has dropped to [X]% over [N] months. All infrastructure metrics remain green: P99 = 45 ms, uptime = 99.9%, error rate = 0.01%. The cause: interaction between INT8 quantization rounding and firmware-specific image preprocessing that shifted the effective input distribution. The Statistical Drift Invariant (Principle 10) and Training-Serving Skew Law (Principle 11) predicted this failure. No single team's metrics caught it."

**Reversibility:** Students increase monitoring_investment to Basic or Full and observe the accuracy axis recover as automated retraining catches the drift. The radar polygon fills back out. The invariant panel row 10 returns to green.

### Structured Reflection

Four-option multiple choice:

> "Every sub-team hit its target: architecture efficiency (8--9x fewer FLOPs), quantization (4x memory reduction, <1% accuracy loss), serving (P99 < 50 ms). Yet accuracy dropped 4 percentage points in 3 months. Which meta-principle explains why?"

- A) Amdahl's Law -- the non-parallelizable monitoring fraction limited overall performance
- B) The Iron Law -- data movement dominated but no team measured it end-to-end
- **C) Conservation of Complexity -- quantization reduced model complexity but shifted complexity to the monitoring and skew-detection pipeline; without investing in that pipeline, the shifted complexity manifested as silent accuracy degradation** <-- correct
- D) The Pareto Frontier -- the team picked a suboptimal point on the accuracy-latency curve

**Math Peek (collapsible):**

$$C_{total} = C_{data} + C_{algorithm} + C_{machine} \approx \text{constant}$$

$$\text{Quantization: } C_{algorithm} \downarrow \;\implies\; C_{data}(\text{retraining frequency}) \uparrow + C_{machine}(\text{monitoring}) \uparrow$$

$$\text{Acc}(t) \approx \text{Acc}_0 - \lambda \cdot D(P_t \| P_0) \qquad \text{(Principle 10: Statistical Drift Invariant)}$$

> **Source:** The chapter states: "Quantization may have shifted load to the accuracy monitoring pipeline. Engineers who celebrate gains in one metric without tracing the compensating costs elsewhere build systems that fail in unexpected ways. Every invariant connects to others; optimizing one in isolation creates technical debt that compounds over time."

---

## 5. Visual Layout Specification

### Act 1: Iron Law Decomposition

1. **Primary: Stacked horizontal bar chart**
   - Chart type: Stacked horizontal bar
   - X-axis: Time (ms), range 0--100 ms
   - Y-axis: Iron Law components: Data Movement ($T_{mem}$), Compute ($T_{comp}$), Overhead ($L_{lat}$)
   - Data series: BlueLine for $T_{mem}$, GreenLine for $T_{comp}$, OrangeLine for $L_{lat}$
   - Annotation: "40x" ratio label between $T_{mem}$ and $T_{comp}$ bars, live-updating
   - Student's prediction overlaid as a vertical marker on the $T_{mem}$ bar
   - Failure state: N/A (Act 1 is calibration only)

2. **Secondary: Roofline operating point**
   - Chart type: Log-log scatter on Roofline axes
   - X-axis: Arithmetic Intensity (FLOP/byte), range 0.1--1000
   - Y-axis: Attainable Performance (TFLOP/s), range 0.1--1000
   - Data series: H100 roofline ceiling lines (memory bandwidth slope, compute ceiling); current config as a dot (RedLine if memory-bound, GreenLine if compute-bound)
   - Ridge point annotation at ~295 FLOP/byte
   - Batch size moves the dot rightward along the x-axis

### Act 2: Conservation of Complexity Radar

1. **Primary: Radar chart (5-axis constraint satisfaction)**
   - Chart type: Radar / spider chart with 5 axes
   - Axes: Accuracy (80--99%), P99 Latency (10--200 ms, inverted), Memory (1--500 MB, inverted), Power (0.5--10 W, inverted), Drift Resilience (0--100%)
   - Data series: Current config polygon (BlueLine fill), SLA minimum envelope (dashed RedLine pentagon)
   - Failure state: Accuracy axis silently contracts when drift_resilience < 50% and months > 3. No alarm until student clicks "Diagnose System."

2. **Secondary: Invariant activation table**
   - Chart type: 12-row status table
   - Columns: Invariant # and Name, Status (green check / amber / red X / gray), Current Value, Threshold
   - Newly activated rows flash when a slider changes

3. **Tertiary: Accuracy decay timeline**
   - Chart type: Line chart
   - X-axis: Months deployed (0--12)
   - Y-axis: Accuracy (80--99%)
   - Data series: Projected accuracy under current monitoring level (BlueLine), 93% threshold (dashed RedLine)
   - Shows how monitoring investment flattens the decay curve ($\lambda$ decreases from 1.3%/month to 0.1%/month)

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|:--------|:-------|:----|:-------------|:---------------|
| **Training Node (H100)** | H100 (80 GB HBM3) | 80 GB | 700 W TDP | Memory bandwidth is the wall for LLM inference; ~40x memory-bound at batch=1; generous on every other axis |
| **Edge Inference (Mobile GPU)** | Mobile GPU (2 GB) | 2 GB | 3 W thermal envelope | Multi-constraint squeeze: accuracy, latency, power, and drift resilience must all be satisfied simultaneously within a tight envelope where INT8 quantization is necessary and monitoring overhead competes with the latency SLA |

The two contexts demonstrate the Conservation of Complexity at different scales. On the Training Node, complexity concentrates in the memory-bandwidth bottleneck (Act 1: a single Iron Law term dominates by 40x). On the Edge, complexity is distributed across all five constraint axes (Act 2: no single term dominates, and optimizing any one shifts cost to others). The transition from Act 1 to Act 2 mirrors the chapter's arc from single-invariant diagnosis to multi-invariant systems thinking.

---

## 7. Design Ledger Output

```json
{
  "chapter": 16,
  "dominant_bottleneck": "memory_bandwidth",
  "memory_compute_ratio_prediction_error": 5.0,
  "system_config": {
    "architecture": "MobileNetV2",
    "quantization": "INT8",
    "monitoring_level": "basic",
    "deployment_context": "mobile_gpu"
  },
  "invariants_activated_count": 7,
  "all_constraints_satisfied_at_month_6": true,
  "conservation_reflection_answer": "C"
}
```

This is the final lab in Volume 1. No subsequent Vol 1 lab reads these fields. The `dominant_bottleneck` and `system_config` fields serve as the student's "graduation profile" for the Volume 1 to Volume 2 transition. The Volume 2 opening lab (`v2_lab_01`) reads `dominant_bottleneck` to initialize the fleet-scale context: a student who identified `memory_bandwidth` as dominant sees how that same bottleneck manifests at rack-scale when the system bus becomes the network fabric.

The capstone lab also reads from all prior labs:
- **Lab 11** `bottleneck_type` and `roofline_position` feed the Act 1 Roofline dot
- **Lab 13** `sla_type` and `p99_latency_ms` feed the Latency Budget axis threshold
- **Lab 14** `drift_rate_lambda` and `psi_threshold` feed the drift resilience axis and the accuracy decay formula
- **Lab 15** `fairness_disparity_ratio` feeds the sub-group accuracy check within the radar

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|:---|:---|:---|
| Llama-2-70B: 70B params, FP16, $D_{vol}$ = 140 GB | `ConclusionRoofline` LEGO cell in conclusion.qmd | `d_vol = model.parameters * precision_bytes` where `precision_bytes = 2.0` (FP16) |
| H100: 3.35 TB/s bandwidth, ~990 TFLOPS FP16 | `ConclusionRoofline` LEGO cell | `gpu = Hardware.H100`; exports `h100_bw_tb_str` and `h100_peak_tflops_str` |
| $T_{mem} / T_{comp} \approx 40\times$ ratio (memory-bound) | "Cost of a Token" callout in `@sec--synthesizing-ml-systems-ee10` | "The memory time $T_{mem}$ is [ratio]$\times$ larger than compute time $T_{comp}$. The system is heavily memory-bound (arithmetic intensity $\approx$ 1)." |
| "Wastes 100% of their effort" optimizing compute when memory-bound | "Cost of a Token" callout | "A systems engineer who optimizes compute kernels ($T_{comp}$) without addressing memory ($T_{mem}$) wastes 100% of their effort." |
| Accuracy dropped 4 percentage points on certain device populations | Opening paragraph of `@sec--synthesizing-ml-systems-ee10` | "accuracy has dropped by four percentage points on certain device populations" |
| MobileNetV2: depthwise separable, 8--9x FLOPs reduction | `@tbl-lighthouse-journey-mobilenet` | "Depthwise Separable Convolutions: 8--9$\times$ reduction in FLOPs vs ResNet-50" |
| INT8 quantization: 4x memory reduction, <1% accuracy loss | `@tbl-lighthouse-journey-mobilenet` | "INT8 Quantization: 4$\times$ memory reduction with minimal accuracy loss (<1%)" |
| P99 < 50 ms latency constraint for mobile serving | `@tbl-lighthouse-journey-mobilenet` | "P99 < 50 ms constraint; optimizing preprocessing (resize/normalize) to avoid CPU bottlenecks" |
| Conservation of Complexity: complexity moved, not destroyed | `@sec-twelve-quantitative-invariants-0dd2` | "Complexity in an ML system cannot be destroyed; it can only be moved between Data, Algorithm, and Machine." |
| Quantization ripples through Pareto, Silicon Contract, AI Law, Skew, Latency Budget | Integrated Framework discussion (line ~240) | "A single quantization decision ripples through the Pareto Frontier, Silicon Contract, and Latency Budget simultaneously, where a win in one (bandwidth) must be validated against a risk in another (numerical skew)." |
| "Integration complexity exceeds the sum of component complexities" | Fallacies section, `@sec--fallacies-pitfalls-12ef` | "Integration complexity exceeds the sum of component complexities because interfaces multiply failure modes." |
| Aggregate metrics conceal 40x error rate disparities | Fallacies section | "aggregate metrics conceal 40$\times$ error rate disparities across demographic groups" |
| Amdahl's Law: 10x inference optimization yields 1.1x system speedup when data loading is 90% of time | Fallacies section | "Optimizing inference latency by 10$\times$ yields only 1.1$\times$ system speedup if data loading accounts for 90% of end-to-end latency." |
| P99 tail latency 40x higher than mean (50 ms mean, 2000 ms P99) | `TailLatencyRatio` LEGO cell in conclusion.qmd | `mean_latency_ms = 50.0; p99_latency_ms = 2000.0; ratio = 40` |
| Statistical Drift: $\text{Acc}(t) \approx \text{Acc}_0 - \lambda \cdot D(P_t \| P_0)$ | `@tbl-twelve-principles`, Principle 10 | "Models decay without code changes; the world drifts away from training data" |
| Energy-Movement: data movement costs 100--1000x more than compute | `@tbl-twelve-principles`, Principle 7 | "$E_{move} \gg E_{compute}$ (100--1,000$\times$)" |

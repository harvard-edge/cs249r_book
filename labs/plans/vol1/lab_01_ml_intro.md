# Mission Plan: lab_01_ml_intro

## 1. Chapter Alignment

- **Chapter:** Introduction (`@sec-introduction`)
- **Core Invariant:** ML hardware spans nine orders of magnitude in power (MW to mW) and memory (TB to KB); the same model cannot run everywhere because physics partitions the deployment landscape into fundamentally distinct operating regimes.
- **Central Tension:** Students believe ML is primarily an algorithm problem and that a single trained model can be deployed anywhere. The chapter's data reveals that the D-A-M Triad (Data, Algorithm, Machine) is inseparable: compressing a model to fit on a mobile device changes its accuracy, doubling the training data demands more compute, and the Verification Gap makes exhaustive testing impossible ($256^{150{,}528}$ possible inputs vs. 50,000 test images). The landscape is governed by physics, not software preferences.
- **Target Duration:** 35--40 minutes (2 acts)

---

## 2. The Two-Act Structure Overview

**Act 1 (Calibration, 12 min):** Students believe that faster hardware simply means "more of the same" -- that a cloud GPU is just a faster version of a microcontroller. The chapter's Hardware Twins (H100 at 989 TFLOPS FP16 vs. ESP32-S3 at 0.0005 TFLOPS) reveal a 2-million-to-one compute gap and a 160,000-to-one memory gap (80 GB vs. 512 KB). This act forces students to predict the magnitude of the gap before seeing it, calibrating their intuition for the nine-order-of-magnitude span that governs every subsequent chapter.

**Act 2 (Design Challenge, 22 min):** Students apply the Degradation Equation ($\text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda \cdot D(P_t \| P_0)$) to discover that a deployed model silently loses accuracy while all infrastructure metrics remain green. They must find the retraining trigger threshold ($\tau$) that keeps accuracy above a specified floor, discovering that traditional software monitoring is blind to ML-specific failures.

---

## 3. Act 1: The Magnitude Gap (Calibration -- 12 minutes)

### Pedagogical Goal

Students dramatically underestimate the hardware diversity of the ML landscape. They assume the difference between a cloud GPU and a microcontroller is "maybe 10x or 100x." The chapter's five Lighthouse Models (ResNet-50, GPT-2/Llama, DLRM, MobileNet, KWS) each stress different extremes of the Iron Law. This act uses two hardware endpoints -- H100 (Cloud) and ESP32-S3 (TinyML) -- to establish the scale of the gap before any optimization discussion begins.

### The Lock (Structured Prediction)

Present a **multiple-choice prediction** before any instruments unlock:

> "An NVIDIA H100 datacenter GPU costs ~$30,000 and consumes 700W. An ESP32-S3 microcontroller costs ~$10 and consumes 1.2W. By what factor does the H100 exceed the ESP32 in peak compute (FLOPS)?"

Options:
- A) About 100x -- one is a GPU, the other is a microcontroller, but both do math
- B) About 10,000x -- GPUs are specialized but the gap has limits
- **C) About 2,000,000x -- the gap spans six orders of magnitude** <-- correct
- D) About 10,000,000x -- the gulf is essentially infinite

The correct answer is ~2,000,000x (989 TFLOPS / 0.0005 TFLOPS = 1,978,000x). Most students pick A or B because they underestimate how specialized GPU silicon is. The ratio is computed from the Hardware Registry: H100 FP16 Tensor = 989 TFLOPS, ESP32-S3 = 0.0005 TFLOPS.

### The Instrument: Hardware Comparison Dashboard

A **log-scale comparison bar chart** showing four hardware tiers side by side:

- **X-axis:** Hardware device (ESP32-S3, iPhone 15 Pro, Jetson Orin NX, H100)
- **Y-axis (log scale):** Metric value
- **Switchable metric:** Toggle between Compute (TFLOPS), Memory (GB), Power (W), Cost ($)

Controls:
- **Metric selector** (radio): Compute / Memory / Power / Cost
- **Highlight pair** (toggle): Select any two devices to show the ratio badge between them

When Compute is selected: ESP32-S3 = 0.0005, iPhone = 35, Jetson = 25, H100 = 989 TFLOPS. The log-scale bars show the staircase visually. A ratio badge between the selected pair displays "H100 is 1,978,000x the ESP32 in compute."

**Secondary:** A **D-A-M Triangle** (static reference) showing how each Lighthouse Model maps to the three axes, with the currently selected hardware pair highlighted to show which models are feasible on each device.

### The Reveal

After interaction:
> "You predicted [X]x. The actual compute ratio is **1,978,000x** (989 TFLOPS / 0.0005 TFLOPS). The memory ratio is **163,840x** (80 GB / 512 KB). This nine-order-of-magnitude span is why no single software stack, model architecture, or deployment strategy works across the entire ML landscape."

### Reflection (Structured)

Four-option multiple choice:

> "The H100 has ~2 million times more compute than the ESP32. Why can't we simply run a cloud-trained model on a microcontroller by 'making it smaller'?"

- A) Microcontrollers lack GPU cores, so the model would run slowly but correctly
- B) The model would need to be retrained from scratch for the microcontroller
- **C) Compression sufficient to bridge a 2,000,000x gap would destroy the model's learned representations -- the D-A-M axes are coupled** <-- correct
- D) The microcontroller's operating system cannot run Python, which is required for ML

**Math Peek (collapsible):**
$$\text{Compute Ratio} = \frac{R_{\text{peak}}^{\text{H100}}}{R_{\text{peak}}^{\text{ESP32}}} = \frac{989 \text{ TFLOPS}}{0.0005 \text{ TFLOPS}} \approx 2 \times 10^6$$
$$\text{Memory Ratio} = \frac{80 \text{ GB}}{512 \text{ KB}} \approx 1.6 \times 10^5$$

---

## 4. Act 2: Silent Degradation (Design Challenge -- 22 minutes)

### Pedagogical Goal

Students believe that once an ML model is deployed and reporting healthy infrastructure metrics (100% uptime, low latency, zero errors), it remains accurate indefinitely -- just like traditional software. The chapter's Degradation Equation ($\text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda \cdot D(P_t \| P_0)$) quantifies how accuracy erodes silently as data distributions shift, even while all infrastructure dashboards remain green. Students must find the retraining threshold that prevents unacceptable accuracy loss.

### The Lock (Numeric Prediction)

Before instruments unlock:

> "A recommendation system is deployed at 85% accuracy. Your monitoring dashboard shows 100% uptime, < 50 ms latency, and 0.01% error rate for 6 months. Estimate the model's accuracy after 6 months (enter a number between 0 and 100)."

Expected wrong answers: 80--85% (students assume metrics would catch problems, or that accuracy barely moves). Actual: with $\lambda = 0.10$ and moderate drift ($D = 0.30$ after 6 months), accuracy drops to ~82%. With faster drift ($\lambda = 0.15$, $D = 0.40$), it drops to ~79%. The key insight is that the decline is invisible to traditional monitoring.

### The Instrument: Degradation Simulator

A **dual-panel dashboard** showing:

**Left panel (Infrastructure Metrics):** Four metric cards that never change:
- Uptime: 100%
- P99 Latency: 48 ms
- Error Rate: 0.01%
- Memory Usage: 72%

**Right panel (Model Accuracy):** A time-series line chart:
- **X-axis:** Months since deployment (0 --> 24)
- **Y-axis:** Model accuracy (40% --> 100%)
- **Accuracy line** (BlueLine): Starts at Accuracy_0, curves downward per the Degradation Equation
- **Accuracy floor line** (RedLine, dashed): The minimum acceptable accuracy (configurable)
- **Retraining trigger line** (GreenLine, dashed): The threshold $\tau$ the student sets

Controls:
- **Initial accuracy** ($\text{Accuracy}_0$): slider, 80%--99%, default 85%, step 1%
- **Drift sensitivity** ($\lambda$): slider, 0.01--0.20, default 0.10, step 0.01
- **Drift rate** (controls $D(P_t \| P_0)$ accumulation): Low / Medium / High (maps to linear drift of 0.05, 0.10, 0.15 per month)
- **Retraining threshold** ($\tau$): slider, sets the accuracy floor for trigger, range 60%--95%, default 75%
- **Deployment context toggle**: H100 (Cloud) / ESP32 (TinyML) -- affects Accuracy_0 default and lambda range

When the accuracy line crosses the retraining threshold, a GreenLine annotation appears: "Retrain triggered at Month [X]." When accuracy crosses the floor without a threshold set, the RedLine failure state activates.

### The Scaling Challenge

**"Find the retraining threshold ($\tau$) that keeps the recommendation system above 75% accuracy for the entire 24-month window, using the fewest retraining cycles."**

Students must balance: setting $\tau$ too high triggers unnecessary retraining (costly); setting it too low allows accuracy to fall below the floor. The minimum viable $\tau$ depends on $\lambda$ and drift rate. With $\lambda = 0.10$ and medium drift, accuracy hits 75% at approximately month 10. Setting $\tau = 78%$ triggers retraining at month 7, resetting accuracy to $\text{Accuracy}_0$ and requiring approximately 3 retraining cycles over 24 months.

### The Failure State

**Trigger condition:** `accuracy(t) < accuracy_floor AND retraining_threshold is None`

**Visual change:** The accuracy line turns RedLine below the floor. The infrastructure metrics panel stays entirely green. A banner appears:

> "**SILENT FAILURE -- Model accuracy dropped to [X]% while all infrastructure metrics remained healthy.** Traditional monitoring detected nothing. The Degradation Equation predicted this at Month [Y]."

The failure state is reversible: setting a retraining threshold and sliding it above the current accuracy triggers a simulated retrain, resetting the accuracy curve.

### Structured Reflection

Four-option multiple choice:

> "Traditional software monitoring showed 100% uptime throughout the simulation. Why did it fail to detect the accuracy drop?"

- A) The monitoring tools were misconfigured and needed accuracy-specific alerts
- B) The accuracy drop was too small for any monitoring system to detect
- **C) Infrastructure metrics (uptime, latency, error rate) measure the Machine axis but are blind to the Data axis -- distribution drift is invisible to systems that only monitor code execution** <-- correct
- D) The model's predictions were still technically valid, just less optimal

**Math Peek:**
$$\text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda \cdot D(P_t \| P_0)$$
$$\text{Retrain when } D(P_t \| P_0) > \tau \implies \text{Accuracy}(t) > \text{Accuracy}_0 - \lambda \cdot \tau$$

---

## 5. Visual Layout Specification

### Act 1: Hardware Comparison Dashboard
- **Primary:** Log-scale bar chart
  - X-axis: Device name (4 devices)
  - Y-axis: Metric value (log scale, range depends on selected metric: Compute 0.0001--1000 TFLOPS; Memory 0.0005--100 GB; Power 0.005--1000 W; Cost 1--50000 $)
  - Data series: one bar per device, colored by deployment tier (TinyML = purple, Mobile = green, Edge = orange, Cloud = blue)
  - Ratio badge between selected pair
- **Secondary:** Static D-A-M triangle reference diagram

### Act 2: Degradation Simulator
- **Primary (Left):** Four metric cards (static green indicators) -- never enter failure state
- **Primary (Right):** Time-series line chart
  - X-axis: Months (0--24)
  - Y-axis: Accuracy (40%--100%)
  - Series: Accuracy curve (BlueLine), accuracy floor (RedLine dashed), retraining threshold (GreenLine dashed)
  - Failure state: accuracy line turns RedLine below floor; banner appears above chart
- **Annotations:** Retraining trigger markers (vertical GreenLine dashed lines at each retrain point)

---

## 6. Deployment Context Definitions

| Context | Device | RAM | Power Budget | Key Constraint |
|---|---|---|---|---|
| **Cloud (H100)** | NVIDIA H100 | 80 GB HBM3 | 700 W | Compute throughput; memory is abundant; degradation cost = wasted GPU-hours |
| **TinyML (ESP32-S3)** | ESP32-S3 | 512 KB SRAM | 1.2 W | Memory capacity; model must fit in KB; retraining requires cloud round-trip |

The two contexts demonstrate that the magnitude gap is not just about speed -- it changes what models are feasible, what monitoring is possible, and how retraining is triggered. On H100, retraining is a cluster scheduling decision. On ESP32, retraining requires uploading data to the cloud, retraining remotely, and flashing new firmware -- a fundamentally different operational model.

---

## 7. Design Ledger Output

```json
{
  "chapter": 1,
  "context": "cloud | tinyml",
  "hardware_compute_ratio_log10": 6.3,
  "hardware_memory_ratio_log10": 5.2,
  "degradation_lambda": 0.10,
  "retraining_threshold_pct": 78,
  "retraining_cycles_24mo": 3
}
```

The `context` and `degradation_lambda` fields feed forward to:
- **Lab 02 (ML Systems):** The hardware selection initializes the default deployment context for Iron Law analysis
- **Lab 14 (ML Ops):** The retraining threshold and lambda inform the monitoring configuration baseline

---

## 8. Traceability Table

| Lab Element | Chapter Section | Exact Claim Being Tested |
|---|---|---|
| H100 peak compute: 989 TFLOPS FP16 | `@sec-introduction`, line 1387; Hardware Registry `H100_FLOPS_FP16_TENSOR` | "NVIDIA H100" as Hardware Twin; 989 TFLOPS from `constants.py` |
| ESP32-S3 peak compute: 0.0005 TFLOPS | `@sec-introduction`, line 1387; Hardware Registry `ESP32_S3` | "ESP32-S3" as Hardware Twin; 0.0005 TFLOPS from registry |
| Nine orders of magnitude span | `@sec-ml-systems-deployment-paradigm-framework-0d25` | "These four paradigms span nine orders of magnitude in power consumption (megawatts to milliwatts) and memory capacity (terabytes to kilobytes)" |
| Verification Gap: $256^{150{,}528}$ possible inputs | `@sec-introduction-datacentric-paradigm-shift-4eca`, Verification Gap callout | "a number with over 361,000 digits. ImageNet's entire test set covers only 50,000 of them" |
| Degradation Equation | `@sec-introduction-ml-vs-traditional-software-e19a`, @eq-degradation | "$\text{Accuracy}(t) \approx \text{Accuracy}_0 - \lambda \cdot D(P_t \| P_0)$" |
| Recommendation system: 85% to below 40% in 6 months | `@sec-introduction-ml-vs-traditional-software-e19a`, line 1450 | "a recommendation system...might decline from 85% to below 40% accuracy over six months" |
| Silent degradation: no crash, no error log | `@sec-introduction-ml-vs-traditional-software-e19a`, line 1420 | "They can continue functioning while their performance degrades silently, without triggering conventional error detection mechanisms" |
| ML code is 5% of total codebase | `@sec-introduction-datacentric-paradigm-shift-4eca`, Hidden Technical Debt callout | "the ML Code (the model itself) is only a tiny fraction (~5%) of the total code base" |
| D-A-M Triad: inseparable axes | `@sec-introduction`, D-A-M taxonomy section | "compressing a model to fit on a mobile device changes its accuracy, doubling the training data demands more compute" |
| Retraining threshold $\tau$ | `@sec-introduction-ml-vs-traditional-software-e19a`, line 1446 | "A system that retrains when $D(P_t \| P_0) > \tau$ for some threshold $\tau$ maintains accuracy within bounds" |

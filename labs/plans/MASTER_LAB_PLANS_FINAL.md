---
title: "Master Lab Plans"
---

# MASTER LAB PLANS — FINAL VERSION

Generated: 2026-03-15
Status: All structural fixes applied, ed-tech reviewed, narrative style enforced.

Total: 32 labs (16 Vol1 + 16 Vol2) after V2-03/V2-06 merger.

# VOLUME I: Introduction to Machine Learning Systems

# Volume I Labs 01-08: Final Cleaned Plans

Generated: 2026-03-15
Status: FINAL -- ready for lab-developer handoff

These plans incorporate all fixes from the ed-tech review:
- Lab 01 Part C replaced (Silent Decay -> Triad Across Targets)
- Lab 02 Part C thermal model simplified to piecewise
- Lab 03 grounded in Engine.solve() and Engine.sweep()
- Lab 04 Part A and Part D explicitly use Engine.solve()
- All parts capped at 12 min; Synthesis sections added to Labs 01-04
- ESP32 replaces all Cortex-M7 references
- Narrative transitions between all parts

---

## Lab 01: The AI Triad

**Story arc**: A single model (ResNet-50) is deployed across three radically different hardware targets, and students discover that the *same* model fails for *different* physical reasons at each point on the deployment spectrum -- establishing the D-A-M triad as a diagnostic framework that governs the entire course.

**Time budget**: 51 min (12 + 12 + 12 + 9 + 6)

### Part A -- Three Axes, One System (~12 min)

**Concept**: The AI Triad (Data, Algorithm, Machine) are interdependent -- optimizing one axis shifts the bottleneck to another. A system performing poorly cannot be fixed by throwing resources at the wrong axis. Students must diagnose *which* axis is binding before they can improve anything. This is the foundational diagnostic skill of the entire course.

**Prediction**: "A recommendation system is showing poor accuracy. The team proposes buying 4x more GPUs. What happens to accuracy?" Options: (a) Improves proportionally (~4x better), (b) Improves modestly (~1.3x better), (c) No change -- the bottleneck is elsewhere, (d) Gets worse due to overfitting. Most students pick (a) or (b) because they assume more compute always helps. The correct answer is (c): the system has stale training data (Data axis), so more compute (Machine axis) changes nothing. Students learn that diagnosis must precede investment.

**Instrument**: Three system scenarios are presented as "patient charts" (recommendation system with stale data, vision model with insufficient FLOPs, language model that exceeds device memory). For each, students select a D-A-M axis as the bottleneck. A "prescribe fix" toggle lets them apply fixes to the wrong axis first -- observing zero improvement -- then apply the correct fix and watch the system recover. A diagnostic summary card shows which axis was binding and why.

**mlsysim grounding**: Conceptual framing exercise; no Engine.solve() calls yet (that comes in Part B). The scenarios are grounded in the textbook's three worked examples from the introduction chapter.

**Transition to B**: "You just diagnosed bottlenecks by intuition. But how do you *quantify* which axis is binding? The Iron Law decomposes system performance into three physical terms -- one for each axis. Let us see what it reveals about a real model on real hardware."

---

### Part B -- The Iron Law Surprise (~12 min)

**Concept**: The Iron Law of ML Systems (Latency = Data_volume/Bandwidth + Operations/(Peak_compute * efficiency) + Latency_overhead) decomposes performance into three physical terms. At batch=1 on an H100, ResNet-50 inference is *memory-bound*, not compute-bound -- a result that violates the naive assumption that GPUs are "compute machines." The bottleneck shifts as batch size increases, demonstrating that the binding constraint is a function of *how* you use the hardware, not just *which* hardware you have.

**Prediction**: "For ResNet-50 inference at batch=1 on an H100 GPU, which Iron Law term dominates latency?" Options: (a) Data loading (D_vol/BW), (b) Compute (O/R_peak), (c) Framework overhead, (d) They are roughly equal. Most students pick (b) because GPUs are "compute machines." The correct answer is (a): at batch=1, the memory bandwidth term dominates because the compute is too small to amortize the cost of loading 25M parameters from HBM.

**Instrument**: A batch size slider (1 to 256) drives a live Engine.solve() call. A latency waterfall bar decomposes total latency into its three Iron Law components, color-coded (blue=compute, green=memory, orange=overhead). A "bottleneck" indicator flips from "MEMORY-BOUND" to "COMPUTE-BOUND" as batch size increases past ~32. Students watch the crossover happen in real time. An MFU gauge shows utilization climbing from <10% at batch=1 to >50% at batch=64.

**mlsysim grounding**:
```python
profile = Engine.solve(Models.ResNet50, Hardware.H100, batch_size=bs, precision="fp16")
# Read: profile.bottleneck, profile.latency_compute, profile.latency_memory, profile.mfu
```
Sweep batch_size from 1 to 256 to generate the crossover curve.

**Transition to C**: "The Iron Law just told you that ResNet-50 on an H100 is memory-bound at batch=1. But what happens when you take that *same* model and deploy it on an edge device? Or a microcontroller? The physics changes dramatically -- and one of those deployments will fail entirely."

---

### Part C -- The Triad Across Targets (~12 min)

**Concept**: The same model (ResNet-50) deployed on three hardware targets (H100, Jetson Orin NX, ESP32) is diagnosed by Engine.solve() as having *different* binding constraints at each tier. On the H100, it is memory-bound (the Machine axis is fine but underutilized). On the Jetson, it fits but is severely memory-bound (bandwidth is the wall). On the ESP32, it is flatly infeasible -- 100 MB of weights vs. 512 KB of SRAM, an OOM by 200x. This demonstrates that the D-A-M triad diagnosis depends entirely on deployment context, and that "the model works" is meaningless without specifying *where*.

**Prediction**: "ResNet-50 requires ~100 MB in FP16. The ESP32-S3 has 512 KB of SRAM. What is the ratio of model size to available memory?" Options: (a) ~2x over budget (need a small trim), (b) ~10x over budget (need significant compression), (c) ~200x over budget (fundamentally infeasible), (d) It fits with INT8 quantization. Most students pick (a) or (b) -- they underestimate the gap because they have never worked with microcontroller-scale memory. The correct answer is (c): 200x, meaning no amount of compression will make ResNet-50 run on this device. You need a *different algorithm entirely*.

**Instrument**: A hardware target selector (H100 / Jetson Orin NX / ESP32) with a fixed model (ResNet-50, FP16). For each target, Engine.solve() produces a performance profile displayed as a diagnostic card: feasibility (pass/fail), bottleneck (compute/memory/OOM), latency breakdown, MFU. The ESP32 card shows a red "INFEASIBLE" banner with the memory ratio. A side-by-side comparison table shows all three targets simultaneously, with the binding D-A-M axis highlighted per target (H100: Machine underutilized, Jetson: Machine bandwidth-limited, ESP32: Machine capacity-exceeded).

**mlsysim grounding**:
```python
for hw in [Hardware.H100, Hardware.Jetson, Hardware.ESP32]:
    profile = Engine.solve(Models.ResNet50, hw, batch_size=1, precision="fp16")
    # Display: profile.feasible, profile.bottleneck, profile.memory_footprint vs hw capacity
```

**Transition to D**: "The ESP32 result is dramatic: a 200x memory gap. But the deployment spectrum is not just three points -- it spans nine orders of magnitude. Let us zoom out and see the full picture."

---

### Part D -- The Deployment Spectrum (~9 min)

**Concept**: The deployment spectrum spans 9 orders of magnitude in compute (H100: ~1,000 TFLOPS vs. ESP32: ~0.5 GFLOPS) and 5 orders of magnitude in memory (80 GB vs. 512 KB). This gap is so vast that a universal ML software stack is physically impossible -- each deployment tier requires fundamentally different choices on every D-A-M axis. The log-scale visualization is the only way to even *see* TinyML devices on the same chart as cloud accelerators.

**Prediction**: "What is the compute ratio between an H100 GPU and an ESP32 microcontroller?" Options: (a) ~100x, (b) ~10,000x, (c) ~1,000,000x, (d) ~1,000,000,000x. Most students pick (a) or (b) because they have never confronted the full hardware spectrum. The correct answer is (c): approximately one million times.

**Instrument**: A scale toggle switches a bar chart between linear view (where TinyML devices are invisible -- a flat line at zero) and log view (where the full spectrum becomes visible). Hardware entries from the mlsysim registry are plotted: H100, A100, Jetson, iPhone, ESP32, Himax. Two bars per device: compute (TFLOPS) and memory (GB). The 1,000,000x compute gap and ~160,000x memory gap are annotated. A second panel shows what this means for each D-A-M axis at each tier: Cloud uses full models with full data; Edge uses compressed models with filtered data; TinyML uses purpose-built models with preprocessed features.

**mlsysim grounding**:
```python
targets = [Hardware.H100, Hardware.A100, Hardware.Jetson, Hardware.iPhone, Hardware.ESP32, Hardware.Himax]
for hw in targets:
    # Read: hw.compute.peak_flops, hw.memory.capacity, hw.tdp
```

---

### Synthesis (~6 min)

**Prompt**: Your team has trained ResNet-50 to 96% accuracy for a medical imaging task. The customer now tells you it must run on a Jetson Orin NX at the edge (no cloud). Using the Iron Law decomposition from Part B and the cross-target diagnosis from Part C, answer: (1) Is ResNet-50 feasible on the Jetson? (2) What is the binding constraint? (3) If the customer then asks for deployment on an ESP32, what changes -- a different model, different data, or different hardware? Justify each answer with a specific number from the lab.

---
---

## Lab 02: The Physics of Deployment

**Story arc**: Students discover that deployment decisions are not engineering preferences but *physical constraints* -- the speed of light, the power wall, and the energy of data transmission each impose hard limits that no software optimization can overcome, forcing the existence of Edge and TinyML as deployment paradigms.

**Time budget**: 51 min (12 + 12 + 12 + 9 + 6)

### Part A -- The Memory Wall Revelation (~12 min)

**Concept**: Arithmetic Intensity (AI = FLOPs / Bytes moved) determines whether a workload is compute-bound or memory-bound. At AI = 5 FLOPs/Byte -- far below the H100 ridge point of ~295 -- a 6x GPU upgrade (A100 to H100) yields only ~8% latency improvement because the Memory bandwidth term dominates the Iron Law. The correct diagnostic: identify the binding term before spending money. Hardware upgrades only help when they improve the *binding* resource.

**Prediction**: "Your company is considering upgrading from A100 ($15K) to H100 ($30K) GPUs -- a 6x compute increase. For a workload with Arithmetic Intensity of 5 FLOPs/Byte, what latency improvement do you expect?" Options: (a) ~6x (proportional to compute increase), (b) ~3x (half the compute gain), (c) ~1.5x (modest improvement), (d) <1.1x (almost no improvement). Most students pick (a) or (b), reasoning that a better GPU should give a proportional or at least significant speedup. The correct answer is (d): at AI=5, the workload is deeply memory-bound, and the H100's memory bandwidth is only ~1.08x faster than the A100's. The 6x compute upgrade is almost entirely wasted.

**Instrument**: An Arithmetic Intensity slider (1-400 FLOPs/Byte) drives a latency waterfall chart that decomposes into compute and memory terms. The ridge point for each GPU is marked as a vertical line. Below the ridge point, the memory bar dominates and the compute bar is invisible. Above it, compute dominates. A "GPU upgrade comparison" panel shows A100 vs H100 latency at the current AI, with the improvement factor displayed as a large number. Students watch the improvement factor climb from <1.1x at low AI to ~6x at high AI.

**mlsysim grounding**:
```python
# Compare A100 vs H100 at varying arithmetic intensity
# Use Engine.solve() with models of different AI characteristics
for hw in [Hardware.A100, Hardware.H100]:
    profile = Engine.solve(model, hw, batch_size=bs, precision="fp16")
    # Ridge point = hw.compute.peak_flops / hw.memory.bandwidth
```

**Transition to B**: "The memory wall limits what hardware upgrades can achieve. But there is an even more fundamental wall -- one that no amount of money can overcome. The speed of light sets an absolute floor on latency that determines whether cloud deployment is even *physically possible*."

---

### Part B -- The Light Barrier (~12 min)

**Concept**: The speed of light in fiber (~200,000 km/s) sets an irreducible latency floor: at 3,000 km, the minimum round-trip time is 30 ms. For an autonomous vehicle with a 10 ms safety SLA, this means cloud inference is physically impossible beyond ~1,000 km -- the photons literally cannot travel fast enough. This is why Edge ML exists as a deployment paradigm: not preference, but physics. No software optimization, no hardware upgrade, no amount of money can make light travel faster.

**Prediction**: "An autonomous vehicle requires 10 ms end-to-end latency for obstacle detection. The nearest cloud datacenter is 1,500 km away. Is cloud inference feasible if the model runs in 1 ms on the cloud GPU?" Options: (a) Yes -- 1 ms compute leaves 9 ms for network, (b) Yes -- but barely, with ~1 ms margin, (c) No -- propagation delay alone exceeds the SLA, (d) Depends on network congestion. Most students pick (a) or (b) because they focus on compute time and assume the network is "fast enough." The correct answer is (c): 1,500 km round-trip at light speed takes 15 ms in fiber, which alone exceeds the 10 ms SLA before any compute, serialization, or queueing.

**Instrument**: A distance slider (0-5,000 km) drives a stacked latency bar showing: propagation delay (c/distance, irreducible), serialization overhead, compute time, and queueing delay, all against a 10 ms SLA reference line. When the total bar crosses the SLA line, a failure banner reads "SLA VIOLATED -- the speed of light cannot be optimized." A threshold marker shows the maximum feasible distance (~1,000 km for 10 ms SLA, accounting for compute and overhead). An SLA dropdown (10 ms / 50 ms / 200 ms) lets students explore how different applications have different feasibility horizons.

**mlsysim grounding**: Propagation formula: `latency_propagation = 2 * distance / (c * 0.67)` where 0.67 is the fiber refractive index factor. Compute latency from Engine.solve() on cloud hardware. Source: textbook @sec-ml-systems-light-barrier.

**Transition to C**: "The speed of light tells you *where* to deploy. But once you choose a deployment target, a new wall appears: the power budget. A mobile phone running at 5W will thermally throttle within seconds if you push it too hard. Let us see what happens when the same model meets different power envelopes."

---

### Part C -- The Power Wall: Why You HAVE to Pick (~12 min)

**Concept**: Each deployment paradigm (Cloud, Edge, Mobile, TinyML) exists because of a distinct physical constraint, and each comes with both capabilities the others lack and limitations the others do not face. A mobile device running a heavy model at 5W hits a thermal throttle point: sustained power draw above the thermal design point causes the processor to clock down, dropping throughput from peak to ~25% within 60 seconds. The thermal throttling model is a simple piecewise function: full speed below the thermal threshold, linear degradation above it.

**Prediction**: "A ResNet-50 model achieves 60 FPS on a mobile phone's NPU. After running continuously for 90 seconds, what frame rate do you expect?" Options: (a) Still 60 FPS -- the hardware is designed for this, (b) ~45 FPS -- slight thermal degradation, (c) ~15 FPS -- severe thermal throttling, (d) 0 FPS -- the phone shuts down. Most students pick (a) or (b), assuming mobile hardware is designed to sustain peak performance. The correct answer is (c): sustained compute at full power exceeds the thermal envelope, and the OS throttles the processor to prevent damage.

**Instrument**: A deployment target selector (Cloud 300W / Edge 25W / Mobile 5W / TinyML 1mW) with a model selector (ResNet-50, MobileNetV2, DS-CNN). A constraint dashboard shows which physical walls are binding for each combination: light barrier (latency), power wall (thermal budget), memory wall (capacity). For mobile, a "sustained performance" timeline shows FPS over 120 seconds, with a thermal throttle knee at ~30 seconds (piecewise model: full performance below thermal threshold, linear drop to 25% above it). A "deploy" button checks feasibility per target and reports pass/fail with the binding constraint identified.

Thermal throttle model (piecewise):
```
if power_sustained <= thermal_design_point:
    performance = peak_performance
else:
    performance = peak_performance * (thermal_design_point / power_sustained)
    # Clamped to floor of 0.25 * peak_performance
```
Source: simplified from textbook @sec-ml-systems-deployment-paradigms. Simplification: real throttling involves thermal mass and time constants; this model captures the steady-state behavior.

**mlsysim grounding**:
```python
for hw in [Hardware.H100, Hardware.Jetson, Hardware.iPhone, Hardware.ESP32]:
    profile = Engine.solve(model, hw, batch_size=1, precision="fp16")
    # Read: profile.feasible, profile.energy, hw.tdp
    # Compute sustained FPS: if profile.energy / latency > hw.tdp, throttle
```

**Transition to D**: "The power wall constrains *sustained* performance. But for battery-powered devices, there is something even more fundamental than sustained power: total energy. Transmitting data to the cloud costs energy too -- and the ratio is shocking."

---

### Part D -- The Energy of Transmission (~9 min)

**Concept**: For battery-powered devices, transmitting 1 MB of raw data to the cloud costs approximately 1,000x more energy than processing it locally on a low-power NPU. This energy asymmetry is why TinyML exists -- even if cloud inference were instantaneous and free, the energy cost of wireless transmission makes cloud offloading physically impossible for always-on sensing applications that must run for months on a coin cell battery.

**Prediction**: "A battery-powered wildlife sensor captures 1 second of audio (16 KB). It can either (A) transmit the raw audio to the cloud for classification or (B) run a KWS model locally. What is the energy ratio of cloud vs. local processing?" Options: (a) ~2x (cloud is slightly more expensive), (b) ~10x, (c) ~100x, (d) ~1,000x. Most students pick (a) or (b), assuming modern wireless is energy-efficient. The correct answer is (d): wireless transmission energy dominates by three orders of magnitude, because radio power amplifiers are energy-hungry relative to low-power inference accelerators.

**Instrument**: A comparison panel showing two energy budgets: "Send to Cloud" (radio power * transmission_time, computed from data_size / wireless_bandwidth) vs. "Process Locally" (inference energy from Engine.solve()). Sliders for data size (1 KB to 1 MB) and wireless bandwidth (BLE 1 Mbps / WiFi 50 Mbps / LTE 10 Mbps). The ratio is displayed as a large number. A battery life calculator shows how many classifications per day each strategy permits on a 250 mAh coin cell. Cloud offloading drains the battery in hours; local inference lasts months.

**mlsysim grounding**:
```python
profile = Engine.solve(Models.Tiny.DS_CNN, Hardware.ESP32, batch_size=1, precision="int8")
local_energy = profile.energy  # ~0.01 mJ per inference
# Wireless energy: radio_power * (data_size / wireless_bw)
# radio_power ~ 100 mW for BLE, data_size = 16 KB, BLE BW = 1 Mbps -> 128 ms -> 12.8 mJ
```

---

### Synthesis (~6 min)

**Prompt**: A hospital wants to deploy a patient fall-detection system. Three options: (1) cameras streaming to a cloud GPU, (2) a Jetson edge device per floor, (3) a wearable with an ESP32. For each option, identify the binding physical wall (light barrier, power wall, memory wall, or energy wall) and state whether the constraint is *fundamental* (no engineering can fix it) or *economic* (more money could fix it). Which option would you recommend for a rural clinic with unreliable internet, and why?

---
---

## Lab 03: The Constraint Tax -- Orchestrating the ML Lifecycle

**Story arc**: Students follow a DR (diabetic retinopathy) screening team from confident model training to devastating deployment failure, discovering that constraints discovered late cost exponentially more than constraints discovered early -- and that quantifying this cost with Engine.solve() turns "best practices" into hard engineering requirements.

**Time budget**: 51 min (12 + 12 + 12 + 9 + 6)

### Part A -- Constraint Propagation: The DR Clinic Disaster (~12 min)

**Concept**: The cost of discovering a deployment constraint at lifecycle stage N grows exponentially as 2^(N-1). The DR screening case study makes this concrete: a team spends 5 months building a 4 GB model to 95% accuracy, then discovers at deployment that rural clinic tablets have hardware equivalent to the Jetson Orin NX (16 GB RAM, 25W) -- or worse, mobile phones (8 GB RAM, 5W). Engine.solve() proves the infeasibility *quantitatively*: the model does not fit, and this could have been discovered on day one with a single API call.

**Prediction**: "A team has trained a 95%-accurate DR screening model (ResNet-50 backbone, ~100 MB FP16). They plan to deploy it on rural clinic tablets with specs similar to an ESP32 (512 KB SRAM). At which lifecycle stage is the deployment constraint cheapest to address?" Options: (a) During model architecture selection (Stage 2, cost 2x), (b) During training (Stage 3, cost 4x), (c) During evaluation (Stage 4, cost 8x), (d) It does not matter -- the cost is similar at all stages. Most students pick (d), assuming good engineering practices make late changes manageable. The correct answer is (a): the exponential cost curve means that discovering a 200x memory gap at Stage 5 (deployment) forces a complete restart -- 16x the cost of specifying the constraint at Stage 1.

But the deeper lesson is that Engine.solve() can quantify the infeasibility *before any training begins*:

**Instrument**: A "discovery stage" slider (1-6: Problem Definition, Data Engineering, Model Development, Evaluation, Deployment, Monitoring) shows the exponential cost curve 2^(N-1). At each stage, the display shows what artifacts must be rebuilt. A parallel panel runs Engine.solve() live:

```python
profile = Engine.solve(Models.ResNet50, Hardware.ESP32, batch_size=1, precision="fp16")
# Result: feasible=False, memory_footprint=~100 MB vs capacity=512 KB
```

The OOM result appears in red: "INFEASIBLE: 200x over memory budget." A timestamp annotation shows: "This diagnosis took 0.003 seconds. The team spent 150 days before discovering it." A cost counter shows person-days wasted at each discovery stage.

**mlsysim grounding**:
```python
# Day-one feasibility check that would have saved the project
profile = Engine.solve(Models.ResNet50, Hardware.ESP32, batch_size=1, precision="fp16")
# Also show it IS feasible on cloud:
profile_cloud = Engine.solve(Models.ResNet50, Hardware.H100, batch_size=1, precision="fp16")
```

**Transition to B**: "Engine.solve() just proved that one model-hardware pair is infeasible. But a real project must evaluate *many* combinations -- different models, precisions, and hardware targets. How many configurations does a team actually need to explore? Engine.sweep() answers that question."

---

### Part B -- The Iteration Velocity Race (~12 min)

**Concept**: Iteration velocity dominates starting accuracy over realistic development windows. Using the DR screening scenario: Team A (large ensemble, 95% start, 1-week training cycle) vs. Team B (lightweight edge model, 90% start, 1-hour training cycle). Team B overtakes Team A because 100 experiments explore more of the design space than 26 experiments. Engine.sweep() grounds this concretely: evaluating the full design space of models x hardware x precisions reveals how many configurations exist and how long each evaluation takes.

**Prediction**: "Team A starts at 95% accuracy with 1-week training cycles. Team B starts at 90% with 1-hour cycles. After 26 weeks, which team has higher accuracy?" Options: (a) Team A -- 5% head start is insurmountable, (b) Team A -- but barely (within 1%), (c) Team B -- faster iteration wins, (d) They converge to the same accuracy. Most students pick (a) because a 5% accuracy gap feels large. The correct answer is (c): Team B runs ~100 experiments to Team A's 26, exploring far more of the design space and finding better configurations.

**Instrument**: A dual-line accuracy chart shows both teams over a 26-week window, with an accuracy improvement model: `accuracy(t) = accuracy_0 + alpha * log(1 + experiments(t))`. Sliders for cycle time (1 hour to 2 weeks) and starting accuracy (85% to 98%) let students find the crossover point and the failure condition where iteration speed stops compensating.

A second panel shows Engine.sweep() output: given a list of candidate models and hardware targets, how many configurations exist and what is the total evaluation time?

```python
results = Engine.sweep(
    model=Models.ResNet50,
    hardware_list=[Hardware.H100, Hardware.Jetson, Hardware.ESP32],
    batch_sizes=[1, 8, 32],
    precisions=["fp32", "fp16", "int8"]
)
# 3 hardware * 3 batch * 3 precision = 27 configurations
# Show: how many are feasible, which hardware fails, which precision helps
```

**mlsysim grounding**: Engine.sweep() across 3 hardware targets, 3 batch sizes, 3 precisions = 27 configs. Display feasibility matrix and latency heatmap.

**Transition to C**: "You now know *how fast* you can iterate and *how many* configurations to explore. But where should the team spend its time? Most students -- and most teams -- dramatically underestimate the time spent on data."

---

### Part C -- The Whole System View: Where Does the Time Go? (~12 min)

**Concept**: Data-related activities consume 60-80% of ML project effort. Model development, despite receiving the most research attention, is only 10-20%. The DR case study illustrates: expert ophthalmologist annotation ($200/hour), image quality validation across heterogeneous clinic equipment, privacy compliance for medical data, and data transfer from clinics with 2 Mbps uplinks. The ML workflow is the entire system, not just model optimization.

**Prediction**: "In a production ML project, what fraction of total engineering effort goes to model development (architecture search, training, hyperparameter tuning)?" Options: (a) 50-60% (it is the core task), (b) 30-40% (significant but not dominant), (c) 10-20% (surprisingly small), (d) <5% (negligible). Most students pick (a) because model development is what courses and papers focus on. The correct answer is (c): data collection, cleaning, labeling, and validation dominate the effort budget.

**Instrument**: Students allocate a 10-person team (person-months) across five project phases using sliders: Data Collection, Data Labeling/Validation, Model Development, Deployment/Infrastructure, Monitoring/Maintenance. A "project outcome simulator" shows consequences of each allocation: underfunding data leads to "team ran out of clean training data in month 2 -- three modelers idle"; underfunding deployment leads to "model achieves 95% accuracy in development, fails deployment feasibility check (see Part A)." A side-by-side bar compares student allocation vs. published industry data (Hidden Technical Debt paper, MLCommons surveys).

The DR case study shows specifically where that 60-80% goes:
- Data collection: 15 clinics, 5 different camera models, 3 countries
- Labeling: Expert ophthalmologists at $200/hour, 2-minute review per image
- Quality validation: 8% of images rejected for blur, exposure, or field-of-view
- Data infrastructure: 2 Mbps uplink at rural clinics = 46 hours to transfer 40 GB

**mlsysim grounding**: Data transfer time calculation: `transfer_time = dataset_size / bandwidth`. For 40 GB at 2 Mbps: ~46 hours per clinic. This is the Iron Law's D_vol/BW term applied to data logistics rather than training I/O.

**Transition to D**: "You have seen that most effort goes to data, and that constraints must be discovered early. But what happens *after* deployment? Unlike traditional software, ML systems create feedback loops that never stop."

---

### Part D -- Feedback Loops: Why the Lifecycle Never Ends (~9 min)

**Concept**: Unlike traditional software where deployment is the end, ML systems require continuous feedback loops. Production data reveals distributional properties invisible during development. The DR system scaling from 5 pilot clinics to 200 clinics changes everything: new camera equipment, new demographics, new failure modes. Problem definitions evolve. Each feedback event triggers re-entry into an earlier lifecycle stage.

**Prediction**: "After deploying the DR system to 200 clinics (up from 5 pilot clinics), how many complete lifecycle iterations (data collection through deployment) will be needed before the system stabilizes?" Options: (a) 0 -- it was validated during pilot, (b) 1 -- one round of fixes, (c) 2-3 -- a few adjustments, (d) 4-8 -- continuous iteration. Most students pick (a) or (b), assuming the pilot validates the system. The correct answer is (d): 4-8 iterations, because scaling reveals problems that are statistically invisible at pilot scale (rare subgroups, equipment drift, regional differences).

**Instrument**: A timeline simulation of the DR system scaling from pilot to production. A "months since launch" slider reveals feedback events as they occur: "Month 3: Camera model changed at clinic #47 -- accuracy drops 8%." "Month 6: New demographic subgroup (elderly patients with cataracts) has 40% higher error rate." "Month 9: Regulatory audit requires model retraining with balanced demographics." Each event triggers a feedback arrow back to an earlier lifecycle stage (rendered as a Sankey-style flow). A counter tracks total iteration cycles, converging toward the chapter's "4-8 complete iterations for production readiness."

**mlsysim grounding**: Conceptual simulation; no Engine.solve() calls. The feedback events are grounded in the chapter's discussion of ML workflow feedback loops.

---

### Synthesis (~6 min)

**Prompt**: You are the ML lead for the DR screening project. Before writing a single line of training code, you run Engine.solve() on your candidate model (ResNet-50) against three deployment targets (H100, Jetson, ESP32). Based on the results from Part A, draft a one-paragraph "Deployment Constraint Specification" document that your team will use as a gate before model development begins. Include: (1) the target hardware and its binding constraint, (2) the maximum model memory footprint, (3) the latency SLA, and (4) the consequence of violating each constraint (quantified in person-days using the exponential cost curve from Part A).

---
---

## Lab 04: The Data Gravity Trap

**Story arc**: Students discover that data is not just input to ML systems -- it is the heaviest object in the system, governed by physics (bandwidth), economics (egress pricing), and compounding error dynamics that make data engineering the dominant cost and risk factor in production ML.

**Time budget**: 51 min (12 + 12 + 12 + 9 + 6)

### Part A -- The Feeding Tax: When Your GPU Starves (~12 min)

**Concept**: A standard cloud disk (250 MB/s) leaves an A100 GPU idle over 95% of the time during ResNet-50 training. The bottleneck is not compute but I/O -- the data pipeline cannot feed the accelerator fast enough. The Iron Law's D_vol/BW term dominates. Buying more GPUs just means more expensive hardware sitting idle. Engine.solve() reveals the compute latency for a single batch is tiny; the question is whether the data pipeline can deliver batches fast enough.

**Prediction**: "You are training ResNet-50 on an A100 GPU with a standard cloud SSD (250 MB/s sequential read). What percentage of wall-clock time is the GPU actually computing?" Options: (a) 80-90% (GPUs are expensive, they should be busy), (b) 50-60% (some I/O overhead is expected), (c) 20-30% (significant I/O bottleneck), (d) <5% (GPU is almost entirely idle). Most students pick (a) or (b), reasoning that expensive hardware should be well-utilized. The correct answer is (d): at 250 MB/s, loading a batch of 224x224 images takes far longer than the A100 takes to compute the forward+backward pass.

**Instrument**: A storage type dropdown (HDD 100 MB/s / SSD 250 MB/s / NVMe 3 GB/s / RAM disk 25 GB/s) and a DataLoader workers slider (1-16) control I/O bandwidth. Engine.solve() provides the compute latency per batch. A batch timeline bar shows GPU compute (green) vs. I/O wait (red). At default (SSD, 1 worker), the red bar dominates. A GPU utilization gauge turns green only when utilization exceeds 80%. Students discover that the fix is faster storage + parallel loading, not more GPUs.

**mlsysim grounding**:
```python
profile = Engine.solve(Models.ResNet50, Hardware.A100, batch_size=32, precision="fp16", is_training=True)
compute_per_batch = profile.latency  # The GPU's share of the step
# I/O time = batch_data_size / (storage_bw * num_workers)
# batch_data_size = batch_size * 224 * 224 * 3 * 4 bytes
# GPU utilization = compute_per_batch / (compute_per_batch + io_time)
```

**Transition to B**: "You just discovered that the GPU starves when storage is slow. But what if the data is not even on the same machine? Moving data across the network introduces a new cost -- and the economics are brutal."

---

### Part B -- Data Gravity: Move the Compute, Not the Data (~12 min)

**Concept**: Moving 50 TB across cloud regions costs $4,000 in egress fees alone -- 20x the compute cost of the training job itself. The physics: T = D_vol/BW governs transfer time, and cloud egress pricing ($0.08/GB) creates an economic gravity that pins computation to wherever data resides. Above a crossover dataset size, it is always cheaper to provision GPUs near the data than to move the data to existing GPUs.

**Prediction**: "You need to train on a 50 TB dataset. The data is in us-east-1 but your GPU cluster is in eu-west-1. The network link is 10 Gbps. How long does the data transfer take?" Options: (a) ~30 minutes, (b) ~5 hours, (c) ~11 hours, (d) ~3 days. Most students dramatically underestimate transfer time because they think in terms of small files. The correct answer is (c): 50 TB / 10 Gbps = ~11.1 hours, and that assumes sustained full-bandwidth utilization with no protocol overhead.

**Instrument**: Sliders for dataset size (1 GB to 500 TB, log scale) and network bandwidth (1 Gbps to 100 Gbps). A cost comparison chart shows two bars: "Move Data" (egress cost + remote GPU cost) vs. "Move Compute" (local GPU cost + provisioning premium). At default settings (50 TB, 10 Gbps), the "Move Data" bar is dominated by the $4,000 egress charge. A crossover finder shows the dataset size where the strategies break even. Below ~1 TB, moving data is cheaper; above ~5 TB, moving compute wins.

**mlsysim grounding**: Transfer time: `T = dataset_size / bandwidth`. Egress cost: `cost = dataset_size * $0.08/GB`. Compute cost for training: derived from Engine.solve() runtime estimate. Source: textbook @sec-data-engineering-data-gravity.

**Transition to C**: "Data gravity tells you where to put your compute. But what happens when the data itself is wrong? A small error at ingestion can amplify through the pipeline and silently destroy model accuracy."

---

### Part C -- Data Cascades: The 2% Error That Ate 15% Accuracy (~12 min)

**Concept**: Data quality errors introduced at early pipeline stages amplify through downstream stages. A 2% schema error at ingestion (e.g., zip code loses leading zero, corrupting geographic features) compounds to ~15% accuracy degradation at deployment. Detection takes a median 4 weeks (industry data from Sambasivan et al. 2021), during which the model silently makes degraded predictions on every request. This is the Pipeline Jungle: without data contracts and schema validation, upstream changes cause catastrophic silent failures.

**Prediction**: "A data pipeline has a 2% error rate at ingestion (some records have a corrupted feature). What accuracy degradation do you expect at the model's output?" Options: (a) ~2% (error passes through linearly), (b) ~5% (slight amplification), (c) ~15% (significant amplification), (d) ~50% (catastrophic). Most students pick (a) or (b), assuming errors propagate linearly. The correct answer is (c): errors compound because corrupted features affect feature interactions, which affect learned representations, which affect all predictions that depend on those features -- not just the 2% of corrupted records.

**Instrument**: An error injection point selector (Ingestion / Feature Engineering / Training Data / Validation Set) and a pipeline depth slider (2-8 stages) let students trace how errors amplify. A cascade amplification chart shows error rate growing through the pipeline. An amplification factor display shows the ratio of output error to input error. A silent degradation timeline shows accuracy declining over weeks before detection, with a "detection latency" slider (1-8 weeks) showing cumulative damage. The key visual: the gap between "when the error was introduced" (day 0) and "when it was detected" (week 4) -- shaded red to show the damage window.

**mlsysim grounding**: Amplification model: `error_stage_n = error_0 * amplification_factor^n` where amplification_factor ~ 1.3-1.5 per stage (from chapter discussion of data cascade dynamics). Source: textbook @sec-data-engineering-data-cascades, Sambasivan et al. (2021).

**Transition to D**: "Data cascades show how small errors amplify. But even with perfect data, some deployment contexts impose requirements so extreme that standard accuracy metrics are meaningless. An always-on sensor evaluates millions of windows per month -- what does 99% accuracy actually mean at that scale?"

---

### Part D -- The False Positive Trap: When 99% Is Not Enough (~9 min)

**Concept**: For always-on KWS (Keyword Spotting) systems, standard accuracy metrics are meaningless at deployment scale. An always-on device evaluates ~2.6 million 1-second windows per month (30 days * 24 hours * 3600 seconds * duty_cycle). A tolerance of 1 false wake-up per month requires a rejection rate of 99.99996% -- far beyond what "99% accuracy" suggests. Engine.solve() shows the inference latency per window, which combined with duty cycle determines total energy and the feasibility of continuous operation.

**Prediction**: "An always-on smart speaker evaluates 1-second audio windows continuously. You want at most 1 false wake-up per month. What rejection rate (true negative rate) is required?" Options: (a) 99% (one nine), (b) 99.9% (three nines), (c) 99.999% (five nines), (d) 99.99996% (effectively six nines). Most students pick (a) or (b) because "99%" sounds very accurate. The correct answer is (d): 2.6 million windows per month means even 99.999% would produce 26 false wakes.

**Instrument**: Students set a false positive tolerance (false wake-ups per month: 1, 5, 10, 50) and a duty cycle (hours per day: 1-24). The lab computes windows per month = duty_hours * 3600 * 30, then required rejection rate = 1 - (tolerance / windows). A "nines counter" displays the number of nines after the decimal. Students discover that "1 false wake per month" requires six nines.

A second panel shows the per-window inference cost from Engine.solve():
```python
profile = Engine.solve(Models.Tiny.DS_CNN, Hardware.ESP32, batch_size=1, precision="int8")
# energy_per_window = profile.energy
# energy_per_month = energy_per_window * windows_per_month
# battery_life = battery_capacity / energy_per_month
```

**mlsysim grounding**: Engine.solve() for DS-CNN on ESP32 gives energy per inference. Multiply by 2.6M windows/month for total energy budget. Compare against coin cell capacity.

---

### Synthesis (~6 min)

**Prompt**: You are building a data pipeline for the DR screening system from Lab 03. Using numbers from this lab, answer: (1) Given 40 GB of retinal images spread across 15 clinics with 2 Mbps uplinks, what is the total data transfer time and should you move data or move compute? (2) If a 2% labeling error enters at ingestion, what accuracy degradation should you expect and when will you detect it? (3) If the system runs continuously screening patients, with a tolerance of 1 missed diagnosis per 10,000 scans, what is the required sensitivity and does your data pipeline need to support this level of quality?

---
---

## Lab 05: Neural Computation

**Story arc**: Students trace the computational life of a neural network from individual operations (transistors) through memory hierarchy (cache tiers) to full forward and backward passes, discovering that the dominant costs are not where they expect -- width scales quadratically, memory tiers create cliffs not slopes, and training requires storing everything the forward pass could discard.

**Time budget**: 50 min (10 + 10 + 10 + 10 + 5 + 5)

### Part A -- The Transistor Tax (~10 min)

**Concept**: Activation functions have wildly different silicon costs: ReLU requires ~50 transistors (a single comparison), while Sigmoid requires ~2,500 transistors (exponentiation, division). This 50x gap is negligible on a cloud GPU (activation compute is <1% of total) but becomes dominant on a mobile NPU, where activation functions can consume 23% of total inference time. The deployment context determines whether this "free" design choice has a real cost.

**Prediction**: "On a mobile NPU, what fraction of inference time comes from activation functions if you use Sigmoid instead of ReLU in every layer?" Options: (a) <1% (negligible, like on GPUs), (b) ~5% (noticeable but small), (c) ~23% (significant cost), (d) ~50% (dominant cost). Most students pick (a) because activation functions feel like a footnote in neural network design. The correct answer is (c): on resource-constrained hardware, the 50x transistor cost difference becomes a meaningful fraction of the total compute budget.

**Instrument**: A per-layer activation function dropdown selector (4 layers; choices: ReLU, GELU, Sigmoid, Swish) with a context toggle (Cloud GPU vs. Mobile NPU). A stacked bar chart shows total inference time decomposed into: matrix multiply, activation functions, normalization, other. On Cloud, swapping activations barely changes the chart. On Mobile, switching all layers to Sigmoid visibly expands the activation bar from <1% to ~23%. A silicon cost counter shows total transistors committed to activation compute.

**mlsysim grounding**: Transistor cost ratios from textbook constants. Latency impact modeled as activation_time = (transistors_per_activation * activations_per_layer * layers) / hardware_clock. Source: textbook @sec-nn-computation-activation-functions.

**Transition to B**: "Activation choice affects compute cost. But there is a more dangerous cost hiding in the memory system. When your activation tensors grow past a cache tier boundary, latency does not increase gradually -- it jumps by 10-100x."

---

### Part B -- The Memory Hierarchy Cliff (~10 min)

**Concept**: Activation tensor size determines memory tier placement (L1 cache / L2 cache / HBM / DRAM), and crossing a tier boundary is a 10-100x latency step function, not a gradual slope. A 16 KB tensor fits in L2 cache (fast). Doubling batch size pushes it to 32 KB -- still L2 on a cloud GPU, but spilling to HBM on a mobile device, triggering a 10x latency cliff. Students learn that memory-aware architecture design is not about total capacity but about tier boundaries.

**Prediction**: "A layer produces a 16 KB activation tensor that fits in L2 cache. You double the batch size, making the tensor 32 KB. How does latency change?" Options: (a) 2x (linear with data size), (b) 1.5x (some overhead), (c) 10x (cache tier boundary crossed), (d) No change (hardware handles it). Most students pick (a) because they model memory as a flat hierarchy. The correct answer depends on context: on a cloud GPU with large L2, (a) is roughly correct. On a mobile NPU, (c) is correct because 32 KB exceeds L2 and spills to HBM.

**Instrument**: Batch size slider (1-512) and layer width slider (64-4096) with context toggle (Cloud GPU / Mobile NPU). A stacked bar chart shows tensor size colored by memory tier (green=L1, blue=L2, orange=HBM, red=DRAM), with horizontal threshold lines at each tier boundary. The latency curve has visible step-function jumps at tier boundaries. Students see that modest increases in batch size or width push past L2 on mobile, triggering the cliff.

**mlsysim grounding**: Tier capacities from hardware specs (L1: ~128 KB, L2: ~6 MB for cloud GPU, ~512 KB for mobile). Latency per tier: L1 ~1 ns, L2 ~5 ns, HBM ~100 ns, DRAM ~200 ns. Source: textbook @sec-nn-computation-memory-hierarchy.

**Transition to C**: "Memory tiers create cliffs. But how much compute does each layer actually require? The answer involves a scaling law that catches most students off guard: width does not scale linearly."

---

### Part C -- The Width-Squared Surprise (~10 min)

**Concept**: Dense layer FLOPs scale as O(width^2): FLOPs = 2 * width_in * width_out. Doubling hidden layer width from 128 to 256 does not double FLOPs -- it quadruples them for the layers connected to the widened layer. For a 3-layer MLP (784 -> hidden -> hidden -> 10), doubling hidden width from 128 to 256 increases total FLOPs by ~3.8x, not 2x. Architecture decisions are the dominant variable in the Iron Law's Operations term.

**Prediction**: "A 3-layer MLP has hidden layers of width 128. You double the hidden width to 256. By how much do total FLOPs increase?" Options: (a) 2x (linear with width), (b) 3x, (c) ~4x (quadratic), (d) 8x (cubic). Most students pick (a) because "double the width, double the work" feels obvious. The correct answer is (c): the hidden-to-hidden layer FLOPs scale as width^2, and the input-to-hidden and hidden-to-output layers scale linearly with width, so the total increase is approximately 3.8x.

**Instrument**: Layer width sliders for a 3-layer MLP (input fixed at 784, output at 10, hidden adjustable 32-2048). A live FLOP counter per layer and a stacked bar chart show how each layer's compute scales. The total FLOP multiplier is displayed as a large number. An arithmetic intensity readout shows AI changing with width, connecting to why narrow layers are bandwidth-bound.

**mlsysim grounding**: FLOPs = 2 * input_dim * output_dim per dense layer. Source: textbook @sec-nn-computation-flop-counting.

**Transition to D**: "Width-squared scaling governs inference compute. But training has a much larger surprise: the forward pass can discard intermediate results layer by layer, but backpropagation cannot."

---

### Part D -- Forward vs. Backward: Where the Memory Goes (~10 min)

**Concept**: During inference, the forward pass can discard each layer's activations after the next layer consumes them -- memory usage is constant regardless of depth. During training, backpropagation requires every intermediate activation to compute gradients, so all activations must be stored simultaneously. For a 20-layer network, training memory is dominated by activations (scaling linearly with depth * batch_size), not weights. This Part covers the conceptual memory multiplier only -- optimizer state details are Lab 08's territory.

**Prediction**: "A 20-layer model uses 50 MB for inference. How much memory does training require (just for weights, gradients, and activations -- ignoring optimizer state)?" Options: (a) ~50 MB (same as inference), (b) ~100 MB (2x for gradients), (c) ~200 MB (4x), (d) ~500 MB+ (10x+). Most students pick (b), reasoning that gradients double the memory. The correct answer is (c) or (d) depending on batch size: activations stored for all 20 layers dominate the memory budget, and this scales with both depth and batch size.

**Instrument**: A depth slider (3-20 layers) and batch size slider (1-128) with a phase toggle (Inference vs. Training). In inference mode, a memory ledger shows only current-layer activations + weights = small constant. In training mode, the ledger accumulates all layers' activations simultaneously, and students watch the stacked bar grow linearly with depth. The training-to-inference memory ratio is displayed as a multiplier. At depth=20 and batch=32, the ratio reaches 4-10x.

**mlsysim grounding**: Memory model: inference = weight_bytes + max_single_layer_activation. Training = weight_bytes + gradient_bytes + sum_all_layer_activations. Activation per layer = batch_size * width * bytes_per_element. Source: textbook @sec-nn-computation-backprop-memory.

---

### Synthesis (~5 min)

**Prompt**: Deploy a 10-layer model on a mobile NPU with 8 GB RAM and 5W power budget for both inference (30 FPS) and on-device fine-tuning. Using numbers from this lab, specify: (1) activation function choice and its impact on mobile inference time, (2) maximum batch size before crossing the L2 cache cliff, (3) total training memory (weights + gradients + activations, ignoring optimizer state), and (4) whether on-device fine-tuning is feasible given the forward-vs-backward memory multiplier.

---
---

## Lab 06: Network Architectures

**Story arc**: Students discover that architecture is not just an accuracy choice -- it is a *systems* choice that determines parameter count, memory access patterns, parallelism potential, and hardware utilization, with each architecture family occupying a distinct point in the compute-memory trade-off space.

**Time budget**: 52 min (12 + 12 + 10 + 8 + 5 + 5)

### Part A -- The Cost of No Structure (~12 min)

**Concept**: Inductive bias is not an abstract concept -- it is a physical memory constraint. An MLP processing a 224x224 RGB image requires a first-layer weight matrix of 150,528 x hidden_dim. At hidden_dim = 150,528 (to match input), that is 22.7 billion parameters -- 91 GB in FP32. A CNN with 3x3 filters requires 1,728 parameters in its first layer, a 13.1 million-fold reduction. Inductive bias (locality, weight sharing) is the mechanism that makes computer vision physically feasible.

**Prediction**: "An MLP takes a 224x224 RGB image as a flattened input vector (150,528 dimensions). The first hidden layer also has 150,528 neurons. How many parameters does this single layer have?" Options: (a) ~150K (about the input size), (b) ~23M (like ResNet-50 total), (c) ~1B (a billion), (d) ~22.7B (twenty-two billion). Most students pick (a) or (b), vastly underestimating because they do not internalize the O(d^2) scaling of dense layers. The correct answer is (d): 150,528^2 = 22.66 billion parameters in one layer.

**Instrument**: Architecture toggle (MLP / CNN 3x3 / CNN 5x5) and image resolution slider (28x28 to 512x512). A parameter count bar chart on log scale shows the three architectures at current resolution. Memory bars with device threshold lines (H100 80 GB, Jetson 16 GB, iPhone 8 GB) show feasibility. At 224x224, the MLP bar extends past all device thresholds. A "fold reduction" counter shows the CNN-to-MLP parameter ratio.

**mlsysim grounding**: Parameter calculation: MLP first layer = input_dim * hidden_dim. CNN first layer = kernel_h * kernel_w * channels_in * channels_out. Source: textbook @sec-architectures-inductive-bias.

**Transition to B**: "CNNs solve the parameter explosion for images by exploiting spatial locality. But Transformers face a different scaling problem: attention scores grow quadratically with sequence length. This creates a hard memory ceiling on how much context a model can process."

---

### Part B -- The Quadratic Wall (~12 min)

**Concept**: Transformer self-attention creates an N x N score matrix that scales quadratically with sequence length, imposing a hard OOM ceiling on context window size. Doubling context from 4K to 8K tokens quadruples attention memory, not doubles it. At 128K tokens (modern LLM context lengths), the attention matrix alone requires ~64 GB in FP16 for a single head -- and with 32 heads, this exceeds even an H100.

**Prediction**: "Doubling a Transformer's context length from 4,096 to 8,192 tokens -- how much more memory does the attention mechanism require?" Options: (a) 2x (linear with tokens), (b) 4x (quadratic), (c) 8x, (d) 16x. Most students pick (a) because "double the input, double the memory" is the default intuition. The correct answer is (b): the N*N attention matrix means 2N * 2N = 4N^2 -- a 4x increase.

**Instrument**: Sequence length slider (512 to 131,072 tokens, log scale) and attention heads slider (1-32). An attention memory curve (GB vs. tokens) plotted on log-log axes with H100 (80 GB) and Jetson (16 GB) threshold lines. Memory formula: `attention_memory = 2 * seq_len^2 * heads * bytes_per_element`. Students watch the curve cross device thresholds. At 128K tokens with 32 heads, the curve hits ~64 GB for attention alone.

**mlsysim grounding**: Attention memory formula from textbook @sec-architectures-attention-complexity. Can cross-reference with Engine.solve() for Transformer workloads at varying seq_len.

**Transition to C**: "The quadratic wall constrains how long a sequence can be. But there is another architectural trade-off hiding in plain sight: two networks with identical parameter counts and total FLOPs can have *dramatically* different latencies."

---

### Part C -- Depth vs. Width: The Sequential Bottleneck (~10 min)

**Concept**: Two networks with identical parameter counts and total FLOPs can have dramatically different latencies because depth imposes O(L) sequential steps while width exposes parallelism. A deep-narrow network (128 layers, width 32) has the same FLOPs as a shallow-wide network (2 layers, width 512), but the deep network is ~10x slower due to sequential layer dispatch overhead and reduced per-layer parallelism. FLOPs are a necessary but insufficient proxy for latency.

**Prediction**: "Two networks have identical total FLOPs and parameter counts: one is 128 layers deep with width 32, the other is 2 layers deep with width 512. Which is faster at inference?" Options: (a) Same speed -- same FLOPs means same time, (b) The deep network -- more layers means more specialized, (c) The shallow-wide network -- by ~2x, (d) The shallow-wide network -- by ~10x. Most students pick (a) because they equate FLOPs with latency. The correct answer is (d): dispatch overhead (per-layer tax) and reduced parallelism within narrow layers make the deep network dramatically slower.

**Instrument**: Depth slider (2-128 layers) and width slider (32-2048), with a constraint that total parameters remain approximately constant (width adjusts automatically when depth changes, or vice versa). Context toggle (Cloud/Edge). A latency waterfall chart decomposes per-layer compute, memory load, and dispatch overhead. The dispatch overhead bars (one per layer) visually accumulate for deep networks.

**mlsysim grounding**: Per-layer dispatch overhead from hardware.dispatch_tax. Total latency = sum of per-layer (max(compute, memory) + dispatch_tax). Source: textbook @sec-architectures-depth-vs-width.

**Transition to D**: "Depth vs. width determines latency for a *given* architecture. But different architecture *families* have fundamentally different relationships with the hardware. Let us see why CNNs and Transformers use GPUs in completely different ways."

---

### Part D -- Workload Signatures (~8 min)

**Concept**: Each architecture family has a characteristic arithmetic intensity (FLOPs per byte of data moved) that determines whether it is compute-bound or memory-bound. CNNs have high AI (>20 FLOPs/byte, compute-bound), Transformers at inference have low AI (<1 FLOPs/byte for attention, memory-bound), and MLPs at batch=1 have very low AI (~0.5). The hardware ridge point determines the crossover. Students who assume Transformers are the most hardware-efficient (because they are "modern") discover that CNNs actually achieve the highest utilization.

**Prediction**: "Which architecture family achieves the highest GPU utilization (MFU) at batch=1?" Options: (a) Transformer (modern and optimized), (b) CNN (high arithmetic intensity), (c) MLP (simplest architecture), (d) All roughly equal. Most students pick (a) because Transformers dominate recent research. The correct answer is (b): CNNs have the highest arithmetic intensity due to weight reuse across spatial positions, making them the most compute-efficient architecture family on GPUs.

**Instrument**: Architecture selector (MLP / CNN / Transformer / DLRM) with batch size slider (1-256). Engine.solve() runs for representative models of each family. A horizontal bar chart shows arithmetic intensity for each architecture at the current batch size, with a vertical "ridge point" line for the target hardware. Operations left of the ridge point are memory-bound; right is compute-bound. An MFU column shows utilization for each.

**mlsysim grounding**:
```python
models = [Models.ResNet50, Models.Language.GPT2, Models.Tiny.DS_CNN, Models.DLRM]
for m in models:
    profile = Engine.solve(m, Hardware.A100, batch_size=bs, precision="fp16")
    # Read: profile.arithmetic_intensity, profile.mfu, profile.bottleneck
```

---

### Synthesis (~5 min)

**Prompt**: A wildlife conservation project needs to classify animals from camera trap images on a 16 GB edge device (Jetson Orin NX) with a 50 ms latency SLA. Using the four analyses from this lab, justify: (1) Why not an MLP? (cite the parameter explosion from Part A), (2) Why not a large Transformer? (cite the quadratic wall from Part B), (3) Should the CNN be deep-narrow or shallow-wide? (cite the dispatch overhead from Part C), (4) What MFU should you expect? (cite the arithmetic intensity from Part D).

---
---

## Lab 07: ML Frameworks

**Story arc**: Students discover that frameworks are not just programming conveniences -- they are *execution engines* whose architectural decisions (eager vs. compiled, fused vs. unfused, cloud vs. edge runtime) determine whether the same model runs 17x faster or fails entirely, without changing a single weight.

**Time budget**: 50 min (10 + 10 + 12 + 8 + 5 + 5)

### Part A -- The Dispatch Tax (~10 min)

**Concept**: Python dispatch overhead (~10 us per operation) makes models with many small kernels overhead-bound, not compute-bound, regardless of how fast the GPU is. GPU utilization depends on the ratio of compute-per-kernel to dispatch cost, not total operation count. A KWS model with 1,000 tiny kernels (~5 us each) achieves <1% GPU utilization because dispatch overhead exceeds compute time for every kernel. A GPT-2-like model with 20 large kernels (~500 us each) achieves >90% utilization on the same hardware.

**Prediction**: "Two models run on the same GPU. Model A (KWS) has 1,000 operations. Model B (GPT-2) has 20 operations. Which achieves higher GPU utilization?" Options: (a) Model A -- more operations means the GPU is busier, (b) Model B -- fewer but larger operations, (c) About the same -- same GPU, (d) Depends on batch size. Most students pick (a) because "more operations = more work = higher utilization" seems logical. The correct answer is (b): Model B's 20 large kernels each amortize the ~10 us dispatch overhead, while Model A's 1,000 tiny kernels spend more time on dispatch than on compute.

**Instrument**: Kernel count slider (10-2000) and compute-per-kernel slider (1-10,000 us, log scale). A dispatch overhead display (fixed at ~10 us per kernel). A GPU utilization gauge: utilization = total_compute / (total_compute + total_dispatch). Two preset buttons: "KWS-like" (1000 kernels, 5 us each) and "GPT-2-like" (20 kernels, 500 us each). A stacked timeline bar shows green (compute) vs. orange (dispatch) for each preset.

**mlsysim grounding**: Dispatch tax from hardware specs: `hardware.dispatch_tax`. Per-kernel overhead model: total_overhead = num_kernels * dispatch_tax. Source: textbook @sec-frameworks-dispatch-overhead.

**Transition to B**: "The dispatch tax wastes time launching kernels. But there is a deeper waste: between every kernel launch, data is written to HBM and then read back. What if you could fuse multiple operations into a single kernel and eliminate those intermediate memory round-trips?"

---

### Part B -- The Fusion Dividend (~10 min)

**Concept**: Kernel fusion eliminates intermediate HBM writes between operations. Element-wise operations like ReLU have arithmetic intensity <1 FLOPs/byte -- they are permanently memory-bound and achieve <1% compute utilization when executed individually. Fusing a sequence of LayerNorm + Dropout + ReLU into a single kernel reduces HBM traffic by 3x (one read + one write instead of three of each), yielding a 3-5x speedup. The speedup comes from memory traffic reduction, not compute reduction.

**Prediction**: "Fusing 3 element-wise operations (LayerNorm, Dropout, ReLU) into one kernel eliminates intermediate HBM writes. What speedup do you expect?" Options: (a) ~1.2x (minor improvement), (b) ~1.5x (noticeable), (c) ~3x (significant), (d) ~10x. Most students pick (a) or (b) because they focus on compute savings (there are none) and underestimate the memory wall. The correct answer is (c): memory traffic drops by ~3x, and since these operations are entirely memory-bound, latency drops proportionally.

**Instrument**: An operation sequence builder where students chain 2-5 element-wise operations from a menu (ReLU, LayerNorm, Dropout, Add, Multiply, GELU). Toggle between "Eager" (each op: read from HBM, compute, write to HBM) and "Fused" (single kernel: one HBM read, all compute, one HBM write). A timeline visualization shows HBM access bars (red) vs. compute bars (green) for each mode. A speedup counter and a memory traffic counter (GB) show the quantitative difference. An arithmetic intensity readout shows unfused AI ~0.5 vs. fused AI ~1.5.

**mlsysim grounding**: Memory traffic model: unfused = 2 * tensor_size * num_ops (read+write per op). Fused = 2 * tensor_size (one read + one write). Latency = memory_traffic / bandwidth. Source: textbook @sec-frameworks-kernel-fusion, referencing the H100 ridge point.

**Transition to C**: "Kernel fusion is done by the compiler. But compilation itself has a cost -- sometimes tens of seconds of upfront compilation time. When is that investment worth it?"

---

### Part C -- The Compilation Break-Even (~12 min)

**Concept**: torch.compile (and similar JIT compilers) have a fixed upfront cost -- 30 seconds for ResNet-50, potentially minutes for larger models. This cost must be amortized over production inferences. The break-even point depends on deployment volume, not model quality. A 48% speedup on ResNet-50 requires ~134,000 inferences to recover the 30-second compilation cost. For a research notebook that runs 10 inferences, compilation makes things *slower*. For a production endpoint serving 1M requests/day, it pays for itself in seconds.

**Prediction**: "torch.compile gives a 48% speedup on ResNet-50 but takes 30 seconds to compile. How many inferences do you need before compilation pays off?" Options: (a) ~100 (almost immediately), (b) ~1,000, (c) ~10,000, (d) ~134,000. Most students pick (a) or (b) because a 48% speedup sounds large enough to amortize quickly. The correct answer is (d): each inference saves ~2 ms (from ~4.2 ms to ~2.2 ms), so recovering 30 seconds requires 30,000 / 0.002 = 15,000... wait, let me recalculate. If eager = 4.2 ms and compiled = 2.2 ms, savings per inference = 2.0 ms. Break-even = 30,000 ms / 2.0 ms = 15,000 inferences. (The exact number depends on the specific speedup and compile time; the key lesson is that break-even is measured in *tens of thousands*, not hundreds.)

**Instrument**: Deployment volume slider (10 to 10M requests/hour, log scale) and compile time slider (5-300 seconds). A break-even timeline chart shows two cumulative time curves: eager (linear, steep) and compiled (starts 30s higher, then grows with a shallower slope). The crossover point is highlighted and labeled with the exact inference count. An ROI gauge turns from red (net loss) to green (net gain) as volume increases past break-even. Context toggle (Cloud high-volume / Edge low-volume) shows how deployment context determines whether compilation makes sense.

**mlsysim grounding**: Eager and compiled latencies from Engine.solve() at different efficiency levels (efficiency=0.5 for eager, efficiency=0.75 for compiled as a proxy for compiler optimization). Source: textbook @sec-frameworks-compilation.

**Transition to D**: "Compilation speed is a framework-level optimization. But the choice of framework itself can be an even bigger lever -- the same model spans 17x latency across frameworks, and on a microcontroller, most frameworks do not fit at all."

---

### Part D -- The Deployment Spectrum (~8 min)

**Concept**: Framework selection determines feasibility, not just speed. The same ResNet-50 model -- identical weights, identical architecture -- spans 17x latency (PyTorch eager 52 ms to TensorRT 3 ms) and 56x memory (1,800 MB PyTorch runtime to 32 MB TF Lite Micro) across frameworks. On a 512 KB microcontroller (ESP32), the question is not "which framework is fastest" but "which framework fits at all" -- and the answer is TF Lite Micro or equivalent, because every other framework exceeds device memory just for its *runtime*, before loading any model.

**Prediction**: "The PyTorch runtime alone (before loading any model) requires ~1,800 MB of memory. The ESP32 has 512 KB. By what factor does the runtime exceed device memory?" Options: (a) ~10x, (b) ~100x, (c) ~3,500x, (d) It fits with optimization. Most students dramatically underestimate the gap because they have never thought about framework runtime footprint as a deployment constraint. The correct answer is (c): 1,800 MB / 0.5 MB = 3,600x.

**Instrument**: Framework dropdown (PyTorch, TensorFlow, ONNX Runtime, TF Lite, TensorRT, TF Lite Micro) and deployment target toggle (Cloud / Edge / MCU). Metric cards show: runtime memory, model memory, total memory, latency, and energy per inference. Feasibility banner turns red on OOM with annotation: "Framework runtime alone exceeds device memory by Nx." A comparison table shows all frameworks simultaneously for the selected target. On MCU, only TF Lite Micro shows green.

**mlsysim grounding**:
```python
# Model memory from Engine.solve()
profile = Engine.solve(Models.ResNet50, hw, batch_size=1, precision="fp16")
# Framework runtime overhead: lookup table from textbook Table 7.x
# Total = framework_runtime + profile.memory_footprint
```

---

### Synthesis (~5 min)

**Prompt**: You manage two deployments: (1) A KWS model (1,000 small kernels, 5 us each) on an ESP32, and (2) ResNet-50 on a cloud endpoint serving 500K requests/day. For each, specify: the framework you would choose, whether you would compile, whether kernel fusion helps, and the expected GPU/NPU utilization. Justify each choice with specific numbers from the lab.

---
---

## Lab 08: Model Training

**Story arc**: Students discover that training a neural network is not just "forward pass but bigger" -- it is a four-stage pipeline where memory budgets, pipeline bottlenecks, precision traps, and communication overhead each create surprising walls, and optimizing the wrong stage wastes resources while the true bottleneck goes untouched.

**Time budget**: 52 min (10 + 12 + 10 + 10 + 5 + 5)

### Part A -- The Memory Budget Shock (~10 min)

**Concept**: Training memory = Weights + Gradients + Optimizer State + Activations. For Adam optimizer in FP32, the static state alone (before any activations) requires 16 bytes per parameter: 4B weights + 4B gradients + 4B momentum + 4B variance. A 7B-parameter model needs 112 GB just for parameter state -- exceeding even an 80 GB H100 before storing a single activation. Students who completed Lab 05 Part D (forward vs. backward memory) already know activations dominate for deep networks; this Part reveals that *optimizer state* is the hidden giant for large models.

**Prediction**: "A 7B-parameter model is trained with Adam in FP32. What is the minimum memory required just for parameter state (weights + gradients + optimizer moments), before any activations?" Options: (a) 28 GB (7B * 4 bytes), (b) 56 GB (weights + gradients), (c) 84 GB (weights + gradients + one momentum buffer), (d) 112 GB (weights + gradients + two momentum buffers). Most students pick (a) or (b) because they forget that Adam maintains two additional state tensors (momentum and variance) per parameter. The correct answer is (d): 7B * 16 bytes/param = 112 GB.

**Instrument**: Model size slider (0.1B to 70B parameters, log scale), optimizer dropdown (SGD 8 bytes/param / SGD+Momentum 12 bytes/param / Adam 16 bytes/param / Adafactor ~10 bytes/param), precision toggle (FP32 / BF16). A stacked bar chart shows four components: weights (blue), gradients (green), optimizer state (orange), and a hatched region for "activations (depends on batch size, see Lab 05)." Device RAM threshold lines for H100 (80 GB), A100 (80 GB), Jetson (16 GB). At 7B + Adam + FP32, the bar exceeds the H100 line before activations.

**mlsysim grounding**:
```python
profile = Engine.solve(Models.Language.Llama3_8B, Hardware.H100,
                       batch_size=1, precision="fp32", is_training=True)
# profile.memory_footprint includes training memory via model.training_memory()
# Also: profile.feasible reveals whether it fits
```

**Transition to B**: "You now know that memory is the first wall in training. But even when the model fits in memory, training involves a four-stage pipeline -- and the stage you think is the bottleneck almost certainly is not."

---

### Part B -- The Training Pipeline: Finding the Bubble (~12 min)

**Concept**: Training is a four-stage pipeline: Data Loading, Host-to-Device Transfer, Forward+Backward Pass, and Gradient Synchronization. Total throughput is limited by the *slowest* stage, creating "accelerator bubbles" -- intervals where the GPU sits idle waiting for data or communication. The chapter's key insight: most training runs are NOT compute-bound. On a V100 with spinning disk storage, data loading can take 10x longer than the forward+backward pass. With 8 GPUs and slow interconnect, gradient sync can dominate. The fix depends on correctly identifying which stage is binding.

**Prediction**: "For GPT-2 training on a V100 with standard SSD storage and 4 GPUs over PCIe, which stage is the bottleneck?" Options: (a) Data loading (disk I/O), (b) Host-to-device transfer (PCIe), (c) Forward + backward pass (compute), (d) Gradient synchronization (inter-GPU). Most students pick (c) because "training = compute" is the default assumption. The correct answer is (a) or (d) depending on the specific configuration: with standard SSD and no prefetching, data loading often dominates; with 4+ GPUs over PCIe (not NVLink), gradient sync can dominate. The compute stage is rarely the bottleneck in practice.

**Instrument**: Four sliders representing stage latencies: data loading (1-100 ms), PCIe transfer (0.1-10 ms), forward+backward (5-200 ms), gradient sync (0-50 ms). Preset buttons load realistic configurations: "V100 + SSD + 1 GPU" (data-bound), "V100 + NVMe + 4 GPU PCIe" (sync-bound), "H100 + NVMe + 1 GPU" (compute-bound). A pipeline Gantt chart shows sequential execution (total = sum of stages) vs. overlapped/prefetched execution (total = max of stages + overlap overhead). A "bottleneck indicator" highlights the binding stage in red. An "accelerator bubble" percentage shows what fraction of GPU time is idle.

**mlsysim grounding**: Forward+backward latency from Engine.solve() with is_training=True. Data loading and sync latencies computed from storage bandwidth and interconnect bandwidth:
```python
profile = Engine.solve(Models.Language.GPT2, Hardware.V100,
                       batch_size=8, precision="fp16", is_training=True)
# Compute stage = profile.latency
# Data loading = batch_data_size / storage_bw
# Gradient sync = model_size_bytes / interconnect_bw * (num_gpus - 1) / num_gpus
```

**Transition to C**: "You now know how to find the bottleneck in the training pipeline. One common optimization is mixed precision -- using FP16 for forward/backward and FP32 for weight updates. But the memory savings are not what you expect."

---

### Part C -- Mixed Precision: The FP32 Master Copy Trap (~10 min)

**Concept**: Mixed-precision training uses FP16 or BF16 for forward and backward passes but retains FP32 master copies of weights and Adam optimizer state. This is required for numerical stability (FP16 gradients can underflow). The result: actual memory savings are ~1.5-1.7x, not the expected 2x. The FP32 "tail" (master weights + Adam state = 12 bytes/param) is identical in both full-precision and mixed-precision modes. Savings come entirely from activations and gradient accumulation buffers being stored in FP16.

**Prediction**: "GPT-2 (1.5B parameters) requires ~77 GB for training in full FP32 (including activations). How much memory does mixed-precision (FP16 forward/backward + FP32 master weights + FP32 Adam state) require?" Options: (a) ~38 GB (exactly half), (b) ~45 GB (~1.7x savings), (c) ~55 GB (~1.4x savings), (d) ~70 GB (almost no savings). Most students pick (a) because "half precision = half memory" seems obvious. The correct answer is (b): the FP32 master weights and Adam state persist unchanged; only activations and gradient buffers shrink to FP16.

**Instrument**: Precision toggle (Full FP32 / Mixed BF16+FP32 master / Pure BF16) and model size slider (0.1B to 13B). Side-by-side stacked bars show FP32 baseline vs. mixed precision, with the FP32 components (master weights, Adam momentum, Adam variance) highlighted with a hatched pattern to show they persist at full precision in both modes. The "actual savings" ratio is displayed as a large number (typically 1.5-1.7x, not 2x). A "what did shrink" annotation highlights the activation and gradient buffer bars that did change size.

**mlsysim grounding**:
```python
# Compare FP32 vs mixed precision
for prec in ["fp32", "fp16"]:
    profile = Engine.solve(Models.Language.GPT2, Hardware.H100,
                           batch_size=8, precision=prec, is_training=True, seq_len=1024)
    # Compare profile.memory_footprint across precision modes
```

**Transition to D**: "Mixed precision saves some memory, but not enough for large models. The next step is multi-GPU training -- but adding GPUs introduces a new cost that most students do not expect."

---

### Part D -- The Communication Tax: When More GPUs Hurt (~10 min)

**Concept**: Multi-GPU data-parallel training follows `Speedup = N / (1 + (N-1) * r)`, where r is the fraction of step time spent on gradient synchronization (AllReduce). For compute-bound workloads with fast interconnect (r = 0.05), scaling is near-linear. For bandwidth-bound workloads on slow interconnect (r = 0.40), diminishing returns set in quickly: 8 GPUs yield only 4.7x speedup, and going from 8 to 16 GPUs adds just 1.5x more. The "communication tax" is the shaded gap between ideal linear scaling and actual throughput.

**Prediction**: "You train a model on 8 GPUs where gradient synchronization takes 15% of each step (r = 0.15). What speedup do you achieve over 1 GPU?" Options: (a) ~8x (linear scaling), (b) ~6.5x (slight overhead), (c) ~4.7x (significant overhead), (d) ~2x (communication dominates). Most students pick (a) or (b), assuming that modern interconnects make communication overhead negligible. The correct answer is (c): `8 / (1 + 7 * 0.15) = 8 / 2.05 = 3.9x` (exact value depends on formula variant; the key lesson is that it is far from linear).

**Instrument**: GPU count slider (1-256, log scale) and communication fraction slider (r: 0.01-0.50). Preset buttons: "ResNet + NVLink" (r=0.05), "LLM + NVLink" (r=0.10), "LLM + PCIe" (r=0.25), "Slow network" (r=0.40). A scaling curve shows effective throughput vs. GPU count with an "ideal linear" reference line. The shaded gap between ideal and actual is the "communication tax," labeled with the wasted GPU-hours. Students watch the curve flatten as they increase either GPU count or r.

**mlsysim grounding**: Scaling formula: `Speedup(N, r) = N / (1 + (N-1) * r)`. Communication fraction r can be estimated from Engine.solve():
```python
profile = Engine.solve(model, hw, batch_size=bs, precision="fp16", is_training=True)
compute_time = profile.latency
# Gradient sync time ~ model_bytes / interconnect_bw
# r = sync_time / (compute_time + sync_time)
```
Source: textbook @sec-training-communication-tax.

---

### Synthesis (~5 min)

**Prompt**: You must train a 7B parameter model on a cluster of H100 GPUs. Using numbers from this lab, specify: (1) the precision mode (full FP32 or mixed BF16) and justify whether the model fits in 80 GB with your chosen optimizer, (2) the training pipeline stage you would optimize first and why, (3) the number of GPUs and your expected scaling efficiency, and (4) the approximate communication fraction r for your setup. Show your work with specific numbers.


# Final Lab Plans: Volume I, Labs 09--16

Generated: 2026-03-15
Status: FINAL (post-review, all fixes applied from MASTER_LAB_PARTS_REVIEW and EDTECH_REVIEW_SUMMARY)

---

## Lab 09: The Data Selection Paradox
**Story arc**: Less data trains faster -- until the cost of *choosing* that data exceeds the savings, and meanwhile your GPU starves waiting for the CPU to finish preprocessing.
**Time budget**: 52 min total -- A(12) + B(12) + C(12) + D(10) + Synthesis(6)

### Part A -- The ICR Frontier: Diminishing Returns (~12 min)

**Concept**: The Information-Compute Ratio (ICR) decays as 1/(O x D) -- most data in a large dataset contributes near-zero learning signal. A 50% coreset retains ~99% of accuracy, but the curve is logarithmic, not linear. Students carry a "more data = better model" prior that breaks against the flat tail of the ICR curve.

**Prediction**: "You have a 1M-image dataset. You want to maintain 99% of full-dataset accuracy. What fraction of the data can you discard?"
- A: 10% (keep 900K) -- conservative, assumes every image matters
- B: 30% (keep 700K) -- moderate, acknowledges some redundancy
- **C: 50% (keep 500K)** -- correct: the ICR curve flattens dramatically past the knee
- D: 80% (keep 200K) -- aggressive, underestimates the tail

**Common wrong answer**: A or B. Students overvalue individual data points because they think about datasets like code -- every line matters.

**Why wrong**: Redundancy in natural image datasets is massive. The ICR curve shows that gradient contributions from the bottom 50% of samples are near-zero after the first few epochs. The distribution of "informativeness" follows a power law, not a uniform distribution.

**Instrument**: ICR curve plot with a "dataset fraction" slider (5%--100%) and a "redundancy level" toggle (low/medium/high, simulating different dataset characteristics). A horizontal threshold line marks 99% of baseline accuracy. A "knee" marker shows where the curve transitions from steep to flat. The student's prediction is overlaid as a vertical line against the actual knee.

**mlsysim grounding**: Use Engine.solve() with MobileNetV2 on Jetson Orin NX to compute training time for the full dataset vs. the coreset. The time savings are real and grounded in hardware: T_train(500K) on edge hardware vs. T_train(1M).

**Transition to B**: "You have just discovered that half your data is dead weight. The obvious move: score every sample and keep the informative ones. But scoring has a cost. Is the selection investment worth it?"

---

### Part B -- The Selection Inequality: When Optimization Backfires (~12 min)

**Concept**: The Selection Inequality T_selection + T_train(subset) < T_train(full) must hold for coreset selection to be systems-efficient. Using a full model to score 1M images takes 2.8 hours; a proxy model takes 0.6 hours. On edge hardware with expensive scoring, the inequality can *break* -- selection costs more than just training on everything.

**Prediction**: "You use a ResNet-50 to score 1M images for coreset selection on an A100 GPU. Scoring takes 2.8 hours. Training on the full dataset takes 8.0 hours. Training on the 50% coreset takes 4.2 hours. Is the selection strategy worth it?"
- A: Yes, you save 1.0 hours (8.0 - 2.8 - 4.2)
- **B: Yes, but barely -- you save only 1.0 hours (12.5% time reduction)**
- C: No, you lose 0.2 hours (scoring + subset training > full training)
- D: It depends on the proxy model cost

**Common wrong answer**: Students pick A without computing the actual savings, or pick D because they sense a catch but cannot articulate it. The real lesson is that the margin is thin even in the best case.

**Why wrong**: Students mentally subtract the coreset training time from full training time and forget to add back the scoring cost. The Selection Inequality forces them to account for the full pipeline.

**Instrument**: Waterfall bar chart showing T_selection + T_train(subset) vs. T_train(full). Controls: coreset fraction slider (10%--90%), scoring model dropdown (Full ResNet-50 / Proxy MobileNetV2 / Cached embeddings), deployment context toggle (Cloud A100 / Edge Jetson). On Edge with full-model scoring, the inequality breaks -- the failure state turns the bar red and displays "SELECTION INEQUALITY VIOLATED."

**mlsysim grounding**: Engine.solve(ResNet50, A100) for T_score_full, Engine.solve(MobileNetV2, A100) for T_score_proxy, Engine.solve(ResNet50, JetsonOrinNX) for the edge case that breaks the inequality. All times are grounded in real hardware specs.

**Transition to C**: "You now know that data selection saves time only when the scoring cost is cheap relative to training savings. But there is a more fundamental bottleneck hiding in every training pipeline -- one that has nothing to do with which data you select."

---

### Part C -- The Preprocessing Tax (~12 min)

**Concept**: CPU-side data augmentation and preprocessing is often the true training bottleneck, not GPU compute. A training pipeline that applies RandAugment + normalization + resizing on CPU can starve an A100 that finishes its forward-backward pass in 12 ms while preprocessing takes 45 ms per batch. The GPU sits idle 73% of the time. This is the data pipeline problem from the Data Engineering chapter manifested specifically in the data selection context: the augmentation strategies that prevent overfitting on small coresets (Part A) are themselves a systems bottleneck.

**Prediction**: "Your coreset training pipeline applies RandAugment (5 transforms) + resize + normalize on 8 CPU workers, feeding an A100 GPU. GPU forward-backward takes 12 ms per batch. What fraction of total step time is GPU compute?"
- A: ~80% (GPU dominates, preprocessing is fast)
- B: ~50% (roughly balanced)
- **C: ~25% (GPU waits for CPU most of the time)**
- D: ~10% (GPU is almost entirely idle)

**Common wrong answer**: A. Students assume GPUs are the bottleneck because they are the expensive component. They do not think about CPU preprocessing as a pipeline stage with its own latency.

**Why wrong**: Data augmentation is CPU-bound sequential work. RandAugment applies 5 random transforms per image, each involving pixel-level operations. With 8 workers, the pipeline throughput is 8 x (1000ms / 45ms) = ~178 batches/sec from CPU, but the GPU can consume 1000ms / 12ms = ~83 batches/sec. Wait -- in this case the GPU is actually the bottleneck? No: the 45ms is per-batch with 8 workers, meaning effective CPU throughput is 8/45ms = 177 images/sec, but with heavy augmentation the per-worker time can be 200ms+, making effective throughput 8/200ms = 40 batches/sec vs. GPU capacity of 83 batches/sec. The GPU idle fraction depends on the specific augmentation pipeline.

**Instrument**: Pipeline Gantt chart showing CPU preprocessing (orange) and GPU compute (blue) stages per step. Controls: number of CPU workers slider (1--16), augmentation complexity dropdown (None / Basic flip+crop / RandAugment-5 / RandAugment-10 + MixUp), model dropdown (MobileNetV2 / ResNet-50). A GPU utilization gauge shows the fraction of time the GPU is actually computing. At high augmentation complexity with few workers, the gauge drops below 30%.

**mlsysim grounding**: Engine.solve() provides the GPU forward-backward time for each model. CPU preprocessing time is modeled as T_preprocess = N_transforms x T_per_transform / N_workers, using documented per-transform costs from the chapter (resize: 2ms, flip: 0.5ms, RandAugment transform: 8ms each). The ratio T_gpu / (T_gpu + max(0, T_preprocess - T_gpu)) gives effective utilization when pipelining is imperfect.

**Transition to D**: "Parts A through C revealed three costs hidden inside 'data selection': the diminishing returns of data, the cost of scoring, and the preprocessing tax. But all of these operate within a larger question: given a fixed compute budget, should you spend it on more data or a bigger model?"

---

### Part D -- The Cost-Optimal Frontier: Data-Starved vs. Compute-Starved (~10 min)

**Concept**: The compute-optimal frontier (Chinchilla scaling) determines whether a training run is data-starved (more data would help more than more compute) or compute-starved (more compute would help more than more data). Most teams misdiagnose their position because they think about data and compute as independent resources rather than as coupled terms in a scaling law.

**Prediction**: "You have a fixed compute budget of 10^21 FLOPs. Which configuration achieves lower loss?"
- A: 10B parameter model trained on 200B tokens
- **B: 3B parameter model trained on 660B tokens**
- C: 30B parameter model trained on 66B tokens
- D: All achieve roughly the same loss

**Common wrong answer**: A or C. Students anchor on model size ("bigger model = better") or on round numbers. They do not expect that the optimal point requires 3.3x more data tokens than model parameters.

**Why wrong**: The Chinchilla scaling law shows D proportional to N^0.74, meaning data must scale nearly linearly with model size. Most teams over-allocate to model size and under-allocate to data, placing themselves in the compute-starved regime where additional FLOPs go to waste.

**Instrument**: 2D scatter plot of dataset size (tokens) vs. model size (parameters), with IsoFLOP contour curves. The student's configuration appears as a draggable dot. The Chinchilla-optimal line cuts through the landscape. Regions above the line are labeled "Data-starved" (more data would help), regions below are labeled "Compute-starved" (more compute would help). A loss readout shows the current configuration's predicted loss.

**mlsysim grounding**: The scaling law L(N, D) = A/N^alpha + B/D^beta + E (with Chinchilla parameters) is computed directly. Engine.solve() is not directly applicable here, but the FLOP budget is grounded in real hardware: 10^21 FLOPs = approximately 1,000 A100-hours at 50% MFU (computed from A100 peak FLOPS from the hardware registry).

---

### Synthesis (~6 min)

**Mission**: You manage a computer vision pipeline for a wildlife monitoring system deployed on Jetson Orin NX edge devices. You have 2M images collected over 3 years, a training budget of 500 A100-hours, and a target model (MobileNetV2 with RandAugment).

Write a data selection strategy that addresses:
1. What coreset fraction to use (reference the ICR curve from Part A)
2. Which scoring model to use and whether the Selection Inequality holds on your hardware (reference Part B)
3. How many CPU preprocessing workers you need to keep the GPU fed (reference Part C)
4. Whether your compute budget is better spent on more data or a larger model (reference Part D)

Justify each choice with a specific number from the lab.

**Design Ledger entry**: Record which prediction you got most wrong and what mental model it revealed.

---
---

## Lab 10: The Compression Paradox
**Story arc**: Compression looks like free performance -- until you discover that the hardware ignores your zeros, the accuracy cliff is vertical, and the only reliable shortcut is stealing knowledge from a model too large to deploy.
**Time budget**: 56 min total -- A(12) + B(12) + C(12) + D(8) + E(12) + Synthesis(6)

### Part A -- The Quantization Free Lunch (~12 min)

**Concept**: Reducing precision from FP32 to INT8 costs under 1% accuracy (the "Free Lunch Zone"), then accuracy collapses catastrophically at 3--4 bits (the "Quantization Cliff"). The curve is flat-then-vertical, not gradual. Students expect a smooth accuracy-compression trade-off.

**Prediction**: "You quantize ResNet-50 from FP32 to INT8 (4x compression). How much accuracy do you lose on ImageNet?"
- A: ~5% (noticeable but tolerable)
- B: ~2% (moderate degradation)
- **C: < 1% (essentially free)**
- D: ~0.1% (unmeasurable)

**Common wrong answer**: A or B. Students assume compression must cost something proportional to the compression ratio. "4x smaller must mean significant quality loss."

**Why wrong**: The representational capacity of INT8 (256 discrete levels) is far more than sufficient for the weight distributions of most trained neural networks, which cluster tightly around zero. The information content of the weights is much lower than the precision used to store them.

**Instrument**: Precision selector (FP32 / FP16 / INT8 / INT4 / INT2) with model dropdown (ResNet-50 / MobileNetV2 / BERT-Base). Table and bar chart showing accuracy, model size, latency, and energy for each precision. The INT8 row glows green as the practical sweet spot. Below INT4, accuracy collapses -- the bar turns red and a "QUANTIZATION CLIFF" banner fires.

**mlsysim grounding**: Engine.solve(ResNet50, H100, precision="fp32") vs. Engine.solve(ResNet50, H100, precision="int8") for latency comparison. Memory footprint from the engine's weight_bytes calculation at each precision. The accuracy curve uses documented values from the chapter (@tbl-quantization-accuracy).

**Transition to B**: "Quantization gave you a free 4x compression. Pruning promises even more -- removing 90% of weights sounds like a 10x speedup. But there is a hardware trap waiting."

---

### Part B -- The Pruning Hardware Trap (~12 min)

**Concept**: Unstructured pruning at 90% sparsity gives zero latency speedup on standard GPU kernels because dense GEMM iterates over every element including zeros. The hardware cannot skip sparse multiplications without specialized support. Structured pruning removes entire channels, yielding real speedup, but destroys accuracy faster.

**Prediction**: "You prune ResNet-50 to 90% sparsity (remove 90% of weights, set to zero) using unstructured magnitude pruning. What inference speedup do you get on an H100 GPU?"
- A: ~10x (removed 90% of work)
- B: ~5x (some overhead from sparse format)
- C: ~2x (significant overhead)
- **D: ~1.0x (no speedup at all)**

**Common wrong answer**: A. This is the strongest wrong-prediction moment in the entire lab suite. Students reason linearly: "90% fewer multiplications should mean 90% less time." It is an entirely reasonable assumption that is entirely wrong on standard hardware.

**Why wrong**: GPU GEMM kernels use dense matrix multiplication. Every element of the weight matrix is loaded from HBM and multiplied, including the zeros. The zeros save no memory bandwidth (the matrix is the same size) and no compute (the multiply still executes). Without hardware support for sparse matrix formats (like NVIDIA's 2:4 structured sparsity), unstructured pruning is invisible to the hardware.

**Instrument**: Sparsity slider (0%--95%) with pruning type toggle (Unstructured / Structured / 2:4 Structured). Two charts: (1) Speedup vs. sparsity -- flat line at 1.0x for unstructured, rising curve for structured, step function for 2:4; (2) Accuracy vs. sparsity -- gradual decline for unstructured, steeper decline for structured. Pushing structured pruning past 85% triggers an accuracy collapse failure state.

**mlsysim grounding**: Engine.solve(ResNet50, H100, precision="fp16") gives the baseline latency. Structured pruning at X% reduces the effective parameter count by X%, which reduces inference_flops proportionally -- Engine.solve() with a modified workload shows real speedup. Unstructured pruning does not change the workload spec at all because the hardware sees the same dense matrix.

**Transition to C**: "Quantization compresses 4x for free. Structured pruning buys speedup but costs accuracy. What if you need to fit an 8B-parameter LLM on a phone with 4 GB of RAM? No single technique spans that gap."

---

### Part C -- The Compression Pareto Frontier (~12 min)

**Concept**: Deploying an 8B-parameter LLM across three mobile memory tiers (8 GB / 4 GB / 2 GB) requires composing INT8, INT4, and structured pruning along a Pareto frontier. No single technique spans all tiers. The Pareto frontier reveals that some compression combinations are dominated (worse accuracy AND worse compression than an alternative).

**Prediction**: "You need to deploy Llama-3 8B on a device with 4 GB RAM. Which compression strategy fits?"
- A: FP16 (16 GB) -- does not fit
- B: INT8 (8 GB) -- does not fit
- **C: INT4 (4 GB) -- fits, with measurable accuracy loss**
- D: INT4 + 50% structured pruning (2 GB) -- fits, but accuracy may be unacceptable

**Common wrong answer**: B. Students underestimate the memory footprint of INT8 for an 8B model (8B x 1 byte = 8 GB, which exceeds the 4 GB budget when you include KV cache and runtime overhead).

**Why wrong**: Memory accounting requires including not just weights but also the KV cache, activation memory, and runtime overhead. INT8 weights alone consume 8 GB, leaving zero room for anything else.

**Instrument**: Scatter plot showing accuracy vs. model size for all compression configurations. Per-tier memory budget lines at 8 GB, 4 GB, and 2 GB. Students select a compression strategy per tier from dropdowns. Pareto-optimal configurations are highlighted; dominated configurations are grayed out. Selecting a configuration that exceeds a tier's budget triggers an OOM failure state.

**mlsysim grounding**: Engine.solve(Llama3_8B, iPhone15Pro, precision="fp16") triggers OOM. Engine.solve(Llama3_8B, iPhone15Pro, precision="int4") shows feasibility. The memory_footprint field from the PerformanceProfile grounds every point on the Pareto frontier.

**Transition to D**: "Compression reduces model size. But there is a second dividend that matters more on battery devices: every bit you do not move from memory saves energy."

---

### Part D -- The Energy Dividend: Bits are Joules (~8 min)

**Concept**: Moving data costs 40,000x more energy than computing it (DRAM read: 640 pJ vs. integer add: 0.015 pJ). INT8 inference uses up to 20x less energy than FP32 because the dominant energy cost is data movement, not arithmetic. On a battery device, this is the difference between 1 hour and 20 hours of operation.

**Prediction**: "For a MobileNetV2 inference on a mobile device, what fraction of total energy is spent on memory access vs. computation?"
- A: 50/50 -- balanced between memory and compute
- B: 70/30 -- memory dominates somewhat
- C: 90/10 -- memory dominates strongly
- **D: 99/1 -- memory access is essentially all of the energy**

**Common wrong answer**: A or B. Students think of GPUs as "compute engines" and assume computation is the primary energy consumer.

**Why wrong**: The energy hierarchy is: DRAM read (640 pJ) >> SRAM read (5 pJ) >> FP32 multiply (3.7 pJ) >> INT8 multiply (0.2 pJ) >> integer add (0.015 pJ). For a network doing millions of multiplies, each requiring a weight loaded from DRAM, memory access energy dominates by 100x or more.

**Instrument**: Energy breakdown stacked bar showing DRAM access energy vs. compute energy for each precision format (FP32 / FP16 / INT8 / INT4). Deployment target toggle (Cloud H100 / Mobile iPhone / IoT Cortex-M). A battery life estimate shows hours of continuous inference. The visual starkness of the 40,000x DRAM-to-compute ratio makes the "Memory Wall" visceral.

**mlsysim grounding**: Engine.solve(MobileNetV2, iPhone15Pro, precision="fp32").energy vs. Engine.solve(MobileNetV2, iPhone15Pro, precision="int8").energy provides the energy comparison. The per-operation energy constants are from the chapter (@tbl-energy-per-operation).

**Transition to E**: "Quantization and pruning modify the original model. But what if you could train a *different*, smaller model that inherits the knowledge of a large one? That is knowledge distillation -- and it unlocks deployment scenarios that compression alone cannot reach."

---

### Part E -- Dark Knowledge Transfer (~12 min)

**Concept**: Knowledge distillation trains a small "student" model to mimic the soft probability outputs of a large "teacher" model. The student learns not just the correct class but the teacher's uncertainty structure ("dark knowledge") -- which wrong classes are similar, how confident to be. A distilled MobileNetV2 student can match 95% of a ResNet-50 teacher's accuracy at 1/10th the compute, enabling deployment on edge hardware where the teacher cannot physically run.

**Prediction**: "A ResNet-50 teacher achieves 76.1% ImageNet accuracy. You distill into a MobileNetV2 student. What accuracy does the student achieve?"
- A: ~65% (significant knowledge loss from compression)
- B: ~70% (moderate knowledge transfer)
- **C: ~73% (retains 95% of teacher accuracy)**
- D: ~76% (perfect knowledge transfer)

**Common wrong answer**: A or B. Students expect that a 10x smaller model must lose proportionally more accuracy. They do not realize that the soft labels from the teacher carry far more information than hard labels alone.

**Why wrong**: Hard labels (one-hot vectors) discard the teacher's uncertainty. Soft labels with temperature scaling preserve the relative probabilities across all classes -- a "dark" signal that tells the student which classes the teacher found confusable. This information-dense supervision signal closes most of the gap between student and teacher capacity.

**Instrument**: Teacher-student comparison dashboard. Teacher model dropdown (ResNet-50 / BERT-Large / Llama-3 70B), student model dropdown (MobileNetV2 / DistilBERT / Llama-3 8B). Temperature slider (1.0--20.0). Accuracy comparison bar chart showing: student trained on hard labels, student trained with distillation, teacher. A deployment feasibility check shows which hardware the teacher fits on vs. the student.

**mlsysim grounding**: Engine.solve(ResNet50, JetsonOrinNX, precision="fp16") shows the teacher is feasible on edge but slow. Engine.solve(MobileNetV2, JetsonOrinNX, precision="int8") shows the distilled student runs 5x faster and fits comfortably. Engine.solve(ResNet50, CortexM7) triggers OOM -- the teacher *cannot run at all* on TinyML, but the distilled student can. This is the deployment unlock that compression alone cannot achieve.

---

### Synthesis (~6 min)

**Mission**: You must deploy a text classification model across three tiers: Cloud (H100), Mobile (iPhone 15 Pro), and IoT (Cortex-M7). Your teacher model is BERT-Large (340M parameters, 93% accuracy).

Design a compression strategy for each tier:
1. Which technique(s) to apply (quantization, pruning, distillation, or combination)
2. Expected accuracy at each tier
3. Whether the Energy Dividend (Part D) or the Deployment Unlock (Part E) is the primary motivation
4. The specific mlsysim feasibility check that validates your choice

**Design Ledger entry**: Record the "Pruning Hardware Trap" prediction. How far off were you?

---
---

## Lab 11: The Hardware Roofline
**Story arc**: Your code is not broken -- your hardware has a ceiling, and that ceiling changes shape depending on what chip you are running on, what operation you are executing, and whether you fuse your kernels.
**Time budget**: 52 min total -- A(12) + B(12) + C(10) + D(10) + E(8) + Synthesis(6)

### Part A -- The Memory Wall (Roofline Diagnosis) (~12 min)

**Concept**: A GEMM kernel at N=512 achieves only 31.5% MFU on the H100 -- not because the code is broken, but because arithmetic intensity (170 FLOP/byte) falls below the ridge point (295 FLOP/byte). The kernel is correctly hitting the memory bandwidth ceiling. The Roofline model is a diagnostic tool, not a performance target.

**Prediction**: "A GEMM kernel (N=512, FP16) on an H100 achieves 31.5% of peak TFLOPS. What is the problem?"
- A: The code has a bug -- it should be 90%+
- B: The GPU is thermally throttling
- **C: The kernel is memory-bandwidth-bound -- it is hitting the Roofline ceiling correctly**
- D: FP16 precision reduces peak throughput

**Common wrong answer**: A. Students equate low utilization with bad code. They do not yet have the mental model that the hardware has two ceilings (compute and bandwidth) and the lower one wins.

**Instrument**: Log-log Roofline plot with a matrix dimension slider (N=128 to 8192) and precision toggle (FP32 / FP16 / INT8). The operation point slides along the bandwidth slope as N increases, crossing the ridge into the compute-bound regime. Metric cards display arithmetic intensity, MFU, and regime classification ("Memory-bound" / "Compute-bound") live.

**mlsysim grounding**: Engine.solve() computes arithmetic_intensity, mfu, and bottleneck classification directly. The ridge point is computed from hardware specs: H100_FLOPS_FP16_TENSOR / H100_MEM_BW.

**Transition to B**: "You now know that a single GEMM kernel can be diagnosed with the Roofline. But real models contain dozens of different operations. What happens when you mix GEMM with LayerNorm and Softmax?"

---

### Part B -- Kernel Fusion: The Elementwise Trap (~12 min)

**Concept**: LLM inference mixes kernels spanning 3 orders of magnitude in arithmetic intensity. GEMM can become compute-bound at large batch, but LayerNorm (AI ~ 0.83 FLOP/byte) and Softmax are permanently memory-bound. These operations dominate inference time not because they do a lot of work, but because each one requires a full round-trip to HBM. Kernel fusion eliminates intermediate HBM writes, collapsing 3 memory-bound kernels into 1.

**Prediction**: "You fuse LayerNorm + Dropout + ReLU into a single kernel, eliminating 2 intermediate HBM writes. What speedup do you get for this fused sequence?"
- A: ~1.3x (modest improvement)
- B: ~2x (eliminated half the work)
- **C: ~3--5x (eliminated most of the memory traffic)**
- D: ~10x (eliminated nearly all overhead)

**Common wrong answer**: A or B. Students think of fusion as eliminating "overhead" (small), not as eliminating *memory traffic* (which is the dominant cost for elementwise ops).

**Instrument**: Multi-operation Roofline showing three markers (GEMM, LayerNorm, Softmax) at current batch size. Toggle between "Eager" (each op is a separate kernel with separate HBM read/write) and "Fused" (single kernel, single HBM round-trip). Timeline visualization shows GPU compute bars vs. memory stalls. Batch size slider and sequence length slider. Increasing batch too far triggers an OOM failure state when KV cache exceeds device RAM.

**mlsysim grounding**: Engine.solve() gives total latency, which is dominated by the memory term at low AI. The per-kernel decomposition models each operation's AI separately and computes latency as sum of per-kernel max(compute, memory) in eager mode vs. combined in fused mode.

**Transition to C**: "The Roofline plot and fusion strategy both depend on the specific hardware. What happens when you move the same kernel from cloud to edge?"

---

### Part C -- The Hardware Balance Shift (~10 min)

**Concept**: The same GEMM kernel can be compute-bound on an edge device (Jetson Orin NX ridge point ~118 FLOP/byte) but memory-bound on the H100 (ridge point ~295 FLOP/byte). More powerful accelerators are *paradoxically harder to saturate*, because compute grows faster than bandwidth across generations. This means that code optimized for one hardware target may have a completely different bottleneck on another.

**Prediction**: "A GEMM at N=1024 is compute-bound on a Jetson Orin NX. Is it compute-bound on an H100?"
- A: Yes -- if it is compute-bound on weaker hardware, it must be compute-bound on stronger hardware
- **B: No -- the H100 has a higher ridge point, so the same operation is now memory-bound**
- C: It depends on the precision
- D: It is memory-bound on both

**Common wrong answer**: A. The intuition "more powerful = more headroom" is wrong because the ridge point also rises. Students need to see the Roofline redraw with different ceilings.

**Instrument**: Hardware toggle (Cloud H100 / Edge Jetson Orin NX / Mobile A17 Pro). The Roofline redraws with different compute ceilings and bandwidth slopes, shifting the ridge point. A fixed GEMM operation point stays at the same AI but changes regime classification. Side-by-side metric cards show MFU on each platform for the same operation.

**mlsysim grounding**: Engine.solve(workload, H100) vs. Engine.solve(workload, JetsonOrinNX) vs. Engine.solve(workload, iPhone15Pro). The bottleneck field directly shows the regime shift. Ridge points computed from each hardware's peak_flops / memory.bandwidth.

**Transition to D**: "You can now diagnose whether an operation is compute-bound or memory-bound on any hardware. But there is a cost your Roofline plot does not show: energy."

---

### Part D -- The Energy Roofline (~10 min)

**Concept**: Energy efficiency has its own Roofline. Operations in the memory-bound regime waste energy on data movement (640 pJ per DRAM access) rather than useful computation (3.7 pJ per FP32 multiply). The energy-optimal operating point is deep in the compute-bound regime, where most Joules go to arithmetic. This is why hardware architects build larger caches and higher-bandwidth memory: to push more operations into the energy-efficient compute-bound regime.

**Prediction**: "A memory-bound operation (AI = 10 FLOP/byte) and a compute-bound operation (AI = 500 FLOP/byte) both perform the same total FLOPs. Which uses more energy?"
- **A: The memory-bound operation uses ~10x more energy**
- B: They use roughly the same energy (same FLOPs = same work)
- C: The compute-bound operation uses more (higher utilization = more power)
- D: It depends on the clock frequency

**Common wrong answer**: B. "Same FLOPs = same energy" is a deeply held intuition. Students do not account for the energy cost of the *data movement* required to feed those FLOPs.

**Instrument**: Energy Roofline plot (energy per FLOP vs. arithmetic intensity) with the same hardware toggle from Part C. The energy floor is in the compute-bound regime; the energy penalty rises linearly in the memory-bound regime. Operations from the previous parts are plotted on this new Roofline. A total energy gauge shows Joules per inference.

**mlsysim grounding**: Engine.solve().energy provides the total energy. The energy decomposition uses hardware TDP and the compute/memory time split from the latency breakdown.

**Transition to E**: "The Roofline tells you where you are. Fusion moves you right along the AI axis. But there is one more trick: tiling -- computing on data that fits in cache before it escapes to HBM."

---

### Part E -- The Tiling Dividend (~8 min)

**Concept**: Matrix multiplication on large matrices that exceed L2 cache forces repeated HBM accesses for the same data. Tiling (blocking) the computation into cache-sized chunks keeps data in fast SRAM, effectively increasing arithmetic intensity by reducing redundant memory traffic. FlashAttention is the canonical example: by tiling the attention computation to fit in SRAM, it achieves 2--4x speedup over the standard implementation.

**Prediction**: "FlashAttention tiles the attention computation to fit in GPU SRAM. Compared to standard attention, how much faster is it at sequence length 4096?"
- A: ~1.2x (minor improvement)
- B: ~1.5x (moderate)
- **C: ~2--4x (significant, from eliminating HBM round-trips)**
- D: ~10x (transformative)

**Common wrong answer**: A. Students who learned about fusion in Part B may think tiling is just another incremental optimization. They underestimate how many redundant HBM reads standard attention performs.

**Instrument**: Tile size slider (32 to 2048 elements) and sequence length slider (512 to 16384). A memory traffic counter shows bytes read from HBM for standard vs. tiled attention. The ratio is displayed prominently. A simplified Roofline shows the effective arithmetic intensity increasing as tile size grows (because fewer bytes are transferred for the same FLOPs).

**mlsysim grounding**: Engine.solve() provides the baseline. The tiling model computes effective AI as total_flops / (bytes_accessed_with_tiling), where bytes_accessed scales with tile_size relative to SRAM capacity. The speedup is the ratio of memory-bound latency at the two AI values.

---

### Synthesis (~6 min)

**Mission**: You must deploy a Transformer inference service on both H100 (cloud) and Jetson Orin NX (edge). For each target, specify:
1. Which operations are memory-bound vs. compute-bound (Roofline diagnosis, Part A/C)
2. Which kernels to fuse (Part B)
3. Whether FlashAttention tiling helps (Part E)
4. The energy cost per inference on each platform (Part D)

Use specific ridge points and arithmetic intensities from the lab.

**Design Ledger entry**: Record your Part A prediction (GEMM utilization). Most students think low utilization = broken code.

---
---

## Lab 12: The Benchmarking Trap
**Story arc**: Vendor benchmarks are designed to make hardware look good. Your job is to make hardware look *honest* -- and the gap between the two is where millions of dollars disappear.
**Time budget**: 52 min total -- A(12) + B(12) + C(10) + D(12) + Synthesis(6)

### Part A -- The Amdahl Ceiling (~12 min)

**Concept**: A 10x inference speedup with 45% non-inference overhead yields only 2.0x end-to-end improvement. Amdahl's Law caps system speedup at 1/(1-f) where f is the non-optimized fraction. The remaining 8x of hardware investment is wasted on a bottleneck that has already moved.

**Prediction**: "You replace your inference GPU with one that is 10x faster. Your pipeline is 45% preprocessing + 55% inference. What is your end-to-end speedup?"
- A: ~10x (the new GPU is 10x faster)
- B: ~5x (about half the pipeline speeds up)
- **C: ~2.0x (Amdahl's Law caps the gain)**
- D: ~1.5x (even worse than expected)

**Common wrong answer**: A or B. Students think component speedup translates to system speedup. They have not internalized that the non-accelerated fraction becomes the new bottleneck.

**Why wrong**: Amdahl's Law: Speedup = 1 / (0.45 + 0.55/10) = 1 / 0.505 = 1.98x. The 45% preprocessing fraction is now 89% of total time. The $30K GPU upgrade delivered a 2x improvement, not 10x.

**Instrument**: Before/after waterfall bar chart showing preprocessing (fixed) + inference (accelerated). Sliders: inference speedup (1x--100x), non-inference fraction (0.05--0.80). An Amdahl saturation curve shows speedup approaching the asymptote 1/f. A "wasted speedup" metric shows the gap between component improvement and system improvement. A dollar-per-improvement gauge shows cost efficiency declining as speedup increases.

**mlsysim grounding**: Engine.solve(ResNet50, H100) vs. Engine.solve(ResNet50, A100) provides the actual speedup ratio for the inference component. The non-inference fraction is parameterized. Total system time = T_preprocess + T_inference / speedup_factor.

**Transition to B**: "Amdahl showed that one fast component does not make a fast system. But even the fast component may be lying about its speed. Vendor benchmarks measure *burst* performance. Production runs are *sustained*."

---

### Part B -- Peak vs. Sustained: The Thermal Cliff (~12 min)

**Concept**: A vendor advertises 30 FPS for an edge chip. After 5 minutes of continuous inference in a fanless enclosure, thermal throttling halves throughput to 15 FPS. Vendor benchmarks are burst measurements; production runs are sustained. The gap between peak and sustained performance can be 2x or more.

**Prediction**: "An edge device benchmarks at 30 FPS in a 1-minute vendor test. What sustained FPS do you get after 10 minutes of continuous inference in a fanless enclosure at 35C ambient?"
- A: ~28 FPS (minor degradation)
- B: ~24 FPS (some thermal throttling)
- **C: ~15 FPS (thermal throttle halves performance)**
- D: ~8 FPS (severe throttling)

**Common wrong answer**: A. Students trust vendor numbers because they assume benchmarks represent production conditions. They do not account for thermal accumulation over time.

**Why wrong**: Thermal energy accumulates as Q = P x t. In a fanless enclosure, convective cooling is limited. Junction temperature rises linearly until hitting the thermal throttle threshold (typically 85--100C), at which point the chip reduces clock frequency to stay within its thermal envelope.

**Instrument**: Time scrubber (0--10 minutes). A piecewise thermal model: Phase 1 (0 to T_throttle): FPS constant at peak, junction temperature rising linearly as T_junction(t) = T_ambient + (TDP / thermal_conductance) x (1 - e^(-t/tau)). Phase 2 (T_throttle onward): FPS drops to TDP_throttled / TDP x peak_FPS. Controls: ambient temperature slider (20--45C), cooling type toggle (active fan / passive heatsink / fanless), TDP slider.

**mlsysim grounding**: Engine.solve(MobileNetV2, JetsonOrinNX).latency provides the per-inference time at peak. The thermal model uses hardware.tdp from the registry. The piecewise model: T_junction = T_ambient + (TDP_watts / G_thermal) x (1 - exp(-t / RC_thermal)), where G_thermal and RC_thermal are parameters for each cooling type.

**Transition to C**: "Vendor benchmarks lie about sustained performance. They also lie about *which metric matters*. A chip that wins on accuracy may fail on latency, and a chip that wins on throughput may fail on power."

---

### Part C -- The Multi-Metric Trap (~10 min)

**Concept**: The model configuration with the best single-metric score (94% accuracy, 1200 QPS) violates latency and power SLOs. Only the balanced configuration (91% accuracy, 95 ms p99, 600 QPS, 4.5 W) passes all four deployment gates simultaneously. Optimizing a single metric while ignoring the others is the most common benchmarking mistake.

**Prediction**: "You have four model configurations. Configuration A has the highest accuracy (94%). Is it deployable?"
- A: Yes -- highest accuracy is always the best choice
- B: Probably -- the other metrics are secondary
- **C: No -- it violates the latency SLO (p99 > 100 ms)**
- D: No -- it violates the power budget (> 5 W)

**Common wrong answer**: A. In academic settings, accuracy is the only metric. Students have not yet internalized that production deployment has simultaneous constraints on multiple axes.

**Instrument**: Batch size slider, precision dropdown, model variant dropdown. A radar chart with four axes (accuracy, latency, throughput, power) overlays SLO threshold rings. An SLO compliance table shows pass/fail per metric with green/red indicators. Any violated SLO triggers a "DEPLOYMENT BLOCKED" failure state with a red banner naming the violated constraints.

**mlsysim grounding**: Engine.solve() at each configuration provides latency, throughput (from batch_size / latency), and energy (proxy for power). Accuracy values come from the chapter's documented configurations. The SLO thresholds are parameters: p99_latency < 100ms, power < 5W, accuracy > 90%, throughput > 500 QPS.

**Transition to D**: "You now know that single-metric benchmarks are misleading and that thermal throttling hides in sustained runs. The final trap: average latency hides catastrophic tail behavior."

---

### Part D -- The Tail Latency Diagnostic (~12 min)

**Concept**: A system with 50 ms average latency but 500 ms p99 violates a 200 ms SLO for 1% of requests. At 10K requests/sec, that is 100 failures per second. Average latency is not just insufficient -- it is *actively misleading* because it hides the distribution shape. The latency distribution for real inference systems follows a log-normal distribution with a heavy right tail.

**Prediction**: "Your inference service reports 50 ms average latency. Your SLO is 200 ms p99. Is the SLO satisfied?"
- A: Yes -- 200 ms is 4x the average, which provides plenty of headroom
- B: Probably -- p99 is usually ~2x the average
- **C: No -- the p99 is ~500 ms due to the heavy tail of the latency distribution**
- D: Cannot determine without more information

**Common wrong answer**: A or B. Students assume latency distributions are symmetric (like a Gaussian) where p99 is close to the mean. Real inference latency distributions are heavy-tailed.

**Why wrong**: Inference latency has multiple sources of variability: garbage collection pauses, memory allocation, cache misses, OS scheduling, thermal throttling. These create a log-normal or worse distribution where p99 can be 5--10x the median.

**Instrument**: Latency distribution histogram generated from Engine.solve() base latency + log-normal noise (sigma parameter from slider). Mean, p50, p95, p99, and p99.9 vertical lines on the histogram. SLO threshold line (adjustable). A violation counter shows "X% of requests exceed SLO." Toggle between training mode (shows throughput + time-to-accuracy + scaling efficiency) and inference mode (shows the latency distribution).

**mlsysim grounding**: Engine.solve(ResNet50, H100, batch_size=B) provides the deterministic base latency. The lab adds log-normal noise: latency_sample = base_latency x exp(N(0, sigma^2)), where sigma controls tail heaviness (slider: 0.1 to 1.0). This generates a realistic latency distribution. The p99 is computed from the distribution.

---

### Synthesis (~6 min)

**Mission**: A vendor claims their new edge chip delivers "30 FPS, 94% accuracy, 50 ms average latency" for your computer vision pipeline. You must write a 3-point rebuttal explaining why this benchmark is insufficient for your production deployment:

1. What is the *sustained* FPS after thermal steady-state? (Part B)
2. Does the 94% accuracy configuration pass all four SLO gates? (Part C)
3. What is the *p99* latency, and does it satisfy your 200 ms SLO? (Part D)
4. What is the *end-to-end* speedup given your preprocessing overhead? (Part A)

This is the "debunk a vendor claim" exercise. It is the most directly career-applicable synthesis in the lab series.

**Design Ledger entry**: Record your Amdahl prediction. The gap between component speedup and system speedup is the single most common mistake in systems engineering.

---
---

## Lab 13: The Tail Latency Trap
**Story arc**: Your serving system looks healthy at 50% utilization, manageable at 70%, and is on fire at 80% -- and the fire is invisible to every metric except the one you are not measuring.
**Time budget**: 50 min total -- A(12) + B(12) + C(12) + D(8) + Synthesis(6)

### Part A -- The Tail Latency Explosion (~12 min)

**Concept**: The M/M/1 queuing model predicts that P99 latency diverges nonlinearly from mean latency as server utilization increases. At 80% utilization, P99 is 23x the service time while the mean is only 5x. A system reporting "healthy 25 ms average latency" is simultaneously delivering 115 ms tail latency that violates a 50 ms SLO for 1 in 100 users.

**Prediction**: "Your ResNet-50 server has a 5 ms service time and runs at 80% utilization. What P99 latency do your slowest 1-in-100 users experience?"
- A: ~10 ms (2x service time)
- B: ~25 ms (5x service time, matches the mean)
- C: ~50 ms (10x service time)
- **D: ~115 ms (23x service time)**

**Common wrong answer**: B. Students confuse mean latency (25 ms at rho=0.8) with P99 latency. They do not expect the ln(100) = 4.6 multiplier from the exponential tail.

**Instrument**: Utilization slider (rho, 0.10--0.95). Live P99 latency histogram with mean (blue dashed) and P99 (red solid) vertical markers diverging as utilization increases. SLO threshold line (green dashed, adjustable). Service time slider (1--20 ms). Metric row: mean latency, P99 latency, P99/mean ratio (always 4.6x for M/M/1, but absolute values change dramatically).

**mlsysim grounding**: Engine.solve(ResNet50, H100, batch_size=1).latency provides the service time. The queuing model: W_mean = service_time / (1 - rho), W_p99 = 4.6 x service_time / (1 - rho). Source: @eq-mm1-wait and @eq-p99-latency in the chapter.

**Transition to B**: "You now know that utilization above 70% is dangerous for tail latency. The natural response: batch requests to improve throughput. But batching has its own hidden tax."

---

### Part B -- The Batching Tax (~12 min)

**Concept**: Larger batch sizes improve GPU throughput (batch-32 achieves 6.4x over batch-1) but impose a formation delay tax of (B-1)/(2 x lambda) that can exceed the SLO before inference even begins. The latency-throughput Pareto frontier has a sharp "knee" where throughput gains plateau and latency explodes.

**Prediction**: "You increase batch size from 1 to 32 to improve throughput. Your arrival rate is 500 QPS and your SLO is 50 ms. What happens?"
- A: Throughput improves 6.4x and latency stays under SLO
- B: Throughput improves 6.4x but latency slightly exceeds SLO
- **C: Throughput improves but the batch formation delay alone (31 ms) consumes 62% of the SLO budget before inference starts**
- D: The system crashes from memory overflow

**Common wrong answer**: A. Students think of batching as pure throughput gain without a latency cost. They do not account for the time requests spend waiting for the batch to fill.

**Why wrong**: Formation delay = (B-1) / (2 x lambda) = 31 / (2 x 500) = 31 ms. This is 62% of the 50 ms SLO consumed before the GPU even starts. Add inference time (which increases with batch size), and the SLO is violated.

**Instrument**: Batch size slider (1--64), arrival rate slider (100--2000 QPS), SLO budget slider (10--100 ms). Dual-axis chart: throughput curve (rising, saturating) and p99 latency curve (U-shaped, with minimum at optimal batch). Latency waterfall showing formation delay + inference time + queuing delay. SLO violation banner when total latency exceeds budget.

**mlsysim grounding**: Engine.solve(ResNet50, H100, batch_size=B).latency provides the inference time component. Engine.solve(ResNet50, H100, batch_size=B).throughput provides the throughput component. Formation delay = (B-1) / (2 x arrival_rate). Total latency = formation_delay + inference_latency + queuing_delay(rho).

**Transition to C**: "For vision models, the batching trade-off is between latency and throughput. For LLMs, there is a third constraint: the KV cache memory wall."

---

### Part C -- The LLM Memory Wall (~12 min)

**Concept**: LLM decode-phase token generation is entirely memory-bandwidth-bound (T_token = Model_Size / Memory_BW). KV cache capacity -- not compute TFLOPS -- determines maximum concurrent batch size. At 128K context length, even an 8xH100 node can serve only 1 concurrent request for a 70B model.

**Prediction**: "You serve Llama-2 70B at 128K context on 8xH100 (640 GB total HBM). What is the maximum concurrent batch size?"
- A: 8 (one per GPU)
- B: 4 (memory split between model and cache)
- **C: 1 (the KV cache for a single 128K context nearly fills all available memory)**
- D: 16 (tensor parallelism across 8 GPUs frees memory)

**Common wrong answer**: A or B. Students think about compute parallelism (8 GPUs = 8 requests) rather than memory capacity. They do not realize that KV cache at 128K tokens for a 70B model consumes hundreds of GB.

**Why wrong**: KV cache size = 2 x layers x heads x head_dim x seq_len x batch x bytes_per_element. For Llama-2 70B at 128K in FP16: approximately 320 GB for a single request. Model weights in FP16: 140 GB. Total: 460 GB, leaving only 180 GB headroom out of 640 GB -- not enough for a second request with KV cache.

**Instrument**: Model size dropdown (7B / 13B / 70B), precision dropdown (FP16 / INT8 / INT4 for weights), context length slider (2K--128K tokens), concurrent batch size slider (1--128), GPU count toggle (1 / 4 / 8). Stacked memory bar: weights (blue, fixed) + KV cache (orange, growing with context x batch) + available VRAM. Red "OOM ZONE" when total exceeds HBM capacity. T_token readout showing memory-bandwidth-bound decode speed.

**mlsysim grounding**: Engine.solve(Llama2_70B, H100, batch_size=B, seq_len=S, precision=P) provides memory_footprint and latency. The feasible field turns False when memory exceeds capacity. KV cache size computation uses the TransformerWorkload fields (layers, hidden_dim, heads).

**Transition to D**: "The KV cache wall limits concurrent requests. But there is another capacity killer for serving: cold starts."

---

### Part D -- The Cold Start Tax (~8 min)

**Concept**: Loading a model from disk to GPU memory before the first inference adds 5--30 seconds of latency for large models. For serverless or auto-scaling deployments, this cold start latency is experienced by real users during traffic spikes. The cold start time is dominated by PCIe or NVMe bandwidth, not compute -- it is another manifestation of the data movement wall.

**Prediction**: "You auto-scale a Llama-2 70B serving instance during a traffic spike. How long does the first user wait for a response?"
- A: ~200 ms (normal inference latency)
- B: ~2 seconds (some model loading overhead)
- **C: ~15 seconds (loading 140 GB of weights over PCIe Gen5)**
- D: ~60 seconds (loading from network storage)

**Common wrong answer**: A or B. Students do not think about model loading as a latency component because in their experience, models are already in memory.

**Why wrong**: Cold start = model_size_bytes / loading_bandwidth. For 70B in FP16 (140 GB) over PCIe Gen5 (64 GB/s): 140/64 = 2.2 seconds just for the transfer. With NVMe reads, model deserialization, CUDA context initialization, and warmup inference: 10--15 seconds total.

**Instrument**: Model size dropdown, storage type dropdown (NVMe SSD / Network FS / Cached in RAM), interconnect dropdown (PCIe Gen4 / Gen5). A timeline showing: disk read + PCIe transfer + CUDA init + warmup inference. Total cold start time displayed prominently against a "user patience threshold" of 3 seconds.

**mlsysim grounding**: Cold start time = model_memory_footprint / min(storage_bandwidth, interconnect_bandwidth) + CUDA_init_overhead. Model memory from Engine.solve().memory_footprint. Storage bandwidth from hardware registry storage.bandwidth. PCIe bandwidth from hardware registry interconnect.bandwidth.

---

### Synthesis (~6 min)

**Mission**: You must design a serving deployment for two workloads: (1) a ResNet-50 vision classifier serving 10K QPS with a 50 ms p99 SLO, and (2) a Llama-2 70B chat endpoint serving 100 concurrent users with 32K context.

For each workload, specify:
1. Maximum utilization target (Part A: where does P99 become unacceptable?)
2. Optimal batch size (Part B: where is the knee of the Pareto frontier?)
3. GPU memory budget breakdown: weights vs. KV cache vs. headroom (Part C)
4. Cold start mitigation strategy (Part D: pre-warm, keep-alive, or accept the tax?)

**Design Ledger entry**: Record your P99 latency prediction from Part A. The ratio between mean and P99 is the most underestimated quantity in serving.

---
---

## Lab 14: The Silent Degradation Problem
**Story arc**: Your model shipped on Monday. By Friday, it has silently lost 3 accuracy points. By month six, it has lost 7. Your monitoring dashboard has been green the entire time.
**Time budget**: 54 min total -- A(12) + B(12) + C(12) + D(12) + Synthesis(6)

### Part A -- PSI Drift Detection (The Silent Drift) (~12 min)

**Concept**: A fraud detection model loses 7 percentage points of accuracy over 6 months while every infrastructure metric (uptime, latency, error rate) stays green. PSI-based monitoring of input feature distributions detects drift weeks before accuracy degradation becomes visible. Infrastructure health and model health are decoupled.

**Prediction**: "Your fraud detection model has been deployed for 6 months. Your monitoring dashboard shows 100% uptime, <50 ms latency, 0.01% error rate. What has happened to model accuracy?"
- **A: Still ~95% -- the system is healthy (this is what students pick)**
- B: Down to ~91% -- slight degradation
- C: Down to ~88% -- significant degradation
- D: Model has crashed

**Common wrong answer**: A. Students conflate infrastructure health with model health. Green dashboards create false confidence.

**Why wrong**: Data distributions shift while models stay static. Transaction patterns change (new fraud vectors, seasonal spending shifts, economic changes). The model was trained on a distribution that no longer exists. PSI for the "transaction_amount" feature crossed the 0.2 alert threshold at week 8, but without PSI monitoring, no one noticed.

**Instrument**: Time slider (0--26 weeks). Split dashboard: left panel shows infrastructure metrics (all permanently green -- uptime, latency, error rate). Right panel shows PSI trajectories for three features (transaction_amount, merchant_category, time_of_day) crossing the 0.2 threshold at different weeks. Accuracy decay card shows the gap between "system healthy" and "model degraded." A "detection gap" counter shows weeks between PSI alert and accuracy drop below threshold.

**mlsysim grounding**: The PSI and accuracy decay model uses documented drift formulas from the chapter: PSI(t) = sum((p_t(i) - p_0(i)) x ln(p_t(i)/p_0(i))), with feature distributions shifting at configurable rates. Accuracy(t) = Accuracy_0 - lambda x cumulative_drift(t). The drift rate lambda and feature shift rates are parameterized.

**Transition to B**: "You now know the model is degrading. The question is: how often should you retrain? Too often wastes compute. Too rarely wastes accuracy. There is a mathematically optimal answer."

---

### Part B -- Optimal Retraining Cadence (The Half-Life of a Model) (~12 min)

**Concept**: The staleness cost model T* = sqrt(2C / C_drift) produces a sublinear relationship: 4x more expensive retraining only doubles the interval. This transforms retraining from an ad hoc decision into a quantitative economic optimization with a U-shaped cost curve.

**Prediction**: "Retraining costs $10K and your model loses $500/day in accuracy-related revenue losses. Your current retraining cadence is every 30 days. What is the optimal interval?"
- A: 7 days (retrain weekly to stay fresh)
- **B: ~6 days (T* = sqrt(2 x 10000 / 500) = 6.3 days)**
- C: 14 days (biweekly is a reasonable compromise)
- D: 30 days (your current cadence is fine)

**Common wrong answer**: C or D. Students guess round numbers based on intuition rather than computing the formula. They dramatically underestimate how frequently high-drift, high-value models should retrain.

**Why wrong**: The square-root law means T* is much more sensitive to drift rate and value than students expect. When C_drift = $500/day, even expensive retraining ($10K) should happen every 6 days.

**Instrument**: U-shaped total annual cost curve. Sliders: drift rate ($100--$5000/day in accuracy losses), retraining cost ($1K--$50K), accuracy threshold (80%--95%). The T* marker moves along the curve. Three cost components visible: checkpoint overhead (decreasing hyperbola), staleness cost (increasing line), total (U-curve). Failure state when accuracy at T* falls below minimum threshold.

**mlsysim grounding**: Engine.solve(model, hardware, is_training=True) computes the actual training time and GPU-hours for retraining, converting to dollar cost via hardware.unit_cost. The retraining cost C is grounded in real hardware: C = training_hours x GPU_hourly_rate x N_GPUs. For a ResNet-50 retrain on 4xA100: Engine.solve(ResNet50, A100, is_training=True, batch_size=64) gives step time, from which total training time = steps x step_time.

**Transition to C**: "The optimal cadence for one deployment target is clear. But the same model deployed on Cloud, Edge, and Mobile has different retraining costs at each tier. How does T* change across the deployment spectrum?"

---

### Part C -- Deployment Cost Asymmetry (Same Model, Different Economics) (~12 min)

**Concept**: The same model with identical accuracy requirements produces dramatically different optimal retraining intervals across Cloud, Edge, and Mobile purely because of deployment cost differences. Cloud T* of 7--14 days vs. Edge T* of 60--90 days, driven by T* scaling with sqrt(cost), not cost itself. A 100x cost difference produces only a 10x cadence difference.

**Prediction**: "Retraining costs $1K on Cloud and $100K on Edge (including OTA update logistics). If Cloud T* is 7 days, what is Edge T*?"
- A: 700 days (100x cost = 100x interval)
- B: 140 days (linearly proportional to cost ratio)
- **C: ~70 days (sqrt(100) = 10x interval)**
- D: 14 days (edge needs more frequent updates because devices are less reliable)

**Common wrong answer**: A or B. Students assume linear scaling between cost and interval. The square-root law is deeply counterintuitive.

**Instrument**: Three-column comparison: Cloud / Edge / Mobile. Per-environment sliders for retraining cost and drift rate. T* computed independently for each. The comparison panel highlights that a 100x cost difference produces only a 10x cadence difference. A "square root surprise" annotation shows the formula.

**mlsysim grounding**: Engine.solve() for training on each deployment target provides the base training cost. Cloud: Engine.solve(model, H100, is_training=True). Edge: same plus OTA deployment cost multiplier (documented in chapter). The sqrt relationship is the core formula.

**Transition to D**: "You now have a retraining cadence for each tier. But what happens when you defer retraining too long? The technical debt compounds -- not linearly, but as a cascade."

---

### Part D -- The Debt Cascade (~12 min)

**Concept**: Technical debt in ML systems compounds through two mechanisms simultaneously: (1) correction cascades, where patching Model A with a post-hoc fix creates a dependency that makes Model B's predictions less reliable, and (2) the snowball effect, where each deferred maintenance item increases the cost of all future maintenance. A system with 3 deferred retraining cycles does not have 3x the debt -- it has ~5x, because each cycle of drift makes the next retraining harder (more distribution shift to bridge, more stale features to update, more downstream models affected).

**Prediction**: "You defer retraining for 3 consecutive cycles (3 x T*). What is the total accumulated accuracy loss compared to a single missed cycle?"
- A: 3x (linear accumulation)
- B: 4x (slightly superlinear)
- **C: ~5--6x (debt compounds due to cascading effects)**
- D: 9x (quadratic in number of missed cycles)

**Common wrong answer**: A. Students assume drift accumulates linearly. They do not account for the compounding effects: each cycle of drift moves the production distribution further from the training distribution, making the next retraining less effective (larger domain gap) and the downstream models less calibrated.

**Why wrong**: Debt compounds through two mechanisms. First, accuracy loss is not linear in drift magnitude -- the model's performance degrades faster as the distribution shifts further from training. Second, correction cascades mean that patching one model after significant drift often requires retraining or recalibrating dependent models, multiplying the cost.

**Instrument**: Timeline visualization showing 1, 2, and 3 missed retraining cycles. For each scenario: accuracy trajectory, accumulated revenue loss (area under the curve), and a "cascade counter" showing how many downstream models are affected. Controls: number of dependent downstream models (0--5), drift rate, base retraining cost. A "debt multiplier" gauge shows total accumulated cost / (N_missed x single_cycle_cost).

**mlsysim grounding**: The base drift model from Part A provides accuracy decay. The cascade model: for each downstream model, accuracy_downstream(t) = f(accuracy_upstream(t)), where f captures the dependency. Retraining cost after N missed cycles: C_retrain(N) = C_base x (1 + cascade_factor x N_downstream), because each downstream model may need recalibration. The total debt = integral of accumulated accuracy loss + sum of cascade retraining costs.

---

### Synthesis (~6 min)

**Mission**: You operate a fraud detection system with 3 dependent models (transaction scoring, account risk, merchant reputation). The system is deployed on Cloud (retrain weekly) and Edge (retrain quarterly).

Write a complete monitoring and retraining policy:
1. Which PSI thresholds trigger alerts for each feature type (Part A)
2. Optimal retraining cadence for each deployment tier, with cost justification (Parts B and C)
3. Maximum number of retraining cycles you can safely defer before the debt cascade makes recovery uneconomical (Part D)
4. The rollback strategy when a retrained model performs worse than the stale one

**Design Ledger entry**: Record your Part A prediction about the "green dashboard." The gap between infrastructure health and model health is the defining failure mode of ML systems.

---
---

## Lab 15: No Free Fairness
**Story arc**: Fairness is mathematically impossible to achieve perfectly, fixing it costs accuracy, explaining it costs latency, and all of it costs carbon -- and none of these costs are optional.
**Time budget**: 48 min total -- A(12) + B(12) + C(10) + D(8) + Synthesis(6)

### Part A -- The Fairness Illusion (~12 min)

**Concept**: The Chouldechova impossibility theorem: when base rates differ between demographic groups, a calibrated classifier with equal accuracy on both groups is mathematically guaranteed to produce unequal error rates (FPR, FNR, PPV). Equal accuracy does not mean equal treatment. You cannot have equal FPR, equal FNR, and equal PPV simultaneously when base rates differ. This is not an engineering limitation -- it is a mathematical impossibility.

**Prediction**: "You train a loan approval model with 92% accuracy for both Group A (30% base rate) and Group B (10% base rate). Are the false positive rates equal?"
- A: Yes -- equal accuracy guarantees equal error rates
- B: Nearly equal -- small differences from random noise
- **C: No -- Group B's FPR is 3x higher than Group A's despite equal accuracy**
- D: No -- Group A's FPR is higher because it has more positive cases

**Common wrong answer**: A. Students believe accuracy is a sufficient fairness metric. "If the model is equally accurate for both groups, it must be equally fair."

**Why wrong**: When base rates differ, equal accuracy requires different threshold placements for each group. Group B (10% base rate) has far more true negatives, so achieving 92% accuracy requires a lower threshold, which increases the false positive rate. The math is inescapable.

**Instrument**: Base rate sliders for two groups (5%--50%), shared classification threshold slider (0.10--0.90). Grouped bar chart of per-group metrics (accuracy, TPR, FPR, FNR, PPV). Color-coded gap cards flag violations (>5% gap turns yellow, >10% turns red). The key visual: accuracy bars are equal height while FPR bars are dramatically different.

**mlsysim grounding**: This part uses pure statistical computation (not Engine.solve()), but the scenario is framed around a deployment context: the model runs on Edge hardware with a latency SLO that constrains threshold search. Engine.solve(model, hardware) provides the inference latency that determines how many threshold evaluations can run within the SLO.

**Transition to B**: "You have just proved that perfect fairness is impossible. The next question: how much accuracy must you sacrifice to *reduce* (not eliminate) the disparity?"

---

### Part B -- The Price of Fairness (~12 min)

**Concept**: The fairness-accuracy Pareto frontier quantifies the accuracy cost of different fairness criteria. The frontier has a "sweet spot" where the first large fairness gains cost only 3--5% accuracy, then a "cliff" where strict equality demands disproportionate sacrifice. Three mitigation strategies (threshold adjustment, reweighting, adversarial debiasing) trace different paths along this frontier.

**Prediction**: "You apply equalized odds constraints to reduce the FPR gap from 15% to 5%. How much accuracy do you lose?"
- A: ~1% (fairness is nearly free at this level)
- **B: ~3--5% (the sweet spot of the Pareto frontier)**
- C: ~10% (significant accuracy sacrifice)
- D: ~15% (proportional to the gap reduction)

**Common wrong answer**: A (optimistic) or D (pessimistic). Students do not have a mental model for the shape of the Pareto frontier. Some assume fairness is free; others assume it is proportionally expensive.

**Why wrong**: The Pareto frontier has a specific shape: concave near the sweet spot (large fairness gains for small accuracy cost) and steep near strict equality (small fairness gains for large accuracy cost). The 15% to 5% gap reduction is in the sweet spot; the 5% to 0% reduction is on the cliff.

**Instrument**: Fairness criterion radio buttons (Demographic Parity / Equal Opportunity / Equalized Odds). Mitigation method dropdown (Threshold Adjustment / Reweighting / Adversarial Debiasing). Pareto frontier chart with three annotated points: A (unconstrained, highest accuracy, largest disparity), B (sweet spot), C (strict equality, lowest accuracy). Students select a target fairness level and see the accuracy cost. Failure state when equalized odds gap exceeds 10 percentage points.

**mlsysim grounding**: The Pareto frontier is computed from the statistical model in Part A. The accuracy cost at each fairness level is deterministic given the base rates and the mitigation method.

**Transition to C**: "Reducing disparity costs accuracy. But there is another cost: explaining the model's decisions to affected users. And explanations cost compute."

---

### Part C -- The Explainability Tax (~10 min)

**Concept**: Regulatory requirements (GDPR right to explanation, ECOA adverse action notices) mandate that certain model decisions be accompanied by per-instance explanations. SHAP explanations for a single prediction require running the model hundreds of times (one per feature permutation). For a loan approval model with 50 features, a SHAP explanation requires ~50 additional forward passes, adding 50x latency overhead to the serving path. Simpler methods (LIME, feature importance) are faster but less faithful.

**Prediction**: "Your loan model has 50 input features. A SHAP explanation requires computing the model output for each feature permutation subset. How much does explanation add to inference latency?"
- A: ~2x (modest overhead)
- B: ~10x (significant but manageable)
- **C: ~50x (one forward pass per feature)**
- D: ~2500x (all permutation subsets)

**Common wrong answer**: A. Students think of explanations as a lightweight post-hoc step, not as additional model executions.

**Why wrong**: SHAP computes marginal contributions by running the model with each feature masked/permuted. For 50 features, this requires approximately 50 forward passes (for the Kernel SHAP approximation; exact SHAP requires 2^50 which is infeasible). Each forward pass has the same latency as the original prediction.

**Instrument**: Feature count slider (10--200), explanation method dropdown (SHAP / LIME / Feature Importance / None). Latency waterfall showing: base inference + explanation overhead. SLO compliance check against a 100 ms budget. At 50 features with SHAP, the total latency is 50x base, easily blowing a 100 ms SLO. Method comparison table shows fidelity vs. compute cost.

**mlsysim grounding**: Engine.solve(model, hardware) provides base inference latency. Explanation overhead = base_latency x explanation_multiplier, where multiplier depends on method and feature count: SHAP ~ N_features, LIME ~ 100 (fixed sample count), Feature Importance ~ 1 (no additional forward passes).

**Transition to D**: "Fairness costs accuracy. Explanations cost latency. And every model you train, retrain, or serve costs energy -- which becomes carbon."

---

### Part D -- The Carbon Cost of Responsibility (~8 min)

**Concept**: Every fairness mitigation, every retraining cycle, every SHAP explanation consumes energy that translates to carbon emissions. A model that retrains weekly to maintain fairness across shifting demographics emits 52x more carbon than one trained once. The question is not whether to pay this cost, but how to budget it against the harms of an unfair or unexplainable system.

**Prediction**: "Your fair model retrains weekly (to track demographic shifts) and computes SHAP explanations for 10% of predictions (regulatory requirement). How does total carbon compare to a train-once, no-explanation baseline?"
- A: ~2x (modest increase)
- B: ~10x (significant but necessary)
- **C: ~60x (52x from weekly retraining + 5x from explanations on 10% of traffic)**
- D: ~100x (even higher due to compounding)

**Common wrong answer**: A or B. Students dramatically underestimate the cumulative carbon cost of operational ML.

**Instrument**: Carbon budget dashboard. Controls: retraining frequency (weekly / monthly / quarterly / never), explanation coverage (0% / 10% / 50% / 100%), explanation method (SHAP / LIME / None). Stacked bar showing: training carbon (one-time) + retraining carbon (cumulative over 1 year) + serving carbon (per-query x query volume) + explanation carbon. A carbon intensity toggle (grid mix: clean hydro / mixed / coal-heavy) shows geographic impact.

**mlsysim grounding**: Engine.solve(model, hardware, is_training=True).energy provides per-training-run energy. Serving energy from Engine.solve(model, hardware).energy x queries_per_year. Carbon = energy x grid_carbon_intensity (gCO2/kWh, parameterized by region).

---

### Synthesis (~6 min)

**Mission**: You are deploying a loan approval model under ECOA regulations. You must present a system specification to your compliance officer that addresses:

1. Which fairness criterion you chose and why (Part A: impossibility constrains your options)
2. The accuracy cost of your fairness constraint and why it is at the sweet spot (Part B)
3. Your explanation strategy: which method, for which percentage of predictions, and the latency impact (Part C)
4. The annual carbon budget of the entire responsible-AI stack: training + retraining + serving + explanations (Part D)

Frame each choice as a trade-off, not a solution. There is no free fairness.

**Design Ledger entry**: Record your Part A prediction about equal accuracy implying equal fairness. The impossibility theorem is the most counterintuitive result in the lab series.

---
---

## Lab 16: The Architect's Audit (Capstone)
**Story arc**: Everything you have learned in 15 labs collapses into one question: given a real system with real constraints, where do you invest your engineering effort? The answer requires every tool in your toolkit -- and the discipline to recognize which tool does not apply.
**Time budget**: 58 min total -- A(12) + B(12) + C(10) + D(8) + E(10) + Synthesis(6)

### Part A -- The Cost of a Token (Calibration) (~12 min)

**Concept**: The Iron Law decomposition for single-token LLM inference reveals that memory-to-compute time ratio is ~40x on H100, not the ~5x most students expect. Autoregressive decoding has arithmetic intensity of ~1 FLOP/byte, placing it deep in the memory-bandwidth-bound regime. This means compute kernel optimization is futile without addressing data movement. This Part calibrates students' quantitative intuition one final time.

**Prediction**: "For Llama-2 70B at batch=1 on H100, what fraction of token generation time is spent waiting for memory?"
- A: ~50% (roughly balanced)
- B: ~75% (memory dominates moderately)
- C: ~90% (memory dominates strongly)
- **D: ~98% (memory access is essentially all of the time)**

**Common wrong answer**: A or B. Even after 15 labs, students underestimate the severity of the memory wall for autoregressive decoding.

**Instrument**: Model size selector (7B / 13B / 70B), precision toggle (FP32 / FP16 / INT8 / INT4), batch size slider (1--256). Iron Law decomposition bar chart: memory time (blue) vs. compute time (orange) vs. overhead (gray). Roofline operating point dot. At batch=1, memory dominates by 40x. At batch=64, the bars approximately equalize -- making the crossover from memory-bound to compute-bound visible.

**mlsysim grounding**: Engine.solve(Llama2_70B, H100, batch_size=1, precision="fp16") provides latency_compute, latency_memory, and arithmetic_intensity directly. The ratio latency_memory / latency_compute is the key number. Sweeping batch size from 1 to 256 shows the crossover.

**Transition to B**: "You now know the cost of a single token. But a deployed system is not just a single inference -- it is a living organism that degrades, drifts, and accumulates debt. The Conservation of Complexity governs what happens next."

---

### Part B -- The Conservation of Complexity (~12 min)

**Concept**: Complexity in an ML system cannot be destroyed, only moved between Data, Algorithm, and Machine. Quantizing MobileNetV2 to INT8 reduces model complexity (Algorithm axis) but creates monitoring debt (Machine axis) -- without monitoring investment, accuracy degrades ~1.3%/month while all infrastructure metrics remain green. Students must track 4--5 key invariants simultaneously to see that optimizing one axis shifts the burden to another.

**Prediction**: "You quantize MobileNetV2 to INT8 and deploy on mobile. After 6 months without monitoring, what has happened?"
- A: Nothing -- INT8 is stable and the model works fine
- B: Latency increased due to hardware wear
- **C: Accuracy silently degraded ~8% while all other metrics stayed green**
- D: The model crashed from numerical instability

**Common wrong answer**: A. Students think deployment is the end of the engineering process. They do not connect quantization (which reduces model robustness to distribution shift) with monitoring investment (which detects that shift).

**Instrument**: Configuration panel: architecture dropdown, quantization level, monitoring investment level (None / Basic / Comprehensive). Time slider (0--12 months deployed). 5-axis radar chart: accuracy, latency, memory, power, drift resilience. Without monitoring, the accuracy axis silently shrinks while all others stay green. An invariant activation table shows which of 4--5 invariants are triggered: (1) Amdahl's Law, (2) Memory Wall, (3) Silent Degradation, (4) Conservation of Complexity, (5) No Free Fairness.

**mlsysim grounding**: Engine.solve(MobileNetV2, iPhone15Pro, precision="int8") provides the initial latency, memory, and energy. The drift model from Lab 14 Part A provides accuracy decay over time. The monitoring investment parameter determines whether PSI alerts fire (Lab 14) or not.

**Transition to C**: "You have seen how one system degrades. Now turn the lens on yourself: across 15 labs, where did *your* predictions consistently fail?"

---

### Part C -- Design Ledger Archaeology (~10 min)

**Concept**: A student's own prediction history across Labs 01--15 reveals systematic blind spots -- which invariants their intuition consistently underweights. The gap between self-assessed and data-revealed weaknesses is itself an engineering insight: mental models degrade just like deployed models.

**Prediction**: "Before viewing your data, which invariant do you think you most consistently underestimated across all 15 labs?"
- Self-assessment via dropdown: Amdahl's Law / Memory Wall / Silent Degradation / Conservation of Complexity / No Free Fairness

(No "correct" answer -- the reveal comes from comparing self-assessment to actual data.)

**Instrument**: 5-axis radar chart computed from Design Ledger history (prediction error magnitude per invariant category). The student's self-assessment is overlaid as a second polygon. The gap between the two reveals metacognitive blind spots.

**Fallback (Design Ledger unavailable)**: If a student does not have complete Ledger data (e.g., they skipped labs or started late), the instrument loads a pre-generated "typical student" dataset showing the median prediction error profile across the class. The student can still self-assess and compare against the class median. A banner notes: "Using class median data. Complete your Design Ledger entries for a personalized analysis."

**mlsysim grounding**: The Design Ledger is a lab-framework feature (mlsysim.labs.state), not an Engine.solve() computation. The radar chart aggregates prediction errors from the ledger entries written at the end of each prior lab.

**Transition to D**: "Your personal blind spots are one piece of the puzzle. The other piece: the fundamental limits that constrain *every* architect. Remember the Amdahl Ceiling from Lab 12?"

---

### Part D -- The Amdahl Ceiling Revisited (~8 min)

**Concept**: This Part explicitly calls back to Lab 12 Part A (The Amdahl Ceiling), but now applies it to a complete system design rather than a single pipeline. The question shifts from "what is the speedup?" to "where should I invest my engineering effort?" Amdahl's Law is not just a speedup formula -- it is a resource allocation framework. The component with the largest time fraction has the highest-leverage optimization.

**Prediction**: "Your ML system has 4 components: preprocessing (35%), inference (40%), postprocessing (15%), logging (10%). You can invest engineering effort to optimize ONE component by 5x. Which yields the largest end-to-end speedup?"
- A: Preprocessing (largest non-inference component)
- **B: Inference (largest overall component: 1/(0.6 + 0.4/5) = 1.47x)**
- C: Postprocessing (seems like low-hanging fruit)
- D: It does not matter -- 5x on any component yields roughly the same system speedup

**Common wrong answer**: D. Students who remember Amdahl's Law from Lab 12 may think "it is always about the serial fraction" without computing which component IS the serial fraction.

**Instrument**: System component breakdown with editable time fractions (must sum to 100%). "Optimize" dropdown selects one component, speedup slider (1x--20x). Amdahl speedup readout and a "dollars per 1% improvement" cost efficiency metric. The visual callback to Lab 12: same Amdahl saturation curve, but now with real system components instead of abstract fractions.

**mlsysim grounding**: Engine.solve() provides the inference time component. The other components (preprocessing, postprocessing, logging) are parameterized. Total system time = sum of all components, with the optimized component divided by the speedup factor.

**Transition to E**: "You know where to invest effort. But what happens when you optimize one component and it shifts the constraint to another? This is the Constraint Cascade -- and it is the final lesson of the course."

---

### Part E -- The Constraint Cascade (~10 min)

**Concept**: Optimizing one constraint in an ML system does not solve the problem -- it moves the binding constraint to the next-tightest axis. Quantizing to INT4 solves the memory constraint but creates a precision constraint. Increasing batch size solves the throughput constraint but creates a latency constraint. Retraining more frequently solves the drift constraint but creates a carbon constraint. This is the Conservation of Complexity made dynamic: the system is always constrained, and the architect's job is to choose which constraint to live with.

**Prediction**: "You quantize Llama-2 70B from FP16 to INT4, solving the memory constraint on a single H100 (70B x 0.5 bytes = 35 GB < 80 GB HBM). What is the new binding constraint?"
- A: Compute (INT4 is slower per operation)
- **B: KV cache memory (weights fit, but KV cache at long context still fills HBM)**
- C: Accuracy (INT4 quality is insufficient)
- D: No constraint -- the system now works

**Common wrong answer**: D. Students think solving the immediate constraint means the system works. They do not trace the cascade to the next binding constraint.

**Why wrong**: INT4 weights for 70B fit in 35 GB. But KV cache for a 32K context in FP16 is still ~40 GB. Total: 75 GB, leaving only 5 GB headroom. At 64K context, the system OOMs again. The constraint has moved from "model too large" to "context too long."

**Instrument**: Constraint dashboard with 5 axes: Memory, Latency, Accuracy, Carbon, Fairness. Students apply optimizations one at a time via checkboxes: INT4 quantization, structured pruning, knowledge distillation, continuous retraining, SHAP explanations. Each optimization resolves one constraint (turns it green) but shifts the binding to another (turns it yellow, then red). The cascade is visible as a chain of constraint migrations.

**mlsysim grounding**: Engine.solve() at each configuration provides memory_footprint, latency, and energy. Feasibility checks (feasible field) track which constraints are satisfied. The cascade: Engine.solve(Llama2_70B, H100, precision="int4", batch_size=1, seq_len=32768) shows the KV cache exceeding available memory even with INT4 weights.

---

### Synthesis (~6 min)

**Mission**: You are the ML systems architect for a startup deploying Llama-2 70B as a chat service on a fleet of 8xH100 nodes. Write a deployment specification that addresses:

1. Precision and quantization strategy (Part A: what is the actual cost of a token?)
2. Monitoring investment to prevent silent degradation (Part B: Conservation of Complexity)
3. Your personal blind spot and how you will compensate for it (Part C: Design Ledger)
4. Where to invest your next engineering sprint (Part D: Amdahl's resource allocation)
5. The constraint cascade: which constraint are you choosing to live with? (Part E)

This is the "graduation specification." It integrates every lab into one document. Students who complete it have demonstrated quantitative reasoning about ML systems across the full stack.

**Design Ledger entry (final)**: Record your cumulative prediction accuracy across all 16 labs. Compute your mean absolute error per invariant category. This is your engineering calibration score.

---
---

# Cross-Lab Reference Table

| Lab | Title | Time | Parts | Key Invariants |
|-----|-------|------|-------|---------------|
| 09 | Data Selection Paradox | 52 min | A(ICR) + B(Selection Inequality) + C(Preprocessing Tax) + D(Chinchilla) + Synthesis | Memory Wall, Amdahl |
| 10 | Compression Paradox | 56 min | A(Quantization) + B(Pruning Trap) + C(Pareto) + D(Energy) + E(Distillation) + Synthesis | Memory Wall, Conservation of Complexity |
| 11 | Hardware Roofline | 52 min | A(Roofline) + B(Fusion) + C(Balance Shift) + D(Energy Roofline) + E(Tiling) + Synthesis | Memory Wall, Amdahl |
| 12 | Benchmarking Trap | 52 min | A(Amdahl) + B(Thermal) + C(Multi-Metric) + D(Tail Latency) + Synthesis | Amdahl, Silent Degradation |
| 13 | Tail Latency Trap | 50 min | A(Queuing) + B(Batching) + C(KV Cache) + D(Cold Start) + Synthesis | Memory Wall, Amdahl |
| 14 | Silent Degradation | 54 min | A(PSI Drift) + B(Retraining Cadence) + C(Cost Asymmetry) + D(Debt Cascade) + Synthesis | Silent Degradation, Conservation of Complexity |
| 15 | No Free Fairness | 48 min | A(Impossibility) + B(Pareto) + C(Explainability Tax) + D(Carbon) + Synthesis | No Free Fairness, Conservation of Complexity |
| 16 | Architect's Audit | 58 min | A(Token Cost) + B(Conservation) + C(Ledger) + D(Amdahl Revisited) + E(Cascade) + Synthesis | All invariants |

# Redundancy Resolution

| Concept | Owned By | Other Labs Reference (Do Not Re-teach) |
|---------|----------|--------------------------------------|
| Amdahl's Law | Lab 12 Part A | Lab 16 Part D explicitly references Lab 12 |
| Precision effects (quantization) | Lab 10 Part A | Lab 13 Part E DROPPED (was redundant) |
| Silent degradation (PSI drift) | Lab 14 Part A | Lab 16 Part B references Lab 14 |
| Energy/carbon | Lab 10 Part D (per-inference) | Lab 15 Part D (system-level) -- different scope |
| Roofline model | Lab 11 Part A | Lab 12 Part D uses it but does not re-teach |
| TCO | MOVED to Lab 12/13 territory | Lab 15 Part C DROPPED (was TCO) |
| Curriculum learning | DROPPED | Lab 09 Part C REPLACED with Preprocessing Tax |



---

# VOLUME II: Machine Learning Systems at Scale

# Volume 2 Labs: Final Cleaned Plans (V2-01 through V2-07)

Generated: 2026-03-15
Status: FINAL -- ready for lab-developer handoff
Numbering: Post-merger (old V2-03 + V2-06 merged into new V2-03; old V2-07 becomes V2-06; old V2-08 becomes V2-07)

---

## Lab V2-01: The Scale Illusion

**Story arc**: Students arrive believing that distributed training is "just more of the same" -- that 1,000 GPUs deliver 1,000x speedup and that hardware reliability at scale is a non-issue. Over five parts, they watch reliability collapse exponentially, discover that communication devours compute gains, learn that scaling laws punish naive resource allocation, confront Amdahl's Law with real communication overhead, and finally classify workloads by their dominant bottleneck. They leave understanding that scale creates qualitative change, not just quantitative increase.

**Time budget**: 55 min (12 + 12 + 10 + 12 + 8 = 54 min + 1 min transitions)

---

### Part A -- The Reliability Collapse (~12 min)

**Concept**: Fleet-wide availability decays exponentially with fleet size. A 1,000-GPU cluster with 99.9% per-node reliability is healthy only 36.8% of the time. At GPT-4 scale (25,000 GPUs), a hardware failure occurs every ~4.4 hours. Failure is the common case, not the exception.

**Prediction**: "Your cluster has 1,000 GPUs, each with 99.9% individual uptime. What fraction of the time is the entire cluster healthy?"

| Option | Value |
|--------|-------|
| A | ~99% -- nearly always healthy |
| B | ~90% -- healthy most of the time |
| C | ~60% -- healthy more often than not |
| **D (correct)** | **~37% -- healthy barely a third of the time** |

**Common wrong answer**: A or B. Students anchor on the per-node reliability (99.9%) and assume fleet reliability degrades linearly or gently.

**Why wrong**: The exponential in P_fleet = (P_node)^N makes even tiny per-node failure rates catastrophic at scale. (0.999)^1000 = 0.368.

**Instrument**:
- Slider 1: Fleet size N (1 to 25,000, step 100, default 1,000)
- Slider 2: Per-node reliability P_node (0.990 to 0.9999, step 0.001, default 0.999)
- Chart: Fleet availability vs. fleet size (line chart, with reference line at P_node = 0.9999)
- Metric row: Fleet availability %, MTBF (= GPU_MTTF_HOURS / N), failures per day

**mlsysim grounding**: `calc_failure_probability(mtbf, job_duration)` from `mlsysim.core.formulas`; `GPU_MTTF_HOURS` from `mlsysim.core.defaults` (50,000 hours). `calc_mtbf_cluster(GPU_MTTF_HOURS, N)` gives system MTBF.

**Transition to B**: "So the cluster fails every few hours. But when it is running, surely 1,000 GPUs give you 1,000x speedup? Let us check what happens to your training step time as you scale."

---

### Part B -- The Coordination Tax (~12 min)

**Concept**: The Fleet Law decomposes distributed step time into Compute + Communication + Coordination. The "Conservation of Overhead" means you cannot eliminate overhead, only redistribute it. At 1,000 GPUs, communication can consume 40% or more of step time, capping effective speedup far below 1,000x.

**Prediction**: "You scale a 175B model training job from 1 GPU to 256 GPUs on InfiniBand NDR. What fleet efficiency do you expect?"

| Option | Value |
|--------|-------|
| A | ~95% -- InfiniBand is fast enough |
| B | ~80% -- some communication overhead |
| **C (correct)** | **~55-65% -- communication is substantial** |
| D | ~30% -- communication dominates |

**Common wrong answer**: A or B. Students overestimate InfiniBand's ability to hide 175B-parameter gradient synchronization.

**Why wrong**: Ring AllReduce for 175B FP16 parameters transfers ~700 GB per step. Even at 50 GB/s (IB NDR), this takes ~14 seconds, a significant fraction of compute time.

**Instrument**:
- Slider 1: Number of GPUs (1 to 1,024, powers of 2)
- Slider 2: Model gradient size (select: 1B/7B/70B/175B, maps to bytes)
- Toggle: Network type (IB NDR 50 GB/s vs. IB HDR 25 GB/s vs. 100GbE 12.5 GB/s)
- Chart: Stacked bar of T_compute, T_communication, T_coordination per step
- Gauge: Fleet efficiency eta = T_compute / T_step

**mlsysim grounding**: `calc_ring_allreduce_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s)` from `mlsysim.core.formulas`. Fleet defined using `Fabrics.InfiniBand_NDR` (bandwidth=400 Gbps, latency=5 us) and `Nodes.DGX_H100` from `mlsysim.systems.registry`.

**Transition to C**: "Communication is expensive. But before you can even estimate costs, you need to know how much compute to buy. Scaling laws tell you the optimal allocation of compute between model size and data -- get it wrong and you waste millions."

---

### Part C -- The Scaling Law Budget Planner (~10 min)

**Concept**: Compute-optimal resource allocation (Chinchilla scaling) requires coordinated scaling of model size and dataset size. Scaling one dimension alone wastes resources. The optimal ratio is approximately D = 20P (20 tokens per parameter).

**Prediction**: "You have a fixed compute budget of 10^23 FLOPs. Which achieves lower loss: a 10B model trained on 200B tokens, or a 3B model trained on 600B tokens?"

| Option | Value |
|--------|-------|
| A | 10B on 200B tokens -- bigger models are always better |
| **B (correct)** | **3B on 600B tokens -- balanced allocation wins** |
| C | Both achieve the same loss -- total FLOPs is what matters |
| D | Neither -- you need at least 70B parameters |

**Common wrong answer**: A. Students assume that model size is the dominant factor in capability.

**Why wrong**: The Chinchilla scaling law shows that for a fixed compute budget, there is a unique optimal (P, D) pair. Over-allocating to model size under-trains the model; over-allocating to tokens under-parameterizes it.

**Instrument**:
- Slider 1: Model parameters P (1B to 100B, log scale)
- Slider 2: Training tokens D (10B to 10T, log scale)
- Constraint indicator: C = 6PD FLOPs (shows current compute use vs. budget)
- Chart: IsoFLOP loss curves with current allocation marked
- Metric: Distance from Chinchilla optimal point

**mlsysim grounding**: `CHINCHILLA_TOKENS_PER_PARAM` (value: 20) and `CHINCHILLA_COMPUTE_CONSTANT` (value: 6) from `mlsysim.core.defaults`. Loss approximation uses the Hoffmann et al. parametric form.

**Transition to D**: "Now you know how much compute you need. But how many GPUs should you buy? At some point, adding GPUs costs more than the speedup is worth. Let us find that point."

---

### Part D -- The Iron Law of Scale (~12 min)

**Concept**: Distributed training speedup is limited by an extended Amdahl's Law where the serial fraction includes communication overhead. Beyond a critical GPU count, adding hardware reduces cost-efficiency. The communication fraction r determines where the speedup curve bends.

**Prediction**: "For a workload with 20% communication overhead (r = 0.20), how many GPUs does it take before scaling efficiency drops below 50%?"

| Option | Value |
|--------|-------|
| A | ~512 GPUs -- efficiency holds a long time |
| B | ~128 GPUs -- moderate scale |
| **C (correct)** | **~32-64 GPUs -- surprisingly few** |
| D | ~8 GPUs -- almost immediately |

**Common wrong answer**: A or B. Students overestimate how far linear scaling extends.

**Why wrong**: At r = 0.20, the communication term grows linearly with N while compute per GPU shrinks as 1/N. The crossover arrives faster than intuition suggests. By 64 GPUs, the communication term consumes half of step time.

**Instrument**:
- Slider 1: Communication fraction r (0.01 to 0.50)
- Slider 2: Number of GPUs (1 to 512, log scale)
- Slider 3: Overlap percentage (0% to 80%)
- Chart: Log-log speedup chart (ideal linear vs. actual), echoing @fig-scaling-tax
- Chart 2: Cost per sample vs. GPU count (shows $ wasted on idle GPUs)
- Marker: Efficiency = 50% line

**mlsysim grounding**: Formulas: T_step(N) = T_compute/N + T_comm(N) - T_overlap. Scaling efficiency from `SCALING_EFF_32GPU` (0.90), `SCALING_EFF_256GPU` (0.70), `SCALING_EFF_1024GPU` (0.50) in `mlsysim.core.defaults`. GPU cost from `Hardware.H100.unit_cost` ($30,000).

**Transition to E**: "You have seen four failure modes of scale: reliability collapse, communication tax, misallocated compute budgets, and diminishing GPU returns. But in any real system, one of these dominates. The C-Cubed diagnostic tells you which."

---

### Part E -- The C-Cubed Diagnostic (~8 min)

**Concept**: The C-Cube taxonomy (Computation, Communication, Coordination) provides a diagnostic framework for identifying the dominant bottleneck in any distributed system. The Conservation of Overhead sits at the center: reducing one C causes another to become dominant.

**Prediction**: Students classify three archetype workloads by dominant bottleneck via radio buttons (not drag-and-drop):

"For each workload, select the dominant bottleneck: Computation, Communication, or Coordination."

| Workload | Student's likely guess | Actual dominant bottleneck |
|----------|----------------------|--------------------------|
| GPT-4 LLM training (175B, 25K GPUs) | Computation | **Communication** (gradient sync) |
| DLRM recommendation (embedding-heavy) | Communication | **Coordination** (All-to-All embedding lookups) |
| Federated MobileNet (edge devices) | Communication | **Coordination** (straggler handling, privacy overhead) |

**Common wrong answer**: Students guess Computation for LLM training because "it has the most FLOPs."

**Why wrong**: At 25K GPUs, the per-GPU compute is small but gradient synchronization of 175B parameters is massive. The Communication term dominates.

**Instrument**:
- Radio buttons: For each of 3 archetypes, select Computation / Communication / Coordination
- Reveal: Stacked bar showing actual time breakdown for each archetype
- Visual: C-Cube triangle with each workload plotted at its actual position

**mlsysim grounding**: Uses `calc_ring_allreduce_time()` for LLM communication estimate. Fleet configurations from `Clusters.Frontier_8K` and `Clusters.Mega_100K`. Compute time from `Engine.solve(model=..., hardware=Hardware.H100)`.

---

## Lab V2-02: The Compute Infrastructure Wall

**Story arc**: Students discover that even the fastest accelerator in the world spends most of its time waiting for data, that the roofline model reveals why, that bandwidth drops by orders of magnitude at each physical boundary, and that even a full DGX node cannot hold a frontier model without memory optimization. They close by discovering that the real cost of scale is not GPUs but everything around them.

**Time budget**: 57 min (12 + 12 + 12 + 12 + 9 = 57 min)

---

### Part A -- The Memory Wall (~12 min)

**Concept**: Token generation latency is dominated by memory bandwidth, not compute. Even with infinite compute throughput, token latency barely improves because data delivery from HBM is the binding constraint. At batch=1, an H100 achieves less than 1% of peak FLOPS.

**Prediction**: "During single-token generation of a 70B model on an H100, what fraction of the time are the arithmetic units idle?"

| Option | Value |
|--------|-------|
| A | ~10% idle -- GPUs are mostly computing |
| B | ~50% idle -- memory and compute are balanced |
| C | ~80% idle -- memory is a drag |
| **D (correct)** | **~99% idle -- the GPU is almost entirely waiting for data** |

**Common wrong answer**: A or B. Students assume GPUs are "compute machines."

**Why wrong**: 70B parameters at 2 bytes = 140 GB. At 3.35 TB/s HBM bandwidth, loading takes ~42 ms. Compute for one token takes ~0.07 ms. The GPU is 99.8% idle.

**Instrument**:
- Select: Model size (7B, 70B, 175B)
- Select: Accelerator (A100, H100, B200)
- Slider: Batch size (1 to 128, powers of 2)
- Chart: Latency waterfall (compute bar vs. memory bar) with ridge point marked
- Gauge: MFU (turning green as batch size increases past ridge point)

**mlsysim grounding**: `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100, batch_size=1, precision="fp16")` returns `PerformanceProfile` with `latency_compute`, `latency_memory`, `mfu`, and `bottleneck`. Hardware specs from `Hardware.H100` (3.35 TB/s HBM, 989 TFLOPS FP16).

**Transition to B**: "At batch=1, the H100 is a $30,000 space heater. The roofline model explains exactly why -- and reveals which workloads can actually use the hardware you paid for."

---

### Part B -- The Roofline Diagnostic (~12 min)

**Concept**: The Roofline Model determines whether a workload is compute-bound or memory-bound. The ridge point separates regimes. Fleet-scale workloads (175B training, not ResNet-50) demonstrate that even on frontier hardware, most LLM operations are memory-bound.

**Prediction**: "Place these fleet-scale workloads on the roofline: LLM decode at batch=1, LLM decode at batch=32, 175B LLM training (forward pass), 175B LLM prefill. Which falls above the ridge point?"

| Option | Value |
|--------|-------|
| A | All of them -- H100 is always compute-bound for large models |
| B | LLM training and prefill -- they have large batch sizes |
| **C (correct)** | **Only LLM prefill and training at large batch -- decode is always below** |
| D | None -- all LLM workloads are memory-bound |

**Common wrong answer**: A. Students conflate "large model" with "compute-bound."

**Why wrong**: Arithmetic intensity depends on the ratio of FLOPs to bytes moved, not model size alone. LLM decode at batch=1 has arithmetic intensity ~0.5 (each weight is loaded once for one multiply), far below the H100 ridge point of ~295 FLOPs/byte at FP16.

**Instrument**:
- Interactive roofline plot: Students drag workload dots to predicted positions
- Workloads: 175B decode B=1, 175B decode B=32, 175B training, 175B prefill, DLRM embedding lookup
- Hardware selector: V100, A100, H100, B200 (shifts the roofline and ridge point)
- Metric: Achieved TFLOPS vs. peak for each workload

**mlsysim grounding**: `Engine.solve()` for each workload returns `arithmetic_intensity` and `peak_flops_actual`. Hardware ridge points calculated as `H100_FLOPS_FP16_TENSOR / H100_MEM_BW` (~295). Model specs from `Models.Llama3_8B`, `Models.Llama3_70B`. Note: uses fleet-scale models (175B), NOT ResNet-50, to differentiate from V1-11.

**Transition to C**: "The roofline tells you about one chip. But a training cluster has thousands of chips connected by a bandwidth staircase where each step drops speed by 10-100x. Let us see how this hierarchy dictates which parallelism strategy goes where."

---

### Part C -- The Bandwidth Staircase (~12 min)

**Concept**: Data transfer speed drops by orders of magnitude at each physical boundary (HBM to NVLink to PCIe to InfiniBand), and this hierarchy dictates which parallelism strategy operates at which level. NVLink-to-IB is an 18x cliff.

**Prediction**: "How much slower is a 10 GB gradient AllReduce over InfiniBand NDR compared to NVLink 4.0?"

| Option | Value |
|--------|-------|
| A | ~2x slower -- InfiniBand is fast |
| B | ~5x slower -- significant but manageable |
| **C (correct)** | **~18x slower -- an order of magnitude gap** |
| D | ~100x slower -- completely different regime |

**Common wrong answer**: A or B. Students underestimate the NVLink-to-IB bandwidth gap.

**Why wrong**: NVLink 4.0 = 900 GB/s. IB NDR = 50 GB/s. Ratio = 18x. For a 10 GB transfer: 11 ms (NVLink) vs. 200 ms (IB NDR). This is why tensor parallelism is confined to within a node.

**Instrument**:
- Slider: Transfer size (1 MB to 10 GB, log scale)
- Staircase bar chart: Transfer time at HBM (3.35 TB/s), NVLink (900 GB/s), PCIe Gen5 (64 GB/s), IB NDR (50 GB/s), IB HDR (25 GB/s)
- Parallelism strategy mapping: TP -> NVLink, PP -> IB (small transfers), DP -> IB (large transfers with compression)

**mlsysim grounding**: `NVLINK_H100_BW` (900 GB/s), `INFINIBAND_NDR_BW` (400 Gbps = 50 GB/s), `PCIE_GEN5_BW` (64 GB/s), `H100_MEM_BW` (3.35 TB/s) from `mlsysim.core.constants`. `Nodes.DGX_H100.intra_node_bw` (900 GB/s).

**Transition to D**: "The bandwidth hierarchy tells you where to put each parallelism strategy. But before you parallelize, you need to know: does the model even fit? Let us check the memory budget for a frontier 175B model."

---

### Part D -- The Node Memory Budget (~12 min)

**Concept**: Training a 175B model requires careful memory budgeting. A single accelerator cannot hold the model; a full 8-GPU DGX H100 node barely suffices even with ZeRO-3, because activation memory pushes total past HBM limits.

**Prediction**: "Can a single 8-GPU DGX H100 node (640 GB total HBM) train a 175B model with Adam in FP16 without ZeRO?"

| Option | Value |
|--------|-------|
| A | Yes -- 640 GB is plenty for a 175B model |
| B | Barely -- it fits with ~50 GB headroom |
| **C (correct)** | **No -- static memory alone exceeds 640 GB** |
| D | No, but ZeRO-1 fixes it |

**Common wrong answer**: A. Students compute only weight memory (175B x 2 bytes = 350 GB) and think it fits.

**Why wrong**: Training memory = weights (350 GB) + gradients (350 GB) + Adam optimizer states (700 GB FP32) = 1,400 GB static memory. This is 2.2x the total HBM of a DGX H100 node.

**Instrument**:
- Slider: Model size (1B to 175B)
- Toggle: Precision (FP32 / FP16 / INT8)
- Select: Optimizer (SGD / Adam / Adafactor)
- Slider: GPUs per node (1, 2, 4, 8)
- Select: ZeRO stage (0, 1, 2, 3)
- Stacked bar: Per-GPU memory (weights, gradients, optimizer, activations) with HBM capacity line

**mlsysim grounding**: `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100, batch_size=1, precision="fp16", is_training=True, zero_stage=3, dp_size=64)` returns `memory_footprint` and `feasible`. Hardware from `Hardware.H100` (80 GB HBM).

**Transition to E**: "So you need multiple nodes, ZeRO optimization, and careful parallelism. But what does all of this cost? The GPU price tag is only the beginning."

---

### Part E -- TCO: The Hidden Cost of Scale (~9 min)

**Concept**: Total Cost of Ownership goes far beyond GPU purchase price. Power, cooling, networking, and utilization efficiency determine the real cost per useful FLOP. A 1,000-GPU H100 cluster costs ~$3M/year in electricity alone. Utilization rate often matters more than hardware generation.

**Prediction**: "For a 1,000-GPU H100 inference cluster, which costs more over 3 years: the GPUs themselves, or the electricity to run them?"

| Option | Value |
|--------|-------|
| A | GPUs by far -- $30M vs. ~$3M electricity |
| **B (correct)** | **GPUs cost more, but electricity is ~30% of total TCO -- surprisingly close** |
| C | Electricity is more expensive -- power costs dominate |
| D | They are roughly equal |

**Common wrong answer**: A. Students drastically underestimate operational costs.

**Why wrong**: 1,000 H100s at 700W = 700 kW. With PUE 1.12, facility power = 784 kW. At $0.12/kWh, annual electricity = $824K. Over 3 years: $2.5M electricity + maintenance + cooling + networking brings OpEx to ~40% of total.

**Instrument**:
- Configure: GPU count, GPU type (A100/H100/B200), networking tier, cooling type
- Slider: Utilization (30% to 90%)
- Slider: PUE (1.1 to 1.6)
- Chart: TCO breakdown (CapEx: GPUs, networking, storage; OpEx: power, cooling, staff) over 3-year lifecycle
- Metric: Cost per useful FLOP at current utilization

**mlsysim grounding**: `Hardware.H100.unit_cost` ($30,000), `Hardware.H100.tdp` (700W), `PUE_BEST_AIR` (1.12), `CLOUD_ELECTRICITY_PER_KWH` ($0.12). `calc_fleet_tco()` from `mlsysim.core.formulas`. Infrastructure costs via `Racks.AI_Standard.power_kw`.

---

## Lab V2-03: Communication at Scale

**Story arc**: This lab merges the old Network Fabrics (V2-03) and Collective Communication (V2-06) labs into a single narrative. Students start with the alpha-beta model to understand the fundamental physics of network transfer, then discover that the choice of AllReduce algorithm depends on a quantitative crossover point. They confront the NVLink-to-InfiniBand cliff and see how hierarchical communication exploits it. They learn when gradient compression helps (and when it hurts). Finally, they assemble a complete communication budget for a 70B model and try to get it under 20% of step time.

**Time budget**: 58 min (10 + 12 + 12 + 12 + 12 = 58 min)

---

### Part A -- The Network Time Budget (~10 min)

**Concept**: The alpha-beta model (T(n) = alpha + n/beta) separates network communication into a latency-dominated regime and a bandwidth-dominated regime. For LLM-scale gradients (hundreds of GB), the bandwidth term dominates by four to five orders of magnitude. This calibration is essential before any algorithm selection can be meaningful.

**Prediction**: "A 70B model trains with data parallelism across 64 GPUs on InfiniBand NDR (50 GB/s, 5 us latency). Gradients are FP32 (4 bytes per parameter). How long does one Ring AllReduce take?"

| Option | Value |
|--------|-------|
| A | ~0.5 ms -- network latency dominates |
| B | ~50 ms -- bandwidth matters, but IB is fast |
| C | ~1,100 ms (~1 second) -- the 280 GB payload starts to show |
| **D (correct)** | **~11,000 ms (~11 seconds) -- bandwidth completely dominates** |

**Common wrong answer**: B or C. Students anchor on IB latency (microseconds) and underestimate by 100-1000x.

**Why wrong**: 70B params x 4 bytes = 280 GB. Ring AllReduce transfers 2(N-1)/N x M bytes. T_bandwidth = 2 x (63/64) x 280 GB / 50 GB/s = 11,032 ms. T_latency = 2 x 63 x 5 us = 0.63 ms. Bandwidth is 99.994% of total.

**Instrument**:
- Select: Model parameters (1B, 7B, 13B, 70B, 175B)
- Toggle: Precision (FP32, BF16, FP8)
- Select: GPU count (8, 16, 32, 64, 128, 256, 512, 1024)
- Toggle: Interconnect (NVLink 4.0 / IB NDR / IB HDR)
- Chart: Stacked bar showing bandwidth term vs. latency term (log scale)
- Metric: Total AllReduce time, bandwidth fraction %

**mlsysim grounding**: `calc_ring_allreduce_time(message_bytes, n_gpus, bandwidth_bytes_s, latency_s)` from `mlsysim.core.formulas`. `Fabrics.InfiniBand_NDR` (bandwidth=400 Gbps, latency=5 us). `INFINIBAND_NDR_BW` (400 Gbps = 50 GB/s), `IB_NDR_LATENCY_US` (5 us) from constants/defaults.

**Transition to B**: "Now you know AllReduce is a bandwidth problem. But Ring AllReduce is not the only algorithm. Tree AllReduce trades bandwidth for latency. When does each win? The answer depends on a crossover formula."

---

### Part B -- Ring vs. Tree: The Algorithm Crossover (~12 min)

**Concept**: Ring AllReduce is bandwidth-optimal but latency-poor (O(N) steps). Tree AllReduce has logarithmic latency but O(log N) bandwidth overhead. The crossover point M_crossover = N x alpha x beta determines which algorithm wins. An algorithm that is optimal at 8 GPUs can be catastrophically wrong at 1,024 GPUs.

**Prediction**: "For 256 GPUs on InfiniBand NDR, which AllReduce algorithm is faster for a 1 MB message?"

| Option | Value |
|--------|-------|
| A | Ring -- it is always bandwidth-optimal |
| **B (correct)** | **Tree -- at 1 MB and 256 GPUs, Tree's O(log N) latency advantage wins** |
| C | They are identical for this message size |
| D | Neither -- you need hierarchical AllReduce |

**Common wrong answer**: A. Students learn that Ring is "bandwidth-optimal" and assume it always wins.

**Why wrong**: At 1 MB with 256 GPUs, Ring incurs 2 x 255 = 510 latency steps (2,550 us) while Tree incurs 2 x 8 = 16 steps (80 us). The bandwidth overhead of Tree (log_2(256) = 8x) on 1 MB is only 160 us vs. Ring's 40 us. Tree total: 240 us. Ring total: 2,590 us. Tree wins by >10x. At 10 GB, Ring wins because bandwidth dominates.

**Instrument**:
- Slider: Message size (1 KB to 10 GB, log scale)
- Select: GPU count (64, 256, 1024)
- Animated time bars: Ring vs. Tree completion time side-by-side
- Chart: Transfer time vs. message size (two curves with crossover point marked)
- Crossover marker: M_crossover = N x alpha x beta (draggable to verify formula)

**mlsysim grounding**: `calc_ring_allreduce_time()` and `calc_tree_allreduce_time()` from `mlsysim.core.formulas`. Constants: `IB_NDR_LATENCY_US` (5 us), `INFINIBAND_NDR_BW_GBS` (50 GB/s).

**Transition to C**: "So algorithm choice depends on message size and GPU count. But there is another variable: the physical topology. NVLink within a node is 18x faster than InfiniBand between nodes. What happens when the algorithm is aware of this hierarchy?"

---

### Part C -- Topology and Hierarchy Effects (~12 min)

**Concept**: The NVLink-to-InfiniBand bandwidth cliff (900 GB/s vs. 50 GB/s = 18x gap) means a flat AllReduce that ignores hierarchy wastes up to 50% of training throughput. Hierarchical AllReduce (local reduce within NVLink, then global AllReduce over IB) achieves 5-6x speedup by reducing inter-node traffic by a factor of G (GPUs per node). Fat-tree bisection bandwidth determines the inter-node ceiling; oversubscription creates multiplicative slowdown.

**Prediction**: "A flat Ring AllReduce across 64 GPUs (8 nodes x 8 GPUs) mixes NVLink and InfiniBand links. A hierarchical 2-level AllReduce does local reduce first, then inter-node AllReduce. What speedup does the hierarchical approach achieve?"

| Option | Value |
|--------|-------|
| A | ~1.5x -- marginal improvement |
| B | ~2x -- moderate improvement |
| **C (correct)** | **~5-6x -- dramatic improvement** |
| D | ~18x -- full NVLink/IB ratio |

**Common wrong answer**: A or B. Students underestimate the multiplicative effect of reducing inter-node traffic.

**Why wrong**: Hierarchical AllReduce reduces the data sent over InfiniBand by 8x (one reduce per node before global AllReduce). Since inter-node communication is the bottleneck at 50 GB/s, reducing it by 8x while the intra-node reduce at 900 GB/s is nearly free yields ~5-6x total speedup.

**Instrument**:
- Select: Topology (flat ring, hierarchical 2-level, hierarchical 3-level)
- Slider: GPUs per node (2, 4, 8)
- Slider: Oversubscription ratio (1:1, 2:1, 4:1)
- Chart: AllReduce time breakdown (intra-node component vs. inter-node component)
- Metric: Effective bandwidth, speedup vs. flat ring
- Failure state: Oversubscription ratio > 1 shows proportional slowdown on inter-node component

**mlsysim grounding**: `calc_hierarchical_allreduce_time(message_bytes, n_nodes, gpus_per_node, intra_bw, inter_bw, intra_latency, inter_latency)` from `mlsysim.core.formulas`. `Nodes.DGX_H100.intra_node_bw` (900 GB/s), `Fabrics.InfiniBand_NDR.bandwidth` (400 Gbps = 50 GB/s).

**Transition to D**: "Hierarchical AllReduce is a huge win. But what if you could make the data smaller before sending it? Gradient compression promises exactly this -- but the physics of convergence fight back."

---

### Part D -- Gradient Compression: When Does It Pay Off? (~12 min)

**Concept**: Gradient compression (quantization, sparsification) trades bandwidth savings for convergence slowdown. It is only worthwhile when the communication-to-computation ratio is high. On fast networks, compression hurts because the extra convergence steps outweigh per-step savings.

**Prediction**: "You apply INT8 gradient compression (4x bandwidth reduction) to a 70B model training on 64 GPUs with InfiniBand NDR. Does total training time decrease?"

| Option | Value |
|--------|-------|
| A | Yes, by ~4x -- you saved 75% of communication |
| B | Yes, by ~2x -- significant improvement |
| **C (correct)** | **It depends -- on IB NDR, compression barely helps because the extra convergence steps nearly cancel the per-step savings** |
| D | No -- compression always hurts convergence |

**Common wrong answer**: A or B. Students assume bandwidth savings translate directly to total time savings.

**Why wrong**: INT8 compression reduces per-step communication by ~4x, but typically requires 1.1-1.5x more training steps to converge. On IB NDR (50 GB/s), where communication is already only 30-40% of step time, the net improvement is small. On slow networks (100GbE, 12.5 GB/s), where communication is 70%+ of step time, compression provides substantial benefit.

**Instrument**:
- Select: Compression method (None, FP16, INT8, Top-K 1%, 1-bit)
- Slider: Network bandwidth (10 GB/s to 100 GB/s)
- Slider: Model size (7B, 70B, 175B)
- Chart 1: Per-step waterfall (compute vs. communication bars)
- Chart 2: Total training time curve accounting for extra convergence steps
- Toggle: Error feedback on/off (shows loss plateau when feedback is disabled)

**mlsysim grounding**: Communication time from `calc_ring_allreduce_time()` with modified message_bytes. Convergence penalty modeled as multiplicative step increase (1.0x for None, 1.05x for FP16, 1.15x for INT8, 1.3x for Top-K, 1.5x for 1-bit). Network bandwidth from `Fabrics.InfiniBand_NDR.bandwidth` and `Fabrics.Ethernet_100G.bandwidth`.

**Transition to E**: "You now have four tools: algorithm choice, hierarchical decomposition, compression, and overlap. Let us put them all together and build a complete communication budget for a production 70B training job."

---

### Part E -- Communication Budget Optimization (~12 min)

**Concept**: For a 70B model on 64 GPUs over IB NDR, raw Ring AllReduce takes ~11 seconds per step. Students assemble a communication strategy by toggling optimizations one at a time and watching each chip away at the budget. The goal: reduce communication to under 20% of total step time.

**Prediction**: "Starting from the raw 11-second AllReduce, how many optimizations must you stack to get communication under 20% of step time?"

| Option | Value |
|--------|-------|
| A | Just one -- hierarchical AllReduce is enough |
| B | Two -- hierarchical + FP16 gradients |
| **C (correct)** | **Three or four -- you need hierarchical + FP16 + overlap + bucket fusion** |
| D | It is impossible on IB NDR -- you need XDR |

**Common wrong answer**: A. Students overestimate the impact of a single optimization.

**Why wrong**: Hierarchical AllReduce gives ~5-6x reduction (11s to ~2s). FP16 halves it (~1s). Overlap hides 50-85% behind backward pass. Bucket fusion reduces latency overhead. You need all of them stacked to reach <20%.

**Instrument**:
- Starting point: 11-second AllReduce (from chapter napkin math)
- Checkboxes: Hierarchical AllReduce, FP16 gradients, Bucket fusion, Backward overlap
- Chart: Stacked step-time bar updating as each optimization is toggled
- Metric: Communication as % of total step time
- Target line: 20% threshold (green when met)
- Preset: "Megatron-LM configuration" button that toggles all optimizations

**mlsysim grounding**: `calc_ring_allreduce_time()` and `calc_hierarchical_allreduce_time()` for base and hierarchical times. `DEFAULT_OVERLAP_EFFICIENCY` (0.85) from `mlsysim.core.defaults`. `INFINIBAND_NDR_BW_GBS` (50 GB/s).

---

## Lab V2-04: The Data Pipeline Wall

**Story arc**: Students discover that storage -- the least glamorous infrastructure component -- can silently determine whether a training cluster is productive or an expensive space heater. The gap between compute consumption and storage delivery has widened 60x in seven years and is getting worse. Over five parts they watch the chasm widen, learn the pipeline equation, encounter the birthday problem in shard contention, diagnose stalls that prefetching cannot fix, and finally confront the checkpoint frequency trade-off. This is the best story arc in Vol 2.

**Time budget**: 58 min (12 + 12 + 10 + 12 + 12 = 58 min)

---

### Part A -- The Storage-Compute Chasm (~12 min)

**Concept**: Accelerator throughput has grown 236x (P100 to B200) while NVMe bandwidth grew only 4x over the same period. The resulting 60x widening gap means that data pipeline engineering is a first-order concern. Faster GPUs make the storage problem worse, not better.

**Prediction**: "Your current cluster is storage-bottlenecked at 30% GPU utilization. You upgrade from A100s to H100s (2x more TFLOPS). What happens to GPU utilization?"

| Option | Value |
|--------|-------|
| A | ~60% -- faster GPUs process data faster, so utilization improves |
| B | ~30% -- no change, storage is the bottleneck |
| **C (correct)** | **~15% -- utilization drops because GPUs are faster but storage isn't** |
| D | ~5% -- catastrophic collapse |

**Common wrong answer**: A. Students assume faster GPUs always improve the system.

**Why wrong**: If the GPU processes data 2x faster but the storage delivers at the same rate, the GPU waits twice as long relative to its compute time. Utilization = T_compute / (T_compute + T_IO). Halving T_compute while keeping T_IO constant reduces utilization.

**Instrument**:
- Select: GPU generation (V100, A100, H100, B200)
- Dual-axis timeline chart: Compute throughput (TFLOPS) vs. storage bandwidth (GB/s) across generations
- Metric: Compute-to-storage bandwidth ratio (showing 60x widening)
- Gauge: GPU utilization at current storage bandwidth

**mlsysim grounding**: Hardware specs from `Hardware.V100` through `Hardware.B200` (peak_flops and memory bandwidth). Storage bandwidth via `NVME_SEQUENTIAL_BW` from constants. Compute-to-storage ratio computed as accelerator HBM bandwidth / storage bandwidth.

**Transition to B**: "The chasm is real and getting worse. So how much storage bandwidth do you actually need? The pipeline equation tells you exactly -- and the answer depends on how many GPUs you are feeding."

---

### Part B -- The Data Pipeline Equation (~12 min)

**Concept**: Required storage bandwidth = N_GPUs x U_target x S_batch / T_iteration. Under-provisioning starves accelerators; over-provisioning wastes money. Doubling GPUs without upgrading storage causes a cliff-like utilization drop.

**Prediction**: "You have a 128-GPU cluster with 80% GPU utilization, well-provisioned with NVMe storage. You double to 256 GPUs without upgrading storage. What happens to utilization?"

| Option | Value |
|--------|-------|
| A | ~80% -- utilization is independent of GPU count |
| B | ~60% -- moderate drop from increased demand |
| **C (correct)** | **~40% -- the storage bandwidth is now split across twice as many GPUs** |
| D | ~20% -- catastrophic starving |

**Common wrong answer**: A. Students do not realize that storage bandwidth is a shared resource.

**Why wrong**: BW_required doubles when GPU count doubles, but BW_available stays constant. The utilization drops proportionally.

**Instrument**:
- Slider: GPU count (8 to 1024)
- Select: Model type (vision/language -- affects batch data size)
- Slider: Target utilization (50% to 95%)
- Chart: Data Stall Frontier S-curve (GPU utilization vs. storage bandwidth)
- Metric: Required bandwidth, current bandwidth, utilization, stall %
- Failure state: Banner when utilization drops below 50%

**mlsysim grounding**: Storage bandwidth from `NVME_SEQUENTIAL_BW` in constants. GPU specs from `Hardware.H100`. Pipeline throughput formula: BW_required = N x U x S_batch / T_iter.

**Transition to C**: "Even with enough aggregate bandwidth, random access patterns create a hidden bottleneck. When hundreds of GPUs independently read dataset shards, collisions are surprisingly common. This is the birthday problem at datacenter scale."

---

### Part C -- The Shard Contention Birthday Problem (~10 min)

**Concept**: Even with many dataset shards, random access by many GPUs creates surprisingly high collision probability (birthday problem), causing tail-latency spikes that stall the entire BSP-synchronized cluster. With 64 workers and 1,000 shards, collision probability exceeds 87%.

**Prediction**: "Your 256-GPU cluster reads from a dataset with 1,000 shards. Each GPU randomly selects a shard at the start of each step. What is the probability that at least two GPUs collide on the same shard?"

| Option | Value |
|--------|-------|
| A | ~10% -- 1,000 shards is plenty for 256 workers |
| B | ~50% -- borderline |
| **C (correct)** | **~100% (near certainty) -- collisions are essentially guaranteed** |
| D | ~75% -- high but not certain |

**Common wrong answer**: A. Students think 1,000 shards / 256 workers = ~4 shards per worker, so collisions should be rare.

**Why wrong**: P(collision) = 1 - e^(-n^2 / 2N). With n=256 and N=1,000: exponent = -256^2 / 2000 = -32.8. P = 1 - e^(-32.8) = ~100%. The birthday problem strikes at n = sqrt(N), which is 32 -- far below 256.

**Instrument**:
- Slider: GPU workers (8 to 256)
- Slider: Dataset shards (100 to 10,000)
- Animation: Workers selecting shards with collisions highlighted in red
- Probability meter: Theoretical collision probability
- Toggle: Random vs. deterministic shard assignment (shows collisions drop to zero)

**mlsysim grounding**: Birthday collision formula from textbook. Shard counts and worker counts grounded in typical ImageNet/C4 dataset partitioning.

**Transition to D**: "Collisions create tail latency. But even without contention, can prefetching eliminate stalls entirely? Only if I/O time is shorter than compute time. Let us check."

---

### Part D -- The Data Stall Diagnostic (~12 min)

**Concept**: Pipelining and prefetching can hide storage latency, but only when I/O time does not exceed compute time. When I/O exceeds compute, no amount of overlap eliminates the stall. Without pipelining: T_step = T_IO + T_compute. With pipelining: T_step = max(T_compute, T_IO).

**Prediction**: "Your training step has 200 ms compute and 300 ms I/O. You add prefetching with depth 4. Does the stall disappear?"

| Option | Value |
|--------|-------|
| A | Yes -- 4 batches of prefetch hide the 300 ms I/O |
| **B (correct)** | **No -- stall drops from 60% to 33% but never reaches zero because I/O > compute** |
| C | Partially -- stall drops to ~10% |
| D | No effect -- prefetching only helps random access patterns |

**Common wrong answer**: A. Students believe prefetching can always hide I/O latency.

**Why wrong**: With perfect pipelining, T_step = max(T_compute, T_IO) = max(200, 300) = 300 ms. Stall = (300 - 200) / 300 = 33%. The only fix is faster storage or slower compute (i.e., the pipeline can only hide I/O when T_IO < T_compute).

**Instrument**:
- Slider: Compute time (100-500 ms)
- Slider: I/O time (50-1000 ms)
- Slider: Prefetch buffer depth (0 to 8 batches)
- Timeline animation: Pipeline execution showing overlap (compute green, I/O wait red)
- Gauge: Stall percentage
- Metric: Effective step time, utilization

**mlsysim grounding**: Formulas from textbook: T_step_sequential = T_IO + T_compute; T_step_pipelined = max(T_compute, T_IO); Stall% = (T_step - T_compute) / T_step.

**Transition to E**: "I/O is not just about feeding the GPUs -- it is also about saving progress. Every checkpoint is a massive write that competes with training data reads. How often should you save?"

---

### Part E -- Checkpoint Economics (~12 min)

**Concept**: Checkpoint frequency trades recovery granularity against I/O overhead. Too frequent: checkpointing steals storage bandwidth from training. Too infrequent: failures waste millions in recomputation. The optimal frequency depends on MTBF (provided directly as a parameter, not requiring V2-01 completion) and checkpoint write time.

**Prediction**: "A 1,000-GPU cluster has MTBF of 5 hours (given). Checkpoints for a 175B model take 2 minutes to write. What is the optimal checkpoint interval?"

| Option | Value |
|--------|-------|
| A | Every 5 minutes -- minimize lost work |
| **B (correct)** | **Every ~27 minutes -- the Young-Daly sweet spot** |
| C | Every hour -- minimize I/O overhead |
| D | Every 2 hours -- checkpoints are expensive |

**Common wrong answer**: A. Students prioritize minimizing lost work without considering I/O cost.

**Why wrong**: Young-Daly optimal interval = sqrt(2 x T_write x MTBF) = sqrt(2 x 120s x 18000s) = sqrt(4,320,000) = ~2,078s = ~35 minutes. (Exact value depends on parameter choices; ~27 min for slightly different MTBF.) Too-frequent checkpointing saturates storage bandwidth.

**Instrument**:
- Slider: Cluster size (determines MTBF; or direct MTBF slider, default 5 hours)
- Slider: Checkpoint write time (30s to 5 min; depends on model size and storage BW)
- Slider: Checkpoint interval (1 min to 2 hours)
- Chart: U-shaped waste curve with three components: checkpoint overhead (decreasing), expected rework (increasing), total (U-curve)
- Metric: Optimal interval (Young-Daly), waste %, dollar cost per day
- Failure state: Banner when checkpoint write time exceeds interval

**mlsysim grounding**: `calc_young_daly_interval(checkpoint_cost_s, mtbf_s)` from `mlsysim.core.formulas`. `calc_checkpoint_size(n_params, bytes_per_param=14)` for 175B model. `calc_mtbf_cluster(GPU_MTTF_HOURS, N)` for deriving MTBF. Note: MTBF is provided as a given value (5 hours for 1,000 GPUs) so this lab does not depend on V2-01 completion. Forward reference to V2-06 (Fault Tolerance) for deeper treatment.

---

## Lab V2-05: The Parallelism Puzzle

**Story arc**: Students scale from a single GPU to a 1,024-GPU cluster by trying each parallelism strategy in sequence and discovering that each one solves one constraint while creating another. Data parallelism hits a communication wall. ZeRO trades communication for memory. Pipeline parallelism creates bubbles. Extended Amdahl's Law reveals the cost-efficiency optimum. Finally, 3D parallelism maps strategies to the bandwidth hierarchy. The Conservation of Overhead governs every choice.

**Time budget**: 60 min (12 + 12 + 12 + 12 + 12 = 60 min)

---

### Part A -- The Communication Wall (~12 min)

**Concept**: Data parallelism scales throughput linearly only until gradient synchronization dominates step time. Ring AllReduce communication overhead grows relative to shrinking per-GPU compute, creating a "Communication Wall." Even with InfiniBand, communication consumes 40%+ of step time for a 175B model at 256 GPUs.

**Prediction**: "You train a 175B model with pure data parallelism on 256 GPUs with InfiniBand NDR. What scaling efficiency do you achieve?"

| Option | Value |
|--------|-------|
| A | ~90% -- InfiniBand handles the gradients |
| B | ~70% -- moderate overhead |
| **C (correct)** | **~50-55% -- communication is nearly half of step time** |
| D | ~25% -- communication dominates |

**Common wrong answer**: A. Students who completed V2-03 may have better calibration, but many still overestimate.

**Why wrong**: 175B FP16 gradients = 350 GB. Ring AllReduce: 2(255/256) x 350 GB / 50 GB/s = ~14 seconds. If compute per GPU is ~10 seconds, efficiency = 10 / (10 + 14) = ~42%. With overlap, ~55%.

**Instrument**:
- Slider: GPU count (1 to 512, powers of 2)
- Select: Model size (1B, 7B, 70B, 175B)
- Toggle: Interconnect (100G Ethernet, IB HDR, IB NDR)
- Chart: Speedup (linear vs. actual) and efficiency gauge
- Metric: Communication fraction, step time breakdown

**mlsysim grounding**: `calc_ring_allreduce_time()` from formulas. `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100)` for per-GPU compute time. `Fabrics.InfiniBand_NDR` for network params. `SCALING_EFF_256GPU` (0.70) as reference.

**Transition to B**: "Communication is expensive because every GPU holds a full copy of the model. What if you sharded the model across GPUs so each holds only a fraction? That is ZeRO -- but it trades memory savings for more communication."

---

### Part B -- ZeRO: Trading Communication for Memory (~12 min)

**Concept**: ZeRO optimization shards optimizer states, gradients, and parameters across workers. Each stage reduces per-GPU memory but increases communication volume, embodying the Conservation of Overhead. Even ZeRO-3 on 64 A100s cannot train a 175B model because activation memory pushes total past 80 GB.

**Prediction**: "Can ZeRO-3 on 64 A100 GPUs (80 GB each) train a 175B model?"

| Option | Value |
|--------|-------|
| A | Yes -- ZeRO-3 shards everything across 64 GPUs |
| B | Yes, but only with FP16 precision |
| **C (correct)** | **No -- activation memory (~50 GB) pushes per-GPU total past 80 GB even with ZeRO-3** |
| D | No -- 64 GPUs is not enough for any ZeRO stage |

**Common wrong answer**: A. Students compute static memory only: 175B x 14 bytes / 64 = 38 GB, well within 80 GB.

**Why wrong**: ZeRO-3 shards static memory (parameters + gradients + optimizer) to 38 GB per GPU. But activations are NOT sharded -- each GPU stores its own activations for the micro-batch. At seq_len=2048, batch_size=1, activations ~50 GB for 175B model. Total = 38 + 50 = 88 GB > 80 GB HBM.

**Instrument**:
- Slider: Model size (1B to 175B)
- Slider: Number of GPUs (8 to 256)
- Select: ZeRO stage (0, 1, 2, 3)
- Stacked bar: Per-GPU memory (parameters, optimizer, gradients, activations) at each stage
- Second chart: Communication volume per step at each stage
- HBM capacity line (80 GB for A100, 80 GB for H100)
- Failure state: OOM banner when total exceeds HBM

**mlsysim grounding**: `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100, is_training=True, zero_stage=3, dp_size=64)` returns `memory_footprint` and `feasible`. Activation memory via `calc_activation_memory()`.

**Transition to C**: "ZeRO cannot do it alone for frontier models. Pipeline parallelism takes a different approach: split the model into stages across GPUs. But idle GPUs create 'bubbles' that waste compute."

---

### Part C -- Pipeline Bubbles: The Idle GPU Problem (~12 min)

**Concept**: Pipeline parallelism overlaps computation across model stages using microbatches, but fill and drain phases create idle GPU time (bubbles). Bubble fraction = (P-1)/(P+M-1). With 4 stages and 4 microbatches, bubble fraction = 43%. Reducing bubbles below 10% requires very large numbers of microbatches.

**Prediction**: "With 8 pipeline stages, how many microbatches do you need to keep bubble fraction below 10%?"

| Option | Value |
|--------|-------|
| A | 8 -- one per stage |
| B | 16 -- twice the stages |
| C | 32 -- four times the stages |
| **D (correct)** | **72+ microbatches -- far more than intuition suggests** |

**Common wrong answer**: A or B. Students think matching microbatches to stages is sufficient.

**Why wrong**: Bubble fraction = (P-1)/(P+M-1). For P=8 and target <10%: 7/(7+M) < 0.10, so M > 63. With M=72: bubble = 7/78 = 9%. This implies a very large global batch size (72 x micro_batch_size), which may hurt convergence.

**Instrument**:
- Select: Pipeline stages (2, 4, 8, 16)
- Slider: Microbatches per step (1 to 64)
- Animated Gantt chart: Forward and backward passes flowing through stages, bubbles in red
- Gauge: Bubble percentage
- Metric: Global batch size implied, per-GPU utilization

**mlsysim grounding**: `calc_pipeline_bubble(n_stages, n_microbatches)` from `mlsysim.core.formulas`. `OVERHEAD_PIPELINE_BUBBLE` (0.05) as reference for well-tuned systems.

**Transition to D**: "Remember the scaling tax from V2-01 Part D? Now let us apply it with real parallelism strategies. Each workload type has a different communication fraction, and that determines where the cost-efficiency optimum falls."

---

### Part D -- The Scaling Tax: Amdahl Meets Communication (~12 min)

**Concept**: Distributed training obeys an extended Amdahl's Law where the serial fraction includes communication overhead. The cost-efficiency optimal point differs dramatically by workload type. A bandwidth-bound embedding model hits diminishing returns at just 4 GPUs, while a compute-bound ResNet scales to 128+.

**Prediction**: "For a bandwidth-bound DLRM embedding model (r = 0.50), at what GPU count does cost-per-sample reach its minimum?"

| Option | Value |
|--------|-------|
| A | ~64 GPUs -- standard cluster size |
| B | ~16 GPUs -- moderate scale |
| **C (correct)** | **~4 GPUs -- essentially single-node** |
| D | ~1 GPU -- never parallelize |

**Common wrong answer**: A. Students assume all models benefit equally from parallelism.

**Why wrong**: At r = 0.50, maximum theoretical speedup = 1/r = 2x regardless of GPU count. The cost-efficiency optimum is where the marginal speedup gain equals the marginal GPU cost, which is ~4 GPUs for r=0.50.

**Instrument**:
- Select: Workload type with preset r values:
  - Compute-bound ResNet (r = 0.05)
  - Balanced LLM (r = 0.20)
  - Bandwidth-bound DLRM (r = 0.50)
- Slider: GPU count (1 to 512)
- Toggle: Communication-computation overlap (0%, 50%, 80%)
- Chart 1: Speedup vs. ideal (log-log)
- Chart 2: Cost per sample vs. GPU count (U-curve with optimum marked)

**mlsysim grounding**: Extended Amdahl's: T_step(N) = T_compute/N + T_comm(N) - T_overlap. Cost = N x T_step(N) x GPU_hourly_rate. GPU cost from `Hardware.H100.unit_cost`. Communication fraction from `calc_ring_allreduce_time()`.

**Transition to E**: "No single parallelism strategy works alone. Production training combines tensor, pipeline, and data parallelism -- each mapped to the bandwidth hierarchy you explored in V2-02 Part C. Now you design that mapping."

---

### Part E -- 3D Parallelism: The Hierarchy-Aware Design (~12 min)

**Concept**: Production training combines TP (within NVLink), PP (across nearby nodes on IB), and DP (across all nodes). The constraint is TP x PP x DP = total GPUs. The correct mapping to the bandwidth hierarchy can yield 2-3x efficiency improvement over naive DP.

**Prediction**: "For a 175B model on 256 H100 GPUs, which 3D configuration maximizes training efficiency?"

Students choose from 3-4 preset configurations to reduce cognitive load:

| Config | TP | PP | DP | Notes |
|--------|----|----|-----|-------|
| A | 1 | 1 | 256 | Pure DP |
| B | 8 | 1 | 32 | TP within node + DP |
| **C (correct)** | **8** | **4** | **8** | **Full 3D parallelism** |
| D | 8 | 32 | 1 | TP + aggressive PP |

**Common wrong answer**: A (pure DP) or B (TP + DP). Students either default to the simplest strategy or forget pipeline parallelism.

**Why wrong**: Pure DP requires AllReduce of 350 GB FP16 gradients across 256 GPUs -- prohibitively slow. TP=8 within NVLink is fast, but DP=32 still requires large AllReduce over IB. Adding PP=4 reduces the DP degree to 8, shrinking the AllReduce to 8 nodes while PP's small activation transfers (200 MB) tolerate IB latency.

**Instrument**:
- Select: TP degree (1, 2, 4, 8)
- Select: PP degree (1, 2, 4, 8, 16, 32)
- Auto-calculated: DP = 256 / (TP x PP)
- Topology diagram: Physical mapping of TP/PP/DP to nodes and racks
- Metrics: Per-GPU memory, communication volume per step, pipeline bubble fraction, scaling efficiency
- Preset buttons: 3-4 configurations (listed above) to reduce cognitive load

**mlsysim grounding**: `Engine.solve(model=Models.Llama3_70B, hardware=Hardware.H100, is_training=True, zero_stage=3, dp_size=DP)` for memory. `calc_ring_allreduce_time()` for DP communication. `calc_pipeline_bubble(PP, microbatches)` for PP overhead. `Nodes.DGX_H100.intra_node_bw` (900 GB/s) for TP bandwidth.

---

## Lab V2-06: When Failure is Routine

**Story arc**: Students confront the reality that at fleet scale, failure is not an exception but a statistical certainty. They briefly recall the exponential reliability collapse (from V2-01), then spend the bulk of the lab on the Young-Daly checkpoint optimization (the U-shaped cost curve), the checkpoint storm problem, serving fault tolerance (where millisecond recovery is required), and the reliability budget trade-off. The Young-Daly sweet spot is one of the best parts in the entire suite -- protect it.

**Time budget**: 56 min (6 + 14 + 12 + 14 + 10 = 56 min)

---

### Part A -- Failure as Routine: A Recall (~6 min)

**Concept**: Rapid recall of the reliability collapse from V2-01 Part A, grounded in this chapter's MTBF equation: MTBF_system = MTBF_component / N. A 10,000-GPU cluster with 50,000-hour GPU MTTF experiences a failure every 5 hours. This is a brief warm-up, NOT a re-teach.

**Prediction**: "A 10,000-GPU cluster uses GPUs with MTBF of 50,000 hours. Approximately how often does the cluster experience a failure?"

| Option | Value |
|--------|-------|
| A | Once a week |
| B | Once a day |
| **C (correct)** | **Every ~5 hours** |
| D | Every ~30 minutes |

**Common wrong answer**: A or B. Even students who saw V2-01 may not remember the exact math.

**Why wrong**: MTBF_cluster = 50,000 / 10,000 = 5 hours. At this rate, a 30-day training run will experience ~144 failures.

**Instrument**:
- Slider: Cluster size (100 to 25,000 GPUs)
- Metric: System MTBF in hours, expected failures per training day/week/month
- Chart: Probability of surviving T hours without failure (exponential decay curve)

**mlsysim grounding**: `calc_mtbf_cluster(GPU_MTTF_HOURS, N)` from `mlsysim.core.formulas`. `GPU_MTTF_HOURS` = 50,000 from `mlsysim.core.defaults`.

**Transition to B**: "Failures every 5 hours mean you must save your progress regularly -- checkpoint. But checkpoint too often and you waste time writing. Checkpoint too rarely and you lose days of work. Where is the sweet spot?"

---

### Part B -- The Young-Daly Sweet Spot (~14 min)

**Concept**: The optimal checkpoint interval tau_opt = sqrt(2 x T_write x MTBF) minimizes total wasted work (checkpoint overhead + rework from failures). The U-shaped cost curve has a clear minimum. This is one of the best pedagogical moments in the entire lab suite.

**Prediction**: "A 16,000-GPU cluster has MTBF of ~3 hours. Checkpoint writes take 2 minutes. What is the optimal checkpoint interval?"

| Option | Value |
|--------|-------|
| A | Every 2 minutes -- match the write time |
| B | Every 10 minutes -- frequent saves |
| **C (correct)** | **Every ~27 minutes -- the square-root law** |
| D | Every 90 minutes -- halfway to MTBF |

**Common wrong answer**: B. Students either checkpoint too aggressively (afraid of failures) or aim for the midpoint of MTBF.

**Why wrong**: tau_opt = sqrt(2 x 120s x 10,800s) = sqrt(2,592,000) = 1,610s = ~27 minutes. The square root law means the optimal interval is geometrically between write time and MTBF, not linearly.

**Instrument**:
- Slider: Cluster size (determines MTBF via MTBF = GPU_MTTF / N)
- Slider: Checkpoint write time (10s to 5 min)
- Slider: Checkpoint interval (draggable, 1 min to 3 hours)
- Chart: U-shaped waste curve with three visible components:
  - Checkpoint overhead (decreasing hyperbola in blue)
  - Expected rework (increasing line in red)
  - Total waste (U-curve in black, minimum marked)
- Metric: Optimal interval, total waste %, dollar cost of waste per day
- Annotation: Young-Daly formula on chart

**mlsysim grounding**: `calc_young_daly_interval(checkpoint_cost_s, mtbf_s)` from `mlsysim.core.formulas`. `calc_mtbf_cluster(GPU_MTTF_HOURS, N)` for MTBF derivation. GPU cost from `Hardware.H100.unit_cost` for dollar cost of waste.

**Transition to C**: "The Young-Daly formula gives the optimal interval. But it assumes checkpointing is free once you start -- it is not. Writing a 175B checkpoint to storage takes time, bandwidth, and money, and at scale, all GPUs write simultaneously."

---

### Part C -- The Checkpoint Storm (~12 min)

**Concept**: When thousands of GPUs write checkpoints simultaneously, storage saturates. The "stop-the-world" cost scales with model size and cluster size. If storage bandwidth is too low, checkpoint write time exceeds the optimal interval, creating a pathological state where the system spends more time checkpointing than computing.

**Prediction**: "A 175B model checkpoints on a 1,000-GPU cluster with NFS storage (1 GB/s aggregate write). How long does one checkpoint take?"

| Option | Value |
|--------|-------|
| A | ~10 seconds -- fast with modern storage |
| B | ~2 minutes -- manageable |
| **C (correct)** | **~41 minutes -- longer than the Young-Daly optimal interval** |
| D | ~5 minutes -- within budget |

**Common wrong answer**: A or B. Students underestimate checkpoint size.

**Why wrong**: 175B params x 14 bytes (weights + Adam states) = 2.45 TB per checkpoint. At 1 GB/s NFS: 2,450 seconds = ~41 minutes. With MTBF of ~5 hours, the optimal interval is ~27 minutes. The checkpoint takes LONGER than the optimal interval -- the system is in a pathological state.

**Instrument**:
- Slider: Model size (1B to 175B)
- Select: Storage type (NFS 1 GB/s / Parallel FS 10 GB/s / NVMe RAID 100 GB/s)
- Slider: GPU count
- Metrics: Checkpoint size (TB), write time, dollar cost per checkpoint, daily checkpoint cost
- Failure state: Banner when write time > Young-Daly optimal interval

**mlsysim grounding**: `calc_checkpoint_size(n_params, bytes_per_param=14)` from `mlsysim.core.formulas`. Storage bandwidth from `CHECKPOINT_WRITE_BW_GBS` (100 GB/s default) in `mlsysim.core.defaults`, with overrides for NFS/parallel FS.

**Transition to D**: "Training fault tolerance is about saving progress and restarting. Serving fault tolerance is fundamentally different: failures must be invisible to users, with millisecond recovery. The strategies are different too."

---

### Part D -- Graceful Degradation in Serving (~14 min)

**Concept**: Serving fault tolerance differs from training: failures must be invisible to users with millisecond-scale recovery. Strategies include model fallback (smaller model), feature fallback (drop expensive features), and load shedding. For LLM serving, simple request redirection fails because KV cache state is lost on the failed replica.

**Prediction**: "An LLM serving replica fails mid-generation. You redirect the in-progress request to another replica. What happens?"

| Option | Value |
|--------|-------|
| A | Seamless recovery -- the user notices nothing |
| B | Brief pause (~100 ms) while the new replica catches up |
| **C (correct)** | **The request must restart from scratch -- KV cache is lost, doubling latency** |
| D | The request fails with an error |

**Common wrong answer**: A or B. Students from web-service backgrounds expect stateless redirection to work.

**Why wrong**: LLM generation is stateful -- the KV cache stores all context computed so far. When a replica fails, this state is lost. The new replica must re-process the entire prompt (prefill) before resuming generation. For a 4K-token context, this adds seconds of latency.

**Instrument**:
- Configure: Replica count (2 to 16), failure rate, SLO budget (P99 latency)
- Failure injection: Toggle to kill a random replica
- Strategy selector: Redirect / Fallback model (smaller) / Load shed
- Live dashboard: P99 latency, accuracy/quality, request success rate
- Metric: KV cache reconstruction time for redirect strategy

**mlsysim grounding**: `calc_availability_stacked(single_availability, n_replicas)` from `mlsysim.core.formulas`. Serving latency decomposition from `Engine.solve()` for LLM decode latency. KV cache size from `calc_kv_cache_size()`.

**Transition to E**: "You have two domains of fault tolerance: training (checkpoint frequency) and serving (replica count). Both cost GPUs. With a fixed GPU budget, how should you allocate between productive compute and fault tolerance overhead?"

---

### Part E -- The Reliability Budget (~10 min)

**Concept**: Fault tolerance investment has diminishing returns. The economic framework balances the cost of redundancy against the cost of downtime, with different optimal points for training and serving. Larger clusters require proportionally more fault tolerance investment, creating a "reliability tax."

**Prediction**: "You have a 1,024-GPU budget. How many GPUs should be dedicated to fault tolerance overhead (spare capacity, checkpointing bandwidth, replicas)?"

| Option | Value |
|--------|-------|
| A | ~2% (20 GPUs) -- minimize overhead |
| **B (correct)** | **~10-15% (100-150 GPUs) -- the diminishing-returns sweet spot** |
| C | ~30% (300 GPUs) -- safety first |
| D | ~50% (512 GPUs) -- maximum reliability |

**Common wrong answer**: A. Students want to maximize productive compute.

**Why wrong**: At 2% overhead, the cost of failures (recomputation, downtime) far exceeds the savings. At 30%, most additional reliability provides negligible benefit. The knee is ~10-15% where marginal reliability gain equals marginal compute cost.

**Instrument**:
- Slider: Total GPU budget (256 to 8,192)
- Slider: Fault tolerance allocation (0% to 50%)
- Chart: Pareto curve (effective throughput vs. reliability)
- Metric: Productive GPUs, fault tolerance GPUs, effective TFLOPS, expected uptime %
- Optimal point marker

**mlsysim grounding**: `OVERHEAD_CHECKPOINT` (0.03), `OVERHEAD_FAILURE_RECOVERY` (0.10), `OVERHEAD_MAINTENANCE` (0.05) from `mlsysim.core.defaults`. `calc_effective_flops(peak_flops, mfu, scaling_eff, goodput_ratio)` for net throughput.

---

## Lab V2-07: The Scheduling Trap

**Story arc**: Students discover that scheduling GPUs is fundamentally harder than scheduling CPUs because of heavy-tailed job distributions, multi-dimensional packing constraints, topology sensitivity, and the impossible trade-off between utilization, fairness, and latency. Over four parts (reduced from five -- deadlock simulation dropped as too complex and OS-specific), they encounter the queuing wall, the allocation problem, topology-aware placement, and the utilization paradox.

**Time budget**: 48 min (10 + 14 + 12 + 12 = 48 min)

---

### Part A -- The Queuing Wall (~10 min)

**Concept**: ML workloads have heavy-tailed duration distributions (coefficient of variation C_s = 3-5) that make queue wait times explode at utilizations where web servers feel responsive. At 80% utilization, ML queue wait is 5x worse than uniform workloads. This is the strongest part of the lab.

**Prediction**: "Your GPU cluster runs at 80% utilization. Web service engineers say 80% is comfortable. What is the average queue wait time for an ML job?"

| Option | Value |
|--------|-------|
| A | ~5 minutes -- similar to web service queuing |
| **B (correct)** | **~25 minutes -- 5x worse than uniform workloads** |
| C | ~1 hour -- significant delay |
| D | ~2 minutes -- GPUs are fast |

**Common wrong answer**: A. Students from web-service backgrounds assume 80% utilization is normal.

**Why wrong**: The Pollaczek-Khinchine formula: W_q = (rho / (1-rho)) x ((1 + C_s^2) / (2 x mu)). For C_s = 3 (ML heavy tail), the (1 + C_s^2)/2 factor = 5, making wait times 5x worse than uniform (C_s = 1). The heavy tail means rare but massive training jobs block hundreds of short experiments.

**Instrument**:
- Slider: Cluster utilization (0% to 99%)
- Toggle: Workload type (Uniform C_s=1 vs. ML C_s=3 vs. Research C_s=5)
- Animation: Queue depth showing jobs arriving and being served, with heavy-tail jobs visually large
- Chart: Wait time vs. utilization for each workload type (diverging curves)
- Metric: Average wait, P99 wait, queue depth

**mlsysim grounding**: `calc_queue_latency_mmc()` from `mlsysim.core.formulas` for baseline. Heavy-tail correction via C_s coefficient. Arrival rate and service rate calibrated from `AVERAGE_RESEARCHER_JOB_DAYS` (2.0 days) and `TARGET_CLUSTER_UTILIZATION` (0.80) in `mlsysim.core.defaults`.

**Transition to B**: "The heavy tail makes queuing painful. But even when a job reaches the front of the queue, it might not run. GPU, CPU, memory, and topology constraints create a multi-dimensional packing problem where the cluster has free GPUs but cannot schedule any pending job."

---

### Part B -- The Allocation Problem (~14 min)

**Concept**: This part merges the fragmentation and gang scheduling problems. Multi-dimensional bin packing with GPU, CPU, memory, and topology constraints creates fragmentation: 30% of GPUs can be free but unusable because they are scattered across nodes. Gang scheduling (all-or-nothing allocation) prevents deadlock but increases fragmentation by requiring contiguous blocks. The combined effect: effective capacity is far less than physical capacity.

**Prediction**: "Your 256-GPU cluster shows 30% free capacity (77 GPUs idle). A researcher submits a 64-GPU training job. Can it be scheduled?"

| Option | Value |
|--------|-------|
| A | Yes immediately -- 77 > 64 idle GPUs |
| **B (correct)** | **No -- the 77 idle GPUs are scattered across 12 nodes in fragments of 1-4 GPUs each, and the job requires 8 contiguous 8-GPU nodes** |
| C | Yes, but with 50% reduced performance due to fragmentation |
| D | Yes, after a brief 5-minute repack |

**Common wrong answer**: A. Students see 77 > 64 and assume scheduling is trivial.

**Why wrong**: Gang scheduling requires all 64 GPUs allocated simultaneously. Topology-aware placement requires them in contiguous nodes (8 per node). With fragments of 1-4 GPUs scattered across nodes, no contiguous block of 64 exists. This is the fragmentation tax: physical capacity != effective capacity.

**Instrument**:
- Cluster heatmap: 32 nodes x 8 GPUs, showing occupied (blue) and free (gray) GPUs
- Job queue: Jobs of varying sizes (1, 2, 4, 8, 64 GPUs) with arrival times
- Scheduling heuristic toggle: First-fit / Best-fit / First-fit-decreasing
- Metric: Effective capacity, fragmentation ratio (stranded GPUs / total), largest contiguous block
- Toggle: Gang scheduling on/off (shows deadlock when off, fragmentation when on)

**mlsysim grounding**: Cluster topology from `Clusters.Research_256` (32 nodes x 8 GPUs). `Nodes.DGX_H100.accelerators_per_node` (8) defines the packing unit.

**Transition to C**: "Even when a contiguous block exists, where you place the job matters enormously. Random placement across racks can degrade throughput by 30-50% compared to topology-aware placement. Let us see why."

---

### Part C -- Topology-Aware Placement (~12 min)

**Concept**: Random GPU placement across a datacenter can degrade training throughput by 30-50% compared to topology-aware placement. The NVLink-to-InfiniBand bandwidth cliff (18x from V2-03) compounds at every communication step. Placement alone, with zero code changes, can match the impact of an algorithmic optimization.

**Prediction**: "You place a 64-GPU training job randomly across the cluster vs. topology-optimally (all within 8 adjacent nodes on the same rack). What speedup does optimal placement achieve?"

| Option | Value |
|--------|-------|
| A | ~1.1x -- placement barely matters |
| B | ~1.5x -- moderate improvement |
| C | ~2x -- significant |
| **D (correct)** | **~3-5x -- placement matches the impact of a major algorithmic optimization** |

**Common wrong answer**: A. Students assume that with InfiniBand everywhere, placement is irrelevant.

**Why wrong**: Random placement crosses rack boundaries, adding hops and hitting the spine oversubscription ratio. With 2:1 oversubscription at spine level, cross-rack AllReduce takes 2x longer than intra-rack. With 3-hop paths vs. 1-hop, latency triples. For TP-heavy workloads, the NVLink/IB cliff (18x) between intra-node and inter-node further amplifies the difference.

**Instrument**:
- Topology visualization: Nodes within racks, racks connected by spine switches
- Placement toggle: Random / Rack-aware / Topology-optimal
- Slider: Job size (8 to 256 GPUs)
- Metric: AllReduce latency, training throughput (samples/sec)
- Congestion heatmap: Shows traffic hot spots for each placement strategy

**mlsysim grounding**: Bandwidth hierarchy from `Nodes.DGX_H100.intra_node_bw` (900 GB/s NVLink), `Fabrics.InfiniBand_NDR.bandwidth` (50 GB/s IB NDR). Cross-rack penalty modeled via `NetworkFabric.oversubscription_ratio`. `calc_hierarchical_allreduce_time()` for communication time at different topological placements.

**Transition to D**: "You now understand that queuing, fragmentation, and placement all constrain scheduling. But here is the trap: optimizing one metric (utilization, fairness, or latency) necessarily hurts the others. You cannot make all stakeholders happy simultaneously."

---

### Part D -- The Utilization Paradox (~12 min)

**Concept**: Maximizing GPU utilization, fairness, job latency, and cost efficiency simultaneously is impossible. Every scheduling policy represents a trade-off point. The conflict between throughput (favor large jobs) and latency (favor small jobs) is the central tension. This is the synthesis of the lab.

**Prediction**: "You operate a shared GPU cluster. Can you achieve >90% utilization AND keep average wait time under 10 minutes AND ensure fair access across 5 research teams?"

| Option | Value |
|--------|-------|
| A | Yes -- a good scheduler can do all three |
| **B (correct)** | **No -- these goals are fundamentally in conflict; improving one degrades another** |
| C | Yes, but only with preemption |
| D | Yes, but only at 50% utilization |

**Common wrong answer**: A. Students believe scheduling is a solved problem from operating systems courses.

**Why wrong**: High utilization requires keeping GPUs busy, which means running large jobs that block queues. Low latency requires running small jobs first, which fragments the cluster and reduces large-job throughput. Fairness requires equal access, which may starve the most productive teams. Preemption helps but adds recomputation cost (lost work since last checkpoint).

**Instrument**:
- Sliders: Priority weight for throughput vs. fairness vs. latency (3 sliders summing to 100%)
- Job queue: Mixed workload (one 512-GPU month-long run + hundreds of 8-GPU 1-hour experiments)
- Live dashboard with 4 metrics: Cluster utilization, average wait time, max wait time, Jain's fairness index
- Color coding: Green when metric meets target, red when violated
- Key insight: It is impossible to turn all 4 metrics green simultaneously

**mlsysim grounding**: Queuing model from `calc_queue_latency_mmc()`. GPU costs from `Hardware.H100.unit_cost` for dollar cost of preemption. `AVERAGE_RESEARCHER_JOB_DAYS` and `TARGET_CLUSTER_UTILIZATION` from defaults for calibrating arrival rates.

---

# Cross-Lab Reference Map

| New Number | Old Number(s) | Title | Key mlsysim Functions |
|------------|--------------|-------|----------------------|
| V2-01 | V2-01 | The Scale Illusion | `calc_failure_probability`, `calc_ring_allreduce_time`, `Engine.solve` |
| V2-02 | V2-02 | The Compute Infrastructure Wall | `Engine.solve`, `Hardware.*`, `Nodes.DGX_H100`, `calc_fleet_tco` |
| V2-03 | V2-03 + V2-06 | Communication at Scale | `calc_ring_allreduce_time`, `calc_tree_allreduce_time`, `calc_hierarchical_allreduce_time`, `Fabrics.InfiniBand_NDR` |
| V2-04 | V2-04 | The Data Pipeline Wall | `calc_young_daly_interval`, `calc_checkpoint_size`, `calc_mtbf_cluster` |
| V2-05 | V2-05 | The Parallelism Puzzle | `Engine.solve(zero_stage=3, dp_size=64)`, `calc_pipeline_bubble`, `calc_ring_allreduce_time` |
| V2-06 | V2-07 | When Failure is Routine | `calc_young_daly_interval`, `calc_mtbf_cluster`, `calc_checkpoint_size`, `calc_kv_cache_size`, `calc_availability_stacked` |
| V2-07 | V2-08 | The Scheduling Trap | `calc_queue_latency_mmc`, `Clusters.Research_256`, `Fabrics.*` |

# Dropped Content (with rationale)

| Content | Was in | Why dropped |
|---------|--------|------------|
| RDMA deep-dive (Go-Back-N, GPUDirect) | Old V2-03 Part B | Too specialized; protocol internals are not durable knowledge |
| Rail-optimized topology | Old V2-03 Part D | Niche; applies only to specific vendor configurations |
| Deadlock simulation | Old V2-08 Part C | Too complex for 12 min; too OS-specific; merged into Part B |
| Redundant alpha-beta intro | Old V2-06 Part A | Merged with V2-03 Part A (single treatment) |
| Separate bandwidth hierarchy | Old V2-06 Part C | Covered by V2-03 Part C (topology + hierarchy in one part) |

# Dependency Chain

```
V2-01 (Scale Illusion)
  |-- V2-02 (Compute Infrastructure) [builds on reliability + efficiency concepts]
  |     |-- V2-03 (Communication at Scale) [builds on bandwidth staircase]
  |           |-- V2-05 (Parallelism Puzzle) [builds on AllReduce + hierarchy]
  |-- V2-04 (Data Pipeline Wall) [independent, references V2-01 MTBF as given]
  |     |-- V2-06 (Fault Tolerance) [forward-referenced by V2-04 Part E]
  |-- V2-07 (Scheduling Trap) [references V2-03 bandwidth for topology placement]
```


# Volume 2 Labs V2-08 through V2-16: Final Cleaned Plans

Post-merger numbering. Each lab specifies a 5-part story arc that maps onto the 2-Act implementation structure (Parts A-B = Act 1 Calibration; Parts C-E = Act 2 Design Challenge). Total target per lab: 55-60 minutes across all parts. Each part has a prediction, instrument, mlsysim grounding, and an explicit narrative transition to the next part.

Cross-references use `@sec-` notation per editorial standards. All formulas and thresholds trace to chapter content.

---

## Lab V2-08: The Inference Economy
**Chapter**: `inference.qmd`
**Story arc**: You trained a 70B model for $2M. Congratulations -- that was the cheap part. Students discover that serving cost eclipses training cost within weeks, that the KV cache memory wall (not compute) is the binding constraint on concurrent serving, and that continuous batching transforms a stop-and-go assembly line into a flowing pipeline. The lab ends with a fleet design challenge where quantization, batching, and replica count interact non-linearly under a latency SLO.
**Time budget**: 10 + 12 + 14 + 12 + 12 = 60 min

### Part A -- The Serving Cost Inversion (~10 min)
**Concept**: Over a model's lifetime, inference OpEx exceeds training CapEx by 10-1000x. A 10% inference optimization saves more money per month than the entire training run cost. Students arrive thinking "training is the expensive part" and leave understanding that the economics flip the moment you deploy.
**Prediction**: "Your team spent $2M training a 70B LLM. You serve it to 1M daily active users at 100 QPS. After how many weeks does cumulative serving cost exceed the training cost?" Options: (a) 6 months, (b) 3 months, (c) 6 weeks, (d) Never -- training dominates. Common wrong answer: (a) 6 months. Students anchor on how expensive training *felt* and underestimate the relentless compounding of per-query cost at scale.
**Instrument**: Cumulative cost chart with two curves: a flat horizontal line for training cost ($2M) and a rising serving cost line. Students set QPS (10-1000), cost-per-query ($0.001-$0.05), and deployment duration (1-52 weeks). An "inference optimization" slider (0-50%) tilts the serving line and shows annual savings. At default settings, the crossover occurs around week 6. A large annotation reads: "Serving cost exceeds training cost here."
**mlsysim grounding**: `Engine.solve()` computes per-query latency and FLOP cost for the selected model; `kv_cache_size()` provides the memory baseline that drives cost-per-query.
**Transition to B**: "So serving dominates the budget. The obvious lever is batching -- serve more requests per GPU cycle. But how large should the batch be? Your intuition will betray you."

### Part B -- The Queuing Hockey Stick (~12 min)
**Concept**: Batching trades latency for throughput. But the relationship is not linear -- it follows a queuing hockey stick. At high GPU utilization (small batches, high arrival rate), queuing delay explodes exponentially. A batch size with *slower* per-request compute can yield *faster* total response time because it drives utilization down into the safe zone. The "knee" of the batching efficiency curve differs by 40x between model types (vision CNNs saturate at batch 4-8; LLM decode needs batch 64-128).
**Prediction**: "A GPT-3 class model serves 100 QPS on 8xA100. Service time is S(B) = 50 + 0.5B ms. Which batch size minimizes total response time (including queuing delay)?" Options: (a) Batch 1 -- minimal compute per request, (b) Batch 8 -- balanced trade-off, (c) Batch 32 -- despite longer compute, queuing drops, (d) Batch 200 -- the unconstrained mathematical optimum. Common wrong answer: (a) Batch 1. Students equate "fast per-request" with "fast system" and do not account for the queuing penalty.
**Instrument**: Dual-axis chart. X: GPU utilization (0-100%). Left Y: total response time (ms). Right Y: throughput (req/s). A hockey-stick latency curve (Kingman's formula) with a 200ms SLO line. Batch size radio selector (1/4/8/16/32/64) highlights the operating point on the curve. At B=1, utilization is 505% (unstable -- infinite queue). At B=8, utilization is 67.5% with 42ms queuing delay. At B=32, utilization drops to 20.6% with 8.9ms queuing delay. A model-type toggle (Vision CNN / LLM Decode / RecSys) shifts the curve, showing the 40x difference in optimal batch size across architectures.
**mlsysim grounding**: `Engine.solve()` with batch_size parameter computes service time per model; Kingman's formula applied to arrival rate and service time.
**Transition to C**: "You found the batch sweet spot. But there is a hidden wall you have not yet hit: the KV cache. Every concurrent request needs its own cache, and that cache is enormous."

### Part C -- The KV Cache Wall (~14 min)
**Concept**: For autoregressive LLMs, the KV cache grows linearly with both sequence length and batch size: `KV = 2 * L * H * S * B * P` bytes. At 128K context, even 8xH100 (640 GB total HBM) can serve only 1 concurrent request for a 70B model because the KV cache consumes all available memory after weights are loaded. This is the memory wall of inference -- not compute, but cache capacity determines maximum concurrency.
**Prediction**: "A 70B model in FP16 on 8xH100 (640 GB HBM). Weights consume ~140 GB. At 128K context length, how many concurrent requests can you serve?" Options: (a) 16-32, (b) 4-8, (c) 2-3, (d) Just 1. Common wrong answer: (a) 16-32. Students think of GPUs as "compute machines" and do not mentally model the KV cache scaling.
**Instrument**: Stacked memory bar chart. Fixed segment: model weights. Growing segment: KV cache (colored by request count). Horizontal line: total HBM capacity. Students set model size (7B/70B/175B), precision (FP16/INT8/INT4 for weights), context length (2K-128K on log slider), and GPU count (1-8). As context length increases, the KV cache bar grows and squeezes the available capacity for concurrent requests. At 128K/70B/FP16/8xH100, max batch = 1. OOM banner triggers when KV + weights exceed HBM. A secondary display shows the `kv_cache_size()` formula with live values.
**Note for implementer**: Add `kv_cache_size(seq_len, batch_size, precision)` helper to `TransformerWorkload` in mlsysim. This method returns bytes = 2 * num_layers * hidden_dim * seq_len * batch_size * bytes_per_element.
**mlsysim grounding**: `kv_cache_size()` on TransformerWorkload; `Engine.solve()` for weight memory; device HBM from hardware registry.
**Transition to D**: "So the KV cache caps your concurrency. But there is a scheduling insight that partially breaks this wall: what if you did not wait for an entire batch to finish before starting new requests?"

### Part D -- Continuous Batching: The Assembly Line (~12 min)
**Concept**: Static batching wastes GPU cycles because requests in a batch finish at different times (short completions wait for long ones). Continuous batching (iteration-level scheduling) inserts new requests into freed slots every decode step, keeping the GPU saturated. This transforms inference from a stop-and-go batch system into a flowing pipeline, increasing throughput by 2-4x at the same latency SLO. The key insight: the scheduling granularity (per-batch vs per-iteration) matters more than raw hardware speed.
**Prediction**: "With static batching, a batch of 32 requests finishes when the *longest* request completes (say, 512 tokens). The average request generates only 128 tokens. What fraction of GPU cycles are wasted on padding?" Options: (a) ~10%, (b) ~25%, (c) ~50%, (d) ~75%. Common wrong answer: (a) ~10%. Students do not realize that the average request finishes 4x earlier than the longest, leaving 75% of its slot idle.
**Instrument**: Two side-by-side timeline visualizations. Left: static batching -- a grid of request slots where completed requests sit idle (grayed out) until the batch finishes. Right: continuous batching -- completed slots immediately fill with new requests (green). A throughput gauge below each shows the 2-4x improvement. Students set request length distribution (uniform, skewed, bimodal) and batch size. With skewed distributions (a few very long requests), the waste in static batching becomes visually dramatic -- most slots are gray for most of the timeline. A "scheduling granularity" toggle (per-batch / per-iteration) switches between the two views.
**mlsysim grounding**: Parametric model of static vs continuous batching using `Engine.solve()` per-token latency and a request-length distribution sampler.
**Transition to E**: "You now understand the cost structure (A), the queuing physics (B), the memory wall (C), and the scheduling trick (D). Time to put it all together."

### Part E -- The Inference Fleet Design Challenge (~12 min)
**Concept**: Designing a serving system requires jointly optimizing quantization, batch size, parallelism degree, and replica count under a latency SLO and cost budget. These interact non-linearly: INT4 quantization frees KV cache memory for larger batches, which increases throughput, which reduces required replicas, which reduces cost. But INT4 also introduces quality degradation. The challenge is finding the narrow corridor where all constraints are simultaneously satisfied.
**Instrument**: Students configure: model precision (FP16/INT8/INT4), batch size (1-128), GPUs per replica (1/2/4/8), replica count (1-64), and scheduling mode (static/continuous). A dashboard shows: throughput (QPS), P99 latency, cost per 1M tokens, memory utilization, and a quality indicator (perplexity delta from FP16 baseline). Constraint: serve 10,000 QPS at <200ms P99 for minimum cost. Failure states: (1) OOM if KV cache exceeds HBM, (2) SLA violation if P99 > 200ms, (3) budget exceeded if monthly cost > threshold. Students discover that INT4 + continuous batching + 4 GPUs/replica achieves the target at 40% lower cost than FP16 + static batching + 8 GPUs/replica.
**mlsysim grounding**: `Engine.solve()` across configurations; `kv_cache_size()` for memory constraint; Kingman queuing for P99 estimation.
**Takeaway**: "Inference is not a forward pass -- it is an economic system where memory, scheduling, and quantization interact to determine whether you can afford to serve your model at all."

---

## Lab V2-09: The Optimization Trap
**Chapter**: `performance_engineering.qmd`
**Story arc**: A junior engineer profiles a Transformer layer and declares it "compute-bound because Transformers are compute-heavy." Wrong. Students discover that most ML operations live far below the roofline ridge point, that fusion saves orders of magnitude more than faster math, and that applying the wrong optimization yields zero improvement. The lab builds from diagnosis (roofline) through treatment (fusion, FlashAttention, quantization) to the meta-skill: always profile before optimizing.
**Time budget**: 10 + 12 + 12 + 12 + 12 = 58 min

### Part A -- The Roofline Diagnostic (~10 min)
**Concept**: The roofline model reveals whether a workload is compute-bound or memory-bound. The ridge point (peak FLOPS / peak bandwidth) has shifted 4x across GPU generations (V100: 69, A100: 156, H100: 295 FLOP/byte), making *more* workloads memory-bound with each generation. LLM decode at batch=1 sits at ~1 FLOP/byte -- 295x below the H100 ridge point.
**Prediction**: "LLM decode (batch=1) on an H100. Is this workload compute-bound or memory-bound?" Options: (a) Compute-bound -- H100 has 1979 TFLOPS, (b) Slightly memory-bound -- close to the ridge, (c) Deeply memory-bound -- 100x+ below the ridge, (d) It depends on the model size. Common wrong answer: (a). Students associate "powerful GPU" with "compute-bound" and do not compute arithmetic intensity.
**Instrument**: Log-log roofline plot. X: arithmetic intensity (FLOP/byte). Y: achievable FLOPS. Sloped line (memory-bound regime) meets flat line (compute-bound regime) at the ridge point. Students select GPU generation (V100/A100/H100/B200) and ML operation (LayerNorm/Attention/large GEMM/LLM decode). Each operation appears as a dot on the plot. LLM decode at batch=1 appears far left of every GPU's ridge point. Toggling GPU generations shows the ridge shifting *rightward*, moving the same workload further into memory-bound territory. A batch-size slider (1-256) lets students watch LLM decode's arithmetic intensity increase and cross the ridge.
**mlsysim grounding**: Hardware registry provides peak FLOPS and HBM bandwidth per GPU; `Engine.solve()` provides per-operation arithmetic intensity.
**Transition to B**: "So LLM decode is memory-bound -- the GPU is starving for data. The fix is not faster math. The fix is moving less data. Enter operator fusion."

### Part B -- The Fusion Dividend (~12 min)
**Concept**: A naive Transformer layer executes 50+ kernel launches, each materializing intermediate tensors in HBM. Fusion eliminates these HBM round-trips by keeping intermediates in SRAM. The savings are not uniform: fusing three elementwise operations saves 64 MB/layer; FlashAttention (fusing the full attention block) saves 4 GB/layer. Not all fusions are equal.
**Prediction**: "A Transformer layer has 50+ operations. Fusing three elementwise ops (GELU + LayerNorm + Dropout) into one kernel yields measurable speedup. FlashAttention fuses the entire attention block. What is the ratio of FlashAttention savings to elementwise fusion savings?" Options: (a) ~2x, (b) ~10x, (c) ~60x, (d) ~1000x. Common wrong answer: (a) ~2x. Students think "fusion is fusion" and do not realize that the attention score matrix is quadratically larger than elementwise intermediates.
**Instrument**: Three-level comparison. Students see a Transformer layer computation graph with three toggle states: (1) No fusion -- every operation materializes in HBM. HBM traffic counter shows total bytes moved. (2) Elementwise fusion -- GELU+LayerNorm+Dropout fused into one kernel. HBM traffic drops by ~64 MB/layer. (3) Full FlashAttention -- attention block fused with tiled SRAM computation. HBM traffic drops by ~4 GB/layer. A stacked bar chart shows HBM bytes for each regime. The visual punch: the elementwise fusion bar is barely visible next to the FlashAttention savings. A sequence length slider (512-128K) amplifies the gap quadratically.
**mlsysim grounding**: Per-operator HBM traffic computed from tensor dimensions in `TransformerWorkload`; naive vs fused traffic formulas from @sec-performance-engineering.
**Transition to C**: "FlashAttention dominates because it eliminates the N-squared attention matrix from HBM. But how does it achieve O(N) memory? Through tiling -- and the savings curve has a shape worth understanding."

### Part C -- FlashAttention: The Savings Curve (~12 min)
**Concept**: Standard attention materializes an N x N score matrix in HBM, costing O(N^2) memory. FlashAttention tiles the computation to SRAM, reducing memory to O(N). The savings ratio is approximately N/(2d), growing linearly with sequence length. At 8K tokens, savings are 32x. At 64K tokens, 512x. Students need to see the *curve* -- it is the shape that teaches the physics.
**Prediction**: "Standard attention at 32K tokens uses ~32 GB of HBM for the score matrix (FP16, 64 heads). FlashAttention uses tiled SRAM computation. What are the memory savings?" Numeric entry (students type a multiplier). Most predict 4-8x (thinking linearly). Actual: ~256x at 32K tokens with d=128.
**Instrument**: A savings curve chart. X: sequence length (512 to 128K, log scale). Y: memory savings ratio (1x to 10,000x, log scale). Two curves: naive memory (quadratic, steep) and FlashAttention memory (linear, shallow). The *gap* between them is the savings, annotated at key points (8K: 32x, 32K: 256x, 128K: 1024x). Students drag the sequence length slider and watch the savings ratio grow. A secondary display shows absolute memory values (GB) so students see that at 128K, naive attention needs >500 GB while Flash needs <1 GB.
**mlsysim grounding**: Naive HBM = 2 * N^2 * bytes * heads; Flash HBM = 4 * N * d * bytes * heads. Both computed from TransformerWorkload dimensions.
**Transition to D**: "Fusion attacks the memory-bound bottleneck by moving less data. Quantization attacks it differently: by making each datum smaller. But naive quantization has a trap."

### Part D -- Precision Engineering: Naive vs Outlier-Aware (~12 min)
**Concept**: Reducing precision from FP16 to INT4 quadruples effective memory bandwidth, shifting the workload rightward on the roofline (closer to compute-bound). But transformer models have outlier features -- a handful of channels with values 100x larger than the rest. Naive quantization clips these outliers, causing catastrophic accuracy loss. Outlier-aware methods (which protect the salient 1% of weights at full precision) preserve quality while still capturing most of the bandwidth gain.
**Prediction**: "You quantize a 70B LLM from FP16 to INT4. Naive (uniform) quantization vs outlier-aware quantization. Both use 4 bits per weight on average. What is the perplexity difference?" Options: (a) Both are similar -- 4 bits is 4 bits, (b) Naive is 1-2 points worse, (c) Naive is 5-10 points worse -- catastrophic for large models, (d) Outlier-aware is worse because the overhead of protecting outliers costs more than it saves. Common wrong answer: (a). Students think quantization precision is the only variable that matters.
**Instrument**: Two panels. Left: roofline plot showing the workload shifting rightward as precision decreases (FP16 -> FP8 -> INT4). The effective bandwidth doubles with each halving of precision. Right: quality metric (perplexity) for two methods: "naive" (uniform quantization of all weights) and "outlier-aware" (protect top 1% of salient weights at higher precision). Students toggle precision (FP16/FP8/INT8/INT4) and method (naive/outlier-aware). At INT4, naive perplexity spikes by 8+ points for 70B models (due to outlier clipping); outlier-aware stays within 0.5 points of FP16. The roofline shift is identical for both methods -- the bandwidth gain is real regardless. The lesson: the quality cost of quantization is in the *method*, not the bit-width.
**mlsysim grounding**: Effective bandwidth = HBM_BW * (FP16_bytes / quant_bytes) shifts the roofline. Quality impact modeled as a lookup table from published quantization benchmarks.
**Transition to E**: "You now have four tools: roofline diagnosis, fusion, FlashAttention, and quantization. But which do you apply to *this* workload? The wrong choice yields zero improvement."

### Part E -- The Optimization Playbook (~12 min)
**Concept**: Applying the wrong optimization yields zero improvement. FlashAttention on a compute-bound prefill workload? Negligible gain. INT4 quantization on a memory-bound decode workload? Huge gain. The meta-skill is: diagnose first, then treat. The diagnostic sequence: (1) profile on roofline, (2) identify bottleneck (compute/memory/overhead), (3) select matching optimization.
**Instrument**: Students receive a randomly assigned "mystery workload" (LLM decode at batch 1, LLM prefill at batch 64, or vision inference at batch 32). A profiling view shows the time breakdown (compute / memory / overhead). Students must first place the workload on the roofline (drag-and-drop or multiple choice), then select optimizations from a menu: FlashAttention, INT4 quantization, operator fusion, CUDA Graphs, larger batch size. The simulation shows speedup for each choice. Correct diagnosis + matching optimization: 2-4x speedup. Wrong optimization: <5% improvement. A structured reflection asks students to match each bottleneck type to its correct treatment.
**mlsysim grounding**: `Engine.solve()` with optimization flags toggled on/off; roofline classification from arithmetic intensity vs ridge point.
**Takeaway**: "The roofline is not a chart -- it is a diagnostic tool. Profile first. The optimization that helps depends entirely on which side of the ridge your workload sits."

---

## Lab V2-10: The Edge Thermodynamics Lab
**Chapter**: `edge_intelligence.qmd`
**Story arc**: A product manager wants to add on-device fine-tuning to a smartphone app. "It is just inference with a backward pass, right?" Wrong. Students discover that training memory is 4-12x inference memory (the amplification tax), that battery drain makes naive on-device training physically impossible without NPU acceleration, and that federated learning's communication cost explodes under non-IID data -- but compression is the natural engineering response. The lab merges the federation paradox and communication-compression trade-off into a single unified part.
**Time budget**: 10 + 10 + 10 + 15 + 10 = 55 min
**Note**: Requires `mlsysim/sim/federated.py` module for Part D.

### Part A -- The Memory Amplification Tax (~10 min)
**Concept**: On-device training requires 4-12x more memory than inference due to activation caching, gradients, and optimizer state. For a 10M-parameter model on a smartphone with 300 MB available RAM: inference needs ~40 MB; full fine-tuning needs ~200-360 MB. The model that *runs* on the device cannot *learn* on the device without adaptation strategies.
**Prediction**: "A 10M-parameter model runs inference comfortably on a smartphone (40 MB). How much memory does full fine-tuning (Adam optimizer, batch size 8) require?" Options: (a) ~60 MB -- gradients add a bit, (b) ~120 MB -- double the inference footprint, (c) ~200-360 MB -- 5-9x amplification, (d) ~1 GB -- training is always an order of magnitude more. Common wrong answer: (b) ~120 MB. Students think "training = inference + gradients" and forget optimizer state and activation storage.
**Instrument**: Stacked memory bar chart. Segments: weights (fixed), gradients (= weights), optimizer state (2x weights for Adam), activations (scales with batch size and depth). A horizontal line marks the 300 MB smartphone RAM ceiling. Model scale slider (1M-100M params) and adaptation strategy toggle (Full / LoRA rank-16 / Bias-Only). At Full fine-tuning for 10M params, the bar exceeds the ceiling -- OOM banner. Switching to LoRA collapses the trainable parameter count by 100x, pulling the bar well below the ceiling.
**mlsysim grounding**: `training_memory()` from Engine with optimizer_type and adaptation_strategy parameters.
**Transition to B**: "LoRA makes it *fit*. But does it make it *practical*? A fine-tuning session that drains 30% of the battery is a product-killing feature, not a product feature."

### Part B -- The Adaptation Strategy Selector (~10 min)
**Concept**: LoRA and weight freezing reduce trainable parameters by 100-1000x, making on-device learning feasible. But the trade-off is not just memory: storage for multi-context personalization (10 user profiles) explodes to 400 MB with full fine-tuning but stays at 42 MB with adapters (200x savings per context). Convergence quality also differs.
**Prediction**: "You need to store personalized models for 10 user contexts. Full fine-tuning stores a complete model per context (40 MB each = 400 MB). LoRA stores only the adapter weights. What is the total LoRA storage?" Options: (a) ~200 MB -- half the full model cost, (b) ~100 MB -- 4x savings, (c) ~42 MB -- nearly 10x savings, (d) ~4 MB -- adapters are negligible. Common wrong answer: (b) ~100 MB. Students underestimate the compression ratio of low-rank adapters.
**Instrument**: Storage bar chart comparing Full / LoRA / Bias-Only across 1-20 user contexts. Storage grows linearly with context count for all methods, but the slopes differ by 200x. A secondary panel shows convergence quality (accuracy after N steps) for each method, revealing that LoRA reaches 95% of full fine-tuning quality with 100x less storage. Compute cost bars (FLOPS per fine-tuning step) complete the three-way trade-off: memory, storage, quality.
**mlsysim grounding**: `Engine.solve()` with adaptation_strategy parameter; storage computed from trainable parameter count.
**Transition to C**: "LoRA makes it fit in memory and storage. But the phone is a thermal system, not just a computational one. How long until the battery dies?"

### Part C -- The Battery Drain Reality (~10 min)
**Concept**: Energy consumption for on-device training is 10-50x worse than inference. An NPU achieves 20x latency speedup and 50x energy gain over CPU for the same fine-tuning workload. On CPU, a single fine-tuning session drains ~15% of battery. On NPU, it drains ~0.3%. The hardware execution target -- not the algorithm -- determines whether on-device training is a product feature or a battery-killing liability.
**Prediction**: "A LoRA fine-tuning session takes 30 seconds on CPU at 3W. The phone has a 15 Wh battery. What percentage of battery does this session consume?" Numeric entry. Most predict 1-3%. Actual: ~15% when accounting for sustained power draw, thermal throttling extending duration, and system overhead. The NPU alternative: 0.3%.
**Instrument**: Battery drain gauge (0-100%) that depletes in real time as students configure: execution target (CPU / GPU / NPU), training duration, and power draw. A thermal throttle indicator turns orange when sustained power exceeds the device's thermal design point, extending effective duration by 2-3x. Two comparison bars: CPU fine-tuning (15% drain, 30s + thermal extension) vs NPU fine-tuning (0.3% drain, 1.5s). A "sessions per full charge" counter drives the point home: CPU = ~6 sessions; NPU = ~300 sessions. Failure state: battery drops below 20%, device enters low-power mode, training aborted.
**mlsysim grounding**: `EdgeNpuSpeedup` class for latency and energy ratios; battery model: drain_pct = (power_W * duration_s) / (battery_Wh * 3600) * 100.
**Transition to D**: "On-device training works -- with LoRA, on an NPU, for a few minutes. But what if you need to learn from *thousands* of devices without collecting their data? That is federated learning, and it has a communication wall that makes the memory wall look small."

### Part D -- The Federation Paradox (~15 min)
**Concept**: Federated learning's communication cost explodes under non-IID data (up to 28x more rounds than IID). Increasing local epochs (E) reduces communication rounds but introduces client drift that degrades convergence beyond E=2-5. The natural engineering response is gradient compression: quantizing or sparsifying updates reduces per-round bytes by 4-100x, but can degrade convergence fidelity, requiring more rounds. The total communication cost (rounds x bytes/round) has a U-shaped optimum that depends on data heterogeneity, local epochs, and compression level -- three knobs that interact non-linearly. This part merges the original Parts D (Federation Paradox) and E (Communication-Compression Trade-off) into a single 15-minute exploration.
**Prediction**: "100 clients, non-IID data (beta=0.5). IID convergence takes 50 rounds. How many rounds does non-IID require to reach the same accuracy?" Options: (a) 60-80 rounds -- modest increase, (b) 100-150 rounds -- 2-3x more, (c) 200-400 rounds -- 4-8x more, (d) 1000+ rounds -- effectively never converges. Common wrong answer: (a) 60-80 rounds. Students underestimate the impact of data heterogeneity on FedAvg convergence.
**Instrument**: Two linked panels. Left panel: convergence plot (accuracy vs communication rounds) with three curves -- IID baseline, non-IID at current settings, non-IID with compression. Students set: data heterogeneity (beta: 0.1-2.0), local epochs (E: 1-20), number of clients (10-500), and compression method (None / INT8 quantized / INT4 quantized / Top-K sparse). Right panel: total communication budget bar (rounds x bytes_per_round) with a budget cap line. As beta decreases (more heterogeneous), the non-IID curve shifts right dramatically. As E increases past 5, client drift causes the curve to flatten or diverge. Compression reduces per-round bytes (shown as a shrinking bar width) but can add rounds (shown as bar growing taller). The U-shaped optimum in total communication cost is visible when students sweep E. A budget cap failure state triggers when total bytes exceed the bandwidth budget before accuracy reaches the target. The key discovery: for highly non-IID data, INT8 compression with E=3 minimizes total communication -- neither extreme (no compression / aggressive compression) is optimal.
**mlsysim grounding**: `mlsysim/sim/federated.py` module -- parametric FedAvg convergence model with heterogeneity penalty (beta), client drift factor, and compression degradation curves.
**Transition to Synthesis**: "You have mapped the full thermodynamic landscape of edge intelligence: memory amplification, battery drain, and communication cost. Each has a natural engineering response (LoRA, NPU, compression). The art is combining them."

### Synthesis (~5 min)
**Prompt**: Deploy a personalized keyboard prediction model (10M params) across 1,000 smartphones. Specify: adaptation strategy (Full/LoRA/Bias-Only), execution target (CPU/NPU), federated configuration (local epochs, compression method), and justify why your combination satisfies memory (< 300 MB), battery (< 1% drain per session), and communication (< 50 MB per round) constraints simultaneously.
**Takeaway**: "On-device training is not inference-plus-a-backward-pass. It is a thermodynamic problem where memory, energy, and communication constraints interact -- and the engineering solution at every level is the same principle: reduce what you move."

---

## Lab V2-11: The Silent Fleet
**Chapter**: `ops_scale.qmd`
**Story arc**: Your company has 200 models in production. Everything looks green on the dashboard. You are losing $1M per day. Students discover that operational complexity grows quadratically with model count (crossing team capacity at ~50 models), that silent model regressions cost millions before detection, and that hierarchical monitoring (the same principle as hierarchical AllReduce from @sec-collective-communication) is the only viable architecture for fleet-scale alerting. The lab swaps the original Part B (Platform ROI) and Part C (Silent Failure) order so that silent failure *motivates* the platform investment.
**Time budget**: 10 + 14 + 10 + 12 + 10 = 56 min

### Part A -- The Complexity Explosion (~10 min)
**Concept**: Operational complexity scales superlinearly with model count: per-model alerts grow O(N), coordination grows O(N log N), inter-model dependencies grow O(N^2). Total operational load crosses a typical team's capacity (~4,000 person-hours/year) at roughly 50 models -- far fewer than most organizations expect.
**Prediction**: "Your ML platform team has capacity for ~4,000 person-hours/year of operational work. At what model count does operational load exceed capacity?" Options: (a) ~200 models, (b) ~100 models, (c) ~50 models, (d) ~20 models. Common wrong answer: (a) ~200 models. Students assume operational cost scales linearly and estimate from "20 hours per model."
**Instrument**: Three stacked area curves (alerts O(N), coordination O(N log N), dependencies O(N^2)) with a horizontal team capacity line. Model count slider (1-500). The O(N^2) curve dominates above ~30 models, making the total shoot past capacity around N=50. A "Platform ON/OFF" toggle flattens the dependency curve from O(N^2) to O(N log N), pushing the capacity crossing point to ~150 models. This foreshadows Part C's platform ROI argument. Failure state: total line crosses capacity with banner "OPERATIONAL OVERLOAD -- team cannot maintain fleet."
**mlsysim grounding**: Complexity model: `C_total = 20N + N*log2(N) + 0.5*N^2` from @tbl-ops-scale-complexity; platform reduces dependency term.
**Transition to B**: "Your team is overloaded. But the real danger is not that they are busy -- it is that the failures they miss are *silent*. No crash. No error. Just money disappearing."

### Part B -- The Silent Failure Tax (~14 min, anchor part)
**Concept**: Model regressions are silent: no crashes, no error logs, no alerts. At scale, the financial cost accumulates catastrophically. A 0.5% CTR drop at 5,000 QPS costs $1.08M in 24 hours. The cost multiplier is *detection latency*, not the regression magnitude. A small regression detected in 1 hour costs $45K. The same regression detected in 24 hours costs $1.08M. This is the defining operational risk of ML systems at scale.
**Prediction**: "A recommendation model silently drops 0.5% in CTR. Traffic is 5,000 QPS. Average revenue per click: $0.50. The regression goes undetected for 24 hours. What is the total revenue loss?" Numeric entry. Most predict $10K-$50K (anchoring on "0.5% sounds small"). Actual: $1,080,000.
**Instrument**: A real-time dollar counter that accumulates loss per second. Students set: QPS (100-10,000), CTR drop magnitude (0.1%-2%), revenue per click ($0.10-$2.00), and detection time (1-48 hours). The counter ticks up visibly -- at default settings, it crosses $1K in under a minute of simulation. A dual-bar comparison shows the same regression at different detection latencies: 1 hour ($45K) vs 6 hours ($270K) vs 24 hours ($1.08M). The visual punch: detection latency is a 24x cost multiplier for the same regression. A "monitoring investment" toggle shows that $50K/year in monitoring infrastructure pays for itself in a single prevented 24-hour incident. Formula displayed: `Loss = QPS * 3600 * T_detection * delta_CTR * revenue_per_click`.
**mlsysim grounding**: `SilentFailure` class from @sec-ml-operations-scale-staged-rollout-strategies-2d1f; parametric loss model.
**Transition to C**: "Silent failures cost millions. The obvious fix is monitoring. But at fleet scale, monitoring itself becomes a problem."

### Part C -- The Platform ROI Calculator (~10 min)
**Concept**: A shared ML platform breaks even at far fewer models than organizations expect (~20 models for a $2M/year platform) because per-model operational savings compound against a fixed platform cost. Multi-tenant resource sharing converts 70% idle utilization to 30% idle, saving 57% on infrastructure.
**Prediction**: "A shared ML platform costs $2M/year. It saves approximately $100K/year per model in reduced operational overhead. How many models before the platform breaks even?" Options: (a) 100+ models, (b) 50 models, (c) 20 models, (d) 5 models. Common wrong answer: (a) 100+. Students anchor on the "$2M price tag" and underestimate per-model savings.
**Instrument**: ROI gauge with break-even indicator. Students set model count (1-200) and platform cost tier ($1M/$2M/$5M). The gauge turns green at break-even. A secondary display shows multi-tenant utilization gain: dedicated per-model allocation (70% idle) vs shared platform (30% idle) = 57% infrastructure savings. The $2M platform breaks even at 20 models; at 200 models, annual savings exceed $18M. A "total silent failure cost avoided" counter (using Part B's per-incident cost) shows that preventing even 2 silent failure incidents per year justifies the platform cost independently.
**mlsysim grounding**: `ROI = (N * T_saved * C_engineer) / C_platform` from @eq-platform-roi; `MultiTenantEfficiency` class.
**Transition to D**: "The platform handles deployment, monitoring, and resource sharing. But the alerting layer has its own scaling problem."

### Part D -- The Canary Duration Designer (~12 min)
**Concept**: Staged rollouts (canary deployments) require a statistically determined minimum observation window. Too short: miss the regression. Too long: deployment stalls. The canary duration formula connects sample size requirements to traffic volume and canary percentage: `t_stage = n_samples / (r_requests * p_stage)`. At 1% canary with 1M req/hour, minimum observation is 1 hour. At 5% canary, it drops to 12 minutes.
**Prediction**: "You roll out a model update to 1% of traffic (1% canary). Your service handles 1M requests/hour. You need 10,000 samples for statistical significance. How long is the minimum canary observation window?" Options: (a) ~6 minutes, (b) ~1 hour, (c) ~6 hours, (d) ~24 hours. Common wrong answer: (a) ~6 minutes. Students forget that 1% of traffic means only 10K requests/hour reach the canary.
**Instrument**: Students design a multi-stage rollout: 1% -> 5% -> 25% -> 50% -> 100%. For each stage, they configure canary percentage and request rate. A timeline visualization shows stage durations and total rollout time. Two failure states: (1) regression slips through if observation window < minimum (banner: "UNDETECTED REGRESSION -- canary duration too short"), (2) deployment stalls if total rollout > 48 hours (banner: "DEPLOYMENT STALL -- reduce stages or increase canary %").
**Cross-lab reference**: The hierarchical monitoring principle here (aggregate metrics across canary/production populations) parallels hierarchical AllReduce from @sec-collective-communication -- both solve the same mathematical problem of efficient aggregation across distributed entities.
**mlsysim grounding**: `t_stage = n_samples / (r_requests * p_stage)` from @eq-canary-duration; sample size from detection sensitivity requirements.
**Transition to E**: "Individual canary deployments work. But at fleet scale with 200 models, even precise per-model monitoring drowns in false alarms."

### Part E -- The Alert Fatigue Wall (~10 min)
**Concept**: With 3-sigma alerting (alpha = 0.003), 1,000 metrics monitored every 5 minutes, the fleet produces ~864 false alerts per day. Per-metric raw alerting is mathematically useless at fleet scale. Hierarchical monitoring -- grouping correlated metrics and alerting on aggregate anomaly scores -- reduces effective N by an order of magnitude.
**Instrument**: Students configure: model count (10-500), metrics per model (5-20), alert threshold (2/3/4-sigma), and check interval (1/5/15 min). A daily false alarm counter updates live. At default (100 models, 10 metrics, 3-sigma, 5-min checks): ~864 false alarms/day. Toggle "Hierarchical Aggregation ON" -- groups correlated metrics, reduces effective N by 10x, false alarms drop to ~86/day. A second toggle for "Adaptive Thresholds" (per-metric learned baselines) further reduces to ~20/day. The visual: a fire alarm panel with lights flashing -- at 864/day, the entire panel is red (meaningless); at 20/day, individual alerts are actionable.
**mlsysim grounding**: `P(>=1 false) = 1 - (1-alpha)^N` from @eq-false-alert-rate; `FalseAlarmTax` class.
**Takeaway**: "Managing 200 models is not 200x the work of one model. It is a qualitatively different problem where silent failures cost millions, false alarms drown real signals, and the solution -- hierarchical monitoring on a shared platform -- mirrors the same aggregation principles you saw in collective communication."

---

## Lab V2-12: The Price of Privacy
**Chapter**: `security_privacy.qmd`
**Story arc**: Your company's privacy officer says "turn on differential privacy." Your product manager says "maintain 800 tokens/sec throughput." These are not independent requirements -- they are in direct tension, and the tension has a price that scales with dataset size, task complexity, and query volume. Students discover that privacy noise destroys utility for small datasets, that the privacy-accuracy frontier has a catastrophic knee, that every defense layer extracts measurable throughput, and that the privacy budget *depletes* with use. The original Part C (Model Extraction Economy) is dropped; the arc restructures around "the cost of privacy" from noise scaling through budget depletion.
**Time budget**: 10 + 10 + 12 + 14 + 10 = 56 min

### Part A -- The Privacy Scaling Wall (~10 min)
**Concept**: Differential privacy noise magnitude is Sensitivity/epsilon, independent of dataset size. But the *error per person* scales as 1/N. At N=1,000, per-person error is $200. At N=100, per-person error is $2,000 -- ten times worse. Privacy "kills utility" for small datasets because the noise is constant while the signal-per-record shrinks.
**Prediction**: "You run a salary analysis with differential privacy (epsilon=1, sensitivity=$200K). At N=1,000 records, per-person error is $200. At N=100 records, what is the per-person error?" Options: (a) ~$200 -- same noise, same error, (b) ~$500 -- modest increase, (c) ~$2,000 -- 10x worse, (d) ~$20,000 -- unusable. Common wrong answer: (a) ~$200. Students think "same epsilon = same quality" and miss the 1/N scaling.
**Instrument**: Log-log plot of error-per-person vs dataset size (N) for three epsilon values (0.1, 1.0, 10.0). Green/yellow/red utility zones mark where the analysis is useful, marginal, or destroyed. Students drag epsilon and N sliders. At epsilon=1, N=100, the curve enters the red zone. At epsilon=1, N=10,000, the curve is solidly in the green zone. The visual teaches that DP is a technique for large datasets, not a universal privacy switch.
**mlsysim grounding**: `DPCostAnalysis` class: `error_per_person = (sensitivity / epsilon) / N`.
**Transition to B**: "Large datasets survive the noise. But even with millions of records, there is a point where tightening epsilon causes accuracy to collapse. That point is the privacy-accuracy frontier."

### Part B -- The Privacy-Accuracy Frontier (~10 min)
**Concept**: The epsilon parameter controls a continuous trade-off. Published results: MNIST retains 95% accuracy at epsilon~1; CIFAR-10 struggles to reach 82% at epsilon=8. The "knee" at epsilon 1-3 marks the transition from practical to catastrophic quality loss. The noise-to-signal ratio in DP-SGD (sigma ~4.8 at epsilon=1) means the gradient is nearly 5x more noise than signal.
**Prediction**: "CIFAR-10 achieves 93% accuracy without DP. At epsilon=8 (weak privacy), what accuracy does DP-SGD achieve?" Options: (a) ~90% -- minimal loss, (b) ~82% -- significant but usable, (c) ~65% -- severely degraded, (d) ~40% -- unusable. Common wrong answer: (a) ~90%. Students expect "epsilon=8 is weak privacy, so little cost."
**Instrument**: Privacy-accuracy curve for two tasks (MNIST, CIFAR-10). X: epsilon (0.1-100, log scale). Y: accuracy. Both curves show a knee region (epsilon 1-3) where accuracy drops steeply. Annotation: noise-to-signal ratio at each epsilon value. A task complexity toggle shows that harder tasks (CIFAR-10) lose accuracy faster because the gradient signal is weaker relative to the DP noise. The visual: MNIST stays above 90% down to epsilon~1; CIFAR-10 drops below 80% at epsilon=3.
**mlsysim grounding**: DP-SGD noise formula: `sigma = (C * sqrt(2*ln(1.25/delta))) / epsilon`; accuracy curves from published benchmarks parameterized by task complexity.
**Transition to C**: "You now understand the accuracy cost of privacy. But DP-SGD is not just a mathematical abstraction -- it runs on real hardware, and that hardware overhead stacks with every other defense you deploy."

### Part C -- The Defense Overhead Stack (~12 min)
**Concept**: Every security and privacy measure extracts measurable throughput. MIG isolation: 15% throughput reduction. DP-SGD per-step overhead: 15-30% training time increase (gradient clipping + noise addition + privacy accounting). Monitoring overhead: 1.5ms per request. Output perturbation: 1.0ms per request. Full defense stack: 30-40% total throughput reduction. Students must find the combination that satisfies both the privacy officer (epsilon < 1.0) and the product manager (throughput > 800 tokens/sec).
**Prediction**: "Baseline serving throughput is 1,000 tokens/sec. You add MIG isolation (-15%), DP-SGD noise injection for training, monitoring (+1.5ms/request), and output perturbation (+1.0ms/request). What is the resulting throughput?" Options: (a) ~900 tok/s -- overhead is small, (b) ~850 tok/s -- MIG dominates, (c) ~700-750 tok/s -- compound overhead, (d) ~500 tok/s -- security halves throughput. Common wrong answer: (a) ~900 tok/s. Students add overheads linearly rather than multiplicatively and underestimate compound effects.
**Instrument**: Waterfall chart. Starting bar: 1,000 tok/s baseline. Each defense layer (MIG isolation, noise injection, monitoring, output perturbation, rate limiting) is a toggle that subtracts from throughput. The bar shrinks with each toggle. A horizontal line at 800 tok/s marks the product SLO. Dual failure states: (1) throughput drops below 800 tok/s (SLO VIOLATED), (2) privacy budget (epsilon) exceeds 1.0 (PRIVACY REQUIREMENT NOT MET). Students must find the combination that keeps throughput above 800 AND epsilon below 1.0. The discovery: you cannot have maximum security AND maximum throughput. The art is choosing which defenses to prioritize.
**mlsysim grounding**: `DefenseOverhead` class; `MultiTenantIsolation` class; compound throughput: `T_secure = T_peak * product(1 - overhead_i)`.
**Transition to D**: "You found a feasible defense configuration for today. But there is a time bomb ticking: every query consumes privacy budget, and that budget does not regenerate."

### Part D -- The Privacy Budget Depletion (~14 min, expanded anchor part)
**Concept**: The epsilon budget is finite and non-renewable across queries. Basic composition: `epsilon_total = T * epsilon_per_query`. Advanced composition: `epsilon_total = sqrt(2T * ln(1/delta)) * epsilon_per_query`. After enough queries, the cumulative privacy loss exceeds any meaningful guarantee. Systems must enforce explicit query budgets or face a choice: shut down the API (availability = 0) or continue serving with no privacy guarantee. This is the defining operational tension of privacy at scale -- privacy is a depletable resource, not a configuration setting.
**Prediction**: "Your system serves 10,000 queries/day with epsilon=0.01 per query. Total privacy budget is epsilon=10. Using basic composition, how many days until the budget is exhausted?" Numeric entry (days). Most predict months or "never" (not understanding that 10,000 * 0.01 = 100 epsilon/day, exhausting budget in 2.4 hours, not days). Actual with advanced composition: ~3.7 days. With basic composition: 0.1 days (2.4 hours).
**Instrument**: A circular gauge showing epsilon budget depleting in real time. Students set: daily query volume (100-100,000), per-query epsilon (0.001-0.1), total budget (1-100), and composition theorem (basic/advanced). The gauge drains visibly. When exhausted, all subsequent queries are rejected -- a large "SERVICE UNAVAILABLE: PRIVACY BUDGET EXHAUSTED" banner appears, and the availability counter drops to 0%. A timeline chart shows cumulative epsilon consumption over days/weeks. Students discover that basic composition drains the budget in hours; advanced composition extends it to days; but both eventually exhaust. A "Renyi Differential Privacy" toggle shows how tighter accounting extends the budget further but still faces eventual depletion. The design challenge: find the per-query epsilon and daily volume that sustains service for at least 30 days. This forces students to quantify the trade-off between per-query privacy, query volume, and service lifetime.
**mlsysim grounding**: Basic: `eps_total = T * eps_q`. Advanced: `eps_total = sqrt(2T * ln(1/delta)) * eps_q + T * eps_q * (exp(eps_q) - 1)`. Both from @sec-security-privacy-privacy-budget-composition-edbe.
**Transition to Synthesis**: "Privacy is not a switch. It is a budget -- finite, depletable, and in direct tension with every other system requirement."

### Synthesis (~5 min)
**Prompt**: Your company serves a healthcare ML model to 50,000 daily users. Privacy regulation requires epsilon < 1.0 cumulative per patient per year. The product team requires 800 tok/s throughput and 99.9% availability. Design the defense stack (MIG, noise level, monitoring, rate limits) and query budget that satisfies all three requirements simultaneously. State which requirement you would relax first if forced to choose.
**Takeaway**: "Privacy has a price that compounds across three dimensions: dataset size (noise scaling), task complexity (accuracy frontier), and time (budget depletion). Every defense you add pays for privacy in throughput. The engineer's job is not to eliminate the cost but to budget it."

---

## Lab V2-13: The Robustness Budget
**Chapter**: `robust_ai.qmd`
**Story arc**: Your safety team mandates adversarial robustness. Your compute team has a fixed GPU budget. Your ops team needs models that run efficiently in INT8. These three demands collide. Students discover that adversarial training costs 26 percentage points of clean accuracy *and* 8x compute, that silent hardware errors are statistically certain at cluster scale, and that the economically rational strategy is external guardrails (detection + monitoring), not universal hardening. The lab clarifies: PGD cost is a *training* tax; randomized smoothing is an *inference* tax; and the quantization techniques that make deployment efficient also narrow the robustness margin.
**Time budget**: 10 + 10 + 12 + 15 + 8 = 55 min

### Part A -- The Robustness Tax (~10 min)
**Concept**: Adversarial training (PGD-7 at epsilon=8/255) drops ResNet-50 clean accuracy from 76% to 50% (a 26 percentage point loss) and costs 8x compute per training epoch (1 standard forward/backward + 7 attack iterations). Randomized smoothing costs 100,000x inference compute (sampling many noisy copies at test time) but preserves clean accuracy. The two defenses have fundamentally different cost profiles: PGD is a *training-time* tax; smoothing is an *inference-time* tax.
**Prediction**: "You adversarially train ResNet-50 with PGD-7 (epsilon=8/255). Standard accuracy is 76%. What clean accuracy do you expect after adversarial training?" Options: (a) ~73% -- small tax for robustness, (b) ~65% -- moderate trade-off, (c) ~50% -- massive 26pp loss, (d) ~35% -- model is barely functional. Common wrong answer: (a) ~73%. Students expect adversarial training to be a low-cost add-on, not a fundamental accuracy sacrifice.
**Instrument**: Grouped bar chart with four defense types (None, Adversarial Training PGD-7, Randomized Smoothing, Feature Squeezing). Two bar groups per defense: clean accuracy (left) and robust accuracy (right). Below: compute cost bars (1x, 8x, 100,000x, ~1x). An annotation explicitly labels where the cost is incurred: "TRAINING COST" for PGD, "INFERENCE COST" for smoothing. Students toggle epsilon (perturbation budget) to see how the accuracy-robustness trade-off shifts. At epsilon=0, all methods achieve 76% clean accuracy. As epsilon increases, the gap between clean and robust accuracy widens, with PGD paying in clean accuracy and smoothing paying in compute.
**mlsysim grounding**: `RobustnessTaxAnalysis` class; `AdversarialPayback` class; compute multiplier = 1 + K for PGD-K.
**Transition to B**: "Adversarial attacks are intentional threats you can train against. But at cluster scale, there are *unintentional* threats that are mathematically certain and just as devastating."

### Part B -- Silent Errors at Scale (~10 min)
**Concept**: Silent Data Corruption (SDC) -- a single bit flip in a weight or activation -- can drop ResNet-50 accuracy from 76% to 11%. At Meta's reported rate (p=10^-4/hour per device), a 10,000-GPU cluster experiences at least one SDC with probability 0.63 per hour. At 100,000 GPUs, SDC is effectively certain every hour. Redundancy and ECC are not optional at scale -- they are mathematically mandatory.
**Prediction**: "A cluster has 10,000 GPUs, each with SDC rate 10^-4 per hour. What is the probability of at least one SDC in the cluster per hour?" Options: (a) ~0.1% -- very rare, (b) ~1% -- occasional, (c) ~63% -- more likely than not, (d) ~99.99% -- virtually certain. Common wrong answer: (a) ~0.1%. Students multiply probabilities rather than using the complement formula and dramatically underestimate fleet-scale risk.
**Instrument**: Three S-curves (for three per-device rates: 10^-3, 10^-4, 10^-5) showing P(>=1 SDC) vs cluster size (1-100,000, log x-axis). A vertical line at the student's cluster size highlights the probability. At 10,000 GPUs with rate 10^-4: probability = 0.63 (annotation: "more likely than not"). At 100,000 GPUs: probability > 0.9999. A secondary display shows the impact of a single SDC: a bar chart of model accuracy before/after a bit flip in the most sensitive layer (76% -> 11%).
**mlsysim grounding**: `SilentErrorProbability` class: `P = 1 - (1-p)^N`; accuracy impact from @fig-silent-error-probability.
**Transition to C**: "Hardware failures are random and can be caught with checksums and ECC. But there is a slower, more insidious failure: the world changes, and your model does not. That is distribution drift."

### Part C -- The Distribution Drift Timeline (~12 min)
**Concept**: Unmonitored models silently degrade 20-40% over 6-12 months under distribution shift. PSI (Population Stability Index) monitoring detects drift 3-6 weeks before accuracy breaches the SLA, enabling preemptive retraining. The detection latency depends on labeled sample volume: at 1,000 labeled samples/hour, detecting a 2% accuracy drop requires ~2,200 samples (~2.2 hours).
**Note on concept ownership**: V2-13 owns PSI as a robustness monitoring tool. V2-11 (ML Ops) references PSI when discussing monitoring infrastructure but does not re-teach the statistical mechanics. Students arriving from V2-11 will recognize PSI; this part extends their understanding to the detection latency calculation.
**Prediction**: "A model deployed without PSI monitoring degrades due to distribution shift. After 6 months, what accuracy remains?" Numeric entry (%). Most predict 85-90% (expecting gradual, mild degradation). Actual: 55-70% for moderate drift rates -- a 20-40% loss that occurred silently with zero alerts.
**Instrument**: Dual-timeline chart. Top: unmonitored model (accuracy degrades smoothly, no alerts, no recovery). Bottom: monitored model (PSI crosses threshold, triggers retraining, accuracy recovers). Students set: monitoring frequency (None / Monthly / Weekly / Daily), drift rate (slow/medium/fast), and labeled sample rate (100-10,000/hour). The gap between the two timelines is the "monitoring dividend" -- the accuracy preserved by detecting drift early. A detection latency calculator shows: at 1,000 samples/hour, detecting a 2% accuracy drop takes ~2.2 hours; detecting a 0.5% drop takes ~35 hours. The trade-off: more frequent monitoring catches smaller regressions but requires more labeled data infrastructure.
**mlsysim grounding**: `DriftLatency` class: `N = (Z_a + Z_b)^2 * (p1*q1 + p2*q2) / (p1-p2)^2`; PSI formula: `PSI = sum((A_i - E_i) * ln(A_i/E_i))`.
**Transition to D**: "You now know the cost of three threats: adversarial attacks (Part A), hardware failures (Part B), and drift (Part C). The question is: which defenses do you invest in? You cannot afford all of them at maximum strength."

### Part D -- The Defense Stack Builder (~15 min)
**Concept**: Every defense layer has a quantifiable cost. Adversarial training: 8x training compute, 26pp clean accuracy loss. Feature squeezing: eliminates 70-90% of attacks at <2x overhead and 95%+ clean accuracy. Confidence thresholds: reject 5-15% of traffic. Monitoring: 5-15% inference overhead. The only economically viable strategy for most production systems is external guardrails (detection + monitoring + feature squeezing), not universal adversarial training. Combined guardrail overhead: ~1.2x vs adversarial training at 8x.
**Prediction**: "To achieve 80%+ robustness against adversarial inputs while keeping clean accuracy above 70%, which strategy has the lowest total compute cost?" Options: (a) Adversarial training -- direct defense is always cheapest, (b) Randomized smoothing -- certifiable guarantees, (c) Feature squeezing + confidence thresholds + monitoring -- layered guardrails, (d) All of the above combined. Common wrong answer: (a). Students intuitively prefer "direct defense" and underestimate the compute and accuracy cost.
**Instrument**: Layered defense builder with toggles: Adversarial Training (ON/OFF), Feature Squeezing (ON/OFF), Confidence Threshold (slider 0.5-0.99), Monitoring Frequency (None/Hourly/Real-time). Left axis: accuracy under three conditions (clean inputs, adversarial inputs, OOD inputs). Right axis: cumulative compute overhead (1x-10x). Dual failure states: (1) accuracy below 60% safety floor on any input type, (2) compute exceeds 10x budget. Students discover that guardrails (Feature Squeezing + Confidence 0.9 + Hourly Monitoring) achieve 85% clean accuracy, 75% adversarial accuracy at 1.2x compute -- while adversarial training alone achieves 50% clean, 42% adversarial at 8x compute. The economics are not close.
**mlsysim grounding**: Defense cost and accuracy parameters from `RobustnessTaxAnalysis`, `DefenseOverhead`, and feature squeezing literature values.
**Transition to E**: "The defense stack is set. But your ops team just told you: they are deploying in INT8 for throughput. Does quantization interact with robustness?"

### Part E -- The Compression-Robustness Collision (~8 min)
**Concept**: INT8 quantization reduces numerical headroom. Adversarial perturbations that a FP32 model absorbs (within the representational margin) can flip INT8 predictions because the quantized model has less room to absorb input noise. On clean inputs, INT8 matches FP32 within 1-3%. Under adversarial perturbation, INT8 accuracy collapses faster. Deployment efficiency and robustness are in direct tension.
**Instrument**: Two overlaid accuracy curves (FP32 and INT8) vs adversarial perturbation magnitude (epsilon). On clean inputs (epsilon=0), INT8 is within 1-3% of FP32. As epsilon increases, INT8 accuracy drops sharply while FP32 degrades more gracefully. Students toggle between FP32 and INT8 and sweep epsilon. The visual: at epsilon=4/255, FP32 retains 68% accuracy; INT8 drops to 52%. The gap widens as perturbations increase. A secondary display shows the deployment benefit: INT8 uses 75% less memory and achieves 2-4x throughput. The trade-off is explicit and quantified.
**mlsysim grounding**: Accuracy-vs-epsilon curves parameterized by precision; memory and throughput from `Engine.solve()` with precision flag.
**Takeaway**: "Robustness is not a switch you flip. It is a budget you spend. Universal adversarial training is too expensive for most production systems. The economically rational strategy is external guardrails: detect, monitor, reject -- and accept that quantization narrows your margin."

---

## Lab V2-14: The Carbon Budget
**Chapter**: `sustainable_ai.qmd`
**Story arc**: An executive announces: "We will make our AI 2x more efficient, cutting our carbon footprint in half." The math says otherwise. Students discover that AI compute demand has outpaced hardware efficiency by 195,000x, that geographic site selection is the single highest-leverage carbon intervention (40x), that embodied carbon dominates in clean-grid regions, and that the Jevons Paradox means efficiency gains can *increase* total consumption when demand is elastic. Only absolute carbon caps guarantee net reduction.
**Time budget**: 10 + 12 + 12 + 14 + 12 = 60 min

### Part A -- The Energy Wall (~10 min)
**Concept**: AI compute demand has grown ~350,000x (2012-2019, doubling every ~3.4 months) while hardware efficiency improves at ~1.5x/year (doubling every ~24 months). The energy deficit -- the gap between demand growth and efficiency growth -- widens exponentially.
**Caveat note for implementer**: The "350,000x growth" figure (from Amodei/Hernandez 2018 and subsequent analyses) measures training compute for frontier models, not industry-wide energy consumption. The lab should annotate: "This figure represents the compute growth for the largest training runs, not total AI energy consumption. Industry-wide growth is substantial but less extreme." This prevents students from conflating frontier model scaling with aggregate energy trends.
**Prediction**: "AI compute demand doubles every ~3.4 months. Hardware efficiency doubles every ~24 months. Over 7 years (2012-2019), how large is the gap between demand growth and efficiency growth?" Options: (a) ~10x, (b) ~1,000x, (c) ~100,000x+, (d) They are roughly even -- Moore's Law keeps up. Common wrong answer: (a) ~10x. Students intuitively expect hardware to "keep up" with demand.
**Instrument**: Dual-curve log-scale plot. X: years. Y: relative scale (log). Demand curve (steep, ~3.4-month doubling) vs efficiency curve (gradual, ~24-month doubling). A shaded "energy deficit" region between the curves grows dramatically. Students set a target model scale (10x, 100x, 1000x over GPT-3) and a hardware timeline (1-10 years). An annotation at 2019 shows the 195,000x gap. A secondary counter asks: "At current efficiency trends, when does AI compute demand exceed projected global datacenter capacity?" Students discover the answer is uncomfortably soon.
**mlsysim grounding**: Exponential growth models: `demand(t) = 2^(t/0.283)` (years, 3.4-month doubling); `efficiency(t) = 2^(t/2)` (24-month doubling).
**Transition to B**: "The energy wall is real. You cannot outrun it with better chips. So where you compute and what powers it becomes the dominant variable."

### Part B -- The Geography of Carbon (~12 min)
**Concept**: Grid carbon intensity varies 40x across regions (Quebec hydro: 20 gCO2/kWh; Poland coal: 800 gCO2/kWh). For a 10,000 MWh training run, this is the difference between 200 tonnes and 8,000 tonnes of CO2. Site selection is the single highest-leverage sustainability intervention -- it rivals the *entire* algorithmic optimization toolkit (pruning + quantization + distillation ~ 160x compound savings) in a single decision.
**Prediction**: "A 10,000 MWh training run. Quebec (hydro, 20 gCO2/kWh) vs Poland (coal, 800 gCO2/kWh). What is the carbon ratio?" Options: (a) ~2-3x, (b) ~5-10x, (c) ~40x, (d) ~100x. Common wrong answer: (b) ~5-10x. Students anchor on algorithmic speedup scales (2-5x) and do not realize that grid carbon varies by more than an order of magnitude.
**Instrument**: Bar chart of carbon emissions by region. Students select region (dropdown with 8-10 options spanning 20-800 gCO2/kWh), training energy (1,000-100,000 MWh), and PUE (1.1-2.0). Carbon total updates live. The 40x gap between Quebec and Poland is the visual punch. A secondary comparison bar shows the compound savings from the best-case algorithmic optimization (pruning + quantization + distillation ~ 160x) alongside the geographic 40x. Annotation: "A single site selection decision achieves 25% of the total savings from the entire algorithmic toolkit."
**mlsysim grounding**: `C_operational = E_total * CI_grid * PUE` from @eq-operational-carbon; grid carbon intensity lookup table from chapter data.
**Transition to C**: "In Quebec, operational carbon is low. Problem solved? Not quite. When you reduce operational carbon, a different term dominates: the carbon cost of manufacturing the hardware itself."

### Part C -- The Lifecycle Carbon Shift (~12 min)
**Concept**: As grids decarbonize, the dominant carbon term shifts from operational emissions to embodied carbon (hardware manufacturing). An H100 has 150-200 kg CO2 embodied. On a coal grid, operational carbon exceeds embodied in months. On a hydro grid, embodied carbon can represent 30-50%+ of total lifecycle emissions. Hardware longevity and utilization become the binding sustainability levers in clean-grid regions.
**Prediction**: "In a datacenter powered by 100% renewable energy, what fraction of total lifecycle carbon comes from hardware manufacturing?" Options: (a) <5% -- hardware is a rounding error, (b) ~10-15%, (c) ~30-50%, (d) ~80%+. Common wrong answer: (a) <5%. Students assume "green energy = zero carbon" and forget the physical carbon cost of fabricating silicon.
**Instrument**: Two side-by-side lifecycle bar charts: coal-grid deployment vs hydro-grid deployment. Each bar has two segments: operational carbon (proportional to grid CI) and embodied carbon (fixed, proportional to hardware count and refresh cycle). In coal-grid mode, operational dominates (~85%). In hydro-grid mode, embodied dominates (~30-50%+). Students adjust: hardware refresh cycle (2-5 years) and utilization rate (30-90%). Longer refresh cycles amortize embodied carbon; higher utilization amortizes both. The discovery: in a clean-grid datacenter, the most effective carbon intervention is *keeping hardware running longer at higher utilization*, not buying newer, more efficient chips.
**mlsysim grounding**: `C_embodied_daily = C_manufacturing / (L_lifetime * 365)` from @eq-embodied-daily; lifecycle model with operational + embodied components.
**Transition to D**: "You have optimized where you compute (geography) and how long you keep hardware (lifecycle). Your efficiency has doubled. But total energy consumption just went *up*. Welcome to the Jevons Paradox."

### Part D -- The Jevons Trap (~14 min)
**Concept**: Jevons Paradox applied to AI: making inference more efficient reduces cost-per-query, which stimulates demand. If demand elasticity > 1, total energy consumption *increases* despite per-unit efficiency gains. The formula: `E_total = (E_baseline / Efficiency) * V_baseline * Efficiency^Elasticity`. At Efficiency=2x, Elasticity=2.0: E_total = 0.5 * 1 * 4 = 2.0 (a 100% *increase* in total energy). Only absolute caps guarantee net reduction when demand is elastic.
**Empirical justification for elasticity parameter**: The elasticity of AI inference demand is estimated from observed patterns: (1) OpenAI API pricing dropped ~10x from GPT-3 to GPT-3.5-turbo (2022-2023), while API call volume increased ~50-100x; (2) inference cost reductions in recommendation systems at Meta/Google consistently drove proportionally larger increases in model serving volume. Published estimates suggest AI inference demand elasticity is 1.5-3.0, firmly in the Jevons danger zone. The lab should annotate these empirical anchors so students understand the elasticity parameter is grounded in observed market behavior, not hypothetical.
**Prediction**: "You double inference efficiency (cost per query halves). Demand increases 3x (elastic market). What happens to total energy consumption?" Numeric entry (% change). Most predict -25% to -50% reduction ("we are more efficient, so less energy"). Actual: +50% increase (E = 0.5 * 3 = 1.5x baseline).
**Instrument**: Jevons dashboard with three demand curves (inelastic: elasticity 0.3, unit-elastic: 1.0, elastic: 2.0). X: efficiency improvement (1x-10x). Y: total energy consumption (normalized to baseline). For inelastic demand, the curve goes down (efficiency works as expected). For unit-elastic, the curve is flat (savings exactly offset by demand growth). For elastic demand, the curve goes *up* (efficiency backfires). Students manipulate efficiency factor (1x-10x) and demand elasticity (0.1-3.0). A carbon cap toggle adds a horizontal ceiling on total compute. With the cap, the elastic curve bends flat at the cap regardless of demand growth. Without the cap, the elastic curve rises indefinitely. Failure state: total energy exceeds 2x baseline, chart turns red with "JEVONS REBOUND" banner.
**mlsysim grounding**: `E_total = (E_baseline / eff) * V_baseline * eff^elasticity`; carbon cap as `min(E_total, E_cap)`.
**Transition to E**: "Efficiency alone is not enough. You need a fleet-level strategy that combines geography, scheduling, and hard caps."

### Part E -- Carbon-Aware Fleet Design (~12 min)
**Concept**: Combine geographic optimization (site selection), temporal scheduling (align with renewable availability windows), efficiency optimization, and carbon caps into a coherent strategy that achieves a 50% emission reduction target without exceeding a 48-hour project delay constraint.
**Instrument**: A 24-hour carbon intensity time series for two regions (one clean, one dirty) with diurnal renewable variation. Students place training jobs on the timeline by dragging job blocks to low-CI hours. Available levers: geographic shift, temporal scheduling, efficiency optimization, and a carbon cap slider. A carbon counter tracks total emissions vs a 50% reduction target line. A latency counter tracks total project delay. Students discover: efficiency alone fails (Jevons, from Part D); geographic shift alone may add unacceptable latency; temporal scheduling alone misses the target if the dirty grid has no clean windows; only the combination of geographic shift + temporal scheduling + carbon cap reliably hits 50% reduction. A Design Ledger records the student's carbon reduction strategy and carbon cap value, feeding forward to V2-16 (Capstone).
**mlsysim grounding**: Grid CI time series data; carbon-aware scheduling model; Design Ledger persistence.
**Takeaway**: "Efficiency is necessary but not sufficient. The Jevons Paradox means that without absolute caps on total compute, efficiency gains stimulate demand that overwhelms the savings. Sustainable AI requires governance -- hard limits -- not just better engineering."

---

## Lab V2-15: The Fairness Budget
**Chapter**: `responsible_ai.qmd`
**Story arc**: A product manager asks: "Can we add a fairness constraint to the model?" The answer is yes -- but it costs accuracy, latency, compute, and the cost scales with demographic heterogeneity. Students discover the Fairness Impossibility Theorem (you literally cannot satisfy all fairness metrics simultaneously), that the "fairness tax" is proportional to base rate divergence, that deployed models create feedback loops that amplify bias exponentially, and that responsible AI infrastructure has real system costs that must be budgeted alongside latency and memory.
**Time budget**: 12 + 10 + 12 + 14 + 12 = 60 min

### Part A -- The Impossibility Wall (~12 min)
**Concept**: The Fairness Impossibility Theorem (Kleinberg 2016, Chouldechova 2017) proves that Demographic Parity, Equalized Odds, and Calibration cannot be simultaneously satisfied when base rates differ between groups. This is not an engineering limitation -- it is a mathematical impossibility. The only escape: equal base rates across groups, which is rarely the case in practice.
**Prediction**: "A classifier has two demographic groups with different base rates (60% vs 30% positive rate). Can you find a single threshold that satisfies both Demographic Parity AND Equalized Odds?" Options: (a) Yes -- with careful threshold tuning, (b) Yes -- with separate thresholds per group, (c) No -- it is mathematically impossible when base rates differ, (d) No -- but only because the model is poorly trained. Common wrong answer: (a). Students believe fairness is a tuning problem, not a mathematical constraint.
**Instrument**: A threshold sweep chart. X: classification threshold (0.0-1.0). Y: three fairness metrics plotted simultaneously (Demographic Parity gap, Equalized Odds gap, Calibration gap). Students drag the threshold slider and watch: at every threshold, at least one metric is substantially violated. The three curves never all cross zero at the same point. A base rate control lets students adjust group base rates. When base rates are equalized, the impossibility vanishes -- all three metrics can be simultaneously satisfied. Side-by-side confusion matrices update in real time, making the trade-off concrete: satisfying DP requires different acceptance rates that violate Equalized Odds, and vice versa.
**mlsysim grounding**: Fairness metric calculations from confusion matrix parameters; impossibility theorem constraints from @sec-responsible-ai.
**Transition to B**: "You cannot satisfy all fairness metrics. You must choose one and pay a cost. How large is that cost?"

### Part B -- The Fairness Tax (~10 min)
**Concept**: Enforcing any fairness constraint reduces overall accuracy by a quantifiable amount (the "fairness tax"). This tax scales with the base rate divergence between groups. Small gap: small tax. Large gap: large tax. At 60% vs 30% base rates, enforcing demographic parity drops accuracy from 85% to ~75-78% -- a 7-10 percentage point tax that scales with heterogeneity.
**Prediction**: "Baseline accuracy is 85%. Groups have base rates of 60% and 30%. After enforcing Demographic Parity, what is the resulting accuracy?" Numeric entry. Most predict 82-84% (expecting a small, fixed cost). Actual: ~75-78%.
**Instrument**: Two linked displays. Top: accuracy vs base rate gap chart. X: base rate divergence (0-0.5). Y: accuracy (60%-90%). Three curves for three fairness constraints (DP, Equalized Odds, Equal Opportunity), each with a different tax profile. Students drag the base rate gap slider and watch accuracy drop along each curve. DP has the steepest tax; Equal Opportunity has the shallowest. Bottom: a "cost calculator" showing the fairness tax in absolute terms for the current configuration. At gap=0.3 (60% vs 30%), DP tax is ~8pp, Equalized Odds tax is ~5pp, Equal Opportunity tax is ~3pp. The lesson: the choice of fairness metric is itself a design decision with quantifiable cost.
**mlsysim grounding**: Fairness tax model: `Tax(metric) ~ k * |base_rate_gap|` with k calibrated per metric from chapter examples.
**Transition to C**: "You chose a metric and accepted the tax. The model is deployed. But deployed models do not just predict -- they *shape* the world that generates their future training data."

### Part C -- The Feedback Loop (~12 min)
**Concept**: ML systems create feedback loops: biased predictions generate biased outcomes, which generate biased retraining data, compounding the initial skew exponentially across iterations. A 5% initial bias amplifies to 40-60% disparity within 10 retraining cycles without intervention. Breaking the loop requires intervention at multiple stages simultaneously -- post-hoc audits alone are insufficient.
**Prediction**: "Initial data has 5% bias between two groups. The model retrains on its own predictions every month. After 10 retraining cycles without intervention, what is the group disparity?" Options: (a) ~8% -- modest growth, (b) ~15% -- doubles, (c) ~40-60% -- exponential amplification, (d) ~5% -- bias is stable. Common wrong answer: (a) ~8%. Students expect linear drift, not exponential amplification.
**Instrument**: A simplified simulation with 5-10 pre-computed iteration snapshots (not a live simulation, to manage complexity). Students see: iteration 0 (5% bias) through iteration 10, with disparity plotted at each step. The curve is exponential. Students can intervene at specific iteration points by toggling four controls: data auditing (ON/OFF), fairness constraint (ON/OFF), output monitoring (ON/OFF), and feedback governance (ON/OFF). Each combination produces a different pre-computed trajectory. Students discover that: (1) no intervention = exponential growth; (2) fairness constraint alone slows but does not stop growth; (3) data auditing alone catches the problem late; (4) only the combination of fairness constraint + data auditing + output monitoring breaks the loop. A counter shows the number of interventions required vs the cost savings from avoided disparity. The key insight: feedback loops require *structural* breaks, not point fixes.
**Note for implementer**: Use 5-10 pre-computed iteration snapshots with student-controlled intervention toggles rather than a live iterative simulation. This avoids complex state management while preserving the pedagogical arc. Each combination of interventions maps to a pre-computed trajectory.
**mlsysim grounding**: Pre-computed trajectories from the Sociotechnical Feedback Invariant: `P_{t+1}(X) = f(P_t(X), model_t(X))` with intervention factors as multiplicative dampening coefficients.
**Transition to D**: "Breaking the feedback loop costs compute: monitoring, auditing, re-evaluation. How much does responsible AI *actually cost* in system resources?"

### Part D -- The Responsible AI Overhead Budget (~14 min)
**Concept**: Responsible AI techniques impose quantifiable computational overhead. DP-SGD: 15-30% training time increase. SHAP explainability: 50-200% inference cost increase. Fairness monitoring: 10-20ms per request. At fleet scale (10B inferences/day), even 10ms overhead = 100M GPU-seconds/day of responsible AI compute. Students must design a fleet that meets accuracy, fairness, and latency requirements simultaneously -- and discover that the overhead budget forces hard choices.
**Prediction**: "Inference latency: 30ms. Fairness monitoring: 15ms. SHAP explainability: 50ms (on-demand). Latency SLA: 100ms. Can you fit all three?" Options: (a) Yes -- 30+15+50 = 95ms, under budget, (b) Yes -- but only with optimized monitoring, (c) No -- 95ms is too close to 100ms with no margin, (d) No -- you must choose between monitoring and explainability. Common wrong answer: (a). Students add the numbers but forget queuing variance, network overhead, and the need for safety margin.
**Instrument**: Latency waterfall chart for two regions (homogeneous demographics, heterogeneous demographics). Segments: inference (30ms), monitoring (toggle: none/basic 10ms/full 20ms), explainability (toggle: none/LIME 25ms/SHAP 50ms), overhead (5-10ms). SLA line at 100ms. Students configure fairness metric (DP/Equalized Odds/Equal Opportunity), monitoring level, and explainability level. A radar plot shows: accuracy, fairness disparity, latency, and memory across both regions. Failure state: total > 100ms triggers "SLA VIOLATED" banner. The design challenge: accuracy > 80% in both regions, fairness disparity < 0.05, total latency < 100ms. Students discover that Full SHAP + Full Monitoring = SLA violation. The solution: Equal Opportunity (less restrictive than DP, lower accuracy tax) + Basic Monitoring + On-demand LIME. A Design Ledger records the fairness metric, monitoring level, and overhead budget for the Capstone lab.
**mlsysim grounding**: Overhead values from @tbl-responsible-ai-overhead; latency composition from `Engine.solve()` + overhead lookup; `ResponsibleEngineeringModel` class for radar plot inputs.
**Transition to E**: "You have designed the monitoring and fairness infrastructure. But there is a paradox at the human-AI interface that no amount of infrastructure can solve."

### Part E -- The Fairness Audit Pipeline (~12 min)
**Concept**: Rather than the original Automation Bias framing (which is important but not systems-focused), this part reframes around a Fairness Audit Pipeline -- the end-to-end system infrastructure required to detect, measure, and remediate fairness violations at fleet scale. This connects directly to `mlsysim`'s `ResponsibleEngineeringModel` and the operational reality that fairness is not a one-time certification but a continuous monitoring and remediation process. The audit pipeline must: (1) collect demographic-stratified performance metrics, (2) detect statistically significant disparity, (3) trigger remediation (retraining, threshold adjustment, or model rollback), and (4) verify that remediation actually reduced disparity without degrading other metrics. Each stage has compute cost and detection latency.
**Instrument**: A pipeline builder where students configure 4 audit stages: data collection (sampling rate: 1%/5%/10% of traffic), disparity detection (statistical test: chi-squared / bootstrap), remediation trigger (disparity threshold: 0.01-0.10), and verification (A/B test duration: 1-7 days). Two outputs: (1) detection latency (how quickly a fairness violation is caught) and (2) remediation cost (compute + revenue impact of A/B test). Students discover the trade-off: aggressive auditing (high sampling, low threshold) catches violations fast but costs significant compute; conservative auditing (low sampling, high threshold) is cheap but misses small but persistent disparities. A timeline shows: violation introduced -> detected -> remediated -> verified. The gap between "introduced" and "verified" is the fairness debt accumulation window. Students optimize to minimize this window within a compute budget. This directly parallels the silent failure detection architecture from V2-11.
**mlsysim grounding**: `ResponsibleEngineeringModel` for audit pipeline cost modeling; detection latency formula analogous to canary duration from V2-11.
**Takeaway**: "Fairness is not a metric you optimize once. It is a budget -- mathematical impossibility forces a choice of metric, demographic heterogeneity determines the accuracy tax, feedback loops demand continuous monitoring, and the infrastructure to detect and remediate violations has real system costs. Every responsible AI decision is a systems engineering decision."

---

## Lab V2-16: The Fleet Synthesis (Capstone)
**Chapter**: `conclusion.qmd`
**Story arc**: Everything connects. Students have spent 15 labs learning individual principles: communication, fault tolerance, inference, performance, edge intelligence, operations, security, robustness, sustainability, and fairness. This capstone reveals that these principles interact as a coupled system where optimizing one degrades another. The binding constraint shifts with scale. The art of distributed ML engineering is not maximizing any single dimension but finding the best compromise across all of them. Cut to 4 parts (from original 5) by dropping the speculative "100x Challenge."
**Time budget**: 12 + 10 + 15 + 15 = 52 min

### Part A -- The Sensitivity Wall (~12 min)
**Concept**: At fleet scale (1,000+ GPUs), communication -- not computation -- is the most sensitive system dimension. A 10% degradation in network bandwidth causes a larger throughput drop than 10% fewer FLOPS, because synchronization barriers amplify communication bottlenecks nonlinearly. But this is *scale-dependent*: at 8 GPUs, compute dominates. The sensitivity ordering flips as you scale.
**Prediction**: "A 1,000-GPU training cluster. Which 10% improvement yields the largest throughput gain?" Options: (a) 10% more FLOPS per GPU, (b) 10% more network bandwidth, (c) 10% better fault tolerance (fewer restarts), (d) 10% better scheduling (less idle time). Common wrong answer: (a) 10% more FLOPS. Students default to "compute is king" even after 15 labs of learning that it usually is not.
**Instrument**: Tornado sensitivity chart showing the throughput impact of 10% improvement in each of 6 dimensions (compute, communication, fault tolerance, scheduling, sustainability overhead, fairness overhead). At 1,000 GPUs, communication is 2-3x more sensitive than compute. A fleet size toggle (8/64/1,000/10,000) lets students watch the sensitivity ordering flip: at 8 GPUs, compute dominates; at 64, they are roughly equal; at 1,000+, communication dominates decisively. This directly demonstrates the book's thesis: scale creates qualitative change. A secondary display decomposes the communication sensitivity: at large N, the `2(n-1)/n * G/BW` term in the training step time equation approaches `2G/BW`, making bandwidth the irreducible floor.
**mlsysim grounding**: `Engine.solve()` with fleet_size parameter; sensitivity computed as `delta_throughput / delta_parameter` for each dimension.
**Transition to B**: "Communication dominates at scale. But the fleet also fails at scale -- with mathematical certainty."

### Part B -- The Failure Budget (~10 min)
**Concept**: At fleet scale, component failure is routine. Meta's Llama 3 training on 16,384 GPUs experienced 419 failures in 54 days (one every 3 hours). Cluster MTBF = component MTBF / N. For 10,000 GPUs with individual MTBF of 10,000 hours: cluster MTBF = 1 hour. Checkpointing overhead and recovery strategy dominate system design, not raw performance.
**Prediction**: "10,000 GPUs, each with 10,000-hour MTBF. What is the cluster MTBF (time between any failure)?" Numeric entry (hours). Most predict days or weeks. Actual: 1 hour. Even 99.99% per-GPU uptime yields only 37% probability that all 10,000 GPUs are simultaneously operational.
**Instrument**: A calculator showing MTBF composition. Students set fleet size (10-100,000) and per-GPU MTBF (1,000-100,000 hours). Cluster MTBF = per-GPU MTBF / N. A secondary display shows: at 10,000 GPUs, `P(all up) = (1 - 1/MTBF)^N ~ 0.37`. Students then adjust checkpoint frequency (every 1/5/15/30 minutes). A goodput chart (useful work / total work) shows the Young-Daly optimal checkpoint interval that maximizes goodput, balancing checkpoint overhead (time spent saving state) against wasted computation (work lost to failures between checkpoints). Cross-reference: this directly builds on V2-07's Young-Daly exploration, now at fleet scale.
**mlsysim grounding**: `MTBF_cluster = MTBF_component / N`; goodput model from Young-Daly: `goodput = 1 - T_ckpt / MTBF_cluster`.
**Transition to C**: "You have seen how individual principles (communication sensitivity, fault tolerance) behave. Now: how do they interact?"

### Part C -- The Principle Interaction Map (~15 min)
**Concept**: The C-Cube framework (Computation, Communication, Coordination) with sustainability and fairness as cross-cutting constraints. The six principles interact as a coupled system: optimizing communication (aggressive gradient compression) increases failure exposure; sustainability constraints (carbon caps from V2-14) limit available compute; fairness overhead (monitoring from V2-15) consumes latency budget. No configuration achieves maximum on all six axes simultaneously. The effective system gain is multiplicative: `G_effective = G_total * eta_comm * eta_fault * (1 - delta_fairness) * (1 - delta_carbon)`.
**Prediction**: "You push communication efficiency to 99% (aggressive gradient compression). What happens to fault tolerance?" Options: (a) No effect -- they are independent, (b) Slight degradation -- compressed checkpoints are less reliable, (c) Significant degradation -- compressed gradients increase sensitivity to bit errors and reduce error detection capability, (d) Improvement -- faster communication means faster checkpointing. Common wrong answer: (a). Students treat principles as independent knobs.
**Instrument**: Hexagonal radar plot with six axes: Computation, Communication, Fault Tolerance, Scheduling, Sustainability, Fairness. Students push individual axes to their maximum and observe which other axes degrade. A coupling matrix (6x6 heatmap) shows positive and negative correlations between principles. For example: Communication + vs Fault Tolerance - (aggressive compression reduces error margins); Sustainability + vs Computation - (carbon caps limit GPU-hours); Fairness + vs Latency - (monitoring overhead consumes the latency budget). Students can drag each axis independently and watch the effective system gain (displayed as a large number above the plot) change. The discovery: maximum on any single axis never produces maximum total gain. The best configuration is a compromise polygon that stays within all constraint zones simultaneously.
**mlsysim grounding**: Coupling matrix with interaction coefficients from chapter principles; `G_effective = product(eta_i)` across all principle efficiencies.
**Transition to D**: "You understand the interactions. Now design the fleet."

### Part D -- The Fleet Architecture Blueprint (~15 min)
**Concept**: The terminal synthesis of the entire two-volume curriculum. Students design a fleet configuration that achieves a target effective system gain (>= 50x over single-GPU baseline) while keeping all six radar axes within acceptable zones. Sustainability constraints from V2-14's Design Ledger (carbon cap, geographic strategy) and fairness overhead from V2-15's Design Ledger (fairness metric, monitoring level, latency overhead) feed in as starting constraints. Students who have not completed V2-14 or V2-15 receive default values (carbon cap = 80% of baseline; fairness overhead = 15ms; fairness metric = Equal Opportunity).
**Default Design Ledger values** (for students who have not completed prior labs):
- Carbon cap: 80% of unconstrained compute budget
- Carbon strategy: Geographic optimization (Quebec) + temporal scheduling
- Fairness metric: Equal Opportunity (lowest accuracy tax)
- Fairness monitoring overhead: 15ms per request
- Fairness disparity threshold: 0.05

**Instrument**: The full fleet configurator. Students set: GPU count (8-10,000), communication strategy (AllReduce ring/tree, gradient compression level), checkpoint frequency (1-30 min), scheduling mode (static/elastic), carbon cap (from Ledger or default), and fairness overhead (from Ledger or default). The hexagonal radar plot from Part C shows the current configuration polygon against a target polygon. Axes turn green when satisfied, red when below threshold. An effective gain counter (displayed prominently) shows the multiplicative product of all principle efficiencies. The constraint: effective gain >= 50x with no axis in the red zone. Students discover that V2-14's carbon cap reduces available compute (requiring more efficient communication to compensate), and V2-15's fairness monitoring overhead eats into the latency budget (constraining batch size). The solution requires balancing all six principles -- no single-axis maximization works. A structured reflection asks: "Which principle was the binding constraint for your fleet? Would you trade fairness overhead for communication efficiency? Why or why not?" The Design Ledger saves the final fleet configuration.
**mlsysim grounding**: `Engine.solve()` at fleet scale with all parameters; `SynthesisSolver` coupling matrix for principle interactions; Design Ledger input/output.
**Takeaway**: "Distributed ML engineering is not the art of maximizing any single metric. It is the art of balancing computation, communication, and coordination under sustainability and fairness constraints -- where every optimization creates a new bottleneck, and the binding constraint shifts with every decision you make. That is the physics of AI engineering."

---

## Cross-Lab Reference Map

| Lab | Feeds Into | Receives From |
|-----|-----------|---------------|
| V2-08 (Inference) | V2-09 (roofline context for inference workloads) | V2-05 (model parallelism context) |
| V2-09 (Performance) | V2-08 (optimization targets for inference) | V2-08 (inference workload profiles) |
| V2-10 (Edge) | V2-11 (on-device monitoring needs) | V2-08 (inference cost baseline) |
| V2-11 (Ops) | V2-13 (monitoring as robustness tool) | V2-07 (hierarchical aggregation principle) |
| V2-12 (Security) | V2-13 (defense overhead context) | V2-11 (silent failure detection) |
| V2-13 (Robust) | V2-16 (robustness as fleet constraint) | V2-11 (PSI reference), V2-12 (defense cost) |
| V2-14 (Carbon) | V2-16 (carbon cap in Design Ledger) | V2-09 (efficiency baseline) |
| V2-15 (Fairness) | V2-16 (fairness overhead in Design Ledger) | V2-11 (feedback loop detection) |
| V2-16 (Capstone) | Terminal | V2-14 + V2-15 Design Ledger values, all prior principles |

## mlsysim Module Requirements

| Module | Needed By | Status |
|--------|-----------|--------|
| `kv_cache_size()` on TransformerWorkload | V2-08 Part C | New method needed |
| `mlsysim/sim/federated.py` | V2-10 Part D | New module needed |
| `mlsysim/sim/policy.py` | V2-11, V2-12, V2-15 | New module needed |
| `SynthesisSolver` coupling matrix | V2-16 Parts C-D | New class needed |
| `DPCostAnalysis`, `DefenseOverhead` | V2-12 | Verify existing or create |
| `RobustnessTaxAnalysis`, `AdversarialPayback` | V2-13 | Verify existing or create |
| `ResponsibleEngineeringModel` | V2-15 Parts D-E | Verify existing or create |
| `SilentFailure`, `FalseAlarmTax` | V2-11 | Verify existing or create |

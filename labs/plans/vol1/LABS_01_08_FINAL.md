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

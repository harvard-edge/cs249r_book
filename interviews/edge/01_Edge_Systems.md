# Round 1: Edge Systems & Real-Time Physics 🤖

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Edge_Systems.md">🤖 Edge Round 1</a> ·
  <a href="02_Edge_Advanced.md">🏭 Edge Round 2</a>
</div>

---

The domain of the Edge ML Systems Engineer. This round tests your understanding of what happens when ML meets hard physics at the point of action: thermal envelopes, real-time deadlines, integer-only silicon, and sensor pipelines that cannot drop a frame.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/edge/01_Edge_Systems.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 📐 Roofline & Integer Compute

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The TOPS Illusion</b> · <code>roofline</code></summary>

**Interviewer:** "Your team is evaluating edge accelerators for a robotics perception stack. The Hailo-8 datasheet says 26 TOPS. The Jetson Orin datasheet says 275 TOPS. Your product manager says 'just buy the Orin — it's 10x faster.' What critical distinction is the PM missing?"

**Common Mistake:** "TOPS is TOPS — more is faster." This treats peak throughput as the only metric, ignoring the constraint that actually matters at the edge.

**Realistic Solution:** The PM is comparing raw TOPS but ignoring TOPS per Watt — the metric that determines what you can actually *sustain* inside a thermal envelope. The Hailo-8 delivers 26 TOPS at 2.5W = **10.4 TOPS/W**. The Jetson Orin delivers 275 TOPS at 60W = **4.6 TOPS/W**. If your robot's power budget is 10W for the AI module, the Hailo-8 can sustain ~100% of its peak, while the Orin must be power-capped to ~46 TOPS — less than 2x the Hailo, not 10x. At the edge, the correct comparison is always TOPS/W × available power budget.

> **Napkin Math:** Hailo-8: 26 TOPS / 2.5W = 10.4 TOPS/W. Orin: 275 TOPS / 60W = 4.6 TOPS/W. At a 10W budget: Hailo delivers ~26 TOPS, Orin delivers ~46 TOPS. The "10x advantage" shrinks to 1.8x.

> **Key Equation:** $\text{Sustainable TOPS} = \text{TOPS/W} \times \text{Power Budget (W)}$

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Integer Roofline</b> · <code>roofline</code></summary>

**Interviewer:** "You're profiling a YOLOv8 detection model on a Jetson Orin NX. The Orin's DLA (Deep Learning Accelerator) is rated at 100 TOPS INT8 with 102 GB/s of memory bandwidth. Your model achieves 18 TOPS. The team says 'we're only at 18% utilization — we need to optimize the kernels.' Construct the integer roofline and explain why the team may be wrong."

**Common Mistake:** "18% utilization means the kernels are inefficient — fuse more layers." This assumes the workload is compute-bound. On edge NPUs, the ridge point is far lower than on data center GPUs.

**Realistic Solution:** First, build the integer roofline. The ridge point = 100 TOPS / 102 GB/s ≈ **980 INT8 Ops/Byte**. Now compute the model's arithmetic intensity. YOLOv8-S has ~28.4 billion INT8 Ops and loads ~11.2 million parameters (11.2 MB at INT8) plus activations (~40 MB for 640×640 input). Total memory traffic ≈ 55 MB per inference. Arithmetic intensity = 28.4 × 10⁹ / 55 × 10⁶ ≈ **516 Ops/Byte**. At 516 Ops/Byte, the workload sits *below* the ridge point — it is **memory-bandwidth bound**, not compute-bound. The attainable throughput = 102 GB/s × 516 ≈ 52.6 TOPS. Achieving 18 TOPS means ~34% of the memory-bandwidth ceiling, which suggests memory access inefficiency (strided access, activation spills), not compute underutilization. The fix is tiling and layer fusion to reduce memory traffic, not faster math.

> **Napkin Math:** Ridge point = 100 TOPS / 102 GB/s ≈ 980 Ops/Byte. YOLOv8-S intensity ≈ 516 Ops/Byte → memory-bound. Bandwidth ceiling = 102 × 516 = 52.6 TOPS. Actual = 18 TOPS → 34% of bandwidth ceiling. Kernel optimization should target memory access patterns, not compute throughput.

> **Key Equation:** $\text{Ridge Point}_{\text{INT8}} = \frac{\text{Peak INT8 TOPS}}{\text{Memory Bandwidth (GB/s)}}$

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

---

### ⏱️ Real-Time Inference & Scheduling

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Frame Budget</b> · <code>real-time</code></summary>

**Interviewer:** "Your autonomous vehicle's perception stack must run at 30 FPS on a Jetson Orin. You have a detection model (18ms), a tracking module (5ms), and a planning module (8ms). Your colleague says 'that's 31ms total — we're fine, it's under 33ms.' Why is your colleague dangerously wrong?"

**Common Mistake:** "31ms < 33ms, so we meet the deadline." This treats the frame budget as if average-case execution is all that matters.

**Realistic Solution:** Your colleague is reasoning about *average* execution time, but real-time systems must guarantee **Worst-Case Execution Time (WCET)**. In a safety-critical pipeline, you must account for: (1) WCET of each stage, not average — detection might spike to 25ms on dense scenes, (2) memory contention from concurrent sensor DMA transfers, (3) OS scheduling jitter (even on Linux RT patches, expect 1–2ms), and (4) thermal throttling that can increase latencies by 30–50%. A real-time budget must include margins. The industry standard is to design for ≤70% of the frame budget in average case, leaving 30% headroom for WCET spikes. That means your pipeline must average ≤23ms, not 31ms.

> **Napkin Math:** Frame budget at 30 FPS = 33.3ms. WCET margin = 30% → usable budget = 23.3ms. Average pipeline = 31ms → **over budget by 33%**. Under thermal throttling (+40%): pipeline = 43ms → drops to ~23 FPS, violating the hard real-time contract.

> **Key Equation:** $\text{Usable Budget} = \frac{1}{\text{FPS}} \times (1 - \text{WCET Margin})$

**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Pipeline Overlap</b> · <code>real-time</code></summary>

**Interviewer:** "Your edge perception system runs three stages sequentially: camera preprocessing (8ms), neural network inference (20ms), and post-processing/NMS (5ms) — totaling 33ms for a single frame. You need to hit 30 FPS. Without changing any model or buying new hardware, how do you cut the per-frame latency?"

**Common Mistake:** "Optimize the neural network to run faster." The question explicitly says no model changes. The answer is architectural.

**Realistic Solution:** Pipeline the stages across frames. While the neural network processes frame N, the camera preprocessor ingests frame N+1, and post-processing finalizes frame N−1. In a 3-stage pipeline, the throughput is limited by the *slowest stage* (20ms), not the sum (33ms). This gives you 1 frame every 20ms = **50 FPS** — well above the 30 FPS requirement. The trade-off is increased *latency* (each individual frame takes 33ms from capture to output), but the *throughput* (frames per second delivered to the planner) meets the deadline.

> **Napkin Math:** Sequential: 33ms/frame = 30.3 FPS (barely meets deadline, no margin). Pipelined: max(8, 20, 5) = 20ms/frame = 50 FPS throughput. Latency per frame = 33ms (unchanged), but throughput headroom = 50/30 = 67% margin for WCET spikes.

> **Key Equation:** $\text{Throughput}_{\text{pipelined}} = \frac{1}{\max(t_{\text{stage}_1}, t_{\text{stage}_2}, \ldots, t_{\text{stage}_n})}$

**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
</details>

---

### 🔢 Quantization & Thermal Headroom

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The QAT Cliff</b> · <code>quantization</code></summary>

**Interviewer:** "You post-training quantized (PTQ) your autonomous driving detection model from FP16 to INT4 on a Jetson Orin. Overall mAP dropped only 2% — acceptable. But when you test on nighttime scenes with low-contrast pedestrians, recall for the 'pedestrian' class drops from 94% to 71%. Your safety team blocks deployment. What went wrong with PTQ, and what is the principled fix?"

**Common Mistake:** "INT4 is too aggressive — go back to INT8." This may work but doesn't explain *why* PTQ failed selectively, and it leaves performance on the table.

**Realistic Solution:** PTQ calibrates quantization ranges using a representative dataset, but the calibration statistics are dominated by the *majority distribution* (daytime, high-contrast). Nighttime pedestrians have activations in the long tail of the distribution — small magnitude, low contrast. INT4's 16 discrete levels crush these subtle features into the same quantization bin, destroying the signal. The fix is **Quantization-Aware Training (QAT)**: insert fake-quantization nodes during fine-tuning so the network learns to be robust to INT4 discretization. QAT forces the gradient updates to account for quantization error, effectively widening the activation distributions for hard classes. Additionally, use per-channel quantization on the critical detection head layers and mixed-precision: INT4 for the backbone (where features are robust) and INT8 for the detection head (where precision matters for safety-critical classes).

> **Napkin Math:** INT8 = 256 levels. INT4 = 16 levels. A nighttime pedestrian might produce activations in the range [0.01, 0.05]. INT8 step size for range [0, 1] = 1/256 ≈ 0.004 → 10 distinct levels in [0.01, 0.05]. INT4 step size = 1/16 = 0.0625 → **zero** distinct levels in [0.01, 0.05] — the entire signal collapses to one bin.

> **Key Equation:** $\text{Quantization Step} = \frac{x_{\max} - x_{\min}}{2^b - 1}$

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 📡 Sensor Fusion & Synchronization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Timestamp Drift</b> · <code>sensor-fusion</code></summary>

**Interviewer:** "Your autonomous vehicle fuses camera images (30 FPS, MIPI CSI) with LiDAR point clouds (10 Hz, PCIe). In testing, 3D bounding boxes are accurate. In deployment at highway speed, the boxes consistently lag behind the actual vehicle positions by about 1–2 meters. The detection model hasn't changed. What is causing the spatial error?"

**Common Mistake:** "The model needs retraining on highway-speed data." The model is fine — the error is in the data pipeline, not the neural network.

**Realistic Solution:** Sensor timestamp misalignment. The camera and LiDAR have independent clocks and different capture latencies. The camera exposes a frame in ~5ms (rolling shutter), while the LiDAR completes a 360° sweep in 100ms. If you naively pair "the latest frame from each sensor," you can have 50–100ms of temporal misalignment. At highway speed (30 m/s or ~108 km/h), a 50ms timestamp drift produces a **1.5 meter spatial offset** — exactly the error you're seeing. The fix is hardware-triggered synchronization (PPS signal from GPS to both sensors) or software compensation (ego-motion interpolation using IMU data to project the LiDAR cloud to the camera's exact capture timestamp).

> **Napkin Math:** Highway speed = 30 m/s. LiDAR sweep = 100ms. Worst-case misalignment = 100ms. Spatial error = 30 m/s × 0.1s = **3.0m**. Average misalignment (~50ms) = 30 × 0.05 = **1.5m**. At 60 km/h (urban): 16.7 m/s × 0.05s = 0.83m — still enough to misclassify lane position.

> **Key Equation:** $\text{Spatial Error} = v_{\text{ego}} \times \Delta t_{\text{sync}}$

**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

---

### 🌡️ Thermal Management & Sustained Performance

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Thermal Staircase</b> · <code>thermal</code></summary>

**Interviewer:** "Your edge AI box (Jetson Orin, 60W TDP, passive cooling) runs a multi-model perception stack for a security system: object detection + face recognition + behavior analysis. In benchmarks, the system processes 25 FPS. After 3 minutes of sustained operation in a 40°C outdoor enclosure, FPS drops to 25 → 20 → 14 in discrete steps. Why does performance degrade in steps rather than gradually, and how do you design around it?"

**Common Mistake:** "Thermal throttling is gradual — the clock speed decreases linearly with temperature." This misunderstands how modern SoCs manage thermals.

**Realistic Solution:** Modern SoCs like the Orin use **DVFS (Dynamic Voltage and Frequency Scaling)** with discrete power states (P-states), not continuous scaling. When the junction temperature hits a threshold (e.g., 80°C on Jetson Orin), the SoC drops to the next lower P-state — a discrete frequency/voltage step. Each step reduces both performance and power dissipation. The "staircase" pattern occurs because: (1) the system runs at full speed until T_junction hits 80°C, (2) drops to P-state 1 (lower clock → less heat → temperature stabilizes temporarily), (3) if ambient heat accumulation continues, temperature rises again and hits the next threshold, triggering P-state 2. To design around this: (a) profile your workload's *sustained* thermal power, not peak; (b) use the Orin's power mode presets (e.g., 30W mode) to voluntarily cap below the thermal ceiling from the start — a steady 20 FPS is better than 25 FPS that decays to 14; (c) implement workload shedding: drop the behavior analysis model when thermal headroom is low, preserving detection and face recognition at full frame rate.

> **Napkin Math:** Orin at MAXN (60W): 275 TOPS, T_junction hits 80°C in ~120s at 40°C ambient. P-state 1 (~40W): ~180 TOPS → 20 FPS. P-state 2 (~25W): ~110 TOPS → 14 FPS. Voluntary 30W mode from boot: ~150 TOPS sustained indefinitely → steady 18 FPS with no degradation. The "slow but steady" mode delivers more total frames over 10 minutes: 18 × 600 = 10,800 vs (25 × 180 + 20 × 180 + 14 × 240) = 4,500 + 3,600 + 3,360 = 11,460 — and without the unpredictable drops that break downstream tracking.

> **Key Equation:** $P_{\text{dynamic}} = C \times V^2 \times f \quad \Rightarrow \quad \text{halving } f \text{ allows } V \text{ to drop, giving cubic power reduction}$

**📖 Deep Dive:** [Volume I: Benchmarking](https://mlsysbook.ai/vol1/benchmarking.html)
</details>

---

### 🛡️ Functional Safety & Graceful Degradation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Degradation Ladder</b> · <code>functional-safety</code></summary>

**Interviewer:** "You're the ML systems architect for an autonomous delivery robot. Your perception stack runs three models on a Jetson Orin: a primary YOLOv8-L detection model (22ms, 43.7 mAP), a semantic segmentation model (15ms), and a depth estimation model (12ms). During a delivery, the Orin's GPU develops a hardware fault — the DLA is still functional but the GPU CUDA cores are offline. Your total compute budget just dropped from 275 TOPS to 100 TOPS (DLA only). Design the graceful degradation strategy from first principles."

**Common Mistake:** "Switch to a smaller model" or "Just run everything on the DLA." Neither addresses the systematic design of a degradation ladder that preserves safety invariants.

**Realistic Solution:** Design a **degradation ladder** — a pre-planned sequence of capability reductions that preserves safety invariants at each level.

**Level 0 (Nominal):** All three models on GPU — full perception, 30 FPS, 49ms total.

**Level 1 (GPU fault → DLA only, 100 TOPS):** The DLA supports INT8 only and has limited layer support. Pre-compile a DLA-optimized YOLOv8-S (INT8, 7ms on DLA, 37.4 mAP) and drop segmentation entirely. Depth estimation is replaced by stereo disparity (classical algorithm on CPU, ~10ms). Total: 17ms/frame = 58 FPS. You trade 6 mAP points and lose semantic segmentation, but maintain the safety-critical function: obstacle detection + distance estimation.

**Level 2 (DLA overtemp → CPU only):** Fall back to a MobileNet-SSD (INT8, ~80ms on CPU ARM cores). Frame rate drops to ~12 FPS. The robot reduces speed to 0.5 m/s (walking pace) and activates ultrasonic proximity sensors as primary collision avoidance. The neural network becomes advisory, not primary.

**Level 3 (Complete compute failure):** Pure reactive safety — ultrasonic stop, hazard lights, cellular alert to operator. No ML inference.

The key principle: each level must be **pre-validated** (models pre-compiled, latency pre-measured, safety cases pre-certified). You cannot compile a TensorRT engine on the fly during a fault — that takes minutes. Every fallback model must be resident on disk and loadable in <500ms.

> **Napkin Math:** DLA-only budget: 100 TOPS INT8. YOLOv8-S INT8: ~7 GOPS × 1000/7ms = ~1 TOPS utilized → 1% of DLA capacity. Stereo disparity on 4× ARM A78AE cores: ~10ms. Total pipeline: 17ms → 58 FPS. Storage for fallback models: YOLOv8-S INT8 (~6 MB) + MobileNet-SSD INT8 (~3 MB) = 9 MB — negligible on a 64 GB eMMC.

**📖 Deep Dive:** [Volume I: Robust AI](https://mlsysbook.ai/vol1/robust_ai.html)
</details>

---

### 🔄 Model Updates & OTA Deployment

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The OTA Brick Risk</b> · <code>model-update</code></summary>

**Interviewer:** "You manage a fleet of 10,000 edge AI cameras running object detection for a smart city project. Each camera has a Hailo-8 accelerator (26 TOPS), 4 GB RAM, and 16 GB eMMC storage. You need to deploy an updated detection model. Your colleague suggests: 'Just push the new model binary over the air and restart inference.' What can go wrong, and what is the correct deployment strategy?"

**Common Mistake:** "OTA model updates are just file transfers — what could go wrong?" This ignores the failure modes unique to constrained, remote devices.

**Realistic Solution:** A naive OTA push has at least four catastrophic failure modes on edge devices: (1) **Power loss during write** — if the camera loses power mid-flash, the model file is corrupted and the device is bricked with no local operator to recover it. (2) **Storage exhaustion** — the new model (say 12 MB) must coexist with the old model during the update; on 16 GB eMMC already 80% full with video buffer, you may not have room for both. (3) **Incompatible runtime** — the new model was compiled for Hailo RT v4.18 but the device runs v4.16; inference crashes silently or produces garbage outputs. (4) **Fleet-wide correlated failure** — pushing to all 10,000 devices simultaneously means a bad model bricks the entire fleet at once.

The correct strategy is **A/B partitioned deployment**: maintain two model slots (A and B) on each device. Write the new model to the inactive slot. Validate it by running inference on a test image with a known-good output hash. Only then atomically swap the active pointer. If validation fails, the device continues running the old model and reports the failure. Roll out in waves: 1% → 10% → 50% → 100%, with automatic rollback triggers (e.g., if >5% of a wave reports validation failure, halt the rollout).

> **Napkin Math:** Model size: 12 MB. A/B slots: 24 MB reserved. Remaining on 16 GB eMMC: ~12.8 GB for video buffer (sufficient). OTA bandwidth per device: 12 MB over LTE at 5 Mbps = ~20 seconds. Fleet of 10,000 in 4 waves: wave 1 (100 devices) = 20s, validate 10 min, wave 2 (900) = 20s, validate 10 min, wave 3 (4,000) = 20s, wave 4 (5,000) = 20s. Total rollout: ~40 minutes with validation gates.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

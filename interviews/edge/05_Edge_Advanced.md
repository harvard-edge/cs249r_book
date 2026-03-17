# Round 5: Advanced Edge Systems 🔬

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Edge_Systems.md">🤖 Round 1</a> ·
  <a href="02_Edge_Constraints.md">⚖️ Round 2</a> ·
  <a href="03_Edge_Ops_and_Deployment.md">🚀 Round 3</a> ·
  <a href="04_Edge_Visual_Debugging.md">🖼️ Round 4</a> ·
  <a href="05_Edge_Advanced.md">🔬 Round 5</a>
</div>

---

This round covers the hardest problems in edge ML systems: safety certification, multi-sensor architectures at scale, adversarial robustness in the physical world, and designing systems that must operate for years without human intervention. These are the questions that separate senior engineers from architects.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/edge/05_Edge_Advanced.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 🛡️ Safety Certification & Functional Safety

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The ISO 26262 Neural Network Problem</b> · <code>functional-safety</code></summary>

**Interviewer:** "Your autonomous vehicle uses a neural network for pedestrian detection. The safety team says you need ISO 26262 ASIL-D certification for this function (the highest automotive safety integrity level). But ISO 26262 was written for deterministic software — it requires 100% code coverage, formal verification, and traceable requirements. Neural networks are stochastic, opaque, and their 'requirements' are learned from data. How do you certify a neural network under ISO 26262?"

**Common Mistake:** "Apply formal verification to the neural network." Formal verification of a 25-million-parameter network is computationally intractable — the state space is too large.

**Realistic Solution:** You don't certify the neural network itself to ASIL-D. You certify the **system architecture** to ASIL-D, with the neural network as an ASIL-QM (no safety rating) component wrapped in safety mechanisms:

**(1) ASIL decomposition.** The pedestrian detection function is ASIL-D, but you decompose it into: (a) a neural network detector (ASIL-QM — no safety claim on the NN itself), (b) a plausibility checker (ASIL-B — deterministic software that validates NN outputs against physical constraints), (c) a safety monitor (ASIL-D — a simple, formally verifiable system that triggers emergency braking if the NN + plausibility checker disagree or fail to produce output within the WCET deadline).

**(2) The plausibility checker** is deterministic and certifiable: it verifies that detections are physically consistent (bounding boxes have reasonable aspect ratios, objects don't teleport between frames, detection counts are within expected ranges for the scene type). It rejects ~2% of NN outputs as implausible.

**(3) The safety monitor** is a small, formally verified state machine (~500 lines of C, 100% MC/DC coverage) that monitors: (a) NN output within WCET, (b) plausibility checker agreement, (c) sensor health (camera not occluded, LiDAR returning points). If any check fails, it triggers the safe state (emergency braking via a hardwired path that bypasses all software).

**(4) Validation through testing.** Since you can't formally verify the NN, you validate it empirically: run it on millions of test scenarios (real + synthetic) and demonstrate that the residual risk (probability of undetected pedestrian × severity) is below the ASIL-D target ($10^{-8}$ per hour of operation). This requires ~$10^9$ test miles or equivalent simulation.

> **Napkin Math:** ASIL-D target: <10⁻⁸ failures/hour. At 30 FPS: 108,000 frames/hour. Allowed undetected pedestrians: <1 per 10⁸ hours = <1 per 11,415 years. To validate this statistically at 95% confidence: need ~3 × 10⁸ / failure_rate test frames ≈ 3 × 10¹⁶ frames. At 30 FPS: 3.2 × 10⁷ years of real driving. This is why simulation is mandatory — you can run 10,000× real-time in parallel.

**📖 Deep Dive:** [Volume II: Robust AI](https://mlsysbook.ai/vol2/robust_ai.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Graceful Degradation Under Sensor Failure</b> · <code>functional-safety</code></summary>

**Interviewer:** "Your autonomous delivery robot has 3 sensors: a stereo camera pair, a 2D LiDAR, and an ultrasonic array. During a delivery, mud splashes onto the left stereo camera lens. Your perception stack loses stereo depth estimation. The robot is 500 meters from its destination on a busy sidewalk. What happens next?"

**Common Mistake:** "Stop immediately and wait for human intervention." Stopping on a busy sidewalk creates its own safety hazard (tripping, blocking wheelchair access). The system must degrade gracefully, not fail catastrophically.

**Realistic Solution:** The system enters a **degraded perception mode** based on a pre-defined sensor health matrix:

**Sensor health assessment (runs every 100ms):** The stereo camera self-test detects the occlusion by comparing left/right image brightness histograms. When they diverge beyond a threshold (left camera brightness drops 80%), the system flags "left camera degraded" within 200ms.

**Degradation response:** (1) Disable stereo depth — switch to monocular depth estimation from the right camera only. Monocular depth is less accurate (±30% at 5m vs ±5% for stereo) but sufficient for obstacle avoidance at reduced speed. (2) Increase reliance on 2D LiDAR for obstacle distance — LiDAR provides accurate range in a horizontal plane, compensating for monocular depth uncertainty in the vertical axis. (3) Reduce speed from 1.5 m/s to 0.5 m/s — at lower speed, the reduced perception accuracy still provides sufficient stopping distance. (4) Expand the ultrasonic safety envelope from 0.5m to 1.5m — ultrasonic provides reliable close-range detection regardless of visual conditions. (5) Alert the fleet management system — request remote operator oversight. If the operator doesn't respond in 60 seconds, navigate to the nearest safe parking spot (pre-mapped) and stop.

The key principle: every degradation level must be pre-validated. You can't design the fallback behavior at runtime — it must be tested and certified before deployment.

> **Napkin Math:** Full perception: stereo depth ±5% at 5m, 1.5 m/s, stopping distance = 0.5m. Degraded: monocular depth ±30% at 5m (effective range uncertainty: 3.5-6.5m), 0.5 m/s, stopping distance = 0.17m. Safety margin with ultrasonic (1.5m envelope): even with 30% depth error, the ultrasonic provides a hard 1.5m safety boundary. Time to destination at 0.5 m/s: 500m / 0.5 = 1000s = 16.7 minutes (vs 5.6 minutes at full speed). Acceptable for a delivery robot.

**📖 Deep Dive:** [Volume I: Robust AI](https://mlsysbook.ai/vol1/robust_ai.html)
</details>

---

### 📡 Multi-Sensor Architecture

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Early vs Late Fusion Trade-off</b> · <code>sensor-fusion</code> <code>architecture</code></summary>

**Interviewer:** "Your autonomous vehicle has a camera and a LiDAR. You need to detect and localize objects in 3D. Your team is debating two architectures: early fusion (concatenate raw camera pixels + LiDAR points, feed to one model) vs late fusion (run separate camera and LiDAR detectors, merge results). What are the system-level trade-offs, and which do you recommend for a Jetson Orin deployment?"

**Common Mistake:** "Early fusion is always better because the model sees all the data." This ignores the computational and engineering costs.

**Realistic Solution:** The trade-offs are fundamentally about compute, robustness, and maintainability:

**Early fusion (single model, raw inputs):**
- *Pros:* The model can learn cross-modal features (e.g., a shadow in the camera image correlates with a gap in the LiDAR point cloud). Higher theoretical accuracy ceiling.
- *Cons:* (1) Massive input tensor — a 1080p image (6 MB) + a 100K-point LiDAR cloud (1.2 MB) = 7.2 MB input per frame. The model must process heterogeneous data types, requiring custom architectures (BEVFusion, TransFusion) that are 3-5× more compute-intensive than single-modal detectors. (2) Single point of failure — if either sensor degrades, the entire model's accuracy drops because it was trained on paired data. (3) Retraining required if you change either sensor (new camera resolution, new LiDAR model).

**Late fusion (separate models, merged outputs):**
- *Pros:* (1) Each model is optimized for its modality — camera detector runs on GPU, LiDAR detector runs on DLA. Total compute is often *less* than early fusion. (2) Sensor independence — if the camera fails, the LiDAR detector still produces 3D detections (at reduced accuracy). (3) Modular — swap the camera or LiDAR model independently without retraining the other.
- *Cons:* Cannot learn cross-modal correlations. Merging 2D camera detections with 3D LiDAR detections requires geometric projection (camera intrinsics/extrinsics), which introduces calibration errors.

**Recommendation for Jetson Orin:** Late fusion. The Orin's DLA + GPU architecture naturally maps to two independent detectors. Camera YOLO on GPU (18ms) + LiDAR PointPillars on DLA (12ms) = 18ms total (parallel execution). BEVFusion on GPU alone: ~45ms. Late fusion is 2.5× faster and provides sensor-failure resilience. The 2-3% accuracy gap vs early fusion is acceptable given the robustness and latency advantages.

> **Napkin Math:** Early fusion (BEVFusion): ~120 GFLOPs, 45ms on Orin GPU. Late fusion: Camera YOLO (28 GFLOPs, 18ms GPU) + LiDAR PointPillars (18 GFLOPs, 12ms DLA) = 18ms parallel. Speedup: 2.5×. Accuracy: early fusion 72.9 NDS on nuScenes, late fusion ~70.1 NDS. The 2.8 NDS gap costs 2.5× the compute.

**📖 Deep Dive:** [Volume I: Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Sensor Calibration Drift</b> · <code>sensor-fusion</code></summary>

**Interviewer:** "Your autonomous vehicle's camera-LiDAR fusion system was calibrated in the factory. After 6 months of operation (vibration, temperature cycles, minor impacts), the extrinsic calibration between the camera and LiDAR has drifted by 0.3°. Your 3D bounding boxes are now offset by 0.5m at 10m range. The safety team wants recalibration every month, which requires taking the vehicle offline for 2 hours. Design an online calibration system that maintains accuracy without downtime."

**Common Mistake:** "Run a calibration routine at startup." Startup calibration helps but doesn't catch drift that occurs during operation (thermal expansion, vibration settling).

**Realistic Solution:** Design a **continuous online calibration** system that runs in the background during normal operation:

**(1) Calibration signal extraction:** During normal driving, identify naturally occurring calibration targets: lane markings (visible in both camera and LiDAR), building edges (strong features in both modalities), and pole-like objects (traffic signs, lamp posts). These features are detected independently in each modality and provide correspondence points.

**(2) Incremental optimization:** Accumulate correspondence points over a sliding window (last 1000 frames, ~33 seconds). Run a lightweight optimization (Levenberg-Marquardt on 6-DOF extrinsics) on the CPU in a background thread. The optimization takes ~50ms on 4 ARM cores — negligible compared to the 33ms frame budget on the GPU.

**(3) Smooth update:** Don't apply the new calibration instantly — that would cause a discontinuity in the fusion output. Interpolate between the old and new extrinsics over 30 frames (1 second) using spherical linear interpolation (SLERP) for rotation and linear interpolation for translation.

**(4) Validation gate:** Before applying any calibration update, verify that the reprojection error decreased. If the new calibration is worse (optimization diverged, or the "calibration targets" were actually moving objects), reject the update and keep the current calibration.

**(5) Drift monitoring:** Track the magnitude of calibration corrections over time. If corrections exceed 1° in any axis over a week, flag the vehicle for physical inspection — the sensor mount may be loose.

> **Napkin Math:** 0.3° rotation drift at 10m range: offset = 10m × tan(0.3°) = 0.052m per 0.1° ≈ 0.16m at 0.3°. At 50m: 0.26m. At 100m: 0.52m. Online calibration accuracy: ±0.05° (reduces offset to <0.1m at 100m). Compute cost: 50ms CPU every 33 seconds = 0.15% CPU utilization. Downtime saved: 2 hours/month × 12 months × $200/hour (vehicle opportunity cost) = $4,800/year per vehicle.

**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

---

### 🏭 Industrial Edge & Long-Term Reliability

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The eMMC Wear-Out Problem</b> · <code>reliability</code></summary>

**Interviewer:** "Your edge device writes inference logs to its eMMC flash storage at 10 MB/minute. The eMMC is rated for 3,000 P/E (program/erase) cycles. The device has a 32 GB eMMC. How long until the storage fails, and how do you prevent it?"

**Common Mistake:** "32 GB × 3,000 cycles = 96 TB total writes. At 10 MB/min = 14.4 GB/day. 96 TB / 14.4 GB = 6,667 days = 18 years. No problem." This assumes perfect wear leveling.

**Realistic Solution:** The 96 TB calculation assumes writes are evenly distributed across all blocks (perfect wear leveling). In practice, log files are written as small synchronous writes (4 KB each), but eMMC must erase entire blocks (128-512 KB) before rewriting. This creates a write amplification factor (WAF) of 10-50× for small writes — writing 4 KB of log data may erase and rewrite a 512 KB block. Additionally: (1) ext4 filesystem journal doubles effective writes, (2) temporary files from TensorRT (workspace, profiling), (3) OS swap if memory pressure occurs. Realistic WAF for logging workloads: 10-30×. Effective writes: 10 MB/min × 20 (WAF) = 200 MB/min = 288 GB/day. With imperfect wear leveling (hot spot factor 2×): effective endurance = 96 TB / 2 = 48 TB. Lifetime: 48 TB / 288 GB = 167 days = **5.5 months**. For a device expected to last 5-7 years, this is a ticking time bomb that will kill your fleet within the first year.

Fixes: (1) **Log to RAM (tmpfs)** and only flush aggregates to eMMC hourly. Reduces writes from 288 GB/day to ~100 MB/day. (2) **Read-only root filesystem** — mount the OS partition as read-only, eliminating journal writes. Use an overlay filesystem (overlayfs) for temporary state. (3) **Disable swap** — if the ML workload OOMs, it should crash and restart, not swap to eMMC. (4) **Monitor eMMC health** — read the eMMC's internal wear indicators (SMART-like attributes via MMC_IOC commands) and alert when remaining life drops below 20%.

> **Napkin Math:** Naive with WAF: 96 TB / 288 GB/day = 333 days (under 1 year). With tmpfs logging: writes drop to 100 MB/day. New lifetime: 96 TB / 100 MB = 960,000 days = **2,630 years**. Even with 10× residual WAF: 263 years. The fix extends lifetime by 1000×.

**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The 5-Year Edge Device Lifecycle</b> · <code>reliability</code> <code>deployment</code></summary>

**Interviewer:** "You're designing an edge AI system for industrial quality inspection. The customer requires a 5-year operational lifetime with <1 hour of unplanned downtime per year (99.99% availability). The system runs 24/7 in a factory with ambient temperatures of 30-45°C. What are the failure modes you must design for, and how do you achieve the availability target?"

**Common Mistake:** "Use enterprise-grade hardware and it'll be fine." Enterprise hardware helps, but software failures dominate in ML systems.

**Realistic Solution:** Failure mode analysis for a 5-year edge ML deployment:

**Hardware failures (MTBF-driven):**
- eMMC wear-out: mitigate with read-only rootfs + tmpfs logging (see previous question).
- Fan failure (if applicable): use fanless design with passive cooling. Fans have MTBF of ~50,000 hours (5.7 years) — too close to the 5-year target.
- Power supply degradation: use industrial-rated PSU with >100,000 hour MTBF. Add a UPS (supercapacitor) for graceful shutdown during power glitches.
- DRAM bit errors: enable ECC RAM. At 45°C, soft error rate increases 10× vs 25°C.

**Software failures (the dominant source):**
- Model accuracy drift: the factory environment changes (new product variants, lighting changes, conveyor speed changes). Plan for quarterly model retraining with on-site data collection.
- Memory leaks: long-running inference processes accumulate leaked memory over weeks. Implement a scheduled daily restart during the maintenance window (e.g., 2 AM shift change).
- OTA update failures: use A/B partitioning with automatic rollback.
- Dependency rot: pin all software versions. A system that auto-updates CUDA or TensorRT will eventually break.

**Availability math:** 99.99% = 52.6 minutes of downtime/year. Budget: planned restarts (365 × 30s = 3 hours — must be during scheduled maintenance, not counted as "unplanned"). Unplanned: hardware failure (1 event/year × 30 min recovery with hot spare) + software crash (12 events/year × 2 min auto-restart) = 30 + 24 = **54 minutes**. Barely meets target. To add margin: keep a hot spare device that takes over within 5 seconds via a hardware failover switch.

> **Napkin Math:** 5 years = 43,800 hours. 99.99% availability = 4.38 hours allowed downtime. Unplanned budget: 52.6 min/year × 5 years = 263 minutes total. With hot spare (5s failover): each failure costs 5s instead of 30 min. Can tolerate 263 min / 0.083 min = 3,168 failures over 5 years = 1.7 failures/day. Extremely robust.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

---

### 🔐 Privacy-Preserving Edge ML

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Privacy-Preserving Drift Correction</b> · <code>privacy</code> <code>monitoring</code></summary>

**Interviewer:** "Your edge AI system monitors patients in a hospital for fall detection. After 6 months, you detect model drift — accuracy has dropped 8%. The obvious fix is to collect recent data from the devices and retrain. But HIPAA regulations forbid uploading patient images to your cloud training servers. How do you fix the model without ever seeing the raw data?"

**Common Mistake:** "Anonymize the images before uploading." De-identification of medical images is legally complex, often insufficient (faces can be reconstructed from body shape), and doesn't address the fundamental privacy constraint.

**Realistic Solution:** Three approaches, in order of increasing sophistication:

**(1) Federated Learning.** Each edge device fine-tunes the model locally on its own data. Instead of uploading raw data, devices upload only gradient updates (or gradient differences) to a central server, which aggregates them into a global model update. Privacy guarantee: the server never sees raw images. Add **differential privacy** (clip gradients and add calibrated noise) to prevent gradient inversion attacks that could reconstruct training images from gradients. Trade-off: convergence is 3-5× slower than centralized training, and DP noise reduces final accuracy by 1-3%.

**(2) On-device active learning with embedding upload.** Each device runs inference and identifies "hard" samples (low confidence, high entropy). Instead of uploading the raw image, the device uploads only the penultimate-layer embedding (a 512-dimensional vector, ~2 KB). The cloud uses these embeddings to identify distribution shift (cluster analysis, drift detection) and selects which synthetic training examples to generate. The synthetic data is used for retraining. Privacy: embeddings are much harder to invert than raw images, and adding noise to embeddings provides additional protection.

**(3) Synthetic data augmentation.** Use the drift signal (confidence distributions, detection count anomalies) to characterize *what kind* of data the model is failing on (e.g., "nighttime scenes with wheelchair users"). Generate synthetic training data matching these characteristics using a generative model. Retrain on the synthetic data. No real patient data ever leaves the device. Trade-off: synthetic data may not capture the full complexity of real-world distribution shift.

In practice, combine all three: federated learning for model updates, embedding-based drift analysis for diagnosis, and synthetic data to supplement the federated training.

> **Napkin Math:** Federated learning: 50 edge devices × 100 local training steps × 10 MB gradient update = 500 MB total upload per round. With gradient compression (top-k sparsification): 50 MB per round. 10 rounds to convergence: 500 MB total. DP noise (ε=8, δ=10⁻⁵): accuracy cost ~2%. Embedding upload: 512 floats × 4 bytes × 1000 hard samples = 2 MB per device. Total for 50 devices: 100 MB. Synthetic data generation: 10,000 images × 100 KB = 1 GB training set, generated in ~2 hours on a cloud GPU.

**📖 Deep Dive:** [Volume II: Security & Privacy](https://mlsysbook.ai/vol2/security_privacy.html)
</details>

---

### 🌐 Edge-Cloud Hybrid Architecture

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Edge-Cloud Split Inference</b> · <code>architecture</code> <code>latency</code></summary>

**Interviewer:** "Your edge camera needs to run a large model (ResNet-152, 60 GFLOPs) that doesn't fit the 33ms latency budget on a Jetson Nano (128 CUDA cores). The camera has a 50ms round-trip to a cloud GPU. Your colleague says 'just run inference in the cloud.' Why might splitting the model between edge and cloud be better than either extreme?"

**Common Mistake:** "Cloud inference is always faster because cloud GPUs are bigger." This ignores network latency and bandwidth constraints.

**Realistic Solution:** Pure cloud inference: 50ms network RTT + 5ms cloud inference = 55ms — over budget. Pure edge inference: 80ms on the Nano — over budget. But the model has a natural split point.

**Split inference:** Run the first 50 layers (the feature extractor, 20 GFLOPs) on the edge device. This produces a 7×7×2048 feature tensor = 401 KB. Transmit this tensor to the cloud (401 KB at 10 Mbps = 0.3ms). Run the remaining 102 layers (classifier head, 40 GFLOPs) in the cloud (3ms). Total: 30ms (edge) + 0.3ms (upload) + 25ms (half-RTT) + 3ms (cloud) = **58.3ms**. Still over budget.

The real win: the edge device compresses the feature tensor before transmission. Apply 4× channel pruning to the split point: 7×7×512 = 100 KB. With INT8 quantization of the feature tensor: 50 KB. Upload: 0.04ms. And the edge portion is now faster because fewer output channels: 15ms. Total: 15ms + 0.04ms + 25ms + 3ms = **43ms**. Still over 33ms.

Final optimization: the edge device runs inference on frame N while uploading frame N-1's features. Pipelined: effective latency = max(15ms edge, 28ms cloud) = **28ms**. Under budget. The edge device does useful work (feature extraction) while waiting for the network, and the cloud does the heavy computation. Neither device is idle.

> **Napkin Math:** Pure edge (Nano): 80ms. Pure cloud: 55ms. Split (naive): 58ms. Split (compressed + pipelined): 28ms. Bandwidth: 50 KB × 30 FPS = 1.5 MB/s = 12 Mbps — fits in a typical LTE connection. Cloud cost: 1 GPU serves ~30 edge devices (3ms per inference, 30 FPS = 90ms/s of GPU time per device, 1000ms/90ms = 11 devices per GPU, with batching: ~30).

**📖 Deep Dive:** [Volume II: Inference at Scale](https://mlsysbook.ai/vol2/inference.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Offline-First Edge Design</b> · <code>deployment</code> <code>reliability</code></summary>

**Interviewer:** "Your edge AI system monitors a remote oil pipeline in the Alaskan wilderness. It has satellite connectivity that works 4 hours per day (weather-dependent) with 256 Kbps bandwidth. The system must detect pipeline leaks 24/7. How do you design the system to operate independently of cloud connectivity?"

**Common Mistake:** "Buffer all data and upload when connectivity is available." At 30 FPS of inference results, you'd generate gigabytes per day — impossible to upload at 256 Kbps.

**Realistic Solution:** Design for **offline-first operation** where the cloud is a luxury, not a dependency:

**(1) On-device inference and decision-making.** The leak detection model runs entirely on-device. Detection → alert → actuator (close valve) happens locally with zero cloud dependency. Latency: <100ms from detection to valve closure command.

**(2) Tiered data storage.** Store 3 tiers locally: (a) Last 24 hours of raw video on a 256 GB NVMe SSD (at 2 Mbps compressed: 21.6 GB/day). (b) Last 30 days of detection events with metadata (timestamps, confidence, bounding boxes): ~50 MB. (c) Last 365 days of hourly aggregates: ~5 MB.

**(3) Satellite upload priority queue.** During the 4-hour connectivity window at 256 Kbps = 115 MB total: Priority 1: alert notifications (leak detected, system health critical) — <1 KB each, sent immediately. Priority 2: daily aggregate report (detection counts, system health, model metrics) — ~50 KB. Priority 3: thumbnail images of detected events — 10 KB each × 100 events = 1 MB. Priority 4: model update download (if available) — up to 10 MB. Priority 5: raw video clips of critical events — remaining bandwidth. Total: ~12 MB of high-priority data easily fits in the 115 MB window, leaving ~100 MB for video clips.

**(4) Autonomous model health monitoring.** Without cloud-based drift detection, the device must self-monitor: track confidence score distributions, detection frequency, and inference latency. If any metric deviates >3σ from the 30-day rolling baseline, flag for priority upload and request human review during the next connectivity window.

> **Napkin Math:** Satellite window: 4h × 256 Kbps = 460 MB. Usable (protocol overhead): ~350 MB. Daily upload: 50 KB (report) + 1 MB (thumbnails) + 10 MB (model update, weekly amortized = 1.4 MB/day) = ~2.5 MB/day. Remaining: 347 MB for video = 347 MB / 2 Mbps compression = 23 minutes of video per day. Sufficient for all critical events.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

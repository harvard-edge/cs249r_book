# Round 3: Operations & Deployment 🚀

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

Deploying a model to one edge device is engineering. Deploying it to 10,000 devices — and keeping them running for years — is operations. This round tests whether you can reason about fleet management, OTA updates, model optimization pipelines, monitoring without cloud connectivity, and security in physically accessible environments.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/edge/03_Edge_Ops_and_Deployment.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 🔧 Model Optimization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Optimization Ladder</b> · <code>optimization</code></summary>

- **Interviewer:** "Your YOLOv8-S runs at 15 FPS on a Jetson Orin NX. You need 30 FPS. Your team immediately starts designing a custom smaller architecture. Why is this the wrong first step?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We need a smaller model — let's try YOLOv8-N or train a custom architecture." Architecture changes are the most expensive optimization and should be the last resort.

  **Realistic Solution:** Follow the optimization ladder — a prioritized sequence from cheapest to most expensive:

  **Step 1: TensorRT with FP16** (effort: minutes). Export your PyTorch model to ONNX, compile with TensorRT. Layer fusion, kernel auto-tuning, and FP16 Tensor Core utilization typically give 1.5-2× speedup for free. 15 FPS → 22-30 FPS.

  **Step 2: INT8 quantization** (effort: hours). Calibrate with 1000 representative images. Another 1.5-2× on top of FP16. 22 FPS → 33-44 FPS.

  **Step 3: Input resolution reduction** (effort: minutes). Drop from 640×640 to 512×512 or 480×480. FLOPs scale quadratically with resolution: (480/640)² = 0.56× FLOPs. Accuracy drops ~1-2% mAP.

  **Step 4: Structured pruning** (effort: days). Remove channels with lowest L1-norm. 20-40% channel reduction for 1-2% mAP loss. Requires fine-tuning.

  **Step 5: Architecture change** (effort: weeks-months). Only if steps 1-4 are insufficient. Design or adopt a smaller architecture (YOLOv8-N, EfficientDet-Lite).

  Most teams jump straight to Step 5, leaving 3-4× of free performance on the table from Steps 1-2.

  > **Napkin Math:** Baseline (PyTorch FP32): 15 FPS. After Step 1 (TensorRT FP16): 15 × 1.8 = 27 FPS. After Step 2 (INT8): 27 × 1.7 = 46 FPS. Already 1.5× over target — no need for Steps 3-5. Engineering time: 2 hours vs 2 months for a custom architecture.

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Pruning Paradox</b> · <code>optimization</code></summary>

- **Interviewer:** "You structured-pruned 40% of channels from your detection model. FLOPs dropped 40%. But when you benchmark on the Hailo-8, latency only dropped 10%. On the Jetson Orin, it dropped 35%. Why does the same pruning give wildly different speedups on different hardware?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The Hailo's compiler isn't optimized for pruned models." The compiler is fine — the issue is architectural.

  **Realistic Solution:** Edge accelerators have fundamentally different execution models. The **Jetson Orin (GPU)** executes layers as CUDA kernels with configurable thread blocks. Fewer channels = fewer threads = proportional speedup (with some overhead). The **Hailo-8 (dataflow)** maps the model onto a fixed spatial pipeline at compile time. Each layer is assigned physical compute units based on its original shape. When you prune a layer from 64 to 38 channels, the Hailo's compiler must still allocate compute units in multiples of its native SIMD width (typically 8 or 16). A 38-channel layer executes as if it had 48 channels (rounded up to the next multiple of 16), wasting 21% of the allocated compute. Across many layers, these rounding penalties accumulate, eroding the theoretical 40% FLOP reduction to a 10% latency reduction. Fix: use **hardware-aware pruning** that constrains channel counts to multiples of the target hardware's native width. Prune from 64 to 32 channels (50% reduction, but hardware-aligned) instead of 64 to 38 (41% reduction, but misaligned).

  > **Napkin Math:** 10 layers, each pruned from 64 to 38 channels. Hailo SIMD width = 16. Effective channels per layer: 48 (rounded up). Effective FLOP reduction: (64-48)/64 = 25%, not 40%. With hardware-aware pruning to 32 channels: effective reduction = 50%, and every channel is utilized. Latency improvement: Hailo ~45% (vs 10% naive), Orin ~48% (vs 35% naive).

  📖 **Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The TensorRT Engine Portability Trap</b> · <code>optimization</code></summary>

- **Interviewer:** "You compiled a TensorRT INT8 engine on your development Jetson AGX Orin (64 GB, Ampere GPU). You copy the engine file to a production Jetson Orin NX (8 GB, same Ampere architecture). Inference crashes immediately. The GPU architecture is the same. What went wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model is too large for 8 GB." The model is only 12 MB — memory isn't the issue.

  **Realistic Solution:** TensorRT engines are **not portable** between different GPU configurations, even within the same architecture family. The engine encodes hardware-specific optimizations at compile time: (1) **CUDA core count** — the AGX Orin has 2048 CUDA cores; the Orin NX has 1024. Kernel launch configurations (thread blocks, grid dimensions) are baked into the engine for the specific core count. (2) **Memory bandwidth** — the AGX has 204.8 GB/s; the NX has 102.4 GB/s. TensorRT's auto-tuner selects different kernel implementations based on the compute-to-bandwidth ratio. (3) **DLA availability** — if the engine was compiled with DLA layers on the AGX (which has 2 DLAs), but the NX has different DLA capabilities, those layers fail. (4) **TensorRT version** — even minor version differences can break engine compatibility. The fix: compile engines on the target hardware, or use ONNX as the portable format and compile on each device at first boot (caching the result).

  > **Napkin Math:** ONNX model: 12 MB (portable). TensorRT compilation on Orin NX: ~45 seconds at first boot. Cached engine: loads in <1 second on subsequent boots. Fleet of 1000 devices: each compiles its own engine once. Total fleet compilation time: 45s × 1000 = 12.5 hours if sequential, but each device compiles independently at boot.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)

  </details>

</details>

---

### 🚀 Deployment & Fleet Management

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The A/B Partition Scheme</b> · <code>deployment</code></summary>

- **Interviewer:** "You manage 10,000 edge cameras deployed across a city. You need to update the detection model on all of them. Your colleague says 'just SSH in and copy the new model file.' What are the four ways this can catastrophically fail, and what is the correct deployment architecture?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "OTA updates are just file copies — what could go wrong?" Everything.

  **Realistic Solution:** Four failure modes of naive OTA: (1) **Power loss during write** — the camera loses power mid-copy. The old model is partially overwritten, the new model is incomplete. The device boots to a corrupted model and is bricked. No one can physically access a camera on a pole 30 feet up. (2) **Storage exhaustion** — the new model must coexist with the old model during the copy. If storage is 80% full with video buffer, there's no room for both. (3) **Runtime incompatibility** — the new model was compiled for TensorRT 8.6 but the device runs 8.5. Inference crashes or produces garbage. (4) **Fleet-wide failure** — pushing to all 10,000 devices simultaneously means a bad model bricks the entire fleet at once.

  The correct architecture is **A/B partitioning**: the device has two model slots (A and B). The running system uses slot A. The new model is written to slot B while A continues serving. After the write completes, the device runs a validation check — inference on a test image with a known-good output hash. Only if validation passes does the bootloader atomically switch the active pointer to B. If validation fails, or if the device fails to boot from B (watchdog timeout), it automatically reverts to A. Roll out in waves: 1% → 10% → 50% → 100%, with automatic rollback if >5% of any wave reports failure.

  > **Napkin Math:** Model: 12 MB. A/B slots: 24 MB reserved. OTA over LTE (5 Mbps): 12 MB × 8 / 5 Mbps = 19.2 seconds per device. Wave 1 (100 devices): 20s transfer + 10 min validation. Wave 2 (900): 20s + 10 min. Wave 3 (4000): 20s + 10 min. Wave 4 (5000): 20s. Total safe rollout: ~40 minutes. Naive push to all 10,000: 20 seconds — but one bad model = 10,000 bricked cameras.

  📖 **Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Fleet Heterogeneity Problem</b> · <code>deployment</code></summary>

- **Interviewer:** "Your company deployed edge AI cameras over 3 years. The fleet now contains: 2,000 devices with Jetson Nano (128 CUDA cores, 4 GB RAM), 5,000 with Jetson Xavier NX (384 CUDA cores, 8 GB RAM), and 3,000 with Jetson Orin NX (1024 CUDA cores, 16 GB RAM). You need to deploy a single updated detection model across the entire fleet. How do you handle the 8× compute gap between the weakest and strongest devices?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Train one model and compile it for each platform." A model that runs well on the Orin will OOM or miss deadlines on the Nano.

  **Realistic Solution:** You need a **model tiering strategy** — multiple model variants compiled from the same training run, each targeting a hardware tier:

  **Tier 1 (Nano):** YOLOv8-N, INT8, 320×320 input. ~6 MB, ~8 FPS. Detects large/medium objects only. Confidence threshold raised to 0.6 to reduce NMS load.

  **Tier 2 (Xavier NX):** YOLOv8-S, INT8, 480×480 input. ~12 MB, ~25 FPS. Full object detection with moderate resolution.

  **Tier 3 (Orin NX):** YOLOv8-M, INT8, 640×640 input. ~25 MB, ~45 FPS. Full resolution, all object classes, lowest confidence threshold.

  All three tiers are distilled from the same teacher model to ensure consistent detection behavior (same class taxonomy, similar confidence calibration). The OTA system tags each device with its hardware tier and delivers the appropriate model variant. Critically, the backend analytics pipeline must normalize results across tiers — Tier 1 devices will miss small objects, so coverage metrics must account for per-tier detection envelopes.

  > **Napkin Math:** Nano: 472 GFLOPS FP16 / ~2× for INT8 = ~940 GOPS. YOLOv8-N at 320×320: ~3.2 GOPS → 3.2/940 = 3.4ms compute + overhead = ~8 FPS. Xavier NX: ~21 TOPS INT8. YOLOv8-S at 480×480: ~16 GOPS → 16/21000 = 0.76ms + overhead = ~25 FPS. Orin NX: ~100 TOPS INT8. YOLOv8-M at 640×640: ~39 GOPS → 39/100000 = 0.39ms + overhead = ~45 FPS. Storage for all 3 tiers on each device: 6 + 12 + 25 = 43 MB (only the matching tier is active).

  📖 **Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Canary Deployment Gone Wrong</b> · <code>deployment</code></summary>

- **Interviewer:** "You roll out a new detection model to 1% of your edge camera fleet (100 devices) as a canary. After 24 hours, accuracy metrics look identical to the old model. You proceed to 100% rollout. Within a week, customer complaints spike — the model misses vehicles in parking garages. Your canary didn't catch this. What went wrong with your canary strategy?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The canary sample was too small." 100 devices is statistically sufficient — the problem is selection bias.

  **Realistic Solution:** Your canary devices were not representative of the fleet's deployment diversity. The 100 canary devices were likely selected randomly or from a single geographic region. If they were all outdoor intersection cameras (well-lit, standard angles), the canary would never encounter parking garage conditions (low light, tight angles, reflective surfaces, unusual vehicle orientations). The new model may have been trained on a dataset that underrepresented indoor/garage scenes, or the quantization calibration was biased toward outdoor distributions. Fixes: (1) **Stratified canary selection** — ensure the canary includes devices from every deployment category (outdoor, indoor, garage, highway, loading dock). (2) **Synthetic stress testing** — before any canary, run the model against a curated test suite that covers known hard cases (low light, rain, snow, unusual angles). (3) **Per-segment metrics** — don't just track aggregate accuracy. Track accuracy per scene type, per lighting condition, per object class. A 0.1% aggregate drop can hide a 30% drop in a critical segment.

  > **Napkin Math:** Fleet: 10,000 devices. Deployment categories: outdoor-intersection (40%), outdoor-highway (20%), indoor-garage (15%), loading-dock (10%), other (15%). Random canary of 100: expected garage devices = 15. But if canary is from one region with no garages: 0 garage devices tested. Stratified canary: 15 garage + 40 intersection + 20 highway + 10 dock + 15 other = 100 devices covering all segments.

  📖 **Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)

  </details>

</details>

---

### 📊 Monitoring & Reliability

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Silent Accuracy Drift</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your edge detection system has been deployed for 6 months. There's no ground truth labeling pipeline — you can't afford human annotators for 10,000 cameras. Customer complaints about missed detections have increased 40% over the last month. How do you detect and diagnose accuracy drift without ground truth labels?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "You can't measure accuracy without ground truth." You can't measure *absolute* accuracy, but you can detect *drift* using proxy signals.

  **Realistic Solution:** Use **distributional proxy metrics** that don't require ground truth:

  (1) **Confidence distribution shift** — track the histogram of detection confidence scores over time. If the model is degrading, the confidence distribution shifts left (more low-confidence detections). Use KL divergence or Population Stability Index (PSI) between the current week's distribution and the baseline.

  (2) **Detection count anomaly** — if a camera that typically detects 200 vehicles/hour suddenly drops to 120, something changed. Either traffic patterns shifted (verifiable from other sensors) or the model is missing detections.

  (3) **Temporal consistency** — track objects across frames. If a tracked vehicle "disappears" for 3 frames then "reappears," those are likely missed detections. The ratio of track fragmentations to total tracks is a proxy for recall.

  (4) **Cross-device comparison** — if 9 out of 10 cameras at an intersection detect a vehicle but 1 doesn't, the outlier camera likely has a model or hardware issue.

  (5) **Periodic spot-check** — label 100 random frames per camera per month (~30 minutes of annotator time per camera). Not full ground truth, but enough to estimate drift with confidence intervals.

  Root cause investigation: the 40% complaint increase correlates with a seasonal change (summer → fall). Shorter days mean more nighttime operation. If the model was calibrated on summer data, nighttime performance degrades — the same calibration bias from the quantization question, but manifesting as operational drift.

  > **Napkin Math:** PSI threshold for "investigate": >0.1. PSI threshold for "alarm": >0.25. Confidence distribution baseline: mean=0.72, std=0.15. Current: mean=0.61, std=0.18. PSI = 0.19 → "investigate" triggered. Spot-check cost: 100 frames × 30s/frame = 50 min/camera/month. For 100 sampled cameras: 83 hours/month of annotation = ~0.5 FTE.

  📖 **Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Fleet Health Dashboard</b> · <code>monitoring</code></summary>

- **Interviewer:** "You're the ML platform architect for a fleet of 50,000 edge devices across 200 cities. Design the monitoring system. What metrics do you collect, how do you aggregate them, and what are your alerting thresholds? Assume each device has intermittent cellular connectivity (uploads at most 1 MB/day)."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Stream all inference results to the cloud for analysis." At 30 FPS × 50,000 devices, that's 1.5 million frames/second — impossible over cellular.

  **Realistic Solution:** Design a **hierarchical edge-cloud monitoring architecture**:

  **On-device (real-time, no connectivity needed):**
  - Compute per-hour aggregates: detection count, mean confidence, confidence histogram (10 bins), track fragmentation rate, inference latency P50/P95/P99, GPU temperature, memory utilization, model version hash.
  - Store 24 hours of hourly aggregates locally (~50 KB/day).
  - On-device anomaly detector: if any metric deviates >3σ from the device's own 7-day rolling baseline, flag for priority upload.

  **Daily upload (≤1 MB/day per device):**
  - 24 hourly aggregate records (~50 KB).
  - 10 flagged anomaly frames with metadata (~500 KB).
  - Device health: uptime, reboot count, thermal throttle events, OTA status.
  - Remaining bandwidth (~450 KB): random sample of 50 full detection outputs for spot-check labeling.

  **Cloud aggregation:**
  - Per-city dashboards: aggregate device metrics by deployment category.
  - Fleet-wide alerts: if >5% of devices in a city show confidence drift (PSI > 0.1), trigger investigation.
  - Cohort analysis: compare metrics across model versions, hardware tiers, and deployment dates.
  - Automated retraining trigger: if fleet-wide recall proxy drops >5% from baseline, queue a retraining job with the latest spot-check labels.

  > **Napkin Math:** 50,000 devices × 1 MB/day = 50 GB/day cloud ingestion. Storage: 50 GB × 365 days × 3 years = 54.75 TB. At $0.023/GB/month (S3): $1,260/month. Cellular cost: 1 MB/day × 30 days × $0.01/MB × 50,000 = $15,000/month. Total monitoring cost: ~$16,260/month for 50,000 devices = **$0.33/device/month**.

  📖 **Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Watchdog Timer</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your edge device runs inference in a loop. Occasionally, the TensorRT engine hangs — the CUDA kernel never returns. The device appears healthy (network up, OS responsive) but produces no detections. How do you detect and recover from this failure mode?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Check if the process is running." The process *is* running — it's blocked inside a CUDA call. Standard health checks (process alive, port open) won't catch this.

  **Realistic Solution:** Implement a **hardware watchdog timer** — a dedicated hardware peripheral (present on most embedded SoCs including Jetson) that must be "kicked" (written to) at regular intervals. If the kick doesn't arrive within the timeout period, the watchdog triggers a hard reset.

  Design: (1) The inference loop writes to the watchdog after each successful inference. Timeout: 2× the worst-case inference time (e.g., 200ms if WCET is 100ms). (2) If the CUDA kernel hangs, the watchdog isn't kicked, and the device reboots after 200ms. (3) On reboot, the system checks a "crash counter" in persistent storage. If it exceeds 3 crashes in 10 minutes, the system falls back to a known-good model version (the A/B partition's backup slot). (4) A software watchdog (separate thread) provides a faster, less disruptive recovery: if no inference result arrives in 150ms, kill the inference process and restart it without rebooting the entire device.

  The two-tier approach (software watchdog for fast recovery, hardware watchdog as last resort) minimizes downtime while guaranteeing recovery from any failure mode, including kernel panics.

  > **Napkin Math:** Normal inference: 30ms. Software watchdog timeout: 150ms → detects hang in 150ms, restarts process in ~2s (TensorRT reload). Hardware watchdog timeout: 200ms → full reboot in ~30s. Without watchdog: device hangs indefinitely until manual intervention. With 10,000 devices and a 0.1% daily hang rate: 10 devices/day hang. Without watchdog: 10 devices need manual reboot (hours of downtime each). With watchdog: 10 devices auto-recover in <35s each.

  📖 **Deep Dive:** [Volume I: Robust AI](https://mlsysbook.ai/vol1/robust_ai.html)

  </details>

</details>

---

### 🔒 Security

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Adversarial Patch Attack</b> · <code>security</code></summary>

- **Interviewer:** "Your autonomous vehicle's camera-based detection system correctly identifies stop signs 99.9% of the time. A security researcher demonstrates that a carefully designed sticker (an adversarial patch) placed on a stop sign causes your model to classify it as a speed limit sign with 95% confidence. Your model is state-of-the-art. How do you defend against this?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Retrain the model with adversarial examples" or "Add input preprocessing to detect patches." Adversarial training helps but is an arms race — new patches can always be designed. Input preprocessing is easily circumvented.

  **Realistic Solution:** Defense in depth — no single layer is sufficient:

  (1) **Multi-sensor fusion** — LiDAR and radar see a physical object at the stop sign's location regardless of the visual patch. If camera says "speed limit" but LiDAR says "vertical planar object at expected stop sign height," the fusion layer flags a conflict.

  (2) **Temporal consistency** — a real stop sign doesn't change classification frame-to-frame. If the sign is "stop" for 28 frames, "speed limit" for 2 frames, then "stop" again, the temporal filter rejects the transient misclassification.

  (3) **HD map priors** — the map database says there's a stop sign at this GPS coordinate. If the model disagrees with the map, trust the map for safety-critical decisions and flag the discrepancy for review.

  (4) **Ensemble disagreement** — run two architecturally different models (e.g., CNN and ViT). Adversarial patches are typically crafted for a specific architecture. If the two models disagree, escalate to the safety system.

  (5) **Behavioral safety** — regardless of classification, if the vehicle is approaching an intersection, reduce speed. The sign classification informs behavior but doesn't override geometric safety rules.

  > **Napkin Math:** Single-model vulnerability: 1 adversarial patch defeats 1 model. With 2 independent models: attacker must defeat both simultaneously — success rate drops from ~95% to ~5% (assuming independent failure). With map prior: attacker must also spoof GPS or compromise the map database. With temporal filter (majority vote over 30 frames): attacker must sustain misclassification for >15 consecutive frames — much harder with a physical patch that only works at specific viewing angles.

  📖 **Deep Dive:** [Volume II: Robust AI](https://mlsysbook.ai/vol2/robust_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Model Theft from Edge Device</b> · <code>security</code></summary>

- **Interviewer:** "Your company spent $2M training a proprietary detection model. It's deployed on 5,000 edge devices running Jetson Orin. A competitor buys one of your devices on the secondary market. How do they extract your model, and what can you do to prevent it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Encrypt the model file on disk." Encryption at rest is necessary but not sufficient — the model must be decrypted into GPU memory to run inference, and that's where extraction happens.

  **Realistic Solution:** Attack vectors for model extraction from a physical device:

  (1) **Disk extraction** — mount the eMMC on another system and copy the model file. If unencrypted, trivial. If encrypted, the attacker needs the decryption key.

  (2) **Memory dump** — while the model is loaded in DRAM for inference, use JTAG or a cold boot attack to dump GPU memory. The weights are in plaintext in VRAM.

  (3) **API extraction** — send thousands of carefully chosen inputs through the inference API and use model distillation to train a clone. No physical access needed if the device has a network interface.

  (4) **Side-channel** — measure power consumption or electromagnetic emissions during inference to reconstruct weight values (demonstrated in academic papers on embedded ML).

  Defense layers: (a) **Secure boot chain** — ensure only signed firmware can boot. Prevents loading a modified OS that dumps memory. (b) **Hardware security module (HSM)** — store the model decryption key in the Orin's Trusted Platform Module (fTPM). The key never leaves the secure enclave. (c) **Encrypted model loading** — decrypt the model inside a Trusted Execution Environment (TEE) and load directly to GPU memory. The plaintext model never touches the filesystem. (d) **Rate limiting + anomaly detection** — detect API extraction attempts by monitoring for unusual query patterns (high volume, systematically varied inputs). (e) **Model watermarking** — embed a cryptographic watermark in the model weights that survives distillation, enabling you to prove theft in court.

  No defense is absolute against a determined attacker with physical access. The goal is to raise the cost of extraction above the cost of training their own model.

  > **Napkin Math:** Training cost: $2M. Disk extraction (unencrypted): $500 for the device + 10 minutes. Disk extraction (encrypted, no HSM): $5,000 for JTAG equipment + 1 week. API distillation: 100,000 queries × $0.01/query = $1,000 + 1 month of training = ~$50,000. With all defenses: physical extraction requires defeating secure boot + TEE + HSM ≈ $500,000+ and specialized expertise. The goal: make extraction cost > $2M.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://mlsysbook.ai/vol2/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Supply Chain Attack</b> · <code>security</code></summary>

- **Interviewer:** "Your edge device runs a detection model compiled with TensorRT, which depends on cuDNN, CUDA, and the Linux kernel. Your security team asks: 'How do we know the model running on the device is the model we trained?' What are they worried about, and how do you provide this guarantee?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We check the model file hash at deployment time." This verifies the file was delivered correctly, but doesn't verify what actually runs.

  **Realistic Solution:** The security team is worried about **supply chain integrity** — the possibility that any component in the software stack has been tampered with:

  (1) **Compromised model file** — an attacker modifies the model weights to insert a backdoor (e.g., a specific pixel pattern triggers a misclassification). The file hash at deployment time was correct, but the file was modified after deployment.

  (2) **Compromised runtime** — a malicious TensorRT or cuDNN library could modify inference results regardless of the model weights. If the attacker controls the shared library, they control the output.

  (3) **Compromised OS** — a rootkit could intercept the inference pipeline and substitute results.

  The guarantee requires **measured boot with attestation**: (a) The bootloader measures (hashes) each component as it loads: kernel → drivers → CUDA → TensorRT → model file. (b) Each hash is stored in the TPM's Platform Configuration Registers (PCRs). (c) Before inference starts, the device sends the PCR values to a remote attestation server, which compares them against known-good values. (d) If any hash doesn't match, the device is quarantined and flagged for investigation. This creates a chain of trust from hardware to model.

  > **Napkin Math:** Components in the chain: bootloader (1 hash) + kernel (1) + 5 drivers (5) + CUDA (1) + cuDNN (1) + TensorRT (1) + model (1) = 11 hashes. PCR measurement time: ~10ms total. Attestation check over cellular: ~200ms. Total boot overhead: <1 second. Storage for attestation records: 11 × 32 bytes × 50,000 devices = 17.6 MB in the attestation database.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://mlsysbook.ai/vol2/security_privacy.html)

  </details>

</details>

---

### 💰 Economics & TCO

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Edge vs Cloud Cost Crossover</b> · <code>economics</code></summary>

- **Interviewer:** "Your company processes security camera feeds. Currently, each camera streams video to the cloud for inference (AWS, $0.50/hour per GPU instance, 4 cameras per GPU). Your team proposes adding a $300 Jetson Orin NX to each camera for on-device inference, eliminating cloud costs. With 1,000 cameras, when does edge break even?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "1,000 cameras × $300 = $300,000 upfront. Cloud cost = 1,000/4 × $0.50 × 24 × 365 = $1,095,000/year. Edge pays for itself in 3.3 months." This ignores the hidden costs of edge.

  **Realistic Solution:** The naive calculation misses significant edge costs:

  **Cloud (annual):** 250 GPU instances × $0.50/hr × 8,760 hrs = $1,095,000. Plus network egress: 1,000 cameras × 5 Mbps × $0.09/GB = ~$178,000. Total cloud: **$1,273,000/year**.

  **Edge (Year 1):** Hardware: 1,000 × $300 = $300,000. Integration engineering (mount, power, network per camera): 1,000 × $150 = $150,000. OTA infrastructure (build/maintain update system): $100,000. Edge monitoring platform: $50,000. Replacement units (5% failure rate): 50 × $300 = $15,000. Power (25W × 24h × 365d × $0.12/kWh × 1,000): $26,280. Reduced but not zero cloud (model training, fleet management, analytics): $100,000. Total edge Year 1: **$741,280**.

  **Edge (Year 2+):** No hardware cost. Replacements: $15,000. Power: $26,280. OTA maintenance: $30,000. Cloud (training/analytics): $100,000. Total: **$171,280/year**.

  Breakeven: Edge saves $531,720 in Year 1. By end of Year 1, edge is already cheaper. By Year 3, cumulative savings = $531,720 + $1,101,720 + $1,101,720 = **$2,735,160**.

  But the real decision factor isn't just cost — it's latency. Cloud inference adds 50-200ms of network round-trip. For real-time security alerts, edge inference (30ms) is the only option that meets the SLA.

  > **Napkin Math:** Cloud: $1.27M/year. Edge Year 1: $741K. Edge Year 2+: $171K. Breakeven: Month 7. 3-year TCO: Cloud = $3.82M. Edge = $1.08M. Edge saves **$2.74M over 3 years** for 1,000 cameras.

  📖 **Deep Dive:** [Volume II: Sustainable AI](https://mlsysbook.ai/vol2/sustainable_ai.html)

  </details>

</details>

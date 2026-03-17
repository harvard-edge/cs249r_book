# Round 3: Operations & Deployment 🚀

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Mobile_Systems.md">📱 Round 1</a> ·
  <a href="02_Mobile_Constraints.md">⚖️ Round 2</a> ·
  <a href="03_Mobile_Ops_and_Deployment.md">🚀 Round 3</a> ·
  <a href="04_Mobile_Visual_Debugging.md">🖼️ Round 4</a> ·
  <a href="05_Mobile_Advanced.md">🔬 Round 5</a>
</div>

---

Shipping an ML model to one phone is a demo. Shipping it to a billion phones — across iOS versions, Android OEMs, chipset generations, and app store policies — is operations. This round tests whether you can reason about model delivery pipelines, app store constraints, A/B testing without ground truth, crash reporting for silent ML failures, and building ML platforms that scale across dozens of features in a single app.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/mobile/03_Mobile_Ops_and_Deployment.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 🔧 Model Optimization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Single-Op Delegation Fix</b> · <code>optimization</code></summary>

**Interviewer:** "Your TFLite model runs in 4ms when fully delegated to the Snapdragon NPU. After adding a single custom GELU activation layer, inference jumps to 38ms. The GELU itself takes <0.1ms on CPU. Why did one tiny op cause a 9.5× slowdown, and how do you fix it?"

**Common Mistake:** "The GELU op is slow on CPU." It's not — the op itself is trivial. The cost is in what it does to the execution graph.

**Realistic Solution:** NPU delegation is all-or-nothing per subgraph. When TFLite encounters an op the NPU delegate doesn't support, it **partitions the graph** at that point. The supported prefix runs on the NPU, then the intermediate tensor is copied back to CPU RAM for the unsupported op, and the remaining ops may or may not return to the NPU. One incompatible op in the middle of the graph shatters it into three segments with two expensive data transfers across the on-chip NoC.

Fixes, in order of preference: (1) **Replace with a supported approximation** — GELU(x) ≈ x × σ(1.702x). Sigmoid is NPU-supported. The approximation error is <0.01 for all practical input ranges. The entire graph stays on the NPU: 4ms. (2) **Move the unsupported op to the end** — restructure the model so GELU is the last operation. Only the tail falls back to CPU, and there's only one NPU→CPU transfer instead of two. (3) **Custom NPU kernel** — if the SoC vendor provides a custom op API (Qualcomm QNN SDK), implement GELU as a native NPU op. High effort but zero overhead. (4) **Use a different runtime** — ONNX Runtime's QNN execution provider may support GELU natively where TFLite's delegate doesn't.

> **Napkin Math:** Fully delegated: 4ms (all on NPU). With GELU partition: 2ms (NPU prefix) + 1.5ms (NPU→CPU transfer) + 0.1ms (GELU on CPU) + 1.5ms (CPU→NPU transfer) + 2ms (NPU suffix) + overhead = 7.1ms minimum. But the partition also breaks NPU graph optimizations (layer fusion, buffer reuse), inflating the NPU segments from 2+2ms to 15+15ms. Total: ~38ms. With sigmoid approximation: 4ms (no partition). **9.5× speedup from changing one activation function.**

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 🚀 Deployment

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The App Size Limit</b> · <code>deployment</code></summary>

**Interviewer:** "Your model is 500 MB (FP32 weights). The App Store allows up to 200 MB for cellular downloads — anything larger requires WiFi. Your PM wants the app to work immediately after download, even on cellular. What are your options?"

**Common Mistake:** "Compress the model file with zip." Model weights don't compress well — they're essentially random floating-point numbers. Zip might save 5-10%.

**Realistic Solution:** Four strategies, from simplest to most complex:

(1) **Quantize to INT8** — 500 MB FP32 → 125 MB INT8. Fits under 200 MB. Accuracy loss: typically <1% for classification, <2% for detection. This is the first thing to try.

(2) **Ship a small model in the bundle, download the full model on WiFi** — include a MobileNet-V3-Small (~8 MB) in the app bundle for immediate functionality. On first WiFi connection, download the full 500 MB model in the background. The user gets instant (lower quality) results that upgrade transparently.

(3) **On-demand model download** — use Apple's On-Demand Resources (ODR) or Android's Play Asset Delivery (PAD) to stream model chunks after install. The model is not in the initial download. First inference triggers a download of the required model shard.

(4) **Knowledge distillation** — train a smaller student model (~50 MB) that mimics the 500 MB teacher. Ship the student in the bundle. This requires ML engineering effort but produces a permanently smaller model.

> **Napkin Math:** FP32: 500 MB (over limit). INT8: 125 MB ✓ (under 200 MB). INT4: 62.5 MB ✓. Distilled student: ~50 MB ✓. Bundle + background download: 8 MB initial + 500 MB on WiFi. Cellular download at 10 Mbps: 200 MB = 160 seconds. 500 MB = 400 seconds (requires WiFi). User experience: INT8 is the best trade-off — immediate, no WiFi dependency, minimal accuracy loss.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The App Store Model Size Rejection</b> · <code>deployment</code></summary>

**Interviewer:** "Your image generation app bundles a 350 MB diffusion model. Apple approves the app, but analytics show 60% of users never complete the download — they abandon when iOS shows 'This app is over 200 MB and requires WiFi.' Your PM says 'just make the model smaller.' What's the real solution?"

**Common Mistake:** "Quantize the model to fit under 200 MB." Aggressive quantization of a diffusion model destroys output quality — users will see blocky, artifacted images and uninstall.

**Realistic Solution:** Separate the model from the app binary entirely using a **staged delivery architecture**:

(1) **Thin app bundle** — ship the app at ~40 MB with UI, a tiny preview model (~5 MB, generates low-res 128×128 previews), and a download manager. This installs over cellular instantly.

(2) **Background model download** — on first launch, show a "Preparing your AI engine" screen while downloading the 350 MB model via Apple's Background Assets framework (iOS 16+) or Android's Play Asset Delivery. The download resumes across app kills and network interruptions. Store the model in the app's `Library/Caches` directory (iOS) or internal storage (Android).

(3) **Progressive quality** — while the full model downloads, let users generate images with the preview model. Low-res results set expectations and demonstrate value, reducing abandonment. When the full model arrives, seamlessly upgrade.

(4) **Model compression for the wire** — apply weight clustering (256 clusters per tensor) + Huffman coding to the model file. This doesn't change inference precision but compresses the download from 350 MB to ~180 MB. Decompress on-device after download.

> **Napkin Math:** App Store cellular limit: 200 MB. Thin bundle: 40 MB ✓ (cellular OK). Model download: 350 MB raw, 180 MB compressed. On WiFi (50 Mbps): 180 MB / 6.25 MB/s = 29 seconds. On LTE (20 Mbps): 180 MB / 2.5 MB/s = 72 seconds. User abandonment: 60% at 350 MB forced-WiFi → estimated 15% at 40 MB cellular + background download. Retention improvement: 4× more users complete setup.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The On-Device Model Hot-Swap</b> · <code>deployment</code></summary>

**Interviewer:** "Your app's ML model needs to be updated without an App Store update. The current model has a critical misclassification bug — it labels some food items as non-food, breaking your calorie tracking feature. An App Store update takes 24-48 hours for review. How do you push a model fix in under 1 hour?"

**Common Mistake:** "Ship the model as a server-side config and swap it remotely." This violates Apple's App Store Review Guidelines §2.5.2 — apps may not download executable code. Model weights are not executable code, but the approach needs careful implementation.

**Realistic Solution:** Design a **dynamic model delivery system** that stays within platform guidelines:

(1) **Model registry service** — host versioned models on your CDN (CloudFront, Firebase Hosting). Each model has a manifest: version, hash (SHA-256), minimum app version, target OS versions, file size, and a rollout percentage.

(2) **On-device model manager** — at app launch (and periodically in background), the app checks the manifest. If a new model is available and the device matches the targeting criteria, download it. Verify the SHA-256 hash before accepting. Store in the app's sandboxed storage.

(3) **Atomic swap** — the app keeps the current model loaded in memory. The new model is loaded into a second buffer, validated with a test inference on a known input/output pair. Only if validation passes does the app atomically swap the model pointer. The old model file is kept as a rollback target for 7 days.

(4) **Platform compliance** — Apple allows downloading "data" (including model weights) post-install. The model format (Core ML `.mlmodelc`, TFLite `.tflite`) is interpreted by Apple/Google's own frameworks, not custom executable code. Firebase ML and Apple's Core ML Model Deployment (CloudKit) are first-party solutions that handle this pattern.

(5) **Staged rollout** — push to 1% of users, monitor crash rates and engagement metrics for 1 hour, then expand to 10% → 50% → 100%. Total time from model fix to full rollout: ~4 hours.

> **Napkin Math:** Model size: 25 MB (INT8 classification model). CDN download at 20 Mbps LTE: 25 MB / 2.5 MB/s = 10 seconds. Validation inference: 50ms. Swap: <1ms (pointer swap). Total update time per device: ~11 seconds. Staged rollout: 1% (1 hour) → 10% (1 hour) → 50% (1 hour) → 100% (1 hour) = 4 hours to full fleet. vs App Store: 24-48 hours review + user must manually update.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Cross-Version Compatibility Maze</b> · <code>deployment</code></summary>

**Interviewer:** "Your app supports iOS 16-18 and Android 10-15. You ship a Core ML 7 model (iOS 18 feature: stateful KV-cache for your on-device LLM). Users on iOS 16 crash on launch. Your Android TFLite model uses INT4 quantization, but devices running Android 10 with TFLite 2.11 don't support INT4. How do you ship one app that works across all these OS versions?"

**Common Mistake:** "Target the lowest common denominator — use only features available on iOS 16 and Android 10." This means no stateful models, no INT4, no recent optimizations — your app is 3× slower than competitors who target newer OS versions.

**Realistic Solution:** Implement a **model tiering system** keyed to runtime capabilities, not OS version:

(1) **Capability detection at launch** — query the actual runtime version and hardware capabilities. On iOS: check `MLModel.availableComputeDevices` and Core ML version via `MLModelConfiguration`. On Android: check TFLite runtime version, GPU delegate availability, and NNAPI level.

(2) **Model tiers:**
- **Tier 1 (Legacy):** Core ML 5 model (FP16, no stateful ops) for iOS 16. TFLite INT8 model for Android 10-12. Widest compatibility, slowest performance.
- **Tier 2 (Modern):** Core ML 6 model (INT8 + palettized weights) for iOS 17. TFLite INT8 with GPU delegate for Android 13-14. Good balance.
- **Tier 3 (Cutting-edge):** Core ML 7 model (INT4 + stateful KV-cache) for iOS 18. TFLite INT4 with NNAPI 1.3 for Android 15+. Maximum performance.

(3) **Bundle strategy** — don't ship all tiers in the app bundle (3× the size). Ship Tier 1 in the bundle (guaranteed to work everywhere). Download the appropriate higher tier on first launch using the dynamic model delivery system. If download fails, Tier 1 works indefinitely.

(4) **Feature flags** — the LLM feature (requiring stateful KV-cache) is only enabled on Tier 3 devices. Tier 1-2 users see a "Cloud AI" fallback or the feature is hidden entirely.

> **Napkin Math:** Tier 1 model: 25 MB (INT8, universal). Tier 2: 15 MB (palettized, 30% smaller). Tier 3: 8 MB (INT4, 70% smaller). App bundle with Tier 1: 25 MB. Post-install download for Tier 2/3: 8-15 MB. User base split (typical 2026): iOS 18 (60%), iOS 17 (25%), iOS 16 (15%). Android 14-15 (45%), Android 12-13 (35%), Android 10-11 (20%). Tier 3 eligible: ~52% of users. Tier 2: ~30%. Tier 1: ~18%.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

### 📊 Monitoring & Reliability

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Silent Accuracy Degradation</b> · <code>monitoring</code></summary>

**Interviewer:** "Your on-device image classification model was deployed 6 months ago. Users haven't complained, but your A/B test shows the new model version improves engagement by 15%. This suggests the old model has silently degraded. You have no server-side ground truth labels for on-device predictions. How do you detect model degradation without ground truth?"

**Common Mistake:** "Monitor accuracy using a held-out test set." You don't have labels for on-device predictions — there's no test set.

**Realistic Solution:** Four proxy signals that detect degradation without ground truth:

(1) **Confidence distribution shift** — track the distribution of the model's softmax confidence scores over time. A healthy model has a bimodal distribution (high confidence for easy inputs, low for hard). If the distribution shifts toward uniform confidence (the model becomes "confused"), it's seeing out-of-distribution data. Use KL divergence between the current week's confidence distribution and the baseline.

(2) **User implicit feedback** — track behavioral proxies: if the model powers a photo search feature, monitor search refinement rate (user searches again immediately = bad result), feature abandonment rate, and time-to-action after a prediction. A 15% engagement improvement from the new model implies the old model's predictions were increasingly irrelevant.

(3) **Lightweight anomaly detector** — deploy a small autoencoder (~1 MB) alongside the main model. Train it on the same distribution as the main model. If reconstruction error exceeds a threshold, the input is OOD. Track the OOD rate over time — a rising rate indicates distribution drift.

(4) **Federated evaluation** — periodically sample a small subset of users, send them a labeled evaluation batch (e.g., 100 images with known labels), and compare the model's predictions. This gives you direct accuracy measurement without collecting user data. Privacy-preserving: the evaluation data comes from you, not from users.

> **Napkin Math:** Confidence monitoring: ~0.1 KB per inference (just the max softmax score). 1000 inferences/day × 30 days = 30,000 data points. KL divergence computation: trivial. Storage: 30 KB/month. Anomaly detector: 1 MB model, 0.5ms per inference. Federated evaluation: 100 labeled images × 4 evaluations/year = 400 labeled predictions per user per year. With 10,000 users: 4M labeled predictions — statistically powerful.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The A/B Test Without Ground Truth</b> · <code>monitoring</code></summary>

**Interviewer:** "You have two versions of your on-device recommendation model. In cloud ML, you'd A/B test by comparing click-through rates against a server-side holdout. But your model runs entirely on-device — predictions never hit your server. How do you A/B test two on-device models when you can't observe the predictions?"

**Common Mistake:** "Send all predictions to the server for analysis." This defeats the purpose of on-device inference (privacy, latency) and may violate your privacy policy.

**Realistic Solution:** Design an **on-device A/B testing framework** that measures outcomes, not predictions:

(1) **Assignment** — at app install (or experiment start), assign users to cohorts using a deterministic hash of their anonymous device ID + experiment ID. No server round-trip needed. 50/50 split: `hash(device_id + "exp_042") % 2`.

(2) **Model deployment** — both model variants are included in the app bundle (or downloaded via dynamic delivery). The on-device experiment manager loads the correct model for the user's cohort.

(3) **Outcome metrics (not prediction metrics)** — you can't observe predictions, but you can observe *consequences*: session duration, feature usage frequency, conversion events (purchase, share, save), retention (day 1, day 7, day 30), and crash rate. These are standard analytics events that don't expose model predictions.

(4) **Guardrail metrics** — monitor latency (P50, P95 inference time), memory usage, battery drain, and crash rate per cohort. A model that improves engagement but drains 2× more battery is not a winner.

(5) **Statistical rigor** — with on-device experiments, you need larger sample sizes because outcome metrics are noisier than prediction metrics. A 2% CTR improvement requires ~50,000 users per cohort to detect at 95% confidence (power analysis). Plan for 2-4 week experiment duration.

> **Napkin Math:** Experiment: 100,000 users, 50/50 split. Metric: 7-day retention. Baseline: 40%. Minimum detectable effect: 2% (absolute). Required sample: n = 16 × p(1-p) / δ² = 16 × 0.4 × 0.6 / 0.02² = 9,600 per cohort. With 50,000 per cohort: can detect 0.9% absolute difference. Duration: 7 days (retention window) + 7 days (data collection) = 14 days minimum. Analytics payload per user: ~5 KB/day (outcome events only, no predictions).

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The ML Crash vs Silent Failure</b> · <code>monitoring</code></summary>

**Interviewer:** "Your app uses an on-device model for real-time object detection in AR. Users report 'the app doesn't detect anything sometimes.' Your crash reporting dashboard (Crashlytics) shows zero crashes. How do you detect and diagnose ML failures that don't crash the app?"

**Common Mistake:** "If there are no crashes, the model is working fine." ML models fail silently — they return valid tensors filled with garbage instead of throwing exceptions.

**Realistic Solution:** ML failures are invisible to standard crash reporting because the model always returns *something* — a valid tensor of zeros, random confidences, or stale results. You need **ML-specific health monitoring**:

(1) **Output validation layer** — wrap every inference call with sanity checks: Are all confidence scores zero? (model returned empty detections — likely input preprocessing failure). Are confidence scores all >0.99? (model is saturated — likely quantization error). Is the output identical to the previous 10 frames? (model is frozen — likely a threading deadlock or stale buffer). Are bounding boxes outside the image bounds? (coordinate space mismatch after a camera resolution change).

(2) **Inference health heartbeat** — log a lightweight event every 60 seconds: model version, inference count since last heartbeat, mean confidence, mean latency, and a boolean "healthy" flag from the validation layer. This costs ~200 bytes per minute. Ship these events with your standard analytics.

(3) **Structured ML error taxonomy** — define error codes: `ML_EMPTY_OUTPUT`, `ML_STALE_OUTPUT`, `ML_LATENCY_SPIKE` (>3× P95), `ML_DELEGATE_FALLBACK` (NPU failed, fell back to CPU), `ML_MODEL_LOAD_FAILURE`. Report these as non-fatal events in Crashlytics/Sentry.

(4) **Reproduction pipeline** — when a silent failure is detected, capture the input tensor (or a hash of it) and the model output. Upload these on WiFi for offline debugging. This lets you reproduce the exact failure in your test environment.

> **Napkin Math:** Silent failure rate (typical): 0.1-1% of inferences. At 30 FPS for 10 minutes: 18,000 inferences. Silent failures: 18-180 per session. None appear in crash logs. Health heartbeat: 10 events per 10-minute session × 200 bytes = 2 KB. Analytics cost: negligible. Detection latency: 60 seconds (heartbeat interval). With output validation: detected within 1 frame (33ms).

**📖 Deep Dive:** [Volume I: Robust AI](https://mlsysbook.ai/vol1/robust_ai.html)
</details>

---

### 🔒 Security & Privacy

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Federated Keyboard</b> · <code>privacy</code></summary>

**Interviewer:** "You're building a next-word prediction model for a mobile keyboard. The model must improve from user typing patterns, but you cannot collect user keystrokes on your servers — that's a privacy violation and a regulatory risk. How do you train the model?"

**Common Mistake:** "Anonymize the data before uploading." Anonymization is insufficient — keystroke patterns can re-identify users, and regulators (GDPR, CCPA) consider this personal data regardless of anonymization.

**Realistic Solution:** **Federated Learning with Differential Privacy (DP-FedAvg):**

(1) **Local training:** Each phone fine-tunes a local copy of the model on the user's recent typing data. Training happens on-device during charging (to avoid battery drain). The phone computes a gradient update (the difference between the local model and the global model).

(2) **Gradient clipping:** Before sending anything, clip the gradient to a maximum L2 norm $C$. This bounds the influence of any single user's data on the global model.

(3) **Noise injection:** Add calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ to the clipped gradient. This provides differential privacy — mathematically guaranteeing that the server cannot determine whether any specific user participated in training.

(4) **Secure aggregation:** The server collects noised gradients from thousands of phones and averages them. Individual gradients are encrypted so the server only sees the aggregate. The noise cancels out in the average (law of large numbers), but protects individual contributions.

(5) **Privacy budget:** The privacy guarantee is measured by $\epsilon$ (epsilon). Lower $\epsilon$ = stronger privacy but more noise = slower learning. Typical production values: $\epsilon = 8$ per training round, with a total budget of $\epsilon = 100$ per year. At these levels, accuracy loss vs non-private training is ~2%.

> **Napkin Math:** 10,000 phones per round. Each sends a 5 MB gradient update (clipped + noised). Server bandwidth: 50 GB per round. Noise per phone: σ = 1.0, C = 1.0. After averaging 10,000 updates: effective noise = σ/√10,000 = 0.01 — negligible. Privacy: ε = 2 per round (strong). 50 rounds/year: total ε = 100 (within budget). Accuracy: ~2% worse than centralized training, but zero user data leaves the device.

> **Key Equation:** $\tilde{g}_{\text{user}} = \text{clip}(g, C) + \mathcal{N}(0, \sigma^2 C^2 I)$

**📖 Deep Dive:** [Volume II: Security & Privacy](https://mlsysbook.ai/vol2/security_privacy.html)
</details>

---

### 🏗️ Platform & Scale

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The 50-Feature Mobile ML Platform</b> · <code>platform</code></summary>

**Interviewer:** "You're the ML platform architect for a super-app (like WeChat or Grab) that has 50 different ML features: face filters, speech recognition, recommendation, fraud detection, OCR, translation, smart replies, and more. Each feature team wants to ship their own model. The app is already 300 MB. Users on budget phones with 3 GB RAM complain about performance. Design the mobile ML platform."

**Common Mistake:** "Let each team bundle their own model and runtime." 50 models × 20 MB average = 1 GB of models. 50 separate TFLite/CoreML runtimes competing for memory. The app becomes unusable.

**Realistic Solution:** Design a **shared ML infrastructure layer** that all 50 features consume:

**(1) Shared backbone architecture** — many features need similar low-level features (edges, textures, object boundaries). Train a single multi-task backbone (e.g., EfficientNet-B0, 20 MB) that produces a shared feature tensor. Each feature team trains only a lightweight task head (0.5-2 MB each). Total: 20 MB backbone + 50 × 1 MB heads = 70 MB, not 1 GB.

**(2) Model loading scheduler** — not all 50 models need to be in memory simultaneously. The face filter model loads when the camera opens. The OCR model loads when the scanner opens. The platform maintains a priority queue: active feature models are resident, recently-used models are cached, and cold models are evicted. Maximum concurrent models: 3-5 (based on available RAM).

**(3) Unified runtime** — one TFLite/CoreML instance serves all models. Shared thread pool, shared memory allocator, shared NPU delegate. This eliminates per-model runtime overhead (~15 MB per runtime × 50 = 750 MB saved).

**(4) Model delivery service** — models are not in the app bundle. They're downloaded on-demand from a CDN, keyed by feature + device capability + OS version. A budget phone with 3 GB RAM gets INT4 models; a flagship gets INT8 with larger heads.

**(5) Resource governor** — a central controller monitors total ML memory usage, inference latency, thermal state, and battery level. If the phone is overheating, it reduces model quality (switch to smaller heads) or increases inference intervals. If RAM is low, it evicts cached models more aggressively.

**(6) Observability** — each inference call is tagged with feature ID, model version, latency, and device state. A central dashboard shows per-feature ML health across the entire user base.

> **Napkin Math:** Naive approach: 50 models × 20 MB = 1 GB models + 50 × 15 MB runtime = 1.75 GB total. Platform approach: 20 MB backbone + 50 × 1 MB heads (70 MB on disk, ~25 MB resident) + 15 MB shared runtime = 40 MB resident. Memory reduction: **44×**. App size: 300 MB app + 70 MB models (downloaded) vs 300 MB + 1 GB (bundled). Budget phone (3 GB RAM, ~1.5 GB available): naive approach OOMs. Platform approach: 40 MB ML + 200 MB app = 240 MB. Fits with 1.26 GB headroom.

**📖 Deep Dive:** [Volume II: Edge Intelligence](https://mlsysbook.ai/vol2/edge_intelligence.html)
</details>

# Ship & Update

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> · <a href="../edge/README.md">🤖 Edge</a> · <b>📱 Mobile</b> · <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

*How you ship models to a billion phones and keep them current*

App store constraints, model delivery, A/B testing, monitoring, privacy, and on-device training — the lifecycle of ML on mobile devices.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/mobile/03_ship_and_update.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---


### Deployment & Model Delivery


#### 🟢 L1/L2


<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The App Store Diet</b> · <code>on-demand-model-download</code></summary>

- **Interviewer:** "You are a Mobile ML Engineer building a new "Magic Stylus" feature that uses an on-device model with 70 million parameters. Your base app is already 150 MB. To avoid exceeding the app store's ~200 MB cellular download limit and forcing users onto Wi-Fi, you plan to use on-demand resource APIs to download the model after the initial install.

For your technical design document, you need to state the storage footprint of the model. Calculate the size of the 70M parameter model if stored in FP16 precision."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is confusing the byte requirements for different precisions. Engineers often default to thinking '1 parameter = 1 byte', which is only true for INT8 quantization. This leads to an answer that is half the correct size. A less common error is using 4 bytes per parameter, the standard for FP32, which would be correct for training weights but is overkill for most mobile inference scenarios and results in a 2x overestimate.

  **Realistic Solution:** The correct answer is 140 MB. A 70 million parameter model requires 140 MB of storage in FP16 precision. This is because each FP16 parameter requires 2 bytes of storage. Adding this to the 150 MB base app size would create a 290 MB binary, which is significantly over the cellular download limit, confirming that an on-demand download strategy is necessary.

  > **Napkin Math:** 1. **Identify parameters:** 70,000,000
2. **Identify bytes per parameter:** FP16 precision uses 16 bits, which is 2 bytes.
3. **Calculate total size:** 70,000,000 parameters × 2 bytes/parameter = 140,000,000 bytes
4. **Convert to megabytes:** 140,000,000 bytes ≈ 140 MB

  > **Key Equation:** $\text{Model Size (Bytes)} = \text{Number of Parameters} \times \text{Bytes per Parameter}$

  > **Options:**
  > [ ] 70 MB
  > [ ] 280 MB
  > [x] 140 MB
  > [ ] 17.5 MB

  📖 **Deep Dive:** [Mobile: Shipping and Updating](https://mlsysbook.ai/mobile/03_ship_and_update.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The OTA Cellular Limit</b> · <code>ota-updates</code></summary>

- **Interviewer:** "You are an engineer on an automotive team, and you need to push a critical update to a driver-assistance model. To avoid waiting for a full app store review, you plan to use an Over-the-Air (OTA) update. To ensure the update reaches the maximum number of vehicles, including those not on Wi-Fi, you must stay within the cellular download limit imposed by the mobile OS. What is the typical size limit you must stay under for a cellular download that doesn't require explicit user opt-in?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers often confuse the large multi-gigabyte *initial install* size with the much stricter *cellular update* size. They might assume that since the app is 2 GB, a 500 MB update is fine, failing to realize OSes strictly cap non-Wi-Fi downloads to protect users' data plans. This leads to failed rollouts for users who aren't on Wi-Fi.

  **Realistic Solution:** Most mobile operating systems, like iOS and Android, impose a limit of around 200-250 MB for automatic downloads over a cellular network. To be safe, an engineering team should target a budget of ~200 MB for any OTA update that needs to be deployed urgently and widely.

  > **Napkin Math:** If your new uncompressed driver-assistance model is 900 MB, you need to achieve a significant compression ratio to meet the OTA limit.

`Required Ratio = Uncompressed Size / Target Size`
`Required Ratio = 900 MB / 200 MB = 4.5x`

You would need to apply quantization (e.g., FP16 to INT4 is a 4x reduction), pruning, or other techniques to meet this budget.

  > **Key Equation:** $\text{Update Size} \le \text{Cellular Limit}$

  > **Options:**
  > [ ] 20 MB
  > [ ] 2 GB
  > [x] ~200 MB
  > [ ] Unlimited, as long as the user has a data plan.

  📖 **Deep Dive:** [Mobile: Ship and Update](https://mlsysbook.ai/mobile/03_ship_and_update.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The OTA Update Budget</b> · <code>ota-update-cost</code></summary>

- **Interviewer:** "You are an ML systems engineer for an automotive company deploying a new driver-assistance feature. The model has 250 million parameters and is stored in FP16 precision. To update the fleet, the model must be sent over-the-air (OTA) via cellular networks. Calculate the size of the model update package that needs to be transmitted."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** The most common mistake is to confuse the storage requirements for different numerical precisions. Engineers often default to thinking in terms of INT8 (1 byte/param) or FP32 (4 bytes/param), leading to a 2x error in either direction. Another mistake is to confuse bits and bytes, accidentally multiplying by 16 (for FP16) instead of 2, resulting in an 8x overestimation.

  **Realistic Solution:** Each parameter in an FP16 (half-precision float) model requires 2 bytes of storage. To calculate the total size, you multiply the number of parameters by the size of each parameter.

250,000,000 parameters × 2 bytes/parameter = 500,000,000 bytes.

Converting bytes to megabytes (dividing by 1,000,000), you get 500 MB. This is a significant download size for a cellular network, impacting data costs and update reliability across a large fleet of vehicles.

  > **Napkin Math:** $\text{Model Size} = \text{Number of Parameters} \times \text{Bytes per Parameter}$

$\text{Model Size} = 250,000,000 \times 2 \text{ bytes} = 500,000,000 \text{ bytes}$

$500,000,000 \text{ bytes} / 10^6 \text{ bytes/MB} = 500 \text{ MB}$

  > **Key Equation:** $\text{Model Size (Bytes)} = \text{Parameters} \times \text{Bytes per Parameter}$

  > **Options:**
  > [ ] 250 MB
  > [ ] 1 GB
  > [x] 500 MB
  > [ ] 4 GB

  📖 **Deep Dive:** [Mobile: Ship and Update](https://mlsysbook.ai/mobile/03_ship_and_update.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L1_Foundation-brightgreen?style=flat-square" alt="Level 1" align="center"> The OTA Update Budget</b> · <code>ota-update-size</code></summary>

- **Interviewer:** "Your team is updating a keyword spotting model that runs continuously in a mobile app. The update will be pushed Over-the-Air (OTA) to millions of users, many on cellular data. The model is quantized to INT8. To avoid user complaints and high CDN costs, you must estimate the update size. State the most realistic size for this OTA model update package."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** Engineers accustomed to cloud models (GBs) or even larger mobile vision models (10s of MB) often misjudge the scale required for ubiquitous, low-friction mobile features. They forget that for OTA updates on cellular, especially for a background service, the size budget is extremely strict. A multi-megabyte update would be considered prohibitively expensive and intrusive by the product team.

  **Realistic Solution:** A quantized keyword spotting model is designed for extreme efficiency and must be very small. A realistic size is in the hundreds of kilobytes. This ensures the OTA update is fast, cheap, and goes unnoticed by the user, which is critical for a background feature on a cellular-connected device.

  > **Napkin Math:** From the `NUMBERS.md` guide, the *entire Flash budget* for a TinyML device (where keyword spotting is a canonical task) after accounting for the OS and bootloader is only ~454 KB. While a mobile phone has more memory, the model architecture for this task remains fundamentally small to conserve power and CPU. Therefore, a size in the hundreds of kilobytes is the only plausible answer.

  > **Options:**
  > [ ] ~50 MB
  > [ ] ~5 GB
  > [x] ~500 KB
  > [ ] ~50 KB

  📖 **Deep Dive:** [Ship and Update](https://mlsysbook.ai/mobile/03_ship_and_update.html)
  </details>
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L2_Analytical-blue?style=flat-square" alt="Level 2" align="center"> The OTA Data Budget</b> · <code>ota-update-cost</code></summary>

- **Interviewer:** "You're a Staff ML Engineer on the mobile team for a popular social media app. Your team is launching a new generative AI feature for creating image filters. The v1 model is 80 MB. After a month of user feedback and fine-tuning, the data science team has produced a v2 model that is 10% more accurate but now weighs 120 MB. Your product manager is concerned about the data cost for users when rolling out this over-the-air (OTA) update. Explain the data consumption for a single user if you ship the full update."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** A common mistake is to only consider the difference in size (120 MB - 80 MB = 40 MB), assuming a differential 'diff' update mechanism is in place. While possible, diffing is complex to engineer. The default, and most common, OTA update strategy for mobile ML models is to replace the entire file. Another mistake is to ignore the user cost entirely in favor of the accuracy gain, which is not a user-centric or systems-aware tradeoff.

  **Realistic Solution:** The update will download the entire 120 MB model file to the user's device, consuming 120 MB of their cellular or Wi-Fi data plan. This is a significant data cost, especially for users on limited or prepaid mobile plans in emerging markets. It could lead to unexpected data charges or cause users to disable background updates, harming future feature adoption. A Staff engineer must flag this cost and propose mitigations.

  > **Napkin Math:** The calculation is a direct interpretation of the model's storage size as network transfer size.

1. **Identify model size:** The v2 model is 120 MB (MegaBytes).
2. **Assume full update:** For a standard OTA rollout where the entire asset is replaced, the data downloaded is equal to the full size of the new asset.
3. **Calculate data cost:** `Data Transferred = v2 Model Size = 120 MB`.

This is not a trivial amount of a monthly data allocation for many users.

  > **Key Equation:** $\text{Data}_{\text{OTA}} = \text{Size}_{\text{Model}}$

  > **Options:**
  > [ ] The update will only transfer the 40 MB difference, which is a manageable size.
  > [ ] The 10% accuracy gain is worth the user data cost; we should ship the 120 MB update immediately.
  > [x] The update will consume 120 MB of the user's data plan, a significant cost we must address.
  > [ ] The update is about 960 Megabits (Mb), but this is standard and shouldn't be a concern.

  📖 **Deep Dive:** [Mobile: Ship and Update](https://mlsysbook.ai/mobile/03_ship_and_update.html)
  </details>
</details>





#### 🟢 L3
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Conversion Cliff</b> · <code>deployment</code></summary>

- **Interviewer:** "You trained a PyTorch model with a custom GELU activation and grouped query attention. You convert it to CoreML for iPhone deployment. The conversion succeeds with no errors, but on-device accuracy is significantly worse than your PyTorch baseline. What went wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Quantization during conversion caused the accuracy drop." But you haven't quantized — the model is FP16 in both cases.

  **Realistic Solution:** Silent operator approximation. CoreML (and TFLite) converters handle unsupported ops by substituting approximate implementations. GELU might be replaced with a sigmoid-based approximation (`x * sigmoid(1.702 * x)`) instead of the exact erf-based version your model was trained with. Grouped query attention may be decomposed into a sequence of reshapes and standard attention ops that introduces numerical drift. The conversion "succeeds" because the graph is structurally valid, but the mathematical behavior has shifted. The fix: always run a numerical comparison (max absolute error, cosine similarity) between the original and converted model on a reference dataset *before* deployment.

  > **Napkin Math:** GELU exact: $x \cdot \Phi(x)$ where $\Phi$ is the Gaussian CDF. GELU approximate: $x \cdot \sigma(1.702x)$. Max absolute difference between the two: ~0.004 at $x \approx -1.5$. Across millions of activations over dozens of layers, these errors compound. A model with 24 transformer layers can accumulate enough drift to shift top-1 accuracy by 1–3%.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> The App Store Privacy Rejection</b> · <code>deployment</code> <code>privacy</code></summary>

- **Interviewer:** "You submit your iOS app to the App Store. It uses an on-device CoreML model for face-based age estimation to apply age-appropriate content filters. Apple rejects it with: 'Guideline 5.1.2 — Data Use and Sharing: Your app collects face data without adequate purpose string.' The model runs entirely on-device and you never send face data to a server. Why was it rejected, and what do you need to change?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We don't collect face data — the model runs on-device, so there's no privacy issue." Apple's privacy framework doesn't distinguish between on-device processing and server-side collection for certain sensitive data categories. Face geometry is treated as biometric data regardless of where it's processed.

  **Realistic Solution:** Apple's App Store Review Guidelines treat face data as a special category:

  (1) **`NSFaceIDUsageDescription` is not enough** — this key is for Face ID authentication. Your app uses the TrueDepth camera or Vision framework to detect face geometry, which requires `NSCameraUsageDescription` with a purpose string that explicitly mentions face analysis. Generic strings like "Camera access needed" are rejected — you must say "This app analyzes your face to estimate age for content filtering."

  (2) **Privacy Nutrition Label** — your App Store Connect listing must declare "Face or Head Data" under "Data Linked to You" or "Data Not Linked to You." Even though the data never leaves the device, Apple requires disclosure of what data types are *processed*, not just *transmitted*. Missing this declaration triggers automatic rejection.

  (3) **Kids category conflict** — if your app targets children (content filtering implies this), Apple's Guideline 1.3 (Kids Category) prohibits collecting biometric data from children under 13, even on-device. You must either: (a) remove the age estimation feature for users under 13, or (b) use a non-biometric signal (e.g., parental gate, date-of-birth entry) for the initial age check, and only use the ML model for adult users who consent.

  (4) **EU Digital Services Act** — if distributed in the EU, on-device age estimation for content filtering may require a Data Protection Impact Assessment (DPIA) under GDPR Article 35, even without data transmission.

  **Fix:** Add the correct `NSCameraUsageDescription` with explicit face analysis mention. Update the Privacy Nutrition Label. Add a consent dialog before first face scan. For kids: use a parental gate instead of face-based age estimation.

  > **Napkin Math:** App Store review cycle: ~24-48 hours per submission. Rejection → fix → resubmit: 3-5 days lost. If you're on a launch deadline, this is catastrophic. Prevention: Apple's App Review team publishes a pre-submission checklist. The privacy nutrition label takes 30 minutes to fill out correctly. The `NSCameraUsageDescription` string takes 5 minutes. Total prevention cost: **35 minutes**. Cost of getting it wrong: **3-5 days** + potential PR damage if users see "rejected by Apple" in tech press.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The App Size Limit</b> · <code>deployment</code></summary>

- **Interviewer:** "Your model is 500 MB (FP32 weights). The App Store allows up to 200 MB for cellular downloads — anything larger requires WiFi. Your PM wants the app to work immediately after download, even on cellular. What are your options?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Compress the model file with zip." Model weights don't compress well — they're essentially random floating-point numbers. Zip might save 5-10%.

  **Realistic Solution:** Four strategies, from simplest to most complex:

  (1) **Quantize to INT8** — 500 MB FP32 → 125 MB INT8. Fits under 200 MB. Accuracy loss: typically <1% for classification, <2% for detection. This is the first thing to try.

  (2) **Ship a small model in the bundle, download the full model on WiFi** — include a MobileNet-V3-Small (~8 MB) in the app bundle for immediate functionality. On first WiFi connection, download the full 500 MB model in the background. The user gets instant (lower quality) results that upgrade transparently.

  (3) **On-demand model download** — use Apple's On-Demand Resources (ODR) or Android's Play Asset Delivery (PAD) to stream model chunks after install. The model is not in the initial download. First inference triggers a download of the required model shard.

  (4) **Knowledge distillation** — train a smaller student model (~50 MB) that mimics the 500 MB teacher. Ship the student in the bundle. This requires ML engineering effort but produces a permanently smaller model.

  > **Napkin Math:** FP32: 500 MB (over limit). INT8: 125 MB ✓ (under 200 MB). INT4: 62.5 MB ✓. Distilled student: ~50 MB ✓. Bundle + background download: 8 MB initial + 500 MB on WiFi. Cellular download at 10 Mbps: 200 MB = 160 seconds. 500 MB = 400 seconds (requires WiFi). User experience: INT8 is the best trade-off — immediate, no WiFi dependency, minimal accuracy loss.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The App Store Model Size Rejection</b> · <code>deployment</code></summary>

- **Interviewer:** "Your image generation app bundles a 350 MB diffusion model. Apple approves the app, but analytics show 60% of users never complete the download — they abandon when iOS shows 'This app is over 200 MB and requires WiFi.' Your PM says 'just make the model smaller.' What's the real solution?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Quantize the model to fit under 200 MB." Aggressive quantization of a diffusion model destroys output quality — users will see blocky, artifacted images and uninstall.

  **Realistic Solution:** Separate the model from the app binary entirely using a **staged delivery architecture**:

  (1) **Thin app bundle** — ship the app at ~40 MB with UI, a tiny preview model (~5 MB, generates low-res 128×128 previews), and a download manager. This installs over cellular instantly.

  (2) **Background model download** — on first launch, show a "Preparing your AI engine" screen while downloading the 350 MB model via Apple's Background Assets framework (iOS 16+) or Android's Play Asset Delivery. The download resumes across app kills and network interruptions. Store the model in the app's `Library/Caches` directory (iOS) or internal storage (Android).

  (3) **Progressive quality** — while the full model downloads, let users generate images with the preview model. Low-res results set expectations and demonstrate value, reducing abandonment. When the full model arrives, seamlessly upgrade.

  (4) **Model compression for the wire** — apply weight clustering (256 clusters per tensor) + Huffman coding to the model file. This doesn't change inference precision but compresses the download from 350 MB to ~180 MB. Decompress on-device after download.

  > **Napkin Math:** App Store cellular limit: 200 MB. Thin bundle: 40 MB ✓ (cellular OK). Model download: 350 MB raw, 180 MB compressed. On WiFi (50 Mbps): 180 MB / 6.25 MB/s = 29 seconds. On LTE (20 Mbps): 180 MB / 2.5 MB/s = 72 seconds. User abandonment: 60% at 350 MB forced-WiFi → estimated 15% at 40 MB cellular + background download. Retention improvement: 4× more users complete setup.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The ML Crash vs Silent Failure</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your app uses an on-device model for real-time object detection in AR. Users report 'the app doesn't detect anything sometimes.' Your crash reporting dashboard (Crashlytics) shows zero crashes. How do you detect and diagnose ML failures that don't crash the app?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "If there are no crashes, the model is working fine." ML models fail silently — they return valid tensors filled with garbage instead of throwing exceptions.

  **Realistic Solution:** ML failures are invisible to standard crash reporting because the model always returns *something* — a valid tensor of zeros, random confidences, or stale results. You need **ML-specific health monitoring**:

  (1) **Output validation layer** — wrap every inference call with sanity checks: Are all confidence scores zero? (model returned empty detections — likely input preprocessing failure). Are confidence scores all >0.99? (model is saturated — likely quantization error). Is the output identical to the previous 10 frames? (model is frozen — likely a threading deadlock or stale buffer). Are bounding boxes outside the image bounds? (coordinate space mismatch after a camera resolution change).

  (2) **Inference health heartbeat** — log a lightweight event every 60 seconds: model version, inference count since last heartbeat, mean confidence, mean latency, and a boolean "healthy" flag from the validation layer. This costs ~200 bytes per minute. Ship these events with your standard analytics.

  (3) **Structured ML error taxonomy** — define error codes: `ML_EMPTY_OUTPUT`, `ML_STALE_OUTPUT`, `ML_LATENCY_SPIKE` (>3× P95), `ML_DELEGATE_FALLBACK` (NPU failed, fell back to CPU), `ML_MODEL_LOAD_FAILURE`. Report these as non-fatal events in Crashlytics/Sentry.

  (4) **Reproduction pipeline** — when a silent failure is detected, capture the input tensor (or a hash of it) and the model output. Upload these on WiFi for offline debugging. This lets you reproduce the exact failure in your test environment.

  > **Napkin Math:** Silent failure rate (typical): 0.1-1% of inferences. At 30 FPS for 10 minutes: 18,000 inferences. Silent failures: 18-180 per session. None appear in crash logs. Health heartbeat: 10 events per 10-minute session × 200 bytes = 2 KB. Analytics cost: negligible. Detection latency: 60 seconds (heartbeat interval). With output validation: detected within 1 frame (33ms).

  📖 **Deep Dive:** [Volume I: Robust AI](https://harvard-edge.github.io/cs249r_book_dev/contents/robust_ai/robust_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The App Store ML Review Trap</b> · <code>deployment</code></summary>

- **Interviewer:** "Your team ships an iOS app with a Core ML model that performs on-device face stylization. The app works perfectly on your iPhone 15 Pro test devices. Apple rejects the app during review because it crashes immediately on launch. You realize Apple's review team tested it on an older iPhone 8. Why does the Core ML model compilation step fail on older ANE hardware, and how must you design your model delivery to handle Apple's heterogeneous hardware matrix?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just add a try/catch block around the model load." This prevents the crash but leaves the user with a broken feature, guaranteeing a rejection anyway.

  **Realistic Solution:** Core ML models (`.mlpackage`) are compiled on-device during the first load into a hardware-specific `.mlmodelc` format. The iPhone 15 Pro has a 16-core A17 Pro Neural Engine (ANE) that supports modern operators (like grouped convolutions or specific attention mechanisms). The iPhone 8 (A11 Bionic) has a primitive 2-core ANE with a very restricted operator set. When Core ML attempts to compile the modern model on the A11, it encounters unsupported ops. If the fallback to GPU/CPU is disabled or exceeds memory limits, the compilation fails and the app crashes.

  To pass review and handle the hardware matrix, you must implement **hardware-aware model delivery**:
  (1) **Tiered Models:** Train multiple variants of the model. A high-res, complex model for A14+ devices, and a simplified, highly quantized fallback model (or a completely different architecture like MobileNet instead of ViT) for older devices.
  (2) **Device Capability Checks:** At runtime, query the device's SoC capabilities (using `sysctlbyname` for hardware string, or checking `MLModel` configuration options) before attempting to load or download the model.
  (3) **Graceful Degradation:** If the device is too old even for the fallback model, the app must gracefully disable the feature in the UI rather than crashing or showing a loading spinner forever. Apple reviewers test on a wide range of devices specifically to catch these hardware-capability crashes.

  > **Napkin Math:** iPhone 15 Pro ANE: ~35 TOPS. iPhone 8 ANE: ~0.6 TOPS. That's a 58× compute difference. A model taking 33ms (30 FPS) on the 15 Pro would take ~2 seconds per frame on the iPhone 8, even if it could compile. The memory gap is also massive: 8GB RAM vs 2GB RAM. A 150MB model easily fits in the 15 Pro's memory budget but will trigger an immediate OOM kill on the iPhone 8 before compilation even finishes.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Feature Flag Footgun</b> · <code>deployment</code></summary>

- **Interviewer:** "Your app has 6 ML features — face detection, background blur, scene recognition, smart HDR, night mode, and document scanning. Each can be independently toggled via feature flags from your server. A junior engineer enables all 6 simultaneously on a Pixel 6a (6 GB RAM, Tensor G1). Users report the phone becomes unusable — launcher crashes, apps get killed. What happened?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Each model is only 20-30 MB, so 6 models = 180 MB. That should fit in 6 GB RAM." This counts only weight storage and ignores runtime memory — activation buffers, delegate memory, and framework overhead.

  **Realistic Solution:** The problem is **cumulative ML runtime memory**, not model file size:

  (1) **Per-model runtime cost** — each loaded TFLite model allocates: weight tensors (20-30 MB mapped from disk), activation buffers (15-40 MB depending on input resolution), GPU/NNAPI delegate workspace (30-50 MB for the Tensor G1's GPU delegate), and TFLite interpreter overhead (~5 MB). Total per model: 70-125 MB *resident*.

  (2) **Six models loaded** — 6 × ~100 MB = 600 MB of ML runtime memory. The Pixel 6a with 6 GB RAM has ~2.5 GB available after Android OS, system services, and the launcher. Your app consuming 600 MB triggers Android's low-memory killer (LMK), which starts killing background apps — including the launcher.

  (3) **Feature flag discipline** — feature flags must be *resource-aware*. Define a per-device memory budget (e.g., 200 MB for ML on 6 GB devices, 400 MB on 12 GB devices). The feature flag system checks available RAM before enabling each feature. Use a priority queue: face detection > smart HDR > background blur > others. Load models on-demand and evict idle models after 30 seconds.

  (4) **Shared backbone** — if multiple features share similar low-level features, use a single shared backbone with lightweight task heads. This reduces 6 × 100 MB to 100 MB backbone + 6 × 10 MB heads = 160 MB.

  > **Napkin Math:** Pixel 6a: 6 GB total, ~2.5 GB available. Android LMK threshold: ~200 MB free. Safe ML budget: 2.5 GB - 200 MB buffer - 500 MB app = 1.8 GB. Six independent models: 600 MB (fits, but leaves only 1.2 GB for everything else). With shared backbone: 160 MB (leaves 2.14 GB). Maximum concurrent models at 100 MB each: floor(1.8 GB / 100 MB) = 18 models. But with delegates: floor(1.8 GB / 100 MB) = only 3-4 models safely.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The ML Error That Looks Like a Feature</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your camera app uses an on-device portrait mode model. After a routine iOS update, users start posting on social media that your app's portrait mode 'looks different — more dreamy.' Engagement actually increases 5%. Your QA team didn't catch any issue. But your ML engineer says the model is producing wrong results. What happened, and do you fix it or ship it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "If engagement is up, the model is working better — don't touch it." This is dangerous. An uncontrolled change to model behavior means you don't understand your system.

  **Realistic Solution:** Diagnose first, then decide:

  (1) **Root cause** — the iOS update changed the camera pipeline's default color space from sRGB to Display P3 (wider gamut). Your model was trained on sRGB images. When fed P3 inputs, the model's depth estimation shifts — it interprets the wider color values as different depth cues, producing a softer, more diffused bokeh. The "dreamy" look is a systematic error in depth estimation, not an intentional effect.

  (2) **Why this is dangerous** — the error is *input-dependent*. For most scenes, the softer bokeh looks pleasant. But for high-contrast edges (person against a bright window), the depth error causes the model to blur the subject's hair and ears into the background. Users haven't noticed yet because it's subtle, but a competitor will screenshot it.

  (3) **The right decision** — fix the model, but *keep the aesthetic as an option*:
  - Immediate fix: add color space normalization in the preprocessing pipeline. Convert P3 → sRGB before inference. This restores correct depth estimation.
  - Product opportunity: the "dreamy" look has user demand. Train a deliberate soft-bokeh variant and offer it as a "Soft Focus" mode. This gives users the aesthetic they liked while maintaining correct behavior as the default.

  (4) **Prevention** — add a preprocessing assertion that checks the input color space and logs a warning if it doesn't match the training distribution. Add an integration test that runs inference on both sRGB and P3 inputs and asserts output similarity within a tolerance.

  > **Napkin Math:** Color space difference: sRGB gamut covers 35% of CIE 1931. Display P3 covers 45%. Pixel values in P3 are ~15% lower for the same perceived color (wider range, same 8-bit encoding). Depth estimation error: ~8% systematic bias toward "farther" (lower values interpreted as less saturated = less foreground). Bokeh radius increase: 8% wider blur kernel. Visually: subtle but measurable. Edge artifact rate: ~3% of photos have visible subject-background bleed.

  📖 **Deep Dive:** [Volume I: Robust AI](https://harvard-edge.github.io/cs249r_book_dev/contents/robust_ai/robust_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The ML Notification Backlash</b> · <code>deployment</code> <code>monitoring</code></summary>

- **Interviewer:** "Your news app uses an on-device ML model to predict which articles a user will click, and sends push notifications for high-confidence predictions. After launching the feature on the Google Pixel 8 (Tensor G3), your app's rating drops from 4.5 to 3.2 stars in two weeks. Reviews say 'too many notifications' and 'the app thinks it knows what I want but it's wrong.' The model has 78% precision on your test set. What went wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "78% precision means 78% of notifications are relevant — that's pretty good." 78% precision means 22% of notifications are *irrelevant*. At high notification volume, that's a disaster.

  **Realistic Solution:** The problem is a mismatch between ML metrics and user experience metrics:

  (1) **Volume amplification** — your model scores every article in the user's feed (200 articles/day). At a confidence threshold of 0.7, it flags 40 articles as "high confidence." 78% precision: 31 relevant + 9 irrelevant notifications per day. Users tolerate 3-5 notifications/day from a news app. You're sending 8× too many, and 9 of them are wrong. Each wrong notification erodes trust multiplicatively — one bad notification makes users doubt the next three good ones.

  (2) **Precision ≠ user satisfaction** — your test set measures "would the user click this article?" But clicking ≠ wanting a notification. Users click articles they see in-feed (low commitment) but resent being *interrupted* for the same content. The notification threshold should be much higher than the in-feed recommendation threshold.

  (3) **Fix: notification budget** — cap notifications at 3/day regardless of model confidence. Rank all candidates by confidence and send only the top 3. This transforms the problem from "is this article relevant?" (precision) to "which 3 articles are *most* relevant?" (ranking). Even with imperfect ranking, 3 notifications/day is within user tolerance.

  (4) **Fix: user feedback loop** — track notification dismissal rate (swipe away without opening). If a user dismisses 3 consecutive notifications, suppress notifications for 24 hours and raise the confidence threshold by 0.1 for that user. If they open a notification, lower the threshold by 0.05. This creates a personalized threshold that converges to each user's tolerance.

  (5) **Fix: time-aware sending** — the model predicts *what* to send but not *when*. Send notifications during the user's historically active hours (learned from app usage patterns). A notification at 3 AM about a relevant article is still a bad notification.

  > **Napkin Math:** Before fix: 40 notifications/day, 22% irrelevant = 8.8 bad notifications/day. User tolerance: 3-5/day total. Annoyance factor: 40/5 = 8× over tolerance. After budget cap (3/day): top-3 from 40 candidates. Precision of top-3: ~95% (high-confidence items are more accurate). Bad notifications: 0.15/day. User satisfaction: dramatically improved. Rating recovery timeline: 4-6 weeks (users slowly update reviews). Notification open rate: before (12% — notification fatigue), after (38% — fewer, better notifications). Revenue impact: 38% × 3 = 1.14 opened/day vs 12% × 40 = 4.8 opened/day. Fewer opens but higher quality engagement and retention.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The App Size Bloat</b> · <code>deployment</code> <code>persistent-storage</code></summary>

- **Interviewer:** "Your team is adding an offline translation feature to your iOS app. The models total 300 MB. You bundle them directly into the app binary and ship the update. The next week, your product manager reports that app installations from new users have plummeted by 40%. Why did bundling the model destroy your acquisition metrics?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model must be making the app slow or crashing on older devices." It's not a performance issue—the users aren't even getting the app onto their phones.

  **Realistic Solution:** You hit the **Cellular Download Limit**. Both Apple's App Store and Google Play have strict (or strongly warned) limits on how much data a user can download over a cellular network (historically 150 MB or 200 MB). By bundling a 300 MB model into the binary, your app exceeds this limit.

  When a user tries to download your app while waiting for a bus, the OS blocks the download and prompts them to connect to Wi-Fi. Most users dismiss the prompt and forget to download the app later.

  **The Fix:** Decouple the ML models from the app binary. Ship the app with the bare minimum UI and business logic (keeping it under 100 MB). Once the app is installed and launched, use **On-Demand Resources (ODR)** or a background download service to fetch the 300 MB model from a CDN, ideally pausing if the user drops off Wi-Fi.

  > **Napkin Math:** App binary: 50 MB + 300 MB models = 350 MB. 350 MB > 200 MB cellular limit. Conversion drop-off for Wi-Fi prompts is typically 30-50%. Moving the model to ODR keeps the initial download at 50 MB, ensuring 100% conversion on cellular.

  📖 **Deep Dive:** [Volume I: Optimizing AI](https://harvard-edge.github.io/cs249r_book_dev/contents/optimizing_ai/optimizing_ai.html)

  </details>

</details>


#### 🔵 L4
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Secure Enclave Boundary</b> · <code>security</code></summary>

- **Interviewer:** "Your banking app uses on-device face verification (FaceID/BiometricPrompt). The ML model runs on the main NPU and returns a 128-dim embedding. The embedding is compared against the enrolled template stored in the Secure Enclave (Apple) or StrongBox (Android). A security researcher points out that the embedding travels from the NPU through main memory to the Secure Enclave — and can be intercepted. How do you protect the embedding in transit, and why can't you just run the ML model inside the Secure Enclave?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run the face verification model inside the Secure Enclave for end-to-end security." The Secure Enclave is a tiny, fixed-function processor — it can't run neural networks.

  **Realistic Solution:** The Secure Enclave (Apple) / StrongBox (Android) is a dedicated security processor with ~256 KB of memory and a simple RISC core. It handles cryptographic operations (AES, RSA, ECC) and key storage. It cannot run a face verification model that requires ~50 MB of working memory and millions of MAC operations.

  The architectural challenge: the embedding must cross the **trust boundary** between the rich execution environment (REE — where the NPU lives) and the trusted execution environment (TEE — where the Secure Enclave lives). During this transit, the embedding is in main DRAM and theoretically accessible to a compromised OS.

  **Defense layers**: (1) **Apple's approach (Face ID)**: the entire face verification pipeline runs inside the Secure Neural Engine — a dedicated portion of the ANE that operates within the Secure Enclave's trust domain. The raw camera data flows directly from the TrueDepth sensor to the Secure Neural Engine via a hardware-isolated data path. The embedding never enters main memory. This is a hardware solution — Apple designed the silicon specifically for this.

  (2) **Android's approach (less integrated)**: the embedding transits through main memory. Mitigations: (a) encrypt the embedding with a session key derived from the StrongBox before it leaves the NPU driver. (b) Use Android's `KeyStore` with hardware-backed keys — the comparison happens inside the TEE (TrustZone), and the enrolled template never leaves TrustZone. (c) Time-bound the embedding — it's valid for only 100ms, making replay attacks impractical.

  (3) **Emerging approach**: ARM's Confidential Compute Architecture (CCA) creates "realms" — isolated execution environments that can run larger workloads (including small ML models) with hardware-enforced memory encryption. This could eventually allow the face model to run entirely within a CCA realm.

  > **Napkin Math:** Secure Enclave memory: ~256 KB. Face verification model (MobileFaceNet): ~5 MB weights + ~20 MB activations = 25 MB. Ratio: 25 MB / 256 KB = **100× too large** for the Secure Enclave. Embedding size: 128 × 4 bytes = 512 bytes. Transit time through DRAM: ~0.1µs. Attack window: 0.1µs — extremely narrow, but a kernel-level attacker with DMA access could capture it. Apple's hardware isolation: 0µs in DRAM — the embedding never leaves the secure data path. Cost of Apple's approach: custom silicon design ($100M+ NRE). Cost of Android's approach: software encryption ($0, but weaker guarantee).

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Cold Start Jitter</b> · <code>serving</code></summary>

- **Interviewer:** "Your ML-powered camera app experiences a noticeable 'hiccup' the very first time a user opens it and tries to snap a photo, even though subsequent photo captures are instantaneous. The inference itself is fast (5ms). What system-level factors contribute to this initial delay, and how would you optimize them?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The first inference is just slower because of cache misses." While cache misses play a role, the initial delay is far more significant and involves several system-level setup costs.

  **Realistic Solution:** The initial delay, often called "cold start latency," is due to several one-time setup costs:
  1.  **Model Loading from Storage:** The `.tflite` or `.mlmodel` file must be read from flash storage (eMMC/UFS) into RAM.
  2.  **Memory Allocation:** Significant chunks of memory are allocated for the model weights, intermediate tensors, and NPU/GPU context. This can involve page faults as the OS maps virtual memory to physical pages.
  3.  **Runtime/Delegate Initialization:** The ML runtime (e.g., TFLite, Core ML) and its hardware delegates (NPU, GPU) need to be initialized. This includes setting up the execution graph, compiling/optimizing the model for the specific hardware, and potentially loading accelerator-specific firmware or drivers.
  4.  **Accelerator Context Setup:** The NPU or GPU needs to set up its execution context, allocate internal buffers, and "warm up" its internal state for the first inference.
  5.  **JIT Compilation (if applicable):** Some runtimes might perform Just-In-Time compilation of parts of the graph on the first run.

  **Optimization Strategies:**
  *   **Pre-load/Pre-initialize:** Load the model and initialize the runtime/delegate in the background during app startup or a non-critical phase.
  *   **Memory-map Model:** Use `mmap` to load the model file directly into memory, avoiding a full copy and allowing the OS to page it in on demand.
  *   **Smaller Model Variants:** Use a smaller, faster model for the initial experience, then swap to a larger, more accurate one in the background.
  *   **Model Caching:** For models downloaded over-the-air, ensure they are cached locally in an optimized format.
  *   **Quantization/Pruning:** Reduce model size to speed up loading and reduce memory footprint.
  *   **Runtime Optimizations:** Utilize runtime features like pre-compiled delegates or AOT (Ahead-Of-Time) compilation where available.

  > **Napkin Math:** A 50MB model takes $\frac{50MB}{500MB/s} = 0.1s$ (100ms) to load from UFS 3.1. Add another 50-100ms for runtime/delegate initialization and context setup, and the total cold start could easily be 150-200ms, which is a noticeable hiccup.

  > **Key Equation:** $T_{cold\_start} = T_{disk\_read} + T_{mem\_alloc} + T_{runtime\_init} + T_{accelerator\_setup}$

  📖 **Deep Dive:** [Volume I: Model Deployment](https://mlsysbook.ai/vol1/model-deployment)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Offline-First ML Design</b> · <code>deployment</code> <code>serving</code></summary>

- **Interviewer:** "You're building a plant identification app that must work in remote areas with no cellular or WiFi connectivity. The cloud model (ResNet-152, 230 MB, 96% accuracy) is too large for on-device. Design the offline-first ML architecture: what model do you ship, how do you handle the accuracy gap, and what's the total on-device storage budget? Target device: a mid-range phone with 4 GB RAM and 64 GB storage (MediaTek Dimensity 7200)."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just compress the ResNet-152 to fit on-device." Compressing a large model designed for cloud inference produces a worse accuracy/size trade-off than using a model architecturally designed for mobile from the start.

  **Realistic Solution:** Design a tiered offline-first system:

  **On-device model:** EfficientNet-B0 (5.3M params, INT8 = 5.3 MB). Accuracy: 77.1% top-1 on ImageNet, ~85% on a curated plant dataset (narrower domain = higher accuracy). Latency on Dimensity 7200 NPU (~8 TOPS): ~6ms. Fits comfortably in 4 GB RAM.

  **Accuracy gap mitigation:** (1) **Domain-specific fine-tuning** — train on plant images only (10K species). Domain-specific models outperform general models: 85% → 92% accuracy on plant ID. (2) **Confidence thresholding** — if model confidence < 80%, show top-3 predictions instead of top-1. Users accept "it might be one of these three" offline. (3) **Deferred cloud verification** — queue low-confidence predictions. When connectivity returns, send the image to the cloud model for verification. Update the on-device result retroactively.

  **Storage budget:** Model: 5.3 MB. Plant database (10K species × 500-byte description + 50 KB thumbnail): 10K × 50.5 KB = 492 MB. Offline maps for GPS context: ~200 MB. App binary: ~30 MB. Image cache (user photos): ~500 MB. Total: **~1.23 GB** out of 64 GB (1.9%).

  **Sync strategy:** When online, download model updates (delta patches: ~500 KB), sync deferred verifications, and update the plant database. Use background fetch to avoid blocking the user.

  > **Napkin Math:** Cloud model: 230 MB, 96% accuracy. On-device: 5.3 MB, 92% accuracy (domain-tuned). Size reduction: **43×**. Accuracy gap: **4%** (mitigated by confidence thresholding + deferred verification). Effective accuracy with deferred verification: ~95% (cloud corrects the 8% of low-confidence predictions within 24 hours). Storage: 1.23 GB / 64 GB = 1.9%. RAM: 5.3 MB model + 50 MB runtime = 55 MB / 4 GB = 1.4%. The offline model is viable on even budget hardware.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The On-Device Model Hot-Swap</b> · <code>deployment</code></summary>

- **Interviewer:** "Your app's ML model needs to be updated without an App Store update. The current model has a critical misclassification bug — it labels some food items as non-food, breaking your calorie tracking feature. An App Store update takes 24-48 hours for review. How do you push a model fix in under 1 hour?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Ship the model as a server-side config and swap it remotely." This violates Apple's App Store Review Guidelines §2.5.2 — apps may not download executable code. Model weights are not executable code, but the approach needs careful implementation.

  **Realistic Solution:** Design a **dynamic model delivery system** that stays within platform guidelines:

  (1) **Model registry service** — host versioned models on your CDN (CloudFront, Firebase Hosting). Each model has a manifest: version, hash (SHA-256), minimum app version, target OS versions, file size, and a rollout percentage.

  (2) **On-device model manager** — at app launch (and periodically in background), the app checks the manifest. If a new model is available and the device matches the targeting criteria, download it. Verify the SHA-256 hash before accepting. Store in the app's sandboxed storage.

  (3) **Atomic swap** — the app keeps the current model loaded in memory. The new model is loaded into a second buffer, validated with a test inference on a known input/output pair. Only if validation passes does the app atomically swap the model pointer. The old model file is kept as a rollback target for 7 days.

  (4) **Platform compliance** — Apple allows downloading "data" (including model weights) post-install. The model format (Core ML `.mlmodelc`, TFLite `.tflite`) is interpreted by Apple/Google's own frameworks, not custom executable code. Firebase ML and Apple's Core ML Model Deployment (CloudKit) are first-party solutions that handle this pattern.

  (5) **Staged rollout** — push to 1% of users, monitor crash rates and engagement metrics for 1 hour, then expand to 10% → 50% → 100%. Total time from model fix to full rollout: ~4 hours.

  > **Napkin Math:** Model size: 25 MB (INT8 classification model). CDN download at 20 Mbps LTE: 25 MB / 2.5 MB/s = 10 seconds. Validation inference: 50ms. Swap: <1ms (pointer swap). Total update time per device: ~11 seconds. Staged rollout: 1% (1 hour) → 10% (1 hour) → 50% (1 hour) → 100% (1 hour) = 4 hours to full fleet. vs App Store: 24-48 hours review + user must manually update.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Silent Accuracy Degradation</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your on-device image classification model was deployed 6 months ago. Users haven't complained, but your A/B test shows the new model version improves engagement by 15%. This suggests the old model has silently degraded. You have no server-side ground truth labels for on-device predictions. How do you detect model degradation without ground truth?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Monitor accuracy using a held-out test set." You don't have labels for on-device predictions — there's no test set.

  **Realistic Solution:** Four proxy signals that detect degradation without ground truth:

  (1) **Confidence distribution shift** — track the distribution of the model's softmax confidence scores over time. A healthy model has a bimodal distribution (high confidence for easy inputs, low for hard). If the distribution shifts toward uniform confidence (the model becomes "confused"), it's seeing out-of-distribution data. Use KL divergence between the current week's confidence distribution and the baseline.

  (2) **User implicit feedback** — track behavioral proxies: if the model powers a photo search feature, monitor search refinement rate (user searches again immediately = bad result), feature abandonment rate, and time-to-action after a prediction. A 15% engagement improvement from the new model implies the old model's predictions were increasingly irrelevant.

  (3) **Lightweight anomaly detector** — deploy a small autoencoder (~1 MB) alongside the main model. Train it on the same distribution as the main model. If reconstruction error exceeds a threshold, the input is OOD. Track the OOD rate over time — a rising rate indicates distribution drift.

  (4) **Federated evaluation** — periodically sample a small subset of users, send them a labeled evaluation batch (e.g., 100 images with known labels), and compare the model's predictions. This gives you direct accuracy measurement without collecting user data. Privacy-preserving: the evaluation data comes from you, not from users.

  On mobile, a common silent degradation source is OS-level framework updates. When iOS 18 updates the CoreML runtime, the Neural Engine's operator fusion patterns change. A model that was calibrated for iOS 17's fusion graph may see 2-3% accuracy drift because intermediate activations are computed in a different order, accumulating different floating-point rounding errors. Monitor per-OS-version accuracy, not just global accuracy.

  > **Napkin Math:** Confidence monitoring: ~1 byte per inference (just the max softmax score). 1000 inferences/day × 30 days = 30,000 data points. KL divergence computation: trivial. Storage: ~30 KB/month. Anomaly detector: 1 MB model, 0.5ms per inference. Federated evaluation: 100 labeled images × 4 evaluations/year = 400 labeled predictions per user per year. With 10,000 users: 4M labeled predictions — statistically powerful.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The A/B Test Without Ground Truth</b> · <code>monitoring</code></summary>

- **Interviewer:** "You have two versions of your on-device recommendation model. In cloud ML, you'd A/B test by comparing click-through rates against a server-side holdout. But your model runs entirely on-device — predictions never hit your server. How do you A/B test two on-device models when you can't observe the predictions?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Send all predictions to the server for analysis." This defeats the purpose of on-device inference (privacy, latency) and may violate your privacy policy.

  **Realistic Solution:** The key insight is that on mobile, the *hardware itself* generates the telemetry you need. Design an **on-device A/B testing framework** that uses hardware-level signals as proxy metrics when ground truth is unavailable:

  (1) **Assignment** — at app install (or experiment start), assign users to cohorts using a deterministic hash of their anonymous device ID + experiment ID. No server round-trip needed. 50/50 split: `hash(device_id + "exp_042") % 2`.

  (2) **Hardware telemetry as proxy metrics** — you can't observe prediction correctness, but the SoC tells you everything about the *cost* of each prediction. Collect per-cohort: NPU utilization (via `PowerMetrics` on iOS, `dumpsys` on Android), thermal state transitions (how often the thermal governor throttles), inference latency percentiles (P50, P95, P99 — measured at the delegate level, not the app level), and battery drain per inference (energy impact gauge). These hardware signals reveal whether a model is *systemically better* — a model that produces the same engagement at lower NPU utilization and lower thermal pressure is objectively superior even without ground truth labels.

  (3) **Outcome metrics layered on top** — observe behavioral *consequences*: session duration, feature usage frequency, conversion events (purchase, share, save), retention (day 1, day 7, day 30). These are standard analytics events that don't expose model predictions.

  (4) **Guardrail metrics** — a model that improves engagement but drains 2× more battery is not a winner. Gate experiment winners on *both* engagement improvement AND hardware cost parity.

  (5) **Statistical rigor** — with on-device experiments, you need larger sample sizes because outcome metrics are noisier than prediction metrics. Plan for 2-4 week experiment duration with 50,000+ users per cohort.

  > **Napkin Math:** If model A averages 8ms inference on the Snapdragon Hexagon NPU and model B averages 12ms, but model B's engagement rate is 5% higher, the 4ms difference costs 0.02 mAh per inference × 1000 inferences/day = 20 mAh/day extra battery. Is the 5% engagement worth 2% battery life? That's the systems trade-off. Experiment sizing: 100,000 users, 50/50 split. Required sample for 2% absolute retention difference: n = 16 × p(1-p) / δ² = 16 × 0.4 × 0.6 / 0.02² = 9,600 per cohort. With 50,000 per cohort: can detect 0.9% absolute difference. Duration: 14 days minimum.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The On-Device Model Encryption Dilemma</b> · <code>security</code></summary>

- **Interviewer:** "Your company spent $2M training a proprietary image segmentation model. You ship it as a Core ML `.mlmodelc` bundle on iOS and a TFLite `.tflite` flatbuffer on Android. A competitor extracts your model from the IPA/APK in 10 minutes using standard tools. How do you protect your model IP on-device?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Encrypt the model file with AES-256 and decrypt it at runtime." This protects the file at rest, but the decrypted model must live in memory during inference. On a jailbroken iPhone or rooted Android, an attacker can dump the process memory and recover the full model in plaintext. You've added complexity without real protection.

  **Realistic Solution:** Defense-in-depth with hardware-backed protection:

  (1) **iOS: Core ML encrypted model** — since iOS 16, Core ML supports compiled model encryption. The model is encrypted with a key stored in the Secure Enclave. Decryption happens inside the Neural Engine's trusted execution path — the plaintext weights never appear in user-space memory. This is the strongest protection available on mobile. The key is tied to your Team ID; even Apple cannot extract it.

  (2) **Android: split architecture** — Android lacks an equivalent to Core ML encryption. Instead, split the model: ship the architecture (layer graph) in the APK, but keep the final 2-3 classification layers' weights on your server. At inference time, the on-device model produces a 512-dim embedding, which is sent to your server for the final classification. An attacker who extracts the on-device model gets a feature extractor, not your proprietary classifier. Latency cost: ~50 ms round-trip on LTE.

  (3) **Obfuscation layer** — rename all tensor names to random hashes, strip metadata, and apply weight permutation with a device-specific seed derived from the Android Keystore. This doesn't prevent extraction but makes reverse-engineering the architecture significantly harder.

  (4) **Watermarking** — embed a statistical watermark in the model weights (a specific pattern in the least significant bits). If a competitor ships a suspiciously similar model, you can prove provenance by detecting the watermark in their outputs.

  > **Napkin Math:** Extraction time without protection: ~10 minutes (unzip IPA, find `.mlmodelc`). With Core ML encryption: requires Secure Enclave exploit (estimated cost: $500K+ for a zero-day). With Android split architecture: attacker gets embedding model but not classifier. Server round-trip: 512 floats × 4 bytes = 2 KB upload, ~50 ms on LTE. At 100 inferences/day: 200 KB bandwidth/day — negligible. Watermark detection: 95% confidence with 50 output samples from the suspect model.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Mobile ML Telemetry Budget</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your photo app runs a style transfer model on-device. You want to collect telemetry — inference latency, confidence scores, model version, device thermal state — to monitor model health across 5 million DAU. Your analytics pipeline charges $4 per million events. The CFO says your ML telemetry budget is $500/month. Design the telemetry system."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Log every inference and sample server-side." At 5M DAU × 20 inferences/day = 100M events/day × 30 days = 3B events/month × $4/M = $12,000/month. That's 24× over budget.

  **Realistic Solution:** Telemetry must be designed as a systems problem with on-device aggregation:

  (1) **On-device aggregation** — don't send per-inference events. Instead, accumulate a local summary every 6 hours: P50/P95/P99 latency, mean confidence, inference count, error count, thermal throttle count, model version, OS version, device model. One summary event = ~500 bytes. Four events per device per day.

  (2) **Stratified sampling** — not all devices need to report. Sample 2% of users deterministically (hash of device ID). 5M × 2% = 100K reporting devices × 4 events/day × 30 days = 12M events/month × $4/M = $48/month. Well under budget.

  (3) **Anomaly escalation** — the 98% of non-sampled devices still monitor locally. If a device detects an anomaly (latency > 3× P95 baseline, error rate > 5%, thermal shutdown during inference), it promotes itself to the reporting cohort and sends a detailed diagnostic event. This captures rare failures without paying for healthy-device telemetry.

  (4) **Cohort-level dashboards** — segment by device model (iPhone 13 vs Pixel 7), OS version, and model version. With 100K reporting devices, each cohort has thousands of data points — statistically sufficient for detecting 2% regression with 95% confidence.

  > **Napkin Math:** Naive: 3B events/month = $12,000. Aggregated: 5M × 4 events/day × 30 = 600M events = $2,400. Sampled 2%: 12M events = $48. With anomaly escalation (assume 0.5% anomaly rate): 12M + 0.5% × 5M × 30 × 1 = 12.75M events = $51/month. Budget: $500. Remaining $449 for storage, dashboards, and alerting.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Model Rollback Nightmare</b> · <code>deployment</code></summary>

- **Interviewer:** "You pushed a new on-device food recognition model to 100% of users via your dynamic model delivery system. Within 2 hours, your telemetry shows a 40% spike in 'no detection' events on Samsung Galaxy S21 devices running Android 13. Other devices are fine. You need to rollback — but only for affected devices. How?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Roll back to the previous model for all users." This punishes 90% of users who are working fine. The new model may be significantly better for them — a global rollback costs real engagement.

  **Realistic Solution:** Implement **targeted rollback** using your model manifest system:

  (1) **Root cause first** — the Galaxy S21 uses the Samsung Exynos 2100 (in non-US markets) or Snapdragon 888. The Exynos NNAPI driver on Android 13 has a known bug with grouped convolution operators that your new model uses. The Snapdragon 888 driver handles them correctly. This is a *driver-specific* failure, not a model bug.

  (2) **Manifest-based targeting** — your model manifest already includes targeting criteria. Add a negative targeting rule: `exclude: {soc: "exynos2100", os: "android_13"}`. Devices matching this rule receive the previous model version. All other devices keep the new model.

  (3) **Atomic rollback on-device** — your model manager keeps the previous model file for 7 days. When the manifest tells an Exynos 2100 device to use v1.2 instead of v1.3, the device checks its local cache. If v1.2 is still cached, it swaps immediately (pointer swap, <1 ms). If evicted, it re-downloads v1.2 from the CDN.

  (4) **Permanent fix** — file a bug with Samsung's NNAPI team. In the meantime, export a variant of v1.3 that replaces grouped convolutions with equivalent depthwise-separable convolutions (slightly larger, but NNAPI-compatible). Ship this as v1.3.1 targeted specifically at Exynos devices.

  > **Napkin Math:** Affected population: Galaxy S21 Exynos on Android 13 ≈ 3% of user base. Global rollback cost: 97% of users lose new model improvements (estimated 8% engagement lift). Targeted rollback cost: 3% of users temporarily on old model. Net engagement preserved: 97% × 8% = 7.76% lift retained vs 0% with global rollback. Rollback speed: manifest update propagates in <1 hour (next app launch checks manifest). CDN cache invalidation: 5 minutes. 90% of affected devices rolled back within 2 hours.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The ML Crash Report Black Hole</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your on-device speech recognition model causes the app to crash on 0.3% of sessions. Crashlytics shows the crash stack trace ends in `CoreML::Espresso::generic_reshape_kernel` — deep inside Apple's private framework. You can't read Apple's source code. You can't reproduce it locally. How do you diagnose and fix this?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "File a bug with Apple and wait." Apple's Core ML bug turnaround is 3-6 months. Your users are crashing now.

  **Realistic Solution:** Diagnose by correlating crash metadata with model execution state:

  (1) **Enrich crash reports** — before every inference call, write a breadcrumb to Crashlytics: input tensor shape, input value range (min/max), model version, delegate type (ANE vs GPU vs CPU), and available memory. When the crash occurs, these breadcrumbs survive and appear in the crash report.

  (2) **Pattern analysis** — filter crashes by device model, iOS version, and your breadcrumbs. The `generic_reshape_kernel` crash on the Apple Neural Engine (ANE) is a known class of bugs triggered by specific input shapes. If 95% of crashes occur when the input audio length is not a multiple of 64 samples, you've found the trigger: the ANE's reshape kernel doesn't handle non-aligned tensor dimensions.

  (3) **Defensive padding** — pad all input tensors to the nearest multiple of 64 before inference. Cost: a few microseconds of memset. This eliminates the crash without understanding Apple's internal code.

  (4) **Delegate fallback** — wrap the inference call in a crash-safe harness. If the ANE crashes, catch the signal, mark the device as "ANE-unsafe" in UserDefaults, and fall back to GPU or CPU delegate on next launch. Latency increases from 8 ms (ANE) to 25 ms (GPU) — acceptable for speech recognition.

  > **Napkin Math:** Crash rate: 0.3% of sessions. At 1M DAU with 3 sessions/day: 9,000 crashes/day. After input padding fix: crash rate drops to 0.01% (residual edge cases). Remaining 300 crashes/day handled by delegate fallback. ANE latency: 8 ms. GPU fallback: 25 ms. CPU fallback: 120 ms. User-perceptible impact of GPU fallback: none (speech recognition tolerance is ~200 ms).

  📖 **Deep Dive:** [Volume I: Robust AI](https://harvard-edge.github.io/cs249r_book_dev/contents/robust_ai/robust_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The App Size Audit</b> · <code>deployment</code></summary>

- **Interviewer:** "Your app is 280 MB. Analytics show that for every 10 MB increase in app size, install conversion drops by 1.5%. Marketing wants to add a new 45 MB generative AI feature. Engineering says the app is already at the limit. Find 60 MB of savings without removing any ML features — the app has 4 models totaling 120 MB."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Quantize all models from FP16 to INT8 — that halves the model size from 120 MB to 60 MB." Not all models are FP16. Two of your four models are already INT8. Blindly re-quantizing INT8 models to INT4 will destroy accuracy on the two sensitive models (OCR and face detection).

  **Realistic Solution:** Systematic app size audit targeting all binary components, not just models:

  (1) **Model-level savings** — Model A (text detection, 45 MB FP16): quantize to INT8 → 22.5 MB. Saves 22.5 MB. Model B (face mesh, 35 MB INT8): already optimized. Model C (style transfer, 25 MB FP16): quantize to INT8 → 12.5 MB. Saves 12.5 MB. Model D (scene classifier, 15 MB INT8): already optimized. Total model savings: 35 MB.

  (2) **Framework deduplication** — your app links both TensorFlow Lite (8 MB) and Core ML runtime shims (3 MB). If all models can run on Core ML (iOS) or TFLite (Android), remove the redundant framework. Savings: 3-8 MB per platform.

  (3) **Asset stripping** — run `xcrun bitcode_strip` (iOS) and `R8 shrinking` (Android). Strip debug symbols (`-s` flag), remove unused architectures (arm64 only, drop armv7). Typical savings: 10-15 MB.

  (4) **On-demand delivery** — move the style transfer model (used by <5% of users) to on-demand resources. It downloads only when the user first opens the style transfer feature. Savings: 12.5 MB from the initial bundle.

  (5) **Shared weight deduplication** — if models A and C share a MobileNet backbone (common in multi-task apps), extract the shared backbone as a single file and load it once. Savings: ~10 MB if the backbone is 50% of each model.

  > **Napkin Math:** Starting: 280 MB. Model quantization: -35 MB. Framework dedup: -5 MB. Asset stripping: -12 MB. On-demand style transfer: -12.5 MB. Total savings: 64.5 MB. New size: 215.5 MB. Adding 45 MB AI feature: 260.5 MB. Net change from original: -19.5 MB. Install conversion impact: +3% improvement (from 280 → 260 MB). Revenue at 100K installs/month, $2 LTV: 3% × 100K × $2 = $6,000/month additional revenue.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Mobile ML CI/CD Pipeline</b> · <code>deployment</code></summary>

- **Interviewer:** "Your ML team trains models in PyTorch on cloud GPUs. Your mobile team consumes Core ML and TFLite models. Currently, the handoff is a Slack message with a Google Drive link. Last month, someone shipped a model trained on the wrong dataset, and it took 3 days to notice. Design a CI/CD pipeline that prevents this."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Add the model to the app's git repo and use pull requests for review." ML models are 20-200 MB binary blobs. Git is not designed for this — your repo will balloon, clones will take forever, and diffs are meaningless on binary files.

  **Realistic Solution:** Build a **model-aware CI/CD pipeline** with automated quality gates:

  (1) **Model registry** — store models in a versioned artifact registry (MLflow, Weights & Biases, or S3 with DynamoDB metadata). Each model version records: training dataset hash, training config, evaluation metrics, conversion format, and the git SHA of the training code. No model enters the registry without this metadata.

  (2) **Automated conversion** — CI pipeline converts PyTorch → ONNX → Core ML + TFLite. This runs in CI, not on a developer's laptop. The conversion script is version-controlled. If conversion fails (unsupported ops), the pipeline blocks.

  (3) **Quality gates** — before a model is eligible for mobile integration: (a) accuracy on the canonical test set must be ≥ threshold (e.g., 91% for your food classifier), (b) model size must be ≤ budget (e.g., 25 MB INT8), (c) inference latency on a reference device (tested via Firebase Test Lab or AWS Device Farm) must be ≤ 50 ms on a Pixel 6, (d) output parity check — run 1000 test inputs through both the PyTorch model and the converted Core ML/TFLite model; max absolute difference must be < 0.01.

  (4) **Staged rollout integration** — the CI pipeline doesn't just build the model; it pushes it to your model delivery CDN with a 0% rollout. A separate release pipeline (triggered by a human) ramps from 1% → 10% → 100% with automated rollback if telemetry degrades.

  (5) **Lineage tracking** — every model in production links back to: the training run, the dataset version, the conversion pipeline version, and the CI build. When the wrong-dataset incident happens, you can trace it in minutes instead of days.

  > **Napkin Math:** Manual handoff: 3 days to detect wrong dataset. CI/CD pipeline: quality gate catches it in ~15 minutes (accuracy test fails because wrong dataset → different accuracy). Conversion pipeline runtime: PyTorch → ONNX (2 min) → Core ML (3 min) → TFLite (3 min) → device testing (10 min) = 18 minutes total. Cost: Firebase Test Lab at $5/device-hour × 0.2 hours × 3 devices = $3 per model version. At 10 model versions/week: $30/week.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Model Cache Eviction Problem</b> · <code>deployment</code></summary>

- **Interviewer:** "Your messaging app downloads ML models on-demand — a sticker suggestion model (8 MB), a smart reply model (12 MB), and a photo enhancement model (30 MB). You store them in the iOS Caches directory. Users complain that the photo enhancement feature 'takes forever to load' every few days. What's happening?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The CDN is slow — optimize the download speed." The download speed is fine. The problem is that the model keeps getting *deleted* and re-downloaded.

  **Realistic Solution:** iOS automatically purges the Caches directory when the device is low on storage. Your 30 MB photo enhancement model is the largest file in Caches and gets evicted first:

  (1) **iOS storage pressure** — when an iPhone with 64 GB storage drops below ~5 GB free, iOS begins purging app caches. The system targets the largest files first. Your 30 MB model is evicted while the 8 MB and 12 MB models survive. The user opens photo enhancement → model is gone → re-download → "takes forever."

  (2) **Tiered storage strategy** — not all models deserve the same storage treatment. Classify models by usage frequency and download cost:
  - **Hot models** (used daily: smart reply, sticker) — store in Application Support directory (not purgeable). Total: 20 MB. Acceptable permanent cost.
  - **Warm models** (used weekly: photo enhancement) — store in Caches but implement a pre-fetch strategy. When the user opens the photo tab, check if the model exists. If not, begin downloading immediately with a progress indicator. Cache a compressed version (gzip: 30 MB → 18 MB on disk) to reduce eviction priority.
  - **Cold models** (used monthly) — always download on-demand. Don't cache.

  (3) **Predictive pre-loading** — track usage patterns. If the user typically uses photo enhancement on weekends, pre-download the model Friday evening during charging + WiFi. Use `BGProcessingTaskRequest` (iOS) for this.

  (4) **Partial model caching** — for the 30 MB model, cache only the shared backbone (15 MB, also used by other features) permanently. Download only the task-specific head (15 MB) on-demand. Head download: 15 MB / 2.5 MB/s (LTE) = 6 seconds vs 12 seconds for the full model.

  > **Napkin Math:** iPhone 64 GB, user has 4 GB free. iOS purge threshold: ~5 GB. Purge targets Caches directory. Your models: 8 + 12 + 30 = 50 MB. After purge: 30 MB model evicted first (largest). Re-download on LTE (20 Mbps): 30 MB / 2.5 MB/s = 12 seconds. User perception: "feature is broken." With partial caching: 15 MB download = 6 seconds. With predictive pre-load: 0 seconds (already downloaded). Storage cost of permanent models: 20 MB out of 64 GB = 0.03% of device storage.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Device-Free ML Testing Strategy</b> · <code>monitoring</code> <code>deployment</code></summary>

- **Interviewer:** "Your team ships an ML-powered camera feature to 200 million users across iOS and Android. You don't have a device farm — you have 5 test phones on your desk. How do you test the ML pipeline across devices without a device farm?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Test on our 5 phones and rely on crash reports from production." This means your 200 million users are your test farm. You'll discover bugs from 1-star reviews.

  **Realistic Solution:** Build a **multi-layer testing pyramid** that catches ML failures before they reach users:

  **(1) Unit tests (no device needed).** Test the model in isolation on your CI server (x86 CPU). Run inference on a golden test set (1000 images with known outputs). Assert: output tensor shape is correct, confidence scores sum to 1.0, known inputs produce expected outputs within tolerance (cosine similarity >0.999 vs reference). This catches model conversion errors, quantization bugs, and preprocessing mismatches. Run on every PR.

  **(2) Integration tests (emulator/simulator).** Run the full inference pipeline (preprocessing → model → postprocessing) on iOS Simulator and Android Emulator. These don't have NPU/GPU delegates, so inference runs on CPU — but they catch: memory leaks (run 10,000 consecutive inferences, assert RSS doesn't grow), threading bugs (run inference from multiple threads simultaneously), lifecycle bugs (simulate backgrounding/foregrounding during inference), and API compatibility (test against multiple OS versions).

  **(3) Cloud device testing (no physical farm needed).** Use Firebase Test Lab (Android, 100+ real devices) or AWS Device Farm (iOS + Android). Run a 5-minute smoke test on 30 representative devices: 10 flagships, 10 mid-range, 10 budget. Assert: inference completes without crash, latency is within tier expectations, output matches reference within tolerance. Cost: ~$150/test run. Run on every release candidate.

  **(4) Canary release (production testing).** Release to 1% of users with enhanced telemetry: per-inference latency, confidence distribution, crash rate, and delegate fallback rate. Monitor for 48 hours. Alert thresholds: crash rate >0.1%, median latency >2× baseline, confidence distribution KL divergence >0.1 vs previous version. If any threshold is breached, halt rollout automatically.

  **(5) Synthetic stress testing.** Generate adversarial inputs programmatically: all-black images, all-white images, 1×1 pixel images, maximum-resolution images, images with extreme aspect ratios, corrupted JPEG headers. Assert the model returns gracefully (empty output or error code) instead of crashing or hanging. Run in CI.

  > **Napkin Math:** Testing pyramid coverage: Unit tests catch ~60% of bugs (model conversion, preprocessing). Integration tests catch ~20% (memory, threading, lifecycle). Cloud device tests catch ~15% (hardware-specific failures, delegate bugs). Canary catches ~4% (real-world edge cases). Remaining ~1%: discovered in production. Cost: Unit + integration: $0 (CI time). Cloud devices: $150 × 4 releases/month = $600/month. Canary: $0 (uses production infrastructure). Total: **$600/month** to test across 100+ devices without owning any.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>


#### 🟡 L5

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Cross-Version Compatibility Maze</b> · <code>deployment</code></summary>

- **Interviewer:** "Your app supports iOS 16-18 and Android 10-15. You ship a Core ML 7 model (iOS 18 feature: stateful KV-cache for your on-device LLM). Users on iOS 16 crash on launch. Your Android TFLite model uses INT4 quantization, but devices running Android 10 with TFLite 2.11 don't support INT4. How do you ship one app that works across all these OS versions?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

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

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Mobile A/B Testing Infrastructure at Scale</b> · <code>monitoring</code> <code>deployment</code></summary>

- **Interviewer:** "You run 12 concurrent ML model experiments across your app — different architectures, quantization levels, and preprocessing pipelines. Each experiment needs statistically significant results within 2 weeks. You have 2M DAU. But your experiments keep producing contradictory results — model A beats model B in experiment 1, but model B beats A in experiment 3. Your data scientist says the experiments are 'interfering.' What's happening and how do you fix it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run each experiment on a separate user cohort." With 12 experiments and 2M DAU, each cohort gets ~167K users. For experiments measuring small effects (1-2% improvement), you need 50K+ per variant — so each experiment can only have 3 variants. This severely limits your experimentation velocity.

  **Realistic Solution:** The interference comes from **overlapping experiments sharing system resources** on the same device:

  (1) **Resource contention** — experiment 1 tests a larger model (50 MB, 15 ms inference). Experiment 3 tests a different preprocessing pipeline. When both run on the same device, the larger model from experiment 1 evicts experiment 3's cached tensors from the Qualcomm Hexagon NPU's L2 cache (1 MB on Snapdragon 8 Gen 2). Experiment 3's latency increases by 40%, making model B appear worse — but the regression is caused by experiment 1, not model B.

  (2) **Layered experiment design** — partition experiments into non-interfering layers. Layer 1: model architecture experiments (mutually exclusive). Layer 2: preprocessing experiments (mutually exclusive within layer, but can overlap with Layer 1 if they don't share hardware resources). Layer 3: UI/UX experiments (no ML resource contention). Each user participates in one experiment per layer.

  (3) **Hardware-aware experiment isolation** — before assigning a user to an experiment, check the device's thermal state and available memory. If the device is already thermally throttled (common on budget phones running multiple experiments), exclude it from latency-sensitive experiments. Log the device's hardware state as a covariate in your analysis.

  (4) **Interaction testing** — for experiments that *must* overlap, use a factorial design. Instead of 12 independent experiments, identify the 3-4 that might interact and run a 2×2 factorial. This explicitly measures interaction effects at the cost of 4× the sample size for those experiments.

  > **Napkin Math:** 2M DAU, 12 experiments. Naive independent: 167K per experiment, 83K per variant (A/B). Minimum detectable effect (MDE) at 80% power: δ = √(16p(1-p)/n) = √(16 × 0.5 × 0.5 / 83,000) = 0.55%. Layered (3 layers, 4 experiments each): 2M per layer, 500K per experiment, 250K per variant. MDE = 0.20%. Experiment duration: 14 days. Thermal throttling prevalence: ~15% of budget Android devices during daytime. Excluding throttled devices: lose 15% sample but eliminate hardware confound.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Performance Regression Detective</b> · <code>monitoring</code></summary>

- **Interviewer:** "Your latest app update didn't change the ML model, but inference latency on iPhone 14 Pro increased from 12 ms to 28 ms — a 2.3× regression. The model binary is identical (same SHA-256). No code changes touched the inference path. Your team is baffled. Find the regression."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "If the model didn't change and the inference code didn't change, the latency can't have changed." This assumes the model and code are the only variables. On mobile, the *runtime environment* is a massive variable.

  **Realistic Solution:** The regression is in the **build configuration or framework version**, not the model or inference code:

  (1) **Xcode version change** — check the build logs. If the team updated from Xcode 15.2 to 15.4, the bundled Core ML compiler (`coremltools` backend) may have changed how it compiles the `.mlmodel` to `.mlmodelc`. The compiled model is different even though the source model is identical. The new compiler may have disabled an ANE operator fusion that was combining three layers into one ANE instruction.

  (2) **Compiler optimization flags** — a seemingly unrelated build setting change (e.g., switching from `-O2` to `-Os` for size optimization, or enabling Link-Time Optimization) can change how the Core ML runtime's C++ code is compiled. LTO can inline functions differently, changing memory access patterns and cache behavior.

  (3) **Delegate routing change** — Core ML dynamically decides whether to run each layer on the ANE, GPU, or CPU. A new iOS point release (e.g., 17.3 → 17.4) may have changed the routing heuristics. Layers that previously ran on the ANE (2 ms) now route to the GPU (8 ms) because Apple's profiler determined the GPU is more energy-efficient for that layer shape.

  (4) **Diagnosis** — use Instruments → Core ML Performance template. Compare the per-layer execution plan between the fast and slow builds. Look for layers that moved from ANE to GPU/CPU. Check `MLComputePlan` (iOS 17+) to see the planned execution device for each operation.

  (5) **Fix** — pin the Core ML compilation step to a specific `coremltools` version in your CI pipeline. Add a latency regression test: run inference on a reference device in CI and fail the build if P95 latency exceeds 1.5× the baseline.

  > **Napkin Math:** ANE inference: 12 ms (all layers on ANE). After delegate rerouting: 3 layers moved to GPU. ANE layers: 8 ms. GPU layers: 20 ms (but GPU and ANE can partially overlap). Total: 28 ms. The 16 ms regression is entirely from 3 layers moving off the ANE. Fix: force ANE execution via `MLModelConfiguration.computeUnits = .cpuAndNeuralEngine` (exclude GPU). Or: file a radar with Apple and pin to the previous iOS version's routing behavior.

  📖 **Deep Dive:** [Volume I: ML Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Production Mobile ML Observability Stack</b> · <code>monitoring</code></summary>

- **Interviewer:** "You're the ML platform lead for a ride-sharing app with 10M DAU. On-device ML powers: ETA prediction, route optimization, driver face verification, and fraud detection. Each model runs on wildly different hardware — from iPhone 15 Pro (ANE, 8 GB RAM) to Samsung Galaxy A14 (Mali GPU, 4 GB RAM) to Xiaomi Redmi Note 12 (Adreno 610, 4 GB RAM). Design the observability system that tells you, within 5 minutes, if any model is degrading on any device segment."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Collect latency and accuracy metrics from all devices and build a dashboard." At 10M DAU × 4 models × ~50 inferences/day = 2B inference events/day. Even at 100 bytes per event, that's 200 GB/day of telemetry. Your data pipeline will collapse and your bill will be astronomical.

  **Realistic Solution:** Build a **hierarchical observability system** with on-device intelligence:

  (1) **On-device health scoring** — each device computes a per-model health score locally every hour. The score combines: latency ratio (current P95 / baseline P95 for this device class), confidence entropy (high entropy = model uncertainty increasing), error rate (silent failures from the output validation layer), and thermal impact (inference-induced temperature rise). Score range: 0-100. Healthy = 80+. Degraded = 50-80. Critical = <50.

  (2) **Adaptive reporting** — healthy devices report a 200-byte summary once per day. Degraded devices report every hour with per-model breakdowns. Critical devices report immediately with full diagnostic payloads (input/output samples, delegate routing, thermal logs). This creates a natural funnel: 10M devices → 9.5M daily summaries + 400K hourly reports + 100K critical alerts.

  (3) **Device segment indexing** — pre-define device segments by SoC family (A17 Pro, Snapdragon 8 Gen 3, Exynos 2400, Dimensity 9300, Tensor G3), RAM tier (4 GB, 6 GB, 8 GB+), and OS version. Each segment has a baseline health score computed from the first week of deployment. Alert when any segment's rolling 1-hour average drops below 2 standard deviations from its baseline.

  (4) **Anomaly detection pipeline** — server-side, run a lightweight anomaly detector on the incoming health scores. Group by (model_version, device_segment, os_version). If the ETA model's health score drops on Snapdragon 8 Gen 2 + Android 14 but not on any other segment, alert the on-call engineer with the specific segment, the degradation magnitude, and the top 10 diagnostic payloads from critical devices in that segment.

  (5) **5-minute detection SLA** — critical alerts arrive in real-time (within seconds). The anomaly detector runs on 1-minute sliding windows. Detection latency: 1 minute (window) + 2 minutes (statistical significance with 100+ data points from the affected segment) + 1 minute (alert routing) = 4 minutes. Under the 5-minute SLA.

  > **Napkin Math:** Telemetry volume: 9.5M × 200 bytes/day = 1.9 GB/day (daily summaries). 400K × 500 bytes × 24/day = 4.8 GB/day (hourly). 100K × 5 KB = 500 MB/day (critical). Total: ~7.2 GB/day. At $0.10/GB ingestion (Datadog): $0.72/day = $22/month. vs naive approach: 200 GB/day = $600/month. Cost reduction: 28×. Device segments: ~50 SoC families × 3 RAM tiers × 5 OS versions = 750 segments. At 10M DAU: ~13,300 devices per segment — sufficient for statistical significance within minutes.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>


#### 🔴 L6+

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Delivery Paradox</b> · <code>deployment</code></summary>

- **Interviewer:** "You're shipping an on-device LLM assistant (3B parameters, INT4) to a global Android app with 500 million installs. The quantized model is 1.7 GB. Google Play's install-time APK limit is 150 MB, and on-demand delivery via Play Asset Delivery has a 2 GB limit. Design the model delivery system. What breaks at this scale that doesn't break in a demo?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just use on-demand download after install — it's under 2 GB." This works for a prototype but collapses at 500M-device scale.

  **Realistic Solution:** At 500M installs, every decision multiplies by 500M. The system design must address five interacting constraints:

  **1. Bandwidth economics:** 500M devices × 1.7 GB = 850 PB of transfer. At $0.01/GB CDN cost, that's $8.5M per model version. You need delta updates — ship weight diffs, not full models. A well-structured INT4 checkpoint with grouped quantization typically yields ~60% compression with brotli, and delta updates between versions compress to ~15% of full size.

  **2. Storage heterogeneity:** 500M Android devices span 32 GB budget phones to 512 GB flagships. You can't assume 1.7 GB of free space. Design a tiered model system: a 200 MB "core" model that ships with the APK (handles basic queries), a 600 MB "standard" model downloaded on WiFi, and the full 1.7 GB "premium" model for flagships. Use Android's `StorageStatsManager` to pick the tier at runtime.

  **3. Model versioning and rollback:** With 500M devices, you can't push a broken model and "just roll back." Use content-addressed storage (hash the model weights) and keep the previous version on-device until the new one passes a local validation suite (canary inference on reference inputs). A/B test new models on 1% of devices for 48 hours before full rollout.

  **4. Integrity and security:** The model is intellectual property sitting on user devices. Sign model chunks with Ed25519, verify at load time. Use Android's `EncryptedFile` API for at-rest encryption. Chunk the model into 50 MB segments for resumable download — mobile connections drop constantly.

  **5. Cold start after download:** The user downloads 1.7 GB and expects instant use. But the first inference requires loading the model into memory and compiling the NPU graph. Pre-compile the QNN/TFLite graph during download (background `WorkManager` task) and memory-map the weights so first inference doesn't require a full 1.7 GB read.

  > **Napkin Math:** CDN cost per version: 500M × 1.7 GB × $0.01/GB = **$8.5M**. With brotli compression (60%): $3.4M. With delta updates (15% of full): **$510K** — a 17× cost reduction. Storage: median Android device has ~15 GB free. 1.7 GB = 11% of free space — unacceptable for budget phones. Tiered delivery is mandatory.

  📖 **Deep Dive:** [Volume I: ML Operations](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_ops/ml_ops.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Multi-Model Orchestration Nightmare</b> · <code>serving</code></summary>

- **Interviewer:** "Your team is building a sophisticated AR application that requires running three ML models concurrently: a body pose estimation (high-res video input, GPU/NPU), a small speech recognition (audio input, CPU), and a gesture classifier (low-res video region, NPU). All must run in real-time. Describe how you would design the inference pipeline to manage resource contention, prioritize tasks, and ensure real-time performance on a single mobile SoC."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just run them all in parallel on different threads." This overlooks shared resources, scheduling, and power management.

  **Realistic Solution:** Designing such a pipeline requires sophisticated orchestration and resource management:
  1.  **Task Prioritization & Scheduling:**
    *   **Prioritize Critical Paths:** Identify the most latency-sensitive model (e.g., pose estimation for visual feedback). Assign higher OS thread priorities for its pre/post-processing and accelerator dispatch threads.
    *   **Asynchronous Execution:** All models should ideally run asynchronously. Use separate threads for each model's pre-processing, inference submission, and post-processing.
    *   **OS Scheduler Hints:** Utilize Android's `setThreadPerfPolicy` or `setThreadScheduler` (if available/permitted) to give hints to the OS scheduler for critical threads.
  2.  **Accelerator Allocation & Contention Management:**
    *   **Dedicated Accelerators (where possible):** Speech recognition is CPU-bound; it should run on dedicated "big" CPU cores. Body pose (large CNN) and gesture (small CNN) should ideally target the NPU.
    *   **NPU Time Slicing:** If two NPU-bound models (pose and gesture) run concurrently, the NPU driver will time-slice them. This introduces latency. Consider:
        *   **Sequential Execution:** If the gesture model is very small, run it sequentially *after* pose estimation if its total latency fits the budget.
        *   **Dynamic Prioritization:** If one model is more critical at a given moment, dynamically adjust its priority with the NPU driver (if API supports it) or suspend the less critical one.
        *   **Batching (if applicable):** If gesture uses a sub-region of the pose model's output, consider fusing them or batching if input frames align.
    *   **GPU as Fallback/Alternative:** If NPU contention is too high, one model (e.g., gesture classifier) could be delegated to the GPU, leveraging its independent execution units, but this might increase power consumption.
  3.  **Memory Bandwidth Management:**
    *   **Minimize Data Copies:** Reduce transfers between CPU, GPU, and NPU. Use shared memory buffers (e.g., `AHardwareBuffer` on Android) when passing tensors between components.
    *   **Data Locality:** Design models to access data efficiently from caches.
    *   **Pre-fetching:** Proactively load model weights or upcoming input data into memory if possible.
  4.  **Power & Thermal Management:**
    *   **DVFS Awareness:** Monitor SoC temperature and clock frequencies. If throttling occurs, dynamically reduce model complexity (e.g., lower resolution, smaller model variant) or drop frames for less critical models.
    *   **Power Gating:** Ensure accelerators are power-gated when idle.
  5.  **Synchronization & Communication:** Use efficient inter-thread communication mechanisms (e.g., lock-free queues, condition variables) to pass data between pipeline stages.

  > **Napkin Math:** Body pose (200ms latency budget): NPU inference 30ms, pre/post-processing 20ms. Speech recognition (100ms latency budget): CPU inference 40ms. Gesture (50ms latency budget): NPU inference 10ms. If pose and gesture both hit the NPU, their combined NPU time might be 40ms, but context switching and driver overhead could push it to 60-80ms. If the speech model uses 2 "big" CPU cores, it leaves fewer for pre/post-processing of other models. The shared LPDDR5 memory (e.g., 60 GB/s) could be saturated if pose (e.g., 20 GB/s) and speech (e.g., 5 GB/s) and gesture (e.g., 5 GB/s) are all active, plus OS overhead.
  > **Key Equation:** $Pipeline\;Throughput = \min(\frac{1}{Latency_{stage_1}}, \frac{1}{Latency_{stage_2}}, ...)$

  📖 **Deep Dive:** [Volume I: Chapter 6.3 Real-time Inference Systems](https://mlsysbook.ai/vol1/systems/real-time.html)

  </details>

</details>


---


### Privacy & Security


#### 🟢 L1/L2

#### 🟢 L3
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The User Consent Minefield</b> · <code>privacy</code></summary>

- **Interviewer:** "Your fitness app wants to use on-device federated learning to improve its activity recognition model. The model trains locally on the user's accelerometer data and only sends gradient updates to the server. Your legal team says you still need explicit user consent under GDPR. Your PM argues 'the data never leaves the device, so GDPR doesn't apply.' Who's right?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Data stays on-device, so no consent needed." GDPR applies to *processing* of personal data, not just *transmission*. On-device training is processing. Gradient updates, even noised, are derived from personal data and constitute personal data under GDPR Article 4(1).

  **Realistic Solution:** The PM is wrong. You need consent, and the implementation has hardware implications:

  (1) **Consent architecture** — present a clear opt-in dialog (not buried in ToS) explaining: what data is used (accelerometer patterns), what processing occurs (local model training), what leaves the device (aggregated gradient updates), and how to opt out. Store consent state in the Keychain (iOS) or EncryptedSharedPreferences (Android) — not in plain UserDefaults, which can be read by device backups.

  (2) **Consent-gated training** — the on-device training pipeline checks consent state before every training round. If consent is revoked mid-round, abort immediately and delete the local gradient. The training scheduler must be interruptible — don't use a non-cancellable background task.

  (3) **Data minimization** — even with consent, GDPR requires you to minimize data processing. Only train on the last 7 days of accelerometer data. Auto-delete training data older than 7 days. Log the deletion for audit purposes.

  (4) **Right to erasure** — if a user requests data deletion (GDPR Article 17), you must delete their local training data AND ensure their gradient contributions are removed from the global model. In practice, this means retraining the global model excluding their contributions — or using machine unlearning techniques to approximate removal.

  (5) **Hardware cost of consent** — consent checking adds ~2 ms per training round (Keychain/Keystore read). Consent UI adds ~50 KB to the app binary. The real cost is opt-in rate: expect 30-40% opt-in for health data, meaning your federated learning pool is 60-70% smaller than your user base.

  > **Napkin Math:** User base: 500K DAU. Opt-in rate (health data): 35% = 175K participating devices. Minimum for federated learning convergence: 10,000 devices per round at 1% sampling = need 1M eligible. With 175K eligible: 1% sampling = 1,750 per round. Convergence: possible but 5-6× slower than with full participation. Gradient noise at 1,750 devices: σ/√1,750 = 0.024 (acceptable). GDPR fine for non-compliance: up to 4% of global revenue or €20M.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The App Store Privacy Rejection</b> · <code>privacy</code> <code>security</code></summary>

- **Interviewer:** "Your camera app uses an on-device face detection model to apply AR filters. You don't send any images to a server — everything runs locally on the Apple A18 Pro. Apple rejects your app update, citing 'Insufficient purpose string for camera and face data usage.' How does on-device ML face processing create a unique privacy challenge vs server-side, and why does Apple's privacy framework struggle to distinguish 'processed locally and discarded' from 'processed and exfiltrated'?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We process everything on-device, so privacy rules don't apply." This assumes the OS can perfectly audit what your model does with the data after inference.

  **Realistic Solution:** On-device ML creates a fundamental tension: the app has direct access to raw biometric data (the camera feed and the resulting 3D face mesh), but the OS cannot mathematically prove that the app *deletes* that data after applying the AR filter. If you send data to a server, the network request is obvious. But on-device, the ML model generates a face mesh tensor in RAM. The OS doesn't know if that tensor is used solely for rendering the next frame, or if it's being serialized and written to a hidden file for later exfiltration.

  Because the OS cannot enforce "process and discard" at the memory level, Apple's privacy framework requires explicit *declarations* of processing, regardless of transmission. You must update your `PrivacyInfo.xcprivacy` manifest to declare face data processing, and your `NSCameraUsageDescription` must explicitly state that the camera is used for ML face tracking, even if local. The ML architecture (local inference) provides the *actual* privacy, but you still must satisfy the platform's *declarative* privacy requirements.

  > **Napkin Math:** App Store review rejection turnaround: 24-48 hours per submission. Typical fix cycle: 2-3 resubmissions = 4-9 days delay. Revenue impact for an app with 50K DAU at $0.30 ARPDAU: $15K/day × 6 days = $90K lost. Prevention cost: 2 hours of engineering to add privacy manifest entries. The architectural choice to run ML on-device saves ~$5,000/month in server GPU costs, but requires navigating strict client-side biometric compliance.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>


#### 🔵 L4
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Federated Keyboard</b> · <code>privacy</code></summary>

- **Interviewer:** "You're building a next-word prediction model for a mobile keyboard. The model must improve from user typing patterns, but you cannot collect user keystrokes on your servers — that's a privacy violation and a regulatory risk. How do you train the model?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Anonymize the data before uploading." Anonymization is insufficient — keystroke patterns can re-identify users, and regulators (GDPR, CCPA) consider this personal data regardless of anonymization.

  **Realistic Solution:** **Federated Learning with Differential Privacy (DP-FedAvg):**

  (1) **Local training:** Each phone fine-tunes a local copy of the model on the user's recent typing data. Training happens on-device during charging (to avoid battery drain). The phone computes a gradient update (the difference between the local model and the global model).

  (2) **Gradient clipping:** Before sending anything, clip the gradient to a maximum L2 norm $C$. This bounds the influence of any single user's data on the global model.

  (3) **Noise injection:** Add calibrated Gaussian noise $\mathcal{N}(0, \sigma^2 C^2 I)$ to the clipped gradient. This provides differential privacy — mathematically guaranteeing that the server cannot determine whether any specific user participated in training.

  (4) **Secure aggregation:** The server collects noised gradients from thousands of phones and averages them. Individual gradients are encrypted so the server only sees the aggregate. The noise cancels out in the average (law of large numbers), but protects individual contributions.

  (5) **Privacy budget:** The privacy guarantee is measured by $\epsilon$ (epsilon). Lower $\epsilon$ = stronger privacy but more noise = slower learning. Typical production values: $\epsilon = 8$ per training round, with a total budget of $\epsilon = 100$ per year. At these levels, accuracy loss vs non-private training is ~2%.

  > **Napkin Math:** 10,000 phones per round. Each sends a 5 MB gradient update (clipped + noised). Server bandwidth: 50 GB per round. Noise per phone: σ = 1.0, C = 1.0. After averaging 10,000 updates: effective noise = σ/√10,000 = 0.01 — negligible. Privacy: ε = 2 per round (strong). 50 rounds/year: total ε = 100 (within budget). Accuracy: ~2% worse than centralized training, but zero user data leaves the device.

  > **Key Equation:** $\tilde{g}_{\text{user}} = \text{clip}(g, C) + \mathcal{N}(0, \sigma^2 C^2 I)$

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The On-Device Personalization Privacy</b> · <code>privacy</code> <code>data-parallelism</code></summary>

- **Interviewer:** "Your photo search app needs to learn each user's personal categories — 'my dog,' 'my office,' 'my kids' soccer games.' You'll fine-tune a CLIP-like vision encoder (150M params, FP16 = 300 MB) on-device using LoRA. The adapted weights never leave the device. Walk me through the LoRA adapter sizing, training memory, time to convergence, and what happens when the user gets a new phone."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "LoRA is tiny, so training is free." LoRA reduces trainable parameters dramatically, but training still requires: forward pass through the full frozen model (300 MB resident), backward pass through LoRA layers (storing activations for all layers), and an optimizer state (Adam stores 2× the trainable parameters). The memory cost is dominated by the frozen model and activations, not the LoRA weights.

  **Realistic Solution:** Design the full on-device training pipeline with realistic resource accounting:

  **(1) LoRA adapter sizing.** CLIP vision encoder: 12 transformer layers, each with Q/K/V/O projections (768 × 768). LoRA at rank 4: each adapter = 2 matrices of (768 × 4) and (4 × 768). Per projection: 768 × 4 + 4 × 768 = 6,144 parameters. 4 projections × 12 layers = 48 adapters × 6,144 = **294,912 trainable parameters**. Storage: 295K × 2 bytes (FP16) = **590 KB**. Negligible.

  **(2) Training memory budget.** Frozen model weights (mmap'd, ~200 MB resident): 200 MB. LoRA trainable weights: 590 KB. Activations for backward pass (stored per layer): 12 layers × batch_size × seq_len × hidden_dim × 2 bytes. For batch=1, 50 image patches (ViT): 12 × 1 × 50 × 768 × 2 = 921 KB. Adam optimizer state (m and v for each trainable param): 295K × 2 × 2 bytes = 1.18 MB. Gradients: 590 KB. **Total training memory: ~203 MB**. Fits on any phone with 4+ GB RAM.

  **(3) Training data and convergence.** Source: user's photo library with implicit labels (album names, faces, locations, user search corrections). Typical user: 5,000 photos, 20-50 personal categories. Training set: ~500 (image, text) pairs generated from metadata. Training: 50 epochs × 500 pairs / batch_size 8 = 3,125 gradient steps. Per step on ANE: forward (12ms) + backward (18ms) = 30ms. Total training time: 3,125 × 30ms = **93.75 seconds ≈ 1.5 minutes**. Power: 2W × 94s = 188J = 0.4% of a 12.8 Wh battery. Schedule during overnight charging.

  **(4) New phone migration.** The LoRA adapter (590 KB) is backed up to iCloud/Google Drive as part of app data backup. On the new phone: download the base model (same version from the app), apply the 590 KB adapter. Instant personalization without retraining. If the base model version changes (app update with new model): the old LoRA adapter is incompatible. Trigger a retraining cycle on the new phone — 1.5 minutes, invisible to the user during first overnight charge.

  **(5) Privacy analysis.** The 590 KB adapter encodes correlations between visual features and text labels. Without the base model and the user's photos, the adapter is meaningless — it's a set of low-rank perturbations to attention projections. Even with the base model, reconstructing training images from LoRA weights is computationally infeasible (the adapter compresses 500 images into 295K parameters — a 10,000× information bottleneck). Differential privacy isn't needed because the weights never leave the device.

  > **Napkin Math:** LoRA params: 295K (0.2% of 150M). Adapter size: 590 KB. Training memory: 203 MB. Training time: 94 seconds. Battery: 0.4%. Migration: 590 KB backup. Personalization gain: +15% recall on user-specific categories (internal benchmark). Retraining frequency: weekly or on 50+ new photos. Annual training cost per user: 52 weeks × 188J = 9.8 kJ = 0.27% of annual phone energy budget (assuming 10 Wh/day usage).

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>


#### 🟡 L5

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Keyboard Prediction Privacy Leak</b> · <code>privacy</code> <code>deployment</code></summary>

- **Interviewer:** "Your on-device keyboard prediction model (LSTM, 20M params) on a Snapdragon 8 Gen 3 learns from user typing to improve suggestions. A security researcher demonstrates that by observing the model's top-5 prediction probabilities over 1000 queries, they can reconstruct fragments of the user's private text — including passwords typed into non-password fields and private messages. The model never leaves the device. How is private data leaking through predictions?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model is on-device, so data can't leak." On-device doesn't mean private. The model's *outputs* (predictions) are observable by any app using the keyboard, and those outputs encode information about the training data.

  **Realistic Solution:** This is a **model inversion / membership inference attack** on a personalized on-device model:

  (1) **Memorization in small models** — a 20M parameter LSTM trained on one user's data (relatively small corpus: ~50K unique sentences) memorizes specific sequences. If the user types "my password is" frequently (e.g., in notes, messages), the model learns to predict the actual password with high confidence. Any app that reads the keyboard's suggestion bar can observe this.

  (2) **Prediction probability leakage** — the top-5 predictions with confidence scores reveal the training distribution. If "meeting with [CEO_NAME]" appears with 0.85 confidence, an attacker knows the user frequently types that CEO's name — leaking business relationships.

  (3) **Gradient leakage in federated learning** — if the keyboard participates in federated learning, the gradient updates sent to the server can be inverted to reconstruct training examples. A 20M parameter gradient update contains enough information to reconstruct ~100 training sentences with high fidelity.

  **Mitigations:**

  (1) **Differential privacy (DP) in training** — add calibrated Gaussian noise to gradients during on-device training. With ε = 8 (moderate privacy): prediction quality drops ~3%, but memorization of specific sequences drops >90%. The model learns general patterns ("meeting with [PERSON]") but not specific names.

  (2) **Prediction filtering** — never suggest tokens that appear fewer than 5 times in the user's history. This prevents one-off sensitive entries (passwords, SSNs) from appearing in suggestions. Maintain a frequency counter per token.

  (3) **Sensitive field detection** — detect password fields (`isSecureTextEntry`), credit card fields, and SSN fields via input type attributes. Disable personalized predictions in these fields — use only the base (non-personalized) model.

  (4) **Temporal decay** — exponentially decay the influence of old training data. Tokens typed >30 days ago have their contribution reduced by 90%. This limits the window of vulnerability.

  > **Napkin Math:** LSTM with 20M params, trained on 50K sentences. Memorization capacity: a 20M param model can memorize ~200K tokens verbatim (1 param ≈ 10 bits, 200K tokens × 100 bits/token = 20M × 10 bits). User types ~10K words/day × 90 days = 900K tokens. The model can memorize ~22% of the user's typing history verbatim. With DP (ε = 8): effective memorization drops to <2%. Prediction quality (perplexity): base model 45, personalized 32, personalized + DP 35. The 3-point perplexity cost buys 10× privacy improvement.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The On-Device Differential Privacy Budget</b> · <code>privacy</code></summary>

- **Interviewer:** "Your health app uses on-device ML to classify sleep stages from Apple Watch accelerometer data. You want to improve the model using federated learning, but your privacy team says you need formal differential privacy guarantees — HIPAA requires it for health data. Your privacy budget is ε = 1 per user per year. The model needs 100 federated rounds to converge. How do you make the math work?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Set ε = 1 per round. After 100 rounds, total ε = 100." This violates the privacy budget by 100×. Differential privacy composes — each round consumes part of the budget, and the total must stay under ε = 1.

  **Realistic Solution:** Use **privacy amplification** techniques to stretch the budget:

  (1) **Subsampling amplification** — in each round, randomly select only q = 1% of eligible devices. By the privacy amplification theorem, the effective per-round ε drops from ε₀ to approximately 2q·ε₀. If you need total ε = 1 over 100 rounds using Rényi DP composition: per-round ε₀ ≈ 0.5, but with q = 0.01 subsampling, effective per-round ε ≈ 2 × 0.01 × 0.5 = 0.01. After 100 rounds with advanced composition: total ε ≈ √(2 × 100 × ln(1/δ)) × 0.01 + 100 × 0.01² ≈ 0.3 (for δ = 10⁻⁶). Well under budget.

  (2) **Gradient clipping** — clip each device's gradient update to L2 norm C = 0.1. This bounds sensitivity. Combined with Gaussian noise σ = C × √(2 ln(1.25/δ)) / ε₀, each device's contribution is provably private.

  (3) **Secure aggregation** — Apple's on-device framework (or your custom implementation) ensures the server only sees the aggregate of ≥1000 devices' noised gradients. Individual gradients are never observable, providing an additional layer beyond DP.

  (4) **Privacy accounting** — use the Rényi DP accountant (not naive composition) to track the running privacy cost. Stop training when the accountant reaches ε = 0.9, leaving a 0.1 buffer for any future model updates that year.

  (5) **Convergence under noise** — with 1M eligible Apple Watch users, 1% subsampling = 10,000 devices per round. The noise per device is high (σ = 1.0), but after averaging 10,000 updates, effective noise = σ/√10,000 = 0.01 — the model converges, just 2-3× slower than non-private training.

  > **Napkin Math:** Privacy budget: ε = 1/year. Rounds: 100. Subsampling rate: q = 1%. Per-round noise multiplier: σ = 1.0. Devices per round: 10,000. Gradient size: 500 KB (sleep stage classifier). Upload per round: 10,000 × 500 KB = 5 GB. Total training bandwidth: 100 × 5 GB = 500 GB. Convergence: ~100 rounds × 2.5× slowdown = equivalent to 250 non-private rounds. Final accuracy: ~2% below non-private baseline. Privacy guarantee: ε = 0.85 (under budget).

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Keyboard Prediction Privacy Leak</b> · <code>privacy</code> <code>security</code></summary>

- **Interviewer:** "Your mobile keyboard app uses an on-device next-word prediction model that personalizes to each user's typing patterns. A security researcher publishes a paper showing they can extract a user's private information (email addresses, passwords typed into non-password fields, medical terms) by querying your model's prediction API 10,000 times with crafted prefixes. The model runs entirely on-device — no data leaves the phone. How is this possible, and how do you fix it without destroying prediction quality?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model is on-device, so there's no privacy risk." On-device doesn't mean private. If the model *memorizes* training data, anyone with access to the model (or its API) can extract that data — including malicious apps using accessibility services or a stolen/seized device.

  **Realistic Solution:** This is a *memorization attack* against the personalized model:

  (1) **Attack mechanism** — the personalized model fine-tunes on the user's typing history. If the user typed "my password is hunter2" in a notes app, the model learns this sequence. The attacker queries the prediction API with the prefix "my password is" and the model confidently predicts "hunter2." By systematically querying with common prefixes ("my SSN is", "my address is", "dear Dr."), the attacker extracts memorized private data. This works even on-device if a malicious app can invoke the keyboard's prediction API programmatically.

  (2) **Fix: differential privacy during personalization** — add calibrated noise to the gradient updates during on-device fine-tuning. Use DP-SGD with per-example gradient clipping (C = 1.0) and Gaussian noise (σ = 0.8). This provides (ε = 4, δ = 10⁻⁵) differential privacy per training epoch. The model learns general typing patterns but cannot memorize specific sequences. Prediction quality drops ~3% (measured by keystroke savings rate).

  (3) **Fix: output filtering** — before returning predictions, check if the predicted sequence matches a sensitive pattern (email regex, phone number format, high-entropy strings that look like passwords). Suppress these predictions. This is a heuristic defense — not mathematically guaranteed like DP — but catches the most damaging leaks.

  (4) **Fix: training data curation** — exclude text typed into password fields (already standard), but also exclude: text from banking/health apps (detected via app package name), text matching PII patterns (SSN, credit card regex), and any text the user explicitly deletes within 5 seconds (likely a mistake or sensitive content).

  (5) **Fix: model capacity limiting** — use a small personalization layer (64-dim embedding, ~50 KB) on top of a frozen base model. The small capacity physically limits how much user data can be memorized. Information-theoretic bound: a 50 KB model can memorize at most 50 KB of user data — roughly 50,000 characters. With DP noise, effective memorization drops to <1 KB.

  > **Napkin Math:** Personalized model vocabulary: 30,000 tokens. Attacker queries: 10,000 prefixes × 10 predictions each = 100,000 queries. Without DP: ~200 unique private sequences extractable (email, addresses, names). With DP (ε = 4): extractable sequences drop to ~3 (and with low confidence). Prediction quality: baseline keystroke savings rate 45%. With DP: 42% (-3%). With output filtering: 44.5% (-0.5%, only suppresses sensitive completions). Combined: 41.5%. User-perceptible difference: minimal — users don't notice 3% fewer correct predictions.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>


#### 🔴 L6+

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Federated Learning System for a Social Media App</b> · <code>privacy</code> <code>data-parallelism</code></summary>

- **Interviewer:** "Design a federated learning system for a social media app with 500 million daily active users. The model personalizes the content feed. You must improve the model without collecting user interaction data on your servers. Cover device selection, communication efficiency, convergence, and privacy guarantees."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Run FedAvg across all 500M devices." Naive FedAvg at this scale would require petabytes of gradient uploads per round and wouldn't converge due to extreme data heterogeneity (non-IID user behavior).

  **Realistic Solution:** Design a **hierarchical federated system** with four key subsystems:

  **(1) Device selection and eligibility.** Not all devices can participate. Eligibility criteria: charging, on WiFi, idle (screen off for >5 minutes), sufficient RAM (>2 GB free), OS version supports on-device training (iOS 16+ with Core ML Training, Android 12+ with TFLite training API). At any given moment, ~5% of DAU meets all criteria = 25M eligible devices. Per round, sample 10,000 devices (stratified by region, device tier, and user activity level to ensure representativeness).

  **(2) Communication efficiency.** A feed ranking model has ~50M parameters. FP32 gradients: 200 MB per device per round. At 10,000 devices: 2 TB per round. Unacceptable. Compression stack: (a) Top-k sparsification (k=1%) — send only the 1% largest gradient values + their indices. 200 MB → 4 MB. (b) INT8 quantization of gradient values: 4 MB → 2 MB. (c) Error feedback — accumulate the unsent 99% of gradients locally and add them to the next round's update. This preserves convergence despite aggressive sparsification. Total upload per device: **2 MB**. Per round: 20 GB. Manageable.

  **(3) Convergence under non-IID data.** User behavior is extremely heterogeneous — a teenager's feed interactions look nothing like a retiree's. Standard FedAvg diverges under this heterogeneity. Solutions: (a) **FedProx** — add a proximal term that penalizes local models from drifting too far from the global model. (b) **Clustered FL** — group users into behavioral clusters (identified by embedding similarity, computed on-device). Run separate federated models per cluster. 10-20 clusters typically suffice. (c) **Personalization layers** — the base model (shared backbone) is trained federally. The final 2-3 layers are trained locally and never uploaded. This gives each user a personalized model while the shared layers capture global patterns.

  **(4) Privacy guarantees.** Differential privacy with (ε=8, δ=10⁻¹⁰) per round. Gradient clipping norm C=1.0. Gaussian noise σ = C × √(2 ln(1.25/δ)) / ε ≈ 0.7. With 10,000 devices per round, the noise in the aggregate is σ/√10,000 = 0.007 — negligible impact on model quality. Secure aggregation ensures the server never sees individual gradients — only the encrypted sum. Annual privacy budget: 50 rounds × ε=8 = ε=400 (with advanced composition, actual ε ≈ 120 using Rényi DP accounting).

  > **Napkin Math:** 500M DAU. 25M eligible per moment. 10,000 sampled per round. Upload: 2 MB × 10,000 = 20 GB/round. Server aggregation: ~5 minutes on 8 GPUs. Rounds per day: 4 (every 6 hours). Daily bandwidth: 80 GB. Monthly: 2.4 TB. CDN cost: ~$200/month. On-device training time: ~3 minutes per round (50 local steps on 50M-param model). Battery cost per round: 3 min × 2W = 0.1 Wh = 0.8% of a 12.8 Wh battery. Convergence: ~200 rounds (50 days) to match centralized training quality within 1%.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>


---


### On-Device Training & Federated Learning


#### 🔵 L4
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The On-Device Fine-Tuning Pipeline</b> · <code>data-parallelism</code> <code>privacy</code></summary>

- **Interviewer:** "Your photo app has a 'search by description' feature powered by a CLIP-like model. Users in Japan complain it doesn't understand Japanese food categories. Users in Brazil say it misclassifies local bird species. You can't collect user photos on your servers. Design an on-device personalization system that adapts the model to each user's photo library."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Fine-tune the entire model on-device." A CLIP model has 150M+ parameters. Fine-tuning all of them on a phone would take hours and drain the battery.

  **Realistic Solution:** Use **parameter-efficient fine-tuning (PEFT)** on-device:

  **(1) Freeze the backbone, train adapters.** Add LoRA (Low-Rank Adaptation) layers to the model's attention projections. LoRA adds two small matrices (rank 4-8) per attention layer. For a 150M-param model with 12 attention layers: trainable parameters = 12 × 2 × (768 × 4 + 4 × 768) = 73,728 parameters. That's 0.05% of the full model. Training 73K parameters is feasible on-device.

  **(2) Training data from the user's library.** Use the user's existing photo-label associations as training signal: album names ("Tokyo Trip 2025"), faces (contacts), locations (geo-tags), and user corrections (when the user searches for "ramen" and taps a photo the model ranked low). Generate (image, text) training pairs automatically. No manual labeling needed.

  **(3) Training schedule.** Fine-tune during charging + idle, using Core ML's `MLUpdateTask` (iOS) or TFLite's training API (Android). 50 gradient steps on 200 training pairs: ~90 seconds on the A17 Pro Neural Engine. Power: 2W × 90s = 180J = 0.4% battery. Run weekly or when the user adds >50 new photos.

  **(4) Quality safeguard.** After fine-tuning, run the adapted model on a held-out validation set (20 photos the user has previously searched for successfully). If search recall drops >10% vs the base model, discard the adaptation and keep the base model. This prevents catastrophic forgetting.

  **(5) Privacy guarantee.** The LoRA weights (73K × 4 bytes = 294 KB) stay on-device. They encode user-specific knowledge but are meaningless without the base model and the user's photo library. Even if exfiltrated, they reveal minimal information about the user's data.

  > **Napkin Math:** Base model: 150M params, 300 MB (FP16). LoRA adapters: 73K params, 294 KB. Training: 200 pairs × 50 steps = 10,000 forward+backward passes. Time: 90 seconds on NPU. Memory during training: 300 MB (frozen weights, mmap'd) + 294 KB (trainable) + 50 MB (activations + gradients for trainable params only) = ~350 MB. Fits on any phone with 4+ GB RAM. Personalization improvement: +12% recall on user-specific categories (measured in internal testing with Japanese and Brazilian photo libraries).

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>


#### 🟡 L5

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Privacy-Utility Squeeze</b> · <code>continual-learning</code></summary>

- **Interviewer:** "You're implementing federated learning for a next-word prediction model across 100 million Android devices. Each device fine-tunes locally and uploads gradient updates. The privacy team mandates differential privacy with ε = 2. After deployment, the model's perplexity is 40% worse than centralized training. The PM asks you to 'just increase epsilon.' What do you tell them?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "ε = 2 is too strict, we should relax to ε = 8 or 10." This treats ε as a tuning knob without understanding the system-level trade-offs.

  **Realistic Solution:** The problem isn't just ε — it's the interaction between DP noise, client heterogeneity, and the communication budget. With ε = 2, the per-update Gaussian noise scale is large relative to the gradient signal from a single device's small local dataset (maybe a few hundred sentences). But simply increasing ε is a one-way door: once you ship a weaker privacy guarantee, you can't retroactively strengthen it for data already collected.

  The staff-level approach: (1) **Increase the number of clients per round** — DP-SGD noise scales as $\sigma \propto 1/\sqrt{n}$, so aggregating 10,000 clients instead of 1,000 reduces noise by ~3.2× at the same ε. (2) **Use secure aggregation** so the server only sees the sum, allowing tighter per-client clipping without the server learning individual updates. (3) **Reduce the model's update surface** — fine-tune only an adapter (LoRA) with ~1% of parameters, concentrating the gradient signal and reducing the dimensionality that noise must cover. (4) **Increase local epochs** (with careful clipping) to improve signal-to-noise per client.

  > **Napkin Math:** DP-SGD noise: $\sigma = \frac{C \cdot \sqrt{2 \ln(1.25/\delta)}}{\varepsilon \cdot \sqrt{n}}$ where $C$ = clipping norm, $n$ = clients per round. At ε=2, δ=10⁻⁸, C=1.0, n=1000: σ ≈ 0.003. At n=10000: σ ≈ 0.001. 3× less noise, same privacy. Adding LoRA (rank 8 on a 768-dim model) reduces trainable params from 85M to ~600K — the gradient signal is 140× more concentrated.

  > **Key Equation:** $\sigma_\text{noise} \propto \frac{C}{\varepsilon \cdot \sqrt{n}}$

  📖 **Deep Dive:** [Volume I: Responsible AI](https://harvard-edge.github.io/cs249r_book_dev/contents/responsible_engr/responsible_engr.html)

  </details>

</details>


#### 🔴 L6+

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> Designing Cross-Device Federated Personalization</b> · <code>data-parallelism</code> <code>privacy</code></summary>

- **Interviewer:** "You're the ML systems architect for a health platform spanning Apple Watch (S9 chip, 1 GB RAM), iPhone (A18 Pro, 8 GB RAM), and iPad (M4, 16 GB RAM). Each device collects different health signals: the Watch has heart rate and motion, the iPhone has location and app usage, and the iPad has extended workout videos. You want to train a personalized health model that fuses data from all three devices — but Apple's privacy framework prohibits sending raw health data between devices, even within the same user's iCloud account. Design a federated personalization system that trains across a single user's device ecosystem without centralizing raw data."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use standard federated learning across the three devices." Standard FL is designed for many users with similar devices, not one user with heterogeneous devices. The Watch can't run the same model as the iPad — it has 1/16th the RAM.

  **Realistic Solution:** Design a **split federated learning** architecture with device-capability-aware model partitioning:

  (1) **Hierarchical model architecture** — decompose the health model into three sub-models matched to device capabilities: (a) **Watch model** (150 KB, runs on S9's ANE at 1 TOPS): processes heart rate + accelerometer → produces a 64-dim "physiological embedding" every 5 minutes. (b) **Phone model** (8 MB, runs on A18 Pro's ANE at 35 TOPS): takes the Watch's embedding + location + app context → produces a 256-dim "activity embedding" every 15 minutes. (c) **iPad model** (45 MB, runs on M4's ANE at 38 TOPS): takes the Phone's embedding + workout video features → produces health risk predictions and personalized recommendations.

  (2) **Embedding exchange, not data exchange** — devices share only low-dimensional embeddings (64-256 floats = 256 bytes to 1 KB), not raw data. The Watch sends its 64-dim embedding to the iPhone via Bluetooth LE (encrypted, within Apple's device-to-device framework). The iPhone sends its 256-dim embedding to the iPad via local WiFi. Embeddings are computed on-device from raw data that never leaves the originating device. Privacy: embeddings are lossy compressions — you cannot reconstruct heart rate time series from a 64-dim vector.

  (3) **Distributed backpropagation** — to train the system end-to-end: the iPad computes the loss and backpropagates through its model, producing a gradient for the Phone's embedding. This gradient (256 floats = 1 KB) is sent to the Phone. The Phone backpropagates through its model, producing a gradient for the Watch's embedding (64 floats = 256 bytes), sent to the Watch. Each device updates its own model locally. No raw data moves; only gradients of embeddings.

  (4) **Asynchronous training** — devices aren't always co-located. The Watch buffers embeddings locally (64 bytes × 288 per day = 18 KB/day). When the iPhone is in Bluetooth range, it syncs the day's embeddings in one burst (18 KB in <1 second over BLE). Training happens on the iPhone/iPad when all devices have synced and are charging.

  (5) **Privacy guarantees** — add local differential privacy to embeddings before sharing: each device adds Gaussian noise calibrated to ε = 2 per embedding. Even if an attacker intercepts the Bluetooth transmission, they cannot infer specific health events. The noise averages out over the 288 daily embeddings during training.

  > **Napkin Math:** Watch model: 150 KB weights, 0.5 ms inference, 0.1 mW. Phone model: 8 MB, 3 ms inference, 50 mW. iPad model: 45 MB, 15 ms inference, 200 mW. Daily embedding data: Watch → Phone: 18 KB. Phone → iPad: 72 KB. Total cross-device data: 90 KB/day (vs raw data: Watch heart rate at 1 Hz × 86,400 sec × 4 bytes = 345 KB, accelerometer at 50 Hz × 86,400 × 12 bytes = 51.8 MB). Data reduction: 575×. BLE transfer: 90 KB / 125 KB/s = 0.72 seconds. Training: 1 epoch over 1 day's data on iPad: ~30 seconds during charging. Privacy budget: ε = 2 per embedding × 288 embeddings/day. With subsampling (train on 10% of embeddings): effective ε = 0.2 per day. Annual budget: 73 — well within typical ε = 100/year targets.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> Designing an On-Device Continual Learning System</b> · <code>data-parallelism</code> <code>security</code></summary>

- **Interviewer:** "You're designing a continual learning system for a mobile keyboard that adapts to each user's evolving vocabulary — new slang, work jargon, names of new contacts, trending topics. The model must learn new patterns without forgetting old ones, train entirely on-device for privacy, resist adversarial manipulation (a malicious app trying to poison the model via accessibility services), and work across the Snapdragon 8 Gen 3 (12 GB RAM) down to the Helio G99 (4 GB RAM). Design the system."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Fine-tune the full model on recent data with a replay buffer." Full model fine-tuning on a 4 GB RAM phone is infeasible (the optimizer state alone would exceed available memory), and a naive replay buffer doesn't protect against adversarial poisoning.

  **Realistic Solution:** Design a **modular continual learning architecture** with security-aware training:

  (1) **Architecture: frozen backbone + adaptive heads** — the base language model (50M params, 100 MB INT8) is frozen and never modified on-device. Personalization happens through three lightweight adapters: (a) **Vocabulary adapter** (500 KB): a hash-based embedding table for new words not in the base vocabulary. When the user types a new word 3+ times, it gets an embedding entry. Capacity: 10,000 new words. (b) **Context adapter** (2 MB): a LoRA module (rank 4) applied to the last 2 transformer layers. Captures user-specific writing style (formal vs casual, emoji usage, sentence length patterns). (c) **Temporal adapter** (200 KB): a small MLP that adjusts prediction probabilities based on time-of-day and recent conversation context. Total trainable parameters: 2.7 MB. Fits on any device.

  (2) **Continual learning without catastrophic forgetting** — the frozen backbone guarantees that base language knowledge is never lost. The adapters use **elastic weight consolidation (EWC)** with a twist: the Fisher information matrix is computed *once* at adapter initialization (from 1000 synthetic examples generated by the base model) and stored (2.7 MB). During training, the EWC penalty prevents the adapters from drifting too far from their initialization. New knowledge is added by growing the vocabulary adapter (append-only) and slowly updating the context adapter.

  (3) **Training schedule** — train during charging + idle, using the last 4 hours of typing data (stored in an encrypted ring buffer, max 500 KB). Training: 1 epoch over the buffer, batch size 8, learning rate 1e-4. On Snapdragon 8 Gen 3: 15 seconds. On Helio G99: 90 seconds. Both fit within Android's 10-minute background job limit. Frequency: once per day.

  (4) **Adversarial poisoning defense** — a malicious app could inject text via accessibility services (simulating keystrokes) to poison the training data. Defenses: (a) **Input validation**: reject training samples from accessibility-injected events (detectable via `InputDevice.SOURCE_KEYBOARD` vs `SOURCE_UNKNOWN`). (b) **Anomaly detection**: compute the perplexity of each training sample under the base model. Samples with perplexity > 100 (gibberish or adversarial strings) are excluded. Cost: one forward pass per sample (~5 ms). (c) **Gradient clipping**: clip per-sample gradients to L2 norm 0.1. Even if a poisoned sample passes validation, its influence on the model is bounded. (d) **Rollback trigger**: if the adapter's validation perplexity (measured on a held-out set of 100 synthetic sentences) increases by >20% after a training session, automatically rollback to the previous adapter checkpoint.

  (5) **Device-adaptive training** — on 4 GB RAM devices (Helio G99): train only the vocabulary adapter (500 KB) and temporal adapter (200 KB). Skip the LoRA context adapter (saves 2 MB of optimizer state). On 8+ GB devices: train all three adapters. On 12+ GB devices: use rank-8 LoRA instead of rank-4 for richer personalization. The adapter architecture is the same; only the training scope varies.

  > **Napkin Math:** Base model: 100 MB (frozen, memory-mapped). Adapters: 2.7 MB (resident). Optimizer state (Adam): 2 × 2.7 MB = 5.4 MB. Training buffer: 500 KB. Total training memory: 2.7 + 5.4 + 0.5 = 8.6 MB. On 4 GB device (700 KB adapters only): 0.7 + 1.4 + 0.5 = 2.6 MB. Fits anywhere. Training time: Snapdragon 8 Gen 3: 500 samples × 5 ms/sample = 2.5 sec forward + 2.5 sec backward = 5 sec + overhead = 15 sec. Helio G99: 5× slower = 75 sec + overhead = 90 sec. Poisoning defense: perplexity check = 500 × 5 ms = 2.5 sec. Gradient clipping: negligible. Total training with defenses: 17.5 sec (flagship), 92.5 sec (budget). New word learning latency: typed 3× → next training session (within 24 hours) → available for prediction.

  📖 **Deep Dive:** [Volume II: Security & Privacy](https://harvard-edge.github.io/cs249r_book_dev/contents/security_privacy/security_privacy.html)

  </details>

</details>


---


### Economics & Platform Constraints


#### 🔴 L6+

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The 50-Feature Mobile ML Platform</b> · <code>deployment</code></summary>

- **Interviewer:** "You're the ML platform architect for a super-app (like WeChat or Grab) that has 50 different ML features: face filters, speech recognition, recommendation, fraud detection, OCR, translation, smart replies, and more. Each feature team wants to ship their own model. The app is already 300 MB. Users on budget phones with 3 GB RAM complain about performance. Design the mobile ML platform."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Let each team bundle their own model and runtime." 50 models × 20 MB average = 1 GB of models. 50 separate TFLite/CoreML runtimes competing for memory. The app becomes unusable.

  **Realistic Solution:** Design a **shared ML infrastructure layer** that all 50 features consume:

  **(1) Shared backbone architecture** — many features need similar low-level features (edges, textures, object boundaries). Train a single multi-task backbone (e.g., EfficientNet-B0, 20 MB) that produces a shared feature tensor. Each feature team trains only a lightweight task head (0.5-2 MB each). Total: 20 MB backbone + 50 × 1 MB heads = 70 MB, not 1 GB.

  **(2) Model loading scheduler** — not all 50 models need to be in memory simultaneously. The face filter model loads when the camera opens. The OCR model loads when the scanner opens. The platform maintains a priority queue: active feature models are resident, recently-used models are cached, and cold models are evicted. Maximum concurrent models: 3-5 (based on available RAM).

  **(3) Unified runtime** — one TFLite/CoreML instance serves all models. Shared thread pool, shared memory allocator, shared NPU delegate. This eliminates per-model runtime overhead (~15 MB per runtime × 50 = 750 MB saved).

  **(4) Model delivery service** — models are not in the app bundle. They're downloaded on-demand from a CDN, keyed by feature + device capability + OS version. A budget phone with 3 GB RAM gets INT4 models; a flagship gets INT8 with larger heads.

  **(5) Resource governor** — a central controller monitors total ML memory usage, inference latency, thermal state, and battery level. If the phone is overheating, it reduces model quality (switch to smaller heads) or increases inference intervals. If RAM is low, it evicts cached models more aggressively.

  **(6) Observability** — each inference call is tagged with feature ID, model version, latency, and device state. A central dashboard shows per-feature ML health across the entire user base.

  > **Napkin Math:** Naive approach: 50 models × 20 MB = 1 GB models + 50 × 15 MB runtime = 1.75 GB total. Platform approach: 20 MB backbone + 50 × 1 MB heads (70 MB on disk, ~25 MB resident) + 15 MB shared runtime = 40 MB resident. Memory reduction: **44×**. App size: 300 MB app + 70 MB models (downloaded) vs 300 MB + 1 GB (bundled). Budget phone (3 GB RAM, ~1.5 GB available): naive approach OOMs. Platform approach: 40 MB ML + 200 MB app = 240 MB. Fits with 1.26 GB headroom.

  📖 **Deep Dive:** [Volume II: Edge Intelligence](https://harvard-edge.github.io/cs249r_book_dev/contents/edge_intelligence/edge_intelligence.html)

  </details>

</details>


---


### Additional Topics


#### 🔵 L4
<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Model Loading I/O Cliff</b> · <code>persistent-storage</code></summary>

- **Interviewer:** "Your Android app loads a 200 MB on-device LLM from internal storage. On a Pixel 8 Pro (UFS 4.0), the model loads in 180ms. On a Samsung Galaxy A15 (eMMC 5.1), the same model takes 4.2 seconds. The user sees a blank screen for 4 seconds on the budget phone. Both devices have enough RAM. Why is there a 23× difference, and how do you fix the user experience?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use a loading spinner — the user will wait." A 4-second blank screen causes 25% of users to abandon the feature (Google's research on mobile latency).

  **Realistic Solution:** The 23× gap comes from the storage technology difference:

  **UFS 4.0** (Pixel 8 Pro): sequential read = 4,200 MB/s. Random 4K read = 100K IOPS × 4 KB = 400 MB/s. 200 MB model load: 200 / 4,200 = 48ms sequential + overhead = ~180ms.

  **eMMC 5.1** (Galaxy A15): sequential read = 250 MB/s. Random 4K read = 7K IOPS × 4 KB = 28 MB/s. If the model file is fragmented on flash (common on budget phones with nearly-full storage): effective read = ~50 MB/s. 200 MB / 50 = 4,000ms + overhead = ~4,200ms.

  **Fixes — a layered approach:**

  (1) **Memory-mapped loading** (`mmap`): instead of reading the entire file into RAM, memory-map it. The OS loads pages on demand as the model accesses them. First inference is slower (page faults), but the app becomes responsive immediately — the user sees the UI while the model loads in the background.

  (2) **Progressive model loading**: load the embedding layer first (needed for the first token), then stream remaining layers as inference progresses. The user sees the first token in ~500ms instead of waiting 4.2s for the full model.

  (3) **Model file defragmentation**: on first install, write the model file sequentially to a dedicated file (using `fallocate` to pre-allocate contiguous blocks). This ensures sequential read speeds even on eMMC.

  (4) **Tiered model strategy**: on eMMC devices, use a smaller model (50 MB INT4 distilled variant). 50 MB / 250 MB/s = 200ms — acceptable. Detect storage type at runtime: `StorageManager.getUuidForPath()` + benchmark a 1 MB read to classify UFS vs eMMC.

  > **Napkin Math:** UFS 4.0 sequential: 4,200 MB/s. eMMC 5.1 sequential: 250 MB/s. Ratio: 16.8×. With fragmentation on eMMC: 50 MB/s. Ratio: 84×. mmap approach on eMMC: first page fault = 0.1ms. First inference (touches ~20 MB of weights): 20 MB / 50 MB/s = 400ms. Subsequent inferences: pages cached in RAM, ~0ms I/O. User-perceived latency: 400ms (first inference) vs 4,200ms (full load). Android device market share by storage type: UFS 3.0+ = 45%, UFS 2.1 = 25%, eMMC = 30%. **30% of your users hit the slow path.**

  📖 **Deep Dive:** [Volume I: Model Serving](https://harvard-edge.github.io/cs249r_book_dev/contents/model_serving/model_serving.html)

  </details>

</details>

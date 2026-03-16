# Topic Map — ML Systems Interview Playbook

This document is the **master plan** for the playbook. Every question we write traces back to a competency area defined here. When expanding the playbook, consult this map first — don't add questions that don't fill a gap.

---

## The Framework

An ML systems interviewer evaluates **10 core competency areas**. These are universal — they apply regardless of whether you deploy to a data center or a microcontroller. What changes across deployment tracks is *how* each competency manifests: the hardware, the constraints, the failure modes.

Each competency is tested at **4 mastery levels** (L3 → L6+), reflecting increasing cognitive depth:

| Level | Cognitive Skill | Scope | Industry Mapping |
|---|---|---|---|
| **L3** | Recall & Define | Task-level | Junior / New Grad |
| **L4** | Apply & Identify | Component-level | Mid-level (2–4 yrs) |
| **L5** | Analyze & Predict | System-level | Senior (5–8 yrs) |
| **L6+** | Synthesize & Derive | Architecture-level | Staff / Principal (8+ yrs) |

---

## The 10 Competency Areas

### 1. Compute Analysis

**What it tests:** Can you reason about whether a workload is compute-bound or memory-bound? Can you use the roofline model to diagnose performance?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | GPU roofline, Tensor Cores, HBM bandwidth, FP16/FP8 peak TFLOPS |
| 🤖 Edge | Integer roofline, TOPS/W, accelerator comparison under power caps |
| 📱 Mobile | NPU delegation, subgraph partitioning, heterogeneous scheduling (CPU/GPU/NPU) |
| 🔬 TinyML | MFLOPS on Cortex-M, no FPU, CMSIS-NN SIMD utilization |

**Textbook grounding:** Hardware Acceleration, Benchmarking, Compute Infrastructure

### 2. Memory Systems

**What it tests:** Can you account for where every byte lives and moves? Can you diagnose memory bottlenecks?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | VRAM accounting (weights + optimizer + activations + KV-cache), HBM tiers, gradient checkpointing |
| 🤖 Edge | DRAM budgets shared with OS/sensors, DMA transfers, memory-mapped I/O |
| 📱 Mobile | Shared RAM with OS and apps, no dedicated VRAM, memory-mapped weights, app lifecycle eviction |
| 🔬 TinyML | SRAM partitioning, flat tensor arena, flash vs SRAM, operator scheduling for peak RAM |

**Textbook grounding:** Neural Computation, Model Training, Hardware Acceleration, Data Systems

### 3. Numerical Representation

**What it tests:** Do you understand precision formats, quantization, and their system-level effects?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | FP16/BF16 mixed precision training, FP8 inference, loss scaling, underflow |
| 🤖 Edge | INT8 quantization-aware training, calibration strategies, per-channel vs per-tensor |
| 📱 Mobile | Float16 on NPU, quantized CPU fallback, silent accuracy loss from format conversion |
| 🔬 TinyML | INT8/INT4 only, zero-point arithmetic, requantization between layers, no floating point |

**Textbook grounding:** Model Compression, Performance Engineering

### 4. Model Architecture → System Cost

**What it tests:** Can you map architecture choices to resource consumption? Can you estimate cost before training?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | Transformer scaling laws, MoE routing overhead, attention complexity (O(n²) vs linear) |
| 🤖 Edge | CNN vs Transformer for real-time vision, model size vs frame budget trade-off |
| 📱 Mobile | MobileNet/EfficientNet design, on-device LLM feasibility, operator support constraints |
| 🔬 TinyML | Depthwise separable convolutions, inverted residuals, NAS for MCUs (MCUNet), operator support |

**Textbook grounding:** Network Architectures, Model Compression, Neural Computation

### 5. Latency & Throughput

**What it tests:** Can you decompose end-to-end latency and identify bottlenecks? Do you understand queueing theory?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | TTFT/TPOT, tail latency, continuous batching, queueing theory (Little's Law), Amdahl's law |
| 🤖 Edge | Worst-case execution time (WCET), frame deadlines (33ms at 30 FPS), pipeline overlap |
| 📱 Mobile | UI jank budgets (16ms at 60 FPS), ANR timeouts, async inference, interaction latency |
| 🔬 TinyML | Microsecond inference, duty cycle timing, interrupt-driven pipelines |

**Textbook grounding:** Model Serving, Benchmarking, Inference at Scale

### 6. Power & Thermal

**What it tests:** Can you reason about energy as a first-class constraint, not an afterthought?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | 700W–1000W TDP per chip, PUE, liquid cooling, TCO dominated by electricity |
| 🤖 Edge | 15–75W thermal envelope, DVFS P-states, sustained vs burst performance |
| 📱 Mobile | 3–5W total device power, thermal throttling, Jevons paradox in battery drain |
| 🔬 TinyML | Milliwatts, energy harvesting, duty cycling, active vs sleep power budgets |

**Textbook grounding:** Sustainable AI, Compute Infrastructure, Hardware Acceleration

### 7. Model Optimization

**What it tests:** Can you make a model smaller/faster without destroying accuracy? Do you know when each technique applies?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | Knowledge distillation, speculative decoding, MoE, FlashAttention, kernel fusion |
| 🤖 Edge | QAT, structured pruning for accelerator alignment, TensorRT optimization |
| 📱 Mobile | Delegation-aware pruning, Core ML / TFLite optimization, operator fusion |
| 🔬 TinyML | Mixed-precision quantization, operator scheduling for peak RAM, binary/ternary networks |

**Textbook grounding:** Model Compression, Performance Engineering, Data Selection

### 8. Deployment & Serving

**What it tests:** Can you get a model into production and keep it running? Do you understand the deployment lifecycle?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | Kubernetes autoscaling, A/B rollout, canary deploys, cold start optimization, model registries |
| 🤖 Edge | OTA updates, A/B partitioned firmware, functional safety certification, rollback mechanisms |
| 📱 Mobile | App store delivery, on-demand model download, tiered models by device capability, model versioning |
| 🔬 TinyML | Flash programming, FOTA (firmware over-the-air), model fits in flash, bootloader constraints |

**Textbook grounding:** ML Operations, Model Serving, Fleet Orchestration, Edge Intelligence

### 9. Monitoring & Reliability

**What it tests:** Can you detect when a system is failing silently? Can you design for graceful degradation?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | Data drift detection (KL divergence, PSI), training-serving skew, MTBF/MTTR, straggler detection |
| 🤖 Edge | Degradation ladders, fail-safe vs fail-operational, sensor fusion validation, watchdog timers |
| 📱 Mobile | Crash reporting, silent accuracy loss, federated analytics, thermal state monitoring |
| 🔬 TinyML | Watchdog timers, hard real-time guarantees, self-test routines, anomaly detection on-device |

**Textbook grounding:** ML Operations, Fault Tolerance, Robust AI, Operations at Scale

### 10. Security, Privacy & Fairness

**What it tests:** Can you reason about trust boundaries, data protection, and societal impact?

| Track | Manifestation |
|---|---|
| ☁️ Cloud | Prompt injection, DP-SGD, membership inference, model theft, bias amplification, subgroup evaluation |
| 🤖 Edge | Physical tampering, adversarial patches, safety certification (ISO 26262), supply chain integrity |
| 📱 Mobile | On-device differential privacy, federated learning, user data isolation, app sandboxing |
| 🔬 TinyML | Side-channel attacks, model extraction from flash, physical access threats, resource-constrained crypto |

**Textbook grounding:** Security & Privacy, Responsible Engineering, Robust AI

---

## Coverage Matrix — Current State

This matrix shows what we have (✅), what's thin (🟡), and what's missing (❌) for each competency × track.

### ☁️ Cloud Track (36 questions + 7 visual challenges)

| Competency | L3 | L4 | L5 | L6+ | Status |
|---|---|---|---|---|---|
| 1. Compute Analysis | — | The Profiling Crisis | — | The Roofline Shift | 🟡 Need L3, L5 |
| 2. Memory Systems | — | The Sequence Length Trap | The Energy-Movement Invariant | — | 🟡 Need L3, L6+ |
| 3. Numerical Representation | — | The Underflow Crisis | — | The Precision Trade-off, The Sparsity Fallacy | 🟡 Need L3, L5 |
| 4. Architecture → Cost | — | — | — | The Amdahl Ceiling | ❌ Need L3, L4, L5 |
| 5. Latency & Throughput | The Serving Inversion | The LLM Metrics | The Black Friday Collapse | The Decoding Bottleneck | ✅ Full |
| 6. Power & Thermal | — | — | — | The Energy Economics | ❌ Need L3, L4, L5 |
| 7. Model Optimization | The Compilation Overhead | The Pre-computation Trade-off | — | The Batching Dilemma, Speculative Decoding | 🟡 Need L5 |
| 8. Deployment & Serving | The Serverless Freeze | The Deployment Risk | — | The Disaggregated Serving | 🟡 Need L5 |
| 9. Monitoring & Reliability | The '95% Problem', The Data Pipeline Stall | The Silent Failure, The Training-Serving Skew, The Straggler Problem | — | The MTBF Crisis, The Retraining Math | 🟡 Need L5 |
| 10. Security/Privacy/Fairness | — | The Trust Boundary | — | The Bias Amplifier, The Privacy Audit | ❌ Need L3, L5 |

**Cloud gaps to fill:** 14 new questions needed for full L3–L6+ coverage across all 10 competencies.

### 🤖 Edge Track (9 questions)

| Competency | L3 | L4 | L5 | L6+ | Status |
|---|---|---|---|---|---|
| 1. Compute Analysis | The TOPS Illusion | — | — | The Integer Roofline | 🟡 Need L4, L5 |
| 2. Memory Systems | — | — | — | — | ❌ Need all levels |
| 3. Numerical Representation | — | — | The QAT Cliff | — | ❌ Need L3, L4, L6+ |
| 4. Architecture → Cost | — | — | — | — | ❌ Need all levels |
| 5. Latency & Throughput | The Frame Budget | The Pipeline Overlap | — | — | 🟡 Need L5, L6+ |
| 6. Power & Thermal | — | — | The Thermal Staircase | — | ❌ Need L3, L4, L6+ |
| 7. Model Optimization | — | — | — | — | ❌ Need all levels |
| 8. Deployment & Serving | — | The OTA Brick Risk | — | — | ❌ Need L3, L5, L6+ |
| 9. Monitoring & Reliability | — | The Timestamp Drift | — | The Degradation Ladder | 🟡 Need L3, L5 |
| 10. Security/Privacy/Fairness | — | — | — | — | ❌ Need all levels |

**Edge gaps to fill:** ~20 new questions needed.

### 📱 Mobile Track (9 questions)

| Competency | L3 | L4 | L5 | L6+ | Status |
|---|---|---|---|---|---|
| 1. Compute Analysis | The Delegation Lottery | The Heterogeneous Scheduling Trap | — | — | 🟡 Need L5, L6+ |
| 2. Memory Systems | — | The Silent Eviction | — | — | ❌ Need L3, L5, L6+ |
| 3. Numerical Representation | The Conversion Cliff | — | — | — | ❌ Need L4, L5, L6+ |
| 4. Architecture → Cost | — | — | — | — | ❌ Need all levels |
| 5. Latency & Throughput | — | The Jank Budget | — | — | ❌ Need L3, L5, L6+ |
| 6. Power & Thermal | — | — | The Thermal Cliff | The Battery Accounting Inversion | 🟡 Need L3, L4 |
| 7. Model Optimization | — | — | — | — | ❌ Need all levels |
| 8. Deployment & Serving | — | — | — | The Delivery Paradox | ❌ Need L3, L4, L5 |
| 9. Monitoring & Reliability | — | — | The Privacy-Utility Squeeze | — | ❌ Need L3, L4, L6+ |
| 10. Security/Privacy/Fairness | — | — | — | — | ❌ Need all levels |

**Mobile gaps to fill:** ~22 new questions needed.

### 🔬 TinyML Track (9 questions)

| Competency | L3 | L4 | L5 | L6+ | Status |
|---|---|---|---|---|---|
| 1. Compute Analysis | — | The CMSIS-NN Speedup | — | — | ❌ Need L3, L5, L6+ |
| 2. Memory Systems | The Flat Memory Reality | — | The Peak RAM Puzzle | The Keyword Spotting Pipeline | ✅ Strong (need L4) |
| 3. Numerical Representation | — | The Quantization Cliff | — | The Integer Arithmetic Engine | 🟡 Need L3, L5 |
| 4. Architecture → Cost | — | The Depthwise Separable Advantage | — | — | ❌ Need L3, L5, L6+ |
| 5. Latency & Throughput | — | — | — | — | ❌ Need all levels |
| 6. Power & Thermal | The Battery Life Equation | — | The Energy Harvesting Wall | — | 🟡 Need L4, L6+ |
| 7. Model Optimization | — | — | — | — | ❌ Need all levels |
| 8. Deployment & Serving | — | — | — | — | ❌ Need all levels |
| 9. Monitoring & Reliability | — | — | — | — | ❌ Need all levels |
| 10. Security/Privacy/Fairness | — | — | — | — | ❌ Need all levels |

**TinyML gaps to fill:** ~24 new questions needed.

---

## Question Generation Plan

The following tables specify exactly what questions to write. Each row is a gap in the coverage matrix above.

### ☁️ Cloud — New Questions (14)

| # | Competency | Level | Topic Tag | Scenario Seed |
|---|---|---|---|---|
| C1 | Compute Analysis | L3 | `roofline` | "What is arithmetic intensity and why does it matter more than raw TFLOPS?" |
| C2 | Compute Analysis | L5 | `roofline` | "Your model's arithmetic intensity shifts from 50 to 200 after enabling FlashAttention. Explain what changed in the memory access pattern." |
| C3 | Memory Systems | L3 | `memory` | "How much VRAM does a 7B parameter model need for inference in FP16? Walk through the calculation." |
| C4 | Memory Systems | L6+ | `memory` | "Design a memory budget for training a 70B model on 8×H100s with gradient checkpointing. Where do you place the recomputation boundaries?" |
| C5 | Numerical Representation | L3 | `precision` | "What is the difference between FP16 and BF16, and when would you choose one over the other?" |
| C6 | Numerical Representation | L5 | `precision` | "Your FP16 training run diverges at step 50k but works fine in BF16. What numerical property explains this?" |
| C7 | Architecture → Cost | L3 | `architecture` | "A PM asks you to estimate GPU-hours for fine-tuning a 13B model on 1M examples. Walk through the FLOPs calculation." |
| C8 | Architecture → Cost | L4 | `architecture` | "Your team switches from a dense 7B model to a 47B MoE with 8 experts (6B active). Memory goes up but FLOPs stay similar. Why?" |
| C9 | Architecture → Cost | L5 | `architecture` | "Quadratic attention is killing your 128k context serving cost. Compare three approaches to reduce it and their system trade-offs." |
| C10 | Power & Thermal | L3 | `power` | "An H100 draws 700W. Your data center rack has a 40kW power budget. How many GPUs can you physically fit?" |
| C11 | Power & Thermal | L4 | `power` | "Your cluster's PUE is 1.4. What does that mean in dollars, and what is the single biggest lever to reduce it?" |
| C12 | Power & Thermal | L5 | `power` | "Two clusters have identical hardware but one trains 30% faster. The slower cluster is in Phoenix, AZ in August. Explain." |
| C13 | Security/Privacy/Fairness | L3 | `security` | "What is the fundamental difference between a SQL injection and a prompt injection? Why can't you fix prompt injection with input validation?" |
| C14 | Security/Privacy/Fairness | L5 | `fairness` | "Your model passes aggregate fairness metrics but fails on intersectional subgroups. Explain the Fairness Gerrymandering problem." |

### 🤖 Edge — New Questions (Round 2, ~18)

| # | Competency | Level | Topic Tag | Scenario Seed |
|---|---|---|---|---|
| E1 | Compute Analysis | L4 | `roofline` | "Your Jetson Orin hits 70% of peak INT8 TOPS but your model runs 3× slower than expected. Use the roofline to diagnose." |
| E2 | Compute Analysis | L5 | `roofline` | "Compare the roofline of a Hailo-8 (dataflow) vs Jetson Orin (GPU) for a YOLO model. Which wins and why?" |
| E3 | Memory Systems | L3 | `memory` | "Your edge box has 8 GB DRAM shared between Linux, camera drivers, and your model. How do you budget it?" |
| E4 | Memory Systems | L4 | `memory` | "Your model loads 200 MB of weights from eMMC into DRAM at startup. First inference takes 3 seconds. How do you fix cold start?" |
| E5 | Memory Systems | L5 | `memory` | "Your multi-model pipeline (detection + tracking + planning) exceeds DRAM. Design a memory-sharing strategy." |
| E6 | Numerical Representation | L3 | `quantization` | "What is the difference between post-training quantization and quantization-aware training? When does PTQ fail?" |
| E7 | Numerical Representation | L4 | `quantization` | "Your INT8 model has 2% accuracy drop on easy scenes but 15% drop on night scenes. Diagnose the calibration failure." |
| E8 | Numerical Representation | L6+ | `quantization` | "Design a mixed-precision strategy for a perception stack: detection in INT8, depth estimation in FP16, planning in FP32. Justify each choice." |
| E9 | Architecture → Cost | L3 | `architecture` | "Why do edge vision systems use YOLO instead of ViT? Frame it in terms of FLOPs per frame and latency budget." |
| E10 | Architecture → Cost | L4 | `architecture` | "Your team wants to add a Transformer-based tracker alongside your CNN detector. Estimate the combined FLOPs and check if it fits in the frame budget." |
| E11 | Latency & Throughput | L5 | `latency` | "Your 30 FPS pipeline occasionally drops to 25 FPS under complex scenes. Design an adaptive quality system that maintains the deadline." |
| E12 | Latency & Throughput | L6+ | `latency` | "Design the worst-case execution time (WCET) analysis for a safety-critical perception pipeline. What must you guarantee and what can you shed?" |
| E13 | Power & Thermal | L3 | `thermal` | "Your edge box runs fine in the lab but throttles in the field. What environmental factor did you miss?" |
| E14 | Power & Thermal | L4 | `thermal` | "Your device has a 30W steady-state thermal budget but your model needs 45W. Design a duty-cycling strategy." |
| E15 | Model Optimization | L4 | `optimization` | "Your YOLO model is 15 FPS on the Jetson. You need 30 FPS. Walk through the optimization ladder: what do you try first, second, third?" |
| E16 | Model Optimization | L5 | `optimization` | "You pruned 40% of channels but latency only dropped 10%. Why doesn't structured pruning give linear speedups on edge accelerators?" |
| E17 | Deployment & Serving | L3 | `deployment` | "What is an A/B partition scheme for OTA updates, and why is it critical for edge devices that can't be physically accessed?" |
| E18 | Security/Privacy/Fairness | L4 | `security` | "Your autonomous vehicle's camera sees a stop sign with an adversarial patch. How does your system detect and handle this?" |

### 📱 Mobile — New Questions (Round 2, ~18)

| # | Competency | Level | Topic Tag | Scenario Seed |
|---|---|---|---|---|
| M1 | Compute Analysis | L5 | `compute` | "Your model runs 4ms on Pixel 8 but 40ms on a budget phone with the same NPU spec sheet. Diagnose the discrepancy." |
| M2 | Compute Analysis | L6+ | `compute` | "Design a heterogeneous execution strategy that splits a single model across CPU, GPU, and NPU based on operator characteristics." |
| M3 | Memory Systems | L3 | `memory` | "Your app uses 800 MB of RAM for the model. On a 4 GB phone, what happens when the user switches to the camera app?" |
| M4 | Memory Systems | L5 | `memory` | "Design a memory-mapped weight loading strategy that avoids the 2-second cold start when iOS kills your background process." |
| M5 | Memory Systems | L6+ | `memory` | "Your on-device LLM needs 4 GB for weights but the phone has 6 GB total. Design a paged weight loading system." |
| M6 | Numerical Representation | L4 | `precision` | "Your model converts from PyTorch FP32 → CoreML FP16. One layer's output changes by 12%. Which layer type is most likely affected and why?" |
| M7 | Numerical Representation | L5 | `precision` | "Design a mixed-precision deployment: which layers stay FP16 on the NPU and which fall back to FP32 on CPU? What's the latency impact?" |
| M8 | Architecture → Cost | L3 | `architecture` | "Why does MobileNetV3 use squeeze-and-excitation blocks? Frame it in terms of accuracy per MFLOP." |
| M9 | Architecture → Cost | L4 | `architecture` | "Your PM wants an on-device LLM. A 3B model needs 6 GB in FP16. Walk through why this is infeasible on most phones and what alternatives exist." |
| M10 | Architecture → Cost | L5 | `architecture` | "Compare the system cost of running Whisper-small vs a custom streaming ASR model for real-time transcription on mobile." |
| M11 | Latency & Throughput | L3 | `latency` | "What is 'jank' in mobile ML, and why does a 20ms inference time cause dropped frames at 60 FPS?" |
| M12 | Latency & Throughput | L6+ | `latency` | "Design an async inference pipeline for a camera app that maintains 60 FPS preview while running a 50ms segmentation model." |
| M13 | Power & Thermal | L3 | `power` | "Your ML feature drains 1% battery per minute. The PM says 'optimize the model.' Is the model actually the problem?" |
| M14 | Power & Thermal | L4 | `power` | "Explain why running inference on the NPU at 2 TOPS uses less battery than running the same model on the GPU at 2 TOPS." |
| M15 | Model Optimization | L4 | `optimization` | "Your TFLite model is 95% NPU-delegated but one custom op forces CPU fallback. What are your options to fix this?" |
| M16 | Deployment & Serving | L3 | `deployment` | "Your model is 500 MB. The App Store cellular download limit is 200 MB. How do you ship it?" |
| M17 | Monitoring & Reliability | L4 | `monitoring` | "Your on-device model's accuracy degrades over 6 months but you have no server-side ground truth. How do you detect this?" |
| M18 | Security/Privacy/Fairness | L4 | `privacy` | "Your keyboard prediction model learns from user typing. How do you improve the model without collecting user data on your servers?" |

### 🔬 TinyML — New Questions (Round 2, ~18)

| # | Competency | Level | Topic Tag | Scenario Seed |
|---|---|---|---|---|
| T1 | Compute Analysis | L3 | `compute` | "Your Cortex-M4 runs at 168 MHz with no FPU. How many INT8 multiply-accumulate operations can it do per second?" |
| T2 | Compute Analysis | L5 | `compute` | "Your model needs 10M MACs per inference. At 168 MHz with SIMD, can you hit 10 inferences per second? Show the math." |
| T3 | Compute Analysis | L6+ | `compute` | "Design a roofline analysis for a Cortex-M7 with 512 KB SRAM and 64-bit AXI bus. Where is the ridge point?" |
| T4 | Memory Systems | L4 | `memory` | "Your tensor arena is 200 KB but one layer's activation peak is 210 KB. What scheduling trick can you use without changing the model?" |
| T5 | Numerical Representation | L3 | `quantization` | "What is a zero-point in quantized inference, and why can't you just round FP32 weights to INT8?" |
| T6 | Numerical Representation | L5 | `quantization` | "Your per-tensor quantized model loses 5% accuracy. Per-channel recovers 3%. Explain the trade-off in code size and inference speed." |
| T7 | Architecture → Cost | L3 | `architecture` | "Why is a depthwise separable Conv2D cheaper than a standard Conv2D? Calculate the FLOP ratio for a 3×3 kernel on 64 channels." |
| T8 | Architecture → Cost | L5 | `architecture` | "Your person detection model uses MobileNetV2 but only 60% of operators are supported by the MCU runtime. Design a fallback strategy." |
| T9 | Architecture → Cost | L6+ | `architecture` | "Design a NAS search space for a Cortex-M4 with 256 KB SRAM. What constraints must the search respect that desktop NAS ignores?" |
| T10 | Latency & Throughput | L3 | `latency` | "Your keyword spotting model must respond within 200ms of the wake word. Budget the time: audio capture, feature extraction, inference, action." |
| T11 | Latency & Throughput | L4 | `latency` | "Your sensor samples at 100 Hz but inference takes 50ms. Design a pipeline that doesn't drop samples." |
| T12 | Latency & Throughput | L6+ | `latency` | "Design an interrupt-driven inference pipeline for a vibration anomaly detector that must respond within 1ms of a fault." |
| T13 | Power & Thermal | L4 | `power` | "Your device runs on a 225 mAh coin cell at 3.3V. Inference costs 50 mW for 10ms. How many inferences per day to last 1 year?" |
| T14 | Power & Thermal | L6+ | `power` | "Design a power budget for a solar-harvesting wildlife monitor: solar panel, supercapacitor, MCU, sensor, radio. When can you run inference?" |
| T15 | Model Optimization | L4 | `optimization` | "Your model fits in flash but the tensor arena exceeds SRAM by 20%. Walk through the optimization options in order of effort." |
| T16 | Deployment & Serving | L3 | `deployment` | "How do you update a model on 10,000 deployed sensor nodes? What happens if the update fails mid-flash?" |
| T17 | Monitoring & Reliability | L4 | `monitoring` | "Your deployed anomaly detector starts producing false positives after 3 months. You have no cloud connection. How do you detect and handle drift?" |
| T18 | Security/Privacy/Fairness | L4 | `security` | "An attacker has physical access to your MCU. How do they extract your model weights, and what can you do to prevent it?" |

---

## Target Question Counts

| Track | Current | Planned New | Target Total |
|---|---|---|---|
| ☁️ Cloud | 36 + 7 visual | 14 | 50 + 7 visual |
| 🤖 Edge | 9 | 18 | 27 |
| 📱 Mobile | 9 | 18 | 27 |
| 🔬 TinyML | 9 | 18 | 27 |
| **Total** | **70** | **68** | **138** |

---

## Level Distribution Targets

Each track should have a roughly balanced distribution across levels, weighted slightly toward L4 (the most common interview level):

| Level | Target % | Cloud Target | Edge Target | Mobile Target | TinyML Target |
|---|---|---|---|---|---|
| L3 | ~25% | 12 | 7 | 7 | 7 |
| L4 | ~30% | 15 | 8 | 8 | 8 |
| L5 | ~25% | 12 | 7 | 7 | 7 |
| L6+ | ~20% | 11 | 5 | 5 | 5 |

---

## How to Use This Map

1. **Before writing a question:** Find the competency × track × level cell in the coverage matrix. If it's already ✅, don't add another question there — fill a gap instead.
2. **When reviewing a question:** Check that the scenario seed matches the competency it claims to test. A question tagged `roofline` should actually require roofline reasoning, not just mention the word.
3. **When expanding the playbook:** Update this map first, then generate questions. The map is the source of truth; the questions are the implementation.

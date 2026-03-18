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

This matrix shows the coverage for each competency × track. Following the March 2026 expansion, all tracks have been fully fleshed out with 150+ questions each.

### ☁️ Cloud Track (217 questions)
✅ **Status:** Fully fleshed out across 6 rounds + visual debugging.

### 🤖 Edge Track (189 questions)
✅ **Status:** Fully fleshed out across 5 rounds.

### 📱 Mobile Track (174 questions)
✅ **Status:** Fully fleshed out across 5 rounds.

### 🔬 TinyML Track (168 questions)
✅ **Status:** Fully fleshed out across 5 rounds.

---

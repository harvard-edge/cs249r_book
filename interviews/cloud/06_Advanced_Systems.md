# Round 6: Advanced Systems & Cross-Cutting Concerns ⚙️

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Single_Node_Physics.md">🧱 Round 1</a> ·
  <a href="02_Distributed_Infrastructure.md">🚀 Round 2</a> ·
  <a href="03_Production_Serving.md">⚡ Round 3</a> ·
  <a href="04_Operations_and_Economics.md">💼 Round 4</a> ·
  <a href="05_Visual_Architecture_Debugging.md">🖼️ Round 5</a> ·
  <a href="06_Advanced_Systems.md">⚙️ Round 6</a>
</div>

---

This round fills the gaps across compute analysis, memory systems, numerical representation, model architecture costs, power and thermal constraints, and security/fairness — topics that real interviewers test but that were underrepresented in earlier rounds.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/cloud/06_Advanced_Systems.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 📐 Compute Analysis & Roofline

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Arithmetic Intensity Question</b> · <code>roofline</code></summary>

**Interviewer:** "A junior engineer on your team is frustrated. They just deployed a model on an H100 and the profiler reports only 30% of peak TFLOPS. They want to file a bug against NVIDIA. Before they embarrass themselves, explain why their GPU isn't broken."

**Common Mistake:** "The CUDA kernels aren't optimized" or "We need to increase the batch size to saturate the ALUs." Both assume the workload is compute-bound without checking.

**Realistic Solution:** The GPU is not broken — the workload is memory-bound. The metric that determines whether you hit peak TFLOPS is Arithmetic Intensity: the ratio of compute operations to bytes moved from memory. Every workload lives on a Roofline: below the ridge point, memory bandwidth is the ceiling; above it, compute is the ceiling. If your model's arithmetic intensity is below the GPU's ridge point, no amount of kernel tuning will help — you're waiting on HBM, not ALUs.

> **Napkin Math:** H100 peak = 989 TFLOPS FP16, HBM bandwidth = 3.35 TB/s. Ridge point = $989 \times 10^{12} / 3.35 \times 10^{12}$ = **295 Ops/Byte**. If the model's arithmetic intensity is 50 Ops/Byte, the bandwidth ceiling is $50 \times 3.35$ = 167.5 TFLOPS, or $167.5 / 989$ = **17% of peak**. The engineer's 30% is actually *above* the theoretical bandwidth ceiling for their workload — the kernels are fine.

> **Key Equation:** $\text{Attainable FLOPS} = \min\bigl(\text{Peak FLOPS},\; \text{BW}_{\text{HBM}} \times I\bigr)$ where $I = \text{FLOPs} / \text{Bytes moved}$

**📖 Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The FlashAttention Shift</b> · <code>roofline</code></summary>

**Interviewer:** "We enabled FlashAttention on our Transformer and the profiler shows the same number of FLOPs, but throughput jumped 3×. The team is confused — how can you get faster without doing less math?"

**Common Mistake:** "FlashAttention must use some approximation that reduces FLOPs" or "It's just better CUDA kernels." FlashAttention computes mathematically identical attention — no approximation. And the speedup is too large to explain by kernel micro-optimization alone.

**Realistic Solution:** FlashAttention doesn't reduce FLOPs — it reduces *bytes moved*. Standard attention materializes the full $N \times N$ attention matrix to HBM: for a 4096-token sequence, that's $4096^2 \times 2$ bytes = 32 MB written and read per head per layer. FlashAttention tiles the computation into SRAM-sized blocks, computing softmax incrementally and never writing the full attention matrix to HBM. The FLOPs are identical, but HBM traffic drops by 4–10×. On the Roofline, this shifts the workload's effective arithmetic intensity from ~50 Ops/Byte to ~200 Ops/Byte, moving it from the memory-bound regime toward the compute-bound regime.

> **Napkin Math:** Standard attention at seq_len=4096, 32 heads, 80 layers: HBM traffic for attention matrices alone = $32 \times 80 \times 4096^2 \times 2 \times 3$ (Q·K, softmax, attn·V) ≈ **258 GB**. FlashAttention keeps intermediates in 20 MB of SRAM per SM — HBM traffic drops to ~30 GB. Same FLOPs, ~8× fewer bytes moved. Effective arithmetic intensity jumps from ~50 to ~400 Ops/Byte, crossing the H100's ridge point of 295.

> **Key Equation:** $I_{\text{effective}} = \frac{\text{FLOPs}_{\text{attention}}}{\text{Bytes}_{\text{HBM}}}$ — FlashAttention holds $\text{FLOPs}$ constant while shrinking $\text{Bytes}_{\text{HBM}}$

**📖 Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

---

### 🧠 Memory Systems

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The VRAM Budget</b> · <code>memory</code></summary>

**Interviewer:** "Your PM asks: 'Can we run Llama-2-7B inference on a single A100 40 GB?' Give me the exact memory breakdown — don't just say 'it fits.'"

**Common Mistake:** "7 billion parameters × 2 bytes = 14 GB, so yes, plenty of room." This accounts for weights only and ignores the two other major memory consumers during inference.

**Realistic Solution:** You need to budget three components: (1) **Model weights** — the static cost. (2) **KV-cache** — grows with batch size and sequence length. (3) **Activation memory** — temporary buffers for the forward pass. For a single request at 2048 tokens, the budget is tight but feasible on 40 GB. The real question is how many *concurrent* requests you can serve before the KV-cache pushes you into OOM.

> **Napkin Math:** Llama-2-7B in FP16: **Weights** = 7B × 2 bytes = **14 GB**. **KV-cache** (32 layers, 32 heads, head_dim=128, seq_len=2048, batch=1) = $2 \times 32 \times 32 \times 128 \times 2048 \times 2$ bytes = **1.07 GB**. **Activations** (peak intermediate buffers) ≈ **0.5 GB**. **Total ≈ 15.6 GB** — fits on A100 40 GB with 24.4 GB free for batching. At batch_size=16: KV-cache balloons to 17.1 GB, total = 31.6 GB — still fits. At batch_size=24: KV-cache = 25.7 GB, total = 40.2 GB — **OOM**. Scale this to Llama-2-70B at 4K context: a single sequence's KV-cache alone is ~2 GB. 32 concurrent users need 64 GB of KV-cache — nearly all of an H100's 80 GB HBM, leaving almost nothing for the 140 GB of weights (which must be sharded across GPUs).

> **Key Equation:** $\text{VRAM}_{\text{inference}} = \underbrace{P \times b_w}_{\text{weights}} + \underbrace{2 \times L \times H \times d_h \times S \times B \times b_w}_{\text{KV-cache}} + \underbrace{A(B)}_{\text{activations}}$

**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Gradient Checkpointing Boundary</b> · <code>memory</code></summary>

**Interviewer:** "We need to train a 70B model on 8× H100 80 GB GPUs. The team enabled ZeRO-3 and claims it'll fit. Walk me through the actual per-GPU memory budget and tell me where it breaks."

**Common Mistake:** "ZeRO-3 shards everything evenly — 70B × 16 bytes / 8 GPUs = 140 GB per GPU. It doesn't fit, so we need more GPUs." This is the right calculation for the *sharded state*, but it misses that gradient checkpointing changes the equation for activations, which is where the real design decision lives.

**Realistic Solution:** With ZeRO-3, the model state (weights + gradients + optimizer) is sharded across all 8 GPUs. The mixed-precision training footprint per parameter is 2 (FP16 weights) + 2 (FP16 gradients) + 4 (FP32 master weights) + 8 (FP32 Adam m + v) = 16 bytes. Sharded: 70B × 16 / 8 = **140 GB** — exceeds 80 GB. But ZeRO-3 doesn't hold all shards resident simultaneously; during a forward/backward pass, parameters are gathered on-demand and discarded. Peak *resident* model state is much lower (~20–25 GB). The real memory pressure comes from **activations**: without checkpointing, storing all intermediate activations for 80 transformer layers at reasonable batch sizes can consume 60+ GB. Gradient checkpointing trades 33% extra compute to store activations only at checkpoint boundaries (every $\sqrt{L}$ ≈ 9 layers), reducing activation memory by ~9×.

> **Napkin Math:** Per-GPU with ZeRO-3 (resident): sharded optimizer ≈ 70B × 12 / 8 = **105 GB** — too much if fully materialized, but ZeRO-3 streams shards. Peak resident model state ≈ **20–25 GB** (one layer's params gathered at a time). Activations without checkpointing (batch=1, seq=2048): ~**60 GB**. With checkpointing every 9 layers: ~**7 GB**. Total per-GPU: 25 + 7 + overhead ≈ **35–40 GB**. Fits in 80 GB with room for micro-batch size 2–4. The design knob: checkpoint every $N$ layers, where $N$ balances recomputation overhead (33% at $N = \sqrt{L}$) against activation memory.

> **Key Equation:** $\text{Activation memory} \propto \frac{L}{N} \times B \times S \times d_{\text{model}}$ where $N$ = checkpoint interval; optimal $N = \sqrt{L}$ minimizes $\text{memory} \times \text{recompute}$

**📖 Deep Dive:** [Volume II: Distributed Training](https://mlsysbook.ai/vol2/distributed_training.html)
</details>

---

### 🔢 Numerical Representation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The FP16 vs BF16 Question</b> · <code>precision</code></summary>

**Interviewer:** "A new hire asks you: 'FP16 and BF16 are both 16 bits. Why does everyone use BF16 for training? Aren't they the same?' Give them the 2-minute explanation."

**Common Mistake:** "BF16 is just a newer, better version of FP16" or "They're basically interchangeable — BF16 is just what NVIDIA recommends now." Both miss the fundamental bit-layout trade-off.

**Realistic Solution:** FP16 and BF16 allocate their 16 bits differently, and that difference is existential for training stability. FP16 uses 5 exponent bits and 10 mantissa bits: high precision (~3 decimal digits) but narrow range (±65,504). BF16 uses 8 exponent bits and 7 mantissa bits: lower precision (~2 decimal digits) but the same range as FP32 (±3.4 × 10³⁸). For training, *range* matters more than *precision* because gradients can span dozens of orders of magnitude. FP16's narrow range causes small gradients to underflow to zero, requiring complex loss scaling. BF16 "just works" because its 8 exponent bits match FP32's dynamic range.

> **Napkin Math:** FP16 min subnormal ≈ $5.96 \times 10^{-8}$, max = 65,504. BF16 min subnormal ≈ $9.18 \times 10^{-41}$, max ≈ $3.39 \times 10^{38}$. Gradients in layer 80 of a deep Transformer routinely hit $10^{-12}$ to $10^{-20}$. In FP16: **underflow to zero** → training collapses. In BF16: represented exactly (within precision) → training proceeds. The precision loss (10 → 7 mantissa bits) is irrelevant because SGD's stochastic noise already exceeds 2 decimal digits of precision.

> **Key Equation:** $\text{Dynamic range} = 2^{2^{e-1}}$ where $e$ = exponent bits. FP16: $2^{16} = 65536$. BF16: $2^{128} \approx 3.4 \times 10^{38}$.

**📖 Deep Dive:** [Volume I: Neural Computation](https://mlsysbook.ai/vol1/nn_computation.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The FP16 Divergence</b> · <code>precision</code></summary>

**Interviewer:** "We're pre-training a 13B model. It trains perfectly in BF16 for 100k steps. When we switch to FP16 with dynamic loss scaling, it diverges around step 50k with loss spikes and eventual NaNs. The hyperparameters are identical. What is happening at step 50k that wasn't happening at step 1k?"

**Common Mistake:** "The loss scaler isn't tuned properly — just increase the scaling factor" or "FP16 training is inherently unstable at scale." The first treats the symptom; the second is defeatist and wrong.

**Realistic Solution:** Early in training, gradients are large (the model is far from any minimum) and comfortably within FP16's representable range. As training converges, gradient magnitudes shrink — by step 50k, many gradients have decayed below FP16's minimum representable value of ~$5.96 \times 10^{-8}$. Dynamic loss scaling tries to compensate by multiplying the loss by a large factor before backprop, but this is a balancing act: scale too high and activations overflow to Inf; scale too low and gradients underflow to zero. Around step 50k, the gradient distribution straddles FP16's narrow range — some values need high scaling (small gradients) while others need low scaling (large activations). No single scale factor works for all parameters simultaneously, and training oscillates between overflow and underflow until it diverges. BF16 never has this problem because its 8 exponent bits give the same range as FP32.

> **Napkin Math:** At step 1k: gradient magnitudes ≈ $10^{-3}$ to $10^{-1}$ — well within FP16's range. At step 50k: gradient magnitudes ≈ $10^{-6}$ to $10^{-12}$. FP16 floor = $5.96 \times 10^{-8}$. Everything below that floor → **zero**. With 70% of gradients underflowing, the effective learning rate collapses for most parameters while a few with large gradients still update — the model destabilizes. BF16 floor = $9.18 \times 10^{-41}$ — gradients of $10^{-12}$ are represented with no issue.

> **Key Equation:** $\text{Loss scale} \times \nabla_\theta \mathcal{L} \in [\text{FP16}_{\min},\, \text{FP16}_{\max}]$ must hold for *all* parameters simultaneously — an increasingly impossible constraint as gradient variance grows during training

**📖 Deep Dive:** [Volume I: Neural Computation](https://mlsysbook.ai/vol1/nn_computation.html)
</details>

---

### 🏗️ Architecture → System Cost

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Fine-Tuning Estimate</b> · <code>architecture</code></summary>

**Interviewer:** "The PM just walked over and asked: 'How much will it cost to fine-tune Llama-2-13B on our 1M-example dataset?' They need a number by end of day. Walk me through how you'd estimate this on a napkin."

**Common Mistake:** "I'd need to run a benchmark first — I can't estimate without profiling." In an interview (and in real planning), you must be able to estimate from first principles. Another common mistake: forgetting to multiply by 3 for the backward pass (it's ~2× the forward pass cost).

**Realistic Solution:** The standard approximation for training FLOPs is $6 \times N$ FLOPs per token, where $N$ is the parameter count. The factor of 6 comes from: 2× for the forward pass (multiply-accumulate = 2 ops per parameter per token) and 4× for the backward pass (2× for gradient w.r.t. activations + 2× for gradient w.r.t. weights). Multiply by total tokens, divide by GPU throughput at realistic utilization, convert to hours, multiply by price.

> **Napkin Math:** Params = 13B. Dataset = 1M examples × 512 avg tokens = 512M tokens. Total FLOPs = $6 \times 13 \times 10^9 \times 512 \times 10^6$ = $3.99 \times 10^{19}$ FLOPs. H100 at ~50% MFU = 989 × 0.5 = **495 TFLOPS** effective. Time = $3.99 \times 10^{19} / 4.95 \times 10^{14}$ = 80,600 seconds ≈ **22.4 GPU-hours**. At $2.50/GPU-hr (on-demand H100) = **$56**. On spot instances at $0.90/hr = **$20**. The PM's answer: "Under $60 on-demand, under $25 on spot."

> **Key Equation:** $\text{GPU-hours} = \frac{6 \times N \times T}{\text{GPU}_{\text{TFLOPS}} \times \text{MFU} \times 3600 \times 10^{12}}$ where $N$ = params, $T$ = tokens, MFU = model FLOP utilization

**📖 Deep Dive:** [Volume I: Model Training](https://mlsysbook.ai/vol1/model_training.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The MoE Memory Trap</b> · <code>architecture</code></summary>

**Interviewer:** "The team is excited: they switched from a dense 7B model to a 47B Mixture-of-Experts model with 8 experts, 2 active per token. 'Same inference speed, 7× more knowledge!' they claim. Then they try to deploy it on a single A100 40 GB and it OOMs. What did they get wrong?"

**Common Mistake:** "MoE only activates 2 of 8 experts, so memory usage should be similar to a 12B dense model (6B active params × 2 for FP16)." This confuses *compute* with *memory*.

**Realistic Solution:** MoE saves compute but not memory. During inference, only 2 experts are active per token (~6B active parameters), so the FLOPs per token are comparable to a ~6B dense model. But **all 47B parameters must reside in VRAM** because the router can select *any* expert for *any* token — you don't know which experts will be needed until runtime. The memory footprint is the full 47B × 2 bytes = 94 GB in FP16, regardless of how many experts are active. This is the MoE Memory Trap: the compute profile of a 6B model with the memory profile of a 47B model.

> **Napkin Math:** Dense 7B: FLOPs/token ≈ $2 \times 7\text{B}$ = 14 GFLOPs. Memory = 7B × 2 = **14 GB**. MoE 47B (2/8 active): FLOPs/token ≈ $2 \times 6\text{B}$ = 12 GFLOPs (similar). Memory = 47B × 2 = **94 GB** (6.7× more). The 47B MoE needs at least 2× A100 80 GB with tensor parallelism, or quantization to INT4 (47B × 0.5 = 23.5 GB) to fit on a single GPU. Compute is cheap; memory is the constraint.

> **Key Equation:** $\text{VRAM}_{\text{MoE}} = N_{\text{total}} \times b_w$ (not $N_{\text{active}} \times b_w$)

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Attention Cost Explosion</b> · <code>architecture</code></summary>

**Interviewer:** "We're deploying a model that supports 128k context. Users love it, but our serving costs are 10× higher than the 4k-context version, and P99 latency is through the roof. The model has the same number of parameters. Why does 32× more context cost 10× more to serve, and what are our architectural options?"

**Common Mistake:** "Context length only affects KV-cache memory, so we just need more VRAM." Memory is part of it, but the compute cost of attention itself is the dominant factor at 128k.

**Realistic Solution:** Standard self-attention is $O(n^2)$ in both compute and memory with respect to sequence length. Going from 4k to 128k tokens is a 32× increase in $n$, which means a $32^2 = 1024\times$ increase in attention FLOPs and memory. The attention matrix alone at 128k is enormous. Three architectural mitigations exist, each with different trade-offs: (1) **FlashAttention** — eliminates the $O(n^2)$ memory by tiling to SRAM, but compute is still $O(n^2)$; it buys you memory headroom, not compute savings. (2) **Sliding window / local attention** — reduces to $O(n \times w)$ where $w$ is the window size, but sacrifices global context; tokens beyond the window are invisible. (3) **Sparse attention with global sentinel tokens** — most tokens attend locally, but a small set of global tokens attend to everything, preserving long-range dependencies at sub-quadratic cost.

> **Napkin Math:** Attention FLOPs per layer per head at 4k tokens: $2 \times 4096^2 \times 128$ = 4.3 GFLOPs. At 128k tokens: $2 \times 131072^2 \times 128$ = **4.4 TFLOPs** — a **1024× increase**. Attention matrix memory at 128k in FP16: $131072^2 \times 2$ = **32 GB per head per layer**. With 32 heads and 80 layers: 32 × 80 × 32 GB = **81.9 TB** — obviously impossible without FlashAttention's tiling. Sliding window with $w = 4096$: attention FLOPs drop back to $2 \times 131072 \times 4096 \times 128$ = 137 GFLOPs per layer per head — a **32× reduction** from full attention.

> **Key Equation:** $\text{Attention FLOPs} = O(n^2 \cdot d)$ for full attention; $O(n \cdot w \cdot d)$ for sliding window, where $w \ll n$

**📖 Deep Dive:** [Volume I: Neural Network Architectures](https://mlsysbook.ai/vol1/nn_architectures.html)
</details>

---

### ⚡ Power & Thermal

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Rack Power Budget</b> · <code>power</code></summary>

**Interviewer:** "Facilities just told us our new rack has a 40 kW power budget. The H100 SXM has a 700W TDP. How many GPUs can we fit in this rack?"

**Common Mistake:** "40,000 W / 700 W = 57 GPUs." This treats the GPU as the only power consumer in the rack.

**Realistic Solution:** A GPU doesn't run in a vacuum. Each GPU sits in a server node that also draws power for CPUs, DRAM, NVLink switches, NVMe storage, cooling fans, and network interface cards. A typical DGX H100 node (8 GPUs) draws ~10.2 kW total: 5.6 kW for GPUs + 4.6 kW for everything else — roughly 575W overhead per GPU. Add top-of-rack networking switches (~2 kW per rack). The real calculation must account for the full system power, not just the accelerator TDP.

> **Napkin Math:** Per-GPU system power = 700W (GPU) + ~275W (CPU/RAM/NIC/fan share) = **~975W**. Rack networking overhead = ~2 kW. Usable rack power = 40 - 2 = 38 kW. GPUs per rack = $38000 / 975$ ≈ **39 GPUs** (4 DGX nodes × 8 GPUs + room for 7 more, but DGX nodes come in 8-GPU units, so realistically **32 GPUs** = 4 nodes). Naive estimate of 57 was off by **44%**. For next-gen B200 at 1000W TDP, the gap widens further.

> **Key Equation:** $N_{\text{GPU}} = \frac{P_{\text{rack}} - P_{\text{network}}}{\text{TDP}_{\text{GPU}} + P_{\text{host-per-GPU}}}$

**📖 Deep Dive:** [Volume II: Compute Infrastructure](https://mlsysbook.ai/vol2/compute_infrastructure.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The PUE Dollar Cost</b> · <code>power</code></summary>

**Interviewer:** "Our CFO asks: 'We're spending $86M a year on electricity for our 10,000-GPU training cluster. The facilities team says they can reduce PUE from 1.4 to 1.1 by switching to liquid cooling, but the retrofit costs $5M. Is it worth it?'"

**Common Mistake:** "PUE is a facilities metric — it doesn't affect ML engineering decisions." In reality, PUE directly determines your electricity bill and can make or break the ROI of a training cluster.

**Realistic Solution:** PUE (Power Usage Effectiveness) is the ratio of total facility power to IT equipment power. A PUE of 1.4 means for every 1W of compute, you pay for 1.4W total — the extra 0.4W goes to cooling, lighting, and power distribution. For a 10,000-GPU cluster, that 0.4W multiplier translates to millions of dollars per year. The CFO's question is a straightforward ROI calculation: compute the annual savings from the PUE reduction and compare to the retrofit cost.

> **Napkin Math:** 10,000 H100s × 700W = **7 MW** IT load. At PUE 1.4: total = 7 × 1.4 = **9.8 MW**. At PUE 1.1: total = 7 × 1.1 = **7.7 MW**. Savings = 2.1 MW. Annual electricity savings = 2.1 MW × 8,760 hrs × $0.10/kWh = **$1.84M/year**. $5M retrofit payback period = $5M / $1.84M ≈ **2.7 years**. For a cluster with a 5-year lifespan, total savings = $1.84M × 5 - $5M = **$4.2M net**. Tell the CFO: "Yes, the retrofit pays for itself in under 3 years."

> **Key Equation:** $\text{Annual cost} = P_{\text{IT}} \times \text{PUE} \times 8760 \times C_{\text{kWh}}$ and $\Delta\text{Cost} = P_{\text{IT}} \times (\text{PUE}_{\text{old}} - \text{PUE}_{\text{new}}) \times 8760 \times C_{\text{kWh}}$

**📖 Deep Dive:** [Volume II: Compute Infrastructure](https://mlsysbook.ai/vol2/compute_infrastructure.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Thermal Throttling Mystery</b> · <code>power</code></summary>

**Interviewer:** "We have two identical 1,000-GPU H100 clusters — same hardware, same software, same model. Cluster A in Phoenix, Arizona consistently trains 30% slower than Cluster B in The Dalles, Oregon. The ops team has checked everything: drivers, firmware, network config — all identical. What's going on?"

**Common Mistake:** "There must be a software misconfiguration or a bad batch of GPUs" or "Network latency between racks is higher in Phoenix." Both ignore the physical environment the hardware sits in.

**Realistic Solution:** Thermal throttling. Phoenix ambient temperature in summer reaches 45°C (113°F). The Dalles averages 20°C (68°F). Data center cooling systems have finite capacity — they can only reject a certain number of watts of heat to the outside air. When ambient temperature rises, the temperature delta between the coolant and the outside air shrinks, reducing cooling effectiveness. When GPU junction temperatures exceed ~83°C, the hardware automatically reduces clock speeds and power draw (from 700W to ~500W) to prevent damage. This is thermal throttling, and it directly reduces training throughput. The 30% slowdown maps almost exactly to the power reduction: $500/700 = 71\%$ of peak performance.

> **Napkin Math:** H100 TDP = 700W, throttle point = ~83°C junction. Phoenix data center: 45°C ambient → cooling struggles → GPU junction hits 83°C → throttles to 500W → **71% of peak throughput**. Oregon: 20°C ambient → GPU junction stays at ~65°C → full 700W → **100% throughput**. The 29% gap matches the reported 30% slowdown. Fix options: (1) liquid cooling (removes ambient dependency), (2) schedule heavy training jobs at night (Phoenix drops to 25°C), (3) over-provision cooling capacity (expensive). This is why hyperscalers build in Oregon, Iowa, and the Nordics — not Phoenix.

> **Key Equation:** $P_{\text{cooling}} = \dot{m} \times c_p \times (T_{\text{coolant}} - T_{\text{ambient}})$ — as $T_{\text{ambient}} \uparrow$, cooling capacity $\downarrow$ at fixed infrastructure

**📖 Deep Dive:** [Volume II: Compute Infrastructure](https://mlsysbook.ai/vol2/compute_infrastructure.html)
</details>

---

### 🔒 Security, Privacy & Fairness

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Prompt Injection Paradox</b> · <code>security</code></summary>

**Interviewer:** "We solved SQL injection 20 years ago with parameterized queries. Why can't we solve prompt injection the same way?"

**Common Mistake:** "We can — just sanitize the user input before passing it to the model" or "Use a system prompt that says 'ignore malicious instructions.'" Input sanitization treats symptoms, and instruction-based defenses are trivially bypassed because the model processes the defense and the attack in the same context.

**Realistic Solution:** SQL injection was solvable because SQL engines have a hard architectural boundary between *code* (the query template) and *data* (the parameters). The engine never interprets parameter values as SQL commands. LLMs have no such boundary. The system prompt, user input, retrieved documents, and tool outputs are all concatenated into a single token sequence and processed identically by the attention mechanism. The model *cannot distinguish* between "instructions from the developer" and "instructions injected by the user" — they're all just tokens. This is a fundamental architectural limitation, not a bug to be patched. Defense requires defense-in-depth: input classifiers that detect injection attempts *before* they reach the model, output filters that catch policy violations, privilege separation at the orchestration layer (the model can *request* tool calls but a separate system *approves* them), and limiting the blast radius of any successful injection.

> **Napkin Math:** A parameterized SQL query processes data through a separate code path — 0% chance of code execution from user input. An LLM processes system prompt + user input through the *same* attention layers — the model assigns attention weights to both equally. If the system prompt is 500 tokens and the user input is 2000 tokens, the "attack surface" has 4× more tokens than the "defense." No amount of prompt engineering changes this ratio.

**📖 Deep Dive:** [Volume II: Security and Privacy](https://mlsysbook.ai/vol2/security_privacy.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Fairness Gerrymandering Problem</b> · <code>fairness</code></summary>

**Interviewer:** "Our hiring model passes demographic parity for gender (male/female approval rates within 2%) and for race (all racial groups within 3%). Legal says we're compliant. Then an audit reveals the model rejects Black women at 2× the rate of any other group. How is this mathematically possible if we pass fairness checks on both attributes independently?"

**Common Mistake:** "The audit must be wrong — if we're fair on gender and fair on race, we must be fair on their intersection" or "This is just noise from a small sample size." The first is provably false; the second dismisses a real structural failure.

**Realistic Solution:** This is Fairness Gerrymandering (Kearns et al., 2018). A model can satisfy fairness constraints on every individual protected attribute while systematically discriminating against their intersections. Mathematically: let approval rate for women = 50% and for men = 50% (gender parity). Let approval rate for Black applicants = 48% and White applicants = 52% (within 3% race parity). But internally: Black women = 30%, Black men = 66%, White women = 70%, White men = 34%. Both marginal constraints are satisfied, but the intersectional disparity is enormous. The model learned to use the *intersection* as a proxy, which marginal fairness metrics are blind to. The fix requires evaluating fairness on intersectional subgroups — but the number of subgroups grows combinatorially ($2^k$ for $k$ binary attributes), making exhaustive testing infeasible for many attributes.

> **Napkin Math:** With 4 protected attributes (gender, race, age bracket, disability), each binary: $2^4 = 16$ intersectional subgroups. With 10 attributes at 3 categories each: $3^{10} = 59,049$ subgroups. Testing each with statistical significance (n ≥ 100 per group) requires 5.9M samples — often more data than the entire training set. Practical approaches: multi-calibration (Hébert-Johnson et al., 2018) provides fairness guarantees for *all* identifiable subgroups simultaneously, or use Gerrymandering-aware auditing that searches for the worst-off subgroup algorithmically rather than exhaustively.

> **Key Equation:** $\max_{g \in \mathcal{G}} \bigl| P(\hat{Y}=1 \mid G=g) - P(\hat{Y}=1) \bigr| \leq \epsilon$ must hold for all subgroups $g$ in a rich class $\mathcal{G}$, not just marginal groups

**📖 Deep Dive:** [Volume II: Robust AI](https://mlsysbook.ai/vol2/robust_ai.html)
</details>

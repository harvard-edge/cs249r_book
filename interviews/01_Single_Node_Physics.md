# Round 1: Single-Node Systems & Silicon Physics 🧱

<div align="center">
  <a href="README.md">🏠 Home</a> ·
  <a href="00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Single_Node_Physics.md">🧱 Round 1</a> ·
  <a href="02_Distributed_Infrastructure.md">🚀 Round 2</a> ·
  <a href="03_Production_Serving.md">⚡ Round 3</a> ·
  <a href="04_Operations_and_Economics.md">💼 Round 4</a> ·
  <a href="05_Visual_Architecture_Debugging.md">🖼️ Round 5</a>
</div>

---

The domain of the ML Systems Engineer. This round tests your understanding of what happens *inside* a single server chassis: memory hierarchies, compute bounds, and arithmetic intensity.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/01_Single_Node_Physics.md)** (Edit in Browser) — see [README](README.md#question-format) for the template.

---

### 📐 Roofline & Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Profiling Crisis</b> · <code>roofline</code></summary>

**Interviewer:** "You've deployed a custom recommendation model. The profiling dashboard shows you are achieving 120 TFLOPS out of a possible 300 TFLOPS on your GPU. Your tech lead suggests buying a faster GPU to fix the latency. Why is your tech lead wrong?"

**Common Mistake:** "We need to optimize the CUDA kernels" or "The GPU isn't being fully utilized, so a faster one will help." Both assume the problem is compute — it isn't.

**Realistic Solution:** The tech lead hasn't checked the Arithmetic Intensity ($Ops/Bytes$). If the model is memory-bound (intensity is lower than the ridge point of the roofline), a GPU with faster ALUs will do nothing. You must buy a GPU with higher *Memory Bandwidth (HBM)*, or optimize the model to move fewer bytes (e.g., quantization).

> **Key Equation:** $\text{Attainable FLOPS} = \min(\text{Peak FLOPS},\ \text{Bandwidth} \times \text{Arithmetic Intensity})$

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Roofline Shift</b> · <code>roofline</code></summary>

**Interviewer:** "We upgraded our serving fleet from A100s to H100s. The H100 has 3x more FP16 TFLOPS. However, our LLM decode throughput got exactly 0% faster. How is that physically possible?"

**Common Mistake:** "The drivers must be misconfigured" or "We need to re-optimize the CUDA kernels for H100." Both miss the fundamental physics.

**Realistic Solution:** The ridge point of the Roofline shifted right. Your workload was already memory-bandwidth bound on the A100 (Arithmetic Intensity < Ridge Point). The H100 has 3x more compute, but only 1.5x more memory bandwidth. Because LLM decoding is bandwidth-starved ($I \approx 1$), faster ALUs literally cannot do any work.

> **Napkin Math:** A100 ridge point = 312 TFLOPS / 2 TB/s ≈ 153 Ops/Byte. H100 ridge point = 989 TFLOPS / 3.35 TB/s ≈ 295 Ops/Byte. LLM decode has $I \approx 1$ Ops/Byte — orders of magnitude below both ridge points. You're deep in the memory-bandwidth ceiling. The only path to faster decoding is more bandwidth, not more FLOPS.

> **Key Equation:** $\text{Ridge Point} = \frac{\text{Peak FLOPS}}{\text{Memory Bandwidth}}$

**📖 Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

---

### 🔢 Numerical Precision & Quantization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Underflow Crisis</b> · <code>precision</code></summary>

**Interviewer:** "We switched our pre-training job from FP32 to FP16 to save memory, but the loss is returning NaNs within the first 100 steps. What numerical property is failing?"

**Common Mistake:** "The learning rate is too high for FP16." Learning rate matters, but the root cause is the format itself.

**Realistic Solution:** Gradient underflow. FP16 only has 5 bits for the exponent, giving it a very narrow dynamic range ($6 \times 10^{-8}$ to $65504$). Small gradients in deep networks underflow to exactly 0.0, causing training to collapse. You must switch to BF16 (which uses 8 bits for the exponent, matching FP32's range) or implement dynamic loss scaling.

> **Napkin Math:** FP16 smallest subnormal ≈ $6 \times 10^{-8}$. BF16 smallest subnormal ≈ $9 \times 10^{-41}$. Gradients in deep networks routinely hit $10^{-12}$ to $10^{-20}$ — well below FP16's floor but safely within BF16's range.

**📖 Deep Dive:** [Volume I: Neural Computation](https://mlsysbook.ai/vol1/nn_computation.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Precision Trade-off</b> · <code>precision</code> <code>hardware</code></summary>

**Interviewer:** "We are quantizing our model from FP16 to INT8. The memory footprint drops by 2x, but the throughput doesn't improve at all. Assuming we are compute-bound, what hardware architectural detail did we forget?"

**Common Mistake:** "INT8 should always be 2x faster because it's half the bits." This confuses memory savings with compute savings.

**Realistic Solution:** You forgot to check if the silicon has dedicated INT8 Tensor Cores. Without specific hardware paths for 8-bit integer math, the GPU must upcast the INT8 values back to FP16 in the registers to perform the multiply-accumulate (MAC), yielding zero compute speedup despite the memory savings.

> **Napkin Math:** A100 INT8 Tensor Cores: 624 TOPS. A100 FP16 Tensor Cores: 312 TFLOPS. If your hardware has INT8 paths, you get a 2x compute boost. Without them, you get 1x compute at 0.5x memory — a pure memory optimization, not a compute one.

**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

---

### 🧠 Memory Hierarchy & KV-Cache

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Sequence Length Trap</b> · <code>kv-cache</code> <code>memory</code></summary>

**Interviewer:** "You need to increase your LLM's context window from 4k to 128k tokens. The model weights fit perfectly in your 80GB VRAM. What hidden memory cost will cause your node to OOM (Out of Memory) during generation?"

**Common Mistake:** "Activations will use more memory." Activations matter during training, but during inference the dominant hidden cost is something else entirely.

**Realistic Solution:** The KV-Cache. While weights are static, the KV-cache grows linearly with sequence length and batch size. At 128k context, the memory required to store the attention keys and values for a single request will massively exceed the size of the model weights themselves.

> **Napkin Math:** For Llama 70B (80 layers, 64 heads, head_dim=128) at 128k tokens in FP16: KV-cache = $2 \times 80 \times 64 \times 128 \times 128000 \times 2$ bytes ≈ **335 GB** for a single request. The weights are only ~140 GB. The cache is 2.4× larger than the model.

> **Key Equation:** $\text{KV-cache} = 2 \times L \times H \times d_h \times S \times b \times \text{bytes}$
> where $L$ = layers, $H$ = heads, $d_h$ = head dim, $S$ = sequence length, $b$ = batch size

**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Energy-Movement Invariant</b> · <code>memory</code> <code>energy</code></summary>

**Interviewer:** "We pruned 50% of the weights from our model, cutting the total MAC (Multiply-Accumulate) operations in half. However, the energy consumption of the node barely dropped. Why?"

**Common Mistake:** "The GPU is still drawing idle power." Idle power exists, but the real issue is where the energy actually goes.

**Realistic Solution:** You forgot the Energy-Movement Invariant. Fetching a bit of data from off-chip DRAM costs roughly 100-200x more energy than the math operation (MAC) itself. If your pruning was unstructured, you still have to load the same dense matrices from memory before applying a sparse mask, yielding zero energy savings.

> **Napkin Math:** A MAC operation costs ~1 pJ. A DRAM access costs ~200 pJ. If your model does 1 TFLOP of math but moves 100 GB of data, the energy split is: compute = 1 mJ, data movement = 20 J. Data movement dominates by 20,000×. Cutting compute in half saves almost nothing.

**📖 Deep Dive:** [Volume I: Neural Computation](https://mlsysbook.ai/vol1/nn_computation.html)
</details>

---

### ⚙️ Hardware & Compilation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Compilation Overhead</b> · <code>frameworks</code></summary>

**Interviewer:** "You move a PyTorch training loop from a CPU to a GPU. The first few batches take 500ms each, but suddenly the latency drops to 10ms per batch. What happened inside the framework?"

**Common Mistake:** "The GPU was warming up" or "caches were cold." GPUs don't have a warm-up period in the CPU sense.

**Realistic Solution:** Just-In-Time (JIT) compilation overhead. The framework spends the first few iterations tracing the computation graph and compiling optimized CUDA kernels for the specific tensor shapes you provided. Once cached, the dispatch overhead disappears.

**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Sparsity Fallacy</b> · <code>hardware</code> <code>sparsity</code></summary>

**Interviewer:** "You wrote a custom CUDA kernel that skips calculating zeros (90% sparse). But it runs slower than a dense PyTorch matrix multiply that does 10x more math. Why does doing 'less work' take more time?"

**Common Mistake:** "The custom kernel must have a bug" or "We need to optimize the memory access pattern." The issue is more fundamental than code quality.

**Realistic Solution:** Hardware acceleration relies on dense, regular computation blocks. Tensor Cores and Systolic Arrays are physically hard-wired to perform $16\times16$ matrix blocks in a single instruction. Irregular sparse operations force the hardware to fall back to standard, un-fused CUDA cores, completely destroying the throughput advantage of the specialized silicon.

> **Napkin Math:** H100 Tensor Core throughput: 990 TFLOPS. H100 CUDA core throughput: ~60 TFLOPS. By going sparse and irregular, you traded a 990 TFLOPS engine for a 60 TFLOPS engine — a 16.5× slowdown that easily overwhelms the 10× reduction in operations.

**📖 Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

---

### 📊 Data Pipelines

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Data Pipeline Stall</b> · <code>data-pipeline</code></summary>

**Interviewer:** "You are training a vision model on high-resolution medical images. `nvidia-smi` shows GPU utilization fluctuating violently between 0% and 100%. What is the most likely bottleneck in your node?"

**Common Mistake:** "The model is too small for the GPU" or "We need a bigger batch size." Both assume the GPU is the problem — it's actually starving.

**Realistic Solution:** CPU Starvation (The Transformation Wall). The GPU finishes its math instantly, then sits at 0% waiting for the CPU to decode, crop, and augment the next batch of JPEGs. You must offload preprocessing to the GPU (like NVIDIA DALI) or increase your CPU worker count.

> **Napkin Math:** An H100 can process a ResNet-50 forward pass in ~2ms. JPEG decoding + augmentation on CPU takes ~10-50ms per image. With a batch of 64 images and 8 CPU workers, preprocessing takes $64 \times 30\text{ms} / 8 = 240\text{ms}$. The GPU finishes in 2ms and waits 238ms — that's 99% idle time.

**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

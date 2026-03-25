# Round 1: Single-Node Systems & Silicon Physics 🧱

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_compute_and_memory.md">🧱 1. Compute & Memory</a> ·
  <a href="02_network_and_distributed.md">🚀 2. Network & Distributed</a> ·
  <a href="03_inference_and_serving.md">⚡ 3. Inference & Serving</a> ·
  <a href="04_data_and_mlops.md">💼 4. Data & MLOps</a> ·
  <a href="05_visual_debugging.md">🖼️ 5. Visual Debugging</a> ·
  <a href="06_advanced_systems.md">⚙️ 6. Advanced Systems</a>
</div>

---

The domain of the ML Systems Engineer. This round tests your understanding of what happens *inside* a single server chassis: memory hierarchies, compute bounds, and arithmetic intensity.

> **[➕ Add a Flashcard](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/cloud/01_compute_and_memory.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

### 📐 Roofline & Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Profiling Crisis</b> · <code>roofline</code></summary>

- **Interviewer:** "You've deployed a custom recommendation model. The profiling dashboard shows you are achieving 120 TFLOPS out of a possible 300 TFLOPS on your GPU. Your tech lead suggests buying a faster GPU to fix the latency. Why is your tech lead wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "We need to optimize the CUDA kernels" or "The GPU isn't being fully utilized, so a faster one will help." Both assume the problem is compute — it isn't.

  **Realistic Solution:** The tech lead hasn't checked the Arithmetic Intensity ($Ops/Bytes$). If the model is memory-bound (intensity is lower than the ridge point of the roofline), a GPU with faster ALUs will do nothing. You must buy a GPU with higher *Memory Bandwidth (HBM)*, or optimize the model to move fewer bytes (e.g., quantization).

  > **Key Equation:** $\text{Attainable FLOPS} = \min(\text{Peak FLOPS},\ \text{Bandwidth} \times \text{Arithmetic Intensity})$

  📖 **Deep Dive:** [Volume I: HW Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Roofline Shift</b> · <code>roofline</code></summary>

- **Interviewer:** "We upgraded our serving fleet from A100s to H100s. The H100 has 3x more FP16 TFLOPS. However, our LLM decode throughput got exactly 0% faster. How is that physically possible?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The drivers must be misconfigured" or "We need to re-optimize the CUDA kernels for H100." Both miss the fundamental physics.

  **Realistic Solution:** The ridge point of the Roofline shifted right. Your workload was already memory-bandwidth bound on the A100 (Arithmetic Intensity < Ridge Point). The H100 has 3x more compute, but only 1.5x more memory bandwidth. Because LLM decoding is bandwidth-starved ($I \approx 1$), faster ALUs literally cannot do any work.

  > **Napkin Math:** A100 ridge point = 312 TFLOPS / 2 TB/s ≈ 153 Ops/Byte. H100 ridge point = 989 TFLOPS / 3.35 TB/s ≈ 295 Ops/Byte. LLM decode has $I \approx 1$ Ops/Byte — orders of magnitude below both ridge points. You're deep in the memory-bandwidth ceiling. The only path to faster decoding is more bandwidth, not more FLOPS.

  > **Key Equation:** $\text{Ridge Point} = \frac{\text{Peak FLOPS}}{\text{Memory Bandwidth}}$

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

---

### 🔢 Numerical Precision & Quantization

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Underflow Crisis</b> · <code>precision</code></summary>

- **Interviewer:** "We switched our pre-training job from FP32 to FP16 to save memory, but the loss is returning NaNs within the first 100 steps. What numerical property is failing?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The learning rate is too high for FP16." Learning rate matters, but the root cause is the format itself.

  **Realistic Solution:** Gradient underflow. FP16 only has 5 bits for the exponent, giving it a very narrow dynamic range ($6 \times 10^{-8}$ to $65504$). Small gradients in deep networks underflow to exactly 0.0, causing training to collapse. You must switch to BF16 (which uses 8 bits for the exponent, matching FP32's range) or implement dynamic loss scaling.

  > **Napkin Math:** FP16 smallest subnormal ≈ $6 \times 10^{-8}$. BF16 smallest subnormal ≈ $9 \times 10^{-41}$. Gradients in deep networks routinely hit $10^{-12}$ to $10^{-20}$ — well below FP16's floor but safely within BF16's range.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Precision Trade-off</b> · <code>precision</code> <code>hardware</code></summary>

- **Interviewer:** "We are quantizing our model from FP16 to INT8. The memory footprint drops by 2x, but the throughput doesn't improve at all. Assuming we are compute-bound, what hardware architectural detail did we forget?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "INT8 should always be 2x faster because it's half the bits." This confuses memory savings with compute savings.

  **Realistic Solution:** You forgot to check if the silicon has dedicated INT8 Tensor Cores. Without specific hardware paths for 8-bit integer math, the GPU must upcast the INT8 values back to FP16 in the registers to perform the multiply-accumulate (MAC), yielding zero compute speedup despite the memory savings.

  > **Napkin Math:** A100 INT8 Tensor Cores: 624 TOPS. A100 FP16 Tensor Cores: 312 TFLOPS. If your hardware has INT8 paths, you get a 2x compute boost. Without them, you get 1x compute at 0.5x memory — a pure memory optimization, not a compute one.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

---

### 🧠 Memory Hierarchy & KV-Cache

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Sequence Length Trap</b> · <code>kv-cache</code> <code>memory</code></summary>

- **Interviewer:** "You need to increase your LLM's context window from 4k to 128k tokens. The model weights fit perfectly in your 80GB VRAM. What hidden memory cost will cause your node to OOM (Out of Memory) during generation?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Activations will use more memory." Activations matter during training, but during inference the dominant hidden cost is something else entirely.

  **Realistic Solution:** The KV-Cache. While weights are static, the KV-cache grows linearly with sequence length and batch size. At 128k context, the memory required to store the attention keys and values for a single request will massively exceed the size of the model weights themselves.

  > **Napkin Math:** For Llama 70B (80 layers, 64 heads, head_dim=128) at 128k tokens in FP16: KV-cache = $2 \times 80 \times 64 \times 128 \times 128000 \times 2$ bytes ≈ **335 GB** for a single request. The weights are only ~140 GB. The cache is 2.4× larger than the model.

  > **Key Equation:** $\text{KV-cache} = 2 \times L \times H \times d_h \times S \times b \times \text{bytes}$
  > where $L$ = layers, $H$ = heads, $d_h$ = head dim, $S$ = sequence length, $b$ = batch size

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Energy-Movement Invariant</b> · <code>memory</code> <code>energy</code></summary>

- **Interviewer:** "We pruned 50% of the weights from our model, cutting the total MAC (Multiply-Accumulate) operations in half. However, the energy consumption of the node barely dropped. Why?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The GPU is still drawing idle power." Idle power exists, but the real issue is where the energy actually goes.

  **Realistic Solution:** You forgot the Energy-Movement Invariant. Fetching a bit of data from off-chip DRAM costs roughly 100-200x more energy than the math operation (MAC) itself. If your pruning was unstructured, you still have to load the same dense matrices from memory before applying a sparse mask, yielding zero energy savings.

  > **Napkin Math:** A MAC operation costs ~1 pJ. A DRAM access costs ~200 pJ. If your model does 1 TFLOP of math but moves 100 GB of data, the energy split is: compute = 1 J, data movement = 20 J. Data movement dominates by 20×. Cutting compute in half saves almost nothing.

  📖 **Deep Dive:** [Volume I: Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

---

### ⚙️ Hardware & Compilation

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Compilation Overhead</b> · <code>frameworks</code></summary>

- **Interviewer:** "You move a PyTorch training loop from a CPU to a GPU. The first few batches take 500ms each, but suddenly the latency drops to 10ms per batch. What happened inside the framework?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The GPU was warming up" or "caches were cold." GPUs don't have a warm-up period in the CPU sense.

  **Realistic Solution:** Just-In-Time (JIT) compilation overhead. The framework spends the first few iterations tracing the computation graph and compiling optimized CUDA kernels for the specific tensor shapes you provided. Once cached, the dispatch overhead disappears.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Sparsity Fallacy</b> · <code>hardware</code> <code>sparsity</code></summary>

- **Interviewer:** "You wrote a custom CUDA kernel that skips calculating zeros (90% sparse). But it runs slower than a dense PyTorch matrix multiply that does 10x more math. Why does doing 'less work' take more time?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The custom kernel must have a bug" or "We need to optimize the memory access pattern." The issue is more fundamental than code quality.

  **Realistic Solution:** Hardware acceleration relies on dense, regular computation blocks. Tensor Cores and Systolic Arrays are physically hard-wired to perform $16\times16$ matrix blocks in a single instruction. Irregular sparse operations force the hardware to fall back to standard, un-fused CUDA cores, completely destroying the throughput advantage of the specialized silicon.

  > **Napkin Math:** H100 Tensor Core throughput: 990 TFLOPS. H100 CUDA core throughput: ~60 TFLOPS. By going sparse and irregular, you traded a 990 TFLOPS engine for a 60 TFLOPS engine — a 16.5× slowdown that easily overwhelms the 10× reduction in operations.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

---

### 📊 Data Pipelines

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Data Pipeline Stall</b> · <code>data-pipeline</code></summary>

- **Interviewer:** "You are training a vision model on high-resolution medical images. `nvidia-smi` shows GPU utilization fluctuating violently between 0% and 100%. What is the most likely bottleneck in your node?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model is too small for the GPU" or "We need a bigger batch size." Both assume the GPU is the problem — it's actually starving.

  **Realistic Solution:** CPU Starvation (The Transformation Wall). The GPU finishes its math instantly, then sits at 0% waiting for the CPU to decode, crop, and augment the next batch of JPEGs. You must offload preprocessing to the GPU (like NVIDIA DALI) or increase your CPU worker count.

  > **Napkin Math:** An H100 can process a ResNet-50 forward pass in ~2ms. JPEG decoding + augmentation on CPU takes ~10-50ms per image. With a batch of 64 images and 8 CPU workers, preprocessing takes $64 \times 30\text{ms} / 8 = 240\text{ms}$. The GPU finishes in 2ms and waits 238ms — that's 99% idle time.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>


### 🧮 Hardware-Aware Tensor Shaping

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Silent Padding Tax</b> · <code>tpu-architecture</code></summary>

- **Interviewer:** "Your team is migrating an NLP workload from Nvidia A100s to Google TPU v4s to save costs. You are using a batch size of 15 and a hidden dimension of 768. XLA compiles the graph successfully, but the profiler shows the MXU (Matrix Multiply Unit) utilization is sitting at a dismal 11%. What is destroying your performance?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming that because the XLA compiler succeeds, the framework is efficiently mapping the math to the hardware just like CUDA does on a GPU."

  **Realistic Solution:** TPUs are built around rigid, physical systolic arrays. The TPU v4 Matrix Multiply Unit (MXU) operates strictly on 128x128 tiles of bfloat16 numbers. If your tensor dimensions are not multiples of 128, XLA does not fail—it silently pads your tensors with zeros to make them fit the physical hardware grid. A batch size of 15 is aggressively padded up to 128, meaning the MXU is spending the vast majority of its time doing math on zeros, burning power and cycle time for nothing.

  > **Napkin Math:** Your actual workload per step requires `15 * 768 = 11,520` operations. The XLA compiler pads the batch size to the nearest MXU tile boundary, forcing the hardware to compute `128 * 768 = 98,304` operations. You are achieving exactly `11,520 / 98,304 = 11.7%` utilization. The fix is to pad your batch size to 16 (for sub-tile optimization) or 128 to saturate the array.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>


### 🧠 Quantization & Underflow

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-gold?style=flat-square" alt="Level 3" align="center"> The FP8 Underflow Crash</b> · <code>numerical-precision</code></summary>

- **Interviewer:** "You are migrating a large transformer model from FP16 to FP8 (specifically the E4M3 format) for training on H100 GPUs. The model runs twice as fast, but after 500 steps, the gradients vanish entirely, and the model's loss plateaus. You verify your learning rate is correct. What numerical physics destroyed your training?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Assuming FP8 is just a smaller FP16 and that standard gradient scaling will magically handle the precision drop without adjusting the dynamic range."

  **Realistic Solution:** You suffered catastrophic underflow due to the physical limits of the E4M3 format. Gradients in deep networks are notoriously small numbers (often around $10^{-5}$ to $10^{-7}$). The FP16 format can represent numbers down to roughly $6 \times 10^{-5}$. However, FP8 E4M3 (4 exponent bits) has a much narrower dynamic range and can only represent numbers down to roughly $1.5 \times 10^{-2}$. When the small gradients were calculated, they physically could not fit into the E4M3 representation and were clamped to exactly `0.0`. Once gradients become zero, the model stops learning.

  > **Napkin Math:** A typical gradient might be $0.00045$. In FP16, this is easily represented. In FP8 E4M3, the smallest representable positive normal number is $2^{-6} = 0.015625$. Since $0.00045 < 0.015625$, the hardware flushes the value to 0. You must use massive Loss Scaling (multiplying the loss by a huge constant like $1024$ before backprop, then dividing after) to physically shift the gradients up into the representable range of the FP8 format.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>


### 🧮 Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-gold?style=flat-square" alt="Level 3" align="center"> The Strided Memory Fetch</b> · <code>memory-bandwidth</code></summary>

- **Interviewer:** "You are writing a custom CUDA kernel to perform a simple element-wise addition of two 1GB tensors. Version A reads the tensors sequentially (`A[i] + B[i]`). Version B reads the tensors with a stride of 1024 (`A[i*1024] + B[i*1024]`). Both perform the exact same number of floating-point operations. Why does Version B run 30x slower?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Thinking of memory as an array of independent bytes, ignoring the physical hardware of the memory bus and cache lines."

  **Realistic Solution:** You destroyed memory coalescing. GPUs do not fetch individual bytes from HBM; they fetch massive contiguous blocks called cache lines (typically 32 or 128 bytes at a time). In Version A, when thread 0 requests `A[0]`, the GPU pulls a 128-byte block into SRAM, meaning the next 31 threads get their data for 'free' from the ultra-fast cache. In Version B, thread 0 requests `A[0]`, pulling 128 bytes. But thread 1 requests `A[1024]`, which is in a completely different cache line. The GPU must issue a brand new fetch to HBM for every single thread. You are forcing the memory bus to transfer 128 bytes to utilize only 4 bytes, wasting 97% of your bandwidth.

  > **Napkin Math:** If HBM bandwidth is 2,000 GB/s. Version A utilizes 100% of the fetched bytes, achieving the full 2,000 GB/s effective bandwidth. Version B fetches 128 bytes to use a 4-byte float. `4 / 128 = 0.031`. Your effective bandwidth plummets to `2,000 * 0.031 = 62 GB/s`. The kernel becomes disastrously memory-bound.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>


### 💾 Compute Analysis

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-gold?style=flat-square" alt="Level 3" align="center"> The HBM vs SRAM Bandwidth Gap</b> · <code>memory-hierarchy</code></summary>

- **Interviewer:** "You profile a custom attention kernel on an A100 GPU. The kernel takes 10ms. You move a 2MB intermediate activation tensor from High Bandwidth Memory (HBM) into the GPU's Shared Memory (SRAM). The kernel time instantly drops to 2ms. Why did a seemingly small change in where a tensor is stored yield a 5x speedup?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Knowing that SRAM is faster than HBM, but failing to quantify *how much* faster, treating all 'GPU memory' as roughly the same."

  **Realistic Solution:** The physical bandwidth gap between HBM and SRAM on an A100 is over 10x. If your kernel repeatedly reads and writes that 2MB intermediate tensor (e.g., during a softmax reduction), leaving it in HBM forces the streaming multiprocessors (SMs) to traverse the slower global memory bus over and over. By explicitly pinning the tensor into the SM's local Shared Memory (which is physically located right next to the ALUs), you unlock roughly 19,000 GB/s of bandwidth compared to HBM's 1,555 GB/s. You moved the workload from being memory-bound to compute-bound.

  > **Napkin Math:** A100 HBM2 bandwidth is `~1.5 TB/s`. A100 L1/Shared Memory bandwidth (aggregate across all SMs) is `~19 TB/s`. If an algorithm makes 50 passes over a 2MB tensor, reading from HBM requires `100MB` of data movement at `1.5 TB/s` (plus massive latency per fetch). Reading from SRAM does the same movement at `19 TB/s` with near-zero latency, easily accounting for a 5x total kernel speedup.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

---

### 🆕 Extended Compute & Memory

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The HBM3e Bandwidth Ceiling</b> · <code>memory-hierarchy</code></summary>

- **Interviewer:** "Your team just got a cluster of H200 GPUs with 4.8 TB/s HBM3e, up from the H100's 3.35 TB/s. Your manager expects a 43% speedup across all workloads. After benchmarking, some models got the full speedup while others saw zero improvement. Which workloads actually benefit from the extra bandwidth, and which don't?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "More bandwidth means everything runs faster." This confuses memory-bound and compute-bound regimes. If a workload is already compute-bound on the H100, extra bandwidth is wasted silicon.

  **Realistic Solution:** Only memory-bound workloads benefit. The key is the arithmetic intensity relative to the roofline ridge point. The H100 ridge point is $989 \text{ TFLOPS} / 3.35 \text{ TB/s} \approx 295 \text{ Ops/Byte}$. The H200 ridge point (same compute, more bandwidth) is $989 / 4.8 \approx 206 \text{ Ops/Byte}$. Any workload with arithmetic intensity below 295 but above 206 was memory-bound on H100 and becomes compute-bound on H200 — these see the full 43% gain. Workloads already above 295 (large-batch GEMM training) see nothing. Workloads far below 206 (LLM decode at $I \approx 1$) see the full 43% because they remain deep in the bandwidth-limited regime.

  > **Napkin Math:** LLM decode (batch=1): $I \approx 1$ Ops/Byte. Throughput scales linearly with bandwidth: $4.8 / 3.35 = 1.43\times$ — full 43% speedup. Large-batch training GEMM (batch=256, hidden=4096): $I \approx 4096$ Ops/Byte, deep in compute-bound territory — 0% speedup. Embedding lookups: $I \approx 0.25$ Ops/Byte — full 43% speedup. The H200 is a decode accelerator, not a training accelerator.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The NVLink Domain Boundary</b> · <code>network-fabric</code></summary>

- **Interviewer:** "You're running tensor-parallel inference for a 70B model across 8 GPUs in a DGX H100. All-reduce latency is 15μs. Your team wants to scale to 16 GPUs by adding a second DGX node. After connecting them, the all-reduce latency jumps to 150μs — a 10x increase — even though you're using 400 Gbps InfiniBand. What physical boundary did you cross?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "InfiniBand is slow" or "We need to tune the network." 400 Gbps InfiniBand has excellent bandwidth, but the issue is latency, not throughput.

  **Realistic Solution:** You crossed the NVLink domain boundary. Inside a single DGX H100, 8 GPUs are connected via NVLink with 900 GB/s bisection bandwidth and ~1-2μs latency — the GPUs share a flat, switch-less memory fabric (NVSwitch). The moment you add a 9th GPU on a different node, communication must traverse: GPU → NVLink → NIC → InfiniBand switch → NIC → NVLink → GPU. Each hop adds latency. InfiniBand's one-way latency is ~1-2μs per hop, but the full software stack (NCCL → libibverbs → RDMA → remote NCCL) adds 50-100μs of overhead. For tensor parallelism, which requires an all-reduce after every layer, this latency is catastrophic.

  > **Napkin Math:** 70B model with 80 layers. Tensor-parallel all-reduce per layer: ~2 MB payload. Intra-node (NVLink): $2\text{ MB} / 900\text{ GB/s} + 1.5\mu s \approx 4\mu s$ per layer. Inter-node (IB): $2\text{ MB} / 50\text{ GB/s} + 100\mu s \approx 140\mu s$ per layer. Over 80 layers: intra-node total = $320\mu s$, inter-node total = $11.2\text{ ms}$. The inter-node path adds 35× more communication time. This is why tensor parallelism stays within a node and pipeline parallelism goes across nodes.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> The CPU-GPU Data Transfer Tax</b> · <code>memory-hierarchy</code></summary>

- **Interviewer:** "You're deploying a Llama 70B model for inference. The model weights are loaded into CPU RAM. Your server has PCIe 5.0 x16 connecting the CPU to an H100 GPU. A user sends the first request and waits 45 seconds before getting any response. Subsequent requests take only 200ms. What happened during that first request, and how do you fix it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The model needs to compile/warm up on the first request." JIT compilation adds seconds, not tens of seconds. The dominant cost is data movement.

  **Realistic Solution:** The 70B model must be physically transferred from CPU DRAM to GPU HBM over the PCIe bus before any computation can begin. PCIe 5.0 x16 has a theoretical peak of 64 GB/s, but real-world sustained throughput is typically 50-55 GB/s due to protocol overhead and TLP (Transaction Layer Packet) framing. A 70B model in FP16 is ~140 GB. That transfer dominates the cold-start latency.

  > **Napkin Math:** Model size: 70B params × 2 bytes (FP16) = 140 GB. PCIe 5.0 x16 effective throughput: ~50 GB/s. Transfer time: $140 / 50 = 2.8$ seconds. But wait — if you're loading from disk (NVMe SSD at 7 GB/s) to CPU RAM first: $140 / 7 = 20$ seconds for disk-to-RAM, then 2.8 seconds for RAM-to-GPU. Total cold start: ~23 seconds. Fix: pre-load the model into GPU memory at server startup, use model sharding across multiple GPUs to parallelize the transfer, or use memory-mapped loading with `torch.load(mmap=True)`.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The NUMA Penalty</b> · <code>memory-hierarchy</code></summary>

- **Interviewer:** "You have a dual-socket server with two AMD EPYC CPUs. Each socket has 256 GB of DDR5 RAM and one A100 GPU attached via PCIe. Your data loading pipeline reads training images from an NVMe drive attached to Socket 0, but the training runs on the GPU attached to Socket 1. The data loading throughput is 40% lower than expected. What physical topology problem are you hitting?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The NVMe is too slow" or "We need more DataLoader workers." Both ignore the physical memory topology of the server.

  **Realistic Solution:** You're paying the NUMA (Non-Uniform Memory Access) penalty. In a dual-socket server, each CPU socket has its own local memory controller. When data is read from the NVMe on Socket 0, it lands in Socket 0's local DRAM. But the GPU on Socket 1 must access that data across the inter-socket link (AMD Infinity Fabric or Intel UPI), which has roughly half the bandwidth and double the latency of local memory access. Every training batch must traverse this cross-socket hop before reaching the GPU.

  > **Napkin Math:** DDR5 local bandwidth per socket: ~200 GB/s. Cross-NUMA bandwidth (Infinity Fabric): ~100 GB/s. If each training batch is 512 MB of image data: local path = $512\text{ MB} / 200\text{ GB/s} = 2.56\text{ ms}$, cross-NUMA path = $512\text{ MB} / 100\text{ GB/s} = 5.12\text{ ms}$. That's a 2× slowdown on the data transfer alone. With 8 workers filling a prefetch queue, the cross-NUMA bottleneck caps your pipeline at $100\text{ GB/s} / 200\text{ GB/s} = 50\%$ of peak throughput. Fix: pin the DataLoader workers to the same NUMA node as the target GPU using `numactl --cpunodebind=1 --membind=1`.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Tensor Core Utilization Gap</b> · <code>compute</code> <code>architecture</code></summary>

- **Interviewer:** "You're training two transformer models on an H100. Model A has hidden_dim=768 and Model B has hidden_dim=1024. Both have the same number of layers and heads. Model B trains 25% faster per step despite having 78% more parameters. How is the bigger model faster?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Model B must have better parallelism" or "The batch sizes are different." The real issue is how tensor dimensions map to the physical hardware.

  **Realistic Solution:** H100 Tensor Cores operate on tiles of specific dimensions — FP16 requires matrix dimensions that are multiples of 8, and optimal throughput requires multiples of 64 for warp-level scheduling. Hidden_dim=1024 is a perfect power of 2 and a multiple of 64, so every Tensor Core tile is fully occupied. Hidden_dim=768 is a multiple of 8 but not 64 — the hardware must pad the last tile in each dimension, wasting compute cycles on zeros.

  > **Napkin Math:** For a GEMM of shape $[B, 768] \times [768, 768]$: the 768 dimension requires $768/64 = 12$ tiles — clean, no waste. But for the head dimension: $768 / 64 = 12$ heads of dim 64 — fine. However, many fused kernels tile on 128-wide blocks: $768 / 128 = 6.0$ — clean. Now consider $[B, 768] \times [768, 3072]$ (FFN): $3072 / 128 = 24$ — clean. The real waste comes from attention: with 12 heads of dim 64, the per-head GEMM is $[S, 64] \times [64, S]$. On 128-wide tiles, the 64-wide matrix wastes 50% of each tile. Model B with dim 1024 and 16 heads of dim 64 has the same per-head waste, but its FFN GEMMs ($[B, 1024] \times [1024, 4096]$) have 33% more arithmetic intensity, pushing it firmly into compute-bound territory where Tensor Cores run at full throughput.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The FP8 Training Frontier</b> · <code>quantization</code> <code>training</code></summary>

- **Interviewer:** "Your team successfully trained a 13B parameter LLM in FP8 on H100s with no accuracy loss. Encouraged, you try FP8 training on a 125M parameter BERT model. The loss diverges within 200 steps. Both use the same FP8 recipe with per-tensor scaling. Why does FP8 work for the large model but fail for the small one?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "FP8 is FP8 — if it works for large models, it should work for small ones too." This ignores how model size affects gradient statistics.

  **Realistic Solution:** Large models have a statistical advantage: with billions of parameters, the gradient distributions within each tensor are smooth and approximately Gaussian, making per-tensor scaling effective. A single scale factor can cover the dynamic range of the entire tensor. Small models have far fewer parameters per layer, leading to spiky, heavy-tailed gradient distributions where a few outlier values dominate. Per-tensor scaling must accommodate these outliers, crushing the majority of small gradients below FP8's representable range. The fix for small models is per-channel or per-group scaling (finer granularity), or simply using BF16.

  > **Napkin Math:** 13B model FFN layer: weight matrix is $[5120, 20480]$ = 104M elements. Gradient distribution is smooth; 99.9th percentile is ~3× the median. A single scale factor wastes $\log_2(3) \approx 1.6$ bits of range — acceptable with FP8's 4 exponent bits. 125M BERT FFN layer: weight matrix is $[768, 3072]$ = 2.4M elements. Gradient distribution has outliers at 100× the median. Per-tensor scale wastes $\log_2(100) \approx 6.6$ bits — more than FP8 E4M3's entire 4-bit exponent range. Memory savings from FP8: model states drop from $16\Phi$ bytes (Adam FP32) to $\approx 10\Phi$ bytes — a 37% reduction. Throughput gain on H100: FP8 Tensor Cores deliver 1,979 TFLOPS vs 989 TFLOPS FP16 — a 2× compute boost, but only if the numerics converge.

  📖 **Deep Dive:** [Volume I: Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Inference Batch Size Sweet Spot</b> · <code>serving</code> <code>roofline</code></summary>

- **Interviewer:** "You're serving a 7B parameter LLM on a single H100 for a chat application. At batch size 1, you get 40 tokens/second but the GPU is only 5% utilized. At batch size 256, you get 2,000 tokens/second but latency per request is 3 seconds. Your SLA requires <500ms latency. Find the batch size sweet spot using roofline analysis."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just pick the largest batch size that meets the latency SLA." This ignores the phase transition between memory-bound and compute-bound regimes and leaves throughput on the table.

  **Realistic Solution:** The sweet spot is where the workload transitions from memory-bound to compute-bound — the ridge point of the roofline. Below this point, increasing batch size is free throughput (you're bandwidth-limited, and batching increases arithmetic intensity without proportionally increasing latency). Above this point, latency grows linearly with batch size because you're now compute-limited.

  > **Napkin Math:** 7B model in FP16: 14 GB of weights. Each decode step reads all weights once. At batch=1: arithmetic intensity $I = 2 \text{ FLOPs/param} \times 1 / 2 \text{ bytes/param} = 1$ Ops/Byte (deep memory-bound). H100 ridge point: $989 \text{ TFLOPS} / 3.35 \text{ TB/s} \approx 295$ Ops/Byte. To reach the ridge: batch size $\approx 295$. But latency at batch=295: each token generation reads 14 GB and does $295 \times 2 \times 7\text{B} = 4.13\text{ TFLOPS}$. Time = $4.13\text{T} / 989\text{T/s} \approx 4.2\text{ ms}$ per token. For 100 output tokens: $420\text{ ms}$ — just under the 500ms SLA. Practical sweet spot: batch ≈ 64-128 gives 80%+ of peak throughput while keeping latency at 150-300ms with comfortable margin.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Multi-Tenant GPU Sharing Problem</b> · <code>serving</code> <code>economics</code></summary>

- **Interviewer:** "You manage an inference platform serving 7 different small models (each <10GB). Your finance team wants to consolidate them onto a single A100-80GB using MIG (Multi-Instance GPU) to save costs. Each MIG slice gets 1/7th of the GPU: ~11 GB memory, ~45 TFLOPS. After deployment, two of the seven models have 3x worse tail latency than on dedicated GPUs. What went wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "MIG provides perfect isolation, so each model should perform as if it has a dedicated 1/7th GPU." MIG isolates compute and memory capacity, but not all resources.

  **Realistic Solution:** MIG does not isolate the shared L2 cache or the memory bandwidth. The A100's 40 MB L2 cache is shared across all MIG instances. Two of the seven models are memory-bandwidth-bound (low arithmetic intensity), and they're now competing for the same 2 TB/s of HBM bandwidth. With 7 tenants, each effectively gets ~285 GB/s — far less than the 2 TB/s they had on a dedicated GPU. The compute-bound models are fine because their bottleneck (Tensor Cores) is properly partitioned.

  > **Napkin Math:** Dedicated A100 per model: $2,000 \text{ GB/s}$ bandwidth, $312 \text{ TFLOPS}$. Cost: 7 × \$2.50/hr = \$17.50/hr. MIG 1g.10gb slice: $2,000 / 7 \approx 285 \text{ GB/s}$ effective bandwidth (shared), $312/7 \approx 45 \text{ TFLOPS}$ (isolated). Cost: 1 × \$2.50/hr = \$2.50/hr. For a memory-bound model with $I = 5$ Ops/Byte: dedicated throughput = $\min(312\text{T}, 2000 \times 5) = 312\text{ TFLOPS}$. MIG throughput = $\min(45\text{T}, 285 \times 5) = 45\text{ TFLOPS}$ — a 7× slowdown, not the expected 7× from compute partitioning alone. Effective cost: \$2.50/hr but 7× slower = \$2.50 × 7 = \$17.50/hr equivalent. No savings for bandwidth-bound models.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Activation Recomputation Trade-off</b> · <code>memory-hierarchy</code> <code>training</code></summary>

- **Interviewer:** "You're training a 30B model on 8× A100-80GB GPUs. With full activation storage, you OOM at batch size 2. Your colleague suggests gradient checkpointing, claiming it 'trades a little compute for a lot of memory.' After enabling it, you can fit batch size 8. But your training time per step increased by 40%, not the 33% your colleague promised. Why is the overhead higher than expected?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Gradient checkpointing adds exactly one extra forward pass, so overhead is always 33%." This assumes the recomputed forward pass is free of memory-system effects.

  **Realistic Solution:** The theoretical 33% overhead assumes the recomputed forward pass runs at the same speed as the original. In practice, the recomputed activations must be regenerated from the checkpoint boundaries, and these recomputations compete with the backward pass for GPU resources — HBM bandwidth, L2 cache, and Tensor Core scheduling. The larger batch size (8 vs 2) also increases the memory pressure: the KV-cache and optimizer states for batch=8 consume more HBM, reducing the effective bandwidth available for activation recomputation. Additionally, each checkpoint segment must reload its input activations from HBM, adding memory traffic that didn't exist in the non-checkpointed version.

  > **Napkin Math:** Without checkpointing: activation memory for 30B model ≈ $2 \times S \times B \times H \times L \times \text{bytes}$. For sequence=2048, batch=2, hidden=7168, 60 layers in FP16: $2 \times 2048 \times 2 \times 7168 \times 60 \times 2 \approx 7\text{ GB}$. With checkpointing (every 5 layers): store only 12 checkpoints instead of 60 layers of activations. Memory drops to $\approx 7 \times 12/60 = 1.4\text{ GB}$ for activations, freeing ~5.6 GB. This allows batch=8 ($4\times$ more), but recomputation adds 48 extra layer forward passes. Theoretical overhead: $48/60 \times 50\% = 40\%$ of total step time (forward is ~50% of forward+backward). The 40% matches — the "33% rule" only applies when checkpointing every single layer.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Sequence Parallelism Necessity</b> · <code>parallelism</code> <code>memory-hierarchy</code></summary>

- **Interviewer:** "You're fine-tuning a 70B model with 128k context length using tensor parallelism across 8 H100 GPUs. The model weights fit comfortably — only 17.5 GB per GPU in FP16. But you OOM immediately. Your colleague says 'just add more tensor parallelism.' Why is tensor parallelism alone insufficient, and what additional technique do you need?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Tensor parallelism shards everything across GPUs, so 8 GPUs should have 8× the memory." Tensor parallelism shards weights and compute, but NOT activations from LayerNorm, Dropout, and the input/output embeddings.

  **Realistic Solution:** Tensor parallelism partitions the weight matrices and their corresponding GEMMs, but the activations between the partitioned regions — specifically LayerNorm inputs, dropout masks, and the residual stream — are replicated on every GPU. These "non-tensor-parallel" activations scale with sequence length and are identical on all 8 GPUs. At 128k context, they dominate memory. You need Sequence Parallelism (SP), which partitions these replicated activations along the sequence dimension across the tensor-parallel group.

  > **Napkin Math:** 70B model, 80 layers, hidden=8192, 128k sequence, batch=1, FP16. Tensor-parallel weight memory per GPU: $140\text{ GB} / 8 = 17.5\text{ GB}$. Replicated activation memory per layer (LayerNorm input + dropout + residual): $3 \times 128000 \times 8192 \times 2 \text{ bytes} = 6.3\text{ GB}$ per layer. Over 80 layers: $6.3 \times 80 = 504\text{ GB}$ — replicated on every GPU. Each H100 has 80 GB. Even with only 17.5 GB of weights, 504 GB of replicated activations is 6.3× the total GPU memory. With sequence parallelism: $504 / 8 = 63\text{ GB}$ per GPU. Combined with gradient checkpointing: feasible.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The PagedAttention Memory Savings</b> · <code>kv-cache</code> <code>serving</code></summary>

- **Interviewer:** "You're running a 13B LLM serving endpoint. With the naive KV-cache implementation, you can serve 12 concurrent requests before OOMing on an A100-80GB. After switching to vLLM with PagedAttention, you can serve 48 concurrent requests. The model weights didn't change. Where did the extra memory come from?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "PagedAttention compresses the KV-cache" or "It uses less precision." PagedAttention doesn't change the data — it changes how memory is allocated.

  **Realistic Solution:** The naive approach pre-allocates a contiguous KV-cache buffer for the maximum possible sequence length for every request. If max_seq_len=2048 but the average request only generates 400 tokens, 80% of the allocated memory is wasted as internal fragmentation. PagedAttention borrows the idea of virtual memory paging from operating systems: it allocates KV-cache in small, fixed-size blocks (pages) and maps them via a block table. Memory is allocated on-demand as tokens are generated, eliminating internal fragmentation.

  > **Napkin Math:** 13B model (40 layers, 40 heads, head_dim=128) in FP16. KV-cache per token: $2 \times 40 \times 40 \times 128 \times 2 = 819\text{ KB}$. Naive allocation (max_seq=2048): $819\text{ KB} \times 2048 = 1.64\text{ GB}$ per request. With 80 GB total and 26 GB for weights: $54\text{ GB} / 1.64\text{ GB} = 32$ max requests — but only 12 fit because of memory allocator overhead and fragmentation (60% waste). PagedAttention (avg 400 tokens, 4% waste): effective per-request = $819\text{ KB} \times 400 \times 1.04 = 341\text{ MB}$. Capacity: $54\text{ GB} / 0.341\text{ GB} \approx 158$ requests. Even accounting for block table overhead and page-level fragmentation, 48 concurrent requests is conservative.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> The Prefetch Buffer Sizing</b> · <code>memory-hierarchy</code> <code>training</code></summary>

- **Interviewer:** "You're training a vision model on ImageNet. Your GPU processes each batch in 50ms, but `nvidia-smi` shows utilization dropping to 0% every few seconds. You're using PyTorch DataLoader with `num_workers=2` and `prefetch_factor=2`. Your colleague says 'just set num_workers=32.' Why might that make things worse, and what's the right way to size the prefetch pipeline?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "More workers is always better" or "Set num_workers to the number of CPU cores." Both ignore the memory cost of prefetch buffers and the diminishing returns of over-provisioning.

  **Realistic Solution:** Each DataLoader worker maintains its own prefetch buffer in CPU pinned memory. With `prefetch_factor=2`, each worker holds 2 batches ready. The total pinned memory consumed is $\text{num\_workers} \times \text{prefetch\_factor} \times \text{batch\_memory}$. Setting `num_workers=32` with large image batches can exhaust CPU RAM, causing the OS to swap to disk — which is catastrophically slower than the original bottleneck. The right approach is to calculate the minimum workers needed to keep the GPU fed.

  > **Napkin Math:** GPU processes a batch in 50ms. Each worker takes ~200ms to load + decode + augment a batch (ImageNet JPEGs at 224×224). To keep the GPU continuously fed: need $200 / 50 = 4$ workers minimum. With `prefetch_factor=2`: 4 workers × 2 prefetched batches = 8 batches in flight. Batch size 256 at 224×224×3 in FP32: $256 \times 224 \times 224 \times 3 \times 4 = 154\text{ MB}$ per batch. Total pinned memory: $8 \times 154 = 1.23\text{ GB}$ — reasonable. At `num_workers=32`: $32 \times 2 \times 154 = 9.9\text{ GB}$ of pinned memory — wasteful, and the extra 28 workers sit idle because the GPU only consumes one batch per 50ms.

  📖 **Deep Dive:** [Volume I: Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> The Gradient Accumulation Equivalence</b> · <code>training</code> <code>memory-hierarchy</code></summary>

- **Interviewer:** "You can't fit batch size 1024 on your single A100 — it OOMs. So you use batch size 64 with 16 gradient accumulation steps to simulate an effective batch of 1024. Your colleague claims this is 'mathematically identical' to true batch 1024. Name two cases where this equivalence breaks down."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Gradient accumulation is always equivalent to large-batch training." The gradients are mathematically averaged the same way, so people assume everything else follows.

  **Realistic Solution:** Two key cases where equivalence breaks: (1) **Batch Normalization** — BatchNorm computes running mean and variance over the micro-batch (64), not the effective batch (1024). The normalization statistics are different, leading to different activations and gradients. This can cause significant accuracy degradation, especially in vision models. Fix: use SyncBatchNorm or switch to LayerNorm/GroupNorm. (2) **Stochastic regularization** — Dropout and data augmentation are applied per micro-batch. With true batch 1024, each sample sees one dropout mask per step. With 16 accumulation steps, the model sees 16 different dropout masks per effective step, subtly changing the regularization dynamics.

  > **Napkin Math:** Memory savings: true batch 1024 activation memory ≈ $1024 \times S \times H \times 2$ bytes. With accumulation at micro-batch 64: $64 \times S \times H \times 2$ bytes — a 16× reduction in peak activation memory. For a model with S=512, H=4096 in FP16: true batch activations = $1024 \times 512 \times 4096 \times 2 = 4\text{ GB}$. Micro-batch activations = $64 \times 512 \times 4096 \times 2 = 256\text{ MB}$. The trade-off: 16× more forward/backward passes means 16× longer step time, but the gradient result is identical (for models without BatchNorm).

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 6+" align="center"> The Expert Parallelism Communication</b> · <code>parallelism</code> <code>network-fabric</code></summary>

- **Interviewer:** "You're training a Mixture-of-Experts model with 64 experts, placing one expert per GPU across 64 H100s connected via 400 Gbps InfiniBand. Each token is routed to 2 experts. During training, you notice the all-to-all communication takes longer than the expert computation itself. At what point does the network become the bottleneck, and how do you fix it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "All-to-all is just like all-reduce — it scales well with more GPUs." All-to-all has fundamentally different scaling properties than all-reduce.

  **Realistic Solution:** In expert parallelism, every GPU must send a fraction of its tokens to every other GPU (the tokens routed to remote experts) and receive tokens back. This is an all-to-all communication pattern where the total data volume scales as $O(N)$ with the number of GPUs, but each GPU's network port bandwidth is fixed. With 64 GPUs, each GPU sends tokens to 63 other GPUs simultaneously, creating massive incast at the network switches. Unlike all-reduce (which can use ring or tree topologies to limit per-link traffic), all-to-all requires full bisection bandwidth.

  > **Napkin Math:** Hidden dim=4096, FP16. Each token's activation: $4096 \times 2 = 8\text{ KB}$. Batch per GPU: 4096 tokens. Top-2 routing: each token goes to 2 of 64 experts. Expected tokens sent per GPU: $4096 \times 2 \times (63/64) \approx 8064$ tokens to remote GPUs. Data sent per GPU: $8064 \times 8\text{ KB} = 63\text{ MB}$. At 400 Gbps (50 GB/s) per link: $63\text{ MB} / 50\text{ GB/s} = 1.26\text{ ms}$ — if you had a dedicated link to every peer. But with a 2-level fat-tree network, bisection bandwidth is typically 2:1 oversubscribed: effective = 25 GB/s, time = 2.52 ms. Expert compute per token (one FFN layer): $\approx 0.5\text{ ms}$. Communication (2.52 ms) > compute (0.5 ms) — network-bound. Fix: expert-parallel groups of 8 within NVLink domains, data-parallel across nodes.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 6+" align="center"> The Disaggregated Memory Architecture</b> · <code>memory-hierarchy</code> <code>architecture</code></summary>

- **Interviewer:** "Your company is evaluating CXL-attached memory pools for ML inference. The pitch: 'Expand GPU-accessible memory to terabytes at DRAM cost.' CXL 2.0 memory has ~200ns access latency vs HBM's ~50ns and local DDR5's ~100ns. For which ML workloads does CXL memory make sense, and for which is it a trap?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "200ns is close enough to 100ns DRAM — CXL memory is basically the same as local RAM." This ignores how latency compounds in bandwidth-sensitive workloads and the difference between latency and throughput.

  **Realistic Solution:** CXL memory makes sense for capacity-bound workloads with low bandwidth requirements: embedding tables for recommendation models (terabytes of parameters, sparse random access), KV-cache overflow for long-context LLM serving (sequential writes, infrequent reads), and model weight storage for rarely-accessed experts in MoE models. CXL is a trap for bandwidth-bound workloads: LLM decode (needs maximum streaming bandwidth), training (activations streamed at full speed), and any workload where the memory bus is the bottleneck.

  > **Napkin Math:** CXL 2.0 bandwidth per link: ~64 GB/s (PCIe 5.0 x16). HBM3 on H100: 3,350 GB/s. Ratio: CXL is 52× slower in bandwidth. For embedding lookup (recommendation model): 1000 lookups/batch × 128-dim × 4 bytes = 512 KB per batch. At 64 GB/s: $0.5\text{ MB} / 64\text{ GB/s} = 8\mu s$ — acceptable for a 10ms inference budget. For LLM decode (7B model): must stream 14 GB of weights per token. At 64 GB/s: $14 / 64 = 219\text{ ms}$ per token — catastrophic (HBM does it in $14 / 3350 = 4.2\text{ ms}$). CXL makes the decode 52× slower. For KV-cache overflow: writing 819 KB per token at 64 GB/s = $13\mu s$ — fine for prefill, where you generate many tokens in parallel and the write is amortized.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 3" align="center"> The Warmup Learning Rate Schedule</b> · <code>training</code></summary>

- **Interviewer:** "You're training a transformer with batch size 4096 on 32 GPUs. Without learning rate warmup, the loss explodes to NaN within 50 steps. With a 2000-step linear warmup, training is stable. Your colleague says 'warmup is just a training trick.' What is the systems-level reason warmup is physically necessary for large-batch training?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Warmup helps the optimizer explore the loss landscape slowly." This is the ML intuition, but it misses the numerical precision reason.

  **Realistic Solution:** The systems reason is gradient variance and numerical precision. At initialization, weights are random, so gradients have extremely high variance across the batch. With a large batch of 4096, the gradient estimate is a sum of 4096 highly variable terms. If the learning rate is large, the weight update magnitude ($\text{lr} \times \text{gradient}$) can exceed the representable range of FP16/BF16, causing overflow → NaN. Warmup keeps the update magnitude small while the gradient variance is high (early training), then increases the learning rate as the model converges toward a region where gradients become more consistent and smaller in magnitude.

  > **Napkin Math:** Random initialization: gradient std ≈ $1/\sqrt{d} \approx 1/\sqrt{4096} \approx 0.016$. With batch 4096, gradient mean estimate has std $\approx 0.016 / \sqrt{4096} = 0.00025$. But outlier gradients can be 10-100× larger: $0.025$. With lr=1e-3: update = $0.025 \times 0.001 = 2.5 \times 10^{-5}$ — safe. With lr=1e-1 (target): update = $0.025 \times 0.1 = 0.0025$ — still safe for weights, but the Adam moment estimates in FP16 can overflow when accumulating squared gradients: $(0.025)^2 \times 4096 = 2.56$ — this approaches FP16 max ($65504$) when multiplied across layers. After warmup, gradient variance drops 10-100× as the model leaves the random initialization regime.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The ZeRO-3 Communication Overhead</b> · <code>parallelism</code> <code>network-fabric</code></summary>

- **Interviewer:** "You're training a 175B parameter model using ZeRO Stage 3 (DeepSpeed) across 64 A100-80GB GPUs. ZeRO-3 shards weights, gradients, AND optimizer states across all GPUs, so each GPU only stores 1/64th of the model. But your training throughput is 40% lower than ZeRO Stage 1 (which only shards optimizer states). Where is the time going?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "ZeRO-3 just uses less memory per GPU — communication should be the same." This confuses memory savings with communication cost. ZeRO-3 trades memory for bandwidth.

  **Realistic Solution:** ZeRO-3 must perform an all-gather to reconstruct the full weight tensor before every forward and backward layer computation, then a reduce-scatter to redistribute gradients after each layer's backward pass. This means 3× the communication volume of ZeRO-1 (which only communicates gradients once at the end of the backward pass). For a 175B model, this communication happens for every single layer, every single step.

  > **Napkin Math:** 175B model in FP16 = 350 GB. ZeRO-1 communication per step: one all-reduce of gradients = $2 \times 350\text{ GB}$ (reduce-scatter + all-gather) = 700 GB. ZeRO-3 communication per step: for each of ~96 layers: one all-gather (forward) + one all-gather (backward) + one reduce-scatter (gradients) = $3 \times 350\text{ GB}$ = 1,050 GB total. With 64 GPUs on 8 nodes (8 GPUs/node), inter-node bandwidth = 400 Gbps = 50 GB/s per link. ZeRO-1: $700 / 50 \approx 14\text{ s}$ communication (overlapped with compute). ZeRO-3: $1050 / 50 \approx 21\text{ s}$, but critically, the all-gathers cannot be fully overlapped because each layer needs its weights before computing. Memory savings: ZeRO-1 stores full weights (350 GB) + 1/64th optimizer (82 GB) = 432 GB — doesn't fit on 80 GB. ZeRO-3 stores 1/64th of everything: $(350 + 350 + 700) / 64 = 21.9\text{ GB}$ — fits easily.

  📖 **Deep Dive:** [Volume I: Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Spot Instance Checkpoint Strategy</b> · <code>fault-tolerance</code> <code>economics</code></summary>

- **Interviewer:** "You're training a 7B model on 8× H100 spot instances on AWS. Spot instances cost \$8/hr (vs \$25/hr on-demand) but can be preempted with a 2-minute warning. Your training run takes 72 hours. Checkpointing takes 3 minutes and pauses training. How often should you checkpoint, and what's the expected cost savings?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Checkpoint every step to minimize lost work" or "Checkpoint every hour like on-demand." Too frequent checkpointing wastes compute on I/O; too infrequent risks losing hours of work.

  **Realistic Solution:** This is a classic cost-optimization problem. The optimal checkpoint interval balances the cost of checkpointing (paused training) against the expected cost of lost work (time since last checkpoint when preemption occurs). AWS spot interruption rate for GPU instances is roughly 5-10% per hour. The optimal interval minimizes: $\text{checkpoint overhead} + \text{expected lost work}$.

  > **Napkin Math:** Spot interruption probability: ~5%/hr. Checkpoint cost: 3 min pause = 0.05 hr. If checkpoint interval = $T$ hours: checkpoints per 72 hr = $72/T$. Total checkpoint overhead = $72/T \times 0.05\text{ hr}$. Expected preemptions in 72 hr: $72 \times 0.05 = 3.6$ events. Expected lost work per preemption: $T/2$ hours (uniform distribution). Total expected lost work: $3.6 \times T/2 = 1.8T$ hours. Total wasted time: $f(T) = 3.6/T + 1.8T$. Minimize: $f'(T) = -3.6/T^2 + 1.8 = 0 \Rightarrow T = \sqrt{2} \approx 1.4$ hours. Optimal: checkpoint every ~85 minutes. Cost comparison: On-demand: $72 \times \$25 = \$1,800$. Spot with optimal checkpointing: effective training time = $72 + 3.6/1.4 \times 0.05 + 1.8 \times 1.4 \approx 74.6$ hours. Cost = $74.6 \times \$8 = \$597$. Savings: 67%.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 4" align="center"> The Inference Compiler Optimization</b> · <code>compiler-runtime</code> <code>serving</code></summary>

- **Interviewer:** "You compile your 7B LLM with TensorRT before deploying on an H100. The model has 1,200 individual operations in eager PyTorch. After TensorRT compilation, it's reduced to 80 fused kernels. Inference latency drops from 15ms to 6ms. Your colleague says 'operator fusion just reduces Python overhead.' What is the real hardware-level reason fusion is so effective?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Fusion eliminates Python interpreter overhead" or "It reduces function call overhead." Python overhead is real but small (~microseconds). The dominant savings come from the memory hierarchy.

  **Realistic Solution:** Each CUDA kernel launch requires: (1) writing the input tensor to HBM, (2) launching the kernel (5-10μs overhead), (3) reading the input from HBM, computing, (4) writing the output to HBM. When two sequential operations are separate kernels, the intermediate tensor makes a round-trip to HBM between them. Fusion eliminates these intermediate HBM round-trips by keeping data in registers and shared memory (SRAM). For a chain of 15 elementwise operations (LayerNorm → GELU → Dropout → Add), unfused execution writes and reads the full activation tensor 15 times from HBM. Fused execution reads once, computes everything in SRAM, and writes once.

  > **Napkin Math:** 7B model, sequence=2048, hidden=4096, FP16. Activation tensor per layer: $2048 \times 4096 \times 2 = 16\text{ MB}$. Unfused: 1,200 kernels, assume ~800 intermediate HBM round-trips. Data moved: $800 \times 16\text{ MB} \times 2 \text{ (read+write)} = 25.6\text{ GB}$. At 3.35 TB/s: $25.6 / 3350 = 7.6\text{ ms}$ just for intermediate data movement. Fused: 80 kernels, ~60 intermediate round-trips. Data moved: $60 \times 16 \times 2 = 1.92\text{ GB}$. Time: $1.92 / 3350 = 0.57\text{ ms}$. Savings from reduced data movement alone: $7.6 - 0.57 = 7.0\text{ ms}$. Kernel launch overhead savings: $(1200 - 80) \times 7\mu s = 7.8\text{ ms}$. Total: ~15ms → ~6ms matches observed speedup.

  📖 **Deep Dive:** [Volume I: ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 5" align="center"> The Memory Bandwidth Roofline Shift</b> · <code>roofline</code> <code>architecture</code></summary>

- **Interviewer:** "Your company is planning GPU purchases for the next 3 years. You need to decide between A100 (2 TB/s, 312 TFLOPS), H100 (3.35 TB/s, 989 TFLOPS), and H200 (4.8 TB/s, 989 TFLOPS). Your workload mix is 60% LLM inference (decode-heavy), 30% training (large-batch GEMM), and 10% embedding lookups. Which GPU gives the best performance-per-dollar for this mix, and why does the roofline analysis change the answer from what you'd expect?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The H100 has 3× the FLOPS of the A100, so it's 3× faster for everything." This ignores that the workload mix is dominated by memory-bound operations.

  **Realistic Solution:** The roofline ridge points tell the story. A100: $312/2.0 = 156$ Ops/Byte. H100: $989/3.35 = 295$ Ops/Byte. H200: $989/4.8 = 206$ Ops/Byte. The H100 actually made the memory-bound problem worse — it added 3× compute but only 1.7× bandwidth, pushing the ridge point higher. The H200 corrected this by adding bandwidth without more compute, pulling the ridge point back down. For a decode-heavy workload mix, the H200 is the clear winner despite having identical FLOPS to the H100.

  > **Napkin Math:** LLM decode ($I \approx 1$, 60% of workload): A100 throughput = $\min(312\text{T}, 2000 \times 1) = 2000$ Gops/s → bandwidth-limited. H100 = $\min(989\text{T}, 3350) = 3350$ → 1.68× A100. H200 = $\min(989\text{T}, 4800) = 4800$ → 2.4× A100. Training GEMM ($I \approx 500$, 30%): A100 = $312\text{T}$ (compute-bound). H100 = $989\text{T}$ → 3.17× A100. H200 = $989\text{T}$ → 3.17× A100. Embedding ($I \approx 0.25$, 10%): A100 = $500$ Gops/s. H100 = $838$ → 1.68×. H200 = $1200$ → 2.4×. Weighted speedup over A100: H100 = $0.6(1.68) + 0.3(3.17) + 0.1(1.68) = 2.13\times$. H200 = $0.6(2.4) + 0.3(3.17) + 0.1(2.4) = 2.63\times$. The H200 wins by 23% over the H100 for this workload mix, entirely due to the bandwidth advantage on the 70% of workloads that are memory-bound.

  📖 **Deep Dive:** [Volume I: Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

---

### 🆕 Napkin Math Drills

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The GEMM Bandwidth Ceiling</b> · <code>roofline</code></summary>

- **Interviewer:** "You're running a matrix multiply $C = A \times B$ where $A$ is $[4096 \times 4096]$ and $B$ is $[4096 \times 4096]$ in FP16 on an H100. Calculate the arithmetic intensity and determine whether this operation is compute-bound or memory-bound."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "A 4096×4096 GEMM is always compute-bound on a GPU." This is usually true for large GEMMs, but you must verify by computing the arithmetic intensity and comparing to the ridge point.

  **Realistic Solution:** Arithmetic intensity = total FLOPs / total bytes moved. For a square GEMM of dimension $N$: FLOPs = $2N^3$ (each output element requires $N$ multiply-adds). Bytes = $3N^2 \times \text{element\_size}$ (read $A$, read $B$, write $C$). The intensity grows linearly with $N$, so larger matrices are more compute-bound.

  > **Napkin Math:** FLOPs = $2 \times 4096^3 = 137.4 \times 10^9 = 137.4$ GFLOPS. Bytes = $3 \times 4096^2 \times 2 = 100.7$ MB. Arithmetic intensity = 137.4 GFLOP / 0.1007 GB = **1,364 Ops/Byte**. H100 ridge point (BF16) = 989 TFLOPS / 3.35 TB/s = 295 Ops/Byte. Since 1,364 >> 295, this is **solidly compute-bound**. Expected time: 137.4 GFLOP / 989 TFLOPS = 0.14 ms. Now compare: a $[512 \times 4096] \times [4096 \times 4096]$ GEMM has intensity = $2 \times 512 \times 4096^2 / ((512 \times 4096 + 4096^2 + 512 \times 4096) \times 2) \approx 409.6$ Ops/Byte — still compute-bound on BF16 but memory-bound on INT8 (ridge = 591).

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Cache Line Waste</b> · <code>memory-hierarchy</code></summary>

- **Interviewer:** "You have a tensor of shape $[1024, 3]$ stored in row-major (C-contiguous) FP32 format. You iterate over it column-by-column (reading all 1024 elements of column 0, then column 1, etc.). The CPU cache line is 64 bytes. What fraction of each cache line fetch is wasted, and how does this compare to row-major iteration?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Cache lines don't matter for GPU workloads" or "The stride is only 3 elements, so it's almost sequential." The stride of 3 FP32 elements = 12 bytes, which means each 64-byte cache line contains data from ~5 rows, but you only need 1 element per fetch when iterating column-wise.

  **Realistic Solution:** In row-major layout, consecutive elements of a row are contiguous in memory. When you iterate column-wise, consecutive accesses are separated by the row stride (3 elements × 4 bytes = 12 bytes). Each cache line (64 bytes) holds 16 FP32 values = 5.33 rows of data. But column iteration only uses 1 element per row, so you use 1 out of every 3 elements fetched — 67% waste. For row-major iteration, you use all 16 elements per cache line — 0% waste.

  > **Napkin Math:** Column iteration: each access fetches 64 bytes but uses 4 bytes → **93.75% waste** (1/16 utilization per cache line). Wait — that's if stride exceeds cache line. Here stride = 12 bytes < 64 bytes, so each cache line serves $\lfloor 64/12 \rfloor = 5$ useful accesses before eviction. Utilization: $5 \times 4 / 64 = 31.25\%$ — **68.75% waste**. Row iteration: 16 consecutive FP32 values per cache line, all used → **0% waste**. For the full tensor: column iteration fetches $1024 \times 3 \times 4 / 0.3125 = 39.3$ KB effective. Row iteration fetches $1024 \times 3 \times 4 = 12.3$ KB. Column-wise reads **3.2× more data** from memory. On GPU, the analogous problem is coalesced vs uncoalesced global memory access — uncoalesced access wastes up to 97% of bandwidth.

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Attention Arithmetic Intensity</b> · <code>roofline</code> <code>kv-cache</code></summary>

- **Interviewer:** "Calculate the arithmetic intensity of the self-attention mechanism for a single head during the decode phase (single new token, attending to $S$ cached tokens). Is attention compute-bound or memory-bound during decoding? How does this change during prefill?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Attention is always compute-bound because it's $O(n^2)$." The quadratic scaling applies to sequence length, but during decode you're computing attention for a single query token against the cached keys — this is a matrix-vector product, not a matrix-matrix product.

  **Realistic Solution:** During decode, attention for one head computes: (1) $q \cdot K^T$ — a vector-matrix product: $[1 \times d_h] \times [d_h \times S]$. (2) Softmax over $S$ scores. (3) $\text{scores} \times V$ — a vector-matrix product: $[1 \times S] \times [S \times d_h]$. Both are matrix-vector products with arithmetic intensity ~1 Op/Byte — deeply memory-bound.

  > **Napkin Math:** Decode (single token, $S=4096$, $d_h=128$, FP16): FLOPs = $2 \times 1 \times 128 \times 4096 + 2 \times 1 \times 4096 \times 128 = 2.1$ MFLOP. Bytes = KV-cache read: $2 \times 4096 \times 128 \times 2 = 2.1$ MB. Intensity = 2.1 MFLOP / 2.1 MB = **1.0 Ops/Byte** → deeply **memory-bound**. Prefill ($S=4096$ query tokens): FLOPs = $2 \times 4096 \times 128 \times 4096 + 2 \times 4096 \times 4096 \times 128 = 8.6$ GFLOP. Bytes = Q, K, V reads: $3 \times 4096 \times 128 \times 2 = 3.1$ MB. Intensity = 8.6 GFLOP / 0.0031 GB = **2,773 Ops/Byte** → solidly **compute-bound**. This is why FlashAttention matters most during prefill (reduces memory writes) while decode optimization focuses on KV-cache compression and batching.

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Adam Memory Multiplier</b> · <code>memory-hierarchy</code> <code>training</code></summary>

- **Interviewer:** "You want to fine-tune a 7B model using Adam optimizer with mixed-precision training (BF16 forward/backward, FP32 optimizer). Your GPU has 80 GB of VRAM. How much memory does the optimizer alone consume, and what fraction of VRAM does it take?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The optimizer just stores gradients — same size as the model." Adam stores far more than gradients.

  **Realistic Solution:** Mixed-precision Adam maintains: (1) FP32 master copy of weights (for numerical stability), (2) FP32 first moment (running mean of gradients), (3) FP32 second moment (running mean of squared gradients). That's 3 copies of the model in FP32, plus the BF16 working copy and BF16 gradients.

  > **Napkin Math:** BF16 weights: 7B × 2 = 14 GB. BF16 gradients: 7B × 2 = 14 GB. FP32 master weights: 7B × 4 = 28 GB. FP32 first moment: 7B × 4 = 28 GB. FP32 second moment: 7B × 4 = 28 GB. **Optimizer states alone: 84 GB.** Total: 14 + 14 + 84 = **112 GB** — 1.4× the VRAM of an 80 GB GPU. The optimizer is 75% of total memory. Solutions: AdaFactor (no second moment → saves 28 GB), 8-bit Adam (moments in INT8 → saves 42 GB), or ZeRO Stage 2 (shard optimizer across GPUs).

  📖 **Deep Dive:** [Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The INT8 Throughput Advantage</b> · <code>quantization</code> <code>roofline</code></summary>

- **Interviewer:** "Your colleague claims 'INT8 inference is always 2× faster than FP16 because the data is half the size.' On an H100 with dedicated INT8 Tensor Cores, is this claim correct? When is it wrong?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "2× smaller data = 2× faster" assumes the speedup comes from memory savings alone. The actual speedup depends on whether the workload is compute-bound or memory-bound.

  **Realistic Solution:** The H100 has 989 TFLOPS for FP16 and 1,979 TOPS for INT8 — a 2× compute advantage. It also has 2× memory efficiency (half the bytes per element). But the speedup you actually get depends on the bottleneck. If compute-bound: up to 2× from doubled TOPS. If memory-bound: up to 2× from halved data size. If balanced: somewhere in between. The claim is approximately right but for the wrong reason — and it breaks down for operations that can't use Tensor Cores (e.g., softmax, layer norm, which remain in FP16/FP32).

  > **Napkin Math:** Large-batch GEMM ($I = 500$ Ops/Byte): FP16 → compute-bound at 989 TFLOPS. INT8 → still compute-bound (500 < 591 ridge? Actually 500 is close). Speedup ≈ 1,979/989 = **2.0×**. LLM decode ($I = 1$ Ops/Byte): FP16 → bandwidth-bound, throughput = 3.35 TB/s × 1 = 3.35 TOPS. INT8 → bandwidth-bound, but reads half the bytes, so effective throughput = 3.35 TB/s × 2 = 6.7 TOPS. Speedup = **2.0×**. Softmax/LayerNorm (remain in FP32): speedup = **1.0×**. End-to-end transformer inference: ~70% time in GEMMs (2× faster), ~30% in non-quantized ops (1× speed). Net speedup: $0.7 \times 2 + 0.3 \times 1 = 1.7\times$ — not 2×.

  📖 **Deep Dive:** [Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Data Loading Wall</b> · <code>data-pipeline</code> <code>memory-hierarchy</code></summary>

- **Interviewer:** "You're training a vision model on 8× A100 GPUs. Each GPU processes a batch of 64 images (224×224×3, FP32) every 200ms. Your training data sits on a networked filesystem (NFS) with 2 GB/s read throughput. Is the data pipeline a bottleneck?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Images are small, so NFS is fine." This ignores the aggregate bandwidth demand of 8 GPUs running at full speed.

  **Realistic Solution:** You must calculate the data consumption rate of the GPU cluster and compare it to the storage throughput. If the GPUs consume data faster than storage can deliver, the GPUs starve and utilization drops.

  > **Napkin Math:** Per image (raw): 224 × 224 × 3 × 4 bytes = 602 KB. Per batch: 64 × 602 KB = 37.6 MB. Per GPU per second: 37.6 MB / 0.2 sec = 188 MB/s. 8 GPUs: 8 × 188 = **1,504 MB/s ≈ 1.5 GB/s**. NFS throughput: 2 GB/s. Headroom: only 33%. But images are typically stored as compressed JPEG (~50 KB each), so raw read is lower: 64 × 50 KB / 0.2s × 8 = 128 MB/s — NFS handles this easily. The bottleneck shifts to **CPU decoding**: JPEG decode at ~5 ms/image × 64 images = 320 ms per batch on 1 core. Need 64 images / 200ms = 320 images/sec per GPU × 8 = 2,560 images/sec. At 5ms/decode: need 13 CPU cores. Typical server has 64 cores → fine. But with augmentation (random crop, color jitter): 15ms/image → need 38 cores. Now you're CPU-bound.

  📖 **Deep Dive:** [Data Engineering](https://harvard-edge.github.io/cs249r_book_dev/contents/data_engineering/data_engineering.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Profiler Trace Puzzle</b> · <code>roofline</code> <code>monitoring</code></summary>

- **Interviewer:** "You profile a transformer training step on an H100 and see: GEMM kernels occupy 45% of wall time, attention kernels 20%, elementwise/normalization 10%, and 25% is 'gaps' (no kernel running). The achieved TFLOPS during GEMM kernels is 790 out of 989 peak. What's the overall GPU compute utilization, and where should you optimize first?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "GPU utilization is 790/989 = 80% — that's great." This only measures utilization during GEMM kernels, ignoring the 55% of time spent on non-GEMM work and gaps.

  **Realistic Solution:** Overall utilization must account for all time, not just the time when compute-heavy kernels are running. The 25% gap time (no kernel running) is pure waste — caused by CPU-side overhead (Python, framework dispatch), synchronization barriers, or data loading stalls.

  > **Napkin Math:** Effective FLOPS during each phase: GEMM (45% of time): 790 TFLOPS. Attention (20%): ~500 TFLOPS (memory-bound portions drag it down). Elementwise (10%): ~50 TFLOPS (bandwidth-bound, low arithmetic intensity). Gaps (25%): 0 TFLOPS. Weighted average: $0.45 \times 790 + 0.20 \times 500 + 0.10 \times 50 + 0.25 \times 0 = 355.5 + 100 + 5 + 0 = 460.5$ TFLOPS. **Overall utilization: 460.5 / 989 = 46.6%**. Optimization priority: (1) **Eliminate gaps** (25% → 0% with torch.compile, CUDA graphs, or overlapped data loading) — potential 33% speedup. (2) **Fuse elementwise ops** into GEMM kernels — saves memory round-trips. (3) **FlashAttention** for attention kernels — reduces memory-bound overhead. Fixing gaps alone would push utilization to $460.5 / (989 \times 0.75) = 62\%$.

  📖 **Deep Dive:** [Benchmarking](https://harvard-edge.github.io/cs249r_book_dev/contents/benchmarking/benchmarking.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Gradient Checkpoint Trade-off</b> · <code>memory-hierarchy</code> <code>training</code></summary>

- **Interviewer:** "You're training a 32-layer transformer and running out of memory. Your colleague suggests gradient checkpointing. How much memory does it save, and what's the compute cost? Calculate both for checkpointing every $k$ layers."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Gradient checkpointing saves all activation memory." It doesn't save all — it trades recomputation for memory by only storing activations at checkpoint boundaries.

  **Realistic Solution:** Without checkpointing, you store activations for all 32 layers. With checkpointing every $k$ layers, you store activations at $32/k$ checkpoint boundaries plus recompute the $k-1$ layers between checkpoints during the backward pass. Memory scales as $O(32/k + k)$ instead of $O(32)$. The optimal $k = \sqrt{32} \approx 6$ minimizes the sum.

  > **Napkin Math:** Per-layer activation size for a transformer (batch=8, seq=2048, hidden=4096, FP16): $8 \times 2048 \times 4096 \times 2 = 128$ MB. 32 layers: **4,096 MB = 4 GB** of activations. **No checkpointing:** 4 GB activations, 0% recompute overhead. **Checkpoint every $k=4$:** Store 8 checkpoints × 128 MB + 3 layers of recompute buffer = 1,408 MB ≈ **1.4 GB**. Savings: 65%. Recompute: 75% of forward pass → ~33% total training time increase. **Checkpoint every $k=6$ (optimal):** Store ~5.3 checkpoints × 128 MB + 5 layers buffer = 1,320 MB ≈ **1.3 GB**. Savings: 68%. Recompute: 83% of forward → ~29% overhead (optimal balances memory and compute). **Checkpoint every layer ($k=32$):** Store 1 checkpoint + recompute all 31 layers = 128 + 128 = **256 MB**. Savings: 94%. Recompute: 97% of forward → ~50% overhead.

  📖 **Deep Dive:** [Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The AllReduce Tax</b> · <code>network-fabric</code> <code>parallelism</code></summary>

- **Interviewer:** "You're training a 7B model with data parallelism across 8 GPUs connected via NVLink (900 GB/s bidirectional). After each backward pass, you must AllReduce 14 GB of gradients (FP16). How long does the AllReduce take, and what fraction of the training step is spent on communication?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "14 GB / 900 GB/s = 15.6 ms — trivial." This uses the raw bandwidth and ignores the ring AllReduce algorithm's communication volume.

  **Realistic Solution:** Ring AllReduce sends $2 \times (N-1)/N \times \text{data\_size}$ total bytes per GPU, where $N$ is the number of GPUs. For $N=8$: each GPU sends and receives $2 \times 7/8 \times 14 = 24.5$ GB. The effective bandwidth is limited by the slowest link in the ring.

  > **Napkin Math:** Ring AllReduce volume per GPU: $2 \times (7/8) \times 14$ GB = 24.5 GB. NVLink bandwidth (unidirectional, per GPU): 450 GB/s. AllReduce time: 24.5 / 450 = **54.4 ms**. Typical training step (forward + backward) for 7B model, batch=32, seq=2048: ~200 ms on 8× H100. Communication fraction: 54.4 / (200 + 54.4) = **21.4%**. With gradient overlap (start AllReduce for early layers while computing backward for later layers): effective communication time ≈ 15 ms (only the last bucket is exposed). Fraction drops to: 15 / 215 = **7%**. Without overlap: 21% of training is pure communication. This is why frameworks like PyTorch DDP use bucketed, overlapped AllReduce by default.

  📖 **Deep Dive:** [Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Batch Size Sweet Spot</b> · <code>roofline</code> <code>serving</code></summary>

- **Interviewer:** "You're serving a 7B LLM on a single H100. At batch=1, you measure 35 tokens/sec. At batch=64, you measure 1,800 tokens/sec. Why does throughput increase 51× with only 64× more work? At what batch size do you expect diminishing returns?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Throughput should scale linearly with batch size." It does initially (when memory-bound), but saturates when the workload becomes compute-bound.

  **Realistic Solution:** At batch=1, LLM decode is deeply memory-bound (intensity ≈ 1 Op/Byte). The GPU reads all weights from HBM for each token but does minimal compute. Increasing batch size reuses the same weight reads across multiple requests, increasing arithmetic intensity linearly. Once intensity exceeds the ridge point, the workload becomes compute-bound and throughput stops scaling.

  > **Napkin Math:** 7B model, FP16 = 14 GB weights. H100: 3.35 TB/s bandwidth, 989 TFLOPS. Batch=1: intensity = $2 \times 7\text{B} / 14\text{GB} = 1$ Op/Byte. Throughput = 3.35T × 1 / (2 × 7B) = 239 tokens/sec theoretical (35 actual due to overhead). Batch=$B$: intensity = $B$ Ops/Byte. Ridge point = 295 Ops/Byte. **Diminishing returns start at $B \approx 295$** — but you'll hit VRAM limits first. KV-cache per request at 2k context: ~0.5 GB. At batch=64: 32 GB KV-cache + 14 GB weights = 46 GB → fits in 80 GB. At batch=295: 147 GB KV-cache → **OOM**. Practical maximum batch: $(80 - 14 - 2) / 0.5 \approx 128$. Throughput at batch=128: ~3,400 tokens/sec. The 51× scaling at batch=64 reflects moving from 1/295 = 0.34% compute utilization to 64/295 = 21.7%.

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Transformer FLOP Count</b> · <code>roofline</code> <code>compute</code></summary>

- **Interviewer:** "Calculate the total FLOPs for a single forward pass through a transformer layer with hidden dimension $H=4096$, sequence length $S=2048$, batch size $B=8$, and 32 attention heads. Break it down by component: QKV projection, attention scores, attention output, and MLP."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just count the attention as $O(S^2)$ and call it a day." This misses that the MLP typically dominates FLOPs in modern transformers (it's 2/3 of the compute), and the QKV projections are large GEMMs.

  **Realistic Solution:** A standard transformer layer has four major GEMM operations plus the attention score computation. Each GEMM of shape $[M \times K] \times [K \times N]$ costs $2MKN$ FLOPs (multiply + add).

  > **Napkin Math:** Let $B=8$, $S=2048$, $H=4096$, $d_h=128$, heads=32. Tokens = $B \times S = 16{,}384$. **(1) QKV projection** ($[BS \times H] \times [H \times 3H]$): $2 \times 16384 \times 4096 \times 12288 = 1.65$ TFLOP. **(2) Attention scores** ($[BS \times S] \times [S \times d_h]$ per head, 32 heads): $32 \times 2 \times 8 \times 2048 \times 2048 \times 128 = 274.9$ GFLOP. **(3) Attention output projection** ($[BS \times H] \times [H \times H]$): $2 \times 16384 \times 4096 \times 4096 = 549.8$ GFLOP. **(4) MLP** (up-project $H \to 4H$, down-project $4H \to H$): $2 \times (2 \times 16384 \times 4096 \times 16384) = 4.40$ TFLOP. **Total per layer: 1.65T + 0.27T + 0.55T + 4.40T ≈ 6.87 TFLOP.** MLP is **64%** of compute. For 32 layers: 220 TFLOP. On H100 at 989 TFLOPS: 220 / 989 = **222 ms** minimum (compute-bound).

  📖 **Deep Dive:** [Neural Computation](https://harvard-edge.github.io/cs249r_book_dev/contents/neural_computation/neural_computation.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Decode Bandwidth Demand</b> · <code>memory-hierarchy</code> <code>kv-cache</code></summary>

- **Interviewer:** "During LLM decoding, every output token requires reading the full KV-cache in addition to the model weights. For a 13B model serving a single request at 8k context length, calculate the total memory bandwidth consumed per token and the resulting tokens-per-second on an A100 (2 TB/s)."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Only the model weights need to be read — the KV-cache is small." At 8k context, the KV-cache is no longer negligible and adds significant bandwidth demand per token.

  **Realistic Solution:** Each decode step reads: (1) all model weights (for the linear projections), and (2) the full KV-cache (for attention computation). Both are streamed from HBM. The total bandwidth demand per token is the sum of both.

  > **Napkin Math:** 13B model in FP16: 26 GB weights. KV-cache at 8k context (40 layers, 40 heads, dim 128, FP16): $2 \times 40 \times 40 \times 128 \times 8192 \times 2 = 6.7$ GB. Total bytes read per token: 26 + 6.7 = **32.7 GB**. A100 bandwidth: 2 TB/s. Time per token: 32.7 / 2,000 = **16.4 ms**. Tokens/sec: **~61 tokens/sec**. Without KV-cache overhead (weights only): 26 / 2,000 = 13 ms → 77 tokens/sec. The KV-cache adds **26% latency** at 8k context. At 32k context: KV-cache = 26.8 GB, total = 52.8 GB, time = 26.4 ms → 38 tokens/sec. The KV-cache overhead doubles, cutting throughput by 38%.

  📖 **Deep Dive:** [Frameworks](https://harvard-edge.github.io/cs249r_book_dev/contents/frameworks/frameworks.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Quantization Noise Floor</b> · <code>quantization</code> <code>roofline</code></summary>

- **Interviewer:** "You're quantizing a model to INT8 (256 levels). The weight distribution is approximately Gaussian with mean 0 and standard deviation 0.02. Calculate the signal-to-quantization-noise ratio (SQNR) and explain at what point adding more quantization levels stops helping."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "More bits always means better quality — go to INT16 if INT8 isn't good enough." Beyond a certain point, quantization noise drops below other noise sources (gradient noise, data noise) and additional bits are wasted.

  **Realistic Solution:** For uniform quantization of a Gaussian signal, the SQNR depends on the number of quantization levels and how well the range covers the distribution. With $2^b$ levels covering $\pm 3\sigma$ (99.7% of values), the quantization step is $\Delta = 6\sigma / 2^b$. The quantization noise power is $\Delta^2 / 12$. SQNR = signal power / noise power.

  > **Napkin Math:** $\sigma = 0.02$, range $= \pm 3\sigma = \pm 0.06$. INT8 (256 levels): $\Delta = 0.12 / 256 = 4.69 \times 10^{-4}$. Noise power = $\Delta^2/12 = 1.83 \times 10^{-8}$. Signal power = $\sigma^2 = 4 \times 10^{-4}$. SQNR = $4 \times 10^{-4} / 1.83 \times 10^{-8} = 21,858 \approx$ **43.4 dB**. INT4 (16 levels): $\Delta = 0.12/16 = 7.5 \times 10^{-3}$. SQNR = $4 \times 10^{-4} / (4.69 \times 10^{-6}) = 85 \approx$ **19.3 dB**. Each additional bit adds ~6 dB. Gradient noise during training is typically 20–30 dB SQNR. So INT8 (43 dB) is well above the noise floor — additional bits are wasted. INT4 (19 dB) is below the gradient noise floor, explaining why INT4 training fails but INT4 inference (no gradients) can work with careful calibration.

  📖 **Deep Dive:** [Model Compression](https://harvard-edge.github.io/cs249r_book_dev/contents/model_compression/model_compression.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Multi-GPU Scaling Curve</b> · <code>parallelism</code> <code>network-fabric</code></summary>

- **Interviewer:** "You scale a training job from 1 to 2, 4, 8, 16, and 32 H100 GPUs using data parallelism. At 1 GPU, throughput is 1,000 samples/sec. Predict the throughput at each scale, accounting for communication overhead. The interconnect is NVLink within 8-GPU nodes and InfiniBand (400 Gb/s) between nodes."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Linear scaling — 32 GPUs = 32,000 samples/sec." This ignores that communication overhead grows with GPU count and that crossing the node boundary (NVLink → InfiniBand) causes a discontinuous drop in scaling efficiency.

  **Realistic Solution:** Scaling efficiency = actual throughput / (N × single-GPU throughput). Within a node (NVLink), AllReduce is fast and overlaps well with compute. Across nodes (InfiniBand), bandwidth drops ~18× and latency increases, creating a scaling cliff at the node boundary.

  > **Napkin Math:** Model: 7B params, 14 GB gradients (FP16). Single-GPU step: 100 ms compute. **Intra-node (NVLink 900 GB/s):** AllReduce time for $N$ GPUs: $2(N-1)/N \times 14\text{GB} / 450\text{GB/s}$. 2 GPUs: 15.6 ms. 4 GPUs: 23.3 ms. 8 GPUs: 27.2 ms. With overlap: ~30% exposed. Effective overhead: 2→4.7ms, 4→7ms, 8→8.2ms. Throughput: 2→1,953 (97.7%), 4→3,738 (93.5%), 8→7,073 (88.4%). **Inter-node (InfiniBand 50 GB/s effective):** 16 GPUs (2 nodes): AllReduce across nodes: 14 GB / 50 GB/s = 280 ms — longer than compute! Must use gradient compression or hierarchical AllReduce. With hierarchical: intra-node AllReduce (8.2 ms) + inter-node AllReduce of reduced gradients (28 ms) = 36.2 ms exposed. Throughput: 16→12,800 (80%). 32 GPUs (4 nodes): inter-node AllReduce grows: ~45 ms exposed. Throughput: 32→22,400 (70%). **The NVLink→InfiniBand boundary at GPU 9 causes scaling to drop from 88% to 80%.**

  📖 **Deep Dive:** [Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The GPU Power Budget</b> · <code>power-thermal</code> <code>economics</code></summary>

- **Interviewer:** "You're running a 72-hour training job on 8× H100 GPUs (700W TDP each). Your data center charges \$0.12/kWh for electricity. Estimate the energy cost of this single training run. How does this compare to the cloud compute cost at \$3.50/GPU-hour?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "8 × 700W × 72 hours × $0.12 = $48.38. Electricity is negligible." This forgets PUE and that GPUs don't run at exactly TDP.

  **Realistic Solution:** GPUs under sustained training load typically draw 80–90% of TDP. Data centers have PUE overhead (cooling, networking, power conversion) that multiplies the IT power draw. The electricity cost is a small fraction of cloud pricing — the rest is amortized hardware, networking, staff, and profit margin.

  > **Napkin Math:** GPU power: 8 × 700W × 85% utilization = 4,760W. PUE of 1.2: total facility power = 4,760 × 1.2 = 5,712W = 5.71 kW. Energy: 5.71 kW × 72 hrs = **411 kWh**. Electricity cost: 411 × \$0.12 = **\$49.34**. Cloud compute cost: 8 GPUs × 72 hrs × \$3.50 = **\$2,016**. Electricity is only **2.4%** of the cloud bill. The other 97.6% covers: GPU depreciation (~40%), networking/storage (~15%), data center lease (~10%), staff (~10%), and profit margin (~22%). This is why on-premise makes financial sense at scale: if you own the GPUs, your marginal cost per training run is ~\$50 instead of ~\$2,000.

  📖 **Deep Dive:** [Sustainable AI](https://harvard-edge.github.io/cs249r_book_dev/contents/sustainable_ai/sustainable_ai.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Pipeline Parallelism Micro-Batch</b> · <code>parallelism</code> <code>memory-hierarchy</code></summary>

- **Interviewer:** "You're using pipeline parallelism with 4 stages across 4 GPUs to train a 30B model. The global batch size is 32. How many micro-batches should you use to minimize the pipeline bubble, and what's the bubble overhead as a fraction of total compute time?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Use micro-batch size of 1 for maximum pipeline utilization." While more micro-batches reduce the bubble fraction, each micro-batch has fixed overhead (kernel launch, synchronization) and very small micro-batches underutilize the GPU's compute units.

  **Realistic Solution:** With $P$ pipeline stages and $M$ micro-batches, the pipeline bubble is $(P-1)$ micro-batch times at the start and end of each step. The bubble fraction is $(P-1) / (M + P - 1)$. More micro-batches → smaller bubble, but each micro-batch must be large enough to saturate the GPU.

  > **Napkin Math:** $P=4$ stages, global batch = 32. If $M$ micro-batches, each has size $32/M$. Bubble fraction = $(4-1)/(M+4-1) = 3/(M+3)$. **$M=4$:** micro-batch=8, bubble = 3/7 = **42.9%** — nearly half the time wasted. **$M=8$:** micro-batch=4, bubble = 3/11 = **27.3%**. **$M=16$:** micro-batch=2, bubble = 3/19 = **15.8%**. **$M=32$:** micro-batch=1, bubble = 3/35 = **8.6%**. But micro-batch=1 means batch GEMM of size $[1 \times S \times H]$ — terrible GPU utilization (arithmetic intensity too low). Sweet spot: $M=16$ (micro-batch=2) gives 15.8% bubble with reasonable GPU utilization. With 1F1B scheduling (interleaved forward/backward), memory is bounded at $P$ micro-batches instead of $M$, allowing $M=32$ without OOM. Interleaved 1F1B with $M=32$: **8.6% bubble, bounded memory** — the production choice.

  📖 **Deep Dive:** [Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The CXL Memory Tier</b> · <code>memory-hierarchy</code> <code>architecture</code></summary>

- **Interviewer:** "Your inference server has 80 GB of HBM (GPU VRAM) and 512 GB of CXL-attached memory at ~200 ns latency and ~64 GB/s bandwidth. You're serving a 70B model (140 GB FP16). Can you use CXL memory to avoid tensor parallelism? Calculate the performance impact."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "CXL memory is just slower DRAM — put the overflow weights there and it'll work fine with maybe 2× slowdown." The actual impact depends on which layers you offload and the access pattern.

  **Realistic Solution:** CXL memory has ~3× the latency of DDR5 and ~50× less bandwidth than HBM3. You can store the model across both tiers, but the layers served from CXL will be dramatically slower. The key insight: not all layers are accessed equally during decode. The embedding table and output projection are accessed once per token, while each transformer layer is accessed once per layer per token. Offloading the right layers minimizes impact.

  > **Napkin Math:** 140 GB model: 80 GB in HBM, 60 GB in CXL. HBM bandwidth: 3.35 TB/s. CXL bandwidth: 64 GB/s. If layers are split proportionally (57% HBM, 43% CXL): per-token time = 80/3,350 + 60/64 = 0.024 + 0.938 = **0.96 sec/token** — the CXL portion is 39× slower and dominates. That's **~1 token/sec** vs 24 tokens/sec with pure HBM (TP=2). **Smarter approach:** put attention layers (which access KV-cache frequently) in HBM and MLP layers (which are pure GEMMs) in CXL. MLP is ~67% of weights = 94 GB. Put 80 GB of MLP + attention in HBM, 60 GB of MLP in CXL. With prefetching (overlap CXL reads with HBM compute): effective overhead = max(HBM time, CXL time) per layer group. Still limited to ~3–5 tokens/sec. **Verdict: CXL helps for batch inference (high arithmetic intensity hides latency) but not for low-batch decode.**

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Bisection Bandwidth Requirement</b> · <code>network-fabric</code> <code>parallelism</code></summary>

- **Interviewer:** "You're designing the network fabric for a 1,024-GPU training cluster. The workload uses 3D parallelism: TP=8 (within node), PP=4 (across nodes), DP=32 (across nodes). Calculate the minimum bisection bandwidth needed to avoid communication bottlenecks, and explain why a fat-tree topology might not be sufficient."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Just add up the per-GPU bandwidth requirements and buy enough switches." This ignores that different parallelism dimensions have vastly different communication patterns and bandwidth needs.

  **Realistic Solution:** Each parallelism dimension has a different communication pattern: TP uses AllReduce within 8-GPU NVLink domains (handled by NVLink, not the network). PP uses point-to-point sends between adjacent stages (low bandwidth, latency-sensitive). DP uses AllReduce across 32 groups (high bandwidth, latency-tolerant). The network must handle PP and DP traffic simultaneously.

  > **Napkin Math:** **PP traffic:** Each pipeline stage sends activations to the next. Activation size: batch × seq × hidden × bytes = $32 \times 2048 \times 8192 \times 2 = 1$ GB per micro-batch. With 16 micro-batches in flight: 16 GB/s sustained per PP link. 4 PP stages × 32 DP groups = 128 PP links. Total PP bandwidth: 128 × 16 GB/s = **2.05 TB/s**. **DP traffic:** AllReduce of gradients. Model size / PP stages = 70B/4 = 17.5B params per stage = 35 GB. Ring AllReduce across 32 GPUs: $2 \times 31/32 \times 35 = 67.8$ GB per GPU. Training step = 500 ms. Required bandwidth: 67.8 / 0.5 = **135.6 GB/s per GPU**. 128 nodes × 135.6 = **17.4 TB/s** aggregate DP bandwidth. **Bisection bandwidth:** half the cluster must communicate with the other half. Minimum bisection = max(PP, DP) concurrent traffic across the bisection = ~**10 TB/s**. A standard 3-tier fat-tree with 400 Gb/s (50 GB/s) links needs 200 spine-to-leaf links at the bisection — feasible but expensive. A rail-optimized topology (separate networks for DP and PP) reduces cost by 30% because PP is latency-sensitive but low-bandwidth while DP is bandwidth-heavy but latency-tolerant.

  📖 **Deep Dive:** [Training](https://harvard-edge.github.io/cs249r_book_dev/contents/training/training.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Optimal Checkpoint Interval</b> · <code>fault-tolerance</code> <code>economics</code></summary>

- **Interviewer:** "You're training on 512 GPUs. The mean time between failures (MTBF) for the cluster is 2 hours. Checkpointing takes 5 minutes and pauses all training. Derive the optimal checkpoint interval that minimizes total wasted time (checkpoint overhead + expected lost work from failures)."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Checkpoint every 5 minutes to minimize lost work." This maximizes checkpoint overhead — you'd spend 50% of time checkpointing.

  **Realistic Solution:** This is a classic optimization problem. Let $\tau$ = checkpoint interval, $\delta$ = checkpoint duration (5 min), $\lambda$ = failure rate (1/MTBF = 0.5/hr). The total wasted time per unit of training has two components: (1) checkpoint overhead = $\delta / \tau$, and (2) expected lost work per failure = $\tau/2$ (uniform distribution of failure within interval) × failure rate = $\lambda \tau / 2$. Minimize $f(\tau) = \delta/\tau + \lambda\tau/2$.

  > **Napkin Math:** $f(\tau) = \delta/\tau + \lambda\tau/2$. Take derivative: $f'(\tau) = -\delta/\tau^2 + \lambda/2 = 0$. Solve: $\tau^* = \sqrt{2\delta/\lambda}$. With $\delta = 5$ min = 1/12 hr, $\lambda = 0.5$/hr: $\tau^* = \sqrt{2 \times (1/12) / 0.5} = \sqrt{1/3} = 0.577$ hr ≈ **34.6 minutes**. Wasted time at optimal: $f(\tau^*) = (1/12)/0.577 + 0.5 \times 0.577/2 = 0.144 + 0.144 = 0.289$ → **28.9% overhead**. Compare: checkpoint every 10 min: $f = (1/12)/(1/6) + 0.5 \times (1/6)/2 = 0.5 + 0.042 = 54.2\%$ — checkpoint overhead dominates. Checkpoint every 2 hrs: $f = (1/12)/2 + 0.5 \times 2/2 = 0.042 + 0.5 = 54.2\%$ — lost work dominates. The optimal balances both at 28.9%. With **async checkpointing** ($\delta_{\text{effective}} \approx 0.5$ min): $\tau^* = \sqrt{2 \times (1/120) / 0.5} = 0.183$ hr ≈ **11 min**, overhead drops to **9.1%**.

  📖 **Deep Dive:** [ML Systems](https://harvard-edge.github.io/cs249r_book_dev/contents/ml_systems/ml_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Serving Fleet Memory Plan</b> · <code>memory-hierarchy</code> <code>economics</code></summary>

- **Interviewer:** "You need to serve three models in production: a 7B model (1,000 QPS, <100ms latency), a 13B model (200 QPS, <200ms latency), and a 70B model (50 QPS, <2s latency). All in FP16. Plan the GPU fleet: how many H100 GPUs (80 GB each) do you need, and what's the monthly cost at \$2.80/GPU-hour?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "Add up the model sizes, divide by 80 GB per GPU." This ignores that you need to account for KV-cache memory, concurrent request batching, and per-model latency SLAs that constrain batch sizes.

  **Realistic Solution:** Each model needs enough GPUs to (1) fit the weights + KV-cache for the target batch size, and (2) meet the latency SLA at that batch size. The batch size is determined by the QPS target divided by the throughput per GPU.

  > **Napkin Math:** **7B model (14 GB FP16):** Fits on 1 GPU with 66 GB for KV-cache. At 4k context, KV-cache ~0.5 GB/request → batch of 120. Throughput at batch=120: ~3,000 tokens/sec → ~100 requests/sec (30 tokens avg). For 1,000 QPS: 10 GPUs. Latency at batch=120: ~40 ms/token × 30 tokens = 1.2s — exceeds 100ms SLA! Need batch=4 for <100ms: throughput = 120 req/sec/GPU. GPUs needed: 1,000/120 = **9 GPUs**. **13B model (26 GB):** 1 GPU, 54 GB for KV. Batch=60 for <200ms. Throughput: ~40 req/sec/GPU. 200/40 = **5 GPUs**. **70B model (140 GB):** TP=2 minimum (70 GB/GPU). 2 GPUs per replica, ~10 GB KV headroom → batch=10. Throughput: ~8 req/sec per replica. 50/8 = 7 replicas = **14 GPUs**. **Total: 9 + 5 + 14 = 28 GPUs.** Monthly cost: 28 × 720 hrs × \$2.80 = **\$56,448/month**. With INT8 quantization: 7B fits with batch=200 (1 GPU serves 200 QPS), 13B fits on 1 GPU with more headroom. Optimized fleet: ~18 GPUs → **\$36,288/month** (36% savings).

  📖 **Deep Dive:** [Hardware Acceleration](https://harvard-edge.github.io/cs249r_book_dev/contents/hw_acceleration/hw_acceleration.html)

  </details>

</details>


<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Stuttering Training Loop</b> · <code>cache-hierarchy</code></summary>

- **Interviewer:** "You're training a small neural network on a GPU, but you notice the GPU utilization is low (e.g., 30-40%), and the training progress seems to 'stutter' periodically, despite using a small dataset that easily fits into GPU memory. What's a common, non-obvious reason for this behavior?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The GPU drivers are bad" or "The CPU is bottlenecking data loading." While possible, for a small dataset fitting GPU memory, these are less likely primary causes than memory hierarchy issues.

  **Realistic Solution:** The problem is likely due to frequent cache misses and poor data locality, leading to "data thrashing." Even if the entire dataset fits in GPU HBM, specific parts of the model (e.g., large embedding tables or activation maps) might not fit into the GPU's faster, smaller on-chip caches (L1/L2). When the GPU repeatedly accesses data that isn't in cache, it incurs the higher latency of fetching from HBM. The "stuttering" occurs when the GPU's compute units frequently stall, waiting for data to be loaded from slower memory, leading to low utilization. Optimizing data access patterns or restructuring the model to improve cache locality can mitigate this.

  > **Napkin Math:** A typical GPU L2 cache might be 6-12MB. If an embedding table for a small model has 100,000 embeddings, each 128 floats (FP32), its size is $100,000 \times 128 \times 4 \text{ bytes} = 51.2 \text{ MB}$. This clearly exceeds L2 cache, forcing frequent HBM fetches.

  > **Key Equation:** $Memory\_Access\_Time = (\text{Cache Hit Rate} \times \text{Cache Latency}) + (\text{Cache Miss Rate} \times \text{Main Memory Latency})$

  📖 **Deep Dive:** [Volume I: Memory Systems](https://mlsysbook.ai/vol1/memory_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Half-Baked Speedup</b> · <code>mixed-precision</code></summary>

- **Interviewer:** "You've been tasked with optimizing a large language model's training throughput. You successfully converted most of your model to use BF16 (bfloat16) precision instead of FP32. You expected a near 2x speedup in training time, but you're only observing about a 1.4x improvement. What's a likely technical explanation for this discrepancy?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "There must be some FP32 operations remaining in the graph" or "The BF16 kernels aren't fully optimized." While these can contribute, the primary issue is often more fundamental.

  **Realistic Solution:** The model is likely **memory-bound** rather than compute-bound. Switching from FP32 to BF16 effectively halves the data size for weights, activations, and gradients. This directly benefits compute-bound operations by allowing more data to be processed per clock cycle (potentially 2x more floating-point operations). However, if the limiting factor is the rate at which data can be fetched from High Bandwidth Memory (HBM) to the GPU's Streaming Multiprocessors (SMs), then halving the data size only helps if the model's arithmetic intensity ($Ops/Bytes$) is high enough to shift it out of the memory-bound region on the Roofline model. For models with low arithmetic intensity or very large memory footprints (like large language models with extensive embedding tables or large context windows), memory bandwidth becomes the bottleneck. The 1.4x speedup indicates some benefit from reduced data movement, but not the full 2x compute-bound speedup, confirming the memory bottleneck.

  > **Napkin Math:** An A100 GPU has 19.5 TFLOPS (FP32) and 312 TFLOPS (BF16, sparse) theoretical peak compute, but a memory bandwidth of 2 TB/s. If an operation has an arithmetic intensity of 10 FLOP/Byte (e.g., a simple matrix-vector multiply), then for FP32, it needs $19.5 \text{ TFLOP/s} / 10 \text{ FLOP/byte} = 1.95 \text{ TB/s}$ bandwidth. For BF16, it needs $312 \text{ TFLOP/s} / 10 \text{ FLOP/byte} = 31.2 \text{ TB/s}$. Since the GPU only has 2 TB/s, both are memory-bound. While BF16 halves the data transfer, it only helps proportionally to the reduction in bytes moved, not necessarily doubling throughput if memory bandwidth is the hard limit.

  > **Key Equation:** $Achieved\_Performance = \min(\text{Compute Throughput}, \text{Memory Bandwidth} \times \text{Arithmetic Intensity})$

  📖 **Deep Dive:** [Volume I: Roofline Model](https://mlsysbook.ai/vol1/roofline_model.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The NUMA Nightmare</b> · <code>NUMA</code></summary>

- **Interviewer:** "You're scaling up a gradient boosting model training job on a powerful dual-socket CPU server (e.g., two Intel Xeon CPUs). When you move from utilizing a single CPU socket to attempting to use both sockets in parallel, you observe that the performance improvement is far less than 2x, sometimes even degrading. What technical architectural detail of modern multi-socket systems is most likely causing this, and how would you diagnose and mitigate it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "There's a lock contention issue in the gradient aggregation" or "The parallelization library isn't efficient." While possible, the primary architectural culprit for non-linear scaling on multi-socket systems is often NUMA.

  **Realistic Solution:** The issue is almost certainly **NUMA (Non-Uniform Memory Access)**. In a dual-socket system, each CPU socket has its own local DRAM controller and directly connected memory. Accessing memory attached to the *local* socket is significantly faster than accessing memory attached to the *remote* socket via the inter-socket interconnect (e.g., Intel UPI, AMD Infinity Fabric). If your process or threads are not NUMA-aware, they might allocate memory on one socket and then have threads on the *other* socket frequently access that remote memory. This "remote access penalty" can severely degrade performance, eating into any gains from parallelism.

  **Diagnosis:**
  1.  Use `numactl --hardware` to inspect the NUMA topology (nodes, distances).
  2.  Use tools like `numastat` or `perf c-state` to monitor NUMA hits/misses and remote memory accesses.
  3.  Monitor CPU core usage and memory allocation patterns (e.g., via `/proc/pid/numa_maps`).

  **Mitigation:**
  1.  **NUMA-aware memory allocation and process/thread pinning:** Use `numactl --membind` and `numactl --cpunodebind` to ensure that memory is allocated on the same NUMA node as the CPU cores that will primarily access it.
  2.  **Thread Affinity:** Explicitly pin threads to specific CPU cores within their respective NUMA nodes using tools like `taskset` or `sched_setaffinity`.
  3.  **Data Partitioning:** Design your algorithm to partition data such that each NUMA node primarily operates on data local to its memory, minimizing cross-node communication.

  > **Napkin Math:** Local memory access latency might be ~80 ns. Remote memory access could be ~150-200 ns (a 1.8x to 2.5x penalty). If 30% of memory accesses become remote due to poor NUMA awareness, the effective memory latency could increase by $(0.7 \times 80 \text{ns}) + (0.3 \times 180 \text{ns}) \approx 56 \text{ns} + 54 \text{ns} = 110 \text{ns}$, a 37.5% increase over purely local access, significantly impacting overall throughput.

  > **Key Equation:** $Effective\_Memory\_Latency = (\text{Local Access Ratio} \times \text{Local Latency}) + (\text{Remote Access Ratio} \times \text{Remote Latency})$

  📖 **Deep Dive:** [Volume I: Memory Systems](https://mlsysbook.ai/vol1/memory_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Paging Paradox</b> · <code>virtual-memory</code></summary>

- **Interviewer:** "You're working on a large-scale recommendation system that uses a massive embedding table (e.g., 1TB in size) loaded into a server with 2TB of RAM. Despite having ample physical memory, you observe very high CPU utilization but surprisingly low throughput, and profiling tools like `perf` show a significant number of `dTLB-load-misses`. What is the underlying cause of this performance bottleneck, and what specific operating system feature would you leverage to mitigate it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "It's a memory bandwidth issue" or "The CPU cache (L1/L2/L3) is too small." While these can be bottlenecks, the `dTLB-load-misses` metric specifically points to a different problem in the memory hierarchy.

  **Realistic Solution:** The bottleneck is due to **Translation Lookaside Buffer (TLB) misses**. Modern CPUs use virtual memory, where applications address memory using virtual addresses, which are then translated by the Memory Management Unit (MMU) to physical addresses. The TLB is a high-speed cache that stores recent virtual-to-physical address translations. When a process accesses a virtual address, the CPU first checks the TLB. If there's a hit, the translation is fast. If there's a miss (a `dTLB-load-miss` in `perf`), the CPU must perform a costly **page table walk** through main memory to find the physical address.
  A 1TB embedding table, if managed with standard 4KB pages, would require $1 \text{ TB} / 4 \text{ KB/page} = 256 \text{ million}$ distinct pages. Even if the entire table fits in RAM, frequently accessing disparate parts of this vast table will quickly exhaust the TLB's limited capacity (e.g., 1024 entries). This leads to continuous TLB misses and expensive page table walks, causing high CPU utilization (due to MMU work) but low effective throughput as the CPU waits for translations.
  The solution is to use **huge pages** (also known as large pages, e.g., 2MB or 1GB pages). By increasing the page size, the number of virtual-to-physical mappings required to cover the 1TB embedding table is drastically reduced ($1 \text{ TB} / 2 \text{ MB/page} = 500,000$ pages). This significantly reduces TLB pressure, leading to a much higher TLB hit rate, fewer page table walks, and consequently, much better performance.

  > **Napkin Math:**
  > - Standard 4KB pages: 1TB table needs $256 \times 10^6$ pages. If each access causes a TLB miss and page table walk costs 100 CPU cycles, and you have $10^9$ accesses, that's $10^{11}$ cycles of overhead.
  > - With 2MB huge pages: 1TB table needs $5 \times 10^5$ pages. The number of TLB misses for the same $10^9$ accesses could be reduced by orders of magnitude, say to $10^7$ misses, reducing overhead to $10^9$ cycles. This is a 100x improvement.

  > **Key Equation:** $Total\_Memory\_Access\_Latency = (\text{TLB Hit Rate} \times \text{TLB Latency}) + (\text{TLB Miss Rate} \times (\text{Page Table Walk Latency} + \text{DRAM Latency}))$

  📖 **Deep Dive:** [Volume I: Memory Systems](https://mlsysbook.ai/vol1/memory_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Small Batch Anomaly</b> · <code>cpu-gpu-arch</code></summary>

- **Interviewer:** "You're running inference for a small convolutional neural network (CNN) with a batch size of 1. To your surprise, you find that a powerful CPU completes the inference faster than a high-end GPU. Explain why this counter-intuitive result might occur."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "The GPU isn't being utilized enough" or "The GPU drivers are not optimized for small models." While true that utilization is low, the core reason for the CPU's advantage is architectural.

  **Realistic Solution:** GPUs are designed for **massive parallel throughput** (SIMT - Single Instruction, Multiple Thread) and achieve efficiency by hiding memory and compute latency with a large number of concurrent threads. For very small workloads, like a batch size of 1 inference:
  1.  **Overhead:** There's a significant overhead associated with using a GPU: data transfer from CPU RAM to GPU VRAM (via PCIe), kernel launch overhead, and context switching. For tiny tasks, this overhead can easily outweigh the GPU's raw computational speed.
  2.  **Latency vs. Throughput:** GPUs are optimized for throughput, not necessarily single-task latency. CPUs, with their sophisticated branch prediction, larger caches per core, and strong single-thread performance, can process small, latency-critical tasks very efficiently.
  3.  **Underutilization:** A batch size of 1 doesn't provide enough work to fully saturate the GPU's thousands of cores, leaving most of them idle. The benefits of parallel execution are minimal, while the overhead remains.
  For small batch sizes, the CPU's lower overhead and higher single-core performance often make it the faster choice.

  > **Napkin Math:**
  > - PCIe transfer overhead: ~10-100 microseconds.
  > - GPU kernel launch overhead: ~10-100 microseconds.
  > - If a small CNN inference takes 5 microseconds on a CPU, the GPU's total time would be $5 \text{ us (kernel)} + 10 \text{ us (transfer)} + 10 \text{ us (launch)} = 25 \text{ us}$, making the CPU 5x faster.

  > **Key Equation:** $Total\_GPU\_Latency = Data\_Transfer\_Time + Kernel\_Launch\_Overhead + Kernel\_Execution\_Time$

  📖 **Deep Dive:** [Volume I: CPU vs. GPU Architecture](https://mlsysbook.ai/vol1/cpu_gpu_architecture.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L4_Mid-blue?style=flat-square" alt="Level 2" align="center"> The Striding Stumble</b> · <code>data-locality</code></summary>

- **Interviewer:** "You're comparing two C++ implementations of a matrix multiplication kernel on a CPU. Both use standard nested loops and perform the exact same arithmetic operations. However, one is consistently 3-5x slower than the other. Upon inspection, you notice the inner loop access patterns are `matrix[i][j]` in the fast version and `matrix[j][i]` in the slow version. Explain the technical reason for this significant performance difference."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "It's a compiler optimization difference" or "One uses SIMD instructions better." While compilers do optimize, the fundamental difference here lies in memory access.

  **Realistic Solution:** The performance difference is due to **cache locality** and the memory layout of multi-dimensional arrays. In C/C++, 2D arrays are stored in **row-major order**. This means elements in the same row (`matrix[i][j]`, `matrix[i][j+1]`, `matrix[i][j+2]`, etc.) are contiguous in memory.
  -   **Fast Version (`matrix[i][j]` access):** When the inner loop iterates through `j`, it accesses elements that are physically adjacent in memory. This exhibits excellent **spatial locality**. The CPU can fetch a cache line (e.g., 64 bytes) containing multiple elements with a single memory access. Subsequent accesses hit the cache, leading to very fast data retrieval.
  -   **Slow Version (`matrix[j][i]` access):** When the inner loop iterates through `j` while `i` is fixed, it's effectively trying to access elements column-wise. For row-major arrays, `matrix[j][i]` and `matrix[j+1][i]` are far apart in memory. Each access likely results in a **cache miss**, forcing the CPU to fetch a new cache line from main memory. This constant fetching and evicting of cache lines (cache thrashing) introduces significant latency, making the operation much slower.

  > **Napkin Math:** Assume a cache line is 64 bytes and `float` is 4 bytes. A fast version might fetch 16 floats per cache miss. A slow version, if the matrix width is large, might incur a cache miss for almost every single float accessed, leading to 16x more main memory accesses. If a cache hit is 1 ns and a miss is 100 ns, the performance difference is substantial.

  > **Key Equation:** $Performance \propto \frac{1}{\text{Memory Access Latency}} \text{ where } \text{Memory Access Latency} = (\text{Cache Hit Rate} \times \text{Cache Latency}) + (\text{Cache Miss Rate} \times \text{Main Memory Latency})$

  📖 **Deep Dive:** [Volume I: Memory Systems](https://mlsysbook.ai/vol1/memory_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 3" align="center"> The Memory Dilemma</b> · <code>memory-technologies</code></summary>

- **Interviewer:** "Your team is designing a custom ML accelerator optimized for large Transformer models with exceptionally long context windows. You need to choose between integrating HBM (High Bandwidth Memory) or GDDR6 for the on-board memory. Discuss the technical trade-offs of each memory technology in the context of this specific workload, and justify your recommendation."

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "HBM is always superior due to higher bandwidth" or "GDDR6 is cheaper, so it's the practical choice." The optimal choice depends heavily on the specific workload characteristics.

  **Realistic Solution:**
  For large Transformer models with long context windows, the primary performance bottleneck often shifts from pure compute to **memory bandwidth** and **memory capacity**.
  1.  **HBM (High Bandwidth Memory - e.g., HBM2e, HBM3):**
      *   **Pros:** Extremely high bandwidth (e.g., 2-4 TB/s for HBM3), lower power consumption per bit, smaller footprint due to 3D stacking. This is critical for memory-bound operations like large embedding table lookups, attention mechanisms (which involve frequent, large data movements), and loading massive model weights and activations for long sequences.
      *   **Cons:** Lower total capacity per stack (e.g., 16-128GB), higher cost per GB, more complex integration (requires interposer).
  2.  **GDDR6 (e.g., GDDR6X):**
      *   **Pros:** Higher total capacity (e.g., 24-48GB per chip, multiple chips can be used), lower cost per GB, simpler integration into PCBs.
      *   **Cons:** Lower bandwidth compared to HBM (e.g., 1 TB/s for GDDR6X), higher power consumption per bit, larger physical footprint.

  **Recommendation for Large Transformers with Long Context Windows:**
  I would recommend **HBM**. Large Transformer models, especially with long context windows, are incredibly memory-hungry. The attention mechanism's quadratic complexity with sequence length means activations grow rapidly, leading to massive data movement requirements. While dense matrix multiplications are theoretically compute-bound, the sheer size of matrices for long contexts means they quickly become memory-bound due to the need to fetch large weights and activations. The extremely high bandwidth of HBM is paramount for feeding the compute units efficiently, preventing them from stalling while waiting for data. Although HBM has lower total capacity per stack, multiple HBM stacks can be integrated to meet the overall capacity requirement for even very large models (e.g., 80GB on an A100, 192GB on an H100). The performance gain from superior bandwidth will likely outweigh the higher cost and capacity limitations of HBM for this specific, demanding workload.

  > **Napkin Math:** A 175B parameter FP16 model requires $\sim 350 \text{ GB}$ of memory for weights. If activations for a long context window add another 100GB, total memory is 450GB. If your accelerator has 100 TFLOPS of compute, and the average arithmetic intensity is 10 FLOP/Byte, you need $100 \text{ TFLOP/s} / 10 \text{ FLOP/Byte} = 10 \text{ TB/s}$ memory bandwidth to keep it compute-bound. HBM (e.g., 4TB/s) gets closer to this ideal than GDDR6 (e.g., 1TB/s), making it less memory-bound.

  > **Key Equation:** $Memory\_Bandwidth\_Required = \frac{Peak\_Compute\_Throughput}{\text{Arithmetic Intensity}}$

  📖 **Deep Dive:** [Volume I: Memory Systems](https://mlsysbook.ai/vol1/memory_systems.html)

  </details>

</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 4" align="center"> The Phantom Update</b> · <code>cache-coherence</code></summary>

- **Interviewer:** "You are developing a high-performance, multi-threaded C++ training loop for a small MLP on a single server with multiple CPU cores. In your custom gradient aggregation step, each thread computes its local gradients and then writes them to distinct, non-overlapping sections of a shared global gradient array. Despite each thread writing to its own designated memory region, you observe inconsistent or incorrect gradient updates and significant performance degradation. There are no explicit race conditions using mutexes or atomics. What is the most likely underlying hardware-level issue, and how would you diagnose and mitigate it?"

  <details>
  <summary><b>🔍 Reveal Answer</b></summary>

  **Common Mistake:** "It's a subtle race condition, I need to add locks." While race conditions are common, the scenario explicitly states distinct, non-overlapping writes, pointing to a more insidious problem.

  **Realistic Solution:** This is a classic case of **false sharing**. Modern CPUs use cache lines (typically 64 bytes) as the smallest unit of data transfer between main memory and CPU caches. When multiple threads access different variables that happen to reside within the *same cache line*, even if those variables are logically distinct and accessed without explicit contention, the cache coherence protocol (e.g., MESI) will cause the cache line to "ping-pong" between the CPU cores.
  Specifically:
  1.  Thread A writes to `global_gradients[X]`. The cache line containing `global_gradients[X]` (and neighboring elements) is marked `Modified` in Thread A's cache.
  2.  Thread B (on a different core) then writes to `global_gradients[Y]`, where `Y` is distinct from `X` but falls within the *same cache line* as `X`.
  3.  Before Thread B can write, its cache must obtain the latest copy of that cache line. This invalidates Thread A's cache line and forces Thread B to fetch it from memory or Thread A's cache.
  4.  This constant invalidation and re-fetching of the shared cache line (even though the threads are operating on logically independent data) incurs significant latency, leading to performance degradation and potentially stale data if the coherence protocol isn't strictly enforced or if reads occur between the ping-pongs.

  **Diagnosis:**
  1.  **Profiling Tools:** Use performance counters (e.g., `perf`) to look for high rates of `cache-misses`, `cache-lines-invalidated`, and `LLC-load-misses` specifically when the gradient aggregation occurs.
  2.  **Memory Layout Inspection:** Examine the memory addresses of the variables being accessed by different threads to see if they are co-located within a 64-byte cache line boundary.

  **Mitigation:**
  1.  **Padding:** Pad the data structures or arrays so that variables frequently accessed by different threads reside in distinct cache lines. For example, ensure each thread's gradient segment starts on a 64-byte boundary.
  2.  **Thread-Local Buffers:** Each thread aggregates its gradients into a *thread-local* buffer first. Only after all threads have completed their local aggregation (and possibly reduced the data size) is a final aggregation performed into the global array, minimizing contention on shared cache lines.
  3.  **Align to Cache Line:** For dynamic allocations, use `posix_memalign` or `std::aligned_alloc` to ensure data structures are aligned to cache line boundaries.

  > **Napkin Math:** A cache line is 64 bytes. If `float` is 4 bytes, 16 floats fit in a cache line. If thread A writes to `gradients[0]` and thread B writes to `gradients[15]`, they share a cache line. Each write triggers an invalidation and fetch for the other core, costing ~100-200 cycles per write instead of a few cycles for a cache hit. This can easily turn a few-cycle operation into a hundreds-of-cycles stall.

  > **Key Equation:** $Cache\_Line\_Size = \text{typically 64 bytes}$

  📖 **Deep Dive:** [Volume I: Cache Coherence](https://mlsysbook.ai/vol1/memory_systems.html#cache-coherence)

  </details>

</details>

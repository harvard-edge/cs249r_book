# Round 1: Single-Node Systems & Silicon Physics 🧱

<div align="center">
  <a href="README.md">🏠 Hub Home</a> |
  <a href="00_The_Architects_Rubric.md">📋 The Rubric</a> |
  <a href="01_Single_Node_Physics.md">🧱 Round 1</a> |
  <a href="02_Distributed_Infrastructure.md">🚀 Round 2</a> |
  <a href="03_Production_Serving.md">⚡ Round 3</a> |
  <a href="04_Operations_and_Economics.md">💼 Round 4</a> |
  <a href="05_Visual_Architecture_Debugging.md">🖼️ Round 5</a>
</div>

---


The domain of the ML Systems Engineer. This round tests your understanding of what happens *inside* a single server chassis: memory hierarchies, compute bounds, and arithmetic intensity.

> **[➕ Add a Flashcard to this section](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/01_Single_Node_Physics.md)** (Edit in Browser)

### 📋 Copy-Paste Template
```markdown
<details>
<summary><b>[LEVEL]: [Your Question Title Here]</b></summary>

**Interviewer:** [The scenario or crisis]
**Realistic Solution:** [Explain the physics/logic here]
**📖 Deep Dive:** [Optional: Link to Volume I or II Chapter]
</details>
```

---

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Profiling Crisis</b></summary>

**Interviewer:** "You've deployed a custom recommendation model. The profiling dashboard shows you are achieving 120 TFLOPS out of a possible 300 TFLOPS on your GPU. Your tech lead suggests buying a faster GPU to fix the latency. Why is your tech lead wrong?"
**Realistic Solution:** The tech lead hasn't checked the Arithmetic Intensity ($Ops/Bytes$). If the model is memory-bound (intensity is lower than the ridge point of the roofline), a GPU with faster ALUs will do nothing. You must buy a GPU with higher *Memory Bandwidth (HBM)*, or optimize the model to move fewer bytes (e.g., quantization).
**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Data Pipeline Stall</b></summary>

**Interviewer:** "You are training a vision model on high-resolution medical images. `nvidia-smi` shows GPU utilization fluctuating violently between 0% and 100%. What is the most likely bottleneck in your node?"
**Realistic Solution:** CPU Starvation (The Transformation Wall). The GPU finishes its math instantly, then sits at 0% waiting for the CPU to decode, crop, and augment the next batch of JPEGs. You must offload preprocessing to the GPU (like NVIDIA DALI) or increase your CPU worker count.
**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Precision Trade-off</b></summary>

**Interviewer:** "We are quantizing our model from FP16 to INT8. The memory footprint drops by 2x, but the throughput doesn't improve at all. Assuming we are compute-bound, what hardware architectural detail did we forget?"
**Realistic Solution:** You forgot to check if the silicon has dedicated INT8 Tensor Cores. Without specific hardware paths for 8-bit integer math, the GPU must upcast the INT8 values back to FP16 in the registers to perform the multiply-accumulate (MAC), yielding zero compute speedup despite the memory savings. 
**📖 Deep Dive:** [Volume I: Model Compression](https://mlsysbook.ai/vol1/model_compression.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Sequence Length Trap</b></summary>

**Interviewer:** "You need to increase your LLM's context window from 4k to 128k tokens. The model weights fit perfectly in your 80GB VRAM. What hidden memory cost will cause your node to OOM (Out of Memory) during generation?"
**Realistic Solution:** The KV-Cache. While weights are static, the KV-cache grows linearly with sequence length and batch size. At 128k context, the memory required to store the attention keys and values for a single request will massively exceed the size of the model weights themselves. 
**📖 Deep Dive:** [Volume I: Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L3_Junior-brightgreen?style=flat-square" alt="Level 1" align="center"> The Compilation Overhead</b></summary>

**Interviewer:** "You move a PyTorch training loop from a CPU to a GPU. The first few batches take 500ms each, but suddenly the latency drops to 10ms per batch. What happened inside the framework?"
**Realistic Solution:** Just-In-Time (JIT) compilation overhead. The framework spends the first few iterations tracing the computation graph and compiling optimized CUDA kernels for the specific tensor shapes you provided. Once cached, the dispatch overhead disappears. 
**📖 Deep Dive:** [Volume I: ML Systems](https://mlsysbook.ai/vol1/ml_systems.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Energy-Movement Invariant</b></summary>

**Interviewer:** "We pruned 50% of the weights from our model, cutting the total MAC (Multiply-Accumulate) operations in half. However, the energy consumption of the node barely dropped. Why?"
**Realistic Solution:** You forgot the Energy-Movement Invariant. Fetching a bit of data from off-chip DRAM costs roughly 100-200x more energy than the math operation (MAC) itself. If your pruning was unstructured, you still have to load the same dense matrices from memory before applying a sparse mask, yielding zero energy savings.
**📖 Deep Dive:** [Volume I: Neural Computation](https://mlsysbook.ai/vol1/nn_computation.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Roofline Shift</b></summary>

**Interviewer:** "We upgraded our serving fleet from A100s to H100s. The H100 has 3x more FP16 TFLOPS. However, our LLM decode throughput got exactly 0% faster. How is that physically possible?"
**Realistic Solution:** The ridge point of the Roofline shifted right. Your workload was already memory-bandwidth bound on the A100 (Arithmetic Intensity < Ridge Point). The H100 has 3x more compute, but only 1.5x more memory bandwidth. Because LLM decoding is bandwidth-starved ($I \approx 1$), faster ALUs literally cannot do any work. 
**📖 Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L5_Senior-yellow?style=flat-square" alt="Level 2" align="center"> The Underflow Crisis</b></summary>

**Interviewer:** "We switched our pre-training job from FP32 to FP16 to save memory, but the loss is returning NaNs within the first 100 steps. What numerical property is failing?"
**Realistic Solution:** Gradient underflow. FP16 only has 5 bits for the exponent, giving it a very narrow dynamic range. Small gradients in deep networks underflow to exactly 0.0, causing training to collapse. You must switch to BF16 (which uses 8 bits for the exponent, matching FP32's range) or implement dynamic loss scaling.
**📖 Deep Dive:** [Volume I: Neural Computation](https://mlsysbook.ai/vol1/nn_computation.html)
</details>

<details>
<summary><b><img src="https://img.shields.io/badge/Level-L6+_Principal-red?style=flat-square" alt="Level 3" align="center"> The Sparsity Fallacy</b></summary>

**Interviewer:** "You wrote a custom CUDA kernel that skips calculating zeros (90% sparse). But it runs slower than a dense PyTorch matrix multiply that does 10x more math. Why does doing 'less work' take more time?"
**Realistic Solution:** Hardware acceleration relies on dense, regular computation blocks. Tensor Cores and Systolic Arrays are physically hard-wired to perform $16\times16$ matrix blocks in a single instruction. Irregular sparse operations force the hardware to fall back to standard, un-fused CUDA cores, completely destroying the throughput advantage of the specialized silicon.
**📖 Deep Dive:** [Volume I: Hardware Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

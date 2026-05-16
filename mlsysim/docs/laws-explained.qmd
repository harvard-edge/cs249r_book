# The 22 Laws of ML Systems — Plain English

> Every equation in mlsysim answers one question. Here is what each one says,
> why it matters, and how to explain it to anyone in 30 seconds.

---

## Domain 1: Node — What Can One Chip Do?

### Wall 1: The Compute Wall
**Equation:** `Time = Operations / (Peak_FLOPS × Efficiency)`

**Plain English:** How long does it take to do all the math? Take the total number
of multiply-adds your model needs, divide by how fast your chip can do math
(adjusted for real-world inefficiency), and that's your compute time.

**30-second version:** "If your model needs 16 billion multiplications and your GPU
can do 989 trillion per second at 50% efficiency, the math takes 0.03 milliseconds."

**Why it matters:** This is the *speed limit* for any workload. No software optimization
can make your model run faster than the hardware can crunch numbers.

---

### Wall 2: The Memory Wall
**Equation:** `Time = Weight_Bytes / Memory_Bandwidth`

**Plain English:** How long does it take to load the model from memory? Your model's
weights sit in HBM (the GPU's memory). To use them, the GPU must read them through
a pipe with a fixed width (bandwidth). Bigger model = longer loading time.

**30-second version:** "A 16 GB model on a GPU with 3.35 TB/s bandwidth takes 4.8 ms
just to load the weights — before any computation even starts."

**Why it matters:** For most LLM inference at batch size 1, this is THE bottleneck.
The GPU finishes the math long before it finishes reading the weights. This is why
memory bandwidth matters more than FLOPS for serving chatbots.

---

### Wall 3: The Software Wall
**Equation:** `MFU = Achieved_FLOPS / Peak_FLOPS`

**Plain English:** How much of the chip's power are you actually using? MFU (Model
FLOPs Utilization) measures the gap between what the hardware *could* do and what
your software *actually* achieves. Low MFU means wasted silicon.

**30-second version:** "If your GPU can do 989 TFLOPS but your training only achieves
450 TFLOPS, your MFU is 45%. The other 55% is lost to kernel launches, memory stalls,
and framework overhead."

**Why it matters:** A 50% MFU means you're paying for twice the hardware you're using.
Improving MFU is often cheaper than buying more GPUs.

---

### Wall 4: The Serving Wall
**Equation:** `TTFT = Prefill_FLOPs / Peak; ITL = Weights / Bandwidth`

**Plain English:** LLM serving has two phases with completely different bottlenecks.
*Prefill* (processing your prompt) is compute-bound — it's doing tons of math.
*Decode* (generating each token) is memory-bound — it's loading weights for every
single token. These two phases need different optimizations.

**30-second version:** "Prefill is like reading a book fast (CPU-intensive). Decode is
like looking up one word at a time in a dictionary (memory-intensive). That's why your
first token takes 50ms but each subsequent token takes 5ms."

---

### Wall 5: The Batching Wall
**Equation:** `KV_cache = 2 × Layers × Heads × Dim × SeqLen × Batch × BytesPerElem`

**Plain English:** Each active request in an LLM server needs its own "memory" of the
conversation (the KV cache). This memory grows with sequence length and eats into the
GPU's finite HBM. More concurrent requests = more KV cache = less room for new requests.

**30-second version:** "Each Llama-3-8B request at 4K context needs ~500 MB of KV cache.
An 80 GB GPU can only serve ~100 concurrent requests before running out of memory."

---

### Wall 6: The Streaming Wall
**Equation:** `Time_per_layer = max(Weight_Injection_Time, Compute_Time)`

**Plain English:** Wafer-scale chips (like Cerebras) flip the normal GPU bottleneck.
Instead of weights sitting in memory and being read by compute, weights are *streamed*
from external memory nodes while activations live on-chip in SRAM. The bottleneck shifts
from memory bandwidth to the injection interconnect speed.

---

### Wall 7: The Tail Latency Wall
**Equation:** Erlang C queueing model (M/M/c)

**Plain English:** When your servers are busy, new requests have to wait in line. The
waiting time grows *non-linearly* — at 80% utilization, the P99 latency might be 2x
the average. At 95% utilization, it can be 10x. This is why you can't run servers at
100% — the tail latency destroys your SLA.

**30-second version:** "It's like a highway. At 70% capacity, traffic flows. At 95%
capacity, everyone's stuck. Servers work the same way."

---

## Domain 2: Data — Can You Feed the Beast?

### Wall 8: The Ingestion Wall
**Equation:** `Utilization = Data_Demand / Storage_Supply`

**Plain English:** Your GPU is hungry. Can your storage feed it fast enough? If the
GPU consumes data faster than the disk can supply it, the GPU sits idle waiting for
food. This ratio tells you if your data pipeline is the bottleneck.

---

### Wall 9: The Transformation Wall
**Equation:** `Time = Batch × Sequence / CPU_Throughput`

**Plain English:** Before data reaches the GPU, CPUs must decode images, tokenize text,
and apply augmentations. If these CPU operations can't keep up with GPU consumption,
the GPU starves. This is why training pipelines often need 10-100x more CPU cores
than you'd expect.

---

### Wall 10: The Locality Wall
**Equation:** `Effective_BW = Link_BW × Bisection_Fraction / Oversubscription`

**Plain English:** Not all network bandwidth is created equal. A fat-tree network
gives full bisection bandwidth (any node can talk to any other at full speed). A ring
or torus does not — distant nodes communicate through intermediate hops, reducing
effective bandwidth. The topology determines how much of the raw link speed you
actually get for collective communication.

---

## Domain 3: Algorithm — How Much Compute Do You Need?

### Wall 11: The Complexity Wall (Chinchilla Scaling)
**Equation:** `Compute = 6 × Parameters × Tokens; Optimal_Params = √(Budget/120)`

**Plain English:** There's a sweet spot between model size and training data. The
Chinchilla scaling law says: for a fixed compute budget, the optimal model size is
proportional to the square root of your budget, and you should train on 20 tokens
per parameter. Train a too-big model on too-little data = waste. Train a too-small
model on too-much data = also waste.

**30-second version:** "With $10M of compute, you should train a ~90B parameter model
on ~1.8T tokens. Going bigger without more data just wastes money."

---

### Wall 12: The Reasoning Wall [Emerging]
**Equation:** `Time = Steps × Time_per_Step`

**Plain English:** Chain-of-thought reasoning makes models think longer before answering.
Each reasoning step costs compute. More steps = better answers but higher cost. This is
inference-time scaling — spending more compute at serving time instead of training time.

---

### Wall 13: The Fidelity Wall (Compression)
**Equation:** `Compression = 32/bits (quantization); Compression = 1/(1-sparsity) (pruning)`

**Plain English:** You can shrink a model by using fewer bits (quantization: FP32→INT8 = 4x
smaller) or by removing weights (pruning: 50% sparse = 2x smaller). But shrinking costs
accuracy. The question is always: how much accuracy can you afford to lose?

**Key insight:** Storage shrinks for all compression, but *inference speedup* depends on
the method. Unstructured pruning saves storage but gives zero speedup on GPU GEMM. Only
structured pruning and N:M sparsity accelerate actual hardware execution.

---

## Domain 4: Fleet — What Happens at Scale?

### Wall 14: The Communication Wall (AllReduce)
**Equation:** `Time = 2(N-1)/N × Message/Bandwidth + 2(N-1) × Latency`

**Plain English:** Distributed training means every GPU must share its gradients with
every other GPU after each training step. This "AllReduce" operation moves data through
the network. The time depends on how much data (gradient size) and how fast the network
(bandwidth). As you add more GPUs, the communication cost grows, eventually dominating
over the compute savings.

**30-second version:** "1 GB of gradients across 8 GPUs at 50 GB/s NVLink takes 35ms.
That's time your GPUs are waiting instead of computing."

---

### Wall 15: The Fragility Wall (Reliability)
**Equation:** `Cluster_MTBF = Component_MTBF / N_components`

**Plain English:** If one GPU fails every 50,000 hours, a 1000-GPU cluster has a failure
every 50 hours. At 10,000 GPUs, you get a failure every 5 hours. This is why large training
runs need checkpointing — without it, a single GPU failure wastes days of compute.

**30-second version:** "At 1000 GPUs, something breaks every 2 days. At 100,000 GPUs,
something breaks every 30 minutes."

---

### Wall 16: The Multi-tenant Wall (Queueing)
**Equation:** `Wait_Time = Utilization / [2 × Service_Rate × (1 - Utilization)]`

**Plain English:** When multiple teams share a cluster, jobs wait in line. The wait time
grows non-linearly with utilization — at 80% cluster utilization, the average wait is
2x the job duration. This is why shared research clusters feel slow even when they're
"only" 80% busy.

---

## Domain 5: Operations — Is It Worth It?

### Wall 17: The Capital Wall (TCO)
**Equation:** `TCO = CapEx + OpEx_energy + OpEx_maintenance`

**Plain English:** The total cost of running ML infrastructure is hardware purchase
(CapEx, amortized over 3-5 years) plus electricity plus maintenance. A 1024-GPU H100
cluster costs ~$30M in hardware. Electricity is surprisingly small (~10% of total).
The dominant cost lever is utilization — idle GPUs cost the same as busy ones.

---

### Wall 18: The Sustainability Wall
**Equation:** `Carbon = Energy × PUE × Carbon_Intensity`

**Plain English:** Every GPU-hour converts electricity into carbon emissions. How much
depends on three things: how much energy (determined by power and time), how efficiently
the datacenter uses it (PUE — a 1.4 PUE wastes 40% on cooling), and how dirty the grid
is (Quebec hydro = 20 gCO₂/kWh vs Poland coal = 820 gCO₂/kWh).

**30-second version:** "Training the same model in Quebec produces 41x less carbon than
training it in Poland. Geography is the biggest lever for sustainable AI."

---

### Wall 19: The Checkpoint Wall
**Equation:** `MFU_penalty = Write_Time / Checkpoint_Interval`

**Plain English:** Long training runs must periodically save their state to disk in case
of failure. This I/O burst interrupts training and reduces effective utilization. The
optimal checkpoint frequency balances the cost of saving too often (I/O overhead) against
the cost of saving too rarely (wasted compute when failures hit).

---

### Wall 20: The Safety Wall (Privacy)
**Equation:** `Slowdown ∝ 1/ε (DP-SGD)`

**Plain English:** Training with differential privacy (adding noise to protect individual
data points) makes training slower. Stronger privacy guarantees (smaller ε) mean more
noise, which means more training steps to converge. Privacy is not free — it's a compute
tax.

---

## Domain 6: Meta-Analysis — Cross-Cutting Tools

### Wall 21: Sensitivity Analysis
**Equation:** `∂Time/∂parameter` (partial derivatives)

**Plain English:** Which constraint is actually binding? Sensitivity analysis perturbs
each parameter (bandwidth, FLOPS, network speed) by a small amount and measures how
much the total time changes. The parameter with the largest derivative is your
bottleneck — that's where to invest next.

---

### Wall 22: Synthesis (Inverse Roofline)
**Equation:** `Required_BW = Weights / Target_Latency`

**Plain English:** Given a latency target (e.g., "50ms per token"), what hardware specs
do you need? This flips the Roofline model backwards — instead of predicting performance
from hardware, it derives the minimum hardware specs from a performance requirement.

**30-second version:** "If you need 50ms decode latency for a 140 GB model, you need
at least 2.8 TB/s of memory bandwidth. That's more than one A100 — you need tensor
parallelism across two H100s."

---

## The Iron Law (The Master Equation)

**Equation:** `Time = FLOPs / (N × Peak × MFU × η_scaling × Goodput)`

**Plain English:** This one equation governs all of ML systems. Training time equals the
total compute divided by how much useful compute you can actually deliver. Each term in
the denominator represents a different loss:

| Term | What it means | What reduces it |
|------|--------------|----------------|
| N | Number of devices | Budget |
| Peak | Raw hardware speed | GPU generation |
| MFU | Software efficiency | Kernel optimization, FlashAttention |
| η_scaling | Communication overhead | Network bandwidth, gradient compression |
| Goodput | Time lost to failures | Checkpointing, fault tolerance |

**Every wall in the taxonomy maps to one of these five terms.** That's the whole framework.

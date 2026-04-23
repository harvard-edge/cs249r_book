# Understanding the Efficiency Parameter

> Every equation in mlsysim needs one number it cannot derive from first
> principles. That number is **efficiency**.

---

## What Is Efficiency?

The efficiency parameter, written as **eta** in equations and `efficiency` in code,
is the ratio of achieved FLOPS to peak FLOPS:

```
eta = Achieved_FLOPS / Peak_FLOPS
```

When you write:

```python
Engine.solve(model=llama70b, hardware=h100, batch_size=32, precision="fp16", efficiency=0.45)
```

you are telling the simulator: "assume this workload achieves 45% of the H100's
peak FP16 throughput." The simulator uses this to convert theoretical compute
time into realistic wall-clock time:

```
Time = FLOPs / (Peak_FLOPS x eta)
```

Efficiency is always between 0.0 and 1.0. A value of 1.0 would mean the
hardware is computing at its absolute theoretical maximum every single cycle --
something that never happens in practice.

---

## Why Is It a Single Number?

Because it compresses 12+ sub-factors that are impossible to predict
analytically for an arbitrary workload on arbitrary hardware. Here is what
eta actually absorbs:

| Sub-factor | What it means |
|-----------|--------------|
| **Kernel launch overhead** | Each CUDA kernel launch has microseconds of latency. CUDA Graphs help amortize this, but eager-mode PyTorch pays it on every op. |
| **SM occupancy** | Register pressure and shared memory usage determine how many warps can run concurrently on each streaming multiprocessor. Low occupancy means idle compute lanes. |
| **Memory coalescing** | Misaligned or scattered memory accesses waste bandwidth. A perfectly coalesced access pattern can be 10x faster than a naive one. |
| **Instruction-level parallelism** | The GPU's pipeline depth means instructions must be independent to keep all stages busy. Data dependencies create bubbles. |
| **Pipeline bubbles** | In pipeline-parallel training, micro-batch scheduling leaves some stages idle while others compute. The bubble fraction depends on the number of micro-batches relative to pipeline stages. |
| **Communication overlap** | Modern training overlaps AllReduce with backward computation. How well this overlap works depends on network bandwidth, gradient sizes, and scheduling. |
| **Framework overhead** | Python GIL contention, graph compilation time (for torch.compile or XLA), operator dispatch overhead, and autograd bookkeeping all steal cycles from useful compute. |
| **Attention implementation** | FlashAttention achieves 2-4x the throughput of naive attention by fusing operations and reducing HBM reads. The choice of attention kernel alone can shift eta by 0.15-0.25. |
| **Quantization runtime overhead** | INT8/INT4 kernels may not achieve the same fraction of peak as FP16 GEMM, especially for non-standard shapes or when dequantization is on the critical path. |
| **Data loader stalls** | If the CPU cannot tokenize, augment, and transfer data to the GPU fast enough, the GPU sits idle between batches. |
| **Garbage collection pauses** | Python's GC and CUDA's memory allocator can cause periodic stalls, especially with large batch sizes and frequent tensor allocation. |
| **OS/driver interrupt handling** | Context switches, NUMA effects, PCIe configuration space accesses, and driver-level synchronization all introduce non-deterministic latency. |

No analytical model can predict the combined effect of all twelve factors for
a given workload. Even NVIDIA's own performance models use empirical
calibration. The honest thing to do is to measure eta, not pretend to derive it.

---

## How Patterson and Hennessy Handled the Same Problem

This is not a new challenge. In *Computer Architecture: A Quantitative Approach*
(CAQDA), Patterson and Hennessy face the identical issue with **CPI** (Cycles Per
Instruction).

CPI is the average number of clock cycles each instruction takes. It depends on
the instruction mix, cache hit rates, branch prediction accuracy, pipeline
hazards, memory latency, and a dozen other microarchitectural details. Patterson
and Hennessy do not try to predict CPI from first principles. Instead, they:

1. **Measure it** on real hardware with real workloads.
2. **Teach students what affects it** -- so they can reason about why CPI
   differs across programs and architectures.
3. **Use it as a parameter** in the performance equation:
   `Time = Instructions x CPI x Clock_Period`

This is exactly what eta is for ML systems. It is our CPI. The performance
equation has the same structure:

```
Time = FLOPs x (1 / Peak_FLOPS) x (1 / eta)
       ^^^^^   ^^^^^^^^^^^^^^^^^   ^^^^^^^^^
       work    clock period         CPI equivalent
```

Just as CAQDA students learn to reason about *why* a SPEC benchmark has CPI = 1.2
on one processor and CPI = 0.8 on another, mlsysim users learn to reason about
*why* Megatron-LM training achieves eta = 0.50 while eager-mode PyTorch on a small
model achieves eta = 0.10.

The value of the simulator is not in predicting eta. It is in answering: **"Given
this eta, what happens when I change the hardware, the model, or the batch size?"**

---

## The Efficiency Guide Table

Use these empirically-grounded ranges as starting points:

| Scenario | Efficiency | Rationale |
|----------|-----------|-----------|
| Training (Megatron-LM, large Transformer) | 0.40-0.55 | Well-optimized GEMM + FlashAttention |
| Training (PyTorch eager, small model) | 0.08-0.15 | Kernel launch overhead dominates |
| Inference decode, batch=1 | 0.01-0.05 | Memory-bound; compute nearly idle |
| Inference decode, batch=32+ | 0.15-0.35 | Batch amortizes weight loading |
| Inference prefill, long context | 0.30-0.50 | Compute-bound GEMM + attention |
| TinyML (TFLite Micro on ESP32) | 0.05-0.15 | Interpreter overhead, no tensor cores |

These ranges come from published benchmarks (MLPerf, PaLM training reports,
Megatron-LM papers) and our own measurements. They are not universal truths --
they are informed defaults.

---

## How to Calibrate Efficiency for Your Workload

If you have access to real hardware, you can measure eta directly. This is the
gold standard -- it turns the simulator from an estimation tool into a
calibrated prediction tool.

### Step 1: Run Your Workload

Run your actual training or inference workload on the target hardware. Use
your real code, your real data, and your real batch size. Do not use synthetic
benchmarks -- they measure a different workload.

### Step 2: Measure Actual Throughput

Record the achieved throughput in tokens/second (for language models) or
samples/second (for vision models). Use steady-state measurements after
warmup, excluding the first few batches.

### Step 3: Back-Calculate Efficiency

```python
# Your measurements
measured_throughput = 1200  # tokens/sec (from step 2)

# Known quantities
model_flops_per_token = 1.4e12    # FLOPs per token for your model
peak_flops = 989e12               # H100 SXM FP16 peak

# Back-calculate
theoretical_max_throughput = peak_flops / model_flops_per_token
eta = measured_throughput / theoretical_max_throughput

print(f"eta = {eta:.3f}")  # e.g., eta = 0.423
```

### Step 4: Use Calibrated Efficiency for What-If Analysis

Now you have a grounded eta. Use it to explore counterfactuals:

```python
# "What if I moved from H100 to B200?"
result = Engine.solve(
    model=my_model,
    hardware=Hardware.B200,
    batch_size=32,
    precision="fp16",
    efficiency=0.423  # calibrated from H100 measurement
)
```

**Important caveat:** eta does not transfer perfectly across GPU generations
(see the fallacies section below). But it transfers much better than having no
measurement at all. A calibrated eta from one GPU gives you a reasonable
starting point for another, which you can then refine.

---

## Common Mistakes with Efficiency

### Fallacy: "eta = 0.5 is always a good default"

It is not. The appropriate eta varies by **orders of magnitude** depending on
the workload:

- LLM decode at batch=1: eta ~ 0.01-0.05
- Optimized large-batch training: eta ~ 0.40-0.55
- Small model in eager mode: eta ~ 0.08-0.15

Using eta = 0.5 for decode-phase inference will overestimate throughput by
10-50x. Always match your efficiency to the workload regime.

### Fallacy: "Higher eta is always better"

Not necessarily. A workload running at INT4 precision might report higher
hardware utilization (more ops/sec relative to INT4 peak) while producing
lower-quality results than the same model at FP16 with lower eta. Efficiency
measures hardware utilization, not model quality.

Similarly, a workload that maximizes eta by using enormous batch sizes may
achieve high throughput but unacceptable latency for real-time serving. The
right eta depends on the objective.

### Fallacy: "eta transfers across GPU generations"

It does not -- at least not exactly. Different architectures have different
bottleneck profiles:

- **A100 to H100:** The Transformer Engine and higher memory bandwidth on H100
  shift many workloads from memory-bound to compute-bound, changing eta
  significantly.
- **Dense to sparse:** Architectures with hardware sparsity support (like
  NVIDIA's 2:4 structured sparsity) achieve different eta for the same
  workload.
- **GPU to TPU:** Completely different memory hierarchy, interconnect topology,
  and programming model. An eta measured on GPU tells you almost nothing about
  TPU performance.

When moving across architectures, use your calibrated eta as a *starting point*,
then adjust based on known architectural differences.

### Fallacy: "You can improve eta just by buying faster hardware"

Eta is a *ratio*. Buying a faster GPU increases both numerator and denominator.
If the bottleneck is software overhead (Python dispatch, kernel launch latency),
faster hardware may actually *decrease* eta because the hardware ceiling rises
but the software overhead stays constant.

Improving eta requires optimizing the software stack: better kernels, operation
fusion, reduced framework overhead, efficient communication overlap. This is
the domain of ML systems engineering.

---

## Further Reading

- **PaLM Training Report** (Chowdhery et al., 2022) -- Reports MFU of 46.2%
  for PaLM 540B on TPUv4, one of the most carefully measured large-scale
  training efficiency numbers published.
- **Megatron-LM** (Shoeybi et al., 2020) -- The reference implementation for
  efficient large model training. Published efficiency numbers across model
  sizes and parallelism configurations.
- **MLPerf Training** (mattson et al., 2020) -- Standardized benchmarks that
  provide comparable efficiency measurements across hardware platforms.
- **Computer Architecture: A Quantitative Approach** (Hennessy & Patterson) --
  Chapter 1's treatment of CPI and the CPU performance equation is the direct
  intellectual ancestor of how mlsysim treats efficiency.

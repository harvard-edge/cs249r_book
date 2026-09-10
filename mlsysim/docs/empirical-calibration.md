# Empirical Calibration: mlsysim vs. Published Benchmarks

This document compares mlsysim analytical predictions against published benchmark
results for common ML workloads. The goal is to validate that the framework's
first-principles models produce results in the correct physical range, and to
characterize the accuracy envelope that users should expect.

**Date:** 2026-04-01
**mlsysim version:** 0.1.0
**Methodology:** Each configuration was run through the appropriate solver
(`SingleNodeModel` for training throughput, `ServingModel` for inference latency,
`calc_transformer_training_flops` for FLOPs estimates). Default efficiency
parameters were used (eta=0.10 for training, eta=0.50 for serving) unless
otherwise noted.

The benchmark scalar values live in `Literature.Benchmarks`, beside the other
cited literature anchors. The executable validation contract in
`mlsysim.engine.empirical` only composes those literature values with canonical
`Models.*` and `Hardware.*` registry entries, plus the tolerance/range and review
rationale. This keeps model definitions, hardware specifications, and published
benchmark numbers single-sourced inside MLSys·im.

---

## Calibration Table

| # | Configuration | mlsysim Prediction | Benchmark Anchor | Source | Delta |
|---|---|---|---|---|---|
| 1 | ResNet-50 A100 bs=256 training (eta=0.10) | 2,499 img/s | `Literature.Benchmarks.ResNet50A100TrainThroughput` | MLPerf Training v3.0 (single A100-80GB) | -22% |
| 2 | ResNet-50 H100 bs=256 training (eta=0.10) | 7,677 img/s | `Literature.Benchmarks.ResNet50H100TrainThroughput` | MLPerf Training v3.1 (single H100-80GB) | +54% |
| 3 | Llama-3-8B H100 bs=1 decode ITL (eta=0.50) | 5.2 ms | `Literature.Benchmarks.Llama3_8B_H100_ITLLower/Upper` | TensorRT-LLM benchmarks (FP16, H100-80GB) | within range |
| 4 | Llama-3-8B H100 bs=32 decode ITL (eta=0.50) | 7.7 ms | `Literature.Benchmarks.Llama3_8B_H100_ITLLower/Upper` | vLLM / TensorRT-LLM batched benchmarks | within range |
| 5 | Llama-3-8B H100 bs=1 prefill TTFT (eta=0.50) | 66.5 ms | `Literature.Benchmarks.Llama3_8B_H100_ITLLower/Upper` | TensorRT-LLM benchmarks (seq=2048) | within range |
| 6 | GPT-3 175B training FLOPs (6PD rule) | 3.15e23 | `Models.Language.GPT3.training_ops` | Brown et al. (2020), Table D.1 | +0.3% |

---

## Analysis

### What works well

**LLM inference latency (Configs 3-5).** The decode ITL predictions land squarely
within published ranges. This is expected: decode is memory-bandwidth-bound, and
the model correctly computes `(weights + KV_cache) / bandwidth`. The decode ITL
is insensitive to the `efficiency` parameter because compute is not on the
critical path -- exactly matching the physical regime where auto-regressive
decoding is a streaming memory read, not a compute problem.

**Training FLOPs (Config 6).** The `6ND` approximation (6 * parameters * tokens)
for transformer training FLOPs matches the GPT-3 paper's reported value to within
0.3%. This validates the fundamental FLOP counting formula.

### Where the model diverges

**CNN training throughput (Configs 1-2).** The default efficiency parameter
(eta=0.10) produces predictions that bracket reality but don't hit both targets
simultaneously:

- **A100** prediction is 22% low at eta=0.10. Setting eta=0.13 brings the
  prediction into the `Literature.Benchmarks.ResNet50A100TrainThroughput`
  validation band.
- **H100** prediction is 54% high at eta=0.10. Setting eta=0.065 brings the
  prediction into the `Literature.Benchmarks.ResNet50H100TrainThroughput`
  validation band.

This asymmetry reveals a real insight: **the efficiency parameter is not a universal
constant.** It encodes the gap between peak datasheet FLOP/s and actual sustained
throughput, which depends on:

1. **Tensor core utilization** -- ResNet-50 has many small convolutions that may
   not saturate H100's larger tensor cores as efficiently as A100's.
2. **Memory system pressure** -- H100's 3x higher peak FLOP/s amplifies any
   memory bottleneck (the model uses peak FP16 FLOP/s of 989 TFLOP/s for H100
   vs. 312 TFLOP/s for A100, but real ResNet-50 kernels achieve a lower fraction
   of the H100 peak).
3. **Framework overhead** -- Batch normalization layers, data loading, and
   gradient synchronization are not modeled in the roofline calculation.

### Key takeaway

The single-parameter efficiency model works well within a hardware family but
does not transfer across hardware generations without re-calibration. This is a
known limitation of roofline-based analytical models and is explicitly surfaced
to students in the textbook.

---

## Per-configuration efficiency calibration

For users who need higher accuracy, the following per-configuration efficiency
values minimize error against published benchmarks:

| Configuration | Calibrated eta | mlsysim Result | Benchmark Anchor | Status |
|---|---|---|---|---|
| ResNet-50 A100 bs=256 training | 0.13 | 3,234 img/s | `Literature.Benchmarks.ResNet50A100TrainThroughput` | within band |
| ResNet-50 H100 bs=256 training | 0.065 | 5,070 img/s | `Literature.Benchmarks.ResNet50H100TrainThroughput` | within band |
| Llama-3-8B H100 bs=1 decode | any (memory-bound) | 5.2 ms | `Literature.Benchmarks.Llama3_8B_H100_ITLLower/Upper` | within range |
| Llama-3-8B H100 bs=32 decode | any (memory-bound) | 7.7 ms | `Literature.Benchmarks.Llama3_8B_H100_ITLLower/Upper` | within range |
| GPT-3 training FLOPs | N/A (closed-form) | 3.15e23 | `Models.Language.GPT3.training_ops` | within band |

---

## Methodology notes

1. **Efficiency parameter (eta).** This is the fraction of peak hardware
   throughput (FLOP/s or GB/s) that the workload actually achieves. It
   consolidates all sources of inefficiency: tensor core utilization, memory
   stalls, kernel launch overhead, data pipeline stalls, and framework
   overhead. It is NOT the same as MFU (Model FLOPs Utilization), which is
   defined as `observed_throughput / theoretical_peak_throughput`.

2. **Published benchmarks.** We compare against:
   - **MLPerf Training v3.0/v3.1**: Industry-standard training benchmarks.
     Single-accelerator numbers used (not multi-GPU or multi-node). ResNet-50
     throughput measured as images/second during convergence training.
   - **TensorRT-LLM benchmarks**: NVIDIA's optimized inference engine.
     H100-80GB SXM, FP16 precision, KV-cache enabled.
   - **vLLM benchmarks**: Open-source LLM serving engine. Batched decode
     latency with PagedAttention.
   - **Brown et al. (2020)**: "Language Models are Few-Shot Learners."
     GPT-3's benchmark target is exposed as
     `Models.Language.GPT3.training_ops`.

3. **Decode ITL is efficiency-insensitive.** The serving model correctly
   identifies decode as memory-bandwidth-bound. The ITL formula is:
   `ITL = (model_weights + KV_cache) / memory_bandwidth + framework_tax`.
   Since this does not involve the compute efficiency parameter, varying eta
   from 0.3 to 0.8 produces identical ITL values. This matches the physical
   reality of auto-regressive decoding.

4. **Hardware specs used.** Hardware values come from the Silicon Zoo:
   `Hardware.Cloud.A100` and `Hardware.Cloud.H100`. Do not copy peak FLOP/s,
   memory bandwidth, or capacity numbers into calibration scripts.

---

## Reproducing these results

```python
import mlsysim
from mlsysim.solvers import SingleNodeModel, ServingModel
from mlsysim.physics import calc_transformer_training_flops

# Config 1: ResNet-50 / A100 / training
resnet = mlsysim.Models.Vision.ResNet50
a100 = mlsysim.Hardware.Cloud.A100
r1 = SingleNodeModel().solve(
    resnet, a100,
    batch_size=256, efficiency=0.10, is_training=True
)
print(f"ResNet-50 A100: {r1.throughput.m_as('1/s'):.0f} img/s")

# Config 2: ResNet-50 / H100 / training
h100 = mlsysim.Hardware.Cloud.H100
r2 = SingleNodeModel().solve(
    resnet, h100,
    batch_size=256, efficiency=0.10, is_training=True
)
print(f"ResNet-50 H100: {r2.throughput.m_as('1/s'):.0f} img/s")

# Config 3: Llama-3-8B / H100 / decode bs=1
llama = mlsysim.Models.Language.Llama3_8B
r3 = ServingModel().solve(
    llama, h100,
    seq_len=2048, batch_size=1, efficiency=0.50
)
print(f"Llama-3-8B bs=1 ITL: {r3.itl.m_as('ms'):.1f} ms")

# Config 4: Llama-3-8B / H100 / decode bs=32
r4 = ServingModel().solve(
    llama, h100,
    seq_len=2048, batch_size=32, efficiency=0.50
)
print(f"Llama-3-8B bs=32 ITL: {r4.itl.m_as('ms'):.1f} ms")

# Config 6: GPT-3 training FLOPs
gpt3 = mlsysim.Models.Language.GPT3
target = mlsysim.Models.Language.GPT3.training_ops
flops = calc_transformer_training_flops(gpt3.parameters, gpt3.training_tokens)
print(f"GPT-3 FLOPs: {flops.to('flop').magnitude:.2e}")
print(f"GPT-3 target: {float(target):.2e}")
```

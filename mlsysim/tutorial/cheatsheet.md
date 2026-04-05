# mlsysim Cheat Sheet

Single-page reference for the ISCA tutorial.

---

## The Iron Law of ML Training

```
Time = FLOPs / (N x Peak_FLOPS x MFU x eta_scaling x Goodput)
```

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| FLOPs | Total operations for the workload | 6PD for training (Chinchilla) |
| N | Number of accelerators | 1 to 100,000+ |
| Peak_FLOPS | Hardware peak (per device) | 312 TFLOPS (H100 FP16) |
| MFU | Model FLOPs Utilization | 0.30 - 0.55 |
| eta_scaling | Scaling efficiency (communication overhead) | 0.70 - 0.95 |
| Goodput | Fraction of time doing useful work (1 - failures - checkpoints) | 0.85 - 0.98 |

---

## The 5 Key Equations

### 1. Roofline Bottleneck

```
T = max(FLOPs / Peak_effective, Bytes / BW_effective)
```

If compute time > memory time, you are **compute-bound** (increase FLOPS).
If memory time > compute time, you are **memory-bound** (increase bandwidth).

### 2. KV-Cache Memory (PagedAttention)

```
KV_bytes = 2 x L x H_kv x D_head x S x B x bytes_per_param
```

L = layers, H_kv = KV heads, D_head = head dimension, S = sequence length, B = batch size.
Factor of 2 accounts for both Key and Value tensors.

### 3. Ring AllReduce Communication

```
T_allreduce = 2(N-1)/N x M/BW + 2(N-1) x alpha
```

N = workers, M = message bytes, BW = link bandwidth, alpha = per-message latency.
As N grows large, volume term approaches 2M/BW (bandwidth-optimal).

### 4. Chinchilla Scaling Law

```
C = 6PD        (compute-optimal training cost)
P* = sqrt(C/120)  (optimal parameter count for budget C)
```

P = parameters, D = tokens, C = total FLOPs. Training is compute-optimal when
D ~ 20P (20 tokens per parameter).

### 5. Carbon Footprint

```
CO2_kg = Energy_kWh x PUE x Carbon_Intensity_gCO2/kWh / 1000
```

PUE = Power Usage Effectiveness (1.0 = perfect, 1.1 = typical hyperscale).
Carbon intensity varies 100x by region (1 gCO2/kWh hydro vs. 800 gCO2/kWh coal).

---

## Efficiency Parameter Guide

| Parameter | Description | Low | Typical | High |
|-----------|-------------|-----|---------|------|
| `efficiency` | MFU (fraction of peak FLOPS achieved) | 0.10 | 0.30-0.50 | 0.65 |
| `mfu` | Same as efficiency, used in fleet solvers | 0.10 | 0.35 | 0.55 |
| Batch=1 inference | LLM decode (memory-bound) | 0.01 | 0.05 | 0.15 |
| Batched inference | LLM prefill / CNN inference | 0.20 | 0.40 | 0.60 |
| Training (single node) | Typical training loop | 0.20 | 0.40 | 0.55 |
| Training (distributed) | Large cluster with comms | 0.15 | 0.30 | 0.45 |
| FlashAttention | Fused attention kernel | 0.50 | 0.60 | 0.70 |
| TinyML (MCU) | Microcontroller inference | 0.05 | 0.15 | 0.30 |

---

## Quick API Reference

### 1. Single-Node Roofline

```python
from mlsysim import Engine, Hardware, Models

profile = Engine.solve(
    model=Models.Llama3_8B,
    hardware=Hardware.H100,
    batch_size=1,
    precision="fp16",       # "fp32", "fp16", "int8", "int4", "fp8"
    efficiency=0.5,
    is_training=False,      # True for training memory/FLOPs
)
# Returns: PerformanceProfile with .latency, .throughput, .bottleneck,
#          .memory_footprint, .mfu, .energy, .feasible
```

### 2. LLM Serving (Prefill + Decode)

```python
from mlsysim import ServingModel, Hardware, Models

result = ServingModel().solve(
    model=Models.Llama3_8B,
    hardware=Hardware.H100,
    seq_len=4096,
    batch_size=32,
    precision="fp16",
)
# Returns: ServingResult with .ttft, .itl, .kv_cache_size,
#          .total_memory_required, .feasible
```

### 3. Distributed Training (3D Parallelism)

```python
from mlsysim import DistributedModel, Models, Systems

result = DistributedModel().solve(
    model=Models.Llama3_70B,
    fleet=Systems.Clusters.Research_256,
    batch_size=1024,
    tp_size=8, pp_size=4,
    precision="fp16",
    efficiency=0.4,
    overlap_comm=True,
)
# Returns: DistributedResult with .scaling_efficiency,
#          .step_latency_total, .dp_communication_latency,
#          .bubble_fraction, .effective_throughput, .parallelism
```

### 4. Compression (Quantization / Pruning)

```python
from mlsysim import CompressionModel, Hardware, Models

result = CompressionModel().solve(
    model=Models.Llama3_8B,
    hardware=Hardware.H100,
    method="quantization",  # "quantization", "pruning", "distillation"
    target_bitwidth=4,      # 4, 8, 16
)
# Returns: CompressionResult with .compression_ratio, .compressed_size_gb,
#          .memory_savings_pct, .inference_speedup, .estimated_accuracy_delta
```

### 5. Sustainability and Economics

```python
from mlsysim import SustainabilityModel, EconomicsModel, Systems, Infra

fleet = Systems.Clusters.Research_256

co2 = SustainabilityModel().solve(fleet, duration_days=30, datacenter=Infra.Quebec, mfu=0.4)
# Returns: SustainabilityResult with .total_energy_kwh, .carbon_footprint_kg,
#          .water_usage_liters, .pue

tco = EconomicsModel().solve(fleet, duration_days=365, mfu=0.4)
# Returns: EconomicsResult with .tco_usd, .capex_usd, .total_opex_usd,
#          .opex_energy_usd, .carbon_footprint_kg
```

---

## The 22 Walls at a Glance

| # | Wall | One-Liner |
|---|------|-----------|
| 1 | Compute | Peak FLOPS ceiling of a single accelerator |
| 2 | Memory | HBM capacity and bandwidth ceilings |
| 3 | Software | Gap between peak and achieved FLOPS (MFU) |
| 4 | Serving | LLM inference: compute-bound prefill vs. memory-bound decode |
| 5 | Batching | Static batching wastes memory through KV-cache fragmentation |
| 6 | Streaming | Wafer-scale shifts bottleneck from HBM to injection interconnect |
| 7 | Tail Latency | P99 latency grows non-linearly as utilization approaches 1.0 |
| 8 | Ingestion | Storage I/O must supply data at the rate the accelerator consumes it |
| 9 | Transformation | CPU preprocessing cannot keep pace with accelerator throughput |
| 10 | Locality | Network topology limits bisection bandwidth between nodes |
| 11 | Complexity | Chinchilla scaling laws govern compute-optimal training |
| 12 | Reasoning | Inference-time compute scales with reasoning chain length |
| 13 | Fidelity | Compression trades model fidelity for efficiency |
| 14 | Communication | Distributed training requires synchronization across N nodes |
| 15 | Fragility | Component failures are inevitable at scale (MTBF/N) |
| 16 | Multi-tenant | Shared clusters introduce queueing delays |
| 17 | Capital | Total cost of ownership bounds what is economically feasible |
| 18 | Sustainability | Energy consumption converts to carbon and water footprint |
| 19 | Checkpoint | Periodic state saves impose I/O burst penalties on training MFU |
| 20 | Safety | Privacy and fairness guarantees impose computational overhead |
| 21 | Sensitivity | Identifies the binding constraint via numerical partial derivatives |
| 22 | Synthesis | Inverse Roofline: derive hardware specs from an SLA target |

---

## Hardware Quick Reference

| Accelerator | Peak FP16 (TFLOPS) | HBM (GiB) | BW (TB/s) | TDP (W) |
|-------------|-------------------|------------|-----------|---------|
| V100 | 125 | 32 | 0.9 | 300 |
| A100 | 312 | 80 | 2.0 | 400 |
| H100 | 989 | 80 | 3.35 | 700 |
| H200 | 989 | 141* | 4.8 | 700 |
| B200 | 2,250 | 192 | 8.0 | 1,000 |
| nRF52840 | 0.000064 | 0.001 (1 MB flash) | 0.000064 | 0.015 |

*H200 capacity listed as 141 GB in registry (non-binary).

Access via: `Hardware.A100`, `Hardware.H100`, `Hardware.Tiny.nRF52840`, etc.

---

## Model Quick Reference

| Model | Parameters | Architecture | Access |
|-------|-----------|--------------|--------|
| ResNet-50 | 25.6M | CNN | `Models.ResNet50` |
| Llama-3-8B | 8.03B | Transformer | `Models.Llama3_8B` |
| Llama-3-70B | 70.6B | Transformer | `Models.Llama3_70B` |
| GPT-3 | 175B | Transformer | `Models.GPT3` |
| DS-CNN (KWS) | 26K | CNN | `Models.Tiny.DS_CNN` |
| MobileNetV2 | 3.4M | CNN | `Models.MobileNetV2` |

# mlsysim Cheat Sheet

Single-page reference for the ISCA tutorial.

---

## The Iron Law of ML Training

```
Time = FLOPs / (N x Peak_FLOP/s x MFU x eta_scaling x Goodput)
```

| Symbol | Meaning | Typical Range |
|--------|---------|---------------|
| FLOPs | Total operations for the workload | 6PD for training (Chinchilla) |
| N | Number of accelerators | 1 to 100,000+ |
| Peak_FLOP/s | Hardware peak (per device) | 312 TFLOP/s (A100 FP16), 989 TFLOP/s (H100 FP16) |
| MFU | Model FLOPs Utilization | 0.30 - 0.50 |
| eta_scaling | Scaling efficiency (communication overhead) | 0.70 - 0.95 |
| Goodput | Fraction of time doing useful work (`1 - waste`, Young-Daly ~ `delta/tau + tau/(2M)`) | 0.85 - 0.98 |

---

## The 5 Key Equations

### 1. Roofline Bottleneck

```
T = max(FLOPs / Peak_effective, Bytes / BW_effective)
```

If compute time > memory time, the workload is **compute-bound** (increase FLOP/s).
If memory time > compute time, the workload is **memory-bound** (increase bandwidth).

### 2. Dense KV-Cache Footprint

```
KV_bytes = 2 x L x H_kv x D_head x S x B x bytes_per_elem
```

L = layers, H_kv = KV heads, D_head = head dimension, S = sequence length, B = batch size, `bytes_per_elem` = activation element bytes (`2` for FP16/BF16, `1` for FP8/INT8).
Factor of 2 accounts for both Key and Value tensors (PagedAttention eliminates contiguous over-reservation waste across variable-length sequences).

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

### 5. Carbon & Water Footprint

```
CO2_kg  = IT_Energy_kWh x PUE x Carbon_Intensity_gCO2/kWh / 1000
Water_L = IT_Energy_kWh x WUE_site
```

PUE = Power Usage Effectiveness (1.0 = ideal, 1.12 = typical hyperscale).
WUE_site = on-site Water Usage Effectiveness (L / kWh_IT, applied to pre-PUE IT equipment energy).
Carbon intensity varies by 41x–82x across registry grids (10 gCO2/kWh hydro in Norway and 20 gCO2/kWh in Quebec vs. 820 gCO2/kWh coal in Poland).

---

## Efficiency Parameter Guide

| Parameter | Description | Low | Typical | High |
|-----------|-------------|-----|---------|------|
| `efficiency` | MFU (fraction of peak FLOP/s achieved) | 0.10 | 0.30-0.50 | 0.55 |
| `mfu` | Same as efficiency, used in fleet solvers | 0.10 | 0.35 | 0.50 |
| Batch=1 inference | LLM decode (memory-bound) | 0.01 | 0.05 | 0.15 |
| Batched inference | LLM prefill / CNN inference | 0.20 | 0.35 | 0.45 |
| Training (single node) | Typical training loop | 0.20 | 0.40 | 0.50 |
| Training (distributed) | Large cluster with comms | 0.15 | 0.30 | 0.45 |
| FlashAttention | Fused attention kernel | 0.50 | 0.60 | 0.70 |
| TinyML (MCU) | Microcontroller inference | 0.05 | 0.15 | 0.30 |

---

## Quick API Reference

### 1. Single-Node Roofline

```python
from mlsysim import Engine, Hardware, Models

profile = Engine.solve(
    model=Models.Language.Llama3_8B,
    hardware=Hardware.Cloud.H100,
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
from mlsysim import Hardware, Models
from mlsysim.solvers import ServingModel

result = ServingModel().solve(
    model=Models.Language.Llama3_8B,
    hardware=Hardware.Cloud.H100,
    seq_len=4096,
    batch_size=32,
    precision="fp16",
)
# Returns: ServingResult with .ttft, .itl, .kv_cache_size,
#          .total_memory_required, .feasible
```

### 3. Distributed Training (3D Parallelism)

```python
from mlsysim import Models, Systems
from mlsysim.solvers import DistributedModel

result = DistributedModel().solve(
    model=Models.Language.Llama3_70B,
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
from mlsysim import Hardware, Models
from mlsysim.solvers import CompressionModel

result = CompressionModel().solve(
    model=Models.Language.Llama3_8B,
    hardware=Hardware.Cloud.H100,
    method="quantization",      # "quantization", "pruning", "distillation"
    target_bitwidth=4,          # 4, 8, 16
    baseline_precision="fp16",  # defaults to "fp32" if omitted
    group_size=128,             # optional FP16 group-scale overhead (4.125 b/w -> 3.88x)
    accuracy_delta=-0.015,      # caller-supplied empirical quality delta (-1.5%)
)
# Returns: CompressionResult with .compression_ratio, .compressed_size_gb,
#          .memory_savings_pct, .inference_speedup, .estimated_accuracy_delta
```

### 5. Sustainability and Economics

```python
from mlsysim import Systems, Infrastructure
from mlsysim.solvers import SustainabilityModel, EconomicsModel

fleet = Systems.Clusters.Research_256

co2 = SustainabilityModel().solve(fleet, duration_days=30, datacenter=Infrastructure.Grids.Quebec, mfu=0.4)
# Returns: SustainabilityResult with .total_energy_kwh, .carbon_footprint_kg,
#          .water_usage_liters, .pue

tco = EconomicsModel().solve(fleet, duration_days=365, mfu=0.4)
# Returns: EconomicsResult with .tco_usd, .capex_usd, .total_opex_usd,
#          .opex_energy_usd, .carbon_footprint_kg
```

---

## The 20 Walls + 2 Analysis Solvers at a Glance

`mlsysim.engine.walls.ALL_WALLS` contains 22 entries: 20 physical/operational walls (rows 1–20) plus 2 cross-cutting sensitivity and inverse-synthesis solvers (rows 21–22).

| # | Wall / Solver | One-Liner |
|---|---------------|-----------|
| 1 | Compute | Peak FLOP/s ceiling of a single accelerator |
| 2 | Memory | HBM capacity and bandwidth ceilings |
| 3 | Software | Gap between peak and achieved FLOP/s (MFU) |
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

| Accelerator | Peak FP16 (TFLOP/s) | Memory | BW (TB/s) | TDP (W) |
|-------------|---------------------|--------|-----------|---------|
| V100 | 125 | 32 GiB | 0.90 | 300 |
| A100 | 312 | 80 GiB | 2.04 | 400 |
| H100 | 989 | 80 GiB | 3.35 | 700 |
| H200 | 989 | 141 GiB | 4.80 | 700 |
| B200 | 2,250 | 180 GiB | 8.00 | 1,000 |
| nRF52840 | 0.000064 | 1 MiB | 0.000064 | 0.015 |

Access via: `Hardware.Cloud.A100`, `Hardware.Cloud.H100`, `Hardware.Tiny.nRF52840`, etc.

---

## Model Quick Reference

| Model | Parameters | Architecture | Access |
|-------|-----------|--------------|--------|
| ResNet-50 | 25.6M | CNN | `Models.Vision.ResNet50` |
| Llama-3-8B | 8.03B | Transformer | `Models.Language.Llama3_8B` |
| Llama-3-70B | 70.6B | Transformer | `Models.Language.Llama3_70B` |
| GPT-3 | 175B | Transformer | `Models.Language.GPT3` |
| DS-CNN (KWS) | 200K | CNN | `Models.Tiny.DS_CNN` |
| MobileNetV2 | 3.5M | CNN | `Models.Vision.MobileNetV2` |

---

## Textbook & Curriculum Cross-Links

See <https://mlsysbook.ai/mlsysim/curriculum-map.html> for direct links from every equation, wall, and solver to the corresponding *Machine Learning Systems* textbook chapter, interactive Marimo lab, and lecture slides.

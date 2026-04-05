---
title: "How Much Memory Does Llama-3 70B Actually Need?"
description: "A 3-minute calculation with mlsysim that answers the question every ML engineer asks."
date: 2026-04-01
categories: [serving, memory, llm]
---

# How Much Memory Does Llama-3 70B Actually Need?

Every ML engineer eventually asks: *"Can I serve Llama-3 70B on my hardware?"*

The answer depends on three things: **precision**, **KV cache**, and **batch size**.
Let's calculate it in 30 seconds.

## The Weights

```python
import mlsysim

llama70b = mlsysim.Models.Language.Llama3_70B

# FP16: 2 bytes per parameter
fp16_size = llama70b.size_in_bytes()
print(f"FP16 weights: {fp16_size.to('GB'):.1f}")
# → 140.0 GB

# INT4: 0.5 bytes per parameter
int4_size = llama70b.size_in_bytes(mlsysim.ureg("0.5 byte"))
print(f"INT4 weights: {int4_size.to('GB'):.1f}")
# → 35.0 GB
```

**Result:** 140 GB in FP16, 35 GB in INT4.

An H100 has 80 GB. So Llama-3 70B in FP16 **does not fit on one GPU**.
You need either tensor parallelism (TP=2) or quantization to INT4.

## The KV Cache (The Hidden Memory Consumer)

Weights are only half the story. Each active request needs a KV cache:

```python
from mlsysim.core.formulas import calc_kv_cache_size

# Llama-3 70B: 80 layers, 8 KV heads (GQA), 128 dim per head
kv_per_request = calc_kv_cache_size(
    n_layers=80, n_heads=8, head_dim=128,
    seq_len=4096, batch_size=1, bytes_per_elem=2
)
print(f"KV cache per request (4K context): {kv_per_request.to('MB'):.0f}")
# → 160 MB per request
```

At 160 MB per request, an 80 GB GPU serving INT4 Llama-3 70B has:
- 35 GB for weights
- 45 GB remaining for KV cache
- **45,000 MB / 160 MB ≈ 280 concurrent requests max**

But at 32K context? KV cache is 1.28 GB per request → only **35 requests**.

## The Full Picture

```python
from mlsysim.core.solver import ServingModel

result = ServingModel().solve(
    mlsysim.Models.Language.Llama3_70B,
    mlsysim.Hardware.Cloud.H100,
    seq_len=4096,
    batch_size=1,
    precision="fp16"
)
print(f"Feasible: {result.feasible}")        # → False (doesn't fit!)
print(f"Memory util: {result.memory_utilization:.1%}")
```

**The punchline:** A "70B model" doesn't just need 140 GB. It needs
140 GB + (KV cache × concurrent requests). At production batch sizes,
the KV cache can consume MORE memory than the weights.

## What To Do About It

| Strategy | Memory Impact | Trade-off |
|----------|-------------|-----------|
| INT4 quantization | 4× smaller weights | ~2-5% accuracy loss |
| GQA (already in Llama-3) | 8× smaller KV cache | None (architectural) |
| KV cache INT8 | 2× smaller KV cache | Negligible quality loss |
| Tensor parallelism (TP=2) | Split across 2 GPUs | Adds NVLink communication |
| PagedAttention (vLLM) | Eliminates KV fragmentation | ~20-40% more concurrent requests |

## Try It Yourself

```bash
pip install mlsysim
mlsysim serve Llama3_70B H100 --seq-len 4096 --batch-size 1
```

---

*This analysis was computed with [mlsysim](https://mlsysbook.ai/mlsysim),
a first-principles analytical calculator for ML systems.
All constants are traceable to hardware datasheets.*

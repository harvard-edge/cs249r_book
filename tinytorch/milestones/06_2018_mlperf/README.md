# Milestone 06: MLPerf - The Optimization Era (2018)

## Historical Context

As ML models grew larger and deployment became critical, the community needed **systematic optimization methodologies**. MLCommons' MLPerf (2018) established standardized benchmarking and optimization workflows, shifting the focus from "can we build it?" to "can we deploy it efficiently?"

This milestone teaches **production optimization** - the systematic process of profiling, compressing, and accelerating models for real-world deployment.

## What You're Building

A complete MLPerf-style optimization pipeline that takes YOUR networks from previous milestones and makes them production-ready!

## Required Modules

| Module | Component | What It Provides |
|--------|-----------|------------------|
| Module 01-03 | Tensor, Linear, ReLU | YOUR base components |
| Module 11 | Embeddings | YOUR token embeddings |
| Module 12 | Attention | YOUR multi-head attention |
| Module 14 | Profiling | YOUR profiler for measurement |
| Module 15 | Quantization | YOUR INT8/FP16 implementations |
| Module 16 | Compression | YOUR pruning techniques |
| Module 17 | Acceleration | YOUR vectorized operations |

## Milestone Structure

This milestone has **two scripts**, each covering different optimization techniques:

### 01_optimization_olympics.py
**Purpose:** Optimize static models (MLP, CNN)

Uses YOUR implementations:
- **Module 14 (Profiling)**: Measure parameters, latency, size
- **Module 15 (Quantization)**: FP32 → INT8 (4× compression)
- **Module 16 (Compression)**: Pruning (remove weights)

Networks from:
- DigitMLP (Milestone 03)
- SimpleCNN (Milestone 04)

### 02_generation_speedup.py
**Purpose:** Speed up Transformer generation

Uses YOUR implementations:
- **Module 11 (Embeddings)**: Token embeddings
- **Module 12 (Attention)**: Multi-head attention
- **Module 14 (Profiling)**: Measure speedup
- **Module 18 (KV Cache)**: Cache K,V for 6-10× speedup

Networks from:
- MinimalTransformer (Milestone 05)

## Expected Results

### Static Model Optimization (01)
| Optimization | Size | Accuracy | Notes |
|-------------|------|----------|-------|
| Baseline | 100% | 85-90% | Full precision |
| + Quantization | 25% | 84-89% | INT8 weights |
| + Pruning | 12.5% | 82-87% | 50% weights removed |

### Generation Speedup (02)
| Mode | Time/Token | Speedup |
|------|-----------|---------|
| Without Cache | ~10ms | 1× |
| With KV Cache | ~1ms | 6-10× |

## Running the Milestone

```bash
# Optimize MLP/CNN (profiling + quantization + pruning)
python milestones/06_2018_mlperf/01_optimization_olympics.py

# Speed up Transformer generation (KV caching)
python milestones/06_2018_mlperf/02_generation_speedup.py
```

Or via tito:
```bash
tito milestone run 06
```

## Key Learning

Unlike earlier milestones where you "build and run," optimization requires:
1. **Measure** (profile to find bottlenecks)
2. **Optimize** (apply targeted techniques)
3. **Validate** (check accuracy didn't degrade)
4. **Repeat** (iterate until deployment targets met)

This is **ML systems engineering** - the skill that ships products!

## Further Reading

- **MLPerf**: https://mlcommons.org/en/inference-edge-11/
- **Deep Compression** (Han et al., 2015): https://arxiv.org/abs/1510.00149
- **Efficient Transformers Survey**: https://arxiv.org/abs/2009.06732

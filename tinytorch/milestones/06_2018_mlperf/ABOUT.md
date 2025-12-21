# Milestone 06: MLPerf - The Optimization Era (2018)

```{tip} What You'll Learn
- The systematic optimization workflow: measure, optimize, validate, repeat
- Why profiling before optimizing beats heroic rewrites
- How to achieve 8× compression and 10× speedup with minimal accuracy loss
```

## Overview

2018. The ML world has a dirty secret: everyone's publishing state-of-the-art results, but nobody can deploy them. GPT-2 takes 30 seconds to generate a single sentence. BERT won't fit on edge devices. Production teams are stuck using years-old models because new ones are too slow, too big, too expensive.

MLCommons launches MLPerf - the first systematic benchmark for ML systems engineering. The message is clear: research breakthroughs don't matter if you can't ship them. Optimization isn't an afterthought; it's a core competency.

This milestone teaches you the same systematic approach production ML engineers use. You'll compress YOUR models 8× and speed up YOUR transformer generation 10×. That's the difference between research demos and shipped products.

## What You'll Build

A complete MLPerf-style optimization pipeline:
1. **Static Model Optimization**: Profile, quantize, and prune MLP/CNN
2. **Generation Speedup**: KV-cache acceleration for transformers

```
Measure --> Optimize --> Validate --> Repeat
```

## Prerequisites

| Module | Component | What It Provides |
|--------|-----------|------------------|
| 01-13 | Foundation + Architectures | Models to optimize |
| 14 | Profiling | YOUR measurement tools |
| 15 | Quantization | YOUR INT8/FP16 implementations |
| 16 | Compression | YOUR pruning techniques |
| 17 | Acceleration | YOUR vectorized operations |
| 18 | Memoization | YOUR KV-cache for generation |

## Running the Milestone

Before running, ensure you have completed Modules 01-18. You can check your progress:

```bash
tito module status
```

```bash
cd milestones/06_2018_mlperf

# Part 1: Optimize MLP/CNN (profiling + quantization + pruning)
python 01_optimization_olympics.py
# Expected: 4-8x compression with <2% accuracy loss

# Part 2: Speed up Transformer generation (KV caching)
python 02_generation_speedup.py
# Expected: 6-10x faster generation
```

## Expected Results

**Static Model Optimization (Script 01)**

| Optimization | Size | Accuracy | Notes |
|--------------|------|----------|-------|
| Baseline (FP32) | 100% | 85-90% | Full precision |
| + Quantization (INT8) | 25% | 84-89% | 4x smaller |
| + Pruning (50%) | 12.5% | 82-87% | 8x smaller total |

**Generation Speedup (Script 02)**

| Mode | Time/Token | Speedup |
|------|------------|---------|
| Without KV-Cache | ~10ms | 1x |
| With KV-Cache | ~1ms | 6-10x |

## The Aha Moment: Systematic Beats Heroic

**The Wrong Way (Heroic Optimization)**:
```
"It's too slow! Let me rewrite everything in C++!"
"Memory is too high! Let me redesign the architecture!"
"KV-cache sounds complex! Let me try CUDA kernels first!"
```
Result: Weeks of work, marginal gains, introduced bugs.

**The Right Way (Systematic Optimization)**:
```
1. MEASURE: Profile shows 70% of time is in attention, 80% of memory is Linear layers
2. OPTIMIZE: Add KV-cache (targets the 70%), quantize Linear layers (targets the 80%)
3. VALIDATE: Accuracy drops 1.5% (acceptable), 8× faster (huge win)
4. REPEAT: Profile again, find next bottleneck
```
Result: 10× faster, 8× smaller, 2% accuracy cost - achieved in days.

**This is what separates ML researchers from ML engineers:**
- YOUR Profiler (Module 14) identifies real bottlenecks (not assumed ones)
- YOUR Quantization (Module 15) reduces memory 4×
- YOUR Pruning (Module 16) reduces parameters 50%+
- YOUR KV-Cache (Module 18) speeds up generation 10×

The complete workflow, measure, optimize, validate, using YOUR tools.

## YOUR Code Powers This

This is the capstone of the entire TinyTorch journey. Every optimization tool comes from YOUR implementations:

| Component | Your Module | What It Does |
|-----------|-------------|--------------|
| `Profiler` | Module 14 | YOUR measurement and bottleneck identification |
| `quantize()` | Module 15 | YOUR INT8/FP16 conversion |
| `prune()` | Module 16 | YOUR weight pruning |
| `vectorize()` | Module 17 | YOUR accelerated operations |
| `KVCache` | Module 18 | YOUR key-value caching for generation |

**No external optimization libraries. Just YOUR code making models production-ready.**

## Historical Context

MLPerf (MLCommons) standardized ML benchmarking across hardware and software stacks. Before MLPerf, comparing ML systems was nearly impossible: different datasets, metrics, and conditions.

The "Optimization Era" marks when ML engineering became as important as ML research. Building a model is step one; deploying it efficiently is where production value lives.

## Systems Insights

- **Memory**: 4-16x compression achievable without significant accuracy loss
- **Latency**: 10-40x speedup with caching + batching
- **Trade-offs**: Every optimization has an accuracy/speed/memory trade-off

## What's Next

Milestone 06 completes the TinyTorch journey from tensors to production. You've now:
- Built every core component (Modules 01-13)
- Optimized for deployment (Modules 14-18)
- Proven mastery through historical recreations (Milestones 01-06)

The Capstone (Module 20) puts it all together in the Torch Olympics competition.

## Further Reading

- **MLPerf**: https://mlcommons.org/
- **Deep Compression**: Han et al. (2015). "Deep Compression: Compressing DNNs with Pruning, Trained Quantization and Huffman Coding"
- **Efficient Transformers**: Tay et al. (2020). "Efficient Transformers: A Survey"

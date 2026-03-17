# Milestone 06: MLPerf - The Optimization Era (2018)

## Historical Context

As ML models grew larger and deployment became critical, the community needed **systematic optimization methodologies**. MLCommons' MLPerf (2018) established standardized benchmarking and optimization workflows, shifting the focus from "can we build it?" to "can we deploy it efficiently?"

This milestone teaches **production optimization** - the systematic process of profiling, compressing, and accelerating models for real-world deployment.

## What You're Building

A complete MLPerf-style optimization pipeline that takes YOUR networks from previous milestones and makes them production-ready!

## Required Modules

<table>
<thead>
<tr>
<th width="25%"><b>Module</b></th>
<th width="25%">Component</th>
<th width="50%">What It Provides</th>
</tr>
</thead>
<tbody>
<tr><td><b>Module 01-03</b></td><td>Tensor, Linear, ReLU</td><td>YOUR base components</td></tr>
<tr><td><b>Module 11</b></td><td>Embeddings</td><td>YOUR token embeddings</td></tr>
<tr><td><b>Module 12</b></td><td>Attention</td><td>YOUR multi-head attention</td></tr>
<tr><td><b>Module 14</b></td><td>Profiling</td><td>YOUR profiler for measurement</td></tr>
<tr><td><b>Module 15</b></td><td>Quantization</td><td>YOUR INT8/FP16 implementations</td></tr>
<tr><td><b>Module 16</b></td><td>Compression</td><td>YOUR pruning techniques</td></tr>
<tr><td><b>Module 17</b></td><td>Acceleration</td><td>YOUR vectorized operations</td></tr>
</tbody>
</table>

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
<table>
<thead>
<tr>
<th width="25%"><b>Optimization</b></th>
<th width="15%">Size</th>
<th width="20%">Accuracy</th>
<th width="40%">Notes</th>
</tr>
</thead>
<tbody>
<tr><td><b>Baseline</b></td><td>100%</td><td>85-90%</td><td>Full precision</td></tr>
<tr><td><b>+ Quantization</b></td><td>25%</td><td>84-89%</td><td>INT8 weights</td></tr>
<tr><td><b>+ Pruning</b></td><td>12.5%</td><td>82-87%</td><td>50% weights removed</td></tr>
</tbody>
</table>

### Generation Speedup (02)
<table>
<thead>
<tr>
<th width="40%"><b>Mode</b></th>
<th width="30%">Time/Token</th>
<th width="30%">Speedup</th>
</tr>
</thead>
<tbody>
<tr><td><b>Without Cache</b></td><td>~10ms</td><td>1×</td></tr>
<tr><td><b>With KV Cache</b></td><td>~1ms</td><td>6-10×</td></tr>
</tbody>
</table>

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

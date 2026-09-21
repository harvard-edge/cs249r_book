# Milestone 06: Optimization and measurement

Compare a trained model with changed versions of itself, and measure a cache
without changing the computation. These are classroom experiments inspired by
MLPerf's measurement discipline, not official MLPerf submissions.

## Part 1: Optimization Olympics across Three Benchmark Divisions

`01_optimization_olympics.py` evaluates three distinct architectural categories across their natural physical constraints:

1. **Division 1: Edge & Embedded Inference — `DigitMLP` (Dense, Parameter-Bound)**
   - Primary constraint: Weight storage capacity in SRAM/Flash.
   - Evaluates: FP32 Baseline, INT8 Quantization (4× smaller), 50% Magnitude Pruning.
   - Trade-off curve: Memory Footprint (KB) vs. Classification Accuracy (%).

2. **Division 2: Spatial Vision & Compute — `SimpleCNN` (Spatial, Compute-Bound)**
   - Primary constraint: Spatial convolution loops and compute throughput.
   - Evaluates: FP32 Baseline, INT8 Quantization, 50% Magnitude Pruning.
   - Trade-off curve: Memory Footprint (KB) vs. Classification Accuracy (%).

3. **Division 3: Generative LLM Serving — `TinyGPT` (Autoregressive, Prefix-Bound)**
   - Primary constraint: $O(N^2)$ prefix recomputation and DRAM weight streaming.
   - Evaluates: Full Prefix Recompute, INT8 Quantization, KV-Cache Memoization, Full Stack (Quant + Cache).
   - Trade-off curve: Step Latency (ms) vs. Memory Footprint (KB).

Each division computes its own **Pareto frontier** using Module 19 (`pareto_frontier`) and plots a tailored ASCII trade-off curve, illustrating why optimization strategies must match the workload category.

Required modules: 01–08 and 14–19.

## Part 2: Cached inference

`02_generation_speedup.py` uses the GPT from Module 13 and the cache from Module
18. It replays one fixed token sequence with the same model weights: first by
recomputing complete causal prefixes, then by processing one token at a time.
Both paths run embedding, attention, feed-forward layers, and output projection.

The script checks logits at every position before comparing median timings from
repeated replays. Each cached replay resets its state and advances the cursor
once per token. Speedup depends on machine and sequence length; a small workload
may become slower. This untrained-model microbenchmark tests inference mechanics,
not language quality.

Required modules: 01–08, 11–13, and 18.

## Run

From the TinyTorch project root:

```bash
tito milestone run 06
tito milestone run 06 --part 1
tito milestone run 06 --part 2
```

Or run the scripts directly with the environment's Python:

```bash
python3 milestones/06_2018_mlperf/01_optimization_olympics.py
python3 milestones/06_2018_mlperf/02_generation_speedup.py
```

A successful run means the experiments completed with their correctness checks.
It does not certify a deployment target or guarantee a fixed compression/speed
ratio. Inspect the measured candidate accuracy and the preserved-computation
check before interpreting the performance numbers.

# Milestone 06: Optimization and measurement

Compare a trained model with changed versions of itself, and measure a cache
without changing the computation. These are classroom experiments inspired by
MLPerf's measurement discipline, not official MLPerf submissions.

## Part 1: Optimization Olympics & Architectural Triad

`01_optimization_olympics.py` evaluates the **Architectural Triad** built across the curriculum:

1. **DigitMLP** (Milestone 03) — Dense, parameter-bound
2. **SimpleCNN** (Milestone 04) — Spatial, compute-bound
3. **TinyGPT** (Milestone 05) — Autoregressive, memory-bandwidth & prefix-bound

The benchmark profiles each baseline and creates optimization candidates across the stack:
- **INT8 Quantization** (Module 15): 4× modeled storage compression
- **Magnitude Pruning** (Module 16): 50% weight sparsity
- **Vectorized Acceleration** (Module 17): SIMD-style matrix operations
- **KV-Cache Memoization** (Module 18): Recomputation-free autoregressive decoding

It computes the **Pareto frontier** using Module 19 (`pareto_frontier`), renders an ASCII **Pareto Trade-Off Curve**, and surfaces the **Asymmetric Architectural Bottlenecks**:
- **MLP**: Memory is dominated by dense weight matrices — quantization compresses storage 4× with negligible accuracy drop.
- **CNN**: Execution time is dominated by spatial convolution loops — vectorized matrix routines eliminate Python loop overhead.
- **Transformer**: Autoregressive decoding is bound by prefix recomputation and memory bandwidth — KV-cache memoization eliminates quadratic latency slowdown.

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

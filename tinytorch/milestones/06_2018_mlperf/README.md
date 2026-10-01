# Milestone 06: MLPerf to Generative Serving (2018)

Compare a trained model with changed versions of itself, and measure cache
acceleration without changing the mathematical output. These experiments bridge
MLPerf's measurement discipline (2018) to modern ChatGPT-scale production serving (2022).

## Part 1: Optimization Olympics across Three Benchmark Divisions

`01_optimization_olympics.py` evaluates three distinct architectural categories across their natural physical constraints:

1. **Division 1: Edge & Embedded Inference : `DigitMLP` (Dense, Parameter-Bound)**
   - Primary constraint: Weight storage capacity in SRAM/Flash.
   - Evaluates: FP32 Baseline, INT8 Quantization (4× smaller), 50% Magnitude Pruning.
   - Trade-off curve: Memory Footprint (KB) vs. Classification Accuracy (%).

2. **Division 2: Spatial Vision & Compute : `SimpleCNN` (Spatial, Compute-Bound)**
   - Primary constraint: Spatial convolution loops and compute throughput.
   - Evaluates: FP32 Baseline, INT8 Quantization, 50% Magnitude Pruning.
   - Trade-off curve: Memory Footprint (KB) vs. Output Signal Fidelity (%).

3. **Division 3: Generative LLM Serving : `TinyGPT` (Autoregressive, Prefix-Bound)**
   - Primary constraint: $O(N^2)$ prefix recomputation and DRAM weight streaming.
   - Evaluates: Full Prefix Recompute, INT8 Quantization, KV-Cache Memoization, Full Stack (Quant + Cache).
   - Trade-off curve: Replay Latency (ms) vs. Memory Footprint (KB).

Each division computes its own **Pareto frontier** using Module 19 (`pareto_frontier`) and plots a tailored ASCII trade-off curve, illustrating why optimization strategies must match the workload category.

Required modules: 01-04, 06, 07, 09, and 11-19.

## Part 2: Cached inference

`02_generation_speedup.py` uses the GPT from Module 13 and the cache from Module
18. It replays one fixed token sequence with the same model weights: first by
recomputing complete causal prefixes, then by processing one token at a time.
Both paths run embedding, attention, feed-forward layers, and output projection.

The script first checks that the model behaves like a language model (its logits vary, no position reads later tokens, and earlier tokens change the predictions), then checks cached against recomputed logits at every position before comparing median timings from
repeated replays. Each cached replay resets its state and advances the cursor
once per token. Speedup depends on machine and sequence length; a small workload
may become slower. This untrained-model microbenchmark tests inference mechanics,
not language quality.

**Try it:** once Part 2 passes in a terminal, a `Sentence >` prompt times your
cache on any sentence you type. Try a short word and then a long sentence: the
gap between recomputing and caching grows with length. Press Enter on an empty
line to finish. The prompt never affects the result and is skipped in CI, in
piped runs, and with `tito milestone run --non-interactive`.

Required modules: 01-04, 06, 11-14, and 18.

## Pre-Built Architectures (`networks.py`)

`networks.py` provides clean reference implementations of the architectures assembled across Milestones 01 to 05, allowing students to focus purely on systems optimization:
- `Perceptron`: Single-layer linear classifier (Milestone 01)
- `DigitMLP`: Multi-layer perceptron with ReLU (Milestone 03)
- `SimpleCNN`: LeNet-style spatial convolutional network (Milestone 04)
- `MinimalTransformer`: Toy sequence transformer with self-attention (Milestone 05)
- `TinyGPT`: Decoder-only generative transformer with Pre-LN blocks and causal masking (Milestone 05)

Import them directly:
```python
from networks import DigitMLP, SimpleCNN, MinimalTransformer, TinyGPT
```

## Run

From the TinyTorch project root:

```bash
tito milestone run 06            # both required parts, in order
tito milestone run 06 --part 1   # records Part 1 only
tito milestone run 06 --part 2   # records Part 2 only
```

Milestone 06 completes once both parts have passed.

Or run the scripts directly with the environment's Python:

```bash
python3 milestones/06_2018_mlperf/01_optimization_olympics.py
python3 milestones/06_2018_mlperf/02_generation_speedup.py
```

## Pass Gates

Part 1 stops with a teaching message at the first gate that fails:

- Loss: YOUR `CrossEntropyLoss` on the baseline's first training batch matches a NumPy computation.
- Profiler: parameter and FLOP counts match counts derived from the layer shapes (2,410 parameters; 4,736 FLOPs at 2·in·out per `Linear`), and the measured latency is positive and finite.
- Baseline `DigitMLP` test accuracy at least 80%.
- INT8: every parameter becomes integer codes in [-128, 127] that dequantize back to the weights, and INT8 accuracy stays within 3 points of the baseline.
- Pruning: zero fraction at 50% ± 2%.
- GPT behavior: before the GPT is cached, timed, or quantized, its logits vary across tokens and positions, changing later tokens never moves earlier predictions, a single token repeated along the sequence gives different predictions at different positions, and changing earlier tokens does.
- KV cache: returns what was stored, `reset` rewinds it, and cached logits match recomputed logits within 1e-4.
- Kernels: `vectorized_matmul` matches a loop reference and `im2col_conv2d` matches YOUR `Conv2d`, within 1e-3.
- Benchmarking: `BenchmarkResult` statistics and `pareto_frontier` match hand-worked fixtures before they score any candidate, and each measured latency summary is consistent (min ≤ mean, median ≤ max).

INT8 byte counts come from the stored code arrays (one byte per code plus 8 bytes of scale and zero point per tensor), not from the compression ratio the quantizer reports. Timing ratios print "slower" when a candidate is slower, and ⚡ marks only a speedup of 2× or more. Part 2 first checks the same GPT behavior (logits vary, no position reads later tokens, every position reads earlier ones), because an all-zero model would match its own cache trivially. It then passes when cached replay reproduces the recomputed logits at every position; a mismatch prints a teaching message and exits 1.

A successful run means the experiments completed with their correctness checks.
It does not certify a deployment target or guarantee a fixed compression/speed
ratio. Inspect the measured candidate accuracy and the preserved-computation
check before interpreting the performance numbers.

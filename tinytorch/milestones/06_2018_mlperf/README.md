# Milestone 06: Optimization and measurement

Compare a trained model with changed versions of itself, and measure a cache
without changing the computation. These are classroom experiments inspired by
MLPerf's measurement discipline, not official MLPerf submissions.

## Part 1: Model compression

`01_optimization_olympics.py` trains DigitMLP on TinyDigits, profiles the baseline,
and creates two independent candidates: rounded weights and pruned weights.
Each candidate runs on the same held-out data and receives its own accuracy and
latency measurements. The original model remains intact.

Both candidates still execute dense float32 operations. The report distinguishes
actual parameter-array bytes from modeled packed INT8 code size. Zeroing weights
does not shrink their arrays. Better accuracy, smaller resident storage, and a
speedup are outcomes to measure, not benefits the script assumes.

Separate cache-lifecycle and matrix-multiplication checks exercise Modules 18
and 17. They are not additional transformations of the MLP.

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

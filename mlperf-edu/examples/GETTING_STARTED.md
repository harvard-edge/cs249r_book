# MLPerf EDU Classroom Guide

MLPerf EDU has two deliberately separate teaching surfaces. Registered
workloads create benchmark reports and provenance artifacts. The three scripts
in this directory are standalone classroom experiments that demonstrate
optimization ideas but never claim canonical benchmark status.

## Setup

Follow the quickstart in the [project README](../README.md) first, then add the
dev extra and get oriented.

```bash
uv sync --locked --extra dev
uv run mlperf health
uv run mlperf list workloads
uv run mlperf show image-classification
```

`mlperf health` checks the environment, runs all fourteen min paths, verifies
their provenance manifests, and writes the suite health report without opening
it. A passing health report establishes functional readiness only. It does not
evaluate any max quality target.

The v0.1 registry contains fourteen workload identities.

- `image-classification`
- `keyword-spotting`
- `anomaly-detection`
- `visual-wake-words`
- `causal-language-modeling`
- `text-classification`
- `information-retrieval`
- `graph-node-classification`
- `time-series-forecasting`
- `code-generation`
- `function-calling`
- `recommendation`
- `image-generation`
- `reinforcement-learning`

Training and inference are modes. Full, prefill, and decode are phases of
causal-language-modeling inference. They are not separate workload IDs.

## First Registered Run

The `min` profile is the fast functional path for setup, instruction, and CI.
It exercises real model code and the artifact pipeline, but it is not a
canonical quality or performance baseline.

```bash
OUTPUT_DIR="submissions/first-run"
uv run mlperf run \
  --workload time-series-forecasting \
  --profile min \
  --output-dir "$OUTPUT_DIR"
uv run mlperf report "$OUTPUT_DIR/time-series-forecasting_min_report.json"
uv run mlperf verify "$OUTPUT_DIR/time-series-forecasting_min.provd.json"
```

The run emits JSON, CSV, HTML, and provenance outputs. Use `max` for the
authoritative quality contract when its declared environment is available. Use
`pro` for controlled single-node research experiments under the same workload
identity. Repetition is optional until the later stability phase.

## Causal Language Modeling

The canonical training path creates the quality-approved checkpoint required
by full, prefill, and decode inference. Keep every phase in the same output
directory so the harness can verify training lineage.

```bash
OUTPUT_DIR="submissions/causal-max"
uv run mlperf fetch \
  --workload causal-language-modeling \
  --mode training \
  --profile max
uv run mlperf run \
  --workload causal-language-modeling \
  --mode training \
  --profile max \
  --output-dir "$OUTPUT_DIR"
uv run mlperf run \
  --workload causal-language-modeling \
  --mode inference \
  --phase decode \
  --profile max \
  --output-dir "$OUTPUT_DIR"
```

Run `uv run mlperf info --workload causal-language-modeling` or `uv run mlperf
list matrix` to inspect valid mode and phase combinations.

## Standalone Lab Preflight

Each lab has a deterministic, CPU-only, network-free smoke path. The smoke
executes real PyTorch operations and fails on a broken functional contract.

```bash
uv run python examples/lab1_optimization.py --smoke
uv run python examples/lab2_inference_sut.py --smoke
uv run python examples/lab3_arch_comparison.py --smoke
```

Lab JSON records `canonical_result` as `false`. These files are classroom
measurements and cannot be submitted as MLPerf EDU reference evidence.

## Lab 1. Training-Loop Optimization

Lab 1 compares two ResNet-18 training-loop configurations on Fashion-MNIST.
It demonstrates batching, data loading, augmentation, optimizer, and schedule
effects. This model and dataset pair is not the registered MLPerf Tiny
image-classification contract.

```bash
uv run python examples/lab1_optimization.py \
  --preset baseline \
  --epochs 1 \
  --max-train-batches 100 \
  --max-validation-batches 50 \
  --output submissions/lab1-baseline.json
```

Instructors may set a course-specific `--target-accuracy`. The lab does not
invent a universal quality threshold for a bounded exercise.

## Lab 2. KV-Cache Inference

Lab 2 implements `mlperf.sut.SUT_Interface` and compares naïve autoregressive
decode with KV-cache decode. It passes only when the two paths emit identical
tokens for every measured query.

```bash
uv run python examples/lab2_inference_sut.py \
  --mode compare \
  --queries 3 \
  --prompt-length 32 \
  --generated-tokens 8 \
  --repeats 3 \
  --output submissions/lab2-kv-cache.json
```

The registered benchmark counterpart is
`causal-language-modeling --mode inference --phase decode`. The product CLI
does not load arbitrary SUT plugins.

## Lab 3. Dense and Sparse Architectures

Lab 3 compares dense and mixture-of-experts language-model training on the
same fixed Tiny Shakespeare batches. It reports total parameters, active
parameters per token, loss, and token throughput. It is an architecture lesson
rather than an admitted benchmark workload.

```bash
uv run python examples/lab3_arch_comparison.py \
  --epochs 1 \
  --max-batches 20 \
  --top-k 2 \
  --output submissions/lab3-dense-sparse.json
```

Use multiple seeds and a declared quality protocol before drawing a
model-quality conclusion from any bounded classroom comparison.

## Interpretation Rules

- Compare timing only after the applicable quality or functional gate passes.
- Keep workload identity separate from precision, compilation, batching, and
  other configurations.
- Treat `min` output as functional evidence only.
- Preserve the JSON report and provenance manifest together.
- Use the registry and generated site as the source of current workload names.

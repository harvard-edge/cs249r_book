# MLPerf EDU Lab Guide

The three labs in this directory are complete classroom experiments with a
shared execution contract. Each accepts `--smoke`, runs real model code on CPU
without network access, performs a functional validity check, and returns a
nonzero exit status on failure.

Lab output is labeled as a classroom measurement. It is not a canonical MLPerf
EDU submission and should not be used as public baseline evidence. The
registered `mlperf run` workflows produce the benchmark reports and provenance
artifacts used for review.

## Setup

Clone the MLSysBook repository and install MLPerf EDU from its project
directory.

```bash
git clone https://github.com/harvard-edge/cs249r_book.git
cd cs249r_book/mlperf-edu
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Check the installation and inspect the registry before running a workload.

```bash
mlperf doctor
mlperf list workloads
mlperf show resnet18-train
```

## Canonical CLI Workflow

The `min` profile is a quick representative execution path. It is useful for
checking installation and artifact plumbing, but a passing `min` run does not
automatically establish the quality of a `max` baseline.

```bash
mlperf run \
  --workload micro-lstm-train \
  --profile min \
  --output-dir submissions/first-run
```

Inspect the generated report and verify the paired provenance manifest.

```bash
mlperf report submissions/first-run/micro-lstm-train_min_report.json
mlperf verify submissions/first-run/micro-lstm-train_min.provd.json
```

Discover workload and variant identifiers from the registry instead of copying
a static workload list from this guide.

```bash
mlperf list workloads
mlperf list variants --workload nanogpt-inference
mlperf info --workload nanogpt-inference --variant decode
```

## Fast Preflight

Run all three lab entry points before a class or documentation release.

```bash
python examples/lab1_optimization.py --smoke
python examples/lab2_inference_sut.py --smoke
python examples/lab3_arch_comparison.py --smoke
```

The smoke paths use deterministic synthetic inputs. Their accuracy and loss
values are functional checks, not quality targets.

## Lab 1. Training-Loop Optimization

Lab 1 trains the repository's complete CIFAR-style ResNet-18 architecture on
the registered Fashion-MNIST dataset. Images are resized to 32 by 32 and
expanded to three channels, matching the canonical workload preprocessing. The
batch size, worker count, augmentation, optimizer, and learning-rate schedule
settings are wired to the executed code. A full run may download Fashion-MNIST
on the first invocation.

Start with the baseline preset and save the classroom result.

```bash
python examples/lab1_optimization.py \
  --preset baseline \
  --epochs 1 \
  --max-train-batches 100 \
  --max-validation-batches 50 \
  --output submissions/lab1-baseline.json
```

Run the optimized preset with the same seed and limits.

```bash
python examples/lab1_optimization.py \
  --preset optimized \
  --epochs 1 \
  --max-train-batches 100 \
  --max-validation-batches 50 \
  --output submissions/lab1-optimized.json
```

Compare throughput and validation accuracy together. A faster run is not an
improvement if the model's quality falls outside the assignment's allowed
range. Instructors may set a course-specific check with `--target-accuracy`,
but the script deliberately does not invent a universal target for every
laptop and run budget.

The canonical registered workload is separate.

```bash
mlperf run --workload resnet18-train --profile min
```

## Lab 2. KV-Cache Inference

Lab 2 implements `mlperf.sut.SUT_Interface` and measures two real autoregressive
decode paths. The baseline recomputes the full sequence at every step. The
optimized path reuses attention keys and values. The comparison passes only if
both paths produce identical tokens for every measured query.

```bash
python examples/lab2_inference_sut.py \
  --mode compare \
  --queries 3 \
  --prompt-length 32 \
  --generated-tokens 8 \
  --repeats 3 \
  --output submissions/lab2-kv-cache.json
```

The default lab-scale model uses deterministic random initialization and is
appropriate for latency and token-parity instruction. Pass a compatible
checkpoint with `--checkpoint` when the assignment also needs trained-model
behavior. Neither mode reports placeholder accuracy.

The current product CLI runs registered SUTs. It does not accept an arbitrary
`--sut` plugin path. Use the built-in canonical decode workload as follows.

```bash
mlperf run \
  --workload nanogpt-inference \
  --variant decode \
  --profile min
```

## Lab 3. Dense and Sparse Architectures

Lab 3 trains NanoGPT and Nano-MoE on the same fixed TinyShakespeare batches.
The `--top-k` value directly changes the sparse router. The result reports total
parameters, active parameters per token, loss, and measured token throughput.

```bash
python examples/lab3_arch_comparison.py \
  --epochs 1 \
  --max-batches 20 \
  --top-k 2 \
  --output submissions/lab3-dense-sparse.json
```

A bounded classroom run does not prove that one architecture converges better
than the other. Use multiple seeds and a declared quality protocol before
drawing a model-quality conclusion. The registered workload commands are:

```bash
mlperf run --workload nanogpt-train --profile min
mlperf run --workload nano-moe-train --profile min
```

## Result Interpretation

Every lab JSON includes its scope, seed, device, effective configuration,
measured metrics, and functional check. The field `canonical_result` is always
`false`. Canonical workload reports have a different schema and are paired with
`.provd.json` provenance manifests.

Useful CLI discovery commands include `mlperf --help`, `mlperf list workloads`,
`mlperf list variants`, `mlperf show WORKLOAD`, and `mlperf info --workload
WORKLOAD`.

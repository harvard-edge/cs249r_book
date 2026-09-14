# MLPerf EDU Tutorials

This directory contains runnable teaching material for ML systems benchmarking.
Tutorial 01 is implemented and tested. Later sessions remain a roadmap and are
not included in the current release.

## Setup

Start from the quickstart in the [project README](../README.md), then add the
tutorial dependency and launch the marimo notebook from the `mlperf-edu`
directory.

```bash
uv sync --extra tutorial
uv run marimo edit tutorials/01_first_benchmark.py
```

The notebook invokes the public `mlperf` command surface through a subprocess
and reads its report artifacts. It does not import runner internals.

## Implemented Material

| **Session** | **Status** | **Duration** | **Entry Point** |
|:---|:---|:---|:---|
| Anatomy of a benchmark run | Implemented and smoke-tested | 30–45 minutes | `01_first_benchmark.py` |

Tutorial 01 runs the `time-series-forecasting` `min` profile. That profile is a
deterministic functional run over a synthetic smoke fixture. It is not a
quality baseline. Students inspect the metrics, run fingerprint, report views,
and provenance manifest, then verify the manifest with the CLI.

The complete noninteractive path is suitable for local preflight and CI.

```bash
python tutorials/smoke_first_benchmark.py
```

The command exits successfully only when the workload passes, the JSON, HTML,
CSV, and provenance files exist, the report includes metrics and a run
fingerprint, and `mlperf verify` accepts the fresh manifest.

## Roadmap

The following sessions describe intended future material. No notebook files or
runnable exercises are claimed for them in this release.

| **Proposed Session** | **Intended Lesson** | **Status** |
|:---|:---|:---|
| Systems regimes | Compare compute, memory, and dispatch behavior using measured telemetry | Roadmap |
| Optimize a SUT | Measure baseline and KV-cache decode while preserving token parity | Roadmap; the complete command-line exercise is available as `examples/lab2_inference_sut.py` |
| Research variants | Run controlled variant sweeps and cite result artifacts | Roadmap |

These sessions should not be advertised as a half-day conference program until
their notebooks, instructor notes, timing checks, and smoke tests are committed.

## Offline Behavior

Tutorial 01 uses a deterministic synthetic smoke fixture and needs no dataset
download. After Python dependencies are installed, its benchmark and
verification steps run without network access. A distributable wheel can be
built ahead of a class with `uv build`, but a complete offline teaching bundle
is not yet part of this repository.

## Teaching Rules

1. **Laptop budget.** The implemented smoke must complete quickly on a laptop
   CPU and must not require an accelerator.
2. **Reports are the interface.** Exercises read generated report artifacts
   instead of scraping console text for benchmark data.
3. **Measurement scope stays explicit.** Functional smoke results must not be
   described as quality baselines or public benchmark scores.
4. **Provenance is checked.** An exercise that creates a manifest must verify it
   before reporting success.
5. **Roadmap material stays labeled.** A session becomes implemented only after
   its entry point and smoke test are present.

## Course Use

Tutorial 01 can be used as a standalone introductory module. The three complete
command-line labs under `examples/` provide longer exercises for training-loop
optimization, KV-cache inference, and dense-versus-sparse comparison. Their
outputs are classroom measurements, not canonical MLPerf EDU submissions.

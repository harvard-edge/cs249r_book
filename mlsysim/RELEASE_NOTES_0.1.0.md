# MLSys·im 0.1.0 — Initial Release

**Release date:** 2026-04-01

MLSys·im is a first-principles analytical engine for predicting performance, cost, and carbon footprint of ML systems. It is the computational companion to the [Machine Learning Systems](https://mlsysbook.ai) textbook.

This is the first public release. The API surface, the 22-wall taxonomy, and the solver portfolio are all considered stable for the 0.1.x line.

## Install

```bash
pip install mlsysim
```

Verify the install:

```bash
python -c "import mlsysim; print(mlsysim.__version__)"
mlsysim eval Llama3_8B H100 --batch-size 32
```

Requires Python 3.10+. No GPU required — the engine computes from closed-form equations.

## Five-line quickstart

```python
import mlsysim
from mlsysim import Engine

profile = Engine.solve(
    model      = mlsysim.Models.ResNet50,
    hardware   = mlsysim.Hardware.Cloud.A100,
    batch_size = 1,
    precision  = "fp16",
)

print(f"Bottleneck: {profile.bottleneck}")              # → Memory
print(f"Latency:    {profile.latency.to('ms'):~.2f}")   # → 0.54 ms
print(f"Throughput: {profile.throughput:.0f}")          # → 1843 / second
```

## Highlights

### Core framework

- **22-wall taxonomy** organizing every constraint that bounds ML system performance, across six domains (Node, Data, Algorithm, Fleet, Ops, Analysis).
- **26 analytical solvers** (`Model`, `Solver`, `Optimizer` classes) covering all 22 walls.
- **Pint unit system** with dimensional analysis enforced at runtime.
- **TraceableConstant** pattern — every default carries a citation.
- **Pipeline composer** for chaining solvers with `explain()` and `run()`.
- **3-tier evaluation scorecard:** Feasibility → Performance → Macro/Economics.
- **Design Space Exploration (DSE) engine** with declarative search and constraint evaluation.

### Hardware Registry (15+ accelerators)

V100, A100, H100, H200, B200, GB200 NVL72, MI300X, TPUv5p, T4, Cerebras CS-3, Jetson Orin NX, ESP32-S3, nRF52840, Himax WE-I Plus, DGX Spark, MacBook M3 Max, iPhone 15 Pro, Pixel 8.

Full precision support: FP32, TF32, BF16, FP16, FP8, INT8, INT4. Multi-level memory hierarchy with HBM, SRAM, and Flash (TinyML). All specifications verified against manufacturer datasheets.

### Model Registry

GPT-2/3/4, Llama-2/3 (7B/8B/70B), BERT Base/Large, ResNet-50, MobileNetV2, AlexNet, Mamba, Stable Diffusion v1.5, DS-CNN, WakeVision. HuggingFace importer included for arbitrary Transformer workloads.

### Analytical models

`SingleNodeModel`, `NetworkRooflineModel`, `EfficiencyModel`, `ForwardModel`, `ServingModel`, `ContinuousBatchingModel`, `WeightStreamingModel`, `TailLatencyModel`, `DataModel`, `TransformationModel`, `TopologyModel`, `ScalingModel`, `InferenceScalingModel`, `CompressionModel`, `DistributedModel`, `ReliabilityModel`, `OrchestrationModel`, `EconomicsModel`, `SustainabilityModel`, `CheckpointModel`, `ResponsibleEngineeringModel`, `SensitivitySolver`, `SynthesisSolver`, `ParallelismOptimizer`, `BatchingOptimizer`, `PlacementOptimizer`.

### CLI

```text
mlsysim eval        Evaluate the analytical physics of an ML system (YAML or CLI flags)
mlsysim serve       Evaluate LLM serving (prefill + decode)
mlsysim optimize    Search the design space for optimal configurations
mlsysim zoo         Explore the built-in registries
mlsysim audit       Profile your local hardware against the Iron Law
mlsysim schema      Export the JSON Schema for the mlsys.yaml configuration file
```

### Testing

367 tests, 100% pass rate. Coverage includes formula unit tests with known answers, full solver suite, physics-bound validation across all registry hardware, wall-taxonomy completeness, pipeline composition, and three optimization backends (exhaustive, OR-tools, scipy).

## Documentation

- Site: [mlsysbook.ai/mlsysim/](https://mlsysbook.ai/mlsysim/)
- Tutorials: roofline, memory wall, KV cache, scaling to 1000 GPUs, geography, sensitivity, full-stack audit
- API reference: [mlsysbook.ai/mlsysim/api/](https://mlsysbook.ai/mlsysim/api/)

## Known limitations & gotchas

- **First-order analytical model.** Predictions are typically within 15–30% of measured throughput on well-optimized workloads. Use MLSys·im to *compare* options and identify *bottlenecks*; validate with empirical benchmarks before committing to a production SLA. See `accuracy.qmd` for full validation against MLPerf v4.0.
- **Slide PDFs.** Many tutorials cross-link to lecture slides at `github.com/harvard-edge/cs249r_book/releases/download/slides-latest/*.pdf`. The `slides-latest` release tag is not yet published; these links will resolve once the slides ship.
- **Hosted notebook launchers.** Google Colab and Binder buttons are planned but not wired up for 0.1.0. Tutorials run locally on any Python 3.10+ environment.

## Project links

- Source: [github.com/harvard-edge/cs249r_book/tree/dev/mlsysim](https://github.com/harvard-edge/cs249r_book/tree/dev/mlsysim)
- Issues: [github.com/harvard-edge/cs249r_book/issues](https://github.com/harvard-edge/cs249r_book/issues)
- License: CC BY-NC-SA 4.0
- Citation: see [`CITATION.cff`](CITATION.cff)

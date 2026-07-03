# MLPerf EDU North Star

## Two-Year Goal

Two years from now, an ML systems, architecture, compiler, edge AI, or efficient
LLM paper should be expected to answer:

> Did you run MLPerf EDU, and if not, why not?

MLPerf EDU should become the practical academic benchmark substrate that MLPerf
never became for most papers: SPEC-like in usability and citation value, but
MLPerf-like in discipline around workloads, scenarios, quality targets,
measurement, provenance, and reviewable artifacts.

The suite should be easy enough to run in a course and serious enough to anchor
research claims in ISCA, MICRO, HPCA, ASPLOS, MLSys, NeurIPS Systems, and related
venues.

## Core Thesis

MLPerf EDU is not just "MLPerf but smaller" and not just a classroom demo.

It is a runnable academic research suite for ML systems. Classroom usability is
the forcing function that keeps the benchmark from becoming too complicated for
students, reviewers, artifact evaluators, and researchers to use.

The project exists because official MLPerf is too hard for most academic groups
to run, modify, cite, and compare against. MLPerf EDU should preserve the useful
parts of MLPerf methodology while removing enough operational friction that it
can become common in academic papers.

## Profile Semantics

Profiles express how much of the benchmark and research surface is exercised.
They are not marketing tiers.

| Profile | North-star meaning | Primary use |
|---|---|---|
| `min` | Minimum representative run, likely one small path from each major suite | install check, smoke test, classroom demo, CI |
| `max` | Full MLPerf EDU benchmark suite at the standard comparable scale | course assignments, artifact evaluation, paper baselines |
| `pro` | Research envelope exposing controlled variants and optimization knobs | architecture, systems, compiler, backend, pruning, quantization, and serving studies |

The implementation may use internal validation shortcuts, but the public mental
model should stay this simple: `min` proves it runs, `max` runs the suite, and
`pro` opens the research space.

## What `pro` Means

`pro` is not merely a longer version of `max`. It should expose the dimensions
that researchers actually study:

- Precision and quantization: fp32, fp16, bf16, int8, int4, weight-only, KV-cache quantization.
- Sparsity and pruning: unstructured, structured, 2:4, channel pruning, block sparsity.
- SLM serving: prefill, decode, long context, batching, KV cache, speculative decode.
- Fine-tuning: LoRA, QLoRA-style paths where feasible, adapter rank sweeps.
- Backend comparison: PyTorch, ONNX Runtime, MLX, llama.cpp, TVM/IREE where feasible.
- Memory behavior: embedding tables, KV cache size, sequence length, batch size, activation memory.
- Distributed/local parallelism: DDP, tensor/model sharding stand-ins, communication/computation tradeoffs.
- Edge and TinyML behavior: compression, on-device memory, small-batch latency, sensor-style inputs.
- Agentic workloads: RAG, tool calls, ReAct loops, code generation, retrieval/generation balance.
- Power and energy: aggregate estimates first, hardware counters where available.

Every `pro` variant should be controlled enough that another group can reproduce
the comparison and understand what changed.

## Full Suite Meaning

A full MLPerf EDU suite is not a clone of every official MLPerf workload. It is
coverage of the major educational and research regimes that ML systems papers
need:

- Language model training and serving.
- Small language model inference and optimization.
- Vision training, inference, compression, and mobile models.
- Recommender and sparse-memory behavior.
- TinyML and edge-style models.
- Agent and retrieval/tool-use systems.
- Distributed/local communication behavior.
- Graph, time-series, and reinforcement-learning control-flow workloads.

The current registry is a strong starting point, but the SLM suite is too thin
for the north-star research goal. Its top-level workloads should use names that
researchers recognize, such as SmolLM2, Qwen, and LLaMA-family inference or
fine-tuning benchmarks. Internal serving phases such as prefill, decode,
batched decode, long context, KV-cache behavior, quantized serving, LoRA,
RAG/tool-use integration, and speculative decode should be exposed as measured
phases, variants, or `pro` knobs inside those recognizable workloads.

## Usability Principle

"Runs out of the box" does not mean the benchmark is trivial. It means:

- A new user can install it and get a valid first result.
- A reviewer can reproduce a paper baseline without becoming a benchmark expert.
- A student can run it without a cluster.
- A researcher can scale from a smoke run to a full-suite run to a controlled
  research sweep.
- Reports are usable in a browser, spreadsheet, and artifact package.

If a benchmark cannot be run by a typical academic group, it will not become the
default benchmark for academic papers.

## Public Vocabulary

Keep the user-facing vocabulary small:

- `suite`: workload domain, such as `slm`, `vision`, `language`, or `tiny`.
- `profile`: run scale and research surface: `min`, `max`, `pro`.
- `workload`: one benchmark ID.

Avoid extra public concepts unless they clearly earn their complexity.

## Planning Implication

Implementation should proceed in this order:

1. Make the current registry run cleanly from a fresh clone.
2. Align default `min`, `max`, and `pro` behavior with the profile semantics above.
3. Expand SLM into a serious research suite.
4. Harden quality targets, data policy, reports, provenance, and grading.
5. Add `pro` optimization variants one controlled dimension at a time.
6. Produce MLCommons review materials and academic artifact-evaluation examples.
7. Run pilot papers or course projects that use MLPerf EDU as the baseline.

The work is done when MLPerf EDU is not merely runnable, but expected.

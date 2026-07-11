# MLPerf EDU Design Philosophy

MLPerf EDU explores whether disciplined benchmark methodology can fit inside a
course laptop and remain useful for research review. The project draws
inspiration from SPEC-style reproducibility and MLPerf-style workload, quality,
scenario, and artifact discipline. It is an independent preview, not an
official MLCommons benchmark.

## Inspectable Where It Teaches

Many core teaching models are compact PyTorch implementations kept in this
repository. Students can inspect the model, data loop, checkpoint path, and
measurement code. The SLM suite deliberately uses pinned off-the-shelf Hugging
Face models because model download, serving, quantization, batching, and task
quality are the systems lessons there.

White-box does not mean that every dependency is pure Python or that every
runner is fewer than a fixed number of lines. PyTorch, torchvision,
Transformers, and platform kernels remain real dependencies. Reports disclose
the backend and software environment instead of pretending those layers do not
exist.

## Quality Before Speed

A fast broken model is not a baseline. Score-bearing candidates must pass a
real-data task metric. Performance-bearing candidates must complete meaningful
work and preserve checkpoint or model quality. Synthetic and micro-sharded
paths remain useful for setup and systems instruction, but they are labeled
systems-only.

The current quality boundary is concrete. Five training candidates use
five-seed target protocols. NanoGPT inference inherits quality from a hashed
training checkpoint. The SmolLM2 baseline uses a pinned revision and a bundled
continuation-perplexity fixture. The dynamic-int8 SLM path stays systems-only
because its current calibration fails quality parity.

## Reports Are the Interface

Console output is transient. Every canonical run writes structured JSON, a
human-readable HTML view, a CSV view, and a provenance manifest. The report
contains the workload identity, profile, data mode, seed, metrics, target
status, hardware and software fingerprints, asset dossiers, and artifact
paths.

Students, instructors, and artifact reviewers should reason from these files.
Tutorial 01 follows this rule by invoking the public command, reading the JSON,
and verifying the paired manifest.

## Provenance Without Overclaiming

The `.provd.json` manifest binds available source, dataset, weights, seed,
hardware, optional sidecar, and exact report evidence with SHA-256. Its
integrity digest is unauthenticated. It detects changes but does not prove who
produced the artifact.

Portable packages use relative paths, index every included file by digest and
byte size, and verify again after clean extraction. These checks make review
easier. They do not replace independent execution, measurement governance, or
rights review.

## Training and Inference Stay Connected

Checkpoint-backed NanoGPT prefill and decode require the training checkpoint in
the selected output path or through an explicit environment variable. Reports
record the checkpoint digest and quality dependency. This creates an auditable
training-to-serving chain rather than timing random weights.

External-model serving uses a different provenance shape. The registry pins a
model revision, and the report records the model dossier, fixture digest,
quality result, and timing protocol.

## Measurement Must Be Repeatable

Candidate inference rows separate warmup from measured work, synchronize the
active device, retain a defined sample count, and report median, p90, and p99
latencies. Score-bearing targets run through five fresh processes with explicit
seeds and create-once evidence packets.

Optional power data is coarse platform telemetry. Optional roofline evidence
must come from an existing, digest-checked sidecar before it can support a
claim. Missing measurements are labeled `unmeasured` rather than inferred from
architecture names.

## The Harness Is Fixed for Canonical Runs

Canonical results use registered runners. The repository contains a
`SUT_Interface` and Lab 2 uses it to compare naive and KV-cache decode with
token parity. The product CLI does not currently accept an arbitrary `--sut`
plugin file. A general plugin-loading protocol is roadmap work and must not be
described as shipped.

Likewise, the implemented asset command is `mlperf fetch`. There is no
`mlperf hydrate` command. Fetching prepares datasets or pinned model assets and
keeps network work outside the measured run where supported.

## Laptop-Scale Is an Operational Constraint

The core smoke paths and all lab smokes run on CPU without a network after
dependencies are installed. Candidate `max` runs may download datasets or the
pinned 135M-parameter SLM and can take materially longer. The release process
therefore records measured runtimes instead of promising a universal time
budget.

Laptop-scale means no cluster is required for the standard path. It does not
mean zero installation, zero downloads, identical runtime on every notebook,
or calibrated energy data on unsupported hardware.

## Governance Is Part of Correctness

The repository separates implementation, validation evidence, and external
approval. A green local run cannot close a dataset-rights question. A complete
artifact cannot grant MLCommons endorsement. A registry target cannot become a
canonical baseline without retained multi-seed evidence and review.

The design succeeds when a student can understand the measured system and a
reviewer can reproduce, challenge, and reject a result using the same visible
contract.

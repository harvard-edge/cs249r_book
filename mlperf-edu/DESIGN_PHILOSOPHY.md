# MLPerf EDU Design Philosophy

MLPerf EDU asks whether mature benchmark discipline can fit inside a machine
learning systems course and remain useful for single-node research. It borrows
the reproducibility and comparability posture of SPEC and MLPerf while keeping
the execution boundary practical on a laptop. It is an independent preview,
not an official MLCommons benchmark.

## Curate Rather Than Invent

The suite begins with an authoritative upstream workload. The upstream source
must supply the task, model or reference implementation, dataset and split,
evaluator, quality contract, and credible baseline. MLPerf EDU adds only the
PyTorch execution adapter, laptop measurement protocol, quality gate,
provenance, and report surface needed to run that contract locally.

A missing upstream component is not an invitation to create a convenient
substitute. The task is deferred or rejected. This policy is why v0.1 contains
nine workloads rather than a large coverage matrix, and why MiniGo is the RL
reference without becoming a v0.1 workload.

## Design Backward From Classroom Use

A student should be able to install the project, inspect a workload, fetch its
pinned assets, execute a functional `min` run, perform a canonical `max` run,
and explain the resulting quality, timing, configuration, and provenance. An
instructor should be able to preflight the same path and grade the resulting
artifacts. A researcher should be able to repeat a controlled single-node
experiment without changing workload identity.

The nine-workload portfolio deliberately spans dense and depthwise vision
convolution, compact audio convolution, autoencoder anomaly scoring,
autoregressive Transformer training and inference, encoder classification,
cross-encoder reranking, sparse graph message passing, and long-horizon
forecasting. Each workload earns its place through distinct learning value and
distinct systems behavior.

## Keep Identity Stable

A workload ID names the learning task. Training and inference are modes. Full,
prefill, and decode are inference phases. Precision, quantization, compilation,
batching, context length, scheduling, and other optimization choices are
configurations recorded in reports. They do not create new workload IDs.

The three profiles express execution intent without changing the workload.
`min` is the fast functional path, `max` is the canonical classroom comparison,
and `pro` is the extended single-node research envelope.

## Gate Performance With Quality

A fast invalid model is not a benchmark result. Every score-bearing case must
pass its inherited task-quality contract before its timing is interpreted.
Every performance-bearing phase must pass a functional contract and inherit
the required model lineage. No median or aggregate may hide a failed
individual run.

Canonical reference evidence uses five fresh processes at the canonical seed.
Every run must pass, and the primary timing coefficient of variation must not
exceed 5%. The evidence index records the complete case identity, source
revision, raw values, aggregate, decision, and content digest.

## Treat Reports as the Interface

Console output is transient. A registered run writes structured JSON, a flat
CSV view, a human-readable HTML report, and a provenance manifest. The report
keeps workload, mode, phase, profile, model, data, quality, requested and
executed devices, backend, timing, and configuration together.

The provenance manifest binds the report and retained inputs with SHA-256. It
detects changes but does not authenticate the producer. Portable packages use
relative paths and verify every included byte again after clean extraction.
Independent reproduction remains necessary.

## Preserve Training Lineage

The causal-language-modeling workload keeps training, full inference, prefill,
and decode under one identity. Canonical inference requires a checkpoint from a
passing canonical training run. Reports record the checkpoint, source report,
source manifest, and package digests so a serving result cannot silently use
random or unrelated weights.

Other inference workloads use pinned authoritative checkpoints. Their reports
record the exact revision, model files, dataset files, and evaluator contract.

## Measure the Declared Boundary

Asset fetching, model construction, and untimed warmup stay outside the
canonical measured region unless an upstream contract explicitly includes
them. Accelerator measurements synchronize at each boundary. Reference runs
record the power source and power mode, and an intervening sleep or power-state
change invalidates an attempt.

Optional power data is coarse platform telemetry. Optional roofline claims
need a measured and digest-checked sidecar. Missing information remains
`unmeasured` instead of being inferred from an architecture name.

## Keep the Scale Honest

The standard path targets CPU and laptop accelerators. The suite supports
controlled studies of processors, memory systems, runtimes, compilers, and
model execution. It does not claim to represent distributed training,
datacenter serving, cluster scheduling, or fleet economics.

Local execution does not mean zero downloads or identical runtimes on every
notebook. Canonical workloads may fetch substantial assets and take tens of
minutes. The project publishes observed hardware and runtime evidence instead
of promising a universal duration.

## Separate Technical Readiness From Governance

A green run cannot settle dataset rights, grant MLCommons endorsement, or
authenticate a result producer. Technical release checks and external review
decisions therefore remain separate. The project is ready for design and
implementation review when all nine workloads and twelve evidence cases
execute, every gate result is verified, provisional measurements are
distinguished from five-run evidence, and every open external decision is
stated plainly. Promotion remains a separate five-run requirement for every
case.

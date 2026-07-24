# MLPerf EDU Design Philosophy

MLPerf EDU asks whether mature benchmark discipline can fit inside a machine
learning systems course and remain useful for single-node research. It borrows
the reproducibility and comparability posture of SPEC and MLPerf while keeping
the classroom entry path practical on a laptop and the extended boundary within
one research node. It is an independent preview, not an official MLCommons
benchmark.

## Curate Rather Than Invent

The suite begins with an authoritative upstream workload. The upstream source
must supply the task, model or reference implementation, dataset and split,
evaluator, quality contract, and credible baseline. MLPerf EDU adds only the
execution adapter, declared single-node measurement protocol, quality gate,
provenance, and report surface needed to run that contract.

A missing upstream component is not an invitation to create a convenient
substitute. A bounded functional probe may establish integration plumbing, but
it remains experimental and outside promotion until the authoritative quality
contract executes unchanged in its declared environment. This policy is why
MiniGo remains the reinforcement-learning identity instead of being replaced
by a control task.

## Design Backward From Classroom Use

A student should be able to install the project, inspect a workload, fetch its
pinned assets, execute a functional `min` run, and perform a canonical `max`
run when the declared environment is available. The student should be able to
explain the resulting quality, timing, configuration, and provenance. An
instructor should be able to preflight the selected path and grade its
artifacts. A researcher should be able to repeat a controlled single-node
experiment without changing workload identity.

The fourteen-workload portfolio spans dense and depthwise vision convolution,
compact audio convolution, autoencoder anomaly scoring, autoregressive
Transformer training and inference, encoder classification, cross-encoder
reranking, structured decoding, sparse embeddings, iterative denoising, sparse
graph message passing, long-horizon forecasting, and self-play learning. Each
workload earns its place through distinct learning value and systems behavior.

## Build in Evidence Spirals

Functional integration comes first. It proves that selection, execution,
reporting, and provenance work while explicitly withholding quality and timing
claims. Quality conformance then replaces the bounded probe with the pinned
model, complete dataset, authoritative evaluator, and published target.
Stabilization follows only after correctness is settled. Promotion is the last
step and requires one complete source-locked evidence set.

## Keep Identity Stable

A workload ID names the learning task. Training and inference are modes. Full,
prefill, and decode are inference phases. Precision, quantization, compilation,
batching, context length, scheduling, and other optimization choices are
configurations recorded in reports. They do not create new workload IDs.

The three profiles express execution intent without changing the workload.
`min` is the fast functional path, `max` is the authoritative quality path for
the workload's declared environment, and `pro` is the extended single-node
research envelope.

## Separate Profile Intent From Hardware Envelope

A profile defines the depth of the benchmark contract, not a universal machine
size. Every `min` path must run on classroom hardware and must preserve enough of
the workload identity to verify setup, execution, reporting, and provenance. A
`max` path runs the unchanged authoritative quality contract. Most max paths fit
the laptop envelope. DLRM and MiniGo still require their declared single-node
research environments, but the first milestone now requires local backends that
preserve those same contracts. The `pro` profile adds research controls without
silently changing the task, data, evaluator, or quality target.

This separation keeps all fourteen workloads teachable without pretending that
a bounded classroom probe is a benchmark-quality result. It also keeps current
resource requirements visible while the local DLRM and MiniGo executors are
developed.

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
decisions therefore remain separate. The project is ready for functional
design review when all fourteen workloads execute through the public CLI and
their current evidence stage is explicit. The existing twelve evidence cases
across nine workloads remain the separate quality and repeatability review
surface. Promotion remains a five-run requirement for every eventual case.

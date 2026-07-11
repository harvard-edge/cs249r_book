# MLPerf EDU North Star

## Two-Year Ambition

MLPerf EDU should become a routine academic baseline for ML systems,
architecture, compiler, edge AI, and efficient-model work. A paper should be
able to use it without a cluster, and a reviewer should be able to reproduce
the baseline without becoming a benchmark maintainer.

That is an ambition, not the current release state. Today the project is an
independent review preview with 30 executable registry rows, five
score-bearing candidates, three performance-bearing candidates, and 22
systems-only rows. Eight exact reference summaries are committed and their raw
packets are retained for local handoff. Thirty-five policy-permitted run
packages are verified; the five DLRM packages remain correctly blocked by the
restricted-dataset policy. Local same-revision validation is complete. Hosted
CI, public packet URLs, asset decisions, independent reproduction, and
MLCommons review remain gates.

## Core Thesis

Classroom usability is a forcing function for benchmark quality. A workload
small enough for students to inspect can also be easier for artifact reviewers
to reproduce, vary, and challenge. The project should preserve the useful
discipline of MLPerf methodology while reducing the operational burden that
keeps many academic groups from running full production suites.

MLPerf EDU is not intended to replace MLPerf Training, Inference, Tiny, Client,
or any official submission process. Its first job is to teach and support
reviewable local experiments. Any stronger relationship requires MLCommons
approval.

## Credibility Conditions

The suite becomes canonical only if it earns trust in four dimensions.

| **Dimension** | **Required Outcome** |
|:---|:---|
| Usability | A new user installs from a documented source or release artifact, obtains a valid first result, and understands which downloads and runtimes to expect. |
| Scientific validity | Public candidates use meaningful quality gates, repeated measurements, disclosed variance, and rules that reject empty or degraded work. |
| Reproducibility | Reports, fingerprints, manifests, checkpoints, fixtures, and portable packages let another group repeat the result. |
| Governance | Dataset terms, model revisions, component license, naming, result wording, and promotion decisions are explicit and reviewable. |

Success in only one dimension is insufficient. A convenient demo without
quality evidence is not a benchmark. A rigorous harness that typical academic
groups cannot run will not become a shared baseline.

## Profile Semantics

Profiles express execution scale and research surface. They are not marketing
tiers.

| **Profile** | **North-Star Meaning** | **Current Boundary** |
|:---|:---|:---|
| `min` | Fast representative execution for setup, teaching, and CI | May use deterministic synthetic inputs and must not be presented as a quality score. |
| `max` | Standard comparable scale for assignments, artifact evaluation, and candidate paper baselines | Comparable only when its row-specific quality, data, timing, provenance, and release gates pass. |
| `pro` | Controlled research space for repetitions, backends, model sizes, precision, sparsity, and ablations | Research envelope remains opt-in and must declare what changed. |

Validation presets remain separate. `smoke`, `coverage`, `max`, and `release`
describe which profile paths the harness executes and grades.

## Workload Coverage

A complete educational suite should cover the systems regimes common in
current coursework and academic evaluation.

- Language-model training, prefill, and autoregressive decode.
- Small language model serving, task-quality checks, quantization, batching,
  and long context.
- Vision training and model-efficiency studies.
- Sparse recommendation and memory behavior.
- TinyML and edge-style models.
- Agent, retrieval, and tool-use systems.
- Local distributed execution.
- Graph, time-series, and reinforcement-learning control flow.

Coverage does not require every row to carry a score. Systems-only rows can be
valuable when their inputs and limitations are honest. Promotion should happen
only after the row gains a defensible task and evidence contract.

## Research Envelope

The long-term `pro` surface should expose one controlled dimension at a time.

- fp32, fp16, bf16, int8, int4, weight-only, and KV-cache precision.
- Structured, unstructured, channel, block, and 2:4 sparsity.
- Prefill, decode, batching, context length, KV cache, and speculative decode.
- LoRA and other adapter configurations.
- PyTorch and optional local backends such as ONNX Runtime, MLX, llama.cpp,
  TVM, or IREE when their runners and packaging are stable.
- Embedding, activation, KV-cache, and communication memory behavior.
- Local multi-process and communication-versus-computation studies.
- Small-batch edge behavior and deployable model footprints.
- Retrieval, tool dispatch, and agent-loop systems costs.
- Estimated energy first, calibrated counters only when the platform and
  protocol support them.

The current registry contains examples of many of these directions, but most
remain systems-only. Their presence is not evidence that every research
variant is standardized.

## Laptop Contract

The standard path should need no cluster and no paid external API. CPU must be
a supported functional path. Apple MPS and CUDA may accelerate compatible
workloads. Score-bearing datasets and pinned SLM weights may require a one-time
download that is completed before measurement.

Runtime claims must come from retained reports on named hardware and software.
The project should publish distributions and ranges rather than one universal
number. The five-hour full-workflow ceiling is a CI safety boundary, not a
promise that every laptop completes in the same time.

## Milestones

| **Milestone** | **Exit Evidence** |
|:---|:---|
| Review-ready preview | Final-source tests, actual release validation, five score packets, inference chain, policy-permitted portable packages, rendered site, verified paper, and explicit external questions. |
| Independent teaching release | Authoritative component license, stable install artifact, macOS and Linux evidence, instructor pilot, and no unresolved in-repository gate. |
| MLCommons-reviewed project | Written decisions on name, sponsor, rules, result wording, assets, targets, and scenarios. |
| Research baseline | Multiple independent reproductions and papers that use the suite without private maintainer intervention. |
| Optional submission track | A reviewed subset with stricter system, data, schema, and result-publication rules. |

## Near-Term Order

1. Finish and retain the same-revision local validation ledger, then obtain
   green hosted CI for that review revision.
2. Transfer or publish the eight raw reference packets and run an independent
   reproduction on a second machine.
3. Resolve component licensing and the MovieLens decision.
4. Ask MLCommons for bounded feedback on naming, sponsorship, target policy,
   scenario scope, and result wording.
5. Pilot the exact release in courses and with independent artifact reviewers.
6. Promote systems-only rows one at a time as evidence warrants.

The north star is reached when running MLPerf EDU is ordinary, interpreting it
is disciplined, and challenging its evidence is straightforward.

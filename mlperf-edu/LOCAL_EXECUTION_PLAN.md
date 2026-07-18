# MLPerf EDU Local Execution Plan

## Decision

The first MLPerf EDU milestone requires every one of the fourteen workloads to
complete its authoritative quality path on a supported local machine. Remote
compute is not an acceptable dependency. A long run is acceptable, but a
workload that can only emit an environment handoff is not complete.

The existing benchmark identities and quality targets remain fixed during this
work. In particular, recommendation remains the MLPerf DLRM and Criteo
contract, and reinforcement learning remains the historical MLPerf MiniGo
contract. The implementation may change to fit local memory or use a modern
runtime, but the dataset, model semantics, evaluator, and quality gate may not
change silently.

This milestone is currently **12 of 14 runnable authoritative paths**. DLRM and
MiniGo are the two open local-execution paths. Eight workloads have a passing
quality result, four have a complete target miss under review, and the two open
local paths do not yet have authoritative results.

## Meaning of Local

The supported reference environment is a source checkout on a 64 GB Apple
Silicon machine with sufficient local SSD capacity. CPU and MPS are permitted.
The acceptance boundary is defined below.

| **Requirement** | **Acceptance Rule** |
|:---|:---|
| Compute | Every process runs on the local host. No remote worker or hosted inference endpoint is required. |
| Source | MLPerf EDU orchestration, adapters, evaluators, and compatibility code are committed in this repository. Permissively licensed upstream code needed at runtime is vendored with notices or fetched from an immutable revision and verified before use. |
| Dependencies | A locked installation or a locally built container may provide third-party dependencies. A container cannot require an unavailable accelerator. |
| Assets | `mlperf fetch` downloads and verifies every redistributable asset. Terms-gated assets use a documented manual download followed by the same digest and structure verification. |
| Memory | A workload may use streaming, memory mapping, sharding, or local scratch storage. It may not require more resident memory than the supported host provides. |
| Resume | Any run expected to take more than two hours writes restartable checkpoints and preserves completed work after interruption. |
| Quality | The complete declared dataset, evaluator, metric direction, tolerance, and target are used. A reduced local probe cannot produce a quality pass. |
| Evidence | A successful run emits JSON, CSV, HTML, and provenance artifacts and passes `mlperf verify`. |

The local milestone does not require repeated timing runs, a promoted
performance baseline, package-index publication, or production signing.

## Portfolio Closure Matrix

The first twelve workloads need a clean-cache fetch audit and one later local
acceptance run. Their benchmark design does not need to change. DLRM and
MiniGo require implementation work before those runs begin.

| **Workload** | **Local Path** | **Quality State** | **Milestone Work** |
|:---|:---|:---|:---|
| Image classification | Ready | Pass | Recheck fetch and run from a clean cache. |
| Keyword spotting | Ready | Pass | Recheck fetch and run from a clean cache. |
| Anomaly detection | Ready | Pass | Recheck fetch and run from a clean cache. |
| Visual wake words | Ready | Pass | Recheck fetch and run from a clean cache. |
| Causal language modeling | Ready | Pass | Recheck training, checkpoint lineage, and inference from a clean cache. |
| Text classification | Ready | Pass | Recheck fetch and run from a clean cache. |
| Information retrieval | Ready | Pass | Recheck fetch and run from a clean cache. |
| Graph node classification | Ready | Pass | Complete domain sign-off on the published tolerance. |
| Time-series forecasting | Ready | Target miss | Investigate implementation parity without relaxing the 0.290 MSE target. |
| Code generation | Ready | Target miss | Investigate the three-task gap without relaxing the 94-of-164 gate. |
| Function calling | Ready | Target miss | Inspect category and prompt parity without changing the pinned BFCL evaluator. |
| Image generation | Ready | Target miss | Inspect sampler and numerical parity without changing the three-trial FID contract. |
| Recommendation | Blocked | Not run | Add an out-of-core local DLRM execution path and a verified Criteo asset journey. |
| Reinforcement learning | Blocked | Not run | Add a native CPU or MPS MiniGo execution path with parity evidence. |

## Recommendation Workstream

The current DLRM runner preserves the official full-memory path. It assumes a
roughly 90 GB checkpoint, manually licensed Criteo Terabyte data, a legacy
runtime, and a 256 GB class host. The local implementation must preserve the
same unshuffled day 23 accuracy set and the 0.8025 ROC AUC gate while removing
the resident-memory requirement.

The work proceeds in this order.

1. Add a deterministic asset preparer for the manually downloaded Criteo
   archive and official checkpoint. It must validate terms acknowledgment,
   expected files, sizes, revisions, and digests before preprocessing.
2. Add resumable preprocessing for the exact accuracy split. Intermediate
   files must be content addressed and safe to reuse after interruption.
3. Implement a CPU out-of-core backend. Large embedding tables should use
   memory-mapped or sharded storage, bounded caching, and streamed batches.
   Dense layers and the official evaluator must retain the reference numeric
   behavior.
4. Build a small parity fixture from legally redistributable synthetic indices.
   Compare logits and ROC AUC aggregation against the pinned official backend
   before allowing the out-of-core backend to score the real dataset.
5. Add disk, memory, and estimated-duration checks to `mlperf doctor`. The
   command must explain manual data acquisition without treating it as a
   remote-compute handoff.
6. Run the complete accuracy set locally, verify the 0.8025 target, and retain
   the official full-memory backend as an explicit comparison path.

The feasibility spike passes only when the out-of-core backend can initialize
and evaluate a representative shard while remaining below 48 GiB of resident
memory. If it fails, recommendation returns to portfolio review. A MovieLens
substitution or a reduced Criteo split cannot inherit the DLRM quality claim.

## Reinforcement-Learning Workstream

The current MiniGo runner preserves the historical TensorFlow 1.x and CUDA
path. The local implementation must retain the 9-by-9 game, policy-value
network semantics, professional-move evaluator, 0.40 prediction gate, and 0.55
promotion playoff.

The work proceeds in this order.

1. Isolate the model, feature encoding, loss, optimizer schedule, self-play,
   search, promotion, and evaluation behavior from the pinned MLPerf source.
2. Implement a modern native backend that runs on CPU and Apple MPS. PyTorch is
   the preferred backend because it is already part of the locked suite and
   supports both devices.
3. Add weight and tensor-layout conversion for deterministic parity fixtures.
   Fixed positions must produce matching legal moves, features, policy logits,
   values, losses, and evaluator counts within declared numeric tolerances.
4. Make self-play, training, and playoff phases independently resumable. Every
   generated game, checkpoint, and promotion decision must be content
   addressed in the run manifest.
5. Add a bounded local acceptance fixture to continuous integration. It checks
   semantics and resume behavior but cannot satisfy the quality target.
6. Run the complete historical quality loop locally and retain the legacy
   container as an optional reference backend rather than a requirement.

The feasibility spike passes only when the native backend completes self-play,
one training update, checkpoint reload, professional-move evaluation, and a
playoff on both CPU and MPS where available. If semantic parity cannot be
demonstrated, MiniGo returns to portfolio review. CartPole or another small
control task would be a new benchmark decision, not an implementation fix.

## Asset and CLI Workstream

The user journey should expose one local preparation path for the entire
portfolio.

> **Planned CLI surface.** The following all-workload acceptance sequence is a
> design target, not a list of commands available today. In particular,
> `doctor --local` and `run --resume` are not implemented.

```bash
uv sync --locked
uv run mlperf doctor --profile max --collection all --local
uv run mlperf fetch --profile max --collection all
uv run mlperf doctor --profile max --collection all --local
uv run mlperf run --profile max --collection all \
  --resume --output-dir runs/local-acceptance
uv run mlperf verify runs/local-acceptance
```

The completed CLI must provide the following behavior.

- `doctor --local` checks compute support, memory, disk, dependencies, source
  revisions, assets, and resumability before a long run starts.
- `fetch --collection all` downloads every permitted asset, pauses with exact
  instructions for terms-gated data, and continues after the supplied files
  verify.
- `run --resume` skips verified completed stages and resumes interrupted
  preprocessing, training, generation, or evaluation.
- The suite report distinguishes missing assets, insufficient local resources,
  quality misses, and execution failures.

An asset audit must start with an empty cache and produce a machine-readable
ledger for all fourteen workloads. Each entry records the source URL or manual
source, immutable revision, expected digest, download size, expanded disk
requirement, license policy, and the command that verified it.

## Quality Closure

All fourteen targets already have an upstream basis. Local execution must not
turn target selection into target fitting. The closure rules are listed below.

1. Keep the six inherited benchmark gates unchanged.
2. Keep the seven published-reference reproduction points unchanged while
   implementation parity is investigated.
3. Obtain independent domain sign-off for the graph-classification tolerance
   and the nanoGPT reproduction interpretation.
4. Resolve the four measured gaps through source, preprocessing, model,
   evaluator, seed, or numerical-parity analysis. Any proposed target change
   requires a new source-backed review before another run.
5. Record one complete authoritative local result for DLRM and MiniGo. Repeated
   timing runs remain outside this milestone.

## Acceptance Gate

The milestone is complete only when a clean checkout on the supported local
machine satisfies every item below.

- [ ] The locked environment installs without importing code from another
  checkout.
- [ ] `mlperf health` completes all fourteen functional paths.
- [ ] The clean-cache asset ledger covers all fourteen workloads.
- [ ] Every redistributable asset downloads and verifies through the CLI.
- [ ] Every terms-gated asset has a complete manual download and verification
  path.
- [ ] `mlperf doctor --local` accepts every selected authoritative run.
- [ ] All fourteen authoritative runs complete locally without remote compute.
- [ ] Every run produces and verifies JSON, CSV, HTML, and provenance artifacts.
- [ ] Each run records a quality decision against the unchanged target.
- [ ] The four current target gaps are resolved or receive an independently
  approved source-backed contract decision.
- [ ] DLRM meets ROC AUC of at least 0.8025 with the local out-of-core backend.
- [ ] MiniGo meets professional-move prediction of at least 0.40 and preserves
  the 0.55 playoff rule with the native backend.
- [ ] The suite-level HTML report presents all fourteen outcomes without
  opening a browser automatically.

## Commit Sequence

Implementation should land in reviewable checkpoints.

1. Local milestone contract, asset ledger schema, and CLI preflight.
2. Clean-cache fetch closure for the existing twelve local workloads.
3. DLRM out-of-core feasibility fixture and parity tests.
4. DLRM complete local runner and resumable asset preparation.
5. MiniGo native feasibility fixture and parity tests.
6. MiniGo complete local runner and resumable quality loop.
7. Four target-gap investigations and fixes.
8. Clean-checkout fourteen-workload acceptance packet and dashboard review.

Every checkpoint should include focused tests and update this plan. Long
quality runs may be performed later, but their exact commands, inputs, expected
outputs, resume points, and acceptance rules must be committed before handoff.

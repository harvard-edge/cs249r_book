# MLPerf EDU Public Result Rules

## Status

These rules govern the independent MLPerf EDU v0.1 review candidate. They do
not authorize the use of an official MLPerf result label or imply MLCommons
acceptance.

## Eligible Workloads and Cases

Only registry-defined cases in a complete `reference_results/index.json` can
support a promoted public reference claim. The registered portfolio contains
fourteen workloads, while the current promotion scope contains nine workloads
and twelve evidence cases. The current
`provisional_results/index.json` contains six five-run verified project records
and six provisional records, but it is a draft-review surface and not a public
baseline index. A local run outside the closure can be useful for teaching or
research, but it is not a promoted reference result.

Code generation, function calling, recommendation, image generation, and
reinforcement learning set `promotion_scope` to false and cannot enter
`reference_results/`. They divide into two states. Code generation, function
calling, and image generation have executed their authoritative quality
contract and missed the unchanged target; the registry records this as
`quality-audited-target-not-met`, and the shortfall MUST be reported rather
than reframed as an unexecuted probe. Reinforcement learning has not executed
its authoritative contract locally, because MiniGo requires a legacy CUDA and
TensorFlow 1.x runtime outside the declared envelope.

## Result Roles

| **Role** | **Public Rule** |
|:---|:---|
| `score-bearing` | Reports the canonical task metric and timing after every quality gate passes. |
| `performance-bearing` | Reports timing after every functional gate passes. |
| `systems-only` | May report observations, but must not be presented as a comparable score or baseline. |
| `deferred` | Has an authoritative reference but no admitted laptop contract. |
| `rejected` | Does not satisfy the workload admission rule. |

The registry role is necessary but not sufficient. The complete evidence and
disclosure rules must also pass.

## Profiles

`min` is functional evidence only. It may use a deterministic reduced input
and must never establish a public quality or performance result. `max` is the
canonical real-data contract. `pro` retains the same workload identity while
exposing controlled single-node research configurations.

Profile names do not replace result roles. A `max` run that fails quality,
repeatability, provenance, or data policy is not public evidence.

## Workload Identity

Training and inference are modes. Full, prefill, and decode are phases.
Batching, precision, quantization, compilation, context length, scheduling,
and serving behavior are configurations or scenarios. Public labels must keep
those fields separate from the workload ID.

## Five-Run Promotion Protocol

A promoted case must satisfy every condition below:

1. Five fresh operating-system processes execute the canonical seed.
2. Every process completes without timeout or artifact loss.
3. Every quality or functional gate passes.
4. The declared aggregate gate passes.
5. The sample timing coefficient of variation is at most 5%.
6. Every run uses the same comparison fingerprint.
7. The source tree is clean and bound to one exact Git SHA.
8. The attempt preserves every declared report, manifest, and artifact digest.

The five fresh processes measure execution repeatability. They are not a
five-seed training experiment. A failed process invalidates the entire
attempt. Individual runs cannot be replaced.

## Power and Interruption Policy

Laptop reference campaigns should use AC power with Low Power Mode disabled.
The operator must disclose power source and platform power policy. Sleep,
hibernation, power-mode change, or material concurrent load invalidates the
affected attempt. A rejected attempt may be retained for audit, but it must not
be imported as promotion evidence.

## Quality Rules

A score-bearing case must inherit its task metric and quality reference from
the admitted upstream definition. Every individual run and the aggregate must
pass. A project-created proxy metric or synthetic substitute cannot support a
score-bearing claim.

Performance-bearing cases must pass their functional gate in every run. Their
performance values are measurements, not pass thresholds.

## Causal Lineage

Full, prefill, and decode inference must use one portable package that selects
exactly one committed training execution. For promotion, the selected run must
represent the five-run median quality. The package, checkpoint, source report,
and source provenance digests must match across all three phases. A provisional
lineage package must remain labeled provisional even when every digest check
passes.

## Required Disclosure

A result disclosure must include:

- Workload ID, profile, mode, phase, and scenario
- Result role and promotion status
- Primary metric and all five values
- Quality metric and all five values where applicable
- Median, range, sample standard deviation, and timing CV
- Dataset mode, split, versions, and artifact digests
- Model or checkpoint identity and lineage
- Requested device (`device_requested`), executed device (`device_executed`),
  and executed backend
- Hardware, operating system, Python, PyTorch, and runtime fingerprint
- Relevant precision, compilation, batching, and scheduling configuration
- Source Git SHA and evidence digest
- Any power, sleep, thermal, or background-load qualification
- Independent-preview and non-endorsement notice

The HTML report is the preferred human-readable disclosure. JSON remains the
authoritative machine-readable record.

## Provenance and Verification

Every result must retain its `.provd.json` manifest with the JSON report and
all referenced artifacts. Verification must pass against the original files
and after clean package extraction. SHA-256 checks integrity but does not
authenticate who produced the result.

## Dataset and Package Policy

Assets are fetched from pinned upstream locations. Dataset or model bytes may
be packaged only when their redistribution policy permits it. Otherwise, the
package must retain digests, source metadata, and a reproducible fetch recipe
without redistributing the bytes.

Unresolved licensing or redistribution policy blocks publication or packaging
of the affected bytes. It does not permit silent substitution.

## Prohibited Claims

The following statements are not permitted:

- Calling an independent result an official MLPerf result
- Comparing runs whose case or comparison fingerprints differ without disclosure
- Presenting `min`, synthetic, or systems-only output as task quality
- Reporting timing from a failed quality or functional run
- Hiding an interrupted, sleeping, or power-mode-changing attempt
- Treating a configuration as a new workload to inflate coverage
- Replacing an authoritative task with a smaller project-created proxy
- Claiming distributed or datacenter relevance from the v0.1 suite

Exact promoted values and evidence IDs will come from the complete strict
index. Draft values and evidence classes come from the committed twelve-case
provisional index. Hand-written documents must not maintain competing baseline
tables.

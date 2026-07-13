# MLPerf EDU North Star

> MLPerf EDU is a locally executable, quality-gated benchmark specification for
> teaching and studying single-node ML systems. It transfers the
> reproducibility, verification, disclosure, and comparability discipline of
> mature benchmark suites to classroom-scale PyTorch workloads. It supports
> controlled research on processors, memory systems, runtimes, compilers, and
> model execution while explicitly excluding distributed and datacenter-scale
> claims.

## Design Commitments

### Start From Authoritative Workloads

The project curates and packages established definitions. It does not invent
models, datasets, metrics, reduced tasks, or quality targets to fill a matrix.
The value is disciplined selection, local execution, measurement, provenance,
and reporting.

### Design Backward From Use

A student should be able to install the suite, inspect a workload, fetch its
assets, run it on a laptop, verify task quality, analyze system behavior, and
hand an instructor a reviewable artifact. A researcher should be able to alter
a controlled single-node configuration without losing task identity or
lineage.

### Keep Workload Identity Stable

Training and inference are modes. Full, prefill, and decode are phases.
Precision, quantization, compilation, batching, context length, scheduling,
and serving behavior are configurations. They appear in reports, not in
workload IDs.

### Make Quality a Gate

Timing without task quality is not a benchmark result. Every score-bearing run
must pass its inherited quality contract. Every performance-bearing phase must
pass a functional contract. No aggregate can hide a failed individual run.

### Make Evidence Reviewable

Every promoted case retains five fresh-process results, source and comparison
fingerprints, artifact hashes, acceptance decisions, and timing repeatability.
Training-to-inference dependencies use portable content-addressed lineage.

### Keep the Scale Local

The suite targets CPU and laptop accelerators. It can support architecture,
memory, compiler, runtime, and execution studies. It does not claim to model
distributed training or datacenter serving.

## Profiles

| **Profile** | **Design Intent** |
|:---|:---|
| `min` | Fast setup, teaching, and CI confidence. |
| `max` | Canonical classroom and comparison contract. |
| `pro` | Extended single-node research envelope under the same workload identity. |

These profiles describe execution scale and research intent. They do not
replace result roles or quality gates.

## v0.1 Portfolio Test

An admitted workload must answer yes to every question:

1. Is the task significant and established?
2. Is the upstream model or implementation authoritative?
3. Are the dataset, split, evaluator, metric, and target fixed upstream?
4. Can the unchanged contract run credibly on laptop-class hardware?
5. Does it add distinct classroom value?
6. Does it expose distinct single-node systems behavior?
7. Can all assets, versions, hashes, and adaptations be disclosed?
8. Can five fresh processes meet quality and repeatability gates?

The seven admitted workloads satisfy this test. The ten evidence cases cover
their canonical `max` paths plus three causal inference phases.

## Success Criteria

MLPerf EDU v0.1 succeeds when a machine-learning systems instructor can use it
without explaining away synthetic scores, arbitrary targets, broken setup, or
opaque provenance, and when a systems researcher can reproduce a local result
without guessing which task or weights were measured.

Initial MLCommons review is the governance milestone. It is not assumed in
advance and is never implied by the project name alone.

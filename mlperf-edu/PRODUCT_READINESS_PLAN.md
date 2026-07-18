# MLPerf EDU Product Readiness Plan

## Product Shape

MLPerf EDU should guide a user through three layers without pretending they
have the same evidentiary strength.

1. The health layer proves that installation, CLI dispatch, all fourteen
   functional paths, result serialization, and provenance verification work.
2. The assignment layer lets a student run an authoritative workload, modify
   one declared component, compare compatible results, explain the change, and
   submit a verifiable package.
3. The research layer lets a user define a controlled collection, preserve
   complete configuration and lineage, compare systems, and export analysis
   without weakening quality gates.

Production publication is a later release layer. It adds security, packaging,
governance, independent reproduction, and promoted stability evidence. The
initial classroom milestone does not depend on it.

## Dashboard Contract

The dashboard must render from result data and registry metadata. It must not
hardcode separate pages for `min`, `max`, or `pro`.

The first screen should answer four questions in order.

- Did the requested work complete?
- What claim is this profile allowed to make?
- Did the authoritative quality target pass?
- What should the user do next?

For a `min` health run, the lead cards should show workloads completed,
manifests verified, failures, and warnings. A stacked categorical bar should
show functional pass, failure, and skipped counts. The page must state that no
quality or performance baseline claim was made.

For a `max` run, the lead card should show the quality metric, target, direction,
and pass decision. A compact target marker or bullet chart can show the result
against its own target. Heterogeneous metrics must not share one raw numeric
axis.

For a `pro` collection, the lead cards should summarize completed workloads,
quality outcomes, environment-gated work, and provenance status. Suite graphs
should use categorical counts, workload-by-status matrices, and per-workload
small multiples. They should not rank unrelated quality metrics as if a FID,
accuracy, and MSE were comparable.

Baseline comparison is valid only when workload, profile, phase, dataset
revision, evaluator, checkpoint lineage, quality contract, and performance
fingerprint are compatible. The page should show quality margin separately
from performance delta. An incompatible result should explain which fields
differ instead of drawing a misleading chart.

## Classroom Journey

The intended student flow is short enough to teach in one lab.

1. Run `mlperf health` and open the generated HTML report only when it is useful.
2. Choose a laptop-capable workload with `mlperf list` and `mlperf show`.
3. Fetch and verify the pinned `max` assets.
4. Run the authoritative baseline once and read its quality decision.
5. Change one declared model, systems, or data-pipeline variable.
6. Run again and compare only if the compatibility check permits it.
7. Explain the quality and performance effects in the result notes.
8. Package the report and provenance manifest for submission.

The instructor should be able to distribute an assignment contract that fixes
the expected workload, profile, mode, phase, configuration, result count,
quality requirement, and allowed modifications. `mlperf grade` should validate
that contract, accept a safe portable package, and produce machine-readable and
HTML feedback.

## Example Curriculum

The next example directory should contain small, complete journeys rather than
isolated commands.

- `examples/01-health-check` for installation and report interpretation.
- `examples/02-inference-tradeoff` for a laptop-capable inference change that
  preserves the quality gate.
- `examples/03-training-tradeoff` for checkpoint lineage and a controlled
  training change.
- `examples/04-result-comparison` for compatible baseline comparison and an
  intentionally incompatible example.
- `examples/05-assignment-package` for submission, verification, and grading.
- `examples/research/pro-collection` for a pinned controlled research plan.

Each example should include learning goals, expected runtime and hardware,
commands, allowed changes, expected report sections, interpretation questions,
and a rubric.

## Research Journey

The pro workflow should accept a versioned experiment plan instead of acting
only as an alias for several `max` runs. A plan should declare workloads,
devices, phases, repetitions when needed, power mode, environment metadata,
and output policy. The result collection should retain every subrun and expose
quality, timing, energy when available, configuration differences, and
provenance compatibility.

Research examples should cover hardware comparison, compiler or precision
changes, training-to-inference lineage, controlled ablation, and export to JSON
or CSV for notebooks. DLRM and MiniGo should remain clearly labeled as
research-environment workloads.

## Work Queue

### Current foundation

- [x] Register all fourteen workloads and explain their portfolio role.
- [x] Provide `min`, `max`, and `pro` CLI paths.
- [x] Run and verify all fourteen `min` paths through `mlperf health`.
- [x] Generate report-driven JSON, CSV, and HTML output.
- [x] Add suite-level health summaries and categorical graphs.
- [x] Audit target authority, datasets, evaluators, and provenance.
- [x] Correct the PatchTST and EDM quality contracts.

### Assignment layer

- [x] Define a versioned assignment contract schema.
- [x] Add safe grading of portable result packages.
- [x] Validate expected workload, profile, phase, configuration, and result
  cardinality.
- [x] Add compatibility-checked baseline comparison to the CLI and dashboard.
- [x] Create the five classroom examples and instructor rubrics.
- [x] Render the numbered sequence in the website navigation.
- [x] Bind allowed plan edits to an instructor reference automatically.
- [x] Add a provenance-bound precomputed-baseline import for candidate-only
  training labs.
- [ ] Publish CPU and accelerator runtime, download, disk, and peak-memory
  budgets measured on course images.
- [x] Repeat simulated student, instructor, and research review with the
  rendered pages; record the supervised-pilot boundary.

### Research layer

- [x] Expose a versioned pro experiment-plan file.
- [x] Add fail-closed controlled configuration comparisons.
- [ ] Add controlled cross-hardware comparisons.
- [x] Add suite-level research quality small multiples and CSV/JSON export
  examples.
- [x] Complete current BFCL and EDM result packets.
- [x] Publish machine-readable DLRM and MiniGo execution handoffs with exact
  environment, asset, source, and command requirements.
- [ ] Execute DLRM and MiniGo in their required environments.

### Production release

- [ ] Close component license, naming, and governance decisions.
- [ ] Harden generated-code execution and untrusted pickle handling.
- [ ] Sign release artifacts and provenance where producer identity matters.
- [ ] Reproduce on independent CPU, Apple Silicon, and CUDA systems.
- [ ] Run the later stability campaign and promote compatible baselines.

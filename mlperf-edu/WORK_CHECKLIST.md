# MLPerf EDU Work Checklist

Last updated on July 18, 2026. This is the maintained progress view for the
fourteen-workload suite. [READINESS.md](READINESS.md) contains the detailed
evidence, and [PRODUCT_READINESS_PLAN.md](PRODUCT_READINESS_PLAN.md) explains
the intended classroom and research experience.

## Current Snapshot

- [x] 14 of 14 workloads are registered with an authoritative task, model,
  dataset, evaluator, and quality target.
- [x] 14 of 14 functional `min` paths pass the suite health workflow.
- [x] 14 of 14 authoritative `max` runners are implemented and fail closed when
  required assets or environments are unavailable.
- [x] 12 of 14 workloads have a complete authoritative quality result from at
  least one run.
- [x] 8 of those 12 results meet the declared target.
- [x] 4 of those 12 results record an honest target miss without weakening the
  target.
- [ ] 2 of 14 quality runs still require their declared external research
  environments.

The initial local suite is workable. Full one-run quality coverage is 12 of 14
until the DLRM and MiniGo handoffs are executed on suitable systems. Stability,
promotion, and production publication are separate later phases.

## Initial Suite

- [x] Provide one CLI for `min`, `max`, and `pro` workflows.
- [x] Pin or fail closed on source, dataset, model, checkpoint, and evaluator
  inputs.
- [x] Set the initial quality acceptance contract to one complete run.
- [x] Generate JSON, CSV, HTML, and provenance artifacts for every run.
- [x] Generate dashboards without opening a browser by default. Open one only
  after an explicit `--open-report` or `mlperf report --open` request.
- [x] Render dashboard content from result and registry data rather than
  hardcoding `min`, `max`, or `pro` pages.
- [x] Show suite health counts, per-workload quality decisions, target markers,
  compatible baseline comparisons, and research small multiples where the
  result data supports them.
- [x] Publish the 38-page website, all workload pages, classroom examples,
  instructor rubrics, and a pro research example.
- [x] Verify desktop and mobile layouts for the website and representative
  standalone dashboards.
- [x] Record simulated student, instructor, and research reviews and retain the
  supervised-pilot boundary.
- [x] Produce complete provenance-bound BFCL and EDM quality packets.
- [x] Publish machine-readable DLRM and MiniGo execution handoffs.
- [ ] Execute the DLRM quality run with the licensed Criteo data, official
  checkpoint, pinned legacy runtime, and a 256-GB-class system.
- [ ] Execute the MiniGo quality run in the reviewed immutable TensorFlow 1.x
  GPU environment.

## Quality Follow-Up

These items refine or approve quality interpretation. They do not justify
lowering a target to fit one local result.

- [ ] Investigate the PatchTST result of 0.292393 MSE against the unchanged
  target of at most 0.290.
- [ ] Investigate the HumanEval+ result of 91 passing tasks against the
  unchanged requirement of at least 94 of 164.
- [ ] Analyze the BFCL category gaps behind the 0.785208 result against the
  unchanged 0.8292 target.
- [ ] Review the EDM sampler and numerical path behind the 1.801554 minimum FID
  against the unchanged target of at most 1.79.
- [ ] Obtain domain approval for the one-sided OGB GCN target interpretation.
- [ ] Obtain independent approval for the nanoGPT target interpretation.
- [ ] Resolve the keyword-spotting adapter's quality-preserving but nonidentical
  promotion boundary.
- [ ] Close dataset redistribution and fetch-only wording before public
  release.

## Classroom and Research Follow-Up

- [x] Bind allowed experiment-plan edits automatically to the instructor's
  reference contract.
- [ ] Add a provenance-bound precomputed-baseline import for training labs that
  cannot produce the candidate checkpoint during class.
- [ ] Measure and publish CPU and accelerator runtime, download, disk, and
  peak-memory budgets on the course images.
- [ ] Add controlled cross-hardware comparisons.
- [ ] Populate measured working-set, arithmetic-intensity, and dispatch
  evidence for research interpretation.
- [ ] Perform an explicitly requested interactive review of a standalone run
  dashboard. Layout automation already passes and no browser should open as a
  side effect of routine verification.

## Stability and Promotion

- [ ] Run the later five-process stability campaign for promotable baselines.
- [ ] Enforce the declared timing-variation contract on promotion candidates.
- [ ] Reproduce compatible results on independent CPU, Apple Silicon, and CUDA
  systems.
- [ ] Promote only results whose quality, provenance, compatibility, and timing
  requirements all pass.

## Production Release

Production publication is not required for the initial classroom and research
preview.

- [ ] Close component licensing, package naming, versioning, and governance.
- [ ] Complete security review of generated-code execution, untrusted pickle
  handling, and legacy container paths.
- [ ] Close MLCommons review of the project name, scope, and result wording.
- [ ] Sign release artifacts and authenticate provenance where producer
  identity matters.
- [ ] Define support, vulnerability response, retention, and release rollback
  procedures.

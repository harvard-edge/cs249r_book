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

- [x] Investigate the PatchTST result. Two exact Apple Silicon reproductions
  reached 0.29168 and 0.29239 MSE. Neither met the unchanged 0.290 point, and
  no post-result tolerance was introduced.
- [x] Investigate the HumanEval+ result. Attention, dtype, Transformers, Qwen
  evaluator, and EvalPlus variants were cross-checked. The authoritative run
  still passed 91 of 164 tasks against the unchanged 94-task requirement.
- [x] Analyze the BFCL category gaps. Java and JavaScript account for the
  largest deficits, while Python, multiple, and parallel-multiple meet or
  exceed the corresponding published category scores. The aggregate remains
  0.785208 against the unchanged 0.8292 target.
- [x] Review the EDM sampler and numerical path. Three complete 50,000-image
  trials were independently rehashed and rescored, the official 18-step and
  35-evaluation schedule was preserved, and the evaluator cross-check differed
  by only 2.53e-8 FID. The best result remains 1.801554 against 1.79.
- [ ] Obtain domain approval for the one-sided OGB GCN target interpretation.
- [ ] Obtain independent approval for the nanoGPT target interpretation.
- [x] Resolve the keyword-spotting adapter boundary. Retain it as a disclosed,
  quality-preserving educational adaptation and block promotion until an
  exact-source execution path or authoritative unquantized weights establish
  parity.
- [x] Use conservative fetch-only wording and reject restricted dataset bytes
  from portable packages while decisions remain open.
- [ ] Close the remaining external dataset redistribution and component-terms
  decisions before public release.

## Classroom and Research Follow-Up

- [x] Bind allowed experiment-plan edits automatically to the instructor's
  reference contract.
- [x] Add a provenance-bound precomputed-baseline import for training labs that
  cannot produce the candidate checkpoint during class.
- [x] Measure and publish one-run CPU and accelerator-requested functional
  runtime, download, disk, and peak-memory budgets on the first course image.
- [ ] Measure authoritative `max` budgets for the workloads selected on each
  actual course image.
- [ ] Add controlled cross-hardware comparisons.
- [ ] Populate measured working-set, arithmetic-intensity, and dispatch
  evidence for research interpretation.
- [ ] Perform an explicitly requested interactive review of a standalone run
  dashboard. Layout automation already passes and no browser should open as a
  side effect of routine verification. The July 18 attempt stopped without
  opening anything because the in-app browser runtime reported no available
  browser.

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
- [x] Harden and document the controlled-preview generated-code and pinned
  pickle paths in [SECURITY_REVIEW.md](SECURITY_REVIEW.md).
- [ ] Replace executable EDM pickle inputs with safe reviewed artifacts and
  qualify independently built, signed DLRM and MiniGo runtime images.
- [ ] Close MLCommons review of the project name, scope, and result wording.
- [ ] Sign release artifacts and authenticate provenance where producer
  identity matters.
- [ ] Define support, vulnerability response, retention, and release rollback
  procedures.

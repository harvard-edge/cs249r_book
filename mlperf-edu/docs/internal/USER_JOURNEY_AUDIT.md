# MLPerf EDU User Journey Audit

> **Superseded in part, 2026-08-04.** Recommendation moved from DLRM on Criteo
> Terabyte to MLPerf Training v0.5 NCF on MovieLens-20M, reinforcement
> learning moved from a CUDA and TensorFlow 1.x container to a PyTorch
> adapter, and the timing protocol dropped from five runs to one. No
> workload is environment-gated. Statements below about gated execution,
> licensed Criteo data, or five-run promotion describe the state at the
> time of the audit and are retained as a record rather than corrected.
> Current state: [WORKLOAD_STATUS.md](WORKLOAD_STATUS.md) and
> [MISS_DIAGNOSIS.md](MISS_DIAGNOSIS.md).


Last updated on July 18, 2026. This checklist follows the product from a fresh
source checkout through setup, benchmark execution, result interpretation,
classroom submission, and research use. A checked item must have direct command
or rendered-page evidence from this audit. Automated layout checks supplement
visual review; they do not replace it.

The audit uses one complete run where a quality run is practical. It does not
repeat runs for stability or promotion. Routine commands must generate HTML
without opening a browser. Pages are opened only as an explicit audit action.

## 1. Installation and First Contact

- [x] Confirm the documented prerequisites match the actual project metadata.
- [x] Create an isolated environment from the source checkout.
- [x] Build and install the wheel in a clean temporary environment.
- [x] Confirm the installed `mlperf` entry point starts without source-tree
  imports.
- [x] Confirm `mlperf --help` presents the primary journey in understandable
  terms.
- [x] Confirm `mlperf doctor` reports Python, hardware, caches, registry, and
  optional dependencies without treating optional tools as failures.
- [x] Confirm setup and diagnostic commands do not open a browser.

## 2. Benchmark Discovery

- [x] Confirm `mlperf list` shows all fourteen workloads and their suite roles.
- [x] Confirm `mlperf show` explains model, dataset, evaluator, target, profile,
  and environment requirements for a laptop workload.
- [x] Confirm `mlperf show` makes DLRM and MiniGo external-environment gates
  clear before a user attempts an expensive run.
- [x] Confirm the generated website agrees with the CLI workload identities and
  target contracts.

## 3. Health and `min` Journey

- [x] Run the complete fourteen-workload health journey once.
- [x] Confirm all fourteen functional paths complete and all child provenance
  manifests verify.
- [x] Confirm the suite writes aggregate JSON, CSV, HTML, and workload rows.
- [x] Confirm a `min` result is labeled as functional setup evidence and makes
  no quality or performance-baseline claim.
- [x] Confirm no report opens automatically.
- [x] Confirm the terminal summary tells a student where each output was
  written and what to do next.

## 4. Authoritative `max` Journey

- [x] Run one practical laptop `max` workload through the public CLI.
- [x] Confirm the complete pinned dataset, model, evaluator, and configuration
  are used rather than the functional probe.
- [x] Confirm the quality metric, direction, target, tolerance, and decision are
  internally consistent across JSON, CSV, and HTML, with provenance binding the
  exact report.
- [x] Confirm a target pass or miss is stated plainly without converting it
  into a production or promotion claim.
- [x] Confirm the standalone dashboard shows the result against its own target
  with an appropriate visual encoding.
- [x] Confirm the report identifies the executed device and preserves source,
  model, dataset, and evaluator provenance.

## 5. Research `pro` Journey

- [x] Validate the packaged research plan in dry-run mode.
- [x] Run one bounded `pro` collection or representative pro fallback once.
- [x] Confirm the aggregate retains every subrun configuration, artifact, and
  child manifest.
- [x] Confirm heterogeneous quality metrics are not placed on a shared raw
  numeric ranking axis.
- [x] Confirm JSON and CSV exports are usable without parsing HTML.
- [x] Confirm environment-gated workloads remain visible and are not silently
  replaced with synthetic tasks.

## 6. Classroom Assignment Journey

- [x] Follow the health-check lab from its documented starting point.
- [x] Validate an instructor reference plan and an allowed student edit.
- [x] Confirm a disallowed plan edit fails before benchmark execution.
- [x] Exercise the provenance-bound instructor baseline import.
- [x] Create or verify a portable submission package.
- [x] Grade the package and confirm the canonical quality decision is
  recomputed rather than trusted from student-controlled fields.
- [x] Confirm restricted dataset bytes are excluded from portable packages.
- [x] Confirm the grade output is understandable to both a student and a TA.

## 7. Result and Comparison Semantics

- [x] Verify every produced provenance manifest through the public CLI.
- [x] Confirm compatible baseline and candidate results receive separate
  quality-margin and performance-delta treatment.
- [x] Confirm an incompatible comparison explains the mismatched fields instead
  of drawing a misleading chart.
- [x] Confirm failed, skipped, environment-gated, target-passed, and
  target-missed states remain visually and semantically distinct.
- [x] Confirm profile, mode, phase, workload, and device labels agree across
  terminal, JSON, CSV, and HTML, while provenance binds the exact report and
  hardware fingerprint.

## 8. Website and Dashboard Visual Review

- [x] Render all website pages from current sources.
- [x] Run automated desktop and mobile overflow/layout checks on every page.
- [x] Inspect the homepage, getting-started page, benchmark index, one workload
  page, labs index, instructor guide, results guide, and readiness page.
- [x] Inspect representative `min`, `max`, `pro`, and validation HTML plus
  grading terminal and JSON output at desktop size.
- [x] Inspect the same representative pages at a narrow mobile size.
- [x] Confirm navigation, headings, cards, charts, target markers, tables,
  warnings, and next-step guidance are readable and correctly prioritized.
- [x] Confirm color is not the only carrier of pass, warning, or failure state.
- [x] Save audit screenshots outside the repository and visually inspect them,
  rather than relying only on DOM assertions.

## 9. Failure Behavior and Final Acceptance

- [x] Confirm a missing external asset or runtime fails closed with a useful
  recovery instruction.
- [x] Confirm routine verification leaves the repository worktree clean.
- [x] Fix any in-scope defect found during the journey and rerun the affected
  path.
- [x] Run the full Python regression suite and generated-document checks.
- [x] Record the final supervised-classroom, research-preview, and production
  boundaries with any remaining external gates.

## Audit Outcome

### Verdict

The supervised classroom journey is ready for preview use from a source checkout.
A student can install the locked environment, run all fourteen functional paths,
read the health dashboard, execute a selected authoritative `max` workload,
inspect and convert its reports, and submit a provenance-bound package. The
bounded `pro` plan journey is also ready for research-preview use with one run
per condition.

All fourteen workloads have a declared initial quality contract with a metric,
direction, target, tolerance, evaluator or pinned runner, dataset policy, and
source rationale. Their current empirical state remains intentionally honest:
eight have at least one target-passing authoritative result, four retain a
recorded target gap, and recommendation plus reinforcement learning remain
external-environment gated. A green `min` result never changes those states.

This is not production or public-benchmark sign-off. Component licensing,
package-index publication, remaining dataset release decisions, MLCommons
governance, security review, independent hardware reproduction, and
authenticated producer identity remain release gates.

### Direct evidence

- A locked source environment and a clean wheel installation both loaded the
  packaged fourteen-workload registry without source-tree imports.
- `mlperf health` completed 14 of 14 functional paths, graded all fourteen,
  and verified every child manifest without opening a browser.
- The representative `max` run reproduced 87.00% image-classification
  accuracy against the unchanged 85.00% target on the pinned model, data,
  index, and evaluator.
- The packaged `pro` plan completed its baseline and candidate conditions
  with two quality passes and preserved both child evidence chains.
- Allowed and disallowed classroom plan edits, an imported instructor
  baseline, portable packaging, grading, restricted-data rejection, compatible
  comparison, and blocked comparison behavior were exercised.
- The public CLI verified 18 final-run manifests after the earlier 41-manifest
  audit pass.
- Playwright checked 40 website pages at two viewports, for 80 clean page runs.
  The dashboard gate checked nine representative result pages at two
  viewports, for 18 clean runs, plus a two-viewport visual-state fixture.
- Screenshots were stored outside the repository and manually reviewed for the
  homepage, getting started, registry, workload, labs, instructor, results,
  readiness, min, max, pro, health, comparison, and failure-state views.
- Registry mirrors, selection ledger, taxonomy, reference claims, generated
  pages, the ten-page review paper, the rendered site, and the clean wheel all
  passed their checks.
- All 391 Python tests passed from the clean audit commit, including the
  measurement-surface source-lock test.

### Defects found and fixed

- The suite-health source path caused a 51 px overflow on a 390 px viewport.
  Long metadata now wraps, and the dashboard layout gate covers the regression.
- Wide mobile website tables looked cut off because overlay scrollbars were
  hidden. They now show a swipe cue, and the site gate verifies that every
  internally scrolling table has one.
- `mlperf show` omitted evaluator, target direction and tolerance, mode,
  phase, and max execution gates. Discovery now exposes the complete contract,
  including the DLRM and MiniGo recovery requirements.
- CSV exports omitted mode, phase, scenario, and requested and executed device.
  JSON, CSV, HTML, and terminal labels now agree, while provenance binds the
  exact report and hardware fingerprint.
- Quality misses, skipped runs, unsupported paths, environment gates, and
  execution failures collapsed into the same dashboard badge. They now retain
  distinct text and color semantics.
- The homepage used a maintainer smoke preset as its primary setup command.
  It now starts students with `mlperf health`.
- Successful health runs now print the next authoritative quality step.
- Two assertions that depended on terminal line wrapping were made stable
  across long worktree paths.

### Remaining boundaries

The next spiral is quality refinement and later stability evidence, not basic
journey repair. It should address the four recorded target gaps, run DLRM and
MiniGo in their declared external environments, collect the required repeated
reference runs, fill the currently unmeasured systems-regime fields, close
dataset and component release decisions, and complete production governance.

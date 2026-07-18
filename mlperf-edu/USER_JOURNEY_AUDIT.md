# MLPerf EDU User Journey Audit

Last updated on July 18, 2026. This checklist follows the product from a fresh
source checkout through setup, benchmark execution, result interpretation,
classroom submission, and research use. A checked item must have direct command
or rendered-page evidence from this audit. Automated layout checks supplement
visual review; they do not replace it.

The audit uses one complete run where a quality run is practical. It does not
repeat runs for stability or promotion. Routine commands must generate HTML
without opening a browser. Pages are opened only as an explicit audit action.

## 1. Installation and First Contact

- [ ] Confirm the documented prerequisites match the actual project metadata.
- [ ] Create an isolated environment from the source checkout.
- [ ] Build and install the wheel in a clean temporary environment.
- [ ] Confirm the installed `mlperf` entry point starts without source-tree
  imports.
- [ ] Confirm `mlperf --help` presents the primary journey in understandable
  terms.
- [ ] Confirm `mlperf doctor` reports Python, hardware, caches, registry, and
  optional dependencies without treating optional tools as failures.
- [ ] Confirm setup and diagnostic commands do not open a browser.

## 2. Benchmark Discovery

- [ ] Confirm `mlperf list` shows all fourteen workloads and their suite roles.
- [ ] Confirm `mlperf show` explains model, dataset, evaluator, target, profile,
  and environment requirements for a laptop workload.
- [ ] Confirm `mlperf show` makes DLRM and MiniGo external-environment gates
  clear before a user attempts an expensive run.
- [ ] Confirm the generated website agrees with the CLI workload identities and
  target contracts.

## 3. Health and `min` Journey

- [ ] Run the complete fourteen-workload health journey once.
- [ ] Confirm all fourteen functional paths complete and all child provenance
  manifests verify.
- [ ] Confirm the suite writes aggregate JSON, CSV, HTML, and workload rows.
- [ ] Confirm a `min` result is labeled as functional setup evidence and makes
  no quality or performance-baseline claim.
- [ ] Confirm no report opens automatically.
- [ ] Confirm the terminal summary tells a student where each output was
  written and what to do next.

## 4. Authoritative `max` Journey

- [ ] Run one practical laptop `max` workload through the public CLI.
- [ ] Confirm the complete pinned dataset, model, evaluator, and configuration
  are used rather than the functional probe.
- [ ] Confirm the quality metric, direction, target, tolerance, and decision are
  internally consistent across JSON, CSV, HTML, and provenance.
- [ ] Confirm a target pass or miss is stated plainly without converting it
  into a production or promotion claim.
- [ ] Confirm the standalone dashboard shows the result against its own target
  with an appropriate visual encoding.
- [ ] Confirm the report identifies the executed device and preserves source,
  model, dataset, and evaluator provenance.

## 5. Research `pro` Journey

- [ ] Validate the packaged research plan in dry-run mode.
- [ ] Run one bounded `pro` collection or representative pro fallback once.
- [ ] Confirm the aggregate retains every subrun configuration, artifact, and
  child manifest.
- [ ] Confirm heterogeneous quality metrics are not placed on a shared raw
  numeric ranking axis.
- [ ] Confirm JSON and CSV exports are usable without parsing HTML.
- [ ] Confirm environment-gated workloads remain visible and are not silently
  replaced with synthetic tasks.

## 6. Classroom Assignment Journey

- [ ] Follow the health-check lab from its documented starting point.
- [ ] Validate an instructor reference plan and an allowed student edit.
- [ ] Confirm a disallowed plan edit fails before benchmark execution.
- [ ] Exercise the provenance-bound instructor baseline import.
- [ ] Create or verify a portable submission package.
- [ ] Grade the package and confirm the canonical quality decision is
  recomputed rather than trusted from student-controlled fields.
- [ ] Confirm restricted dataset bytes are excluded from portable packages.
- [ ] Confirm the grade output is understandable to both a student and a TA.

## 7. Result and Comparison Semantics

- [ ] Verify every produced provenance manifest through the public CLI.
- [ ] Confirm compatible baseline and candidate results receive separate
  quality-margin and performance-delta treatment.
- [ ] Confirm an incompatible comparison explains the mismatched fields instead
  of drawing a misleading chart.
- [ ] Confirm failed, skipped, environment-gated, target-passed, and
  target-missed states remain visually and semantically distinct.
- [ ] Confirm profile, mode, phase, workload, and device labels agree across
  terminal, JSON, CSV, HTML, and provenance outputs.

## 8. Website and Dashboard Visual Review

- [ ] Render all website pages from current sources.
- [ ] Run automated desktop and mobile overflow/layout checks on every page.
- [ ] Inspect the homepage, getting-started page, benchmark index, one workload
  page, labs index, instructor guide, results guide, and readiness page.
- [ ] Inspect representative `min`, `max`, `pro`, validation, and grading HTML
  at desktop size.
- [ ] Inspect the same representative pages at a narrow mobile size.
- [ ] Confirm navigation, headings, cards, charts, target markers, tables,
  warnings, and next-step guidance are readable and correctly prioritized.
- [ ] Confirm color is not the only carrier of pass, warning, or failure state.
- [ ] Save audit screenshots outside the repository and visually inspect them,
  rather than relying only on DOM assertions.

## 9. Failure Behavior and Final Acceptance

- [ ] Confirm a missing external asset or runtime fails closed with a useful
  recovery instruction.
- [ ] Confirm routine verification leaves the repository worktree clean.
- [ ] Fix any in-scope defect found during the journey and rerun the affected
  path.
- [ ] Run the full Python regression suite and generated-document checks.
- [ ] Record the final supervised-classroom, research-preview, and production
  boundaries with any remaining external gates.

## Audit Outcome

Pending. The final section will summarize observed results, visual findings,
defects fixed, commands used, and remaining limitations after every checklist
item has been resolved or explicitly bounded.

# MLPerf EDU Ship Plan

Goal: a benchmark someone can download and run. The paper is the accompanying
specification, not a study of learning outcomes.

## The Acceptance Test

Everything in this plan serves one testable outcome. On a machine that has never
seen this project:

```bash
pip install mlperf-edu
mlperf doctor
mlperf run --workload image-classification
```

This must produce a graded, provenance-verified result in under five minutes,
with no repository clone, no manual dataset step, and no editing of paths. It
must hold on macOS arm64, Linux x86 CPU, and Linux CUDA.

Today none of those three lines works for an outside user. The repository is a
subdirectory of a textbook, there is no published package, and every result was
produced on one Apple M5 Max.

## Status Entering the Plan

Verified against the registry and evidence index on 2026-08-03, recomputing each
retained result against the current gate.

| Tier | Count | Workloads |
|:---|:---|:---|
| Verified (gate + repeated-timing repeatability) | 6 | anomaly-detection, image-classification, information-retrieval, keyword-spotting, text-classification, visual-wake-words |
| Passes gate, under five runs | 2 | causal-language-modeling, graph-node-classification |
| Quality miss | 1 | time-series-forecasting |
| No retained local result | 5 | code-generation, function-calling, image-generation, recommendation, reinforcement-learning |

All six verified benchmarks are inference, and four of the six are inherited
MLPerf Tiny contracts. No training benchmark is verified.

---

## Phase 0 — Decisions That Block Everything Else

These are not engineering tasks. Nothing downstream can ship until they close.

### 0.1 License

Choose and commit an authoritative license for the published artifact, plus a
component-terms statement covering inherited models, datasets, and evaluators.

Acceptance: `LICENSE` exists at the package root; every third-party component is
listed with its own terms; `pip show mlperf-edu` reports a license.

Blocker severity: absolute. No public release without it.

### 0.2 Name and MLCommons Relationship

`MLPerf` is an MLCommons trademark. Publishing a package named `mlperf-edu`
that installs a `mlperf` console script is a trademark and namespace conflict
independent of technical merit.

Three outcomes, in descending preference:

1. MLCommons endorses the name and the project ships as an affiliated
   educational suite.
2. MLCommons permits the name with a disclaimer, and the console script is
   renamed to avoid the bare `mlperf` entry point.
3. The project renames. Working alternatives that keep the lineage legible in
   prose without claiming the mark.

Acceptance: a written answer from MLCommons, and a console-script name that does
not collide with any official tool.

Blocker severity: absolute for public release. Start this first because it has
the longest external latency.

### 0.3 Repository Shape

The suite currently lives inside a textbook repository. An outside user cannot
download it without the book.

Options: extract to a standalone repository with its own issues and CI, or keep
development in place and publish only the built package. Standalone is strongly
preferred; a benchmark that cannot be forked cannot attract submissions.

Acceptance: a clone-and-run path that does not mention the textbook.

### 0.4 v0.1 Suite Scope

Shipping fourteen workloads when two cannot execute locally and one misses its
gate produces a bad first run. Recommendation: ship the eight workloads with a
real local max result as the benchmark suite, and mark the remaining six as
preview. Preview workloads keep their registry identity and `min` probe but are
excluded from the default run set and from any suite-level claim.

Acceptance: `mlperf run` with no workload argument executes only workloads that
are expected to succeed on the user's machine.

---

## Phase 1 — Make It Installable

### 1.1 Publish the package

A wheel already builds. It is not published.

- Reserve the package name on PyPI (gated on 0.2).
- Publish to TestPyPI, install into a clean environment, run the acceptance test.
- Publish to PyPI with pinned lower bounds, not the development lockfile.
- Verify the console script resolves without the source checkout on PATH.

Acceptance: `pip install mlperf-edu` in a fresh venv, then `mlperf doctor` green.

### 1.2 Host the assets

Datasets and checkpoints currently resolve through fetch paths validated on one
machine, and larger evidence packages have no public URL.

- Enumerate every asset a shipped workload fetches, with size, source, license,
  and redistribution status.
- For each: fetch from upstream, mirror, or exclude the workload from v0.1.
- Publish a manifest with digests so a fetch failure is diagnosable.

Acceptance: a clean-cache run of all shipped workloads on a machine with no
prior state, from a network location that is not the author's.

### 1.3 Close the dataset policy

`DATASET_RELEASE_REVIEW.md` leaves redistribution open for CIFAR-10, SST-2,
NanoBEIR, ETTm1, KWS, and visual wake words. Each needs a decision: mirror,
fetch-only, or drop.

Acceptance: no shipped workload depends on an undecided asset.

---

## Phase 2 — Make It Portable

This is the largest engineering block and the biggest risk to the goal. Every
number in the project comes from one laptop.

### 2.1 Hosted CI on Linux

- GitHub Actions job running `mlperf validate coverage` on Linux x86 CPU for
  every pull request.
- Cache assets between runs so the job is minutes, not tens of minutes.
- Fail the build on any workload regression.

Acceptance: a green Linux badge that runs the real suite, not a lint job.

### 2.2 Second and third reference hosts

- Linux x86 CPU: full `max` sweep on shipped workloads.
- Linux CUDA: full `max` sweep on shipped workloads.
- Record quality transfer explicitly. Task metrics should transfer within
  declared tolerance; runtime will not.

Acceptance: the evidence index holds results from at least three distinct hosts,
and the paper reports quality transfer across them.

This item also removes the single most likely paper rejection reason. It is
worth more than any amount of editing.

### 2.3 Numerical tolerance policy

Cross-platform runs will produce small metric differences. Decide in advance,
and in writing, what deviation is acceptable per metric, before seeing the
numbers. Deciding after the fact is exactly the failure the project already
refused once with PatchTST.

Acceptance: a predeclared tolerance rule in the registry, with a reviewer note.

---

## Phase 3 — Make the First Run Good

### 3.1 Progress output

A cold `mlperf run --workload anomaly-detection --profile max` currently prints
two lines and then goes silent for roughly three minutes while it fetches. A
first-time user cannot distinguish that from a hang.

- Emit fetch progress with sizes and destinations.
- Emit a phase line on entering load, warmup, measure, evaluate.
- Print an expected-duration estimate from the registry envelope.

### 3.2 Time to first result

Advertise and enforce a fastest path. `image-classification` at max is roughly
eight seconds of compute once assets are cached.

Acceptance: a documented command that produces a graded result in under five
minutes cold, including fetch, on a mid-range laptop.

### 3.3 Fail closed with an actionable message

No workload fails closed on environment any more. The remaining fail-closed
paths are asset fetches and quality gates, and those messages must still tell a
user what to do next rather than only what went wrong.

### 3.4 Quarantine preview workloads

Implement the 0.4 decision: preview workloads are excluded from default
selection and from suite-level counts.

---

## Phase 4 — Documentation and Website

### 4.1 Fix the known defects

- `SPEC.md:264` states "exactly nine workload definitions" in the normative
  conformance section for a fourteen-workload registry. This is the single most
  damaging line in the docs.
- Six citations describe code-generation, function-calling, and image-generation
  as un-executed probes; the registry marks all three
  `quality-audited-target-not-met`. Correct `PUBLIC_RULES.md:20-23` first, since
  it governs public claims.
- `RELEASE_CHECKLIST.md:212` claims a default-opening dashboard; the CLI
  deliberately does not open a browser.
- `tools/generate_docs.py --check` fails on a clean checkout from argparse line
  wrapping. Three documents advertise it as a green gate. Make the check
  width-stable.
- Page count appears as 38, 40, and 41 across three documents.
- `labs/` contains only an untracked stale `__pycache__`; the real labs are in
  `examples/`. Remove it.

### 4.2 Consolidate the document set

Eighteen top-level status documents is the direct cause of the drift above: the
same claim is restated in six places, so correcting one leaves five wrong.

Ship set: `README.md`, `INSTALL.md`, one rules document, one results document,
and the website. Move working documents to `docs/internal/` or retire them.

Merge candidates identified by audit: NORTH_STAR + PROPOSAL + DESIGN_PHILOSOPHY
share verbatim paragraphs; WORK_CHECKLIST + READINESS + PRODUCT_READINESS_PLAN +
LOCAL_EXECUTION_PLAN are four overlapping ledgers over the same fourteen
workloads; SPEC + PUBLIC_RULES duplicate the protocol tables.

### 4.3 Website

- Build, review at desktop and narrow viewport, and host it.
- The landing page must answer, above the fold: what this is, what it runs on,
  how long a first run takes, and the install command.
- Every shipped workload needs a page stating its upstream authority, gate,
  assets, and current result.
- Publish the results table from the generated index so it cannot drift.

Acceptance: a hosted URL where a stranger can decide in sixty seconds whether
this is useful to them.

---

## Phase 5 — The Paper as Specification

The paper's job is to define the benchmark and give people something to cite. It
is not a learning-outcomes study, and the current subtitle promises pedagogy the
evidence does not support.

### 5.1 Finish generating the numbers

145 hardcoded numeric tokens remain across 116 lines. Two are already wrong:
the paper says "one expected skip" where the suite reports two, and "a clean
Python 3.12 wheel" where the reference machine runs 3.14.4.

- Emit host macros from the run fingerprint, which already records chip, machine
  model, memory, core counts, OS, Python, and PyTorch versions.
- Replace spelled-out counts that duplicate existing macros.
- Add a build check that fails on a bare numeral in result prose.

### 5.2 Retitle and refocus

Drop "Pedagogical Framework." The paper describes a runnable, quality-gated
benchmark for single-node ML systems, its rules, and its reference results.
Education motivates the resource envelope; it is not the claim.

### 5.3 Report what the suite actually is

State plainly that the verified core is six inference benchmarks, four inherited
from MLPerf Tiny. Report the four honest target misses as results. A suite where
everything passes says nothing about whether its gates bind.

### 5.4 Add cross-host results

Fold Phase 2 evidence in. This converts the weakest section into a strength.

---

## Phase 6 — Feedback From Real Users

Only after Phase 1 and 3, because feedback on an uninstallable artifact measures
installation friction rather than benchmark design.

- Three task-based protocols: student running a first benchmark, instructor
  building an assignment, researcher comparing two configurations.
- Each is a scripted task list with timing and a short structured form, not an
  opinion survey.
- Recruit from a course cohort and a small set of external systems researchers.
- Feed results into an issue tracker, not a document.

---

## Ordering

Phase 0 starts immediately and runs in parallel with everything, because the
MLCommons conversation has the longest external latency.

Critical path to the acceptance test: 0.1 and 0.2, then 1.1 and 1.2, then 2.1,
then 3.1 through 3.4. Phases 4 and 5 proceed in parallel. Phase 6 waits.

## What Would Make This Fail

- Shipping without the name and license resolved, and having to withdraw.
- Publishing while all evidence comes from one machine, so the first outside user
  on Linux hits failures the author never saw.
- Shipping fourteen workloads where six do not work for the downloader.
- Continuing to maintain eighteen overlapping status documents, so published
  claims keep drifting from the registry.

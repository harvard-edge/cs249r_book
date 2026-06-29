# MLPerf EDU Iteration Loop

This loop is the operating rhythm for turning MLPerf EDU into a course-safe,
research-useful, MLCommons-reviewable benchmark suite. The first job is not to
add more workloads. The first job is to repeatedly expose the suite to the
right reviewers, capture what breaks, and turn that feedback into runnable
changes.

## Loop Goal

Every iteration should answer one question:

> Can the intended audience run, understand, trust, and compare this benchmark
> without becoming benchmark-infrastructure experts?

If the answer is no, the iteration should produce a small, concrete fix:
better CLI wording, clearer docs, a hardened workload, a stronger report, a
better validation suite, or a demotion from public-score status.

The loop shape is:

```text
Intent -> Prototype -> Private review -> Maintainer synthesis
      -> Stakeholder packet -> Decision -> Evidence
      -> Next intent
```

The important discipline is that internal review can be noisy, but the product
surface must stay plain. Students should see commands that help them finish a
benchmark. Instructors should see grading and runtime affordances. Maintainers
can see validation suites, public-result audits, and release checklists.

## Stakeholder Lanes

Keep these lanes separate. A command or document can serve more than one lane,
but the intended reader must be obvious.

| Lane | Primary Question | Typical Reviewer | Output |
|---|---|---|---|
| Student | Can I install it, run it, and understand the result? | New course user | Confusion log, broken command list, first-run timing |
| Instructor | Can I assign, grade, and debug it in class? | Course staff | Lab fit, grading friction, runtime budget, report needs |
| Architecture researcher | Can I use this for a defensible systems paper? | ISCA/MICRO/HPCA/ASPLOS-style reviewer | Comparability, knobs, counters, workload realism, artifact requirements |
| MLCommons reviewer | Is the naming, result language, and rules posture acceptable? | MLCommons working group reviewer | Rules changes, endorsement blockers, public-result boundaries |
| Maintainer | Can we keep this passing from a fresh clone? | Repo owner / CI owner | Tests, validation suites, schema/versioning, release checklist |

The normal user path should stay small:

```bash
mlperf doctor
mlperf list
mlperf fetch --profile max
mlperf run --profile max --open-report
mlperf report <report.json> --format html --open
```

Maintainer validation is separate:

```bash
mlperf audit
mlperf validate smoke
mlperf validate coverage
mlperf validate max
mlperf validate release
pytest
```

Do not expose internal process labels in the learner path. Use plain language:

| Internal Concept | User-Facing Language |
|---|---|
| validation suite | check this setup or run suite validation |
| public-result audit | rules check |
| reviewer artifact | submission package |
| release blocker | needs changes before release |
| maintainer triage | project follow-up |

## Feedback Intake

Use one visible feedback entry point, then route internally by audience.

The intake form, issue template, or future CLI helper should ask:

- What were you trying to do?
- Which command did you run?
- What happened?
- What did you expect?
- Which role best describes you: student, instructor, researcher, reviewer, or
  maintainer?
- Can the tool attach environment, version, report, and error context?

Do not ask students to write maintainer-quality bug reports. The tool should
collect reproducible context where possible. Maintainers can later map the
feedback to product, docs, curriculum, benchmark rules, infrastructure, or
governance.

## One Iteration

Run this loop for one user-facing problem, one workload, or one release-risk at
a time.

1. **Frame the question.**
   Write the exact question for the round. Examples:
   "Can a student run the default benchmark path from a fresh install?"
   "Can an architecture reviewer compare two quantization papers using the SLM
   suite?"
   "Can MLCommons reviewers tell which results are educational and which are
   official?"

2. **Create review packets.**
   Give each lane the smallest runnable packet:
   commands, expected runtime, expected report path, and the specific thing to
   judge. Do not ask every reviewer to inspect the whole repo.
   Stakeholder packets should be clean and user-facing; internal agent notes,
   half-decisions, and maintainer uncertainty stay out of the packet.

3. **Run parallel reviews.**
   Use humans when available. Use parallel agents for focused drafts or audits
   when humans are not available yet. Assign one lane per reviewer:
   student UX, instructor workflow, architecture/research credibility,
   MLCommons rules posture, or maintainer validation. Do not ask agents to
   solve the same question twice.

4. **Capture feedback as evidence.**
   Every finding should include:
   audience, command or file, expected behavior, observed behavior, severity,
   and proposed fix. "This is confusing" is not enough; capture where the
   confusion happened.

5. **Triage by audience impact.**
   Use this order:
   student cannot run it, instructor cannot assign it, public result would be
   misleading, researcher cannot compare results, maintainer cannot keep it
   passing, nice-to-have polish.

6. **Choose one bounded slice.**
   Implement one slice before broadening scope. Good slices are:
   rename one command, harden one workload, improve one report, add one audit
   rule, promote/demote one public status, or fix one validation suite.

7. **Validate in layers.**
   Use the lightest checks first, then expand:

   ```bash
   mlperf doctor
   mlperf run --workload <id> --profile min
   mlperf run --workload <id> --profile max
   mlperf report <report.json> --format html
   mlperf audit
   mlperf validate smoke
   pytest
   ```

   Run `mlperf validate coverage`, `max`, or `release` when the change
   touches registry policy, reports, public status, or shared runners.

8. **Write the decision.**
   End the iteration with a short decision record:
   what changed, which audience it helped, what was intentionally deferred,
   and which commands passed.
   Close the loop with the original reporter when possible: fixed in a version,
   clarified in docs, deferred with rationale, or rejected because it conflicts
   with the benchmark contract.

9. **Restart with the next highest-risk question.**
   Do not keep adding features while the current user path is confusing or
   while public-result language is ambiguous.

## Parallel Agent Pattern

Parallel agents are useful only when each agent has a distinct lens.

Good agent prompts:

- Student UX reviewer: run the README mentally and flag unclear commands.
- Instructor reviewer: identify grading, runtime, and course failure modes.
- Architecture reviewer: identify comparability gaps for pruning,
  quantization, memory-system, or serving papers.
- MLCommons rules reviewer: identify naming, result-language, and endorsement
  risks.
- Maintainer reviewer: identify missing tests, schema drift, or CI gaps.

Bad agent prompts:

- "Review everything."
- "Make MLPerf EDU better."
- "Find all bugs."
- Asking multiple agents to edit the same files at the same time.

For code changes, give each agent a disjoint write scope. For review-only work,
ask for findings with file/command references and do the integration in one
place.

Agents should not post directly to external stakeholders. They generate private
review notes; maintainers synthesize those notes into a coherent decision or a
clean stakeholder packet.

## Round Template

Use this template for every iteration.

````markdown
# MLPerf EDU Iteration: <short name>

Date:
Owner:
Round question:

## Audience

- Student:
- Instructor:
- Architecture researcher:
- MLCommons reviewer:
- Maintainer:

## Review Packet

Commands:

```bash
<commands under review>
```

Expected artifacts:

- Report:
- HTML:
- CSV:
- Provenance:

Expected runtime:

## Findings

| Audience | Severity | Evidence | Proposed Fix | Decision |
|---|---|---|---|---|
| | | | | |

## Implementation Slice

Files:

-

Decision:

## Verification

```bash
mlperf doctor
mlperf audit
mlperf validate smoke
pytest
```

Result:

## Deferred

-
````

## Current Highest-Value Rounds

1. **Student first run:** fresh install, `doctor`, `list`, `fetch`, `run`,
   `report --open`.
2. **Instructor assignment:** one lab using the default `max` path, grading artifacts,
   and timing expectations.
3. **Research comparability:** one quantization or pruning study using SLM or
   MobileNet, with fixed knobs and report fields.
4. **Public result boundary:** audit which workloads are score-bearing,
   performance-bearing, systems-only, or experimental.
5. **Release validation:** run `audit`, `validate smoke`, `validate coverage`,
   `validate max`, `validate release`, and `pytest` from a clean checkout.

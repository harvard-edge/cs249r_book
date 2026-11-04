# Lab Development Protocol

This public stub tracks the release-facing protocol checks that are enforced by
the labs test suite.

## Release Invariants

1. Every lab must be a valid Marimo app.
2. Browser labs must include the WASM bootstrap and relative `mlsysim` wheel path.
3. Prediction widgets must be structured controls, not free-text prompts.
4. Interactive widgets must be returned through Marimo dataflow.
5. Runtime-installed WASM packages must be imported after `micropip.install(...)`.
6. Labs should render tabbed parts plus a synthesis section.
7. Labs should save decisions to the Design Ledger where applicable.

Run `python3 -m pytest labs/tests/ -v` from the repository root for the current
automated checks.

## Tradeoff Experiments

A lab should leave students able to defend a design choice under stated
constraints. Its calculations support that decision. Completing arithmetic or
selecting the correct prediction is not the learning objective.

### Design from concepts before implementing controls

Read the complete chapter and its concept map. Give each primary concept an
explicit disposition: required investigation, explanation within an
investigation, optional extension, or reading only, with a reason. Identify the
chapter-owned claim and misconception before choosing the interface. The
conclusion integrates earlier owners without introducing new concepts.

Treat worked and napkin examples as starting points. Identify which fixed
assumption, untested consequence, or missing evidence the lab should open for
investigation. A chapter claim can require a new MLSysIM scenario even when the
book has no complete worked example for it. Do not invent empirical quality or
hardware measurements to complete a lesson.

Specify the changed and fixed conditions, an informative counterexample or
limiting case, the teaching instrument, the saved comparison, and a
chapter-specific discussion prompt. Choose the instrument for what students
need to notice. Include a changed-assumption transfer question and distinguish
predicted consequences from observed results. Optional depth must not silently
become a completion requirement.

### Design the submission with the experiment

Support an instructor assigning reading, core investigations, a downloaded
discussion report, and submission through the course's existing system. Retain
actual predictions, tested settings, outcomes, units, and scenario versions.
Keep those captured facts visibly separate from the student's interpretation.
Ask students to explain the mechanism, defend a chosen or rejected alternative,
identify uncertainty, and name a useful follow-up experiment.

LLM assistance can be permitted without making unsupported claims acceptable.
Assess reasoning tied to the captured evidence, comparable conditions,
trade-offs, limitations, and transfer. Do not grade prediction accuracy,
agreement with a universal winner, or prose polish as substitutes for
understanding. A justified hold or baseline retention can be a sound conclusion.
Capture identifiers connect claims to results; they do not prove authorship.

Each substantive part should give students a short experiment:

1. Predict a consequence before seeing the result. An incorrect prediction
   should still unlock exploration.
2. Change a meaningful system choice while holding the comparison workload
   fixed, or explicitly identify the changed workload.
3. Compare the baseline and alternative in real units. Show both the benefit
   and the cost, alongside the constraint that determines feasibility.
4. Cross a failure boundary and recover when the scenario permits it. Explain
   why a better headline metric can still produce an unacceptable design.
5. Save the tested settings and outcomes before making a recommendation.
   Synthesis should use those saved experiments, including a rejected
   alternative, a remaining limitation, and a condition for revisiting the choice.

Charts should make the tradeoff visible: a selected point on a curve, a budget
boundary, or a comparison of competing outcomes. A table of calculated answers
alone does not establish that students explored the design space. Keep formulas
and assumptions available in Calculation Notes.

## Calculation Ownership

MLSysIM owns physical behavior, reusable scenario assumptions, and registry
facts. Notebooks select inputs and render returned results; the lab helper
package owns presentation, evidence capture, and reports. Extend MLSysIM when a
model is missing instead of implementing another calculator in a notebook or
lab helper. Label illustrative scenarios as assumptions rather than measured
hardware behavior.

Browser releases must rebuild the MLSysIM and lab-helper wheels from the source
being released. A successful native run does not verify a checked-in browser
wheel, and structural tests do not establish physical fidelity or teaching quality.

## Review in the Browser

Inspect the rendered lab after unlocking predictions. Exercise controls, check
that outputs change, cross and recover from a constraint violation, and switch
contexts to verify that predictions and evidence reset appropriately. Inspect
both desktop and narrow-screen layouts, including charts, legends, tables, and
tab navigation. Report automated checks separately from these observations.

# Lab 10 Learner Feedback Loop - 2026-06-04

## Scope

- Pilot lab: `labs/vol1/lab_10_model_compress.py`
- Goal: verify that students can read the setup, choose a track, enter the parts, and understand why each compression exercise matters for the chosen deployment.
- Rule: the user should not be the first render check. Browser render, click-through, simulated feedback, and regression tests must happen before asking for review.

## Render Evidence

- Preview URL used locally: `http://localhost:30310`
- Browser screenshots captured under `/tmp/mlsysbook-lab10-learner-loop-v2/`
- Full interaction smoke artifacts captured under `/tmp/mlsysbook-feedback-loop-full/`

Observed checks:

- Four track choices were clickable: iPhone, Oura Ring, RoboTaxi, Cloud Fleet.
- The track mission updated after each track choice.
- Student-facing page text no longer exposed `MLSysIM`, `Source Trace`, `source trace`, or `Evidence Summary`.
- Student-facing setup no longer showed `Hardware source`, `System source`, or `Model source`.
- Layout metrics stayed clean: zero horizontal overflow and zero offscreen panels.
- Parts A-E and synthesis include a scenario thread and a decision path.

Second-pass checks:

- A stricter learner audit switched all four tracks, returned to iPhone, opened Parts A-E and synthesis, clicked the gated answer in each part, and captured screenshots under `/tmp/mlsysbook-lab10-learner-loop-v2/`.
- The stricter audit found that source-like labels were still visible in the track/scenario setup. Those labels were removed from the student-facing flow and replaced with role, metric, guardrail, dominant constraint, model under pressure, first decision, and validation question.
- The audit also showed that part-level scenario threads were useful but too wordy. The threads were shortened so they function as reminders rather than a second scenario brief.

## Simulated Instructor Feedback

### ML systems instructor

Verdict: stronger than the previous version because the lab now separates preparation, mission, case, and work. The green track/case convention is useful because students can scan where the deployment context starts.

Requested improvements:

- Keep visible provenance out of the launch path. It distracts from the learning objective.
- Keep source/provenance inside reports or debug views only.
- Make each part begin by reminding students which deployment decision they are making.
- Do not rely on a Mermaid renderer for the main learning path.

Applied:

- Removed visible source/provenance labels from Lab 10 setup.
- Added `scenario_thread()` and `decision_flow()` reusable helpers.
- Added those helpers to Parts A-E and synthesis.
- Removed visible hardware/model/system source labels from the track and scenario panels.

### Teaching assistant

Verdict: the tabbed structure works, but students need a compact mental model before each part. The decision path helps TAs explain what the student is supposed to do before they click controls.

Requested improvements:

- Shorten part-level scenario reminders.
- Keep the track consequence close to the prediction/control.
- Make report and takeaways visually distinct so students know the exercise has ended.

Applied:

- Tightened the Part A-E scenario-thread copy.
- Added blue treatment for takeaways and amber treatment for report export.
- Kept the visual path renderer-independent instead of adding Mermaid as a dependency.

## Simulated Learner Feedback

### Strong learner

Reaction: understands that compression is not one trick. The part flow makes it clear that quantization, pruning, Pareto selection, energy, and distillation are different tests of the same deployment choice.

Remaining friction:

- Track selector is still visually lighter than the green track mission card.
- Long charts can make the part feel large after the first answer.

### Struggling learner

Reaction: the color coding helps. The decision path gives a checklist before the question.

Remaining friction:

- The report fields still require judgment; students may need examples of acceptable reflection answers in future labs.
- If a student jumps straight to a tab, they may miss the selected track context unless each part repeats it.

Applied:

- Each part now repeats the selected track in the scenario thread.
- Synthesis repeats the expected metric, guardrail, and residual-risk structure.

## Current Acceptance Judgment

Lab 10 is ready as a pilot for the track-first visual flow. It is not the final part-body pattern for every lab, but the direction is strong enough to propagate deliberately:

- Intro/preparation: blue.
- Track mission, case, lab map, and instructions: green.
- Part body: scenario thread, concept, decision path, prediction/control, evidence, reflection.
- Takeaways: blue.
- Download report: amber.

## Verification

- `python3 -m py_compile labs/mlsysbook_labs/ui.py labs/mlsysbook_labs/__init__.py labs/vol1/lab_10_model_compress.py labs/tests/test_ui_helpers.py`
- `python3 -m pytest labs/tests/test_ui_helpers.py labs/tests/test_static.py labs/tests/test_report_contract.py -q`
- `python3 labs/tools/interaction_lab_smoke.py --labs labs/vol1/lab_10_model_compress.py --html-pages labs/lab-plan-dashboard.html --port-start 31200 --output-dir /tmp/mlsysbook-lab10-final-audit --max-radios 0 > /tmp/mlsysbook-lab10-final-audit/results.json`
- `python3 labs/tools/interaction_lab_smoke.py --port-start 31300 --output-dir /tmp/mlsysbook-style-final-all --max-radios 0 > /tmp/mlsysbook-style-final-all/results.json`
- `python3 labs/tools/interaction_lab_smoke.py --labs labs/vol1/lab_10_model_compress.py --html-pages labs/lab-plan-dashboard.html --port-start 31400 --output-dir /tmp/mlsysbook-lab10-feedback-final-audit --max-radios 0 > /tmp/mlsysbook-lab10-feedback-final-audit/results.json`
- `python3 labs/tools/interaction_lab_smoke.py --port-start 31500 --output-dir /tmp/mlsysbook-feedback-loop-full --max-radios 0 > /tmp/mlsysbook-feedback-loop-full/results.json`

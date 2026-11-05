# Reference notebook template

Lab 02 is the visual and interaction prototype for this template. Lab 00 and
Lab 01 use the same opening and outer frame. Further chapter implementation
should follow this template after its rendered experience has been reviewed;
older notebooks are not interchangeable visual examples.

## The structure students learn once

Show the destination and route before the first experiment. Use the same visible labels in each lab:

- **Your scenario:** a short scene from the continuing project, the decision due today, and a photograph that makes the setting tangible. Use the shared `notebook_scenario(...)` helper after the title and project selector, before the learning goals. Reuse the MLSysIM episode or canonical lab narrative; do not maintain a second scene in the notebook.
- **What you will learn:** a short set of observable chapter-owned goals, expressed as things students can compare, explain, or defend. Name the ability without announcing the experimental finding.
- **Your route through this lab:** every part's title, question, and contribution to the decision; mark the current part. Keep later parts visible even when their instruments are not yet available.
- **What you will hand in:** one integrated discussion and its required evidence; explain what a supported submission contains.
- **What comes next:** the next part and next lab question, and what evidence carries forward. A preview is not a claim that the student has already established that evidence.

Reveal consequences and deeper explanations progressively, not the learning goals, route, submission requirements, or destination. Keep section headings and short previews visible from the start. Unavailable controls should state what action opens them. Optional background can collapse; the core learning goals cannot.

Within each investigation, use a narrative learning cycle:

**Scene → short briefing → meaningful choice → consequence feedback → explanation → handoff.**

Give students enough to understand the task, terms, and starting conditions
before acting. The concept map's learning claim and misconception are authoring
guidance; avoid revealing their resolution in the opening. After a choice, show
what actually changed, the competing consequence, and any failure or unknown.
Invite an explanation and a further trial, then connect the result to the next
decision. Feedback is more than a verdict: its content depends on the saved
comparison and remains available after an incorrect prediction. The shared
[narrative contract](vol1/concepts/schema.yml) defines progressive disclosure
and feedback requirements.

Use the same sequence in every chapter lab:

**Project → Your scenario → Learning goals and route → Investigate → Discuss and submit → Next.**

The opening identifies the chosen system, gives the catalog-owned episode
event, and states the chapter's observable goal. Use `notebook_bridge(...)`
for the familiar connection strip. “Previously” describes the project thread;
only verified compatible records may be described as prior supporting evidence.
Otherwise identify the supplied starting case. Orientation uses Start / Goal /
Next and rehearses the interfaces instead of making a deployment claim.

Keep a short core investigation and optional depth in the same places. Let
students choose a useful contrast before opening “Help me design a comparison.”
Its Look at / Compare / Explain / Challenge entries offer attention, a route,
the mechanism, and a transfer probe. Numeric outcomes, failed gates, units, and
evidence limitations remain visible without requesting help.

The closing team decision uses saved compatible evidence. It names the claim or
hold, rejected alternative, unresolved obligation, and reversal condition.
Finish with visible Carry forward / Still open / Next labels; the next chapter
question must be as recognizable as the opening goal. A complete discussion is
not automatic mastery or permission to deploy.

The filled interaction and acceptance contract is in
[the Volume I engagement specification](plans/vol1/engagement_spec.qmd).


The opening tells students what just happened and why a decision is needed now.
Use natural language such as “Save your progress” and “Download your work.”
Keep internal payload names, file schemas, and storage implementation details
out of the story. Technical concepts such as a belief state or data lineage
remain precise where the chapter teaches them.

Use the shared packaged project photograph, with meaningful alt text and a
short caption. Keep source, license, and image modifications in the photo credit.
The photograph illustrates the project setting; it is not a measurement of the
named product. Images must work without a separate image-host request.

Every lab ends with the H2 **Discuss and submit**, rendered by
`notebook_submission(...)` outside completion gates. Explain how to use the
actual download controls and submit through the course system. Provide a
readable discussion alongside structured evidence. Orientation uses
`orientation=True` and saves the project choice and practice work. Do not
promise automatic submission, file import, or a dedicated print interface.

## Design the lesson first

Read the chapter and its concept map. State the core claims, the consequential
choices students will explore, what stays fixed, the observations they need,
and the discussion they will submit. Choose instruments for those purposes.
The chapter determines the experiment; the shared template organizes it.

For Volume I, the [shared lab learning contract](vol1/concepts/schema.yml)
anchors this work in exploring tradeoffs. Every notebook has a matching
`vol1/concepts/<notebook_stem>_concepts.yml` file. Its distilled concepts name
the chapter source and owner, learning claim, misconception, tradeoff, student
action, evidence view, and explanation requirement. These are authoring and
assessment fields; they do not require separate student essays.

Bind those concepts to the selected MLSysIM mission's chapter episode and
`engine/data/deployment_learning/<track>.json` investigations. Preserve the
same conceptual goal across tracks while changing the task, meaningful choices,
observations, and consequences to fit the actual project. The
[complete paradigm guides](plans/vol1/paradigm_learning.qmd) write out those
bindings. Run `python labs/tools/check_lab_concepts.py` and the narrative
generator's `--check` before extending a notebook. Consistency checks do not
establish numerical fidelity or student learning.

## Follow one deployment project through the series

Orientation selects a deployment project within TinyML, MobileML, EdgeML, or
CloudML. Show its user, input, intended outcome, deployed endpoint, and unresolved
requirements. Obtain the versioned mission and chapter episode from MLSysIM;
do not replace it with an unrelated scenario when the next notebook opens.
Keep the category visible so a memorable project does not erase the hierarchy.

Each chapter is an episode in that project. State the decision due now, what
earlier work contributes, and what new evidence this chapter adds. A wearable
can use a separate development host for training; a cloud project stays within
the single-endpoint scope of this volume. State that boundary in the mission.
Controlled teaching fixtures are not measurements of the project or a named
commercial product. Never disguise a generic fixture as product validation.

Persist project identity separately from experimental evidence. The deployment
dossier accumulates scoped decisions and unresolved obligations. A previous
passing experiment is reusable only when its task, artifact, population, and
measurement conditions match. A missing earlier record means a labeled
standalone start, not a fabricated history. Comparing another track must not
relabel earlier saved results.

## One frame and a compact opening

Use `mlsysbook_labs.notebook` for explicit layout containers. Apply
`notebook_shell(...)` to the opening, tabbed investigations, discussion export,
and status. They share a centered maximum width and the same edges. Let prose
use a shorter reading measure inside that frame. Do not make a full-window
hero above a narrow worksheet or match CSS against an inline background string.

The opening contains a small lab/chapter label, title, one driving question,
short concept labels, and context selector. Core learning goals, the route, deliverable, and next-step preview remain visible.
Supporting reading, detailed background, and assumptions can use an accordion. The first meaningful action should
appear without scrolling through a second chapter of prose. Orientation is a
short interface rehearsal; it does not need technical instruments or a physics
failure to look like the chapter labs.

```python
from mlsysbook_labs.notebook import (
    notebook_header, notebook_shell, notebook_part, notebook_workspace,
    notebook_cards, notebook_progress, notebook_status, notebook_table,
)

notebook_shell([
    notebook_header(metadata, driving_question, topics=core_concept_labels),
    context_control,
    mo.accordion({"Reading and assumptions": reading_and_assumptions}),
])
```

These helpers own presentation only. Physical quantities, scenario values,
model behavior, and sweep points come from MLSysIM. Use its formatters for the
returned quantities. Leave legacy global CSS out of these opt-in reference
notebooks so its width heuristics do not compete with the explicit frame.

## Predict, investigate, and retain evidence

Use a structured prediction before revealing instruments. Any submitted
prediction unlocks exploration. Once committed, replace the tall form with a
concise prediction and observation; keep the original prediction in the saved
comparison. Correctness must not gate exploration or completion.

Within a part, use `notebook_workspace(controls, results)` to keep controls
beside the linked instrument on a wide screen and above it on a narrow screen.
Give the learner a clear question and a consequential action. Keep action
buttons adjacent and in order: pin a run, change a choice, save a contrast.
Place status and a short next-step hint nearby.

A prediction is only the opening question. Each part needs a connected
investigation: diagnose the baseline, design a purposeful comparison, test a
boundary or an ineffective remedy, explain the mechanism, and consider a
changed condition. Choose the moves the chapter claim needs; do not fill a
question quota. Titles and prompts must not announce the answer before the
prediction. Keep deeper probes optional when they would overload the core.

Collect a short causal interpretation tied to the saved comparison. When a
comparison changes, require the learner to revisit its interpretation instead
of silently attaching old prose to new evidence. Transfer claims remain
predictions until the student actually runs the changed condition. A sequence
of green checks or any two different settings is not evidence of understanding.

Use the same state vocabulary throughout:

- **Live experiment:** follows the controls and can change.
- **Pinned comparison:** retains a run while the learner tries another choice.
- **Saved evidence:** an immutable comparison consumed by later decisions.
- **Stale evidence:** retained history whose upstream assumptions have changed.

Use reactive notebook capabilities where they clarify the concept: linked
charts, current and pinned markers, changing budget boundaries, capacity views,
dependency traces, and evidence-dependent downstream experiments. Editable code
or chart selection is optional; add it only when it supports a learning goal.
Do not make a table of calculated answers the default instrument for every part.

The Lab 02 prototype demonstrates these choices:

- Locality uses a linked latency decomposition and an ideal-processor contrast.
- Resources show required versus available capacity before timing; an
  overflowed configuration has no executable timing bar.
- Operation uses sourced energy and power curves, an observation boundary,
  current/pinned markers, and separate service, energy, and power verdicts.
- Placement shows the supplied gates in a compact matrix. Excluded candidates
  remain visible, and unavailable evidence stays unresolved.

## Keep the important result visible

Show the active constraint or evidence gap beside the result, with text as well
as color. Use `notebook_cards(...)` for a small set of consequential outcomes.
Use a single concise prediction/observation comparison instead of repeating
several full-width information banners.

Keep Calculation Notes, exact data rows, and deeper explanations expandable.
Small read-only evidence tables use `notebook_table(...)`; use `mo.ui.table`
when sorting, selecting, or inspecting a larger dataset is part of the task.
Avoid search, column management, and per-table downloads when they do not help
the investigation. The discussion export already carries the evidence.

## Make discussion and saving deliberate

The discussion workspace summarizes saved and stale evidence, provides usable
writing fields, and asks for the decision, mechanism, rejected alternative,
uncertainty, and revisit condition. It should remain possible to justify a hold
or retain the baseline.

Generate the downloadable discussion from valid saved comparisons and the
student's text. Keep download readiness separate from persistent-save status.
An explicit **Save discussion** action records the completed discussion;
editing afterward must require saving again. Do not silently mark every draft
keystroke as a final saved discussion. On a storage error, keep the report
available and show an honest unavailable state.

Keep the footer quiet and aligned with the frame. Show the context and actual
evidence/save state. Internal API names, schema versions, and implementation
provenance belong in source and exported metadata, not the opening banner.

## Review the real notebook before extending the template

Inspect native Marimo and the exported browser notebook. Check a wide desktop,
laptop, tablet, and narrow phone viewport. Check the locked and unlocked states,
not only an initial screenshot.

Verify that the opening, context, tabs, instruments, report, and footer align;
controls remain associated with their results when columns stack; and labels,
legends, tables, and tabs do not clip. Check that a first-time student can reach
the first prediction with reading details collapsed. Inspect keyboard use and
mobile scrolling rather than assuming an absence of document overflow proves
usability.

Exercise a wrong prediction, pin and change controls, save a contrast, invalidate
an upstream choice, generate the discussion, save deliberately, edit again,
download, reload, and simulate unavailable storage. Rebuild the browser wheels
after helper or model changes. Source checks, browser checks, and student pilots
answer different questions; report them separately.

Use simulated student, instructor, and classroom walkthroughs to identify
likely problems, revise, and review the actual revision again. Label their
observations as simulated. Real student feedback requires an actual pilot:
observe independent use, ask learners to explain a saved comparison and a new
case, review the exported reasoning, and record confusing controls and stalls.
Do not claim engagement, completion time, or learning gains from agent reviews.

Run the relevant notebook reference tests and the static/style checks, then the
lab suite after integration:

```bash
python3 -m pytest labs/tests/test_placement_reference.py labs/tests/test_static.py
python3 labs/tools/check_browser_wheels.py
python3 -m pytest labs/tests
```

For Lab 02, run `labs/tools/check_placement_discussion.py --url URL` against
the running native or exported notebook. It exercises actual reactive callbacks:
attach, edit, recapture, reattach, download, explicitly save, and edit again.
Headless `app.run()` checks do not exercise the same event loop. A browser
walkthrough caught an attachment state bug that the reference tests missed.


### Track selection and reactive evidence

Maintain one notebook per chapter. A track selects its canonical MLSysIM project narrative, workload, resource profile, constraints, and supported experiments together. TinyML, MobileML, EdgeML, and CloudML share the chapter concept set and investigation components. A workload case, such as a larger model, is a controlled variation within that track, not a separate notebook or a substitute for a distinct deployment context.

After a prediction, changing an experiment control updates live evidence and consequence feedback immediately. Reserve explicit actions for pinning a comparison, recording a decision, and downloading work. Keep saved evidence distinct from live evidence. A track change must not attach the previous track's notes or decisions to the new project.

Use the volume's book accent for headings, links, and navigation; retain semantic resource and outcome colors for instruments. The published lab should retain shared course navigation, a return-to-chapter link, and previous/next lab links around a compact notebook workspace. A downloadable standalone notebook should remain usable without that site shell.


### Shared course shell

The laboratory index uses the shared Quarto navigation and ecosystem styles. Each lab opens as a full-page Marimo notebook, without an iframe or nested course page. The build reuses the actual Quarto header and compiled theme from the landing page: shared shield, brand, ecosystem menus, and responsive navigation. A quiet contextual row adds the lab index, volume/chapter context, reading link, and previous/next labs from the canonical learning catalog. There is no separately authored notebook masthead. Quarto styles are scoped to the header so they cannot change Marimo widgets. Preserve the book's fonts and volume accents; reserve the remaining viewport for the notebook. Keep notebook CSS scoped so it cannot restyle the course navigation.

The reference example demonstrates this shell around an interactive investigation. Production Marimo integration must preserve this shell around the notebook workspace; a standalone downloaded notebook can omit site navigation.

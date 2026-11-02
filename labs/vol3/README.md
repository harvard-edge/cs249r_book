# Volume III agentic labs

Lab 00 rehearses the interface; Labs 01–20 follow the canonical chapter order
and the repair project from a task contract to an agent-system dossier.
Research, remediation, and support are bounded transfer lighthouses from the
book, not substitute deployment tiers.

Each lab has four concept investigations, visible chapter learning goals,
a supplied starting point, one integrated discussion, and an onward question.
The shared `mlsysbook_labs.agentic_companion` module owns interaction behavior.
`mlsysim.labs.experiments.v3_studies` owns study definitions, project identities,
policy effects, and evidence records. Notebooks contain reactive wiring rather
than copied scenario prose or local calculations.

## Evidence boundary

Concept maps specify intended chapter coverage. Implemented studies have
narrower explicit scope: controlled supplied traces, guarded policy replays,
and selected analytical instruments. A trace does not establish production
reliability, statistical significance, model quality, or deployment readiness.
Read each study's scope and unmodeled requirements. Some concepts still need
richer instruments; appearing in a roadmap does not make a concept operational.

Wrong predictions unlock the same study. Policy changes update live evidence
immediately. Saving retains an immutable comparison and interpretation in the
session. Changing a prediction, policy, or interpretation makes the old record
historical. Track changes cannot relabel evidence or decision text. Reports
list required, completed, and missing parts. Explicit saving is separate from
downloading; session saving is not durable storage.

## Run and review

From the repository root:

```bash
marimo run labs/vol3/lab_00_orientation.py
marimo check labs/vol3/*.py
PYTHONPATH=mlsysim:labs python3 -m pytest labs/tests/test_agentic_companion.py
```

Chapter-grounded authoring specifications live in local `labs/plans/vol3/`.
Independent agent critiques are simulated expert reviews, not observed student
feedback. A classroom pilot remains necessary to assess learning and usability.

# Lab Single Source Of Truth Policy

This policy governs all MLSysBook lab implementation work.

## Core Rule

Labs do not own system facts. Labs present scenarios, controls, visuals, reflections, and reports. Quantitative facts and reusable models must live in MLSysIM or in a typed `mlsysbook_labs` metadata layer that points back to MLSysIM.

## Ownership Boundaries

| Information type | Owner | Examples |
|---|---|---|
| Hardware facts | MLSysIM hardware registry | memory, SRAM, flash, battery, TDP, bandwidth, storage, dispatch tax |
| Model facts | MLSysIM model registry | parameters, FLOPs, architecture family, default precision, model size |
| Infrastructure facts | MLSysIM infrastructure/systems registry | fleet topology, accelerator count, datacenter region, PUE, cost assumptions |
| Physics and solver equations | MLSysIM physics/solver APIs | roofline, queueing, compression, energy, carbon, reliability, placement |
| Track identity | `mlsysbook_labs` track profile registry | iPhone, Oura Ring, RoboTaxi, Cloud Fleet |
| Lab scenario defaults | `mlsysbook_labs` lab variant registry | stakeholder, workload, SLO, guardrail, report target |
| Notebook UI state | Notebook/lab layer | selected tab, current slider value, expanded source trace |
| Student evidence | Design Ledger/report layer | prediction, selected candidate, rationale, residual risk |

## Prohibited Patterns

Do not add notebook-local constants for:

- SRAM, flash, memory, storage, or battery capacity.
- TDP, power budget, energy per inference, or thermal limit.
- Hardware peak compute, memory bandwidth, or dispatch tax.
- Model parameters, FLOPs, default model size, or architecture constants.
- Datacenter carbon intensity, PUE, accelerator price, or fleet topology.
- Common solver formulas that should be reusable across labs.

If a lab needs one of these values, add it to MLSysIM first with provenance, then consume it through a registry or solver API.

## Allowed Lab-Local Values

Lab-local values are allowed only when they are clearly scenario choices, not system facts:

- A stakeholder's target, such as "p99 latency must be below 10 ms."
- A pedagogical sweep range, such as showing bit width from 2 to 16.
- A default UI value, if it points to a typed scenario field.
- A report prompt or rubric phrase.

Even these should move into `mlsysbook_labs` track/lab variant metadata once reused by more than one lab.

## Provenance Requirement

Every hardware, model, infrastructure, or empirical assumption must have one of:

- `datasheet`
- `benchmark`
- `paper`
- `vendor documentation`
- `estimate`
- `convention`

Estimates and conventions must be labeled in source traces and exported reports.

## Implementation Notes Requirement

Every implementation pass must update `labs/LAB_IMPLEMENTATION_NOTES.md` with:

- Lab touched.
- Track(s) touched.
- New MLSysIM facts or APIs needed.
- Constants removed from notebook-local code.
- Plan lessons that should be propagated to other labs.
- Follow-up work.

## Structure And Report Requirement

Every lab implementation must also satisfy `labs/LAB_STRUCTURE_AND_REPORT_CONTRACT.md`.

In particular:

- Every part needs a small "What You Need To Know" section before controls.
- Every part needs prediction, evidence, source trace, reflection, and checkpoint/decision.
- Every lab ends with big takeaways and a local-first report download.
- Report generation must work without a hosted backend.

## Track Profiles

The canonical student-facing tracks are:

| Track | Category | Required hardware source |
|---|---|---|
| iPhone | Mobile ML | `Hardware.Mobile.iPhone15Pro` |
| Oura Ring | TinyML / wearable | `Hardware.Tiny.OuraRing` |
| RoboTaxi | Edge AI | `Hardware.Edge.RoboTaxi` |
| Cloud Fleet | Cloud/Fleet | Cloud fleet profile backed initially by `Hardware.Cloud.H100` |

Extra devices can exist in MLSysIM for comparisons, but they are not primary student tracks until a lab variant and report contract support them.

## Review Checklist

Before a lab implementation is considered complete:

- Hardware facts are read from MLSysIM.
- Model facts are read from MLSysIM.
- Reused formulas are in MLSysIM solvers/physics.
- Scenario thresholds are in typed track/lab variant metadata.
- The source trace names the APIs and assumptions.
- The report includes track, hardware ref, scenario, prediction, result, decision, and residual risk.
- Implementation notes record any lesson that should improve other labs.

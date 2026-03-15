# Lab Review Prompt — Educational Quality & mlsysim Integration Audit

Use this prompt with the `lab-designer` or `education-reviewer` agent to systematically audit each Marimo lab for educational quality and mlsysim integration completeness.

---

## How to Use

Pass this prompt (or reference this file) when reviewing any lab. Replace `{LAB_FILE}` with the actual lab path:

```
Review the lab at {LAB_FILE} using the full audit checklist in labs/REVIEW_PROMPT.md.
Produce a structured YAML report with pass/fail/warning for each gate.
```

---

## The Review Prompt

You are an expert educational reviewer and ML systems engineer auditing a Marimo interactive lab for the *Machine Learning Systems* textbook. Your job is to evaluate the lab against two dimensions:

1. **Pedagogical Quality** — Does this lab create productive failure, build intuition, and follow the 2-Act protocol?
2. **mlsysim Integration** — Does this lab use the analytical simulator's solvers, registries, and result types instead of hard-coding physics?

Read the lab file completely. Then evaluate against every gate below. For each gate, report: `PASS`, `FAIL`, or `WARNING` with a one-line justification.

---

### DIMENSION 1: Pedagogical Quality (14 Gates)

#### Structure & Timing

- **P1 — 2-Act Structure**: Lab has exactly 2 acts. Act I = Calibration (12–15 min). Act II = Design Challenge (20–25 min). Total ≤ 40 min. No 3-act labs.
- **P2 — 22-Cell Template**: Cells follow the TEMPLATE.md zones: A (Opening, Cells 0–4), B (Act I, Cells 5–11), C (Act II, Cells 12–19), D (Closing, Cells 20–21). Order is preserved even if some cells are merged.
- **P3 — hide_code=True**: All cells except Cell 0 (SETUP) have `hide_code=True`. Students see outputs, not implementation.

#### Predictions & Learning Moments

- **P4 — Structured Predictions Only**: Every prediction uses `mo.ui.radio()` with exactly 4 options OR `mo.ui.number()` with bounded range. No free-text input fields. No `mo.ui.text()` for predictions.
- **P5 — Prediction Gates**: `mo.stop()` blocks downstream instruments until prediction is committed. Gate displays a `mo.callout(..., kind="warn")` message.
- **P6 — Prediction Overlay**: After each act, the lab shows: "You predicted X. The actual value is Y. You were off by Z×." This is the core learning moment.
- **P7 — MathPeek Accordion**: Each act has a `mo.accordion()` revealing the governing equation with labeled components. Formula matches the chapter's notation exactly.

#### Failure States & Constraints

- **P8 — At Least One Failure State**: Act II has at least one instrument that turns red / shows a danger banner when a physical constraint is violated (OOM, SLA violation, thermal throttle, negative ROI).
- **P9 — Reversible Failure**: The failure state is reversible — students can adjust sliders and watch the system recover. Failure teaches the boundary, not punishes.
- **P10 — Constraint Variety**: Check if the lab uses the appropriate failure type for its domain (memory wall → OOM, serving → SLA violation, sustainability → carbon budget, economics → ROI).

#### Context & Continuity

- **P11 — 2 Deployment Contexts**: Lab offers exactly 2 contexts via `mo.ui.radio()` toggle (e.g., Cloud vs Edge, H100 vs Mobile). Not 4 tracks. Context changes hardware constants, not lab structure.
- **P12 — Design Ledger Integration**: Lab calls `ledger.save()` with chapter number, context, predictions, results, and constraint hits. Later labs can read this state.
- **P13 — Stakeholder Framing**: Each act opens with a stakeholder message (named persona, urgent scenario) that motivates the constraint exploration.

#### Traceability

- **P14 — Every Number Has a Source**: Every hardware constant, slider range, threshold, and chart value has a comment citing its source (`# H100 SXM5 HBM3e, NVIDIA spec` or `# @sec-hw-acceleration`). No magic numbers.

---

### DIMENSION 2: mlsysim Integration (12 Gates)

The architecture document (ARCHITECTURE.md) specifies which solvers each lab should use. Check whether the lab actually calls them vs. hard-coding the math.

#### Registry Usage

- **M1 — Hardware from Registry**: Hardware specs come from `mlsysim.Hardware.*` (e.g., `Hardware.Cloud.H100`, `Hardware.Tiny.ESP32`) instead of locally defined constants like `H100_BW_GBS = 3350`.
- **M2 — Models from Registry**: Workload definitions come from `mlsysim.Models.*` (e.g., `Models.Llama3_8B`, `Models.ResNet50`) instead of manual parameter definitions.
- **M3 — Infrastructure from Registry**: Grid profiles come from `mlsysim.Infra.*` (e.g., `Infra.Quebec`, `Infra.Poland`) instead of hard-coded carbon intensity values.

#### Solver Usage (per ARCHITECTURE.md mapping)

- **M4 — Correct Solvers Used**: The lab imports and calls the solvers specified in ARCHITECTURE.md's "Solvers" column for this lab number. List expected vs. actual.
- **M5 — Typed Results**: Solver results use mlsysim's typed Pydantic result objects (e.g., `PerformanceProfile`, `ServingResult`, `SustainabilityResult`) instead of raw dicts or manual calculations.
- **M6 — Engine.solve() Pattern**: Single-node analysis uses `Engine.solve(workload, hardware)` → `PerformanceProfile` rather than manual Roofline math.

#### Advanced Capabilities

- **M7 — Tier 2 Solvers (Diagnostics)**: Labs that teach sensitivity analysis (V1-12, V1-16, V2-09, V2-17) use `SensitivitySolver` or `SynthesisSolver` instead of manual perturbation.
- **M8 — Tier 3 Optimizers (Design Space Search)**: Labs with design challenges use `ParallelismOptimizer`, `BatchingOptimizer`, or `PlacementOptimizer` where applicable, instead of manual slider sweeps.
- **M9 — Scenario.evaluate()**: Labs that compare workload-on-hardware use the `Scenario` API with `.evaluate()` → `SystemEvaluation` scorecard, not manual feasibility checks.
- **M10 — Formulas Module**: Physics calculations use `mlsysim.core.formulas.*` (e.g., `calc_ring_allreduce_time()`, `calc_kv_cache_size()`, `calc_young_daly_interval()`) instead of inline math.

#### Visualization & Presentation

- **M11 — Dashboard Components**: Labs use `mlsysim.viz.dashboard.*` components (`command_header`, `lever_panel`, `telemetry_panel`, `audit_panel`) for consistent layout, OR document why custom layout is preferred.
- **M12 — Plot Helpers**: Roofline plots use `mlsysim.viz.plots.plot_roofline()` or equivalent instead of building Plotly charts from scratch.

---

### DIMENSION 3: Cross-Lab Coherence (4 Gates)

- **C1 — Instrument Progression**: Lab does NOT introduce instruments before their chapter. Check against LABS_SPEC.md Rule 6 instrument introduction table.
- **C2 — Forward Dependencies**: If this lab reads from Design Ledger state saved by a prior lab, verify the schema matches (e.g., Lab 08 reading Lab 05's activation choice).
- **C3 — Wall Coverage**: The lab exercises the walls listed in ARCHITECTURE.md. List expected walls vs. actually exercised.
- **C4 — Constants Consistency**: Hard-coded constants match `book/quarto/mlsys/constants.py` and `mlsysim.core.constants`. Flag any drift.

---

## Output Format

Produce a structured report per lab:

```yaml
lab: lab_XX_name.py
volume: 1 or 2
reviewer: [agent name]
date: YYYY-MM-DD

pedagogical_quality:
  P1_2act_structure: {status: PASS|FAIL|WARNING, note: "..."}
  P2_22cell_template: {status: PASS|FAIL|WARNING, note: "..."}
  P3_hide_code: {status: PASS|FAIL|WARNING, note: "..."}
  P4_structured_predictions: {status: PASS|FAIL|WARNING, note: "..."}
  P5_prediction_gates: {status: PASS|FAIL|WARNING, note: "..."}
  P6_prediction_overlay: {status: PASS|FAIL|WARNING, note: "..."}
  P7_mathpeek: {status: PASS|FAIL|WARNING, note: "..."}
  P8_failure_state: {status: PASS|FAIL|WARNING, note: "..."}
  P9_reversible_failure: {status: PASS|FAIL|WARNING, note: "..."}
  P10_constraint_variety: {status: PASS|FAIL|WARNING, note: "..."}
  P11_2_contexts: {status: PASS|FAIL|WARNING, note: "..."}
  P12_design_ledger: {status: PASS|FAIL|WARNING, note: "..."}
  P13_stakeholder_framing: {status: PASS|FAIL|WARNING, note: "..."}
  P14_number_traceability: {status: PASS|FAIL|WARNING, note: "..."}

mlsysim_integration:
  M1_hardware_registry: {status: PASS|FAIL|WARNING, note: "..."}
  M2_model_registry: {status: PASS|FAIL|WARNING, note: "..."}
  M3_infra_registry: {status: PASS|FAIL|WARNING, note: "..."}
  M4_correct_solvers: {status: PASS|FAIL|WARNING, expected: [...], actual: [...]}
  M5_typed_results: {status: PASS|FAIL|WARNING, note: "..."}
  M6_engine_solve: {status: PASS|FAIL|WARNING, note: "..."}
  M7_tier2_solvers: {status: PASS|FAIL|WARNING, note: "..."}
  M8_tier3_optimizers: {status: PASS|FAIL|WARNING, note: "..."}
  M9_scenario_evaluate: {status: PASS|FAIL|WARNING, note: "..."}
  M10_formulas_module: {status: PASS|FAIL|WARNING, note: "..."}
  M11_dashboard_components: {status: PASS|FAIL|WARNING, note: "..."}
  M12_plot_helpers: {status: PASS|FAIL|WARNING, note: "..."}

cross_lab_coherence:
  C1_instrument_progression: {status: PASS|FAIL|WARNING, note: "..."}
  C2_forward_dependencies: {status: PASS|FAIL|WARNING, note: "..."}
  C3_wall_coverage: {status: PASS|FAIL|WARNING, expected: [...], actual: [...]}
  C4_constants_consistency: {status: PASS|FAIL|WARNING, drifts: [...]}

summary:
  pedagogical_score: X/14
  mlsysim_score: X/12
  coherence_score: X/4
  total: X/30
  priority_fixes:
    - "[P|M|C]X: description of highest-priority fix"
    - "..."
  mlsysim_integration_plan:
    - "Replace hard-coded H100_BW_GBS with Hardware.Cloud.H100.memory.bandwidth"
    - "Replace manual Roofline math with Engine.solve()"
    - "..."
```

---

## Batch Review Command

To audit all labs in a volume:

```
For each lab in labs/vol1/*.py (excluding __init__.py and __pycache__),
run the full 30-gate audit from labs/REVIEW_PROMPT.md.
Produce individual YAML reports in .claude/_reviews/labs/vol1/
and a summary table ranking labs by total score.
```

---

## Priority Matrix

When triaging fixes across labs, use this priority order:

1. **P8 (Failure States)** — Labs without failure states fail the core pedagogical mission
2. **M4 (Correct Solvers)** — Labs should call the solvers ARCHITECTURE.md specifies
3. **P6 (Prediction Overlay)** — Missing overlays eliminate the primary learning moment
4. **M1/M2 (Registries)** — Constants drift is a silent correctness bug
5. **C4 (Constants Consistency)** — Hard-coded values must match single source of truth
6. **P14 (Traceability)** — Uncommented magic numbers undermine trustworthiness
7. **M10 (Formulas Module)** — Inline physics duplicates solver logic and drifts
8. **Everything else** — Important but lower blast radius

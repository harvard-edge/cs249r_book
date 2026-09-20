# Lab 1 teaching design: diagnose before optimizing

## Main point

An ML system must produce useful predictions within its deployment constraints.
Students should learn to identify what prevents that outcome, compare changes
against the same baseline, and revisit their decision when conditions change.

The textbook develops the principles; TinyTorch develops implementation skill.
This lab develops judgment through experiments with competing requirements.
Successful completion means defending a decision with comparative evidence. It
does not require making every simulated system feasible.

## Chapter alignment

The source is Volume I's Introduction. Its sections establish the following
experimental obligations:

| Chapter section | Claim to investigate | Lab evidence |
|---|---|---|
| Data-Centric Paradigm Shift; ML vs. Traditional Software | Learned behavior can fail with unchanged code and model weights. | Fixed-model quality under two population mixes. |
| Defining ML Systems | Diagnosis separates data, algorithm, and machine causes; relaxing a suspected constraint tests the hypothesis. | A controlled change with before/after outcomes. |
| Iron Law of ML Systems | Sequential execution time includes movement, computation, and overhead; improving one component does not proportionally improve the whole. | Component time bars and end-to-end speedup. |
| Return on compute | Additional model quality must justify additional resource cost. | Candidate quality alongside memory, time, and energy requirements. |
| Deployment context shapes the lifecycle | The operating envelope changes which design is acceptable. | A feasible choice, a failed alternative, and an explicit requirement. |
| Engineering across the ML lifecycle | Deployment produces evidence that can invalidate an earlier decision. | A changed-condition experiment and a reevaluation trigger. |

The historical narrative and the bitter lesson motivate the scaling question in
Part C. Students do not need a separate history exercise. The five pillars guide
the final judgment; they are not five extra dashboards.

## Student sequence

Target approximately 50–55 minutes, including orientation and synthesis. Timing
is a design estimate that requires a student pilot.

### A. Can behavior fail without a code change?

Hold the model and machine fixed. Predict which outcome changes when the
production population differs from the evaluation population. Compare a baseline
and a changed population. Identify the quality requirement that fails and which
additional evidence would distinguish poor coverage from inadequate model
capacity.

The investigation must show the population assumption. A distribution change is
not automatically a quality failure. The modeled outcomes determine whether the
quality requirement is crossed. Monitoring observes outcomes; it does not repair
the model.

**Record:** original prediction, population settings, two quality outcomes, and
the requirement used to judge them.

### B. Which improvement makes the whole system faster?

Predict the end-to-end consequence of doubling compute capability. Compare this
with improving data movement while the task remains fixed. Display movement,
computation, and overhead in a stacked time bar. Identify the dominant term
before and after the intervention.

Use the chapter's sequential model:

`time = bytes moved / bandwidth + operations / effective compute rate + overhead`

The calculation notes must state the no-overlap assumption. Improving a smaller
term can produce a smaller, nonzero benefit. Do not teach a universal
`min(Data, Algorithm, Machine)` performance law.

**Record:** predicted speedup, baseline time, changed time, actual speedup, and a
less useful alternative under these conditions.

### C. Is the higher-quality model deployable?

Compare candidates on the same task and population. A higher-quality candidate
must incur an explicit resource cost. Inspect quality, memory, time, and energy
against requirements. Find both a feasible candidate and an attractive candidate
that fails a requirement, where the scenario permits them.

Offline quality evidence alone does not establish deployment readiness. If no
candidate satisfies all requirements, that is a valid result to carry into Part D.

**Record:** candidate comparison, selected candidate or no-feasible-candidate
conclusion, and the decisive requirement.

### D. What should we improve first?

Carry forward the selected candidate and population. Compare data, model, and
machine interventions from that same baseline under a common intervention
budget. Judge complete outcomes; do not rank arbitrary readiness scores or the
largest post-intervention surplus.

An intervention that helps one requirement can leave another unresolved or make
it worse. Students should explain what their spending forgoes and which
constraint remains afterward.

**Record:** chosen intervention, its cost and outcomes, a rejected alternative
with quantitative justification, and the remaining limitation.

### E. Does the decision survive changed conditions?

Stress the selected design with a changed population or greater work per decision
window. Locate a requirement crossing. Choose a monitoring signal and a condition
that warrants reevaluation. Explain what monitoring can establish and what
additional evidence is needed before corrective action.

**Record:** baseline and stress outcomes, the failed requirement or tested safe
range, reevaluation trigger, and residual uncertainty.

### Synthesis

Produce a short recommendation using this chain:

**Requirement → baseline → hypothesis → intervention → consequence → rejected
alternative → remaining constraint → reevaluation trigger.**

The report contains the student's recommendation, not automatic deployment
authorization. Preserve wrong initial predictions as evidence of learning.

## Tracks

All tracks answer the same questions. Each changes the workload, available
resources, requirements, and consequences. A selected track is a complete path.

| Track | Relevant pressure | Introductory evidence |
|---|---|---|
| TinyML | Always-on sensing with little working memory or energy. | Quality, memory fit, time and energy per sensing window. |
| Mobile | Interactive behavior within a device budget. | Quality, response time, memory and energy per interaction. |
| Edge | Local decisions before a deadline. | Quality, deadline compliance and bounded device resources. |
| Cloud | A service on a single machine. | Quality, per-request time and resource demand. |

Do not compare quality percentages across different tasks as if they measure the
same capability. A cross-track comparison transfers the reasoning process unless
the workload and evaluation population are explicitly matched. Fleet scheduling
and distributed coordination belong to later study.

## Simulator and interface responsibilities

MLSysIM owns scenario assumptions, quantities with units, equations, candidate
outcomes, intervention costs, feasibility, and quantitative comparisons. The lab
owns questions, controls, charts, evidence capture, and report composition.
Illustrative scenario quality rates must be identified as assumptions, not
presented as measured device or model benchmarks.

The interface starts with a compact question, track selector, estimated duration,
and deliverable. Every part follows **Predict → Experiment → Compare → Explain**.
Use one principal chart and a compact evidence table. Keep equations in optional
calculation notes. Avoid introducing Roofline analysis, queueing mathematics,
thermal models, tail-distribution instruments, or fleet coordination in this lab.

## Acceptance criteria

- Every track supports the intended contrasts and reachable failure conditions.
- Monitoring cannot change underlying quality without a modeled intervention.
- Candidate quality benefits have visible resource costs.
- Counterfactual interventions use a common baseline and explicit budget.
- Completion requires comparative evidence and deliberate decisions.
- Changing tracks cannot silently reuse evidence from another workload.
- Predictions remain distinguishable from observations and final explanations.
- The saved recommendation agrees with the recorded experiment state.
- The rendered notebook works on desktop and narrow screens without clipped
  controls, and failure states include text rather than color alone.
- Simulator tests check causal behavior and boundary cases; browser checks
  exercise prediction, comparison, capture, and report generation.

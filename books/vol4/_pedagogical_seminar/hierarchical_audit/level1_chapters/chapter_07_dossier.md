# Chapter 07 Micro Audit: Evaluation

- **Part**: Part II: Teaching
- **Core Thesis**: A neural policy's benchmark score evaluates the coupled dynamics of the entire physical assembly, not an intrinsic property of the model weights; therefore, offline validation and finite physical sampling are mathematically insufficient to certify deployment safety without structured runtime containment.
- **Audited At**: `2026-09-19T15:23:52.552689`

## Pedagogical Strengths
- 🟢 Vividly grounds statistical failures in specific physical realities (e.g., compounding unicycle kinematics error, Joule heating, motor winding thermal gradients).
- 🟢 Clearly demarcates the boundary between classical analytical stability (Lyapunov) and the statistical trial-counting forced upon black-box neural policies.

## Student Cohort Friction Points
- **Priya**: The mention of an image sensor exposure time increasing by 10 ms is excellent, but needs grounding in real-time budgets: at 60Hz (16.6ms), a 10ms exposure increase blows past the compute deadline, causing a dropped frame and pipeline stall.
- **Elena**: The Butler-Finelli exposure limit ($T \ge -\ln(1-C)/\lambda$) appears abruptly in the takeaways without prior definition in the main summary text, violating progressive disclosure.
- **Marcus**: Questions how the statistical 3/N rule for finite trials interacts with deterministic runtime monitoring; if we cannot prove Lyapunov stability, do we enforce an ISS-like envelope via the $1000\text{ Hz}$ MCU supervisor instead?

## Established Budgets & Invariants
- ⚖️ `Unicycle kinematic compounding error ($1.0^\circ$ bias = $17.5\text{ cm}$ lateral drift over $10\text{ m}$)`
- ⚖️ `The 3/N Rule for upper bounding failure probability in finite trials ($p \le 3/N$)`
- ⚖️ `Butler–Finelli exposure limits ($T \ge -\ln(1-C)/\lambda$)`
- ⚖️ `Classical Lyapunov stability conditions ($V(\mathbf{x}) > 0$, $\dot{V}(\mathbf{x}) \le -\alpha V(\mathbf{x})$)`
- ⚖️ `Joule heating ($I^2R$)`

## Conceptual Continuity
### Imported Prerequisites
- ↰ Causal boundary (imported from Chapter on boundaries / @sec-boundary-causal-boundary)
- ↰ Offline training pipelines and mean squared error/cross-entropy loss
- ↰ Real-time MCU supervisor constraints ($1000\text{ Hz}$)
### Exported Downstream Concepts
- ↳ The Evaluation Deliverable Quadrant (Bounded ODD, Reproducible Record, Coverage Gap Catalog, Runtime Specifications)
- ↳ The Fleet Testing Fallacy
- ↳ Closed-loop vs. Open-loop evaluation distinction

## Thematic Threads for Part Synthesis
- 🧵 How does the engineering team specify and enforce the 'Runtime Monitoring Specifications' and 'OOD reject gates' at the hardware level when the neural policy inevitably breaches its bounded Operational Design Domain?

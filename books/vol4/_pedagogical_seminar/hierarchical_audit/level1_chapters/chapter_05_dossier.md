# Chapter 05 Micro Audit: Data

- **Part**: Part II: Teaching
- **Core Thesis**: Physical data collection is an irreversible thermodynamic transaction that binds training datasets to the mechanical constraints, latency budgets, and safety boundaries of the specific hardware that captured them.
- **Audited At**: `2026-09-19T15:24:03.146596`

## Pedagogical Strengths
- 🟢 Grounds the abstract concept of machine learning datasets in physical realities like mechanical fatigue, energy expenditure, and sensor exposure windows.
- 🟢 Clearly contrasts endogenous experience in Physical AI with exogenous data collection in classical CV/NLP to highlight the causal coupling between actions and observations.

## Student Cohort Friction Points
- **Elena**: The 4-stream action tuple (a_req, a_cmd, a_enf, a_meas) appears in the summary as a critical takeaway but lacks sufficient progressive disclosure and definition in the main body text.
- **Marcus**: Control Barrier Functions (CBFs) and forward-invariance are mentioned abruptly in the synchronous capture loop section without grounding in control theory prerequisites.
- **Priya**: The multi-stream telemetry section introduces asynchronous clocks and SPI/CAN-FD buses, but skips how we physically reconcile clock drift before software-level timestamping.
- **Alex**: The ALVINN covariate shift example is intuitive, but I need to know exactly how standard supervised loss functions break when encountering out-of-distribution physical states lacking counterfactuals.

## Established Budgets & Invariants
- ⚖️ `Boundary retention law: Safety interventions and aborted runs must be preserved to teach recovery dynamics.`
- ⚖️ `Thermodynamic and mechanical budgets: Every physical demonstration consumes non-recoverable mechanical life, induces motor heating, and wears gear teeth.`
- ⚖️ `Telemetry timing budgets: Hard temporal constraints on active vision (10-30ms exposure, 4-8ms readout) vs. tactile matrix sampling (10ms intervals).`

## Conceptual Continuity
### Imported Prerequisites
- ↰ Basic closed-loop control and hardware feedback mechanics (e.g., motor drives, torque generation).
- ↰ Kinematic mappings and spatial transformations from end-effector coordinates to joint space.
### Exported Downstream Concepts
- ↳ Empirical support envelopes and convex safety boundaries required to trigger selective runtime abstention.
- ↳ The structural necessity of logging physical counterfactuals and recovery trajectories for downstream policy learning (Sim-to-Real, Behavioral Cloning).

## Thematic Threads for Part Synthesis
- 🧵 How do downstream learning algorithms (like Diffusion or Behavioral Cloning in Chapter 6) practically compensate for or abstain when encountering the physical counterfactual voids defined here?

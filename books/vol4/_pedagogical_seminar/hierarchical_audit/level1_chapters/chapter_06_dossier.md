# Chapter 06 Micro Audit: Training

- **Part**: Part II: Teaching
- **Core Thesis**: Physical AI policy training must reconcile the sample efficiency and safety of offline or simulated optimization with the unyielding reality of unmodeled hardware dynamics and compounding out-of-distribution execution errors.
- **Audited At**: `2026-09-19T15:23:52.526431`

## Pedagogical Strengths
- 🟢 Grounds abstract machine learning failure modes (like L2 mode averaging) in tangible physical consequences, such as destructive maneuvers or gear wear.
- 🟢 Clearly frames the sim-to-real gap as 'modeling debt' that gradient descent naturally and aggressively exploits.

## Student Cohort Friction Points
- **Elena**: The acronym 'ACT' is introduced alongside Diffusion and Flow Matching without prior expansion or grounding, disrupting cognitive flow for readers unfamiliar with recent imitation learning architectures.
- **Marcus**: The introduction of Action Chunking lacks an immediate bridge to classical control theory; students may wonder how this predictive horizon relates to Model Predictive Control (MPC).

## Established Budgets & Invariants
- ⚖️ `Quadratic compounding error bounds (O(T^2 epsilon)) for single-step behavioral cloning policies.`
- ⚖️ `Hardware and thermal budgets as hard constraints against unphysical high-frequency torque chatter.`

## Conceptual Continuity
### Imported Prerequisites
- ↰ Actuator thermal limits and duty cycles (@sec-body-thermal-duty-cycles)
- ↰ Rigid body dynamics and actuator limits (@sec-body-actuator-limits)
### Exported Downstream Concepts
- ↳ Sim-to-real modeling debt and domain randomization limits
- ↳ Compounding covariate shift in closed-loop physical execution
- ↳ Action chunking as a defense against high-frequency mode chatter

## Thematic Threads for Part Synthesis
- 🧵 How do we systematically benchmark, bound, and evaluate policies that carry unmodeled simulation debt before granting them full operational authority on physical hardware?

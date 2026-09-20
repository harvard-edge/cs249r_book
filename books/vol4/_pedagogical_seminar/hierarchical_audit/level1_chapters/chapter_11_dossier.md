# Chapter 11 Micro Audit: Planning

- **Part**: Part III: Running
- **Core Thesis**: Trajectory planning bridges the gap between spatial intent and physical execution by producing time-parameterized contracts that respect kinematic feasibility, strict tail-latency budgeting, and derivative continuity at replacement seams.
- **Audited At**: `2026-09-19T15:24:11.463176`

## Pedagogical Strengths
- 🟢 Successfully grounds abstract latency constraints into tangible physical consequences, such as infinite acceleration demands causing motor current saturation.
- 🟢 Clearly distinguishes geometric spatial paths from time-parameterized trajectories, emphasizing the necessity of the uncompromising clock in physical AI systems.

## Student Cohort Friction Points
- **Elena**: The dense mathematical notation for extreme mission tail quantile ($Q_{1-\epsilon_{\mathrm{mission}}/K}(L)$) is introduced abruptly in the takeaways without prior progressive disclosure or intuitive grounding in the main text.
- **Alex**: How do we map the highly variable, probabilistic inference latencies of modern diffusion policies to the strict deterministic tail-latency budgets ($P_{99.99997}$) demanded by the hardware?

## Established Budgets & Invariants
- ⚖️ `Latency Budgeting: Action chunk horizon $H$ must strictly exceed the extreme tail replacement latency of the compute pipeline to prevent physical starvation.`
- ⚖️ `Kinematic Continuity Law: Boundary derivatives must match exactly across plan seams ($\Delta \dot{\mathbf{q}} = 0$) to prevent infinite instantaneous acceleration and current loop saturation.`
- ⚖️ `Terminal Stopping Budget: Every trajectory must conclude with a physically achievable stopping suffix sized by $T_{\text{stop}} \ge v_{\text{bound}}/a_{\text{dec}}$.`

## Conceptual Continuity
### Imported Prerequisites
- ↰ Spatial intent and goals (@sec-intent)
- ↰ Causal boundary concepts (@sec-boundary-causal-boundary)
- ↰ Configuration space ($\mathcal{C}$-space) and topological manifolds
### Exported Downstream Concepts
- ↳ Time-parameterized action chunks and kinodynamic trajectories
- ↳ Seam lateness and boundary condition matching
- ↳ Mandatory terminal stopping suffixes

## Thematic Threads for Part Synthesis
- 🧵 How do the low-level hardware controllers and power electronics physically realize the high-frequency feedforward torques, and what happens when unmodeled environmental disturbances violate the planner's physical assumptions?

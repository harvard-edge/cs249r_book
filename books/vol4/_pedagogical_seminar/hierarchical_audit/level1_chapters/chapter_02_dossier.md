# Chapter 02 Micro Audit: The Body

- **Part**: Part I: Anatomy
- **Core Thesis**: Physical reality imposes non-negotiable budgets—momentum, thermal limits, actuation bandwidth, and sensor latency—that strictly bound what AI policies can safely command without causing hardware destruction or state loss.
- **Audited At**: `2026-09-19T15:24:05.792274`

## Pedagogical Strengths
- 🟢 Contrasts forgiving software abstractions (infinite holding torque, instant thread exceptions) with unyielding physical realities (continuous phase current, kinetic momentum).
- 🟢 Quantifies abstract constraints using concrete, realistic numbers (e.g., 120 kg payload at 2.0 m/s, J_ref = N^2 J_rotor) to ground the physical invariants.

## Student Cohort Friction Points
- **Elena**: The 'Five Physical Budgets' section launches into a transmission-shattering anecdote without first enumerating the budgets, violating progressive disclosure.
- **Priya**: The mention of PCIe buses and DC bus impedance droop assumes prerequisite knowledge of real-time hardware architecture that hasn't been explicitly grounded yet.

## Established Budgets & Invariants
- ⚖️ `Observation Age and Freshness Deadline (t_now - t_transduction <= \Delta t_fresh)`
- ⚖️ `Kinematic Momentum and Stopping Envelope (d_stop <= D_clear)`
- ⚖️ `Delivered Actuation and Reflected Rotor Inertia (J_ref = N^2 J_rotor)`
- ⚖️ `Energy and Thermal State via Joule Heating (P_loss = I^2 R)`
- ⚖️ `Electrical Power Integrity via DC Bus Impedance Droop (\Delta V_droop)`

## Conceptual Continuity
### Imported Prerequisites
- ↰ The causal boundary (@sec-boundary-causal-boundary)
### Exported Downstream Concepts
- ↳ Numerical physical envelopes bounding downstream cognitive deliberation (@sec-brain)
- ↳ The severe tension requiring arbitration across the causal boundary (@sec-nervous)

## Thematic Threads for Part Synthesis
- 🧵 How must the system architecture arbitrate between the high-capacity, stochastic nature of neural deliberation and the strict, deterministic physical limits of the robotic body?

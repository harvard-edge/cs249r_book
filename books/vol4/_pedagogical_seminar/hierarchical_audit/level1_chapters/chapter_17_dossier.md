# Chapter 17 Micro Audit: The Frontier

- **Part**: Conclusion
- **Core Thesis**: Physical AI systems require an explicit accounting of epistemic uncertainty, using tools like a residual-claims register to enforce operational limits when empirical evidence or runtime observability cannot guarantee safety.
- **Audited At**: `2026-09-19T15:23:51.736195`

## Pedagogical Strengths
- 🟢 Grounds abstract epistemic limits with stark, quantifiable physical examples (e.g., $1.5\text{ m/s}$ transport base requiring $180\text{ ms}$ braking vs $250\text{ ms}$ perception latency).
- 🟢 Synthesizes the entire multi-course systems arc (single machine, fleets, agents) into the embodied frontier, providing a satisfying pedagogical closure.

## Student Cohort Friction Points
- **Elena**: The dense list of requirements in 'What the Method Cannot Establish' creates cognitive overload; breaking it into bulleted invariants would improve progressive disclosure.
- **Marcus**: The 'practical detector' section introduces detection latency vs time-to-harm well, but could better explicitly tie this to the delay distance formula ($\Delta x = \int v \, dt$) mentioned later in the summary.

## Established Budgets & Invariants
- ⚖️ `Time-to-harm latency budget ($t_{\text{distinguish}} + \tau_{\text{det}} + \tau_{\text{act}} > t_{\text{harm}}$)`
- ⚖️ `Kinetic energy dissipation ($\frac{1}{2}mv^2$)`
- ⚖️ `Computational delay spatial consumption ($\Delta x = \int v \, dt$)`
- ⚖️ `Statistical exposure wall for empirical assurance ($p \le 10^{-9}/\text{hour}$ requiring $>3.0 \times 10^9\text{ h}$)`

## Conceptual Continuity
### Imported Prerequisites
- ↰ Proposal-permission authority separation (@sec-brain-embodied)
- ↰ Causal boundary definition (@sec-boundary-causal-boundary)
- ↰ ISO 10218 / ISO/TS 15066 safety envelopes
- ↰ Hardware Root of Trust and real-time fieldbus margins
### Exported Downstream Concepts
- ↳ The Residual-Claims Register as a formal systems engineering ledger
- ↳ The taxonomy of missing evidence (Observability Gap, Evaluation Gap)

## Thematic Threads for Part Synthesis
- 🧵 How does the engineering discipline evolve to ethically and legally manage the residual claims of embodied intelligence operating in open, unconstrained human environments?

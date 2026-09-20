# Physical AI Pedagogical Simulation & Student Agent Pool

This directory provides the multi-agent student simulation pipeline for *Physical AI Systems* (Volume IV of *Machine Learning Systems*).

It simulates a semester-long project seminar (Harvard CS/EE 288 / ETH Zurich 151-0851) where four diverse student learners read assigned chapters each week, record line-by-line margin notes, and convene in a seminar room moderated by a teaching assistant to debate progressive disclosure, cross-disciplinary accessibility, and pedagogical flow.

---

## The Tri-Discipline Challenge & Progressive Disclosure

Volume IV brings together three historically siloed engineering disciplines:
1. **Machine Learning**: Foundation models, diffusion policies, token generation, loss functions, continuous latent representations.
2. **Embedded Systems & Silicon Architecture**: RTOS, deterministic bus scheduling, memory bandwidth walls, DMA, interrupt jitter, bare-metal C.
3. **Control Theory & Robotics Mechanics**: Newton-Euler dynamics, reflected inertia, actuator saturation, thermal impedance, Lyapunov stability, irreversible kinetic energy.

### The Governing Principle: Progressive Disclosure
- **Physical Motivation First**: Never introduce an equation, silicon constraint, or motor parameter without grounding it in the physical problem the system must solve.
- **Cognitive Scaffolding**: Chapter $N$ assumes only $\{1, \dots, N-1\}$ plus explicit prerequisites in `prerequisites.qmd`.
- **Acronym & Jargon Discipline**: Every technical term (e.g. `CBF-QP`, `FOC`, `TSDF`, `QDD`) must be expanded and given a 1-sentence intuitive definition upon first mention.
- **Appendices as Buffers**: Deep mathematical derivations (e.g., Lie algebra in control, transformer attention complexity in ML, bus arbiters in systems) are referenced in `vol4/backmatter/appendix_{control,ml,systems}.qmd` so the main narrative never stalls.

---

## Student Cohort Personas

| Student | Background & Role | Primary Cognitive Lens | Review Focus |
|:---|:---|:---|:---|
| **Alex Chen** | MSc in CS / Deep Learning | Assumes digital idempotency, thinks in tokens, latent embeddings, and loss curves. | Flags unmotivated hardware physics; demands algorithmic intuition for mechanical limits. |
| **Priya Patel** | PhD in Computer Systems | Thinks in memory bandwidth (GB/s), cache lines, PCIe/AXI buses, and WCET. | Flags hand-wavy "real-time" claims, missing bus arbitration, and ungrounded silicon architecture. |
| **Marcus Vance** | PhD in MechE / Control | Thinks in $F=ma$, $\tau=I\alpha$, torque-speed curves, reflected inertia, and stability. | Flags violations of physics, unmodeled inertia, lack of stability proofs, and naive end-to-end ML. |
| **Elena Rostova** | Senior Undergraduate in EECS | The voice of progressive disclosure and fresh cognitive flow. | Flags unannounced acronyms, cognitive overload, inverted ordering, and abrupt leaps. |
| **Dr. Aris Thorne** | Lead TA & Moderator | Pedagogical synthesizer and seminar chair. | Moderates peer debates, adjudicates consensus vs. knowledge gaps, and authors surgical text rewrites. |

---

## Architecture & Workflow

```
                        Weekly Chapter Source (.qmd)
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │        curriculum.py / Parser            │
                │  - Maps 14-week syllabus to chapters     │
                │  - Chunks text while tracking line #s    │
                └────────────────────┬─────────────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │        student_reader.py                 │
                │  Simulates 4 student readers reading     │
                │  line-by-line with personal system prompt│
                └────────────────────┬─────────────────────┘
                                     │ (Line-by-line margin notes)
                                     ▼
                ┌──────────────────────────────────────────┐
                │         seminar_room.py                  │
                │  Dr. Aris convenes multi-turn seminar:   │
                │  - Peer debate & disciplinary exchange   │
                │  - Consensus on progressive disclosure   │
                │  - Formulates surgical text rewrites     │
                └────────────────────┬─────────────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │       dossier_generator.py               │
                │  Outputs:                                │
                │  - Markdown Audit Dossier (.md)          │
                │  - Machine-Readable Ledger (.json)       │
                │  - Tri-Discipline Balance Scorecard      │
                └────────────────────┬─────────────────────┘
                                     │
                                     ▼
                ┌──────────────────────────────────────────┐
                │          apply_edits.py                  │
                │  Surgically updates .qmd source with     │
                │  verified progressive disclosure fixes   │
                └──────────────────────────────────────────┘
```

---

## CLI Usage

### Run weekly pedagogical simulation:
```bash
# Run simulation for Week 1 (Chapter 1) using LLM
.venv/bin/python -m scripts.pedagogy_sim.orchestrator --week 1 --chapter 1 --mode api

# Run offline fast heuristic simulation without API calls
.venv/bin/python -m scripts.pedagogy_sim.orchestrator --week 1 --chapter 1 --mode heuristic
```

### Preview and apply proposed progressive disclosure edits:
```bash
# Dry run preview of consensus edits
.venv/bin/python -m scripts.pedagogy_sim.apply_edits books/vol4/_pedagogical_seminar/week_01_ch01_pedagogical_seminar_audit.json

# Live surgical application to the .qmd source (with automatic .bak backup)
.venv/bin/python -m scripts.pedagogy_sim.apply_edits books/vol4/_pedagogical_seminar/week_01_ch01_pedagogical_seminar_audit.json --apply
```

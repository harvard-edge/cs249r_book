# Physical AI Systems — detailed notes

> **Student-facing syllabus:** start at [`README.md`](README.md) (prerequisites, bigger picture, chapters, labs).
> This file keeps ETH packaging detail (assessment, schedule TBD, teaching-team notes).

**ETH Zurich · Project seminar & hardware studio · Draft syllabus**
**Portal:** [physical.mlsysbook.ai](https://physical.mlsysbook.ai)

| | |
| --- | --- |
| **Course** | Physical AI Systems |
| **Credits** | 6 ECTS *(proposed · ≈ 150–180 h)* |
| **Format** | Weekly seminar + hardware studio · **no written exam** |
| **Language** | English |
| **Level** | Advanced Bachelor & Master |
| **Kit** | **Physical AI Kit** (Arduino UNO Q dual-brain) |
| **Book** | *Physical AI: Machine Learning Systems That Sense and Act* |

> Items marked **TBD** lock when course number, room, and semester are confirmed.
> Intended as a **recurring** offering; open materials at `physical.mlsysbook.ai`.


## Teaching team

### Prof. Vijay Janapa Reddi — Lecturer

Gordon McKay Professor of Electrical Engineering, Harvard University · Visiting Professor, ETH Zurich

| | |
| --- | --- |
| **Office** | ETZ F 83 |
| **Email** | [vjanapa@ethz.ch](mailto:vjanapa@ethz.ch) |

### Dr. Andrea Mattia Garavagno — Co-teacher

Postdoctoral researcher, IIS / D-ITET, ETH Zurich · Physical AI Kit & studio lead

| | |
| --- | --- |
| **Email** | TBD *(course page)* |
| **Office** | TBD |

**Subject line:** `[Physical AI] …` · Office hours by appointment.


## What we teach

Standard ML ends at a digital output. **Physical AI** begins when software commands actuators—mass moves, energy is spent, the world permanently changes ($W_t \to W_{t+1}$). You cannot `ctrl+z` kinetic energy.

**North-star question**

> What must the surrounding system know, measure, enforce, preserve, and prove before an unverified learned proposal may produce a physical consequence?

**Proposal–permission (dual-brain) core**

| Host brain (MPU / Linux) | Reflex brain (MCU / real-time) |
| --- | --- |
| Perception, belief, VLMs, planning | Limits, watchdogs, CBFs, STO |
| **Proposes** expiring intent | **Permits or refuses** at ~1 kHz |

**Three cadences:** slow semantic deliberation (~0.5–2 Hz) · trajectory decode (~20–50 Hz) · bare-metal reflex (~1 kHz).

**Not this course:** robotics kinematics survey · TinyML-only deploy · LLM chat agents · safety certification exam.


## Who should take this

| Good fit | Probably not |
| --- | --- |
| You want ML that may **act** under constraint | Kinematics / ROS deep dive only |
| You like measuring systems (tails, energy, failure) | LLM tool-agent course |
| Teamwork on real hardware | Paper-presentation seminar only |
| ML systems *or* embedded/CPS background (we pair teams) | Classical written-exam lecture |

**Workload.** Project-first 6 ECTS. Seminar teaches method; most hours are the kit. Analytical substitutes exist for some stations—but **measure, runtime continuity, and MCU enforcer** are required.


## Curriculum spine (matches the book)

Eleven teaching chapters + capstone. One cumulative **design dossier**; each chapter freezes one artifact.

### Part I — Foundations *(the laws)*

| Ch | Title | You install | Dossier |
| ---: | --- | --- | --- |
| 1 | Causal boundary | When ML becomes physical AI; loop charter | loop charter |
| 2 | Physical constraints | Freshness, $P_{99}$, $d_{\text{stop}}$, energy, bus contention | requirements ledger |
| 3 | Cognitive dimensions | Co-design matrix; multi-rate lifecycle | workflow charter |

### Part II — Agent architecture *(perceive → permit)*

| Ch | Title | You install | Dossier |
| ---: | --- | --- | --- |
| 4 | Perception | Spatial tokens, DMA / sensing contracts | observation contract |
| 5 | Memory / world models | Frames, clocks, belief validity | state and timing model |
| 6 | Reasoning (intent) | VLMs as expiring proposals | intent schema |
| 7 | Planning | Action chunks / trajectories (proposals only) | planning schema |
| 8 | Enforcement | Independent MCU permission (signature) | enforcement design |

### Part III — Integration & release

| Ch | Title | You install | Dossier |
| ---: | --- | --- | --- |
| 9 | Placement | Heterogeneous map under shared budgets | placement ledger |
| 10 | Governance | Human authority; governed interaction data | authority design |
| 11 | Assurance | Evidence ladder; deploy / condition / refuse | release case |
| — | Capstone | Whole-system defense under seeded fault | Final release |


## Semester plan (14 weeks)

| Wk | Focus | Kit / studio | Due |
| ---: | --- | --- | --- |
| 1 | Kickoff · dual-brain · teams | Kit bring-up `00` | **T0** |
| 2 | Ch 1 boundary | Close the loop `01` | **T1 proposal** |
| 3 | Ch 2 constraints | Freshness + measure `02`–`03` | — |
| 4 | Ch 3 cognition / runtime | Fault containment `04` | **T2 foundations** |
| 5 | Ch 4 perception | Perception frontier `05` | — |
| 6 | Ch 5–6 state + intent | Belief + two-speed intent `06`–`07` | — |
| 7 | **Midterm** · Ch 7 planning preview | Propose ⇢ permit live | **T3 midterm** |
| 8 | Ch 8 enforcement | **MCU enforcer** `08` | — |
| 9 | Ch 9 placement | Placement ripple `09` | — |
| 10 | Ch 10 governance | Authority + learning turn `11`–`12` | — |
| 11 | Ch 11 assurance | Shadow / ship gate `10`/`13` | **T4 release draft** |
| 12 | Dry-run | Freeze dossier | — |
| 13 | **Capstone defense** | Seeded fault | **T5** |
| 14 | Buffer · return kits | — | **T6 dossier** |


## Assessment

| Component | Weight |
| --- | --- |
| Process & studio checkpoints | 20% |
| Midterm system review | 15% |
| Capstone defense | 25% |
| Cumulative design dossier | 40% |

**Pass bar:** real MCU permission path · ≥1 measured claim that changed a design · evidence-backed release verdict · every teammate can explain dual-brain without slides.


## Prerequisites

**Recommended:** ML systems intro (models as measured components) · Python and/or C/C++ · teamwork on hardware.
**Helpful:** embedded/real-time · TinyML · basic estimation.
**Not required:** agentic-LLM course · full robotics sequence.

Baseline compress/serve topics live in [mlsysbook.ai](https://mlsysbook.ai)—linked, not re-taught.


## Registration & contact

| | |
| --- | --- |
| **Registration** | myStudies · places limited by kit *(≈ 8–12 teams)* · TBD |
| **Lecturer** | [vjanapa@ethz.ch](mailto:vjanapa@ethz.ch) · ETZ F 83 |
| **Co-teacher** | Dr. Andrea Mattia Garavagno · email TBD |


## Catalogue blurb

Students design, build, measure, and defend a physical AI system in which a learned component may act only through independent real-time permission. Team projects on the Physical AI Kit (Arduino UNO Q dual-brain); weekly method seminars. Assessment: proposal, midterm demo, capstone defense, design dossier (**deploy / condition / refuse**). English. Proposed 6 ECTS.

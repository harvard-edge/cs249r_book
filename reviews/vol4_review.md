# Volume IV Master Review & Enhancement Blueprint: Physical AI Systems That Sense and Act

> **Document Status:** Active Execution Plan & Subagent Prompt Repository
> **Target Repository:** `MLSysBook` (`books/vol4/`)
> **Canonical Path:** `reviews/vol4_review.md`

---

## 1. Executive Mandate & Textbook Identity

### 1.1 The Physical MLSys Lens
Volume IV is an advanced textbook on **Physical AI: Machine Learning Systems That Sense and Act**. It defines the intersection of machine learning systems, physical hardware, real-time control, and embodied intelligence:
- **Volume I (Foundations):** Mathematical learning principles, optimization, foundational neural architectures, loss formulations.
- **Volume II (Distributed Serving & Infrastructure):** Distributed infrastructure, CUDA kernels, high-throughput model serving engines, data center networks.
- **Volume III (Agent Systems & Stochastic Runtimes):** Software-agent runtimes, tool actuation sandboxes, trajectory supervision, memory hierarchies, multi-agent swarms.
- **Volume IV (Physical AI Systems):** The physical, embodied machine learning system where **software meets atoms**. In Physical AI, actions have irreversible physical consequences; latency equals physical distance traveled; the environment cannot be paused or snapshotted; and safety is bound by conservation laws ($F = ma$, thermal dissipation, energy limits).

### 1.2 The Voice Rule: Avoiding Outdated Metaphors and Disembodied Abstractions
Physical AI possesses its own rigorous systems principles. **We avoid two failure modes:**
1. **Classical Disembodied ML:** Do *not* treat robots as mere "cameras with an LLM attached" or assume infinite compute and zero latency. Real robots face strict compute envelopes ($15\text{ W}$ to $150\text{ W}$), noisy sensors, non-stationary dynamics, and hard real-time deadlines ($1\text{ kHz}$ current loops, $100\text{ Hz}$ impedance control, $10\text{ Hz}$ vision-action policies).
2. **Pre-Deep-Learning Classical Robotics:** Do *not* relegate machine learning to an afterthought on top of 1980s kinematic pipelines. Modern Physical AI co-designs learned foundation policies (VLA models, diffusion policies, learned world models) with deterministic real-time safety envelopes, hardware-in-the-loop simulators, and distributed sensor-actuator networks.
- **Terminology:** Use native Physical MLSys vocabulary: *embodied policy inference, Control Barrier Functions (CBFs), real-time jitter bounds, sensor-actuator synchronization, sim-to-real reality gaps, action chunking, tactile state estimation, hardware-in-the-loop (HIL) testing, and graceful degradation envelopes.*

### 1.3 Pedagogical Scaffolding & Multi-Tier Accessibility
Physical AI spans mechanical engineering, electrical engineering, real-time systems, and deep learning. To make this cross-disciplinary field accessible without sacrificing mathematical depth:
- **Primary Learners:** Senior undergraduates and early graduate students in Computer Science, Mechanical Engineering, Electrical Engineering, and Robotics, alongside software engineers moving into embodied systems.
- **Advanced Specialists:** Roboticists, autonomous vehicle systems architects, and research scientists requiring formal safety guarantees, real-time scheduling proofs, and hardware rooflines.
- **Accessibility Mechanisms:**
  1. *Physical Reality Before Formal Proofs:* Every chapter opens with a concrete physical problem (e.g., drone motor saturation in wind gusts, robot arm vibration from sensor-actuator latency, or policy failure under lighting shifts) before formalizing the mathematics.
  2. *Cross-Discipline Bridge Definitions:* Roboticists may not know continuous batching or diffusion policy scheduling; computer scientists may not know Lie groups ($SE(3)$), Coriolis forces, or CAN bus arbitration. Every domain-specific concept is defined clearly on first use.
  3. *Scholarly Footnotes as the Depth Valve:* Keep the main narrative brisk and accessible. Specialized actuator driver quirks, low-level Linux PREEMPT_RT kernel configurations, detailed Lie algebra proofs, and niche sensor errata are placed in **scholarly footnotes**.
  4. *Concrete Worked Traces:* Provide step-by-step traces showing end-to-end data flow: sensor capture $\to$ temporal synchronization $\to$ neural policy inference $\to$ safety filter projection $\to$ motor torque actuation.

---

## 2. The 8-Stage Execution Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Parallel Chapter Audits (17 Agents, Chapters 01–17)                │
│ Audit against Physical MLSys lens, real-time bounds, and accessibility.    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Parallel Chapter Implementation (Targeted Chapter Edits)           │
│ Implement 5-phase progressive disclosure, inject equations & safety bounds.  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Part-Level Cohesion Audits (4 Agents, Parts 1–4)                   │
│ Verify cross-chapter flow across Anatomy, Teaching, Running, & Governing.   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Whole-Book Global Synthesis (End-to-End Embodied Arc Agent)        │
│ Ensure end-to-end coherence from Causal Boundary to Epistemic Frontier.     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Simulated Classroom Pedagogy (17 Dual-Persona Student Reads)       │
│ Test accessibility across undergraduate learners and systems researchers.   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: Visual Graphic & PNG Inspection (Visual Verification Engine)        │
│ Render kinematic, hardware, and control SVGs to PNG; inspect via view_file. │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: CMOS Reference & Prose Depth Pass (Chicago Manual of Style)        │
│ Standardize in-text figure callouts and ensure balanced technical depth.    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 8: CMOS Caption & Self-Contained Takeaways Verification               │
│ Enforce bold title phrases and complete explanatory systems takeaways.      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chapter & Part Distribution Map

Volume IV consists of **17 Chapters** structured across **4 Core Parts**, flanked by an introductory foundation and capstone frontier:

- **Introduction (Chapter 01): The Causal Boundary**
  - Why physical AI breaks classical digital assumptions: latency as physical displacement ($d = v \cdot \Delta t$), unpausable time, energy constraints, irreversible state changes.
- **Part I: The Machine Anatomy (Chapters 02–04)**
  - *Chapter 02: The Physical Body* — Actuation dynamics, electric/hydraulic/pneumatic drives, torque/speed curves, proprioceptive and tactile sensors, compliance.
  - *Chapter 03: The Compute & Acceleration Substrate* — Embedded SoCs (Jetson, Thor, mobile NPUs), power envelopes ($15\text{ W}–150\text{ W}$), thermal throttling, memory bandwidth on mobile platforms.
  - *Chapter 04: The Real-Time Interconnect & Nervous System* — Real-time buses (CAN, EtherCAT, TSN), ROS2/DDS middleware, clock synchronization (PTP/IEEE 1588), jitter and latency budgets.
- **Part II: Teaching the Machine (Chapters 05–07)**
  - *Chapter 05: Embodied Data Systems* — Teleoperation hardware, bilateral haptics, multi-modal logging, sensor time-synchronization, sim-to-real data curation.
  - *Chapter 06: Policy Learning & Foundation Models* — Imitation learning, Action Chunking Transformers (ACT), diffusion policies, reinforcement learning with physical priors, world models.
  - *Chapter 07: Closed-Loop Evaluation & Benchmarking* — Physics simulation (MuJoCo, Isaac Sim), reality gap metrics, hardware-in-the-loop (HIL) testing, distribution shifts under closed-loop rollout.
- **Part III: Running the Machine (Chapters 08–13)**
  - *Chapter 08: Multi-Modal State Estimation & Sensor Fusion* — Extended Kalman Filters, visual-inertial odometry (VIO), LiDAR SLAM, tactile sensing, 3D representations.
  - *Chapter 09: Spatial, Semantic & Episodic Memory* — Voxel grids, Octrees, 3D scene graphs, topological maps, persistent spatial-temporal state.
  - *Chapter 10: Goal Formulation & Intent Understanding* — Vision-Language-Action (VLA) models, open-vocabulary affordances, 3D geometric task grounding.
  - *Chapter 11: Real-Time Motion & Trajectory Planning* — Model Predictive Control (MPC), trajectory optimization, sampling planners (RRT*/PRM), hierarchical planning.
  - *Chapter 12: Enforcement & Safety Filtering* — Control Barrier Functions (CBFs), runtime reachability analysis, physical limiters, watchdog monitors.
  - *Chapter 13: Workload Placement & Partitioning* — Onboard vs edge vs cloud compute, wireless link dynamics (5G/Wi-Fi packet loss/jitter), fail-safe local reflexes.
- **Part IV: Governing the Machine (Chapters 14–16)**
  - *Chapter 14: Intervention & Shared Autonomy* — Human takeover protocols, teleoperation latency mitigation, impedance control, graceful degradation.
  - *Chapter 15: Verification & Safety Cases* — Formal verification, reachability analysis, safety cases (ISO 26262, ISO 13482), fault tree analysis.
  - *Chapter 16: Release, Fleet Telemetry & Operations* — Over-the-air (OTA) update safety, fleet calibration drift, shadow mode validation, regulatory standards.
- **Conclusion (Chapter 17): The Epistemic Frontier**
  - Synthesis of embodied MLSys, long-horizon autonomy, and open challenges at the physical-digital frontier.

---

## 4. Stage Specifications & Subagent Prompts

### Stage 1: Chapter-Level Architectural Audit (Parallel, 17 Agents)

**Objective:** Audit each of the 17 chapters against the Physical MLSys rubric, verify progressive disclosure, identify missing physical/mathematical equations, and ensure accessibility for undergrads and researchers alike.

#### Subagent Launch Prompt Template:
```markdown
You are an expert Physical Machine Learning Systems researcher and senior textbook author auditing Chapter {CHAPTER_NUM} ({CHAPTER_TITLE}) of Volume IV ("Physical AI: Machine Learning Systems That Sense and Act").

Your job is an exhaustive architectural audit of `books/vol4/{CHAPTER_DIR}/{CHAPTER_FILE}`.

EVALUATION CRITERIA:
1. Native Physical MLSys Lens:
   - Does this chapter bridge machine learning systems with physical, real-time embodiment?
   - Eliminate both purely disembodied ML abstractions (ignoring latency, physical dynamics, and compute budgets) and purely classical 1980s robotics (ignoring modern learned policies and neural perception).
   - Ensure the altitude balances physical physics/kinematics with ML systems execution.
2. Pedagogical Accessibility & Scaffolding:
   - Is the chapter accessible to senior undergraduates and early graduate students in CS/ME/EE?
   - Does it begin with concrete physical motivation (e.g., motor saturation, latency drift, sensor occlusion) before formal mathematical derivations?
   - Are cross-disciplinary concepts (e.g., SE(3), CAN arbitration, diffusion scheduling) clearly defined on first appearance?
3. Quantitative & Physical Rigor:
   - Are physical equations, timing deadlines ($1\text{ kHz}$ vs $100\text{ Hz}$ vs $10\text{ Hz}$), energy budgets ($W$), and compute rooflines provided?
   - Are hardware specifications grounded in modern physical AI platforms (e.g., Jetson Orin/Thor, real-time Ethernet, harmonic drives, industrial force sensors)?
4. Footnote Depth Strategy:
   - Identify dense mathematical proofs, low-level OS/driver errata, and specific hardware SKU quirks that should be moved to footnotes to keep the main narrative clean.
5. Figure & Table Integration:
   - Are all diagrams, timing sequences, and kinematic schematics properly integrated into the explanatory prose?

OUTPUT FORMAT:
Generate a structured audit report with:
- Executive Score (1-10) on Physical MLSys Lens & Accessibility.
- Identified Flaws (Metaphor leakage, missing timing/physical bounds, jargon walls).
- Concrete Chapter Edit Specification (exact sections to add, modify, or streamline).
```

---

### Stage 2: Chapter-Level Implementation & Progressive Disclosure Edits

**Objective:** Implement the revisions identified in Stage 1 directly in the `.qmd` chapter files within isolated task worktrees.

#### Execution Sequence within Each Chapter:
1. **Physical Motivation:** Start with the concrete embodied failure mode or engineering challenge.
2. **Formal Abstraction:** Mathematical model (kinematic/dynamic equations, state-space representations, or policy formulation).
3. **Systems Architecture:** Real-time data pipeline, hardware execution substrate, communication bus, or safety filter.
4. **Empirical Benchmarks & Physical Limits:** Roofline models, latency vs. tracking accuracy trade-offs, sim-to-real transfer curves.
5. **Architectural Principles:** Key takeaway rules for Physical AI systems engineers.

---

### Stage 3: Part-Level Cohesion Audits (4 Agents, Parts 1–4)

**Objective:** Audit each of the 4 Parts to ensure seamless cross-chapter handoffs, consistent notation ($x, u, q, \dot{q}, \tau$), and elimination of redundant introductory text across chapters.

#### Evaluation Directives:
- Verify that Part I (Anatomy) smoothly feeds into Part II (Teaching).
- Verify that Part II (Teaching) policy outputs align with Part III (Running) execution loops.
- Verify that Part III (Running) failure modes motivate Part IV (Governing) safety and verification mechanisms.
- Standardize physical and ML symbols across the Part.

---

### Stage 4: Whole-Book Global Synthesis & Capstone Coherence

**Objective:** Conduct an end-to-end audit verifying the entire narrative arc from Chapter 01 (The Causal Boundary) to Chapter 17 (The Epistemic Frontier).
- Verify that safety cases and control barrier functions introduced in Chapter 12 are carried through to fleet verification in Chapter 15 and 16.
- Ensure consistent cross-references across volumes (e.g., referring to Volume II for distributed GPU training of foundation policies, and Volume III for software agent tool use).

---

### Stage 5: Simulated Classroom Pedagogy (17 Sequential Dual-Persona Student Reads)

**Objective:** Simulate an advanced interdisciplinary course in Physical AI where two student personas read each chapter sequentially.

#### Simulation Personas:
- **Persona A: The Senior Undergrad / Early Grad Student (Interdisciplinary Learner):**
  - *Background:* Has taken undergraduate machine learning, data structures, and basic physics/control.
  - *Evaluation Lens:* "Where do I get lost in the transition from digital ML to physical control? Are sensor-actuator timing concepts explained intuitively? Are mathematical steps motivated before equations appear?"
- **Persona B: The Senior Autonomous Systems Architect / Roboticist:**
  - *Background:* 10+ years deploying autonomous robots, autonomous vehicles, or industrial physical AI in the real world.
  - *Evaluation Lens:* "Are real-time latency numbers realistic? Does the text address actuator saturation, thermal throttling, sensor noise, and safety certification? Are edge cases properly isolated into footnotes?"

---

### Stage 6: Visual Graphic & PNG Inspection Protocol

**Objective:** Visually inspect all diagrams across Volume IV to ensure immaculate publication quality.

#### Verification Protocol:
1. Render every SVG to PNG:
   `rsvg-convert -w 1600 books/vol4/{CHAPTER}/images/svg/{NAME}.svg -o /tmp/vol4_vis_{NAME}.png`
2. Inspect via `view_file` tool call.
3. Validate:
   - **Zero Text Collisions:** Labels, arrows, and coordinate frames must never overlap.
   - **Kinematic & Control Clarity:** Block diagrams (control loops, sensor pipelines) must have clear signal flow arrows and labeled units.
   - **Textbook Styling:** Standardized fonts (Fira Sans / JetBrains Mono) and publication color palette.
   - **Empirical Plots:** Quantitative graphs must be generated via Python/Matplotlib scripts.

---

### Stage 7: CMOS In-Text Reference & Prose Depth Pass

**Objective:** Audit all in-text figure callouts to ensure compliance with the Chicago Manual of Style.
- Replace all generic pointers ("as seen below", "refer to the figure above") with explicit, flowing references ("As shown in Figure 8.4, the visual-inertial odometry pipeline...").
- Calibrate explanatory depth: explain the physical causal mechanics and design trade-offs depicted without mechanically transcribing every visual label.

---

### Stage 8: CMOS Caption & Self-Contained Takeaway Verification

**Objective:** Ensure all captions across Volume IV follow professional CMOS textbook formatting:
- **Structure:** `**Figure X.Y: Bold Synthesizing Title Phrase.** Explanatory sentence detailing the physical and computational context. Concluding sentence highlighting the critical physical trade-off or systems takeaway.`
- **Skimmability:** A reader reviewing only the figures and captions must understand the core architectural insight of each system diagram.

---

## 5. Verification Gates & Execution Checklist

Before concluding any stage or merging changes into local `dev`:
- [ ] `./binder/binder check refs --scope citations` (Zero missing or orphaned citations)
- [ ] `./binder/binder check refs --scope duplicate-key` (Zero duplicate label or bib keys)
- [ ] `./binder/binder check refs --scope manual-bracket` (Zero malformed citation brackets)
- [ ] `view_file` visual verification completed on 100% of modified or newly generated figures.
- [ ] American English spelling verified throughout (`-ize`, `center`, `behavior`, `modeled`).
- [ ] Zero AI co-author or attribution trailers in git commit history.
- [ ] `tree-merge` protocol followed strictly: sync `dev` into worktree, verify hooks, promote via `--no-ff` from main checkout, delete branch with `-d`.

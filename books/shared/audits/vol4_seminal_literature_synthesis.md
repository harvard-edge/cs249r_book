# Master Synthesis: Seminal Literature & Foundations for Volume IV

**Document Version:** 1.0 (Curriculum Synthesis)
**Standard:** Hennessy & Patterson Standard for Physical AI / MIT Press Editorial Spine
**Target Volume:** *Physical AI: Machine Learning Systems That Sense and Act* (Volume IV)

---

## Overview and Purpose

This document provides a comprehensive, chapter-by-chapter synthesis of the **seminal literature**, **canonical engineering papers**, and **foundational scientific principles** that anchor Volume IV.

In keeping with the goal of creating an enduring, timeless textbook—one that stands alongside Hennessy & Patterson’s *Computer Architecture* and avoids ephemeral machine learning fads—every chapter must be firmly rooted in the physical, mathematical, and computer systems canon. For each cited work, this document outlines:
1. **Citation Metadata & BibTeX Key**: Author, Year, Title, and Venue.
2. **Core Theoretical / Systems Insight**: The foundational theorem, mathematical formulation, or physical mechanism.
3. **Why Physical AI Must Cite It**: Why an engineering student or practitioner must know this work, and how it reframes machine learning from disembodied software into grounded physical systems.
4. **Concrete In-Text Placement**: Exactly where and how the reference integrates into the volume narrative.

---

## Part I: Machine Anatomy (The Physical Substrate)

```
Part I System Architecture:
├── Chapter 01: The Boundary of Embodiment  (Thermodynamics, Cybernetics & Irreversibility)
├── Chapter 02: The Physical Body          (Electromechanics, Reflected Inertia & Thermal Physics)
├── Chapter 03: The Nervous System         (Real-Time Scheduling, Microsecond Bus Arbitration)
└── Chapter 04: The Deliberative Brain     (Memory Walls, Rooflines & Systolic Acceleration)
```

---

### Chapter 01: The Boundary of Embodiment (`01-boundary.qmd`)

#### 1. Wiener (1948) — Cybernetics
* **Reference**: Wiener, Norbert. *Cybernetics: Or Control and Communication in the Animal and the Machine*. MIT Press / John Wiley & Sons, 1948.
* **Citekey**: `@wiener1948cybernetics`
* **Core Insight**: Information feedback loops. Wiener established that control in animals and machines is governed by circular causal chains: actions alter sensory feedback, which in turn recalculates actions.
* **Why Physical AI Must Cite It**: In disembodied machine learning, inference is a directed acyclic graph (tokens $\to$ tokens). In Physical AI, the environment is in the loop. Wiener provides the historical and philosophical origin of closed-loop dynamical interaction.
* **Placement**: Section 1.1, opening paragraphs defining the transition from open-loop calculation to closed-loop embodiment.

#### 2. Moravec (1988) — Moravec's Paradox
* **Reference**: Moravec, Hans. *Mind Children: The Future of Robot and Human Intelligence*. Harvard University Press, 1988.
* **Citekey**: `@moravec1988mind`
* **Core Insight**: High-level abstract reasoning (e.g., chess, symbolic logic, medical diagnosis) requires trivial computational power, whereas low-level sensorimotor coordination (e.g., walking, grasping, tactile perception) requires vast computational capacity shaped by a billion years of evolution.
* **Why Physical AI Must Cite It**: Explains to students why LLMs reached superhuman benchmark scores in language while robots still struggle to clear a dinner table. It directly justifies the book’s focus on physical perception and actuation.
* **Placement**: Section 1.1, contrasting digital foundation models with the physical wall.

#### 3. Landauer (1961) — Thermodynamic Limit of Erasure
* **Reference**: Landauer, Rolf. "Irreversibility and Heat Generation in the Computing Process." *IBM Journal of Research and Development* 5, no. 3 (1961): 183–191.
* **Citekey**: `@landauer1961`
* **Core Insight**: Any logically irreversible manipulation of information (such as the erasure of a bit or the merge of two computational paths) must dissipate a minimum thermodynamic energy into the environment:
  $$E_{\text{dissipated}} \ge k_B T \ln 2$$
* **Why Physical AI Must Cite It**: Information in digital memory can be rewritten or reset at near-zero physical cost. In the physical world, kinetic collisions, irreversible friction, and plastic deformation cannot be erased. Landauer provides the thermodynamic grounding for why Physical AI cannot "undo" a physical failure.
* **Placement**: Section 1.2, under the derivation of irreversible state transitions and Newtonian grounding.

#### 4. Brooks (1986, 1991) — Subsumption & Situatedness
* **Reference**: Brooks, Rodney A. "A Robust Layered Control System for a Mobile Robot." *IEEE Journal on Robotics and Automation* 2, no. 1 (1986): 14–23; and "Intelligence Without Representation." *Artificial Intelligence* 47 (1991): 139–159.
* **Citekey**: `@brooks1986robust`, `@brooks1991intelligence`
* **Core Insight**: Physical grounding and the Subsumption Architecture. Intelligent behavior emerges from situated interaction with the physical environment ("the world is its own best model") through fast, uninhibited reactive layers rather than monolithic, centralized symbolic world representations.
* **Why Physical AI Must Cite It**: Brooks justifies the Four-Tier Architectural Stack. Fast reactive enforcers (the nervous system) must operate independently of slow deliberative models (the brain) to preserve survival.
* **Placement**: Section 1.3, motivating the separation of deliberative reasoning from low-level deterministic reflexes.

#### 5. Sutton (2019) — The Bitter Lesson vs. The Physical Wall
* **Reference**: Sutton, Richard. "The Bitter Lesson." Incomplete Ideas (blog), March 13, 2019.
* **Citekey**: `@sutton2019bitter`
* **Core Insight**: In AI research, general-purpose methods that leverage massive computation (search and learning) consistently beat methods relying on human-crafted domain heuristics in the long run.
* **Why Physical AI Must Cite It**: The volume explicitly contrasts Sutton’s Bitter Lesson with the **Physical Wall**: while compute scaling triumphs in digital realms, in embodied hardware, compute hits thermal dissipation ceilings ($15\text{--}60\text{ W}$), battery drains, reflected inertia limits ($N^2 J$), and hard physical safety bounds.
* **Placement**: Section 1.4, closing synthesis on why scaling laws differ in physical reality.

---

### Chapter 02: The Physical Body (`02-body.qmd`)

#### 1. Hogan (1985) — Impedance Control
* **Reference**: Hogan, Neville. "Impedance Control: An Approach to Manipulation." *ASME Journal of Dynamic Systems, Measurement, and Control* 107, no. 1 (1985): 1–24.
* **Citekey**: `@hogan1985impedance`
* **Core Insight**: Physical contact interaction cannot be modeled as pure position control or pure force control. Hogan established the impedance ($F = Z(v)$) and admittance ($v = Y(F)$) duality: a robot must modulate its dynamic relationship with the environment (stiffness, damping, and inertia).
* **Why Physical AI Must Cite It**: Learned neural policies cannot output stiff Cartesian position commands in unknown contact environments without inducing infinite contact forces and breaking gearboxes. Impedance control is the foundational physical control interface.
* **Placement**: Section 2.5, when establishing actuator output modalities and compliant interaction.

#### 2. Spong (1987) — Modeling and Control of Elastic Joint Robots
* **Reference**: Spong, Mark W. "Modeling and Control of Elastic Joint Robots." *ASME Journal of Dynamic Systems, Measurement, and Control* 109, no. 4 (1987): 310–319.
* **Citekey**: `@spong1987modeling`
* **Core Insight**: Flexible joint dynamics. Real robotic transmissions (harmonic drives, cycloidal gears, belts) introduce joint elasticity that decouples the motor rotor inertia from the link dynamics. Spong formulated the singular perturbation model separating the fast joint vibration manifold from slow link motion.
* **Why Physical AI Must Cite It**: Students must understand why rigid-body Euler-Lagrange equations fail at high speeds or under impact, and why joint compliance causes catastrophic chattering if learned policies command high-frequency torque ripples.
* **Placement**: Section 2.3, following the transmission and gearbox mechanics derivation.

#### 3. Seok et al. (2015) & Wensing et al. (2017) — Proprioceptive Quasi-Direct Drive (MIT Cheetah)
* **Reference**: Seok, Sangok et al. "Design Principles for Energy-Efficient Legged Locomotion and Proprioceptive Actuation." *IEEE/ASME Transactions on Mechatronics* 20, no. 3 (2015): 1113–1125; and Wensing, Patrick M. et al. "Proprioceptive Actuator Design in the MIT Cheetah." *IEEE Transactions on Robotics* 33, no. 5 (2017): 1109–1122.
* **Citekey**: `@seok2015design`, `@wensing2017proprioceptive`
* **Core Insight**: Low-gear-reduction ($N \le 10$) proprioceptive actuation. By minimizing the gear ratio, reflected rotor inertia ($N^2 J_m$) is reduced by orders of magnitude, making the actuator physically backdrivable, transparent to ground impact shocks, and capable of sensing external forces directly via motor phase currents ($I = \tau / K_t$) without fragile 6-axis load cells.
* **Why Physical AI Must Cite It**: Explains the universal hardware shift from high-ratio industrial robots (KUKA, FANUC) to modern dynamic quadrupeds (Spot, Cheetah) and collaborative humanoids.
* **Placement**: Section 2.4, under the comparative analysis of actuator gear ratios and reflected inertia.

#### 4. Mason (2001) — Mechanics of Robotic Manipulation
* **Reference**: Mason, Matthew T. *Mechanics of Robotic Manipulation*. MIT Press, 2001.
* **Citekey**: `@mason2001mechanics`
* **Core Insight**: Frictional contact mechanics, Coulomb friction cones, Form closure vs. Force closure, and the mechanics of pushing and grasping.
* **Why Physical AI Must Cite It**: Contact is non-smooth, discontinuous, and governed by friction constraints. Mason provides the analytical mechanics required to understand why physical manipulation cannot be treated as smooth Euclidean regression.
* **Placement**: Section 2.6, introducing contact friction and kinematic constraints.

---

### Chapter 03: The Deliberative Brain (`03-brain.qmd`)

#### 1. Hennessy & Patterson (2019) — Computer Architecture: A Quantitative Approach
* **Reference**: Hennessy, John L., and David A. Patterson. *Computer Architecture: A Quantitative Approach*. 6th ed. Morgan Kaufmann, 2019.
* **Citekey**: `@hennessy2019computer`
* **Core Insight**: The Roofline Model, the Memory Wall, Amdahl’s Law, and Domain-Specific Architectures (DSA). Provides the quantitative framework relating floating-point performance ($P_{\text{peak}}$), memory bandwidth ($B_{\text{mem}}$), and operational arithmetic intensity ($I = \text{FLOP/byte}$).
* **Why Physical AI Must Cite It**: Deep learning models running on edge robots are severely memory-bandwidth bound. Hennessy & Patterson provides the theoretical apparatus to analyze why edge SoCs stall on weight streaming during single-step policy execution.
* **Placement**: Section 3.3, introducing the Roofline model and memory bus bottlenecks.

#### 2. Horowitz (2014) — Computing's Energy Problem
* **Reference**: Horowitz, Mark. "1.1 Computing's Energy Problem (and What We Can Do About It)." In *IEEE International Solid-State Circuits Conference (ISSCC) Digest of Technical Papers*, 10–14. IEEE, 2014.
* **Citekey**: `@horowitz2014computing`
* **Core Insight**: The physics of computational energy consumption. A 16-bit floating-point multiply-accumulate (MAC) consumes $\approx 0.2\text{ pJ}$, whereas fetching that same 16-bit operand from external LPDDR DRAM consumes $\approx 300\text{--}500\text{ pJ}$—a **$1500\times$ to $2500\times$ energy tax** for data movement.
* **Why Physical AI Must Cite It**: In mobile robots operating on battery power, memory access drains battery energy and produces silicon heat far faster than computation itself. Horowitz proves why model quantization and on-chip SRAM reuse are thermodynamic imperatives.
* **Placement**: Section 3.4, discussing thermal dissipation envelopes and edge SoC power budgets.

#### 3. Jouppi et al. (2017, 2023) — Tensor Processing Unit (TPUv1 & TPUv4)
* **Reference**: Jouppi, Norman P. et al. "In-Datacenter Performance Analysis of a Tensor Processing Unit." In *ACM/IEEE International Symposium on Computer Architecture (ISCA)*, 1–12. IEEE, 2017; and "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings." In *ISCA*, 2023.
* **Citekey**: `@jouppi2017`, `@jouppi2023`
* **Core Insight**: 2D Systolic Array architectures for matrix multiplication. Weights are held stationary while activations flow through a grid of arithmetic processing elements (PEs), eliminating intermediate register-file and cache reads.
* **Why Physical AI Must Cite It**: Teaches students the microarchitectural difference between GPU SIMT streaming multiprocessors and dedicated NPU/TPU systolic arrays deployed on edge robots.
* **Placement**: Section 3.5, contrasting edge accelerator dataflows (weight-stationary vs. output-stationary).

#### 4. Yun et al. (2013) — MemGuard: Memory Bandwidth Reservation
* **Reference**: Yun, Heechul et al. "MemGuard: Memory Bandwidth Reservation System for Efficient Performance Isolation in Multi-Core Platforms." In *IEEE Real-Time and Embedded Technology and Applications Symposium (RTAS)*, 55–64. IEEE, 2013.
* **Citekey**: `@yun2013memguard`
* **Core Insight**: Memory bus contention on heterogeneous Systems-on-Chip (SoCs). When an unprivileged GPU/NPU accelerator saturates shared DRAM bandwidth with dense weight fetches, real-time CPU cores suffer massive memory latency spikes ($>10\text{ ms}$).
* **Why Physical AI Must Cite It**: Directly addresses the "silent crash" failure mode in robotics where running an embodied foundation model causes real-time CAN bus or IMU interrupt handlers to miss timing deadlines.
* **Placement**: Section 3.6, under heterogeneous compute SoC scheduling and shared bus contention.

---

### Chapter 04: The Nervous System (`04-nervous.qmd`)

#### 1. Liu & Layland (1973) — Rate Monotonic & EDF Scheduling
* **Reference**: Liu, C. L., and James W. Layland. "Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment." *Journal of the ACM* 20, no. 1 (1973): 46–61.
* **Citekey**: `@liu1973scheduling`
* **Core Insight**: The fundamental mathematical foundations of hard real-time scheduling. Derived the exact CPU utilization bound for static-priority Rate Monotonic Scheduling (RMS):
  $$U = \sum_{i=1}^n \frac{C_i}{T_i} \le n \left(2^{1/n} - 1\right) \xrightarrow{n \to \infty} \ln 2 \approx 0.693$$
  and proved the optimality of dynamic-priority Earliest Deadline First (EDF) ($U \le 1.0$).
* **Why Physical AI Must Cite It**: Robotic control loops must meet strict temporal deadlines (e.g., $1\text{ kHz}$ motor loops, $100\text{ Hz}$ trajectory filters). Liu & Layland proves whether a given set of periodic tasks will provably avoid deadline misses.
* **Placement**: Section 4.2, introducing real-time operating systems (RTOS) scheduling algorithms.

#### 2. Sha, Rajkumar, & Lehoczky (1990) — Priority Inheritance Protocols
* **Reference**: Sha, Lui, Ragunathan Rajkumar, and John P. Lehoczky. "Priority Inheritance Protocols: An Architectural Approach to Real-Time Synchronization." *IEEE Transactions on Computers* 39, no. 9 (1990): 1175–1185.
* **Citekey**: `@sha1990priority`
* **Core Insight**: Solves the Unbounded Priority Inversion problem (the bug that famously froze the Mars Pathfinder spacecraft in 1997). Formalized the Priority Inheritance Protocol (PIP) and Priority Ceiling Protocol (PCP), mathematically bounding the blocking duration of high-priority tasks to at most one lower-priority critical section.
* **Why Physical AI Must Cite It**: In a physical robot, if a high-priority motor control task shares a mutex with a low-priority logging thread, a medium-priority vision thread can preempt the logger, indefinitely stalling the motor loop and causing catastrophic physical failure.
* **Placement**: Section 4.3, covering multi-threaded synchronization and lock contention on microcontrollers.

#### 3. Kopetz (2011) — Time-Triggered Architecture (TTA)
* **Reference**: Kopetz, Hermann. *Real-Time Systems: Design Principles for Distributed Embedded Applications*. 2nd ed. Springer, 2011.
* **Citekey**: `@kopetz2011real`
* **Core Insight**: Time-Triggered Architecture (TTA) vs. Event-Triggered Architecture. Establishes determinism, composability, global clock synchronization, and the "babbling idiot" failure mode where a malfunctioning node floods a bus with noise.
* **Why Physical AI Must Cite It**: Serves as the primary reference for deterministic fieldbus communication (EtherCAT Distributed Clocks, TSN IEEE 802.1Qbv, and Time-Triggered Ethernet).
* **Placement**: Section 4.4, fieldbus comparison and deterministic clock synchronization.

---

## Part II: Teaching the Machine (Behavioral Synthesis)

```
Part II System Architecture:
├── Chapter 05: Physical Data         (Telemetry, Bilateral Passivity & Sensor Logging)
├── Chapter 06: Behavior Learning     (DAgger Proof, Flow Matching & Compounding Shift)
└── Chapter 07: Empirical Evaluation  (Closed-Loop Metrology, Wald SPRT & Rule of Three)
```

---

### Chapter 05: Physical Data (`05-data.qmd`)

#### 1. Pomerleau (1989) — ALVINN
* **Reference**: Pomerleau, Dean A. "ALVINN: An Autonomous Land Vehicle in a Neural Network." In *Advances in Neural Information Processing Systems (NeurIPS)* 1, 305–313. 1989.
* **Citekey**: `@pomerleau1989alvinn`
* **Core Insight**: The historical origin of behavioral cloning on physical hardware. Pomerleau trained a 3-layer neural network to steer an autonomous Chevy van from camera images, discovering that the vehicle quickly drifted off the road because training data contained only expert driving (no recovery trajectories).
* **Why Physical AI Must Cite It**: The seminal paper that exposed the compounding error problem in imitation learning.
* **Placement**: Section 5.1, introducing the unique characteristics and hazards of embodied telemetry data.

#### 2. Mandlekar et al. (2020) — Robot Teleoperation and Demonstration Metrology
* **Reference**: Mandlekar, Ajay et al. "Human-in-the-Loop Robot Learning and Teleoperation." *Robotics: Science and Systems (RSS) Workshops*, 2020; and "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation." In *Conference on Robot Learning (CoRL)*, 2021.
* **Citekey**: `@mandlekar2020human`
* **Core Insight**: Metrology of human demonstrations. Analyzed how operator interface latency, multimodal human strategies, and inconsistent sub-optimal demonstrations degrade downstream policy imitation.
* **Why Physical AI Must Cite It**: Provides the experimental science for physical dataset collection, teleoperation latency budgets, and demonstration quality filtering.
* **Placement**: Section 5.3, teleoperation systems, latency, and demonstrator variance.

#### 3. Zhao et al. (2023) — ALOHA & Action Chunking Data Infrastructure
* **Reference**: Zhao, Tony Z. et al. "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware." In *Robotics: Science and Systems (RSS)*, 2023.
* **Citekey**: `@zhao2023learning`
* **Core Insight**: Bilateral mechanical teleoperation without actuation. ALOHA utilized an unactuated, passively counterbalanced leader-follower mechanism, completely bypassing the software passivity/instability traps of active haptic feedback while capturing precise 50 Hz kinematic telemetry.
* **Why Physical AI Must Cite It**: The ubiquitous open-source standard for modern robotic manipulation data collection.
* **Placement**: Section 5.4, under teleoperation hardware architectures and bilateral feedback.

#### 4. Anderson & Spong (1989) — Bilateral Teleoperation with Time Delay
* **Reference**: Anderson, Robert J., and Mark W. Spong. "Bilateral Control of Teleoperators with Time Delay." *IEEE Transactions on Automatic Control* 34, no. 5 (1989): 494–501.
* **Citekey**: `@anderson1989bilateral`
* **Core Insight**: Passivity theory applied to delayed teleoperation. Demonstrated that network transmission delays transform a passive bilateral teleoperator into an active energy generator, causing destructive mechanical instability. Solved using scattering formulations / wave variables.
* **Why Physical AI Must Cite It**: The fundamental mathematical proof of why remote teleoperation over wireless links destabilizes without explicit passivity observers.
* **Placement**: Section 5.4, explaining the physics of bilateral force-feedback instability.

---

### Chapter 06: Behavior Learning (`06-training.qmd`)

#### 1. Ross, Gordon, & Bagnell (2011) — DAgger
* **Reference**: Ross, Stéphane, Geoffrey J. Gordon, and J. Andrew Bagnell. "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." In *International Conference on Artificial Intelligence and Statistics (AISTATS)*, 627–635. 2011.
* **Citekey**: `@ross2011reduction`
* **Core Insight**: The seminal proof that Behavioral Cloning suffers from quadratic compounding error:
  $$J(\hat{\pi}) - J(\pi^*) \le \mathcal{O}\left(T^2 \epsilon\right)$$
  Introduced **DAgger** (Dataset Aggregation), proving that querying an expert oracle on the learner's visited states collapses compounding regret back to linear: $\mathcal{O}(T \epsilon)$.
* **Why Physical AI Must Cite It**: The foundational mathematical theorem governing imitation learning in closed-loop dynamical systems.
* **Placement**: Section 6.2, establishing the theoretical failure mode of Behavioral Cloning and motivating interactive imitation.

#### 2. Levine et al. (2016) — End-to-End Visuomotor Policies
* **Reference**: Levine, Sergey et al. "End-to-End Training of Deep Visuomotor Policies." *Journal of Machine Learning Research (JMLR)* 17, no. 39 (2016): 1–40.
* **Citekey**: `@levine2016end`
* **Core Insight**: The landmark achievement proving that deep neural networks could be trained end-to-end directly from raw camera RGB pixels to robot joint torques on real physical hardware, bypassing hand-engineered perception pipelines.
* **Why Physical AI Must Cite It**: The foundational historical milestone for modern deep visuomotor policy training.
* **Placement**: Section 6.1, opening the history of learned behavior policies.

#### 3. Florence et al. (2022) — Implicit Behavioral Cloning
* **Reference**: Florence, Pete et al. "Implicit Behavioral Cloning: Dual Representations for Robot Learning." In *Conference on Robot Learning (CoRL)*, 1584–1595. PMLR, 2022.
* **Citekey**: `@florence2022implicit`
* **Core Insight**: Resolving the Multimodality Catastrophe via Energy-Based Models (EBMs). Standard feedforward networks ($a = \pi(s)$) optimize $L_2$ loss, causing mode averaging (e.g., steering directly into a central obstacle when two safe paths exist). Implicit BC models an energy landscape $E(s, a)$, taking actions via argmin energy optimization, naturally handling sharp discontinuities.
* **Why Physical AI Must Cite It**: The bridge between classical optimal control energy minimization and modern continuous generative decoders.
* **Placement**: Section 6.2, overcoming multimodality in continuous action distributions.

#### 4. Chi et al. (2023) — Diffusion Policy
* **Reference**: Chi, Cheng et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." *The International Journal of Robotics Research (IJRR)*, 2024 (ArXiv 2023).
* **Citekey**: `@chi2024diffusionpolicy`
* **Core Insight**: Formulating trajectory generation as a conditional score-based denoising diffusion process over temporal action chunks. Expresses arbitrary multi-modal distributions without mode averaging, handles high-dimensional action spaces, and exhibits smooth closed-loop stability.
* **Why Physical AI Must Cite It**: The state-of-the-art baseline for continuous trajectory generation in modern robotics.
* **Placement**: Section 6.2, presenting continuous trajectory action decoders.

#### 5. Lipman et al. (2022) — Conditional Flow Matching
* **Reference**: Lipman, Yaron et al. "Flow Matching for Generative Modeling." In *International Conference on Learning Representations (ICLR)*, 2023.
* **Citekey**: `@lipman2022flow`
* **Core Insight**: Conditional Flow Matching (CFM). Instead of integrating curved Brownian diffusion paths requiring 20–100 denoising steps, flow matching regresses straight probability vector fields, enabling high-fidelity trajectory generation in only 2 to 4 numerical ODE steps.
* **Why Physical AI Must Cite It**: The premier method for executing fast generative action decoders within hard real-time latency budgets on embedded NPUs.
* **Placement**: Section 6.2, under fast ODE generative trajectory decoders.

---

### Chapter 07: Empirical Evaluation (`07-evaluation.qmd`)

#### 1. Butler & Finelli (1993) — The Infeasibility of Quantifying Ultra-Reliability
* **Reference**: Butler, Ricky W., and George B. Finelli. "The Infeasibility of Quantifying the Reliability of Life-Critical Software." *IEEE Transactions on Software Engineering* 19, no. 1 (1993): 3–12.
* **Citekey**: `@butler1993infeasibility`
* **Core Insight**: Mathematical proof that testing life-critical software (such as commercial fly-by-wire or autonomous vehicles requiring a failure rate $\lambda \le 10^{-9}$ failures per hour) is physically impossible via black-box operational testing: it requires testing for $>10^9$ hours (over 114,000 years).
* **Why Physical AI Must Cite It**: Destroys the myth that running a physical robot for a few thousand hours proves safety. Proves to students why formal verification, out-of-band safety governors, and structured safety cases are mandatory.
* **Placement**: Section 7.1 and 7.5, bounding the limits of physical empirical testing.

#### 2. Wald (1945) — Sequential Probability Ratio Test (SPRT)
* **Reference**: Wald, Abraham. "Sequential Tests of Statistical Hypotheses." *The Annals of Mathematical Statistics* 16, no. 2 (1945): 117–186.
* **Citekey**: `@wald1945sequential`
* **Core Insight**: Sequential hypothesis testing. Rather than fixing sample size $N$ beforehand, the Sequential Probability Ratio Test calculates the log-likelihood ratio after each sample, continuing until crossing an upper boundary $A$ (reject $H_0$) or lower boundary $B$ (accept $H_0$). Reduces average required sample size by $30\text{--}50\%$ while strictly guaranteeing Type I ($\alpha$) and Type II ($\beta$) error limits.
* **Why Physical AI Must Cite It**: Robot hardware wear is expensive. SPRT allows automated test rigs to abort failed policies early and certify successful ones with minimal physical test cycles.
* **Placement**: Section 7.5, automated testbed execution and early stopping criteria.

#### 3. Cohen (1988) — Statistical Power Analysis
* **Reference**: Cohen, Jacob. *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed. Lawrence Erlbaum Associates, 1988.
* **Citekey**: `@cohen1988power`
* **Core Insight**: Formalizing Type II error ($\beta$), Statistical Power ($1 - \beta$), and effect size indices (Cohen's $h$ for binomial proportions).
* **Why Physical AI Must Cite It**: In robotics research, authors frequently test a robot 20 times and claim Policy A is superior to Policy B. Cohen’s framework proves that an A/B test with $N=20$ has statistical power $<25\%$, exposing small-sample claims as unscientific noise.
* **Placement**: Section 7.2, under statistical rigor and sample size determination.

#### 4. Hanley & Lippman-Hand (1983) — The Rule of Three
* **Reference**: Hanley, James A., and Abby Lippman-Hand. "If Nothing Goes Wrong, Is Everything All Right? Interpreting Zero Numerators." *JAMA* 249, no. 13 (1983): 1743–1745.
* **Citekey**: `@hanley1983nothing`
* **Core Insight**: Derivation of the non-asymptotic rule for zero-failure trials: when $N$ trials pass with zero failures, the upper $95\%$ confidence limit on the failure probability is approximately:
  $$p_{\text{failure}} \le \frac{3}{N}$$
* **Why Physical AI Must Cite It**: Demystifies the "twenty clean runs" paradox. 20 consecutive zero-failure trials only establish that the failure rate is $\le 15\%$ ($p \le 3/20 = 0.15$), allowing a failure every 6.7 runs.
* **Placement**: Section 7.1 and 7.10, bounding tail risk under zero observed failures.

---

## Part III: Running the Machine (The Real-Time Loop)

```
Part III System Architecture:
├── Chapter 08: Sensor Perception        (Photons to Voxels, Pinhole Optics & BEV Splatting)
├── Chapter 09: Spatial Memory           (Occupancy, TSDF, 3DGS & Dynamic Scene Graphs)
├── Chapter 10: Grounded Intent          (Symbol Grounding, Affordance Bounds & Leases)
├── Chapter 11: Motion Planning          (C-Space, Kinematic Singularities & CHOMP/MPPI)
├── Chapter 12: Safety Enforcement       (Out-of-Band Reflexes, CBFs & Nagumo Sets)
├── Chapter 13: Compute Placement        (Fading Channels, Zero-Copy IPC & Tail Latency)
└── Chapter 14: Supervisory Intervention (Ironies of Automation, Takeover Lag & MRMs)
```

---

### Chapter 08: Sensor Perception (`08-perception.qmd`)

#### 1. Marr (1982) — Vision
* **Reference**: Marr, David. *Vision: A Computational Investigation into the Human Representation and Processing of Visual Information*. W. H. Freeman, 1982.
* **Citekey**: `@marr1982vision`
* **Core Insight**: The three levels of visual analysis (computational theory, representation & algorithm, hardware implementation) and the progressive representation from raw intensity images to primal sketch, 2.5D sketch, and 3D shape models.
* **Why Physical AI Must Cite It**: The philosophical foundation of computer vision, establishing why robots must infer 3D spatial representations from 2D optical intensity projections.
* **Placement**: Section 8.1, introduction to embodied spatial perception.

#### 2. Hartley & Zisserman (2003) — Multiple View Geometry
* **Reference**: Hartley, Richard, and Andrew Zisserman. *Multiple View Geometry in Computer Vision*. 2nd ed. Cambridge University Press, 2003.
* **Citekey**: `@hartley2003multiple`
* **Core Insight**: Projective geometry, camera matrix decomposition ($\mathbf{P} = \mathbf{K}[\mathbf{R} \mid \mathbf{t}]$), the fundamental matrix $\mathbf{F}$, epipolar geometry, and bundle adjustment.
* **Why Physical AI Must Cite It**: The definitive mathematical bible for multi-camera 3D spatial reconstruction and visual odometry.
* **Placement**: Section 8.3, pinhole camera projections and epipolar geometric constraints.

#### 3. Zhang (2000) — Flexible Camera Calibration
* **Reference**: Zhang, Zhengyou. "A Flexible New Technique for Camera Calibration." *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)* 22, no. 11 (2000): 1330–1334.
* **Citekey**: `@zhang2000flexible`
* **Core Insight**: Closed-form camera intrinsic calibration using planar checkerboard patterns viewed from multiple arbitrary orientations, followed by non-linear Levenberg-Marquardt radial distortion optimization.
* **Why Physical AI Must Cite It**: The universal industry-standard algorithm used on every robotic camera rig in production.
* **Placement**: Section 8.3, sensor calibration procedures.

#### 4. Philion & Fidler (2020) — Lift, Splat, Shoot (LSS)
* **Reference**: Philion, Jonah, and Sanja Fidler. "Lift, Splat, Shoot: Sensor-Free Navigation in Bird's-Eye View." In *European Conference on Computer Vision (ECCV)*, 194–210. Springer, 2020.
* **Citekey**: `@philion2020lift`
* **Core Insight**: The Lift-Splat-Shoot architecture. Explicitly lifts 2D camera feature maps into 3D camera ray frustums by predicting categorical depth distributions ($\mathbf{D} \otimes \mathbf{F}$), splats features into a unified 3D voxel grid, and shoots them via GPU prefix-sum (`cumsum`) pooling into a 2D Bird’s-Eye-View (BEV) plane.
* **Why Physical AI Must Cite It**: The dominant modern architectural paradigm for multi-camera 3D perception in autonomous driving and mobile robotics.
* **Placement**: Section 8.4, explaining Figure 8.4 and camera-to-BEV spatial projection.

---

### Chapter 09: Spatial Memory (`09-memory.qmd`)

#### 1. Elfes (1989) — Occupancy Grid Mapping
* **Reference**: Elfes, Alberto. "Using Occupancy Grids for Mobile Robot Perception and Navigation." *Computer* 22, no. 6 (1989): 46–57.
* **Citekey**: `@elfes1989using`
* **Core Insight**: Probabilistic spatial mapping. Represents spatial volume as a grid of discrete cells, updating the occupancy probability of each cell via recursive Bayesian log-odds estimation under sensor beam models.
* **Why Physical AI Must Cite It**: The fundamental representation of geometric space in mobile robotics and obstacle avoidance.
* **Placement**: Section 9.2, metric spatial representations.

#### 2. Curless & Levoy (1996) — Truncated Signed Distance Fields (TSDF)
* **Reference**: Curless, Brian, and Marc Levoy. "A Volumetric Method for Building Complex Models from Range Images." In *ACM SIGGRAPH*, 303–312. 1996.
* **Citekey**: `@curless1996volumetric`
* **Core Insight**: Volumetric TSDF surface reconstruction. Encodes 3D surfaces as the zero-crossing of a voxelized signed distance function $D(\mathbf{x})$, fusing incoming depth maps via weighted running averages.
* **Why Physical AI Must Cite It**: The foundation of real-time dense 3D reconstruction (e.g., KinectFusion) and distance-field collision checking.
* **Placement**: Section 9.3, volumetric memory and signed distance fields.

#### 3. Durrant-Whyte & Bailey (2006) — SLAM Principles
* **Reference**: Durrant-Whyte, Hugh, and Tim Bailey. "Simultaneous Localization and Mapping: Part I & II." *IEEE Robotics & Automation Magazine* 13, no. 2/3 (2006): 99–110 / 108–117.
* **Citekey**: `@durrant2006simultaneous`
* **Core Insight**: Probabilistic formulation of SLAM. Proves that as a robot traverses an unknown environment, landmark covariance estimates become fully correlated, making loop closure detection and global pose-graph optimization the critical drivers of map consistency.
* **Why Physical AI Must Cite It**: Establishes why spatial memory drifts over time without loop closure.
* **Placement**: Section 9.1, the state estimation and mapping problem.

#### 4. Kerbl et al. (2023) — 3D Gaussian Splatting
* **Reference**: Kerbl, Bernhard et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." *ACM Transactions on Graphics (TOG)* 42, no. 4 (2023): 1–14.
* **Citekey**: `@kerbl20233d`
* **Core Insight**: Explicit volumetric scene representation via parameterized 3D Gaussians (position $\boldsymbol{\mu}$, covariance $\boldsymbol{\Sigma} = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$, opacity $\alpha$, and spherical harmonics color). Enables real-time rendering and direct geometric distance queries without neural ray-marching.
* **Why Physical AI Must Cite It**: Revolutionizing spatial memory by replacing slow, implicit NeRFs with real-time, collision-queryable geometric primitives.
* **Placement**: Section 9.4, neural and explicit radiance field representations.

#### 5. Rosinol et al. (2021) — 3D Dynamic Scene Graphs (Kimera)
* **Reference**: Rosinol, Antoni et al. "Kimera: From SLAM to Dynamic 3D Scene Graphs." *IEEE Transactions on Robotics (T-RO)* 37, no. 4 (2021): 1076–1098.
* **Citekey**: `@rosinol2021kimera`
* **Core Insight**: Multi-layer spatial abstraction. Organizes physical environments into a hierarchical graph: Layer 1 (metric mesh), Layer 2 (3D object bounding boxes), Layer 3 (places & freespace topological nodes), Layer 4 (rooms), Layer 5 (buildings).
* **Why Physical AI Must Cite It**: The architectural bridge connecting low-level metric geometry with high-level semantic reasoning for foundation models.
* **Placement**: Section 9.5, semantic scene graphs and hierarchical spatial memory.

---

### Chapter 10: Grounded Intent (`10-intent.qmd`)

#### 1. Harnad (1990) — The Symbol Grounding Problem
* **Reference**: Harnad, Stevan. "The Symbol Grounding Problem." *Physica D: Nonlinear Phenomena* 42, no. 1–3 (1990): 335–346.
* **Citekey**: `@harnad1990symbol`
* **Core Insight**: Symbolic tokens inside a computer have no intrinsic meaning (Searle's Chinese Room argument / dictionary-go-round) unless they are causally grounded in non-symbolic, sensorimotor interactions with the physical world.
* **Why Physical AI Must Cite It**: Explains why disembodied Large Language Models hallucinate physically impossible actions: their tokens lack sensorimotor grounding.
* **Placement**: Section 10.1, framing why language instructions must be grounded in physical affordances.

#### 2. Gibson (1979) — Affordance Theory
* **Reference**: Gibson, James J. *The Ecological Approach to Visual Perception*. Houghton Mifflin, 1979.
* **Citekey**: `@gibson1979ecological`
* **Core Insight**: Ecological perception of affordances. Animals do not perceive raw geometric coordinates; they perceive what the environment *affords* for action (e.g., surfaces afford supporting weight, handles afford grasping), defined relative to the agent's morphology.
* **Why Physical AI Must Cite It**: Establishes that an "affordance" is not merely a 3D coordinate, but a coupled relationship between target geometry, robot gripper kinematics, and contact force limits.
* **Placement**: Section 10.2, defining spatial affordance masks and interaction envelopes.

#### 3. Winograd (1972) — SHRDLU
* **Reference**: Winograd, Terry. *Understanding Natural Language*. Academic Press, 1972.
* **Citekey**: `@winograd1972`
* **Core Insight**: The first system to ground natural language parsing directly in physical manipulation actions in a blocks world.
* **Why Physical AI Must Cite It**: Provides historical context, contrasting SHRDLU’s closed-world brittle symbolic grounding with modern open-vocabulary stochastic foundation models.
* **Placement**: Section 10.1, historical progression of language-directed manipulation.

#### 4. Gray & Cheriton (1989) — Distributed Leases
* **Reference**: Gray, Cary G., and David R. Cheriton. "Leases: An Efficient Fault-Tolerant Mechanism for Distributed File Cache Consistency." In *ACM Symposium on Operating Systems Principles (SOSP)*, 202–210. 1989.
* **Citekey**: `@gray1989leases`
* **Core Insight**: The Lease Primitive. A time-bounded permission contract that automatically expires upon communication silence, guaranteeing cache consistency and preventing deadlock without requiring active cancellation.
* **Why Physical AI Must Cite It**: The conceptual blueprint adapted in Chapter 10 for physical intent leases ($\mathcal{L}_{\text{intent}}$): a physical goal must fail safe automatically on silence.
* **Placement**: Section 10.4, under the formal derivation of expiring intent leases.

---

### Chapter 11: Motion Planning (`11-planning.qmd`)

#### 1. Lozano-Pérez (1983) — Configuration Space
* **Reference**: Lozano-Pérez, Tomás. "Spatial Planning: A Configuration Space Approach." *IEEE Transactions on Computers* C-32, no. 2 (1983): 108–120.
* **Citekey**: `@lozanoperez1983spatial`
* **Core Insight**: Configuration Space ($\mathcal{C}$-space). Transforms a complex articulated physical robot with geometry into a single point in an $n$-dimensional coordinate space, while obstacles are grown into $\mathcal{C}$-space obstacles ($\mathcal{C}_{\text{obs}}$) via Minkowski difference.
* **Why Physical AI Must Cite It**: The fundamental mathematical breakthrough that enables all modern robotic path planning algorithms.
* **Placement**: Section 11.2, introducing kinematic configurations and obstacle boundaries.

#### 2. Kavraki et al. (1996) — Probabilistic Roadmaps (PRM)
* **Reference**: Kavraki, Lydia E. et al. "Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces." *IEEE Transactions on Robotics and Automation* 12, no. 4 (1996): 566–580.
* **Citekey**: `@kavraki1996probabilistic`
* **Core Insight**: Multi-query sampling-based planning. Randomly samples collision-free points in $\mathcal{C}_{\text{free}}$, connects neighboring points via local planners, and constructs a roadmap graph for fast multi-query path queries.
* **Why Physical AI Must Cite It**: Overcomes the curse of dimensionality for high-DOF robotic arms.
* **Placement**: Section 11.3, sampling-based planning paradigms.

#### 3. LaValle (1998) — Rapidly-Exploring Random Trees (RRT)
* **Reference**: LaValle, Steven M. "Rapidly-Exploring Random Trees: A New Tool for Path Planning." Technical Report TR 98-11, Computer Science Dept., Iowa State University, 1998.
* **Citekey**: `@lavalle1998rapidly`
* **Core Insight**: Incremental single-query sampling biased by construction toward large unvisited Voronoi regions of $\mathcal{C}$-space, finding collision-free paths in non-convex environments without building full roadmaps.
* **Why Physical AI Must Cite It**: The universal baseline path search algorithm in robotics.
* **Placement**: Section 11.3, real-time path planning.

#### 4. Ratliff et al. (2009) — CHOMP
* **Reference**: Ratliff, Nathan et al. "CHOMP: Gradient Optimization Techniques for Efficient Motion Planning." In *IEEE International Conference on Robotics and Automation (ICRA)*, 489–494. 2009.
* **Citekey**: `@ratliff2009chomp`
* **Core Insight**: Covariant Hamiltonian Optimization for Motion Planning. Optimizes entire continuous trajectory splines using functional gradient descent, simultaneously minimizing path curvature (acceleration/jerk) and obstacle proximity on precomputed signed distance fields.
* **Why Physical AI Must Cite It**: Explains how coarse, jagged paths are smoothed into kinematically feasible trajectories.
* **Placement**: Section 11.4, continuous trajectory optimization.

#### 5. Schulman et al. (2014) — TrajOpt
* **Reference**: Schulman, John et al. "Motion Planning with Sequential Convex Optimization and Continuous Collision Checking." *The International Journal of Robotics Research (IJRR)* 33, no. 9 (2014): 1251–1270.
* **Citekey**: `@schulman2014motion`
* **Core Insight**: Sequential Convex Programming (SCP) with exact $L_1$ penalty formulation and swept-volume continuous collision checking.
* **Why Physical AI Must Cite It**: Eliminates "collision tunneling" where high-speed waypoints jump through thin barriers between discrete time steps.
* **Placement**: Section 11.4, continuous-time collision checking.

#### 6. Williams et al. (2017) — Model Predictive Path Integral (MPPI)
* **Reference**: Williams, Grady et al. "Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving." *IEEE Transactions on Robotics (T-RO)* 34, no. 6 (2017): 1603–1622.
* **Citekey**: `@williams2017model`
* **Core Insight**: GPU-accelerated Model Predictive Path Integral control. Samples thousands of randomized candidate control trajectories in parallel, weighting rollouts via an exponential free-energy cost, generating agile trajectories through complex non-differentiable dynamics.
* **Why Physical AI Must Cite It**: Bridges classical trajectory optimization and modern GPU-accelerated neural policy execution.
* **Placement**: Section 11.4, parallel trajectory rollout optimization.

---

### Chapter 12: Safety Enforcement (`12-enforcement.qmd`)

#### 1. Ames et al. (2019) — Control Barrier Functions (CBFs)
* **Reference**: Ames, Aaron D. et al. "Control Barrier Functions: Theory and Applications." In *European Control Conference (ECC)*, 3420–3431. IEEE, 2019.
* **Citekey**: `@ames2019control`
* **Core Insight**: Control Barrier Functions. Enforces forward invariance of safe sets $\mathcal{C}$ ($h(x) \ge 0$) by bounding the derivative of the barrier function along system dynamics:
  $$\dot{h}(x, u) \ge -\alpha(h(x))$$
  Solved at $1\text{ kHz}$ via Quadratic Programming (CBF-QP) as a minimum-intervention safety filter on proposed controls.
* **Why Physical AI Must Cite It**: The definitive mathematical tool used to guarantee that learned neural policy proposals never violate physical safety boundaries.
* **Placement**: Section 12.3, formal set invariance and runtime safety filtering.

#### 2. Wabersich & Zeilinger (2021) — Predictive Safety Filters
* **Reference**: Wabersich, Kim P., and Melanie N. Zeilinger. "A Predictive Safety Filter for Learning-Based Control of Constrained Systems." *Automatica* 129 (2021): 109647.
* **Citekey**: `@wabersich2021predictive`
* **Core Insight**: Model Predictive Safety Filter (MPSF). Solves a receding-horizon optimization that monitors candidate learned inputs, intervening only when unmitigated execution would enter a state from which no safe recovery trajectory exists.
* **Why Physical AI Must Cite It**: Allows learned policies to operate aggressively while mathematically guaranteeing recursive feasibility.
* **Placement**: Section 12.4, predictive safety filtering.

#### 3. Mitchell, Bayen, & Tomlin (2005) — Hamilton-Jacobi Reachability
* **Reference**: Mitchell, Ian M., Alexandre M. Bayen, and Claire J. Tomlin. "A Time-Dependent Hamilton-Jacobi Formulation of Reachable Sets for Continuous Dynamic Games." *IEEE Transactions on Automatic Control* 50, no. 7 (2005): 947–957.
* **Citekey**: `@mitchell2005time`
* **Core Insight**: Level-set methods for Hamilton-Jacobi-Isaacs partial differential equations. Computes the maximal backward reachable set (viability kernel) guaranteeing safety under worst-case bounded environmental disturbances.
* **Why Physical AI Must Cite It**: The mathematical gold standard for computing viability kernels under dynamic disturbances.
* **Placement**: Section 12.3, reachability analysis and viability kernels.

#### 4. NTSB (2019) — Uber ATG Fatal Pedestrian Crash Report
* **Reference**: National Transportation Safety Board (NTSB). *Collision Between a Self-Driving Car and a Pedestrian, Tempe, Arizona, March 18, 2018*. Accident Report NTSB/HAR-19/03. Washington, D.C., 2019.
* **Citekey**: `@ntsb2019uber`
* **Core Insight**: Forensic analysis of a fatal autonomous system failure. The perception system repeatedly toggled classification of a pedestrian (unknown $\to$ vehicle $\to$ bicycle), resetting tracking histories to zero on each toggle. Furthermore, emergency braking was programmatically suppressed for 1.2 seconds to prevent erratic maneuvers, delaying mitigation until impact was inevitable.
* **Why Physical AI Must Cite It**: The definitive engineering case study proving that safety enforcement must be out-of-band and never subordinate to learned perception heuristics.
* **Placement**: Section 12.1 and 12.6, real-world failure case study.

---

### Chapter 13: Compute Placement (`13-placement.qmd`)

#### 1. Saltzer, Reed, & Clark (1984) — End-to-End Arguments
* **Reference**: Saltzer, Jerome H., David P. Reed, and David D. Clark. "End-to-End Arguments in System Design." *ACM Transactions on Computer Systems (TOCS)* 2, no. 4 (1984): 277–288.
* **Citekey**: `@saltzer1984end`
* **Core Insight**: The End-to-End Argument. Functions placed at lower layers of a distributed communication system are often redundant or ineffective compared to providing them at the endpoints where complete application context resides.
* **Why Physical AI Must Cite It**: The classic systems philosophy governing what processing must reside on the physical edge robot versus what can be delegated to the cloud.
* **Placement**: Section 13.1, edge-to-cloud functional partitioning.

#### 2. Satyanarayanan (2001) — Cloudlets & Edge Computing
* **Reference**: Satyanarayanan, Mahadev. "Pervasive Computing: Vision and Challenges." *IEEE Personal Communications* 8, no. 4 (2001): 10–17.
* **Citekey**: `@satyanarayanan2001pervasive`
* **Core Insight**: The three-tier compute hierarchy: mobile device $\to$ local edge cloudlet $\to$ distant hyperscale datacenter. Overcomes the speed-of-light latency penalty for interactive cognitive processing.
* **Why Physical AI Must Cite It**: The foundational architectural blueprint for robotic fleet edge-cloud continuum systems.
* **Placement**: Section 13.2, edge compute topology.

#### 3. Dean & Barroso (2013) — The Tail at Scale
* **Reference**: Dean, Jeffrey, and Luiz André Barroso. "The Tail at Scale." *Communications of the ACM* 56, no. 2 (2013): 74–80.
* **Citekey**: `@dean2013tail`
* **Core Insight**: High-percentile latency ($P_{99}$, $P_{99.9}$) in distributed systems. As a system scales, rare latency spikes on individual microservices dominate overall response times.
* **Why Physical AI Must Cite It**: Proves why closed-loop motor control loops cannot tolerate cloud offloading: a $100\text{ ms}$ tail latency spike will cause mechanical instability.
* **Placement**: Section 13.4, tail latency budgets and network jitter.

---

### Chapter 14: Supervisory Intervention (`14-intervention.qmd`)

#### 1. Bainbridge (1983) — Ironies of Automation
* **Reference**: Bainbridge, Lisanne. "Ironies of Automation." *Automatica* 19, no. 6 (1983): 775–779.
* **Citekey**: `@bainbridge1983ironies`
* **Core Insight**: The fundamental paradox of supervisory control: by automating nominal, predictable tasks, the system leaves the human operator with only the rare, catastrophic edge cases that the automation could not resolve—precisely the moments when the human’s skill, situational awareness, and vigilance are lowest.
* **Why Physical AI Must Cite It**: The classic human factors paper explaining why human emergency takeover in autonomous systems is fundamentally fraught.
* **Placement**: Section 14.1, the human-in-the-loop dilemma.

#### 2. Sheridan (1992) — Telerobotics and Supervisory Control
* **Reference**: Sheridan, Thomas B. *Telerobotics, Automation, and Human Supervisory Control*. MIT Press, 1992.
* **Citekey**: `@sheridan1992telerobotics`
* **Core Insight**: Formal levels of automation taxonomy (from manual teleoperation to autonomous execution) and the cognitive modeling of supervisory handovers.
* **Why Physical AI Must Cite It**: Standard reference for designing human-machine authority boundaries.
* **Placement**: Section 14.2, levels of supervisory authority.

#### 3. Parasuraman, Sheridan, & Wickens (2000) — Human-Automation Interaction
* **Reference**: Parasuraman, Raja, Thomas B. Sheridan, and Christopher D. Wickens. "A Model for Types and Levels of Human Interaction with Automation." *IEEE Transactions on Systems, Man, and Cybernetics* 30, no. 3 (2000): 286–297.
* **Citekey**: `@parasuraman2000model`
* **Core Insight**: Four-stage model of human information processing (sensory acquisition, analysis, decision selection, and action execution) and the automation bias phenomenon.
* **Why Physical AI Must Cite It**: Establishes how alarms, takeover prompts, and situation awareness degradations must be engineered.
* **Placement**: Section 14.3, situation awareness lag and cognitive handover.

#### 4. California DMV / Quinn Emanuel (2024) — Cruise SF Pedestrian Dragging Incident
* **Reference**: Quinn Emanuel Urquhart & Sullivan, LLP. *Report to the Special Committee of the Board of Directors of Cruise LLC Regarding the October 2, 2023 San Francisco Accident*. San Francisco, CA, January 2024.
* **Citekey**: `@quinnemanuel2024cruise`
* **Core Insight**: Post-collision supervisory intervention failure. Following an initial collision with a pedestrian struck by another human-driven vehicle, the autonomous vehicle's secondary response pulled over to the curb, dragging the pedestrian who was occluded in the floorpan blind spot.
* **Why Physical AI Must Cite It**: Crucial real-world forensic case study illustrating the catastrophic danger of secondary maneuvers executing without post-collision sensor verification.
* **Placement**: Section 14.4, minimum risk maneuvers (MRM) and secondary incident hazards.

---

## Part IV: Governing the Machine (Operational Trust)

```
Part IV System Architecture:
├── Chapter 15: Fault Verification   (Butler-Finelli Limits, SMT Solvers & STPA)
├── Chapter 16: Assurance Release    (Toulmin Arguments, GSN & ASIL Decomposition)
└── Chapter 17: The Frontier         (Ashby Variety, Morphological Compute & Fleets)
```

---

### Chapter 15: Fault Verification (`15-verification.qmd`)

#### 1. Leveson (2011) — Engineering a Safer World (STAMP / STPA)
* **Reference**: Leveson, Nancy G. *Engineering a Safer World: Systems Thinking Applied to Safety*. MIT Press, 2011.
* **Citekey**: `@leveson2011engineering`
* **Core Insight**: Systems-Theoretic Accident Model and Processes (STAMP) and System-Theoretic Process Analysis (STPA). Accidents in complex cyber-physical systems do not stem merely from physical component failures; they emerge from dysfunctional interactions and unsafe control actions between non-failed software components.
* **Why Physical AI Must Cite It**: The gold standard systems safety methodology for autonomous machines, moving beyond obsolete linear FMEA/FTA methods.
* **Placement**: Section 15.2, hazard analysis and control loop safety constraints.

#### 2. Katz et al. (2017) — Reluplex: Verifying Deep Neural Networks
* **Reference**: Katz, Guy et al. "Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks." In *International Conference on Computer Aided Verification (CAV)*, 97–117. Springer, 2017.
* **Citekey**: `@katz2017reluplex`
* **Core Insight**: Extending the Simplex algorithm to handle non-linear piecewise ReLU activation functions, enabling formal SMT verification of neural network input-output properties (e.g., verifying ACAS Xu collision avoidance networks).
* **Why Physical AI Must Cite It**: Teaches students how formal mathematical solvers can prove that a neural policy will never output dangerous commands within a bounded input polytope.
* **Placement**: Section 15.4, formal neural network verification.

---

### Chapter 16: Assurance Release (`16-release.qmd`)

#### 1. Toulmin (1958) — The Uses of Argument
* **Reference**: Toulmin, Stephen E. *The Uses of Argument*. Cambridge University Press, 1958.
* **Citekey**: `@toulmin1958uses`
* **Core Insight**: The Toulmin Argument Structure. Formalized rational justification: a Claim must be grounded in Evidence (Grounds), authorized by a Warrant (inference rule), supported by Backing, qualified by Modality, and subject to explicit Rebuttals.
* **Why Physical AI Must Cite It**: The philosophical and structural foundation of all structured safety cases and assurance arguments.
* **Placement**: Section 16.1, the anatomy of an engineering safety claim.

#### 2. Kelly (1998) — Goal Structuring Notation (GSN)
* **Reference**: Kelly, Tim P. "Arguing Safety: A Systematic Approach to Managing Safety Cases." PhD diss., Department of Computer Science, University of York, 1998.
* **Citekey**: `@kelly1998arguing`
* **Core Insight**: Formalized Goal Structuring Notation (GSN) as a graphical modeling standard to represent the hierarchy of safety goals, strategies, context, and underlying evidence solutions.
* **Why Physical AI Must Cite It**: The international standard visual syntax used by safety certification authorities worldwide.
* **Placement**: Section 16.2, graphical assurance case construction.

#### 3. Koopman et al. (2020) — UL 4600 & Autonomous Product Safety
* **Reference**: Koopman, Philip. *How Safe Is Safe Enough? Measuring and Predicting Autonomous Vehicle Safety*. Over the Shoulder Press, 2022; and ANSI/UL 4600: *Standard for Safety for the Evaluation of Autonomous Products*. Underwriters Laboratories, 2020.
* **Citekey**: `@koopman2020how`, `@ul4600`
* **Core Insight**: Goal-based safety assurance for autonomous products operating without human drivers. Emphasizes structured safety cases, coverage metrics, unmitigated hazard analysis, and operational design domain (ODD) boundaries.
* **Why Physical AI Must Cite It**: The primary industry standard defining release criteria for Physical AI systems.
* **Placement**: Section 16.3, release readiness and statutory assurance standards.

---

### Chapter 17: The Frontier (`17-frontier.qmd`)

#### 1. Ashby (1956) — Law of Requisite Variety
* **Reference**: Ashby, W. Ross. *An Introduction to Cybernetics*. Chapman & Hall, 1956.
* **Citekey**: `@ashby1956introduction`
* **Core Insight**: The Law of Requisite Variety: "Only variety can destroy variety." A control system can successfully maintain stability only if its internal state variety matches or exceeds the disturbance variety of the external environment:
  $$\mathcal{V}_{\text{system}} \ge \mathcal{V}_{\text{environment}}$$
* **Why Physical AI Must Cite It**: Explains the fundamental theoretical reason why closed-world policies inevitably fail in open-world physical environments.
* **Placement**: Section 17.1, the open-world distribution shift frontier.

#### 2. von Neumann (1966) — Self-Reproducing Automata
* **Reference**: von Neumann, John. *Theory of Self-Reproducing Automata*. Edited by Arthur W. Burks. University of Illinois Press, 1966.
* **Citekey**: `@vonneumann1966theory`
* **Core Insight**: Complexity thresholds for kinematic automata. Below a critical threshold, machines can only build simpler machines; above the threshold, machines can maintain, replicate, and repair themselves.
* **Why Physical AI Must Cite It**: Points to the ultimate scientific frontier of Physical AI: self-healing, self-repairing, and autonomous infrastructure.
* **Placement**: Section 17.4, long-term autonomous physical automata.

#### 3. Pfeifer & Bongard (2006) — Morphological Computation
* **Reference**: Pfeifer, Rolf, and Josh Bongard. *How the Body Shapes the Way We Think: A New View of Intelligence*. MIT Press, 2006.
* **Citekey**: `@pfeifer2006how`
* **Core Insight**: Morphological Computation. Intelligence is not confined to the brain; physical mechanics, compliance, material damping, and geometry perform physical computation, dramatically simplifying neural policy complexity.
* **Why Physical AI Must Cite It**: The grand synthesis of the textbook: true Physical AI is not just a neural network dropped onto a robot; it is the tight, co-designed integration of Body, Nervous System, and Brain.
* **Placement**: Section 17.5, concluding synthesis of the volume.

---

## Master BibTeX Canon (`references-vol4.bib` Additions)

Below is the verified BibTeX block containing the missing canonical entries ready for addition to `books/references-vol4.bib`:

```bibtex
@book{wiener1948cybernetics,
  title = {Cybernetics: Or Control and Communication in the Animal and the Machine},
  author = {Wiener, Norbert},
  year = {1948},
  publisher = {John Wiley \& Sons},
  address = {New York}
}

@book{moravec1988mind,
  title = {Mind Children: The Future of Robot and Human Intelligence},
  author = {Moravec, Hans},
  year = {1988},
  publisher = {Harvard University Press},
  address = {Cambridge, MA}
}

@article{landauer1961,
  title = {Irreversibility and Heat Generation in the Computing Process},
  author = {Landauer, Rolf},
  journal = {IBM Journal of Research and Development},
  volume = {5},
  number = {3},
  pages = {183--191},
  year = {1961},
  doi = {10.1147/rd.53.0183}
}

@article{liu1973scheduling,
  title = {Scheduling Algorithms for Multiprogramming in a Hard-Real-Time Environment},
  author = {Liu, C. L. and Layland, James W.},
  journal = {Journal of the ACM},
  volume = {20},
  number = {1},
  pages = {46--61},
  year = {1973},
  doi = {10.1145/321738.321743}
}

@article{sha1990priority,
  title = {Priority Inheritance Protocols: An Architectural Approach to Real-Time Synchronization},
  author = {Sha, Lui and Rajkumar, Ragunathan and Lehoczky, John P.},
  journal = {IEEE Transactions on Computers},
  volume = {39},
  number = {9},
  pages = {1175--1185},
  year = {1990},
  doi = {10.1109/12.57058}
}

@article{wald1945sequential,
  title = {Sequential Tests of Statistical Hypotheses},
  author = {Wald, Abraham},
  journal = {The Annals of Mathematical Statistics},
  volume = {16},
  number = {2},
  pages = {117--186},
  year = {1945},
  doi = {10.1214/aoms/1177731118}
}

@book{cohen1988power,
  title = {Statistical Power Analysis for the Behavioral Sciences},
  author = {Cohen, Jacob},
  edition = {2nd},
  year = {1988},
  publisher = {Lawrence Erlbaum Associates},
  address = {Hillsdale, NJ}
}

@article{hanley1983nothing,
  title = {If Nothing Goes Wrong, Is Everything All Right? Interpreting Zero Numerators},
  author = {Hanley, James A. and Lippman-Hand, Abby},
  journal = {JAMA},
  volume = {249},
  number = {13},
  pages = {1743--1745},
  year = {1983},
  doi = {10.1001/jama.1983.03330370053031}
}

@article{lozanoperez1983spatial,
  title = {Spatial Planning: A Configuration Space Approach},
  author = {Lozano-P{\'e}rez, Tom{\'a}s},
  journal = {IEEE Transactions on Computers},
  volume = {C-32},
  number = {2},
  pages = {108--120},
  year = {1983},
  doi = {10.1109/TC.1983.1676196}
}

@article{kavraki1996probabilistic,
  title = {Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces},
  author = {Kavraki, Lydia E. and {\v{S}}vestka, Petr and Latombe, Jean-Claude and Overmars, Mark H.},
  journal = {IEEE Transactions on Robotics and Automation},
  volume = {12},
  number = {4},
  pages = {566--580},
  year = {1996},
  doi = {10.1109/70.508439}
}

@techreport{lavalle1998rapidly,
  title = {Rapidly-Exploring Random Trees: A New Tool for Path Planning},
  author = {LaValle, Steven M.},
  institution = {Computer Science Department, Iowa State University},
  number = {TR 98-11},
  year = {1998}
}

@inproceedings{ratliff2009chomp,
  title = {{CHOMP}: Gradient Optimization Techniques for Efficient Motion Planning},
  author = {Ratliff, Nathan and Zucker, Matt and Bagnell, J. Andrew and Srinivasa, Siddhartha},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  pages = {489--494},
  year = {2009},
  doi = {10.1109/ROBOT.2009.5152817}
}

@article{schulman2014motion,
  title = {Motion Planning with Sequential Convex Optimization and Continuous Collision Checking},
  author = {Schulman, John and Duan, Yan and Ho, Alex and Lee, Alex and Aflalo, Jon and Ding, David and Abbeel, Pieter},
  journal = {The International Journal of Robotics Research},
  volume = {33},
  number = {9},
  pages = {1251--1270},
  year = {2014},
  doi = {10.1177/0278364914528132}
}

@article{williams2017model,
  title = {Information-Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving},
  author = {Williams, Grady and Aldrich, Andrew and Theodorou, Evangelos A.},
  journal = {IEEE Transactions on Robotics},
  volume = {34},
  number = {6},
  pages = {1603--1622},
  year = {2017},
  doi = {10.1109/TRO.2018.2865891}
}

@inproceedings{lipman2022flow,
  title = {Flow Matching for Generative Modeling},
  author = {Lipman, Yaron and Chen, Ricky T. Q. and Ben-Hamu, Heli and Nicklas, Maximilian and Le, Matt},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2023}
}

@book{sutton2018reinforcement,
  title = {Reinforcement Learning: An Introduction},
  author = {Sutton, Richard S. and Barto, Andrew G.},
  edition = {2nd},
  year = {2018},
  publisher = {MIT Press},
  address = {Cambridge, MA}
}

@book{ashby1956introduction,
  title = {An Introduction to Cybernetics},
  author = {Ashby, W. Ross},
  year = {1956},
  publisher = {Chapman \& Hall},
  address = {London}
}

@book{vonneumann1966theory,
  title = {Theory of Self-Reproducing Automata},
  author = {von Neumann, John},
  editor = {Burks, Arthur W.},
  year = {1966},
  publisher = {University of Illinois Press},
  address = {Urbana, IL}
}

@book{pfeifer2006how,
  title = {How the Body Shapes the Way We Think: A New View of Intelligence},
  author = {Pfeifer, Rolf and Bongard, Josh},
  year = {2006},
  publisher = {MIT Press},
  address = {Cambridge, MA}
}
```

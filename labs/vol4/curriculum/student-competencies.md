# Volume IV: Physical AI Competency Matrix & Check-Off Card

**Status:** Course design contract for the teaching team. The competencies below describe transferable engineering capabilities and observable physical evidence without restricting the pedagogy to a single board, sensor bus, or arm. The last section shows where each competency is signed off in the current LeRobot, Seeed SO-101, and Arduino UNO Q lab sequence.

---

## The Graduation Standard: The Three-Part Scope Test

To demonstrate mastery of Physical AI systems, a student team must produce a verifiable end-to-end trace:

$$\text{Physical State } (s_t) \longrightarrow \text{Observation } (I_t, q_t) \longrightarrow \text{Learned Proposal } (a_{\text{req}}) \longrightarrow \text{MCU Permission } (a_{\text{enf}}) \longrightarrow \text{Actuation } (a_{\text{meas}}) \longrightarrow \text{New Observation } (I_{t+1}, q_{t+1}) \longrightarrow \text{Revised Decision}$$

The final system must satisfy the three conditions of the book's [scope test](../../../books/vol4/01_boundary/01_boundary.qmd):
1. **A Learned Decision:** A model whose output is conditioned on high-dimensional physical observations, not a hardcoded trajectory.
2. **Consequential Physical Feedback:** Action changes the world, and subsequent decisions must respond to the actual measured state produced by prior motion or disturbance.
3. **Delegated Actuator Authority:** Low-level actuation executes within an independent microcontroller permission boundary capable of vetoing proposals in real time.

---

## The 2×2 Competency Architecture

An engineer who passes this course can **characterize an unfamiliar physical plant, build and optimize models for it, close an autonomous physical feedback loop, and govern delegated authority under fault**.

The two axes separate the **domain of study** (the physical plant vs. the computing architecture) from the **engineering work** (characterize & model vs. control & govern):

| | **Characterize & Model**<br>*(Observe, Measure, Predict, Profile)* | **Control & Govern**<br>*(Act, Enforce, Adapt, Defend)* |
|:---|:---|:---|
| **Physical Embodiment**<br>*(The World, Mechanics, Senses & Actuators)* | **Quadrant A: Measure the Plant**<br>Kinematics, sensing, timing.<br>**A1, A2, A3** | **Quadrant C: Act in the World**<br>Execution, chunking, replanning.<br>**C1, C2, C3** |
| **Computing Architecture**<br>*(The Board, Brain, Models & Software)* | **Quadrant B: Characterize the Brain**<br>Datasets, baselines, edge profiling.<br>**B1, B2, B3** | **Quadrant D: Govern the System**<br>Authority, safety governor, release.<br>**D1, D2, D3** |

---

## Transferable Competency Definitions

### Quadrant A: Measure the Plant (Physical × Characterize)

| ID | Competency | Observable Physical Evidence (Platform-Independent) |
|:---|:---|:---|
| **A1** | **Plant Kinematics & Safe Envelope**<br>Identify physical degrees of freedom, joint limits, motor power routing, homing calibration, and de-energized rest state. | Measured operating workspace envelope, joint range table, and power-off drop trace proving no uncommanded motion upon boot or power loss. |
| **A2** | **Multi-Modal Sensing & Calibration**<br>Acquire and calibrate two distinct physical feedback streams (e.g., vision and joint telemetry); quantify uncertainty, backlash, and sensor loss. | Extrinsic camera calibration ($T_{\text{cam}}^{\text{base}}$), command-versus-settled error distributions, backlash quantification, and a sensor-dropout trial. |
| **A3** | **Feedback Timing & Latency**<br>Align multi-modal observation timestamps; measure physical sensor-to-torque loop delay; detect stale physical measurements. | Synchronized $(I_t, q_t)$ timestamp alignment trace, end-to-end loop latency histogram, and deliberate stale-data rejection trial. |

### Quadrant B: Characterize the Brain (Computing × Characterize)

| ID | Competency | Observable Physical Evidence (Platform-Independent) |
|:---|:---|:---|
| **B1** | **Physical Dataset Engineering**<br>Capture repeatable demonstration episodes with standardized schemas; log multi-tap action telemetry; construct defensible held-out splits. | Replayable demonstration dataset (e.g., LeRobot Dataset v3) logging all 4 action taps (`a_req`, `a_map`, `a_enf`, `a_meas`), dataset card, and leak-free train/val/test splits. |
| **B2** | **Deterministic Baseline Benchmarking**<br>Construct a non-learned scripted or rule-based controller to serve as an objective performance, latency, and reliability reference. | Matched-start physical trials comparing rule-based controller against human teleoperation and learned policies under identical conditions. |
| **B3** | **Edge Model Profiling & Optimization**<br>Train a compact imitation policy (e.g., ACT); quantize/export (ONNX/INT8); profile memory footprint and inference latency against edge deadline budgets. | Pinned model artifact, training/validation loss curves, host-vs-edge numerical parity check, memory footprint profile, and inference execution time distribution. |

### Quadrant C: Act in the World (Physical × Control)

| ID | Competency | Observable Physical Evidence (Platform-Independent) |
|:---|:---|:---|
| **C1** | **Closed-Loop Physical Action**<br>Close the autonomous physical loop on hardware; stream policy proposals to produce consequential, intended physical change in the workspace. | Witnessed autonomous physical task completion driven entirely by on-board model inference without host PC tethering. |
| **C2** | **Action Chunk Dynamics**<br>Evaluate multi-step action chunk horizons ($K$) versus single-step reactive execution; analyze trade-offs between trajectory smoothness and latency drift. | Trajectory tracking comparison across chunk horizons $K$, tracking error vs. horizon curves, and open-loop drift characterization. |
| **C3** | **Disturbance Recovery & Replanning**<br>Detect physical discrepancies (e.g., moved target mid-trajectory) from fresh sensory feedback; adapt action proposals or deliberately abstain. | Matched trials with mid-trajectory object displacement, comparing open-loop continuation (failure) against closed-loop adaptation (recovery) or justified abstention. |

### Quadrant D: Govern the System (Computing × Control)

| ID | Competency | Observable Physical Evidence (Platform-Independent) |
|:---|:---|:---|
| **D1** | **Hardware Authority Routing**<br>Enforce that all actuation flows exclusively through the independent microcontroller permission boundary (`a_req` $\to$ `a_map` $\to$ `a_enf`); prove zero unmonitored host bypass. | Hardware interface and power routing diagram, verified single-path command trace, and proof of blocked host USB direct motor access. |
| **D2** | **Real-Time Safety Governor & Cutoff**<br>Implement velocity clamps, collision envelopes, communication watchdogs, and emergency cutoffs on the MCU; verify safe-state transition with zero backlog. | Injected fault trials (over-speed request, workspace boundary breach, Linux hang/dropped frames) showing immediate MCU clipping or safe shutdown with 0 queued packets. |
| **D3** | **Physical Release Defense**<br>Conduct a statistically frozen held-out disturbance evaluation; compare against baseline; defend a bounded release dossier. | Held-out physical benchmark report across varied initial positions and disturbances, failure mode taxonomy, and formal Physical Release Dossier oral defense. |

---

## The Physical AI Competency Check-Off Card

This card serves as the concrete, observable sign-off sheet for students and instructors at the lab bench:

```markdown
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PHYSICAL AI STATION COMPETENCY CARD                             │
│ Student / Team: ___________________________   Station ID: ____________________________ │
├──────┬────────────────────────────────────────┬─────────────────────────┬──────────────┤
│ ID   │ Competency Description                 │ Required Physical Proof │ Sign-Off     │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ A1: Plant Kinematics & Safe Envelope   │ Workspace envelope,     │ Date: ______ │
│      │                                        │ joint limits, safe rest │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ A2: Multi-Modal Sensing & Calibration  │ Extrinsic camera matrix,│ Date: ______ │
│      │                                        │ settled error plot      │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ A3: Feedback Timing & Latency          │ Timestamp sync trace,   │ Date: ______ │
│      │                                        │ sensor-to-torque delay  │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ B1: Physical Dataset Engineering       │ LeRobot episodes,       │ Date: ______ │
│      │                                        │ 4 action taps logged    │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ B2: Deterministic Baseline Benchmark   │ Scripted reach baseline │ Date: ______ │
│      │                                        │ matched trials report   │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ B3: Edge Model Profiling & Opt.        │ Edge export, latency    │ Date: ______ │
│      │                                        │ budget on Qualcomm MPU  │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ C1: Closed-Loop Physical Action        │ Live autonomous reach   │ Date: ______ │
│      │                                        │ completion on UNO Q     │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ C2: Action Chunk Dynamics              │ Chunk-horizon sweep,    │ Date: ______ │
│      │                                        │ latency/smoothness plot │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ C3: Disturbance Recovery & Replanning  │ Target moved mid-reach, │ Date: ______ │
│      │                                        │ adaptive correction     │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ D1: Hardware Authority Routing         │ Linux➔Bridge➔MCU trace, │ Date: ______ │
│      │                                        │ host bypass verified cut│ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ D2: Real-Time Safety Governor & Cutoff │ MCU over-speed refusal, │ Date: ______ │
│      │                                        │ watchdog timeout cutoff │ Staff: _____ │
├──────┼────────────────────────────────────────┼─────────────────────────┼──────────────┤
│ [ ]  │ D3: Physical Release Defense           │ Frozen held-out trials  │ Date: ______ │
│      │                                        │ & Release Dossier oral  │ Staff: _____ │
└──────┴────────────────────────────────────────┴─────────────────────────┴──────────────┘
```

---

## Where Each Competency Is Signed Off

| Competency | Signed off in | Milestone |
|:---|:---|:---:|
| **A1** Plant Kinematics & Safe Envelope | [Lab 1: Robot Kinematics & Digital Twin](../labs/lab-01-kinematics-and-twin.md) | M1 |
| **A2** Multi-Modal Sensing & Calibration | [Lab 3: Learned Control with VLAs](../labs/lab-03-learned-control.md) | M2 |
| **A3** Feedback Timing & Latency | [Lab 2: From Simulation to Real Hardware](../labs/lab-02-sim-to-real.md) | M1 |
| **B1** Physical Dataset Engineering | [Lab 4: Real-World Data Collection & Fine-Tuning](../labs/lab-04-real-data-and-fine-tuning.md) | M2 |
| **B2** Deterministic Baseline Benchmarking | [Lab 3](../labs/lab-03-learned-control.md) | M2 |
| **B3** Edge Model Profiling & Optimization | [Lab 3](../labs/lab-03-learned-control.md) | M2 |
| **C1** Closed-Loop Physical Action | [Lab 4](../labs/lab-04-real-data-and-fine-tuning.md) | M2 |
| **C2** Action Chunk Dynamics | [Lab 5: Stress-Testing](../labs/lab-05-stress-testing.md) | M3 |
| **C3** Disturbance Recovery & Replanning | [Lab 5](../labs/lab-05-stress-testing.md) | M3 |
| **D1** Hardware Authority Routing | [Lab 6: Safety Governor](../labs/lab-06-safety-governor.md) (first traced in Lab 1, Part 1A) | M4 |
| **D2** Real-Time Safety Governor & Cutoff | [Lab 6](../labs/lab-06-safety-governor.md) | M4 |
| **D3** Physical Release Defense | [Capstone](../labs/lab-capstone-studio.md) | M5 |

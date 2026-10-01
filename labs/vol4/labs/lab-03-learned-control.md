# Lab 3: Learned Control with Vision-Language-Action Models

**Status:** `skeleton`\
**Weeks:** 5–6 (Parts 3A, 3B)\
**Reading:** [Chapter 5: Physical Data](../../../books/vol4/05_data/05_data.qmd), [Chapter 6: Policy Training](../../../books/vol4/06_training/06_training.qmd)\
**Competencies:** `A2`, `B2`, `B3`\
**Milestone:** Contributes to [Milestone 2](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 8, with Lab 4)

---

## Overview

**The physical question.** Can a neural network that learned to manipulate objects in simulation control a real robot? Where does it fail, and is the gap in perception (lighting, camera angle), dynamics (friction, backlash, gravity), timing, or the policy itself?

**Purpose.** You can move the robot with hand-derived equations. Now you let it learn from data instead. Vision-Language-Action models take camera images and a language instruction and output motor commands directly. You run a staff-provided VLA in simulation, calibrate the real camera it will see through, deploy the same model to the hardware, and measure what happens. The gap between simulated and physical performance, the **sim2real gap**, is the central lesson.

**Comes in from Lab 2:** IK baseline on the MCU, sim-vs-real table, RPC bridge, four-tap logging.\
**Goes out to Lab 4:** camera calibration, a sim-trained VLA deployed on the UNO Q, a sim2real gap table with failure causes, and a VLA-versus-IK comparison.\
**If Lab 2 didn't finish:** staff provide reference IK firmware for the MCU.

### Learning outcomes

By the end of this lab, you will be able to:
- Explain how a VLA combines vision, language, and proprioception into actions
- Evaluate a pre-trained VLA in simulation and compare it with your IK baseline
- Calibrate the camera-to-robot transform and measure its accuracy
- Deploy a sim-trained VLA to the UNO Q's Linux MPU and profile it on the board
- Measure the sim2real gap and sort its failures by cause

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

Additional for this lab:
- GPU workstation or cloud instance for simulation: TBD (staff)
- Sim-trained VLA checkpoint (SmolVLA or ACT): TBD (staff)
- Camera calibration target: TBD (staff)
- The Lab 1 simulation environment

<!-- STAFF: this lab depends on a checkpoint trained on simulated
     demonstrations of the course arm, in a simulated scene whose camera
     matches the real one. Decide SmolVLA vs. ACT and who produces it. -->

---

## 2. Protocol

### Part 3A: VLAs in Simulation (Week 5)

**Goal:** understand the model and measure it where it was trained.

1. **Understanding VLA architecture.** TBD (staff)
2. **Running the pre-trained VLA in simulation.** TBD (staff)
3. **VLA versus IK baseline in simulation.** TBD (staff)

**Checkpoint:** simulated success rate for the VLA and the IK baseline on the same targets.

### Part 3B: The VLA on Real Hardware (Week 6)

**Goal:** give the model real eyes, run it on the robot, and measure what breaks.

1. **Camera configuration and calibration.** TBD (staff)
2. **Model export and on-board profiling.** TBD (staff)
3. **Real hardware trials.** TBD (staff)
4. **Measuring the sim2real gap.** TBD (staff)

**Checkpoint:** camera calibration and a sim2real gap table with each failure sorted into perception, dynamics, timing, or distribution shift.

<!-- STAFF: run the model export between sessions so Part 3B is bench time. -->

---

## 3. Disturbance & Failure Tests

- **Camera occlusion.** Block the camera briefly. Pass condition: TBD (staff).
- **Degraded lighting.** Pass condition: TBD (staff).
- **Shifted target.** Pass condition: TBD (staff).

---

## 4. Telemetry Trace

TBD (staff)

<!-- STAFF: likely an inference-latency and stale-observation trace on real
     hardware, using the four action taps. -->

---

## 5. Common Pitfalls

TBD (staff)

---

## 6. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **VLA architecture sketch** | TBD (staff) |
| 2 | **VLA versus IK comparison** | TBD (staff) |
| 3 | **Camera calibration** | TBD (staff) |
| 4 | **Export and on-board profiling** | TBD (staff) |
| 5 | **Real hardware trials** | TBD (staff) |
| 6 | **Sim2real gap table** | TBD (staff) |

*Staff signs off `A2`, `B2`, and `B3` on the team's [Competency Card](../curriculum/student-competencies.md).*

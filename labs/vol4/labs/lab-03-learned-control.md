# Lab 3: Learned Control with Vision-Language-Action Models

**Status:** `skeleton`\
**Weeks:** 5–6 (Parts 3A, 3B)\
**Reading:** [Chapter 5: Physical Data](../../../books/vol4/05_data/05_data.qmd), [Chapter 6: Policy Training](../../../books/vol4/06_training/06_training.qmd), and, before Part 3B, the calibration section of [Chapter 8: Sensor Perception](../../../books/vol4/08_perception/08_perception.qmd)\
**Competencies:** `A2`, `B2`, `B3`\
**Due:** end of Week 6

---

## 1. Overview

**The physical question.** Can a neural network that learned to manipulate objects in simulation control a real robot? Where does it fail, and is the gap in perception (lighting, camera angle), dynamics (friction, backlash, gravity), timing, or the policy itself?

**Purpose.** You can move the robot with hand-derived equations. Now you let it learn from data instead. Vision-Language-Action models take camera images and a language instruction and output motor commands directly. You run a staff-provided VLA in simulation, calibrate the real camera it will see through, deploy the same model to the hardware, and measure what happens. The gap between simulated and physical performance, the **sim2real gap**, is the central lesson.

**Core concepts:**
- **The classical wall:** where hand-written control gives way to learning
- **Vision-Language-Action policies:** one model from pixels and words to motor commands
- **The learned sim2real gap:** perception, dynamics, timing, and distribution shift

**Your digital twin:** the VLA runs in your twin before it touches the arm. Where the twin and the arm disagree is where to look for the sim2real gap.

**Comes in from the previous lab:** IK baseline on the MCU, sim-vs-real table, RPC bridge, four-tap logging.\
**Goes out to the next lab:** camera calibration, a sim-trained VLA deployed on the UNO Q, a sim2real gap table with failure causes, and a VLA-versus-IK comparison.\
**If the previous lab didn't finish:** staff provide reference IK firmware for the MCU.

---

## 2. Learning Objectives

By the end of this lab, you will be able to:
- Explain how a VLA combines vision, language, and proprioception into actions
- Evaluate a pre-trained VLA in simulation and compare it with your IK baseline
- Calibrate the camera-to-robot transform and measure its accuracy
- Deploy a sim-trained VLA to the UNO Q's Linux MPU and profile it on the board
- Measure the sim2real gap and sort its failures by cause

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

**Additional setup:** TBD (staff)

<!-- STAFF: This lab depends on a checkpoint trained on simulated demonstrations of the course arm, in a simulated scene whose camera matches the real one. Decide SmolVLA vs. ACT and who produces it. Run model export between sessions. -->

### Part 3A: VLAs in Simulation (Week 5)

**Goal:** understand the model and measure it where it was trained. **Checkpoint:** simulated success rate for the VLA and the IK baseline on the same targets.

TBD (staff)

<!-- STAFF: expected to cover VLA architecture walkthrough; running the pre-trained VLA in simulation; VLA versus IK in simulation. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

### Part 3B: The VLA on Real Hardware (Week 6)

**Goal:** give the model real eyes, run it on the robot, and measure what breaks. **Checkpoint:** camera calibration and a sim2real gap table.

TBD (staff)

<!-- STAFF: expected to cover camera configuration and calibration; export and on-board profiling; real hardware trials; the sim2real gap table; camera occlusion, degraded lighting, and shifted-target tests. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **VLA architecture sketch** | TBD (staff) |
| 2 | **VLA versus IK comparison** | TBD (staff) |
| 3 | **Camera calibration** | TBD (staff) |
| 4 | **Export and on-board profile** | TBD (staff) |
| 5 | **Sim2real gap table, each failure sorted into perception, dynamics, timing, or distribution shift** | TBD (staff) |

*Staff signs off `A2`, `B2`, and `B3` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. Which category of sim2real failure dominated, and what kind of data would fix it?
2. Where did the hand-derived IK beat the learned policy, and where did the policy win? What does that say about when learning is worth its cost?

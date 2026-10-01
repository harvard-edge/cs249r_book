# Lab 3: Learned Control with Vision-Language-Action Models

**Status:** `skeleton`\
**Weeks:** 5–6\
**Reading:** [Chapter 5: Physical Data](../../../books/vol4/05_data/05_data.qmd), [Chapter 6: Policy Training](../../../books/vol4/06_training/06_training.qmd)\
**Competencies:** `B2`, `B3`\
**Milestone:** Contributes to [Milestone 2](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 8, with Lab 4)

---

## Overview

**The physical question.** Can a neural network that learned to manipulate objects in simulation control a real robot? Where does it fail, and is the gap in perception (lighting, camera angle), dynamics (friction, backlash, gravity), timing, or the policy itself?

**Purpose.** You can move the robot with hand-derived equations. Now you let it learn from data instead. Vision-Language-Action models take camera images and a language instruction and output motor commands directly. You run a staff-provided VLA in simulation, deploy the same model to the real hardware, and measure what happens. The gap between simulated and physical performance, the **sim2real gap**, is the central lesson.

**Comes in from Lab 2:** calibrated station, IK baseline on the MCU, RPC bridge, sim2real table for IK.\
**Goes out to Lab 4:** sim-trained VLA deployed on the UNO Q, sim2real gap table with categorized failures, VLA-versus-IK comparison.

### Learning outcomes

By the end of this lab, you will be able to:
- Explain how a VLA combines vision, language, and proprioception into actions
- Evaluate a pre-trained VLA in simulation across tasks
- Deploy a sim-trained VLA to the UNO Q's Linux MPU and connect it to the MCU over RPC
- Measure and document the sim2real gap quantitatively
- Compare learned control against your IK baseline on the same physical tasks

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md).

Additional for this lab:
- GPU workstation or cloud instance for simulation: TBD (staff)
- Sim-trained VLA checkpoint (SmolVLA or ACT): TBD (staff)
- The Lab 1 simulation environment

<!-- STAFF: this lab depends on a checkpoint trained on simulated demonstrations
     of the course arm. Decide SmolVLA vs. ACT and who produces it. -->

---

## 2. Protocol

### Phase A: VLAs in Simulation (Week 5)

**Goal:** understand the model and measure it where it was trained.

1. **Understanding VLA architecture.** TBD (staff)
2. **Running the pre-trained VLA in simulation.** TBD (staff)
3. **VLA versus IK baseline in simulation.** TBD (staff)

### Phase B: Deploying the VLA to Real Hardware (Week 6)

**Goal:** run the same model on the robot and measure what breaks.

4. **Model export and edge deployment.** TBD (staff)
5. **VLA running on real hardware.** TBD (staff)
6. **Measuring the sim2real gap.** TBD (staff)

<!-- STAFF: Step 6 should end with failures sorted into perception, dynamics,
     timing, and distribution-shift causes. Lab 4 builds on that table. -->

---

## 3. Disturbance & Failure Tests

- **Degraded lighting.** Pass condition: TBD (staff).
- **Shifted target.** Pass condition: TBD (staff).
- **Out-of-distribution prompt.** Pass condition: TBD (staff).

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
| 2 | **Simulation performance** | TBD (staff) |
| 3 | **VLA versus IK comparison** | TBD (staff) |
| 4 | **Export and on-device profiling** | TBD (staff) |
| 5 | **Real hardware trials** | TBD (staff) |
| 6 | **Sim2real gap table** | TBD (staff) |

*Staff signs off `B2` and `B3` on the team's [Competency Card](../curriculum/student-competencies.md).*

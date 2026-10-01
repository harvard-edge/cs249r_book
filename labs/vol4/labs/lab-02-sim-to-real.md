# Lab 2: From Simulation to Real Hardware

**Status:** `skeleton`\
**Weeks:** 4 (Part 2A)\
**Reading:** [Chapter 3: The Cognitive Brain](../../../books/vol4/03_brain/03_brain.qmd), [Chapter 4: The Nervous System](../../../books/vol4/04_nervous/04_nervous.qmd)\
**Competencies:** `A3`\
**Milestone:** Concludes [Milestone 1](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 4)

---

## Overview

**The physical question.** Your simulation said the robot could reach a target precisely. Does the real hardware agree? Where does the sim2real gap come from (backlash, gravity, encoder resolution, latency), and how large is it?

**Purpose.** Your IK worked in simulation. Now you run it on the real robot and discover what simulation got wrong. Servos have backlash, gravity pulls on real mass, and every command crosses a communication link with its own delay. You meet the dual-brain architecture properly for the first time: the Linux MPU that thinks and the STM32 MCU that governs, with the bridge between them.

**Comes in from Lab 1:** IK solver, digital twin, the targets you tested in simulation.\
**Goes out to Lab 3:** IK running on the MCU (your nonlearned baseline), the sim-vs-real table, RPC timing, and four-tap logging.\
**If Lab 1 didn't finish:** staff provide a reference IK solver.

### Learning outcomes

By the end of this lab, you will be able to:
- Run your IK on the STM32 and command the real arm
- Quantify the gap between your simulation and the real hardware
- Measure inter-core RPC latency and align Linux and MCU timestamps
- Read the four action taps and say where a command was changed or lost

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

Additional for this lab:
- Your Lab 1 IK solver and test targets
- Initial state: motor power off, arm at rest

---

## 2. Protocol

### Part 2A: IK on the Real Arm (Week 4)

**Goal:** reach the same targets you reached in simulation, and measure the difference.

1. **IK on the MCU.** TBD (staff)
2. **Real-world reaching and the sim-vs-real table.** TBD (staff)
3. **Inter-core RPC bridge and timing.** TBD (staff)

**Checkpoint:** a sim-vs-real table for your test targets and an RPC timing trace.

<!-- STAFF: decide whether students port IK to C on the STM32 or load their DH
     parameters into a staff-provided IK module. Name the actual tools and
     commands for flashing and commanding the board. -->

---

## 3. Disturbance & Failure Tests

- **Stale MCU telemetry.** Delay MCU telemetry and check that Linux notices. Pass condition: TBD (staff).

---

## 4. Telemetry Trace

First analysis of the [four action taps](station-reference.md#the-four-action-taps). Capture one IK-commanded reach (does `a_meas` converge to `a_enf`?) and a set of reaches with Linux and MCU timestamps (how far apart are the two clocks?). Details: TBD (staff).

---

## 5. Common Pitfalls

TBD (staff)

---

## 6. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **IK reaching on real hardware** | TBD (staff) |
| 2 | **Sim-vs-real comparison table** | TBD (staff) |
| 3 | **RPC bridge latency** | TBD (staff) |
| 4 | **Timestamp alignment trace** | TBD (staff) |

*Staff signs off `A3` on the team's [Competency Card](../curriculum/student-competencies.md).*

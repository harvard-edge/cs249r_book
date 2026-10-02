# Lab 2: From Simulation to Real Hardware

**Status:** `skeleton`\
**Weeks:** 4 (Part 2A)\
**Reading:** [Chapter 4: The Nervous System](../../../books/vol4/04_nervous/04_nervous.qmd)\
**Competencies:** `A3`\
**Due:** end of Week 4

---

## 1. Overview

**The physical question.** Your simulation said the robot could reach a target precisely. Does the real hardware agree? Where does the sim2real gap come from (backlash, gravity, encoder resolution, latency), and how large is it?

**Purpose.** Your IK worked in simulation. Now you run it on the real robot and discover what simulation got wrong. Servos have backlash, gravity pulls on real mass, and every command crosses a communication link with its own delay. You meet the dual-brain architecture properly for the first time: the Linux MPU that thinks and the STM32 MCU that governs, with the bridge between them.

**Core concepts:**
- **The sim-to-real gap:** what a model of the body leaves out
- **Two processors:** one thinks, one governs, and a bridge carries every command between them
- **Time:** every command crosses a link with its own delay, and two clocks must agree

**Your digital twin:** compare the real arm against it target by target, then feed the backlash and latency you measured back into it. It should predict the real arm better at the end of this lab than at the start.

**Comes in from the previous lab:** IK solver, digital twin, the targets you tested in simulation.\
**Goes out to the next lab:** IK running on the MCU (your nonlearned baseline), the sim-vs-real table, RPC timing, and four-tap logging.\
**If the previous lab didn't finish:** staff provide a reference IK solver.

---

## 2. Learning Objectives

By the end of this lab, you will be able to:
- Run your IK on the STM32 and command the real arm
- Quantify the gap between your simulation and the real hardware
- Measure inter-core RPC latency and align Linux and MCU timestamps
- Read the four action taps and say where a command was changed or lost

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

**Additional setup:** TBD (staff)

<!-- STAFF: Decide whether students port IK to C on the STM32 or load their DH parameters into a staff-provided IK module. Name the real tools for flashing and commanding the board. -->

### Part 2A: IK on the Real Arm (Week 4)

**Goal:** reach the same targets you reached in simulation, and measure the difference. **Checkpoint:** a sim-vs-real table and an RPC timing trace.

TBD (staff)

<!-- STAFF: expected to cover IK on the MCU; real-world reaching against the Lab 1 targets; the RPC bridge and timing; a stale-telemetry test. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **IK reaching on real hardware** | TBD (staff) |
| 2 | **Sim-vs-real comparison table** | TBD (staff) |
| 3 | **RPC bridge latency** | TBD (staff) |
| 4 | **Four-tap trace of one reach, with Linux and MCU timestamps aligned** | TBD (staff) |

*Staff signs off `A3` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. Rank the sources of your sim-vs-real error. Which one would you fix first, and is it a modeling problem or a hardware problem?
2. Your command crossed two processors before it reached a motor. What did each one add, and which delay would matter most for a faster robot?

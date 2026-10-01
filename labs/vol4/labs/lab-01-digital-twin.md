# Lab 1: The Digital Twin

**Status:** `skeleton`\
**Weeks:** 1–3 (Parts 1A, 1B, 1C)\
**Reading:** [Chapter 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd), [Chapter 2: The Physical Body](../../../books/vol4/02_body/02_body.qmd), [Chapter 3: The Cognitive Brain](../../../books/vol4/03_brain/03_brain.qmd)\
**Competencies:** `A1`\
**Milestone:** Contributes to [Milestone 1](../curriculum/syllabus.md#7-assessment-milestones-and-grading) (End of Week 4, with Lab 2)

---

## 1. Overview

**The physical question.** What are the physical limits of this body, and how do you command it to a target within them? How do you translate "move the hand to $(x, y, z)$" into joint angles, and where does that math break down?

**Purpose.** On day one you drive the robot, record a motion, replay it, and cut its power, so you know what the machine does before you study how. In week two you measure the body's own limits: how far it travels before it stops, how hot a servo gets holding a load, how far the supply voltage sags when a joint stalls, and how old a joint reading is by the time you use it. Then you build its digital twin: a simulated copy of the arm that runs on kinematics you derive yourself, so you can command it, push it to its limits, and break it without consequence. By the end you know both what the body can do and how to ask it to do it.

**Core concepts:**
- **The causal boundary:** where computation stops being symbolic and becomes physical force
- **The body's limits:** stopping distance, heat, power, and measurement age bound everything the robot can do
- **The digital twin:** a model of the body you can command, test, and break safely
- **Where the model breaks:** workspace edges and singularities

**Comes in from:** nothing; this is the first lab.\
**Goes out to the next lab:** a replayed episode, a safe-state check, the body's measured limits, your IK solver, the digital twin, and a workspace map.\
**If the previous lab didn't finish:** staff provide a reference IK solver.

---

## 2. Learning Objectives

By the end of this lab, you will be able to:
- Operate the station safely: arm it, teleoperate it, and cut motor power
- Measure the body's physical limits: stopping distance, thermal duty, supply sag under load, and measurement age
- Derive the arm's forward kinematics and solve inverse kinematics for a target pose
- Command a digital twin of the course robot with your own IK
- Identify the workspace boundary, singularities, and joint limits

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). In Part 1A the arm runs under the staff baseline governor.

> **Two rules from day one.** If a servo chatters, buzzes, stalls, or overheats, switch off motor power within two seconds and call staff before re-arming. Never connect a computer directly to the servo bus, around the STM32; doing so fails the current milestone.

**Additional setup:** TBD (staff)

<!-- STAFF: Part 1A must be fun and safe: low speed, soft targets, a staff member at the bench, and every motion through the STM32 baseline governor, never the stock USB path. The book does not teach DH or inverse kinematics, so Part 1C needs a short kinematics primer handout. Needs a teleoperation input, a pinned simulator (MuJoCo or PyBullet), and a robot model matching the delivered SO-101. -->

### Part 1A: Meet the Robot (Week 1)

**Goal:** drive the arm, record a motion, replay it, and know how to stop it. **Checkpoint:** a recorded episode that replays, and a safe-state check.

TBD (staff)

<!-- STAFF: expected to cover station tour and cable trace; teleoperation; record and replay a demonstration; cut motor power at a raised pose and confirm nothing moves until re-arm. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

### Part 1B: Measure the Body (Week 2)

**Goal:** measure the physical limits this body will impose on everything you build later. **Checkpoint:** a table of the body's measured limits.

TBD (staff)

<!-- STAFF: expected to cover stopping distance from several speeds; servo temperature while holding a load; supply voltage sag when a joint stalls; the age of a joint reading by the time Linux sees it. The STS3215 servos report temperature, voltage, and load directly. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

### Part 1C: The Digital Twin (Week 3)

**Goal:** build the arm's digital twin and command it with kinematics you derive yourself. **Checkpoint:** IK reaching results in simulation and a workspace map with singularities marked.

TBD (staff)

<!-- STAFF: expected to cover forward kinematics and inverse kinematics from the primer; a staff-provided digital twin; IK-controlled reaching; workspace mapping; an unreachable target; a singular configuration. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **Safe-state check** | TBD (staff) |
| 2 | **Measured limits of the body: stopping distance, thermal duty, supply sag, measurement age** | TBD (staff) |
| 3 | **Forward and inverse kinematics derivation** | TBD (staff) |
| 4 | **IK solver working in simulation** | TBD (staff) |
| 5 | **Workspace map with singularities marked** | TBD (staff) |

*Staff signs off `A1` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. Which of the limits you measured would bind first if the arm moved twice as fast, and what would you change to move it?
2. Your IK found one joint configuration for each target. Where else could the arm have been, and why did the solver pick the one it did?

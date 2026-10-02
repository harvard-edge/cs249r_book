# Lab 1: The Digital Twin

**Status:** `skeleton`\
**Weeks:** 2–3 (Parts 1A, 1B)\
**Reading:** [Chapter 2: The Physical Body](../../../books/vol4/02_body/02_body.qmd), [Chapter 3: The Cognitive Brain](../../../books/vol4/03_brain/03_brain.qmd)\
**Competencies:** `A1`\
**Milestone:** Contributes to [Milestone 1](../curriculum/syllabus.md#7-assessment-milestones-and-grading) (End of Week 4, with Lab 2)

---

## 1. Overview

**The physical question.** What are the physical limits of this body, and how faithful a model of it can you build? How do you translate "move the hand to $(x, y, z)$" into joint angles, and where does that math break down?

**Purpose.** You have driven the robot; now you model it. First you measure the body's own limits: how far it travels before it stops, how hot a servo gets holding a load, how far the supply voltage sags when a joint stalls, and how old a joint reading is by the time you use it. Then you build its digital twin: a simulated copy of the arm that runs on kinematics you derive yourself, so you can command it, push it to its limits, and break it without consequence. The twin is not finished when this lab ends. You will use it in every lab that follows, and improve it each time the real arm shows you what it was missing.

**Core concepts:**
- **The body's limits:** stopping distance, heat, power, and measurement age bound everything the robot can do
- **The digital twin:** a model of the body you can command, test, and break safely, and that you keep improving all term
- **Where the model breaks:** workspace edges and singularities

**Comes in from:** Getting Started: a working station and a replayed episode.\
**Goes out to the next lab:** the body's measured limits, your IK solver, your digital twin, and a workspace map.\
**If a part didn't finish:** staff provide a reference IK solver.

---

## 2. Learning Objectives

By the end of this lab, you will be able to:
- Measure the body's physical limits: stopping distance, thermal duty, supply sag under load, and measurement age
- Derive the arm's forward kinematics and solve inverse kinematics for a target pose
- Command a digital twin of the course robot with your own IK
- Identify the workspace boundary, singularities, and joint limits

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

**Additional setup:** TBD (staff)

<!-- STAFF: The book does not teach DH or inverse kinematics, so Part 1B needs a short kinematics primer handout. Needs a teleoperation input, a pinned simulator (MuJoCo or PyBullet), and a robot model matching the delivered SO-101. -->

### Part 1A: Measure the Body (Week 2)

**Goal:** measure the physical limits this body will impose on everything you build later. **Checkpoint:** a table of the body's measured limits.

TBD (staff)

<!-- STAFF: expected to cover stopping distance from several speeds; servo temperature while holding a load; supply voltage sag when a joint stalls; the age of a joint reading by the time Linux sees it. The STS3215 servos report temperature, voltage, and load directly. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

### Part 1B: Build the Digital Twin (Week 3)

**Goal:** build the arm's digital twin and command it with kinematics you derive yourself. **Checkpoint:** IK reaching results in simulation and a workspace map with singularities marked.

TBD (staff)

<!-- STAFF: expected to cover forward kinematics and inverse kinematics from the primer; a staff-provided digital twin; IK-controlled reaching; workspace mapping; an unreachable target; a singular configuration. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **Measured limits of the body: stopping distance, thermal duty, supply sag, measurement age** | TBD (staff) |
| 2 | **Forward and inverse kinematics derivation** | TBD (staff) |
| 3 | **IK solver working in simulation** | TBD (staff) |
| 4 | **Workspace map with singularities marked** | TBD (staff) |

*Staff signs off `A1` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. Which of the limits you measured would bind first if the arm moved twice as fast, and what would you change to move it?
2. Your IK found one joint configuration for each target. Where else could the arm have been, and why did the solver pick the one it did?

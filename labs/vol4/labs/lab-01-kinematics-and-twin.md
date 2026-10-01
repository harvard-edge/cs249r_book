# Lab 1: Robot Kinematics & Digital Twin

**Status:** `skeleton`\
**Weeks:** 1–3 (Parts 1A, 1B, 1C)\
**Reading:** [Chapter 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd), [Chapter 2: The Physical Body](../../../books/vol4/02_body/02_body.qmd)\
**Competencies:** `A1`\
**Milestone:** Contributes to [Milestone 1](../curriculum/syllabus.md#7-assessment-milestones-and-grading) (End of Week 4, with Lab 2)

---

## 1. Overview

**The physical question.** How do you translate the intent "move the hand to $(x, y, z)$" into the joint angles $q = (q_1, \ldots, q_6)$ that achieve it? What happens at the edge of the workspace, and where does the math break down?

**Purpose.** On day one you drive the robot, record a motion, replay it, and cut its power, so you know what the machine does before you study how. Then you learn the math that makes it move: you derive the forward and inverse kinematics of a 6-DoF arm by hand and validate your equations on a digital twin in simulation. By the end you can send the simulated robot to any reachable target with equations you derived yourself, and you understand *why* each joint moves the way it does instead of calling a library function.

**Comes in from:** nothing; this is the first lab.\
**Goes out to the next lab:** a replayed episode, a safe-state check, the DH parameter table, your IK solver, the digital twin, and a workspace map.\
**If the previous lab didn't finish:** staff provide a reference DH table and IK solver.

---

## 2. Learning Objectives

By the end of this lab, you will be able to:
- Operate the station safely: arm it, teleoperate it, and cut motor power
- Assign Denavit-Hartenberg parameters to a 6-DoF serial manipulator and derive its forward kinematics $T_0^6(q)$
- Compute the Jacobian and solve inverse kinematics for a target end-effector pose
- Build a digital twin of the course robot in a physics simulator and command it with your own IK
- Identify the workspace boundary, singularities, and joint limits

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). In Part 1A the arm runs under the staff baseline governor.

**Additional setup:** TBD (staff)

<!-- STAFF: Part 1A must be fun and safe: low speed, soft targets, a staff member at the bench, and every motion through the STM32 baseline governor, never the stock USB path. Needs a teleoperation input, a pinned simulator (MuJoCo or PyBullet), a robot model matching the delivered SO-101, and a DH worksheet. -->

### Part 1A: Meet the Robot (Week 1)

**Goal:** drive the arm, record a motion, replay it, and know how to stop it. **Checkpoint:** a recorded episode that replays, and a safe-state check.

TBD (staff)

<!-- STAFF: expected to cover station tour and cable trace; teleoperation; record and replay a demonstration; cut motor power at a raised pose and confirm nothing moves until re-arm. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

### Part 1B: Kinematics on Paper (Week 2)

**Goal:** derive the arm's kinematics by hand. **Checkpoint:** the DH table and FK/IK derivation.

TBD (staff)

<!-- STAFF: expected to cover DH parameter assignment; forward kinematics; Jacobian and inverse kinematics. State the DH convention and provide the mechanical drawing. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

### Part 1C: Digital Twin (Week 3)

**Goal:** validate the hand derivation against a simulated robot. **Checkpoint:** IK reaching results and a workspace map with singularities marked.

TBD (staff)

<!-- STAFF: expected to cover build the simulated environment; IK-controlled reaching; workspace mapping; an unreachable target; a singular configuration. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **Safe-state check** | TBD (staff) |
| 2 | **DH parameter table** | TBD (staff) |
| 3 | **Forward kinematics derivation** | TBD (staff) |
| 4 | **IK solver working in simulation** | TBD (staff) |
| 5 | **Singularity identification** | TBD (staff) |
| 6 | **Workspace map** | TBD (staff) |

*Staff signs off `A1` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. Your IK found one joint configuration for each target. Where else could the arm have been, and why did the solver pick the one it did?
2. In Part 1A the arm stopped safely when you cut power. Which part of the system made that true, and what would happen if it were missing?

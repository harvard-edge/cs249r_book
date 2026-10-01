# Lab 1: Robot Kinematics & Digital Twin

**Status:** `skeleton`\
**Weeks:** 1–3 (Parts 1A, 1B, 1C)\
**Reading:** [Chapter 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd), [Chapter 2: The Physical Body](../../../books/vol4/02_body/02_body.qmd)\
**Competencies:** `A1`\
**Milestone:** Contributes to [Milestone 1](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 4, with Lab 2)

---

## Overview

**The physical question.** How do you translate the intent "move the hand to $(x, y, z)$" into the joint angles $q = (q_1, \ldots, q_6)$ that achieve it? What happens at the edge of the workspace, and where does the math break down?

**Purpose.** On day one you drive the robot, record a motion, replay it, and cut its power, so you know what the machine does before you study how. Then you learn the math that makes it move: you derive the forward and inverse kinematics of a 6-DoF arm by hand and validate your equations on a digital twin in simulation. By the end you can send the simulated robot to any reachable target with equations you derived yourself, and you understand *why* each joint moves the way it does instead of calling a library function.

**Comes in from:** nothing; this is the first lab.\
**Goes out to Lab 2:** a replayed episode, a safe-state check, the DH parameter table, your IK solver, the digital twin, and a workspace map.\
**If a part didn't finish:** staff provide a reference DH table and IK solver.

### Learning outcomes

By the end of this lab, you will be able to:
- Operate the station safely: arm it, teleoperate it, and cut motor power
- Assign Denavit-Hartenberg parameters to a 6-DoF serial manipulator and derive its forward kinematics $T_0^6(q)$
- Compute the Jacobian and solve inverse kinematics for a target end-effector pose
- Build a digital twin of the course robot in a physics simulator and command it with your own IK
- Identify the workspace boundary, singularities, and joint limits

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md). In Part 1A the arm runs under the staff baseline governor.

Additional for this lab:
- Teleoperation input (leader arm or gamepad): TBD (staff)
- Simulator and Python environment: TBD (staff)
- Robot model (URDF/MJCF) of the course arm: TBD (staff)
- DH parameter worksheet: TBD (staff)

<!-- STAFF: Part 1A must be fun and safe: low speed, soft targets, a staff
     member at the bench. Every motion goes through the STM32 baseline
     governor; never the stock USB path to the servos. Pick one simulator
     (MuJoCo or PyBullet) and pin its version. The robot model must match the
     delivered SO-101's geometry and joint limits. -->

---

## 2. Protocol

### Part 1A: Meet the Robot (Week 1)

**Goal:** drive the arm, record a motion, replay it, and know how to stop it.

1. **Station tour.** Trace every cable from the power strip to the arm. Find the motor power switch. TBD (staff)
2. **Teleoperate.** TBD (staff)
3. **Record and replay a demonstration.** TBD (staff)
4. **Cut motor power.** With the arm raised, switch off motor power and watch what happens. Restore it and confirm nothing moves until you re-arm. TBD (staff)

**Checkpoint:** a recorded episode that replays on your station, and a safe-state check signed by staff.

### Part 1B: Kinematics on Paper (Week 2)

**Goal:** derive the arm's kinematics by hand.

1. **DH parameter assignment.** TBD (staff)
2. **Forward kinematics derivation.** TBD (staff)
3. **Jacobian and inverse kinematics.** TBD (staff)

**Checkpoint:** completed DH table and FK/IK derivation.

<!-- STAFF: decide the DH convention (standard or modified) and say so here.
     Provide the mechanical drawing students work from. -->

### Part 1C: Digital Twin (Week 3)

**Goal:** validate the hand derivation against a simulated robot.

1. **Build the simulated environment.** TBD (staff)
2. **IK-controlled reaching in simulation.** TBD (staff)
3. **Workspace mapping.** TBD (staff)

**Checkpoint:** IK reaching results in simulation and a workspace map with singularities marked.

---

## 3. Disturbance & Failure Tests

- **Motor power cut at a raised pose (Part 1A).** Pass condition: TBD (staff).
- **Unreachable target (Part 1C).** Command a target outside the workspace. Pass condition: TBD (staff).
- **Singular configuration (Part 1C).** Drive the arm to a configuration where the Jacobian loses rank. Pass condition: TBD (staff).

---

## 4. Telemetry Trace

The Part 1A episode is your first log. Look at it, but you don't analyze it yet; Lab 2 introduces the [four action taps](station-reference.md#the-four-action-taps).

---

## 5. Common Pitfalls

TBD (staff)

---

## 6. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **Safe-state check** | TBD (staff) |
| 2 | **DH parameter table** | TBD (staff) |
| 3 | **Forward kinematics derivation** | TBD (staff) |
| 4 | **IK solver working in simulation** | TBD (staff) |
| 5 | **Singularity identification** | TBD (staff) |
| 6 | **Workspace visualization** | TBD (staff) |

*Staff signs off `A1` on the team's [Competency Card](../curriculum/student-competencies.md).*

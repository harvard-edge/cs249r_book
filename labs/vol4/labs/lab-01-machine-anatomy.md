# Lab 1: Robot Kinematics & Digital Twin

**Status:** `skeleton`\
**Weeks:** 1–2\
**Reading:** [Chapter 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd), [Chapter 2: The Physical Body](../../../books/vol4/02_body/02_body.qmd)\
**Competencies:** `A1`\
**Milestone:** Contributes to [Milestone 1](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 4, with Lab 2)

---

## Overview

**The physical question.** How do you translate the intent "move the hand to $(x, y, z)$" into the joint angles $q = (q_1, \ldots, q_6)$ that achieve it? What happens at the edge of the workspace, and where does the math break down?

**Purpose.** Before you touch the robot, you need to understand the math that makes it move. You derive the forward and inverse kinematics of a 6-DoF serial arm by hand, then validate your equations on a digital twin in simulation. By the end you can command a simulated robot to any reachable target with equations you derived yourself, and you understand *why* each joint moves the way it does rather than calling a library function.

**Comes in from:** nothing; this is the first lab.\
**Goes out to Lab 2:** DH parameter table, IK solver, digital twin, workspace map.

### Learning outcomes

By the end of this lab, you will be able to:
- Assign Denavit-Hartenberg parameters to a 6-DoF serial manipulator and derive its forward kinematics $T_0^6(q)$
- Compute the Jacobian and solve inverse kinematics for a target end-effector pose
- Build a digital twin of the course robot in a physics simulator
- Command the simulated robot to target positions with your own IK
- Identify the workspace boundary, singularities, and joint limits in simulation

---

## 1. Setup

> **Standard bench:** not needed. This lab runs entirely in simulation.

Additional for this lab:
- Simulator and Python environment: TBD (staff)
- Robot model (URDF/MJCF) of the course arm: TBD (staff)
- DH parameter worksheet: TBD (staff)

<!-- STAFF: pick one simulator (MuJoCo or PyBullet) and pin its version. The
     robot model must match the delivered SO-101's geometry and joint limits. -->

---

## 2. Protocol

### Phase A: Pen-and-Paper Kinematics (Week 1)

**Goal:** derive the arm's kinematics by hand.

1. **DH parameter assignment.** TBD (staff)
2. **Forward kinematics derivation.** TBD (staff)
3. **Jacobian and inverse kinematics.** TBD (staff)

<!-- STAFF: decide the DH convention (standard or modified) and say so here.
     Provide the mechanical drawing students work from. -->

### Phase B: Digital Twin in Simulation (Week 2)

**Goal:** validate the hand derivation against a simulated robot.

4. **Build the simulated environment.** TBD (staff)
5. **IK-controlled reaching in simulation.** TBD (staff)
6. **Workspace mapping.** TBD (staff)

---

## 3. Disturbance & Failure Tests

- **Unreachable target.** Command a target outside the workspace. Pass condition: TBD (staff).
- **Singular configuration.** Drive the arm to a configuration where the Jacobian loses rank. Pass condition: TBD (staff).

Optional exploration: TBD (staff)

---

## 4. Telemetry Trace

Not applicable; there is no hardware in this lab. The deliverable is the workspace map and the IK test results.

---

## 5. Common Pitfalls

TBD (staff)

---

## 6. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **DH parameter table** | TBD (staff) |
| 2 | **Forward kinematics derivation** | TBD (staff) |
| 3 | **IK solver working in simulation** | TBD (staff) |
| 4 | **Singularity identification** | TBD (staff) |
| 5 | **Workspace visualization** | TBD (staff) |

*Staff signs off `A1` on the team's [Competency Card](../curriculum/student-competencies.md).*

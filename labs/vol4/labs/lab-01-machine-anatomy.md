# Lab 1: Kinematics & Simulation

**Schedule:** Weeks 1–2 · Milestone 1 due end of Week 4 (with Lab 2)
**Required Textbook Reading:** [Chapter 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd), [Chapter 2: The Physical Body](../../../books/vol4/02_body/02_body.qmd)
**Target Competencies:** `[ ] A1 Plant Mechanics & Safe Envelope`
**Milestone Alignment:** Contributes to [Milestone 1](../curriculum/syllabus.md#sec-milestones) (End of Week 4)

---

## Overview

### Purpose
Before you touch the robot, you need to understand the math that makes it move. This lab teaches you to derive the forward and inverse kinematics of a 6-DoF serial manipulator by hand, then validate your equations by building a digital twin in simulation. By the end, you will be able to command a simulated robot to reach any target position in its workspace using equations you derived yourself — and you will understand *why* each joint must move the way it does, rather than just calling a library function.

### Prerequisites
None — this is the first lab. You should have read Chapters 1–2 and have Python with a simulation environment (MuJoCo or PyBullet) installed on your workstation.

### Learning Outcomes
By the end of this lab, you will be able to:
- Assign Denavit-Hartenberg (DH) parameters to a 6-DoF serial manipulator and derive the forward kinematics transformation $T_0^6(q)$
- Compute the Jacobian and solve the inverse kinematics for a target end-effector pose
- Build a digital twin of the course robot in a physics simulation environment
- Command the simulated robot to reach specified target positions using your derived IK equations
- Identify the workspace boundary, singularities, and mechanical limits in simulation

### What You Will Do
In the first week, you will work through the kinematics on paper: assigning DH parameters to each joint, deriving the forward kinematics chain, computing the Jacobian, and solving the inverse kinematics for a target pose. In the second week, you will build a digital twin of the robot in simulation (MuJoCo or PyBullet), load the URDF model, implement your IK equations in Python, and command the simulated robot to reach target positions. You will map the reachable workspace, identify singularities, and experiment freely — because in simulation, you cannot break anything.

---

### 1. The Physical Question

How do you translate the abstract intent "move the hand to position $(x, y, z)$" into the concrete joint angles $q = (q_1, q_2, \ldots, q_6)$ that achieve it? What happens at the boundaries of the workspace, and where does the math break down?

---

### 2. Setup

**Software Environment:**
- Python 3.10+ with NumPy and SciPy
- Physics simulator: MuJoCo (recommended) or PyBullet
- Staff-provided URDF/MJCF model of the course robot
- Jupyter notebook or Python IDE

**Materials:**
- DH parameter worksheet (provided)
- Graph paper or tablet for hand derivations

---

### 3. Step-by-Step Protocol

#### Phase A: Pen-and-Paper Kinematics (Week 1)

##### Step 1: DH Parameter Assignment
1. Study the mechanical drawing of the 6-DoF robot arm. Identify each joint axis and link.
2. Assign coordinate frames to each joint following the Denavit-Hartenberg convention.
3. Fill in the DH parameter table:

   | Joint $i$ | $\alpha_{i-1}$ | $a_{i-1}$ | $d_i$ | $\theta_i$ |
   |:---:|:---:|:---:|:---:|:---:|
   | 1 | | | | |
   | 2 | | | | |
   | 3 | | | | |
   | 4 | | | | |
   | 5 | | | | |
   | 6 | | | | |

##### Step 2: Forward Kinematics Derivation
1. For each joint, write the homogeneous transformation matrix $A_i$ using the DH parameters.
2. Multiply the chain: $T_0^6 = A_1 \cdot A_2 \cdot A_3 \cdot A_4 \cdot A_5 \cdot A_6$.
3. Verify your result by computing the end-effector position for known joint configurations (e.g., all zeros, all 90°).

##### Step 3: Jacobian and Inverse Kinematics
1. Compute the geometric Jacobian $J(q) \in \mathbb{R}^{6 \times 6}$ relating joint velocities to end-effector velocities.
2. Implement a numerical IK solver using the Jacobian pseudo-inverse:
   $$\Delta q = J(q)^+ \cdot \Delta x$$
3. Identify at least one singular configuration where $\det(J) \approx 0$ and explain why the IK fails there.

#### Phase B: Digital Twin in Simulation (Week 2)

##### Step 4: Build the Simulated Environment
1. Load the staff-provided URDF/MJCF model into your simulator.
2. Verify that the simulated joint limits match the DH parameters you derived.
3. Command each joint individually through its full range. Confirm the simulated motion matches your kinematic model.

##### Step 5: IK-Controlled Reaching in Simulation
1. Implement your IK solver in Python. Given a target $(x, y, z)$, compute the joint angles and command the simulated robot.
2. Test 10 target positions distributed across the workspace. Record success/failure and final positioning error for each.
3. Attempt a target outside the workspace. Document the solver's behavior (divergence, oscillation, or graceful failure).

##### Step 6: Workspace Mapping
1. Systematically sample the reachable workspace by sweeping joint angles and computing forward kinematics.
2. Visualize the workspace boundary as a 3D point cloud or cross-section.
3. Identify and annotate the singularity configurations you found analytically.

---

### 4. Exploration Challenges

These are optional but encouraged — simulation is your sandbox:
- Command the robot to trace a circle or figure-eight in Cartesian space. Plot the resulting trajectory.
- Add a simple obstacle (a box) to the simulation. Can your IK avoid it?
- Compare your analytical IK with the simulator's built-in IK solver. Where do they disagree?

---

### 5. Common Pitfalls & Debugging

* ⚠️ **DH Convention Variants:** There are two DH conventions (standard and modified). Be consistent. The course uses the modified convention (Craig's formulation).
* ⚠️ **Gimbal Lock at Singularities:** When two joint axes align, the Jacobian loses rank. Your solver will oscillate or diverge. Implement a damped least-squares fallback: $\Delta q = J^T(JJ^T + \lambda^2 I)^{-1} \Delta x$.
* ⚠️ **Multiple IK Solutions:** A 6-DoF arm typically has up to 8 IK solutions for a given end-effector pose. Your solver will find *one* — verify it's the physically reasonable one (no joint wrapping).

---

### 6. Sign-Off Criteria

| # | Criterion | Measurable Threshold |
|:---:|:---|:---|
| 1 | **DH parameter table** | All 6 joints correctly parameterized; verified by instructor |
| 2 | **Forward kinematics derivation** | Hand-derived $T_0^6(q)$ matches simulation FK for 3 test configurations within $< 1\text{ mm}$ |
| 3 | **IK solver working in simulation** | ≥ 8/10 target positions reached with $< 5\text{ mm}$ error |
| 4 | **Singularity identification** | At least 1 singular configuration identified and explained |
| 5 | **Workspace visualization** | 3D plot of reachable workspace with annotated boundaries |

*Staff signs off `[ ] A1` on the team's [Competency Card](../curriculum/student-competencies.md).*

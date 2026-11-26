# Lab 3 - Inverse Kinematics

Status: Draft \
When: Week 3 \
Due: end of Week 4 \
Readings: Part III: Running the Machine - Trajectory Planning

---

## 1. Learning Objectives

By the end of this lab, you will be able to:

- Derive the closed-form inverse kinematics of a planar three-link arm by decoupling the wrist, and explain why the problem can have no solution or multiple solutions (elbow-up and elbow-down).
- Determine the reachable workspace of the planar model and the range of end-effector orientations for which a target remains reachable.
- Derive the Jacobian of the planar model, make it square by adding an orientation constraint, and identify its singularities from the determinant.
- Compare methods for inverting the Jacobian (transpose, Moore–Penrose pseudoinverse, damped least squares) and explain how damping keeps the solution stable near singularities.
- Build a numerical damped-least-squares inverse kinematics solver from the robot's URDF using Pinocchio and CasADi, with exact symbolic Jacobians, joint limits, and weighted position and orientation errors.
- Generate embedded C code for the solver and deploy it to a microcontroller, behind the safety governor of Lab 2, to drive the physical arm to end-effector poses in real time.
- Evaluate the solver on the physical arm by separating the error predicted by the solver from the measured error, and analyze the trade-off between number of iterations, accuracy, and computation time.

---
<<<<<<< HEAD

## 2. Implementation Steps

1. Calculate a trigonometry-based 3D inverse kinematics model of the SO-101 arm (You Do - 30 mins).
2. Derive the 3D inverse kinematics model from the D-H forward model (I Do - 20 mins).
3. Implement a numerical inverse kinematics solver using a finite-difference Jacobian approximation (We Do - 50 mins).
4. Compute the workspace map with singularities marked (You Do - 20 mins).
5. Deploy the numerical inverse kinematics solver onto the microcontroller to drive a physical robotic arm to targeted end-effector poses in real time (We Do - 60 mins).

---

## 3. Deliverables

1. Closed-form inverse kinematic equations derived using trigonometry.
2. Workspace map with singularities marked and a written explanation of what a singularity is.
3. Well-commented source code of the numerical inverse kinematics solver deployed to the MCU.
4. Commanded target vs. caliper-measured end-effector position, in mm, for three poses.

---
=======
>>>>>>> e161925 (Add lab 3 and update labs 1-2)

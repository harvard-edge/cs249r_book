# Lab 3 - Inverse Kinematics

Status: Skeleton \
When: Week 3 \
Due: end of Week 4 \
Readings: Part III: Running the Machine - Trajectory Planning

---

## 1. Learning Objectives

By the end of this lab, you will be able to:

- Derive closed-form inverse kinematic equations for a multi-axis robotic arm using geometric decoupling and trigonometry.
- Develop a numerical inverse kinematics solver using a finite-difference Jacobian approximation.
- Compute the workspace and identify singularities.
- Deploy the solver to a microcontroller to drive a physical robotic arm to targeted end-effector poses in real time.

---

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
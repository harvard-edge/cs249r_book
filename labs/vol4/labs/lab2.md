# Lab 2 - The Safety Governor

Status: Skeleton \
When: Week 2 \
Due: end of Week 3 \
Readings: Part III: Running the Machine - Safety Enforcement

---

## 1. Learning Objectives

By the end of this lab, you will be able to:

- Implement a safety mechanism to stop the robot's operations in case of danger.
- Implement a safety governor that protects the robot from infeasible requests.
- Use forward kinematics on the MCU to monitor and bound the real-time end-effector pose during physical robot actuation.
- Validate the governor's per-joint bounds against real actuation.

---

## 2. Implementation Steps

1. Design and implement on the MCU a safety mechanism that stops the robot's operations, and measure its latency (You Do - 30 mins).
2. Design and implement on the MCU a safety governor that protects the robot from infeasible requests (We Do - 45 mins).
3. Deploy the derived forward kinematics equations onto the MCU alongside the safety mechanism and the safety governor to track the physical arm (We Do - 45 mins).
4. Validate the governor's per-joint bounds against real actuation (You Do - 60 mins).

---

## 3. Deliverables

1. Code for the safety mechanism, including a recorded measurement of its trigger latency.
2. Code for the MCU safety governor and its finite state machine.
3. The deployed forward kinematics code script.
4. A telemetry log of the governor's bounds for each joint.

---
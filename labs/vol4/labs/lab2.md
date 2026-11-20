# Lab 2 - The Safety Governor

Status: Draft \
When: Week 2 \
Due: end of Week 3 \
Readings: Part III: Running the Machine - Safety Enforcement

---

## 1. Learning Objectives

By the end of this lab, you will be able to:

- Read and write motor registers over the serial bus protocol 
- Monitor motor temperature and current draw, and detect motors that stop responding
- Implement a safety governor: a latching finite state machine that halts the robot when a servo leaves its safe operating range
- Measure the range of motion of each joint and clamp commanded targets to it
- Enable torque safely and drive the motors using position control
---

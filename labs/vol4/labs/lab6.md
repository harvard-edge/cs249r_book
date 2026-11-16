# Lab 6 - Reducing the Sim2Real Gap: Real-World Data Collection & Policy Fine-Tuning

Status: Skeleton \
When: Week 6 (Part A) & Week 7 (Part B) \
Due: end of Week 8 \
Readings: Part II: Teaching the Machine - Physical Data; Policy Training

---

## 1. Learning Objectives

By the end of this lab, you will be able to:

- Teleoperate the robot to collect a structured demonstration dataset.
- Fine-tune a simulation-trained VLA on a small real-world dataset.
- Quantify how much fine-tuning reduces the sim2real gap.

---

## 2. Implementation Steps

- Part A
  1. Set up the robot for teleoperation (We Do - 60 mins).
  2. Collect 10 episodes of real-world data to be pooled into a shared class dataset (You Do - 60 mins).
  3. Fine-tune the previously deployed VLA policy on the pooled dataset for a few steps (We Do - 30 mins).
  4. Instructor Presentation: Introduction to the Capstone Project (I Do - 30 mins).
- Part B
  1. Deploy the fine-tuned VLA policy provided by the instructor (You Do - 30 mins).
  2. Quantify the differences between the simulation-trained policy (from the previous lab) and the fine-tuned policy by measuring partial completion rates and collision counts (You Do - 60 mins).
  3. Class Brainstorming & Discussion of Capstone Project Ideas (We Do - 90 mins).

---

## 3. Deliverables

- Part A
  1. Collected samples.
  2. Terminal execution log of the fine-tuning process.
- Part B
  1. A side-by-side quantitative comparison of partial completion rates and collision counts between the policy from the previous lab and the fine-tuned policy (both deployed on the robot) in the form of a table and chart. Include a 1-paragraph qualitative analysis explaining why the fine-tuned policy performed differently.
  2. Capstone Project Idea (draft): Submit a half-page document covering the Problem Statement, Proposed Robot/Hardware, and Target VLA Policy.

---
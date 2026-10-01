# Lab 4: Real-World Data Collection & Fine-Tuning

**Status:** `skeleton`\
**Weeks:** 7–8 (Parts 4A, 4B)\
**Reading:** [Chapter 7: Closed-Loop Evaluation](../../../books/vol4/07_evaluation/07_evaluation.qmd), [Chapter 8: Sensor Perception](../../../books/vol4/08_perception/08_perception.qmd), [Chapter 9: Spatial Memory](../../../books/vol4/09_memory/09_memory.qmd)\
**Competencies:** `B1`, `C1`\
**Milestone:** Concludes [Milestone 2](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 8). Capstone proposal due end of Week 8.

---

## 1. Overview

**The physical question.** Can a small amount of real-world data rescue a model trained entirely in simulation? How many demonstrations do you need, and how much of the sim2real gap does fine-tuning close?

**Purpose.** Your sim-trained VLA struggled on real hardware. Now you solve it the way the field does: collect a small set of real demonstrations and fine-tune the simulation-trained model on them. Pre-train cheaply in simulation, then adapt with real data. By the end you close the loop completely, with a fine-tuned VLA running autonomously on the edge and no host tether. Knowing what the loop can do, you propose your capstone task.

**Comes in from the previous lab:** camera calibration, sim-trained VLA on the UNO Q, sim2real gap table.\
**Goes out to the next lab:** real-world dataset and dataset card, fine-tuned VLA, untethered loop, loop-latency breakdown, and your capstone proposal.\
**If the previous lab didn't finish:** staff provide a reference sim-trained checkpoint already deployed on the board.

---

## 2. Learning Objectives

By the end of this lab, you will be able to:
- Teleoperate the robot to collect a structured demonstration dataset with all four action taps
- Fine-tune a simulation-trained VLA on a small real-world dataset
- Quantify how much fine-tuning reduces the sim2real gap
- Run vision, inference, RPC, safety, and actuation as one untethered loop on the edge
- Test whether the policy keeps track of a target it can no longer see

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

**Additional setup:** TBD (staff)

<!-- STAFF: Set the episode count, task protocol, train/val/test split, and dataset-card fields. Pin the LeRobot release and training recipe. Fine-tuning must finish offline between Part 4A and Part 4B. -->

### Part 4A: Real Demonstrations (Week 7)

**Goal:** collect a clean dataset of real demonstrations. **Checkpoint:** an audited dataset and its dataset card. Fine-tuning runs before Week 8.

TBD (staff)

<!-- STAFF: expected to cover teleoperation setup; recording episodes; dataset audit and dataset card. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

### Part 4B: Fine-Tuned & Untethered (Week 8)

**Goal:** measure the improvement and run the loop with no host. **Checkpoint:** the comparison table, an untethered run, and the latency breakdown.

TBD (staff)

<!-- STAFF: expected to cover sim-only versus fine-tuned comparison; untethered loop; latency decomposition; a spatial-memory test (let the policy see the target, then block the camera briefly); reduced-lighting, delayed-camera, and camera-disconnect tests. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **Dataset and dataset card** | TBD (staff) |
| 2 | **Fine-tuning converges** | TBD (staff) |
| 3 | **Sim-only versus fine-tuned comparison** | TBD (staff) |
| 4 | **Untethered loop and success rate** | TBD (staff) |
| 5 | **Loop-latency breakdown** | TBD (staff) |
| 6 | **Four-tap trace of an autonomous reach** | TBD (staff) |
| 7 | **Capstone proposal** | TBD (staff) |

*Staff signs off `B1` and `C1` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. How much of the sim2real gap did real data close, and what is left? Would more demonstrations fix it, or different ones?
2. When you blocked the camera, did the arm remember where the target was? What would the policy need in order to remember?

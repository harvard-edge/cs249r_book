# Lab 4: Real-World Data Collection & Fine-Tuning

**Status:** `skeleton`\
**Weeks:** 7–8 (Parts 4A, 4B)\
**Reading:** [Chapter 7: Closed-Loop Evaluation](../../../books/vol4/07_evaluation/07_evaluation.qmd), [Chapter 8: Sensor Perception](../../../books/vol4/08_perception/08_perception.qmd), [Chapter 9: Spatial Memory](../../../books/vol4/09_memory/09_memory.qmd)\
**Competencies:** `B1`, `C1`\
**Milestone:** Concludes [Milestone 2](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 8). Capstone proposal due end of Week 8.

---

## Overview

**The physical question.** Can a small amount of real-world data rescue a model trained entirely in simulation? How many demonstrations do you need, and how much of the sim2real gap does fine-tuning close?

**Purpose.** Your sim-trained VLA struggled on real hardware. Now you solve it the way the field does: collect a small set of real demonstrations and fine-tune the simulation-trained model on them. Pre-train cheaply in simulation, then adapt with real data. By the end you close the loop completely, with a fine-tuned VLA running autonomously on the edge and no host tether. Knowing what the loop can do, you propose your capstone task.

**Comes in from Lab 3:** camera calibration, sim-trained VLA on the UNO Q, sim2real gap table.\
**Goes out to Lab 5:** real-world dataset and dataset card, fine-tuned VLA, untethered loop, loop-latency breakdown, and your capstone proposal.\
**If Lab 3 didn't finish:** staff provide a reference sim-trained checkpoint already deployed on the board.

### Learning outcomes

By the end of this lab, you will be able to:
- Teleoperate the robot to collect a structured demonstration dataset with all four action taps
- Fine-tune a simulation-trained VLA on a small real-world dataset
- Quantify how much fine-tuning reduces the sim2real gap
- Run vision, inference, RPC, safety, and actuation as one untethered loop on the edge
- Test whether the policy keeps track of a target it can no longer see

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

Additional for this lab:
- Teleoperation input device: TBD (staff)
- Target objects: TBD (staff)
- GPU workstation for fine-tuning: TBD (staff)

---

## 2. Protocol

### Part 4A: Real Demonstrations (Week 7)

**Goal:** collect a clean dataset of real demonstrations.

1. **Teleoperation setup.** TBD (staff)
2. **Record demonstration episodes.** TBD (staff)
3. **Audit the dataset and write the dataset card.** TBD (staff)

**Checkpoint:** an audited dataset and its dataset card. Fine-tuning runs between Week 7 and Week 8.

<!-- STAFF: set the episode count, task protocol, train/val/test split, and
     the dataset-card fields. Pin the LeRobot release and training recipe.
     Fine-tuning must finish offline before Part 4B. -->

### Part 4B: Fine-Tuned & Untethered (Week 8)

**Goal:** measure the improvement and run the loop with no host.

1. **Sim-only versus fine-tuned comparison.** TBD (staff)
2. **Untethered autonomous loop.** TBD (staff)
3. **Latency decomposition.** TBD (staff)
4. **Spatial memory test.** Let the policy see the target, then block the camera for a moment. Does the arm keep going toward where the target was, or lose it? TBD (staff)

**Checkpoint:** the comparison table, an untethered run, and the latency breakdown.

---

## 3. Disturbance & Failure Tests

- **Reduced lighting mid-trial.** Pass condition: TBD (staff).
- **Delayed camera pipeline.** Pass condition: TBD (staff).
- **Camera disconnected mid-reach.** Pass condition: TBD (staff).

---

## 4. Telemetry Trace

Capture all four [action taps](station-reference.md#the-four-action-taps) during an autonomous reach. This is the first trace where `a_req` comes from your own fine-tuned policy rather than a staff checkpoint. Details: TBD (staff).

---

## 5. Common Pitfalls

TBD (staff)

---

## 6. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **Dataset collected** | TBD (staff) |
| 2 | **Dataset card** | TBD (staff) |
| 3 | **Fine-tuning converges** | TBD (staff) |
| 4 | **Sim-only versus fine-tuned comparison** | TBD (staff) |
| 5 | **Untethered loop runs** | TBD (staff) |
| 6 | **Success rate** | TBD (staff) |
| 7 | **Loop latency** | TBD (staff) |
| 8 | **Capstone proposal** | Submitted; format TBD (staff) |

*Staff signs off `B1` and `C1` on the team's [Competency Card](../curriculum/student-competencies.md).*

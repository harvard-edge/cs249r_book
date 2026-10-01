# Lab 4: Real-World Data Collection & Fine-Tuning

**Status:** `skeleton`\
**Weeks:** 7–8\
**Reading:** [Chapter 7: Closed-Loop Evaluation](../../../books/vol4/07_evaluation/07_evaluation.qmd), [Chapter 8: Sensor Perception](../../../books/vol4/08_perception/08_perception.qmd), [Chapter 9: Spatial Memory](../../../books/vol4/09_memory/09_memory.qmd)\
**Competencies:** `B1`, `C1`\
**Milestone:** Concludes [Milestone 2](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 8)

---

## Overview

**The physical question.** Can a small amount of real-world data rescue a model trained entirely in simulation? How many demonstrations do you need, and how much of the sim2real gap does fine-tuning close?

**Purpose.** Your sim-trained VLA struggled on real hardware. Now you solve it the way the field does: collect a small set of real demonstrations and fine-tune the simulation-trained model on them. Pre-train cheaply in simulation, then adapt with real data. By the end you close the loop completely, with a fine-tuned VLA running autonomously on the edge and no host tether.

**Comes in from Lab 3:** sim-trained VLA on the UNO Q, sim2real gap table.\
**Goes out to Lab 5:** real-world dataset and dataset card, fine-tuned VLA, untethered autonomous loop, loop-latency breakdown.

### Learning outcomes

By the end of this lab, you will be able to:
- Teleoperate the robot to collect a structured demonstration dataset with all four action taps
- Fine-tune a simulation-trained VLA on a small real-world dataset
- Quantify how much fine-tuning reduces the sim2real gap
- Run vision, inference, RPC, safety, and actuation as one untethered loop on the edge
- Evaluate autonomous manipulation through repeated physical trials

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md).

Additional for this lab:
- Teleoperation input device: TBD (staff)
- Target objects: TBD (staff)
- GPU workstation for fine-tuning: TBD (staff)

---

## 2. Protocol

### Phase A: Real-World Data Collection & Fine-Tuning (Week 7)

**Goal:** collect a clean dataset and adapt the model with it.

1. **Teleoperation setup.** TBD (staff)
2. **Record demonstration episodes.** TBD (staff)
3. **Fine-tune the sim-trained VLA.** TBD (staff)

<!-- STAFF: set the episode count, task protocol, train/val/test split, and
     the dataset-card fields. Pin the LeRobot release and training recipe. -->

### Phase B: Autonomous Closed-Loop Operation (Week 8)

**Goal:** measure the improvement and run the loop with no host.

4. **Sim-only versus fine-tuned comparison.** TBD (staff)
5. **Untethered autonomous loop.** TBD (staff)
6. **Latency decomposition.** TBD (staff)

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

*Staff signs off `B1` and `C1` on the team's [Competency Card](../curriculum/student-competencies.md).*

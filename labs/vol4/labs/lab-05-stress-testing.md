# Lab 5: Stress-Testing — Prediction, Language & Recovery

**Status:** `skeleton`\
**Weeks:** 9 (Part 5A)\
**Reading:** [Chapter 10: Grounded Intent](../../../books/vol4/10_intent/10_intent.qmd), [Chapter 11: Trajectory Planning](../../../books/vol4/11_planning/11_planning.qmd)\
**Competencies:** `C2`, `C3`\
**Milestone:** Concludes [Milestone 3](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 9)

---

## Overview

**The physical question.** How far ahead can an embodied model act open-loop before drift or a changing world demands a fresh observation? Does language change physical action on the same scene, and how does a closed-loop system adapt when the world moves mid-reach?

**Purpose.** Your fine-tuned VLA reaches targets on its own. Can it handle the unexpected? The world does not hold still while the model thinks. This lab stresses the system along time (how far ahead it can predict), a changing world (what happens when the target moves mid-reach), and language (whether the instruction changes the motion). This is where you learn the difference between a demo that works once and a system that works reliably.

**Comes in from Lab 4:** fine-tuned VLA running an untethered loop on the UNO Q.\
**Goes out to Lab 6:** chosen action horizon, disturbance-recovery traces, language check.\
**If Lab 4 didn't finish:** staff provide a reference fine-tuned checkpoint.

### Learning outcomes

By the end of this lab, you will be able to:
- Benchmark how action-chunk horizon affects success, smoothness, and recovery
- Evaluate whether the policy replans or drifts when the target moves mid-reach
- Show whether a change of instruction changes the trajectory on the same scene
- Quantify the trade-off between open-loop efficiency and closed-loop adaptation

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

Additional for this lab:
- Two-target task arena: TBD (staff)

---

## 2. Protocol

### Part 5A: Stress the Loop (Week 9)

**Goal:** find where the loop breaks in time, under a moving world, and under a changed instruction.

1. **Action-chunk horizon benchmark.** TBD (staff)
2. **Mid-reach physical disturbance.** TBD (staff)
3. **Language check.** One paired-prompt test on the same scene. TBD (staff)

**Checkpoint:** horizon curves and a recorded disturbance recovery.

<!-- STAFF: this is the one single-week lab and the schedule buffer. Keep the
     horizon sweep and the disturbance test; keep the language check short.
     Choose the horizon values, the size and timing of the target shift, and
     the number of trials each needs. -->

---

## 3. Disturbance & Failure Tests

- **Target removed mid-reach.** Pass condition: TBD (staff).
- **Target moved outside the envelope.** Pass condition: TBD (staff).

---

## 4. Telemetry Trace

A successful disturbance recovery: the shift event, the change in `a_req` after the fresh observation, how the governor smooths `a_enf`, and the final settled position. Details: TBD (staff).

---

## 5. Common Pitfalls

TBD (staff)

---

## 6. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **Horizon analysis report** | TBD (staff) |
| 2 | **Live disturbance recovery** | TBD (staff) |
| 3 | **Language check** | TBD (staff) |

*Staff signs off `C2` and `C3` on the team's [Competency Card](../curriculum/student-competencies.md).*

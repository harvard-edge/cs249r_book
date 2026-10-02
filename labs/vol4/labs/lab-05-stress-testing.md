# Lab 5: Stress-Testing — Prediction, Language & Recovery

**Status:** `skeleton`\
**Weeks:** 9 (Part 5A)\
**Reading:** [Chapter 10: Grounded Intent](../../../books/vol4/10_intent/10_intent.qmd), [Chapter 11: Trajectory Planning](../../../books/vol4/11_planning/11_planning.qmd)\
**Competencies:** `C2`, `C3`\
**Milestone:** Concludes [Milestone 3](../curriculum/syllabus.md#7-assessment-milestones-and-grading) (End of Week 9)

---

## 1. Overview

**The physical question.** How far ahead can an embodied model act open-loop before drift or a changing world demands a fresh observation? Does language change physical action on the same scene, and how does a closed-loop system adapt when the world moves mid-reach?

**Purpose.** Your fine-tuned VLA reaches targets on its own. Can it handle the unexpected? The world does not hold still while the model thinks. This lab stresses the system along time (how far ahead it can predict), a changing world (what happens when the target moves mid-reach), and language (whether the instruction changes the motion). This is where you learn the difference between a demo that works once and a system that works reliably.

**Core concepts:**
- **Open loop versus closed loop:** how far ahead to commit before looking again
- **Recovery:** replanning when the world changes mid-action
- **Grounded language:** whether an instruction changes the physical motion

**Your digital twin:** rehearse each moving-target trial in the twin first, then run it on the arm. Does the twin predict where recovery fails?

**Comes in from the previous lab:** fine-tuned VLA running an untethered loop on the UNO Q.\
**Goes out to the next lab:** chosen action horizon, disturbance-recovery traces, language check.\
**If the previous lab didn't finish:** staff provide a reference fine-tuned checkpoint.

---

## 2. Learning Objectives

By the end of this lab, you will be able to:
- Benchmark how action-chunk horizon affects success, smoothness, and recovery
- Evaluate whether the policy replans or drifts when the target moves mid-reach
- Show whether a change of instruction changes the trajectory on the same scene
- Quantify the trade-off between open-loop efficiency and closed-loop adaptation

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). The arm runs under the staff baseline governor.

**Additional setup:** TBD (staff)

<!-- STAFF: This is the one single-week lab and the schedule buffer. Keep the horizon sweep and the disturbance test; keep the language check short. Choose horizon values, the size and timing of the target shift, and trial counts. Needs a two-target arena. -->

### Part 5A: Stress the Loop (Week 9)

**Goal:** find where the loop breaks in time, under a moving world, and under a changed instruction. **Checkpoint:** horizon curves and a recorded disturbance recovery.

TBD (staff)

<!-- STAFF: expected to cover action-chunk horizon benchmark; mid-reach disturbance; one paired-prompt language check; target-removed and target-out-of-envelope tests. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **Horizon analysis: success and smoothness across horizons** | TBD (staff) |
| 2 | **Live disturbance recovery** | TBD (staff) |
| 3 | **Four-tap trace of the recovery** | TBD (staff) |
| 4 | **Language check** | TBD (staff) |

*Staff signs off `C2` and `C3` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. What horizon would you ship, and what did you give up to get it?
2. When the target moved, what had to happen, and in what order, for the arm to recover? Where in your loop would that chain break first?

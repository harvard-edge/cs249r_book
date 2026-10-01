# Lab 5: Stress-Testing — Prediction, Language & Recovery

**Status:** `skeleton`\
**Weeks:** 9\
**Reading:** [Chapter 10: Grounded Intent](../../../books/vol4/10_intent/10_intent.qmd), [Chapter 11: Trajectory Planning](../../../books/vol4/11_planning/11_planning.qmd)\
**Competencies:** `C2`, `C3`\
**Milestone:** Concludes [Milestone 3](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 9)

---

## Overview

**The physical question.** How far ahead can an embodied model act open-loop before drift or a changing world demands a fresh observation? Does language change physical action on the same scene, and how does a closed-loop system adapt when the world moves mid-reach?

**Purpose.** Your fine-tuned VLA reaches targets on its own. Can it handle the unexpected? The world does not hold still while the model thinks. This lab stresses the system along three axes: how far ahead it can predict (action horizons), whether language redirects its behavior (language conditioning), and what happens when the scene changes mid-action (disturbance recovery). This is where you learn the difference between a demo that works once and a system that works reliably.

**Comes in from Lab 4:** fine-tuned VLA running an untethered loop on the UNO Q.\
**Goes out to Lab 6:** chosen action horizon, language-conditioning evidence, disturbance-recovery traces.

### Learning outcomes

By the end of this lab, you will be able to:
- Benchmark how action-chunk horizon affects success, smoothness, and recovery
- Show that language prompts produce measurably different trajectories on the same scene
- Evaluate whether the policy replans or drifts when the target moves mid-reach
- Quantify the trade-off between open-loop efficiency and closed-loop adaptation

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md).

Additional for this lab:
- Two-target task arena: TBD (staff)

---

## 2. Protocol

### Phase A: Three Experiments (Week 9)

**Goal:** stress the loop along time, language, and a changing world.

1. **Action-chunk horizon benchmark.** TBD (staff)
2. **Language-conditioning evaluation.** TBD (staff)
3. **Mid-reach physical disturbance.** TBD (staff)

<!-- STAFF: choose the horizon values to compare, the prompts, the size and
     timing of the target shift, and how many trials each needs. -->

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
| 2 | **Language-conditioning proof** | TBD (staff) |
| 3 | **Live disturbance recovery** | TBD (staff) |

*Staff signs off `C2` and `C3` on the team's [Competency Card](../curriculum/student-competencies.md).*

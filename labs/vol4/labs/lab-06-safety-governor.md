# Lab 6: Safety Governor — Build It, Then Break It

**Status:** `skeleton`\
**Weeks:** 10–11 (Parts 6A, 6B)\
**Reading:** [Chapter 12: Safety Enforcement](../../../books/vol4/12_enforcement/12_enforcement.qmd), [Chapter 13: Silicon Placement](../../../books/vol4/13_placement/13_placement.qmd), [Chapter 14: Supervisory Intervention](../../../books/vol4/14_intervention/14_intervention.qmd), [Chapter 15: Adversarial Verification](../../../books/vol4/15_verification/15_verification.qmd)\
**Competencies:** `D1`, `D2`\
**Milestone:** Concludes [Milestone 4](../curriculum/syllabus.md#7-assessment-milestones-and-grading) (End of Week 11). Capstone task frozen by end of Week 10.

---

## 1. Overview

**The physical question.** Can an independent microcontroller keep the arm safe even when the Linux brain crashes, lies, or goes silent, and can you **prove** it by trying to break it?

**Purpose.** Since week 1 the arm has run under a staff-written baseline governor, and you have trusted it without looking inside. Now safety is the subject. In the first week you replace that governor with your own: speed limits, a workspace geofence, and joint limits, and you test why it belongs on the MCU rather than in Linux. In the second week you try to defeat it by freezing the brain, corrupting packets, and cutting power. If you can't break it, you can trust it.

**Comes in from the previous lab:** a working, disturbance-tested autonomous loop.\
**Goes out to the next lab:** your governor firmware, watchdog, and fault-injection evidence.\
**If the previous lab didn't finish:** staff provide a reference autonomous loop.

---

## 2. Learning Objectives

By the end of this lab, you will be able to:
- Implement deterministic real-time safety barriers on a microcontroller
- Explain, with a measurement, why the governor runs on the MCU and not on Linux
- Prove that no software or electrical path reaches the actuators without MCU permission
- Design and run adversarial fault-injection tests, including a watchdog that purges stale commands
- Produce telemetry evidence of active safety intervention (`a_enf` ≠ `a_req`)

---

## 3. Instructions

> **Standard bench:** see the [Station Reference Card](station-reference.md). Run the autonomous loop from Labs 4–5.

**Additional setup:** TBD (staff)

<!-- STAFF: Fix the speed limit, geofence height, and joint limits for the reference station, and decide whether students write the governor from scratch or extend the baseline one. Needs the STM32 toolchain and fault-injection tools. -->

### Part 6A: Build the Governor (Week 10)

**Goal:** your governor clamps or vetoes every unsafe proposal. **Checkpoint:** your governor running, a veto log, and the placement result.

TBD (staff)

<!-- STAFF: expected to cover program the STM32 permission barrier; inject out-of-bounds proposals; a placement test (run the same speed clamp in Linux and on the MCU, then freeze Linux); verify the single-path hardware boundary. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

### Part 6B: Break the Governor (Week 11)

**Goal:** the governor holds under every fault you can devise. **Checkpoint:** fault-injection evidence and a zero-backlog trace.

TBD (staff)

<!-- STAFF: expected to cover communication watchdog; host freeze; queue backlog and buffer flush; sensor dropout and packet corruption; motor power cutoff under load; noisy proposals; full-speed drive toward the table. Write steps with real commands, what to record, the expected result, and pitfalls from the Student Zero run. -->

---

## 4. Deliverables

| # | Evidence | Threshold |
|:---:|:---|:---|
| 1 | **Speed clamp and table geofence** | TBD (staff) |
| 2 | **Placement test** | TBD (staff) |
| 3 | **Single command path** | TBD (staff) |
| 4 | **Firmware review** | TBD (staff) |
| 5 | **Veto trace showing `a_enf` ≠ `a_req`** | TBD (staff) |
| 6 | **Watchdog timeout and zero backlog on resume** | TBD (staff) |
| 7 | **Corrupt packet rejection** | TBD (staff) |
| 8 | **Motor power cutoff** | TBD (staff) |

*Staff signs off `D1` and `D2` on the team's [Competency Card](../curriculum/student-competencies.md).*

---

## 5. Reflection

1. Which attack came closest to getting past your governor, and what does that say about what a governor can and cannot guarantee?
2. You ran the same clamp in Linux and on the MCU. Using your placement result, argue where each safety function in your system should live.

# Lab 6: Safety Governor — Build It, Then Break It

**Status:** `skeleton`\
**Weeks:** 10–11 (Parts 6A, 6B)\
**Reading:** [Chapter 12: Safety Enforcement](../../../books/vol4/12_enforcement/12_enforcement.qmd), [Chapter 13: Silicon Placement](../../../books/vol4/13_placement/13_placement.qmd), [Chapter 14: Supervisory Intervention](../../../books/vol4/14_intervention/14_intervention.qmd), [Chapter 15: Adversarial Verification](../../../books/vol4/15_verification/15_verification.qmd), [Chapter 16: Safe Release](../../../books/vol4/16_release/16_release.qmd)\
**Competencies:** `D1`, `D2`\
**Milestone:** Concludes [Milestone 4](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 11). Capstone task frozen by end of Week 10.

---

## Overview

**The physical question.** Can an independent microcontroller keep the arm safe even when the Linux brain crashes, lies, or goes silent, and can you **prove** it by trying to break it?

**Purpose.** Since week 1 the arm has run under a staff-written baseline governor, and you have trusted it without looking inside. Now safety is the subject. In the first week you replace that governor with your own: speed limits, a workspace geofence, and joint limits, and you test why it belongs on the MCU rather than in Linux. In the second week you try to defeat it by freezing the brain, corrupting packets, and cutting power. If you can't break it, you can trust it.

**Comes in from Lab 5:** a working, disturbance-tested autonomous loop.\
**Goes out to the Capstone:** your governor firmware, watchdog, and fault-injection evidence.\
**If Lab 5 didn't finish:** staff provide a reference autonomous loop.

### Learning outcomes

By the end of this lab, you will be able to:
- Implement deterministic real-time safety barriers on a microcontroller
- Explain, with a measurement, why the governor runs on the MCU and not on Linux
- Prove that no software or electrical path reaches the actuators without MCU permission
- Design and run adversarial fault-injection tests, including a watchdog that purges stale commands
- Produce telemetry evidence of active safety intervention (`a_enf` ≠ `a_req`)

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md), with the autonomous loop from Labs 4–5 running.

Additional for this lab:
- STM32 firmware project and toolchain: TBD (staff)
- Fault-injection tools: TBD (staff)

---

## 2. Protocol

### Part 6A: Build the Governor (Week 10)

**Goal:** your governor clamps or vetoes every unsafe proposal.

1. **Program the STM32 permission barrier.** TBD (staff)
2. **Inject malicious and out-of-bounds proposals.** TBD (staff)
3. **Placement test.** Run the same speed clamp in Linux and on the MCU, then freeze Linux. Which one still protects the arm? TBD (staff)
4. **Verify the single-path hardware boundary.** TBD (staff)

**Checkpoint:** your governor running, a veto log, and the placement result.

<!-- STAFF: fix the speed limit, geofence height, and joint limits for the
     reference station, and decide whether students write the governor from
     scratch or extend the baseline one. -->

### Part 6B: Break the Governor (Week 11)

**Goal:** the governor holds under every fault you can devise.

1. **Communication watchdog.** TBD (staff)
2. **Host freeze.** TBD (staff)
3. **Queue backlog and buffer flush.** TBD (staff)
4. **Sensor dropout and packet corruption.** TBD (staff)
5. **Motor power cutoff under load.** TBD (staff)

**Checkpoint:** fault-injection evidence and a zero-backlog trace.

---

## 3. Disturbance & Failure Tests

Part 6B is the disturbance test. Two additional stress tests:

- **Noisy proposals.** Pass condition: TBD (staff).
- **Full-speed drive toward the table.** Pass condition: TBD (staff).

---

## 4. Telemetry Trace

Two traces using the [four action taps](station-reference.md#the-four-action-taps): a clamping event showing `a_enf` ≠ `a_req`, and a watchdog timeout showing the freeze, the cutoff, and no stale command on resume. Details: TBD (staff).

---

## 5. Common Pitfalls

TBD (staff)

---

## 6. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **Speed clamp** | TBD (staff) |
| 2 | **Table geofence** | TBD (staff) |
| 3 | **Placement test** | TBD (staff) |
| 4 | **Single command path** | TBD (staff) |
| 5 | **Firmware review** | TBD (staff) |
| 6 | **Veto telemetry plot** | TBD (staff) |
| 7 | **Watchdog timeout** | TBD (staff) |
| 8 | **Zero backlog on resume** | TBD (staff) |
| 9 | **Corrupt packet rejection** | TBD (staff) |
| 10 | **Motor power cutoff** | TBD (staff) |

*Staff signs off `D1` and `D2` on the team's [Competency Card](../curriculum/student-competencies.md).*

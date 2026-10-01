# Lab 2: From Simulation to Real Hardware

**Status:** `skeleton`\
**Weeks:** 3–4\
**Reading:** [Chapter 3: The Cognitive Brain](../../../books/vol4/03_brain/03_brain.qmd), [Chapter 4: The Nervous System](../../../books/vol4/04_nervous/04_nervous.qmd)\
**Competencies:** `A2`, `A3`, `D1`\
**Milestone:** Concludes [Milestone 1](../curriculum/syllabus.md#4-assessment-milestones-and-grading) (End of Week 4)

---

## Overview

**The physical question.** Your simulation said the robot could reach a target precisely. Does the real hardware agree? Where does the sim2real gap come from (backlash, gravity, encoder resolution, latency), and how large is it?

**Purpose.** Your IK worked in simulation. Now you move to the real robot and discover what simulation got wrong. Servos have backlash, gravity pulls on real mass, communication has latency, and power rails can brown out. You learn to characterize, calibrate, and control a physical system, and you meet the dual-brain architecture for the first time: the Linux MPU that thinks and the STM32 MCU that governs.

**Comes in from Lab 1:** DH parameter table, IK solver, digital twin.\
**Goes out to Lab 3:** wired and calibrated station, IK running on the MCU, camera extrinsics, working RPC bridge, four-tap logging, sim2real comparison table.

### Learning outcomes

By the end of this lab, you will be able to:
- Wire a station with isolated logic and motor power and verify its safe state
- Run your IK on the STM32 and command the real arm
- Calibrate the camera-to-robot transform and measure its accuracy
- Measure inter-core RPC latency between the Linux MPU and the STM32
- Quantify the gap between your simulation and the real hardware
- Verify that the system fails safely under power loss and communication failure

---

## 1. Setup

> **Standard bench:** see the [Station Reference Card](station-reference.md).

Additional for this lab:
- Calibration target: TBD (staff)
- Your Lab 1 IK solver
- Initial state: motor power off, arm at rest

---

## 2. Protocol

### Phase A: Wiring, Enumeration & IK on Hardware (Week 3)

**Goal:** prove the authority path, then reach real targets with your own IK.

1. **Physical authority and power audit.** TBD (staff)
2. **Servo enumeration and mechanical envelope.** TBD (staff)
3. **IK on the MCU, real-world reaching.** TBD (staff)

<!-- STAFF: decide whether students port IK to C on the STM32 or load their DH
     parameters into a staff-provided IK module. Name the actual tools and
     commands for flashing and commanding the board. -->

### Phase B: Sensing, Calibration & Safety (Week 4)

**Goal:** bring up the camera and the bridge, and confirm safe failure.

4. **Camera configuration and calibration.** TBD (staff)
5. **Inter-core RPC bridge.** TBD (staff)
6. **Failure and safe-state tests.** TBD (staff)

---

## 3. Disturbance & Failure Tests

- **Motor power cut at an elevated pose.** Pass condition: TBD (staff).
- **Re-power without re-arm.** Pass condition: TBD (staff).
- **Camera occlusion.** Pass condition: TBD (staff).
- **Stale MCU telemetry.** Pass condition: TBD (staff).

---

## 4. Telemetry Trace

First use of the [four action taps](station-reference.md#the-four-action-taps). Capture one IK-commanded reach (mechanical: does `a_meas` converge to `a_enf`?) and a set of reaches with camera and MCU timestamps (temporal: how far apart are the two clocks?). Details: TBD (staff).

---

## 5. Common Pitfalls

TBD (staff)

---

## 6. Sign-Off

| # | Criterion | Threshold |
|:---:|:---|:---|
| 1 | **Authority route proof** | TBD (staff) |
| 2 | **Servo enumeration** | TBD (staff) |
| 3 | **IK reaching on real hardware** | TBD (staff) |
| 4 | **Sim2real comparison table** | TBD (staff) |
| 5 | **Camera calibration** | TBD (staff) |
| 6 | **RPC bridge latency** | TBD (staff) |
| 7 | **Safe-state verification** | TBD (staff) |
| 8 | **Sensor dropout handled** | TBD (staff) |

*Staff signs off `A2`, `A3`, and `D1` on the team's [Competency Card](../curriculum/student-competencies.md).*

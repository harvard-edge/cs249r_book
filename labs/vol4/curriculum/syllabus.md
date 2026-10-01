# Master Course Syllabus: Physical AI Systems Studio

**Course Title:** Physical AI Systems: Machine Learning Systems That Sense and Act\
**Course Code:** MLSYS-401 / ETH-PAI / SEAS-CS249r\
**Term:** 14-Week Laboratory Seminar & Project Studio\
**Instructors:** Prof. Vijay Janapa Reddi (Harvard University) & Dr. Andrea (ETH Zurich / Lab Director)\
**Primary Textbook:** [MLSysBook Volume IV: Physical AI Systems](https://mlsysbook.ai/vol4/)\
**Studio Repository:** [github.com/harvard-edge/MLSysBook-vol4-labs](https://github.com/harvard-edge/MLSysBook-vol4-labs)

---

## 1. Course Overview

This course teaches how to build, deploy, evaluate, and govern an embodied machine. Teams take a Vision-Language-Action policy (**SmolVLA** or **ACT**), run it on the Qualcomm Linux side of an **Arduino UNO Q**, and delegate the right to move a **Seeed SO-101** arm to an independent **STM32** safety governor. The [studio landing page](../README.md) introduces the system; the [Station Reference Card](../labs/station-reference.md) specifies the hardware, software stack, wiring, and telemetry every lab uses.

---

## 2. Pedagogy & Studio Philosophy

1. **The dual-brain architecture is the subject.** A learned policy needs a probabilistic Linux environment; a motor needs hard real-time determinism and a guaranteed veto. Students build, benchmark, and defend the boundary between the two in every lab.
2. **Bench first.** Each concept from the reading is tested on hardware or in simulation the same week.
3. **Every experiment is one causal loop.** Sense, propose, permit, act, and observe again. The four action taps make each step of that loop visible in the log.
4. **The scope test is the pass bar.** A submission that only classifies frames, or wires a model to a fixed script, does not count. The test is stated on the [Competency Card](student-competencies.md).
5. **Adversarial verification.** A cherry-picked video proves nothing. Instructors and peer teams introduce disturbances (lighting drops, displaced targets, obstacles), and the system must recover or safely abstain.

---

## 3. Learning Objectives

Students are evaluated on the twelve competencies of the [Physical AI Competency Card](student-competencies.md), three in each quadrant: **A** measure the plant, **B** characterize the brain, **C** act in the world, and **D** govern the system. The card defines each competency, the evidence it requires, and the lab where it is signed off.

---

## 4. Assessment, Milestones, and Grading

Grades are earned through witnessed bench demonstrations, milestone packets, and the capstone oral defense, not paper exams.

| Milestone | Due | Weight | Witnessed Evidence | Competencies |
|:---|:---:|:---:|:---|:---:|
| **M1: Station Charter & Envelope** | End of Week 4 | **15%** | Labs 1–2. Safe-state check, hand-derived kinematics validated in simulation, and the sim-vs-real gap measured with your own IK on the real arm. | A1, A3 |
| **M2: Learned Policy on the Edge** | End of Week 8 | **25%** | Labs 3–4. Camera calibration, a sim-trained VLA deployed with a measured sim2real gap, a real LeRobot dataset with all four action taps, and a fine-tuned policy running untethered on the UNO Q. | A2, B1, B2, B3, C1 |
| **M3: Stress-Tested Closed Loop** | End of Week 9 | **10%** | Lab 5. Chunk-horizon benchmark, live recovery from a displaced target, and a language check. | C2, C3 |
| **M4: Certified Governed Station** | End of Week 11 | **25%** | Lab 6. Your own governor clamps or vetoes unsafe proposals, the watchdog halts the arm when Linux freezes, and no stale command runs on resume. | D1, D2 |
| **M5: Capstone Release & Defense** | End of Week 14 | **25%** | Capstone. Proposal (Week 8) and frozen task (Week 10), then live held-out trials under disturbance, baseline comparison, oral defense, and the Physical Release Dossier. | D3 (full card) |

The lab-by-lab schedule and readings are in the [studio landing page](../README.md#3-the-14-week-lab-schedule). The capstone protocol is in the [Capstone Project Studio](../labs/lab-capstone-studio.md).

---

## 5. Team Station Dynamics (Rotating Roles)

Students work in teams of 2 or 3 per bench station. To ensure individual accountability and comprehensive skill mastery, roles rotate on a weekly basis:

1. **The Operator:**
   * Controls hardware power, physical target positioning, teleoperation input devices, and oversees bench motor power cutoffs.
   * Responsible for mechanical calibration, homing checks, and fixture safety.
2. **The Systems Lead:**
   * Operates the Qualcomm Linux terminal, executes LeRobot scripts, manages ONNX model quantization, and monitors inter-core RPC bridge logs.
   * Responsible for runtime latency profiling and software version control.
3. **The Evidence Reviewer / Auditor:**
   * Logs run IDs, verifies timestamp synchronization, records the 4 action taps, captures error plots, and maintains the team's physical lab notebook.
   * Tracks weekly sign-offs on the team's **Competency Card**.

*Individual Accountability:* At the final Capstone defense, each team member is randomly assigned one raw trace or fault recovery to explain and defend individually.

---

## 6. The Physical Safety Contract & Lab Policy

Because actuators impart physical momentum and electrical current:
1. **The Power Separation Mandate:** Motor DC power may only be energized after the STM32 firmware has initialized and the operator confirms the workspace is clear.
2. **The Table Geofence:** Actuator trajectories must remain within the marked table boundaries. Driving the gripper into the tabletop or mounting bracket triggers an immediate hardware disarm.
3. **Hardware Incident Protocol:** If a servo chatters, buzzes, stalls, or overheats, switch off motor power within 2 seconds. A post-incident inspection is mandatory before rearming.
4. **Zero Live Bypass Rule:** Plugging a host USB cable directly into the servo bus to bypass the STM32 MCU permission boundary results in an immediate milestone failure.

---

## 7. Station Logistics, Bench Booking & Spares

* **Supervised Studio Sessions:** Each team receives 3 hours of supervised lab bench time weekly with instructor support.
* **Open Studio Booking:** Stations are available outside class hours via an online reservation portal. Teams may book up to 6 hours of additional bench time weekly.
* **Offline Preparation Requirement:** Students must write code, test preprocessing pipelines, and validate model architectures offline using recorded replay datasets before arriving at the bench. Bench time is reserved for physical trials.
* **Spare Parts Depot:** Teaching staff maintains spare STS3215 servos, gears, cables, and webcams. Damaged components are swapped immediately during office hours.

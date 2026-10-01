# Physical AI Systems: Course Syllabus

**Course:** Physical AI Systems: Machine Learning Systems That Sense and Act\
**Course numbers:** Harvard CS/EE 288 · ETH Zurich 151-0851 (to confirm)\
**Instructor:** Prof. Vijay Janapa Reddi\
**Teaching assistant:** Dr. Andrea Mattia Garavagno\
**Textbook:** [*Physical AI: Machine Learning Systems That Sense and Act*](https://mlsysbook.ai/vol4/) (MLSysBook Volume IV)\
**Status:** Draft. Dates, rooms, and policies marked `TBD` are set before the term.

---

## 1. Course Description

When a chatbot makes a mistake, you retype the prompt. When a robot makes a mistake, something collides or breaks. This course teaches how to build, deploy, evaluate, and govern a machine that senses and acts. Teams take a Vision-Language-Action policy (SmolVLA or ACT), run it on the Qualcomm Linux side of an **Arduino UNO Q**, and give the right to move a **Seeed SO-101** arm to an independent **STM32** safety governor. Over fourteen weeks you go from deriving the arm's kinematics by hand to defending, with your own measurements, whether your system is safe to release.

---

## 2. Course at a Glance

| | |
|:---|:---|
| **Format** | Two meetings a week: a 90-minute systems seminar and a 3-hour supervised lab. Six labs, then a three-week capstone. No written exam. |
| **Length** | 14 teaching weeks |
| **Prerequisites** | Programming in Python and C, basic linear algebra and probability, introductory machine learning, and first-year Newtonian physics. Operating systems or control experience helps but is not required. |
| **Teams** | Two or three students per bench station, with rotating roles |
| **Hardware** | One station per team: Arduino UNO Q, SO-101 arm, overhead camera, and separate motor power. See the [Station Reference Card](../labs/station-reference.md). |
| **Workload** | Seminar and lab each week, plus reading, analysis, and booked bench time. Credits: TBD |
| **Schedule and rooms** | TBD |

---

## 3. Learning Outcomes

By the end of the course, you will be able to:

1. Draw a complete path from sensing to actuation and say which component proposes, which permits, and which executes each physical action.
2. Calibrate sensors, measure actuator limits, safe states, and end-to-end latency, and set an operating envelope from those measurements.
3. Compare a simulation with the real hardware and explain where and why they disagree.
4. Collect and audit robot demonstrations, train and deploy a learned policy on edge hardware, and compare it with a nonlearned baseline.
5. Distinguish offline model metrics from physical task outcomes, including recovery, abstention, and incomplete motion.
6. Build and attack a real-time safety governor, and defend a bounded release or no-release decision with your own evidence.

The twelve competencies on the [Competency Card](student-competencies.md) break these outcomes into skills that staff sign off at the bench.

---

## 4. How a Week Works

**Seminar (about 90 minutes).** The instructor frames one physical AI question from the week's reading. A rotating team then presents one raw trace or failed prediction from its own station, and the class challenges it: what does this trace prove, and what other explanation is left? The seminar ends with the experiment each team will run in lab.

**Lab (about 3 hours).** The teaching assistant opens with the station briefing: safe state, motion limits, and the week's common measurement. Teams then work through that week's **lab part**. Every part ends with a **checkpoint**, an artifact staff look at before the team moves on.

**Before and after.** Before each lab, write down what you predict will happen. After it, add a short entry to your team's shared notebook: the raw trace, what you measured, how it differed from your prediction, and what you will try next. The notebook is checked at the bench and feeds your milestones; there are no separate weekly lab reports.

---

## 5. Schedule

Each chapter has one core section for discussion, announced the week before; the rest is background. The [lab template](../LAB_TEMPLATE.md) explains how each lab is organized.

| Week | Reading | Lab part | Due |
|:---:|:---|:---|:---|
| 1 | [Ch 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd) | [**1A** Meet the robot](../labs/lab-01-kinematics-and-twin.md) | |
| 2 | [Ch 2: The Physical Body](../../../books/vol4/02_body/02_body.qmd) | [**1B** Kinematics on paper](../labs/lab-01-kinematics-and-twin.md) | |
| 3 | Ch 2, continued | [**1C** Digital twin](../labs/lab-01-kinematics-and-twin.md) | |
| 4 | [Ch 3: The Cognitive Brain](../../../books/vol4/03_brain/03_brain.qmd), [Ch 4: The Nervous System](../../../books/vol4/04_nervous/04_nervous.qmd) | [**2A** IK on the real arm](../labs/lab-02-sim-to-real.md) | **M1** |
| 5 | [Ch 5: Physical Data](../../../books/vol4/05_data/05_data.qmd) | [**3A** VLAs in simulation](../labs/lab-03-learned-control.md) | |
| 6 | [Ch 6: Policy Training](../../../books/vol4/06_training/06_training.qmd) | [**3B** The VLA on real hardware](../labs/lab-03-learned-control.md) | |
| 7 | [Ch 7: Closed-Loop Evaluation](../../../books/vol4/07_evaluation/07_evaluation.qmd) | [**4A** Real demonstrations](../labs/lab-04-real-data-and-fine-tuning.md) | |
| 8 | [Ch 8: Sensor Perception](../../../books/vol4/08_perception/08_perception.qmd), [Ch 9: Spatial Memory](../../../books/vol4/09_memory/09_memory.qmd) | [**4B** Fine-tuned & untethered](../labs/lab-04-real-data-and-fine-tuning.md) | **M2**, capstone proposal |
| 9 | [Ch 10: Grounded Intent](../../../books/vol4/10_intent/10_intent.qmd), [Ch 11: Trajectory Planning](../../../books/vol4/11_planning/11_planning.qmd) | [**5A** Stress the loop](../labs/lab-05-stress-testing.md) | **M3** |
| 10 | [Ch 12: Safety Enforcement](../../../books/vol4/12_enforcement/12_enforcement.qmd), [Ch 13: Silicon Placement](../../../books/vol4/13_placement/13_placement.qmd) | [**6A** Build the governor](../labs/lab-06-safety-governor.md) | Capstone task frozen |
| 11 | [Ch 14: Supervisory Intervention](../../../books/vol4/14_intervention/14_intervention.qmd), [Ch 15: Adversarial Verification](../../../books/vol4/15_verification/15_verification.qmd), [Ch 16: Safe Release](../../../books/vol4/16_release/16_release.qmd) | [**6B** Break the governor](../labs/lab-06-safety-governor.md) | **M4** |
| 12 | [Ch 17: The Epistemic Frontier](../../../books/vol4/17_frontier/17_frontier.qmd) | [**C1** Capstone task build](../labs/lab-capstone-studio.md) | |
| 13 | No new reading | [**C2** Peer fault exchange](../labs/lab-capstone-studio.md) | |
| 14 | No new reading | [**C3** Final trials and defense](../labs/lab-capstone-studio.md) | **M5** |

---

## 6. The Capstone

Every team ends the course by defending a release decision for a system it built. You propose a task of your own at the end of week 8, once your autonomous loop works, and freeze it at the end of week 10. Weeks 12–14 are bench time: build the task, survive a peer team's attacks, then run final trials and defend your **Physical Release Dossier** in an individual oral exam. A team that shows, with rigorous evidence, that its system should *not* be released can earn full marks. Details are in the [capstone](../labs/lab-capstone-studio.md).

---

## 7. Assessment, Milestones, and Grading

Grades are earned through witnessed bench demonstrations, milestone packets, and the capstone defense.

| Milestone | Due | Weight | Witnessed Evidence | Competencies |
|:---|:---:|:---:|:---|:---:|
| **M1: Station Charter & Envelope** | End of Week 4 | **15%** | Labs 1–2. Safe-state check, hand-derived kinematics validated in simulation, and the sim-vs-real gap measured with your own IK on the real arm. | A1, A3 |
| **M2: Learned Policy on the Edge** | End of Week 8 | **25%** | Labs 3–4. Camera calibration, a sim-trained VLA deployed with a measured sim2real gap, a real LeRobot dataset with all four action taps, and a fine-tuned policy running untethered on the UNO Q. | A2, B1, B2, B3, C1 |
| **M3: Stress-Tested Closed Loop** | End of Week 9 | **10%** | Lab 5. Chunk-horizon benchmark, live recovery from a displaced target, and a language check. | C2, C3 |
| **M4: Certified Governed Station** | End of Week 11 | **25%** | Lab 6. Your own governor clamps or vetoes unsafe proposals, the watchdog halts the arm when Linux freezes, and no stale command runs on resume. | D1, D2 |
| **M5: Capstone Release & Defense** | End of Week 14 | **25%** | Capstone. Proposal and frozen task, then live held-out trials under disturbance, baseline comparison, oral defense, and the Physical Release Dossier. | D3 (full card) |

**Three grading principles.**
- **Evidence over polish.** A good model metric cannot make up for a bypassed safety path or unexplained motion. Staff may require a repair and a repeated trial before a team moves on.
- **Honest failure counts.** A well-measured negative result, explained, earns more credit than an unmeasured success.
- **Individual accountability.** Teams submit one system, but each student explains their own measurements and a randomly selected trace at the final defense.

The notebook, seminar participation, and the week-13 rehearsal receive feedback but are not graded separately.

---

## 8. Team Roles

Teams of two or three share a station. Roles rotate weekly so every student does every job.

1. **Operator.** Controls motor power, target placement, and teleoperation; responsible for calibration, homing, and fixture safety.
2. **Systems lead.** Runs the Linux side: LeRobot scripts, model export, and the RPC bridge; responsible for latency profiling and version control.
3. **Evidence reviewer.** Logs run IDs, checks timestamp alignment and the four action taps, makes the plots, and keeps the notebook and Competency Card up to date.

---

## 9. Safety Contract

Actuators carry real momentum and current. Every student agrees to these rules before the first lab:

1. **Power order.** Motor power is switched on only after the STM32 firmware has started and the operator confirms the workspace is clear.
2. **Table geofence.** Motion stays inside the marked workspace. Driving the gripper into the table or the mount triggers an immediate disarm.
3. **Incident protocol.** If a servo chatters, buzzes, stalls, or overheats, switch off motor power within two seconds. A staff inspection is required before re-arming.
4. **No bypass.** Connecting a host directly to the servo bus, around the STM32, fails the current milestone.

---

## 10. Station Logistics

- **Supervised lab.** Each team has 3 hours of supervised bench time a week.
- **Open bench booking.** Stations can be booked outside class hours, up to 6 extra hours a week. Booking system: TBD
- **Prepare offline.** Write and test code against recorded replay data before coming to the bench; bench time is for physical trials.
- **Spares.** Staff keep spare servos, cables, and cameras; damaged parts are swapped during office hours.

---

## 11. Course Policies

- **Collaboration:** TBD
- **Use of AI tools:** TBD
- **Late work and repairs:** TBD
- **Accommodations:** TBD
- **Office hours:** TBD

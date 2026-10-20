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
| **Prerequisites** | Programming in Python, basic linear algebra and probability, introductory machine learning, and first-year Newtonian physics. C, operating systems, or control experience helps but is not required; staff provide the baseline firmware and support its extension. |
| **Teams** | Two or three students per bench station, with rotating roles |
| **Hardware** | One station per team: Arduino UNO Q, SO-101 arm, overhead camera, and separate motor power. See the [Station Reference Card](../labs/station-reference.md). |
| **Workload** | Seminar and lab each week, plus reading, analysis, and booked bench time. Credits: TBD |
| **Schedule** | Seminar early in the week (Monday or Tuesday), lab later in the week (Thursday or Friday). Times and rooms: TBD |

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

**Before week 1.** Read the Volume IV [reader guide and prerequisites](../../../books/vol4/frontmatter/prerequisites.qmd) and [Chapter 1](../../../books/vol4/01_boundary/01_boundary.qmd). If you want a refresher, the book's appendices cover [machine learning](../../../books/vol4/backmatter/appendix_ml.qmd), [systems](../../../books/vol4/backmatter/appendix_systems.qmd), and [control](../../../books/vol4/backmatter/appendix_control.qmd); you do not need to master them before the course begins.

---

## 4. How a Week Works

The week has a fixed rhythm: **read the chapter, discuss it in seminar, then put it to work in lab.**

**Seminar (early in the week, about 90 minutes).** Come having read the week's chapter. The instructor frames one physical AI question from the week's reading. A rotating team then presents one raw trace or failed prediction from its own station, and the class challenges it: what does this trace prove, and what other explanation is left? The seminar ends with the experiment each team will run in lab.

**Lab (later in the week, about 3 hours).** The teaching assistant opens with the station briefing: safe state, motion limits, and the week's common measurement. Teams then work through that week's **lab part**. Every part ends with a **checkpoint**, an artifact staff look at before the team moves on.

**Before and after.** Before each lab, write down what you predict will happen. After it, add a short entry to your team's shared notebook: the raw trace, what you measured, how it differed from your prediction, and what you will try next. The notebook is checked at the bench and feeds your lab deliverables; there are no separate weekly lab reports.

---

## 5. Schedule

Read the week's chapter before the seminar: the chapter gives the principles, the seminar tests them, and the lab puts them to work on the bench. Readings follow the book in order, so a chapter sometimes prepares a lab a week or two ahead. The [lab template](../LAB_TEMPLATE.md) explains how each lab is organized.

| Week | Reading | Lab part | Due |
|:---:|:---|:---|:---|
| 1 | [Ch 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd) | [**Getting Started**](../labs/week-01-getting-started.md) |  |
| 2 | [Ch 2: The Physical Body](../../../books/vol4/02_body/02_body.qmd) | [**1A** Measure the body](../labs/lab-01-digital-twin.md) |  |
| 3 | [Ch 3: The Cognitive Brain](../../../books/vol4/03_brain/03_brain.qmd) | [**1B** Build the digital twin](../labs/lab-01-digital-twin.md) | **Lab 1** |
| 4 | [Ch 4: The Nervous System](../../../books/vol4/04_nervous/04_nervous.qmd) | [**2A** IK on the real arm](../labs/lab-02-sim-to-real.md) | **Lab 2** |
| 5 | [Ch 5: Physical Data](../../../books/vol4/05_data/05_data.qmd) | [**3A** VLAs in simulation](../labs/lab-03-learned-control.md) |  |
| 6 | [Ch 6: Policy Training](../../../books/vol4/06_training/06_training.qmd); read ahead: calibration in Ch 8 | [**3B** The VLA on real hardware](../labs/lab-03-learned-control.md) | **Lab 3** |
| 7 | [Ch 7: Closed-Loop Evaluation](../../../books/vol4/07_evaluation/07_evaluation.qmd) | [**4A** Real demonstrations](../labs/lab-04-real-data-and-fine-tuning.md) |  |
| 8 | [Ch 8: Sensor Perception](../../../books/vol4/08_perception/08_perception.qmd), [Ch 9: Spatial Memory](../../../books/vol4/09_memory/09_memory.qmd) | [**4B** Fine-tuned & untethered](../labs/lab-04-real-data-and-fine-tuning.md) | **Lab 4**, capstone proposal |
| 9 | [Ch 10: Grounded Intent](../../../books/vol4/10_intent/10_intent.qmd), [Ch 11: Trajectory Planning](../../../books/vol4/11_planning/11_planning.qmd) | [**5A** Stress the loop](../labs/lab-05-stress-testing.md) | **Lab 5** |
| 10 | [Ch 12: Safety Enforcement](../../../books/vol4/12_enforcement/12_enforcement.qmd), [Ch 13: Silicon Placement](../../../books/vol4/13_placement/13_placement.qmd) | [**6A** Build the governor](../labs/lab-06-safety-governor.md) | Capstone task frozen |
| 11 | [Ch 14: Supervisory Intervention](../../../books/vol4/14_intervention/14_intervention.qmd), [Ch 15: Adversarial Verification](../../../books/vol4/15_verification/15_verification.qmd) | [**6B** Break the governor](../labs/lab-06-safety-governor.md) | **Lab 6** |
| 12 | [Ch 16: Safe Release](../../../books/vol4/16_release/16_release.qmd) | [**C1** Capstone task build](../labs/lab-capstone-studio.md) |  |
| 13 | [Ch 17: Responsible Physical AI](../../../books/vol4/17_responsibility/17_responsibility.qmd) | [**C2** Peer fault exchange](../labs/lab-capstone-studio.md) |  |
| 14 | [Ch 18: The Epistemic Frontier](../../../books/vol4/18_frontier/18_frontier.qmd) | [**C3** Final trials and defense](../labs/lab-capstone-studio.md) | **Capstone** |

---

## 6. The Capstone

Every team ends the course by defending a release decision for a system it built. You propose a task of your own at the end of week 8, once your autonomous loop works, and freeze it at the end of week 10. Weeks 12–14 are bench time: build the task, survive a peer team's attacks, then run final trials and defend your **Physical Release Dossier** in an individual oral exam. A team that shows, with rigorous evidence, that its system should *not* be released can earn full marks. Details are in the [capstone](../labs/lab-capstone-studio.md).

---

## 7. Assessment, Milestones, and Grading

Grades are earned through witnessed bench demonstrations, lab deliverables, and the capstone defense. Labs 1–2 and Labs 3–4 are graded together because each pair builds one working system.

| Graded work | Due | Weight | Witnessed Evidence | Competencies |
|:---|:---:|:---:|:---|:---:|
| **Labs 1–2:** Station & envelope | End of Weeks 3 and 4 | **15%** | Getting Started and Labs 1–2. Safe-state check, the body's measured limits, a digital twin driven by your own kinematics, and the sim-vs-real gap measured with your own IK on the real arm. | A1; A3 timing foundation |
| **Labs 3–4:** Learned policy on the edge | End of Weeks 6 and 8 | **25%** | Labs 3–4. Camera calibration, training/validation curves, a sim-trained VLA deployed with a measured sim2real gap, a real LeRobot dataset with all four action taps, and a fine-tuned policy running untethered on the UNO Q. | A2, A3 (complete), B1, B2, B3, C1 |
| **Lab 5:** Stress-tested loop | End of Week 9 | **10%** | Lab 5. Chunk-horizon benchmark, live recovery from a displaced target, and a language check. | C2, C3 |
| **Lab 6:** Your own governor | End of Week 11 | **25%** | Lab 6. Your own governor clamps or vetoes proposals outside the declared checked limits; staff-approved fault trials measure its response against declared limits, including the watchdog and stale-command rejection on resume. | D1, D2 |
| **Capstone:** Release & defense | End of Week 14 | **25%** | Capstone. Proposal and frozen task, then live held-out trials under disturbance, baseline comparison, oral defense, and the Physical Release Dossier. | D3 (full card) |

**Three grading principles.**
- **Evidence over polish.** A good model metric cannot make up for a bypassed safety path or unexplained motion. Staff may require a repair and a repeated trial before a team moves on.
- **Honest failure counts.** A well-measured negative result, explained, earns more credit than an unmeasured success.
- **Individual accountability.** Teams submit one system, but each student explains their own measurements and a randomly selected trace at the final defense.

The notebook, seminar participation, and the week-13 rehearsal receive feedback but are not graded separately.

---

## 8. Course Policies

**Collaboration.** Collaboration is encouraged. Bench hardware is limited, so all lab work is done in teams; discussing ideas across teams is welcome. Each student must still be able to explain their own team's work at the final defense.

**Use of AI tools.** AI tools are allowed. Say which tools you used and how, in your notebook or with the deliverable. There is no penalty for using them; the point is that you learn, and you will be asked to explain your work without them at the defense.

**Late work and repairs.** The labs build on each other, so falling behind costs more than a late penalty would.
- **Equipment failures** that are not your team's fault (a dead servo, a broken station) get an automatic extension.
- **A missed checkpoint** can be finished in booked bench time before the next lab; if your team falls further behind, staff hand you the reference artifact so you can keep going.
- **Repairs.** Staff may ask you to fix and re-run a result before signing it off. A repair finished within a week earns full credit.
- **Milestones** otherwise lose 10% per day late, unless you arranged an extension before the deadline.

**Accommodations.** Students with approved accommodations should contact the instructor in the first week. Because lab work is a physical demonstration at the bench, lab accommodations are given as extra time on checkpoints and milestones.

**Office hours.** TBD

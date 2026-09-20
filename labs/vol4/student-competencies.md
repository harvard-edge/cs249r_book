# Volume IV physical AI competency checklist

**Status:** Course design contract for the teaching team. The competencies below describe transferable abilities and acceptable evidence without naming a simulator, model library, board, sensor bus, or robot. The later course map shows how the proposed [LeRobot/UNO Q operating plan](course-architecture.md) and [SO-101 lab draft](so101-uno-q-course.md) would teach them. Changing the kit should change that map, not the competencies.

## The minimum graduation demonstration

Given an unfamiliar starting condition, a team must show a trace of **physical state → sensed state → versioned learned model → proposed action → permission → measured physical action → new observation → revised decision**. It must repeat the task, compare against a nonlearned baseline, and handle a seeded fault without a prohibited move. This implements the book's [three-part scope test](../../books/vol4/01_boundary/01_boundary.qmd): learned model, consequential physical feedback, and delegated actuator authority. Individual competencies can be assessed across common labs and the capstone; every model family need not run in the final device.

## Transferable student competencies

An engineer who passes this course should be able to **measure an unfamiliar physical process, build and test models of it, close a feedback loop, and justify the authority and evidence behind a deployed system**. The two axes below separate the object of study (physical process or computing system) from the engineering work (characterize and predict, or decide, act, and govern). The quadrant names are shorthand for the checklist items, not four separate projects.

| | **Characterize and predict** | **Decide and govern** |
|:---|:---|:---|
| **Physical process** | **A. Measure the world**<br>Sense, simulate, predict.<br>C1, C2, C3, C5, C7, C9 | **C. Change the world**<br>Plan, act, recover.<br>C10, C11, C14 |
| **Computing system** | **B. Characterize computation**<br>Compare models and profile runtime.<br>C4, C8, C12 | **D. Integrate and govern**<br>Connect, constrain, verify.<br>C6, C13, C15 |

### A. Measure the world

| ID | Student can... | Evidence that would count on any platform |
|:---|:---|:---|
| [ ] **C1 Body** | Identify the physical state, available actions, energy source, motion limits, and safe state of a mechanism. | A measured operating envelope and observed behavior at startup, shutdown, and loss of power. |
| [ ] **C2 Sensing** | Acquire and calibrate two different kinds of physical feedback; explain units, electrical or communication interface, sampling, uncertainty, and missing data. | Raw readings, calibration procedure, disagreement analysis, and a sensor-loss trial. |
| [ ] **C3 Feedback and time** | Associate an observation, decision, commanded action, measured action, and subsequent observation; detect delay and stale information. | Linked trace with timestamps or cycle IDs, latency distribution, and a stale-data trial. |
| [ ] **C5 Simulation** | Express a task through a reusable observation/action contract, test at least two simulation models, and identify discrepancies against held-out physical trials. | Versioned models, matched initial conditions and actions, prediction errors, and an explanation of failed assumptions. |
| [ ] **C7 Physical data** | Record repeatable episodes with observations, attempted and executed actions, measured state, outcomes, and interventions; construct a defensible held-out split. | Replayable episode sample, schema, dataset card, and split rationale. |
| [ ] **C9 Dynamics** | Learn or identify how state changes after an action and test prediction beyond the training trajectories. | One-step and rollout errors on held-out physical trials, compared with a simple model. |

### B. Characterize the computation

| ID | Student can... | Evidence that would count on any platform |
|:---|:---|:---|
| [ ] **C4 Baseline** | Build a nonlearned rule or controller and evaluate it under the same conditions as a learned system. | Matched trials reporting both policies' actions and physical outcomes. |
| [ ] **C8 Perception** | Version a learned perception model and its preprocessing, evaluate it on held-out physical observations, and deploy inference where the task requires it. | Model artifact, matched development/deployment outputs, error cases, memory use, and inference timing. |
| [ ] **C12 Compute placement** | Assign sensing, inference, planning, and low-level control to available processors and justify the timing and resource budget. | Processor map, memory use, ordinary and loaded latency tails, and the resulting deadline rule. |

### C. Change the world

| ID | Student can... | Evidence that would count on any platform |
|:---|:---|:---|
| [ ] **C10 Planning** | Use current state, a goal, and a prediction to select an admissible action; replan when measured motion differs from prediction. | Open-loop and feedback trials, task error, action count, and response to an omitted or incomplete move. |
| [ ] **C11 Policies** | Compare a reactive action policy with a policy that predicts a sequence of actions; explain drift and when to observe again. | Policy traces on common episodes, including a changed scene or failed action. |
| [ ] **C14 Intervention** | Stop a task, invalidate old intent, recover state, and resume only from a fresh observation. | Cutoff, reset, and rearm trace showing no queued motion. |

### D. Integrate and govern

| ID | Student can... | Evidence that would count on any platform |
|:---|:---|:---|
| [ ] **C6 Hardware integration** | Trace a decision through software, communication, control electronics, actuator, and feedback; identify where state or commands can be lost or bypassed. | Interface and power diagram, software versions, a command trace, and an injected communication fault. |
| [ ] **C13 Authority** | Identify who may issue, permit, and physically execute an action; test what happens when a proposer stalls or sends an invalid request. | Accepted and refused requests paired with measured actuator and power behavior. |
| [ ] **C15 Evaluation** | Compare repeated physical outcomes with the baseline and simulation predictions, including failures and abstentions; state the tested envelope. | Frozen trial protocol, raw traces, fault results, and a bounded release or no-release claim. |

The postdoc should make each row an observable pass/fail check in the student handout. A team may use a different body or simulator if it supplies equivalent evidence. A language model or arm is an optional way to exercise these abilities, not a substitute for any row.

## Course realization: LeRobot, UNO Q, and SO-101

The [course operating plan](course-architecture.md) specifies the seminar rhythm and milestones; the [SO-101 lab draft](so101-uno-q-course.md) details each weekly experiment. The proposed common stack is a LeRobot-compatible robot and dataset, a simulator, a versioned Hugging Face policy, UNO Q Qualcomm-side local inference, and STM32-permitted physical action. Staff must test the SO-101 STM32-to-servo command path before claiming MCU control of the arm. The one-axis [reference sequence](lab-sequence.md) remains a fallback body under the same competency contract.

### A. Measure the world

- **C1 Body (weeks 1–2):** map power, joints, gripper, workspace, and safe state; submit the measured envelope and power-off trace.
- **C2 Sensing (weeks 2–4):** calibrate camera and joint readback as two physical feedback sources; submit units, disagreement, and a sensor-loss trial. An MCU-read SPI sensor can add independent joint feedback after staff qualification.
- **C3 Feedback and time (weeks 3–6):** link frame, action taps, joint response, and next frame; submit clock or cycle alignment and a stale-data trial.
- **C5 Simulation (weeks 7–8):** compare a simple model and a second robot simulator; submit matched-start predictions and measured simulation gaps.
- **C7 Physical data (weeks 4–5):** record LeRobot episodes with requested, mapped, enforced, and measured action taps; submit a replayable dataset, held-out split, intervention fields, and card.
- **C9 Dynamics (week 7):** fit a small next-state predictor; submit one-step and rollout errors on held-out physical trials against a simple model.

### B. Characterize computation

- **C4 Baseline (week 5):** run a scripted or teleoperated reference under a frozen protocol; submit matched baseline and learned-policy physical trials.
- **C8 Perception (weeks 5–6):** freeze visual preprocessing and local model output; submit held-out error cases, host/board agreement, memory, and timing.
- **C12 Compute placement (weeks 6 and 9–10):** profile Qualcomm inference, Bridge, MCU decision, and action age; submit a processor map, latency tails, and deadline rule.

### C. Change the world

- **C10 Planning (weeks 6, 8, and 11):** correct after an incomplete move or changed object state, beginning with the week-6 local learned loop; submit open-loop versus feedback outcomes and a replanning trace.
- **C11 Policies (weeks 8–9):** compare ACT and SmolVLA action chunks with a reactive policy; submit a changed-scene or changed-instruction trace.
- **C14 Intervention (weeks 11–12):** exercise cutoff, reset, correction, and rearm; show that no queued motion executes and resumption uses fresh evidence.

### D. Integrate and govern

- **C6 Hardware integration (weeks 1–3):** trace Qualcomm → Bridge → STM32 → motor bus → feedback; submit command/power diagrams and a communication-fault trace.
- **C13 Authority (weeks 3 and 10):** test MCU acceptance and refusal under deliberate faults; submit enforced-action and measured-motion traces showing no bypass.
- **C15 Evaluation (weeks 12–14):** repeat frozen physical trials; submit raw outcomes, baseline comparison, fault results, and a release or no-release verdict.

The capstone rubric uses the four quadrants equally: **A** measures the world, **B** characterizes computation, **C** changes the world, and **D** integrates and governs. Students still check off the specific competencies within each quadrant. The qualifying episode must contain a learned action proposal, actual world change, a fresh measurement of tool and object/task state, and a state-dependent revised decision under the MCU permission boundary. A model score alone cannot replace C10 correction, C13 authority, or C15 physical evidence. The [course operating plan](course-architecture.md) describes team and individual evidence.

The SO-101 is the book's contact-dominated manipulation example. [LeKiwi or a custom rover](so101-uno-q-course.md) can add a mobility comparison; a qualified thermal cell can add a process/energy comparison. Changing the embodiment changes the lab realization and measured physical law, not the competency definitions above. The staff build report should mark each competency **proved on hardware**, **prepared with replay or simulation**, or **unresolved**, with a linked trace supporting the mark.

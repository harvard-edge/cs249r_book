# Physical AI Systems

**ETH spring project seminar · Volume IV**

**Instructor:** Professor Vijay Janapa Reddi<br>
**Teaching assistant:** Dr. Andrea Mattia Garavagno

> **Course planning status:** This page shows the proposed course and weekly project milestones. The linked lab briefs are drafts while the teaching team qualifies the hardware station. Final meeting times, enrollment, lab access, and motion limits will be announced through the ETH course listing and course staff.

## What this course is about

Students build one physical AI system across the semester. A learned policy on the Arduino UNO Q proposes a bounded action; its STM32 microcontroller permits or refuses it; a robot changes the world; and the system measures what actually happened before deciding again. The first proposed body is a Seeed SO-101 arm using the LeRobot data and policy workflow. The engineering question is whether the whole **sense → propose → permit → act → observe → revise** loop works under physical delay, imperfect motion, and changed scenes.

A capstone must satisfy the [Volume IV scope test](../../books/vol4/01_boundary/01_boundary.qmd): a learned decision, consequential physical feedback, and delegated actuation inside an independent permission boundary. A model score or a camera classifier that triggers a fixed arm script does not establish that result. Teams compare their system with a nonlearned baseline and report measured physical outcomes, refusals, corrections, and failures.

## Who this course is for

This seminar is intended for advanced undergraduate and graduate students in computer science, electrical or computer engineering, machine learning, and robotics. Students may arrive with different strengths; the team project brings those perspectives together. Before the course, you should be able to:

- Write and debug Python, work with arrays, and use Git to share code and results.
- Explain the difference between training a model and running inference, and interpret a held-out test result.
- Use basic vectors, algebra, probability, and rates of change to reason about measurements and motion.
- Read a system diagram or log and identify where data, commands, and delays enter a computation.

Prior experience with robot arms, LeRobot, microcontroller firmware, or advanced control theory is helpful but not required. The course develops those connections through measurements, simulation, data collection, policy deployment, and supervised experiments. Students should expect to learn from teammates whose background differs from their own.

**Before week 1:** Read the Volume IV [reader guide and prerequisites](../../books/vol4/frontmatter/prerequisites.qmd) and [Chapter 1](../../books/vol4/01_boundary/01_boundary.qmd). If you want a refresher, the book includes appendices on [machine learning](../../books/vol4/backmatter/appendix_ml.qmd), [systems](../../books/vol4/backmatter/appendix_systems.qmd), and [control](../../books/vol4/backmatter/appendix_control.qmd). You do not need to master every appendix before the seminar begins.

## Course at a glance

- **Format:** 14 teaching weeks; one team project; two meetings per week; no written exam. The class-free Easter week is outside the numbered teaching weeks.
- **Meeting A — systems seminar:** Provisional 90 minutes, chaired by Professor Reddi. A short framing of one systems question is followed by a rotating team trace and discussion. Teams leave with a question to test; there is no new lecture deck or paper presentation each week.
- **Meeting B — supervised project lab:** Provisional three hours, led by Dr. Garavagno. The lab begins with the qualified station envelope and one common measurement; teams spend the block experimenting on their own project, with short bench reviews. Additional bench time is reserved separately.
- **Teams and workload:** Two or three students per station, subject to the hardware capacity pilot. Final timetable, ECTS, and enrollment follow the ETH course listing.
- **Reading:** Each week focuses on one section of the linked Volume IV chapter; the full chapter and any second chapter are references for deeper study. Research papers are optional unless they serve a project question. LeRobot documentation is a software reference, not an assigned external course.
- **Evidence and assessment:** Teams keep one shared, versioned project notebook with a brief trace and next step each week. Four cumulative milestones are graded: the week-2 task charter, week-6 feedback loop, week-10 authority test, and week-14 capstone. Other bench checks are formative. The final submission is a concise system report and oral demonstration.

## Freedom within a common systems contract

Teams choose a physical question by week 2. The common starting task is a guarded reach toward a soft target; after the first move, staff safely shift the target or cause incomplete motion. By week 6, every team must show that a local learned policy uses the resulting observation to change its proposal or abstain. That is the shared working system that each team develops further for the capstone.

After the week-6 demonstration, teams can pursue recovery from failed grasp or placement, active inspection before grasping, instruction-conditioned manipulation, policy timing, simulation gap, or an approved question of their own. They choose the model, baseline, simulator, data strategy, and recovery method. A different embodiment or sensor module needs its own staff qualification. Every capstone still needs the learned action, MCU permission, measured world change, new decision, matched baseline, and a safe fault trial.

## Draft weekly schedule

Each linked lab is a studio prompt for the same team project. Some later briefs offer choices or rehearsal time; they are not fourteen separate graded assignments. The seminar asks a systems question, and the lab produces evidence the next week can use. The [full draft syllabus](syllabus.qmd) has assessment details.

<table>
  <caption>Fourteen teaching weeks; exact dates follow the ETH course listing.</caption>
  <thead>
    <tr>
      <th scope="col">Week&nbsp;&nbsp;</th>
      <th scope="col">Seminar, lab, and evidence</th>
    </tr>
  </thead>
  <tbody>
    <tr><th colspan="2" scope="colgroup">Weeks 1–6 · Establish the shared physical loop</th></tr>
    <tr><th scope="row">1–2</th><td><strong>Seminar:</strong> <a href="../../books/vol4/01_boundary/01_boundary.qmd">Ch. 1 · The Causal Boundary</a> + <a href="../../books/vol4/02_body/02_body.qmd">Ch. 2 · The Physical Body</a><br>What makes this physical AI?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-01-boundary.md">Lab 1 · The Causal Boundary & Servo Bus Bring-Up</a><br><strong>Evidence:</strong> Command and power diagram; observed safe state.</td></tr>
    <tr><th scope="row">3</th><td><strong>Seminar:</strong> <a href="../../books/vol4/03_brain/03_brain.qmd">Ch. 3 · The Cognitive Brain</a> + <a href="../../books/vol4/04_nervous/04_nervous.qmd">Ch. 4 · The Nervous System</a><br>Who proposes and who permits motion?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-02-body-and-sensors.md">Lab 2 · Multi-Modal Sensing & The Inter-Core Bridge</a><br><strong>Evidence:</strong> Multi-modal sensor sync and inter-core bridge logs.</td></tr>
    <tr><th scope="row">4–5</th><td><strong>Seminar:</strong> <a href="../../books/vol4/05_data/05_data.qmd">Ch. 5 · Physical Data</a> + <a href="../../books/vol4/07_evaluation/07_evaluation.qmd">Ch. 7 · Closed-Loop Evaluation</a><br>Can an episode reconstruct what happened?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-03-physical-episodes.md">Lab 3 · Teleoperation, LeRobot Dataset & Action Taps</a><br><strong>Evidence:</strong> Replayable episode, 4 action taps, dataset card.</td></tr>
    <tr><th scope="row">6</th><td><strong>Seminar:</strong> <a href="../../books/vol4/06_training/06_training.qmd">Ch. 6 · Policy Training</a><br>What does learning improve over a rule?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-04-baseline-and-learning.md">Lab 4 · Deterministic Baseline & SmolVLA/ACT Export</a><br><strong>Evidence:</strong> Held-out split and matched physical baseline.</td></tr>
    <tr><th colspan="2" scope="colgroup">Weeks 7–11 · Autonomy, horizons, and safety enforcement</th></tr>
    <tr><th scope="row">7–8</th><td><strong>Seminar:</strong> <a href="../../books/vol4/08_perception/08_perception.qmd">Ch. 8 · Sensor Perception</a> + <a href="../../books/vol4/11_planning/11_planning.qmd">Ch. 11 · Trajectory Planning</a> + <a href="../../books/vol4/13_placement/13_placement.qmd">Ch. 13 · Silicon Placement</a><br>Does the learned loop revise its decision?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-05-local-feedback-loop.md">Lab 5 · Autonomous Closed-Loop Reach on UNO Q</a><br><strong>Evidence:</strong> Witnessed changed-scene loop on the UNO Q.</td></tr>
    <tr><th scope="row">9</th><td><strong>Seminar:</strong> <a href="../../books/vol4/10_intent/10_intent.qmd">Ch. 10 · Grounded Intent</a> + <a href="../../books/vol4/12_enforcement/12_enforcement.qmd">Ch. 12 · Safety Enforcement</a><br>When must an action chunk expire?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-06-action-chunks.md">Lab 6 · Action Horizons, Language & Disturbances</a><br><strong>Evidence:</strong> Chunk horizon vs reactive recovery trace.</td></tr>
    <tr><th scope="row">10</th><td><strong>Seminar:</strong> <a href="../../books/vol4/12_enforcement/12_enforcement.qmd">Ch. 12 · Safety Enforcement</a> + <a href="../../books/vol4/14_intervention/14_intervention.qmd">Ch. 14 · Supervisory Intervention</a><br>Can the MCU refuse plausible but stale intent?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-07-authority-under-fault.md">Lab 7 · The Microcontroller Safety Governor</a><br><strong>Evidence:</strong> Refusal, cutoff, and no queued motion.</td></tr>
    <tr><th scope="row">11</th><td><strong>Seminar:</strong> <a href="../../books/vol4/15_verification/15_verification.qmd">Ch. 15 · Adversarial Verification</a> + <a href="../../books/vol4/16_release/16_release.qmd">Ch. 16 · Deployment Release</a><br>Does the system fail safely under fault?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-08-verification-and-release.md">Lab 8 · Fault Injection & Fail-Safe Cutoffs</a><br><strong>Evidence:</strong> Injected fault cutoff and safe disarm result.</td></tr>
    <tr><th colspan="2" scope="colgroup">Weeks 12–14 · Capstone project studio, freeze, and defense</th></tr>
    <tr><th scope="row">12–14</th><td><strong>Seminar:</strong> <a href="../../books/vol4/17_frontier/17_frontier.qmd">Ch. 1–17 · Full Curriculum Synthesis</a><br>What claim can the physical evidence support?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-capstone-studio.md">Capstone Project Studio & Release Defense</a><br><strong>Evidence:</strong> 20 live physical trials, written release dossier, oral defense.</td></tr>
  </tbody>
</table>

## What counts as a successful capstone

A team submits its frozen code, model and data revisions, calibration, simulator assumptions, raw physical trials, matched baseline, injected fault result, and a written claim about its tested envelope. In the live defense, each student explains one proposal → permission → measured action → new observation → revised decision trace. A project may conclude that its tested system should not be released; the quality of the evidence and reasoning determines the result.

The [student competency matrix](../../labs/vol4/curriculum/student-competencies.md) supplies a platform-independent 2×2 checklist. The [course operating plan](../../labs/vol4/curriculum/course-architecture.md) and [lab blueprints](../../labs/vol4/labs/lab-blueprints.md) provide instructor detail. The [feasibility plan](../../labs/vol4/staff/feasibility-plan.md) tells the teaching team what must work before these briefs become runnable student assignments.

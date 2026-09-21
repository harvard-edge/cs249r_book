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
    <tr><th scope="row">1</th><td><strong>Seminar:</strong> <a href="../../books/vol4/01_boundary/01_boundary.qmd">Ch. 1 · The Causal Boundary</a><br>What makes this physical AI?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-01-boundary.md">Lab 1 · Boundary and authority</a><br><strong>Evidence:</strong> Command and power diagram; observed safe state.</td></tr>
    <tr><th scope="row">2</th><td><strong>Seminar:</strong> <a href="../../books/vol4/02_body/02_body.qmd">Ch. 2 · The Physical Body</a><br>Where does commanded motion differ from reality?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-02-body-and-sensors.md">Lab 2 · Body and sensors</a><br><strong>Evidence:</strong> Calibration, physical envelope, and project charter.</td></tr>
    <tr><th scope="row">3</th><td><strong>Seminar:</strong> <a href="../../books/vol4/03_brain/03_brain.qmd">Ch. 3 · The Cognitive Brain</a> + <a href="../../books/vol4/04_nervous/04_nervous.qmd">Ch. 4 · The Nervous System</a><br>Who proposes and who permits motion?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-03-action-permission.md">Lab 3 · Action permission</a><br><strong>Evidence:</strong> Four-action-tap trace and MCU refusal.</td></tr>
    <tr><th scope="row">4</th><td><strong>Seminar:</strong> <a href="../../books/vol4/05_data/05_data.qmd">Ch. 5 · Physical Data</a> + <a href="../../books/vol4/08_perception/08_perception.qmd">Ch. 8 · Sensor Perception</a><br>Can an episode reconstruct what happened?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-04-physical-episodes.md">Lab 4 · Physical episodes</a><br><strong>Evidence:</strong> Replayable episode, aligned MCU log, dataset card.</td></tr>
    <tr><th scope="row">5</th><td><strong>Seminar:</strong> <a href="../../books/vol4/06_training/06_training.qmd">Ch. 6 · Policy Training</a> + <a href="../../books/vol4/07_evaluation/07_evaluation.qmd">Ch. 7 · Closed-Loop Evaluation</a><br>What does learning improve over a rule?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-05-baseline-and-learning.md">Lab 5 · Baseline and learning</a><br><strong>Evidence:</strong> Held-out split and matched physical baseline.</td></tr>
    <tr><th scope="row">6</th><td><strong>Seminar:</strong> <a href="../../books/vol4/07_evaluation/07_evaluation.qmd">Ch. 7 · Closed-Loop Evaluation</a><br>Does the learned loop revise its decision?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-06-local-feedback-loop.md">Lab 6 · Local feedback loop</a><br><strong>Evidence:</strong> Witnessed changed-scene loop on the UNO Q.</td></tr>
    <tr><th colspan="2" scope="colgroup">Weeks 7–12 · Investigate and revise each project</th></tr>
    <tr><th scope="row">7</th><td><strong>Seminar:</strong> <a href="../../books/vol4/09_memory/09_memory.qmd">Ch. 9 · Spatial Memory</a> + <a href="../../books/vol4/10_intent/10_intent.qmd">Ch. 10 · Grounded Intent</a><br>Which simulation assumptions fail?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-07-simulation-gap.md">Lab 7 · Simulation gap</a><br><strong>Evidence:</strong> Matched rollouts and measured prediction error.</td></tr>
    <tr><th scope="row">8</th><td><strong>Seminar:</strong> <a href="../../books/vol4/11_planning/11_planning.qmd">Ch. 11 · Trajectory Planning</a><br>When must an action chunk expire?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-08-action-chunks.md">Lab 8 · Action chunks</a> (choose this or the week-9 investigation)<br><strong>Evidence:</strong> One project trace; no separate hand-in.</td></tr>
    <tr><th scope="row">9</th><td><strong>Seminar:</strong> <a href="../../books/vol4/08_perception/08_perception.qmd">Ch. 8 · Sensor Perception</a> + <a href="../../books/vol4/13_placement/13_placement.qmd">Ch. 13 · Silicon Placement</a><br>Does language or model size earn its cost?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-09-language-and-placement.md">Lab 9 · Language and placement</a> (project option or staff trace)<br><strong>Evidence:</strong> Ungraded design review of policy choice and timing.</td></tr>
    <tr><th scope="row">10</th><td><strong>Seminar:</strong> <a href="../../books/vol4/12_enforcement/12_enforcement.qmd">Ch. 12 · Safety Enforcement</a><br>Can the MCU refuse plausible but stale intent?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-10-authority-under-fault.md">Lab 10 · Authority under fault</a><br><strong>Evidence:</strong> Refusal, cutoff, and no queued motion.</td></tr>
    <tr><th scope="row">11</th><td><strong>Seminar:</strong> <a href="../../books/vol4/14_intervention/14_intervention.qmd">Ch. 14 · Supervisory Intervention</a><br>What happens after incomplete motion?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-11-incomplete-motion.md">Lab 11 · Incomplete motion</a><br><strong>Evidence:</strong> Measured discrepancy, correction or abstention.</td></tr>
    <tr><th scope="row">12</th><td><strong>Seminar:</strong> <a href="../../books/vol4/15_verification/15_verification.qmd">Ch. 15 · Adversarial Verification</a><br>Does a documented correction help?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-12-human-correction.md">Lab 12 · Human correction</a> or project repair<br><strong>Evidence:</strong> Before-and-after trace or documented repair; no separate grade.</td></tr>
    <tr><th colspan="2" scope="colgroup">Weeks 13–14 · Freeze, test, and defend</th></tr>
    <tr><th scope="row">13</th><td><strong>Seminar:</strong> <a href="../../books/vol4/16_release/16_release.qmd">Ch. 16 · Deployment Release</a><br>What claim can the evidence support?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-13-release-case.md">Lab 13 · Rehearsal and release case</a><br><strong>Evidence:</strong> Frozen trial packet and formative feedback; no new grade.</td></tr>
    <tr><th scope="row">14</th><td><strong>Seminar:</strong> <a href="../../books/vol4/17_frontier/17_frontier.qmd">Ch. 17 · The Epistemic Frontier</a><br>What happens on an unfamiliar trial?<br><strong>Lab:</strong> <a href="../../labs/vol4/lab-14-capstone-trial.md">Lab 14 · Capstone trial</a><br><strong>Evidence:</strong> Repeated physical trials, written report, oral defense.</td></tr>
  </tbody>
</table>

## What counts as a successful capstone

A team submits its frozen code, model and data revisions, calibration, simulator assumptions, raw physical trials, matched baseline, injected fault result, and a written claim about its tested envelope. In the live defense, each student explains one proposal → permission → measured action → new observation → revised decision trace. A project may conclude that its tested system should not be released; the quality of the evidence and reasoning determines the result.

The [student competency matrix](../../labs/vol4/curriculum/student-competencies.md) supplies a platform-independent 2×2 checklist. The [course operating plan](../../labs/vol4/curriculum/course-architecture.md) and [lab blueprints](../../labs/vol4/labs/lab-blueprints.md) provide instructor detail. The [feasibility plan](../../labs/vol4/staff/feasibility-plan.md) tells the teaching team what must work before these briefs become runnable student assignments.

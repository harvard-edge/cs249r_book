# Volume IV: Physical AI Studio

<h2 align="center">Physics has no <kbd>Ctrl</kbd> + <kbd>Z</kbd>.</h2>

<p align="center"><em>Physical AI begins where computation stops being symbolic and becomes physical force.</em></p>

**Status:** Course structure for the [Volume IV Physical AI seminar](https://mlsysbook.ai/vol4/). The structure, schedule, and lab skeletons are set; the teaching team is writing the lab instructions. Companion to the [MLSysBook Hardware Kits](../../kits/index.qmd).

---

## 1. What This Course Is About

When a chatbot makes a mistake, you retype the prompt. When a robot makes a mistake, something collides, tears, or breaks, and there is no undo. This studio teaches the systems engineering of machines that sense and act, built around one split of responsibility:

1. **A learned brain proposes.** A Vision-Language-Action policy (Hugging Face **SmolVLA** or **ACT**) runs on the UNO Q's Qualcomm Linux processor and proposes the next move.
2. **A deterministic governor permits.** The UNO Q's **STM32** microcontroller checks every proposal against joint limits, speed limits, and a table geofence before it reaches a motor.
3. **A physical body acts.** A 6-DoF **Seeed SO-101** arm moves, and an overhead webcam shows what actually happened.

![The control loop: the camera feeds a learned policy on Linux, which proposes an action across the authority boundary; the STM32 governor permits, clamps, or vetoes it before it reaches the SO-101 arm, and the next image closes the loop.](assets/images/vol4-control-loop.svg)

The four logged actions (`a_req`, `a_map`, `a_enf`, `a_meas`) and the bench wiring are defined once, in the [Station Reference Card](labs/station-reference.md). The course-level graduation standard is the scope test in the [Competency Card](curriculum/student-competencies.md); each lab contributes the competencies needed to demonstrate it.

---

## 2. The Kit

| Board | Arm | Software | Textbook |
| :---: | :---: | :---: | :---: |
| <img src="assets/images/arduino-uno-q.jpg" alt="Arduino UNO Q" width="130" /> | <img src="assets/images/so101-follower.png" alt="Seeed SO-101 follower arm" width="130" /> | <img src="assets/images/lerobot-logo.png" alt="Hugging Face LeRobot" width="150" /> | <img src="assets/images/vol4-textbook-cover.png" alt="MLSysBook Volume IV: Physical AI" width="110" /> |
| **Arduino UNO Q**<br>Linux MPU + STM32 MCU | **Seeed SO-101**<br>6 smart serial servos | **Hugging Face LeRobot**<br>SmolVLA and ACT | **MLSysBook Volume IV**<br>18 chapters, 4 parts |

Specifications, wiring, and the three bench rules are on the [Station Reference Card](labs/station-reference.md).

---

## 3. The 14-Week Lab Schedule

Read the chapter early in the week, discuss it in seminar, then put it to work in lab. Six labs follow the book in order; the final three weeks are the capstone.

![Fourteen-week timeline: Getting Started in week 1, Lab 1 in weeks 2 and 3, Lab 2 in week 4, Labs 3 and 4 in weeks 5 to 8, Lab 5 in week 9, Lab 6 in weeks 10 and 11, and the capstone in weeks 12 to 14.](assets/images/vol4-course-timeline.svg)

<table>
  <thead>
    <tr><th>Week</th><th>Lab</th><th>Book part &amp; chapters</th><th>What you do</th><th>Due</th></tr>
  </thead>
  <tbody>
    <tr><td align="center">1</td><td><a href="labs/week-01-getting-started.md"><strong>Getting Started</strong></a></td><td>Part I · Ch 1 <em>The Causal Boundary</em></td><td>Get the kit and your computer ready, learn how the station works, and learn to stop the robot safely.</td><td></td></tr>
    <tr><td align="center">2</td><td rowspan="2">Lab 1<br><a href="labs/lab-01-digital-twin.md"><strong>The Digital Twin</strong></a></td><td>Part I · Ch 2 <em>The Physical Body</em></td><td><strong>1A</strong> Drive the robot and measure the body: stopping distance, heat under load, supply sag, and measurement age.</td><td></td></tr>
    <tr><td align="center">3</td><td>Part I · Ch 3 <em>The Cognitive Brain</em></td><td><strong>1B</strong> Build the digital twin and command it with kinematics you derive yourself. You will use and improve it all term.</td><td><strong>Lab 1</strong></td></tr>
    <tr><td align="center">4</td><td rowspan="1">Lab 2<br><a href="labs/lab-02-sim-to-real.md"><strong>From Simulation to Real Hardware</strong></a></td><td>Part I · Ch 4 <em>The Nervous System</em></td><td><strong>2A</strong> Run your IK on the real arm and measure what simulation got wrong.</td><td><strong>Lab 2</strong></td></tr>
    <tr><td align="center">5</td><td rowspan="2">Lab 3<br><a href="labs/lab-03-learned-control.md"><strong>Learned Control with Vision-Language-Action Models</strong></a></td><td>Part II · Ch 5 <em>Physical Data</em></td><td><strong>3A</strong> Run a pre-trained VLA in simulation and compare it with your IK.</td><td></td></tr>
    <tr><td align="center">6</td><td>Part II · Ch 6 <em>Policy Training</em></td><td><strong>3B</strong> Calibrate the camera, deploy the VLA to the robot, and measure the sim2real gap.</td><td><strong>Lab 3</strong></td></tr>
    <tr><td align="center">7</td><td rowspan="2">Lab 4<br><a href="labs/lab-04-real-data-and-fine-tuning.md"><strong>Real-World Data Collection &amp; Fine-Tuning</strong></a></td><td>Part II · Ch 7 <em>Closed-Loop Evaluation</em></td><td><strong>4A</strong> Teleoperate the robot and record real demonstrations.</td><td></td></tr>
    <tr><td align="center">8</td><td>Part III · Ch 8 <em>Sensor Perception</em>, Ch 9 <em>Spatial Memory</em></td><td><strong>4B</strong> Fine-tune on real data and run the loop untethered.</td><td><strong>Lab 4<br>Capstone proposal</strong></td></tr>
    <tr><td align="center">9</td><td rowspan="1">Lab 5<br><a href="labs/lab-05-stress-testing.md"><strong>Stress-Testing: Prediction, Language &amp; Recovery</strong></a></td><td>Part III · Ch 10 <em>Grounded Intent</em>, Ch 11 <em>Trajectory Planning</em></td><td><strong>5A</strong> Push the loop until it breaks: horizons, moving targets, language.</td><td><strong>Lab 5</strong></td></tr>
    <tr><td align="center">10</td><td rowspan="2">Lab 6<br><a href="labs/lab-06-safety-governor.md"><strong>Safety Governor: Build It, Then Break It</strong></a></td><td>Part III · Ch 12 <em>Safety Enforcement</em>, Ch 13 <em>Silicon Placement</em></td><td><strong>6A</strong> Replace the staff governor with your own.</td><td><strong>Capstone task frozen</strong></td></tr>
    <tr><td align="center">11</td><td>Part IV · Ch 14 <em>Supervisory Intervention</em>, Ch 15 <em>Adversarial Verification</em></td><td><strong>6B</strong> Attack it: freeze the brain, corrupt packets, cut power.</td><td><strong>Lab 6</strong></td></tr>
    <tr><td align="center">12</td><td rowspan="3"><a href="labs/lab-capstone-studio.md"><strong>Capstone: Physical Release Defense</strong></a></td><td>Part IV · Ch 16 <em>Safe Release</em></td><td><strong>C1</strong> Build your own task on the full pipeline.</td><td></td></tr>
    <tr><td align="center">13</td><td>Part IV · Ch 17 <em>Responsible Physical AI</em></td><td><strong>C2</strong> Survive a peer team's attacks and harden the system.</td><td></td></tr>
    <tr><td align="center">14</td><td>Conclusion · Ch 18 <em>The Epistemic Frontier</em></td><td><strong>C3</strong> Final trials and oral defense of your release decision.</td><td><strong>Capstone</strong></td></tr>
  </tbody>
</table>

Each lab teaches one core set of concepts and is split into weekly **parts**. Every part ends with a **checkpoint**, an artifact staff see before the team moves on.

From week 1 to week 9 the arm runs under a staff-written baseline governor; in Lab 6 students replace it with their own. The digital twin built in Lab 1 is used in every lab after it. Every lab follows the [lab template](LAB_TEMPLATE.md) and carries a status: all are `skeleton` today, then `draft` when every section is written, then `bench-verified` when staff have run them end to end on the reference station.

---

## 4. Reference Documents

| Document | Description |
|:---|:---|
| 📋 **[Master Course Syllabus](curriculum/syllabus.md)** | Course description, weekly schedule, grading, and course policies. |
| 🏷️ **[Physical AI Competency Card](curriculum/student-competencies.md)** | The scope test and the 12-skill rubric signed off at the bench. |
| 🔧 **[Station Reference Card](labs/station-reference.md)** | Bench hardware, software stack, wiring, engineering rules, and telemetry schema. |
| 🧩 **[Lab Template](LAB_TEMPLATE.md)** | The section-by-section structure every lab follows, with guidance for whoever writes it. |

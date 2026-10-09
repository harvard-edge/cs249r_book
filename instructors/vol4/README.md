# Physical AI Systems

**ETH spring project seminar · Volume IV**

**Instructor:** Professor Vijay Janapa Reddi<br>
**Teaching assistant:** Dr. Andrea Mattia Garavagno

> **Course planning status:** The course structure is set and the labs are being written by the teaching team. Final meeting times, enrollment, lab access, and motion limits will be announced through the ETH course listing and course staff.

## What this course is about

Students build one physical AI system across the semester. A learned policy on the Arduino UNO Q proposes a bounded action; its STM32 microcontroller permits or refuses it; a robot changes the world; and the system measures what actually happened before deciding again. The first proposed body is a Seeed SO-101 arm using the LeRobot data and policy workflow. The engineering question is whether the whole **sense → propose → permit → act → observe → revise** loop works under physical delay, imperfect motion, and changed scenes.

A capstone must satisfy the [Volume IV scope test](../../books/vol4/01_boundary/01_boundary.qmd): a learned decision, consequential physical feedback, and delegated actuation inside an independent permission boundary. A model score or a camera classifier that triggers a fixed arm script does not establish that result. Teams compare their system with a nonlearned baseline and report measured physical outcomes, refusals, corrections, and failures.

## Who this course is for

This seminar is intended for advanced undergraduate and graduate students in computer science, electrical or computer engineering, machine learning, and robotics. Students may arrive with different strengths; the team project brings those perspectives together. Before the course, you should be able to:

- Write and debug Python, work with arrays, and use Git to share code and results.
- Explain the difference between training a model and running inference, and interpret a held-out test result.
- Use basic vectors, algebra, probability, and rates of change to reason about measurements and motion.
- Read a system diagram or log and identify where data, commands, and delays enter a computation.

C and prior experience with robot arms, LeRobot, microcontroller firmware, or advanced control theory are helpful but not required. Staff provide the baseline firmware and support its extension. The course develops those connections through measurements, simulation, data collection, policy deployment, and supervised experiments. Students should expect to learn from teammates whose background differs from their own.

**Before week 1:** Read the Volume IV [reader guide and prerequisites](../../books/vol4/frontmatter/prerequisites.qmd) and [Chapter 1](../../books/vol4/01_boundary/01_boundary.qmd). If you want a refresher, the book includes appendices on [machine learning](../../books/vol4/backmatter/appendix_ml.qmd), [systems](../../books/vol4/backmatter/appendix_systems.qmd), and [control](../../books/vol4/backmatter/appendix_control.qmd). You do not need to master every appendix before the seminar begins.

## Syllabus, schedule, and labs

Everything else about the course lives in one place, next to the labs:

- **[Course syllabus](../../labs/vol4/curriculum/syllabus.md):** the weekly rhythm, the 14-week schedule of readings and lab parts, milestones and grading, and course policies.
- **[Studio landing page](../../labs/vol4/README.md):** the system students build, the kit, and how the labs connect week by week.
- **[Competency Card](../../labs/vol4/curriculum/student-competencies.md):** the twelve skills staff sign off at the bench.

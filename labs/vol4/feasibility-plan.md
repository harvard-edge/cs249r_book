# Feasibility plan for the Volume IV labs

**Andrea:** Use the [course page](../../instructors/vol4/README.md), [syllabus](../../instructors/vol4/syllabus.qmd), and linked lab briefs as the proposed class. You have an Arduino UNO Q, an SO-101 arm, and Hugging Face LeRobot. Try the labs as if we had already written the student handouts. Find out which exercises a small team could actually do, what we would have to supply, and what needs to change. You do not need to write fourteen polished handouts now.

For each lab, make the smallest convincing example yourself. Save the artifact a student would produce, note how long it took, and mark **works / works with changes / does not work yet / not tried**. If something fails, say what you tried and what you would try next. The detailed [one-station bench notes](station-pilot/README.md) are a technical reference when you need to investigate hardware; they do not define the student curriculum.

## What the labs are building toward

Students should finish with a system that **senses a physical state, uses a learned model to propose an action, allows the microcontroller to permit or refuse it, acts, measures what happened, and decides again from a new observation**. The common starting example is a slow SO-101 reach toward a soft target. After the first move, someone shifts the target or causes an incomplete move. The next proposal must respond to the observed result or abstain. Students compare that behavior with a simple scripted rule.

This is how the labs exercise the book rather than becoming a model deployment course: early labs establish body, sensing, and physical data; middle labs test learned decisions against real motion, timing, and simulation; later labs test permission, recovery, intervention, and the evidence behind a final claim. The [competency matrix](student-competencies.md) names the abilities students should carry to another robot or simulator.

Training and simulation may run on a workstation. The required local loop asks for a **learned action policy on the UNO Q's Qualcomm side** and physical action that the STM32 can refuse. The stock LeRobot SO-101 USB connection is useful for collecting data, but it does not by itself put the STM32 in the motor command path. Find out whether that arrangement can be built and taught on this kit. Treat local SmolVLA as an experiment until its memory and timing are measured; it is not needed for the first feedback loop.

## Try the proposed labs

The [syllabus](../../instructors/vol4/syllabus.qmd) supplies the weekly readings and meeting format. Each linked brief explains the fuller student exercise. Try each one on a single station and answer the question under it.

1. **[Lab 1 — Boundary and authority](lab-01-boundary.md)**
   - Students show the sensor, processor, motor, and power paths and the arm's state when stopped.
   - Andrea checks whether the command path and safe state are reproducible, and lists missing parts.
2. **[Lab 2 — Body and sensors](lab-02-body-and-sensors.md)**
   - Students calibrate camera and arm feedback, compare requested with measured motion, and define a project task.
   - Andrea checks whether a usable workspace, uncertainty, and sensor loss can be measured in one lab.
3. **[Lab 3 — Action permission](lab-03-action-permission.md)**
   - Students send valid and invalid proposals and show what the STM32 permits or refuses before motion.
   - Andrea checks whether the STM32 can control the SO-101 motor path without a live host bypass, or identifies a simpler setup.
4. **[Lab 4 — Physical episodes](lab-04-physical-episodes.md)**
   - Students record and replay an episode linking observation, requested action, permission, measured motion, and outcome.
   - Andrea checks whether LeRobot data and MCU events can be joined to reconstruct one trial.
5. **[Lab 5 — Baseline and learning](lab-05-baseline-and-learning.md)**
   - Students hold out data, inspect a learned action policy, and compare it with a scripted rule on matched starts.
   - Andrea checks whether a small dataset, model, and baseline can be supplied and used within the allotted time.
6. **[Lab 6 — Local feedback loop](lab-06-local-feedback-loop.md)**
   - Students run a learned action policy on the UNO Q, move with MCU permission, observe a changed scene, then revise or abstain.
   - Andrea checks whether this complete sequence works on the kit and what every team would need to reproduce it.
7. **[Lab 7 — Simulation gap](lab-07-simulation-gap.md)**
   - Students use the same starts and actions in two simulators and on hardware, then explain a measured difference.
   - Andrea checks whether two workable simulations and matching physical traces can be supplied.
8. **[Lab 8 — Action chunks](lab-08-action-chunks.md)**
   - Students compare an ACT action chunk with a policy that observes between moves, including a changed scene.
   - Andrea checks whether a supplied checkpoint and a practical physical trial fit the lab time.
9. **[Lab 9 — Language and placement](lab-09-language-and-placement.md)**
   - Students test whether a task-matched language-conditioned model changes actions and whether its runtime fits the task.
   - Andrea measures SmolVLA on the board and identifies a useful workstation or recorded comparison if it cannot run locally.
10. **[Lab 10 — Authority under fault](lab-10-authority-under-fault.md)**
    - Students show refusal of stale or excessive requests, cutoff behavior, and no old motion after restart.
    - Andrea checks whether students can inject these faults safely and observe both logs and physical behavior.
11. **[Lab 11 — Incomplete motion](lab-11-incomplete-motion.md)**
    - Students detect that the arm or object did something different from the command, then correct or abstain.
    - Andrea finds a repeatable, harmless disturbance whose outcome students can measure.
12. **[Lab 12 — Human correction](lab-12-human-correction.md)**
    - Students record a correction and compare physical results before and after a documented change.
    - Andrea tests whether correction fits the week or whether this time should be used for project repair.
13. **[Lab 13 — Rehearsal and release case](lab-13-release-case.md)**
    - Students freeze the system and trial procedure, exchange a fault case, and explain evidence for a bounded claim.
    - Andrea checks whether another team can run the project from its instructions.
14. **[Lab 14 — Capstone trial](lab-14-capstone-trial.md)**
    - Students run unfamiliar physical trials, compare with the rule, and defend the full sense → propose → permit → act → observe → revise trace.
    - Andrea checks whether raw evidence, including failures and abstentions, is enough to score the outcome.

Weeks 1–6 give everyone the same starting system. After that, teams use the labs to investigate their own project question. Weeks 8–9 offer a choice of deeper study; week 12 can be repair; week 13 is rehearsal. Only weeks **2, 6, 10, and 14** need graded submissions. Test whether this rhythm is realistic rather than turning every row into a separate assignment.

## What to send back

For **each lab**, send a short entry in this form:

- **Result:** works / works with changes / does not work yet / not tried.
- **What you tried:** the task, kit and software used, and approximate active lab time.
- **Student artifact:** a link to the example trace, episode, model output, measurement, or physical video. A failed attempt is useful evidence.
- **What we would supply:** starter code, checkpoint, dataset, fixture, simulator, fault script, or staff setup.
- **What should change:** the smallest revision that would make the exercise teach its intended concept.
- **Next experiment:** if unresolved, what you will try, which part or support it needs, and when you expect an answer.

Also return one short station summary: actual parts and per-station cost; what standard LeRobot already does; whether a learned action policy runs locally on the Q; whether the STM32 can control or refuse the arm's live commands; whether the changed-scene feedback loop works; and how many teams can share a station. Show one other person trying the instructions. Keep the raw evidence linked so we can judge the claims.

If the SO-101 cannot support STM32-controlled actions, tell us whether a [one-axis station](lab-sequence.md) can teach that part while the arm teaches LeRobot data and policy work. If local learned action inference does not run on the Q, state that directly so we can revise the hardware or course objective before giving students the lab.

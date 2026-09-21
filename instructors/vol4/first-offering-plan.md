---
draft: true
---

# First-offering teaching plan

**Status:** Instructor and teaching-assistant planning note. The [student page](README.md) gives the course outline; the [syllabus draft](syllabus.qmd) gives the provisional assessment. Confirm the timetable and station capacity after the [hardware feasibility pilot](../../labs/vol4/staff/feasibility-plan.md).

## Recommendation

Keep fourteen teaching weeks and one continuing team project. Treat the fourteen linked labs as **weekly studio prompts**, not fourteen independently graded assignments. Grade four cumulative milestones: the week-2 task charter, week-6 local feedback loop, week-10 authority test, and week-14 capstone. Week 13 is a rehearsal and repair buffer. Every other week, teams update a shared project notebook and show one raw trace at the bench; staff give a quick formative check rather than grading another report.

This keeps the physical AI standard intact. Every capstone still needs a learned action proposal on the UNO Q, STM32 permission, measured physical change, a fresh observation that changes the next decision or leads to abstention, a matched baseline, and fault evidence. The lighter schedule removes duplicate submissions, not the causal loop.

## Weekly operating rhythm

- **Seminar:** Vijay chairs a provisional 90-minute meeting with a short framing of one systems question, a rotating student trace, and discussion. Reuse the same question template: *What did the system predict? What actually moved? What evidence changed the next decision?* One focused section of the linked chapter is required; a second linked chapter and research papers are background unless announced for a specific project question. No new slide deck or paper presentation is required each week.
- **Studio:** Andrea leads the provisional three-hour supervised lab on a prequalified station. Open with its safe state and one measurement, then let teams work on their own project. Andrea records a brief bench check and routes unresolved hardware or scope decisions to Vijay. The station pilot determines how many teams Andrea can supervise safely; enrollment follows that capacity.
- **Team record:** One shared, versioned notebook accumulates code and model revisions, raw traces, physical outcomes, and next steps. A weekly entry can be a trace and a few sentences. Students submit a polished packet only at the four graded milestones. Each student identifies evidence they personally collected or interpreted for the final defense.
- **First offering:** Use one common arm task, fixture, baseline, simulator fixtures, and starter checkpoint. Allow different project questions and methods inside that station. Do not add a second embodiment, mandatory on-board SmolVLA, or a new STM32-to-servo interface during the semester unless staff have already tested it.

## Load-bearing weeks

| Week | Teaching purpose | Assessment |
|---:|---|---|
| 1–2 | Orient to the station; measure the safe state; choose a bounded task. | **Week 2:** task charter and station signoff. |
| 3–5 | Exercise permission, record episodes, and compare the baseline with a compact policy. | Bench checks and notebook entries. |
| 6 | Demonstrate local learned action, permitted movement, changed observation, and a revised proposal or abstention. | **Week 6:** witnessed feedback loop. |
| 7 | Compare simulation predictions with physical traces. | Notebook evidence carried into the final report. |
| 8–9 | Choose one deeper policy investigation: action chunks or language/model placement. Use the other brief as a staff demonstration or trace discussion. | Ungraded design review; no new deployment requirement. |
| 10 | Inject stale, duplicate, and disarmed proposals; test refusal and cutoff. | **Week 10:** authority under fault. |
| 11–12 | Test incomplete motion and recovery. Use week 12 for a corrective demonstration or project repair, depending on the team's evidence. | Bench checks; no separate retraining requirement. |
| 13 | Freeze versions, rehearse the trial, repair faults, and exchange one safe test case. | Formative rehearsal; no new grade. |
| 14 | Run held-out physical trials and defend a bounded claim. | **Week 14:** concise system report and individual oral defense. |

## Ownership and preparation boundary

| Vijay | Andrea | Teams |
|---|---|---|
| Set the conceptual scope and capstone standard; chair the seminar; review the four milestone rubrics and unresolved project decisions; join the loop, authority, and final defenses. | Qualify and maintain the station before term; prepare one reusable bench card per week from the lab briefs; lead supervised labs and bench checks; organize bookings, spares, replay traces, and first-pass milestone feedback. | Operate within the approved envelope; maintain one shared notebook; collect and interpret physical evidence; present rotating traces; assemble four milestone packets and the final report. |

Before enrollment, Andrea should work through the [lab feasibility plan](../../labs/vol4/staff/feasibility-plan.md) as a student would, reproduce the local policy → MCU permission → physical action → new observation path, time a full student-style lab session, and report how many teams one station and one supervisor can support. The [one-station bench notes](../../labs/vol4/staff/station-pilot/README.md) are available for technical investigation. If capacity is small, cap enrollment or add support. The course should not depend on Vijay debugging student USB, firmware, or arm calibration problems during the term.

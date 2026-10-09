---
draft: true
---

# First-offering teaching plan

**Status:** Instructor and teaching-assistant planning note. The [student page](README.md) gives the course outline; the [syllabus draft](syllabus.qmd) gives the provisional assessment. Confirm the timetable and station capacity after the [hardware feasibility pilot](../../labs/vol4/staff/feasibility-plan.md).

## Recommendation

Keep fourteen teaching weeks and one continuing team project. Follow the [canonical syllabus](../../labs/vol4/curriculum/syllabus.md): six labs and five graded groups. Labs 1–2 carry 15 percent, Labs 3–4 carry 25 percent, Lab 5 carries 10 percent, Lab 6 carries 25 percent, and the capstone carries 25 percent. The paired lab checkpoints fall in weeks 3/4 and 6/8; Lab 5 is due in week 9, Lab 6 in week 11, and the capstone in week 14. The capstone proposal is due in week 8 and the task freezes in week 10. Week 13 combines peer fault exchange with rehearsal and repair. Each week, teams update a shared project notebook and show one raw trace at the bench; staff give a quick formative check rather than grading another report.

This keeps the physical AI standard intact. Every capstone still needs a learned action proposal on the UNO Q, STM32 permission, measured physical change, a fresh observation that changes the next decision or leads to abstention, a matched baseline, and fault evidence. The shared notebook carries formative evidence into the scheduled submissions.

## Weekly operating rhythm

- **Seminar:** Vijay chairs a provisional 90-minute meeting with a short framing of one systems question, a rotating student trace, and discussion. Reuse the same question template: *What did the system predict? What actually moved? What evidence changed the next decision?* Read the scheduled chapters before seminar, as the canonical syllabus requires; focus discussion on one section tied to the week’s systems question. Research papers are background unless announced for a specific project question. No new slide deck or paper presentation is required each week.
- **Studio:** Andrea leads the provisional three-hour supervised lab on a prequalified station. Open with its safe state and one measurement, then let teams work on their own project. Andrea records a brief bench check and routes unresolved hardware or scope decisions to Vijay. The station pilot determines how many teams Andrea can supervise safely; enrollment follows that capacity.
- **Team record:** One shared, versioned notebook accumulates code and model revisions, raw traces, physical outcomes, and next steps. A weekly entry can be a trace and a few sentences. Students submit evidence at the scheduled lab checkpoints and assemble the five graded groups. Each student identifies evidence they personally collected or interpreted for the final defense.
- **First offering:** Use one common baseline teaching task, fixture, baseline controller, simulator fixtures, and starter checkpoint. Allow staff-approved bounded capstone tasks and different project questions and methods inside that station. Do not add a second embodiment, mandatory on-board SmolVLA, or a new STM32-to-servo interface during the semester unless staff have already tested it.

## Load-bearing weeks

| Week | Teaching purpose | Assessment |
|---:|---|---|
| 1–4 | Qualify the station, measure the body, build the digital twin, and compare IK in simulation and on the arm. | **Weeks 3 and 4:** Labs 1–2 checkpoints (15 percent together). |
| 5–6 | Compare a staff-provided learned proposer with the baseline, calibrate the camera, and export/profile on the board. | **Week 6:** Lab 3 checkpoint, carried into Labs 3–4. |
| 7–8 | Collect real demonstrations, fine-tune, and demonstrate the untethered feedback loop. | **Week 8:** Lab 4 and capstone proposal (Labs 3–4: 25 percent together). |
| 9 | Measure chunk-horizon trade-offs and disturbance recovery; retain the short language check. | **Week 9:** Lab 5 (10 percent). |
| 10–11 | Build the governor, review its command paths, and run the declared fault campaign. Freeze the capstone task in week 10. | **Week 11:** Lab 6 (25 percent). |
| 12–13 | Build the frozen task; exchange authorized peer faults; rehearse and repair. | Formative bench checks and notebook evidence. |
| 14 | Run held-out physical trials and defend a bounded claim. | **Week 14:** capstone system report and individual oral defense (25 percent). |

## Ownership and preparation boundary

| Vijay | Andrea | Teams |
|---|---|---|
| Set the conceptual scope and capstone standard; chair the seminar; review the five graded-group rubrics and unresolved project decisions; join the learned-loop, governor, and final demonstrations and defenses. | Qualify and maintain the station before term; prepare one reusable bench card per week from the lab briefs; lead supervised labs and bench checks; organize bookings, spares, replay traces, and first-pass checkpoint and graded-group feedback. | Operate within the approved envelope; maintain one shared notebook; collect and interpret physical evidence; present rotating traces; assemble the scheduled lab evidence and final report. |

Before enrollment, Andrea should work through the [lab feasibility plan](../../labs/vol4/staff/feasibility-plan.md) as a student would, reproduce the local policy → MCU permission → physical action → new observation path, time a full student-style lab session, and report how many teams one station and one supervisor can support. The [one-station bench notes](../../labs/vol4/staff/station-pilot/README.md) are available for technical investigation. If capacity is small, cap enrollment or add support. The course should not depend on Vijay debugging student USB, firmware, or arm calibration problems during the term.

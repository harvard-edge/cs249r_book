# Lab 9 — Does language or model size earn its cost?

**Status:** Draft student lab brief. Andrea will try this exercise on the kit and record the student artifact, time required, starter materials, and any changes in the [feasibility plan](feasibility-plan.md). Staff will set motion limits and reset instructions from the tested station before releasing the handout.

**Read before class:** [Chapter 8: Sensor Perception](../../books/vol4/08_perception/08_perception.qmd); [Chapter 13: Silicon Placement](../../books/vol4/13_placement/13_placement.qmd).

**First-offering role:** Optional deeper investigation. Teams choose this prompt or Lab 8; the other topic is discussed from a staff trace. Record evidence in the shared notebook without a separate graded report.

**Start from:** Lab 7 common policy traces and a task-matched SmolVLA artifact or staff rollout.

## Experiment

Test a task-matched SmolVLA checkpoint or staff-provided rollout with two instructions applied to the same scene. Compare it with ACT and the compact Q policy. Profile local SmolVLA on the Q only if staff have shown a viable runtime; otherwise analyze the measured staff trace and keep the compact policy in the physical loop.

## Notebook evidence and exit check

Record paired-instruction action differences, matched **physical** task outcomes where the model controls the qualified station, memory and latency measurements, and a model placement decision. The bench review checks whether the team distinguishes a language-conditioned change in the physical plan from a visual coincidence and states whether the Q can meet its deadline. If the model cannot control the qualified station, label this an offboard comparison rather than a physical AI demonstration. This exercises **C11, C12**.

**Carry forward:** paired-instruction outcomes and placement verdict. Keep the raw trace and artifact revision so the next lab can reconstruct this result.

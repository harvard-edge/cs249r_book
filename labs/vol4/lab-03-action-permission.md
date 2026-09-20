# Lab 3 — Who permits an action?

**Status:** Draft student lab brief. Andrea will try this exercise on the kit and record the student artifact, time required, starter materials, and any changes in the [feasibility plan](feasibility-plan.md). Staff will set motion limits and reset instructions from the tested station before releasing the handout.

**Read before class:** [Chapter 3: The Cognitive Brain](../../books/vol4/03_brain/03_brain.qmd); [Chapter 4: The Nervous System](../../books/vol4/04_nervous/04_nervous.qmd).

**Start from:** Lab 2 envelope and staff-tested STM32-to-servo interface.

## Experiment

Send synthetic one-joint proposals through the supplied STM32-to-servo interface: valid, duplicate, expired, disarmed, and out of range. Progress to a slow bounded pose only after the one-joint trace is correct. Do not ask students to design an untested servo-bus interface during term.

## Submit and exit check

Submit the proposal schema and linked requested, mapped, enforced, and measured action traces. Pass when the MCU refuses every injected invalid request and no alternate host path can move the arm. This exercises **C3, C6, C13**.

**Carry forward:** proposal schema and refusal trace. Keep the raw trace and artifact revision so the next lab can reconstruct this result.

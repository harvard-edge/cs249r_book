# Lab 1 — Where is the physical AI boundary?

**Status:** Draft student lab brief. Andrea will try this exercise on the kit and record the student artifact, time required, starter materials, and any changes in the [feasibility plan](feasibility-plan.md). Staff will set motion limits and reset instructions from the tested station before releasing the handout.

**Read before class:** [Chapter 1: The Causal Boundary](../../books/vol4/01_boundary/01_boundary.qmd).

**Start from:** station schematic and isolated motor power.

## Experiment

With motor power isolated, inventory the camera, Qualcomm Linux, Bridge, STM32, servo bus, and cutoff. Trace one hypothetical proposal and identify every component that can cause motion. Observe boot, disconnect, and power-off behavior on the qualified station.

## Submit and exit check

Submit a power and command diagram, safe-state observations, and one counterexample in which a model output would not establish physical AI. Pass when the team can identify the only live actuator command route and the measured safe state. This starts **C1, C6, C13**.

**Carry forward:** authority diagram and observed safe state. Keep the raw trace and artifact revision so the next lab can reconstruct this result.

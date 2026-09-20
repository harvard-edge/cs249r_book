# Lab 7 — What do the simulators miss?

**Status:** Draft student lab brief. Andrea will try this exercise on the kit and record the student artifact, time required, starter materials, and any changes in the [feasibility plan](feasibility-plan.md). Staff will set motion limits and reset instructions from the tested station before releasing the handout.

**Read before class:** [Chapter 9: Spatial Memory](../../books/vol4/09_memory/09_memory.qmd); [Chapter 10: Grounded Intent](../../books/vol4/10_intent/10_intent.qmd).

**Start from:** Lab 6 physical traces and two staff simulation fixtures.

## Experiment

Apply identical initial states and actions to the staff-supplied kinematic model, a second pinned simulator, and the arm. Identify one dynamical parameter or fit a small next-state predictor using supplied training trajectories; evaluate the resulting prediction against held-out physical reaches. Students configure and compare the fixtures rather than build two engines.

## Submit and exit check

Submit observation/action contracts, matched rollouts, one-step and multi-step errors, and one physical failure explained by model mismatch. Pass when the team identifies which prediction it will trust for its project. This exercises **C5, C9**.

**Carry forward:** matched rollout and model-error report. Keep the raw trace and artifact revision so the next lab can reconstruct this result.

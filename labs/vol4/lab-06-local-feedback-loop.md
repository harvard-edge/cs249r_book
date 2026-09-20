# Lab 6 — Can the learned system close one physical loop on the Q?

**Status:** Draft student lab brief. Andrea will try this exercise on the kit and record the student artifact, time required, starter materials, and any changes in the [feasibility plan](feasibility-plan.md). Staff will set motion limits and reset instructions from the tested station before releasing the handout.

**Read before class:** [Chapter 7: Closed-Loop Evaluation](../../books/vol4/07_evaluation/07_evaluation.qmd).

**Start from:** Labs 3–5 artifacts and qualified UNO Q station.

## Experiment

Run the compact policy locally on Qualcomm Linux. Compare its outputs with the workstation, send a bounded proposal through the MCU, and measure joint, tool, and target state. Staff then shift the soft target within the qualified workspace or cause a safe incomplete move. Capture a fresh observation and require the learned component to produce a state-dependent correction or abstention. Run the scripted baseline against the same disturbance. Use the slow reach task and a staff checkpoint if a team's model is not ready.

## Submit and exit check

Demonstrate a witnessed model → permission → measured move → changed scene → new observation → revised decision. Submit the model revision, memory and latency trace, four action taps, target/tool state before and after, rule comparison, and a refusal or abstention. The second proposal must respond to measured state rather than execute a queued script. This **week-6 demonstration** exercises **C8, C10, C12, C15**.

**Carry forward:** witnessed changed-scene learned loop. Keep the raw trace and artifact revision so the next lab can reconstruct this result.

# Student station bench card template

**Do not release this card until every blank is filled from the qualified station and a second staff member has followed it.** Keep a printed copy beside each station. The [pilot protocol](README.md) defines the qualification evidence.

## Identity and limits

- Station ID and build revision:
- Board image, MCU firmware, LeRobot, model, and calibration revisions:
- Qualified task and fixture:
- Allowed joints and gripper actions:
- Measured travel, step, speed, duty, and session limits, with units:
- Camera and joint-feedback health check:
- Motor-power cutoff location and tested action:
- Supervisor and incident contact for this lab:

## Before enabling motor power

1. Confirm the fixture, guard, soft target, camera, and cutoff match this card.
2. Confirm the arm is in the documented rest pose and the workspace is clear.
3. Confirm the **only** live command path is the qualified Qualcomm → Bridge → STM32 → servo-bus route. The stock host USB motor command path must be disconnected during governed trials.
4. Start logging and record the run ID, model revision, calibration, operator, and target layout.
5. Check the camera frame, measured joints, MCU `DISARMED` state, and physical cutoff. Ask the supervisor to authorize the first arm session.

## One trial

1. Record the starting image, measured joint state, tool/target position, and predicted outcome.
2. Arm under the station procedure; issue one bounded proposal and record its MCU decision.
3. Measure what moved. After the first action, apply only the disturbance specified for this lab.
4. Record a **new** observation and the next learned proposal or abstention. Record any MCU refusal.
5. Measure final tool and target state. Label success, failure, abstention, and any intervention using the frozen scoring rule.
6. Disarm, return to the qualified rest state, and verify no queued action occurs on rearm.

## Stop and reset

- **Stop immediately when:** [list station-specific motion, camera, logging, power, or communication faults].
- **To stop:** [exact physical cutoff action and expected behavior].
- **To reset:** [exact measured rest-state procedure; no automatic restart].
- **To rearm:** [explicit supervisor-approved procedure and fresh-observation check].
- **If the station is unavailable:** use [named replay artifact] for analysis and book a supervised physical trial; replay does not replace a required live trial.

## Exit check

The team's notebook links one run ID to the initial observation, learned proposal, MCU decision, measured move or refusal, changed scene, new observation, revised decision, and task outcome. Record the next experiment and any discrepancy. The instructor-facing [competency matrix](../student-competencies.md) states which capability this lab exercises.

**Qualified by:** [first staff member, date, trace]<br>
**Reproduced by:** [second staff member, date, trace]

# One-station work report template

Copy this file into the private pilot record for each station. Attach raw traces and uncut videos by path; do not replace measurements with a success summary.

## Station identity

| Field | Measured or observed value |
|:---|:---|
| Pilot date and operators | |
| UNO Q SKU, RAM, board image, serial | |
| App Lab and MCU firmware revision | |
| SO-101 follower, servo, controller, and supply variants | |
| Leader or other teleoperator | |
| Camera, mount, light, and fixture | |
| Motor cutoff, interface board, wiring revision | |
| Workstation OS, Python, LeRobot, Feetech versions | |
| Model, dataset, preprocessing, and simulator revisions | |

## Inventory and missing parts

For each item, record **in hand / acquire / design**, the exact part or open question, cost, and lead time.

- Follower and bus controller:
- Teleoperation input:
- Camera and powered hub:
- Motor supply and independent cutoff:
- MCU-to-servo interface:
- Soft fixture, mount, and guard:
- Spares:

## Work checklist

For **each step**, record **works / does not work / not tried**, an evidence path, the observed failure, and the smallest next experiment. Mark a result as working only after it has been repeated on the delivered station.

0. Inventory and safe state:
1. Stock LeRobot episode and replay:
2. UNO Q Bridge without motors:
3. MCU refusal, one servo, governed arm:
4. Learned action policy on Q:
5. Disturbed loop, baseline, faults:
6. Second-person student rehearsal:

For each numbered step above, add: **Result; exact build and software revisions; measured observation; raw trace or video path; time spent; missing part or failure; next experiment and expected date.**

## Weekly lab preparation

For each [lab brief](../lab-blueprints.md), record **ready / ready with replay / needs revision**, a link to the example student artifact, active bench minutes, and what Andrea had to supply or fix. Prioritize labs 1–6, 10, and 14. Record station booking needs and any exercise that cannot fit the class period.

## Trial protocol and result

Write the frozen held-out start positions, target markers, scripted baseline, learned model, camera outcome rule, disturbance action, and reset procedure here. Record every trial, including failures; use the same starts and disturbance for both controllers. Copy the block below for all eight matched trials.

### Trial ID: [fill in]

- Start and controller:
- Condition: unchanged / target shifted
- First proposed, permitted, and measured move:
- New observation and second proposal or abstention:
- Final measured tool and target state:
- Outcome and raw trace path:

## Refusal and recovery results

For each fault, write the expected behavior **before** the test, then record the MCU decision, measured motion or power, whether a queued command moved after rearm, and the trace path.

- Duplicate or expired request:
- Out-of-range goal:
- Camera loss:
- Linux process stop:
- Physical cutoff:
- Reset and rearm:

## Measurements and teaching capacity

For each measure, include the method, units, result, and raw data path.

- Model load and peak memory on Q:
- Host-versus-Q output difference:
- Capture → proposal latency distribution:
- Proposal → MCU decision → measured motion latency:
- Observation age at each action:
- Setup, reset, and trial minutes per team:
- Staff interventions during second-person run:
- Teams per station and required booking slots:

## Course recommendation

- **Choose one:** SO-101 with STM32-controlled actions and local Q policy / SO-101 for LeRobot plus a separate one-axis STM32-controlled Q station / revise the hardware requirement.
- **Evidence supporting that choice:**
- **Unresolved technical facts and smallest next experiments:**
- **Parts, labor, and lead time before another station can be built:**
- **Student labs ready now; labs requiring replay or redesign:**
- **Second person's reproduction verdict and signature:**

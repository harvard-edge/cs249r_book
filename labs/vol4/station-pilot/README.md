# One-station pilot: bench protocol for Andrea

**Status:** Staff instructions to follow when the kit arrives; no UNO Q/SO-101 integration is claimed to have worked yet. Build **one** station and fill the [work report](report-template.md) after each step. The [feasibility plan](../feasibility-plan.md) connects these tests to the student labs and book principles.

## Target episode

Use one fixed camera, a marked low-energy workspace, and a soft target. From a held-out start, the arm makes one guarded reach. After that move, the operator shifts the target within the approved workspace or produces a qualified incomplete move. A learned policy **running on the UNO Q** sees the new state and proposes a different bounded action or abstains. The STM32 permits or refuses each proposal before the servo bus receives a goal. An independent observation measures the tool and target after motion. Run the same starts and changes with a scripted baseline.

First verify the stock LeRobot USB path. Disconnect it from the live motor bus when testing STM32-controlled actions. A host-side clamp, an LED Bridge demo, a joint-position readback, or a model that selects a fixed script is insufficient for the target episode.

## Sources and version rule

Use the [Arduino UNO Q user manual](https://docs.arduino.cc/tutorials/uno-q/user-manual/), [Arduino App Lab examples](https://docs.arduino.cc/software/app-lab/getting-started/examples), and the [LeRobot installation](https://huggingface.co/docs/lerobot/v0.6.1/installation), [SO-101](https://huggingface.co/docs/lerobot/v0.6.1/so101), and [real-robot workflow](https://huggingface.co/docs/lerobot/il_robots) guides as implementation references. LeRobot documentation has versioned and `main` pages; choose **one tested release**, record the installed package version and its matching guide, and do not combine commands from incompatible versions. The command templates below follow the currently documented LeRobot CLI; confirm each with that release's `--help` before motion.

Record the UNO Q RAM variant, board image, App Lab version, MCU firmware hash, LeRobot version, Feetech adapter variant, motor supply, camera, policy hash, dataset revision, and exact host OS in the report. Keep the pilot dataset local (`--dataset.push_to_hub=False`) until the team decides what may be shared.

## Step 0 — Receive and make the station safe

1. Photograph the box contents and name every delivered part. Verify whether there is a **follower arm, leader or other teleoperator, USB motor adapter, motor supply, camera, cable, mount, and cutoff**. An assembled follower alone does not provide the rest of the station.
2. Draw the as-delivered power and command path. Label the board supply and motor supply separately. Record connector polarity and the delivered servo/controller model from its label, not from a product listing.
3. Mount the arm and soft fixture. Put a reachable physical motor-power cutoff in series with the motor supply. With motor power isolated, verify that the host and UNO Q can boot and log. Then test motor-power removal and return with the arm unloaded; record any unintended movement or retained command.
4. Set a provisional low-speed, low-travel envelope using actual calibration and observed behavior. Write the stop, reset, and rearm steps on a bench card before any trial. Do not copy numerical limits from another station.

**Save and check:** inventory with photographs, power diagram, boot/cutoff video, measured safe state, and a list of absent parts or unknown electrical facts. Stop here if the cutoff or safe reset cannot be demonstrated.

## Step 1 — Reproduce stock LeRobot on a workstation

Install the chosen LeRobot release with its hardware and recording extras in an isolated Python 3.12 environment, as required by the current installation guide. For the current stable documentation, the package shape is:

```bash
python -m pip install 'lerobot[core_scripts,feetech]==0.6.1'
python -m pip show lerobot
lerobot-find-port
lerobot-find-cameras
```

Write down the observed follower, leader, and camera identifiers. **Do not run `lerobot-setup-motors` on a delivered assembled arm until you know whether IDs and baud rates are already configured:** that command writes motor configuration. Follow the version-matched SO-101 guide if setup is actually needed. Calibrate the follower and available teleoperator using their observed ports and stable IDs. Verify the calibration files are retained for the next session.

With the arm inside the marked workspace, run a slow teleoperation session using the documented `lerobot-teleoperate` command for the actual input device. If the kit lacks a leader or another documented teleoperator, record that dependency as **blocked**; do not invent an input path during the pilot. Configure the camera with the port/index returned by `lerobot-find-cameras`.

For a follower plus SO-101 leader, fill these values from the discovered devices. The `:?` checks make the commands stop if a value was not set:

```bash
: "${FOLLOWER_PORT:?Set from lerobot-find-port}"
: "${FOLLOWER_ID:?Use one stable calibration ID}"
: "${LEADER_PORT:?Set from lerobot-find-port}"
: "${LEADER_ID:?Use one stable calibration ID}"
: "${CAMERA_INDEX:?Set from lerobot-find-cameras}"

lerobot-calibrate --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID"
lerobot-calibrate --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID"

lerobot-teleoperate \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID"
```

Once teleoperation and reset are repeatable, record three guarded reach episodes locally. Recheck the selected release's `lerobot-record --help` before using this template:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id="$FOLLOWER_ID" \
  --robot.cameras="{front: {type: opencv, index_or_path: ${CAMERA_INDEX}, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id="$LEADER_ID" \
  --dataset.repo_id=pilot/so101_guarded_reach \
  --dataset.num_episodes=3 \
  --dataset.single_task="Guarded reach to a soft target" \
  --dataset.push_to_hub=False
```

Locate and save the local dataset directory and inspect one episode. Replay it only after checking that its recorded motion stays inside the current envelope. Pin the exact command and dataset path in the report; if the installed release rejects a flag, record the correction rather than silently switching versions.

**Save and check:** exact commands with substituted ports and IDs, calibration files, camera frame, one replayable episode, and an uncut teleoperation/recording video. Log any camera or motor cadence warnings. This step establishes native LeRobot behavior only.

## Step 2 — Test UNO Q Linux ↔ STM32 communication without motors

Set up the delivered UNO Q in [Arduino App Lab](https://docs.arduino.cc/software/app-lab/). With motor power physically disconnected, run a built-in example that sends a Linux-side command through Bridge to an MCU-controlled LED or pin, such as the documented Pin Toggle example. Show an accepted command, a repeated command, and loss/restart of the Linux app. Capture both Linux and MCU logs and the observed output state.

On the Q, save the outputs of these read-only commands with the board image and power setup:

```bash
cat /etc/os-release
uname -a
free -h
df -h
```

**Save and check:** one Bridge request/response trace with MCU-side output, both software versions, and restart behavior. This does **not** prove any arm interface, servo timing, or safety authority.

## Step 3 — Put the STM32 in the motor command path, first with no motor load

The stock SO-101 motor bus uses a host USB adapter. Before connecting a servo to the MCU route, identify its bus controller, signal voltage, protocol, baud rate, servo power, ground, and required transceiver or level conversion from the delivered hardware and documentation. Draw the proposed circuit and have a second staff member check it. The UNO Q header is **not** assumed electrically compatible with the servo bus. Keep the stock USB controller disconnected from the live servo bus during governed tests.

Implement and log an MCU authority state machine with these minimum states: `DISARMED`, `ARMED`, and `FAULT`. The MCU starts disarmed, accepts only bounded fresh proposals while armed, rejects a duplicate sequence or out-of-envelope goal, and returns a reason for every refusal. A Linux timestamp alone cannot prove freshness to the MCU; use an MCU-local receipt deadline/watchdog and record how a session is renewed. On communication loss, reset, or cutoff, show the measured safe state and require an explicit rearm before new motion. The physical cutoff remains independent of the software state machine.

Use the same structured record for every request:

```text
run_id, trial_id, observation_id, proposal_seq, linux_send_time,
mcu_receive_time, mcu_state, requested_action, mapped_action,
decision, refusal_reason, enforced_action, motor_goal,
measured_joint_state, measured_tool_state, measured_target_state,
next_observation_id, outcome
```

First replay synthetic proposals with the servo bus disconnected. Write the expected result before each case:

| Proposal or condition | Required observation |
|:---|:---|
| New sequence, armed, inside measured envelope | Accept once; log the mapped and enforced goal. |
| Same sequence replayed | Refuse; no second motor goal. |
| MCU-local request deadline expired or Linux stops | Refuse or enter the qualified hold/stop state; no new motor goal. |
| Disarmed or faulted | Refuse regardless of model confidence. |
| Goal outside calibrated travel or measured step limit | Refuse or clip according to the frozen policy; record which occurred. |
| Cutoff removed and power returned | Remain disarmed; no queued command executes before deliberate rearm. |

Verify the MCU log and output-enable state for each. Then connect **one unloaded, guarded servo** through the qualified interface and repeat with the minimum measured travel. Finally extend the same route to the arm, rechecking calibration, gripper, cutoff, and reset. No alternate host cable may reach the live servo bus in governed mode.

**Save and check:** circuit and wiring photographs; firmware and adapter revisions; synthetic refusal matrix; one-servo and arm traces containing requested, mapped, enforced, and measured action; cutoff and no-queued-motion video; second-person reproduction notes. If the STM32-to-servo route does not work, do not claim STM32-controlled arm actions; test the [one-axis fallback](../lab-sequence.md).

## Step 4 — Run a learned action policy locally on the Q

Use the local LeRobot episodes plus a linked sidecar if needed for the four action taps and object outcome. For the first policy, constrain the soft target to an arc where one qualified SO-101 joint can reduce image-plane error. From each demonstration cycle, pair the **current** camera-derived target coordinate and measured joint angle with the operator's **next** small joint change. Split by whole recording session or target position, never adjacent frames from the same episode. Fit a small regression policy to propose the signed bounded joint change or abstention; compare it with a geometric rule under the same starts. The camera-derived coordinate may come from a fixed marker detector for this pilot, but the **action decision itself must be learned from demonstrations**, vary with both target and measured arm state, and change after a new observation. A label that dispatches a fixed reach script is a bring-up check only. Train on the workstation; pin units, normalization, dataset split, model, and export. ACT and SmolVLA are separate comparison experiments, not prerequisites for this first local loop.

Before motor power, feed identical held-out observations to the workstation and the UNO Q. Save both output vectors and their difference, model load time, peak memory, and capture-to-proposal latency. The Q must perform the inference without a workstation or remote service. With the governed arm qualified, run one low-energy learned proposal through the MCU and measure the resulting tool and target state.

**Save and check:** exact model/dataset hashes, training or adaptation command, held-out split, host-versus-Q outputs, runtime measurements, and a physical action trace. If no learned action policy runs locally, report the observed obstacle instead of substituting off-board inference.

## Step 5 — Observe again, compare policies, and test faults

Freeze a small pilot trial set: **two held-out starts × two controllers** (learned and scripted) **× two conditions** (unchanged and target shifted after the first action). That gives eight matched physical trials, enough to test whether the protocol can run; it is not an accuracy estimate for a deployed system. Use the same camera, fixture, target positions, and outcome measure. Record the initial image and measured joints, first proposal, MCU decision, measured move, changed scene, second image, revised proposal or abstention, and final tool/target state. Keep failed trials.

After the matched set, run one trial each for an expired/duplicate proposal, out-of-range goal, lost camera, Linux process stop, physical motor cutoff, and reset/rearm. State the expected no-motion or bounded-motion behavior **before** each fault. Verify that no old command executes after rearm.

**Save and check:** eight linked trial traces, baseline comparison, raw camera and MCU logs, timing distribution, uncut video of at least one changed-scene run and one refusal, and a fault table with observed outcomes. A fresh image followed by an unchanged queued action does not show a revised decision.

## Step 6 — Reproduce as a student would

A second staff member follows only the written instructions, without Andrea's private setup knowledge. Fill the [one-page bench card](bench-card-template.md), then time the full setup, calibration check, episode collection, reset, one baseline trial, one learned disturbed trial, and one refusal test. Count hands-on minutes, queue time, and every intervention by staff. This determines whether a two- or three-student team can finish inside the provisional lab block and how many stations or booking slots are needed. Keep one spare or replay path for other weeks; weeks 6, 10, and 14 still require witnessed physical evidence.

**Save and check:** second-person reproduction log, time per stage, exact station image and files, one-page student bench card, support incidents, and a capacity recommendation. Only then convert the draft lab briefs into runnable student handouts or purchase a class set.

## Decision to return

Mark each step **works**, **does not work**, or **not tried**, with a direct link to evidence. Recommend one of three outcomes: (1) SO-101 with STM32-controlled actions and local Q policy; (2) SO-101 for LeRobot learning plus a one-axis STM32-controlled Q station; or (3) revise the hardware requirement. Include the smallest unresolved experiment, the parts and labor it requires, and the date by which that experiment could change the recommendation.

# SO-101 and UNO Q: a physical AI course candidate

**Status:** Proposed center for the Volume IV hardware studio and [instructor syllabus](../../instructors/vol4/README.md), pending staff qualification. The SO-101 supplies the LeRobot manipulation task; the UNO Q supplies local Qualcomm inference and an STM32 command-permission boundary. The [course operating plan](course-architecture.md) explains how teams use one semester project; the [lab blueprints](lab-blueprints.md) state the student experiments and exit checks. Staff must qualify the arm's command path before releasing assignments; the [one-axis reference sequence](lab-sequence.md) is the fallback. The [competency matrix](student-competencies.md) remains platform independent. This page asks whether the arm can become a complete **physical AI system** under the book's scope test, not merely a robot that runs a learned model.

## What makes this a physical AI system

The [book's scope test](../../books/vol4/01_boundary/01_boundary.qmd) requires a learned model, consequential physical feedback, and delegated actuator authority. On the SO-101, a learned policy must choose an action that changes the arm or an object; joint and object observations must then alter the next choice; and the STM32 must be able to refuse the proposal in the actual motor path. Students use the same meanings in a simulator, a LeRobot episode, a policy rollout, and a physical trace. Every lab states which quantities are measured, estimated, commanded, or assumed.

| Term | Meaning and required evidence |
|:---|:---|
| **Body and frame** | Robot links, joints, camera mount, task objects, coordinate frames, units, calibration revision, and reachable workspace. A pixel coordinate is not an end-effector position. |
| **Observation** | Image, joint readback, and any added contact or object sensor, each with identity, acquisition time, units, and validity. An image and joint sample need an explicit alignment rule. |
| **Intent** | Task instruction and success condition, such as which object to move and where. Test paired instructions from the same scene to establish whether language changes the proposed action. |
| **Proposal** | Model-selected joint target or action chunk with task epoch, sequence, observation ID, and expiry. It is a request, not evidence of motion. |
| **Permission** | STM32 decision for the current request and measured envelope, with an accept, clipped command, or refusal reason. The Linux host cannot command the servos through another live path in the governed configuration. |
| **Execution** | Joint positions read back after the command, plus whether an object moved, a grasp held, or contact occurred. SO-101 servo readback is not a calibrated contact-force measurement. |
| **Outcome** | Task success, correction, abstention, intervention, time, and energy or duty where measurable. Compare with a rule or teleoperated baseline from matched starts. |

The [Physical Data chapter](../../books/vol4/05_data/05_data.qmd) specifies four action taps: **requested** policy or operator action, **mapped** joint command, **enforced** command after the governor, and **measured** response. The lab trace should retain all four plus the next observation and task outcome. A policy's `action` and a servo's `Goal_Position` alone cannot establish physical success. Linux and STM32 clocks need an explicit alignment method; the STM32 judges expiry against its own local cycle or deadline.

## One system, three layers

1. **Learning layer:** LeRobot robot adapter, teleoperation, episode dataset, ACT, and SmolVLA. Training and large-model analysis can run on a workstation. A versioned model used for the required physical loop runs locally on the UNO Q.
2. **Permission layer:** UNO Q Qualcomm Linux sends bounded proposals to its STM32. The STM32 verifies state, sequence, age, joint limits, and permitted motion, then commands the servo bus through a staff-built electrical interface. A separate accessible switch cuts motor power.
3. **Physical layer:** SO-101 links, STS3215 servos, gripper, camera, objects, fixtures, and measured joint/object outcomes. Add a contact sensor only if it has a calibrated, useful measurement; do not infer force from a commanded gripper position.

The stock [LeRobot SO-101](https://huggingface.co/docs/lerobot/so101) path uses a host USB connection to a Feetech serial motor bus. [LeRobot's hardware integration guide](https://huggingface.co/docs/lerobot/main/integrate_hardware) permits a custom `Robot` adapter, but software clipping in `send_action()` does not give the STM32 independent command authority. Staff must prove a single physical command route through the MCU before describing the full arm as MCU-governed. The required bus interface, pin levels, power, polling rate, and reboot behavior are engineering measurements, not assumed features of the UNO Q.

The candidate semester task is a disturbance-and-recovery manipulation cell. Begin with a guarded learned reach toward one soft block, safely shift the target or interrupt the first move, and require a new state-dependent proposal or abstention. Add grasp, placement in a marked tray, and paired language instructions only after that loop works. The next camera frame and an explicit object-position check determine the task result; a joint target alone does not. Staff should replace the task if the delivered arm, camera, gripper, or model cannot perform it repeatably within a measured low-energy envelope.

## Hardware demonstrations staff must reproduce

| Test | Demonstration required before using it in student labs |
|:---|:---|
| **A. Native LeRobot arm** | Inventory the delivered kit; calibrate the follower and available teleoperator; record and replay a short episode; confirm camera, USB, power, joint readback, and low-speed teleoperation on the intended host. |
| **B. One-joint STM32 command path** | Remove any direct host-to-servo command path. Send a proposal from Qualcomm Linux through the STM32 to one unloaded, guarded servo; read back actual position. Refuse stale, duplicate, disarmed, and out-of-range requests; verify power cutoff and reset behavior. |
| **C. Governed SO-101** | Extend the same path to all six joints with calibrated limits, speed bounds, collision fixture, gripper behavior, and synchronized camera/joint logging. A second staff member reproduces the build and fault trials. |
| **D. Local policy** | Run a compact LeRobot-compatible action policy on the exact UNO Q and compare its output and latency with the host. Attempt task-matched SmolVLA locally, reporting memory, cold start, action-chunk latency, sustained operation, and a physical deadline verdict. |

Test A supports shared policy exercises. Tests B and C are required for a capstone claiming STM32-permitted SO-101 actions. If they fail, use the [one-axis reference rig](staff-build-brief.md) for the STM32-controlled physical loop while the arm remains a separate LeRobot station. Test D determines whether students can run SmolVLA locally; a failure there still leaves the LeRobot data and policy curriculum intact, but it must not be reported as on-board VLA deployment.

## How this maps to Volume IV

The course follows one physical episode through the book's stack instead of assigning a separate robot to each chapter. The SO-101 is the worked contact example; the same lab method can later be applied to mobility or process systems.

| Book principle and reading | Lab experiment | Evidence that the principle was exercised |
|:---|:---|:---|
| [Causal boundary](../../books/vol4/01_boundary/01_boundary.qmd) and [body](../../books/vol4/02_body/02_body.qmd) | Calibrate joints, frames, and physical workspace; deliberately compare a requested move with measured motion. | A reproducible operating envelope and a case where command and physical result differ. |
| [Nervous system](../../books/vol4/04_nervous/04_nervous.qmd) and [physical data](../../books/vol4/05_data/05_data.qmd) | Align camera and joint samples; log requested, mapped, enforced, and measured actions across Qualcomm Linux, STM32, and servos. | A replayable episode that identifies the origin and time of every action tap. |
| [Training](../../books/vol4/06_training/06_training.qmd) and [closed-loop evaluation](../../books/vol4/07_evaluation/07_evaluation.qmd) | Collect demonstrations, train ACT, compare with a scripted or teleoperated baseline, and evaluate on held-out physical arrangements. | Physical task outcomes and failure cases alongside offline model metrics. |
| [Grounded intent](../../books/vol4/10_intent/10_intent.qmd), [planning](../../books/vol4/11_planning/11_planning.qmd), and [placement](../../books/vol4/13_placement/13_placement.qmd) | Test SmolVLA with paired instructions in the same scene; compare chunked and stepwise actions and measure inference age on the UNO Q. | A language-dependent action difference, action-chunk expiry, and board timing trace. |
| [Enforcement](../../books/vol4/12_enforcement/12_enforcement.qmd), [intervention](../../books/vol4/14_intervention/14_intervention.qmd), [verification](../../books/vol4/15_verification/15_verification.qmd), and [release](../../books/vol4/16_release/16_release.qmd) | Inject stale or excessive proposals, interrupt motion, collect a human correction, and rerun a frozen task protocol. | Refusal and rearm traces, before/after physical outcomes, and a bounded release claim. |

LeRobot contributes [episode datasets](https://huggingface.co/docs/lerobot/lerobot-dataset-v3), policy training, [simulation](https://huggingface.co/docs/lerobot/envhub_leisaac), and [rollout strategies that record human corrections](https://huggingface.co/docs/lerobot/main/inference). Volume IV asks what those tools mean once computation consumes physical time and a requested action may be filtered, delayed, or only partly realized.

Its current ecosystem also includes reinforcement-learning and reward-model paths, but the first offering should complete one demonstration → imitation → physical evaluation → correction cycle before adding those methods. The course needs a measured system outcome more than a larger menu of algorithms.

## Other LeRobot embodiments and course options

### SO-101

The [supported arm](https://huggingface.co/docs/lerobot/so101) gives us contact-dominated manipulation, demonstrations, action chunks, and visual feedback. It is the best first shared robot because the supplied hardware already has a LeRobot path. Staff still need to test the STM32 command path and local policy.

### LeKiwi

[LeKiwi](https://huggingface.co/docs/lerobot/lekiwi) puts an SO-101-style arm on a mobile base. Students could measure clearance and stopping while also manipulating. It is the strongest next embodiment if the arm works. Its documented mobile host is normally a Raspberry Pi; replacing that host with the UNO Q and governing the wheel and arm buses are separate pilots.

### EarthRover Mini Plus

The [EarthRover Mini Plus](https://huggingface.co/docs/lerobot/earthrover_mini_plus) adds wheeled navigation, camera data, and remote-control latency. Its documented control and video path uses a cloud SDK. It suits a remote-data comparison, but it does not directly realize local Qualcomm inference plus STM32 motion permission.

### Custom UNO Q mobility or process station

[LeRobot's `Robot` interface](https://huggingface.co/docs/lerobot/main/integrate_hardware) can wrap a low-energy UNO Q rover with wheel feedback and measured stopping distance. Staff would need to build its body, driver, sensors, adapter, and operating envelope. The same interface could record a low-voltage thermal cell with heat accumulation, cooling lag, sensor delay, and power cutoff. That would cover the process/energy class physically, but the fixture and suitable policy need a separate pilot.

### Simulation

[LeIsaac SO-101 environments](https://huggingface.co/docs/lerobot/envhub_leisaac) and other LeRobot benchmarks provide repeatable simulated trials. They belong on a workstation and add value when students quantify their gap from measured arm behavior. Staff must pin compatible versions.

The simplest scope for the first offering is **one qualified SO-101 manipulation task**, with comparative seminar or simulation exercises for mobility and process systems. A second physical station requires its own qualification and capacity plan. LeKiwi is a possible extension when a mobile manipulator is worth the added mechanics; a small UNO Q rover is more focused when the lesson is stopping clearance. A thermal cell is needed to exercise the book's process/energy law physically. The fourth book archetype is integrative; LeKiwi couples mobility and manipulation but does not by itself reproduce the full multi-physics humanoid case.

## Fourteen-lab progression

This is the proposed weekly arc once staff reproduce tests A–C. Each lab produces an artifact consumed by the next and tests a physical claim.

| Week | Student experiment | Evidence carried forward |
|---:|:---|:---|
| 1 | Map body, processor, servo bus, motor power, and authority. Test power-off and reboot with the arm unloaded. | Command-path and power diagram; observed safe state. |
| 2 | Calibrate joints and camera; measure joint limits, speed, repeatability, and camera-to-robot frame error. | Calibration revision, units, frames, and measured workspace. |
| 3 | Exercise one-joint proposals and MCU permission, then extend to a bounded arm pose. Inject expired and duplicate requests. | Proposal schema and requested/mapped/enforced/measured trace. |
| 4 | Teleoperate a simple pick-and-place or reach-and-touch task; record synchronized images, joint states, all four action taps, instruction, and outcomes. | Replayable LeRobot episodes and task definition. |
| 5 | Audit episodes, hold out capture sessions, define a rule or scripted baseline, and measure task success on the physical station. | Dataset card, split, baseline, and failure inventory. |
| 6 | Deploy a compact policy locally on the Q; after one permitted move, shift the target or interrupt motion, then use measured tool and target state to revise the learned proposal or abstain. | Board memory/latency trace, changed-scene trial, and matched rule comparison. |
| 7 | Build a simple kinematic model and use a second simulator with the SO-101 geometry; compare predicted and measured motion. | Frame agreement, rollout error, and simulation gap. |
| 8 | Train or fine-tune ACT from episodes, then compare open-loop action chunks with stepwise correction in simulation and on the guarded arm. | Policy artifact, matched physical outcomes, and chunk failure trace. |
| 9 | Evaluate task-matched SmolVLA with paired instructions, compare ACT and the compact Q policy, and measure the board's memory and latency budget. Use a staff trace if local SmolVLA cannot run. | Language-conditioning test, task success comparison, and placement verdict. |
| 10 | Delay inference and expire queued chunks; reject stale, duplicate, disarmed, and out-of-envelope proposals. Interrupt Linux and test the physical cutoff. | Four-tap refusal and timing trace; authority report. |
| 11 | Perturb object position or grasp; require the next observation to change the plan or trigger abstention. Add calibrated contact sensing only if available. | Recovery, incomplete-motion, and object-state evidence. |
| 12 | Interrupt, rearm, and record a human correction; revise the dataset or policy and repeat a matched task. | Before/after physical outcomes and intervention log. |
| 13 | Exchange fault scripts, freeze versions, and defend a bounded operating envelope. | Fault manifest and claim–argument–evidence draft. |
| 14 | Run an unfamiliar arrangement with a safe scene change or incomplete move after the first action. Explain the revised proposal, permission, measured action, and object outcome. | Repeated disturbance trials and defended verdict. |

The arm is the course's **contact-dominated manipulation** example. It does not by itself teach stopping clearance of a mobile body or transport lag of a continuous process. The [cart and thermal proxies](archetype-proxies.md) can remain shared comparison stations or capstone extensions that reuse the same action-tap and outcome record, subject to separate staff qualification. A second simulator is useful only when students compare its predictions with the real arm; [LeRobot's LeIsaac SO-101 environments](https://huggingface.co/docs/lerobot/envhub_leisaac) are one candidate, while the course competency stays simulator independent.

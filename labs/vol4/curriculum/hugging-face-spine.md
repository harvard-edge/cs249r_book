# Hugging Face and UNO Q curriculum spine

**Status:** Design decision and staff feasibility plan, not a claim that a VLA runs on the UNO Q today. The [student competencies](student-competencies.md) remain platform independent. LeRobot supplies the common software workflow; the UNO Q supplies the local compute and physical permission boundary. The [SO-101 course candidate](so101-uno-q-course.md) develops this into an arm-centered lab sequence, pending a test of its STM32-to-servo command path.

## The course experiment

The repeated experiment is **demonstrate → record an episode → train a policy → run it in simulation → propose actions on the UNO Q → permit bounded motion on the STM32 → observe the physical result → correct the data or policy → retest**. Students should see the same observation and action contract at each stage. A model that only names an object has not completed the experiment. An action log that records only the command sent to a motor cannot establish what the motor did.

[LeRobot's custom hardware interface](https://huggingface.co/docs/lerobot/main/integrate_hardware) gives us a concrete `Robot` contract, [LeRobotDataset v3](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) gives an episode format, and [policy rollout](https://huggingface.co/docs/lerobot/main/inference) supports evaluation and human corrections. Staff should pin one tested LeRobot release and record any local adapter changes. The Hub can hold versioned data and model artifacts; publication is a separate decision.

The course adds the book's four action taps beside those tools: requested policy or operator action, mapped command, STM32-enforced command, and measured response, with observation ID and timestamp, model revision, intervention, and next observation. Students compare physical task outcomes with held-out policy metrics. The STM32's permission check must occur in the actual actuator command path; clipping an action in Python does not provide that boundary.

## Model progression and placement

| Stage | Why it is taught | Execution target and staff test |
|:---|:---|:---|
| Nonlearned rule and teleoperation | Establish task, action units, failure modes, and a fair baseline. | Real station and matching simulator; record complete episodes. |
| Compact learned perception or action policy | Prove that a versioned Hugging Face artifact can alter a physical decision locally. | Run on the UNO Q's QRB2210 with measured memory, output agreement, and capture-to-action timing. A MobileNet-based visual policy is a candidate, not the course's intellectual endpoint. |
| [ACT](https://huggingface.co/docs/lerobot/act) | Learn imitation, state/action alignment, and the consequences of predicting action chunks. | Train on a workstation; test in simulation or on the shared SO-101. Pilot UNO Q inference separately before assigning it there. ACT is an action policy, not a language-conditioned VLA. |
| [SmolVLA](https://huggingface.co/docs/lerobot/smolvla) | Test whether visual observation, robot state, and task language can jointly select action chunks for a real task. | Fine-tune and validate with LeRobot off board. **Staff target:** local inference on the exact UNO Q variant, followed by a slow, bounded physical trial only if memory and timing allow. Its published SO-101 results do not establish UNO Q performance. |

The VLA feasibility report must include model and dependency memory at load and during inference, cold-start time, action-chunk latency distribution, camera-to-action age, sustained thermal behavior, and behavior after interruption. Test a reduced image resolution, fewer generated actions, or a documented compression route only when the original result identifies the bottleneck. Preserve host-versus-board output checks. If full SmolVLA misses the measured deadline, report that result and keep the required local loop with a smaller task-specific policy. Do not label that smaller policy a VLA unless language measurably conditions its physical actions.

Qualcomm's [QRB2210](https://www.qualcomm.com/internet-of-things/products/q2-series/qrb2210) and the [UNO Q](https://docs.arduino.cc/hardware/uno-q/) make this a credible deployment experiment, but an available accelerator or a runnable LeRobot installation is not proof that SmolVLA meets a robot's action deadline. Start with a reproducible CPU path; measure any later accelerated path on this board.

## Hardware roles

The low-energy one-axis station is the governed fallback for sensors, timing, simulation mismatch, action permission, and fault injection if the SO-101 STM32-to-servo command path does not work. It should expose a LeRobot-compatible observation/action adapter even if staff run collection and training from a workstation. Its first model can be simple so students reach a feedback loop early.

The [SO-101](https://huggingface.co/docs/lerobot/so101) is the proposed central LeRobot manipulation station for demonstrations, episode collection, ACT, and SmolVLA. Staff should test the supplied kit's actual cameras, leader/follower parts, and command path. The stock USB motor-bus path can bypass the UNO Q's STM32. An arm-centered curriculum that claims STM32 permission for each action needs a tested STM32-to-servo command path; an independent motor-power cutoff is a separate safeguard. A slow VLA trial can use a staff-controlled arm station only after the command and cutoff behavior are documented. The [course operating plan](course-architecture.md) keeps those tests explicit.

The [three body proxies](archetype-proxies.md)—a cart, compliant pusher, and thermal cell—remain candidate capstone contexts. They can share LeRobot-style episode records, policy comparison, and the proposed/permitted/measured trace while testing different physical laws. Staff should qualify only the bodies that can be built and reproduced before student selection.

## Weekly arc

| Weeks | LeRobot and model milestone | Physical-systems evidence |
|---:|:---|:---|
| 1–3 | Define the `Robot` observation/action contract and teleoperation or rule baseline. | Body envelope, sensor calibration, simulator contract, and STM32 permission path. |
| 4–6 | Record episodes, create held-out splits, version a compact policy, and run the first local model-to-action loop. | Proposed/permitted/measured action and next observation, plus a matched rule trial. |
| 7–9 | Train and evaluate ACT on shared or simulated trajectories; investigate SmolVLA on the SO-101 and exact UNO Q. | Simulation gap, action-chunk staleness, memory and timing measurements, and deployment verdict. |
| 10–12 | Collect a failure or human correction, revise the dataset or policy, and repeat the physical trial. | MCU refusals, intervention/rearm behavior, and before/after physical outcomes. |
| 13–14 | Freeze the policy, data, simulator, and operating envelope. | Unfamiliar physical trial and trace-backed defense of what the model caused. |

The teaching sequence does not depend on a separate media-processing track. Camera and video are observations; their value is established by whether the policy uses them to improve a measured physical task. The [lab sequence](../labs/lab-sequence.md) and [instructor syllabus](../../../instructors/vol4/README.md) should be finalized after staff return the local-policy and SmolVLA feasibility results.

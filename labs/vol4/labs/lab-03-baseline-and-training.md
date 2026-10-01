# Lab 3: Learned Control with Vision-Language-Action Models

**Schedule:** Weeks 5–6 · Milestone 2 due end of Week 8 (with Lab 4)
**Required Textbook Reading:** [Chapter 5: Physical Data](../../../books/vol4/05_data/05_data.qmd), [Chapter 6: Policy Training](../../../books/vol4/06_training/06_training.qmd)
**Target Competencies:** `[ ] B1 Physical Dataset Engineering & Multi-Tap Logging`, `[ ] B2 Deterministic Baseline Benchmarking`
**Milestone Alignment:** Contributes to [Milestone 2](../curriculum/syllabus.md#sec-milestones) (End of Week 8)

---

## Overview

### Purpose
You can move the robot with hand-derived equations. But what if you want it to learn behaviors from data instead of code? This lab introduces Vision-Language-Action models (VLAs) — neural policies that take camera images and language instructions as input and output motor commands directly. You will first see a staff-provided VLA perform tasks in simulation, then deploy that same model to the real hardware and observe what happens. The gap between simulated performance and physical reality — the **sim2real gap** — is the central lesson.

### Prerequisites
**Lab 2 completed.** You need a fully wired and calibrated station with working IK on the MCU, a verified camera, and a functioning inter-core RPC bridge. You also need your sim2real comparison table from Lab 2.

### Learning Outcomes
By the end of this lab, you will be able to:
- Explain the VLA architecture: how vision, language, and action are combined in a single model
- Evaluate a pre-trained VLA's performance in simulation across multiple tasks
- Deploy a sim-trained VLA to the Arduino UNO Q Dragonwing (Qualcomm Linux MPU) and connect it to the MCU via RPC
- Measure and document the sim2real performance gap quantitatively
- Compare learned VLA control against your Lab 1–2 IK baseline on the same physical tasks

### What You Will Do
In Week 5, you will work in simulation: loading a staff-provided pre-trained VLA (SmolVLA or ACT trained on simulated demonstrations), running it on simulated tasks, evaluating its success rate, and comparing its behavior against your IK baseline. In Week 6, you will export the sim-trained VLA to ONNX, deploy it to the UNO Q Dragonwing, connect it to the STM32 MCU via the RPC bridge, and run the same tasks on the real robot. You will systematically measure the performance drop and identify what the model learned in simulation that doesn't transfer to reality.

---

### 1. The Physical Question

Can a neural network that learned to manipulate objects in simulation actually control a real robot? Where does it fail, and why? Is the sim2real gap in the perception (different lighting, camera angles), the dynamics (different friction, backlash, gravity), or the policy itself?

---

### 2. Hardware Setup

> **Standard Bench Configuration:** See the [Station Reference Card](station-reference.md).

Additional requirements:
1. **GPU Workstation or Cloud Instance:** For VLA inference in simulation (Week 5).
2. **Staff-Provided VLA Checkpoint:** Pre-trained SmolVLA or ACT model trained on simulated demonstrations of the course robot.
3. **Simulation Environment:** Same MuJoCo/PyBullet setup from Lab 1 with the digital twin.

---

### 3. Step-by-Step Protocol

#### Phase A: VLAs in Simulation (Week 5)

##### Step 1: Understanding VLA Architecture
1. Read the provided SmolVLA architecture overview. Identify the three input streams:
   - **Vision:** Camera image $I_t$ (224 × 224 RGB)
   - **Language:** Task instruction $l$ (e.g., "pick up the red block")
   - **Proprioception:** Current joint state $q_t$
2. Identify the output: action chunk $a_t \in \mathbb{R}^{K \times 6}$ — a sequence of $K$ joint position targets.
3. Sketch the data flow from input to output. Where does the language influence the action?

##### Step 2: Running the Pre-Trained VLA in Simulation
1. Load the staff-provided VLA checkpoint in the simulation environment.
2. Set up the standard task: reach and touch a colored foam block placed in the workspace.
3. Run 20 simulated episodes. Record:
   - Success rate (end-effector within 10 mm of target)
   - Mean positioning error
   - Mean episode length (number of steps)
4. Test with different language prompts: "pick up the red block" vs. "pick up the blue block." Does the VLA respond to language in simulation?

##### Step 3: VLA vs. IK Baseline in Simulation
1. Run your Lab 1 IK solver on the same 20 target positions.
2. Compare success rate, positioning error, and episode length.
3. Where does the VLA outperform IK? Where does IK win? (Hint: IK is exact but inflexible; the VLA generalizes but has learned biases.)

#### Phase B: Deploying the VLA to Real Hardware (Week 6)

##### Step 4: Model Export & Edge Deployment
1. Export the sim-trained VLA to ONNX INT8 format. Target: model size $< 50\text{ MB}$.
2. Transfer the ONNX model to the UNO Q Dragonwing (Qualcomm Linux MPU).
3. Profile on-device inference: peak memory, load time, inference latency.
   - Target: $t_{\text{inf}} < 80\text{ ms}$ per forward pass.

##### Step 5: VLA Running on Real Hardware
1. Connect the VLA inference pipeline on the Dragonwing to the STM32 MCU via the RPC bridge.
2. The data flow is now: real camera → VLA on Dragonwing → action proposal via RPC → MCU safety check → servo command.
3. Run 20 physical reach trials using the same task and target positions as the simulation trials.
4. Record: success rate, positioning error, and episode length on real hardware.

##### Step 6: Measuring the Sim2Real Gap
1. Construct the sim2real comparison table:

   | Metric | Simulation | Real Hardware | Gap |
   |:---|:---:|:---:|:---:|
   | Success rate (20 trials) | | | |
   | Mean positioning error (mm) | | | |
   | Mean episode length (steps) | | | |
   | Language responsiveness | | | |

2. Analyze the failures. For each failed real-world trial, categorize the cause:
   - **Perception gap:** Different lighting, camera angle, background
   - **Dynamics gap:** Backlash, friction, gravity effects absent in sim
   - **Timing gap:** Inference latency causing stale observations
   - **Distribution shift:** Object positions or orientations not seen in training
3. Document your findings — these will motivate Lab 4's fine-tuning.

---

### 4. Disturbance & Failure Tests

- Run the sim-trained VLA with deliberately degraded lighting (50% brightness). Compare against simulation performance.
- Shift the target block 30 mm from its expected position. Does the VLA adapt or reach for the old position?
- Feed the VLA an out-of-distribution prompt (e.g., "stack the blocks"). Document the behavior.

---

### 5. Common Pitfalls & Debugging

* ⚠️ **ONNX Quantization Errors:** INT8 quantization can degrade model accuracy. Compare ONNX outputs against the original PyTorch model on 10 held-out frames before deploying. Max divergence should be $< 1\%$.
* ⚠️ **Camera Mismatch:** The simulation camera and the real C270 webcam have different intrinsics, field of view, and color response. The VLA may perform poorly simply because the pixels look different. Document this as a perception gap.
* ⚠️ **Inference Latency Stacking:** If VLA inference takes > 80 ms, the robot is acting on stale observations. The MCU safety governor should flag this — check the multi-tap telemetry for timing violations.

---

### 6. Sign-Off Criteria

| # | Criterion | Measurable Threshold |
|:---:|:---|:---|
| 1 | **VLA architecture sketch** | Correctly identifies vision, language, and proprioception inputs and action chunk output |
| 2 | **Simulation performance** | 20-trial results: success rate, error, and episode length recorded |
| 3 | **VLA vs. IK comparison** | Side-by-side table in simulation with analysis of strengths and weaknesses |
| 4 | **ONNX export and profiling** | Model $< 50\text{ MB}$; $t_{\text{inf}} < 80\text{ ms}$ on the Dragonwing |
| 5 | **Real hardware trials** | 20-trial results on real hardware recorded |
| 6 | **Sim2Real gap table** | Quantified comparison with categorized failure analysis |

*Staff signs off `[ ] B1` and `[ ] B2` on the team's [Competency Card](../curriculum/student-competencies.md).*

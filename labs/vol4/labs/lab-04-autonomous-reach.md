# Lab 4: Closing the Gap — Real Data & Fine-Tuning

**Schedule:** Weeks 7–8 · Milestone 2 due end of Week 8
**Required Textbook Reading:** [Chapter 7: Closed-Loop Evaluation](../../../books/vol4/07_evaluation/07_evaluation.qmd), [Chapter 8: Sensor Perception](../../../books/vol4/08_perception/08_perception.qmd)
**Target Competencies:** `[ ] B3 Edge Model Profiling & Resource Budgets`, `[ ] C1 Closed-Loop Autonomous Action`
**Milestone Alignment:** Concludes [Milestone 2](../curriculum/syllabus.md#sec-milestones) (End of Week 8)

---

## Overview

### Purpose
Your sim-trained VLA struggled on real hardware — the sim2real gap was real and measurable. Now you learn how the field actually solves this: collect a small dataset of real-world demonstrations and use it to fine-tune the simulation-trained model. This is the modern Physical AI workflow: pre-train cheaply in simulation, then adapt efficiently with real data. By the end of this lab, you will close the loop completely — a fine-tuned VLA running autonomously on the edge, with no host tether, achieving real physical tasks.

### Prerequisites
**Lab 3 completed.** You need a sim-trained VLA deployed on the Dragonwing, a documented sim2real gap table, and working inter-core RPC. You also need the full calibrated station from Lab 2.

### Learning Outcomes
By the end of this lab, you will be able to:
- Teleoperate a robot to collect a structured physical demonstration dataset with all four action taps logged
- Fine-tune a simulation-pre-trained VLA on a small real-world dataset
- Quantify how fine-tuning reduces the sim2real gap compared to the sim-only model
- Integrate vision, inference, RPC, safety, and actuation into a single autonomous control loop running untethered on edge hardware
- Evaluate autonomous manipulation through repeated physical trials

### What You Will Do
In Week 7, you will teleoperate the robot with a gamepad, collect 30 real-world demonstration episodes in LeRobot Dataset v3 format, audit the dataset, and fine-tune the Lab 3 sim-trained VLA on this real data. In Week 8, you will deploy the fine-tuned model, disconnect the host PC, and run the full autonomous Sense–Propose–Permit–Act loop untethered. You will measure the improvement over the sim-only model and decompose the end-to-end loop latency.

---

### 1. The Physical Question

Can a small amount of real-world data rescue a model that was trained entirely in simulation? How many demonstrations do you need, and how much of the sim2real gap does fine-tuning actually close?

---

### 2. Hardware Setup

> **Standard Bench Configuration:** See the [Station Reference Card](station-reference.md).

Additional requirements:
1. **USB Gamepad:** Xbox or Logitech controller for teleoperation input.
2. **Foam Block Targets:** Colored foam blocks (red, blue, yellow) for manipulation tasks.
3. **GPU Workstation:** For fine-tuning the VLA on real data.

---

### 3. Step-by-Step Protocol

#### Phase A: Real-World Data Collection & Fine-Tuning (Week 7)

##### Step 1: Teleoperation Setup
1. Configure the LeRobot teleoperation interface with the USB gamepad.
   ```yaml
   robot:
     type: manipulator
     motors:
       bus: uno_q_bridge
       port: /dev/ttyACM0
       baudrate: 1000000
     cameras:
       overhead:
         type: opencv
         index_or_path: /dev/video0
         fps: 30
         width: 640
         height: 480
   ```
2. Map gamepad axes to SO-101 joint velocities. Verify proportional control with zero lag.
3. Practice smooth teleoperation for 10 minutes before recording.

##### Step 2: Record 30 Demonstration Episodes
1. Define the task protocol: reach toward and touch a colored foam block from 5 varied starting positions.
2. Record 30 demonstration episodes. Each episode logs all 4 action taps ($a_{\text{req}}$, $a_{\text{map}}$, $a_{\text{enf}}$, $a_{\text{meas}}$) plus synchronized camera frames in LeRobot Dataset v3 format.
3. Record 3 additional "adversarial" episodes with deliberately poor demonstrations (jerky motion, overshoot). Label them as failure cases.
4. Audit dataset integrity: verify zero dropped frames, confirm timestamp monotonicity, check action tap completeness.
5. Split into train/validation/test (24/3/3) with leak-free episode boundaries. Produce a dataset card.

##### Step 3: Fine-Tune the Sim-Trained VLA
1. Load the sim-trained VLA checkpoint from Lab 3.
2. Fine-tune on your 24-episode real-world training set using the staff-provided training recipe.
3. Monitor train/val loss convergence. Early-stop on validation loss.
4. Export the fine-tuned model to ONNX INT8. Verify model size $< 50\text{ MB}$.
5. Transfer to the UNO Q Dragonwing.

#### Phase B: Autonomous Closed-Loop Operation (Week 8)

##### Step 4: Sim-Only vs. Fine-Tuned Comparison
1. Run the **sim-only VLA** (Lab 3 checkpoint) on 10 physical reach trials. Record success rate and positioning error.
2. Run the **fine-tuned VLA** on the same 10 target positions. Record success rate and positioning error.
3. Construct the improvement table:

   | Model | Success Rate | Mean Error (mm) | Sim2Real Gap Reduction |
   |:---|:---:|:---:|:---:|
   | IK baseline (Lab 2) | | | — |
   | Sim-only VLA (Lab 3) | | | — |
   | Fine-tuned VLA | | | |

##### Step 5: Untethered Autonomous Loop
1. Integrate the full Sense–Propose–Permit–Act pipeline:
   camera capture → VLA inference → RPC proposal → MCU safety check → servo command → telemetry return.
2. **Disconnect the host PC.** Run the loop headlessly on the UNO Q.
3. Execute 10 physical reach trials from varied starting positions. Record success/failure and final positioning error for each.

##### Step 6: Latency Decomposition
1. Break down end-to-end loop latency:
   $$T_{\text{total}} = t_{\text{cam}} + t_{\text{prep}} + t_{\text{inf}} + t_{\text{rpc}} + t_{\text{act}}$$
2. Target: $T_{\text{total}} \le 80\text{ ms}$ (≥ 12.5 Hz loop rate).
3. Identify the bottleneck component. Can you reduce it?

---

### 4. Disturbance & Failure Tests

- Reduce lighting to 50% mid-trial. Does the fine-tuned model handle it better than the sim-only model?
- Introduce a 200 ms artificial delay in the camera pipeline. Does the MCU reject the stale proposal?
- Disconnect the webcam mid-reach. Verify the system enters safe hold.

---

### 5. Multi-Tap Telemetry Trace

Capture the full 4-tap telemetry during an autonomous reach:
* $a_{\text{req}}$: VLA's proposed action (Dragonwing output)
* $a_{\text{map}}$: Mapped joint command sent over RPC
* $a_{\text{enf}}$: MCU's permitted command (after safety checks)
* $a_{\text{meas}}$: Actual servo encoder readback

Verify convergence: $a_{\text{meas}} \to a_{\text{enf}}$ within $\pm 0.5^\circ$.

---

### 6. Common Pitfalls & Debugging

* ⚠️ **Overfitting on Small Datasets:** With only 24 training episodes, the fine-tuned model can overfit. Monitor validation loss carefully and use early stopping.
* ⚠️ **Distribution Mismatch:** If your teleop demonstrations are much smoother or jerkier than the sim training data, the fine-tuned model may learn a mixed style. Be consistent in your demonstration quality.
* ⚠️ **Telemetry Dropped Frames:** At 30 FPS, even one dropped camera frame creates a gap in the action sequence. Audit timestamp monotonicity before training.

---

### 7. Sign-Off Criteria (Milestone 2 Exit Check)

| # | Criterion | Measurable Threshold |
|:---:|:---|:---|
| 1 | **Dataset collected** | 30 complete episodes; all 4 action taps populated per timestep |
| 2 | **Dataset card** | Collection protocol, sensor specs, splits, and known limitations documented |
| 3 | **Fine-tuning converges** | Val loss decreasing; no NaN gradients |
| 4 | **Sim-only vs. fine-tuned comparison** | Quantified improvement table with at least 10 matched physical trials each |
| 5 | **Untethered loop runs** | ≥ 10 consecutive trials without host intervention |
| 6 | **Success rate** | Fine-tuned VLA: ≥ 7/10 reaches within 15 mm of target |
| 7 | **Loop latency** | $T_{\text{total}} \le 80\text{ ms}$ (p50); $< 120\text{ ms}$ (p99) |

*Staff signs off `[ ] B3` and `[ ] C1` on the team's [Competency Card](../curriculum/student-competencies.md).*

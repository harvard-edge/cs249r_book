# Volume IV: Eight Lab Blueprints for the Physical AI Studio

**Status:** Canonical Staff Architecture and Studio Blueprint
**Target Platform:** Arduino UNO Q ("Unikue" QRB2210 Linux + STM32U585) · Seeed Studio SO-101 6-DoF Arm · Hugging Face LeRobot · SmolVLA & ACT Models
**Reference Curriculum:** [Master Landing Page](README.md) · [Course Syllabus](syllabus.md) · [12-Competency Matrix](student-competencies.md) · [Postdoc Pre-Flight Guide](feasibility-plan.md) · [Capstone Studio](lab-capstone-studio.md)

---

## 1. Pedagogical Architecture

The Volume IV studio is structured into **4 foundational parts (8 hands-on labs, Weeks 1–11)** followed by a **3-Week Dedicated Capstone Studio (Weeks 12–14)**. Rather than treating physical AI as abstract software deployment, every lab exercises the closed-loop Sense·Propose·Permit·Act loop across the physical boundary.

```
                    ┌─────────────────────────┐
                    │   1. SENSE (Webcam)     │
                    └───────────┬─────────────┘
                                │ RGB Frames
                                ▼
                    ┌─────────────────────────┐
                    │ 2. PROPOSE (Qualcomm)   │ ◄── SmolVLA / ACT Policy
                    └───────────┬─────────────┘
                                │ Requested Actions (a_req)
                                ▼
                    ┌─────────────────────────┐
                    │  3. PERMIT (STM32 MCU)  │ ◄── Safety Envelope & Veto
                    └───────────┬─────────────┘
                                │ Enforced Actions (a_enf)
                                ▼
                    ┌─────────────────────────┐
                    │    4. ACT (SO-101 Arm)  │ ◄── STS3215 Serial Servos
                    └───────────┬─────────────┘
                                │ Measured Telemetry (a_meas)
                                └────────────────────────► Closed-Loop Revision
```

---

## 2. The Eight Canonical Lab Blueprints

### Part I: The Machine Anatomy (Weeks 1–3) · Chapters 1–4

#### Lab 1: Causal Boundary & Bus Bring-Up
* **Core Systems Question:** Where does software end, where does physical actuation begin, and who holds veto authority?
* **Hands-On Experiment:** Map the physical wiring harness between the Arduino UNO Q, the independent 7.4V/5A motor power rail, the latching mushroom E-stop, and the STS3215 single-wire half-duplex UART bus. Measure cold boot time, reset transients, and verify zero-motion safe hold when motor power is severed.
* **Exit Proof & Artifact:** Submit `wiring_diagram_golden.pdf` and an oscilloscope/logic-analyzer trace demonstrating motor power cutoff within $< 5\text{ ms}$ while Linux logging remains active.
* **Competencies:** A1, D1.

#### Lab 2: Sensing & Inter-Core Bridge
* **Core Systems Question:** What can the camera and joint sensors actually measure, and what is the latency cost of communicating across the inter-core boundary?
* **Hands-On Experiment:** Calibrate the overhead UVC USB webcam (`/dev/video0`) using an AprilTag target; calibrate joint angle zero-offsets across all 6 SO-101 joints ($\pm 1.5^\circ$ repeatability). Benchmark inter-core RPC throughput and round-trip latency between Qualcomm Linux and STM32 MCU over `/dev/ttyRPMSG`.
* **Exit Proof & Artifact:** Submit calibrated camera projection matrices, joint angle calibration offsets, and a latency histogram showing RPC round-trip times $\le 5\text{ ms}$ at $50\text{ Hz}$.
* **Competencies:** A2, A3.

---

### Part II: Teaching the Machine (Weeks 4–6) · Chapters 5–7

#### Lab 3: Teleoperation & Datasets
* **Core Systems Question:** How do human physical demonstrations become causal, auditable datasets for imitation learning?
* **Hands-On Experiment:** Use LeRobot teleoperation (`lerobot-teleoperate`) to record 10 expert pick-and-place demonstration episodes on the SO-101 arm. Synchronize $30\text{ Hz}$ camera frames ($224 \times 224$ RGB) with joint telemetry into the standardized LeRobot v2 dataset schema (`LeRobotDataset`).
* **Exit Proof & Artifact:** Submit a validated 10-episode dataset card (`dataset_golden_10ep/`) containing synchronized observation and action streams with zero dropped frames.
* **Competencies:** B1, B2.

#### Lab 4: Baseline & Policy Export
* **Core Systems Question:** How does an imitation learning policy compare against a classical heuristic, and how is it packaged for edge execution?
* **Hands-On Experiment:** Implement a deterministic heuristic reaching script (`heuristic_reach.py`) using fixed joint interpolation. Train a baseline Action Chunking with Transformers (ACT) policy on the Lab 3 dataset using GPU compute. Export and quantize the model to ONNX INT8 for edge deployment.
* **Exit Proof & Artifact:** Submit comparative offline validation curves (MSE vs ground truth) and the quantized model artifact (`policy_act_int8.onnx`, $< 50\text{ MB}$).
* **Competencies:** B2, B3.

---

### Part III: Running the Machine (Weeks 7–9) · Chapters 8–13

#### Lab 5: Autonomous Closed-Loop Reach
* **Core Systems Question:** Can a learned policy close the feedback loop untethered on edge silicon?
* **Hands-On Experiment:** Deploy the quantized ACT/SmolVLA policy natively onto Qualcomm Linux. Run the autonomous loop (`pai_edge_runtime.py`) without host PC assistance. The arm must observe the block via webcam, infer joint proposals, send them to the STM32, and execute an autonomous reach to arbitrary target positions.
* **Exit Proof & Artifact:** A witnessed untethered reach on physical hardware and a 4-tap telemetry log showing total loop latency $T_{\text{total}} \le 80\text{ ms}$ ($> 12.5\text{ Hz}$).
* **Competencies:** C1, C2.

#### Lab 6: Action Horizons & Disturbances
* **Core Systems Question:** How far into the future can an embodied model predict open-loop actions before physical drift demands re-observation?
* **Hands-On Experiment:** Benchmark policy performance across action chunk horizons $K \in \{1, 8, 16, 32\}$. Inject a physical disturbance mid-trajectory by shifting the target block. Measure the recovery latency and tracking error under temporal ensemble averaging. Test language-prompt variation on SmolVLA.
* **Exit Proof & Artifact:** Submit comparative tracking error curves across chunk horizons and multi-tap telemetry capturing autonomous disturbance recovery.
* **Competencies:** C2, C3.

---

### Part IV: Governing the Machine (Weeks 10–11) · Chapters 14–17

#### Lab 7: Microcontroller Safety Governor
* **Core Systems Question:** How do we deterministically ensure that a probabilistic neural network cannot command destructive physical motion?
* **Hands-On Experiment:** Program the STM32U585 firmware to act as an active safety governor. Implement joint velocity saturation ($\omega_{\text{max}} = 45^\circ/\text{s}$) and a Cartesian forward-kinematics table collision geofence ($z_{\text{tool}} \ge 15\text{ mm}$). Inject illegal commands via Python and measure real-time MCU veto behavior.
* **Exit Proof & Artifact:** Submit STM32 firmware source code and a live demonstration where injected out-of-bounds proposals are clamped or vetoed in $< 5\text{ ms}$ without system crashes.
* **Competencies:** D1, D2.

#### Lab 8: Fault Injection & Cutoffs
* **Core Systems Question:** What happens when the host computer freezes, the camera is disconnected, or communications fail mid-motion?
* **Hands-On Experiment:** Configure a $150\text{ ms}$ hardware SysTick watchdog on the STM32. Inject simulated Linux freezes (`kill -STOP`), USB cable disconnects, and packet corruptions. Verify that the arm decelerates to a safe stop within $150\text{ ms}$ and executes zero stale buffered commands upon resumption.
* **Exit Proof & Artifact:** Submit telemetry traces demonstrating watchdog cutoff and an audit proving zero backlogged action execution after process resumption.
* **Competencies:** D2, D3.

---

### Dedicated Capstone Project Studio (Weeks 12–14)

* **Mission:** Design, train, deploy, and rigorously defend an end-to-end embodied manipulation system capable of robust disturbance recovery under real-world perturbations.
* **Milestone 5 Defense:** 20 consecutive physical trials under staff-injected perturbations (target relocations, surface obstacles, visual occlusions) requiring $\ge 80\%$ success and $0\%$ safety violations.
* **Deliverable:** Claim-Argument-Evidence (CAE) Physical Release Dossier and oral defense.

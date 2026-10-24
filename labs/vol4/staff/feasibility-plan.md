# Postdoc Pre-Flight Implementation & Qualification Guide

**Audience:** Postdoctoral Researcher (Andrea, ETH Zurich) & Lab Staff
**Status:** Canonical Engineering Implementation Checklist & Qualification Plan
**Target Platform:** Arduino UNO Q ("Unikue" QRB2210 Linux + STM32U585 MCU) · Seeed Studio SO-101 6-DoF Arm · Hugging Face LeRobot · SmolVLA & ACT Models
**Reference Curriculum:** [Course Homepage](https://mlsysbook.ai/courses/physical-ai/eth-spring-2027/) · [Course Syllabus](https://mlsysbook.ai/courses/physical-ai/eth-spring-2027/syllabus.html) · [Staff Master Plan (Sep–Dec 2026)](staff-implementation-plan.md) · [12-Competency Matrix](../curriculum/student-competencies.md) · [Capstone Studio](../labs/lab-capstone-studio.md)

---

## 1. Executive Mandate: Prove the Bench Before Teaching It

Andrea, your primary mission before the semester begins is to **build, verify, and qualify one "Golden Reference Station"** from raw hardware to untethered physical AI execution. You must act as **"Student Zero"** across all 6 labs and the capstone, verifying that every exercise functions deterministically, producing the gold-standard artifacts, and isolating hardware failure modes before students arrive.

Students should finish this course with an autonomous embodied system that **senses physical reality via vision, uses a learned model to propose multi-joint action chunks, delegates safety authority to an on-board microcontroller to permit or clamp motion, measures true kinematic outcomes, and revises its next decision from closed-loop feedback**.

Do not attempt to write sixteen different handouts or build custom motor shields. The hardware kit is fixed:
1. **The Board:** Arduino UNO Q (Qualcomm QRB2210 Linux MPU + STM32U585 real-time MCU).
2. **The Body:** Seeed Studio SO-101 6-DoF arm (STS3215 smart serial bus servos) + $20 standard UVC USB webcam.
3. **The Software/Model Spine:** Hugging Face LeRobot (Python API) + SmolVLA / ACT action-chunking policies.

---

## 2. Phase 0: Physical Bench Assembly & Hardware Bring-Up

Complete these physical build steps and verify electrical safety before powering on digital electronics:

- [ ] **SO-101 Arm Mechanical Assembly:**
  - Assemble the 6-DoF follower arm using the Seeed Studio SO-101 Pro kit.
  - Verify mechanical backlash and free range of motion on all 6 joints ($J_1$ base yaw to $J_6$ gripper).
  - Securely clamp the arm baseplate to the lab workbench using heavy-duty C-clamps. The arm must not tip or rock under maximum payload and full acceleration.
- [ ] **Camera & Workspace Rigging:**
  - Mount the standard UVC USB webcam (Logitech C270 or equivalent) on a rigid, vibration-isolated gooseneck arm overlooking the manipulation stage at an oblique angle ($45^\circ$, $40\text{ cm}$ distance).
  - Define a marked physical workspace boundary ($300\text{ mm} \times 200\text{ mm}$) on the bench surface using high-contrast tape.
  - Set up diffused, flicker-free LED task lighting to prevent exposure fluctuations.
- [ ] **Electrical Power Harness & Dedicated Rail:**
  - Connect a regulated external DC bench power supply ($7.4\text{V}$, $5\text{A}$ rating) dedicated exclusively to the STS3215 servo rail.
  - Verify that the motor power supply delivers stable $7.4\text{V}$ without voltage sag under multi-joint motion.
  - **CRITICAL:** Tie the DC motor power ground and Arduino UNO Q ground together into a solid star-ground. **NEVER** draw motor current through the UNO Q headers.
- [ ] **Dual-Core Board & Bus Communication:**
  - Power the Arduino UNO Q via its USB-C port using an official 45W USB-PD adapter through the powered USB-C hub.
  - Plug the UVC webcam into a USB-A port on the hub; verify Linux recognizes the device at `/dev/video0`.
  - Connect the single-wire half-duplex UART communication line from the STS3215 servo bus to the STM32U585 USART pins via the level-shifter circuit.

![Bench wiring: webcam and 45 W logic supply to the UNO Q, STM32 to the servo bus over UART, a separate switched 7.4 V motor supply, and a common star ground.](../assets/images/vol4-bench-wiring.svg)

---

## 3. Phase 1: Hardware Bring-Up and Validation Steps

Before publishing student labs or ordering additional stations, you must successfully complete and log these four sequential validation steps:

```
[ Step 1: Native LeRobot Teleop ]
       │ (Arm moves via standard HF LeRobot scripts)
       ▼
[ Step 2: 1-Joint MCU Interceptor ]
       │ (STM32 intercepts UART, rejects invalid commands)
       ▼
[ Step 3: 6-DoF Governed Arm ]
       │ (UnoQMotorsBus adapter runs full arm under MCU bounds)
       ▼
[ Step 4: Standalone Untethered Reach ]
         (Qualcomm Linux runs INT8 policy at < 100 ms latency)
```

### Bring-Up Step 1: Native LeRobot USB Teleoperation
- [ ] Connect the SO-101 arm to a development workstation using the standard USB BusLinker adapter.
- [ ] Install Hugging Face LeRobot (`pip install lerobot`).
- [ ] Run the official LeRobot joint calibration utility:
  ```bash
  python -m lerobot.scripts.control_robot calibrate --robot.type=so101
  ```
- [ ] Record a 60-second teleoperation sequence (leader arm or keyboard teleop) and execute episode replay (`lerobot-replay`).
- **Success Criteria:** Arm smoothly mirrors teleoperated commands with zero servo jitter or dropped packets.

### Bring-Up Step 2: Single-Joint STM32 Safety Interceptor
- [ ] Disconnect joint 1 (base yaw) from the host USB BusLinker. Connect its serial data line to the STM32U585 USART header.
- [ ] Flash baseline interceptor firmware to the STM32 via Arduino App Lab.
- [ ] Write a 40-line Python test script on Qualcomm Linux that sends velocity requests across the inter-core Bridge (`/dev/ttyRPMSG` or Arduino RPC).
- [ ] Transmit three test requests:
  1. A valid $10^\circ$ rotation at $20^\circ/\text{s}$ $\to$ **STM32 must permit and forward to motor**.
  2. An unsafe $90^\circ$ step requesting $300^\circ/\text{s}$ velocity $\to$ **STM32 must clamp velocity to $45^\circ/\text{s}$ max**.
  3. An out-of-bounds target position ($220^\circ$) $\to$ **STM32 must refuse motion and enter safe hold**.
- **Success Criteria:** The STM32 deterministically filters commands; no software bypass can cause unpermitted physical motion.

### Bring-Up Step 3: Full 6-DoF Governed Arm Integration
- [ ] Connect all 6 SO-101 joints to the governed STM32 bus line.
- [ ] Implement the minimal LeRobot custom motor bus adapter (`UnoQMotorsBus`) in Python:
  - Methods: `connect()`, `disconnect()`, `write("Goal_Position", targets)`, `read("Present_Position")`.
  - The adapter packages joint targets into a lightweight binary struct (`a_req`) and sends it over inter-core RPC.
  - The STM32 evaluates joint limit tables and a simple Cartesian table-collision geofence ($z_{\text{tool}} \ge 15\text{ mm}$), writes permitted targets (`a_enf`) to the STS3215 bus, reads measured positions (`a_meas`), and returns them over RPC.
- [ ] Execute standard LeRobot control commands through the governed adapter:
  ```bash
  python -m lerobot.scripts.control_robot teleoperate --robot.type=so101_unoq
  ```
- **Success Criteria:** All 6 joints operate smoothly through LeRobot while the STM32 intercepts and vetoes any command that would collide with the table surface.

### Bring-Up Step 4: Standalone Untethered Reach on Qualcomm Linux
- [ ] Export a trained ACT or SmolVLA policy checkpoint to ONNX INT8 format.
- [ ] Transfer the model file (`policy_int8.onnx`) and Python inference runtime (`pai_edge_runtime.py`) to the Qualcomm Linux storage.
- [ ] Unplug the USB cable connecting the UNO Q to the host PC. The board must run completely untethered on its USB-PD power supply.
- [ ] Place a soft foam block inside the marked workspace.
- [ ] Launch the autonomous edge runtime on Qualcomm Linux via SSH over Wi-Fi:
  ```bash
  python pai_edge_runtime.py --model policy_int8.onnx --rate 30
  ```
- [ ] Measure end-to-end loop timing across 50 consecutive frames:
  - $t_{\text{capture}} \le 25\text{ ms}$
  - $t_{\text{infer}} \le 45\text{ ms}$ (INT8 quantized policy on QRB2210 CPU/NPU)
  - $t_{\text{bridge}} \le 5\text{ ms}$
  - $t_{\text{mcu}} \le 2\text{ ms}$
  - **Total Loop Latency:** $T_{\text{total}} \le 80\text{ ms}$ ($> 12.5\text{ Hz}$ closed-loop bandwidth).
- **Success Criteria:** The physical arm autonomously reaches and touches the target block without tethered host assistance; total latency stays strictly below $100\text{ ms}$.

---

## 4. Phase 2: "Student Zero" Lab Qualification Runs (Labs 1–6)

For each lab in the 14-week curriculum, Andrea must execute the student protocol end-to-end, identify potential pitfalls, and prepare the "Gold Standard" starter assets. The labs follow the [lab template](../LAB_TEMPLATE.md): the instructor sets each lab's purpose, objectives, weekly parts, checkpoints, and deliverables; staff write the instructions and set the thresholds from the Student Zero run.

![Fourteen-week student timeline: six labs, the capstone, and milestones M1 to M5.](../assets/images/vol4-course-timeline.svg)

**Course assets the labs depend on.** These must exist before the lab that uses them, and several are needed from week 1:

| Asset | First needed | Notes |
|:---|:---|:---|
| **Baseline governor firmware** (joint limits, speed clamp, watchdog) | Getting Started, week 1 | The arm runs under it in weeks 1–9; students replace it with their own in Lab 6. Builds on the Step 3 governed arm. |
| **Pre-wired, pre-calibrated stations** | Getting Started, week 1 | Students trace the wiring in week 1 but do not build it. |
| **Body-limits measurement procedure** | Part 1A, week 2 | Stopping distance, servo temperature under held load, supply sag on stall, joint-reading age. The STS3215 servos report temperature, voltage, and load. |
| **Kinematics primer handout** | Part 1B, week 3 | The book does not teach DH parameters or inverse kinematics. |
| **Digital twin** (URDF/MJCF matching the delivered SO-101) | Part 1B, week 3 | Used in every lab after Lab 1, and in Lab 3's simulation in Lab 3's simulation. |
| **Sim-trained VLA checkpoint** (SmolVLA or ACT, trained on simulated demonstrations of the course arm) | Part 3A, week 5 | The simulated camera should match the real one. Fallback: a suitable public SO-101 checkpoint, if one exists. |
| **Fallback artifacts** for each lab | Each lab | A reference IK solver, reference IK firmware, reference checkpoints, and a reference governor, handed to a team that did not finish the previous lab. |

| Lab & Parts | Pre-Flight Tasks for Andrea ("Student Zero") | Required Staff Deliverables for Students |
|:---|:---|:---|
| **[Lab 1: The Digital Twin](../labs/lab-01-digital-twin.md)** (Getting Started, then 1A measure the body, 1B build the twin) | • Run teleoperation, record and replay, and a motor-power cut under the baseline governor.<br>• Measure the body's limits on the reference station.<br>• Work the kinematics primer and the IK test in the twin. | • Baseline governor firmware<br>• Body-limits procedure<br>• Kinematics primer, digital twin, simulation starter notebook |
| **[Lab 2: Simulation to Real Hardware](../labs/lab-02-sim-to-real.md)** (2A IK on the real arm) | • Run student IK on the STM32 against the Lab 1 targets.<br>• Benchmark inter-core RPC latency and timestamp alignment. | • IK-on-MCU path (student port or staff module)<br>• `test_bridge_latency.py`<br>• Reference IK solver (fallback) |
| **[Lab 3: Learned Control with VLAs](../labs/lab-03-learned-control.md)** (3A sim, 3B hardware) | • Run the sim-trained checkpoint in simulation and compare with IK.<br>• Calibrate the camera; export and profile the model on the UNO Q.<br>• Run the simulation vs. hardware comparison. | • Sim-trained VLA checkpoint<br>• `calibrate_camera.py`, `camera_v4l2_config.sh`<br>• `export_onnx_quantized.py` |
| **[Lab 4: Real-World Data & Fine-Tuning](../labs/lab-04-real-data-and-fine-tuning.md)** (4A data, 4B untethered) | • Record reference teleoperation episodes with all four action taps; validate the dataset schema.<br>• Fine-tune the sim-trained checkpoint offline; build the deterministic reach baseline.<br>• Run the untethered loop and record a four-tap trace. | • `teleop_record.py`, `inspect_dataset.py`, reference dataset<br>• Fine-tuning recipe, `heuristic_reach.py`<br>• `pai_edge_runtime.py`, `telemetry_logger.py` |
| **[Lab 5: Stress-Testing](../labs/lab-05-stress-testing.md)** (5A) | • Benchmark chunk horizons.<br>• Test mid-reach target displacement and removal.<br>• Run one paired-prompt language check. | • `benchmark_chunking.py`<br>• `disturbance_eval.py` |
| **[Lab 6: Safety Governor](../labs/lab-06-safety-governor.md)** (6A build, 6B break) | • Program the governor; run the MCU-vs-Linux placement test.<br>• Program the watchdog; inject freezes, dropouts, and corrupted packets.<br>• Prove zero stale-command execution on resume. | • Governor firmware project<br>• `fault_injection_suite.py`, `recovery_verification.py`<br>• Reference governor (fallback for the capstone) |

---

## 5. Phase 3: Golden System Image & Bench Duplication

Once the single station passes all validation steps and lab qualifications, prepare the infrastructure for the full student cohort:

- [ ] **Qualcomm Linux Golden SD Card Image:**
  - Build a clean Ubuntu/Debian rootfs image for the UNO Q QRB2210.
  - Pre-install dependencies: Python 3.10+, PyTorch ARM64, ONNX Runtime, Hugging Face LeRobot (pinned release), OpenCV, NumPy, V4L2-utils, Git.
  - Pre-clone student course repository and baseline model checkpoints into `/home/arduino/course/`.
  - Configure automatic Wi-Fi joining for the university lab network and fixed static hostname (`unoq-station-XX.local`).
  - Compress and store the golden `.img` file on the lab server for quick flashing.
- [ ] **STM32 Pre-Flashed Baseline Binary:**
  - Create the standard Arduino App Lab sketch containing the inter-core RPC listener, STS3215 bus driver, and hardware watchdog timer.
  - Verify that freshly unboxed UNO Q boards can be flashed with this baseline in $< 2\text{ minutes}$.
- [ ] **Spare Parts Buffer & Tooling Kit:**
  - 4x spare STS3215 smart servos (pre-addressed IDs 1 through 6).
  - 2x backup Logitech C270 USB webcams.
  - 3x spare 7.4V/5A DC motor power supplies.
  - Set of 3D-printed spare brackets, gripper jaws, and base clamps.
  - 2x digital multimeters and logic analyzers (for UART bus debugging).

---

## 6. Phase 4: Risk Mitigation and Fallback Matrix

If specific technical blockers arise during bench bring-up, apply these pre-authorized fallbacks:

| Failure / Risk Event | Primary Diagnostic | Pre-Approved Fallback Action |
|:---|:---|:---|
| **SmolVLA inference latency is too slow on QRB2210 CPU ($> 200\text{ ms}$)** | Profile ONNX execution breakdown (vision encoder vs LLM backbone). | **Fallback to ACT (Action Chunking with Transformers):** ACT uses a lightweight ResNet/MobileNet visual backbone + small transformer decoder, executing in $< 35\text{ ms}$ on ARM64 INT8. Reserve SmolVLA for workstation analysis in Lab 6. |
| **Camera frame drops or V4L2 buffer overflow** | Check whether OpenCV capture is running synchronously in the inference thread. | **Separate Capture Thread:** Run frame acquisition in a dedicated background daemon with double-buffering, locking camera exposure via `v4l2-ctl -c exposure_auto=1`. |
| **Inter-core Bridge latency jitter ($> 15\text{ ms}$)** | High serialization overhead from JSON-RPC. | **Raw Binary RPC:** Switch inter-core messaging to a fixed 24-byte C struct (`float a_req[6]`) transmitted over raw UART shared memory (`/dev/ttyRPMSG`). |
| **STS3215 servo thermal shutdown during teleop** | Measure servo casing temperature after 20 minutes continuous teleoperation. | **Duty Cycle Clamping:** Add software current limit in STM32 firmware and adhere passive aluminum heatsinks to shoulder ($J_2$) and elbow ($J_3$) servos. |

---

## 7. Weekly Reporting Protocol for Andrea

At the conclusion of each lab qualification pass, send a structured report to the teaching team containing:

1. **Status:** `[WORKS]` / `[WORKS WITH CHANGES]` / `[BLOCKED]`
2. **Artifact Evidence:** Link to the generated telemetry CSV, LeRobot dataset slice, or video recording.
3. **Starter Pack Adjustments:** List of starter scripts, configuration defaults, or fixtures that must be supplied.
4. **Next Milestone:** Target completion date for the next phase.

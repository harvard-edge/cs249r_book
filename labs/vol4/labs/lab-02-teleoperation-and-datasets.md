# Lab 2: Hardware Bring-Up & Real-World Control

**Schedule:** Weeks 3–4 · Milestone 1 due end of Week 4
**Required Textbook Reading:** [Chapter 3: The Cognitive Brain](../../../books/vol4/03_brain/03_brain.qmd), [Chapter 4: The Nervous System](../../../books/vol4/04_nervous/04_nervous.qmd)
**Target Competencies:** `[ ] A2 Multi-Modal Sensing & Calibration`, `[ ] A3 Feedback Timing & Latency`, `[ ] D1 Hardware Authority Routing`
**Milestone Alignment:** Concludes [Milestone 1](../curriculum/syllabus.md#sec-milestones) (End of Week 4)

---

## Overview

### Purpose
Your IK equations worked perfectly in simulation. Now you move to the real robot — and discover what simulation got wrong. Servos have backlash. Gravity pulls on real mass. Communication has latency. Power rails can brown out. This lab is where you transition from simulated perfection to physical reality, learning to characterize, calibrate, and control an actual robotic system. You will also encounter the dual-brain architecture for the first time: the Linux MPU that thinks and the STM32 MCU that governs.

### Prerequisites
**Lab 1 completed.** You need working IK equations (validated in simulation) and a completed DH parameter table for the course robot.

### Learning Outcomes
By the end of this lab, you will be able to:
- Wire a robotic workstation with isolated dual-power rails and verify safe-state behavior
- Implement your inverse kinematics on the STM32 MCU and command the real robot via the Arduino UNO Q CLI
- Calibrate an extrinsic camera-to-robot transformation and measure its accuracy
- Measure inter-core RPC latency between the Linux MPU and the STM32 MCU
- Quantify the discrepancies between your simulation model and the real hardware (sim2real gap preview)
- Verify that the system fails safely under power loss and communication failure

### What You Will Do
In Week 3, you will wire the bench station, enumerate and calibrate all six servos, implement your Lab 1 IK equations in the STM32 firmware, and command the robot to reach target positions using the Arduino UNO Q CLI — comparing real-world accuracy against your simulation predictions. In Week 4, you will configure the overhead webcam, calibrate the camera-to-base transformation, bring up the inter-core RPC bridge, measure communication latency, and run failure tests (power cutoff, sensor dropout, stale data injection).

---

### 1. The Physical Question

Your simulation said the robot could reach a target with sub-millimeter accuracy. Does the real hardware agree? Where does the sim2real gap come from — backlash, gravity, encoder resolution, latency — and how bad is it?

---

### 2. Hardware Setup

> **Standard Bench Configuration:** See the [Station Reference Card](station-reference.md) for the full component list, wiring diagram, three engineering rules, and the four-tap telemetry schema used in every lab.

Additional materials for this lab:
1. **Calibration Target:** Printed ChArUco board fixture placed flat on the tabletop.
2. **Your IK Code:** The Python IK solver from Lab 1, ready to port to C for the STM32.
3. **Initial State:** Motor power supply switch **OFF (Disarmed)**. Arm in resting, folded configuration.

---

### 3. Step-by-Step Protocol

#### Phase A: Wiring, Enumeration & IK on Hardware (Week 3)

##### Step 1: Physical Authority & Power Audit
1. Inspect the station wiring. Trace every cable from the power strip to the arm.
2. Verify that the **only** electrical connection to the STS3215 servo bus originates from the STM32 microcontroller header.
3. Confirm that no direct USB-to-UART bridge connects the host workstation to the servo bus.
4. Draw the physical authority diagram showing power rails, data buses, and cutoff paths.

##### Step 2: Servo Enumeration & Mechanical Envelope
1. Flash the staff-provided baseline firmware to the STM32 MCU.
2. Energize motor power. Run the bus scan:
   ```bash
   uno-q-cli bus scan --baud 1000000
   ```
3. Verify all 6 STS3215 servos respond with correct IDs (`1` to `6`).
4. Zero-calibrate each joint using the calibration jig. Record encoder counts.
5. Articulate each joint through its full travel range. Measure $\theta_{i,\min}$ and $\theta_{i,\max}$.
6. **Compare against simulation:** How do the real travel limits differ from your URDF model?

##### Step 3: IK on the MCU — Real-World Reaching
1. Port your Lab 1 IK solver to C and integrate it into the STM32 firmware (or use the staff-provided IK module with your DH parameters).
2. Define 10 target positions in the workspace — the same ones you tested in simulation.
3. Command the robot to each target via the Arduino UNO Q CLI:
   ```bash
   uno-q-cli ik reach --target-xyz 150 0 100
   ```
4. Measure the actual end-effector position (by hand or with the calibration target). Record the positioning error.
5. **Sim2Real Comparison Table:** For each target, record: simulation error vs. real-world error. Identify the dominant sources of discrepancy.

#### Phase B: Sensing, Calibration & Safety (Week 4)

##### Step 4: Camera Configuration & Calibration
1. Log into the Qualcomm Linux terminal. Verify the webcam:
   ```bash
   v4l2-ctl --list-devices
   v4l2-ctl -d /dev/video0 --list-formats-ext
   ```
2. Lock auto-exposure and white balance. Capture test frames at 30 FPS.
3. Place the ChArUco calibration target. Run the staff calibration utility:
   ```bash
   python3 -m pai_tools.calibrate_camera --cam-id 0 --board-type charuco
   ```
4. Compute the transformation matrix $T_{\text{cam}}^{\text{base}} \in SE(3)$. Verify reprojection error $< 2.0\text{ pixels}$.

##### Step 5: Inter-Core RPC Bridge
1. Launch the `arduino-bridge` daemon connecting Qualcomm Linux to the STM32 MCU.
2. Request joint telemetry across the bridge:
   ```python
   from pai_bridge import BridgeClient

   client = BridgeClient()
   telemetry = client.get_joint_state()  # [q1, q2, q3, q4, q5, q6, timestamp_us]
   print("MCU Joint State:", telemetry)
   ```
3. Measure round-trip RPC latency over 1,000 queries. Target: $< 1.0\text{ ms}$.

##### Step 6: Failure & Safe-State Tests
1. **Power-Off Drop Test:** Command the arm to an elevated pose. Cut motor power. Verify safe passive drop with no violent snapping.
2. **Re-Power Surge Audit:** Restore motor power. Verify zero uncommanded motion until explicit re-arm.
3. **Sensor Occlusion:** Block the webcam for 3 seconds. Verify `CAMERA_DROPOUT` error is flagged.
4. **Stale Telemetry:** Inject a 200 ms delay in MCU telemetry. Verify Linux detects stale data ($T_{\text{age}} > 50\text{ ms}$).

---

### 4. Multi-Tap Telemetry Trace

Establish the four-tap telemetry schema that logs every action across the semester:

**Trace A (Mechanical):** Log an IK-commanded reach in CSV format. Verify that $a_{\text{meas}}$ converges to $a_{\text{enf}}$ within $\pm 0.5^\circ$.

**Trace B (Temporal):** Capture 10 reaches. For each, log camera timestamp, MCU timestamp, and inter-modal skew ($\Delta t < 16\text{ ms}$ at 30 FPS).

---

### 5. Common Pitfalls & Debugging

* ⚠️ **TTL Half-Duplex Contention:** The STS3215 bus uses a single bi-directional data line. Ensure the direction-control pin timing is exact to avoid bus collisions.
* ⚠️ **Voltage Sag under Multi-Servo Stall:** Ensure logic power is completely decoupled from motor power to prevent brown-out resets.
* ⚠️ **Sim2Real Backlash:** The STS3215 servos have measurable backlash (~1–2°) that doesn't exist in simulation. This is normal — document it, don't fight it.
* ⚠️ **Clock Drift Between Cores:** Qualcomm Linux uses `CLOCK_MONOTONIC`; the STM32 uses a hardware SysTick timer. Perform a timestamp sync handshake at bridge initialization.

---

### 6. Sign-Off Criteria (Milestone 1 Exit Check)

| # | Criterion | Measurable Threshold |
|:---:|:---|:---|
| 1 | **Authority route proof** | Physical wiring diagram; disconnecting STM32 stops all motor communication |
| 2 | **Servo enumeration** | All 6 joints respond with correct IDs |
| 3 | **IK reaching on real hardware** | ≥ 8/10 targets reached within $< 15\text{ mm}$ via CLI |
| 4 | **Sim2Real comparison table** | 10-target table showing sim error vs. real error with identified discrepancy sources |
| 5 | **Camera calibration** | $T_{\text{cam}}^{\text{base}}$ computed; reprojection error $< 2.0\text{ pixels}$ |
| 6 | **RPC bridge latency** | Round-trip $< 1.5\text{ ms}$ (median over 1,000 samples) |
| 7 | **Safe-state verification** | Zero uncommanded motion across 5 power-cutoff cycles |
| 8 | **Sensor dropout handled** | `CAMERA_DROPOUT` and stale-data flags verified |

*Staff signs off `[ ] A2`, `[ ] A3`, and `[ ] D1` on the team's [Competency Card](../curriculum/student-competencies.md).*

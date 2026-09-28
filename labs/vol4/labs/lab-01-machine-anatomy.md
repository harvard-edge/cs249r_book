# Lab 1: Machine Anatomy & Station Bring-Up

**Schedule:** Weeks 1–3 · Milestone 1 due end of Week 3
**Required Textbook Reading:** [Chapter 1: The Causal Boundary](../../../books/vol4/01_boundary/01_boundary.qmd), [Chapter 2: The Physical Body](../../../books/vol4/02_body/02_body.qmd), [Chapter 3: The Cognitive Brain](../../../books/vol4/03_brain/03_brain.qmd), [Chapter 4: The Nervous System](../../../books/vol4/04_nervous/04_nervous.qmd)
**Target Competencies:** `[ ] A1 Plant Mechanics & Safe Envelope`, `[ ] A2 Multi-Modal Sensing & Calibration`, `[ ] A3 Feedback Timing & Latency`, `[ ] D1 Hardware Authority Routing`
**Milestone Alignment:** Concludes [Milestone 1](../curriculum/syllabus.md#sec-milestones) (End of Week 3)

---

### 1. The Physical Question

Where does computational authority end and physical force begin? If a host computer or neural network issues an erroneous command, what hardware mechanisms guarantee that the machine fails safely? And once you trust the plant, how do you integrate disparate sensory modalities—an external camera perceiving light and an internal encoder measuring angle—across asynchronous processors with aligned spatial frames and synchronized clocks?

Before teaching the machine anything, you must understand its physical limits, its sensor accuracy, and the latency of its internal communication — because in Physical AI, **you cannot `Ctrl+Z` physics.**

---

### 2. Hardware Setup

> **Standard Bench Configuration:** See the [Station Reference Card](station-reference.md) for the full component list, wiring diagram, three engineering rules, and the four-tap telemetry schema used in every lab.

Additional materials for this lab:
1. **Calibration Target:** Printed ChArUco board fixture placed flat on the tabletop.
2. **Initial State:** Motor power supply switch **OFF (Disarmed)**. Arm in resting, folded configuration.

---

### 3. Step-by-Step Protocol

#### Phase A: Physical Authority & Power Audit (Week 1)

##### Step 1: Wiring Inspection & Authority Tracing
1. Inspect the station wiring. Trace every cable from the power strip to the arm.
2. Verify that the **only** electrical connection to the STS3215 servo bus originates from the STM32 microcontroller header.
3. Confirm that no direct USB-to-UART bridge connects the host workstation to the servo bus.
4. Draw the physical authority diagram showing power rails, data buses, and cutoff paths.

##### Step 2: Safe-State Verification
1. With motor power OFF, boot the UNO Q. Verify that no servo moves on power-up.
2. Apply and remove motor power while logging. Verify zero uncommanded motion on power transitions across 5 cycles.
3. Test the physical power cutoff switch. Confirm motor power drops while Linux logging remains active.

#### Phase B: Servo Enumeration & Mechanical Envelope (Week 1–2)

##### Step 3: Servo Bus Bring-Up
1. Flash the staff-provided baseline motion firmware to the STM32 MCU via the Arduino IDE / App Lab CLI.
2. Open the STM32 serial monitor at 115200 baud. Energize motor power via the physical toggle switch.
3. Run the enumeration command:
   ```bash
   uno-q-cli bus scan --baud 1000000
   ```
4. Verify that all 6 STS3215 servos respond with their programmed IDs (`1` to `6`), firmware versions, and current voltages.

##### Step 4: Zero-Homing & Envelope Characterization
1. Use the physical calibration jig to align each joint to its zero-angle mechanical detent.
2. Record the raw 12-bit magnetic encoder counts for each joint.
3. Slowly articulate each joint by hand across its full physical travel range. Measure and record:
   * $\theta_{i,\min}$ and $\theta_{i,\max}$ in mechanical degrees.
   * Hard physical stop angles vs. allowable software travel bounds.
4. Flash the calibrated soft limits into the STM32 non-volatile configuration memory.

#### Phase C: Sensing, Calibration & Inter-Core Bridge (Week 2–3)

##### Step 5: Camera Ingestion & Linux V4L2 Bring-Up
1. Log into the Qualcomm Linux terminal on the UNO Q.
2. Verify video capture device discovery:
   ```bash
   v4l2-ctl --list-devices
   v4l2-ctl -d /dev/video0 --list-formats-ext
   ```
3. Capture a test frame at $640 \times 480$ resolution @ 30 FPS. Verify lighting uniformity and absence of motion blur.

##### Step 6: Extrinsic Camera-to-Base Calibration
1. Place the calibration target in the center of the arm's reachable workspace.
2. Run the staff camera calibration utility on Qualcomm Linux:
   ```bash
   python3 -m pai_tools.calibrate_camera --cam-id 0 --board-type charuco
   ```
3. Command the arm to touch 4 marked fiducial points on the calibration board. Record the forward kinematics end-effector position $(X, Y, Z)_{\text{base}}$ for each point.
4. Solve the Perspective-n-Point (PnP) problem to compute the transformation matrix $T_{\text{cam}}^{\text{base}} \in SE(3)$. Verify reprojection error $< 2.0\text{ pixels}$.

##### Step 7: Inter-Core RPC Bridge Bring-Up
1. Launch the `arduino-bridge` daemon connecting Qualcomm Linux to the STM32 MCU over the high-speed UART link.
2. Write a Python script to request joint telemetry across the bridge:
   ```python
   from pai_bridge import BridgeClient

   client = BridgeClient()
   telemetry = client.get_joint_state()  # [q1, q2, q3, q4, q5, q6, timestamp_us]
   print("MCU Joint State:", telemetry)
   ```
3. Measure round-trip ping-pong RPC latency over 1,000 queries. Verify mean transport delay $< 1.0\text{ ms}$.

---

### 4. Disturbance & Failure Tests

#### Test A: Power-Off Drop Test
Command the arm to a stable elevated test pose (Joint 2 @ $45^\circ$, Joint 3 @ $45^\circ$). While elevated, switch off the 7.4V motor power supply.
* *Observation:* Record the mechanical drop trajectory as gravity pulls the unpowered links down. Verify that no mechanical binding or violent snapping occurs.

#### Test B: Re-Power Surge Audit
With the arm now resting in an arbitrary fallen position, flip the motor power switch back ON.
* *Pass Criteria:* The arm must remain completely limp and passive. The servos must **never** violently jerk, snap to zero, or execute pre-stored moves upon power restoration until an explicit arming handshake is sent from the console.

#### Test C: Sensor Occlusion
While streaming synchronized observations, physically block the webcam with an index card for 3 seconds.
* *Pass Criteria:* The software pipeline flags a `CAMERA_DROPOUT` error rather than blindly feeding blank or stale frames into downstream processing.

#### Test D: Stale Telemetry Detection
Artificially delay STM32 telemetry responses by 200 ms. Verify that Qualcomm Linux detects that observation timestamp age exceeds the allowable threshold ($T_{\text{age}} > 50\text{ ms}$) and marks the observation invalid.

---

### 5. Multi-Tap Telemetry Traces

**Trace A (Mechanical):** Log an elevation move in CSV format and verify that $a_{\text{meas}}$ converges to $a_{\text{enf}}$ within $\pm 0.5^\circ$, establishing the fundamental four-tap telemetry schema used all semester.

**Trace B (Temporal):** Capture 10 repeated commanded reach poses. For each pose, log:
* $t_{\text{cam}}$: Hardware timestamp of image capture.
* $t_{\text{joint}}$: Hardware timestamp of MCU joint readback.
* $\Delta t = |t_{\text{cam}} - t_{\text{joint}}|$: Inter-modal skew (must be $< 16\text{ ms}$ at 30 FPS).
* Commanded tool pose $\mathbf{x}_{\text{cmd}}$ vs. settled optical tool pose $\mathbf{x}_{\text{optical}} = T_{\text{cam}}^{\text{base}} \cdot \mathbf{x}_{\text{pixel}}$.
* Quantify the command-versus-settled error distribution across the workspace.

---

### 6. Common Pitfalls & Debugging

* ⚠️ **TTL Half-Duplex Contention:** The STS3215 bus uses a single bi-directional data line. If the STM32 driver does not disable its transmitter before reading, bus collisions will corrupt packets. Ensure the direction-control pin timing is exact.
* ⚠️ **Voltage Sag under Multi-Servo Stall:** If multiple servos draw stall current simultaneously (>1.5A each), poorly regulated supplies will dip, causing the STM32 or servos to brown-out reset. Ensure logic power is completely decoupled from motor power.
* ⚠️ **Rolling Shutter Warp:** Cheap USB webcams use rolling shutters. Rapid arm movements will cause visual distortion. Always calibrate and evaluate poses when the arm is settled or moving smoothly.
* ⚠️ **Clock Drift Between Cores:** Qualcomm Linux uses a POSIX clock (`CLOCK_MONOTONIC`), while the STM32 uses a hardware SysTick timer. Perform a timestamp sync handshake at bridge initialization to correlate timestamps.

---

### 7. Sign-Off Criteria (Milestone 1 Exit Check)

To receive credit for Lab 1 and complete Milestone 1, demonstrate the following live to the instructor:

| # | Criterion | Measurable Threshold |
|:---:|:---|:---|
| 1 | **Authority Route Proof** | Show physical wiring diagram; prove disconnecting STM32 stops all motor communication |
| 2 | **Safe-State Verification** | Zero uncommanded motion across 5 boot/reset/power-cutoff cycles |
| 3 | **Servo Enumeration** | All 6 joints respond with correct IDs on bus scan |
| 4 | **Measured Operating Envelope** | Travel limits ($\theta_{\min}, \theta_{\max}$) for all 6 joints with $\pm 0.5^\circ$ repeatability |
| 5 | **Extrinsic Matrix Validation** | $T_{\text{cam}}^{\text{base}}$ computed; reprojection error $< 2.0\text{ pixels}$; physical touch accuracy $< 3\text{ mm}$ |
| 6 | **Synchronized Stream Proof** | Live display of synchronized RGB video + 6-axis joint telemetry |
| 7 | **RPC Bridge Latency** | Round-trip $< 1.5\text{ ms}$ (median over 1,000 samples) |
| 8 | **Power-Off & Re-Arm Trace** | Safe passive drop on motor cutoff; zero uncommanded motion on re-power |

*Staff signs off `[ ] A1`, `[ ] A2`, `[ ] A3`, and `[ ] D1` on the team's [Competency Card](../curriculum/student-competencies.md).*

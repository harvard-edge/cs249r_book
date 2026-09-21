# Lab 2: Multi-Modal Sensing & The Inter-Core Bridge

**Schedule:** Week 3 | **Part:** [Part I: The Machine Anatomy](module-1-machine-anatomy.md)
**Required Textbook Reading:** [Chapter 3: The Cognitive Brain](../../../books/vol4/03_brain/03_brain.qmd) & [Chapter 4: The Nervous System](../../../books/vol4/04_nervous/04_nervous.qmd)
**Target Competencies:** `[ ] A2 Multi-Modal Sensing & Calibration`, `[ ] A3 Feedback Timing & Latency`
**Milestone Alignment:** Concludes [Milestone 1](../curriculum/syllabus.md#sec-milestones) (End of Week 3)

---

### 1. The Physical Question
How does an embodied system integrate disparate sensory modalities across asynchronous processors? When an external camera perceives light and an internal encoder measures angle, how do we align their spatial coordinate frames and synchronize their temporal clocks to eliminate latency jitter?

---

### 2. Hardware Setup

| Component | Function in Lab 2 | Interface & Protocol | Visual Reference |
|:---|:---|:---|:---:|
| **Logitech C270 Webcam** | Overhead workspace visual sensing | UVC USB 2.0, 720p @ 30 FPS, rigid overhead clamp mount | <img src="../assets/images/logitech-c270-webcam.png" alt="Logitech C270 Webcam" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |
| **Feetech STS3215 Bus Ports** | Inter-servo daisy-chain communications | Dual 3-pin ports, TTL half-duplex UART @ 1 Mbps | <img src="../assets/images/feetech-sts3215-bus-ports.jpg" alt="STS3215 Bus Ports" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |
| **Arduino UNO Q Board** | Multi-modal edge computer & bridge master | Qualcomm Linux (V4L2 vision) + STM32U585 MCU (STS3215 bus) | <img src="../assets/images/arduino-uno-q.jpg" alt="Arduino UNO Q" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |

1. **Compute Board:** Arduino UNO Q with Qualcomm Debian Linux booted and USB webcam attached.
2. **Camera:** Standard 720p UVC USB webcam mounted on a rigid overhead clamp pointing down at the workspace.
3. **Calibration Target:** Printed OpenCV checkerboard or ArUco board fixture placed flat on the table.
4. **Arm Station:** Seeed SO-101 arm connected to the STM32 MCU via the internal RPC bridge.

![Physical Bench Rigging & Electrical Power Isolation Harness](../assets/images/vol4-bench-wiring-harness.svg)

---

### 3. Step-by-Step Protocol

#### Step 1: Camera Ingestion & Linux V4L2 Bring-Up
1. Log into the Qualcomm Linux terminal on the UNO Q.
2. Verify video capture device discovery:
   ```bash
   v4l2-ctl --list-devices
   v4l2-ctl -d /dev/video0 --list-formats-ext
   ```
3. Capture a test frame at $640 \times 480$ resolution @ 30 FPS. Verify lighting uniformity and absence of motion blur.

#### Step 2: Extrinsic Camera-to-Base Calibration
1. Place the calibration target in the center of the arm's reachable workspace.
2. Run the staff camera calibration utility on Qualcomm Linux:
   ```bash
   python3 -m pai_tools.calibrate_camera --cam-id 0 --board-type charuco
   ```
3. Command the arm to touch 4 marked fiducial points on the calibration board. Record the forward kinematics end-effector position $(X, Y, Z)_{\text{base}}$ for each point.
4. Solve the Perspective-n-Point (PnP) problem to compute the transformation matrix $T_{\text{cam}}^{\text{base}} \in SE(3)$. Verify reprojection error $< 2.0\text{ pixels}$.

#### Step 3: Inter-Core RPC Bridge Bring-Up
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

### 4. The Disturbance & Failure Test
1. **Sensor Occlusion Test:** While streaming synchronized observations, physically block the webcam with an index card for 3 seconds.
   * *Observation:* Verify that the software pipeline flags a `CAMERA_DROPOUT` error rather than blindly feeding blank or stale frames into downstream processing.
2. **Stale Telemetry Injection:** Artificially delay STM32 telemetry responses by 200 ms. Verify that Qualcomm Linux detects that observation timestamp age exceeds the allowable threshold ($T_{\text{age}} > 50\text{ ms}$) and marks the observation invalid.

---

### 5. Multi-Tap Telemetry Trace
Capture 10 repeated commanded reach poses. For each pose, log:
* $t_{\text{cam}}$: Hardware timestamp of image capture.
* $t_{\text{joint}}$: Hardware timestamp of MCU joint readback.
* $\Delta t = |t_{\text{cam}} - t_{\text{joint}}|$: Inter-modal skew (must be $< 16\text{ ms}$ at 30 FPS).
* Commanded tool pose $\mathbf{x}_{\text{cmd}}$ vs. settled optical tool pose $\mathbf{x}_{\text{optical}} = T_{\text{cam}}^{\text{base}} \cdot \mathbf{x}_{\text{pixel}}$.
* Quantify the command-versus-settled error distribution across the workspace.

---

### 6. Common Pitfalls & Debugging
* ⚠️ **Rolling Shutter Warp:** Cheap USB webcams use rolling shutters. Rapid arm movements will cause visual distortion. Always calibrate and evaluate poses when the arm is settled or moving smoothly.
* ⚠️ **Clock Drift Between Cores:** Qualcomm Linux uses a POSIX clock (`CLOCK_MONOTONIC`), while the STM32 uses a hardware SysTick timer. Perform a timestamp sync handshake at bridge initialization to correlate timestamps.

---

### 7. Sign-Off Criteria (The Exit Check)
To receive credit for Lab 2 and complete Milestone 1:
1. [ ] **Extrinsic Matrix Validation:** Present the calculated $T_{\text{cam}}^{\text{base}}$ matrix and demonstrate that reprojected pixel coordinates match physical fingertip touch within 3 mm.
2. [ ] **Synchronized Stream Proof:** Display a live visualization showing synchronized RGB video alongside real-time 6-axis joint angle telemetry ($I_t, q_t$).
3. [ ] **Loop Latency Histogram:** Submit a histogram of 1,000 RPC round-trip latencies proving transport delay $< 1.5\text{ ms}$.
*Staff signs off `[ ] A2` and `[ ] A3` on the team's [Competency Card](../curriculum/student-competencies.md).*

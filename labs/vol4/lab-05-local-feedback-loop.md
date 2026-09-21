# Lab 5: Autonomous Closed-Loop Reach on UNO Q

**Schedule:** Weeks 7–8 | **Part:** [Part III: Running the Machine](module-3-running-the-machine.md)
**Required Textbook Reading:** [Chapter 8: Sensor Perception](../../books/vol4/08_perception/08_perception.qmd), [Chapter 11: Trajectory Planning](../../books/vol4/11_planning/11_planning.qmd) & [Chapter 13: Silicon Placement](../../books/vol4/13_placement/13_placement.qmd)
**Target Competencies:** `[ ] C1 Closed-Loop Autonomous Action`
**Milestone Alignment:** Contributes to [Milestone 3](syllabus.md#sec-milestones) (End of Week 9)

---

### 1. The Physical Question
Can a trained neural policy run completely untethered on edge silicon, closing the physical loop between live camera pixels and motor torques without relying on an external workstation or cloud server?

---

### 2. Hardware Setup
1. **Edge Board:** Arduino UNO Q running standalone (Qualcomm Linux MPU + STM32 MCU).
2. **Network Connection:** Disconnected or headless network (zero tethering to a workstation during task execution).
3. **Camera & Arm:** USB webcam mounted above the table; Seeed SO-101 arm in resting pose.
4. **Target Setup:** A soft colored foam block placed at an arbitrary, unscripted location inside the reachable workspace envelope.

---

### 3. Step-by-Step Protocol

#### Step 1: Autonomous Loop Integration
1. Assemble the autonomous runtime script on Qualcomm Linux (`pai_edge_runtime.py`):
   ```python
   import cv2
   from pai_bridge import BridgeClient
   from pai_onnx import PolicyRunner

   cam = cv2.VideoCapture(0)
   bridge = BridgeClient()
   policy = PolicyRunner("so101_act.onnx")

   # Main closed-loop cycle
   while True:
       ret, frame = cam.read()
       joint_state = bridge.get_joint_state()

       # 1. Propose action chunk
       action_chunk = policy.predict(frame, joint_state)  # K steps

       # 2. Stream chunk across RPC bridge to STM32
       bridge.send_action_chunk(action_chunk)
   ```
2. Verify that the script handles camera frame preprocessing, normalizes joint state inputs, and streams the predicted action chunk over the internal RPC bridge.

#### Step 2: The Untethered Autonomous Reach
1. Disconnect all USB cables connecting the UNO Q to your laptop. The UNO Q must run exclusively on its dedicated power supply.
2. Place the block at an arbitrary starting position in the workspace.
3. Trigger task start via the onboard user button or a headless SSH command.
4. **Witness the Loop:**
   * The camera captures the scene.
   * The ONNX policy processes the frame and current joint state.
   * The policy proposes a 16-step action chunk ($a_{\text{req}}$).
   * The STM32 MCU validates the trajectory and commands the STS3215 servos ($a_{\text{enf}}$).
   * The arm smoothly reaches toward the block and touches it.

#### Step 3: Loop Latency Breakdown
1. Instrument the edge runtime script to log the latency breakdown over 50 consecutive cycles:
   * $t_{\text{cam}}$: Frame capture and V4L2 buffer copy.
   * $t_{\text{inf}}$: Neural policy ONNX inference.
   * $t_{\text{rpc}}$: Inter-core RPC transmission time to STM32.
   * $t_{\text{bus}}$: Half-duplex TTL servo packet transmission time.
2. Plot the end-to-end loop latency histogram. Verify that the 99th-percentile tail latency satisfies $t_{99} < 100\text{ ms}$.

---

### 4. The Disturbance & Failure Test
1. **Unannounced Novel Position:** Place the block in an unfamiliar starting pose never seen in the 30 training demonstrations (e.g., far right boundary).
   * *Observation:* Verify whether the policy generalizes and reaches the new position, or whether it exhibits compounding error.
2. **Visual Clutter Obstacle:** Place a benign, neutral object (e.g., an empty tape roll) adjacent to the block. Verify that the visual perception pipeline does not get distracted or drive into the clutter.

---

### 5. Multi-Tap Telemetry Trace
Record the full multi-tap trace of an untethered autonomous reach:
* Plot the 6 joint angle trajectories over time ($t = 0$ to $t_{\text{end}}$).
* Overlay requested action proposals ($a_{\text{req}}$) and measured physical feedback ($a_{\text{meas}}$).
* Measure the final positioning accuracy: Euclidean distance between arm end-effector and block center of mass.

---

### 6. Common Pitfalls & Debugging
* ⚠️ **Headless Camera Exposure Shifts:** When running headless without an interactive GUI, ensure auto-exposure does not hunt or oscillate between frames. Lock camera exposure and white balance using `v4l2-ctl -c exposure_auto=1`.
* ⚠️ **Thread Starvation:** Ensure camera capture runs in a separate thread from policy inference so that V4L2 buffers do not drop frames while the neural model computes.

---

### 7. Sign-Off Criteria (The Exit Check)
To receive credit for Lab 5:
1. [ ] **Witnessed Untethered Reach:** Demonstrate a live autonomous reach on the SO-101 arm running entirely from the Arduino UNO Q without host PC intervention.
2. [ ] **Physical Task Completion:** The arm must successfully make contact with the block placed at a staff-selected arbitrary position within the envelope.
3. [ ] **End-to-End Latency Profile:** Submit a verified latency breakdown showing total loop execution time $< 100\text{ ms}$.
*Staff signs off `[ ] C1` on the team's [Competency Card](student-competencies.md).*

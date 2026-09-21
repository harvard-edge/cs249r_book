# Lab 8: Fault Injection & Fail-Safe Cutoffs

**Schedule:** Week 11 | **Part:** [Part IV: Governing the Machine](module-4-governing-the-machine.md)
**Required Textbook Reading:** [Chapter 15: Adversarial Verification](../../../books/vol4/15_verification/15_verification.qmd) & [Chapter 16: Safe Release](../../../books/vol4/16_release/16_release.qmd)
**Target Competencies:** `[ ] D2 Real-Time Safety Governor & Fault Isolation`
**Milestone Alignment:** Concludes [Milestone 4](../curriculum/syllabus.md#sec-milestones) (End of Week 11)

---

### 1. The Physical Question
What happens when the host computer freezes, the camera is disconnected, or the communication bus stalls while a robot is moving? How do we implement hardware watchdogs and physical cutoffs that guarantee immediate safe state transitions with zero queued command backlog?

---

### 2. Hardware Setup

| Component | Role in Fault & Release Testing | Fault Injection Mode | Visual Reference |
|:---|:---|:---|:---:|
| **Arduino UNO Q Board** | Dual-silicon brain running Linux runtime & STM32 watchdog | Process suspension (`kill -STOP`), RPC buffer overflow | <img src="../assets/images/arduino-uno-q.jpg" alt="Arduino UNO Q" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |
| **Logitech C270 Webcam** | Visual perception sensor | Live physical USB disconnect during trajectory execution | <img src="../assets/images/logitech-c270-webcam.png" alt="Logitech C270 Webcam" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |
| **Seeed SO-101 6-DoF Arm** | Physical plant under active load | Table collision veto, watchdog freeze, unpowered gravity drop | <img src="../assets/images/so101-follower.png" alt="SO-101 Robot Arm" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |
| **Switched 7.4V DC Rail** | Isolated motor power rail | Live power disconnect while logic telemetry continues uninterrupted | <img src="../assets/images/feetech-sts3215-bus-ports.jpg" alt="Motor Bus Power Ports" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |

1. **Controller Board:** Arduino UNO Q (Qualcomm Linux MPU + STM32U585 MCU).
2. **Motor Power Rail:** Switched 7.4V/5A DC motor power supply sharing a common star ground with the Arduino UNO Q.
3. **Perception Sensor:** Logitech C270 USB webcam connected to Qualcomm Linux.
4. **Robot Station:** Seeed SO-101 6-DoF arm carrying a light payload in active motion.
5. **Fault Injection Toolkit:** Python process-killing scripts and serial corruption utilities.

---

### 3. Step-by-Step Protocol

#### Step 1: Communication Watchdog Implementation
1. Program a hardware SysTick watchdog timer on the STM32 MCU ($T_{\text{watchdog}} = 150\text{ ms}$).
2. The Qualcomm Linux edge runtime must send periodic heartbeat pulses over the RPC bridge alongside action proposals.
3. If no valid heartbeat arrives within $150\text{ ms}$, the STM32 must immediately:
   * Veto all pending motion.
   * Command servos to hold present position or gracefully transition to compliant de-energized rest.
   * Purge all incoming command queues.

#### Step 2: The Host Freeze Stress Test
1. Command the arm to execute an active trajectory.
2. Mid-trajectory, forcefully suspend the Qualcomm Linux Python process:
   ```bash
   kill -STOP $(pgrep -f pai_edge_runtime)
   ```
3. **Observe the Reaction:**
   * Verify that within exactly $150\text{ ms}$, the STM32 watchdog triggers.
   * Confirm that the arm safely freezes or settles without jerking or completing the remaining trajectory open-loop.

#### Step 3: Queue Backlog & Buffer Flush Test
1. While the Python process is frozen, flood the serial bridge with delayed commands.
2. Resume the Python process:
   ```bash
   kill -CONT $(pgrep -f pai_edge_runtime)
   ```
3. **Pass Criteria:** The STM32 must **not** execute the queued, stale commands. It must enforce a clean reset and require an explicit re-arm handshake before accepting fresh proposals.

---

### 4. The Disturbance & Failure Test
1. **Camera Disconnect During Motion:** Physically unplug the USB webcam while the arm is reaching toward an object.
   * *Pass Criteria:* The Linux perception thread must detect the device drop, flag `SENSOR_LOSS`, and signal the STM32 to abort the reach and return to rest pose.
2. **Motor Power Cutoff Under Full Load:** While the arm is lifting a payload at maximum speed, switch off the 7.4V motor DC power supply.
   * *Pass Criteria:* Motor torque must drop to zero immediately ($< 10\text{ ms}$). Logic power on the UNO Q must remain uninterrupted, and telemetry logging must capture the event.

---

### 5. Multi-Tap Telemetry Trace
Capture the complete telemetry trace during an injected watchdog timeout:
* Plot the continuous stream of heartbeat timestamps.
* Mark the exact moment of process suspension ($t_{\text{kill}}$).
* Show the watchdog timeout firing at $t_{\text{kill}} + 150\text{ ms}$.
* Show the enforced command $a_{\text{enf}}$ dropping to zero velocity and the motor readback $a_{\text{meas}}$ holding steady.
* Prove that no queued commands execute upon process resumption.

---

### 6. Common Pitfalls & Debugging
* ⚠️ **UART Buffer Overflow:** When a receiver stops processing, Linux UART buffers can fill up with hundreds of bytes. Upon resumption, reading stale buffer data will cause erratic behavior. Always explicitly flush the input buffer (`tcflush(fd, TCIFLUSH)`) on rearm.
* ⚠️ **Power Cutoff Inductive Kickback:** Cutting high inductive motor currents abruptly can cause inductive kickback spikes. Ensure flyback suppression diodes or snubbers are present across the motor power rail.

---

### 7. Sign-Off Criteria (The Exit Check)
To receive credit for Lab 8 and complete Milestone 4:
1. [ ] **Watchdog Timeout Proof:** Demonstrate that suspending the Linux process halts the arm within $150\text{ ms}$ with zero anomalous motion.
2. [ ] **Zero Backlog Audit:** Prove that resuming the suspended process does not execute stale buffered movements.
3. [ ] **Motor Power Cutoff Verification:** Switch off the 7.4V motor power supply during active motion, proving immediate motor cutoff while system telemetry continues logging.
*Staff signs off `[ ] D2` on the team's [Competency Card](../curriculum/student-competencies.md).*

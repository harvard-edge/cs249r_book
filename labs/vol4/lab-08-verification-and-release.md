# Lab 8: Fault Injection & Fail-Safe Cutoffs

**Schedule:** Week 11 | **Part:** [Part IV: Governing the Machine](module-4-governing-the-machine.md)
**Required Textbook Reading:** [Chapter 15: Adversarial Verification](../../books/vol4/15_verification/15_verification.qmd) & [Chapter 16: Safe Release](../../books/vol4/16_release/16_release.qmd)
**Target Competencies:** `[ ] D2 Real-Time Safety Governor & Fault Isolation`
**Milestone Alignment:** Concludes [Milestone 4](syllabus.md#sec-milestones) (End of Week 11)

---

### 1. The Physical Question
What happens when the host computer freezes, the camera is disconnected, or the communication bus stalls while a robot is moving? How do we implement hardware watchdogs and physical cutoffs that guarantee immediate safe state transitions with zero queued command backlog?

---

### 2. Hardware Setup
1. **Controller Board:** Arduino UNO Q (Qualcomm Linux MPU + STM32 MCU).
2. **Safety Switch:** Accessible physical emergency-stop toggle switch wired in-line with the motor power rail.
3. **Robot Station:** Seeed SO-101 6-DoF arm carrying a light payload in active motion.
4. **Fault Injection Toolkit:** Python process-killing scripts and serial corruption utilities.

---

### 3. Step-by-Step Bench Protocol

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
2. **Physical E-Stop Depressed Under Full Load:** While the arm is lifting a payload at maximum speed, depress the physical emergency-stop switch.
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

### 6. Common Hardware Pitfalls & Debugging
* ⚠️ **UART Buffer Overflow:** When a receiver stops processing, Linux UART buffers can fill up with hundreds of bytes. Upon resumption, reading stale buffer data will cause erratic behavior. Always explicitly flush the input buffer (`tcflush(fd, TCIFLUSH)`) on rearm.
* ⚠️ **E-Stop Induced Brownout:** Cutting high inductive motor currents abruptly can cause inductive kickback spikes. Ensure flyback suppression diodes or snubbers are present across the motor power rail.

---

### 7. Bench Sign-Off Criteria (The Exit Check)
To receive credit for Lab 8 and complete Milestone 4:
1. [ ] **Watchdog Timeout Proof:** Demonstrate that suspending the Linux process halts the arm within $150\text{ ms}$ with zero anomalous motion.
2. [ ] **Zero Backlog Audit:** Prove that resuming the suspended process does not execute stale buffered movements.
3. [ ] **Physical E-Stop Verification:** Depress the physical E-stop switch during active motion, proving immediate motor cutoff while system telemetry continues logging.
*Staff signs off `[ ] D2` on the team's [Competency Card](student-competencies.md).*

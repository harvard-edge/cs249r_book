# Lab 6: Safety Governor — Build It, Then Break It

**Schedule:** Weeks 10–11 · Milestone 4 due end of Week 11
**Required Textbook Reading:** [Chapter 12: Safety Enforcement](../../../books/vol4/12_safety/12_safety.qmd), [Chapter 14: Supervisory Intervention](../../../books/vol4/14_supervisory/14_supervisory.qmd), [Chapter 15: Adversarial Verification](../../../books/vol4/15_adversarial/15_adversarial.qmd), [Chapter 16: Safe Release](../../../books/vol4/16_release/16_release.qmd)
**Target Competencies:** `[ ] D1 Hardware Authority Routing`, `[ ] D2 Deterministic Safety Enforcement`
**Milestone Alignment:** Concludes [Milestone 4](../curriculum/syllabus.md#sec-milestones) (End of Week 11)

---

### 1. The Physical Question

Can an independent microcontroller guarantee physical safety even when the Linux brain crashes, lies, or goes silent — and can you **prove** it by systematically trying to break it?

This lab has two phases. In Phase A, you build the deterministic safety governor on the STM32 MCU: velocity clamps, workspace geofences, and joint limits. In Phase B, you become the adversary — injecting faults, freezing processes, corrupting packets, and cutting power to verify that the governor holds under every failure mode you can devise.

---

### 2. Hardware Setup

> **Standard Bench Configuration:** See the [Station Reference Card](station-reference.md). This lab uses the full station with the autonomous S·P·A loop running (as established in Labs 4–5).

---

### 3. Step-by-Step Protocol

#### Phase A: Build the Governor (Week 10)

##### Step 1: Programming the STM32 Permission Barrier
1. Open the STM32 firmware project in Arduino App Lab / STM32Cube.
2. Implement the **Simplex Safety Governor** filter:
   ```c
   bool evaluate_action_proposal(const ActionProposal* req, EnforcedAction* enf) {
       for (int i = 0; i < 6; i++) {
           // 1. Joint Angle Limit Check
           if (req->target_pos[i] < joint_min[i] || req->target_pos[i] > joint_max[i]) {
               return false; // Veto proposal
           }

           // 2. Velocity Delta Clamp
           float delta = req->target_pos[i] - current_pos[i];
           if (fabs(delta) > max_delta_per_step[i]) {
               delta = copysign(max_delta_per_step[i], delta);
           }
           enf->target_pos[i] = current_pos[i] + delta;
       }

       // 3. Geometric Workspace Barrier (Table Geofence)
       Vector3D tool_pos = forward_kinematics(enf->target_pos);
       if (tool_pos.z < Z_TABLE_LIMIT) {
           return false; // Veto proposal penetrating tabletop
       }
       return true;
   }
   ```
3. Flash the safety firmware to the STM32 MCU.

##### Step 2: Injected Malicious & Out-of-Bounds Proposals
1. Run a synthetic fault-injection script on Qualcomm Linux:
   ```bash
   python3 -m pai_tools.inject_faults --target mcu --type out_of_bounds
   ```
2. Inject three distinct failure cases:
   * *Excessive Joint Velocity:* Send a proposal commanding Joint 1 to jump $90^\circ$ in 10 ms.
   * *Table Collision:* Send a proposal commanding the end-effector to drive 20 mm below table surface.
   * *Exceeded Soft Stop:* Send a proposal commanding Joint 2 beyond its calibrated safe range.
3. Confirm that the STM32 logs an explicit refusal event and clamps or rejects the motion in $< 5\text{ ms}$.

##### Step 3: Verifying the Single-Path Hardware Boundary
1. Audit the physical hardware path.
2. Attempt to bypass the STM32 by issuing commands from Linux directly to serial device nodes.
3. Prove that without STM32 permission arbitration, no electrical command can reach the STS3215 motor bus.

---

#### Phase B: Break the Governor (Week 11)

##### Step 4: Communication Watchdog Implementation
1. Program a hardware SysTick watchdog timer on the STM32 MCU ($T_{\text{watchdog}} = 150\text{ ms}$).
2. The Qualcomm Linux edge runtime must send periodic heartbeat pulses over the RPC bridge alongside action proposals.
3. If no valid heartbeat arrives within $150\text{ ms}$, the STM32 must immediately:
   * Veto all pending motion.
   * Command servos to hold present position or gracefully transition to compliant de-energized rest.
   * Purge all incoming command queues.

##### Step 5: The Host Freeze Stress Test
1. Command the arm to execute an active trajectory using the autonomous S·P·A loop.
2. Mid-trajectory, forcefully suspend the Qualcomm Linux Python process:
   ```bash
   kill -STOP $(pgrep -f pai_edge_runtime)
   ```
3. **Observe the Reaction:**
   * Verify that within exactly $150\text{ ms}$, the STM32 watchdog triggers.
   * Confirm that the arm safely freezes or settles without jerking or completing the remaining trajectory open-loop.

##### Step 6: Queue Backlog & Buffer Flush Test
1. While the Python process is frozen, flood the serial bridge with delayed commands.
2. Resume the Python process:
   ```bash
   kill -CONT $(pgrep -f pai_edge_runtime)
   ```
3. **Pass Criteria:** The STM32 must **not** execute the queued, stale commands. It must enforce a clean reset and require an explicit re-arm handshake before accepting fresh proposals.

##### Step 7: Sensor Dropout & Packet Corruption
1. Physically unplug the USB webcam while the arm is executing a reach.
   * *Pass Criteria:* The Linux perception thread must detect the device drop, flag `SENSOR_LOSS`, and signal the STM32 to abort the reach and return to rest pose.
2. Corrupt a proposal packet by flipping bits in the RPC stream.
   * *Pass Criteria:* The STM32 detects the CRC failure, rejects the packet, and logs the corruption event.

##### Step 8: Motor Power Cutoff Under Full Load
1. While the arm is lifting a payload at maximum speed, switch off the 7.4V motor DC power supply.
   * *Pass Criteria:* Motor torque drops to zero immediately ($< 10\text{ ms}$). Logic power on the UNO Q remains uninterrupted, and telemetry logging captures the event without data loss.

---

### 4. Disturbance & Failure Tests

The entirety of Phase B **is** the disturbance test — this lab's pedagogical core is adversarial verification.

Two additional stress tests:

1. **Adversarial Noise Injection:** Feed Gaussian noise into the action proposals ($a_{\text{req}} \sim \mathcal{N}(0, \sigma^2)$) simulating a corrupted model output.
   * *Pass Criteria:* The arm must exhibit smooth, bounded motion. The MCU velocity clamp must prevent all high-frequency motor chatter or violent vibrations.
2. **Table Crash Prevention:** Command the arm at maximum speed toward the tabletop.
   * *Pass Criteria:* The forward kinematics barrier must halt motion exactly at $Z = Z_{\text{table}} + 5\text{ mm}$, preventing any physical contact with the table surface.

---

### 5. Multi-Tap Telemetry Traces

**Trace A (Veto Authority):** Capture the multi-tap telemetry stream during a velocity clamping event:
* Plot $a_{\text{req}}$ (the jagged or excessive request from Linux).
* Overlay $a_{\text{enf}}$ (the smooth, clamped command issued by the STM32).
* Show $a_{\text{meas}}$ (the physical encoder readback).
* Demonstrate that $a_{\text{enf}} \ne a_{\text{req}}$, confirming that the MCU exercised active veto authority.

**Trace B (Watchdog Cutoff):** Capture the complete telemetry trace during an injected watchdog timeout:
* Plot the continuous stream of heartbeat timestamps.
* Mark the exact moment of process suspension ($t_{\text{kill}}$).
* Show the watchdog timeout firing at $t_{\text{kill}} + 150\text{ ms}$.
* Show the enforced command $a_{\text{enf}}$ dropping to zero velocity and the motor readback $a_{\text{meas}}$ holding steady.
* Prove that no queued commands execute upon process resumption.

---

### 6. Common Pitfalls & Debugging

* ⚠️ **Forward Kinematics Latency on MCU:** Computing full trigonometric forward kinematics on an MCU can be slow if floating-point hardware is not utilized. Ensure the Cortex-M33 hardware FPU is enabled in compiler flags.
* ⚠️ **Deadband Chatter at Limits:** When clamping at boundaries, ensure a hysteresis deadband is implemented so the arm does not oscillate on the edge of the barrier.
* ⚠️ **UART Buffer Overflow:** When a receiver stops processing, Linux UART buffers can fill up with hundreds of bytes. Upon resumption, reading stale buffer data will cause erratic behavior. Always explicitly flush the input buffer (`tcflush(fd, TCIFLUSH)`) on rearm.
* ⚠️ **Power Cutoff Inductive Kickback:** Cutting high inductive motor currents abruptly can cause inductive kickback spikes. Ensure flyback suppression diodes or snubbers are present across the motor power rail.

---

### 7. Sign-Off Criteria (Milestone 4 Exit Check)

To receive credit for Lab 6 and complete Milestone 4, demonstrate the following live to the instructor:

| # | Criterion | Measurable Threshold |
|:---:|:---|:---|
| 1 | **Velocity Clamp** | 10/10 over-speed proposals clamped to $\lvert\omega_i\rvert \le 45°/\text{s}$ |
| 2 | **Table Geofence** | 10/10 below-table proposals vetoed; arm halts at $Z \ge 15\text{ mm}$ |
| 3 | **Single Command Path** | No electrical or software bypass found; verified by instructor |
| 4 | **MCU Firmware Review** | Present STM32 source code for velocity clamp, geofence, and watchdog |
| 5 | **Multi-Tap Veto Plot** | Telemetry plot showing $a_{\text{enf}} \ne a_{\text{req}}$ during safety intervention |
| 6 | **Watchdog Timeout Proof** | Linux `kill -STOP` halts arm within $150\text{ ms}$ with zero anomalous motion |
| 7 | **Zero Backlog Audit** | Resuming suspended process does not execute stale buffered commands |
| 8 | **Packet Corruption Rejection** | CRC-corrupted packets rejected and logged |
| 9 | **Motor Power Cutoff** | Motor power loss does not affect logic telemetry logging |

*Staff signs off `[ ] D1` and `[ ] D2` on the team's [Competency Card](../curriculum/student-competencies.md).*

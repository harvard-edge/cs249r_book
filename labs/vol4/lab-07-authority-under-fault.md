# Lab 7: The Microcontroller Safety Governor

**Schedule:** Week 10 | **Part:** [Part IV: Governing the Machine](module-4-governing-the-machine.md)
**Required Textbook Reading:** [Chapter 12: Safety Enforcement](../../books/vol4/12_enforcement/12_enforcement.qmd) & [Chapter 14: Supervisory Intervention](../../books/vol4/14_intervention/14_intervention.qmd)
**Target Competencies:** `[ ] D1 Hardware Authority Routing & Boundary Enforcement`
**Milestone Alignment:** Contributes to [Milestone 4](syllabus.md#sec-milestones) (End of Week 11)

---

### 1. The Physical Question
How do we mathematically and physically ensure that an unverified, probabilistic neural network never commands a dangerous physical action? How does the real-time microcontroller act as a deterministic safety barrier that clips, clamps, or vetoes proposals before they reach motor coils?

---

### 2. Hardware Setup
1. **Controller Board:** Arduino UNO Q with STM32U585 MCU firmware running the safety governor.
2. **Serial Bridge:** Inter-core RPC bridge connecting Qualcomm Linux to the STM32 MCU.
3. **Robot Arm:** Seeed SO-101 6-DoF arm with 6× Feetech STS3215 bus servos.
4. **Physical Obstacle:** Tabletop acrylic guard defining the forbidden zone ($Z < Z_{\text{table}}$).

---

### 3. Step-by-Step Bench Protocol

#### Step 1: Programming the STM32 Permission Barrier
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

#### Step 2: Injected Malicious & Out-of-Bounds Proposals
1. Run a synthetic fault-injection script on Qualcomm Linux:
   ```bash
   python3 -m pai_tools.inject_faults --target mcu --type out_of_bounds
   ```
2. Inject three distinct failure cases:
   * *Excessive Joint Velocity:* Send a proposal commanding Joint 1 to jump $90^\circ$ in 10 ms.
   * *Table Collision:* Send a proposal commanding the end-effector to drive 20 mm below table surface.
   * *Exceeded Soft Stop:* Send a proposal commanding Joint 2 beyond its calibrated safe range.
3. Confirm that the STM32 logs an explicit refusal event and clamps or rejects the motion.

#### Step 3: Verifying the Single-Path Hardware Boundary
1. Audit the physical hardware path.
2. Attempt to bypass the STM32 by issuing commands from Linux directly to serial device nodes.
3. Prove that without STM32 permission arbitration, no electrical command can reach the STS3215 motor bus.

---

### 4. The Disturbance & Failure Test
1. **Adversarial Noise Injection:** Feed Gaussian noise into the action proposals ($a_{\text{req}} \sim \mathcal{N}(0, \sigma^2)$) simulating a corrupted model output.
   * *Pass Criteria:* The arm must exhibit smooth, bounded motion. The MCU velocity clamp must prevent all high-frequency motor chatter or violent vibrations.
2. **Table Crash Prevention:** Command the arm at maximum speed toward the tabletop.
   * *Pass Criteria:* The forward kinematics barrier must halt motion exactly at $Z = Z_{\text{table}} + 5\text{ mm}$, preventing any physical contact with the acrylic base.

---

### 5. Multi-Tap Telemetry Trace
Capture the multi-tap telemetry stream during a velocity clamping event:
* Plot $a_{\text{req}}$ (the jagged or excessive request from Linux).
* Overlay $a_{\text{enf}}$ (the smooth, clamped command issued by the STM32).
* Show $a_{\text{meas}}$ (the physical encoder readback).
* Demonstrate that $a_{\text{enf}} \ne a_{\text{req}}$, confirming that the MCU exercised active veto authority.

---

### 6. Common Hardware Pitfalls & Debugging
* ⚠️ **Forward Kinematics Latency on MCU:** Computing full trigonometric forward kinematics on an MCU can be slow if floating-point hardware is not utilized. Ensure the Cortex-M33 hardware FPU is enabled in compiler flags.
* ⚠️ **Deadband Chatter at Limits:** When clamping at boundaries, ensure a hysteresis deadband is implemented so the arm does not oscillate on the edge of the barrier.

---

### 7. Bench Sign-Off Criteria (The Exit Check)
To receive credit for Lab 7:
1. [ ] **MCU Firmware Verification:** Present the STM32 firmware source code showing the velocity clamping and table collision geofence implementations.
2. [ ] **Live Veto Demonstration:** The instructor injects an out-of-bounds command and a table-crash proposal. The MCU must reject both proposals in real time.
3. [ ] **Multi-Tap Veto Plot:** Submit a telemetry plot clearly showing the divergence between $a_{\text{req}}$ and $a_{\text{enf}}$ during a safety clamp event.
*Staff signs off `[ ] D1` on the team's [Competency Card](student-competencies.md).*

# Lab 1: The Causal Boundary & Servo Bus Bring-Up

**Schedule:** Weeks 1–2 | **Part:** [Part I: The Machine Anatomy](module-1-machine-anatomy.md)
**Required Textbook Reading:** [Chapter 1: The Causal Boundary](../../books/vol4/01_boundary/01_boundary.qmd) & [Chapter 2: The Physical Body](../../books/vol4/02_body/02_body.qmd)
**Target Competencies:** `[ ] A1 Plant Mechanics & Safe Envelope`, `[ ] D1 Hardware Authority Routing`
**Milestone Alignment:** Contributes to [Milestone 1](syllabus.md#sec-milestones) (End of Week 3)

---

### 1. The Physical Question
Where does computational authority end and physical delegation begin? If a host computer or neural network issues an erroneous command, what hardware mechanisms guarantee that the machine fails safely without causing physical damage or uncommanded motion?

---

### 2. Hardware Setup
1. **Actuator Station:** Seeed Studio SO-101 6-DoF arm mounted securely to the tabletop acrylic baseplate.
2. **Controller Board:** Arduino UNO Q ("Unikue") powered via USB-C PD (45W).
3. **Bus Interface:** Half-duplex TTL serial bus connecting the STM32U585 MCU (TX/RX pin pair) to the first Feetech STS3215 smart servo.
4. **Isolated Power:** 7.4V–12V DC regulated motor power supply wired through an accessible physical emergency-stop toggle switch.
5. **Initial State:** Motor power switch **OFF (Disarmed)**. Arm in resting, folded configuration.

---

### 3. Step-by-Step Bench Protocol

#### Step 1: Physical Authority & Power Audit
1. Inspect the station wiring. Trace every cable from the power strip to the arm.
2. Verify that the **only** electrical connection to the STS3215 servo bus originates from the STM32 microcontroller header.
3. Confirm that no direct USB-to-UART bridge connects the host workstation to the servo bus.

#### Step 2: Servo Enumeration & Firmware Bring-Up
1. Flash the staff-provided baseline motion firmware to the STM32 MCU via the Arduino IDE / App Lab CLI.
2. Open the STM32 serial monitor at 115200 baud. Energize motor power via the physical toggle switch.
3. Run the enumeration command:
   ```bash
   uno-q-cli bus scan --baud 1000000
   ```
4. Verify that all 6 STS3215 servos respond with their programmed IDs (`1` to `6`), firmware versions, and current voltages.

#### Step 3: Zero-Homing & Mechanical Envelope Characterization
1. Use the physical calibration jig to align each joint to its zero-angle mechanical detent.
2. Record the raw 12-bit optical/magnetic encoder counts for each joint.
3. Slowly articulate each joint by hand across its full physical travel range. Measure and record:
   * $\theta_{i,\min}$ and $\theta_{i,\max}$ in mechanical degrees.
   * Hard physical stop angles vs. allowable software travel bounds.
4. Flash the calibrated soft limits into the STM32 non-volatile configuration memory.

---

### 4. The Disturbance & Failure Test
1. **Power-Off Drop Test:** Command the arm to a stable elevated test pose (Joint 2 @ $45^\circ$, Joint 3 @ $45^\circ$). While elevated, depress the physical emergency-stop switch.
   * *Observation:* Record the mechanical drop trajectory as gravity pulls the unpowered links down. Verify that no mechanical binding or violent snapping occurs.
2. **Re-Power Surge Audit:** With the arm now resting in an arbitrary fallen position, flip the motor power switch back ON.
   * *Pass Criteria:* The arm must remain completely limp and passive. The servos must **never** violently jerk, snap to zero, or execute pre-stored moves upon power restoration until an explicit arming handshake is sent from the console.

---

### 5. Multi-Tap Telemetry Trace
In this lab, establish the fundamental telemetry schema that will log the four action taps across the entire semester:
* $a_{\text{req}}$: Target joint vector requested by software.
* $a_{\text{map}}$: Mapped target joint angles bounded by calibration limits.
* $a_{\text{enf}}$: Command permitted by the STM32 MCU.
* $a_{\text{meas}}$: Actual position feedback read back from the STS3215 encoder registers.

Log an elevation move in CSV format and verify that $a_{\text{meas}}$ converges to $a_{\text{enf}}$ within $\pm 0.5^\circ$.

---

### 6. Common Hardware Pitfalls & Debugging
* ⚠️ **TTL Half-Duplex Contention:** The STS3215 bus uses a single bi-directional data line. If the STM32 driver does not disable its transmitter before reading, bus collisions will corrupt packets. Ensure the direction-control pin timing is exact.
* ⚠️ **Voltage Sag under Multi-Servo Stall:** If multiple servos draw stall current simultaneously (>1.5A each), poorly regulated supplies will dip, causing the STM32 or servos to brown-out reset. Ensure logic power is completely decoupled from motor power.

---

### 7. Bench Sign-Off Criteria (The Exit Check)
To receive credit for Lab 1, demonstrate the following live to the instructor:
1. [ ] **Authority Route Proof:** Show the physical wiring diagram and prove that disconnecting the STM32 stops all motor communication.
2. [ ] **Measured Operating Envelope Table:** Present the measured travel limits ($\theta_{\min}, \theta_{\max}$) for all 6 joints and the calibrated resting rest pose.
3. [ ] **Power-Off & Re-Arm Trace:** Demonstrate depressing the E-stop switch during motion, observing a safe passive drop, restoring power, and proving zero uncommanded motion occurs.
*Staff signs off `[ ] A1` and `[ ] D1` on the team's [Competency Card](student-competencies.md).*

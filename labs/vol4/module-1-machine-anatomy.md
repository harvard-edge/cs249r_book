# Part I / Module 1: The Machine Anatomy

**Schedule:** Weeks 1–3
**Milestone Deliverable:** [Milestone 1](syllabus.md#sec-milestones) (End of Week 3) — *Station Charter, Calibrated Operating Envelope & Authority Route*
**Required Textbook Reading:**
- [Chapter 1: The Causal Boundary](../../books/vol4/01_boundary/01_boundary.qmd)
- [Chapter 2: The Physical Body](../../books/vol4/02_body/02_body.qmd)
- [Chapter 3: The Cognitive Brain](../../books/vol4/03_brain/03_brain.qmd)
- [Chapter 4: The Nervous System](../../books/vol4/04_nervous/04_nervous.qmd)
**Associated Studio Labs:** [Lab 1: Causal Boundary & Servo Bus Bring-Up](lab-01-boundary.md) & [Lab 2: Multi-Modal Sensing & The Inter-Core Bridge](lab-02-body-and-sensors.md)
**Target Competency Card Items:** `[ ] A1`, `[ ] A2`, `[ ] A3`, `[ ] D1`

---

## 1. Executive Overview

Before deploying learned policies, robotics engineers must answer two fundamental systems questions:
1. **Where does computational authority end and physical delegation begin?**
2. **How accurately does commanded intent match realized physical state?**

In this module, teams inspect the dual-brain architecture of the **Arduino UNO Q** paired with the **Seeed SO-101 6-DoF arm**, audit physical power and communication routes, calibrate joint telemetry and camera extrinsics, measure loop latencies, and establish their semester project charter.

---

## 2. Hardware Architecture & Authority Route

The physical station enforces an asymmetry between high-level computation and low-level actuation:

```
[ Camera (USB) ] ──▶ [ Qualcomm QRB2210 (Debian Linux) ]
                            │
                            ▼ (Arduino Bridge RPC over UART/SPI)
                     [ STM32U585 Real-Time MCU ] ◄── [ Hardware Watchdog & Power Rail ]
                            │
                            ▼ (Half-Duplex TTL Serial Bus @ 1 Mbps)
                     [ 6× Feetech STS3215 Bus Servos ]
```

### Key Engineering Rules:
- **Zero Host Bypass:** The SO-101 servos must be commanded **exclusively** through the STM32 microcontroller. The host workstation or Qualcomm USB bus must never connect directly to the servo bus via a USB-to-UART adapter.
- **Physical Power Isolation:** The motor supply (7.4V/5A DC) operates on an independent rail with a switched toggle, sharing a common star ground with the Arduino UNO Q. Cutting motor power de-energizes the servos instantly without resetting the Qualcomm Linux MPU or STM32 MCU logic power.

---

## 3. Laboratory Investigations

### Lab 1: The Causal Boundary & Servo Bus Bring-Up (Weeks 1–2)
*Full Brief:* [lab-01-boundary.md](lab-01-boundary.md)
*Textbook Reading:* Chapters 1 & 2
*Competencies Checked:* `[ ] A1 Plant Mechanics & Safe Envelope`, `[ ] D1 Hardware Authority Routing`
1. **Trace the Actuation Chain:** Map each physical layer: Camera $\to$ Qualcomm QRB2210 $\to$ Arduino Bridge $\to$ STM32U585 $\to$ STS3215 Servos. Verify zero host bypass.
2. **Safe-State Transitions:** Measure servo bus behavior during board boot, soft reset, and sudden motor power loss. Confirm that uncommanded motion cannot occur upon power restoration.
3. **The Physical AI Scope Test:** Formulate a team counterexample: identify a system that uses machine learning in robotics but fails the book's 3-part scope test.

### Lab 2: Multi-Modal Sensing & The Inter-Core Bridge (Week 3)
*Full Brief:* [lab-02-body-and-sensors.md](lab-02-body-and-sensors.md)
*Textbook Reading:* Chapters 3 & 4
*Competencies Checked:* `[ ] A2 Multi-Modal Sensing & Calibration`, `[ ] A3 Feedback Timing & Latency`
1. **Multi-Modal Calibration:** Zero revolute joints; calibrate the overhead USB webcam using a checkerboard/ArUco fixture to compute $T_{\text{cam}}^{\text{base}}$.
2. **Commanded vs. Settled Error:** Issue 10 repeated slow movements across the reachable workspace. Measure commanded joint vector $\mathbf{q}_{\text{cmd}}$, encoder readback $\mathbf{q}_{\text{meas}}$, and camera-observed tool position $\mathbf{x}_{\text{cam}}$. Quantify backlash and settling time.
3. **Inter-Core Bridge & Latency:** Establish bidirectional RPC communication between Qualcomm Linux and STM32. Log synchronized observation tuples $(I_t, q_t)$ and measure the end-to-end loop latency distribution.

---

## 4. Milestone 1 Deliverable: Station Charter & Calibrated Envelope

By the end of Week 3, each team submits their **Milestone 1 Packet**:
1. **Power & Command Schematic:** Annotated system diagram showing power rails, logic voltages, physical cutoff placement, and communication buses.
2. **Characterized Envelope Card:**
   - Bounded 3D workspace limits ($X, Y, Z$ coordinates in mm) safe for autonomous motion.
   - Maximum allowable joint velocities ($\text{deg/s}$) and payload capacity ($\text{g}$).
   - Measured repeatability ($3\sigma$ error in mm) and settling delay ($t_{\text{settle}}$ in ms).
3. **Project Task Charter:**
   - 1-page description of the team's chosen semester manipulation task.
   - Definition of the **independent task outcome metric** (how success is physically verified beyond joint encoder readback).
4. **Competency Sign-Off:** Verified sign-off for **A1, A2, A3, and D1** on the team's [Competency Card](student-competencies.md).

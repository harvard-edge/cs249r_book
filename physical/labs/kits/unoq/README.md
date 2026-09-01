# Arduino Uno Q Dual-Brain Physical AI Kit: Postdoc Investigation & Prototyping Blueprint

---

## 1. Executive Context & Mission

The goal of the Physical AI curriculum is to provide university classrooms with **hands-on, hardware-grounded laboratory experiences** that illustrate the 17 chapters of the textbook without requiring $50,000 industrial manipulators.

The **Arduino Uno Q** serves as the canonical low-cost reference hardware platform for this curriculum due to its native **Dual-Brain Architecture**:
* **Host Application MPU (Linux Tier):** Runs high-level deliberation, sensory ingestion, coordinate transforms, and neural action chunk proposals ($1\text{--}20\text{ Hz}$).
* **Real-Time MCU (Bare-Metal Tier):** Executes cycle-deterministic $1000\text{ Hz}$ reflex loops, PWM motor timers, Control Barrier Function (CBF) safety projections, and hardware watchdog interlocks.
* **Shared Inter-Core Boundary:** Memory-mapped mailbox registers / shared SRAM with lock-free seqlocks.

```
   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
   │                          ARDUINO UNO Q DUAL-BRAIN ARCHITECTURE                              │
   └─────────────────────────────────────────────────────────────────────────────────────────────┘

   [ SENSORS: Camera / IMU / ToF ]
                 │
                 ▼
   ┌─────────────────────────────────────────────┐
   │ HOST APPLICATION PROCESSOR (MPU - Linux)    │
   │ • High-level perception & feature encoding  │ ◄── Low Cadence (1–20 Hz)
   │ • Neural action chunk proposals (Tiny-ACT)  │     Stochastic / High Capacity
   │ • Logging 4-stream action telemetry tuples  │
   └──────────────────────┬──────────────────────┘
                          │ Inter-Core Shared SRAM Mailbox (Lock-Free Seqlock)
                          ▼
   ┌─────────────────────────────────────────────┐
   │ REAL-TIME MICROCONTROLLER (MCU - Bare-Metal)│
   │ • 1000 Hz Control Barrier Function (CBF-QP) │ ◄── Hard Real-Time (1 kHz)
   │ • PWM motor timer generation & dead-time    │     Deterministic / Zero Jitter
   │ • Hardware watchdog & Safe Torque Off (STO) │
   └──────────────────────┬──────────────────────┘
                          │
                          ▼
   [ ACTUATORS: Smart Serial Servos / DC Motors / Power H-Bridges ]
```

### The Postdoc Investigation Mandate
This document provides three concrete hardware kit proposals. The postdoc's objective is to evaluate each proposal against:
1. **Physical Pedagogical Coverage:** Which textbook chapters does it illuminate?
2. **Compute & Memory Feasibility:** What model architectures actually fit in memory and execute within latency bounds on the Uno Q?
3. **Classroom Feasibility & Cost:** Can this kit be sourced and assembled for under **$100–$150 per student station** with high mechanical reliability?

---

## 2. Three Candidate Kit Proposals

```
   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
   │                               THREE CANDIDATE KIT PROPOSALS                                 │
   └─────────────────────────────────────────────────────────────────────────────────────────────┘

   PROPOSAL A: The Desktop Manipulator (4–6 DoF Micro-Arm / Serial Bus Servos)
   • Focus: Contact force, action chunking, bilateral teleoperation, and obstacle avoidance.

   PROPOSAL B: The Desktop Self-Balancing Rover (Inverted Pendulum / Micro-AGV)
   • Focus: Momentum, dynamic stopping distance, 1 kHz balance reflexes, and spatial freshness.

   PROPOSAL C: The 1D Dynamical Contact & Thermal Testbed (Motor Dyno with Variable Brake)
   • Focus: Reflected inertia (N²J), stator heat (I²R), sim-to-real friction, and fault injection.
```

---

### PROPOSAL A: The Desktop Manipulator (4–6 DoF Serial Bus Arm)

```
                       [ Overhead / Wrist Camera ]
                                   │
                                   ▼
      ┌─────────────────────────────────────────────────────────┐
      │ Arduino Uno Q Dual-Brain Board                          │
      │ • MPU: Visual Feature Ingestion + Action Chunking (5 Hz)│
      │ • MCU: 1 kHz Trajectory Spline & CBF Joint Limits       │
      └────────────────────────────┬────────────────────────────┘
                                   │ Half-Duplex UART Bus (1 Mbps)
                                   ▼
      [ 4–6x Smart Serial Bus Servos (e.g. Feetech / Waveshare STS3215) ]
                                   │
                                   ▼
      [ 3D-Printed Kinematic Linkage (SO-100 / Micro-ALOHA style) ]
```

#### 1. Hardware Architecture & BOM Target
* **Actuators:** 4 to 6 serial bus servos with position/current/temperature feedback over a half-duplex UART bus ($1\text{ Mbps}$).
* **Sensors:** Single USB/MIPI color camera ($640\times 480$ @ $30\text{ fps}$) + joint current feedback.
* **Teleoperation Interface:** Passive 3D-printed leader arm with potentiometers/encoders for bilateral demonstration logging (GELLO / leader-follower style).
* **Target BOM Cost:** $\approx \$90\text{--}\$130$.

#### 2. Core Textbook Principles Taught
* **Ch 01 & 04 (Dual-Brain & Multi-Rate):** MPU generates $5\text{ Hz}$ trajectory proposals; MCU executes $1000\text{ Hz}$ joint velocity setpoints.
* **Ch 05 (Endogenous Data):** Logging the Four-Stream Action Tap Tuple $(o_t, a_{\text{prop}}, a_{\text{act}}, i_t)$ via leader-follower teleoperation.
* **Ch 06 & 11 (Policy Synthesis & Planning):** Running Action Chunking ($H=16$) with $C^2$ quintic spline interpolation across chunk seams.
* **Ch 12 (Enforcement):** Bare-metal Control Barrier Function (CBF-QP) preventing the gripper from colliding with the tabletop or exceeding joint limits.
* **Ch 14 (Intervention):** Shared authority and bumpless takeover when human moves the arm during autonomous rollout.

#### 3. Neural Model Architecture & Memory Feasibility
* **Visual Encoder:** MobileNetV3-Small or quantized ResNet-18 (spatial softmax pooling $\to 32$-dim latent vector $\mathbf{z}$).
* **Action Policy:** Tiny-ACT (2-layer Transformer decoder, $d_{\text{model}}=128$, $H=16$ chunk horizon) or Quantized 1D-Diffusion Policy (3 denoising steps via Flow Matching).
* **Quantization & Runtime:** INT8 quantization via TFLite-Micro or ONNX Runtime. Target model footprint: $< 15\text{ MB RAM}$.

#### 4. Postdoc Investigation Tasks
* [ ] Benchmark INT8 MobileNetV3 + Tiny-ACT inference latency on the Uno Q Linux MPU. What is the maximum sustainable chunk emission frequency ($f_{\text{chunk}}$)?
* [ ] Measure serial bus servo command jitter over the half-duplex UART at $1\text{ Mbps}$. Can the MCU sustain $100\text{ Hz}$ servo setpoints without packet drops?
* [ ] Test mechanical repeatability and gear backlash of low-cost serial bus servos after 10 hours of continuous teleoperation.

---

### PROPOSAL B: The Desktop Self-Balancing Rover (Inverted Pendulum / Micro-AMR)

```
                            [ ToF Depth Sensor ]
                                     │
                                     ▼
        ┌───────────────────────────────────────────────────────────┐
        │ Arduino Uno Q Dual-Brain Board                            │
        │ • MPU: High-level Goal Navigation & Spatial Memory (10 Hz)│
        │ • MCU: 1 kHz Inverted Pendulum LQR / CBF Balance Shield   │
        └─────────────────────────────┬─────────────────────────────┘
                                      │ PWM / Dual H-Bridge
                                      ▼
        [ 2x Geared DC Motors with Quadrature Encoders (1000 CPR) ]
                                      │
                                      ▼
        [ 2-Wheeled Inverted Pendulum Chassis with 6-DoF IMU ]
```

#### 1. Hardware Architecture & BOM Target
* **Actuators:** 2x high-speed geared DC motors with magnetic quadrature encoders ($1000\text{ CPR}$) driven by a dual MOSFET H-bridge.
* **Sensors:** Onboard 6-DoF IMU (accelerometer + gyroscope @ $1\text{ kHz}$) + forward-facing VL53L5CX Time-of-Flight (ToF) multi-zone depth sensor.
* **Chassis:** 2-wheeled self-balancing inverted pendulum chassis (desktop footprint: $15\text{ cm} \times 10\text{ cm}$).
* **Target BOM Cost:** $\approx \$60\text{--}\$85$.

#### 2. Core Textbook Principles Taught
* **Ch 02 (The Body):** Dynamic stopping distance quadratic ($d_{\text{stop}} = v t_{\text{lag}} + \frac{v^2}{2 a_{\max}}$), motor electrical time constants, and power rail droop ($\Delta V = L \frac{dI}{dt} + IR$) during sudden acceleration.
* **Ch 04 (The Nervous System):** Hard real-time balance loop requiring $< 1\text{ ms}$ latency jitter; demonstrating failure when balance loop is moved to non-real-time Linux.
* **Ch 08 & 09 (Perception & Memory):** 8x8 ToF depth array ingestion, coordinate transform trees (`odom` $\to$ `base_link`), and spatial belief decay under dynamic motion.
* **Ch 10 (Intent):** Expiring Intent Leases ($\mathcal{L}_{\text{intent}}$)—rover safely stops balancing in place if the high-level intent lease expires.
* **Ch 15 (Verification):** Hardware fault injection (injecting synthetic IMU gyro bias, frame dropouts, and motor power brownouts).

#### 3. Neural Model Architecture & Memory Feasibility
* **Policy Architecture:** Tiny State-Space MLP Policy (Ingests $[x, \dot{x}, \theta, \dot{\theta}, d_{\text{ToF}}] \in \mathbb{R}^8 \to$ predicts target velocity $v_{\text{cmd}}$ @ $20\text{ Hz}$).
* **Footprint:** Extremely small ($< 500\text{ KB}$), highly robust, runs natively on MPU or directly on MCU DSP registers.
* **Optional Vision Module:** Tiny 1-channel grayscale camera for visual lane/obstacle following.

#### 4. Postdoc Investigation Tasks
* [ ] Implement 1 kHz complementary filter / EKF on the MCU core and measure worst-case execution time (WCET).
* [ ] Measure voltage droop on the $5\text{V}/3.3\text{V}$ logic rails when both motors reverse direction under full torque. Is a bulk decoupling capacitor or split power rail required to prevent MCU brownout resets?
* [ ] Evaluate student safety: does the robot safely drop to its kickstand when the MCU watchdog trips?

---

### PROPOSAL C: The 1D Dynamical Contact & Thermal Testbed (Motor Dyno)

```
        ┌───────────────────────────────────────────────────────────┐
        │ Arduino Uno Q Dual-Brain Board                            │
        │ • MPU: Learned Dynamics Estimator & Thermal Model (10 Hz) │
        │ • MCU: 1 kHz Current Loop, Hardware Trip Zone, & CBF      │
        └─────────────────────────────┬─────────────────────────────┘
                                      │ PWM Gate Signals
                                      ▼
        [ High-Torque Brushless / DC Motor + Variable Magnetic Friction Brake ]
                                      │
                                      ▼
        [ Sensors: Thermocouple / RTD on Stator + Reaction Torque Cell + Encoder ]
```

#### 1. Hardware Architecture & BOM Target
* **Actuators:** 1x BLDC motor with Field-Oriented Control (FOC) driver or brushed DC motor + 1x adjustable magnetic/hysteresis friction brake.
* **Sensors:** Thermocouple/RTD directly mounted to stator copper windings + non-contact optical encoder + inline torque load cell.
* **Chassis:** Enclosed desktop dynamometer baseplate with acrylic safety shield.
* **Target BOM Cost:** $\approx \$75\text{--}\$110$.

#### 2. Core Textbook Principles Taught
* **Ch 02 (The Body):** Direct measurement of the 5 Physical Budgets:
  * Stator Joulean heating: $\Delta T = \frac{I^2 R \Delta t}{C_{\text{th}}}$.
  * Reflected rotor inertia ($N^2 J$) under high gear reduction vs direct drive.
  * Bus resistance $R_{\text{bus}}$ and voltage sag during peak stall current.
* **Ch 04 (The Nervous System):** Hardware trip-zone register configuration (`EPWM_TZFRC`) tripping motor drive on thermal/overcurrent limit.
* **Ch 06 (Training):** The Sim-to-Real Reality Gap: training a neural dynamics simulator on nominal friction curves and observing model collapse under non-linear Stribeck friction hysteresis and thermal resistance drift ($\alpha_{\text{Cu}} = 0.00393\text{ K}^{-1}$).
* **Ch 15 (Verification):** Executing automated hardware fault injection test manifests (thermal stress cycles, step-torque shocks, encoder phase wire inversion).

#### 3. Neural Model Architecture & Memory Feasibility
* **Dynamics Estimator:** Physics-Informed Neural Network (PINN) or Autoregressive LSTM/GRU predicting future motor temperature $T(t+k)$ and torque response under unmodeled friction.
* **Footprint:** $< 2\text{ MB}$, lightweight execution on Linux MPU.

#### 4. Postdoc Investigation Tasks
* [ ] Validate stator thermal rise curves against the theoretical $\Delta T = \frac{I^2 R \Delta t}{C_{\text{th}}}$ equation under continuous stall current.
* [ ] Verify that the MCU hardware trip zone safely cuts PWM gate drive within $< 10\,\mu\text{s}$ of an overcurrent event without waiting for Linux MPU intervention.
* [ ] Design a simple, foolproof student lab script where students measure the Stribeck friction curve and observe where classical linear models fail.

---

## 3. Comparison Matrix Across Proposals

| Feature / Metric | Proposal A: Manipulator Arm | Proposal B: Balancing Rover | Proposal C: Dynamical Dyno |
| :--- | :--- | :--- | :--- |
| **BOM Cost Target** | $\$90\text{--}\$130$ | $\$60\text{--}\$85$ | $\$75\text{--}\$110$ |
| **Textbook Chapter Coverage** | **Ch 1, 3, 4, 5, 6, 11, 12, 14** | **Ch 1, 2, 4, 8, 9, 10, 15** | **Ch 1, 2, 4, 6, 12, 13, 15** |
| **Primary Physical Archetype** | **Manipulator (Contact & Inertia)** | **Mobile (Momentum & Freshness)** | **Process/Energy (Heat & Power)** |
| **Vision Model Feasibility** | MobileNetV3 / Tiny-ACT (INT8) | Optional 8x8 ToF / Low-res Camera | None required (Pure Physical Telemetry) |
| **Mechanical Fragility** | Moderate (Servo gears can strip) | Low (Durable chassis & kickstand) | Very Low (Enclosed metal/acrylic base) |
| **Student Excitement Factor** | **Very High** (Pick-and-place & teleop) | **High** (Dynamic self-balancing) | **High for Systems/Hardware** |

---

## 4. Postdoc Investigation Workplan & Next Steps

```
   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐
   │                               POSTDOC INVESTIGATION PHASES                                  │
   └─────────────────────────────────────────────────────────────────────────────────────────────┘

   PHASE 1: Silicon & Inter-Core Benchmarking (Weeks 1–2)
   • Benchmark Uno Q MPU neural inference (TFLite-Micro / ONNX Runtime INT8).
   • Benchmark MPU-to-MCU shared SRAM mailbox throughput and round-trip latency.
   • Measure MCU 1 kHz timer jitter and PWM generation stability.

   PHASE 2: Breadboard Prototyping & Actuator Feasibility (Weeks 3–4)
   • Assemble 1 prototype of Proposal A (Serial Bus Servos) and Proposal B (DC Balance Rover).
   • Evaluate serial bus half-duplex communication latency and motor electrical noise isolation.

   PHASE 3: Lab Curriculum & Safety Shield Validation (Weeks 5–6)
   • Implement the bare-metal CBF-QP safety shield on the MCU for Proposal A/B.
   • Write the 3 standard laboratory contract manifests (Dataset Schema, Trajectory Contract, Release Manifest).
   • Report back with final recommendation for the canonical course hardware kit.
```

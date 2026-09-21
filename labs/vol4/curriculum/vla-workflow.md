# Modern Robot Control Workflow: From Analytical Kinematics to Edge VLAs

**Status:** The primary pedagogical spine for the Volume IV Physical AI Studio, proposed by Dr. Andrea Mattia Garavagno and Professor Vijay Janapa Reddi.

---

## The Vision: Controlling Robots in a Modern Fashion

Modern robotics is undergoing a paradigm shift from hand-crafted analytical kinematics and trajectory generators to **Vision-Language-Action (VLA) foundation models** and end-to-end imitation learning. However, students cannot appreciate what modern learned policies solve—or where they fail—without first grounding themselves in physical kinematics, simulation modeling, and microcontroller execution.

The studio follows an intuitive, hands-on **7-Step Modern Control Workflow** that takes students from first-principles analytical mathematics to edge-deployed VLAs on physical hardware:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      THE 7-STEP MODERN CONTROL WORKFLOW                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. Paper & Pen IK         ──▶ Compute analytical kinematics by hand              │
│ 2. Digital Twin (Sim)     ──▶ Model the Hero robot & test IK in modern simulation│
│ 3. Hardware MCU Baseline  ──▶ Deploy IK equations on STM32 MCU via UNO Q CLI    │
│ 4. VLAs in Simulation     ──▶ Control the Hero robot in sim with a pre-trained VLA│
│ 5. Edge VLA & Sim2Real    ──▶ Deploy VLA on Dragonwing via RPC ➔ Witness the GAP! │
│ 6. Real-World Adaptation  ──▶ Collect teleop data with LeRobot & fine-tune VLA   │
│ 7. Architecture Transfer  ──▶ Generalize pipeline to novel robot embodiments     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## The 7-Step Sequence in Detail

### Step 1: Analytical Kinematics (Paper & Pen)
* **What Students Do:** Derive the forward and inverse kinematics (DH parameters, geometric/analytical IK) for the "Hero" robot arm on paper.
* **Concepts:** Coordinate frames, transform matrices, joint limits, singularities, and reachable workspace.
* **Outcome:** Closed-form mathematical equations mapping $(x, y, z, \text{roll}, \text{pitch}, \text{yaw})$ end-effector targets to joint angles $(\theta_1, \dots, \theta_6)$.

### Step 2: Digital Twin in Simulation
* **What Students Do:** Instantiate a URDF/MJCF digital twin of the Hero robot inside a modern physics simulator (MuJoCo, Isaac Gym / LeIsaac). Implement the paper-and-pen IK equations in Python/C++ to drive the simulated arm to interactive target positions.
* **Concepts:** Rigid-body dynamics, joint damping, collision meshes, forward integration.
* **Outcome:** Interactive digital twin responding deterministically to Cartesian target commands.

### Step 3: Hardware Baseline & Real-World MCU Control
* **What Students Do:** Move from simulation to the physical station. Implement the IK equations on the **Arduino UNO Q STM32 microcontroller (MCU)**. Command the physical robot arm via the **Arduino UNO Q CLI**.
* **Concepts:** Microcontroller execution, half-duplex serial bus (Feetech STS3215), gear backlash, commanded vs. settled error, physical safety envelopes.
* **Outcome:** A working, deterministic, nonlearned baseline that moves the physical arm in the real world.

### Step 4: Introducing VLAs in Simulation
* **What Students Do:** Shift from classical Cartesian IK to modern Vision-Language-Action (VLA) policies. Staff provide training recipes, multimodal demonstration datasets, and a pre-trained VLA checkpoint acting in the digital twin environment.
* **Concepts:** Multimodal tokenization, cross-attention between vision/text and action chunks, imitation learning.
* **Outcome:** Students command the simulated robot using natural language and visual observations (e.g., *"reach toward the red cube"*).

### Step 5: Edge Deployment on Dragonwing & The Sim-to-Real Shock
* **What Students Do:** Export the pre-trained VLA policy to the **Arduino UNO Q Qualcomm Dragonwing (QRB2210 Linux MPU)**. Connect the Dragonwing policy to the STM32 MCU via **RPC / Arduino Bridge**. Run the simulation-trained VLA on the physical hero robot via the Arduino UNO Q CLI.
* **The Revelation:** Students witness the **Sim-to-Real gap** firsthand: lighting shifts, camera distortion, table friction, cable drag, and actuator lag cause the simulation-trained policy to miss, drift, or stall.
* **Outcome:** Measurable quantification of the Sim2Real performance drop.

### Step 6: Bridging the Gap with Real-World Adaptation
* **What Students Do:** Collect a small, focused real-world demonstration dataset on the physical hardware using Hugging Face LeRobot teleoperation scripts. Fine-tune the pre-trained VLA on this real-world dataset.
* **Concepts:** Data-centric AI, behavioral cloning, distribution shift, fine-tuning recipes, real-time closed-loop recovery.
* **Outcome:** The fine-tuned VLA successfully executes the physical task on hardware, compensating for real-world friction and camera variations.

### Step 7: Cross-Architecture Generalization & Capstone
* **What Students Do:** Test how this analytical $\to$ simulation $\to$ edge VLA $\to$ fine-tuning pipeline generalizes across varied task conditions, perturbations, and different robot kinematics/architectures (e.g., mobile manipulators, varied payloads, novel compliant fixtures).
* **Outcome:** Final capstone trials, adversarial fault testing, and oral defense.

---

## Alignment with Course Milestones & Weekly Rhythm

Over a 14-week semester (provisional 3 hours of supervised studio lab per week = 42 contact lab hours), the 7 steps map into pairs of weeks synchronized with the 4 course milestones:

| Studio Weeks | Workflow Step | Module Mapping | Milestone Deliverable |
|:---:|:---|:---|:---|
| **Weeks 1–2** | **Step 1:** Paper & Pen IK<br>**Step 2:** Digital Twin Simulation | **Module 1:** Causal Boundary & Plant | **Milestone 1 (Week 2):** Project Charter, IK Validation & Envelope |
| **Weeks 3–4** | **Step 3:** Physical Hardware MCU Baseline (CLI) | **Module 2:** Closing the Autonomous Loop | *Formative Bench Review:* Baseline repeatability & MCU authority |
| **Weeks 5–6** | **Step 4:** VLAs in Simulation<br>**Step 5:** Edge VLA & Sim2Real Gap | **Module 2:** Closing the Autonomous Loop | **Milestone 2 (Week 6):** Witnessed Edge VLA Closed Loop on UNO Q |
| **Weeks 7–8** | **Step 6 (Part A):** Real Data Collection (LeRobot) | **Module 3:** Embodied Policy & Governors | *Formative Bench Review:* Dataset card & alignment audit |
| **Weeks 9–10** | **Step 6 (Part B):** VLA Fine-Tuning & Adaptation | **Module 3:** Embodied Policy & Governors | **Milestone 3 (Week 10):** Authority Under Fault & Adapted VLA Demo |
| **Weeks 11–12** | **Step 7 (Part A):** Disturbance & Generalization | **Module 3:** Embodied Policy & Governors | *Formative Bench Review:* Incomplete motion & recovery traces |
| **Weeks 13–14** | **Step 7 (Part B):** Capstone Defense & Release | **Module 4:** Capstone Defense | **Milestone 4 (Week 14):** System Release Case & Oral Defense |

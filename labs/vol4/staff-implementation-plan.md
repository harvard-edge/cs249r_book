# Volume IV Physical AI Studio: Staff Implementation Master Plan
### Comprehensive 13-Week Execution Roadmap (September 21 – December 18, 2026)

**Audience:** Postdoctoral Researcher (Andrea, ETH Zurich) & Course Instruction Team
**Course Lead:** Prof. Vijay Janapa Reddi
**Target Semester:** Spring 2027 Studio Launch
**Target Hardware:** Arduino UNO Q (Qualcomm QRB2210 Linux + STM32U585 MCU) · Seeed Studio SO-101 6-DoF Follower Arm · Hugging Face LeRobot · SmolVLA & ACT
**Core Reference Architecture:** [Master Landing Page](README.md) · [Course Syllabus](syllabus.md) · [Pre-Flight Guide](feasibility-plan.md) · [12-Competency Matrix](student-competencies.md)

---

## 1. Executive Summary & Operational Mandate

This master plan governs the 13-week physical implementation, hardware bring-up, and pedagogical qualification period between **Monday, September 21, 2026**, and **Friday, December 18, 2026**.

The primary objective is for **Andrea (Postdoctoral Researcher, ETH Zurich)** to act as **"Student Zero"**:
1. Personally build, wire, verify, and qualify one complete **Golden Reference Station**.
2. Pass the **Four Technical Go/No-Go Gate Tests (Gates A–D)**.
3. Execute and qualify all **8 Hands-On Student Labs** end-to-end, producing verified golden artifacts, starter scripts, and datasets.
4. Freeze the **Golden Qualcomm Linux Image** and **STM32 Firmware Binary**.
5. Conduct an independent replication audit with a secondary researcher/TA to ensure every station can be duplicated without staff intervention before spring classes start.

---

## 2. Master 13-Week Timeline Overview

```
SEPTEMBER 2026                  OCTOBER 2026                    NOVEMBER 2026                   DECEMBER 2026
W01       W02       W03       W04       W05       W06       W07       W08       W09       W10       W11       W12       W13
[--- SPRINT 1: HARDWARE ---]  [--- SPRINT 2: DATA & MODEL ---]  [--- SPRINT 3: EDGE LOOP ---]   [-- SPRINT 4: GOV --]   [-- SPRINT 5: PACK --]
Gate A    Gate B    Gate C    Lab 1-2   Lab 3     Lab 4     Gate D    Lab 6     Mid-Term  Lab 7     Lab 8     Capstone  Image Freeze &
Unbox     1-Joint   6-DoF     Calibrate Teleop    ACT Train Untether  Chunking  Stress    Safety    Watchdog  Rehearsal Replication Test
Assembly  Veto      Governor  Rig       Dataset   ONNX INT8 Reach     Perturb   Audit     Governor  Cutoff    20-Trials Ready for Class!
```

---

## 3. Sprint-by-Sprint Weekly Implementation Breakdown

### Sprint 1: Mechanical Bring-Up & Gate Tests A–C (Weeks 1–3)
**Focus:** Mechanical assembly, electrical safety isolation, and establishing the governed servo bus path.

#### Week 1 (Sep 21 – Sep 25, 2026): BOM Verification, Bench Rigging & Gate A
* **Primary Objective:** Assemble the physical arm, establish power safety, and achieve native teleoperation.
* **Tasks:**
  - [ ] Inventory delivered hardware: Seeed SO-101 Pro assembled kit, Arduino UNO Q (4GB), 45W USB-PD adapter, USB hub, Logitech C270 camera.
  - [ ] Clamp arm baseplate rigidly to workbench using heavy-duty C-clamps.
  - [ ] Wire external 7.4V/5A DC motor power supply through physical latching mushroom E-Stop switch to STS3215 servo power rail.
  - [ ] Verify E-Stop cuts motor power in $< 5\text{ ms}$ while leaving logic power untouched.
  - [ ] Connect SO-101 arm to workstation via USB BusLinker; install Hugging Face LeRobot (`pip install lerobot`).
  - [ ] **Pass Gate A:** Run joint calibration and execute 60-second teleoperation replay (`lerobot-replay`).
* **Weekly Deliverable:** Video proof of Gate A teleoperation + E-Stop power cutoff oscilloscope/multimeter trace.

#### Week 2 (Sep 28 – Oct 2, 2026): Inter-Core Bridge Wiring & Gate B (1-Joint Interceptor)
* **Primary Objective:** Route servo communication through the STM32 and prove real-time command veto.
* **Tasks:**
  - [ ] Disconnect Joint 1 (Base Yaw) from USB BusLinker; connect single-wire half-duplex UART line to STM32U585 USART via level-shifter.
  - [ ] Flash baseline interceptor firmware to STM32 using Arduino App Lab.
  - [ ] Write Python bridge script on Qualcomm Linux transmitting target commands across `/dev/ttyRPMSG`.
  - [ ] **Pass Gate B:** Verify STM32 permits valid commands ($10^\circ$ at $20^\circ/\text{s}$), clamps over-speed commands ($> 45^\circ/\text{s}$), and refuses out-of-range targets ($> 180^\circ$).
* **Weekly Deliverable:** Telemetry log showing MCU clamping and rejection of out-of-bounds joint commands.

#### Week 3 (Oct 5 – Oct 9, 2026): Full 6-DoF Governed Arm & Gate C
* **Primary Objective:** Scale the governed bus to all 6 joints and achieve full LeRobot teleoperation under MCU control.
* **Tasks:**
  - [ ] Daisy-chain all 6 STS3215 servos into the STM32 USART bus line.
  - [ ] Implement the 60-line Python `UnoQMotorsBus` custom LeRobot adapter.
  - [ ] Program Cartesian forward kinematics check in STM32 firmware; implement table collision geofence ($z_{\text{tool}} \ge 15\text{ mm}$).
  - [ ] **Pass Gate C:** Run full 6-DoF LeRobot teleoperation through the governed adapter; verify that table-crash command proposals are actively vetoed by the STM32.
* **Weekly Deliverable:** `UnoQMotorsBus.py` source code + demonstration of live table-crash veto.

---

### Sprint 2: Sensing, Data Collection & Policy Baseline (Weeks 4–6)
**Focus:** Sensor calibration, human demonstration recording, and baseline imitation policy training.

#### Week 4 (Oct 12 – Oct 16, 2026): "Student Zero" Run for Labs 1 & 2
* **Primary Objective:** Qualify Lab 1 (Causal Boundary) and Lab 2 (Sensing & Bridge).
* **Tasks:**
  - [ ] Mount UVC USB webcam at $45^\circ$ oblique angle, $40\text{ cm}$ overlooking workspace; tape $300 \times 200\text{ mm}$ boundary.
  - [ ] Write and test AprilTag camera calibration script (`calibrate_camera.py`).
  - [ ] Write joint homing calibration script; verify $\pm 1.5^\circ$ repeatability across 20 cycles.
  - [ ] Benchmark inter-core RPC latency: measure round-trip times across 1,000 packets; confirm $\le 5\text{ ms}$ at $50\text{ Hz}$.
  - [ ] Package student starter assets: `wiring_diagram_golden.pdf`, `calibrate_camera.py`, `test_bridge_latency.py`.
* **Weekly Deliverable:** Milestone 1 qualification package (Lab 1 & 2 starter repo + reference calibration card).

#### Week 5 (Oct 19 – Oct 23, 2026): "Student Zero" Run for Lab 3 (Teleoperation & Datasets)
* **Primary Objective:** Record golden demonstration dataset in standardized LeRobot v2 schema.
* **Tasks:**
  - [ ] Establish reproducible pick-and-place task: pick $30\text{ mm}$ foam cube from marked start zone and place in receptacle.
  - [ ] Record 10 expert demonstration episodes using teleoperation.
  - [ ] Synchronize $30\text{ Hz}$ RGB frames ($224 \times 224$) with the 4 action taps (`a_req`, `a_map`, `a_enf`, `a_meas`).
  - [ ] Format into `LeRobotDataset` v2 schema; verify zero dropped frames and correct chunk indexing.
  - [ ] Package student starter scripts: `teleop_record.py`, `inspect_dataset.py`, `dataset_golden_10ep/`.
* **Weekly Deliverable:** Published 10-episode reference dataset with complete dataset card and inspection plots.

#### Week 6 (Oct 26 – Oct 30, 2026): "Student Zero" Run for Lab 4 & Milestone 2
* **Primary Objective:** Train baseline ACT policy, quantize to ONNX INT8, and build heuristic baseline.
* **Tasks:**
  - [ ] Implement deterministic heuristic reach script (`heuristic_reach.py`) for baseline comparison.
  - [ ] Train Action Chunking with Transformers (ACT) model on workstation/Colab using the 10 golden episodes ($\le 45\text{ min}$ training).
  - [ ] Quantize trained PyTorch model to ONNX INT8 format (`export_onnx_quantized.py`).
  - [ ] Verify model file size is $< 50\text{ MB}$ and validates on held-out evaluation episodes.
  - [ ] Package starter assets: `train_act_baseline.py`, `pretrained_act_int8.onnx`, `heuristic_reach.py`.
* **Weekly Deliverable:** Milestone 2 qualification package (trained checkpoint + offline MSE evaluation curves).

---

### Sprint 3: Untethered Edge Deployment & Dynamic Adaptation (Weeks 7–9)
**Focus:** Natively running neural policies on Qualcomm Linux, action chunk horizons, and closed-loop disturbance recovery.

#### Week 7 (Nov 2 – Nov 6, 2026): Gate D & Lab 5 (Untethered Autonomous Reach)
* **Primary Objective:** Run closed-loop visual reach completely untethered on the Arduino UNO Q.
* **Tasks:**
  - [ ] Transfer `policy_int8.onnx` and `pai_edge_runtime.py` to Qualcomm Linux storage.
  - [ ] Unplug host PC tether; board runs exclusively on 45W USB-PD supply with webcam and governed arm.
  - [ ] Launch autonomous runtime via SSH over Wi-Fi:
    $$\text{Webcam Frames } (30\text{ Hz}) \longrightarrow \text{ONNX Policy } (<45\text{ ms}) \longrightarrow \text{RPC Bridge } (<5\text{ ms}) \longrightarrow \text{STM32 Motion}$$
  - [ ] **Pass Gate D:** Verify end-to-end loop latency $T_{\text{total}} \le 80\text{ ms}$; arm autonomously reaches foam block at arbitrary locations across 10 trials.
  - [ ] Package starter assets: `pai_edge_runtime.py`, `telemetry_logger.py`, `sample_reach_trace.csv`.
* **Weekly Deliverable:** Gate D verification report + multi-tap telemetry CSV of untethered reach.

#### Week 8 (Nov 9 – Nov 13, 2026): "Student Zero" Run for Lab 6 (Action Horizons & Disturbances)
* **Primary Objective:** Evaluate action chunk horizons and closed-loop disturbance recovery.
* **Tasks:**
  - [ ] Benchmark action chunk horizons: $K \in \{1, 8, 16, 32\}$. Measure tracking error and physical trajectory smoothness.
  - [ ] Test mid-trajectory disturbance: shift target block by $50\text{ mm}$ during open-loop chunk execution; measure recovery time.
  - [ ] Evaluate language conditioning: run test prompts ("pick red block" vs "pick blue block") on SmolVLA checkpoint; verify trajectory divergence.
  - [ ] Package starter assets: `benchmark_chunking.py`, `disturbance_eval.py`, `smolvla_eval_harness.py`.
* **Weekly Deliverable:** Milestone 3 qualification package (horizon comparison plots + disturbance recovery video).

#### Week 9 (Nov 16 – Nov 20, 2026): Mid-Term Rig Reliability & Thermal Stress Audit
* **Primary Objective:** Conduct continuous stress testing to isolate hardware wear and thermal bottlenecks.
* **Tasks:**
  - [ ] Run 50 continuous autonomous pick-and-place cycles on the Golden Station.
  - [ ] Measure STS3215 servo casing temperatures on shoulder ($J_2$) and elbow ($J_3$) joints; verify temperature stays below $60^\circ\text{C}$.
  - [ ] Profile V4L2 camera buffer over 2 hours of continuous streaming; confirm zero memory leaks or dropped frame buffers.
  - [ ] Re-verify joint homing calibration offsets; confirm drift is $< 1.0^\circ$ after extended duty cycle.
* **Weekly Deliverable:** 50-cycle reliability report, thermal profile graph, and mechanical wear assessment.

---

### Sprint 4: Real-Time Governance & Fault Hardening (Weeks 10–11)
**Focus:** Active microcontroller safety gating, hardware watchdogs, and zero-backlog recovery.

#### Week 10 (Nov 23 – Nov 27, 2026): "Student Zero" Run for Lab 7 (MCU Safety Governor)
* **Primary Objective:** Program and stress-test the STM32 safety governor.
* **Tasks:**
  - [ ] Finalize STM32 firmware for joint velocity saturation ($\omega_i \le 45^\circ/\text{s}$) and Cartesian floor geofence ($z \ge 15\text{ mm}$).
  - [ ] Write Python fault injection script (`inject_table_crash.py`) that sends deliberate downward velocity spikes and out-of-bound targets.
  - [ ] Demonstrate real-time MCU veto: verify that illegal proposals are rejected in $< 5\text{ ms}$ while legal joints continue moving safely.
  - [ ] Package starter assets: `stm32_governor_firmware/`, `inject_table_crash.py`, `veto_audit_golden.csv`.
* **Weekly Deliverable:** Milestone 4 Part A qualification package (firmware source + veto reaction audit).

#### Week 11 (Nov 30 – Dec 4, 2026): "Student Zero" Run for Lab 8 & Milestone 4
* **Primary Objective:** Hardware watchdog timeout and fail-safe cutoff under system crash.
* **Tasks:**
  - [ ] Program $150\text{ ms}$ hardware SysTick watchdog timer in STM32 firmware.
  - [ ] Inject Linux process freeze (`kill -STOP <pid>`) during active arm motion.
  - [ ] Verify that the arm decelerates to a safe stop within $150\text{ ms}$ of heartbeat loss.
  - [ ] Resume Linux process (`kill -CONT <pid>`); verify that the MCU flushes stale UART queues and executes **zero stale commands**.
  - [ ] Package starter assets: `stm32_watchdog_firmware/`, `fault_injection_suite.py`, `recovery_verification.py`.
* **Weekly Deliverable:** Milestone 4 complete qualification package (watchdog timing trace + zero-backlog proof).

---

### Sprint 5: Capstone Rehearsal & Class-Set Duplication (Weeks 12–13)
**Focus:** Capstone defense simulation, golden disk image freeze, and independent bench replication.

#### Week 12 (Dec 7 – Dec 11, 2026): Capstone Rehearsal & 20-Trial Release Protocol
* **Primary Objective:** Simulate the final Capstone Milestone 5 defense protocol.
* **Tasks:**
  - [ ] Define the official Capstone evaluation benchmark: 20 witnessed physical trials across held-out target locations and surface obstacles.
  - [ ] Execute 20 consecutive trials as "Student Zero"; record success rate (target $\ge 80\%$) and safety violations (must be $0\%$).
  - [ ] Prepare the Claim-Argument-Evidence (CAE) Physical Release Dossier template for students.
  - [ ] Audit the complete student-facing rubric against the 12 competencies (A1–D3).
* **Weekly Deliverable:** Reference Capstone Physical Release Dossier + 20-trial unedited video archive.

#### Week 13 (Dec 14 – Dec 18, 2026): Golden Image Freeze & Independent Replication Audit
* **Primary Objective:** Freeze software images and have a second staff member replicate Station 2 from scratch.
* **Tasks:**
  - [ ] Build Golden Qualcomm Linux SD card image: pre-installed Debian, LeRobot v0.2+, PyTorch ARM64, ONNX Runtime, V4L2 utilities, pre-cloned course repository.
  - [ ] Freeze STM32 pre-flashed binary for one-click flashing via Arduino App Lab.
  - [ ] **Independent Replication Test:** Have a secondary researcher or teaching assistant unbox a second UNO Q and SO-101 arm, follow Andrea's setup guide, flash the images, and achieve Gate D untethered reach.
  - [ ] Assemble spare parts buffer: 4x spare STS3215 servos, 2x spare webcams, replacement 3D-printed brackets, multimeters.
  - [ ] Final sign-off meeting with Prof. Vijay Janapa Reddi: certify lab readiness for Spring 2027.
* **Weekly Deliverable:** Golden `.img` file uploaded to lab server + signed Replication Audit Report.

---

## 4. Master Deliverables & Weekly Sign-Off Checklist

| Week & Date Range | Milestone / Sprint Phase | Primary Physical AI Deliverables | Target Status | Sign-Off Owner |
|:---|:---|:---|:---:|:---:|
| **W01: Sep 21 – Sep 25** | Hardware Bring-Up | • Mechanical arm assembled & clamped<br>• E-stop motor power harness verified<br>• **Gate A:** LeRobot USB teleoperation passing | `[ ]` | Andrea / VJ |
| **W02: Sep 28 – Oct 02** | Bus Authority | • STM32 UART level-shifter circuit wired<br>• Inter-core RPC bridge test script<br>• **Gate B:** 1-joint velocity clamp & veto | `[ ]` | Andrea |
| **W03: Oct 05 – Oct 09** | 6-DoF Integration | • `UnoQMotorsBus` Python adapter deployed<br>• Table geofence ($z \ge 15\text{ mm}$) programmed<br>• **Gate C:** Governed 6-DoF teleoperation | `[ ]` | Andrea / VJ |
| **W04: Oct 12 – Oct 16** | **Milestone 1:** Anatomy | • Lab 1 (Boundary) qualified by Student Zero<br>• Lab 2 (Sensing & Bridge) qualified<br>• AprilTag camera & joint offset calibration | `[ ]` | Andrea |
| **W05: Oct 19 – Oct 23** | Dataset Pipeline | • Lab 3 (Teleoperation) qualified<br>• 10-episode pick-and-place reference dataset<br>• Standardized LeRobot v2 schema validated | `[ ]` | Andrea |
| **W06: Oct 26 – Oct 30** | **Milestone 2:** Training | • Lab 4 (Baseline & Export) qualified<br>• ACT model trained on workstation GPU<br>• Quantized ONNX INT8 policy exported ($< 50\text{ MB}$) | `[ ]` | Andrea / VJ |
| **W07: Nov 02 – Nov 06** | Edge Inference | • **Gate D:** Untethered reach on Qualcomm Linux<br>• End-to-end loop latency $T_{\text{total}} \le 80\text{ ms}$<br>• Lab 5 (Autonomous Reach) qualified | `[ ]` | Andrea |
| **W08: Nov 09 – Nov 13** | **Milestone 3:** Dynamics | • Lab 6 (Action Horizons) qualified<br>• Horizon benchmark curves ($K=1..32$)<br>• Disturbance recovery mid-trajectory verified | `[ ]` | Andrea / VJ |
| **W09: Nov 16 – Nov 20** | Stress & Reliability | • 50-cycle continuous autonomous reliability test<br>• Servo thermal profiling ($< 60^\circ\text{C}$)<br>• V4L2 long-term memory stability audited | `[ ]` | Andrea |
| **W10: Nov 23 – Nov 27** | Safety Governor | • Lab 7 (Safety Governor) qualified<br>• Real-time table geofence & velocity clamp<br>• Python fault injection suite operational | `[ ]` | Andrea |
| **W11: Nov 30 – Dec 04** | **Milestone 4:** Cutoffs | • Lab 8 (Watchdog Cutoffs) qualified<br>• $150\text{ ms}$ hardware SysTick watchdog verified<br>• Linux process freeze & zero backlog proved | `[ ]` | Andrea / VJ |
| **W12: Dec 07 – Dec 11** | Capstone Rehearsal | • Capstone Studio protocol simulated<br>• 20 witnessed physical trials executed<br>• Claim-Argument-Evidence dossier template | `[ ]` | Andrea |
| **W13: Dec 14 – Dec 18** | **Final Release Freeze** | • Golden Linux SD card image compiled<br>• STM32 baseline binary locked<br>• Secondary TA replication test completed | `[ ]` | Andrea / VJ |

---

## 5. Weekly Execution & Reporting Protocol

To ensure rapid resolution of hardware bugs and maintain transparent progress toward the December deadline:

1. **Weekly Friday Check-In (Every Friday by 17:00 CET):**
   - Andrea submits a structured weekly update via GitHub Issue / PR containing:
     - **Status:** `[ON TRACK]` / `[DELAYED]` / `[BLOCKED]`
     - **Completed Tasks:** Checkboxes matching the weekly sprint table above.
     - **Evidence Links:** Commit hash, telemetry CSV, model evaluation curve, or uncut video clip.
     - **Active Blockers:** Detailed diagnostic logs of any hardware/timing failure.
2. **Bi-Weekly Physical Sync with Prof. Vijay Janapa Reddi:**
   - 30-minute bench review demonstrating the gate tests and milestone deliverables on live hardware.
3. **Escalation Trigger:**
   - If any Gate Test (A, B, C, or D) is delayed by more than 4 business days, activate the pre-approved fallback matrix in [`feasibility-plan.md`](feasibility-plan.md#sec-fallbacks) immediately.

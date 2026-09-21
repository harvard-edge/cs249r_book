# Part IV / Module 4: Governing the Machine & Capstone Studio

**Schedule:** Weeks 10–14 (Weeks 10–11: Classroom Instruction · Weeks 12–14: Dedicated Capstone Studio)
**Milestone Deliverables:**
- [Milestone 4](syllabus.md#sec-milestones) (End of Week 11) — *Certified Governed Station Under Fault Injection*
- [Milestone 5](syllabus.md#sec-milestones) (End of Week 14) — *Capstone System Defense & Physical Release Dossier*
**Required Textbook Reading:**
- [Chapter 12: Safety Enforcement](../../books/vol4/12_enforcement/12_enforcement.qmd)
- [Chapter 14: Supervisory Intervention](../../books/vol4/14_intervention/14_intervention.qmd)
- [Chapter 15: Adversarial Verification](../../books/vol4/15_verification/15_verification.qmd)
- [Chapter 16: Safe Release](../../books/vol4/16_release/16_release.qmd)
- [Chapter 17: The Epistemic Frontier](../../books/vol4/17_frontier/17_frontier.qmd)
**Associated Studio Labs:**
- [Lab 7: The Microcontroller Safety Governor](lab-07-authority-under-fault.md) (Week 10)
- [Lab 8: Fault Injection & Fail-Safe Cutoffs](lab-08-verification-and-release.md) (Week 11)
- [Capstone Project Studio & Release Defense](lab-capstone-studio.md) (Weeks 12–14)
**Target Competency Card Items:** `[ ] D1`, `[ ] D2`, `[ ] D3` *(Full 12-Card Mastery)*

---

## 1. Executive Overview

Formal lectures and new conceptual textbook reading conclude in **Week 11**. The final phase of the course shifts from guided weekly assignments into a **dedicated 3-week Capstone Project Studio (Weeks 12–14)**.

In Part IV, teams prove that their system can survive in the wild:
1. **The STM32 Microcontroller Safety Governor:** Enforcing hard real-time velocity clamps, collision envelopes, and communication watchdogs that veto invalid neural proposals in real time.
2. **Adversarial Fault Injection:** Deliberately corrupting action packets, disconnecting sensors, simulating OS freezes, and testing motor rail power cutoff transitions with zero command queue backlog.
3. **The Capstone Project Studio:** Freezing code and weights, swapping adversarial fault challenges with peer teams, executing 20 live held-out physical disturbance trials, and defending an auditable **Physical Release Dossier**.

---

## 2. Laboratory Investigations (Weeks 10–11)

### Lab 7: The Microcontroller Safety Governor (Week 10)
*Full Brief:* [lab-07-authority-under-fault.md](lab-07-authority-under-fault.md)
*Textbook Reading:* Chapters 12 & 14
*Competencies Checked:* `[ ] D1 Hardware Authority Routing & Boundary Enforcement`
1. **The Permission Boundary:** Program the STM32 MCU firmware to inspect every action proposal ($a_{\text{req}}$) arriving from Qualcomm Linux across the RPC bridge.
2. **Constraint Verification:** Implement deterministic filters:
   * *Velocity Saturation:* Clamp joint speed deltas to prevent high-kinetic impact.
   * *Table Collision Geofence:* Veto any end-effector trajectory whose forward kinematics penetrate the tabletop plane ($Z < Z_{\text{table}}$).
3. **Real-Time Veto Test:** Feed out-of-envelope and high-velocity proposals from Linux. Verify the MCU clips or refuses the commands, enforcing $a_{\text{enf}} \ne a_{\text{req}}$ without host bypass.

### Lab 8: Fault Injection & Fail-Safe Cutoffs (Week 11)
*Full Brief:* [lab-08-verification-and-release.md](lab-08-verification-and-release.md)
*Textbook Reading:* Chapters 15 & 16
*Competencies Checked:* `[ ] D2 Real-Time Safety Governor & Fault Isolation`
1. **Communication Watchdogs:** Configure an MCU hardware watchdog timer ($T_{\text{watchdog}} = 150\text{ ms}$). Forcefully kill or suspend the Linux inference process. Verify that missing heartbeats trigger an immediate safe disarm to de-energized rest state.
2. **Queue Backlog Test:** Flood the bridge with delayed packets. Confirm that the MCU never executes a stale command buffer after recovery.
3. **Hardware Power Cutoff Audit:** Switch off the 7.4V motor DC power supply during active trajectory execution. Confirm immediate servo de-energization while logic power and telemetry logging remain active.
*🎯 Milestone 4 Sign-Off (Week 11): Certified Governed Station.*

---

## 3. The Dedicated Capstone Studio Runway (Weeks 12–14)

*Full Brief:* [lab-capstone-studio.md](lab-capstone-studio.md)
*Textbook Reading:* Complete Synthesis (Chapters 1–17)
*Target Competency:* `[ ] D3 Physical Release Defense & Evidence Dossier` *(Mastery of all 12 competencies A1–D3)*

* **Week 12 (Team Build & Real-World Adaptation):** Teams select an independent manipulation challenge (e.g., color-conditioned bin sorting, compliant peg insertion, obstacle avoidance). Collect 50 custom LeRobot demonstration episodes, fine-tune SmolVLA/ACT, and deploy on the UNO Q.
* **Week 13 (Peer Fault Swapping & Rehearsal):** Teams exchange adversarial disturbance challenges with peer groups (e.g., lighting drops, object position randomization, compliant obstacles). Teams harden their STM32 governors and recovery logic.
* **Week 14 (The 20-Trial Physical Benchmark & Oral Defense):**
  1. *20 Live Physical Trials:* 10 baseline trials across randomized initial poses + 10 disturbance trials (target shifted mid-reach, illumination dimmed, obstacle placed in path).
  2. *The Physical Release Dossier:* A formal technical engineering document following the Claim-Argument-Evidence (CAE) framework, specifying the tested operating envelope, latency distributions, failure mode taxonomy, and bounded release claim.
  3. *Live Oral Defense:* Every team member traces a selected episode from camera pixels to motor torque in front of the examination panel.

> **Rigorous Engineering Standard:** A well-documented, evidence-backed conclusion that the system **should NOT be released** due to measured safety or timing limitations receives full credit. In physical systems engineering, proving where a model fails is just as valuable as demonstrating where it succeeds.

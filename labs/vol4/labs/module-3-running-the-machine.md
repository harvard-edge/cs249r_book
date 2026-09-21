# Part III / Module 3: Running the Machine

**Schedule:** Weeks 7–9
**Milestone Deliverable:** [Milestone 3](../curriculum/syllabus.md#sec-milestones) (End of Week 9) — *Autonomous Closed-Loop Manipulation Under Disturbance*
**Required Textbook Reading:**
- [Chapter 8: Sensor Perception](../../../books/vol4/08_perception/08_perception.qmd)
- [Chapter 9: Spatial Memory](../../../books/vol4/09_memory/09_memory.qmd)
- [Chapter 10: Grounded Intent](../../../books/vol4/10_intent/10_intent.qmd)
- [Chapter 11: Trajectory Planning](../../../books/vol4/11_planning/11_planning.qmd)
- [Chapter 12: Safety Enforcement](../../../books/vol4/12_enforcement/12_enforcement.qmd)
- [Chapter 13: Silicon Placement](../../../books/vol4/13_placement/13_placement.qmd)
**Associated Studio Labs:** [Lab 5: Autonomous Closed-Loop Reach on UNO Q](lab-05-local-feedback-loop.md) & [Lab 6: Action Horizons, Language & Disturbances](lab-06-action-chunks.md)
**Target Competency Card Items:** `[ ] C1`, `[ ] C2`, `[ ] C3`

---

## 1. Executive Overview

Deploying an autonomous physical system requires transforming high-dimensional sensory observations into continuous physical forces under strict millisecond deadlines. In this module, teams close the loop in reality:
1. **Deploy live on-board inference on the Arduino UNO Q** without host PC tethering.
2. **Characterize Action Chunk Horizons ($K$)** to understand the fundamental trade-off between open-loop trajectory smoothness and closed-loop reactivity.
3. **Evaluate Grounded Intent (SmolVLA)** to verify that natural language instructions measurably condition physical action choices on identical visual scenes.
4. **Subject the system to physical disturbances** (shifting target objects mid-reach) to prove the policy detects discrepancies from fresh observations and adapts or abstains.

---

## 2. Laboratory Investigations

### Lab 5: Autonomous Closed-Loop Reach on UNO Q (Weeks 7–8)
*Full Brief:* [lab-05-local-feedback-loop.md](lab-05-local-feedback-loop.md)
*Textbook Reading:* Chapters 8, 11, 13
*Competencies Checked:* `[ ] C1 Closed-Loop Autonomous Action`
1. **Edge Deployment Pipeline:** Assemble the end-to-end loop running natively on the UNO Q:
   $$\text{USB Webcam (V4L2)} \longrightarrow \text{Qualcomm Linux (ONNX ACT / SmolVLA)} \longrightarrow \text{RPC Bridge} \longrightarrow \text{STM32 MCU} \longrightarrow \text{STS3215 Servos}$$
2. **Untethered Execution:** Disconnect the workstation USB tether. Verify the UNO Q boots, arms, acquires the target via camera, and autonomously reaches the target block.
3. **Latency Profiling:** Measure the end-to-end loop latency breakdown: camera frame ingestion ($t_{\text{cam}}$), neural inference ($t_{\text{inf}}$), inter-core RPC transport ($t_{\text{rpc}}$), and motor bus transmission ($t_{\text{bus}}$). Verify $t_{\text{total}} < 100\text{ ms}$.

### Lab 6: Action Horizons, Language & Disturbances (Week 9)
*Full Brief:* [lab-06-action-chunks.md](lab-06-action-chunks.md)
*Textbook Reading:* Chapters 10 & 12
*Competencies Checked:* `[ ] C2 Temporal Horizons & Action Dynamics`, `[ ] C3 Disturbance Recovery & Replanning`
1. **Action Chunk Horizon Tuning:** Benchmark tracking error and trajectory jitter across chunk horizons ($K = 1, 8, 16, 32$). Determine the empirical limit where open-loop trajectory drift exceeds tolerance.
2. **SmolVLA Language Conditioning:** Present the arm with a workspace containing two colored blocks (red and blue). Issue paired language prompts to the same scene:
   * *"Pick up the red block"* $\longrightarrow$ Arm generates leftward reach trajectory.
   * *"Pick up the blue block"* $\longrightarrow$ Arm generates rightward reach trajectory.
   Prove that language conditioning actively alters physical joint commands.
3. **The Mid-Reach Disturbance Test:** While the arm executes a reach, displace the target object by 50 mm at step $k=5$. Compare two behaviors:
   * *Open-loop continuation (Failure):* The arm continues to the old target location, grasping empty air.
   * *Closed-loop adaptation (Recovery):* The arm captures a fresh observation, re-plans the action chunk, and successfully reaches the new target position or cleanly abstains.

---

## 3. Milestone 3 Deliverable: Autonomous Closed-Loop Manipulation Under Disturbance

By the end of Week 9, each team submits their **Milestone 3 Packet**:
1. **Witnessed Autonomous Reach Demo:** Live demonstration of the SO-101 completing the reach/manipulation task running entirely on the UNO Q without host PC control.
2. **Disturbance Recovery Trace:** Telemetry log showing the mid-reach target displacement, the fresh observation timestamp, the revised action chunk proposal, and successful physical acquisition.
3. **Horizon & Latency Benchmark:** Comparative report analyzing chunk sizes ($K=1$ vs $16$) and latency budgets.
4. **Competency Sign-Off:** Verified sign-off for **C1, C2, and C3** on the team's [Competency Card](../curriculum/student-competencies.md).

# Lab 6: Action Horizons, Language & Disturbances

**Schedule:** Week 9 | **Part:** [Part III: Running the Machine](module-3-running-the-machine.md)
**Required Textbook Reading:** [Chapter 10: Grounded Intent](../../books/vol4/10_intent/10_intent.qmd) & [Chapter 12: Safety Enforcement](../../books/vol4/12_enforcement/12_enforcement.qmd)
**Target Competencies:** `[ ] C2 Temporal Horizons & Action Dynamics`, `[ ] C3 Disturbance Recovery & Replanning`
**Milestone Alignment:** Concludes [Milestone 3](syllabus.md#sec-milestones) (End of Week 9)

---

### 1. The Physical Question
How far into the future can an embodied model predict open-loop actions before physical drift or world changes require fresh sensory observation? How does natural language condition physical action choices on the same scene, and how does a closed-loop system adapt when the world moves mid-reach?

---

### 2. Hardware Setup

| Component | Station Function | Technical Role | Visual Reference |
|:---|:---|:---|:---:|
| **Seeed SO-101 Arm** | Physical manipulation follower arm | Executes variable action chunk horizons ($K \in \{1, 8, 16, 32\}$) | <img src="assets/images/so101-follower.png" alt="SO-101 Arm" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |
| **Arduino UNO Q Board** | Edge VLA runtime engine | Runs SmolVLA/ACT inference and dynamic chunk replanning | <img src="assets/images/arduino-uno-q.jpg" alt="Arduino UNO Q" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |
| **Logitech C270 Webcam** | Live optical feedback | Provides continuous 30 FPS observation frames for disturbance detection | <img src="assets/images/logitech-c270-webcam.png" alt="Webcam" style="max-height: 115px; max-width: 140px; object-fit: contain; display: block; margin: auto;" /> |

1. **Edge Board:** Arduino UNO Q running the closed-loop runtime pipeline.
2. **Robot Station:** Seeed SO-101 arm and overhead USB webcam.
3. **Task Arena:** Tabletop setup containing two colored target blocks: a **Red Block** (left zone) and a **Blue Block** (right zone).

---

### 3. Step-by-Step Protocol

#### Step 1: Action Chunk Horizon Benchmarking
1. Evaluate the policy across four different action chunk horizon lengths:
   * $K = 1$: Single-step reactive execution (re-observe every 33 ms).
   * $K = 8$: Short horizon action chunk.
   * $K = 16$: Standard ACT action chunk.
   * $K = 32$: Long horizon open-loop action chunk.
2. For each chunk size $K$, execute 10 reach trials. Measure:
   * Trajectory smoothness (spectral arc length or jerk).
   * Total reach time.
   * Accumulated open-loop trajectory tracking drift.
3. Identify the optimal horizon $K^*$ that balances trajectory smoothness with closed-loop responsiveness.

#### Step 2: SmolVLA Language Conditioning Evaluation
1. Set up the two-block scene: Red Block on the left, Blue Block on the right.
2. Feed the exact same camera image into the Vision-Language-Action model (SmolVLA) under two distinct language prompts:
   * Prompt 1: `"reach for the red cube"`
   * Prompt 2: `"reach for the blue cube"`
3. Log the generated action chunk trajectories.
4. Verify that language conditioning produces divergent, task-appropriate physical joint commands on the identical visual scene.

#### Step 3: The Mid-Reach Physical Disturbance Trial
1. Command the arm to reach toward the red block using chunk size $K=16$.
2. At timestep step $k = 5$ (mid-trajectory), course staff safely shifts the target block 50 mm to the right.
3. Compare two policy behaviors:
   * **Open-Loop Failure (Static Policy):** The policy executes the remaining $11$ steps of the old chunk without looking, reaching empty space.
   * **Adaptive Closed-Loop Recovery (Dynamic Policy):** The system invalidates the remainder of the chunk, ingests a fresh camera frame, generates a revised action chunk, and successfully redirects to the new block position.

---

### 4. The Disturbance & Failure Test
1. **Target Removal / Disappearance:** Remove the target block completely mid-reach.
   * *Pass Criteria:* The arm must detect the absence of the target from the fresh observation and safely **abstain** (halt or return to rest pose) rather than violently searching or colliding with the table.
2. **Extreme Displacement:** Move the block completely outside the reachable envelope during execution. Verify that the STM32 MCU safety governor clamps joint angles at the boundary and refuses out-of-envelope motion.

---

### 5. Multi-Tap Telemetry Trace
Record the multi-tap telemetry trace of a successful disturbance recovery:
* Show the target shift event marked with timestamp $t_{\text{shift}}$.
* Plot the abrupt change in requested trajectory $a_{\text{req}}$ following the fresh observation.
* Show how the STM32 MCU enforces smooth velocity transitions ($a_{\text{enf}}$) without sharp jerks.
* Record the final settled position showing successful physical touch at the new coordinates.

---

### 6. Common Pitfalls & Debugging
* ⚠️ **Chunk Horizon Lag:** If $K$ is set too large ($K > 32$), the robot will appear sluggish to react to moving objects because it is committed to executing the old pre-computed trajectory.
* ⚠️ **Re-Planning Jitter:** Re-planning too frequently with high-variance policies can cause the arm to stutter. Use temporal ensemble averaging (exponential moving average over overlapping chunks) to smooth transitions.

---

### 7. Sign-Off Criteria (The Exit Check)
To receive credit for Lab 6 and complete Milestone 3:
1. [ ] **Horizon Analysis Report:** Present the comparative tracking error and smoothness curves across chunk sizes ($K=1, 8, 16, 32$).
2. [ ] **Language Conditioning Proof:** Demonstrate that changing the language prompt on the identical scene produces divergent, verified physical arm paths.
3. [ ] **Live Disturbance Recovery Demo:** The instructor shifts the block mid-reach. The system must adapt autonomously, revising its trajectory to make successful contact at the new position.
*Staff signs off `[ ] C2` and `[ ] C3` on the team's [Competency Card](student-competencies.md).*

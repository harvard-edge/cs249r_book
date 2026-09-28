# Lab 5: Action Horizons, Language & Disturbances

**Schedule:** Week 9 · Milestone 3 due end of week
**Required Textbook Reading:** [Chapter 10: Grounded Intent](../../../books/vol4/10_intent/10_intent.qmd) & [Chapter 12: Safety Enforcement](../../../books/vol4/12_enforcement/12_enforcement.qmd)
**Target Competencies:** `[ ] C2 Temporal Horizons & Action Dynamics`, `[ ] C3 Disturbance Recovery & Replanning`
---

## Overview

### Purpose
Your fine-tuned VLA can reach targets autonomously — but can it handle the unexpected? The real world doesn't hold still while your model thinks. This lab stress-tests your system along three axes: how far ahead the model can predict before physical drift accumulates (action horizons), whether language can redirect physical behavior (language conditioning), and what happens when the environment changes mid-action (disturbance recovery). This is where you learn the difference between a demo that works once and a system that works reliably.

### Prerequisites
**Lab 4 completed.** You need a fine-tuned VLA running in an untethered autonomous S·P·A loop on the UNO Q with ≥ 7/10 reach success rate and verified end-to-end latency ≤ 80 ms.

### Learning Outcomes
By the end of this lab, you will be able to:
- Benchmark how action chunk horizon length (K) affects manipulation success rate, motion smoothness, and recovery ability
- Demonstrate that natural language prompts can produce measurably different physical trajectories on the same scene
- Evaluate closed-loop disturbance recovery by measuring whether the VLA replans or drifts when the target is displaced mid-reach
- Quantify the tradeoff between open-loop prediction efficiency and closed-loop adaptation

### What You Will Do
In one week, you will run systematic experiments across three dimensions. First, you will benchmark four action chunk horizons (K = 1, 8, 16, 32) with matched trials. Second, you will test SmolVLA's language conditioning by giving contrasting prompts ("pick up the red block" vs. "pick up the blue block") and measuring the resulting trajectory divergence. Third, you will physically shift the target object 50 mm mid-reach and evaluate whether the VLA replans toward the new location or completes its original trajectory into empty space.

---

### 1. The Physical Question
How far into the future can an embodied model predict open-loop actions before physical drift or world changes require fresh sensory observation? How does natural language condition physical action choices on the same scene, and how does a closed-loop system adapt when the world moves mid-reach?

---

### 2. Hardware Setup

> **Hardware Setup:** See the [Station Reference Card](station-reference.md) for the standard bench configuration.


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
*Staff signs off `[ ] C2` and `[ ] C3` on the team's [Competency Card](../curriculum/student-competencies.md).*

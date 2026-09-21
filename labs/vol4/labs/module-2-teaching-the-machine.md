# Part II / Module 2: Teaching the Machine

**Schedule:** Weeks 4–6
**Milestone Deliverable:** [Milestone 2](../curriculum/syllabus.md#sec-milestones) (End of Week 6) — *Validated Demonstration Dataset & Edge-Ready SmolVLA / ACT Policy on Qualcomm Linux*
**Required Textbook Reading:**
- [Chapter 5: Physical Data](../../../books/vol4/05_data/05_data.qmd)
- [Chapter 6: Policy Training](../../../books/vol4/06_training/06_training.qmd)
- [Chapter 7: Closed-Loop Evaluation](../../../books/vol4/07_evaluation/07_evaluation.qmd)
**Associated Studio Labs:** [Lab 3: Teleoperation, LeRobot Dataset & Action Taps](lab-03-physical-episodes.md) & [Lab 4: Deterministic Baseline & SmolVLA/ACT Export](lab-04-baseline-and-learning.md)
**Target Competency Card Items:** `[ ] B1`, `[ ] B2`, `[ ] B3`

---

## 1. Executive Overview

A Physical AI system is defined by its ability to learn from physical demonstrations, generalize across perceptual variations, and operate within the strict computational bounds of an edge processor. In this module, teams build the learning engine:
1. **Collect high-fidelity demonstration datasets** on the SO-101 arm using Hugging Face LeRobot.
2. **Log the 4 continuous action taps** (`a_req`, `a_map`, `a_enf`, `a_meas`) to maintain complete physical observability.
3. **Construct a deterministic scripted baseline** to benchmark where classical control works and where machine learning is strictly necessary.
4. **Train, compress, and profile an edge policy (SmolVLA or ACT)** running locally on the Qualcomm QRB2210 processor under strict latency budgets (<100 ms).

---

## 2. The Four Action Taps Schema

To guarantee observability across the asynchronous boundaries of Linux, the microcontroller, and the physical servos, every recorded episode must preserve four explicit action taps:

```
[ Learned Policy (Qualcomm Linux) ]
       │
       ▼ (1) REQUESTED ACTION  (a_req: raw policy joint targets from SmolVLA / ACT)
[ RPC Bridge ]
       │
       ▼ (2) MAPPED ACTION     (a_map: transformed into joint limits, gear ratios & units)
[ STM32 Safety Governor ]
       │
       ▼ (3) ENFORCED ACTION   (a_enf: verified, rate-limited, or clipped safe command)
[ STS3215 Servos & Plant ]
       │
       ▼ (4) MEASURED ACTION   (a_meas: actual settled joint angles & tool trajectory)
```

Each log packet contains: `[cycle_id, epoch_timestamp, a_req, a_map, a_enf, a_meas, refusal_flag, next_obs_id]`.

---

## 3. Laboratory Investigations

### Lab 3: Teleoperation, LeRobot Dataset & Action Taps (Weeks 4–5)
*Full Brief:* [lab-03-physical-episodes.md](lab-03-physical-episodes.md)
*Textbook Reading:* Chapters 5 & 7
*Competencies Checked:* `[ ] B1 Physical Dataset Engineering & Multi-Tap Logging`
1. **Teleoperation Setup:** Configure a leader arm, gamepad, or keyboard teleoperator to drive the SO-101 follower arm at a stable 30 Hz control loop.
2. **Dataset v3 Episode Collection:** Collect 30 clean demonstration episodes of reaching, pushing, or grasping varied target positions in the workspace.
3. **Multi-Tap Logging:** Record synchronized multi-modal streams: RGB images ($640 \times 480$ @ 30 FPS downsampled to $224 \times 224$), joint telemetry ($q_t$), and all four action taps (`a_req`, `a_map`, `a_enf`, `a_meas`).
4. **Leak-Free Partitioning:** Partition episodes strictly by recording session into train/val/test splits, ensuring no frame-level temporal leakage between splits.

### Lab 4: Deterministic Baseline & SmolVLA/ACT Export (Week 6)
*Full Brief:* [lab-04-baseline-and-learning.md](lab-04-baseline-and-learning.md)
*Textbook Reading:* Chapter 6
*Competencies Checked:* `[ ] B2 Deterministic Baseline Benchmarking`, `[ ] B3 Edge Model Profiling & Resource Budgets`
1. **Deterministic Baseline:** Implement an unlearned scripted controller (e.g., fixed geometric visual reach) on identical starting positions. Measure its success rate, execution time, and failure points when targets are displaced.
2. **Policy Training / Fine-Tuning:** Train a compact ACT (Action Chunking with Transformers) policy or fine-tune SmolVLA on the collected LeRobot dataset using workstation/Colab compute. Evaluate validation loss curves.
3. **Model Quantization & Edge Export:** Export the trained policy checkpoint to ONNX / INT8 format. Transfer the model to the Arduino UNO Q.
4. **Edge Profiling:** Profile inference latency distribution, memory footprint (<150 MB), and CPU load on the Qualcomm QRB2210 processor. Verify that inference executes comfortably within the loop deadline (<100 ms).

---

## 4. Milestone 2 Deliverable: Edge-Ready Policy on Qualcomm Linux

By the end of Week 6, each team submits their **Milestone 2 Packet**:
1. **Curated LeRobot Dataset:** Replayable LeRobot Dataset v3 on Hugging Face Hub (or local disk) with 30 episodes, complete Dataset Card, and documented 4-action-tap schema.
2. **Baseline Comparison Report:** Performance benchmark comparing the deterministic scripted baseline against teleoperated human demonstrations.
3. **Quantized Edge Model Artifact:** Pinned ONNX model checkpoint running on the Qualcomm QRB2210 with verified latency and memory profiling traces.
4. **Competency Sign-Off:** Verified sign-off for **B1, B2, and B3** on the team's [Competency Card](../curriculum/student-competencies.md).

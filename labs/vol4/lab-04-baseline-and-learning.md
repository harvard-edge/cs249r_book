# Lab 4: Deterministic Baseline & SmolVLA/ACT Export

**Schedule:** Week 6 | **Part:** [Part II: Teaching the Machine](module-2-teaching-the-machine.md)
**Required Textbook Reading:** [Chapter 6: Policy Training](../../books/vol4/06_training/06_training.qmd)
**Target Competencies:** `[ ] B2 Deterministic Baseline Benchmarking`, `[ ] B3 Edge Model Profiling & Resource Budgets`
**Milestone Alignment:** Concludes [Milestone 2](syllabus.md#sec-milestones) (End of Week 6)

---

### 1. The Physical Question
Why do we use machine learning instead of classical control? Does a learned neural policy (ACT or SmolVLA) actually outperform a deterministic scripted rule on hardware, and can its computational footprint fit comfortably within an edge Linux MPU's real-time deadline budget?

---

### 2. Hardware & Bench Setup
1. **Workstation / GPU Server:** Machine learning training environment (or Google Colab / HF Hub) equipped with PyTorch and LeRobot.
2. **Edge Board:** Arduino UNO Q with Qualcomm Debian Linux.
3. **Robot Station:** Seeed SO-101 arm and USB webcam.
4. **Bench Target:** Foam block placed at 10 fixed coordinate positions in the workspace.

---

### 3. Step-by-Step Bench Protocol

#### Step 1: Deterministic Baseline Implementation
1. Write a non-learned scripted controller: detect the colored block using OpenCV color thresholding or contour centroid estimation from the camera frame.
2. Calculate target joint angles using inverse kinematics or a calibrated linear lookup table.
3. Command the arm to execute the reach via the STM32 MCU.
4. Run 10 repeated physical trials across the 10 marked target positions. Record:
   * Success rate (block touched within $\pm 5\text{ mm}$).
   * Total reach duration ($t_{\text{reach}}$).
   * Failure modes when lighting drops or the block is rotated.

#### Step 2: Policy Training & Validation (ACT or SmolVLA)
1. Train an imitation learning policy on the 30-episode dataset collected in Lab 3 using LeRobot:
   ```bash
   python lerobot/scripts/train.py \
     --dataset_repo_id local/so101_reach_dataset_v3 \
     --policy.type=act \
     --policy.chunk_size=16 \
     --training.batch_size=8 \
     --training.epochs=50 \
     --output_dir=checkpoints/so101_act
   ```
2. Plot validation L1/MSE trajectory error. Verify that the model converges to a stable loss curve without overfitting.

#### Step 3: Quantization & ONNX Export for the Edge
1. Export the PyTorch model checkpoint to ONNX format with dynamic batching:
   ```bash
   python -m pai_tools.export_onnx \
     --checkpoint checkpoints/so101_act/best_model.pth \
     --output checkpoints/so101_act.onnx \
     --quantize int8
   ```
2. Transfer the quantized ONNX artifact (`<150 MB`) to the Arduino UNO Q via `scp`.
3. Benchmark inference execution time on the Qualcomm QRB2210 CPU:
   ```bash
   python3 -m pai_tools.profile_onnx --model so101_act.onnx --iterations 100
   ```
4. Verify that mean inference latency $t_{\text{inf}} < 80\text{ ms}$, satisfying the loop deadline.

---

### 4. The Disturbance & Failure Test
1. **Host-vs-Edge Numerical Parity Test:** Feed an identical test image and joint state vector into both the PyTorch workstation model and the quantized ONNX edge model on the UNO Q.
   * *Pass Criteria:* Max absolute error across joint angle action outputs must be $< 0.05\text{ rad}$ ($< 3^\circ$), proving quantization has not corrupted policy fidelity.
2. **Thermal & Memory Throttling Stress:** Run continuous inference on the Qualcomm chip for 10 minutes. Monitor CPU core temperatures and verify that thermal throttling does not increase latency beyond 100 ms.

---

### 5. Multi-Tap Telemetry Trace
Record the performance table comparing the deterministic baseline against the offline learned policy predictions across 10 identical initial scene images:
* Baseline success rate vs. Policy offline trajectory accuracy.
* Compute time: Baseline visual script ($12\text{ ms}$) vs. ACT inference ($65\text{ ms}$).
* Document where the baseline fails (e.g., lighting variations or shadows) and how the learned visual representation handles the shift.

---

### 6. Common Hardware Pitfalls & Debugging
* ⚠️ **Memory Exhaustion on 2GB Boards:** If PyTorch is imported directly on a 2GB RAM UNO Q, the OS may trigger the Out-Of-Memory (OOM) killer. Always run inference using lightweight **ONNX Runtime**, not full PyTorch, on the edge board.
* ⚠️ **Normalization Mismatch:** Ensure camera images fed to the ONNX model are normalized with the exact same mean and standard deviation vectors used during training.

---

### 7. Bench Sign-Off Criteria (The Exit Check)
To receive credit for Lab 4 and complete Milestone 2:
1. [ ] **Baseline Benchmark Data:** Present the 10-trial performance log of the deterministic baseline controller detailing its limitations.
2. [ ] **Edge Execution Proof:** Execute live inference of the ONNX policy on the Qualcomm Linux processor on the UNO Q, demonstrating inference execution time $< 100\text{ ms}$.
3. [ ] **Memory & Latency Trace:** Submit a profiling report showing memory usage $< 150\text{ MB}$ and stable 10 Hz action chunk generation.
*Staff signs off `[ ] B2` and `[ ] B3` on the team's [Competency Card](student-competencies.md).*

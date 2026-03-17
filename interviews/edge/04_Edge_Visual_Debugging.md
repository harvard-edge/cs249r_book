# Round 4: Visual Architecture Debugging 🖼️

<div align="center">
  <a href="../README.md">🏠 Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="01_Edge_Systems.md">🤖 Round 1</a> ·
  <a href="02_Edge_Constraints.md">⚖️ Round 2</a> ·
  <a href="03_Edge_Ops_and_Deployment.md">🚀 Round 3</a> ·
  <a href="04_Edge_Visual_Debugging.md">🖼️ Round 4</a> ·
  <a href="05_Edge_Advanced.md">🔬 Round 5</a>
</div>

---

The ultimate test of an edge ML systems engineer is spotting the bottleneck in a proposed architecture *before* it ships to 10,000 devices that can't be easily recalled. Each challenge presents a plausible edge system design with a hidden flaw. Try to find it before clicking "Reveal the Bottleneck."

> **[➕ Add a Visual Challenge](https://github.com/harvard-edge/cs249r_book/edit/dev/interviews/edge/04_Edge_Visual_Debugging.md)** (Edit in Browser) — see [README](../README.md#question-format) for the template.

---

## 🛑 Challenge 1: The "High-Performance" Perception Pipeline · `latency` `memory`

**The Scenario:** An autonomous delivery robot team designed their perception pipeline on a Jetson AGX Orin. The pipeline processes camera frames through three sequential models.

```mermaid
flowchart LR
    Camera[Camera
30 FPS
1920x1080] -->|MIPI CSI| Preprocess[CPU Preprocessing
Resize + Normalize
8ms]

    Preprocess -->|"cudaMemcpy H→D"| Detection[GPU: YOLOv8-M
Detection
18ms]

    Detection -->|"cudaMemcpy D→H→D"| Depth[GPU: MiDaS
Depth Estimation
15ms]

    Depth -->|"cudaMemcpy D→H→D"| Tracking[GPU: DeepSORT
Tracking
8ms]

    Tracking -->|Results| Planner[Path Planner]

    classDef error fill:#ffe0e0,stroke:#d32f2f,stroke-width:2px;
    class Preprocess error;
```

**The Question:** The team measures 55ms end-to-end latency (8+18+15+8+6ms overhead) and gets 18 FPS — well below the 30 FPS target. The GPU utilization is only 40%. The profiler shows the GPU is idle for 30% of each frame. Where is the time going?

<details>
<summary><b>🚨 Reveal the Bottleneck</b></summary>

### The Host-Device Memory Bounce

**Common Mistake:** "The models are too slow — use smaller models." The models themselves only take 41ms of compute. The missing time is in data movement.

The pipeline copies data from GPU → CPU → GPU between every model (`cudaMemcpy D→H→D`). Each round-trip takes 2-4ms for a 1080p tensor, and there are 3 such copies. Worse, `cudaMemcpy` is synchronous by default — the GPU sits completely idle during each transfer, creating bubbles.

**The Fix:** Keep all tensors on the GPU. Use CUDA unified memory or explicit device-to-device transfers. Detection's output tensor becomes Depth's input tensor *without ever touching the CPU*. Preprocessing should also move to the GPU (NVIDIA DALI or custom CUDA kernels). This eliminates ~12ms of memory copies and GPU idle time, bringing the pipeline to ~43ms → 23 FPS. Add frame pipelining (overlap frame N's depth with frame N+1's detection) to reach 30+ FPS.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

## 🛑 Challenge 2: The "Redundant" Multi-Camera System · `memory` `architecture`

**The Scenario:** A warehouse robot has 4 cameras (front, back, left, right) for 360° perception. The team runs a separate YOLOv8-S instance for each camera on a Jetson Orin NX.

```mermaid
flowchart TD
    subgraph "Jetson Orin NX (16 GB DRAM)"
        subgraph "Camera Pipelines"
            Cam1[Front Camera] --> Model1[YOLOv8-S Instance 1
Weights: 22 MB
Activations: 80 MB]
            Cam2[Back Camera] --> Model2[YOLOv8-S Instance 2
Weights: 22 MB
Activations: 80 MB]
            Cam3[Left Camera] --> Model3[YOLOv8-S Instance 3
Weights: 22 MB
Activations: 80 MB]
            Cam4[Right Camera] --> Model4[YOLOv8-S Instance 4
Weights: 22 MB
Activations: 80 MB]
        end
        Fusion[Fusion Layer] --> Planner[Path Planner]
    end

    Model1 --> Fusion
    Model2 --> Fusion
    Model3 --> Fusion
    Model4 --> Fusion

    classDef error fill:#ffe0e0,stroke:#d32f2f,stroke-width:2px;
    class Model1,Model2,Model3,Model4 error;
```

**The Question:** The system uses 4 × (22 + 80) = 408 MB for ML alone, and each camera gets only 25% of the GPU's compute time, running at 12 FPS per camera. The team says "we need a bigger GPU." Do they?

<details>
<summary><b>🚨 Reveal the Bottleneck</b></summary>

### Quadruplicated Weights

**Common Mistake:** "4 cameras need 4 model instances." The cameras need 4 inference passes, but NOT 4 copies of the weights.

All four instances load identical YOLOv8-S weights — 22 MB × 4 = 88 MB of duplicated weights in GPU memory. Since the weights are read-only during inference, a single copy serves all four cameras. Load the weights once (22 MB) and run 4 inference passes with different input tensors. Memory drops from 408 MB to 22 + (4 × 80) = 342 MB — a 16% reduction. More importantly, **batch the 4 camera inputs** into a single inference call: batch size 4 with shared weights. On the Orin NX, batched inference is 2-3× more efficient than 4 sequential calls because it amortizes kernel launch overhead and improves GPU occupancy. Expected FPS per camera: 25-30 (up from 12).

**The Fix:** Single model instance, batch size 4. If the cameras have different resolutions or FOVs, use dynamic batching with padding. Memory: 22 MB weights + 320 MB activations (batched) = 342 MB. Throughput: ~28 FPS per camera.

**📖 Deep Dive:** [Volume I: Model Serving](https://mlsysbook.ai/vol1/model_serving.html)
</details>

---

## 🛑 Challenge 3: The "Safe" OTA Update Pipeline · `deployment` `reliability`

**The Scenario:** The fleet management team designed an OTA model update system for 10,000 edge cameras. The system downloads the new model, validates it, and swaps it in.

```mermaid
flowchart TD
    Cloud[Cloud Model Registry] -->|"1. Download new model (LTE)"| Device[Edge Device]

    subgraph "Update Process"
        Device -->|"2. Stop inference"| Stop[Stop Current Model]
        Stop -->|"3. Delete old model"| Delete[Free Storage]
        Delete -->|"4. Write new model"| Write[Write to Disk]
        Write -->|"5. Load new model"| Load[Load into GPU]
        Load -->|"6. Resume inference"| Resume[Resume Service]
    end

    classDef error fill:#ffe0e0,stroke:#d32f2f,stroke-width:2px;
    class Stop,Delete error;
```

**The Question:** The team says "the update takes 45 seconds — the camera is blind for less than a minute." The security team rejects this design. What are the two critical flaws?

<details>
<summary><b>🚨 Reveal the Bottleneck</b></summary>

### Blind Spot + No Rollback

**Common Mistake:** "45 seconds of downtime is acceptable for a software update." For a security camera, 45 seconds of blindness is an exploitable window. For a safety system, it's unacceptable.

**Flaw 1: Inference gap.** Steps 2-6 create a 45-second window where the camera produces no detections. An intruder who knows the update schedule has a guaranteed blind spot. For safety-critical systems (autonomous vehicles, industrial monitoring), any inference gap violates the safety contract.

**Flaw 2: No rollback path.** Step 3 deletes the old model *before* the new model is validated. If the download was corrupted, or the new model fails validation, or power is lost during step 4, the device has *no model at all* — it's bricked.

**The Fix:** A/B partitioning with hot-swap. Keep two model slots. Download the new model to the inactive slot *while the active model continues serving*. Validate the new model by running test inference on a reference image. Only then atomically swap the active pointer. Zero downtime, instant rollback.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

---

## 🛑 Challenge 4: The "Efficient" Sensor Fusion Pipeline · `sensor-fusion` `latency`

**The Scenario:** An autonomous vehicle fuses camera and LiDAR data for 3D object detection. The team designed a straightforward pipeline.

```mermaid
flowchart LR
    Camera[Camera
30 FPS
t=0ms] -->|"Capture"| CamBuf[Frame Buffer]

    LiDAR[LiDAR
10 Hz
t=0ms] -->|"360° sweep
100ms"| LidarBuf[Point Cloud Buffer]

    CamBuf -->|"Wait for both"| Sync[Synchronization
Barrier]
    LidarBuf --> Sync

    Sync -->|"Paired data"| Fusion[3D Fusion Model
25ms]

    Fusion --> Output[3D Detections]

    classDef error fill:#ffe0e0,stroke:#d32f2f,stroke-width:2px;
    class Sync error;
```

**The Question:** The system produces 3D detections at only 10 Hz (limited by LiDAR rate), even though the camera runs at 30 FPS. At highway speed, a 100ms gap between detections means the vehicle travels 3 meters blind. The team says "we need a faster LiDAR." Is that the only option?

<details>
<summary><b>🚨 Reveal the Bottleneck</b></summary>

### The Synchronization Barrier Stalls the Fast Sensor

**Common Mistake:** "The LiDAR is the bottleneck — buy a 30 Hz LiDAR." A faster LiDAR helps but costs $5,000+ more and doesn't solve the fundamental architecture issue.

The synchronization barrier forces the camera to wait for the LiDAR. 20 out of every 30 camera frames are discarded — 67% of the visual information is thrown away. The pipeline's output rate is locked to the slowest sensor.

**The Fix:** Asynchronous fusion with temporal interpolation. Run camera-only 2D detection at 30 FPS (no waiting). Run LiDAR 3D processing at 10 Hz independently. Between LiDAR sweeps, project the 2D camera detections into 3D using the *most recent* LiDAR depth map + ego-motion compensation from the IMU. This gives 30 Hz 3D detections: 10 Hz are "full fusion" (camera + fresh LiDAR), and 20 Hz are "interpolated fusion" (camera + stale LiDAR + IMU correction). At highway speed, the gap between detections drops from 100ms (3m) to 33ms (1m).

**📖 Deep Dive:** [Volume I: Data Engineering](https://mlsysbook.ai/vol1/data_engineering.html)
</details>

---

## 🛑 Challenge 5: The "Optimized" Multi-Model Scheduler · `latency` `thermal`

**The Scenario:** An edge device runs three models sequentially on a single GPU: detection (18ms), segmentation (12ms), and classification (5ms). Total: 35ms per frame. The team wants to add a fourth model (pose estimation, 15ms) but that would push the total to 50ms — well over the 33ms budget.

```mermaid
flowchart LR
    Frame[Input Frame] --> Det[Detection
18ms
GPU 100%]
    Det --> Seg[Segmentation
12ms
GPU 100%]
    Seg --> Class[Classification
5ms
GPU 30%]
    Class --> Pose["Pose Estimation
15ms
GPU 100%
❌ Over Budget"]

    subgraph "Timeline (50ms total)"
        T1[0-18ms: Det] --> T2[18-30ms: Seg] --> T3[30-35ms: Class] --> T4["35-50ms: Pose"]
    end

    classDef error fill:#ffe0e0,stroke:#d32f2f,stroke-width:2px;
    classDef underused fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    class Pose error;
    class Class underused;
```

**The Question:** Look at the GPU utilization numbers. One model is only using 30% of the GPU. The team treats all four models as a flat sequential pipeline. What scheduling optimization can bring the total under 33ms without changing any model?

<details>
<summary><b>🚨 Reveal the Bottleneck</b></summary>

### Wasted GPU Parallelism

**Common Mistake:** "Run all models sequentially — the GPU can only do one thing at a time." Modern GPUs support concurrent kernel execution via CUDA streams.

Classification uses only 30% of the GPU — it's a small model that doesn't saturate the compute units. The remaining 70% of the GPU is idle during those 5ms. Using CUDA Multi-Process Service (MPS) or multiple CUDA streams, you can overlap classification with the beginning of pose estimation. Better yet, overlap classification with segmentation (both together use ~130% of a single GPU's resources, but with proper stream scheduling, they share the SMs efficiently).

**The Fix:** Use CUDA streams to overlap small models with large ones. Run classification concurrently with segmentation (12ms for both, since classification finishes in 5ms within the 12ms segmentation window). Then run pose estimation. New timeline: Detection (18ms) → Segmentation + Classification (12ms) → Pose (15ms) = **45ms**. Still over budget. Next: overlap detection's NMS (CPU, 2ms) with segmentation's first layers (GPU). And run pose estimation at half rate (every other frame, using temporal interpolation). Effective: 18ms + 12ms + 7.5ms (amortized pose) = **37.5ms**. With the stream overlaps: ~33ms. Meets budget.

**📖 Deep Dive:** [Volume I: ML Frameworks](https://mlsysbook.ai/vol1/frameworks.html)
</details>

---

## 🛑 Challenge 6: The "Complete" Edge Monitoring Stack · `monitoring` `economics`

**The Scenario:** The operations team designed a monitoring system for their edge camera fleet. Every camera streams its inference results to the cloud for analysis.

```mermaid
flowchart TD
    subgraph "Edge Device (per camera)"
        Inference[Inference Engine
30 FPS] -->|"Every frame"| Encode[JSON Encode
Detections + Metadata]
        Encode -->|"30 msgs/sec"| Cell[Cellular Modem
LTE Cat-4]
    end

    Cell -->|"Upload"| Cloud[Cloud Analytics
Dashboard]

    subgraph "Per-Device Bandwidth"
        BW["30 detections/sec × 2 KB each
= 60 KB/s = 4.3 GB/month"]
    end

    classDef error fill:#ffe0e0,stroke:#d32f2f,stroke-width:2px;
    class Cell error;
```

**The Question:** With 10,000 cameras, the monthly cellular bill arrives: $430,000. The CFO says "shut it down." What went wrong with this monitoring architecture, and how do you get the same operational visibility for 1/100th the cost?

<details>
<summary><b>🚨 Reveal the Bottleneck</b></summary>

### Streaming Raw Results Over Cellular

**Common Mistake:** "We need all the data for proper monitoring." You need operational *visibility*, not raw data. 99% of the frames contain nothing interesting.

10,000 cameras × 4.3 GB/month × $1/GB cellular = $43,000/month (the $430K figure assumes higher per-GB rates or includes the data plan). Either way, streaming every detection result is economically insane.

**The Fix:** Edge-side aggregation. On each device, compute hourly statistics: detection counts by class, confidence histograms, latency percentiles, thermal state. Upload only the aggregates (~50 KB/day) plus a small sample of anomalous frames (~500 KB/day). Total: ~550 KB/day × 30 days = 16.5 MB/month per device. At $0.01/MB: $0.165/device/month × 10,000 = **$1,650/month** — a 260× reduction. You lose per-frame granularity but gain the same operational insights (drift detection, anomaly alerts, fleet health) at sustainable cost.

**📖 Deep Dive:** [Volume I: ML Operations](https://mlsysbook.ai/vol1/ml_ops.html)
</details>

---

## 🛑 Challenge 7: The "Thermal-Safe" Outdoor Deployment · `thermal` `power`

**The Scenario:** An edge AI box is deployed in a sealed IP67 enclosure on a rooftop in Phoenix, Arizona. The team tested it in the lab at 22°C and it ran perfectly at 30 FPS.

```mermaid
flowchart TD
    subgraph "IP67 Sealed Enclosure"
        SoC["Jetson Orin NX
25W TDP
Small Heatsink"] -->|"Heat"| Air["Trapped Air
No Ventilation"]
        Air -->|"Conduction only"| Enclosure["Aluminum Enclosure
Surface Area: 0.04 m²"]
    end

    Sun["☀️ Arizona Sun
45°C Ambient
+ Solar Load"] -->|"Radiant Heat"| Enclosure

    subgraph "Thermal Timeline"
        T0["t=0: Boot at 45°C ambient"] --> T1["t=2min: Junction 78°C
30 FPS"]
        T1 --> T2["t=5min: Junction 85°C
DVFS → 20 FPS"]
        T2 --> T3["t=10min: Junction 95°C
DVFS → 12 FPS"]
        T3 --> T4["t=15min: Junction 105°C
THERMAL SHUTDOWN"]
    end

    classDef error fill:#ffe0e0,stroke:#d32f2f,stroke-width:2px;
    class Air error;
```

**The Question:** The device works in the lab but shuts down within 15 minutes in the field. The team's thermal design assumed 25°C ambient. What three design changes would you make to survive 45°C ambient with sustained 30 FPS?

<details>
<summary><b>🚨 Reveal the Bottleneck</b></summary>

### Thermal Design for Lab, Not Field

**Common Mistake:** "Add a bigger fan." The enclosure is sealed (IP67) — there's no airflow path for a fan.

Three design changes:

**(1) Reduce power mode from the start.** Run at 15W instead of 25W. At 15W with a 2°C/W heatsink: ΔT = 30°C. Junction = 45 + 30 = 75°C — below the 80°C throttle threshold. Sustained 20 FPS instead of 30→12→shutdown. Predictable is better than fast-then-dead.

**(2) Thermal path to enclosure.** Replace the small heatsink + trapped air gap with a direct thermal path: copper heat pipe from SoC to the aluminum enclosure wall, with thermal interface material (TIM) at both junctions. The enclosure becomes the heatsink. Thermal resistance: ~1°C/W (vs 5°C/W with air gap). At 25W: ΔT = 25°C. Junction = 45 + 25 = 70°C — safe at full power.

**(3) Solar shield.** A white-painted sun shade above the enclosure reduces solar load by 60%. Effective ambient drops from 45°C + 15°C solar load = 60°C to 45°C + 6°C = 51°C. Combined with the thermal path: junction = 51 + 25 = 76°C — sustainable at full power.

**📖 Deep Dive:** [Volume I: HW Acceleration](https://mlsysbook.ai/vol1/hw_acceleration.html)
</details>

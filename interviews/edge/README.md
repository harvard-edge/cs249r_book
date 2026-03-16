# 🤖 Edge Track — Autonomous Systems & Industrial AI

<div align="center">
  <a href="../README.md">🏠 Playbook Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> ·
  <b>🤖 Edge</b> ·
  <a href="../mobile/README.md">📱 Mobile</a> ·
  <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

The Edge track covers ML systems deployed on dedicated hardware at the point of action — autonomous vehicles, robotics platforms, CCTV and surveillance systems, industrial inspection, and medical devices.

### The Constraint Regime

| Dimension | Edge Reality |
|---|---|
| **Compute** | TOPS (Jetson Orin, Hailo-8, Intel Movidius, Google Coral) |
| **Memory** | 8–32 GB DRAM, shared with sensor pipelines |
| **Interconnect** | PCIe, MIPI CSI (camera), CAN bus (automotive) |
| **Power budget** | 15–75W per module |
| **Primary bottleneck** | Thermal envelope and real-time deadlines |
| **Failure mode** | Missing a hard real-time deadline, thermal throttling under sustained load |

### What Makes Edge Different from Cloud

In the cloud, you can always add more GPUs. At the edge, the hardware is fixed and the environment is hostile. An autonomous vehicle running object detection at 30 FPS cannot drop frames when the sun angle changes. A robotic arm running pose estimation cannot pause for garbage collection. The physics of edge is the physics of **hard constraints under uncertainty**.

### Topics That Need Questions

These are the areas where edge-specific interview questions would be most valuable. Each maps to a real interview scenario at companies like Tesla, Waymo, Boston Dynamics, or industrial AI startups.

| Topic | What edge interviews test | Example scenario |
|---|---|---|
| **Roofline** | Integer-only roofline on NPUs, understanding TOPS vs TOPS/W | "Your Jetson Orin hits 50% of peak INT8 TOPS. Is it compute-bound or memory-bound?" |
| **Real-time inference** | Worst-case execution time (WCET), frame budgets, pipeline scheduling | "Your perception stack must run at 30 FPS. You have 33ms per frame. How do you partition detection, tracking, and planning?" |
| **Quantization** | INT8/INT4 for thermal headroom, quantization-aware training | "Quantizing to INT4 saves 15W but drops mAP by 3%. How do you decide?" |
| **Sensor fusion** | Multi-modal pipelines (camera + LiDAR + radar), synchronization | "Your camera and LiDAR timestamps drift by 50ms. What happens to your 3D bounding boxes?" |
| **Thermal management** | Sustained vs burst performance, thermal throttling curves | "Your edge box runs at 200 TOPS for 30 seconds, then throttles to 80 TOPS. How do you design for steady state?" |
| **Functional safety** | Graceful degradation, redundancy, fail-safe vs fail-operational | "Your primary model crashes mid-inference. What does the fallback path look like?" |
| **Model update** | OTA updates, A/B model deployment on constrained devices | "How do you deploy a new model to 10,000 edge devices without bricking any of them?" |

### The Rounds

| Round | Focus | Questions |
|---|---|---|
| [**1. Edge Systems & Real-Time Physics**](01_Edge_Systems.md) | Roofline, real-time deadlines, thermal management, sensor fusion, OTA | 9 |
| [**2. Edge Advanced**](02_Edge_Advanced.md) | Memory management, architecture selection, optimization, deployment, security | 18 |

### Contributing

We need more edge questions — especially from engineers at Tesla, Waymo, Boston Dynamics, and industrial AI companies. See the [question format](../README.md#question-format) and submit a PR.

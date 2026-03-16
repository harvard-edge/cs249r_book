# 📱 Mobile Track — On-Device AI for Smartphones

<div align="center">
  <a href="../README.md">🏠 Playbook Home</a> ·
  <a href="../00_The_Architects_Rubric.md">📋 Rubric</a> ·
  <a href="../cloud/README.md">☁️ Cloud</a> ·
  <a href="../edge/README.md">🤖 Edge</a> ·
  <b>📱 Mobile</b> ·
  <a href="../tinyml/README.md">🔬 TinyML</a>
</div>

---

The Mobile track covers ML systems running on smartphones and tablets — where the model must coexist with the operating system, other apps, and a battery that the user expects to last all day.

### The Constraint Regime

| Dimension | Mobile Reality |
|---|---|
| **Compute** | TOPS shared across CPU, GPU, and NPU (Snapdragon, Apple ANE, MediaTek APU, Samsung NPU) |
| **Memory** | 6–12 GB shared with OS and apps — no dedicated VRAM |
| **Interconnect** | On-chip NoC, no external accelerator bus |
| **Power budget** | 3–5W total device, ML workload gets a fraction |
| **Primary bottleneck** | Battery life and thermal throttling |
| **Failure mode** | Draining the battery, heating the device, jank in the UI thread |

### What Makes Mobile Different from Edge

Edge devices are dedicated to their ML workload. A smartphone is not. The NPU shares silicon and power with the cellular modem, display controller, camera ISP, and whatever else the user is running. Mobile ML is a **resource negotiation problem** — you're always competing for compute, memory, and thermal headroom with the rest of the system.

### Topics That Need Questions

These are the areas where mobile-specific interview questions would be most valuable. Each maps to real interview scenarios at companies like Apple, Google (Android/Pixel), Qualcomm, Samsung, or mobile-first AI startups.

| Topic | What mobile interviews test | Example scenario |
|---|---|---|
| **NPU delegation** | Which ops run on CPU vs GPU vs NPU, operator compatibility | "Your model has 95% NPU-compatible ops but 5% fall back to CPU. What happens to latency?" |
| **Memory pressure** | Model must fit in shared RAM, memory-mapped weights, app lifecycle | "iOS kills your app's background process. How do you handle model reload without a cold-start spike?" |
| **Battery impact** | Energy per inference, sustained vs burst workloads, thermal governors | "Your on-device LLM drains 15% battery per hour. The PM wants 5%. What are your levers?" |
| **Model formats** | CoreML, TFLite, ONNX, QNN — format conversion and operator coverage | "Your PyTorch model uses a custom attention op. How do you get it running on the Apple Neural Engine?" |
| **On-device training** | Federated learning, personalization, differential privacy on-device | "How do you fine-tune a language model on user data without the data leaving the device?" |
| **Latency budgets** | 16ms frame budget (60 FPS), interaction latency, async inference | "Your camera filter model takes 25ms. How do you hit 60 FPS without dropping frames?" |
| **Model delivery** | App size limits, on-demand model download, model versioning | "Your model is 500 MB. The App Store limit is 200 MB for cellular download. What do you do?" |

### The Rounds

| Round | Focus | Questions |
|---|---|---|
| [**1. Mobile Systems & On-Device Inference**](01_Mobile_Systems.md) | NPU delegation, memory pressure, battery, model formats, latency budgets | 9 |
| [**2. Mobile Advanced**](02_Mobile_Advanced.md) | Compute analysis, memory architecture, precision, optimization, deployment, privacy | 18 |

### Contributing

We need more mobile questions — especially from engineers at Apple, Google, Qualcomm, and Samsung. See the [question format](../README.md#question-format) and submit a PR.

# The Physical AI Kit & Hardware Laboratories

This directory contains the firmware contracts, starter checkpoints, hardware wiring specifications, and laboratory execution guides for the **Physical AI Kit** accompanying ***Physical AI: Machine Learning Systems That Sense and Act***.


## The Physical AI Kit (Arduino UNO Q Dual-Brain Reference Platform)

The **Physical AI Kit** is a zero-magic, dual-brain embedded platform engineered specifically to expose real-world systems realities—tail latency ($P_{99}$), memory bus contention, clock skew, thermal limits, and real-time safety enforcement.

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ THE PHYSICAL AI KIT │
│ (Arduino UNO Q Dual-Brain Reference Platform) │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ │
│ ┌──────────────────────────────────┐ ┌──────────────────────────────────────┐ │
│ │ HOST BRAIN: LINUX MPU │ │ REFLEX BRAIN: REAL-TIME MCU │ │
│ │ (Cognitive Cortex) │ │ (Safety Permission Authority) │ │
│ │ • Quad-core Application Proc │ │ • 32-bit ARM Cortex-M4 Micro │ │
│ │ • Gigabytes Shared DRAM (UMA) │ │ • Zero-Dynamic Allocation (TCM) │ │
│ │ • Workloads: Encoders, VLMs, │ │ • Workloads: 1 kHz CBF Enforcer, │ │
│ │ Diffusion Action Chunks │ │ Dynamic Stopping $d_{\text{stop}}$, │ │
│ │ • Role: Untrusted Proposals │ │ Watchdog Leases, Interlock Relay │ │
│ │ • Frequency: 0.5 Hz – 50 Hz │ │ • Frequency: 1000 Hz Hard Loop │ │
│ └─────────────────┬────────────────┘ └──────────────────┬───────────────────┘ │
│ │ │ │
│ └─────────────────┐ ┌─────────────────┘ │
│ ▼ ▼ │
│ ┌───────────────────────────────┐ │
│ │ HETEROGENEOUS SHARED SRAM │ │
│ │ • Lock-free ring buffers │ │
│ │ • Atomic sequence counters │ │
│ │ • Expiring intent leases │ │
│ └───────────────┬───────────────┘ │
│ │ │
│ ════════════════════════════════════╪═════════════════════════════════════════════ │
│ SENSOR INGESTION SUITE │ ACTUATOR & SAFETY SUBSYSTEM │
│ • MIPI CSI-2 Camera (DMA rings) │ • Precision Multi-Axis Motion Stage │
│ • Optical Quadrature Encoders │ • 3-Phase Gate Drivers & Shunt Sense │
│ • 6-DoF Inertial Measurement Unit ▼ • Hardware Emergency Power Interlock │
│ PHYSICAL REALITY │
└────────────────────────────────────────────────────────────────────────────────────────┘
```


## Pedagogical Division of Ownership

The laboratory curriculum connects platform-neutral theory to bench execution:

| Role | Responsibility & Deliverables |
| :--- | :--- |
| **Course Lecturer / Book Author**<br>*(Prof. Vijay Janapa Reddi)* | Owns the pedagogical spine, chapter concepts, and formal **Lab Contracts** (`CONTRACT.md`): learning objectives, target physical phenomena, mathematical formulations, fault injection regimes, and design dossier decision gates. |
| **Kit & Studio Lead**<br>*(Dr. Andrea Mattia Garavagno)* | Owns the **Physical AI Kit** hardware realization: PCB pinouts, power rail isolation, MPU Linux environments, MCU FreeRTOS/bare-metal starter checkpoints, assembly schematics, and bench validation. |
| **Student / Practitioner** | Executes the experiments, measures latency distributions and physical telemetry, diagnoses systems failures, makes architectural trade-offs, and updates the versioned **engineering notebook**. |


## The Signature Dual-Brain Invariant: Propose vs. Permit

The dual-brain split is not a decorative software abstraction—it is a physical and architectural firewall:

1. **The Linux MPU Proposes (Cognitive Cortex):** Runs high-capacity perception models (ViT, DINOv2), multimodal reasoning (VLMs), and trajectory decoders (ACT Action Chunking). It operates under best-effort Linux scheduling and emits typed, timestamped, **expiring intent leases** ($p_t$ with TTL $t_{\text{expire}}$).
2. **The Real-Time MCU Permits (Reflex & Safety Enforcer):** Runs a zero-allocation $1000\text{ Hz}$ bare-metal / FreeRTOS control loop. It verifies Control Barrier Functions ($h(x) \ge 0$), evaluates dynamic stopping distance ($d_{\text{stop}}$), services hardware watchdogs, and commands the 3-phase motor bridge.
3. **No Direct Actuator Access:** No MPU user-space process, cloud API, or Python script possesses direct electrical authority to toggle gate drivers. All physical consequences require MCU permission: $u_t = \text{permit}(p_t)$.
4. **Crash Invariance:** If the Linux MPU experiences an uncaught exception, `SIGKILL`, or kernel panic, the MCU hardware watchdog trips within $50\text{ ms}$ and clamps the actuator power rail to safe de-energization.


## The 14-Week Laboratory Spine

The hardware studio track follows the 14-week course syllabus ([`course/syllabus.md`](../course/syllabus.md)), incrementally transforming the raw **Physical AI Kit** into a certified, autonomous **Physical Agent**:

| Week | Milestone | Lab Directory | Core Systems Focus | Notebook checkpoint |
| :---: | :--- | :--- | :--- | :---: |
| **W1** | **Kit Bring-Up** | `00-kit-bringup/` | Board bring-up, inter-processor link, and safe idle | Hardware Bring-Up |
| **W2** | **Causal Boundary** | `01-close-the-loop/` | Advisory mode vs. closed-loop physical state mutation | loop charter |
| **W3** | **Freshness & Tails** | `02-freshness-wall/`<br>`03-measure-both-brains/`| Information age ($\Delta t$), $P_{99}$ latency tails, and memory bus contention | requirements ledger |
| **W4** | **Runtime Engine** | `04-runtime-fault-containment/` | Multi-rate IPC, seqlocks, and MPU crash survival | runtime skeleton |
| **W5** | **Vision Ingestion** | `05-perception-frontier/` | MIPI CSI-2 DMA ring buffers and spatial tokens | observation contract |
| **W6** | **Spatial Memory** | `06-belief-drift/` | $SE(3)$ frame graphs, clock drift, and TTL belief leases | state and timing model |
| **W7** | **Semantic Intent** | `07-two-speed-intent/` | Edge VLM bounding boxes and expiring intent leases | intent schema |
| **W8** | **Safety Enforcer** | `08-mcu-enforcer/` | **Signature Lab:** 1 kHz MCU Control Barrier Function vetoes | enforcement design |
| **W9** | **Silicon Placement**| `09-placement-ripple/` | Heterogeneous resource ledger (FLOPs, SRAM, Watts, QoS) | placement ledger |
| **W10**| **Faults & Lineage** | `10-shadow-and-faults/`<br>`11-authority-paths/` | Bumpless joystick override and shadow runtime auditing | authority design |
| **W11**| **Release Gate** | `12-learning-turn/`<br>`13-ship-gate/` | Cross-layer seeded fault injection and safety case | release case |
| **W12–14**| **Capstone Jury**| `99-design-review/` | Live unannounced fault defense and jury release verdict | **Final Release** |


## Directory Structure per Lab

Each laboratory directory follows a standardized contract-first structure:

```text
NN-slug/
├── CONTRACT.md # Pedagogical specification (learning goals, phenomenon, math, dossier gate)
├── README.md # Student-facing step-by-step bench guide & hardware wiring schematics
├── mpu/ # Linux application code (Python, TensorRT, PyTorch, IPC endpoints)
├── mcu/ # Real-time microcontroller firmware (C/C++, FreeRTOS, CBF enforcer)
├── checkpoint/ # Known-good starter firmware and configuration snapshots
└── evidence/ # Reference oscilloscope captures, latency histograms, and dossier logs
```


## Shared Firmware & Schemas

The [`labs/shared/`](shared/) directory contains production-grade, reusable headers and schemas shared across all labs:
* **Lock-Free Shared Memory IPC:** Layouts and atomic sequence counters (`ipc_ring.h`).
* **Typed Proposal Schemas:** Intent lease and telemetry message schemas (`intent_lease.h`).
* **Safety Invariant Enforcers:** Control Barrier Function quadratic program solvers and stopping distance estimators (`cbf_enforcer.h`).
* **Hardware Metrology Utilities:** Hardware GPIO profiling and PTP timestamp synchronization utilities.

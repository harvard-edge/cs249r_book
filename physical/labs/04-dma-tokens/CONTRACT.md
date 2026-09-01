# Lab Contract: DMA Ingestion and 3D Affordance Tokens

## Identity

- **Lab ID:** lab
- **Book chapter:** Chapter 4 (*Perception and Spatial Encoders*)
- **Title:** DMA Ingestion, UMA Bus Contention, and 3D Spatial Affordance Tokens
- **Status:** accepted

## Pedagogy (Book Steward)

- **Phenomenon made visible:** Pre-inference ingestion tax and memory bus contention: how scaling camera DMA resolution from 480p to 1080p and unmanaged `memcpy()` thrashing starves NPU weight streaming and real-time control cache lines.
- **Learner prediction (before running):** Increasing camera resolution will improve image sharpness without affecting neural policy inference latency.
- **Controlled perturbation:** Stream raw camera video at 480p @ 15 Hz, 720p @ 60 Hz, and 1080p @ 30 Hz with and without zero-copy DMA coherent ring buffer allocation.
- **Negative control / alternative:** Naive user-space `cv2.VideoCapture` with standard POSIX buffer copies vs. kernel-pinned `dma_alloc_coherent` zero-copy ring buffers.
- **Metric, units, regime, efficacy floor:** Memory bus bandwidth (MB/s), L3 cache miss rate (%), sense-to-token latency ($P_{50}, P_{99}$ in ms), PTP hardware timestamp skew ($\mu\text{s}$). Efficacy floor: $P_{99} \le 20\text{ ms}$ under full 720p @ 60 Hz streaming.
- **Chapter-native failure to diagnose:** Memory bus starvation causing $P_{99}$ inference blowout from $22\text{ ms}$ to $135\text{ ms}$, tripping the watchdog safety veto.
- **Engineering decision:** Lock sensor operating resolution to 720p @ 60 Hz / 1080p @ 30 Hz with zero-copy DMA ring buffers and AXI QoS prioritization.
- **Dossier artifact update:** observation contract (Observation Contract & Token Schema).

## Dual-Brain Responsibilities

- **MPU owns:** Kernel DMA ring buffer driver, V4L2 zero-copy stream, INT8 DINOv2 / ViT spatial token encoder on NPU, metric 3D unprojection via $\mathbf{K}^{-1}$, emission of 3D spatial affordance tokens.
- **MCU owns:** Hardware PTP / timer master clock, GPIO `FSYNC` camera exposure trigger strobe, SPI IMU high-rate sampling ($1000\text{ Hz}$), timestamp coherence audit, dynamic stopping distance check $d_{\text{stop}}$.
- **Messages crossing the boundary:** Spatial affordance token array $o_t$ with 64-bit PTP hardware timestamps, frame sequence IDs, and $SE(3)$ grasp frames via shared SRAM mailbox.
- **What must remain enforceable if the MPU fails:** If the camera stream stalls or latency exceeds $40\text{ ms}$, the MCU drops spatial trust, maintains blind proprioceptive IMU/encoder state propagation, and commands active Category 1 deceleration.

## Realization Constraints

- **Sensors / actuators required:** Arduino UNO Q Dual-Brain Kit, MIPI CSI-2 RGB-D / global-shutter camera module, 6-axis IMU over SPI, logic analyzer / oscilloscope for GPIO strobe verification.
- **Optional vs required hardware:** Required: Arduino UNO Q board and camera. Optional: external logic analyzer for sub-microsecond PTP validation.
- **Starter checkpoint input:** lab (Multi-Rate IPC Mailboxes).
- **Acceptance test on physical kit:** Run `python3 test_ingestion_qos.py`. Verify zero dropped frames over 10,000 cycles, $P_{99} \le 20\text{ ms}$, and PTP timestamp skew $\le 500\,\mu\text{s}$.
- **Analytical / hosted fallback:** Simulated camera DMA trace and synthetic memory bus contention injector running on Linux PREEMPT_RT kernel.

## Out of Scope

- **Action Chunking Decoders:** Multi-step trajectory decoders (owned by Chapter 7).
- **Latent World Models:** Temporal JEPA world model latent belief states (owned by Chapter 5).
- **Multimodal Language Reasoning:** VLM natural language intent reasoning (owned by Chapter 6).
- **Real-Time Safety Solvers:** 1 kHz Control Barrier Function QP solvers (owned by Chapter 8).

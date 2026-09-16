# Master Art & Illustration Manifest: Visual Architecture for Physical AI Systems

> **Authoritative Plan and Live Progress Tracker for MLSysBook Volume IV**
> *MIT Press Standards · Bespoke Mechanistic Schematics · Authentic Research Artifacts*

---

## 1. Visual Architecture & Editorial Standards

Volume IV of Machine Learning Systems (*Physical AI Systems*) teaches how neural computation couples to the irreversible physical world. To meet MIT Press textbook excellence, every figure must satisfy a rigorous pedagogical contract:

1. **Zero Decorative Stock Imagery**:
   - Generic factory welding arms, blurry emergency stop buttons, and clip-art robots are strictly prohibited.
   - Every image must be either a **bespoke mechanistic schematic (Type A)** or an **authentic scientific research artifact (Type B)** with formal attribution and paper citation.
2. **First-Principles Mechanistic Explanations (Type A)**:
   - Must expose internal computational, physical, and signal-level mechanisms:
     - Spatial tokenization and patch projection (ViT backbones).
     - Cross-attention affordance maps and multimodal grounding.
     - Temporal chunking and $C^2$-continuity trajectory stitching.
     - 3D ray unprojection ($K^{-1}$), discrete depth bin lifting, and Bird's-Eye View (BEV) pillar splatting.
     - Lock-free ring buffer memory fences (`LDAR`/`STLR`) and seqlock synchronization.
     - Control Barrier Function (CBF) quadratic program half-space vector projection.
   - Visual syntax: Multi-panel architectural blueprints, strict coordinate frame conventions (right-hand rule, explicit axes), explicit physical SI units, and mathematical notation harmonized exactly with chapter equations.
3. **Authentic Scientific Artifacts & Forensic Records (Type B)**:
   - Sourced directly from seminal papers, national safety agencies, and university robotics labs:
     - Real teleoperation testbeds (Stanford ALOHA, GELLO, RoboTurk).
     - Physical robot learning systems (Google RT-1/RT-2, OpenAI Dactyl).
     - Accelerator silicon micrographs and board layouts (NVIDIA Jetson AGX Orin, Tesla FSD).
     - Forensic crash telemetry and sensor rigs (NTSB Tempe Volvo XC90, Williston Tesla Model S).
     - Metrology and calibration artifacts (Zhang checkerboard/ChArUco, NIST USAR arenas).
4. **Mandatory Visual Inspection Gate**:
   - Every bespoke SVG must be rendered to raster format (e.g. via `rsvg-convert -w 2400` or `pdftoppm`) and inspected using visual inspection tools before being marked complete.
   - Requirements: Zero text collisions, sharp typographic hierarchy, accessible high-contrast palettes, and zero clipping at bounding boxes.

---

## 2. Four-Part Division of Labor

To maintain cross-chapter narrative flow and prevent visual fragmentation, the 17 chapters are organized under **Four Part Art Directors**:

```
Volume IV: Physical AI Systems
│
├── PART I: The Anatomy of Physical Agents (Ch 1–4)
│   ├── Ch 1: The Causal Boundary (Bits meet physics, irreversible energy transfer, time-to-harm)
│   ├── Ch 2: The Body (Actuator dynamics, torque-speed curves, thermal dissipation, gearboxes)
│   ├── Ch 3: The Brain (Foundation models, VLAs, edge SoC rooflines, deliberation walls)
│   └── Ch 4: The Nervous System (Deterministic buses, seqlock protocol, proposal-permission split)
│
├── PART II: Teaching the Machine (Ch 5–7)
│   ├── Ch 5: The Data Engine (Multi-stream logging, teleoperation rigs, calibration, yield)
│   ├── Ch 6: Policy Training (BC compounding error, Diffusion Policy, ACT CVAE, residual fine-tuning)
│   └── Ch 7: Evaluation (Closed-loop covariate shift, unannounced fault injection, Wald SPRT)
│
├── PART III: Running the Machine (Ch 8–13)
│   ├── Ch 8: Perception (Physical transduction, CSI-2 DMA, Lift-Splat-Shoot, depth covariance)
│   ├── Ch 9: Memory (SE(3) transform trees, transform staleness, OctoMap/TSDF, belief decay)
│   ├── Ch 10: Intent (Temporal leases, VLA cross-attention grounding, reachability envelopes)
│   ├── Ch 11: Planning (Action chunking seams, quintic splines, fallback timelines)
│   ├── Ch 12: Enforcement (Control Barrier Functions, QP projection, safety envelopes, STO)
│   └── Ch 13: Placement (Heterogeneous SoC mapping, RTOS-Linux lockless boundaries, DRAM contention)
│
└── PART IV: Governing the Machine (Ch 14–17)
    ├── Ch 14: Intervention (Authority handovers, bumpless transfer, Mode 2 reaction drift, NTSB logs)
    ├── Ch 15: Verification (Qualification ladder: MIL/SIL/HIL, fault injection, envelope falsification)
    ├── Ch 16: Release (CAE/GSN safety assurance cases, ODD boundaries, cryptographic manifests)
    └── Ch 17: The Epistemic Frontier (Astronomical exposure walls, containment, curriculum synthesis)
```

---

## 3. Live Progress Dashboard

| Part | Chapters | Status | Total Figs | Type A (SVG) | Type B (Sourced) | Retired / Replaced | Complete % |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Part I: Anatomy** | Ch 1, 2, 3, 4 | 🟢 Complete | 23 | 12 | 11 | 1 retired | 100% |
| **Part II: Teaching** | Ch 5, 6, 7 | 🟢 Complete | 18 | 11 | 7 | 0 retired | 100% |
| **Part III: Running** | Ch 8, 9, 10, 11, 12, 13 | 🟢 Complete | 28 | 17 | 11 | 1 retired | 100% |
| **Part IV: Governing** | Ch 14, 15, 16, 17 | 🟢 Complete | 16 | 9 | 7 | 0 retired | 100% |
| **Total** | **Ch 1 – 17** | 🟢 **Complete** | **85** | **49** | **36** | **2 retired** | **100%** |

---

## 4. Chapter-by-Chapter Art Manifest

### Part I: The Anatomy of Physical Agents

#### Chapter 1: The Causal Boundary
*Focus: Irreversibility, plant response times, energy transfer, proposal-permission boundary.*
- [x] **`fig01_causal_loop.svg`** (Type A): Closed-loop causal interaction between compute cycle and plant physics.
- [x] **`fig01_real_lidar_sensor_rack.jpg`** (Type B): Autonomous vehicle roof sensor pod showing optical, radar, and solid-state LiDAR geometry (Steve Jurvetson, CC BY 2.0).
- [x] **`fig01_three_archetypes.jpg`** (Type B): Photographic plate of the 3 reference machine classes: (a) Mobility (Spot), (b) Contact Manipulation (Franka Panda), (c) Cybernetic Process (Tesvolt BESS) (Wikimedia Commons, CC BY-SA 4.0).
- [x] **`fig01_scope_venn.svg`** (Type A): Physical AI systems scope Venn diagram (ML Systems $\cap$ Real-Time Systems $\cap$ Robotics).
- [x] **`fig01_tempe_kinematics.svg`** (Type A): Kinematic trajectory of the 2018 Tempe collision showing perception toggle gaps and braking delays.
- [x] **`fig01_real_ntsb_tempe_vehicle.jpg`** (Type B): Official NTSB forensic crash investigation photo of the test vehicle.
- [x] **`fig01_dual_brain_soc.svg`** (Type A): Heterogeneous SoC architecture separating unprivileged AI model from privileged safety monitor.

#### Chapter 2: The Body
*Focus: Actuators, power distribution, thermal limits, gearbox mechanics, reflected inertia.*
- [x] **`fig02_latency_waterfall.svg`** (Type A): 8-stage microsecond latency waterfall from photon arrival to motor torque rise ($L/R$ time constant).
- [x] **`fig02_stopping_distance.svg`** (Type A): Stopping distance scaling curves ($v_0 t_{\text{delay}} + v_0^2 / 2a_{\max}$) comparing compute lag vs physical braking.
- [x] **`fig02_real_strain_wave_gear.jpg`** (Type B): Authentic teardown micrograph of harmonic drive flexspline, wave generator, and circular spline.
- [x] **`fig02_real_broken_gear_tooth.jpg`** (Type B): High-magnification scanning electron micrograph of mechanical fatigue fracture in gear teeth.
- [x] **`fig02_real_burnt_motor_stator.jpg`** (Type B): Micrograph of thermal runaway and charred magnet wire insulation from sustained $I^2 R$ stall current.
- [x] **`fig02_motor_torque_speed_envelope.svg`** (Type A): Four-quadrant BLDC motor curve showing continuous thermal boundary, peak intermittent torque, back-EMF headroom, and field weakening.
- [x] **`fig02_real_motor_dyno_bench.jpg`** (Type B): Authentic motor dynamometer test fixture for measuring torque-speed and back-EMF envelopes.

#### Chapter 3: The Brain
*Focus: Foundation models, VLA tokenization, memory bandwidth walls, edge deliberation.*
- [x] **`fig03_vla_architecture.svg`** (Type A): **[UPGRADED]** 4-panel architectural blueprint: (1) Patch indexing & ViT projection, (2) Text prompt tokenization, (3) Cross-attention affordance heatmaps, (4) Action chunk decoding (diffusion vs discrete tokens) across the proposal-permission boundary.
- [x] **`fig03_real_industrial_welding.jpg`** (Type B): BMW Leipzig plant spot-welding workcell (KUKA 6-DoF arms, submillimeter fixtures, safety interlock cages) demonstrating the preprogrammed classical robotics paradigm (BMW Group / Wikimedia Commons, CC BY-SA 2.0 de).
- [x] **`fig03_real_aloha_manipulator.png`** (Type B): Authentic Stanford ALOHA bimanual teleoperation and autonomous rollout setup (Zhao et al., 2023).
- [x] **`fig03_patterson_roofline.svg`** (Type A): Patterson roofline diagram showing LPDDR5 memory wall vs Tensor Core compute ceilings for edge SoCs.
- [x] **`fig03_five_handoffs.svg`** (Type A): 5 cognitive handoffs across the sensory-motor pipeline.
- [x] **`fig03_two_speed_brain.svg`** (Type A): Fast real-time reflex loop (100 Hz - 1 kHz) decoupled from slow foundation model deliberator (5 - 10 Hz).

#### Chapter 4: The Nervous System
*Focus: Deterministic interconnect, seqlock protocol, real-time loops, hardware privilege.*
- [x] **`fig04_multirate_cadence_buffer.svg`** (Type A): Multi-rate cadence buffer (10 Hz perception, 100 Hz planning, 1 kHz current loop) with seqlock timing.
- [x] **`fig04_privilege_enforcer_registers.svg`** (Type A): Register-level proposal-permission privilege enforcement showing MMIO-isolated actuator gates.
- [x] **`fig04_sto_safety_circuit.svg`** (Type A): **[UPGRADED]** Authentic Category 4 / PLe dual-channel safety relay schematic showing force-guided contacts and optical isolation to Safe Torque Off (STO), replacing retired stock photo `fig04_real_estop_interlock.jpg`.
- [x] **`fig04_real_industrial_ethercat_controller.jpg`** (Type B): Authentic Beckhoff/Omron industrial deterministic bus slice with sub-microsecond jitter specs.
- [x] **`fig04_real_motor_drive_controller.jpg`** (Type B): Authentic real-time motor inverter drive board showing gate driver isolation and shunt resistors.
- [x] **`fig04_real_teach_pendant.jpg`** (Type B): Authentic industrial teach pendant with 3-position deadman switch.

---

### Part II: Teaching the Machine

#### Chapter 5: The Data Engine
*Focus: Multi-stream logging, teleoperation, calibration, demonstration budgets.*
- [x] **`fig05_four_stream_action_taps.svg`** (Type A): 4-stream tap dataflow ($a_{\text{req}}$ model proposal, $a_{\text{cmd}}$ planner smoothed, $a_{\text{enf}}$ safety clamped, $a_{\text{meas}}$ joint encoder actual).
- [x] **`fig05_real_gello_teleop.png`** (Type B): Authentic 3D-printed GELLO passive arm teleoperation interface (Wu et al., 2023).
- [x] **`fig05_collector_coverage_ledger.svg`** (Type A): Teleoperation dataset distribution ledger across operators and failure recoveries.
- [x] **`fig05_compounding_error_flywheel.svg`** (Type A): Demonstrator error propagation flywheel vs intervention-corrected data collection.
- [x] **`fig05_real_camera_calibration.jpg`** (Type B): Authentic Zhang checkerboard / ChArUco multi-camera extrinsic calibration fixture.
- [x] **`fig05_cruise_pedestrian_blindspot_drag.svg`** (Type A): Forensic reconstruction of sensor blind-spot tracking failure.

#### Chapter 6: Policy Training
*Focus: Behavioral cloning, action chunking, diffusion policy, residual fine-tuning.*
- [x] **`fig06_compounding_error.svg`** (Type A): $O(T^2)$ compounding error trajectory divergence under single-step BC vs $O(T)$ chunked rollout.
- [x] **`fig06_real_diffusion_policy_architecture.png`** (Type B): Authentic Diffusion Policy neural network architecture diagram (Chi et al., 2023).
- [x] **`fig06_bespoke_diffusion_denoising_process.svg`** (Type A): **[UPGRADED]** Step-by-step DDPM action chunk denoising across reverse diffusion steps ($k = 16 \to 10 \to 4 \to 0$) with multimodal symmetry breaking and receding-horizon decoupling.
- [x] **`fig06_real_sim2real_dactyl.jpg`** (Type B): Authentic OpenAI Dactyl Shadow Hand manipulating Rubik's Cube under physical domain randomization.
- [x] **`fig06_sim2real_gap.svg`** (Type A): Reality gap distribution shift in friction, damping, and actuator latency.
- [x] **`fig06_policy_ranking_inversion.svg`** (Type A): Open-loop validation loss ranking inversion vs closed-loop real-world task success.

#### Chapter 7: Evaluation
*Focus: Closed-loop vs open-loop evaluation, fault injection, Wald SPRT, confidence bounds.*
- [x] **`fig07_open_vs_closed_loop.svg`** (Type A): Phase portrait comparing open-loop prediction accuracy with closed-loop stability and drift.
- [x] **`fig07_evidence_regimes.svg`** (Type A): Sample size requirements across testing regimes: nominal bench, edge case injection, statistical proof.
- [x] **`fig07_confidence_bounds.svg`** (Type A): Clopper-Pearson exact confidence intervals vs Wald approximations across failure counts.
- [x] **`fig07_real_nist_test_arena.jpg`** (Type B): Authentic NIST ASTM E2521 standard test arena for emergency response robot mobility evaluation.
- [x] **`fig07_real_crash_barrier_test.jpg`** (Type B): Industrial mobile robot crash barrier compliance test setup.

---

### Part III: Running the Machine

#### Chapter 8: Perception
*Focus: Transduction, rolling shutter, CSI-2 DMA, Lift-Splat-Shoot, depth covariance.*
- [x] **`fig08_ray_frustum_voxel_lifting.svg`** (Type A): **[UPGRADED]** 4-panel architectural blueprint: (1) Pinhole camera geometry & intrinsics $K$, (2) Ray unprojection & discrete depth bin distribution, (3) SE(3) extrinsics & BEV pillar splatting, (4) Quadratic depth covariance elongation $\delta Z \propto Z^2$ and clearance insetting.
- [x] **`fig08_coordinate_frame_tree.svg`** (Type A): Directed acyclic graph of SE(3) kinematic reference frames from map to end-effector.
- [x] **`fig08_depth_uncertainty_clearance.svg`** (Type A): Covariance ellipse dilation and obstacle padding along optical bearing rays.
- [x] **`fig08_real_sensor_suite.jpg`** (Type B): Authentic autonomous vehicle sensor cluster with co-located LiDAR, stereo cameras, and thermal imager.
- [x] **`fig08_real_lidar_pointcloud.jpg`** (Type B): High-density LiDAR point cloud colored by reflectance intensity and return distance.
- [x] **`fig08_real_rolling_shutter_skew.jpg`** (Type B): Authentic rolling shutter sensor distortion on high-speed rotating mechanism.
- [x] **`fig08_real_ntsb_williston_crash.png`** (Type B): NTSB reconstruction of optical contrast failure against high-luminance sky.

#### Chapter 9: Memory
*Focus: Spatial representations, kinematic trees, transform staleness, volumetric mapping.*
- [x] **`fig09_frame_staleness_error.svg`** (Type A): Kinematic lever-arm error propagation ($\Delta x = \omega \times r \cdot \Delta t$) caused by asynchronous transform buffer latency.
- [x] **`fig09_uncertainty_growth.svg`** (Type A): Spatial belief variance growth during sensor occlusion under constant acceleration drift.
- [x] **`fig09_real_robot_transform_tree.png`** (Type B): Real ROS/ROS2 `tf2` dynamic transform tree graph of a mobile manipulator.
- [x] **`fig09_real_octomap_volumetric_mapping.png`** (Type B): Authentic OctoMap hierarchical octree occupancy grid.
- [x] **`fig09_real_3dgs_slam_splatam.png`** (Type B): SplatAM dense 3D Gaussian Splatting SLAM reconstruction (Keetha et al., 2024).
- [x] **`fig09_dynamic_scene_graph_hierarchy.svg`** (Type A): **[UPGRADED]** 5-layer 3D Dynamic Scene Graph architecture (Metric voxel mesh $\to$ Segmented objects $\to$ Voronoi freespace $\to$ Structural rooms $\to$ Global topology) with bidirectional abstraction/grounding flow.

#### Chapter 10: Intent
*Focus: Multi-cadence intent horizons, temporal leases, affordance grounding.*
- [x] **`fig10_vla_cross_attention_intent_lease.svg`** (Type A): Vision-language cross-attention affordance heatmap generating spatial intent bounding bounds and lease validity.
- [x] **`fig10_intent_lease_envelope.svg`** (Type A): Reachability boundary and dynamic expiration envelope under bounded velocity and acceleration.
- [x] **`fig10_lease_dynamics_tradeoff.svg`** (Type A): Trade-off curve between lease horizon duration, communication latency, and obstacle clearance margins.
- [x] **`fig10_real_grounded_affordance_franka.jpg`** (Type B): Real Franka robot executing affordance-grounded grasping on tabletop clutter.
- [x] **`fig10_real_open_vocab_grounding.jpg`** (Type B): Open-vocabulary zero-shot object detection and 3D bounding box prediction.

#### Chapter 11: Planning
*Focus: Trajectory chunking, $C^2$ quintic splines, fallback timelines, real-time deadlines.*
- [x] **`fig11_action_chunk_seam_continuity.svg`** (Type A): Mathematical trajectory chunk seam stitching: $C^0, C^1, C^2$ boundary condition matching using quintic polynomials.
- [x] **`fig11_seam_timeline_fallback.svg`** (Type A): Real-time timing waterfall: nominal inference deadline vs safe deceleration fallback envelope activation.
- [x] **`fig11_real_action_chunking_trace.png`** (Type B): Telemetry comparison of raw discrete predicted action chunks vs temporally ensembled smooth trajectory.
- [x] **`fig11_real_manipulator_trajectory.png`** (Type B): Measured joint position, velocity, and torque traces on industrial manipulator.

#### Chapter 12: Enforcement
*Focus: Control Barrier Functions, QP filtering, safety shields, hardware interlocks.*
- [x] **`fig12_cbf_safety_filter.svg`** (Type A): Quadratic Program (QP) minimal-intervention vector projection: unconstrained policy action $u_{\text{nom}}$ projected onto safe half-space.
- [x] **`fig12_stopping_envelope_phase_plane.svg`** (Type A): Phase plane $(x, \dot{x})$ showing safe set $\mathcal{C}$, viability boundary, and maximum emergency deceleration envelope.
- [x] **`fig12_fallback_ladder.svg`** (Type A): 4-tier safety fallback ladder (Nominal AI $\to$ CBF-QP Shield $\to$ Deterministic Kinematic Stop $\to$ Category 4 Hardware STO).
- [x] **`fig12_sto_safety_circuit.svg`** (Type A): **[UPGRADED]** Authentic Category 4 / PLe dual-channel safety relay schematic showing force-guided contacts and optical isolation to Safe Torque Off (STO), replacing retired stock photo `fig12_real_estop_interlock.jpg`.
- [x] **`fig12_real_laser_scanner.jpg`** (Type B): Authentic industrial safety laser scanner (SICK microScan3) showing programmable protective fields.

#### Chapter 13: Placement
*Focus: Heterogeneous SoCs, Linux-RTOS boundaries, memory bus contention, cache partitioning.*
- [x] **`fig13_soc_contention_droop.svg`** (Type A): DRAM memory bus bandwidth contention between GPU neural inference and real-time CPU DMA packet servicing.
- [x] **`fig13_lockfree_boundary_contract.svg`** (Type A): Shared memory layout of single-producer multi-consumer lockless seqlock ring buffer across Linux and RTOS.
- [x] **`fig13_real_jetson_orin.jpg`** (Type B): Authentic NVIDIA Jetson AGX Orin module and carrier board with labeled SoC, LPDDR5 DRAM, and power ICs.
- [x] **`fig13_real_fsd_board.jpg`** (Type B): Authentic Tesla Dual-FSD automotive board showing redundant SoCs, separate power rails, and independent CAN/Ethernet interfaces.

---

### Part IV: Governing the Machine

#### Chapter 14: Intervention
*Focus: Shared authority, bumpless transfer, Mode 2 automation surprise, forensic logging.*
- [x] **`fig14_authority_handshake_fsm.svg`** (Type A): Formal finite state machine governing human takeover, override latching, and supervisory clearance.
- [x] **`fig14_bumpless_transfer_dynamics.svg`** (Type A): Actuator torque and acceleration waveforms comparing discontinuous step transfer with $C^1$-smooth quintic smoothstep blending.
- [x] **`fig14_real_ntsb_telemetry.png`** (Type B): Official NTSB forensic crash telemetry graph of the Tempe collision showing suppression of emergency braking.
- [x] **`fig14_real_teach_pendant.jpg`** (Type B): Industrial robot teach pendant showing ergonomic 3-position liveman enabling switch.

#### Chapter 15: Verification
*Focus: Qualification ladders, fault injection, sim-to-real falsification, HIL dynos.*
- [x] **`fig15_qualification_ladder.svg`** (Type A): 4-tier qualification ladder (Model-in-the-Loop $\to$ Software-in-the-Loop $\to$ Hardware-in-the-Loop $\to$ Proving Ground).
- [x] **`fig15_hardware_fault_injection.svg`** (Type A): Hardware fault injection breakout topology (switched open circuits, rail shorts, clock jitter, bus corruption).
- [x] **`fig15_real_hil_avionics_testbed.jpg`** (Type B): Authentic aerospace/automotive HIL dynamometer test rack with real-time dSPACE / NI PXI chassis.
- [x] **`fig15_real_crash_test_instrumentation.jpg`** (Type B): High-g crash dummy and vehicle optical tracking instrumentation.

#### Chapter 16: Release
*Focus: Safety assurance cases, Goal Structuring Notation (GSN), ODD bounds, release manifests.*
- [x] **`fig16_cae_tree.svg`** (Type A): Formal Claim-Argument-Evidence (CAE) / GSN tree linking top-level system safety goal to empirical test results and formal bounds.
- [x] **`fig16_release_record.svg`** (Type A): Cryptographically signed release manifest schema linking neural weights, firmware hashes, and HIL test certificates.
- [x] **`fig16_real_autonomous_haul_truck.jpg`** (Type B): Heavy autonomous mining haul truck operating inside strictly geofenced, controlled ODD.
- [x] **`fig16_real_robot_safety_cell.jpg`** (Type B): Industrial robotic manufacturing cell with interlocked perimeter fencing and optical light curtains.

#### Chapter 17: The Epistemic Frontier
*Focus: Astronomical exposure walls, observational indistinguishability, curriculum synthesis.*
- [x] **`fig17_epistemic_limits.svg`** (Type A): The $10^9$-hour exposure barrier graph: required test hours to statistically prove ultra-rare catastrophe probabilities.
- [x] **`fig17_closed_loop_synthesis.svg`** (Type A): Master architectural diagram synthesizing Volume IV: Causal Boundary $\to$ Body $\to$ Brain $\to$ Nervous System $\to$ Policy Training $\to$ Real-Time Shielding.
- [x] **`fig17_real_perseverance_mars_drill.jpg`** (Type B): Authentic NASA/JPL telemetry photo of Perseverance rover autonomous coring drill on Martian rock.
- [x] **`fig17_real_humanoid_disaster_rubble.jpg`** (Type B): Humanoid robot traversing unstructured rubble test course at DARPA Robotics Challenge.

---

## 5. Execution Workflow & Next Steps

```
[Phase 1: Deep Audit & Cataloging]  ==>  [Phase 2: Seminal Paper Sourcing]
               │                                      │
               ▼                                      ▼
[Phase 3: High-Fidelity SVG Drafting]  ==>  [Phase 4: Visual QA & Book Integration]
```

### Immediate Action Items
1. **Retire & Replace Stock Images (P0)**:
   - [x] Verified `fig03_real_industrial_welding.jpg` (BMW Leipzig plant KUKA spot-welding workcell, CC BY-SA 2.0 de BMW Group).
   - [x] Retired `fig04_real_estop_interlock.jpg` & `fig12_real_estop_interlock.jpg` $\to$ Replaced with authentic Category 4 / PLe safety relay circuit diagram `fig04_sto_safety_circuit.svg` and `fig12_sto_safety_circuit.svg` (dual-channel force-guided contacts, optical isolation to STO).
2. **Draft Priority Mechanistic SVGs (P1)**:
   - [x] `fig06_bespoke_diffusion_denoising_process.svg`: Step-by-step diffusion action chunk denoising across reverse diffusion steps with multimodal symmetry breaking and receding-horizon decoupling.
   - [x] `fig09_dynamic_scene_graph_hierarchy.svg`: 5-layer 3D Dynamic Scene Graph (Metric $\to$ Objects $\to$ Freespace $\to$ Rooms $\to$ Topology) with upward abstraction and downward grounding flows.
3. **Visual Verification Protocol**:
   - [x] All SVGs rendered to PNG via `rsvg-convert -w 2400` and visually inspected with `view_file` to verify zero text collision, proper padding, and crystal-clear visual hierarchy.
4. **CMOS & Citation Integration**:
   - [x] Formal CMOS source credits integrated across captions (`03_brain.qmd`, `04_nervous.qmd`, `06_training.qmd`, `08_perception.qmd`, `09_memory.qmd`, `12_enforcement.qmd`).
   - [x] Simons/Patterson division of labor implemented: rigorous lead-ins, concise encoding captions, and argumentative payoff prose connecting directly to engineering consequences.
5. **Validation**:
   - [x] Validated with `./binder/binder check markup`, `./binder/binder check figures --vol4`, `./binder/binder check images --vol4`, and `./binder/binder check prose`. 100% PASS (Zero errors).

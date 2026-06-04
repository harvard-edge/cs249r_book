# Lab Deep Migration Checklist

This checklist tracks the remaining work after the catalog-wide baseline migration.
The baseline state is complete: every catalog lab has a track plan, four canonical
track variants, and a notebook-level track/report surface. The remaining work is
the deeper pass where each lab's actual parts, plots, numbers, reports, and
reflections become track-specific.

Canonical tracks:
- Mobile: iPhone
- Tiny: Oura Ring
- Edge: RoboTaxi
- Cloud: Cloud Fleet

Source-of-truth rule:
- Do not add hardware, model, system, cost, memory, latency, energy, or fleet facts
  as notebook-local constants unless they are temporary UI labels.
- If a fact is computed or displayed as evidence, put it in MLSysIM or a shared
  `mlsysbook_labs` registry/helper first.
- Notebook code may compose facts, run solvers, render plots, and collect student
  decisions, but the underlying reference data should be shared.

## Global Completion Gates

- [x] Worktree exists at `/Users/VJ/GitHub/MLSysBook-labs` on branch `codex/labs`.
- [x] Canonical track registry exists in `mlsysbook_labs`.
- [x] Oura Ring and RoboTaxi hardware entries exist in MLSysIM.
- [x] Every catalog lab has a `*.track-plan.md` file.
- [x] Every catalog lab has baseline variants for all four tracks.
- [x] Every catalog lab has a notebook-level track/report surface.
- [x] Static tests guard track plan coverage and track/report surface coverage.
- [ ] Define shared plot/modality catalog for reusable lab evidence displays.
- [ ] Define shared per-part section helpers for:
  - what students need to know,
  - prediction lock,
  - track scenario,
  - computed evidence,
  - checkpoint reflection,
  - source trace,
  - report payload.
- [ ] Add shared report schema checks for required fields across all deep-migrated labs.
- [ ] Add shared tests that every deep-migrated lab's variants resolve all referenced
  hardware, model, system, and infrastructure refs.
- [ ] Add browser-facing smoke coverage for at least one representative lab per track.
- [ ] Keep `labs/LAB_IMPLEMENTATION_NOTES.md` updated after every lab slice.

## Per-Lab Definition Of Done

For each lab:

- [ ] Read the current notebook and the matching `*.track-plan.md`.
- [ ] Identify the core pedagogical spine: the one decision students should learn to make.
- [ ] Decide which parts are invariant across tracks and which parts should change.
- [ ] Replace baseline variant text with lab-specific variants where needed.
- [ ] Add or reuse source-of-truth facts in MLSysIM or `mlsysbook_labs`.
- [ ] Add or reuse solver/helper APIs for computed evidence.
- [ ] Wire selected track into the notebook's parts, plots, tables, and narrative.
- [ ] Ensure every part has a consistent internal structure:
  - Objective
  - What you need to know
  - Prediction or design choice
  - Track-specific scenario
  - Computed evidence
  - Checkpoint reflection
- [ ] Ensure the synthesis/report includes:
  - selected track,
  - predictions,
  - computed evidence summary,
  - final decision,
  - reflection,
  - residual risk,
  - source trace.
- [ ] Add or update tests for any shared helper, solver, registry, or report contract.
- [ ] Run focused tests plus static checks.
- [ ] Commit the lab or coherent small batch.
- [ ] Mark checklist items complete.

## Recommended Implementation Order

- [x] 1. V2-11 Edge Intelligence: closest to device-specific constraints.
- [x] 2. V1-11 Hardware Roofline: pressure-tests hardware registry and roofline plots.
- [x] 3. V2-10 Inference Economy: connects latency, cost, batching, and deployment target.
- [x] 4. V1-12 Benchmarking Trap: standardizes benchmark plots and report evidence.
- [x] 5. V1-13 Tail Latency Trap: extends serving and SLA evidence.
- [x] 6. V1-14 Silent Degradation: adds monitoring and drift evidence.
- [x] 7. V2-06 Collective Communication: upgrade existing rich wrapper to canonical tracks.
- [x] 8. Finish remaining Volume I labs by dependency order.
- [x] 9. Finish remaining Volume II labs by dependency order.
- [ ] 10. Run catalog-wide QA and final cleanup.

## Volume I Labs

### V1-00 - The Architect's Portal

Path: `labs/vol1/lab_00_introduction.py`

Current status:
- [x] Deep orientation structure exists.
- [x] Track selection exists.
- [x] Local report export exists.
- [x] Track profile references come from `mlsysbook_labs`.

Remaining tasks:
- [x] Replace any remaining notebook-local track display facts with profile-derived fields.
- [x] Confirm the track picker writes a ledger value consumed by all later labs.
- [x] Add a regression test that Lab 00 exposes all four canonical track IDs.
- [x] Confirm the downloaded report includes track ID, hardware ref, system ref, and source policy.
- [x] Update the track-plan file if Lab 00 becomes the canonical place for student track selection.

### V1-01 - The AI Triad

Path: `labs/vol1/lab_01_ml_intro.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [x] Map Data, Algorithm, and Machine axes to each canonical track.
- [x] Add track-specific D-A-M examples:
  - iPhone: on-device vision or text classification under memory and latency limits.
  - Oura Ring: always-on sensing under energy and SRAM/flash limits.
  - RoboTaxi: perception stack under safety and real-time latency limits.
  - Cloud Fleet: hosted model service under throughput, cost, and reliability limits.
- [x] Move any displayed device/model facts to shared registries.
- [x] Replace generic baseline scenario text with per-track narrative.
- [x] Add a small track-specific constraint table.
- [x] Update report evidence to include the selected D-A-M bottleneck.
- [x] Add tests for any new helper that summarizes track constraints.

### V1-02 - Physics of Deployment

Path: `labs/vol1/lab_02_ml_systems.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [x] Tie the deployment physics lesson to track-specific resource budgets.
- [x] Add a source-of-truth helper for deployment envelope summaries if one does not exist.
- [x] Compute or display memory, latency, energy, or cost only through shared APIs.
- [x] Give each track a different binding constraint:
  - iPhone: thermal/latency envelope.
  - Oura Ring: battery and memory envelope.
  - RoboTaxi: deterministic latency and sensor throughput.
  - Cloud Fleet: cost, utilization, and failure domain.
- [x] Update parts so the same conceptual lesson produces different track conclusions.
- [x] Add report fields for binding physical constraint and mitigation.

### V1-03 - Constraint Tax

Path: `labs/vol1/lab_03_ml_workflow.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [x] Identify where the workflow creates hidden constraint tax for each track.
- [x] Add shared workflow-stage descriptors if they are reused by later labs.
- [x] Add track-specific examples for data collection, validation, deployment, and monitoring.
- [x] Make the final decision differ by track rather than only by generic workflow stage.
- [x] Ensure the report captures the most expensive workflow constraint and the mitigation plan.
- [x] Add tests for any shared workflow or ledger serialization helper.

### V1-04 - Data Gravity

Path: `labs/vol1/lab_04_data_engr.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [x] Define track-specific data source, data rate, retention, and privacy assumptions.
- [x] Move reusable data-rate or storage assumptions to shared registries/helpers.
- [x] Add evidence for where data should be processed:
  - iPhone: local preprocessing vs upload.
  - Oura Ring: summary features vs raw sensor streams.
  - RoboTaxi: local sensor fusion vs fleet upload.
  - Cloud Fleet: warehouse/lake/feature-store placement.
- [x] Update plots/tables to show storage, bandwidth, or freshness tradeoffs.
- [x] Update report evidence with selected data placement and residual data risk.

### V1-05 - Activation Tax

Path: `labs/vol1/lab_05_nn_compute.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [x] Tie neural computation concepts to per-track activation memory and compute budgets.
- [x] Add shared helper for activation footprint if not already available.
- [x] Use model refs from the shared variant registry.
- [x] Show how batch size, precision, and activation shape affect each track.
- [x] Add track-specific failure state:
  - iPhone: thermal or memory pressure.
  - Oura Ring: SRAM/energy overflow.
  - RoboTaxi: latency miss.
  - Cloud Fleet: utilization/cost miss.
- [x] Include activation-footprint evidence in the report.

### V1-06 - Architecture Tax

Path: `labs/vol1/lab_06_nn_arch.py`

Current status:
- [x] Deep track-aware architecture migration installed.

Deep migration tasks:
- [x] Connect architecture choices to device and service constraints.
- [x] Add or reuse model architecture descriptors in `mlsysbook_labs`.
- [x] Compare track-appropriate model families instead of one generic model list.
- [x] Show why a model that is accurate in one track fails in another.
- [x] Add report fields for architecture choice, rejected alternatives, and dominant constraint.
- [x] Add tests for model-family registry lookups if introduced.

### V1-07 - Framework Tax

Path: `labs/vol1/lab_07_ml_frameworks.py`

Current status:
- [x] Deep track-aware framework/runtime migration installed.

Deep migration tasks:
- [x] Make framework/runtime choice track-specific.
- [x] Add shared runtime/deployment-target catalog if needed.
- [x] Track examples:
  - iPhone: Core ML style deployment and operator support.
  - Oura Ring: TFLite Micro or MCU-oriented runtime.
  - RoboTaxi: TensorRT/accelerated edge runtime.
  - Cloud Fleet: server runtime with batching and observability.
- [x] Display operator coverage, packaging, and portability tradeoffs from shared facts.
- [x] Update report with selected runtime and deployment risk.

### V1-08 - Training Gauntlet

Path: `labs/vol1/lab_08_model_train.py`

Current status:
- [x] Deep track-aware training migration installed.

Deep migration tasks:
- [x] Separate training environment from deployment environment for each track.
- [x] Add shared training budget/cost helper if needed.
- [x] Track examples:
  - iPhone: train centrally, personalize lightly on device if appropriate.
  - Oura Ring: train centrally, deploy tiny model, maybe adapt thresholds.
  - RoboTaxi: simulation/fleet retraining loop.
  - Cloud Fleet: large-scale distributed training and evaluation.
- [x] Show compute, data, and evaluation bottlenecks by track.
- [x] Update report with training strategy and deployment handoff risk.

### V1-09 - Selection Paradox

Path: `labs/vol1/lab_09_data_selection.py`

Current status:
- [x] Deep track-aware data-selection migration installed.

Deep migration tasks:
- [x] Make data/model selection criteria depend on track constraints.
- [x] Add shared candidate-selection helper if useful across labs.
- [x] Include track-specific tradeoffs between accuracy, robustness, latency, memory, and cost.
- [x] Ensure plots compare candidates with track-specific feasibility boundaries.
- [x] Update report with selected candidate, rejected candidate, and reason.
- [x] Add tests for candidate feasibility logic if introduced.

### V1-10 - Compression Paradox

Path: `labs/vol1/lab_10_model_compress.py`

Current status:
- [x] Deep pilot migration exists.
- [x] Compression candidate solver exists in MLSysIM.
- [x] Track-specific variants exist.
- [x] Report synthesis includes computed evidence and reflection fields.

Remaining tasks:
- [ ] Audit device precision facts, especially iPhone int8 support, and decide whether to add
  missing hardware capabilities to MLSysIM.
- [ ] Confirm all four track variants have hand-authored guardrails, validation tests, and residual risks.
- [ ] Add a visual regression or data-shape test for the compression evidence table if useful.
- [ ] Use V1-10 as the reference pattern for later deep migrations.

### V1-11 - Hardware Roofline

Path: `labs/vol1/lab_11_hw_accel.py`

Current status:
- [x] Deep track-aware notebook migration installed.
- [x] Track selector, track context, source trace, and local report export installed.
- [x] Shared roofline helper backs GEMM, fusion, and roofline evidence.
- [x] All displayed hardware roofline facts resolve through canonical track variants and MLSysIM refs.

Deep migration tasks:
- [x] Build track-specific roofline evidence using hardware registry facts.
- [x] Confirm no missing hardware capabilities are needed for this slice:
  - peak ops,
  - memory bandwidth,
  - memory capacity,
  - accelerator type,
  - supported precision.
- [x] Add shared roofline helper if current calculations are notebook-local.
- [x] Track examples:
  - iPhone: Neural Engine/GPU/CPU boundary.
  - Oura Ring: MCU/DSP-style tiny compute boundary.
  - RoboTaxi: edge accelerator throughput and deterministic latency.
  - Cloud Fleet: GPU roofline and utilization.
- [x] Update report with compute-bound vs memory-bound diagnosis.
- [x] Add tests for roofline helper and hardware ref resolution.

### V1-12 - Benchmarking Trap

Path: `labs/vol1/lab_12_perf_bench.py`

Current status:
- [x] Deep track-aware notebook migration installed.
- [x] Track selector, track context, source trace, and local report export installed.
- [x] Shared benchmarking helper backs Amdahl, sustained benchmark, multi-metric, and tail-latency evidence.

Deep migration tasks:
- [x] Standardize benchmark modalities for latency, throughput, memory, energy, and cost.
- [x] Add shared benchmark-result schema if needed.
- [x] Give each track a different "bad benchmark" trap.
- [x] Add track-specific benchmark plots and failure states.
- [x] Ensure report captures benchmark setup, misleading metric, corrected metric, and conclusion.
- [x] Add tests for benchmark schema/report serialization.

### V1-13 - Tail Latency Trap

Path: `labs/vol1/lab_13_model_serving.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [x] Tie serving choices to track-specific SLA and request pattern.
- [x] Add shared latency distribution helper if current logic is notebook-local.
- [x] Track examples:
  - iPhone: interactive on-device response.
  - Oura Ring: periodic background inference.
  - RoboTaxi: hard real-time perception budget.
  - Cloud Fleet: p95/p99 service SLO.
- [x] Display tail latency distributions and mitigation options by track.
- [x] Update report with SLA, tail driver, mitigation, and residual risk.
- [x] Add tests for latency helper/report fields.

### V1-14 - Silent Degradation

Path: `labs/vol1/lab_14_ml_ops.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [x] Make monitoring signals and remediation options track-specific.
- [x] Add shared drift/degradation scenario catalog if useful.
- [x] Track examples:
  - iPhone: app version/device OS drift.
  - Oura Ring: sensor placement or physiology drift.
  - RoboTaxi: weather/geography drift.
  - Cloud Fleet: traffic mix or upstream data drift.
- [x] Add plots for metric drift, alert thresholds, and false alarms.
- [x] Update report with monitoring plan, trigger, action, and residual risk.
- [x] Add tests for operations helper/report fields.

### V1-15 - No Free Fairness

Path: `labs/vol1/lab_15_responsible_engr.py`

Current status:
- [x] Deep track-aware responsibility structure installed.
- [x] Track selector, source trace, ledger save, and local report export installed.

Deep migration tasks:
- [x] Make fairness/responsibility tradeoffs specific to each deployment context.
- [x] Add shared metric-policy helper if the lab needs comparable definitions.
- [x] Tie track scenarios to realistic stakeholders and harm models.
- [x] Ensure any group metric examples come from a shared synthetic dataset/helper.
- [x] Update report with selected metric, tradeoff, mitigation, and residual risk.
- [x] Add tests for report fields and synthetic dataset shape if introduced.

### V1-16 - The Architect's Audit

Path: `labs/vol1/lab_16_ml_conclusion.py`

Current status:
- [x] Deep track-aware capstone structure installed.
- [x] Track selector, source trace, ledger replay, sensitivity audit, memo report export installed.

Deep migration tasks:
- [x] Make the capstone synthesize the selected track across prior Volume I decisions.
- [x] Add ledger reader/helper if cross-lab decisions need normalized access.
- [x] Track examples should summarize the student's chosen device/service constraints.
- [x] Add final architecture report fields that reference prior lab evidence.
- [x] Confirm missing ledger entries degrade gracefully.
- [x] Add tests for any ledger summary helper.

## Volume II Labs

### V2-01 - The Scale Illusion

Path: `labs/vol2/lab_01_introduction.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Reframe scale as track-specific rather than only cloud-scale.
- [x] Add shared scale-envelope helper if reused.
- [x] Track examples:
  - iPhone: millions of installed devices.
  - Oura Ring: always-on fleet telemetry with tiny payloads.
  - RoboTaxi: city-scale autonomous fleet operations.
  - Cloud Fleet: GPU/service fleet scaling.
- [x] Update parts so scale changes reliability, coordination, and cost differently by track.
- [x] Update report with the selected scale failure mode.

### V2-02 - The Compute Infrastructure Wall

Path: `labs/vol2/lab_02_compute_infra.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Tie compute infrastructure limits to track-specific hardware/system refs.
- [x] Add missing infrastructure facts to MLSysIM if displayed.
- [x] Compare compute, bandwidth, memory, and utilization limits by track.
- [x] Show where adding more compute stops helping.
- [x] Update report with infrastructure bottleneck and mitigation.
- [x] Add tests for any shared compute infrastructure helper.

### V2-03 - Network Fabric Design

Path: `labs/vol2/lab_03_communication.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Make communication fabric meaningful for every track.
- [x] Add or reuse network/fabric refs in MLSysIM.
- [x] Track examples:
  - iPhone: device-to-cloud uplink constraints.
  - Oura Ring: BLE/mobile relay payload constraints.
  - RoboTaxi: vehicle-edge-cloud synchronization.
  - Cloud Fleet: east-west datacenter fabric.
- [x] Display bandwidth, latency, payload, and retry tradeoffs.
- [x] Update report with selected communication strategy and residual risk.

### V2-04 - The Data Pipeline Wall

Path: `labs/vol2/lab_04_data_storage.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Connect data storage and freshness to each track.
- [x] Add shared pipeline/storage budget helper if needed.
- [x] Track examples:
  - iPhone: privacy-preserving local cache and upload policy.
  - Oura Ring: compressed time-series summaries.
  - RoboTaxi: high-volume sensor logs and incident upload.
  - Cloud Fleet: feature store and training data lake.
- [x] Add plots/tables for storage growth, freshness, and bandwidth.
- [x] Update report with data pipeline decision and residual risk.

### V2-05 - The Parallelism Puzzle

Path: `labs/vol2/lab_05_dist_train.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Make parallelism choice relevant to track deployment lifecycle.
- [x] Distinguish training-time parallelism from inference-time deployment target.
- [x] Add shared parallelism strategy descriptors if useful.
- [x] Track examples:
  - iPhone/Oura Ring/RoboTaxi: centralized or federated training feeding edge deployment.
  - Cloud Fleet: data/model/pipeline parallelism in service of large model training.
- [x] Update report with training strategy, scaling bottleneck, and residual risk.
- [x] Add tests for any shared parallelism helper.

### V2-06 - Collective Communication

Path: `labs/vol2/lab_06_collective_communication.py`

Current status:
- [x] Rich shared wrapper exists.
- [x] Baseline track/report panel installed.
- [x] MLSysIM collective communication physics functions are already used.

Deep migration tasks:
- [x] Upgrade the existing rich wrapper to canonical track variants.
- [x] Decide which tracks use true collectives and which use communication analogs.
- [x] Track examples:
  - iPhone: federated update aggregation payloads.
  - Oura Ring: tiny update/sensor summaries through a phone relay.
  - RoboTaxi: fleet update synchronization and map/model rollout.
  - Cloud Fleet: GPU collective algorithms.
- [x] Add source-of-truth communication payload assumptions if displayed.
- [x] Ensure the report uses canonical track profile and variant fields.
- [x] Add tests for track-specific communication scenario selection.

### V2-07 - When Failure Is Routine

Path: `labs/vol2/lab_07_fault_tolerance.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Make failure modes and recovery decisions track-specific.
- [x] Add shared reliability/failure-budget helper if needed.
- [x] Track examples:
  - iPhone: offline mode, app crashes, OS updates.
  - Oura Ring: battery depletion, sensor dropout, sync gaps.
  - RoboTaxi: sensor faults, degraded mode, safety fallback.
  - Cloud Fleet: node failures and retry storms.
- [x] Display failure-rate, recovery-time, or availability evidence.
- [x] Update report with failure budget and recovery decision.

### V2-08 - The Scheduling Trap

Path: `labs/vol2/lab_08_fleet_orch.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Connect scheduling/orchestration to each track's fleet shape.
- [x] Add shared fleet/orchestration descriptors if needed.
- [x] Track examples:
  - iPhone: staged rollout and device eligibility.
  - Oura Ring: background job timing and battery-aware scheduling.
  - RoboTaxi: vehicle dispatch, update windows, and safety constraints.
  - Cloud Fleet: GPU scheduling and bin packing.
- [x] Display utilization, queueing, rollout, or availability tradeoffs.
- [x] Update report with scheduling policy and residual risk.

### V2-09 - The Optimization Trap

Path: `labs/vol2/lab_09_perf_engineering.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Make optimization targets differ by track.
- [x] Add shared performance-counter or bottleneck taxonomy helper if useful.
- [x] Track examples:
  - iPhone: optimize latency without thermal regression.
  - Oura Ring: optimize energy without missing sensing events.
  - RoboTaxi: optimize latency while preserving safety margin.
  - Cloud Fleet: optimize throughput/cost without p99 regression.
- [x] Update plots to show local optimum vs system optimum.
- [x] Update report with chosen optimization and unintended side effect.

### V2-10 - The Inference Economy

Path: `labs/vol2/lab_10_inference.py`

Current status:
- [x] Deep track-aware notebook migration installed.
- [x] Track selector, track context, source trace, and local report export installed.
- [x] Shared inference-economy helper backs cost crossover, state/cache capacity, batching, and serving-plan evidence.

Deep migration tasks:
- [x] Build track-specific inference economics.
- [x] Add shared inference cost/latency helper if current logic is notebook-local.
- [x] Track examples:
  - iPhone: local inference cost is battery/thermal/UX.
  - Oura Ring: local inference cost is energy and duty cycle.
  - RoboTaxi: local inference cost is latency and safety margin.
  - Cloud Fleet: service inference cost is dollars, utilization, and p99.
- [x] Display batching, quantization, caching, or placement tradeoffs by track.
- [x] Update report with inference placement and economic constraint.
- [x] Add tests for cost/latency helper.

### V2-11 - The Edge Thermodynamics Lab

Path: `labs/vol2/lab_11_edge_intelligence.py`

Current status:
- [x] Hand-authored variants exist.
- [x] Baseline track/report panel installed.
- [x] Deep track-aware notebook migration installed.

Deep migration tasks:
- [x] Make this the first full device-track deep migration after V1-10.
- [x] Move any displayed device battery, memory, latency, or energy facts into MLSysIM/shared helpers.
- [x] Add a shared edge energy or duty-cycle helper if needed.
- [x] Wire each notebook part to the selected canonical track.
- [x] Track examples:
  - [x] iPhone: on-device adaptation under thermal and battery limits.
  - [x] Oura Ring: always-on sensing and tiny inference energy budget.
  - [x] RoboTaxi: edge perception and vehicle compute envelope.
  - [x] Cloud Fleet: compare edge offload against centralized inference.
- [x] Add track-specific plots for memory, battery/energy, and update payloads.
- [x] Update report with edge placement decision and thermodynamic residual risk.
- [x] Add tests for helper math and variant field completeness.

### V2-12 - The Silent Fleet

Path: `labs/vol2/lab_12_ops_scale.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Make fleet observability and operations track-specific.
- [x] Add shared fleet health/telemetry helper if needed.
- [x] Track examples:
  - iPhone: app/device version segments.
  - Oura Ring: sensor quality and sync coverage.
  - RoboTaxi: route/geography/weather fleet slices.
  - Cloud Fleet: service shards, regions, and model versions.
- [x] Display fleet health dashboard evidence by track.
- [x] Update report with monitoring slice, action threshold, and residual risk.

### V2-13 - The Price of Privacy

Path: `labs/vol2/lab_13_security_privacy.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Make security/privacy controls track-specific.
- [x] Add shared privacy/security control catalog if needed.
- [x] Track examples:
  - iPhone: local processing and permission boundary.
  - Oura Ring: health-derived sensor data and consent.
  - RoboTaxi: location/video logs and incident retention.
  - Cloud Fleet: tenant isolation and data governance.
- [x] Display accuracy, latency, cost, or utility impact of privacy controls.
- [x] Update report with selected privacy control and residual risk.

### V2-14 - The Robustness Budget

Path: `labs/vol2/lab_14_robust_ai.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Tie robustness budget to track-specific failure consequences.
- [x] Add shared robustness scenario/helper if needed.
- [x] Track examples:
  - iPhone: varied lighting/device/user context.
  - Oura Ring: sensor noise and physiological variation.
  - RoboTaxi: weather, occlusion, and out-of-distribution scenes.
  - Cloud Fleet: prompt/user/data distribution shift.
- [x] Display robustness-cost tradeoffs.
- [x] Update report with robustness budget and unresolved hazard.

### V2-15 - The Carbon Budget

Path: `labs/vol2/lab_15_sustainable_ai.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Make sustainability accounting track-specific.
- [x] Add shared energy/carbon helper if the lab displays computed carbon evidence.
- [x] Track examples:
  - iPhone: battery energy and charging externality.
  - Oura Ring: tiny battery and lifecycle duty cycle.
  - RoboTaxi: vehicle compute energy and fleet update cadence.
  - Cloud Fleet: datacenter energy, utilization, and region mix.
- [x] Display energy/carbon tradeoffs with source trace.
- [x] Update report with carbon budget decision and residual risk.

### V2-16 - The Fairness Budget

Path: `labs/vol2/lab_16_responsible_ai.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, and local report export installed.

Deep migration tasks:
- [x] Make fairness tradeoffs specific to each deployment track.
- [x] Reuse or extend the fairness metric helper from V1-15 if created.
- [x] Track examples:
  - iPhone: accessibility and device/user variation.
  - Oura Ring: physiological and demographic variation.
  - RoboTaxi: neighborhood, pedestrian, weather, and safety exposure.
  - Cloud Fleet: service quality across user cohorts/regions.
- [x] Display fairness/utility/safety tradeoffs.
- [x] Update report with fairness budget, mitigation, and residual risk.

### V2-17 - The Fleet Synthesis

Path: `labs/vol2/lab_17_fleet_synthesis.py`

Current status:
- [x] Deep shared system-design renderer installed.
- [x] Typed system-design variants cover all canonical tracks.
- [x] Track selector, source trace, decision frontier, scaling curve, reflection, ledger save, prior Volume II decision summary, and local report export installed.

Deep migration tasks:
- [x] Make this the Volume II capstone for the selected canonical track.
- [x] Add or reuse ledger summary helper to gather prior Volume II decisions.
- [x] Track examples should synthesize scale, infrastructure, communication, failure,
  operations, privacy, robustness, sustainability, and responsibility.
- [x] Confirm missing ledger entries degrade gracefully.
- [x] Generate a final fleet architecture report.
- [x] Add tests for fleet synthesis report schema and ledger fallback.

## Final Catalog QA

- [x] Run `python3 -m pytest labs/tests/test_static.py -q`.
- [x] Run all shared helper tests in `labs/tests`.
- [ ] Run relevant MLSysIM tests separately from `labs/tests`.
- [x] Verify `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` is rebuilt after shared helper changes.
- [ ] Verify `wheels/mlsysim-0.1.2-py3-none-any.whl` is rebuilt after MLSysIM changes.
- [ ] Verify every lab's report can be generated locally.
- [ ] Verify each track has at least one representative visual/evidence modality.
- [ ] Verify no lab embeds source-of-truth hardware/model/system facts locally.
- [ ] Commit final cleanup.

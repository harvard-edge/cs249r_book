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

- [ ] 1. V2-11 Edge Intelligence: closest to device-specific constraints.
- [ ] 2. V1-11 Hardware Roofline: pressure-tests hardware registry and roofline plots.
- [ ] 3. V2-10 Inference Economy: connects latency, cost, batching, and deployment target.
- [ ] 4. V1-12 Benchmarking Trap: standardizes benchmark plots and report evidence.
- [ ] 5. V1-13 Tail Latency Trap: extends serving and SLA evidence.
- [ ] 6. V1-14 Silent Degradation: adds monitoring and drift evidence.
- [ ] 7. V2-06 Collective Communication: upgrade existing rich wrapper to canonical tracks.
- [ ] 8. Finish remaining Volume I labs by dependency order.
- [ ] 9. Finish remaining Volume II labs by dependency order.
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
- [ ] Replace any remaining notebook-local track display facts with profile-derived fields.
- [ ] Confirm the track picker writes a ledger value consumed by all later labs.
- [ ] Add a regression test that Lab 00 exposes all four canonical track IDs.
- [ ] Confirm the downloaded report includes track ID, hardware ref, system ref, and source policy.
- [ ] Update the track-plan file if Lab 00 becomes the canonical place for student track selection.

### V1-01 - The AI Triad

Path: `labs/vol1/lab_01_ml_intro.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Map Data, Algorithm, and Machine axes to each canonical track.
- [ ] Add track-specific D-A-M examples:
  - iPhone: on-device vision or text classification under memory and latency limits.
  - Oura Ring: always-on sensing under energy and SRAM/flash limits.
  - RoboTaxi: perception stack under safety and real-time latency limits.
  - Cloud Fleet: hosted model service under throughput, cost, and reliability limits.
- [ ] Move any displayed device/model facts to shared registries.
- [ ] Replace generic baseline scenario text with per-track narrative.
- [ ] Add a small track-specific constraint table.
- [ ] Update report evidence to include the selected D-A-M bottleneck.
- [ ] Add tests for any new helper that summarizes track constraints.

### V1-02 - Physics of Deployment

Path: `labs/vol1/lab_02_ml_systems.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Tie the deployment physics lesson to track-specific resource budgets.
- [ ] Add a source-of-truth helper for deployment envelope summaries if one does not exist.
- [ ] Compute or display memory, latency, energy, or cost only through shared APIs.
- [ ] Give each track a different binding constraint:
  - iPhone: thermal/latency envelope.
  - Oura Ring: battery and memory envelope.
  - RoboTaxi: deterministic latency and sensor throughput.
  - Cloud Fleet: cost, utilization, and failure domain.
- [ ] Update parts so the same conceptual lesson produces different track conclusions.
- [ ] Add report fields for binding physical constraint and mitigation.

### V1-03 - Constraint Tax

Path: `labs/vol1/lab_03_ml_workflow.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Identify where the workflow creates hidden constraint tax for each track.
- [ ] Add shared workflow-stage descriptors if they are reused by later labs.
- [ ] Add track-specific examples for data collection, validation, deployment, and monitoring.
- [ ] Make the final decision differ by track rather than only by generic workflow stage.
- [ ] Ensure the report captures the most expensive workflow constraint and the mitigation plan.
- [ ] Add tests for any shared workflow or ledger serialization helper.

### V1-04 - Data Gravity

Path: `labs/vol1/lab_04_data_engr.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Define track-specific data source, data rate, retention, and privacy assumptions.
- [ ] Move reusable data-rate or storage assumptions to shared registries/helpers.
- [ ] Add evidence for where data should be processed:
  - iPhone: local preprocessing vs upload.
  - Oura Ring: summary features vs raw sensor streams.
  - RoboTaxi: local sensor fusion vs fleet upload.
  - Cloud Fleet: warehouse/lake/feature-store placement.
- [ ] Update plots/tables to show storage, bandwidth, or freshness tradeoffs.
- [ ] Update report evidence with selected data placement and residual data risk.

### V1-05 - Activation Tax

Path: `labs/vol1/lab_05_nn_compute.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Tie neural computation concepts to per-track activation memory and compute budgets.
- [ ] Add shared helper for activation footprint if not already available.
- [ ] Use model refs from the shared variant registry.
- [ ] Show how batch size, precision, and activation shape affect each track.
- [ ] Add track-specific failure state:
  - iPhone: thermal or memory pressure.
  - Oura Ring: SRAM/energy overflow.
  - RoboTaxi: latency miss.
  - Cloud Fleet: utilization/cost miss.
- [ ] Include activation-footprint evidence in the report.

### V1-06 - Architecture Tax

Path: `labs/vol1/lab_06_nn_arch.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Connect architecture choices to device and service constraints.
- [ ] Add or reuse model architecture descriptors in `mlsysbook_labs`.
- [ ] Compare track-appropriate model families instead of one generic model list.
- [ ] Show why a model that is accurate in one track fails in another.
- [ ] Add report fields for architecture choice, rejected alternatives, and dominant constraint.
- [ ] Add tests for model-family registry lookups if introduced.

### V1-07 - Framework Tax

Path: `labs/vol1/lab_07_ml_frameworks.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make framework/runtime choice track-specific.
- [ ] Add shared runtime/deployment-target catalog if needed.
- [ ] Track examples:
  - iPhone: Core ML style deployment and operator support.
  - Oura Ring: TFLite Micro or MCU-oriented runtime.
  - RoboTaxi: TensorRT/accelerated edge runtime.
  - Cloud Fleet: server runtime with batching and observability.
- [ ] Display operator coverage, packaging, and portability tradeoffs from shared facts.
- [ ] Update report with selected runtime and deployment risk.

### V1-08 - Training Gauntlet

Path: `labs/vol1/lab_08_model_train.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Separate training environment from deployment environment for each track.
- [ ] Add shared training budget/cost helper if needed.
- [ ] Track examples:
  - iPhone: train centrally, personalize lightly on device if appropriate.
  - Oura Ring: train centrally, deploy tiny model, maybe adapt thresholds.
  - RoboTaxi: simulation/fleet retraining loop.
  - Cloud Fleet: large-scale distributed training and evaluation.
- [ ] Show compute, data, and evaluation bottlenecks by track.
- [ ] Update report with training strategy and deployment handoff risk.

### V1-09 - Selection Paradox

Path: `labs/vol1/lab_09_data_selection.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make data/model selection criteria depend on track constraints.
- [ ] Add shared candidate-selection helper if useful across labs.
- [ ] Include track-specific tradeoffs between accuracy, robustness, latency, memory, and cost.
- [ ] Ensure plots compare candidates with track-specific feasibility boundaries.
- [ ] Update report with selected candidate, rejected candidate, and reason.
- [ ] Add tests for candidate feasibility logic if introduced.

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
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Build track-specific roofline evidence using hardware registry facts.
- [ ] Add missing hardware capabilities to MLSysIM if roofline needs them:
  - peak ops,
  - memory bandwidth,
  - memory capacity,
  - accelerator type,
  - supported precision.
- [ ] Add shared roofline helper if current calculations are notebook-local.
- [ ] Track examples:
  - iPhone: Neural Engine/GPU/CPU boundary.
  - Oura Ring: MCU/DSP-style tiny compute boundary.
  - RoboTaxi: edge accelerator throughput and deterministic latency.
  - Cloud Fleet: GPU roofline and utilization.
- [ ] Update report with compute-bound vs memory-bound diagnosis.
- [ ] Add tests for roofline helper and hardware ref resolution.

### V1-12 - Benchmarking Trap

Path: `labs/vol1/lab_12_perf_bench.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Standardize benchmark modalities for latency, throughput, memory, energy, and cost.
- [ ] Add shared benchmark-result schema if needed.
- [ ] Give each track a different "bad benchmark" trap.
- [ ] Add track-specific benchmark plots and failure states.
- [ ] Ensure report captures benchmark setup, misleading metric, corrected metric, and conclusion.
- [ ] Add tests for benchmark schema/report serialization.

### V1-13 - Tail Latency Trap

Path: `labs/vol1/lab_13_model_serving.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Tie serving choices to track-specific SLA and request pattern.
- [ ] Add shared latency distribution helper if current logic is notebook-local.
- [ ] Track examples:
  - iPhone: interactive on-device response.
  - Oura Ring: periodic background inference.
  - RoboTaxi: hard real-time perception budget.
  - Cloud Fleet: p95/p99 service SLO.
- [ ] Display tail latency distributions and mitigation options by track.
- [ ] Update report with SLA, tail driver, mitigation, and residual risk.
- [ ] Add tests for latency helper/report fields.

### V1-14 - Silent Degradation

Path: `labs/vol1/lab_14_ml_ops.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make monitoring signals and remediation options track-specific.
- [ ] Add shared drift/degradation scenario catalog if useful.
- [ ] Track examples:
  - iPhone: app version/device OS drift.
  - Oura Ring: sensor placement or physiology drift.
  - RoboTaxi: weather/geography drift.
  - Cloud Fleet: traffic mix or upstream data drift.
- [ ] Add plots for metric drift, alert thresholds, and false alarms.
- [ ] Update report with monitoring plan, trigger, action, and residual risk.

### V1-15 - No Free Fairness

Path: `labs/vol1/lab_15_responsible_engr.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make fairness/responsibility tradeoffs specific to each deployment context.
- [ ] Add shared metric-policy helper if the lab needs comparable definitions.
- [ ] Tie track scenarios to realistic stakeholders and harm models.
- [ ] Ensure any group metric examples come from a shared synthetic dataset/helper.
- [ ] Update report with selected metric, tradeoff, mitigation, and residual risk.
- [ ] Add tests for report fields and synthetic dataset shape if introduced.

### V1-16 - The Architect's Audit

Path: `labs/vol1/lab_16_ml_conclusion.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make the capstone synthesize the selected track across prior Volume I decisions.
- [ ] Add ledger reader/helper if cross-lab decisions need normalized access.
- [ ] Track examples should summarize the student's chosen device/service constraints.
- [ ] Add final architecture report fields that reference prior lab evidence.
- [ ] Confirm missing ledger entries degrade gracefully.
- [ ] Add tests for any ledger summary helper.

## Volume II Labs

### V2-01 - The Scale Illusion

Path: `labs/vol2/lab_01_introduction.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Reframe scale as track-specific rather than only cloud-scale.
- [ ] Add shared scale-envelope helper if reused.
- [ ] Track examples:
  - iPhone: millions of installed devices.
  - Oura Ring: always-on fleet telemetry with tiny payloads.
  - RoboTaxi: city-scale autonomous fleet operations.
  - Cloud Fleet: GPU/service fleet scaling.
- [ ] Update parts so scale changes reliability, coordination, and cost differently by track.
- [ ] Update report with the selected scale failure mode.

### V2-02 - The Compute Infrastructure Wall

Path: `labs/vol2/lab_02_compute_infra.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Tie compute infrastructure limits to track-specific hardware/system refs.
- [ ] Add missing infrastructure facts to MLSysIM if displayed.
- [ ] Compare compute, bandwidth, memory, and utilization limits by track.
- [ ] Show where adding more compute stops helping.
- [ ] Update report with infrastructure bottleneck and mitigation.
- [ ] Add tests for any shared compute infrastructure helper.

### V2-03 - Network Fabric Design

Path: `labs/vol2/lab_03_communication.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make communication fabric meaningful for every track.
- [ ] Add or reuse network/fabric refs in MLSysIM.
- [ ] Track examples:
  - iPhone: device-to-cloud uplink constraints.
  - Oura Ring: BLE/mobile relay payload constraints.
  - RoboTaxi: vehicle-edge-cloud synchronization.
  - Cloud Fleet: east-west datacenter fabric.
- [ ] Display bandwidth, latency, payload, and retry tradeoffs.
- [ ] Update report with selected communication strategy and residual risk.

### V2-04 - The Data Pipeline Wall

Path: `labs/vol2/lab_04_data_storage.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Connect data storage and freshness to each track.
- [ ] Add shared pipeline/storage budget helper if needed.
- [ ] Track examples:
  - iPhone: privacy-preserving local cache and upload policy.
  - Oura Ring: compressed time-series summaries.
  - RoboTaxi: high-volume sensor logs and incident upload.
  - Cloud Fleet: feature store and training data lake.
- [ ] Add plots/tables for storage growth, freshness, and bandwidth.
- [ ] Update report with data pipeline decision and residual risk.

### V2-05 - The Parallelism Puzzle

Path: `labs/vol2/lab_05_dist_train.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make parallelism choice relevant to track deployment lifecycle.
- [ ] Distinguish training-time parallelism from inference-time deployment target.
- [ ] Add shared parallelism strategy descriptors if useful.
- [ ] Track examples:
  - iPhone/Oura Ring/RoboTaxi: centralized or federated training feeding edge deployment.
  - Cloud Fleet: data/model/pipeline parallelism in service of large model training.
- [ ] Update report with training strategy, scaling bottleneck, and residual risk.
- [ ] Add tests for any shared parallelism helper.

### V2-06 - Collective Communication

Path: `labs/vol2/lab_06_collective_communication.py`

Current status:
- [x] Rich shared wrapper exists.
- [x] Baseline track/report panel installed.
- [x] MLSysIM collective communication physics functions are already used.

Deep migration tasks:
- [ ] Upgrade the existing rich wrapper to canonical track variants.
- [ ] Decide which tracks use true collectives and which use communication analogs.
- [ ] Track examples:
  - iPhone: federated update aggregation payloads.
  - Oura Ring: tiny update/sensor summaries through a phone relay.
  - RoboTaxi: fleet update synchronization and map/model rollout.
  - Cloud Fleet: GPU collective algorithms.
- [ ] Add source-of-truth communication payload assumptions if displayed.
- [ ] Ensure the report uses canonical track profile and variant fields.
- [ ] Add tests for track-specific communication scenario selection.

### V2-07 - When Failure Is Routine

Path: `labs/vol2/lab_07_fault_tolerance.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make failure modes and recovery decisions track-specific.
- [ ] Add shared reliability/failure-budget helper if needed.
- [ ] Track examples:
  - iPhone: offline mode, app crashes, OS updates.
  - Oura Ring: battery depletion, sensor dropout, sync gaps.
  - RoboTaxi: sensor faults, degraded mode, safety fallback.
  - Cloud Fleet: node failures and retry storms.
- [ ] Display failure-rate, recovery-time, or availability evidence.
- [ ] Update report with failure budget and recovery decision.

### V2-08 - The Scheduling Trap

Path: `labs/vol2/lab_08_fleet_orch.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Connect scheduling/orchestration to each track's fleet shape.
- [ ] Add shared fleet/orchestration descriptors if needed.
- [ ] Track examples:
  - iPhone: staged rollout and device eligibility.
  - Oura Ring: background job timing and battery-aware scheduling.
  - RoboTaxi: vehicle dispatch, update windows, and safety constraints.
  - Cloud Fleet: GPU scheduling and bin packing.
- [ ] Display utilization, queueing, rollout, or availability tradeoffs.
- [ ] Update report with scheduling policy and residual risk.

### V2-09 - The Optimization Trap

Path: `labs/vol2/lab_09_perf_engineering.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make optimization targets differ by track.
- [ ] Add shared performance-counter or bottleneck taxonomy helper if useful.
- [ ] Track examples:
  - iPhone: optimize latency without thermal regression.
  - Oura Ring: optimize energy without missing sensing events.
  - RoboTaxi: optimize latency while preserving safety margin.
  - Cloud Fleet: optimize throughput/cost without p99 regression.
- [ ] Update plots to show local optimum vs system optimum.
- [ ] Update report with chosen optimization and unintended side effect.

### V2-10 - The Inference Economy

Path: `labs/vol2/lab_10_inference.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Build track-specific inference economics.
- [ ] Add shared inference cost/latency helper if current logic is notebook-local.
- [ ] Track examples:
  - iPhone: local inference cost is battery/thermal/UX.
  - Oura Ring: local inference cost is energy and duty cycle.
  - RoboTaxi: local inference cost is latency and safety margin.
  - Cloud Fleet: service inference cost is dollars, utilization, and p99.
- [ ] Display batching, quantization, caching, or placement tradeoffs by track.
- [ ] Update report with inference placement and economic constraint.
- [ ] Add tests for cost/latency helper.

### V2-11 - The Edge Thermodynamics Lab

Path: `labs/vol2/lab_11_edge_intelligence.py`

Current status:
- [x] Hand-authored variants exist.
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make this the first full device-track deep migration after V1-10.
- [ ] Move any displayed device battery, memory, latency, or energy facts into MLSysIM/shared helpers.
- [ ] Add a shared edge energy or duty-cycle helper if needed.
- [ ] Wire each notebook part to the selected canonical track.
- [ ] Track examples:
  - iPhone: on-device adaptation under thermal and battery limits.
  - Oura Ring: always-on sensing and tiny inference energy budget.
  - RoboTaxi: edge perception and vehicle compute envelope.
  - Cloud Fleet: compare edge offload against centralized inference.
- [ ] Add track-specific plots for memory, battery/energy, and update payloads.
- [ ] Update report with edge placement decision and thermodynamic residual risk.
- [ ] Add tests for helper math and variant field completeness.

### V2-12 - The Silent Fleet

Path: `labs/vol2/lab_12_ops_scale.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make fleet observability and operations track-specific.
- [ ] Add shared fleet health/telemetry helper if needed.
- [ ] Track examples:
  - iPhone: app/device version segments.
  - Oura Ring: sensor quality and sync coverage.
  - RoboTaxi: route/geography/weather fleet slices.
  - Cloud Fleet: service shards, regions, and model versions.
- [ ] Display fleet health dashboard evidence by track.
- [ ] Update report with monitoring slice, action threshold, and residual risk.

### V2-13 - The Price of Privacy

Path: `labs/vol2/lab_13_security_privacy.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make security/privacy controls track-specific.
- [ ] Add shared privacy/security control catalog if needed.
- [ ] Track examples:
  - iPhone: local processing and permission boundary.
  - Oura Ring: health-derived sensor data and consent.
  - RoboTaxi: location/video logs and incident retention.
  - Cloud Fleet: tenant isolation and data governance.
- [ ] Display accuracy, latency, cost, or utility impact of privacy controls.
- [ ] Update report with selected privacy control and residual risk.

### V2-14 - The Robustness Budget

Path: `labs/vol2/lab_14_robust_ai.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Tie robustness budget to track-specific failure consequences.
- [ ] Add shared robustness scenario/helper if needed.
- [ ] Track examples:
  - iPhone: varied lighting/device/user context.
  - Oura Ring: sensor noise and physiological variation.
  - RoboTaxi: weather, occlusion, and out-of-distribution scenes.
  - Cloud Fleet: prompt/user/data distribution shift.
- [ ] Display robustness-cost tradeoffs.
- [ ] Update report with robustness budget and unresolved hazard.

### V2-15 - The Carbon Budget

Path: `labs/vol2/lab_15_sustainable_ai.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make sustainability accounting track-specific.
- [ ] Add shared energy/carbon helper if the lab displays computed carbon evidence.
- [ ] Track examples:
  - iPhone: battery energy and charging externality.
  - Oura Ring: tiny battery and lifecycle duty cycle.
  - RoboTaxi: vehicle compute energy and fleet update cadence.
  - Cloud Fleet: datacenter energy, utilization, and region mix.
- [ ] Display energy/carbon tradeoffs with source trace.
- [ ] Update report with carbon budget decision and residual risk.

### V2-16 - The Fairness Budget

Path: `labs/vol2/lab_16_responsible_ai.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make fairness tradeoffs specific to each deployment track.
- [ ] Reuse or extend the fairness metric helper from V1-15 if created.
- [ ] Track examples:
  - iPhone: accessibility and device/user variation.
  - Oura Ring: physiological and demographic variation.
  - RoboTaxi: neighborhood, pedestrian, weather, and safety exposure.
  - Cloud Fleet: service quality across user cohorts/regions.
- [ ] Display fairness/utility/safety tradeoffs.
- [ ] Update report with fairness budget, mitigation, and residual risk.

### V2-17 - The Fleet Synthesis

Path: `labs/vol2/lab_17_fleet_synthesis.py`

Current status:
- [x] Baseline track/report panel installed.

Deep migration tasks:
- [ ] Make this the Volume II capstone for the selected canonical track.
- [ ] Add or reuse ledger summary helper to gather prior Volume II decisions.
- [ ] Track examples should synthesize scale, infrastructure, communication, failure,
  operations, privacy, robustness, sustainability, and responsibility.
- [ ] Confirm missing ledger entries degrade gracefully.
- [ ] Generate a final fleet architecture report.
- [ ] Add tests for fleet synthesis report schema and ledger fallback.

## Final Catalog QA

- [ ] Run `python3 -m pytest labs/tests/test_static.py -q`.
- [ ] Run all shared helper tests in `labs/tests`.
- [ ] Run relevant MLSysIM tests separately from `labs/tests`.
- [ ] Verify `wheels/mlsysbook_labs-0.1.0-py3-none-any.whl` is rebuilt after shared helper changes.
- [ ] Verify `wheels/mlsysim-0.1.2-py3-none-any.whl` is rebuilt after MLSysIM changes.
- [ ] Verify every lab's report can be generated locally.
- [ ] Verify each track has at least one representative visual/evidence modality.
- [ ] Verify no lab embeds source-of-truth hardware/model/system facts locally.
- [ ] Commit final cleanup.

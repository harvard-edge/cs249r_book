# Margin Figure Caption And Device Audit

Date: 2026-06-02

Scope: all 216 SVG margin figures referenced from QMD `.column-margin`
blocks after the caption-first and closed-device rules were added.

## Checks

| Check | Result |
|---|---:|
| Referenced margin SVGs inventoried | 216 |
| Missing captions | 0 |
| Captions over 16 words | 0 |
| Captions over 115 characters | 0 |
| Title-like short captions | 0 |
| Generic starts such as "This..." after fixes | 0 |
| Missing referenced SVGs | 0 |
| Live SVG text/font residue | 0 |

Rendered review sheet:
`/tmp/mlsysbook-margin-caption-pass-all-v2.png`

Focused new-figure sheet:
`/tmp/mlsysbook-new-margin-figures.png`

## Caption Tweaks Made

| Chapter | Asset | Old caption | New caption | Reason |
|---|---|---|---|---|
| vol1/hw_acceleration | `hw_acceleration_dam_locator.svg` | `This chapter is the Machine axis.` | `Hardware acceleration turns on the Machine axis.` | Rewrites a generic title-like caption into a paragraph/job statement. |
| vol1/ml_workflow | `ml_workflow_feedback_timescales.svg` | `Feedback loops span roughly five orders of magnitude, from minute-level operational fixes to quarterly architectural review.` | `Feedback loops span minutes to quarters across five orders of magnitude.` | Keeps the visual takeaway while removing footnote-like detail. |

## Strict New-Candidate Pass

Two read-only agents audited Volume 1 and Volume 2 under the updated catalog and
caption rules. The implementation decision was conservative: add only
high-confidence, kit-native figures that were not redundant with a nearby body
figure.

| Decision | Chapter | Candidate | Device | Caption |
|---|---|---|---|---|
| Added | vol1/data_engineering | Data debt compounding | `sparkline-trend` | `Data debt diverges as accumulation rate rises.` |
| Cut | vol1/model_serving | Utilization-latency cliff | `scale-anchor` | Redundant with local body figure `@fig-tail-latency-explosion`. |
| Deferred | vol1/hw_acceleration | Heterogeneous SoC engine assignment | `taxonomy-mini` | Medium confidence; useful only if kept as category assignment, not a mechanism diagram. |
| Added | vol2/compute_infrastructure | CXL bandwidth gap | `hierarchy-ladder` | `HBM is 50-fold faster, so CXL is capacity only.` |
| Added | vol2/network_fabrics | PFC pause propagation | `blast-radius` | `One paused receiver can freeze unrelated flows.` |
| Deferred | vol2/edge_intelligence | Edge memory heterogeneity | `hierarchy-ladder` | Medium confidence; keep only if a future pass needs a single-axis memory span. |
| Cut | vol2/ops_scale | Telemetry hierarchy | `hierarchy-ladder` / `taxonomy-mini` | Redundant with the monitoring pyramid body figure. |
| Added | vol2/security_privacy | Output leakage shrinkage | `hierarchy-ladder` | `Full distributions leak far more than top-k outputs.` |

## New Figures Added

| Chapter | Asset | Device | Geometry contract |
|---|---|---|---|
| vol1/data_engineering | `data_engineering_debt_compounding.svg` | `sparkline-trend` | Quantitative curves from `Debt_n = Debt_0(1+r)^n` using the adjacent 10 percent and 30 percent accumulation rates. |
| vol2/compute_infrastructure | `compute_infrastructure_cxl_bandwidth_gap.svg` | `hierarchy-ladder` | Quantitative bandwidth ladder: H100 HBM from MLSysIM versus adjacent CXL 3.0 64 GB/s prose value. |
| vol2/network_fabrics | `network_fabrics_pfc_pause_blast.svg` | `blast-radius` | Schematic one-source-to-many pause propagation, complementing the adjacent incast body figure. |
| vol2/security_privacy | `security_privacy_output_leakage_ladder.svg` | `hierarchy-ladder` | Quantitative returned-score ladder: 1000-class full distribution versus top-5 output. |

## Existing Non-Kit Exceptions

These figures render cleanly and have useful captions, but they should not be
treated as examples of new production device types. They are grandfathered
exceptions under the closed-kit rule.

| Chapter | Asset | Current device metadata | Decision |
|---|---|---|---|
| vol1/benchmarking | `vol1_benchmarking_margin_004.svg` | `other-new` | Keep as a dominated-point exception; do not generalize Pareto frontiers into a margin device. |
| vol1/conclusion | `vol1_conclusion_margin_001.svg` | `other-new` | Keep as a synthesis cascade; future causal chains should usually be body figures or prose. |
| vol1/introduction | `vol1_introduction_margin_001.svg` | `other-new` | Keep as a nested-system exception; it reinforces the opening scope shift. |
| vol1/nn_architectures | `vol1_nn_architectures_margin_003.svg` | `other-new` | Keep as a localized all-to-all dependence schematic; do not generalize meshes into a margin device. |
| vol2/collective_communication | `vol2_collective_communication_margin_004.svg` | `other-new` | Keep as an error-feedback loop exception; do not add loop as a device unless it recurs and passes the new-device gate. |
| vol2/conclusion | `vol2_conclusion_margin_002.svg` | `other-new` | Keep as a matched-rate synthesis icon; do not generalize matched strips into a device. |
| vol2/responsible_ai | `responsible_ai_monitoring_scale.svg` | `formula-rows` | Keep as an existing compact scale reminder; do not treat formula rows as a production device. |
| vol2/security_privacy | `vol2_security_privacy_margin_003.svg` | `other-new` | Keep as an existing epsilon burn-down exception; future budget ledgers should usually be prose/body figures. |

## Bottom Line

The placed margin figure set is visually sound after the render pass, caption
edits, and four strict additions above. The main discipline issue is
metadata/precedent: existing `other-new` and `formula-rows` figures should remain
exceptions, not templates for new margin visuals.

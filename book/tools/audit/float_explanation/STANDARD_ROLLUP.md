# Float Exposition Audit — full results (standard-based, all 33 chapters)

Graded against `FLOAT_EXPOSITION_STANDARD.md`. Caption, alt-text, in-figure labels, code
comments, and callout interiors are excluded from the prose budget. Per-chapter findings and
suggested rewrites live in `worklists/worklist_<vol>_<chapter>_std.md`.

Grade: ✅ prose carries it · ⚠️ cited but no lead-out (takeaway stuck in caption/cells/code/callout) · 🛑 no prose carries it / orphan / pivot-away.

## Volume 1 (16 chapters)

| chapter | floats | findings | 🛑 |
|---|---|---|---|
| introduction | 23 | 5 | 0 |
| ml_systems | 34 | 8 | 0 |
| ml_workflow | 9 | 0 | 0 |
| data_engineering | 37 | 11 | 0 |
| nn_computation | 64 | 23 | 0 |
| nn_architectures | 49 | 16 | 0 |
| frameworks | 68 | 15 | 0 |
| training | 65 | 7 | 0 |
| data_selection | 38 | 14 | 1 |
| model_compression | 56 | 11 | 0 |
| hw_acceleration | 67 | 16 | 2 |
| benchmarking | 38 | 8 | 0 |
| model_serving | 48 | 13 | 0 |
| ml_ops | 65 | 15 | 0 |
| responsible_engr | 25 | 7 | 0 |
| conclusion | 3 | 1 | 0 |
| **vol1** | **689** | **170** | **3** |

## Volume 2 (17 chapters)

| chapter | floats | findings | 🛑 |
|---|---|---|---|
| introduction | 23 | 11 | 1 |
| compute_infrastructure | 24 | 8 | 0 |
| network_fabrics | 12 | 3 | 0 |
| data_storage | 14 | 5 | 0 |
| distributed_training | 29 | 10 | 1 |
| collective_communication | 25 | 7 | 3 |
| fault_tolerance | 67 | 28 | 0 |
| fleet_orchestration | 21 | 8 | 0 |
| performance_engineering | 16 | 7 | 2 |
| inference | 115 | 29 | 2 |
| edge_intelligence | 28 | 11 | 1 |
| ops_scale | 82 | 17 | 1 |
| security_privacy | 38 | 10 | 0 |
| robust_ai | 28 | 9 | 1 |
| sustainable_ai | 55 | 10 | 2 |
| responsible_ai | 23 | 7 | 0 |
| conclusion | 2 | 1 | 0 |
| **vol2** | **602** | **181** | **14** |

## Total

**1,291 floats · 351 findings · ~17 hard fails (🛑).**

Dominant patterns, by type:
- **Tables (🟠):** the largest category — bare "summarizes / lists / maps / provides guidance" pointers; the load-bearing row or decision lives only in the cells or caption.
- **Figures (🟠):** float-announcer ("illustrates this"), pivot-away ("while Fig X shows…, other…"), or the takeaway sitting only in the caption/alt-text.
- **Equations (🔴):** symbols glossed but the consequence/regime never stated in body prose, or the consequence pushed into a footnote (concentrated in nn_computation, fault_tolerance, sustainable_ai).
- **Listings (🟡):** bare "shows the implementation" pointers with no mechanism or design choice named (concentrated in frameworks, hw_acceleration).

Cross-cutting structural note: ~22 chapters have at least one finding where the takeaway exists
but sits inside a `.callout` box ("Systems insight" / notebook callouts) rather than running body
prose. Whether a deliberate insight callout satisfies the standard is an open calibration call.

Cleanest chapter (prose already carries every float): vol1 ml_workflow (0/9).

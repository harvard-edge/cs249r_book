# LEGO Audit — Ranked Chapter Ledger (2026-06-06)

**Status: P0–P2 complete.** See `lego_audit_signoff.md` for sign-off.

## Phase 0 inventory summary

| Signal | Count |
|--------|------:|
| Chapters scanned | 74 |
| lego_focal_verify FAIL | 9 |
| fmt_prose_contract violations | 0 |
| Phase 1 gate failures (dead-code, prose-units, canonical) | 0 |
| Category A (closed name + open fmt, refined) | 27 |
| Category C (_b/_m scale + bare fmt) | 14 |

## Priority order (fix P0–P2)

| Rank | Chapter | Focal issues | Cat A | Notes |
|------|---------|--------------|------:|-------|
| 1 | vol2/compute_infrastructure | 2 | 2 | power_delta_kw_str, tau_opt_s_str |
| 2 | vol1/model_serving | 0 | 1 | h100_tdp_str P0 (manual W in prose) |
| 3 | vol2/inference | 0 | 8 | l3_all_s_str, parameters_b_str, *_k_str |
| 4 | vol1/introduction | 0 | 4 | google_search_b_str, images_m_str, distance_m_str |
| 5 | vol1/nn_architectures | 2 | 2 | dlrm_entries_b_str ×2 |
| 6 | vol2/edge_intelligence | 3 | 0 | P3 deferred (locality) |
| 7 | vol1/hw_acceleration | 0 | 2 | conv_out_m_str, ln_elements_m_str |
| 8 | vol1/training | 0 | 1 | exp_cycles_min_str |
| 9 | vol2/fault_tolerance | 0 | 1 | target_write_min_str |
| 10 | vol2/fleet_orchestration | 1 | 1 | gs_waste_m_str; dead ClusterEconomicsMigRecap |
| 11 | vol1/ml_systems | 0 | 1 | ww_devices_b_str |
| 12 | vol1/ml_ops | 0 | 1 | dp_m_str |
| 13 | vol2/performance_engineering | 1 | 1 | context_len_tokens_str |
| 14 | vol2/security_privacy | 0 | 1 | top_k_str (dimension K — verify) |
| 15 | vol2/introduction | 0 | 0 | distance_str open-fmt — see P5 backlog |

## Deferred P3–P5

### P5 — Distance / length (`fmt_length` candidate)

- Review `distance_str = fmt(distance_m, ...)` vs closed `fmt_qty(..., km)` (`LightLatency`, `BrakingDistance`, `EdgeLatencyDistance`)
- Comma policy: `commas=False` for small m values; `commas=True` for km / ≥1,000 m
- Consider typed `fmt_length` in MLSysIM; pilot in ml_systems + vol2 introduction

### P6 — Formatter gaps (prose-unit lock-in)

- **First:** migrate `*_ridge_str` / `*_intensity_str` → existing `fmt_arithmetic_intensity` (~30 cells; strip prose ` FLOP/byte`)
- **Then:** `fmt_length` (P5), `fmt_rate`/`fmt_speed` for km/h (3 cells)
- **Audit:** `fmt_temperature`, `fmt_latency` vs `fmt_time`; add binder rules when lanes land
- **Rule:** closed formatter ⇒ bare prose ref; `lego-prose-units` is the backstop

- edge_intelligence: EdgeDeviceSpectrum gap:738, cross_cell
- compute_infrastructure: 108 *Recap classes (locality)
- data_engineering, nn_architectures, conclusion, sustainable_ai: multi_section span
- 4 multi-class cells (responsible_engr, training, distributed_training ×2)

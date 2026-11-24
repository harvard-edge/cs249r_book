#!/usr/bin/env python3
"""Companion calculations for the 12 newly enrolled Explore figure modules.

Computes every derived quantity in code from mlsysim scenarios and closed-form
laws so no hand-calculated numbers enter narration, readouts, or intros.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
for rel in ("books/shared/python", "mlsysim/src", "mlsysim"):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from mlsysim import Embodied, Scenarios  # noqa: E402
from mlsysim.core.units import byte, gigabyte, meter, millimeter, ms, second  # noqa: E402


def compute_all() -> dict:
    # 1. Vol I Ch04: Shared Preprocessing State (concrete toy z-score parity check)
    train_vals = [10.0, 20.0, 30.0]
    holdout_vals = [50.0, 70.0, 90.0]
    raw_query = 50.0
    mu_train = sum(train_vals) / len(train_vals)
    sigma_train = math.sqrt(sum((x - mu_train) ** 2 for x in train_vals) / len(train_vals))
    mu_refit = sum(holdout_vals) / len(holdout_vals)
    sigma_refit = math.sqrt(sum((x - mu_refit) ** 2 for x in holdout_vals) / len(holdout_vals))
    z_shared = (raw_query - mu_train) / sigma_train
    z_refit = (raw_query - mu_refit) / sigma_refit
    z_skew = abs(z_shared - z_refit)

    # 2. Vol I Ch06: Attention QKV (6-token sequence N = 6)
    seq_len = 6
    score_cells = seq_len * seq_len
    proj_matrices = 3  # W_Q, W_K, W_V -> 3 * d^2

    # 3. Vol I Ch06: DLRM Pipeline (BillionUserTable from mlsysim)
    dlrm = Scenarios.BillionUserTable
    dlrm_rows = int(dlrm.rows)
    dlrm_dim = int(dlrm.width)
    dlrm_bpe = int(round(dlrm.bytes_per_value.to(byte).magnitude))
    dlrm_table_gb = round(dlrm.table_bytes.to(gigabyte).magnitude, 1)
    dlrm_row_bytes = int(round(dlrm.row_bytes.to(byte).magnitude))
    # 1 dense vector + 3 sparse vectors = 4 vectors -> C(4, 2) = 6 pairwise dot products
    dlrm_vectors = 4
    dlrm_pairs = dlrm_vectors * (dlrm_vectors - 1) // 2

    # 4. Vol II Ch02: Node PCIe & NUMA (2 sockets, 4 PCIe switches, 8 GPUs, 8 HCAs)
    numa_domains = 2
    gpus_total = 8
    gpus_per_numa = gpus_total // numa_domains
    hcas_per_numa = gpus_per_numa

    # 5. Vol II Ch13: Attestation Key Release (175B reference 14 B/param checkpoint = 2.45 TB)
    ref_params_b = 175
    bytes_per_ckpt_param = 14
    ckpt_tb = round(ref_params_b * 1e9 * bytes_per_ckpt_param / 1e12, 2)

    # 6. Vol II Ch14: Selective Prediction Routing (illustrative risk-coverage curve & arrival rate)
    arrival_rps = 1000
    pts_selective = []
    for tau, cov, risk in [(0.55, 0.90, 0.082), (0.72, 0.78, 0.041), (0.85, 0.62, 0.019)]:
        fallback_frac = round(1.0 - cov, 4)
        fallback_rps = round(fallback_frac * arrival_rps)
        pts_selective.append({
            "tau": tau,
            "coverage": cov,
            "risk": risk,
            "fallback_frac": fallback_frac,
            "fallback_rps": fallback_rps,
        })

    # 7. Vol IV Ch01: Kinematic Roofline (Scenarios.WarehouseAisleStopAtDriveLimit)
    stop = Scenarios.WarehouseAisleStopAtDriveLimit
    half = Scenarios.WarehouseAisleStopHalfSpeed
    aisle = Embodied.Scenario.WarehouseAisle
    rm = Embodied.MobileManipulator.WarehouseMobileManipulator
    mps = meter / second
    tau_s = stop.delay.to(second).magnitude
    tau_ms = round(stop.delay.to(ms).magnitude, 1)
    a_mps2 = round(stop.deceleration.to(mps / second).magnitude, 2)
    d_clear_mm = round(aisle.d_clear.to(millimeter).magnitude, 1)
    d_clear_m = round(aisle.d_clear.to(meter).magnitude, 2)
    v_star = round(2.0 * a_mps2 * tau_s, 4)
    d_star_term_mm = round(v_star * tau_s * 1000.0, 2)
    d_star_total_mm = round(2.0 * d_star_term_mm, 2)
    v_half = round(half.speed.to(mps).magnitude, 2)
    d_half_delay_mm = round(v_half * tau_s * 1000.0, 1)
    d_half_brake_mm = round(v_half ** 2 / (2.0 * a_mps2) * 1000.0, 1)
    d_half_total_mm = round(d_half_delay_mm + d_half_brake_mm, 1)
    v_drive = round(stop.speed.to(mps).magnitude, 2)
    d_drive_delay_mm = round(v_drive * tau_s * 1000.0, 1)
    d_drive_brake_mm = round(v_drive ** 2 / (2.0 * a_mps2) * 1000.0, 1)
    d_drive_total_mm = round(d_drive_delay_mm + d_drive_brake_mm, 1)
    d_drive_spare_mm = round(d_clear_mm - d_drive_total_mm, 1)
    # Solve v * tau + v^2 / (2 a) = d_clear_m
    v_ceiling = round(-a_mps2 * tau_s + math.sqrt((a_mps2 * tau_s) ** 2 + 2.0 * a_mps2 * d_clear_m), 3)
    backbone_ms = round(rm.intent_backbone_latency.to(ms).magnitude, 1) if hasattr(rm, "intent_backbone_latency") else 22.0
    backbone_mm = round(v_drive * (backbone_ms / 1000.0) * 1000.0, 1)

    # 8. Vol IV Ch18: Indistinguishable Observation Deadline (mlsysim Scenarios)
    transit = Scenarios.WarehouseRestrictedTransit
    s_dry = Scenarios.WarehouseFinishedStopAtAisleSpeed.result
    s_oil = Scenarios.WarehouseFinishedStopOilFilm.result
    s_res = Scenarios.WarehouseFinishedStopRestricted.result
    v_aisle = round(s_dry.speed.to(mps).magnitude, 2)
    v_restr = round(s_res.speed.to(mps).magnitude, 2)
    a_oil = round(s_oil.deceleration.to(mps / second).magnitude, 3)
    a_restr = round(s_res.deceleration.to(mps / second).magnitude, 2)
    d_dry_130 = round(s_dry.required_distance.to(meter).magnitude, 3)
    d_oil_130 = round(s_oil.required_distance.to(meter).magnitude, 3)
    d_restr_100 = round(s_res.required_distance.to(meter).magnitude, 3)
    overhead_m = round((s_dry.overhead + s_dry.tracking_inset).to(meter).magnitude, 2)
    speed_cut_pct = round(transit.speed_cut * 100.0, 1)

    return {
        "vol1_preprocessing": {
            "mu_train": round(mu_train, 2),
            "sigma_train": round(sigma_train, 3),
            "mu_refit": round(mu_refit, 2),
            "sigma_refit": round(sigma_refit, 3),
            "raw_query": raw_query,
            "z_shared": round(z_shared, 2),
            "z_refit": round(z_refit, 2),
            "z_skew": round(z_skew, 2),
        },
        "vol1_attention": {
            "seq_len": seq_len,
            "score_cells": score_cells,
            "proj_matrices": proj_matrices,
        },
        "vol1_dlrm": {
            "rows": dlrm_rows,
            "embed_dim": dlrm_dim,
            "bytes_per_value": dlrm_bpe,
            "table_gb": dlrm_table_gb,
            "row_bytes": dlrm_row_bytes,
            "vectors": dlrm_vectors,
            "pairwise_dots": dlrm_pairs,
        },
        "vol2_numa": {
            "numa_domains": numa_domains,
            "gpus_total": gpus_total,
            "gpus_per_numa": gpus_per_numa,
            "hcas_per_numa": hcas_per_numa,
        },
        "vol2_attestation": {
            "ref_params_b": ref_params_b,
            "bytes_per_param": bytes_per_ckpt_param,
            "ckpt_tb": ckpt_tb,
        },
        "vol2_selective": {
            "arrival_rps": arrival_rps,
            "points": pts_selective,
        },
        "vol4_roofline": {
            "tau_ms": tau_ms,
            "a_mps2": a_mps2,
            "d_clear_mm": d_clear_mm,
            "d_clear_m": d_clear_m,
            "v_star": round(v_star, 2),
            "v_star_exact": v_star,
            "d_star_term_mm": round(d_star_term_mm, 1),
            "d_star_total_mm": round(d_star_total_mm, 1),
            "v_half": v_half,
            "d_half_delay_mm": d_half_delay_mm,
            "d_half_brake_mm": d_half_brake_mm,
            "d_half_total_mm": d_half_total_mm,
            "v_drive": v_drive,
            "d_drive_delay_mm": d_drive_delay_mm,
            "d_drive_brake_mm": d_drive_brake_mm,
            "d_drive_total_mm": d_drive_total_mm,
            "d_drive_spare_mm": d_drive_spare_mm,
            "v_ceiling": v_ceiling,
            "backbone_ms": backbone_ms,
            "backbone_mm": backbone_mm,
        },
        "vol4_deadline": {
            "v_aisle": v_aisle,
            "v_restricted": v_restr,
            "tau_ms": tau_ms,
            "d_clear_m": d_clear_m,
            "a_dry": a_mps2,
            "a_oil": a_oil,
            "a_restricted": a_restr,
            "d_dry_m": d_dry_130,
            "d_oil_m": d_oil_130,
            "d_restricted_m": d_restr_100,
            "overhead_m": overhead_m,
            "speed_cut_pct": speed_cut_pct,
        },
    }


def main() -> None:
    data = compute_all()
    assert data["vol1_attention"]["score_cells"] == 36
    assert data["vol1_dlrm"]["pairwise_dots"] == 6
    assert abs(data["vol4_roofline"]["d_drive_total_mm"] - 762.9) < 0.2
    assert data["vol4_roofline"]["d_drive_total_mm"] < data["vol4_roofline"]["d_clear_mm"]
    assert data["vol4_deadline"]["d_dry_m"] < data["vol4_deadline"]["d_clear_m"] < data["vol4_deadline"]["d_oil_m"]
    if "--check" in sys.argv:
        print("new-explore-batch.calc.py: OK")
        return
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()

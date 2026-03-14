#!/usr/bin/env python3
"""
Validation check for the mlsysim paper.

Runs all 7 empirical anchors through mlsysim solvers and compares
the output against the values hardcoded in paper.tex. Flags any
mismatches so you can update the paper or recalibrate the solver.

Usage:
    python3 validate_anchors.py
"""

import sys
import math
from pathlib import Path

# Add repo root to path so mlsysim is importable
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

import mlsysim
from mlsysim.core.constants import Q_, ureg

# ── Reported ground truth values ────────────────────────────────────
REPORTED = {
    "a1_throughput": 38200,       # samples/s, MLPerf v4.0
    "a2_itl_lo": 40,             # ms, vLLM range low
    "a2_itl_hi": 50,             # ms, vLLM range high
    "a3_mfu_lo": 0.38,           # Meta Llama 3 range low
    "a3_mfu_hi": 0.43,           # Meta Llama 3 range high
    "a4_mfu": 0.46,              # PaLM full-scale MFU
    "a5_params_b": 70,           # Chinchilla actual (70B)
    "a6_carbon_t": 552,          # Patterson et al. tonnes CO2
    "a7_tp": 8,                  # Meta Llama 3 parallelism
    "a7_pp": 4,
    "a7_dp": 512,
}

# ── Shared workloads and fleets ─────────────────────────────────────

def _llama_405b():
    return mlsysim.TransformerWorkload(
        name="Llama3_405B",
        architecture="Transformer",
        parameters=Q_("405e9 count"),
        layers=126,
        hidden_dim=16384,
        heads=128,
    )

def _fleet_16k_h100():
    return mlsysim.Fleet(
        name="Llama3_16K",
        node=mlsysim.Node(
            name="H100 Node",
            accelerator=mlsysim.Hardware.Cloud.H100,
            accelerators_per_node=8,
            intra_node_bw=Q_("900 GB/s"),
        ),
        count=2048,
        fabric=mlsysim.NetworkFabric(name="IB", bandwidth=Q_("50 GB/s")),
    )


# ── Anchor implementations ─────────────────────────────────────────

def anchor1_resnet():
    """MLPerf ResNet-50 on DGX A100 (8x A100, batch=2048).

    ResNet-50 achieves ~19% of peak FP16 tensor FLOPS on A100 because
    its small convolution kernels cannot saturate the tensor cores the
    way large GEMM-dominated Transformer layers can. This is consistent
    with published MLPerf submissions showing ~58 TFLOP/s per A100 out
    of 312 TFLOP/s peak.
    """
    model = mlsysim.Models.Vision.ResNet50
    hardware = mlsysim.Hardware.Cloud.A100
    solver = mlsysim.SingleNodeModel()

    per_gpu_batch = 2048 // 8
    res = solver.solve(model, hardware, batch_size=per_gpu_batch,
                       precision="fp16", efficiency=0.19, is_training=True)

    # Scale by 8 for ideal DP within one DGX node (NVLink makes DP overhead negligible)
    fleet_throughput = res.throughput.magnitude * 8
    per_gpu = fleet_throughput / 8

    error_pct = abs(fleet_throughput - REPORTED["a1_throughput"]) / REPORTED["a1_throughput"] * 100

    return {
        "AnchorOneThroughput": int(round(fleet_throughput)),
        "AnchorOnePerGPU": int(round(per_gpu)),
        "AnchorOneError": round(error_pct, 1),
    }


def anchor2_llama_itl():
    """vLLM Llama-2-70B ITL on H100 (TP=2, batch=1).

    At batch=1, decode is purely memory-bandwidth-bound. The raw
    weight-read time on 2x H100 is 140GB / (2 * 3.35 TB/s) = 20.9 ms.
    Real-world overhead (KV-cache, scheduling, NVLink sync) adds ~2x.
    """
    model = mlsysim.Models.Language.Llama2_70B
    hardware = mlsysim.Hardware.Cloud.H100
    solver = mlsysim.ServingModel()

    res = solver.solve(model, hardware, seq_len=1024, batch_size=1, precision="fp16")

    # Raw ITL from single-GPU solve, shard across 2 GPUs (TP=2)
    raw_itl_ms = res.itl.to("ms").magnitude
    sharded_itl = raw_itl_ms / 2.0

    # Empirical overhead factor ~2x for scheduling, NVLink, kernels
    predicted_itl = sharded_itl * 2.0

    in_range = REPORTED["a2_itl_lo"] <= predicted_itl <= REPORTED["a2_itl_hi"]

    return {
        "AnchorTwoITL": int(round(predicted_itl)),
        "AnchorTwoInRange": "yes" if in_range else "no",
    }


def anchor3_llama3_mfu():
    """Meta Llama-3 405B training MFU at 16K H100s.

    Uses DistributedModel with TP=8, PP=4, DP=512 and 64 microbatches
    to minimize pipeline bubble. η=0.42 is the system-level efficiency
    that includes kernel utilization, stragglers, load imbalance,
    checkpointing pauses, and thermal throttling — all effects the
    analytical model does not capture in its communication equations.
    """
    eta = 0.42  # system-level efficiency (kernel η ≈ 0.55 minus real-world overhead)
    solver = mlsysim.DistributedModel()
    res = solver.solve(
        _llama_405b(), _fleet_16k_h100(),
        batch_size=4096, precision="fp16", efficiency=eta,
        tp_size=8, pp_size=4, microbatch_count=64,
        overlap_comm=True, overlap_efficiency=0.85,
    )

    aggregate_mfu = res.scaling_efficiency * eta
    in_range = REPORTED["a3_mfu_lo"] <= aggregate_mfu <= REPORTED["a3_mfu_hi"]
    mfu_pct = aggregate_mfu * 100

    return {
        "AnchorThreeMFU": round(mfu_pct, 1),
        "AnchorThreeScalingEff": round(res.scaling_efficiency * 100, 1),
        "AnchorThreeInRange": "yes" if in_range else "no",
    }


def anchor4_palm():
    """PaLM scaling efficiency at 64K TPU v4s.

    Models the MFU degradation from single-pod to full-scale due to
    inter-pod communication overhead on the ICI fabric.
    """
    tpuv4 = mlsysim.HardwareNode(
        name="TPU v4",
        release_year=2022,
        compute=mlsysim.hardware.types.ComputeCore(
            peak_flops=Q_("275 TFLOP/s"),
            precision_flops={"bf16": Q_("275 TFLOP/s")},
        ),
        memory=mlsysim.hardware.types.MemoryHierarchy(
            capacity=Q_("32 GB"), bandwidth=Q_("1200 GB/s"),
        ),
        tdp=Q_("200 W"),
    )

    palm_540b = mlsysim.TransformerWorkload(
        name="PaLM-540B",
        architecture="Transformer",
        parameters=Q_("540e9 count"),
        layers=118,
        hidden_dim=18432,
        heads=48,
    )

    # Full scale: 64K chips, 4 per host, inter-pod bandwidth limited
    fleet_64k = mlsysim.Fleet(
        name="PaLM_64K",
        node=mlsysim.Node(
            name="TPUv4 Pod Slice",
            accelerator=tpuv4,
            accelerators_per_node=4,
            intra_node_bw=Q_("400 GB/s"),
        ),
        count=64000 // 4,
        fabric=mlsysim.NetworkFabric(
            name="ICI", bandwidth=Q_("24 GB/s"),
            oversubscription_ratio=2.0,
        ),
    )

    eta = 0.47  # system-level efficiency (ICI fabric + stragglers at 64K scale)
    solver = mlsysim.DistributedModel()
    res = solver.solve(
        palm_540b, fleet_64k, batch_size=64000,  # large batch to cover DP
        precision="fp16", efficiency=eta,
        tp_size=4, pp_size=1,
        overlap_comm=True, overlap_efficiency=0.85,
    )

    aggregate_mfu = res.scaling_efficiency * eta
    mfu_pct = aggregate_mfu * 100
    error_pct = abs(aggregate_mfu - REPORTED["a4_mfu"]) / REPORTED["a4_mfu"] * 100

    return {
        "AnchorFourMFU": int(round(mfu_pct)),
        "AnchorFourError": round(error_pct, 1),
    }


def anchor5_chinchilla():
    """Chinchilla scaling law: optimal P* for C = 5e23 FLOPs.

    The ScalingModel implements P* = sqrt(C / 120) from the Chinchilla
    parametric law C = 6PD with D = 20P.
    """
    solver = mlsysim.ScalingModel()
    res = solver.solve(compute_budget=Q_("5e23 flop"))
    p_opt_b = res.optimal_parameters.to("Gcount").magnitude

    error_pct = abs(p_opt_b - REPORTED["a5_params_b"]) / REPORTED["a5_params_b"] * 100

    # Also compute for 1e24 (the larger budget mentioned in the paper)
    res_1e24 = solver.solve(compute_budget=Q_("1e24 flop"))
    p_1e24_b = res_1e24.optimal_parameters.to("Gcount").magnitude
    d_1e24_t = res_1e24.optimal_tokens.to("Tcount").magnitude

    return {
        "AnchorFiveParams": int(round(p_opt_b)),
        "AnchorFiveError": round(error_pct, 1),
        "AnchorFiveLargeP": int(round(p_1e24_b)),
        "AnchorFiveLargeD": round(d_1e24_t, 1),
    }


def anchor6_carbon():
    """GPT-3 training carbon: 10K V100s, 34 days, US grid.

    The paper computes carbon from Patterson et al.'s reported energy
    (1198 MWh) multiplied by US grid carbon intensity (429 gCO2/kWh).
    This is a simple formula validation, not a SustainabilityModel run,
    because Patterson's energy figure is a direct measurement, not
    derivable from TDP × time.
    """
    # Patterson et al. reported values (used in paper's pgfmath constants)
    energy_mwh = 1198          # \GPTenergyMWh in paper.tex
    grid_ci = 429              # \GPTgridCI in paper.tex (gCO2/kWh)

    carbon_t = energy_mwh * grid_ci / 1000  # = 514 tonnes
    error_pct = abs(carbon_t - REPORTED["a6_carbon_t"]) / REPORTED["a6_carbon_t"] * 100

    return {
        "AnchorSixEnergyMWh": int(round(energy_mwh)),
        "AnchorSixCarbonT": int(round(carbon_t)),
        "AnchorSixError": round(error_pct, 1),
    }


def anchor7_parallelism():
    """Llama-3 parallelism optimizer: find TP/PP/DP for 405B on 16K H100s.

    The ParallelismOptimizer searches the discrete space of TP * PP * DP
    factorizations. Note: the optimizer may not find PP=4 if it doesn't
    model memory constraints forcing pipeline parallelism.
    """
    optimizer = mlsysim.ParallelismOptimizer()
    res = optimizer.solve(
        _llama_405b(), _fleet_16k_h100(),
        batch_size=4096,
        precision="fp16", efficiency=0.55, overlap_comm=True,
    )

    best = res.best_config
    match = (best["tp"] == REPORTED["a7_tp"] and
             best["pp"] == REPORTED["a7_pp"] and
             best["dp"] == REPORTED["a7_dp"])

    return {
        "AnchorSevenTP": best["tp"],
        "AnchorSevenPP": best["pp"],
        "AnchorSevenDP": best["dp"],
        "AnchorSevenMatch": "yes" if match else "no",
        "AnchorSevenSearched": res.total_searched,
    }


# ── Paper claims (hardcoded in paper.tex validation table) ──────────
# Update these when you update the paper. The script compares solver
# output against these to flag mismatches.

PAPER_CLAIMS = {
    "Anchor 1": {"key_value": 38500, "key_name": "throughput (s/s)", "reported": 38200},
    "Anchor 2": {"key_value": 43,    "key_name": "ITL (ms)",         "reported": "40-50"},
    "Anchor 3": {"key_value": 40.0,  "key_name": "MFU (%)",          "reported": "38-43"},
    "Anchor 4": {"key_value": 45,    "key_name": "MFU (%)",          "reported": 46},
    "Anchor 5": {"key_value": 65,    "key_name": "P* (B params)",    "reported": 70},
    "Anchor 6": {"key_value": 514,   "key_name": "CO2 (tonnes)",     "reported": 552},
    "Anchor 7": {"key_value": "TP=8,PP=4,DP=512", "key_name": "parallelism", "reported": "TP=8,PP=4,DP=512"},
}

# Map anchor functions to their key output for comparison
KEY_EXTRACTORS = {
    "Anchor 1": lambda r: r["AnchorOneThroughput"],
    "Anchor 2": lambda r: r["AnchorTwoITL"],
    "Anchor 3": lambda r: r["AnchorThreeMFU"],
    "Anchor 4": lambda r: r["AnchorFourMFU"],
    "Anchor 5": lambda r: r["AnchorFiveParams"],
    "Anchor 6": lambda r: r["AnchorSixCarbonT"],
    "Anchor 7": lambda r: f"TP={r['AnchorSevenTP']},PP={r['AnchorSevenPP']},DP={r['AnchorSevenDP']}",
}


def main():
    """Run all anchors, compare solver output vs paper claims."""
    anchors = [
        ("Anchor 1", "ResNet-50 DGX A100",     anchor1_resnet),
        ("Anchor 2", "Llama-2-70B ITL",         anchor2_llama_itl),
        ("Anchor 3", "Llama-3 MFU",             anchor3_llama3_mfu),
        ("Anchor 4", "PaLM scaling",            anchor4_palm),
        ("Anchor 5", "Chinchilla P*",           anchor5_chinchilla),
        ("Anchor 6", "GPT-3 carbon",            anchor6_carbon),
        ("Anchor 7", "Llama-3 parallelism",     anchor7_parallelism),
    ]

    print("mlsysim Validation Report")
    print("=" * 70)
    print(f"  {'Anchor':<30} {'Solver':>10} {'Paper':>10} {'Reported':>10}  Match?")
    print("-" * 70)

    failed = 0
    mismatches = 0
    for key, desc, fn in anchors:
        try:
            result = fn()
            solver_val = KEY_EXTRACTORS[key](result)
            claim = PAPER_CLAIMS[key]
            paper_val = claim["key_value"]

            # Check match (within 5% for numerics, exact for strings)
            if isinstance(solver_val, (int, float)) and isinstance(paper_val, (int, float)):
                match = abs(solver_val - paper_val) / max(abs(paper_val), 1) < 0.05
            else:
                match = str(solver_val) == str(paper_val)

            status = "OK" if match else "MISMATCH"
            if not match:
                mismatches += 1

            print(f"  {key + ': ' + desc:<30} {str(solver_val):>10} {str(paper_val):>10} {str(claim['reported']):>10}  {status}")

        except Exception as e:
            failed += 1
            print(f"  {key + ': ' + desc:<30} {'FAIL':>10} {'':>10} {'':>10}  {e}")

    print("=" * 70)
    if mismatches:
        print(f"  {mismatches} mismatch(es) — update paper.tex or calibrate solver")
    if failed:
        print(f"  {failed} anchor(s) failed to run")
        sys.exit(1)
    elif mismatches == 0:
        print("  All values match.")


if __name__ == "__main__":
    main()

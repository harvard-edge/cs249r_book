"""Validation-anchor scenarios shared by the paper generator and the anchor check.

``generate_paper_values.py`` formats these results into LaTeX macros and
``validate_anchors.py`` compares them with the published values. Both import
this module, so an anchor's configuration is defined exactly once.

Each ``anchor_*`` function runs its solver(s) against the in-repo mlsysim and
returns a plain dict of raw (unrounded) results plus a ``source`` string that
names the call. Published values the anchors are compared against have no
registry home yet; they are the ``*_REPORTED*`` constants below, each with its
citation.
"""

from __future__ import annotations

import sys
import csv
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # directory holding the mlsysim package
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlsysim  # noqa: E402
from mlsysim import Hardware, Infrastructure, Models, Systems  # noqa: E402
from mlsysim.core.units import Q_, resolve_precision  # noqa: E402
from mlsysim.models.types import TransformerWorkload  # noqa: E402
from mlsysim.physics import carbon_from_energy  # noqa: E402
from mlsysim.solvers import (  # noqa: E402
    DistributedModel,
    ParallelismOptimizer,
    ScalingModel,
    SingleNodeModel,
)
from mlsysim.systems.types import Fleet, NetworkFabric, Node  # noqa: E402

# ── Published values (no registry home yet) ─────────────────────────────────
# A1: NVIDIA DeepLearningExamples, TensorFlow ResNet-50 v1.5, AMP + XLA, DGX A100
#     (8x A100 40GB), TensorFlow 20.06-tf1-py3 container, batch 256 per GPU; images/s for the node.
A1_REPORTED_IMG_PER_S = 17_400
# A3: Llama 3 herd paper (Llama Team, 2024), Table 4: 16,384 H100 at 8K sequence, 41% MFU.
A3_REPORTED_MFU = 0.41
# A4: PaLM (Chowdhery et al.), model FLOPs utilization of the 540B run: 46.2%.
A4_REPORTED_MFU = 0.462
# PaLM Sec. 4.1: analytically computed hardware FLOPs utilization, including rematerialization.
A4_REPORTED_HFU = 0.578
# A5: Chinchilla (Hoffmann et al., 2022) trained 70B parameters on 1.4T tokens.
A5_CHINCHILLA_PARAMS = 70e9
A5_CHINCHILLA_TOKENS = 1.4e12
# A6: Patterson et al. (2021) report 552 tCO2 for GPT-3 training.
A6_REPORTED_TONNES = 552
# A7: Meta's Llama 3 405B parallelism backbone (TP=8, PP=16; DP fills the rest).
A7_REPORTED_TP = 8
A7_REPORTED_PP = 16

# ── Configurations ──────────────────────────────────────────────────────────
A1_GPUS = 8
A1_CONFIG = dict(batch_size=256, precision="fp16", efficiency=0.18, is_training=True)
A2_TP_DEGREE = 4  # Match the Argonne request-latency configuration, not minimum capacity.
# A2 decode reference: NVIDIA NIM benchmarks report ITL with TTFT excluded at concurrency 1
# (paper/data/nim-llama-70b-concurrency1.csv). The primary row matches A2's FP16, TP=4, four-H100
# configuration and is the shortest published prompt; the later release and the TP sweep use the
# one sequence length every 16-bit configuration shares.
A2_ITL_PRIMARY = dict(version="1.5.0", gpu="H100", tp=4, precision="fp16", isl=200, osl=200)
A2_ITL_LATER_VERSION = "1.8.0"
A2_ITL_SWEEP_TOKENS = (1000, 1000)
A2_ITL_HARDWARE = {"H100": Hardware.Cloud.H100, "H200": Hardware.Cloud.H200, "A100": Hardware.Cloud.A100}
A3_ETA = 0.50
A3_V_STAGES = 8
# Llama 3 herd paper, Sec. 3.3.2 and Table 4: TP=8, CP=1, PP=16, DP=128 on 16K H100 at 8,192
# tokens, a global batch of 2,048 sequences (16 per DP group). FSDP shards optimizer states and
# gradients but not the weights (ZeRO-2), and training runs without activation checkpointing.
# Sixteen sequences per DP group allow at most 16 one-sequence microbatches. Meta runs an interleaved
# schedule with virtual stages (v=8 virtual stages per rank for a 16-stage pipeline on 126 layers),
# which divides the bubble from 48.4% down to 10.5%.
A3_CONFIG = dict(batch_size=2048, precision="fp16", efficiency=A3_ETA, tp_size=8, pp_size=16,
                 v_stages=A3_V_STAGES, microbatch_count=16, zero_stage=2, overlap_comm=True,
                 overlap_efficiency=0.85, seq_len=8192)
A4_ETA = A4_REPORTED_HFU
# PaLM, Sec. 4: two TPU v4 Pods of 3,072 chips on 768 hosts each (4 chips per host), connected
# over the datacenter network. Inside a pod the chips reach one another over ICI, whose bandwidth
# comes from the registry (Hardware.Cloud.TPUv4.nvlink, Jouppi et al. 2023, Table 4). Cross-pod
# gradient transfers burst to "an aggregate burst of 81 Tbps across all hosts"; spread over the
# 6,144 chips, that measured burst stands in for the per-chip DCN bandwidth (a floor on capacity).
# Sec. 4.1 reports 57.8% hardware FLOPs utilization (HFU) and 46.2% MFU under selective activation
# rematerialization.
A4_CHIPS, A4_POD_CHIPS, A4_CHIPS_PER_HOST = 6144, 3072, 4
A4_DCN_BURST_GBS = 81_000 / 8  # 81 Tbps in GB/s
A4_DCN_PER_CHIP = Q_(A4_DCN_BURST_GBS / A4_CHIPS, "GB/s")
# PaLM, Sec. 4: pipeline-free; each 3,072-chip pod uses 12-way model parallelism and 256-way fully
# sharded data parallelism, with two-way data parallelism across the two pods (512 DP ranks).
# Batch 2,048 sequences of 2,048 tokens, with rematerialization (Sec. 4.1).
A4_CONFIG = dict(batch_size=2048, precision="fp16", efficiency=A4_ETA, tp_size=12, pp_size=1,
                 zero_stage=3, activation_recomputation=True, overlap_comm=True,
                 overlap_efficiency=0.85, seq_len=2048)
# At Meta's 8,192 tokens no split fits mlsysim's activation accounting without recomputation
# (see A3), so the search runs with full recomputation.
A7_CONFIG = dict(batch_size=2048, precision="fp16", efficiency=A3_ETA, overlap_comm=True, seq_len=8192,
                 activation_recomputation=True)
A7_WIDE_MAX_TP = 64


def llama3_16k_fleet() -> Fleet:
    """Meta Llama 3 405B training fleet: 2,048 DGX H100 nodes on InfiniBand NDR."""
    return Systems.Clusters.Meta_Llama3_16K_H100


def palm_540b() -> TransformerWorkload:
    return TransformerWorkload(name="PaLM-540B", architecture="Transformer", parameters=Q_("540e9 param"),
                               layers=118, hidden_dim=18432, heads=48)


def palm_fleet() -> Fleet:
    """Two 3,072-chip TPU v4 pods: ICI inside each pod, the datacenter network between them."""
    tpu = Hardware.Cloud.TPUv4
    node = Node(name="TPU v4 Pod (3,072 chips on ICI)", accelerator=tpu,
                accelerators_per_node=A4_POD_CHIPS, intra_node_bw=tpu.nvlink.bandwidth_per_direction,
                nics_per_node=A4_POD_CHIPS // A4_CHIPS_PER_HOST)
    fabric = NetworkFabric(name="Datacenter network (PaLM cross-pod burst)", bandwidth=A4_DCN_PER_CHIP)
    return Fleet(name="PaLM 6144 TPU v4", node=node, count=A4_CHIPS // A4_POD_CHIPS, fabric=fabric)


def anchor_one() -> dict:
    profile = SingleNodeModel().solve(Models.Vision.ResNet50, Hardware.Cloud.A100, **A1_CONFIG)
    per_gpu = profile.throughput.m_as("1/s")
    node = A1_GPUS * per_gpu
    return {
        "profile": profile, "feasible": profile.feasible, "per_gpu": per_gpu, "node": node,
        "reported": A1_REPORTED_IMG_PER_S,
        "rel_error": abs(node - A1_REPORTED_IMG_PER_S) / A1_REPORTED_IMG_PER_S,
        "source": ("SingleNodeModel().solve(Models.Vision.ResNet50, Hardware.Cloud.A100, "
                   + ", ".join(f"{key}={value!r}" for key, value in A1_CONFIG.items()) + ")"),
    }


def anchor_two() -> dict:
    model, h100 = Models.Language.Llama2_70B, Hardware.Cloud.H100
    _, bpp = resolve_precision("fp16")
    weights = model.size_in_bytes(bpp)
    floor = weights / (A2_TP_DEGREE * h100.memory.bandwidth)
    path = Path(__file__).resolve().parents[1] / "data" / "argonne-llama2-70b-h100.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    short = [r for r in rows if r["Input Output Length"] == "128"]
    if len(short) != 2 or {r["Framework"] for r in short} != {"vLLM", "TensorRT-LLM"}:
        raise ValueError("Expected one short-prompt row per Argonne engine")
    averages = {r["Framework"]: float(r["Latency"]) * 1000 / 128 for r in short}
    return {
        "weights": weights, "floor_ms": floor.m_as("ms"), "tp": A2_TP_DEGREE,
        "request_average_ms": averages,
        "source": "Models.Language.Llama2_70B.size_in_bytes(FP16) / (4 x Hardware.Cloud.H100.memory.bandwidth)",
    }


def anchor_two_itl() -> dict:
    """Weight-read floors against NIM's TTFT-excluded ITL for Llama-3.3-70B at 16 bits."""
    model = Models.Language.Llama3_70B
    _, bpp = resolve_precision("fp16")  # BF16 rows have the same two-byte weights
    weights = model.size_in_bytes(bpp)
    path = Path(__file__).resolve().parents[1] / "data" / "nim-llama-70b-concurrency1.csv"
    with path.open() as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["Model"] == "Llama-3.3-70b-instruct" and r["Precision"] in ("fp16", "bf16")]

    def point(r: dict) -> dict:
        gpu, tp = r["GPU"].split()[0], int(r["TP"])
        floor = (weights / (tp * A2_ITL_HARDWARE[gpu].memory.bandwidth)).m_as("ms")
        itl = float(r["ITL (ms)"])
        return {"version": r["NIM Version"], "gpu": gpu, "tp": tp, "precision": r["Precision"],
                "isl": int(r["Input Tokens"]), "osl": int(r["Output Tokens"]), "itl_ms": itl,
                "floor_ms": floor, "gap_frac": 1 - floor / itl, "residual_ms": itl - floor}

    points = [point(r) for r in rows]
    if not points or any(p["floor_ms"] >= p["itl_ms"] for p in points):
        raise ValueError("A2 ITL: a weight-read floor is not below its measured ITL")

    def pick(**match) -> dict:
        found = [p for p in points if all(p[k] == v for k, v in match.items())]
        if len(found) != 1:
            raise ValueError(f"A2 ITL: expected one row for {match}, found {len(found)}")
        return found[0]

    p = A2_ITL_PRIMARY
    primary = pick(**p)
    same_release = [q for q in points if q["version"] == p["version"] and q["gpu"] == p["gpu"] and q["tp"] == p["tp"]]
    isl, osl = A2_ITL_SWEEP_TOKENS
    earlier = pick(version=p["version"], gpu=p["gpu"], tp=p["tp"], isl=isl, osl=osl)
    later = pick(version=A2_ITL_LATER_VERSION, gpu=p["gpu"], tp=p["tp"], isl=isl, osl=osl)
    sweep = {key: pick(version=A2_ITL_LATER_VERSION, gpu=gpu, tp=tp, isl=isl, osl=osl)
             for key, gpu, tp in [("h200_tp2", "H200", 2), ("h100_tp4", "H100", 4),
                                  ("h100_tp8", "H100", 8), ("a100_tp8", "A100", 8)]}
    return {
        "weights": weights, "points": points, "primary": primary,
        "release_min_ms": min(q["itl_ms"] for q in same_release),
        "release_max_ms": max(q["itl_ms"] for q in same_release),
        "earlier": earlier, "later": later,
        "runtime_shift_frac": 1 - later["itl_ms"] / earlier["itl_ms"],
        "sweep": sweep,
        "source": "Models.Language.Llama3_70B.size_in_bytes(FP16) / (TP x Hardware.Cloud.<GPU>.memory.bandwidth)",
    }


def anchor_three() -> dict:
    fleet = llama3_16k_fleet()
    result = DistributedModel().solve(Models.Language.Llama3_405B, fleet, **A3_CONFIG)
    # Fleet MFU = replica MFU (model FLOPs only, so recomputation is excluded) x scaling efficiency.
    mfu = result.scaling_efficiency * result.node_profile.mfu
    return {
        "fleet": fleet, "result": result, "mfu": mfu, "reported": A3_REPORTED_MFU,
        "feasible": result.node_profile.feasible,
        "memory_gb": result.node_profile.memory_footprint.m_as("GB"),
        "rel_error": abs(mfu - A3_REPORTED_MFU) / A3_REPORTED_MFU,
        "abs_error_points": abs(mfu - A3_REPORTED_MFU) * 100,
        "source": ("DistributedModel().solve(Models.Language.Llama3_405B, Systems.Clusters.Meta_Llama3_16K_H100, "
                   "batch_size=2048, efficiency=0.50, tp_size=8, pp_size=16, v_stages=8, microbatch_count=16, "
                   "zero_stage=2, overlap_comm=True, overlap_efficiency=0.85, seq_len=8192)"),
    }


def anchor_four() -> dict:
    fleet = palm_fleet()
    result = DistributedModel().solve(palm_540b(), fleet, **A4_CONFIG)
    # Replica MFU excludes the rematerialization FLOPs, matching PaLM's MFU definition.
    mfu = result.scaling_efficiency * result.node_profile.mfu
    return {
        "fleet": fleet, "result": result, "mfu": mfu, "reported": A4_REPORTED_MFU,
        "feasible": result.node_profile.feasible,
        "memory_gb": result.node_profile.memory_footprint.m_as("GB"),
        "rel_error": abs(mfu - A4_REPORTED_MFU) / A4_REPORTED_MFU,
        "abs_error_points": abs(mfu - A4_REPORTED_MFU) * 100,
        "source": ("DistributedModel().solve(PaLM-540B [118 layers, d=18432, 48 heads], Fleet(2 x 3072-chip "
                   "Hardware.Cloud.TPUv4 pods on ICI, datacenter network at PaLM's 81 Tbps burst / 6144 chips), "
                   "batch_size=2048, efficiency=0.578, tp_size=12, pp_size=1, zero_stage=3, "
                   "activation_recomputation=True, overlap_comm=True, seq_len=2048)"),
    }


def anchor_five() -> dict:
    lit = mlsysim.Literature.Chinchilla
    budget = float(lit.ComputeConstant) * A5_CHINCHILLA_PARAMS * A5_CHINCHILLA_TOKENS
    result = ScalingModel().solve(compute_budget=Q_(budget, "flop"))
    p_star = result.optimal_parameters.m_as("count")
    return {
        "budget": budget, "result": result, "p_star": p_star,
        "d_star": result.optimal_tokens.m_as("count"), "reported": A5_CHINCHILLA_PARAMS,
        "rel_error": abs(p_star - A5_CHINCHILLA_PARAMS) / A5_CHINCHILLA_PARAMS,
        "source": "ScalingModel().solve(compute_budget=Literature.Chinchilla.ComputeConstant x 70e9 x 1.4e12 FLOPs)",
    }


def anchor_six() -> dict:
    energy = Models.Language.GPT3.training_energy_mwh
    ci = Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh
    tonnes = carbon_from_energy(energy, Q_(ci, "g/kWh")).m_as("t")
    return {
        "energy_mwh": energy.m_as("MWh"), "ci": ci, "tonnes": tonnes, "reported": A6_REPORTED_TONNES,
        "rel_error": abs(tonnes - A6_REPORTED_TONNES) / A6_REPORTED_TONNES,
        "source": ("physics.carbon_from_energy(Models.Language.GPT3.training_energy_mwh, "
                   "Infrastructure.Grids.US_Avg.carbon_intensity_g_kwh)"),
    }


def anchor_seven(fleet: Fleet | None = None) -> dict:
    fleet = fleet or llama3_16k_fleet()
    model = Models.Language.Llama3_405B
    best = ParallelismOptimizer().solve(model, fleet, **A7_CONFIG)
    wide = ParallelismOptimizer().solve(model, fleet, max_tp=A7_WIDE_MAX_TP, **A7_CONFIG)
    top, second = best.top_candidates[0], best.top_candidates[1]
    return {
        "result": best, "wide": wide, "runner_up": second,
        "margin_rel": (top["mfu"] - second["mfu"]) / top["mfu"],
        "margin_points": (top["mfu"] - second["mfu"]) * 100,
        "reported_tp": A7_REPORTED_TP, "reported_pp": A7_REPORTED_PP,
        "match": best.best_config["tp"] == A7_REPORTED_TP and best.best_config["pp"] == A7_REPORTED_PP,
        "source": ("ParallelismOptimizer().solve(Models.Language.Llama3_405B, Systems.Clusters.Meta_Llama3_16K_H100, "
                   "batch_size=2048, efficiency=0.50, overlap_comm=True, seq_len=8192, "
                   "activation_recomputation=True)"),
    }


# ── Measured board energy (paper/data/gpu-energy*) ──────────────────────────
# Four declared tests read GPU board energy from NVML's counter on Modal cloud GPUs. Only some
# runs commit their raw trials, so the measured columns come from each test's committed
# summary.csv, device_choice.csv and comparison tables. Every quantity that depends on the
# package (modeled time, FLOP utilization, the four energy estimates, and the device each estimate
# picks) is recomputed here with the in-repo ServingModel and SingleNodeModel, so a solver change
# moves these values and `--check` reports it.
ENERGY_DATA = Path(__file__).resolve().parents[1] / "data"
ENERGY_TESTS = ("gpu-energy", "gpu-energy-workloads", "gpu-energy-boards",
                "gpu-energy-seeds/seed-2193", "gpu-energy-seeds/seed-2198")
ENERGY_NEW_TESTS = ENERGY_TESTS[1:]  # the three tests that include the H200 (two seed analyses for the last)
ENERGY_DEVICES = ("H100", "A100", "H200")
ENERGY_DENSE = ("Models.Language.Llama3_8B", "Models.Language.Llama2_7B", "Models.Language.Llama2_13B")
ENERGY_MOE = "Models.Language.Mixtral_8x7B"
ENERGY_ESTIMATES = ("rated_measured", "rated_modeled", "proportional_measured", "proportional_modeled")


def _energy_csv(path: Path) -> list[dict]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def _energy_rows(gpu, test: str) -> tuple[list[dict], list[dict]]:
    """One test's per-phase rows and device choices, with model-dependent values recomputed."""
    folder = ENERGY_DATA / test
    plan = json.loads((folder / "plan.json").read_text())
    configs = {c["id"]: c for c in plan["configs"]}
    rows, by_config = [], {}
    efficiency = {(c["device"], c["workload"]): gpu.protocol.efficiency_of(plan, c) for c in plan["configs"]}
    for r in _energy_csv(folder / "summary.csv"):
        config, phase = configs[r["config_id"]], r["phase"]
        tdp = gpu.resolve(plan["devices"][config["device"]]["registry"]).tdp
        rated_w = tdp.m_as("W")
        seconds, joules = float(r["measured_s_per_unit"]), float(r["measured_j_per_unit"])
        modeled = gpu.modeled_seconds(plan, config, phase)
        work = gpu.work_seconds(plan, config, phase)
        energy = {
            "rated_measured": gpu.board_energy_j(seconds, tdp),
            "rated_modeled": gpu.board_energy_j(modeled, tdp),
            "proportional_measured": gpu.board_energy_j(seconds, gpu.proportional_power(tdp, work / seconds)),
            "proportional_modeled": gpu.board_energy_j(modeled, gpu.proportional_power(tdp, work / modeled)),
        }
        row = {"test": test, "device": config["device"], "workload": r["workload"], "phase": phase,
               "batch": config["batch"], "measured_j": joules,
               "efficiency": efficiency[(config["device"], config["workload"])],
               "power_over_tdp": float(r["measured_power_w"]) / rated_w,
               "idle_over_tdp": float(r["idle_power_w"]) / rated_w,
               "utilization": work / seconds, "step_ratio": modeled / seconds, "energy": energy,
               **{f"{name}_over_measured": value / joules for name, value in energy.items()}}
        rows.append(row)
        by_config[(r["config_id"], phase)] = row
    choices = []
    for c in _energy_csv(folder / "device_choice.csv"):
        devices = [d for d in ENERGY_DEVICES if c.get(f"{d}_config")]
        measured = {d: by_config[(c[f"{d}_config"], c["phase"])]["measured_j"] for d in devices}
        if any(not math.isclose(measured[d], float(c[f"{d}_measured_j_per_unit"]), rel_tol=1e-5) for d in devices):
            raise ValueError(f"energy: {test} device_choice.csv disagrees with summary.csv")
        if c["measured_preferred"] != min(measured, key=measured.get):
            raise ValueError(f"energy: {test} measured_preferred is not the lowest measured energy")
        if c["measured_every_run_pair"] not in ("True", "False"):
            raise ValueError(f"energy: {test} measured_every_run_pair is not True or False")
        estimates = {name: {d: by_config[(c[f"{d}_config"], c["phase"])]["energy"][name] for d in devices}
                     for name in ENERGY_ESTIMATES}
        picks = {name: min(e, key=e.get) for name, e in estimates.items()}
        if c["phase"] == "train" and "A100" in devices:
            # TDP over the modeled time again, with the A100 at Benchmark 1's efficiency for the same case.
            a100 = configs[c["A100_config"]]
            if (a100["batch"], plan["workloads"][a100["workload"]]["precision"]) != (A1_CONFIG["batch_size"],
                                                                                     A1_CONFIG["precision"]):
                raise ValueError(f"energy: {test} ResNet-50 training is not Benchmark 1's batch and precision")
            a100_at_benchmark_one = gpu.board_energy_j(
                gpu.modeled_seconds(plan, {**a100, "efficiency": A1_CONFIG["efficiency"]}, "train"),
                gpu.resolve(plan["devices"]["A100"]["registry"]).tdp)
            picks["benchmark_one_a100_highest"] = all(
                a100_at_benchmark_one > e for d, e in estimates["rated_modeled"].items() if d != "A100")
        ranked = sorted(measured.values())
        choices.append({"test": test, "workload": c["workload"], "phase": c["phase"], "batch": int(c["batch"]),
                        "measured": measured, "winner": c["measured_preferred"],
                        "every_pair": c["measured_every_run_pair"] == "True",
                        "lead": 1 - ranked[0] / ranked[1], "estimates": estimates, "picks": picks})
    return rows, choices


def measured_energy() -> dict:
    """Measured GPU board energy against TDP and the proportional power rule (four declared tests).

    Raises if one of the paper's main statements about the measurements no longer holds under
    the in-repo package, so those statements cannot drift from the data.
    """
    from mlsysim.engine import calibration
    from mlsysim.engine.empirical import EMPIRICAL_ANCHORS, RESNET50_A100_TRAIN_BS256, RESNET50_H100_TRAIN_BS256

    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import analyze_gpu_energy as gpu

    rows, choices = [], []
    for test in ENERGY_TESTS:
        test_rows, test_choices = _energy_rows(gpu, test)
        rows += test_rows
        choices += test_choices

    def span(values) -> tuple[float, float]:
        values = list(values)
        if not values:
            raise ValueError("energy: an empty selection")
        return min(values), max(values)

    def require(condition: bool, claim: str) -> None:
        if not condition:
            raise ValueError(f"energy: the paper's statement no longer holds: {claim}")

    # Decode: TDP is above the measured power and the modeled step is short, so the two partly offset.
    decode = [r for r in rows if r["phase"] == "decode"]
    dense = [r for r in decode if r["workload"] in ENERGY_DENSE]
    moe_b1 = [r for r in decode if r["workload"] == ENERGY_MOE and r["batch"] == 1]
    require(dense and all(r["step_ratio"] < 1 < r["rated_measured_over_measured"] for r in dense),
            "every dense decode step is modeled short and TDP is above the measured power")
    decode_prop = span(r["proportional_measured_over_measured"] for r in decode)
    require(decode_prop[1] < 1, "the proportional rule over the measured time falls short in decode")
    moe_step = span(r["step_ratio"] for r in moe_b1)
    require(all(r["rated_modeled_over_measured"] > 1 for r in moe_b1) and moe_step[0] > 1
            and round(moe_step[0], 2) == round(moe_step[1], 2),
            "Mixtral at batch 1 has one modeled step ratio, above 1")
    moe_larger = [r for r in decode if r["workload"] == ENERGY_MOE and r["batch"] > 1]
    require(moe_larger and all(r["step_ratio"] < 1 for r in moe_larger),
            "Mixtral's modeled step is short at the larger batches")
    dense_prop_modeled = span(r["proportional_modeled_over_measured"] for r in dense)
    require(dense_prop_modeled[1] < 1, "the proportional rule over the modeled step stays below the meter")
    prefill_prop = span(r["proportional_measured_over_measured"] for r in rows if r["phase"] == "prefill")
    train_prop = span(r["proportional_measured_over_measured"] for r in rows if r["phase"] == "train")
    require(prefill_prop[1] < 1 and train_prop[1] < 1, "the proportional rule falls short in prefill and training")
    idle = span(r["idle_over_tdp"] for r in rows)
    require(idle[1] < calibration.ENERGY_IDLE_FRACTION, "idle power is below the rule's idle term")

    # Device choice.
    new_decode = [c for c in choices if c["phase"] == "decode" and c["test"] in ENERGY_NEW_TESTS]
    require(new_decode and all(c["winner"] == "H200" and c["every_pair"] and set(c["picks"].values()) == {"H200"}
                               for c in new_decode),
            "the H200 wins every decode comparison on every pairing, and every estimate picks it")
    others = [c for c in choices if c["phase"] != "decode" and c["test"] in ENERGY_NEW_TESTS]
    clear = [c for c in others if c["every_pair"]]
    require(clear and all(c["winner"] == "H200" for c in clear),
            "every prefill or training winner on every pairing is the H200")
    tied = [c for c in choices if c["phase"] != "decode" and {"H100", "H200"} <= c["measured"].keys()]
    require(tied and all(math.isclose(c["estimates"][name]["H100"], c["estimates"][name]["H200"], rel_tol=1e-12)
                         for c in tied for name in ("rated_modeled", "proportional_modeled")),
            "modeled-time estimates give the H100 and H200 the same prefill and training energy")
    train = [c for c in choices if c["phase"] == "train"]
    require(len(train) == len(ENERGY_TESTS) and all(
        c["picks"]["rated_modeled"] == "A100" and max(c["measured"], key=c["measured"].get) == "A100"
        and c["every_pair"] and c["picks"]["rated_measured"] == c["winner"] for c in train),
        "for ResNet-50, TDP over the modeled time picks the A100, which uses the most measured energy")
    require(all(c["picks"]["benchmark_one_a100_highest"] for c in train),
            "at Benchmark 1's A100 efficiency, TDP over the modeled time gives the A100 the most energy")
    first = [c for c in train if c["test"] == "gpu-energy"]
    later = [c for c in train if c["test"] in ENERGY_NEW_TESTS]
    require(len(first) == 1 and first[0]["winner"] == "H100" and all(c["winner"] == "H200" for c in later),
            "the ResNet-50 meter winner is the H100 in the first test and the H200 in the others")
    train_rows = [r for r in rows if r["phase"] == "train"]
    eta = {d: {r["efficiency"] for r in train_rows if r["device"] == d} for d in ENERGY_DEVICES}
    require(all(len(v) == 1 for v in eta.values()) and eta["H100"] == eta["H200"] != eta["A100"],
            "every plan declares one ResNet-50 efficiency per GPU, shared by the H100 and H200")
    require(eta["A100"] == {RESNET50_A100_TRAIN_BS256.solver_kwargs["efficiency"]}
            and eta["H100"] == {RESNET50_H100_TRAIN_BS256.solver_kwargs["efficiency"]}
            and all(a.solver_kwargs["batch_size"] == A1_CONFIG["batch_size"]
                    for a in (RESNET50_A100_TRAIN_BS256, RESNET50_H100_TRAIN_BS256)),
            "the plans' ResNet-50 efficiencies are the engine.empirical anchors for the A100 and H100")
    require(not any({"Hardware.Cloud.H200", "Models.Vision.ResNet50"} <= set(a.registry_paths) for a in EMPIRICAL_ANCHORS),
            "the H200 has no ResNet-50 anchor")

    # Boards and seeds.
    by_board = {}
    for r in _energy_csv(ENERGY_DATA / "gpu-energy-boards" / "board_energy.csv"):
        if r["test"] == "gpu-energy-boards" and r["phase"] == "decode":
            key = (r["device"], r["batch"], r["prompt_tokens"])
            by_board.setdefault(key, []).append(float(r["j_per_unit"]))
    require(by_board and all(len(v) == 4 for v in by_board.values()), "the third test has four boards per GPU")
    widest = max(by_board, key=lambda key: max(by_board[key]) / min(by_board[key]))
    board_spread = max(by_board[widest]) / min(by_board[widest]) - 1
    require(widest[0] == "H200", "the widest board spread in the third test is on the H200")
    seed_shift = {}
    for r in _energy_csv(ENERGY_DATA / "gpu-energy-seeds" / "seed_pairs.csv"):
        if r["phase"] == "decode":
            seed_shift.setdefault(r["device"], []).append(abs(float(r["ratio_b_over_a"]) - 1))
    require(seed_shift, "the seed test has decode pairs")
    seed_mean = max(sum(v) / len(v) for v in seed_shift.values())
    seed_max = max(max(v) for v in seed_shift.values())
    require(seed_max < board_spread, "the largest seed shift is below the widest board spread of the third test")
    seed_summary = [r for r in _energy_csv(ENERGY_DATA / "gpu-energy-seeds" / "seed_summary.csv") if r["phase"] == "decode"]
    require(seed_summary, "the seed test has decode spreads")
    seed_widest = max(seed_summary, key=lambda r: max(float(r["spread_a"]), float(r["spread_b"])))
    seed_board_spread = max(float(seed_widest["spread_a"]), float(seed_widest["spread_b"]))
    require(seed_widest["device"] == "A100", "the widest board spread in the fourth test is on the A100")
    first_decode = [c for c in choices if c["test"] == "gpu-energy" and c["phase"] == "decode"]
    first_a100 = sum(c["winner"] == "A100" and c["every_pair"] for c in first_decode)
    require(first_a100 > 0, "in the first test the A100 wins some decode comparisons on every pairing")
    boards = [r for r in _energy_csv(ENERGY_DATA / "gpu-energy-boards" / "board_summary.csv") if r["phase"] == "decode"]
    require(len(boards) == len(first_decode), "the eight-board comparison covers the first test's decode configurations")
    pairs = {int(r["H100_vs_A100_pairs_all_tests"]) for r in boards}
    lower = span(int(r["H100_lower_all_tests"]) for r in boards)
    require(pairs == {64} and 0 < lower[0] and lower[1] < 64,
            "over eight H100 and eight A100 boards, neither GPU wins a decode comparison on every pairing")
    # A100 power limits above the registry TDP. Each summary row pools two runs on two boards.
    a100_tdp = gpu.resolve("Hardware.Cloud.A100").tdp.m_as("W")
    raised = {}
    for test in ENERGY_TESTS:
        for r in _energy_csv(ENERGY_DATA / test / "summary.csv"):
            limits = sorted(float(x) for x in r["power_limits_w"].split("/"))
            if r["device"] == "A100" and limits[-1] != a100_tdp:
                require(r["runs"] == "2" and limits[0] == a100_tdp < limits[-1] and len(limits) == 2,
                        "a raised A100 power limit is on one of the two boards")
                raised.setdefault(test, set()).add((r["workload"], limits[-1]))
    raised_limits = {limit for rows in raised.values() for _, limit in rows}
    require(set(raised) == {"gpu-energy", "gpu-energy-workloads"} and len(raised_limits) == 1
            and {w for w, _ in raised["gpu-energy-workloads"]} == {"Models.Language.Llama2_7B"},
            "one A100 board in the first test and one Llama-2-7B A100 board in the second report a raised limit")

    return {
        "decode_power_over_tdp": span(r["power_over_tdp"] for r in decode),
        "decode_utilization": span(r["utilization"] for r in decode),
        "decode_proportional_measured": decode_prop,
        "decode_rated_measured": span(r["rated_measured_over_measured"] for r in decode),
        "dense_step_ratio": span(r["step_ratio"] for r in dense),
        "dense_rated_modeled": span(r["rated_modeled_over_measured"] for r in dense),
        "dense_proportional_modeled": dense_prop_modeled,
        "prefill_proportional_measured": prefill_prop, "train_proportional_measured": train_prop,
        "train_efficiency_a100": next(iter(eta["A100"])), "train_efficiency_h100": next(iter(eta["H100"])),
        "seed_board_spread_max": seed_board_spread,
        "moe_b1_step_ratio": sum(moe_step) / 2,
        "moe_b1_rated_modeled": span(r["rated_modeled_over_measured"] for r in moe_b1),
        "idle_over_tdp": idle,
        "h200_decode_lead": span(c["lead"] for c in new_decode),
        "other_clear": len(clear), "other_total": len(others),
        "other_h200_lead": span(c["lead"] for c in clear),
        "train_a100_over_h200": span(c["measured"]["A100"] / c["measured"]["H200"] - 1 for c in later),
        "train_a100_over_h100_first": first[0]["measured"]["A100"] / first[0]["measured"]["H100"] - 1,
        "train_step_a100": span(r["step_ratio"] for r in train_rows if r["device"] == "A100"),
        "train_step_h200": span(r["step_ratio"] for r in train_rows if r["device"] == "H200"),
        "board_spread_max": board_spread,
        "seed_shift_mean_max": seed_mean, "seed_shift_max": seed_max,
        "board_pairs": next(iter(pairs)), "board_h100_lower": lower,
        "first_decode_a100_clear": first_a100, "first_decode_total": len(first_decode),
        "a100_tdp_w": a100_tdp, "a100_raised_limit_w": next(iter(raised_limits)),
        "source": ("paper/data/gpu-energy*/summary.csv measured columns; modeled time, utilization and "
                   "estimates recomputed with analyze_gpu_energy.modeled_seconds / work_seconds / board_energy_j"),
    }

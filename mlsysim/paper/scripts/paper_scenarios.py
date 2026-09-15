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
# A1: NVIDIA DeepLearningExamples, TensorFlow ResNet-50 v1.5, AMP + XLA,
#     8x A100 80GB, batch 256 per GPU; images/s for the whole node.
A1_REPORTED_IMG_PER_S = 17_400
# A3: Llama 3 herd paper (Llama Team, 2024), Table 4: 16,384 H100 at 8K sequence, 41% MFU.
A3_REPORTED_MFU = 0.41
# A4: PaLM (Chowdhery et al.), model FLOPs utilization of the 540B run: 46.2%.
A4_REPORTED_MFU = 0.462
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
A1_CONFIG = dict(batch_size=256, precision="fp16", efficiency=0.19, is_training=True)
A2_TP_DEGREE = 2
A3_ETA = 0.42
A3_CONFIG = dict(batch_size=2048, precision="fp16", efficiency=A3_ETA, tp_size=8, pp_size=16,
                 microbatch_count=64, overlap_comm=True, overlap_efficiency=0.85, seq_len=8192)
A4_ETA = 0.47
A4_CHIPS, A4_CHIPS_PER_HOST = 6144, 4
A4_INTRA_HOST_BW = Q_("400 GB/s")
A4_FABRIC_BW = Q_("24 GB/s")
A4_OVERSUBSCRIPTION = 2.0
A4_CONFIG = dict(batch_size=2048, precision="fp16", efficiency=A4_ETA, tp_size=4, pp_size=1,
                 overlap_comm=True, overlap_efficiency=0.85)
A7_CONFIG = dict(batch_size=2048, precision="fp16", efficiency=A3_ETA, overlap_comm=True)
A7_WIDE_MAX_TP = 64


def llama3_16k_fleet() -> Fleet:
    """2,048 DGX H100 nodes on a 400 Gb/s per-GPU InfiniBand NDR fabric."""
    return Fleet(name="Llama 3 16K H100", node=Systems.Nodes.DGX_H100, count=2048,
                 fabric=Systems.Fabrics.InfiniBand_NDR)


def palm_540b() -> TransformerWorkload:
    return TransformerWorkload(name="PaLM-540B", architecture="Transformer", parameters=Q_("540e9 param"),
                               layers=118, hidden_dim=18432, heads=48)


def palm_fleet() -> Fleet:
    node = Node(name="TPU v4 host (4 chips)", accelerator=Hardware.Cloud.TPUv4,
                accelerators_per_node=A4_CHIPS_PER_HOST, intra_node_bw=A4_INTRA_HOST_BW,
                nics_per_node=A4_CHIPS_PER_HOST)
    fabric = NetworkFabric(name="ICI-class fabric", bandwidth=A4_FABRIC_BW,
                           oversubscription_ratio=A4_OVERSUBSCRIPTION)
    return Fleet(name="PaLM 6144 TPU v4", node=node, count=A4_CHIPS // A4_CHIPS_PER_HOST, fabric=fabric)


def anchor_one() -> dict:
    profile = SingleNodeModel().solve(Models.Vision.ResNet50, Hardware.Cloud.A100, **A1_CONFIG)
    per_gpu = profile.throughput.m_as("1/s")
    node = A1_GPUS * per_gpu
    return {
        "profile": profile, "feasible": profile.feasible, "per_gpu": per_gpu, "node": node,
        "reported": A1_REPORTED_IMG_PER_S,
        "rel_error": abs(node - A1_REPORTED_IMG_PER_S) / A1_REPORTED_IMG_PER_S,
        "source": ("SingleNodeModel().solve(Models.Vision.ResNet50, Hardware.Cloud.A100, batch_size=256, "
                   "precision='fp16', efficiency=0.19, is_training=True)"),
    }


def anchor_two() -> dict:
    model, h100 = Models.Language.Llama2_70B, Hardware.Cloud.H100
    _, bpp = resolve_precision("fp16")
    weights = model.size_in_bytes(bpp)
    floor = weights / (A2_TP_DEGREE * h100.memory.bandwidth)
    return {
        "weights": weights, "floor_ms": floor.m_as("ms"), "tp": A2_TP_DEGREE,
        "source": "Models.Language.Llama2_70B.size_in_bytes(FP16) / (2 x Hardware.Cloud.H100.memory.bandwidth)",
    }


def anchor_three() -> dict:
    fleet = llama3_16k_fleet()
    result = DistributedModel().solve(Models.Language.Llama3_405B, fleet, **A3_CONFIG)
    mfu = result.scaling_efficiency * A3_ETA
    return {
        "fleet": fleet, "result": result, "mfu": mfu, "reported": A3_REPORTED_MFU,
        "rel_error": abs(mfu - A3_REPORTED_MFU) / A3_REPORTED_MFU,
        "abs_error_points": abs(mfu - A3_REPORTED_MFU) * 100,
        "source": ("DistributedModel().solve(Models.Language.Llama3_405B, Fleet(DGX_H100 x 2048, InfiniBand_NDR), "
                   "batch_size=2048, efficiency=0.42, tp_size=8, pp_size=16, microbatch_count=64, "
                   "overlap_comm=True, overlap_efficiency=0.85, seq_len=8192)"),
    }


def anchor_four() -> dict:
    fleet = palm_fleet()
    result = DistributedModel().solve(palm_540b(), fleet, **A4_CONFIG)
    mfu = result.scaling_efficiency * A4_ETA
    return {
        "fleet": fleet, "result": result, "mfu": mfu, "reported": A4_REPORTED_MFU,
        "rel_error": abs(mfu - A4_REPORTED_MFU) / A4_REPORTED_MFU,
        "abs_error_points": abs(mfu - A4_REPORTED_MFU) * 100,
        "source": ("DistributedModel().solve(PaLM-540B [118 layers, d=18432, 48 heads], Fleet(1536 x 4-chip "
                   "Hardware.Cloud.TPUv4 hosts, 400 GB/s intra-host, 24 GB/s ICI-class fabric, 2x oversubscribed), "
                   "batch_size=2048, efficiency=0.47, tp_size=4, pp_size=1, overlap_comm=True)"),
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
        "source": ("ParallelismOptimizer().solve(Models.Language.Llama3_405B, Fleet(DGX_H100 x 2048, InfiniBand_NDR), "
                   "batch_size=2048, efficiency=0.42, overlap_comm=True)"),
    }

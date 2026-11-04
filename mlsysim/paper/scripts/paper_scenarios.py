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

import math
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
# A1: NVIDIA DeepLearningExamples, TensorFlow ResNet-50 v1.5, AMP + XLA, DGX A100
#     (8x A100 40GB), TensorFlow 20.06-tf1-py3 container, batch 256 per GPU; images/s for the node.
A1_REPORTED_IMG_PER_S = 17_400
# A2: LLM-Inference-Bench (Chitty-Venkata et al., 2024, arXiv:2411.00136), raw data in
#     github.com/argonne-lcf/LLM-Inference-Bench at commit 4211f5d4e9bc0145ed541b6d372d5d8223b44235
#     (2025-07-18), file Plots/All_results.csv, rows "Nvidia H100 GPU, 4, <engine>,
#     meta-llama/Llama-2-70b-hf, 128, 1": Llama-2-70B, FP16, batch 1, four H100 SXM5 80GB with
#     TP=4 (the paper's Sec. III states H100 SXM5 80GB and TP = number of GPUs), 128 input and
#     128 output tokens. Values are end-to-end request latency in seconds. The source reports no
#     time-to-first-token for this configuration, so the per-token value below is end-to-end
#     latency divided by output tokens and still contains prefill; A2_PREFILL_REF bounds that share.
#     paper/scripts/check_anchor_two_source.py re-derives every A2 constant from the source files.
A2_SOURCE_COMMIT = "4211f5d4e9bc0145ed541b6d372d5d8223b44235"
A2_REPORTED_E2E_S = {"TensorRT-LLM": 2.3088122606277466, "vLLM": 2.37026520420089}
A2_REPORTED_INPUT_TOKENS = 128
A2_REPORTED_OUTPUT_TOKENS = 128
# Same source, Plots/Fig_22/all_results_TTFT_ITL.csv: measured TTFT for Llama-2-7B on one H100 SXM5,
# FP16, 16 prompts of 1,024 tokens. These are the only H100 TTFT rows in the source; they set the
# measured prefill rate used to bound the prefill share of the A2 end-to-end latencies.
A2_PREFILL_REF = dict(prompts=16, input_tokens=1024, gpus=1,
                      ttft_s={"TensorRT-LLM": 0.38651649951934813, "vLLM": 0.4216477160962919})
# Declared decode memory-bandwidth efficiency for the labelled estimate (not the floor):
# Databricks (Agarwal et al., 2023, "LLM Inference Performance Engineering: Best Practices")
# report a model bandwidth utilization of 60% at batch size 1 on 2x H100-80GB (MPT-7B).
A2_DECODE_BW_EFFICIENCY = 0.60
# Pessimistic factor for the prefill bound: assume TP=4, batch-1 prefill runs this many times
# slower per FLOP than the measured single-GPU, 16-prompt reference.
A2_PREFILL_PESSIMISM = 4.0
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
A1_CONFIG = dict(batch_size=256, precision="fp16", efficiency=0.19, is_training=True)
# Four-way tensor parallelism matches the published A2 measurement; two H100s is the memory minimum.
A2_TP_DEGREE = 4
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
        "source": ("SingleNodeModel().solve(Models.Vision.ResNet50, Hardware.Cloud.A100, batch_size=256, "
                   "precision='fp16', efficiency=0.19, is_training=True)"),
    }


def anchor_two() -> dict:
    """Llama-2-70B FP16 batch-1 decode on H100: theoretical floor, a labelled estimate, and the
    published end-to-end-derived per-token latency (see the A2 constants for provenance).

    * floor: weights streamed once per token at peak aggregate HBM bandwidth (no efficiency term);
    * estimate: the floor divided by the declared A2_DECODE_BW_EFFICIENCY;
    * reported: end-to-end latency / output tokens per engine, which includes prefill;
    * prefill bound: prefill time of the A2 prompt at the source's own measured H100 prefill rate
      (scaled by FLOPs and GPU count), as a fraction of the smaller end-to-end latency.
    """
    model, h100 = Models.Language.Llama2_70B, Hardware.Cloud.H100
    _, bpp = resolve_precision("fp16")
    weights = model.size_in_bytes(bpp)
    min_tp = math.ceil((weights / h100.memory.capacity).m_as("dimensionless"))
    floor_ms = (weights / (A2_TP_DEGREE * h100.memory.bandwidth)).m_as("ms")
    estimate_ms = floor_ms / A2_DECODE_BW_EFFICIENCY
    reported_ms = {eng: 1e3 * e2e / A2_REPORTED_OUTPUT_TOKENS for eng, e2e in A2_REPORTED_E2E_S.items()}
    floor_err = {eng: abs(floor_ms - rep) / rep for eng, rep in reported_ms.items()}
    estimate_err = {eng: abs(estimate_ms - rep) / rep for eng, rep in reported_ms.items()}

    # Prefill bound: measured prefill FLOP rate per GPU from the 7B reference (slowest engine),
    # applied to the 70B prompt on A2_TP_DEGREE GPUs.
    ref = A2_PREFILL_REF
    ref_params = Models.Language.Llama2_7B.parameters.m_as("param")
    ref_flops = 2 * ref_params * ref["prompts"] * ref["input_tokens"]
    rate_per_gpu = ref_flops / (max(ref["ttft_s"].values()) * ref["gpus"])
    prefill_s = 2 * model.parameters.m_as("param") * A2_REPORTED_INPUT_TOKENS / (A2_TP_DEGREE * rate_per_gpu)
    e2e_min = min(A2_REPORTED_E2E_S.values())
    return {
        "weights": weights, "tp": A2_TP_DEGREE, "min_tp": min_tp,
        "floor_ms": floor_ms, "estimate_ms": estimate_ms, "bw_eta": A2_DECODE_BW_EFFICIENCY,
        "reported_ms": reported_ms, "floor_err": floor_err, "estimate_err": estimate_err,
        "prefill_share": prefill_s / e2e_min,
        "prefill_share_pessimistic": A2_PREFILL_PESSIMISM * prefill_s / e2e_min,
        "source": (f"Models.Language.Llama2_70B.size_in_bytes(FP16) / ({A2_TP_DEGREE} x "
                   "Hardware.Cloud.H100.memory.bandwidth)"),
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

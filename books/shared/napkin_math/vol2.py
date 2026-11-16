"""Napkin Math, Volume II: the fleet (hundreds to thousands of accelerators on a network).

Every number below is computed here from the mlsysim registry or scenarios.
Most worked examples reuse the scenarios that Volume II's own LEGO cells use
(the 175B reference job on 1,024 H100s, the 10,000-GPU reliability cases, the
Llama 70B serving replica), so the sheet matches the book. Inputs that a
scenario or this sheet chose only for an example are tagged with assumed() on
the page; the ones this module adds itself are listed in SCENARIO. Constants
from outside the registry would be listed in EXTERNAL with their source.
"""
from __future__ import annotations

import math
import warnings

from mlsysim import (BYTES_ADAM_STATE, BYTES_FP16, BYTES_FP32, HOURS_PER_DAY, SEC_PER_DAY, SEC_PER_HOUR,
                     Hardware as HW, Infrastructure as INF, Models as M,
                     ReferenceStats as RS, Systems as SYS,
                     calc_checkpoint_waste, calc_hierarchical_allreduce_time,
                     calc_mtbf_cluster, calc_pipeline_bubble, calc_ring_allreduce_time,
                     calc_ring_tree_crossover_size, calc_tree_allreduce_time, calc_young_daly_interval)
from mlsysim.engine.scenarios import Scenarios as SC

from .core import (Card, Chart, Col, Know, Note, Page, Plot, Sheet, Table,
                   assumed, fmt, frac, n, v)

warnings.filterwarnings("ignore")  # registry deprecation chatter only

# Megatron-style tensor parallelism: 2 AllReduces per transformer layer forward, 2 backward
# (Shoeybi et al. 2019, Megatron-LM, Sec. 3). A structural count, not a measured value.
EXTERNAL = {"tp_allreduces_per_layer": 4}
# tp_seqs: sequences per tensor-parallel microbatch (matches the reference step's 1).
# tp_ring: tensor-parallel ring size (one DGX node). tiers: cluster rows of the reference table.
SCENARIO = dict(tp_seqs=1, tp_ring=8, tiers=("Research_256", "Training_1K_NDR", "Frontier_8K", "Mega_100K"))


def _m(q, unit):
    return q.to(unit).magnitude


def build() -> Sheet:
    S = SCENARIO
    A = TAG = assumed()

    # ---------------------------------------------------------------- registry
    H = HW.Cloud.H100
    PEAK = _m(H.compute.peak_flops, "flop/s")
    HBM = _m(H.memory.capacity, "B")
    HBM_GIB = _m(H.memory.capacity, "GiB")
    TDP = _m(H.tdp, "W")
    NVL = _m(H.nvlink.bandwidth_per_direction, "B/s")
    NODE = SYS.Nodes.DGX_H100
    AIR_KW = INF.AIR_COOLING_LIMIT_KW
    NDR = SYS.Fabrics.InfiniBand_NDR
    BETA = _m(NDR.bandwidth, "B/s")
    ALPHA_Q = SYS.SwitchFabric.AlphaNdr
    ALPHA = _m(ALPHA_Q, "s")
    GPU_MTTF = SYS.Reliability.Gpu.mttf_hours
    SDC = float(SYS.Reliability.SdcRatePerGpuHr)
    CKPT_BW = SYS.Reliability.Recovery.checkpoint_write_bw_gbs * 1e9
    BPP_STATE = _m(RS.MixedPrecisionAdam.TrainingStateBytesPerParameter, "B/param")
    BPP_CKPT = _m(RS.MixedPrecisionAdam.CheckpointBytesPerParameter, "B/param")
    PUE = {k: getattr(INF.FacilityCooling, k).pue for k in ("LiquidCooled", "Legacy")}
    CI = {k: getattr(INF.Grids, k).carbon_intensity_g_kwh for k in ("Quebec", "Poland", "US_Avg")}

    B16, B32, BADAM = (_m(q, "B") for q in (BYTES_FP16, BYTES_FP32, BYTES_ADAM_STATE))
    H_DAY, S_DAY, S_HR = float(HOURS_PER_DAY), float(SEC_PER_DAY), float(SEC_PER_HOUR)
    TP_AR = EXTERNAL["tp_allreduces_per_layer"]

    G3 = M.Language.GPT3
    P3 = _m(G3.parameters, "param")
    D3 = _m(G3.training_tokens, "count")
    L3, W3 = G3.layers, G3.hidden_dim
    G3_MWH = _m(G3.training_energy_mwh, "MWh")

    # ---------------------------------------------------------------- page 1: the reference step
    ref, ref8 = SC.DenseLLMPretrainingStep, SC.DenseLLMPretrainingStep8K

    def step(sc):
        r = sc.result
        t = dict(comp=_m(r.node_profile.latency, "s"), comm=_m(r.communication_latency, "s"),
                 bub=_m(r.pipeline_bubble_latency, "s"), tot=_m(r.step_latency_total, "s"),
                 eff=r.scaling_efficiency, mem=_m(r.node_profile.memory_footprint, "B"))
        t["ovl"] = t["comp"] + t["comm"] + t["bub"] - t["tot"]   # hidden communication
        t["exp"] = t["comm"] - t["ovl"]                          # exposed communication
        return t
    s1, s8 = step(ref), step(ref8)
    N1, N8 = ref.fleet.total_accelerators, ref8.fleet.total_accelerators
    TP, PP, DP, MB = ref.tensor_parallel, ref.pipeline_depth, ref.data_parallel, ref.microbatches_per_replica
    MB8 = ref8.microbatches_per_replica
    ETA = ref.local_efficiency
    replica = TP * PP
    speedup_8x = s1["tot"] / s8["tot"]

    # training compute and time
    C3 = 6 * P3 * D3
    t_days = C3 / (N1 * PEAK * ETA) / S_DAY
    t_days_fleet = t_days / s1["eff"]

    # training state and ZeRO
    z = SC.Llama2_7B_ZeroStages64
    zr = z.result
    zP, zN = _m(z.workload.parameters, "param"), z.n_devices
    zstage = {k: _m(q, "GB") for k, q in zr.stage_state.items()}
    state_gpu = _m(ref.state_per_gpu, "B")

    # input bandwidth
    img, txt = SC.ImageInputDemand2K, SC.TextInputDemand2K   # per-GPU demand, applied to the reference fleet
    img_kb = _m(img.bytes_per_sample, "kB")
    img_bw = N1 * img.samples_per_step * _m(img.bytes_per_sample, "B") / _m(img.step_time, "s") / 1e9
    txt_bw = N1 * txt.samples_per_step * _m(txt.bytes_per_sample, "B") / _m(txt.step_time, "s") / 1e6

    # alpha-beta and collectives
    ab = SC.AlphaBetaNdrRoce
    msg_small, msg_big = _m(ab.messages["control"], "B"), _m(ab.messages["shard"], "B")
    t_small, t_big = ALPHA + msg_small / BETA, ALPHA + msg_big / BETA
    nstar = ALPHA * BETA
    gr = SC.Llama2_70BGradientRing64Ndr
    grM, grN = _m(gr.message, "B"), gr.group_sizes[0]
    gr_ring = _m(calc_ring_allreduce_time(gr.message, grN, NDR.bandwidth, ALPHA_Q), "s")
    gr_tree = _m(calc_tree_allreduce_time(gr.message, grN, NDR.bandwidth, ALPHA_Q), "s")
    def xover(Nn):  # the book's exact ring/tree crossover, @eq-ring-tree-crossover
        lg = math.log2(Nn)
        return nstar * ((Nn - 1) - lg) / (lg - (Nn - 1) / Nn)
    xover_64, xover_1k = xover(grN), xover(N1)
    hi = SC.HierarchicalAllReduce8x8DgxH100
    hi_t = _m(calc_hierarchical_allreduce_time(hi.message, hi.n_nodes, hi.gpus_per_node, hi.intra_bandwidth,
                                               hi.inter_bandwidth, inter_node_lat=hi.inter_alpha), "s")
    hi_flat = _m(hi.result.flat_ring_time, "s")
    ov = SC.RingAllReduceAcross4to1Spine
    ovr = ov.result
    OVR, OVF = ov.oversubscription_ratio, ov.communication_fraction
    bis_full, bis_over = _m(ovr.bisection_full, "TB/s"), _m(ovr.bisection_oversubscribed, "TB/s")
    thr = 1 / (1 - OVF + OVF * OVR)

    # pipeline bubble
    pb = SC.Pipeline8Stage32Micro
    bub_32 = calc_pipeline_bubble(pb.stages, pb.microbatches)
    bub_v = calc_pipeline_bubble(pb.stages, pb.microbatches, v_stages=2)
    bub_ref = calc_pipeline_bubble(PP, MB)
    bub_8k = calc_pipeline_bubble(PP, MB8)

    # tensor-parallel exchange, one GPT-3 layer
    m_act = S["tp_seqs"] * G3.context_length * W3 * B16
    ALPHA_NVL = _m(SC.CriticalSizeByInterconnect.alphas["NVLink"], "s")   # the book's NVLink alpha, also on the chart
    tp_nvl = _m(calc_ring_allreduce_time(m_act, S["tp_ring"], NVL, ALPHA_NVL), "s")
    tp_layers_stage = L3 // PP
    tp_ndr = _m(calc_ring_allreduce_time(m_act, S["tp_ring"], BETA, ALPHA_Q), "s")

    # straggler, goodput
    st = SC.HalfSpeedStraggler1K
    gp = SC.Llama3_405BGoodput16K
    gp_gpus = SYS.Clusters.Meta_Llama3_16K_H100.total_accelerators

    # ---------------------------------------------------------------- page 2
    gf, ff = SC.GpuFailuresTraining10K, SC.FullSystemFailures10K
    gfr, ffr = gf.result, ff.result
    mtbf_gpu, mtbf_sys = _m(gfr.interval, "h"), _m(ffr.mtbf, "h")
    n10k = gf.n_components
    sdc_h = 1 / (SDC * n10k)

    ck_bytes = P3 * BPP_CKPT
    t_write = ck_bytes / CKPT_BW
    local_bw = SYS.Storage.LocalNvmeGen4x4.devices_per_node * _m(SYS.Storage.LocalNvmeGen4.bandwidth, "B/s")
    nodes1 = ref.fleet.count
    shard1 = ck_bytes / nodes1
    t_local1 = shard1 / local_bw
    tau = _m(calc_young_daly_interval(t_write, mtbf_sys * S_HR), "s")
    waste = calc_checkpoint_waste(t_write, tau, mtbf_sys * S_HR)
    rb = SC.Gpt3RecoveryBudget10K
    rbr = rb.result
    rec_s = _m(rbr.total, "s")
    rec_share = rec_s / (mtbf_sys * S_HR)

    fan, alarms = SC.ServingFanout100Leaves, SC.ThreeSigmaModelFleetAlerts

    kv = SC.Llama3_70BLongContextAdmissionDgxH100
    kvr, wl = kv.result, kv.workload
    dec1 = SC.Llama3_70B_DecodeH100.result
    HBM_BW = _m(H.memory.bandwidth, "B/s")
    dec_tp = SC.Llama2_70BDecodeFloorTp8.tp_size
    dec_w = _m(dec1.weight_bytes, "B")
    dec_tp_s = dec_w / (dec_tp * HBM_BW)   # same Llama 3 70B weights, split over TP8
    ll = SC.VisionServingLittlesLaw
    sv = SC.LlmServingFirstYear
    fed = SC.PhoneImagesVersusQuantizedUpdates

    gpl = SC.QuebecVersusPolandPlacement
    gplr = gpl.result
    g3_t = G3_MWH * CI["US_Avg"] / 1000  # MWh x g/kWh = kg; / 1000 = t
    rack = SYS.Racks.DGX_H100_4Node
    fp = SC.Reference25KFacilityPower
    fpr = fp.result
    it_gpu_kw = TDP * fp.it_ratio / 1e3   # the book's per-GPU IT basis: GPU TDP x rack-profile IT ratio
    rack_gpus = rack.nodes_per_rack * NODE.accelerators_per_node
    rack_kw = rack_gpus * it_gpu_kw
    pr = SC.PueRetrofit2MW
    prr = pr.result
    own = SC.OwnVsRentH100Cluster10K
    assert _m(own.rent_rate, "USD/h") == _m(INF.Pricing.Cloud.GpuTrainingPerHour.rate, "USD/h"), "rent rate is the registry cloud price"
    dp = SC.SalaryMeanLaplaceRelease
    pgd = SC.PgdSevenTraining
    can = SC.CanaryVersusCutoverExposure

    def gb(x, d=1):
        return f"{x / 1e9:,.{d}f} GB"

    def ms(x, d=1):
        return f"{x * 1e3:,.{d}f} ms"

    def us(x, d=1):
        return f"{x * 1e6:,.{d}f} µs"

    # ---------------------------------------------------------------- chart: alpha-beta
    CS = SC.CriticalSizeByInterconnect
    links = [("NVLink", NVL, _m(CS.alphas["NVLink"], "s"), "line"),
             ("IB NDR", BETA, ALPHA, "line warn"),
             ("RoCE 100G", _m(SYS.Fabrics.RoCE_100G.bandwidth, "B/s"), _m(SYS.SwitchFabric.AlphaRoce, "s"), "line muted")]
    ap = Plot((1e2, 1e9), (1, 1e5), xlog=True, ylog=True, W=316, H=77, right=12, bottom=24)
    xl = {1e2: "100 B", 1e3: "1 KB", 1e4: "", 1e5: "100 KB", 1e6: "", 1e7: "10 MB", 1e8: "", 1e9: "1 GB"}
    yl = {1: "1 µs", 10: "", 100: "", 1e3: "1 ms", 1e4: "", 1e5: "100 ms"}
    ap.grid(list(xl), list(yl), xfmt=lambda x: xl[x], yfmt=lambda y: yl[y])
    for name, bw, al, cls in links:
        ap.line([(10 ** (2 + k * 0.1), (al + 10 ** (2 + k * 0.1) / bw) * 1e6) for k in range(71)], cls)
        ap.point(al * bw, 2 * al * 1e6)
    for k, (name, bw, al, cls) in enumerate(links):
        y = ap.t + 3 + k * 9
        ap.parts.append(f'<line x1="{ap.l + 6}" x2="{ap.l + 20}" y1="{y - 3}" y2="{y - 3}" class="{cls}"/>')
        ap.text(ap.l + 24, y, f"{name.split()[0] if name != 'IB NDR' else 'NDR'}: {al * 1e6:g} µs, {bw / 1e9:g} GB/s")
    ap.text(ap.X(8e8), ap.Y(2.2), "β-bound", "end", cls="lab ital")
    ap.axes("message size n · legend: α, β · dot: n* = α·β", "time")

    # ---------------------------------------------------------------- chart: Fleet Law terms at two scales
    def bars():
        W, x0, x1, bh = 340, 54, 334, 14
        out = [f'<svg viewBox="0 0 {W} 64" class="chart" role="img" aria-label="Reference step decomposition at two scales">']
        for k, (lab, t) in enumerate(((f"{fmt(N1)} GPUs", s1), (f"{fmt(N8)} GPUs", s8))):
            y = 3 + k * 31
            out.append(f'<text x="{x0 - 4}" y="{y + 10}" class="lab" text-anchor="end">{lab}</text>')
            xs = x0
            for key, style in (("comp", 'class="fill-accent"'), ("exp", 'class="fill-warn"'), ("bub", 'class="fill-neutral"')):
                w = (x1 - x0) * t[key] / t["tot"]
                out.append(f'<rect x="{xs:.1f}" y="{y}" width="{w:.1f}" height="{bh}" {style}/>')
                xs += w
            out.append(f'<text x="{x0 + 4}" y="{y + 10}" class="mono">compute {fmt(t["comp"] / t["tot"] * 100)}%</text>')
            out.append(f'<text x="{x1}" y="{y + bh + 9}" class="tick" text-anchor="end">step {t["tot"]:.2f} s (modeled) · exposed communication {fmt(t["exp"] / t["tot"] * 100)}% · bubble {fmt(t["bub"] / t["tot"] * 100)}% of step</text>')
        out.append("</svg>")
        return "".join(out)

    # ---------------------------------------------------------------- chart: Young-Daly
    v_txt_tau = "optimum"
    yd = Plot((60, 3600 * 3), (0, 0.25), xlog=True, W=340, H=112, right=14, bottom=25)
    yd.grid([60, 300, 900, 3600, 10800], [0, 0.1, 0.2],
            xfmt=lambda x: {60: "1 min", 300: "5 min", 900: "15 min", 3600: "1 h", 10800: "3 h"}[x],
            yfmt=lambda y: f"{y * 100:.0f}%")
    xs = [60 * 10 ** (k * math.log10(180) / 80) for k in range(81)]
    mtbf_s = mtbf_sys * S_HR
    ymax = yd.y1
    for f_, cls in ((lambda x: t_write / x, "line alt"), (lambda x: x / (2 * mtbf_s), "line alt"),
                    (lambda x: calc_checkpoint_waste(t_write, x, mtbf_s), "line")):
        yd.line([(x, f_(x)) for x in xs if f_(x) <= ymax], cls)  # draw only inside the plot
    yd.point(tau, waste, f"{v_txt_tau} {tau / 60:.0f} min, W = {waste * 100:.1f}%", "middle", 0, -12)
    yd.text(yd.X(140), yd.Y(0.12), "saving", "end", cls="lab ital")
    yd.text(yd.X(4800), yd.Y(0.12), "rework", cls="lab ital")
    yd.axes(f"checkpoint interval (GPT-3 checkpoint, {fmt(n10k)}-GPU fleet, MTBF {mtbf_sys:.2f} h)", "waste W")

    # ---------------------------------------------------------------- page 1
    band = (f'<div class="eqbig">{v("T", "step")} = {frac(v("T", "compute"), v("N"))} + {v("T", "comm")} + {v("T", "sync")} − {v("T", "overlap")}'
            f'<span class="lbl">Compute · Communication · Coordination − Overlap</span></div>'
            f'<div class="txt"><div><b>The Fleet Law (Ch 1).</b> The 175B reference job on {fmt(N1)} H100s ({DP} replicas of TP{TP} × PP{PP}), modeled: '
            f'{s1["comp"]:.2f} s compute + {s1["comm"]:.2f} s {v("T", "comm")} + {s1["bub"]:.2f} s {v("T", "sync")} (pipeline bubble) − {s1["ovl"]:.2f} s {v("T", "overlap")} ≈ {n(f"{s1["tot"]:.2f} s")}. '
            f'Scaling efficiency {v("η", "scaling")} = ({v("T", "compute")}/{v("N")}) / {v("T", "step")} = {n(fmt(s1["eff"] * 100) + "%")}.</div>'
            f'<div><b>{N8 // N1}× the GPUs, same batch:</b> {s8["tot"]:.2f} s per step, only {speedup_8x:.1f}× faster; {v("η", "scaling")} falls to {fmt(s8["eff"] * 100)}%.</div></div>'
            f'<div class="side"><div><b>Pull the largest term\'s lever</b></div><div>{v("T", "comm")}: faster fabric, overlap, compression</div>'
            f'<div>{v("T", "sync")}: fewer barriers; one GPU at half speed halves the fleet</div><div>{v("T", "compute")}/{v("N")}: faster or more accelerators</div></div>')

    zero_rows = [["state per GPU (GB)"] + [fmt(zstage["DDP"])] + [f"{zstage[k]:.1f}" for k in ("ZeRO-1", "ZeRO-2", "ZeRO-3")]]

    p1 = Page(
        question="Will it scale?",
        sub="Read left to right: what one training job needs, how fast bytes move between accelerators, and where each step's time goes.",
        meta=f"reference: {fmt(N1)} × H100, DGX H100 nodes, IB NDR",
        notation=[
            ("Job", f'{v("P")} parameters · {v("D")} tokens · {v("B")} sequences per microbatch · {v("S")} sequence length · {v("d")} width · {v("m")} microbatches · {v("t")} wall-clock time · {v("T", "step")} step time'),
            ("Fleet", f'{v("N")} devices · {v("N", "DP")}, {v("N", "TP")}, {v("N", "PP")} data, tensor, pipeline degrees · {v("R", "peak")} peak FLOP/s · {v("η", "hw")} fraction of peak · {v("η", "scaling")} scaling efficiency · {v("V")} interleaved stages per device'),
            ("Network", f'{v("α")} per-message latency · {v("β")} bytes/s per direction · {v("n")} message size · {v("M")} payload per rank · {v("f")} share of step communicating · R:1 oversubscription · {A}= example input'),
        ],
        know=[
            Know("2(N−1)/N ≈ 2", "bytes sent per payload byte, ring AllReduce"),
            Know(f"{fmt(NVL / 1e9)} · {fmt(BETA / 1e9)} GB/s", "H100 NVLink · NDR port, per direction"),
            Know(f"{ALPHA * 1e6:.1f} µs · {fmt(nstar / 1e3)} KB", "NDR α · crossover size α·β"),
            Know(f"{fmt(BPP_STATE)} · {fmt(BPP_CKPT)} bytes", "per parameter: training state · checkpoint"),
            Know(f"{fmt(mtbf_gpu)} h", f"between failures of {fmt(n10k)} GPUs, {fmt(GPU_MTTF / 1e3)}K h MTTF each"),
            Know(f"{PUE['LiquidCooled']:.2f}–{PUE['Legacy']:.2f}", f"PUE: liquid-cooled to {INF.FacilityCooling.Legacy.name.lower()}"),
            Know(f"{fmt(CI['Quebec'])} · {fmt(CI['Poland'])} g/kWh", f"grid CO₂: Quebec · Poland ({fmt(CI['Poland'] / CI['Quebec'])}×)"),
        ],
        band=band,
        cols=[
            Col("1 What does the job need?", [
                Card("Training compute and time", f'{v("t")} ≈ 6{v("P")}{v("D")} / ({v("N")} · {v("R", "peak")} · {v("η", "hw")} · {v("η", "scaling")})',
                     chapter="Ch 2",
                     ex=f'GPT-3 ({P3 / 1e9:.0f}B params, {D3 / 1e9:.0f}B tokens) is {C3:.2e} FLOPs. On {fmt(N1)} H100s at {A}{v("η", "hw")} = {fmt(ETA * 100)}%: '
                        f'{t_days:.1f} days of compute; ÷ {v("η", "scaling")} = {s1["eff"]:.2f} → {n(f"≈ {t_days_fleet:.1f} days")}, before any failure.'),
                Card("Training state per device", None, chapter="Ch 6",
                     holds=f'Bytes: {fmt(B16)} weights + {fmt(B16)} gradients + {fmt(B32)} FP32 master + {fmt(BADAM)} Adam moments, activations extra. ZeRO-3 moves {zr.traffic_multiple["ZeRO-3"]:.1f}× the bytes of DDP.'),
                Table(["bytes per GPU", "DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"], [
                    ["formula", f"{fmt(BPP_STATE)}<i>P</i>", f"{fmt(2 * B16)}<i>P</i>+{fmt(B32 + BADAM)}<i>P</i>/<i>N</i><sub>DP</sub>",
                     f"{fmt(B16)}<i>P</i>+{fmt(B16 + B32 + BADAM)}<i>P</i>/<i>N</i><sub>DP</sub>", f"{fmt(BPP_STATE)}<i>P</i>/<i>N</i><sub>DP</sub>"],
                    [f"7B, {v('N', 'DP')} = {zN} (GB)"] + [fmt(zstage["DDP"])] + [f"{zstage[k]:.1f}" for k in ("ZeRO-1", "ZeRO-2", "ZeRO-3")]]),
                Card("Smallest replica that fits", f'{frac(fmt(BPP_STATE) + v("P"), v("N", "TP") + " · " + v("N", "PP"))} + activations ≤ HBM, &nbsp; {v("N", "DP")} = {frac(v("N"), v("N", "TP") + " · " + v("N", "PP"))}',
                     chapter="Ch 6",
                     ex=f'GPT-3: {BPP_STATE * P3 / 1e12:.1f} TB of state. TP{TP} × PP{PP} = {replica} GPUs: {gb(state_gpu)} each, {n(gb(s1["mem"]))} with activations, '
                        f'within {fmt(HBM_GIB)} GiB = {HBM / 1e9:.1f} GB. {fmt(N1)} GPUs hold {v("N", "DP")} = {n(str(DP))} replicas.'),
                Card("Can storage feed the fleet?", f'storage ≥ {v("N")} × bytes per GPU per step / {v("T", "step")}',
                     chapter="Ch 4",
                     ex=f'{fmt(N1)} GPUs, {A}per {_m(img.step_time, "ms"):.0f} ms step each reading {img.samples_per_step} images × {img_kb:.2f} KB: {n(f"{img_bw:.1f} GB/s")}; '
                        f'{fmt(txt.samples_per_step)} tokens × {_m(txt.bytes_per_sample, "B"):.0f} bytes: {txt_bw:.1f} MB/s.'),
            ], "Ch 2, 4, 6"),
            Col("2 How fast do bytes move?", [
                Chart(ap.svg("Transfer time versus message size")),
                Card("Point to point: the α-β model", f'{v("T")}({v("n")}) = {v("α")} + {frac(v("n"), v("β"))}, &nbsp; crossover {v("n")}* = {v("α")} · {v("β")}',
                     chapter="Ch 3",
                     ex=f'On NDR a {fmt(msg_small / 1e3)} KB control message takes {us(t_small, 2)}, '
                        f'{n(fmt(ALPHA / t_small * 100) + "% latency")}; a {fmt(msg_big / 1e6)} MB shard, {n(ms(t_big, 2))}.'),
                Table(["collective (Ch 5)", "latency term", "bandwidth term"], [
                    ["Ring AllReduce", "2(<i>N</i>−1)<i>α</i>", "2(<i>N</i>−1)/<i>N</i> · <i>M</i>/<i>β</i>"],
                    ["Tree, unsegmented", "2 log₂<i>N</i> · <i>α</i>", "2 log₂<i>N</i> · <i>M</i>/<i>β</i>"],
                    ["ReduceScatter/AllGather", "(<i>N</i>−1)<i>α</i>", "(<i>N</i>−1)/<i>N</i> · <i>M</i>/<i>β</i>"],
                    ["AllToAll (MoE)", "(<i>N</i>−1)<i>α</i>", "(<i>N</i>−1)/<i>N</i> · <i>M</i>/<i>β</i>"],
                ]),
                f'<p class="ex">Llama 2 70B gradients ({gb(grM, 0)} BF16), {grN} NDR GPUs: ring {n(f"{gr_ring:.2f} s")}, unsegmented tree {gr_tree:.1f} s; pipelined trees near the ring. '
                f'App. D\'s exact crossover: trees win below {fmt(xover_64 / 1e3)} KB here, {xover_1k / 1e6:.2f} MB at {fmt(N1)}.</p>',
                Card("Oversubscription R:1 at the spine", f'throughput = 1 / (1 − {v("f")} + {v("f")} · R)',
                     chapter="Ch 3",
                     ex=f'{fmt(ov.fleet.total_accelerators)} NDR GPUs: bisection {v("N")}{v("β")}/2R = {fmt(bis_full, 1)} → {fmt(bis_over, 1)} TB/s at {OVR:.0f}:1; {A}{v("f")} = {fmt(OVF * 100)}% → {n(fmt(thr * 100) + "%", True)}.'),
            ], "Ch 3, 5"),
            Col("3 Where does the step go?", [
                Chart(bars(), f'Fleet Law terms of the reference step at {fmt(N1)} and {fmt(N8)} GPUs: compute (blue), exposed communication (amber), pipeline bubble (gray).'),
                Card("Pipeline bubble", f'{v("b", "pipe")} = {frac(v("N", "PP") + " − 1", v("N", "PP") + " − 1 + " + v("V") + v("m"))}',
                     chapter="Ch 6",
                     holds=f'Share of pipeline time (fill, drain, {v("m")} microbatches), balanced 1F1B stages. The bars above divide by the whole step instead.',
                     ex=f'{v("N", "PP")} = {pb.stages}, {v("m")} = {pb.microbatches}: {n(fmt(bub_32 * 100, 1) + "%")} idle; {fmt(bub_v * 100, 1)}% at {v("V")} = 2. '
                        f'Reference job: {v("m")} = {MB} → {fmt(bub_ref * 100, 1)}%; at {fmt(N8)} GPUs each replica gets {v("m")} = {MB8} → {n(fmt(bub_8k * 100, 1) + "%", True)}.'),
                Card("Tensor parallelism stays inside the node", f'{TP_AR} AllReduces per layer of {v("B")} · {v("S")} · {v("d")} · {fmt(B16)} bytes',
                     chapter="Ch 6, 10",
                     ex=f'GPT-3 ({v("S")} = {fmt(G3.context_length)}, {v("d")} = {fmt(W3)}, {A}{v("B")} = {S["tp_seqs"]}): {m_act / 1e6:.0f} MB. A ring of {S["tp_ring"]} takes '
                        f'{n(us(tp_nvl, 0))} on NVLink, {n(ms(tp_ndr, 2), True)} on NDR ({tp_ndr / tp_nvl:.0f}×). Per stage: {tp_layers_stage} of {L3} layers, {TP_AR * tp_layers_stage} AllReduces per microbatch.'),
                Card("Goodput: what the held GPU-hours buy", "goodput = scheduling × runtime × MFU",
                     chapter="Ch 9",
                     ex=f'Llama 3 405B on {fmt(gp_gpus)} dedicated H100s: {gp.scheduling:.1f} × {gp.runtime:.2f} × {gp.program:.2f} = '
                        f'{n(fmt(gp.result.mpg * 100, 1) + "%")} of the held capacity becomes progress at peak rate.'),
            ], "Ch 6, 7, 9"),
        ],
    )

    # ---------------------------------------------------------------- page 2
    trows = []
    for k in S["tiers"]:
        c = getattr(SYS.Clusters, k)
        Nc = c.total_accelerators
        mt = _m(calc_mtbf_cluster(GPU_MTTF, Nc), "h")
        ring_alpha = 2 * (Nc - 1) * ALPHA
        fac = Nc * it_gpu_kw * fp.pue / 1e3
        trows.append([c.name.split(" (")[0] + (" (reference)" if Nc == N1 else ""), fmt(Nc), fmt(c.count),
                      f"{mt:,.1f} h" if mt < 10 else f"{fmt(mt)} h", f"{H_DAY / mt:.1f}",
                      ms(ring_alpha, 2 if ring_alpha < 0.01 else 0), f"{Nc / 2 * BETA / 1e12:,.1f} TB/s", f"{fac:,.1f} MW"])

    p2 = Page(
        question="Will it survive and serve?",
        sub="Failures as a steady state, serving at fleet scale, and what the fleet costs in money, energy, carbon, and trust.",
        meta="H100 fleet · Llama 70B serving replica",
        notation=[
            ("Reliability", f'MTTF of one part · MTBF of the system · {v("t")} duration · {v("T", "write")} checkpoint write time · {v("T", "ckpt")} checkpoint interval · {v("W")} fraction of time wasted · {v("k")} parts, each with probability {v("p")}'),
            ("Serving", f'{v("M", "KV")} KV-cache bytes · {v("S")} context tokens · {v("N", "L")} layers · {v("H", "KV")} KV heads · {v("d", "head")} head dim · {v("N", "TP")} tensor-parallel degree · {v("λ")} requests/s · {v("B")} batch served in {v("t", "batch")} · {v("ρ")} target utilization'),
            ("Facility and trust", f'{v("P", "IT")} IT power · PUE = {v("P", "facility")}/{v("P", "IT")} · {v("E")} energy · CI grid g CO₂/kWh · {v("ε")} privacy budget · {v("Δ")} sensitivity · {v("n")} records averaged · {A}= example input'),
        ],
        cols=[
            Col("4 Will it survive?", [
                Card("Fleet MTBF", [f'MTBF = 1 / Σ (1/MTTF<sub>part</sub>) = MTTF / {v("k")} if identical', f'Pr(a failure within {v("t")}) = 1 − e<sup>−{v("t")}/MTBF</sup>'],
                     chapter="Ch 7",
                     ex=f'{fmt(n10k)} H100s at {fmt(GPU_MTTF)} h: GPUs fail every {n(f"{mtbf_gpu:.1f} h")}; with PCIe switches and NICs, every {n(f"{mtbf_sys:.2f} h", True)} ({ffr.failures_per_day:.1f}/day). '
                        f'SDC at {SDC:g}/GPU-h: one per {fmt(sdc_h)} h. Ch 7 cold restart (detect, restart, reload, warm-up): {n(f"{rec_s:.1f} s")}.'),
                Card("Checkpoint size and write time", f'{v("T", "write")} = {fmt(BPP_CKPT)} bytes × params ÷ write bandwidth',
                     chapter="Ch 4, 7",
                     ex=f'GPT-3: {ck_bytes / 1e12:.2f} TB (no gradients); at the book\'s {CKPT_BW / 1e9:.0f} GB/s, {v("T", "write")} = {n(f"{t_write:.1f} s")}. Local NVMe on the reference '
                        f'{nodes1} nodes ({local_bw / 1e9:.0f} GB/s each): {gb(shard1)} per node in {n(f"{t_local1:.2f} s")}, drained later.'),
                Card("Young-Daly checkpoint interval", [f'{v("T", "ckpt")}* = √(2 · {v("T", "write")} · MTBF)', f'{v("W")} ≈ {frac(v("T", "write"), v("T", "ckpt"))} + {frac(v("T", "ckpt"), "2 · MTBF")} &nbsp; (saving + redone work)'],
                     chapter="Ch 7",
                     ex=f'GPT-3 on the {fmt(n10k)}-GPU fleet (MTBF {mtbf_sys:.2f} h, {v("T", "write")} = {t_write:.1f} s): {v("T", "ckpt")}* = {fmt(tau)} s ≈ {n(f"{tau / 60:.0f} min")}, {v("W")} = {n(fmt(waste * 100, 1) + "%")}. '
                        f'Recovery is budgeted separately, as in Ch 7: {rec_s:.1f} s per failure adds {rec_share * 100:.1f}%.'),
                Chart(yd.svg("Checkpoint waste versus interval"), f'Waste {v("W")} (solid) = saving + rework (dashed), equal at the optimum.'),
            ], "Ch 4, 7"),
            Col("5 Can it serve at scale?", [
                Card("KV cache decides how many requests fit", [f'{v("M", "KV")} per token = 2 · {v("N", "L")} · {v("H", "KV")} · {v("d", "head")} · bytes',
                                                                f'requests = {frac("HBM − weights − reserve", v("S") + " · " + v("M", "KV") + " per token")}'],
                     chapter="Ch 10",
                     ex=f'Llama 3 70B ({v("N", "L")} = {wl.layers}, {v("H", "KV")} = {wl.kv_heads}, {v("d", "head")} = {kvr.head_dim}, FP16): {n(fmt(_m(kvr.kv_per_token, "B") / 1024) + " KiB/token")}, '
                        f'{_m(kvr.kv_per_request, "GB"):.1f} GB at {v("S")} = {fmt(kv.context_tokens / 1024)}K; {kv.accelerators} × {fmt(HBM_GIB)} GiB = {fmt(_m(kvr.hbm_total, "GB"))} GB − {_m(kvr.weights, "GB"):.1f} − {A}{_m(kv.reserve, "GB"):.0f} '
                        f'→ {n(f"{kvr.max_requests} requests")}.'),
                Card("Decode floor: every token reads the weights", f'time per token ≥ {frac("weight bytes", v("N", "TP") + " · HBM bandwidth")}',
                     chapter="Ch 10",
                     ex=f'Llama 3 70B in FP16 ({gb(dec_w)}) at {HBM_BW / 1e12:.2f} TB/s: {n(ms(_m(dec1.memory_time, "s")))}/token on one H100; TP{dec_tp}: '
                        f'{n(ms(dec_tp_s, 2))}, ≤ {fmt(math.floor(1 / dec_tp_s))} tokens/s per sequence.'),
                Card("Little's law sizes the replica count", f'replicas = ⌈{v("λ")} · {v("t", "batch")} / ({v("ρ")} · {v("B")})⌉',
                     chapter="Ch 10",
                     ex=f'{A}{v("λ")} = {fmt(_m(ll.arrival_rate, "1/s"))}/s, {v("t", "batch")} = {_m(ll.batch_service_time, "ms"):.0f} ms: Little\'s law puts {v("λ")}{v("t", "batch")} = {fmt(ll.mean_executing)} in service. '
                        f'{A}{v("B")} = {ll.batch_size} per replica at {v("ρ")} = {ll.utilization_target}: {n(f"{ll.replicas_needed} replicas")}.'),
                Card("Fan-out multiplies rare events", f'Pr(none of {v("k")} parts misbehaves) = (1 − {v("p")})<sup>{v("k")}</sup>',
                     chapter="Ch 4, 10, 12",
                     ex=f'{A}{fan.n_components} leaves, each {fmt(fan.availability * 100)}% fast: {n(fmt(fan.all_up * 100, 1) + "%", True)} of requests see none slow. '
                        f'{A}{fmt(alarms.monitors)} monitors checked every {_m(alarms.check_interval, "min"):.0f} min at {v("p")} = {alarms.false_positive_rate:g}: {n(fmt(alarms.daily_false_alerts) + " false alerts/day", True)}.'),
                Card("Federated learning: send updates, not data", 'update bytes = params × bits ÷ 8',
                     chapter="Ch 11",
                     ex=f'{A}{fmt(fed.items_per_week)} photos × {_m(fed.item_size, "kB"):.0f} KB = {_m(fed.raw_volume, "MB"):.0f} MB/week; a {fed.update_bits_per_param}-bit {fed.update_params / 1e6:.0f}M-param update is '
                        f'{_m(fed.update_size, "MB"):.1f} MB, {n(f"{fed.reduction:.0f}× less")}. Past {fed.break_even_rounds:.0f} rounds/week, send photos.'),
            ], "Ch 4, 10, 11"),
            Col("6 What does the fleet cost?", [
                Card("Power and PUE", f'{v("P", "facility")} = PUE · {v("P", "IT")}, &nbsp; overhead = (PUE − 1) · {v("P", "IT")}',
                     chapter="Ch 2, 15",
                     ex=f'IT per H100: {fmt(TDP)} W × {fp.it_ratio:.3f} = {it_gpu_kw * 1e3:.0f} W; a {rack_gpus}-GPU rack: {n(f"{rack_kw:.1f} kW", True)} (air limit {fmt(AIR_KW)} kW). '
                        f'{fmt(fp.devices)} H100s: {_m(fpr.it_power, "MW"):.1f} MW IT, {n(f"{_m(fpr.facility_power, "MW"):.1f} MW")} at PUE {fp.pue:.2f}. '
                        f'PUE {pr.pue_before} → {pr.pue_after} on {A}{_m(pr.it_power, "MW"):.0f} MW: −{fmt(_m(prr.saved_energy, "MWh"))} MWh/yr.'),
                Card("Energy and carbon", f'{v("E")} = PUE · {v("P", "IT")} · {v("t")}, &nbsp; CO₂ = {v("E")} × CI',
                     chapter="Ch 15",
                     ex=f'GPT-3\'s {fmt(G3_MWH)} MWh at the US average {fmt(CI["US_Avg"])} g/kWh: {n(fmt(g3_t) + " t CO₂")}. A {A}{fmt(_m(gpl.energy, "MWh"))} MWh run emits '
                        f'{fmt(_m(gplr.emissions["Quebec"], "t"))} t in Quebec and {n(fmt(_m(gplr.emissions["Poland"], "t")) + " t", True)} in Poland: siting outweighs tuning.'),
                Card("Own or rent?", f'break-even utilization = {frac("owned cost over the life", "GPU-hours × rental rate")}',
                     chapter="Ch 12",
                     ex=f'{fmt(own.gpus)} H100s for {own.years} years (nodes, network, facility, staff, power) against the book\'s ${_m(own.rent_rate, "USD/h"):.0f}/GPU-hour cloud price: '
                        f'own ${_m(own.owned_total, "USD") / 1e6:.0f}M, rent ${_m(own.rented_total, "USD") / 1e6:.0f}M at {A}{fmt(own.utilization * 100)}% use. Owning wins above {n(fmt(own.breakeven_utilization * 100) + "%")}.'),
                Card("Serving cost overtakes training", '$/year = 365 × users × requests/day × $/request',
                     chapter="Ch 10",
                     ex=f'{A}{fmt(sv.daily_users / 1e6)}M users × {sv.requests_per_user_per_day}/day × ${sv.cost_per_request:g} = {n("$" + f"{sv.annual_serving_cost / 1e6:.2f}M/year")}, '
                        f'{sv.serving_to_training:.1f}× a ${sv.training_cost / 1e6:.0f}M training run, overtaken in {sv.crossover_weeks:.1f} weeks.'),
                Card("Differential privacy costs accuracy", f'noise scale = {v("Δ")} / {v("ε")}; for a mean of {v("n")}: {v("Δ")} / ({v("n")}{v("ε")})',
                     chapter="Ch 13",
                     ex=f'{A}Salaries from $0 to ${_m(dp.value_range, "USD") / 1e3:.0f}K at {v("ε")} = {dp.epsilon:g}: the mean of {fmt(dp.group_size)} is off by about '
                        f'{n("$" + fmt(_m(dp.noise_scale, "USD") / dp.group_size))}; of {dp.small_group_size}, by {n("$" + fmt(_m(dp.small_group_error, "USD")), True)}.'),
                Note("Before you trust the number:", 'GB or Gb (÷ 8)? Per direction or both? MTBF of the whole system? KV cache and activations counted? Modeled or measured?'),
            ], "Ch 2, 10, 12–15"),
        ],
        bottom=Table(["Cluster tier (DGX H100, IB NDR)", "GPUs", "nodes", "MTBF, GPUs only", "GPU failures/day", "ring latency 2(N−1)α", "bisection at 1:1", "facility power"], trows),
    )
    return Sheet(volume=2, book="Scaling Machine Learning Systems", scope="The Fleet", pages=[p1, p2],
                 sources=f'GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B · 1 B = 8 b, so 400 Gb/s = 50 GB/s · {TAG}= input chosen for the example · values from the mlsysim registry and Volume II scenarios; step times are modeled.')

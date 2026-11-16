"""Napkin Math, Volume II: the fleet (hundreds to thousands of accelerators on a network).

Every number below is computed here from the mlsysim registry or Volume II's
scenarios (the 175B reference job on 1,024 H100s, the 10,000-GPU reliability
cases, the Llama 70B serving replica), so the sheet matches the book. Inputs
chosen only for an example are tagged with assumed() on the page; the ones
this module adds itself are listed in SCENARIO. Constants from outside the
registry are listed in EXTERNAL with their source. Symbols follow the series
table in README.md and the book's notation registry.
"""
from __future__ import annotations

import math
import warnings

from mlsysim import (BYTES_ADAM_STATE, BYTES_FP16, BYTES_FP32, HOURS_PER_DAY, SEC_PER_DAY, SEC_PER_HOUR,
                     Hardware as HW, Infrastructure as INF, Models as M,
                     ReferenceStats as RS, Systems as SYS,
                     calc_checkpoint_waste, calc_mtbf_cluster, calc_pipeline_bubble, calc_ring_allreduce_time,
                     calc_tree_allreduce_time, calc_young_daly_interval)
from mlsysim.engine.scenarios import Scenarios as SC

from .core import (Card, Chart, Col, Know, Note, Page, Plot, Sheet, Table,
                   assumed, fmt, frac, n, v)

warnings.filterwarnings("ignore")  # registry deprecation chatter only

# Megatron-style tensor parallelism: 2 AllReduces per transformer layer forward, 2 backward
# (Shoeybi et al. 2019, Megatron-LM, Sec. 3). A structural count, not a measured value.
EXTERNAL = {"tp_allreduces_per_layer": 4}
# tp_seqs: sequences per tensor-parallel microbatch (matches the reference step's 1). tp_ring: one DGX node.
# serve_*: the replica-sizing example (Llama 3 70B FP16 on one TP8 node). tiers: rows of the cluster table.
SCENARIO = dict(tp_seqs=1, tp_ring=8, serve_ctx=4096, serve_out=300, serve_slots=64, serve_lam=50, serve_rho=0.8,
                tiers=("Research_256", "Training_1K_NDR", "Frontier_8K", "Mega_100K"))


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
    HBM_BW = _m(H.memory.bandwidth, "B/s")
    TDP = _m(H.tdp, "W")
    NVL = _m(H.nvlink.bandwidth_per_direction, "B/s")
    PRICE = _m(INF.Pricing.Cloud.GpuTrainingPerHour.rate, "USD/h")
    NODE = SYS.Nodes.DGX_H100
    G_NODE = NODE.accelerators_per_node
    AIR_KW = INF.AIR_COOLING_LIMIT_KW
    NDR = SYS.Fabrics.InfiniBand_NDR
    BETA = _m(NDR.bandwidth, "B/s")
    ALPHA_Q = SYS.SwitchFabric.AlphaNdr
    ALPHA = _m(ALPHA_Q, "s")
    CS = SC.CriticalSizeByInterconnect
    ALPHA_NVL = _m(CS.alphas["NVLink"], "s")      # the book's NVLink alpha
    ROCE_BW, ROCE_A = _m(SYS.Fabrics.RoCE_100G.bandwidth, "B/s"), _m(SYS.SwitchFabric.AlphaRoce, "s")
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

    C3 = 6 * P3 * D3
    t_days = C3 / (N1 * PEAK * ETA) / S_DAY
    t_days_fleet = t_days / s1["eff"]

    z = SC.Llama2_7B_ZeroStages64
    zr = z.result
    zN = z.n_devices
    zstage = {k: _m(q, "GB") for k, q in zr.stage_state.items()}
    state_gpu = _m(ref.state_per_gpu, "B")

    img, txt = SC.ImageInputDemand2K, SC.TextInputDemand2K   # per-GPU demand, applied to the reference fleet
    img_kb = _m(img.bytes_per_sample, "kB")
    img_bw = N1 * img.samples_per_step * _m(img.bytes_per_sample, "B") / _m(img.step_time, "s") / 1e9

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
    ov = SC.RingAllReduceAcross4to1Spine
    ovr = ov.result
    OVR, OVF = ov.oversubscription_ratio, ov.communication_fraction
    bis_full, bis_over = _m(ovr.bisection_full, "TB/s"), _m(ovr.bisection_oversubscribed, "TB/s")
    thr = 1 / (1 - OVF + OVF * OVR)

    pb = SC.Pipeline8Stage32Micro
    bub_32 = calc_pipeline_bubble(pb.stages, pb.microbatches)
    bub_v = calc_pipeline_bubble(pb.stages, pb.microbatches, v_stages=2)
    bub_ref = calc_pipeline_bubble(PP, MB)
    bub_8k = calc_pipeline_bubble(PP, MB8)

    m_act = S["tp_seqs"] * G3.context_length * W3 * B16
    tp_nvl = _m(calc_ring_allreduce_time(m_act, S["tp_ring"], NVL, ALPHA_NVL), "s")
    tp_ndr = _m(calc_ring_allreduce_time(m_act, S["tp_ring"], BETA, ALPHA_Q), "s")
    tp_layers_stage = L3 // PP

    gp = SC.Llama3_405BGoodput16K
    gp_gpus = SYS.Clusters.Meta_Llama3_16K_H100.total_accelerators

    # ---------------------------------------------------------------- page 2
    gf, ff = SC.GpuFailuresTraining10K, SC.FullSystemFailures10K
    gfr, ffr = gf.result, ff.result
    mtbf_gpu, mtbf_sys = _m(gfr.interval, "h"), _m(ffr.mtbf, "h")
    n10k = ff.n_devices
    nodes10k = n10k // ff.devices_per_node
    parts = [(lab, n10k // ff.devices_per_node * ff.per_node[k], _m(ff.component_mttf[k], "h"))
             for k, lab in (("gpu", "GPUs"), ("pcie", "PCIe switches"), ("network", "NICs")) if ff.per_node[k]]
    rate_sum = sum(cnt / mt for _, cnt, mt in parts)
    assert abs(1 / rate_sum - mtbf_sys) < 1e-9, "printed parts reproduce the scenario MTBF"
    sdc_h = 1 / (SDC * n10k)
    run_h = _m(gf.run_duration, "h")
    run_fail = run_h / mtbf_sys

    ck_bytes = P3 * BPP_CKPT
    t_write = ck_bytes / CKPT_BW
    local_bw = SYS.Storage.LocalNvmeGen4x4.devices_per_node * _m(SYS.Storage.LocalNvmeGen4.bandwidth, "B/s")
    nodes1 = ref.fleet.count
    shard1 = ck_bytes / nodes1
    t_local1 = shard1 / local_bw
    tau = _m(calc_young_daly_interval(t_write, mtbf_sys * S_HR), "s")
    waste = calc_checkpoint_waste(t_write, tau, mtbf_sys * S_HR)
    rbr = SC.Gpt3RecoveryBudget10K.result
    rec_s = _m(rbr.total, "s")
    rec_share = rec_s / (mtbf_sys * S_HR)

    fan, alarms = SC.ServingFanout100Leaves, SC.ThreeSigmaModelFleetAlerts

    kv = SC.Llama3_70BLongContextAdmissionDgxH100
    kvr, wl = kv.result, kv.workload
    m_tok = _m(kvr.kv_per_token, "B")
    w_bytes = _m(kvr.weights, "B")
    n_tp = kv.accelerators
    # replica sizing: slot-hold time from the decode step, then Little's law
    mkv_req = m_tok * S["serve_ctx"]
    t_step1 = w_bytes / (n_tp * HBM_BW)
    t_stepB = (w_bytes + S["serve_slots"] * mkv_req) / (n_tp * HBM_BW)
    assert w_bytes + S["serve_slots"] * mkv_req <= _m(kvr.hbm_total, "B"), "slots fit in the replica's HBM"
    t_hold = S["serve_out"] * t_stepB
    q_req = S["serve_lam"] * t_hold
    replicas = math.ceil(q_req / (S["serve_rho"] * S["serve_slots"]))

    gpl = SC.QuebecVersusPolandPlacement
    gplr = gpl.result
    g3_t = G3_MWH * CI["US_Avg"] / 1000  # MWh x g/kWh = kg; / 1000 = t
    rack = SYS.Racks.DGX_H100_4Node
    fp = SC.Reference25KFacilityPower
    fpr = fp.result
    rack_gpus = rack.nodes_per_rack * G_NODE
    rack_gpu_kw = rack_gpus * TDP / 1e3
    rack_support_kw = _m(rack.non_accelerator_power - rack.cooling_allocation_power, "kW")  # host, NVSwitch, network, conversion
    it_ratio = (rack_gpu_kw + rack_support_kw) / rack_gpu_kw
    assert abs(it_ratio - fp.it_ratio) < 1e-9, "IT ratio is the book's rack-profile factor"
    rack_kw = rack_gpu_kw + rack_support_kw
    it_gpu_kw = TDP * it_ratio / 1e3
    pr = SC.PueRetrofit2MW
    prr = pr.result
    own = SC.OwnVsRentH100Cluster10K
    assert _m(own.rent_rate, "USD/h") == PRICE, "rent rate is the registry cloud price"
    sv = SC.LlmServingFirstYear
    dp = SC.SalaryMeanLaplaceRelease

    def gb(x, d=1):
        return f"{x / 1e9:,.{d}f} GB"

    def ms(x, d=1):
        return f"{x * 1e3:,.{d}f} ms"

    def us(x, d=1):
        return f"{x * 1e6:,.{d}f} µs"

    NG = v("N", "GPU")
    nm = lambda mdl: mdl.name.replace("-", " ")

    # ---------------------------------------------------------------- chart: alpha-beta
    links = [("NVLink", NVL, ALPHA_NVL, "line"), ("IB NDR", BETA, ALPHA, "line warn"), ("RoCE 100G", ROCE_BW, ROCE_A, "line muted")]
    ap = Plot((1e2, 1e9), (1, 1e5), xlog=True, ylog=True, W=316, H=98, right=12, bottom=34)
    xl = {1e2: "100 B", 1e3: "1 KB", 1e4: "", 1e5: "100 KB", 1e6: "", 1e7: "10 MB", 1e8: "", 1e9: "1 GB"}
    yl = {1: "1 µs", 10: "", 100: "", 1e3: "1 ms", 1e4: "", 1e5: "100 ms"}
    ap.grid(list(xl), list(yl), xfmt=lambda x: xl[x], yfmt=lambda y: yl[y])
    for name, bw, al, cls in links:
        ap.line([(10 ** (2 + k * 0.1), (al + 10 ** (2 + k * 0.1) / bw) * 1e6) for k in range(71)], cls)
        ap.point(al * bw, 2 * al * 1e6)
    ap.text(ap.X(8e8), ap.Y(2.2), "β-bound", "end", cls="lab ital")
    ap.text((ap.l + ap.W - ap.r) / 2, ap.H - 13, "message size n · dot at n* = α·β · α = " + ", ".join(f"{al * 1e6:g}" for _, _, al, _ in links) + " µs", "middle", "ax")
    for k, (name, bw, al, cls) in enumerate(links):   # legend row below the axis, clear of the curves
        x0 = ap.l + k * 90
        ap.parts.append(f'<line x1="{x0}" x2="{x0 + 12}" y1="{ap.H - 5}" y2="{ap.H - 5}" class="{cls}"/>')
        ap.text(x0 + 15, ap.H - 2, f"{name.replace('IB ', '').replace(' 100G', '')} {bw / 1e9:g} GB/s", cls="tick")
    cy = (ap.t + ap.H - ap.b) / 2
    ap.parts.append(f'<text x="9" y="{cy:.1f}" class="ax" text-anchor="middle" transform="rotate(-90 9 {cy:.1f})">time</text>')

    # ---------------------------------------------------------------- chart: Fleet Law terms at two scales
    def bars():
        W, x0, x1, bh = 340, 54, 334, 14
        out = [f'<svg viewBox="0 0 {W} 60" class="chart" role="img" aria-label="Reference step decomposition at two scales">']
        for k, (lab, t) in enumerate(((f"{fmt(N1)} GPUs", s1), (f"{fmt(N8)} GPUs", s8))):
            y = 3 + k * 31
            out.append(f'<text x="{x0 - 4}" y="{y + 10}" class="lab" text-anchor="end">{lab}</text>')
            xs = x0
            for key, cls in (("comp", "fill-accent"), ("exp", "fill-warn"), ("bub", "fill-neutral")):
                w = (x1 - x0) * t[key] / t["tot"]
                out.append(f'<rect x="{xs:.1f}" y="{y}" width="{w:.1f}" height="{bh}" class="{cls}"/>')
                xs += w
            out.append(f'<text x="{x0 + 4}" y="{y + 10}" class="mono">compute {fmt(t["comp"] / t["tot"] * 100)}%</text>')
            out.append(f'<text x="{x1}" y="{y + bh + 9}" class="tick" text-anchor="end">step {t["tot"]:.2f} s (modeled) · exposed communication {fmt(t["exp"] / t["tot"] * 100)}% · bubble {fmt(t["bub"] / t["tot"] * 100)}% of step</text>')
        out.append("</svg>")
        return "".join(out)

    # ---------------------------------------------------------------- chart: Young-Daly
    yd = Plot((60, 3600 * 3), (0, 0.25), xlog=True, W=340, H=104, right=14, bottom=25)
    yd.grid([60, 300, 900, 3600, 10800], [0, 0.1, 0.2],
            xfmt=lambda x: {60: "1 min", 300: "5 min", 900: "15 min", 3600: "1 h", 10800: "3 h"}[x],
            yfmt=lambda y: f"{y * 100:.0f}%")
    xs = [60 * 10 ** (k * math.log10(180) / 80) for k in range(81)]
    mtbf_s = mtbf_sys * S_HR
    for f_, cls in ((lambda x: t_write / x, "line alt"), (lambda x: x / (2 * mtbf_s), "line alt"),
                    (lambda x: calc_checkpoint_waste(t_write, x, mtbf_s), "line")):
        yd.line([(x, f_(x)) for x in xs if f_(x) <= yd.y1], cls)  # draw only inside the plot
    yd.point(tau, waste, f"optimum {tau / 60:.0f} min, W = {waste * 100:.1f}%", "middle", 0, -12)
    yd.text(yd.X(140), yd.Y(0.12), "saving", "end", cls="lab ital")
    yd.text(yd.X(4800), yd.Y(0.12), "rework", cls="lab ital")
    yd.axes(f"checkpoint interval (GPT-3, {fmt(n10k)} GPUs, MTBF {mtbf_sys:.2f} h)", "waste W")

    # ---------------------------------------------------------------- page 1
    band = (f'<div class="eqbig">{v("T", "step")} = {frac(v("T", "compute"), NG)} + {v("T", "comm")} + {v("T", "sync")} − {v("T", "overlap")}'
            f'<span class="lbl">Compute · Communication · Coordination − Overlap</span></div>'
            f'<div class="txt"><div><b>The Fleet Law (Ch 1).</b> The 175B reference job on {NG} = {fmt(N1)} H100s ({DP} replicas of TP{TP} × PP{PP}), modeled: '
            f'{s1["comp"]:.2f} s compute + {s1["comm"]:.2f} s {v("T", "comm")} + {s1["bub"]:.2f} s {v("T", "sync")} (pipeline bubble) − {s1["ovl"]:.2f} s {v("T", "overlap")} ≈ {n(f"{s1["tot"]:.2f} s")}; '
            f'{v("η", "scaling")} = ({v("T", "compute")}/{NG}) / {v("T", "step")} = {n(fmt(s1["eff"] * 100) + "%")}.</div>'
            f'<div><b>{N8 // N1}× the GPUs, same batch:</b> {s8["tot"]:.2f} s, only {speedup_8x:.1f}× faster; {v("η", "scaling")} = {fmt(s8["eff"] * 100)}%.</div></div>'
            f'<div class="side"><div><b>Pull the largest term\'s lever</b></div><div>{v("T", "comm")}: faster fabric, overlap, compression</div>'
            f'<div>{v("T", "sync")}: fewer barriers; one GPU at half speed halves the fleet</div><div>{v("T", "compute")}/{NG}: faster or more accelerators</div></div>')

    def mrow(mdl):
        kvh = mdl.kv_heads or mdl.heads   # no grouped-query attention: every head keeps K and V
        return [nm(mdl), f"{_m(mdl.parameters, 'param') / 1e9:,.1f}B".replace(".0B", "B"), str(mdl.layers), fmt(mdl.hidden_dim),
                str(kvh), str(mdl.hidden_dim // mdl.heads), f"{_m(mdl.training_tokens, 'count') / 1e12:g}T"]
    ref_models = Table(["reference model", v("P"), v("N", "L"), v("d"), v("H", "KV"), v("d", "head"), v("D")],
                       [mrow(G3), mrow(M.Language.Llama2_7B), mrow(M.Language.Llama2_70B), mrow(wl)],
                       caption=f'H100: {fmt(PEAK / 1e12)} TFLOP/s FP16, {HBM_BW / 1e12:.2f} TB/s, {fmt(HBM_GIB)} GiB = {HBM / 1e9:.1f} GB, {fmt(TDP)} W, ${PRICE:.0f}/GPU-h (cloud). DGX node: {G_NODE} GPUs.')

    p1 = Page(
        question="Will it scale?",
        sub="Read left to right: what one training job needs, how fast bytes move between accelerators, and where each step's time goes.",
        meta=f"reference: {fmt(N1)} × H100, DGX H100 nodes, IB NDR",
        notation=[
            ("Job", f'{v("P")} parameters · {v("D")} training tokens · {v("S")} sequence length · {v("d")} width · {v("N", "L")} layers · {v("m")} microbatches per step · {v("T")} time'),
            ("Fleet", f'{NG} accelerators · {v("N", "DP")}, {v("N", "TP")}, {v("N", "PP")} data, tensor, pipeline degrees · {v("R", "peak")} peak FLOP/s · {v("η", "hw")} fraction of peak · {v("η", "scaling")} scaling efficiency'),
            ("Network", f'{v("α")} per-message latency · {v("β")} link bandwidth per direction · {v("M")} collective payload per rank · {A}= example input'),
        ],
        know=[
            Know(f"{fmt(PEAK / 1e12)} · {HBM_BW / 1e12:.2f}", "H100: FP16 TFLOP/s · HBM TB/s"),
            Know(f"{fmt(NVL / 1e9)} · {fmt(BETA / 1e9)} GB/s", "NVLink · NDR port, per direction"),
            Know(f"{ALPHA * 1e6:.1f} µs", "NDR per-message latency α"),
            Know(f"{fmt(BPP_STATE)} · {fmt(BPP_CKPT)} bytes", "per parameter: training state · checkpoint"),
            Know(f"{fmt(GPU_MTTF)} h", "MTTF of one GPU"),
            Know(f"{PUE['LiquidCooled']:.2f}–{PUE['Legacy']:.2f}", f"PUE: liquid-cooled to {INF.FacilityCooling.Legacy.name.lower()}"),
            Know(f"{fmt(CI['Quebec'])}–{fmt(CI['Poland'])} g/kWh", "grid CO₂: hydro to coal"),
        ],
        band=band,
        cols=[
            Col("1 What does the job need?", [
                ref_models,
                Card("Training time", f'{v("T")} ≈ 6{v("P")}{v("D")} / ({NG} · {v("R", "peak")} · {v("η", "hw")} · {v("η", "scaling")})', chapter="Ch 2",
                     ex=f'GPT-3, {NG} = {fmt(N1)}, {A}{v("η", "hw")} = {ETA:.2f}, {v("η", "scaling")} = {s1["eff"]:.2f} → {n(f"{t_days_fleet:.1f} days")}.'),
                Card("Training state per device", None, chapter="Ch 6",
                     holds=f'Bytes per parameter: {fmt(B16)} weights + {fmt(B16)} gradients + {fmt(B32)} master + {fmt(BADAM)} Adam moments. ZeRO-3 moves {zr.traffic_multiple["ZeRO-3"]:.1f}× DDP\'s bytes.'),
                Table(["bytes per GPU", "DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"], [
                    ["formula", f"{fmt(BPP_STATE)}<i>P</i>", f"{fmt(2 * B16)}<i>P</i>+{fmt(B32 + BADAM)}<i>P</i>/<i>N</i><sub>DP</sub>",
                     f"{fmt(B16)}<i>P</i>+{fmt(B16 + B32 + BADAM)}<i>P</i>/<i>N</i><sub>DP</sub>", f"{fmt(BPP_STATE)}<i>P</i>/<i>N</i><sub>DP</sub>"],
                    [f"7B, {v('N', 'DP')} = {zN} (GB)"] + [fmt(zstage["DDP"])] + [f"{zstage[k]:.1f}" for k in ("ZeRO-1", "ZeRO-2", "ZeRO-3")]]),
                Card("Smallest replica that fits", f'{frac(fmt(BPP_STATE) + v("P"), v("N", "TP") + " · " + v("N", "PP"))} + activations ≤ HBM, &nbsp; {v("N", "DP")} = {frac(NG, v("N", "TP") + " · " + v("N", "PP"))}',
                     chapter="Ch 6",
                     ex=f'GPT-3, TP{TP} × PP{PP}: {gb(state_gpu)} + activations = {n(gb(s1["mem"]))} ≤ {HBM / 1e9:.1f} GB; {fmt(N1)} GPUs → {v("N", "DP")} = {n(str(DP))}.'),
            ], "Ch 2, 6"),
            Col("2 How fast do bytes move?", [
                Chart(ap.svg("Transfer time versus message size")),
                Card("Point to point: the α-β model", f'{v("T")}({v("n")}) = {v("α")} + {frac(v("n"), v("β"))}, &nbsp; crossover {v("n")}* = {v("α")} · {v("β")}', chapter="Ch 3",
                     ex=f'NDR, {v("n")} = {fmt(msg_small / 1e3)} KB → {us(t_small, 2)} ({n(fmt(ALPHA / t_small * 100) + "% α")}); {v("n")} = {fmt(msg_big / 1e6)} MB → {n(ms(t_big, 2))}.'),
                Table(["collective (Ch 5)", "time, payload <i>M</i> per rank"], [
                    ["Ring AllReduce", f"2({NG}−1)(<i>α</i>&nbsp;+&nbsp;<i>M</i>/{NG}<i>β</i>)"],
                    ["Tree, unsegmented", f"2&nbsp;log₂{NG}&nbsp;(<i>α</i>&nbsp;+&nbsp;<i>M</i>/<i>β</i>)"],
                    ["ReduceScatter, AllGather", f"({NG}−1)(<i>α</i>&nbsp;+&nbsp;<i>M</i>/{NG}<i>β</i>)"],
                    ["AllToAll (MoE dispatch)", f"({NG}−1)(<i>α</i>&nbsp;+&nbsp;<i>M</i>/{NG}<i>β</i>)"],
                ]),
                f'<p class="ex">{nm(M.Language.Llama2_70B)}, {v("M")} = {gb(grM, 0)} (BF16), {NG} = {grN}, NDR → ring {n(f"{gr_ring:.2f} s")}, unsegmented tree {gr_tree:.1f} s (pipelined trees near the ring). '
                f'Trees win below App. D\'s exact crossover: {fmt(xover_64 / 1e3)} KB at {grN}, {xover_1k / 1e6:.2f} MB at {fmt(N1)}.</p>',
                Card("Can storage feed the fleet?", f'BW<sub>storage</sub> ≥ {NG} × bytes per GPU per step / {v("T", "step")}', chapter="Ch 4",
                     ex=f'{fmt(N1)} GPUs, {A}{img.samples_per_step} images × {img_kb:.2f} KB per {_m(img.step_time, "ms"):.0f} ms step → {n(f"{img_bw:.1f} GB/s")}.'),
            ], "Ch 3, 4, 5"),
            Col("3 Where does the step go?", [
                Chart(bars(), 'Compute (blue), exposed communication (amber), pipeline bubble (gray).'),
                Card("Pipeline bubble", f'{v("b", "pipe")} = {frac(v("N", "PP") + " − 1", v("N", "PP") + " − 1 + " + v("V") + v("m"))}', chapter="Ch 6",
                     holds=f'Share of pipeline time (the bars use the whole step); 1F1B; {v("V")} interleaved stages per device.',
                     ex=f'{v("N", "PP")} = {pb.stages}, {v("m")} = {pb.microbatches} → {n(fmt(bub_32 * 100, 1) + "%")} ({fmt(bub_v * 100, 1)}% at {v("V")} = 2). Reference: {v("m")} = {MB} → {fmt(bub_ref * 100, 1)}%; '
                        f'{v("m")} = {MB8} at {fmt(N8)} GPUs → {n(fmt(bub_8k * 100, 1) + "%", True)}.'),
                Card("Tensor parallelism stays inside the node", f'{TP_AR} per layer, {v("M")} = {v("B", "micro")} · {v("S")} · {v("d")} · {fmt(B16)} bytes', chapter="Ch 6, 10",
                     holds=f'AllReduces over a ring of {v("N", "TP")}; {v("B", "micro")}: sequences per microbatch.',
                     ex=f'GPT-3, {A}{v("B", "micro")} = {S["tp_seqs"]} → {v("M")} = {m_act / 1e6:.0f} MB; ring of {S["tp_ring"]}: NVLink {n(us(tp_nvl, 0))}, NDR {n(ms(tp_ndr, 2), True)}. '
                        f'{TP_AR * tp_layers_stage} per stage per microbatch.'),
                Card("Oversubscription R:1 at the spine", f'throughput = 1 / (1 − {v("f")} + {v("f")} · R)', chapter="Ch 3",
                     holds=f'{v("f")}: step share crossing the spine; bisection = {NG}{v("β")}/2R.',
                     ex=f'{NG} = {fmt(ov.fleet.total_accelerators)}, NDR, R = {OVR:.0f}, {A}{v("f")} = {OVF:.1f} → bisection {fmt(bis_full, 1)} → {fmt(bis_over, 1)} TB/s, throughput {n(fmt(thr * 100) + "%", True)}.'),
            ], "Ch 3, 6"),
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

    T_txt = v("T")
    part_txt = ", ".join(f"{fmt(cnt)} {lab} at {fmt(mt)} h" for lab, cnt, mt in parts)

    p2 = Page(
        question="Will it survive and serve?",
        sub="Failures as a steady state, serving at fleet scale, and what the fleet costs in money, energy, carbon, and trust.",
        meta=f"H100 fleet · {nm(wl)} serving replica",
        notation=[
            ("Reliability", f'MTTF of one part · MTBF of the whole system · {v("T", "write")} checkpoint write time · {v("T", "ckpt")} checkpoint interval · {v("W")} fraction of time wasted · {v("T")} time'),
            ("Serving", f'{v("M", "KV")} KV bytes per request · {v("S")} context tokens · {v("B")} slots per replica · {v("N", "TP")} tensor-parallel GPUs · BW HBM bandwidth · {v("λ")} requests/s · {v("t", "s")} service time · {v("ρ")} utilization'),
            ("Facility", f'{v("P", "IT")} IT power · PUE = {v("P", "facility")}/{v("P", "IT")} · {v("E")} energy · CI grid g CO₂/kWh · {A}= example input'),
        ],
        cols=[
            Col("4 Will it survive?", [
                Card("Fleet MTBF", [f'MTBF = 1 / Σ<sub>parts</sub> (count / MTTF)', f'Pr(failure within {v("T")}) = 1 − e<sup>−{v("T")}/MTBF</sup>'], chapter="Ch 7",
                     ex=f'{fmt(n10k)} H100s ({fmt(nodes10k)} nodes): {part_txt} → {n(f"{mtbf_sys:.2f} h", True)} ({mtbf_gpu:.1f} h from GPUs alone). '
                        f'Over {A}{T_txt} = {run_h / H_DAY:.0f} days: {fmt(run_fail)} expected failures. SDC at {SDC:g}/GPU-h → one per {fmt(sdc_h)} h.'),
                Card("Checkpoint size and write time", f'{v("T", "write")} = {fmt(BPP_CKPT)} bytes × {v("P")} ÷ write bandwidth', chapter="Ch 4, 7",
                     ex=f'GPT-3 ({ck_bytes / 1e12:.2f} TB) at {CKPT_BW / 1e9:.0f} GB/s → {n(f"{t_write:.1f} s")}; local NVMe, {nodes1} nodes at {local_bw / 1e9:.0f} GB/s → {n(f"{t_local1:.2f} s")}.'),
                Card("Young-Daly checkpoint interval", [f'{v("T", "ckpt")}* = √(2 · {v("T", "write")} · MTBF)', f'{v("W")} ≈ {frac(v("T", "write"), v("T", "ckpt"))} + {frac(v("T", "ckpt"), "2 · MTBF")} &nbsp; (saving + rework)'], chapter="Ch 7",
                     ex=f'{v("T", "write")} = {t_write:.1f} s, MTBF {mtbf_sys:.2f} h → {fmt(tau)} s ≈ {n(f"{tau / 60:.0f} min")}, {v("W")} = {n(fmt(waste * 100, 1) + "%")}. '
                        f'Restart budgeted apart (Ch 7): detect + restart + reload + warm-up = {rec_s:.1f} s per failure, +{rec_share * 100:.1f}%.'),
                Chart(yd.svg("Checkpoint waste versus interval"), f'Waste {v("W")} (solid) = saving + rework (dashed), equal at the optimum.'),
            ], "Ch 4, 7"),
            Col("5 Can it serve at scale?", [
                Card("KV cache per request", f'{v("M", "KV")} = 2 · {v("N", "L")} · {v("H", "KV")} · {v("d", "head")} · {v("s", "elem")} · {v("S")}', chapter="Ch 10",
                     holds=f'{v("s", "elem")}: bytes per element.',
                     ex=f'{nm(wl)} FP16, {v("S")} = {fmt(kv.context_tokens / 1024)}K → {n(f"{_m(kvr.kv_per_request, "GB"):.1f} GB")}; '
                        f'{n_tp} × {fmt(HBM_GIB)} GiB − {gb(w_bytes)} weights − {A}{_m(kv.reserve, "GB"):.0f} GB → {n(f"{kvr.max_requests} requests")}.'),
                Card("Decode step and replica count", [f'{v("t", "step")} = (weights + {v("B")} · {v("M", "KV")}) / ({v("N", "TP")} · BW)',
                                                       f'{v("t", "s")} = output tokens × {v("t", "step")}, &nbsp; replicas = ⌈{frac(v("λ") + v("t", "s"), v("ρ") + v("B"))}⌉'],
                     chapter="Ch 10",
                     holds=f'{v("t", "s")}: how long a request holds a slot. Little\'s law: {v("λ")}{v("t", "s")} requests are in service.',
                     ex=f'{nm(wl)}, TP{n_tp}: {v("B")} = 1 → {ms(t_step1, 2)}. {A}{v("S")} = {fmt(S["serve_ctx"])}, {v("B")} = {S["serve_slots"]} → {v("M", "KV")} = {gb(mkv_req, 2)}, {n(ms(t_stepB, 2))}; '
                        f'{A}{S["serve_out"]} tokens → {v("t", "s")} = {t_hold:.2f} s; {A}{v("λ")} = {S["serve_lam"]}/s, {v("ρ")} = {S["serve_rho"]} → {fmt(q_req)} in service, {n(f"{replicas} replicas")}.'),
                Card("Fan-out multiplies rare events", f'Pr(none of {v("k")} parts misbehaves) = (1 − {v("p")})<sup>{v("k")}</sup>', chapter="Ch 4, 10, 12",
                     ex=f'{A}{v("k")} = {fan.n_components}, {v("p")} = {1 - fan.availability:.2f} → {n(fmt(fan.all_up * 100, 1) + "%", True)} of requests see no slow leaf. '
                        f'{A}{fmt(alarms.monitors)} monitors × {fmt(alarms.checks_per_day)} checks/day × {alarms.false_positive_rate:g} → {n(fmt(alarms.daily_false_alerts) + " false alerts/day", True)}.'),
                Card("Serving cost overtakes training", '$/year = 365 × users × requests/day × $/request', chapter="Ch 10",
                     ex=f'{A}{fmt(sv.daily_users / 1e6)}M × {sv.requests_per_user_per_day} × ${sv.cost_per_request:g} → {n("$" + f"{sv.annual_serving_cost / 1e6:.2f}M/year")}, '
                        f'{sv.serving_to_training:.1f}× a ${sv.training_cost / 1e6:.0f}M training run.'),
                Note("Before you trust the number:", 'GB or Gb (÷ 8)? Per direction or both? MTBF of the whole system? KV cache counted in the step? Modeled or measured?'),
            ], "Ch 4, 10, 12"),
            Col("6 What does the fleet cost?", [
                Card("Power and PUE", [f'{v("P", "IT")} = {NG} × GPU TDP × IT ratio', f'{v("P", "facility")} = PUE · {v("P", "IT")}'], chapter="Ch 2, 15",
                     holds=f'IT ratio: the book\'s DGX H100 rack profile, ({rack_gpus} × {fmt(TDP)} W GPUs + {rack_support_kw:.1f} kW host, NVSwitch, network, conversion) ÷ {rack_gpu_kw:.1f} kW = {it_ratio:.3f}. Cooling is in PUE.',
                     ex=f'Rack: {n(f"{rack_kw:.1f} kW", True)} vs {fmt(AIR_KW)} kW air limit. {fmt(fp.devices)} H100s → {_m(fpr.it_power, "MW"):.1f} MW IT, PUE {fp.pue:.2f} → {n(f"{_m(fpr.facility_power, "MW"):.2f} MW")}.'),
                Card("Goodput: what the held GPU-hours buy", "goodput = scheduling × runtime × MFU", chapter="Ch 9",
                     ex=f'{nm(M.Language.Llama3_405B)} on {fmt(gp_gpus)} H100s: {gp.scheduling:.1f} × {gp.runtime:.2f} × {gp.program:.2f} = {n(fmt(gp.result.mpg * 100, 1) + "%")} of held capacity.'),
                Card("Energy and carbon", f'{v("E")} = PUE · {v("P", "IT")} · {v("T")}, &nbsp; CO₂ = {v("E")} × CI', chapter="Ch 15",
                     ex=f'GPT-3, {fmt(G3_MWH)} MWh × {fmt(CI["US_Avg"])} g/kWh → {n(fmt(g3_t) + " t")}. {A}{fmt(_m(gpl.energy, "MWh"))} MWh: '
                        f'Quebec {fmt(_m(gplr.emissions["Quebec"], "t"))} t, Poland {n(fmt(_m(gplr.emissions["Poland"], "t")) + " t", True)}.'),
                Card("Own or rent?", f'break-even utilization = {frac("owned cost over the life", "GPU-hours × rental rate")}', chapter="Ch 12",
                     ex=f'{fmt(own.gpus)} H100s, {own.years} yr: ${_m(own.node_price, "USD") / 1e3:.0f}K/node, ${_m(own.network_capex + own.facility_capex, "USD") / 1e6:.0f}M network + facility, ${_m(own.staffing_per_year, "USD") / 1e6:.0f}M/yr staff, '
                        f'{_m(own.power, "MW"):.0f} MW × PUE {own.pue} at ${_m(own.electricity_price, "USD/kWh"):.2f}/kWh → ${_m(own.owned_total, "USD") / 1e6:.0f}M; vs ${PRICE:.0f}/GPU-h, own above {n(fmt(own.breakeven_utilization * 100) + "%")}.'),
                Card("Differential privacy costs accuracy", f'noise scale = Δ / {v("ε")}; error of a mean of {v("n")} ≈ Δ / ({v("n")}{v("ε")})', chapter="Ch 13",
                     holds=f'Δ: largest change one person can make (sensitivity); {v("ε")}: privacy budget.',
                     ex=f'{A}Δ = ${_m(dp.value_range, "USD") / 1e3:.0f}K, {v("ε")} = {dp.epsilon:g}: {v("n")} = {fmt(dp.group_size)} → {n("$" + fmt(_m(dp.noise_scale, "USD") / dp.group_size))}; '
                        f'{v("n")} = {dp.small_group_size} → {n("$" + fmt(_m(dp.small_group_error, "USD")), True)}.'),
            ], "Ch 2, 9, 12–15"),
        ],
        bottom=Table(["Cluster tier (DGX H100, IB NDR)", "GPUs", "nodes", "MTBF, GPUs only", "GPU failures/day", "ring latency 2(N<sub>GPU</sub>−1)α", "bisection at 1:1", "facility power"], trows),
    )
    return Sheet(volume=2, book="Scaling Machine Learning Systems", scope="The Fleet", pages=[p1, p2],
                 sources=f'GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B · 1 B = 8 b, so 400 Gb/s = 50 GB/s · {TAG}= input chosen for the example · values from the mlsysim registry and Volume II scenarios; step times are modeled.')

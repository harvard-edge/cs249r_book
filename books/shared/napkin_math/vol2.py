"""Napkin Math, Volume II: the fleet (hundreds to thousands of accelerators on a network).

Every number below is computed here from the mlsysim registry or Volume II's
scenarios (the 175B reference job on 1,024 H100s, the 10,000-GPU reliability
cases, the Llama 70B serving replica), so the sheet matches the book. Inputs
chosen only for an example are tagged with assumed() on the page; the ones
this module adds itself are listed in SCENARIO. Constants from outside the
registry are listed in EXTERNAL with their source. Symbols follow the series
table in README.md and the book's notation registry.

Pages 1-2 are the core handout (front and back). Pages 3-4 carry the cards
the core cannot hold: cheaper communication and coordination, then serving
responsibly.
"""
from __future__ import annotations

import math
import warnings

from mlsysim import (BYTES_ADAM_STATE, BYTES_FP16, BYTES_FP32, HOURS_PER_DAY, SEC_PER_DAY, SEC_PER_HOUR,
                     Hardware as HW, Infrastructure as INF, Models as M,
                     ReferenceStats as RS, Systems as SYS,
                     calc_checkpoint_waste, calc_mg1_wait_ratio, calc_mtbf_cluster, calc_pipeline_bubble,
                     calc_ring_allreduce_time, calc_tree_allreduce_time, calc_young_daly_interval)
from mlsysim.engine.scenarios import Scenarios as SC

from .core import (Card, Chart, Col, Know, Note, Page, Plot, Sheet, Table,
                   assumed, fmt, frac, n, v)

warnings.filterwarnings("ignore")  # registry deprecation chatter only

# Megatron-style tensor parallelism: 2 AllReduces per transformer layer forward, 2 backward
# (Shoeybi et al. 2019, Megatron-LM, Sec. 3). A structural count, not a measured value.
EXTERNAL = {"tp_allreduces_per_layer": 4}
# tp_seqs: sequences per tensor-parallel microbatch (matches the reference step's 1). tp_ring: one DGX node.
# serve_*: the replica-sizing example (Llama 3.1 70B FP16 on one TP8 node). tiers: rows of the cluster table.
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
    scale_x = N8 // N1
    speedup_8x = s1["tot"] / s8["tot"]
    ideal_share = speedup_8x / scale_x

    C3 = 6 * P3 * D3
    t_days = C3 / (N1 * PEAK * ETA) / S_DAY
    t_days_fleet = t_days / s1["eff"]

    z = SC.Llama2_7B_ZeroStages64
    zr = z.result
    zN = z.n_devices
    zstage = {k: _m(q, "GB") for k, q in zr.stage_state.items()}
    state_gpu = _m(ref.state_per_gpu, "B")
    act_gpu = s1["mem"] - state_gpu   # activations and buffers per GPU

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

    pb = SC.Pipeline8Stage32Micro
    bub_32 = calc_pipeline_bubble(pb.stages, pb.microbatches)
    bub_ref = calc_pipeline_bubble(PP, MB)
    bub_8k = calc_pipeline_bubble(PP, MB8)

    m_act = S["tp_seqs"] * G3.context_length * W3 * B16
    tp_nvl = _m(calc_ring_allreduce_time(m_act, S["tp_ring"], NVL, ALPHA_NVL), "s")
    tp_ndr = _m(calc_ring_allreduce_time(m_act, S["tp_ring"], BETA, ALPHA_Q), "s")
    tp_layers_stage = L3 // PP

    # ---------------------------------------------------------------- page 2
    gf, ff = SC.GpuFailuresTraining10K, SC.FullSystemFailures10K
    gfr, ffr = gf.result, ff.result
    mtbf_gpu, mtbf_sys = _m(gfr.interval, "h"), _m(ffr.mtbf, "h")
    n10k = ff.n_devices
    nodes10k = n10k // ff.devices_per_node
    parts = [(lab, nodes10k * ff.per_node[k], _m(ff.component_mttf[k], "h"))
             for k, lab in (("gpu", "GPUs"), ("pcie", "PCIe switches"), ("network", "NICs")) if ff.per_node[k]]
    rate_sum = sum(cnt / mt for _, cnt, mt in parts)
    assert abs(1 / rate_sum - mtbf_sys) < 1e-9, "printed parts reproduce the scenario MTBF"
    run_h = _m(gf.run_duration, "h")
    run_fail = run_h / mtbf_sys

    ck_bytes = P3 * BPP_CKPT
    t_write = ck_bytes / CKPT_BW
    tau = _m(calc_young_daly_interval(t_write, mtbf_sys * S_HR), "s")
    waste = calc_checkpoint_waste(t_write, tau, mtbf_sys * S_HR)
    rbr = SC.Gpt3RecoveryBudget10K.result
    rec_s = _m(rbr.total, "s")
    rec_share = rec_s / (mtbf_sys * S_HR)

    kv = SC.Llama3_70BLongContextAdmissionDgxH100
    kvr, wl = kv.result, kv.workload
    m_tok = _m(kvr.kv_per_token, "B")
    w_bytes = _m(kvr.weights, "B")
    n_tp = kv.accelerators
    mkv_req = m_tok * S["serve_ctx"]
    t_step1 = (w_bytes + mkv_req) / (n_tp * HBM_BW)
    t_stepB = (w_bytes + S["serve_slots"] * mkv_req) / (n_tp * HBM_BW)
    assert w_bytes + S["serve_slots"] * mkv_req <= _m(kvr.hbm_total, "B"), "slots fit in the replica's HBM"
    t_hold = S["serve_out"] * t_stepB
    q_req = S["serve_lam"] * t_hold
    replicas = math.ceil(q_req / (S["serve_rho"] * S["serve_slots"]))

    fan = SC.ServingFanout100Leaves

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
    gp = SC.Llama3_405BGoodput16K
    gp_gpus = SYS.Clusters.Meta_Llama3_16K_H100.total_accelerators
    gpl = SC.QuebecVersusPolandPlacement
    gplr = gpl.result
    g3_t = G3_MWH * CI["US_Avg"] / 1000  # MWh x g/kWh = kg; / 1000 = t
    own = SC.OwnVsRentH100Cluster10K
    assert _m(own.rent_rate, "USD/h") == PRICE, "rent rate is the registry cloud price"
    own_hw = _m(own.hardware_capex, "USD")
    own_nf = _m(own.network_capex + own.facility_capex, "USD")
    own_staff = _m(own.staffing_per_year, "USD") * own.years
    own_power = _m(own.electricity_per_year_full, "USD") * own.years
    own_total = own_hw + own_nf + own_staff + own_power
    assert abs(own_total - _m(own.owned_total, "USD")) < 1, "the printed parts sum to the scenario total"
    rent_full = own.gpus * _m(own.lifetime_hours, "h") * PRICE
    breakeven = own_total / rent_full
    assert abs(breakeven - own.breakeven_utilization) < 1e-9

    # ---------------------------------------------------------------- page 3
    hi = SC.HierarchicalAllReduce8x8DgxH100
    hir = hi.result
    hi_rs, hi_inter, hi_ag = _m(hir.reduce_scatter_time, "s"), _m(hir.inter_time, "s"), _m(hir.all_gather_time, "s")
    hi_t, hi_flat = _m(hir.total_time, "s"), _m(hir.flat_ring_time, "s")
    moe = SC.MoeDispatch8Gpu64Experts
    moer = moe.result
    moe_bytes = _m(moer.sent_bytes, "B")
    moe_t, moe_hot = _m(moer.injection_bound, "s"), _m(moer.hot_receive_bound, "s")
    cp = SC.CompressionPayback8x
    cp_t, cp_codec = _m(cp.communication_time, "s"), _m(cp.codec_time, "s")
    cp_new = _m(cp.total_time, "s")
    cp_fast = _m(cp.fast_communication_time, "s")
    cp_fast_new = cp_fast / cp.compression_ratio + cp_codec
    ob = SC.OverlapBudget7B64Gpu
    obr = ob.result
    ob_L, ob_bwd, ob_comm = obr.layers, _m(ob.backward_per_layer, "s"), _m(obr.communication_per_layer, "s")
    ob_exp = _m(obr.exposed_per_layer, "s")
    ob_seq, ob_ovl = _m(obr.sequential_time, "s"), _m(obr.overlapped_time, "s")
    assert abs(ob_L * ob_bwd + (ob_L - 1) * ob_exp + ob_comm - ob_ovl) < 1e-9, "printed overlap terms reproduce the total"
    ov = SC.RingAllReduceAcross4to1Spine
    ovr = ov.result
    OVR, OVF = ov.oversubscription_ratio, ov.communication_fraction
    bis_full, bis_over = _m(ovr.bisection_full, "TB/s"), _m(ovr.bisection_oversubscribed, "TB/s")
    thr = 1 / (1 - OVF + OVF * OVR)
    img = SC.ImageInputDemand2K   # per-GPU demand, applied to the reference fleet
    img_kb = round(_m(img.bytes_per_sample, "kB"))
    img_bw = N1 * img.samples_per_step * img_kb * 1e3 / _m(img.step_time, "s") / 1e9   # from the printed inputs
    st = SC.HalfSpeedStraggler1K
    st_n, st_rate = st.n_devices, _m(st.device_hour_rate, "USD/h")
    st_idle = st_n * (1 - _m(st.step_time, "s") / _m(st.slow_step, "s")) * st_rate
    assert abs(st_idle - _m(st.idle_cost, "USD")) < 1e-9
    q1, q3 = SC.Mg1ExponentialJobQueueAt80, SC.Mg1MixedJobQueueAt80
    el = SC.ElasticResizeDecision
    sh = SC.ServingConsolidationA100
    local_bw = SYS.Storage.LocalNvmeGen4x4.devices_per_node * _m(SYS.Storage.LocalNvmeGen4.bandwidth, "B/s")
    nodes1 = ref.fleet.count
    shard1 = ck_bytes / nodes1
    t_local1 = shard1 / local_bw
    alarms = SC.ThreeSigmaModelFleetAlerts
    pr = SC.PueRetrofit2MW
    prr = pr.result
    pr_usd = _m(prr.saved_energy, "kWh") * _m(pr.electricity_price, "USD/kWh")

    # ---------------------------------------------------------------- page 4
    px = SC.ChatbotSharedSystemPrompt
    sv = SC.LlmServingFirstYear
    can = SC.CanaryVersusCutoverExposure
    pgd = SC.PgdSevenTraining
    dp = SC.SalaryMeanLaplaceRelease
    fed = SC.PhoneImagesVersusQuantizedUpdates
    loan = SC.LoanScreeningTwoGroups
    ga, gb_ = loan.groups[loan.first], loan.groups[loan.second]
    appr = {k: (g["tp"] + g["fp"]) / sum(g.values()) for k, g in loan.groups.items()}
    tpr = {k: loan.tpr(k) for k in loan.groups}
    fpr_ = {k: loan.fpr(k) for k in loan.groups}
    assert abs(appr[loan.second] / appr[loan.first] - loan.disparate_impact) < 1e-9
    par = SC.CreditParityAccuracyCost
    lc = SC.Lab64A100TrainingFootprint
    lcr = lc.result

    def gb(x, d=1):
        return f"{x / 1e9:,.{d}f} GB"

    def ms(x, d=1):
        return f"{x * 1e3:,.{d}f} ms"

    def us(x, d=1):
        return f"{x * 1e6:,.{d}f} µs"

    NG = v("N", "GPU")
    nm = lambda mdl: mdl.name.replace("-", " ")

    # ---------------------------------------------------------------- chart: alpha-beta
    links = [("NVLink", NVL, ALPHA_NVL, "line"), ("NDR", BETA, ALPHA, "line cat3"), ("RoCE", ROCE_BW, ROCE_A, "line muted")]
    ap = Plot((1e2, 1e9), (1, 1e5), xlog=True, ylog=True, W=316, H=112, right=12, bottom=34)
    xl = {1e2: "100 B", 1e3: "1 KB", 1e4: "", 1e5: "100 KB", 1e6: "", 1e7: "10 MB", 1e8: "", 1e9: "1 GB"}
    yl = {1: "1 µs", 10: "", 100: "", 1e3: "1 ms", 1e4: "", 1e5: "100 ms"}
    ap.grid(list(xl), list(yl), xfmt=lambda x: xl[x], yfmt=lambda y: yl[y])
    for name, bw, al, cls in links:
        ap.line([(10 ** (2 + k * 0.1), (al + 10 ** (2 + k * 0.1) / bw) * 1e6) for k in range(71)], cls)
        ap.point(al * bw, 2 * al * 1e6)
    ap.text(ap.X(8e8), ap.Y(2.2), "β-bound", "end", cls="lab ital")
    ap.text(ap.X(1.4e2), ap.Y(25), "α-bound", cls="lab ital")
    ap.text((ap.l + ap.W - ap.r) / 2, ap.H - 13, "message size n · dot at n* = α·β · α = " + ", ".join(f"{al * 1e6:g}" for _, _, al, _ in links) + " µs", "middle", "ax")
    for k, (name, bw, al, cls) in enumerate(links):   # legend row below the axis, clear of the curves
        x0 = ap.l + k * 90
        ap.parts.append(f'<line x1="{x0}" x2="{x0 + 12}" y1="{ap.H - 5}" y2="{ap.H - 5}" class="{cls}"/>')
        ap.text(x0 + 15, ap.H - 2, f"{name} {bw / 1e9:g} GB/s", cls="tick")
    cy = (ap.t + ap.H - ap.b) / 2
    ap.parts.append(f'<text x="9" y="{cy:.1f}" class="ax" text-anchor="middle" transform="rotate(-90 9 {cy:.1f})">time</text>')

    # ---------------------------------------------------------------- chart: Fleet Law terms at two scales
    def bars():
        W, x0, x1, bh = 340, 54, 334, 14
        styles = (("comp", 'class="fill-accent"'), ("exp", 'style="fill:var(--accent-mid);stroke:var(--accent);stroke-width:.6"'),
                  ("bub", 'class="fill-neutral"'))
        out = [f'<svg viewBox="0 0 {W} 60" class="chart" role="img" aria-label="Reference step decomposition at two scales">']
        for k, (lab, t) in enumerate(((f"{fmt(N1)} GPUs", s1), (f"{fmt(N8)} GPUs", s8))):
            y = 3 + k * 31
            out.append(f'<text x="{x0 - 4}" y="{y + 10}" class="lab" text-anchor="end">{lab}</text>')
            xs = x0
            for key, style in styles:
                w = (x1 - x0) * t[key] / t["tot"]
                out.append(f'<rect x="{xs:.1f}" y="{y}" width="{w:.1f}" height="{bh}" {style}/>')
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

    # ---------------------------------------------------------------- chart: M/G/1 wait
    qp = Plot((0, 0.95), (0, 40), W=340, H=100, right=14, bottom=25)
    qp.grid([0, 0.2, 0.4, 0.6, 0.8], [0, 10, 20, 30, 40])
    for cv, cls in ((q1.service_cv, "line"), (q3.service_cv, "line cat3")):
        pts = [(r / 100, calc_mg1_wait_ratio(r / 100, cv)) for r in range(0, 95)]
        qp.line([p for p in pts if p[1] <= 40], cls)
    qp.point(q1.utilization, q1.wait_ratio, f"{q1.wait_ratio:.0f}×", "end", -6, -5)
    qp.point(q3.utilization, q3.wait_ratio, f"{q3.wait_ratio:.0f}×", "end", -6, 4)
    for k, (cv, cls) in enumerate(((q1.service_cv, "line"), (q3.service_cv, "line cat3"))):
        y = qp.t + 6 + k * 11
        qp.parts.append(f'<line x1="{qp.l + 8}" x2="{qp.l + 22}" y1="{y - 3}" y2="{y - 3}" class="{cls}"/>')
        qp.text(qp.l + 26, y, f"job-length cv = {cv:g}")
    qp.axes("fleet utilization ρ", "wait ÷ mean job length")

    # ---------------------------------------------------------------- page 1
    band = (f'<div class="eqbig">{v("T", "step")} = {frac(v("T", "compute"), NG)} + {v("T", "comm")} + {v("T", "sync")} − {v("T", "overlap")}'
            f'<span class="lbl">Compute · Communication · Coordination − Overlap</span></div>'
            f'<div class="txt"><div><b>The Fleet Law (Ch 1).</b> 175B reference job, {NG} = {fmt(N1)} H100s ({DP} replicas of TP{TP} × PP{PP}), modeled: '
            f'{s1["comp"]:.2f} s compute + {s1["comm"]:.2f} s {v("T", "comm")} + {s1["bub"]:.2f} s {v("T", "sync")} (pipeline bubble) − {s1["ovl"]:.2f} s {v("T", "overlap")} ≈ {n(f"{s1["tot"]:.2f} s")}. '
            f'Compute share {v("η", "scaling")} = {s1["comp"]:.2f} ÷ {s1["tot"]:.2f} = {n(fmt(s1["eff"] * 100) + "%")}.</div>'
            f'<div><b>{scale_x}× GPUs:</b> {s8["tot"]:.2f} s, {speedup_8x:.1f}× faster ({fmt(ideal_share * 100)}% of ideal); compute share {s8["comp"]:.2f} ÷ {s8["tot"]:.2f} = {fmt(s8["eff"] * 100)}%.</div></div>'
            f'<div class="side"><div><b>Pull the largest term\'s lever</b></div><div>{v("T", "comm")}: faster fabric, overlap, compression</div>'
            f'<div>{v("T", "sync")}: more microbatches, fewer barriers</div><div>{v("T", "compute")}/{NG}: faster or more accelerators</div></div>')

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
            ("Job", f'{v("P")} parameters · {v("D")} training tokens · {v("S")} sequence length · {v("d")} width · {v("N", "L")} layers · {v("T")} time'),
            ("Fleet", f'{NG} accelerators · {v("N", "DP")}, {v("N", "TP")}, {v("N", "PP")} data, tensor, pipeline degrees · {v("R", "peak")} peak FLOP/s · {v("η", "hw")} fraction of peak · {v("η", "scaling")} compute share of the step'),
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
                Card("Training time", f'{v("T")} ≈ 6{v("P")}{v("D")} / ({NG} · {v("R", "peak")} · {v("η", "hw")} · {v("η", "scaling")})', chapter="Ch 2",
                     holds=f'6 FLOPs per parameter per token: 2 forward + 4 backward.',
                     ex=f'GPT-3: 6 × {P3:.3g} × {D3:.3g} / ({fmt(N1)} × {PEAK:.3g} FLOP/s × {A}{ETA:.2f} × {s1["eff"]:.2f}) = {n(f"{t_days_fleet:.1f} days")}.',
                     so=f'Before failures. Raising {v("η", "hw")} or {v("η", "scaling")} buys as much as more GPUs.'),
                Card("Training state per device", None, chapter="Ch 6",
                     holds=f'Bytes per parameter: {fmt(B16)} weights + {fmt(B16)} gradients + {fmt(B32)} FP32 master + {fmt(BADAM)} Adam moments = {fmt(BPP_STATE)}; each ZeRO stage shards more of it over {v("N", "DP")} ranks.',
                     so=f'ZeRO-3 cuts state per GPU {zstage["DDP"] / zstage["ZeRO-3"]:.0f}× but moves {zr.traffic_multiple["ZeRO-3"]:.1f}× DDP\'s bytes.'),
                Table(["bytes per GPU", "DDP", "ZeRO-1", "ZeRO-2", "ZeRO-3"], [
                    ["formula", f"{fmt(BPP_STATE)}<i>P</i>", f"{fmt(2 * B16)}<i>P</i>+{fmt(B32 + BADAM)}<i>P</i>/<i>N</i><sub>DP</sub>",
                     f"{fmt(B16)}<i>P</i>+{fmt(B16 + B32 + BADAM)}<i>P</i>/<i>N</i><sub>DP</sub>", f"{fmt(BPP_STATE)}<i>P</i>/<i>N</i><sub>DP</sub>"],
                    [f"Llama 2 7B, {v('N', 'DP')} = {zN} (GB)"] + [fmt(zstage["DDP"])] + [f"{zstage[k]:.1f}" for k in ("ZeRO-1", "ZeRO-2", "ZeRO-3")]]),
                Card("Smallest replica that fits", f'{frac(fmt(BPP_STATE) + v("P"), v("N", "TP") + " · " + v("N", "PP"))} + activations ≤ HBM, &nbsp; {v("N", "DP")} = {frac(NG, v("N", "TP") + " · " + v("N", "PP"))}',
                     chapter="Ch 6",
                     ex=f'GPT-3, TP{TP} × PP{PP}: {fmt(BPP_STATE)} × {P3:.3g} / {TP * PP} = {gb(state_gpu)} + {gb(act_gpu)} activations and buffers = {n(gb(s1["mem"]))} ≤ {HBM / 1e9:.1f} GB. '
                        f'{v("N", "DP")} = {fmt(N1)} / {TP * PP} = {n(str(DP))}.',
                     so='TP fills one node; PP adds the fewest stages that fit.'),
            ], "Ch 2, 6"),
            Col("2 How fast do bytes move?", [
                Chart(ap.svg("Transfer time versus message size")),
                Card("Point to point: the α-β model", f'{v("T")}({v("n")}) = {v("α")} + {v("n")} / {v("β")}, &nbsp; crossover {v("n")}* = {v("α")} · {v("β")}', chapter="Ch 3",
                     holds=f'{v("n")}: message size in bytes.',
                     ex=f'NDR: {fmt(msg_small / 1e3)} KB → {ALPHA * 1e6:.1f} µs + {msg_small / BETA * 1e6:.2f} µs = {us(t_small, 2)}; {fmt(msg_big / 1e6)} MB → {n(ms(t_big, 2))}; {v("n")}* = {ALPHA * 1e6:.1f} µs × {fmt(BETA / 1e9)} GB/s = {fmt(nstar / 1e3)} KB.',
                     so=f'Below {fmt(nstar / 1e3)} KB, buy lower latency or batch messages; above it, buy bandwidth.'),
                Table(["collective (Ch 5)", f"time over {NG} ranks, payload <i>M</i> each"], [
                    ["Ring AllReduce", f"2({NG}−1)(<i>α</i>&nbsp;+&nbsp;<i>M</i>/{NG}<i>β</i>)"],
                    ["Tree, unsegmented", f"2&nbsp;log₂{NG}&nbsp;(<i>α</i>&nbsp;+&nbsp;<i>M</i>/<i>β</i>)"],
                    ["ReduceScatter, AllGather", f"({NG}−1)(<i>α</i>&nbsp;+&nbsp;<i>M</i>/{NG}<i>β</i>)"],
                    ["AllToAll (MoE dispatch)", f"({NG}−1)(<i>α</i>&nbsp;+&nbsp;<i>M</i>/{NG}<i>β</i>)"],
                ]),
                '<p class="so"><b>So</b> A ring sends each byte about twice whatever the GPU count; a tree pays fewer α but resends the payload at every level.</p>',
            ], "Ch 3, 5"),
            Col("3 Where does the step go?", [
                Chart(bars(), 'Compute (dark blue), exposed communication (light blue), bubble (gray), as shares of the step.'),
                Card("Pipeline bubble", f'{v("b", "pipe")} = {frac(v("N", "PP") + " − 1", v("N", "PP") + " − 1 + " + v("V") + v("m"))}', chapter="Ch 6",
                     holds=f'{v("m")}: microbatches per step; {v("V")}: interleaved stages per device (else 1). Share of pipeline time.',
                     ex=f'{v("N", "PP")} = {pb.stages}, {v("m")} = {pb.microbatches}: {pb.stages - 1} / {pb.stages - 1 + pb.microbatches} = {n(fmt(bub_32 * 100, 1) + "%")}. Reference: {v("m")} = {MB} → {fmt(bub_ref * 100, 1)}%; '
                        f'at {fmt(N8)} GPUs each replica gets {v("m")} = {MB8} → {n(fmt(bub_8k * 100, 1) + "%")} of pipeline time.',
                     so=f'A fixed batch on more replicas means fewer {v("m")} each: the bubble grows (the bars count it per step).'),
                Card("Tensor parallelism stays inside the node", f'{TP_AR} per layer, {v("M")} = {v("B", "micro")} · {v("S")} · {v("d")} · {fmt(B16)} bytes', chapter="Ch 6, 10",
                     holds=f'AllReduces over a ring of {v("N", "TP")}; {v("B", "micro")}: sequences per microbatch.',
                     ex=f'GPT-3, {A}{v("B", "micro")} = {S["tp_seqs"]}: {fmt(G3.context_length)} × {fmt(W3)} × {fmt(B16)} = {m_act / 1e6:.0f} MB. Ring of {S["tp_ring"]}: NVLink {n(us(tp_nvl, 0))}, NDR {n(ms(tp_ndr, 2))}; '
                        f'{tp_layers_stage} layers per stage: {TP_AR * tp_layers_stage} per microbatch.',
                     so=f'{tp_ndr / tp_nvl:.0f}× slower on NDR: TP stays inside the node.'),
            ], "Ch 6"),
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

    part_txt = " + ".join(f"{fmt(cnt)}/{fmt(mt)}" for _, cnt, mt in parts)
    part_names = ", ".join(f"{lab} at {fmt(mt)} h" for lab, _, mt in parts)

    p2 = Page(
        question="Will it survive and serve?",
        sub="Columns 4–6 continue page 1: failures as a steady state, serving at scale, and what the fleet costs. ▲ marks a result that misses a limit.",
        meta=f"H100 fleet · {nm(wl)} serving replica",
        notation=[
            ("Reliability", f'MTTF of one part · MTBF of the whole system · {v("T", "write")} checkpoint write time · {v("T", "ckpt")} checkpoint interval · {v("W")} fraction of time wasted · {v("T")} time'),
            ("Serving", f'{v("M", "KV")} KV bytes per request · {v("S")} context tokens · {v("B")} slots per replica · {v("N", "TP")} tensor-parallel GPUs · BW HBM bandwidth · {v("λ")} requests/s · {v("t", "s")} service time · {v("ρ")} utilization'),
            ("Facility", f'{v("P", "IT")} IT power · PUE = {v("P", "facility")}/{v("P", "IT")} · {v("E")} energy · CI grid g CO₂/kWh · {A}= example input'),
        ],
        cols=[
            Col("4 Will it survive?", [
                Card("Fleet MTBF", [f'MTBF = 1 / Σ<sub>parts</sub> (count / MTTF)', f'expected failures in {v("T")} = {v("T")} / MTBF'], chapter="Ch 7",
                     ex=f'{fmt(n10k)} H100s ({fmt(nodes10k)} nodes; {part_names}): 1 / ({part_txt}) = {n(f"{mtbf_sys:.2f} h")} ({mtbf_gpu:.1f} h from GPUs alone). '
                        f'{A}{v("T")} = {run_h / H_DAY:.0f} days → {fmt(run_fail)} failures.',
                     so='At fleet scale, failure every few hours is normal: plan for it.'),
                Card("Checkpoint write time", f'{v("T", "write")} = {fmt(BPP_CKPT)} bytes × {v("P")} ÷ write bandwidth', chapter="Ch 4, 7",
                     ex=f'GPT-3: {fmt(BPP_CKPT)} × {P3:.3g} = {ck_bytes / 1e12:.2f} TB ÷ {CKPT_BW / 1e9:.0f} GB/s = {n(f"{t_write:.1f} s")}.',
                     so=f'No gradients in the checkpoint: {fmt(BPP_CKPT)}, not {fmt(BPP_STATE)}, bytes per parameter.'),
                Card("Young-Daly checkpoint interval", [f'{v("T", "ckpt")}* = √(2 · {v("T", "write")} · MTBF)', f'{v("W")} ≈ {frac(v("T", "write"), v("T", "ckpt"))} + {frac(v("T", "ckpt"), "2 · MTBF")} &nbsp; (saving + rework)'], chapter="Ch 7",
                     ex=f'√(2 × {t_write:.1f} s × {mtbf_sys * S_HR:,.0f} s) = {fmt(tau)} s ≈ {n(f"{tau / 60:.0f} min")}; {v("W")} = {n(fmt(waste * 100, 1) + "%")}.',
                     so=f'Restart is budgeted apart (Ch 7): {rec_s:.1f} s per failure adds {rec_share * 100:.1f}%. Faster writes shorten the interval and cut both.'),
                Chart(yd.svg("Checkpoint waste versus interval")),
            ], "Ch 4, 7"),
            Col("5 Can it serve at scale?", [
                Card("KV cache per request", f'{v("M", "KV")} = 2 · {v("N", "L")} · {v("H", "KV")} · {v("d", "head")} · {v("s", "elem")} · {v("S")}', chapter="Ch 10",
                     holds=f'{v("s", "elem")}: bytes per element ({fmt(B16)} in FP16).',
                     ex=f'{nm(wl)}: 2 × {wl.layers} × {wl.kv_heads} × {kvr.head_dim} × 2 = {fmt(m_tok / 1024)} KiB/token; × {fmt(kv.context_tokens)} = {n(f"{_m(kvr.kv_per_request, "GB"):.1f} GB")}. '
                        f'{n_tp} × {fmt(HBM_GIB)} GiB = {fmt(_m(kvr.hbm_total, "GB"))} GB − {gb(w_bytes)} weights − {A}{_m(kv.reserve, "GB"):.0f} GB → {n(f"{kvr.max_requests} requests")}.',
                     so='At long context the KV cache, not the weights, limits how many users a replica holds.'),
                Card("Decode step and replica count", [f'{v("t", "step")} = (weights + {v("B")} · {v("M", "KV")}) / ({v("N", "TP")} · BW)',
                                                       f'{v("t", "s")} = output tokens × {v("t", "step")}, &nbsp; replicas = ⌈{frac(v("λ") + v("t", "s"), v("ρ") + v("B"))}⌉'],
                     chapter="Ch 10",
                     holds=f'{v("t", "s")}: how long a request holds one of {v("B")} slots. Little\'s law: {v("λ")}{v("t", "s")} requests are in service. '
                           f'{A}{v("S")} = {fmt(S["serve_ctx"])}, {v("B")} = {S["serve_slots"]}, {S["serve_out"]} output tokens, {v("λ")} = {S["serve_lam"]}/s, {v("ρ")} = {S["serve_rho"]}.',
                     ex=f'{nm(wl)}, TP{n_tp}, {v("M", "KV")} = {gb(mkv_req, 2)}. ① {v("t", "step")} = ({gb(w_bytes)} + {S["serve_slots"]} × {gb(mkv_req, 2)}) / ({n_tp} × {HBM_BW / 1e12:.2f} TB/s) = {ms(t_stepB, 2)} '
                        f'({ms(t_step1, 2)} at {v("B")} = 1). ② {v("t", "s")} = {S["serve_out"]} × {ms(t_stepB, 2)} = {t_hold:.2f} s. ③ {v("λ")}{v("t", "s")} = {fmt(q_req)} in service. '
                        f'④ ⌈{fmt(q_req)} / ({S["serve_rho"]} × {S["serve_slots"]})⌉ = {n(f"{replicas} replicas")}.',
                     so=f'{S["serve_slots"]} users cost only {(t_stepB / t_step1 - 1) * 100:.0f}% more step time than one: the weights are read once per step.'),
                Card("Fan-out multiplies rare events", f'Pr(at least one of {v("k")} is slow) = 1 − (1 − {v("p")})<sup>{v("k")}</sup>', chapter="Ch 4, 10",
                     ex=f'{A}{v("k")} = {fan.n_components} leaves, {v("p")} = {1 - fan.availability:.2f}: 1 − {fan.availability:.2f}<sup>{fan.n_components}</sup> = {n(fmt((1 - fan.all_up) * 100, 1) + "%")} of requests hit a slow leaf.',
                     so='A rare per-part event becomes the common case for the request: hedge or replicate the tail.'),
            ], "Ch 4, 10"),
            Col("6 What does the fleet cost?", [
                Card("Power and PUE", [f'{v("P", "IT")} = {NG} × GPU TDP × IT ratio', f'{v("P", "facility")} = PUE · {v("P", "IT")}'], chapter="Ch 2, 15",
                     holds=f'IT ratio, from the book\'s DGX H100 rack profile: ({rack_gpus} × {fmt(TDP)} W GPUs + {rack_support_kw:.1f} kW host, NVSwitch, network, power conversion) ÷ {rack_gpu_kw:.1f} kW = {it_ratio:.3f}. Cooling is in PUE.',
                     ex=f'Rack: {rack_gpu_kw:.1f} + {rack_support_kw:.1f} = {n(f"{rack_kw:.1f} kW", True)} > {fmt(AIR_KW)} kW air limit. {fmt(fp.devices)} H100s: {fmt(fp.devices)} × {fmt(TDP)} W × {it_ratio:.3f} = {_m(fpr.it_power, "MW"):.1f} MW × {fp.pue:.2f} = {n(f"{_m(fpr.facility_power, "MW"):.2f} MW")}.',
                     so='Dense AI racks need liquid cooling; the facility draws a small city\'s power.'),
                Card("Energy and carbon", f'{v("E")} = PUE · {v("P", "IT")} · {v("T")}, &nbsp; CO₂ = {v("E")} × CI', chapter="Ch 15",
                     ex=f'GPT-3: {fmt(G3_MWH)} MWh × {fmt(CI["US_Avg"])} g/kWh = {n(fmt(g3_t) + " t")}. {A}{fmt(_m(gpl.energy, "MWh"))} MWh: Quebec × {fmt(CI["Quebec"])} = {fmt(_m(gplr.emissions["Quebec"], "t"))} t, '
                        f'Poland × {fmt(CI["Poland"])} = {n(fmt(_m(gplr.emissions["Poland"], "t")) + " t")}.',
                     so=f'Where you run changes carbon {gplr.spread:.0f}×; tuning changes it by percent.'),
                Card("Own or rent?", f'break-even use = {frac("owned cost", "GPUs × hours × rent rate")}', chapter="Ch 12",
                     ex=f'{fmt(own.gpus)} H100s, {own.years} years: ${own_hw / 1e6:.1f}M nodes + ${own_nf / 1e6:.0f}M network and facility + ${own_staff / 1e6:.0f}M staff + ${own_power / 1e6:.1f}M power '
                        f'({_m(own.power, "MW"):.0f} MW × PUE {own.pue}, ${_m(own.electricity_price, "USD/kWh"):.2f}/kWh) = ${own_total / 1e6:.0f}M. Renting all of it: {fmt(own.gpus)} × {fmt(_m(own.lifetime_hours, "h"))} h × ${PRICE:.0f} = ${rent_full / 1e9:.2f}B. '
                        f'{own_total / 1e6:.0f} / {rent_full / 1e6:,.0f} = {n(fmt(breakeven * 100) + "%")}.',
                     so=f'Own if you will keep the GPUs busy more than {fmt(breakeven * 100)}% of the time.'),
                Note("Before you trust the number:", 'GB or Gb (÷ 8)? Per direction or both? MTBF of the whole system? KV cache counted in the step? Modeled or measured?'),
            ], "Ch 2, 9, 12, 15"),
        ],
        bottom=ref_models,
    )

    # ---------------------------------------------------------------- page 3
    p3 = Page(
        question="Can the overhead shrink?",
        sub="Columns 7–9 extend pages 1–2 on the Fleet Law's overhead terms: cheaper collectives (communication), hidden transfers, and busy GPUs (coordination).",
        meta="communication and coordination levers",
        notation=[
            ("Network", f'{NG} accelerators · {v("G", "node")} GPUs per node · {v("α")} per-message latency · {v("β")} link bandwidth per direction · {v("M")} collective payload per rank'),
            ("Time", f'{v("T", "step")} training step · {v("T", "comm")} communication time · {v("ρ")} utilization · {A}= example input'),
        ],
        cols=[
            Col("7 Can the collective be cheaper?", [
                Card("Ring or tree?", [f'{v("M")}* = {v("α")}{v("β")} · {frac("(N − 1) − log₂N", "log₂N − (N − 1)/N")}'], chapter="Ch 5, App. D",
                     holds=f'Trees win below {v("M")}*. N = {NG} ranks; App. D\'s exact crossover for unsegmented trees.',
                     ex=f'{nm(M.Language.Llama2_70B)} gradients, {v("M")} = {gb(grM, 0)} BF16, {grN} NDR GPUs: ring 2 × {grN - 1}/{grN} × {gb(grM, 0)} / {fmt(BETA / 1e9)} GB/s = {n(f"{gr_ring:.2f} s")}; unsegmented tree {gr_tree:.1f} s. '
                        f'Crossover: {fmt(xover_64 / 1e3)} KB at {grN} GPUs, {xover_1k / 1e6:.2f} MB at {fmt(N1)}.',
                     so='Rings for gradient buckets, trees for small messages; pipelined trees close most of the gap.'),
                Card("Hierarchical AllReduce", [f'RS → ring on {v("M")}/{v("G", "node")} → AG'], chapter="Ch 5",
                     holds=f'ReduceScatter (RS) and AllGather (AG) run on NVLink; only 1/{v("G", "node")} of the payload crosses the network ring.',
                     ex=f'{gb(_m(hi.message, "B"), 0)} on {hi.n_nodes} nodes × {hi.gpus_per_node}: {ms(hi_rs, 2)} + {ms(hi_inter, 2)} + {ms(hi_ag, 2)} = {n(ms(hi_t, 2))}, vs one flat ring over {hi.n_devices} GPUs {ms(hi_flat, 1)}.',
                     so=f'{hi_flat / hi_t:.1f}× faster: keep as many bytes as possible on NVLink.'),
                Card("MoE dispatch is an AllToAll", [f'bytes out = {frac(NG + " − 1", NG)} × tokens × {v("d")} × {v("s", "elem")}', f'{v("T")} ≈ {v("α")} + bytes / {v("β")}'], chapter="Ch 5",
                     holds=f'Per GPU; tokens on that GPU; {v("d")}: hidden width; {v("s", "elem")}: bytes per element.',
                     ex=f'{A}{moe.total_tokens} tokens over {moe.n_devices} GPUs, {moe.n_experts} experts, {v("d")} = {moe.hidden_dim}, BF16: {moer.tokens_per_device} × {moe.hidden_dim} × {fmt(_m(moe.bytes_per_element, "B"))} × {moe.n_devices - 1}/{moe.n_devices} = {moe_bytes / 1e6:.2f} MB → '
                        f'{n(us(moe_t))} on NDR. One expert {moe.receive_skew}× hot: {n(us(moe_hot))} to receive.',
                     so='The hottest expert sets the step: balance routing before buying bandwidth.'),
                Card("Does gradient compression pay?", f'{v("T", "comm")}′ = {v("T", "comm")} / ratio + codec time', chapter="Ch 5",
                     ex=f'{A}{ms(cp_t, 0)} sync, {cp.compression_ratio:.0f}× smaller, {ms(cp_codec, 0)} codec: {ms(cp_t, 0)}/{cp.compression_ratio:.0f} + {ms(cp_codec, 0)} = {n(ms(cp_new, 0))} ({cp.speedup:.1f}× faster). '
                        f'A {ms(cp_fast, 0)} sync: {n(ms(cp_fast_new, 2), True)} > {ms(cp_fast, 0)}.',
                     so='Compress only when the link, not the codec, is the bottleneck.'),
            ], "Ch 5"),
            Col("8 Can transfers hide or keep up?", [
                Card("Overlap budget", [f'{v("T")} ≈ {v("N", "L")}{v("t", "bwd")} + ({v("N", "L")} − 1) · exposed + {v("t", "comm")}', f'exposed = max(0, {v("t", "comm")} − {v("t", "bwd")})'], chapter="Ch 5",
                     holds=f'{v("t", "bwd")}, {v("t", "comm")}: backward and gradient-AllReduce time per layer; each layer\'s AllReduce runs under the next layer\'s backward.',
                     ex=f'{nm(ob.workload)}, {ob.n_devices} GPUs, {A}{v("t", "bwd")} = {ms(ob_bwd, 0)}, {v("t", "comm")} = {ms(ob_comm, 0)}: {ob_L} × {ms(ob_bwd, 0)} + {ob_L - 1} × {ms(ob_exp, 0)} + {ms(ob_comm, 0)} = {n(ms(ob_ovl, 0))}, '
                        f'vs {ob_L} × ({ms(ob_bwd, 0)} + {ms(ob_comm, 0)}) = {ms(ob_seq, 0)} in sequence.',
                     so=f'Overlap saves {obr.savings_fraction * 100:.0f}%; with {v("t", "comm")} > {v("t", "bwd")}, {ms(ob_exp, 0)} per layer still shows.'),
                Card("Oversubscription R:1 at the spine", [f'bisection = {NG}{v("β")} / 2R', f'throughput = 1 / (1 − {v("f")} + {v("f")} · R)'], chapter="Ch 3",
                     holds=f'R: endpoint to uplink bandwidth ratio; {v("f")}: share of the step whose traffic crosses the spine.',
                     ex=f'{fmt(ov.fleet.total_accelerators)} NDR GPUs: {fmt(ov.fleet.total_accelerators)} × {fmt(BETA / 1e9)} GB/s / 2 = {fmt(bis_full, 1)} TB/s; ÷ {OVR:.0f} = {fmt(bis_over, 1)} TB/s. '
                        f'{A}{v("f")} = {OVF:.1f}: 1 / ({1 - OVF:.1f} + {OVF:.1f} × {OVR:.0f}) = {n(fmt(thr * 100) + "%")}.',
                     so='A cheaper spine taxes only the communicating share, but that share grows with scale.'),
                Card("Can storage feed the fleet?", f'BW<sub>storage</sub> ≥ {NG} × bytes per GPU per step / {v("T", "step")}', chapter="Ch 4",
                     ex=f'{fmt(N1)} GPUs × {A}{img.samples_per_step} images × {img_kb} KB / {_m(img.step_time, "ms"):.0f} ms = {n(f"{img_bw:.0f} GB/s")}.',
                     so='Images need a parallel file system or local NVMe cache; token IDs barely load the disk.'),
                Card("Stage checkpoints on local NVMe", 'write time = checkpoint / (nodes × BW<sub>NVMe</sub>)', chapter="Ch 4, 7",
                     ex=f'GPT-3, {ck_bytes / 1e12:.2f} TB over the reference job\'s {nodes1} nodes, {SYS.Storage.LocalNvmeGen4x4.devices_per_node} × {_m(SYS.Storage.LocalNvmeGen4.bandwidth, "GB/s"):.0f} GB/s each: {gb(shard1)} / {local_bw / 1e9:.0f} GB/s = {n(f"{t_local1:.2f} s")}, vs {t_write:.1f} s to shared storage.',
                     so='Training pauses for under a second; the copy to durable storage drains in the background.'),
            ], "Ch 3, 4, 5"),
            Col("9 Can the scheduler keep GPUs busy?", [
                Card("Stragglers: the slowest GPU sets the step", f'idle $/h = {NG} × (1 − {v("T", "step")}/{v("T", "step")}′) × $/GPU-h', chapter="Ch 7",
                     holds=f'{v("T", "step")}′: step time with the straggler (synchronous training waits for it).',
                     ex=f'{A}one of {fmt(st_n)} GPUs at {fmt(st.slow_speed * 100)}% speed: {_m(st.step_time, "s"):.0f} s → {_m(st.slow_step, "s"):.0f} s; {fmt(st_n)} × (1 − {_m(st.step_time, "s"):.0f}/{_m(st.slow_step, "s"):.0f}) × ${st_rate:.0f} = {n("$" + fmt(st_idle) + "/hour")}.',
                     so='Detect and evict slow GPUs fast; one bad part idles the whole fleet.'),
                Card("Queue wait in a shared fleet (M/G/1)", f'wait ÷ mean job length = {frac(v("ρ"), "1 − " + v("ρ"))} × {frac("1 + cv²", "2")}', chapter="Ch 8",
                     holds='cv: coefficient of variation (std ÷ mean) of job length.',
                     ex=f'{A}{v("ρ")} = {q1.utilization}: {q1.load_factor:.0f} × {q1.variability_factor:.0f} = {n(f"{q1.wait_ratio:.0f}×")} at cv = {q1.service_cv:g}; {q3.load_factor:.0f} × {q3.variability_factor:.0f} = {n(f"{q3.wait_ratio:.0f}×")} at cv = {q3.service_cv:g}.',
                     so='Wait explodes as ρ → 1, faster when job lengths vary: separate short and long jobs.'),
                Chart(qp.svg("Queue wait versus utilization")),
                Card("Goodput: what held GPU-hours buy", "goodput = scheduling × runtime × MFU", chapter="Ch 9",
                     ex=f'{nm(M.Language.Llama3_405B)}, {fmt(gp_gpus)} dedicated H100s: {gp.scheduling:.1f} × {gp.runtime:.2f} × {gp.program:.2f} = {n(fmt(gp.result.mpg * 100, 1) + "%")}.',
                     so='Most capacity is lost inside the step (MFU).'),
                Card("Grow now or wait for GPUs?", ['epochs = Σ throughput × hours', 'cost = Σ GPUs × hours'], chapter="Ch 8",
                     ex=f'{A}{el.window_h} h window, throughput {el.small_throughput:g} / {el.step_up_throughput:g} / {el.full_throughput:g} epochs/h on {el.small_gpus} / {el.step_up_gpus} / {el.full_gpus} GPUs, {el.wait_h} h queue for {el.full_gpus}. '
                        f'Epochs / GPU-h: start small {el.start_small_epochs:.0f} / {fmt(el.gpu_hours["start_small"])}; wait {el.wait_run_h} h × {el.full_throughput:g} = {n(f"{el.wait_epochs:.0f}")} / {fmt(el.gpu_hours["wait"])}; '
                        f'step up {el.step_up_epochs:.1f} / {fmt(el.gpu_hours["step_up"])}.',
                     so='Wait for the deadline; start small for the budget.'),
            ], "Ch 7, 8"),
        ],
    )

    # ---------------------------------------------------------------- page 4
    p4 = Page(
        question="Can it serve responsibly?",
        sub="Columns 10–12: serving with fewer GPUs, changing a live system safely and privately, and measuring fairness and lifecycle carbon.",
        meta="Volume II, Part III and Part IV",
        notation=[
            ("Serving and trust", f'{v("M", "KV")} KV bytes · {v("ε")} privacy budget · Δ sensitivity (most one person can change a result) · TPR true-positive rate · {A}= example input'),
        ],
        cols=[
            Col("10 Can serving use fewer GPUs?", [
                Card("Prefix caching", ['tokens, unshared = users × (prefix + own)', 'tokens, shared = prefix + users × own'], chapter="Ch 10",
                     holds='prefix: system-prompt tokens every user shares; own: tokens unique to one user. KV memory scales with tokens.',
                     ex=f'{A}{fmt(px.users)} users, {fmt(px.prefix_tokens)}-token system prompt, {px.unique_tokens_per_user} own tokens: {fmt(px.tokens_unshared)} → {fmt(px.tokens_shared)} tokens; '
                        f'full-width {nm(px.workload)} KV {_m(px.kv_unshared, "TB"):.2f} → {n(f"{_m(px.kv_shared, "TB"):.2f} TB")}.',
                     so=f'{px.reduction * 100:.0f}% less KV memory, {px.capacity_ratio:.1f}× the users per replica, and {fmt(px.prefill_tokens_skipped)} prefill tokens skipped.'),
                Card("Serving cost overtakes training", '$/year = 365 × users × requests/day × $/request', chapter="Ch 10",
                     ex=f'{A}365 × {fmt(sv.daily_users)} × {sv.requests_per_user_per_day} × ${sv.cost_per_request:g} = {n("$" + f"{sv.annual_serving_cost / 1e6:.2f}M/year")}; ÷ ${sv.training_cost / 1e6:.0f}M training = {sv.serving_to_training:.1f}×.',
                     so=f'Serving passes the training bill in {sv.crossover_weeks:.1f} weeks; a {fmt(sv.efficiency_gain * 100)}% serving gain saves ${sv.efficiency_savings / 1e6:.2f}M a year.'),
                Card("Federated learning: send updates, not data", 'update bytes = parameters × bits ÷ 8', chapter="Ch 11",
                     ex=f'{A}{fmt(fed.items_per_week)} photos × {_m(fed.item_size, "kB"):.0f} KB = {_m(fed.raw_volume, "MB"):.0f} MB a week; {fed.update_params / 1e6:.0f}M × {fed.update_bits_per_param} bits ÷ 8 = '
                        f'{n(f"{_m(fed.update_size, "MB"):.1f} MB")} per round, {fed.reduction:.0f}× less.',
                     so=f'Past {fed.break_even_rounds:.0f} rounds a week the updates outweigh the photos: round frequency is the lever.'),
                Card("Shared GPUs serve more models", 'GPUs = Σ ⌈footprint ÷ memory share⌉', chapter="Ch 8",
                     ex=f'{A}{sh.n_small} {nm(sh.small_model)} models ({_m(sh.small_footprint, "GB"):.0f} GB) + {sh.n_large} {nm(sh.large_model)} ({sh.gpus_per_large} GPUs each): {sh.exclusive_gpus} GPUs alone, '
                        f'{sh.shared_gpus} with {sh.tenants} small models per {_m(sh.device_memory, "GB"):.0f} GB GPU → {n(f"{sh.saved_gpus} GPUs")}, ${sh.annual_value / 1e3:.0f}K/year at ${_m(sh.device_hour_rate, "USD/h"):.0f}/GPU-h.',
                     so='Pack small models only after load tests show the pair meets both latency targets.'),
            ], "Ch 10, 11"),
            Col("11 Is it safe and private?", [
                Card("Staged rollout limits exposure", 'exposure ∝ traffic share × time to roll back', chapter="Ch 12",
                     holds='exposure: requests served by a bad model before it is rolled back.',
                     ex=f'{A}{fmt(can.canary_share * 100)}% canary vs {fmt(can.full_share * 100)}% cutover, both {_m(can.time_to_rollback, "h"):.0f} h to roll back: {fmt(can.full_share * 100)} / {fmt(can.canary_share * 100)} = {n(f"{can.exposure_ratio:.0f}× less")} exposure.',
                     so='Only true if the canary detects the fault as fast; a smaller slice needs a longer watch.'),
                Card("Robustness costs compute", 'cost ≈ (attack steps + 1) × standard training', chapter="Ch 14",
                     holds='PGD-k crafts each adversarial batch with k gradient steps, then trains on it.',
                     ex=f'PGD-{pgd.attack_passes}: {pgd.attack_passes} attack passes + {pgd.update_passes} update = {n(f"{pgd.slowdown:.0f}×")} the compute per batch.',
                     so='Budget robust training like a larger model; weaker attacks are cheaper but defend less.'),
                Card("Differential privacy costs accuracy", [f'Laplace noise scale = Δ / {v("ε")}', f'error of a mean of {v("n")} values ≈ Δ / ({v("n")} · {v("ε")})'], chapter="Ch 13",
                     holds=f'Smaller {v("ε")} is more private and noisier.',
                     ex=f'{A}salaries $0–{_m(dp.value_range, "USD") / 1e3:.0f}K, {v("ε")} = {dp.epsilon:g}: {v("n")} = {fmt(dp.group_size)} → ${_m(dp.value_range, "USD") / 1e3:.0f}K / {fmt(dp.group_size)} = {n("$" + fmt(_m(dp.noise_scale, "USD") / dp.group_size))}; '
                        f'{v("n")} = {dp.small_group_size} → {n("$" + fmt(_m(dp.small_group_error, "USD")))}.',
                     so='Privacy is cheap for large groups and expensive for small ones.'),
                Card("False alarms across many monitors", 'false alerts/day = monitors × checks/day × rate', chapter="Ch 12",
                     ex=f'{A}{alarms.models} models × {alarms.metrics_per_model} metrics, every {_m(alarms.check_interval, "min"):.0f} min, 3σ ({alarms.false_positive_rate:g}): {fmt(alarms.monitors)} × {fmt(alarms.checks_per_day)} × {alarms.false_positive_rate:g} = {n(fmt(alarms.daily_false_alerts) + " per day")}.',
                     so='Thresholds that are fine for one metric drown on-call at fleet scale: group, deduplicate, or raise them.'),
            ], "Ch 12, 13, 14"),
            Col("12 Is it fair and sustainable?", [
                Card("Fairness metrics from a confusion matrix", ['approval = (TP + FP) / all, &nbsp; TPR = TP / (TP + FN)', 'disparate impact = approval B / approval A'], chapter="Ch 16",
                     holds='TP: approved and repaid · FP: approved, defaulted · FN: rejected, would have repaid.',
                     ex=f'{A}{sum(ga.values()):.0f} applicants per group. A: TP {ga["tp"]:.0f}, FP {ga["fp"]:.0f}, FN {ga["fn"]:.0f}; B: TP {gb_["tp"]:.0f}, FP {gb_["fp"]:.0f}, FN {gb_["fn"]:.0f}. '
                        f'Approval {appr["A"] * 100:.0f}% vs {appr["B"] * 100:.0f}% → disparate impact {n(f"{loan.disparate_impact:.2f}")}; TPR {tpr["A"]:.2f} vs {tpr["B"]:.2f} → equal-opportunity gap {n(f"{tpr["A"] - tpr["B"]:.2f}")}.',
                     so='Different metrics can disagree; pick the one the decision\'s harm calls for before you tune.'),
                Card("Fairness constraints cost accuracy", 'cost = baseline accuracy − constrained accuracy', chapter="Ch 16",
                     ex=f'{A}demographic parity on a credit model: {par.baseline_accuracy:.0f}% − {par.constrained_accuracy:.0f}% = {n(f"{par.accuracy_cost:.0f} points")} ({par.relative_cost * 100:.1f}% relative).',
                     so='State the trade-off explicitly; the cost lands on the groups whose approvals change.'),
                Card("Lifecycle carbon: operations + manufacturing", ['CO₂ = energy × CI + embodied × hours / life'], chapter="Ch 15",
                     holds='energy: facility energy of the run; embodied: CO₂ to make the GPUs; life: their service life in hours.',
                     ex=f'{A}{nm(M.Language.Llama2_7B)}, {lc.devices:.0f} A100s for {_m(lcr.duration, "h") / H_DAY:.0f} days: {_m(lcr.facility_energy, "MWh"):.1f} MWh → '
                        f'US {_m(lcr.carbon["US average"], "t"):.2f} t, Quebec {_m(lcr.carbon["Quebec"], "t"):.2f} t. Embodied {_m(lcr.embodied_manufactured, "t"):.1f} t × {_m(lcr.duration, "h"):.0f} / {_m(lc.service_life, "h"):,.0f} h = {n(f"{_m(lcr.embodied_allocated, "t"):.3f} t")}.',
                     so=f'Embodied carbon is {lcr.embodied_share["US average"] * 100:.1f}% of the total on the US grid but {lcr.embodied_share["Quebec"] * 100:.0f}% on hydro: clean power makes manufacturing matter.'),
                Card("PUE upgrade payback", 'saved = (PUE<sub>before</sub> − PUE<sub>after</sub>) × IT power × hours', chapter="Ch 15",
                     ex=f'{A}{_m(pr.it_power, "MW"):.0f} MW IT, PUE {pr.pue_before} → {pr.pue_after}: ({pr.pue_before} − {pr.pue_after}) × {_m(pr.it_power, "MW"):.0f} MW × {fmt(_m(pr.period, "h"))} h = {n(fmt(_m(prr.saved_energy, "MWh")) + " MWh/yr")}, '
                        f'× ${_m(pr.electricity_price, "USD/kWh"):.2f}/kWh = ${pr_usd / 1e3:,.0f}K/yr.',
                     so='Every 0.1 of PUE costs 10% of the IT load.'),
            ], "Ch 15, 16"),
        ],
        bottom=Table(["Cluster tier (DGX H100, IB NDR)", "GPUs", "nodes", "MTBF, GPUs only", "GPU failures/day", "ring latency 2(N<sub>GPU</sub>−1)α", "bisection at 1:1", "facility power"], trows),
    )
    return Sheet(volume=2, book="Scaling Machine Learning Systems", scope="The Fleet", pages=[p1, p2, p3, p4],
                 sources=f'GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B, {_m(NDR.bandwidth, "Gbit/s"):.0f} Gb/s = {BETA / 1e9:.0f} GB/s · {TAG}= example input · values: mlsysim registry and Volume II scenarios; step times modeled')

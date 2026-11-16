"""Napkin Math, Volume I: one node (one machine, one to eight accelerators).

Every number below is computed here from the mlsysim registry or scenarios.
Inputs chosen only for an example are tagged with assumed() on the sheet and
listed in SCENARIO. Constants from outside the registry are listed in EXTERNAL
with their source.
"""
from __future__ import annotations

import math

from mlsysim import Hardware as HW, Infrastructure as INF, Literature as LIT, Models as M, Systems as SYS
from mlsysim.engine.scenarios import Scenarios as SC

from .core import (Card, Chart, Col, Know, Note, Page, Plot, Sheet, Table,
                   assumed, fmt, frac, n, v)

EXTERNAL = {"llama3_vocab": 128256}  # Llama 3 tokenizer vocabulary (Meta Llama 3 model card)
SCENARIO = dict(ctx=2048, prompt=2048, eta_prefill=0.50, batches=(1, 32, 128), t_s=0.010, lam=80,
                mcu_macs_per_cycle=2, fps=30, eta_edge=0.30, conv=(56, 3, 128, 128), q_range=(-6.0, 6.0),
                prune=0.5, idx_bytes=2, amd_p=0.9, amd_a=10, ssd=550e6, f_loss=0.10, sample=110e3,
                # Page 3, matching the book's LEGO cells: the target effective batch of Ch 8's
                # GradientAccumulation cell, the per-GPU batches of GPT2NodeAllReduce plotted in the
                # sync-share chart, and the three grids compared in the carbon card.
                eff_batch=512, sync_batches=(1, 2, 4, 8, 16, 32, 64), grids=("Quebec", "US_Avg", "Poland"))


def build() -> Sheet:
    S = SCENARIO
    A = TAG = assumed()
    H = HW.Cloud.H100
    PEAK = {k: q.magnitude * 1e12 for k, q in H.compute.precision_flops.items()}  # TFLOP/s or TOPS
    PEAK["fp16"] = H.compute.peak_flops.to("flop/s").magnitude
    BW = H.memory.bandwidth.to("B/s").magnitude
    CAP = H.memory.capacity.to("B").magnitude
    CAP_GIB = H.memory.capacity.to("GiB").magnitude
    PCIE = H.interconnect.bandwidth.to("B/s").magnitude
    TDP = H.tdp.to("W").magnitude
    LAUNCH_US = H.dispatch_tax.to("microsecond").magnitude

    L70 = M.Language.Llama3_70B
    LL = M.Language.Llama3_8B
    N = LL.parameters.magnitude
    NL, D, NH, NKV = LL.layers, LL.hidden_dim, LL.heads, LL.kv_heads
    DH = D // NH
    G2 = M.Language.GPT2.parameters.magnitude
    TR = SC.Gpt2XlRunOnOneH100
    USD_HR = TR.hourly_price.to("USD/hour").magnitude

    J = HW.Edge.JetsonOrinNano
    J_PEAK, J_BW = J.compute.peak_flops.to("flop/s").magnitude, J.memory.bandwidth.to("B/s").magnitude
    J_TDP, J_LAUNCH_MS = J.tdp.to("W").magnitude, J.dispatch_tax.to("ms").magnitude
    YOLO = M.Vision.YOLOv8_Nano
    Y_FLOP, Y_N = YOLO.inference_flops.to("flop").magnitude, YOLO.parameters.magnitude
    MCU = HW.Tiny.STM32F4
    MCU_PEAK, MCU_BW = MCU.compute.peak_flops.to("flop/s").magnitude, MCU.memory.bandwidth.to("B/s").magnitude
    MCU_SRAM, MCU_FLASH = MCU.memory.sram_capacity.to("KiB").magnitude, MCU.memory.flash_capacity.to("KiB").magnitude
    MCU_CLK = MCU.compute.clock_rate.to("MHz").magnitude
    KWS = M.Tiny.DS_CNN
    K_N, K_FLOP = KWS.parameters.magnitude, KWS.inference_flops.to("flop").magnitude
    T = HW.Tech
    E_DRAM_WORD = T.Memory.DRAM.energy_per_access.to("pJ").magnitude
    E_FP16 = T.Op.FlopFp16.energy.to("pJ/flop").magnitude
    E_FP32 = T.Op.FlopFp32.energy.to("pJ/flop").magnitude
    E_INT8 = T.Op.OpInt8.energy.to("pJ/flop").magnitude

    # ---- model anatomy
    est_layers = 12 * NL * D * D
    est_embed = 2 * EXTERNAL["llama3_vocab"] * D
    est_err = abs(est_layers + est_embed - N) / N * 100
    # ---- memory
    w = {"FP32": 4 * N, "FP16": 2 * N, "INT8": N, "INT4": 0.5 * N}
    kv_tok = 2 * NL * NKV * DH * 2
    kv_seq = kv_tok * S["ctx"]
    kv_mha = 2 * NL * NH * DH * 2
    bmax = math.floor((CAP - w["FP16"]) / kv_seq)
    act_layer = 34 * S["ctx"] * D
    # ---- time
    ridge = {k: PEAK[k] / BW for k in ("fp32", "tf32", "fp16", "fp8", "int8")}
    rows = []
    for b in list(S["batches"]) + [bmax]:
        byts = w["FP16"] + b * kv_seq
        t = byts / BW
        rows.append(dict(b=b, ms=t * 1e3, tps=b / t, I=2 * N * b / byts, usd=USD_HR / (b / t * 3600) * 1e6))
    prefill_flop = 2 * N * S["prompt"] + 4 * NL * S["prompt"] ** 2 * D
    prefill_ms = prefill_flop / (PEAK["fp16"] * S["eta_prefill"]) * 1e3
    train_s_per_flop = 1 / (PEAK["fp16"] * TR.efficiency)
    decode_s_per_flop = 1 / (BW * 1.0)       # batch 1, FP16 weights: 1 FLOP per byte
    coupling = decode_s_per_flop / train_s_per_flop
    amd = 1 / ((1 - S["amd_p"]) + S["amd_p"] / S["amd_a"])
    # ---- serving
    rho = S["lam"] * S["t_s"]
    W_ms = S["t_s"] / (1 - rho) * 1e3
    p99_ms = W_ms * math.log(100)
    p99_half = S["t_s"] / 0.5 * math.log(100) * 1e3
    inflight = S["lam"] * W_ms / 1e3
    # ---- edge, tiny, data
    frame_ms = 1e3 / S["fps"]
    yolo_ms = Y_FLOP / (J_PEAK * S["eta_edge"]) * 1e3
    yolo_w_ms = Y_N * 2 / J_BW * 1e3
    kws_ms = K_FLOP / MCU_PEAK * 1e3
    kws_simd = (K_FLOP / 2) / (S["mcu_macs_per_cycle"] * MCU_CLK * 1e6) * 1e3
    supply = S["ssd"] * (1 - S["f_loss"]) / S["sample"]
    # ---- shrinking
    hw_, k_, ci, co = S["conv"]
    conv_std, conv_dw = hw_ * hw_ * k_ * k_ * ci * co, hw_ * hw_ * ci * (k_ * k_ + co)
    lo, hi = S["q_range"]
    d8, d4 = (hi - lo) / 255, (hi - lo) / 15
    prune_idx = (1 - S["prune"]) * N * (2 + S["idx_bytes"])
    prune_mask = (1 - S["prune"]) * N * 2 + N / 8
    int4_cut = 1 - (w["INT4"] + bmax * kv_seq) / (w["FP16"] + bmax * kv_seq)
    sparse_ratio = H.compute.precision_flops["fp16_sparse"].magnitude / H.compute.peak_flops.magnitude
    e_byte = E_DRAM_WORD / 4
    w16 = w['FP16']
    usd_b1, usd_bmax = f"${rows[0]['usd']:.2f}", f"${rows[-1]['usd']:.2f}"
    mcu_w_kib = K_N / 1024                          # INT8: one byte per parameter
    flop_tok = 2 * N
    assert mcu_w_kib > MCU_SRAM, "MCU card says the INT8 weights exceed SRAM and run from flash"

    # ---- page 3: training on one node (Ch 8 GPT2SqrtCheckpoint, GPT2NodeAllReduce, GradientAccumulation)
    G2L = M.Language.GPT2.layers
    n_ckpt = round(math.sqrt(G2L))
    ckpt_layers = n_ckpt + G2L / n_ckpt                # layers' worth of activations kept
    ckpt_saved = 1 - ckpt_layers / G2L
    ckpt_extra = 1 / 3                                 # one extra forward over forward 1 + backward 2
    NODE = SYS.Nodes.DGX_H100
    NG = NODE.accelerators_per_node
    LINK = NODE.intra_node_bw.to("B/s").magnitude      # per accelerator, one direction
    ETA_T = LIT.Training.MfuHigh
    blk = SC.Gpt2XlBatch32Block
    SEQ, B_FULL = blk.context_tokens, blk.batch_size
    B_SPLIT = B_FULL // NG
    grad_b = 2 * G2                                    # FP16 gradients
    ring = 2 * (NG - 1) / NG
    t_ar = ring * grad_b / LINK
    t_cmp = lambda b: 6 * G2 * b * SEQ / (PEAK["fp16"] * ETA_T)
    share = lambda b: t_ar / (t_ar + t_cmp(b))
    state16 = 16 * N
    zero3 = state16 / NG
    zero3_traffic = 3 / 2                              # three parameter-sized passes instead of two
    mb = SC.Gpt2MixedPrecisionTwoMicroBatchesOnH100
    B_MICRO, MB_GB = mb.batch_size, mb.training_total.to("GB").magnitude
    k_acc = S["eff_batch"] // (NG * B_MICRO)
    assert MB_GB * 1e9 < CAP and k_acc * NG * B_MICRO == S["eff_batch"], "accumulation card: micro-batch fits, batch divides"
    SCL = SC.ImageClassifierEightGpuScaling
    t1_h, tn_h, n_scl = SCL.base_time.to("hour").magnitude, SCL.scaled_time.to("hour").magnitude, SCL.scaled_devices
    eff_scl = t1_h / (n_scl * tn_h)
    CS = SC.ResNet50TenPercentCoresetH100
    cs_full, cs_sub = CS.full_time.to("minute").magnitude, CS.subset_time.to("minute").magnitude
    cs_score = CS.score_time().to("minute").magnitude
    cs_total = cs_sub + cs_score
    # ---- page 3: operating it (Ch 7 compile break-even, Ch 15 carbon, Ch 14 A/B and retraining)
    CB = SC.ResNet50CompileBreakeven
    cb_te, cb_tc = CB.eager_item_time.to("ms").magnitude, CB.compiled_item_time.to("ms").magnitude
    cb_T = CB.compile_time.to("s").magnitude
    cb_items = math.ceil(CB.breakeven_items / 1e3) * 1e3        # round up: fewer items do not repay
    E_RUN = TR.energy.to("kWh").magnitude
    grid = {g: getattr(INF.Grids, g).carbon_intensity_kg_kwh for g in S["grids"]}
    co2 = {g: E_RUN * ci for g, ci in grid.items()}
    AB = SC.ConversionRelativeLiftExperiment
    ab_sig2 = AB.baseline_rate * (1 - AB.baseline_rate)
    ab_delta = AB.target_rate - AB.baseline_rate
    ab_n = math.ceil(AB.per_variant / 1e3) * 1e3                # round up: fewer users miss the lift
    ab_small = AB.detectable_lift()
    RT = SC.RankingRetrainingInterval
    rt_C, rt_Q, rt_V = RT.run_cost.to("USD").magnitude, RT.query_rate.to("1/day").magnitude, RT.value_per_query.to("USD").magnitude
    rt_A0, rt_g = RT.drift.initial_accuracy, RT.drift.decay_rate.to("1/day").magnitude
    rt_T = math.sqrt(2 * rt_C / (rt_Q * rt_V * rt_A0 * rt_g))
    rt_T4 = SC.RankingRetrainingIntervalFourfoldTraffic.optimal_interval.to("day").magnitude

    def gb(x, d=1):
        return f"{x / 1e9:,.{d}f} GB"

    # ---------------------------------------------------------------- charts
    rp = Plot((0.1, 1e4), (0.1, 4e3), xlog=True, ylog=True, W=360, H=132, right=20)
    rp.grid([0.1, 1, 10, 100, 1000, 1e4], [1, 10, 100, 1000])
    for key, cls, yoff in (("fp8", "roof alt", -4), ("fp16", "roof", 12)):
        pk = PEAK[key]
        rg = pk / BW
        rp.line([(0.1, BW * 0.1 / 1e12), (rg, pk / 1e12), (1e4, pk / 1e12)], cls)
        rp.text(rp.X(1e4) - 2, rp.Y(pk / 1e12) + yoff, f"{key.upper()} {fmt(pk / 1e12)}", "end")
    rg = ridge["fp16"]
    rp.parts.append(f'<line x1="{rp.X(rg):.1f}" x2="{rp.X(rg):.1f}" y1="{rp.Y(PEAK["fp16"] / 1e12):.1f}" y2="{rp.H - rp.b}" class="ridge"/>')
    rp.text(rp.X(rg) + 3, rp.H - rp.b - 5, f"ridge {fmt(rg)}")
    att = lambda i: min(PEAK["fp16"], BW * i) / 1e12
    rp.point(rows[0]["I"], att(rows[0]["I"]), f"decode B={rows[0]['b']}")
    rp.point(rows[2]["I"], att(rows[2]["I"]), f"B={rows[2]['b']}", "end", -5, -3)
    rp.point(4096 / 3, att(4096 / 3), "GEMM n=4096", "start", -28, 26)
    rp.text(rp.X(0.12), rp.Y(400), "bandwidth-bound", cls="lab ital")
    rp.axes("arithmetic intensity I (FLOP/byte)", "TFLOP/s")

    qp = Plot((0, 1), (0, 500), W=340, H=140, bottom=28)
    qp.grid([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 100, 200, 300, 400, 500])
    ts = S["t_s"] * 1e3
    for kmul, cls in ((1.0, "line muted"), (math.log(100), "line")):
        pts, r = [], 0.0
        while r < 0.999:
            ms = ts * kmul / (1 - r)
            pts.append((r, ms))
            if ms > 500:
                break
            r += 0.005
        qp.line(pts, cls)
    for r in (0.5, rho):
        p = ts * math.log(100) / (1 - r)
        qp.point(r, p, f"{fmt(p)} ms", "end", -5, -5)
    qp.text(qp.X(0.68), qp.Y(ts / (1 - 0.68)) - 5, "mean")
    qp.text(qp.X(0.2), qp.Y(ts * math.log(100) / 0.8) - 5, "p99")
    qp.axes(f"utilization ρ (service time {fmt(ts)} ms)", "ms")

    sp = Plot((1, 64), (0, 45), xlog=True, W=340, H=90, bottom=26)
    sp.grid(list(S["sync_batches"]), [0, 10, 20, 30, 40], xfmt=lambda b: f"{b:g}", yfmt=lambda y: f"{y:g}%")
    bs = [2 ** (i / 8) for i in range(49)]
    sp.line([(b, share(b) * 100) for b in bs])
    sp.point(B_SPLIT, share(B_SPLIT) * 100, f"B = {B_SPLIT}: {share(B_SPLIT) * 100:.0f}%", "start", 4, -6)
    sp.point(B_FULL, share(B_FULL) * 100, f"B = {B_FULL}: {share(B_FULL) * 100:.0f}%", "end", -4, -7)
    sp.axes(f"sequences per GPU per step (S = {fmt(SEQ)})", "sync share")

    anatomy = '''<svg viewBox="0 0 330 96" class="chart" role="img" aria-label="One transformer layer">
<defs><marker id="ar" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L6,3 L0,6 z" class="arrowhead"/></marker></defs>
<text x="4" y="42" class="lab">x</text><line x1="14" y1="39" x2="34" y2="39" class="flow" marker-end="url(#ar)"/>
<rect x="36" y="14" width="112" height="50" rx="3" class="blk"/>
<text x="92" y="30" class="lab b" text-anchor="middle">Attention</text>
<text x="92" y="44" class="mono" text-anchor="middle">Q K V O: 4 × d×d</text>
<text x="92" y="57" class="mono acc" text-anchor="middle">≈ 4 d² params</text>
<line x1="148" y1="39" x2="172" y2="39" class="flow" marker-end="url(#ar)"/>
<rect x="174" y="14" width="112" height="50" rx="3" class="blk"/>
<text x="230" y="30" class="lab b" text-anchor="middle">MLP (width 4d)</text>
<text x="230" y="44" class="mono" text-anchor="middle">d→4d→d</text>
<text x="230" y="57" class="mono acc" text-anchor="middle">≈ 8 d² params</text>
<line x1="286" y1="39" x2="306" y2="39" class="flow" marker-end="url(#ar)"/><text x="310" y="42" class="lab">x′</text>
<rect x="36" y="72" width="112" height="20" rx="3" class="kv"/>
<text x="92" y="85" class="mono" text-anchor="middle">KV cache: K, V kept</text>
<line x1="92" y1="64" x2="92" y2="71" class="flow" marker-end="url(#ar)"/></svg>'''

    # ---------------------------------------------------------------- page 1
    band = (f'<div class="eqbig">{v("T")} = {frac("samples", "task")} × {frac("FLOPs", "sample")} × {frac("seconds", "FLOP")}'
            f'<span class="lbl">Data × Algorithm × Machine</span></div>'
            f'<div class="txt"><div><b>The iron law (Ch 1).</b> {v("T")} = {v("D", "vol")}/{"BW"} + {v("O")}/({v("R", "peak")}·{v("η", "hw")}) + {v("L", "lat")}; per FLOP, 1/({v("I")}·{"BW"}) + 1/({v("R", "peak")}·{v("η", "hw")}) + {v("L", "lat")}/{v("O")}. '
            f'So seconds per FLOP depends on the workload. On one H100, batch-1 decode ({v("I")} ≈ 1): 1/(1 × {BW / 1e12:.2f} TB/s) = {decode_s_per_flop * 1e12:.2f} ps per FLOP; '
            f'training at {fmt(TR.efficiency * 100)}% of peak: 1/({TR.efficiency:g} × {fmt(PEAK["fp16"] / 1e12)} TFLOP/s) = {train_s_per_flop * 1e12:.4f} ps; ratio ≈ {n(fmt(round(coupling, -1)) + "×")}.</div>'
            f'<div><b>Levers:</b> fewer samples · fewer FLOPs or bytes per sample · fewer seconds per FLOP (batch; raise {"BW"}, {v("R", "peak")}, {v("η", "hw")}).</div></div>'
            f'<div class="side"><div><b>Amdahl (Ch 11)</b> &nbsp;speedup = 1 / ((1 − {v("f")}) + {v("f")} / {v("G")})</div>'
            f'<div>{A}{v("f")} = {S["amd_p"]:g} of the time sped up by {v("G")} = {fmt(S["amd_a"])}: 1/({1 - S["amd_p"]:.1f} + {S["amd_p"] / S["amd_a"]:.2f}) = {n(f"{amd:.1f}×")}; cap 1/(1 − {v("f")}) = {n(fmt(1 / (1 - S["amd_p"])) + "×")}</div></div>')

    p1 = Page(
        question="Will it run?",
        sub="Read left to right: what is in the model, whether it fits in memory, and how fast it goes on one accelerator.",
        meta="reference node: NVIDIA H100",
        notation=[
            ("Model", f'{v("P")} parameters · {v("N", "L")} layers · {v("d")} width · {v("H", "KV")} KV heads · {v("d", "head")} head dim'),
            ("Workload", f'{v("S")} tokens of context · {v("B")} batch · {v("D", "vol")} bytes moved · {v("O")} FLOPs · {A}= example input'),
            ("Machine", f'{v("R", "peak")} peak FLOP/s · {"BW"} bandwidth (bytes/s) · {v("η", "hw")} fraction of peak sustained · {v("I")} = {v("O")}/{v("D", "vol")} FLOP/byte'),
        ],
        know=[
            Know("2 / 16 bytes", "per parameter: FP16 weights / Adam training state"),
            Know("2P / 6P", "FLOPs per token: inference / training"),
            Know(f"{fmt(PEAK['fp16'] / 1e12)} TFLOP/s", "H100 FP16 dense peak"),
            Know(f"{BW / 1e12:.2f} TB/s", "H100 HBM bandwidth"),
            Know(f"{fmt(ridge['fp16'])} FLOP/B", "H100 ridge point: below it, bandwidth-bound"),
            Know(f"{fmt(int(E_DRAM_WORD / E_FP16))}×", "energy: 32-bit DRAM read vs FP16 multiply (45 nm)"),
            Know("4.6 = ln 100", "p99 ÷ mean time in an M/M/1 queue"),
        ],
        band=band,
        cols=[
            Col("1 What is in the model?", [
                Card("Parameters from the architecture", f'{v("P")} ≈ 12 · {v("N", "L")} · {v("d", sup="2")} + 2 · vocab · {v("d")}',
                     ex=f'Llama 3 8B, vocab {fmt(EXTERNAL["llama3_vocab"])}: {est_layers / 1e9:.2f}B + {est_embed / 1e9:.2f}B = '
                        f'{n(f"{(est_layers + est_embed) / 1e9:.2f}B")} vs {N / 1e9:.2f}B actual. Within {fmt(est_err)}%: shared KV heads and a wider gated MLP shift the 12.'),
                Card(f'Weights: {v("M", "w")} = {v("P")} × bytes per parameter',
                     holds="FP32 4 · FP16 2 · INT8 1 · INT4 ½ bytes. Serving adds the KV cache; training adds 16P state and activations (column 2)."),
                Table(["Reference model", f'{v("P")}', f'{v("N", "L")}', f'{v("d")}', f'{v("H", "KV")}', f'{v("d", "head")}', "FP16 weights"],
                      [[m.name.replace("Llama-3.1-", "Llama 3 "), f"{m.parameters.magnitude / 1e9:.1f}B", str(m.layers), fmt(m.hidden_dim), str(m.kv_heads), str(m.hidden_dim // m.heads), gb(2 * m.parameters.magnitude)] for m in (LL, L70)]),
                Card("FLOPs per token", [f'forward ≈ 2{v("P")} &nbsp;·&nbsp; training ≈ 6{v("P")} = 2{v("P")} fwd + 4{v("P")} bwd',
                                         f'attention adds {v("O", "attn")} ≈ 4 · {v("N", "L")} · {v("S", sup="2")} · {v("d")} per sequence'],
                     holds=f'One multiply-add (2 FLOPs) per parameter forward; backward ≈ 2× forward. Attention is small until {v("S")} nears {v("P")} / (2{v("N", "L")}{v("d")}); causal masking halves it.',
                     ex=f'Llama 3 8B: 2 × {N / 1e9:.2f}B = {n(f"{flop_tok / 1e9:.1f} GFLOP")} per token forward, {3 * flop_tok / 1e9:.0f} GFLOP training.'),
            ], "Ch 6, App. Algorithm"),
            Col("2 Will it fit?", [
                Card("KV cache per sequence (inference)", f'{v("M", "KV")}({v("S")}) = 2 · {v("N", "L")} · {v("H", "KV")} · {v("d", "head")} · bytes · {v("S")}',
                     ex=f'Llama 3 8B FP16 ({v("H", "KV")} = {NKV}, {v("d", "head")} = {DH}): {n(fmt(kv_tok / 1024) + " KiB per token")}; {A}{v("S")} = {fmt(S["ctx"])} → '
                        f'{n(fmt(kv_seq / 2**20) + " MiB per sequence")}. All {NH} heads would need {fmt(kv_mha / 1024)} KiB.'),
                Card("Decode: every step reads the weights and the KV cache",
                     f'{v("T", "step")} ≈ {frac(v("M", "w") + " + " + v("B") + " · " + v("M", "KV") + "(" + v("S") + ")", "BW")}, &nbsp; tokens/s = {v("B")} / {v("T", "step")}',
                     ex=f'{v("B")} = 1: ({w16 / 1e9:.1f} + {kv_seq / 1e9:.2f}) GB ÷ {BW / 1e12:.2f} TB/s = {n(f"{rows[0]['ms']:.1f} ms")} → {n(fmt(rows[0]["tps"]) + " tokens/s")}; {v("I")} = 2{v("P")}·{v("B")} ÷ bytes ≈ {fmt(rows[0]["I"])} FLOP/byte.',
                     so=f'Decode is bandwidth-bound: bytes, not FLOPs, set the time per token.'),
                Table(["batch <i>B</i>", "ms/step", "tokens/s", "<i>I</i>", "$ / M tok"],
                      [[fmt(r["b"]), f"{r['ms']:.1f}", fmt(r["tps"]), fmt(r["I"]), f"${r['usd']:.2f}"] for r in rows],
                      caption=f'Llama 3 8B FP16, {v("S")} = {fmt(S["ctx"])}, ${USD_HR:.2f}/GPU-hour. {v("I")} stays below the ridge: batching buys throughput, not speed per token.'),
            ], f"Ch 8, 10, 13 · H100 {fmt(CAP_GIB)} GiB = {CAP / 1e9:.1f} GB"),
            Col("3 How fast?", [
                Card("Largest batch that fits", f'{v("B", "max")} = {frac("capacity − " + v("M", "w") + " − overhead", v("M", "KV") + "(" + v("S") + ")")}',
                     ex=f'({CAP / 1e9:.1f} GB − {gb(w["FP16"])}) ÷ {kv_seq / 1e9:.3f} GB = {n(fmt(bmax) + " sequences")} at {v("S")} = {fmt(S["ctx"])}, an upper bound before activations. Decode in column 2 ←'),
                Chart(rp.svg("H100 roofline")),
                Card(f'Roofline: attainable = min({v("R", "peak")}, {"BW"} · {v("I")}), ridge = {v("R", "peak")} / {"BW"}',
                     holds=f'An ({v("M")}×{v("K")})·({v("K")}×{v("N")}) matrix multiply does 2{v("MNK")} FLOPs; {v("I")} = 2{v("MNK")} ÷ bytes moved. Square FP16, {v("n")} = 4096: {v("n")}/3 = {fmt(4096 / 3)} &gt; {fmt(ridge["fp16"])}, compute-bound.',
                     so=f'Above the ridge ({fmt(ridge["fp16"])} FLOP/byte) the math units set the speed; below it, memory bandwidth does.'),
                Table(["H100 dense", "FP32", "TF32", "FP16", "FP8", "INT8"], [
                    ["peak (TFLOP/s, TOPS)", fmt(PEAK["fp32"] / 1e12), f"{PEAK['tf32'] / 1e12:.1f}", fmt(PEAK["fp16"] / 1e12), fmt(PEAK["fp8"] / 1e12), fmt(PEAK["int8"] / 1e12)],
                    ["ridge (FLOP/B)"] + [fmt(ridge[k]) for k in ("fp32", "tf32", "fp16", "fp8", "int8")]]),
            ], f"Ch 8, 11, 13 · H100 {BW / 1e12:.2f} TB/s"),
        ],
    )

    # ---------------------------------------------------------------- page 2
    p2 = Page(
        question="Will it meet its target?",
        sub="Serving under load, deadlines on edge and tiny devices, and what shrinking the model buys.",
        meta="H100 · Jetson Orin Nano · STM32F4",
        notation=[
            ("Queues", f'{v("D", "vol")}, {v("O")}, {v("I")}, {v("L", "lat")} as on page 1 · {v("λ")} arrivals/s · {v("t", "s")} service time · {v("ρ")} = {v("λ")}{v("t", "s")} busy fraction · {v("T", "lat")} time in system · {v("Q", "req")} requests in system'),
            ("Shrinking", f'{v("b")} bits per value · {v("s")} quantization step · {v("σ")} fraction of weights pruned'),
            ("Not the same “utilization”", f'{v("η", "hw")} (MFU) = useful FLOP/s ÷ peak · {v("ρ")} = fraction of time the server is busy · nvidia-smi “GPU-Util” = time any kernel was running'),
        ],
        cols=[
            Col("4 Can it serve the load?", [
                Card("Prefill (time to first token): compute-bound",
                     f'{v("T", "prefill")} ≈ (2{v("P")} · {v("S")} + 4{v("N", "L")}{v("S", sup="2")}{v("d")}) / ({v("R", "peak")} · {v("η", "hw")})',
                     ex=f'{fmt(S["prompt"])}-token prompt: {prefill_flop / 1e12:.1f} TFLOP at {A}{v("η", "hw")} = {fmt(S["eta_prefill"] * 100)}% → {n("≈ " + fmt(prefill_ms) + " ms")} (no causal saving).'),
                Card("Queueing (M/M/1: random arrivals, one server)",
                     f'{v("ρ")} = {v("λ")} · {v("t", "s")}, &nbsp; {v("T", "lat")} = {frac(v("t", "s"), "1 − " + v("ρ"))}, &nbsp; p99 ≈ {frac("4.6 · " + v("t", "s"), "1 − " + v("ρ"))}',
                     holds=f'4.6 = ln 100. Latency explodes as {v("ρ")} → 1; keep headroom for a tail target.'),
                Chart(qp.svg("Latency versus utilization")),
                f'<p class="ex">{A}{v("λ")} = {fmt(S["lam"])}/s, {v("t", "s")} = {fmt(ts)} ms → {v("ρ")} = {fmt(S["lam"])} × {S["t_s"]:g} s = {rho:.1f}: {v("T", "lat")} = {fmt(ts)} ms ÷ (1 − {rho:.1f}) = {n(fmt(W_ms) + " ms")}; '
                f'p99 ≈ 4.6 × {fmt(W_ms)} ms = {n(fmt(p99_ms) + " ms")}. At {v("ρ")} = 0.5: 4.6 × {fmt(ts)} ÷ 0.5 = {fmt(p99_half)} ms.</p>',
                f'<p class="so"><b>So</b> Holding {v("ρ")} at 0.5 instead of {rho:.1f} cuts the p99 {p99_ms / p99_half:.1f}×.</p>',
                Card("Little's law", f'{v("Q", "req")} = {v("λ")} · {v("T", "lat")}',
                     ex=f'{fmt(S["lam"])}/s × {fmt(W_ms)} ms = {n(fmt(inflight) + " requests")} in the system on average, {rho:.1f} in service and the rest waiting. '
                        f'Size queues with {v("Q", "req")}; a batching server holds a KV cache per running request.'),
                Note("Before you trust the number:", f'every term in seconds or bytes? GB vs GiB? Compute-bound only if {v("I")} &gt; ridge? Memory counted with KV cache and activations? {v("ρ")} below about 0.7 for a tail target?'),
            ], "Ch 13, 11"),
            Col("5 Does it meet the deadline?", [
                Card("Real-time frame budget", f'{v("T", "frame")} = 1 / fps &nbsp; ≥ &nbsp; Σ stage times (worst case)',
                     ex=f'{fmt(S["fps"])} fps → {n(f"{frame_ms:.1f} ms")} for capture, preprocessing, inference, and post-processing together.'),
                Card("Inference on an edge GPU", f'{v("T")} ≈ max( {frac(v("O"), v("R", "peak") + " · " + v("η", "hw"))}, {frac("weights + activations", "BW")} ) + {v("L", "lat")}',
                     ex=f'YOLOv8-Nano ({Y_FLOP / 1e9:.1f} GFLOP, {Y_N / 1e6:.1f}M params) on Jetson Orin Nano ({fmt(J_PEAK / 1e12)} TFLOP/s FP16-equiv., {fmt(J_BW / 1e9)} GB/s) at {A}{v("η", "hw")} = {fmt(S["eta_edge"] * 100)}%: '
                        f'compute {n(f"{yolo_ms:.1f} ms")}; FP16 weights alone read in {yolo_w_ms:.2f} ms. Launches cost {J_LAUNCH_MS:g} ms each: run one compiled engine. '
                        f'Energy {v("E")} = power · {v("T")} ≤ {fmt(J_TDP)} W × {yolo_ms:.1f} ms ≈ {n(fmt(J_TDP * yolo_ms) + " mJ")} per frame.'),
                Card("Will it fit on a microcontroller?", ["flash ≥ weights + code", "SRAM ≥ largest layer's (input + output) + runtime"],
                     ex=f'DS-CNN keyword spotter ({fmt(K_N / 1e3)}K params, INT8) on STM32F4: {n(fmt(mcu_w_kib) + " KiB")} of weights in {fmt(MCU_FLASH)} KiB flash; '
                        f'too big for the {fmt(MCU_SRAM)} KiB SRAM, which holds only activations and runtime.'),
                Card("Microcontroller time", f'{v("T")} ≈ {v("O")} / (FLOP per cycle × clock)',
                     ex=f'{fmt(K_FLOP / 1e6)} MFLOP at 1 FLOP/cycle ({fmt(MCU_CLK)} MHz) ≈ {n(fmt(kws_ms) + " ms")}; INT8 SIMD kernels at {A}{S["mcu_macs_per_cycle"]} MACs/cycle ≈ {n(fmt(kws_simd) + " ms")}.'),
                Card("Can the data pipeline keep up?", f'samples/s ≤ min(accel., CPU, {frac("BW<sub>disk</sub>" + " (1 − loss)", "bytes/sample")})',
                     ex=f'{A}{fmt(S["ssd"] / 1e6)} MB/s SSD, {fmt(S["f_loss"] * 100)}% loss, {fmt(S["sample"] / 1e3)} KB images → disk supplies {n(fmt(supply) + " samples/s")}. CPU = workers × decode rate per worker. '
                        f'Loading {gb(w16)} of weights over PCIe ({fmt(PCIE / 1e9)} GB/s) takes ≥ {w16 / PCIE:.2f} s.'),
            ], "Ch 2, 4, 13"),
            Col("6 What does shrinking buy?", [
                Card("Convolution, standard vs depthwise-separable",
                     [f'standard: MACs = {v("H", "o")}{v("W", "o")} · {v("K", sup="2")} · {v("C", "in")} · {v("C", "out")}',
                      f'separable: MACs = {v("H", "o")}{v("W", "o")} · {v("C", "in")} · ({v("K", sup="2")} + {v("C", "out")})'],
                     holds=f'Per image, per layer; {v("H", "o")}×{v("W", "o")} output size. Savings ≈ 1 / (1/{v("C", "out")} + 1/{v("K", sup="2")}).',
                     ex=f'{hw_}×{hw_}, {v("K")} = {k_}, {ci}→{co} channels: {fmt(conv_std / 1e6)}M → {fmt(conv_dw / 1e6)}M MACs, {n(f"{conv_std / conv_dw:.1f}× fewer")}.'),
                Card("Quantization", f'bytes × {frac(v("b"), "16")} from FP16, &nbsp; {v("s")} = {frac("max − min", "2" + "<sup>" + v("b") + "</sup> − 1")}',
                     ex=f'Only quantized tensors shrink: INT4 weights cut decode bytes at {v("B", "max")} by just {fmt(int4_cut * 100)}%, since the FP16 KV cache dominates. '
                        f'Range [{lo:g}, {hi:g}]: {v("s")} = {n(f"{d8:.3f}")} at INT8, {n(f"{d4:.2f}")} at INT4. Error per value ≤ {v("s")}/2.'),
                Card("Pruning", [f'with index: (1 − {v("σ")}) · {v("P")} · (value + index) bytes', f'with mask: (1 − {v("σ")}) · {v("P")} · value + {v("P")}/8 bytes'],
                     ex=f'{A}{v("σ")} = {fmt(S["prune"] * 100)}%, FP16: a {S["idx_bytes"]}-byte relative index per value gives {n(gb(prune_idx), True)}, no saving; a 1-bit mask gives {gb(prune_mask)}. '
                        f'2:4 sparsity doubles H100 peak math ({fmt(sparse_ratio)}×); bandwidth-bound decode gains only the fewer bytes.'),
                Card("Movement costs more than math", f'{v("E")} ≈ {v("D", "vol")} · {v("e", "byte")} + {v("O")} · {v("e", "op")}', chapter="Ch 1"),
                Table(["45 nm (Horowitz)", "INT8 mult", "FP16 mult", "FP32 mult", "DRAM, per byte"],
                      [["pJ", f"{E_INT8:g}", f"{E_FP16:g}", f"{E_FP32:g}", fmt(e_byte)]]),
                f'<p class="holds">A DRAM byte ≈ {fmt(e_byte / E_FP16)}× an FP16 multiply. At {v("I")} ≈ 1 (batch-1 decode), energy is almost all movement.</p>',
            ], "Ch 6, 10, 11"),
        ],
        bottom=Table(["Reference node", "peak", "memory bandwidth", "memory", "ridge (FLOP/B)", "launch", "power"], [
            ["NVIDIA H100 (cloud)", f"{fmt(PEAK['fp16'] / 1e12)} TFLOP/s FP16", f"{BW / 1e12:.2f} TB/s HBM", f"{fmt(CAP_GIB)} GiB", fmt(ridge["fp16"]), f"{fmt(LAUNCH_US)} µs", f"{fmt(TDP)} W"],
            ["Jetson Orin Nano (edge)", f"{fmt(J_PEAK / 1e12)} TFLOP/s FP16-equiv.", f"{fmt(J_BW / 1e9)} GB/s", f"{fmt(J.memory.capacity.to('GiB').magnitude)} GiB", fmt(J_PEAK / J_BW), f"{J_LAUNCH_MS:g} ms", f"{fmt(J_TDP)} W"],
            ["STM32F4, Cortex-M4F (tiny)", f"{fmt(MCU_PEAK / 1e6)} MFLOP/s FP32", f"{fmt(MCU_BW / 1e6)} MB/s", f"{fmt(MCU_SRAM)} KiB SRAM, {fmt(MCU_FLASH)} KiB flash", f"{MCU_PEAK / MCU_BW:.2f}", "—", "—"],
        ]),
    )

    def sci(x, d=1):
        m, e = f"{x:.{d}e}".split("e")
        return f"{m} × 10<sup>{int(e)}</sup>"

    p3 = Page(
        question="What does it cost?",
        sub="Read left to right: whether training fits and finishes on one accelerator, how to spread it across the node's eight, and what a model costs once it is in service.",
        meta="reference node: NVIDIA H100 · DGX H100 (8 GPUs)",
        notation=[
            ("Training", f'{v("P")} parameters · {v("D")} training tokens · {v("N", "L")} layers · {v("B")} batch · {v("S")} sequence length · {v("d")} width · {v("η", "hw")} fraction of peak sustained'),
            ("Node", f'{v("N", "GPU")} accelerators in the node · {v("D", "vol")} bytes moved · BW link bandwidth per GPU, one direction · {v("T")} time, subscript says which'),
        ],
        cols=[
            Col("7 Will training fit and finish?", [
                Card("Training state (mixed-precision Adam)", f'{v("M", "train")} ≈ (2 + 2 + 4 + 8) {v("P")} = 16{v("P")} bytes',
                     ex=f'FP16 weights + gradients, FP32 master copy + two Adam moments. Llama 3 8B: 16 × {N / 1e9:.2f}B = {n(gb(state16, 0), True)} &gt; {CAP / 1e9:.1f} GB before any activation.',
                     so='Fully fine-tuning 8B does not fit one H100; column 8 shards it.'),
                Card("Activations (training)", f'{v("M", "act")} ≈ 34 · {v("B")} · {v("S")} · {v("d")} bytes per layer',
                     ex=f'Ch 8\'s GPT-style constant, an estimate for Llama 3 8B, {v("B")} = 1, {v("S")} = {fmt(S["ctx"])}: 34 × {fmt(S["ctx"])} × {fmt(D)} = {fmt(act_layer / 2**20)} MiB × {NL} layers = {n(f"{act_layer * NL / 1e9:.1f} GB")}.'),
                Card("Recompute activations (checkpointing)", [f'keep {v("n", "ckpt")} = √{v("N", "L")} segment inputs', f'memory ≈ ({v("n", "ckpt")} + {v("N", "L")} / {v("n", "ckpt")}) layers of activations'],
                     ex=f'GPT-2 XL, {v("N", "L")} = {G2L}: √{G2L} ≈ {n_ckpt}, so {n_ckpt} + {G2L}/{n_ckpt} = {ckpt_layers:.1f} layers instead of {G2L}: '
                        f'{n(f"{ckpt_saved * 100:.0f}% less")} activation memory for one extra forward pass, about {n(f"{ckpt_extra * 100:.0f}% more")} compute (forward 1 + backward 2).'),
                Card("Training time", f'{v("T", "train")} ≈ 6 · {v("P")} · {v("D")} / ({v("R", "peak")} · {v("η", "hw")})',
                     ex=f'The Ch 8 run, GPT-2 XL: 6 × {G2 / 1e9:.1f}B × {fmt(TR.training_tokens.magnitude / 1e9)}B tokens = {sci(6 * G2 * TR.training_tokens.magnitude)} FLOPs ÷ '
                        f'({fmt(PEAK["fp16"] / 1e12)} TFLOP/s × {TR.efficiency:g}) = {n("≈ " + fmt(TR.training_time.to("day").magnitude) + " days")} on one H100; '
                        f'{TR.device_hours.to("hour").magnitude:,.0f} h × {TDP / 1000:.1f} kW = {fmt(E_RUN)} kWh; × ${USD_HR:.2f}/h = ${fmt(TR.rental_cost.to("USD").magnitude)}.',
                     so=f'About {TR.training_time.to("day").magnitude / 7:.0f} weeks on one GPU: the node\'s other {NG - 1} are the next lever.'),
                Card("Does data selection pay?", f'{v("T", "selection")} + {v("T", "train")}({v("D", "subset")}) &lt; {v("T", "train")}({v("D", "total")})', chapter="Ch 9",
                     ex=f'ResNet-50, ImageNet, {CS.epochs} epochs on one H100, {A}{CS.keep_fraction * 100:.0f}% coreset: {cs_score:.1f} min scoring + {cs_sub:.1f} min training = '
                        f'{cs_total:.1f} min vs {fmt(cs_full)} min, {n(f"{math.floor((1 - cs_total / cs_full) * 100)}% saved")}.',
                     so='At matched quality only; rescoring every epoch can erase the saving.'),
            ], "Ch 8, 9"),
            Col("8 Can one machine use all its accelerators?", [
                Card("Ring AllReduce: synchronizing gradients", f'{v("T", "comm")} ≈ {frac("2(" + v("N", "GPU") + " − 1)", v("N", "GPU"))} · {frac(v("D", "vol"), "BW")}',
                     ex=f'GPT-2 XL FP16 gradients, {v("D", "vol")} = {grad_b / 1e9:.1f} GB, {v("N", "GPU")} = {NG}: {ring:.2f} × {grad_b / 1e9:.1f} GB ÷ {fmt(LINK / 1e9)} GB/s NVLink = {n(f"{t_ar * 1e3:.0f} ms")}. '
                        f'Compute per GPU, 6{v("P")}·{v("B")}·{v("S")} ÷ ({fmt(PEAK["fp16"] / 1e12)} TFLOP/s × {ETA_T:g}): {v("B")} = {B_FULL} → {t_cmp(B_FULL) * 1e3:,.0f} ms, sync {n(f"{share(B_FULL) * 100:.0f}%")} of the step; '
                        f'{B_FULL} split {NG} ways ({v("B")} = {B_SPLIT}) → {t_cmp(B_SPLIT) * 1e3:,.0f} ms, sync {n(f"{share(B_SPLIT) * 100:.0f}%")}.'),
                Chart(sp.svg("Share of the step spent synchronizing versus batch per GPU"),
                      caption=f'{v("T", "comm")} depends only on {v("P")}; compute shrinks with the batch per GPU.'),
                Card("Shard the training state (ZeRO-3)", f'{frac("16" + v("P"), v("N", "GPU"))} bytes per GPU, &nbsp; {zero3_traffic:g}× the AllReduce bytes',
                     ex=f'Llama 3 8B from column 7: {gb(state16, 0)} ÷ {NG} = {n(gb(zero3))} per GPU, under {CAP / 1e9:.1f} GB with {math.floor((CAP - zero3) / 1e9)} GB left for activations.'),
                Card("Gradient accumulation", f'{v("B")} = {v("N", "GPU")} · {v("k")} · {v("B", "micro")}',
                     ex=f'GPT-2 XL: {v("B", "micro")} = {B_MICRO} fits ({MB_GB:.1f} GB). {A}{v("B")} = {S["eff_batch"]} on {NG} GPUs: '
                        f'{v("k")} = {S["eff_batch"]} ÷ ({NG} × {B_MICRO}) = {n(str(k_acc))} micro-batches per update.',
                     so='Big-batch gradient at micro-batch memory; one AllReduce per update.'),
                Card("Scaling efficiency (strong scaling)", f'Eff({v("N", "GPU")}) = {frac(v("T") + "(1)", v("N", "GPU") + " · " + v("T") + "(" + v("N", "GPU") + ")")}', chapter="Ch 12",
                     ex=f'{A}{fmt(t1_h)} h on 1 GPU, {fmt(tn_h)} h on {n_scl}: {fmt(t1_h)} ÷ ({n_scl} × {fmt(tn_h)}) = {n(f"{eff_scl * 100:.0f}%")}; ideal is {t1_h / n_scl:g} h.',
                     so=f'Profile the missing {(1 - eff_scl) * 100:.0f}%: exposed sync, copies, imbalance.'),
            ], "Ch 8, 12"),
            Col("9 What does running it cost?", [
                Card("Cost per token", f'$ per token = {frac("$ per hour", "tokens/s × 3600")}',
                     ex=f'${USD_HR:.2f} ÷ ({fmt(rows[0]["tps"])} tokens/s × 3,600) × 10<sup>6</sup> = {n(usd_b1)} per million tokens at {v("B")} = 1; '
                        f'at {v("B", "max")} ({fmt(rows[-1]["tps"])} tokens/s): {n(usd_bmax)}.',
                     so=f'Batching is the lever: one GPU-hour, {rows[-1]["tps"] / rows[0]["tps"]:.0f}× the tokens.'),
                Card("When does compiling pay?", f'break-even items = {frac(v("T", "compile"), v("t", "eager") + " − " + v("t", "compiled"))}', chapter="Ch 7",
                     ex=f'ResNet-50, {A}{fmt(CB.eager_rate.magnitude)} → {fmt(CB.compiled_rate.magnitude)} images/s, {fmt(cb_T)} s compile: {fmt(cb_T)} s ÷ ({cb_te:.3f} − {cb_tc:.3f}) ms = {n("≈ " + fmt(cb_items) + " images")}.',
                     so='Compile for serving, not for code that recompiles often.'),
                Card("Carbon", f'CO₂ = energy × grid intensity', chapter="Ch 15",
                     ex=f'Column 7\'s GPT-2 XL run, {fmt(E_RUN)} kWh at GPU TDP, × {grid["Quebec"]:g} / {grid["US_Avg"]:g} / {grid["Poland"]:g} kg/kWh (Quebec / US / Poland) = '
                        f'{n(fmt(co2["Quebec"]))} / {n(fmt(co2["US_Avg"]))} / {n(fmt(co2["Poland"]) + " kg")}.',
                     so=f'Same run, {co2["Poland"] / co2["Quebec"]:.0f}× apart: the grid is a lever.'),
                Card("A/B test size", f'{v("n")} = {frac("2(" + v("z", "α/2") + " + " + v("z", "β") + ")<sup>2</sup> · " + v("σ", sup="2"), "(" + v("δ", "MDE") + ")<sup>2</sup>")} users per variant', chapter="Ch 14",
                     ex=f'{A}{AB.baseline_rate * 100:.0f}% conversion, {AB.relative_lift * 100:.0f}% relative lift: {v("δ", "MDE")} = {ab_delta:.3f}, {v("σ", sup="2")} = {AB.baseline_rate:g} × {1 - AB.baseline_rate:g}; '
                        f'{AB.confidence * 100:.0f}% confidence, {AB.power * 100:.0f}% power: 2 × ({AB.z_alpha:.3f} + {AB.z_beta:.3f})² × {ab_sig2:.4f} ÷ {ab_delta:.3f}² = {n("≈ " + fmt(ab_n))}.',
                     so=f'Halving {v("δ", "MDE")} takes 4× the users; {fmt(AB.available_per_variant)} see only ≥ {ab_small:.4f}.'),
                Card("How often to retrain?", f'{v("T")}* ≈ √( 2{v("C")} ÷ ({v("Q", "day")} · {v("V")} · Acc<sub>0</sub> · {v("γ")}) )', chapter="Ch 14",
                     ex=f'{A}{v("C")} = ${fmt(rt_C)}/run, {v("Q", "day")} = {fmt(rt_Q)}/day, {v("V")} = ${rt_V:.2f} per unit accuracy, Acc<sub>0</sub> = {rt_A0:g}, decay {v("γ")} = {rt_g:.5f}/day: '
                        f'√({fmt(2 * rt_C)} ÷ {fmt(rt_Q * rt_V * rt_A0 * rt_g)}) = {n(f"{rt_T:.1f} days")}; 4× traffic → {rt_T4:.1f}.',
                     so='Square roots damp error: rough inputs still give the right cadence.'),
            ], "Ch 7, 14, 15"),
        ],
    )
    return Sheet(volume=1, book="Introduction to Machine Learning Systems", scope="One Node", pages=[p1, p2, p3],
                 sources=f'GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B · 1 MAC = 2 FLOPs · {TAG}= input chosen for the example · values from the mlsysim registry; Llama 3 vocabulary from its model card.')

"""Napkin Math, Volume I: one node (one machine, one to eight accelerators).

Every number below is computed here from the mlsysim registry or scenarios.
Inputs chosen only for an example are tagged with assumed() on the sheet and
listed in SCENARIO. Constants from outside the registry are listed in EXTERNAL
with their source.
"""
from __future__ import annotations

import math

from mlsysim import Hardware as HW, Models as M
from mlsysim.engine.scenarios import Scenarios as SC

from .core import (Card, Chart, Col, Know, Note, Page, Plot, Sheet, Table,
                   assumed, fmt, frac, n, v)

EXTERNAL = {"llama3_vocab": 128256}  # Llama 3 tokenizer vocabulary (Meta Llama 3 model card)
SCENARIO = dict(ctx=2048, prompt=2048, eta_prefill=0.50, batches=(1, 32, 128), t_s=0.010, lam=80,
                mcu_macs_per_cycle=2, fps=30, eta_edge=0.30, conv=(56, 3, 128, 128), q_range=(-6.0, 6.0),
                prune=0.5, idx_bytes=2, amd_p=0.9, amd_a=10, ssd=550e6, f_loss=0.10, sample=110e3)


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
    usd_b1, usd_bmax = f"${rows[0]['usd']:.2f}", f"${rows[-1]['usd']:.2f}"

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
    rp.point(rows[0]["I"], att(rows[0]["I"]), f"decode b={rows[0]['b']}")
    rp.point(rows[2]["I"], att(rows[2]["I"]), f"b={rows[2]['b']}", "end", -5, -3)
    rp.point(4096 / 3, att(4096 / 3), "GEMM n=4096", "start", -28, 26)
    rp.text(rp.X(0.12), rp.Y(400), "bandwidth-bound", cls="lab ital")
    rp.axes("arithmetic intensity I (FLOP/byte)", "TFLOP/s")

    qp = Plot((0, 1), (0, 500), W=340, H=140, bottom=28)
    qp.grid([0, 0.2, 0.4, 0.6, 0.8, 1.0], [0, 100, 200, 300, 400, 500])
    ts = S["t_s"] * 1e3
    for kmul, cls in ((1.0, "line muted"), (math.log(100), "line warn")):
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
    band = (f'<div class="eqbig">{v("t")} = {frac("samples", "task")} × {frac("FLOPs", "sample")} × {frac("seconds", "FLOP")}'
            f'<span class="lbl">Data × Algorithm × Machine</span></div>'
            f'<div class="txt"><div><b>The iron law (Ch 1).</b> {v("t")} = {v("D")}/{v("B")} + {v("O")}/({v("R")}·{v("η")}) + {v("L")}; per FLOP, 1/({v("I")}·{v("B")}) + 1/({v("R")}·{v("η")}) + {v("L")}/{v("O")}. '
            f'So seconds per FLOP depends on the workload: bandwidth-bound batch-1 decode on one H100 takes ≈ {n(fmt(round(coupling, -1)) + "×")} that of training at {fmt(TR.efficiency * 100)}% MFU. '
            f'{v("L")} is fixed overhead; a kernel launch adds ≈ {fmt(LAUNCH_US)} µs.</div>'
            f'<div><b>Levers:</b> fewer samples · fewer FLOPs or bytes per sample · fewer seconds per FLOP (batch; raise {v("B")}, {v("R")}, {v("η")}).</div></div>'
            f'<div class="side"><div><b>Amdahl (Ch 11)</b> &nbsp;speedup = 1 / ((1 − {v("p")}) + {v("p")} / {v("a")})</div>'
            f'<div>{A}{v("p")} = {fmt(S["amd_p"] * 100)}% sped up {v("a")} = {fmt(S["amd_a"])}×: {n(f"{amd:.1f}×")}, cap {n(fmt(1 / (1 - S["amd_p"])) + "×")}</div></div>')

    p1 = Page(
        question="Will it run?",
        sub="Read left to right: what is in the model, whether it fits in memory, and how fast it goes on one accelerator.",
        meta="reference node: NVIDIA H100",
        notation=[
            ("Model", f'{v("N")} parameters · {v("ℓ")} layers · {v("d")} width · {v("h", "kv")} KV heads · {v("d", "h")} head dim'),
            ("Workload", f'{v("S")} tokens of context · {v("b")} batch · {v("D")} bytes moved · {v("O")} FLOPs · {A}= example input'),
            ("Machine", f'{v("R")} peak FLOP/s · {v("B")} bandwidth (bytes/s) · {v("η")} fraction of peak sustained · {v("I")} = {v("O")}/{v("D")} FLOP/byte'),
        ],
        know=[
            Know("2 B / 16 B", "per parameter: FP16 weights / Adam training state"),
            Know("2N / 6N", "FLOPs per token: inference / training"),
            Know(f"{fmt(PEAK['fp16'] / 1e12)} TFLOP/s", "H100 FP16 dense peak"),
            Know(f"{BW / 1e12:.2f} TB/s", "H100 HBM bandwidth"),
            Know(f"{fmt(ridge['fp16'])} FLOP/B", "H100 ridge point: below it, bandwidth-bound"),
            Know(f"{fmt(e_byte / E_FP16)}×", "energy of a DRAM byte vs an FP16 multiply"),
            Know("4.6", "p99 ÷ mean in an M/M/1 queue (ln 100)"),
        ],
        band=band,
        cols=[
            Col("1 What is in the model?", [
                Chart(anatomy),
                Card("Parameters from the architecture", f'{v("N")} ≈ 12 · {v("ℓ")} · {v("d", sup="2")} + 2 · vocab · {v("d")}',
                     ex=f'Llama 3 8B ({v("ℓ")} = {NL}, {v("d")} = {fmt(D)}, vocab {fmt(EXTERNAL["llama3_vocab"])}): {est_layers / 1e9:.2f}B + {est_embed / 1e9:.2f}B = '
                        f'{n(f"{(est_layers + est_embed) / 1e9:.2f}B")} vs {N / 1e9:.2f}B actual. Within {fmt(est_err)}%: shared KV heads and a wider gated MLP shift the 12.'),
                Card("FLOPs per token", f'forward ≈ 2{v("N")} &nbsp;·&nbsp; training ≈ 6{v("N")} = 2{v("N")} fwd + 4{v("N")} bwd',
                     holds="One multiply-add (2 FLOPs) per parameter forward; backward costs about twice the forward."),
                Card("Attention adds a term that grows with context", f'{v("O", "attn")} ≈ 4 · {v("ℓ")} · {v("S", sup="2")} · {v("d")} &nbsp;per sequence',
                     holds=f'Small next to 2{v("N")}·{v("S")} until {v("S")} nears {v("N")} / (2{v("ℓ")}{v("d")}); causal masking halves it.'),
                Card("Training time", f'{v("t", "train")} ≈ 6 · {v("N")} · tokens / ({v("R")} · {v("η")})', chapter="Ch 8",
                     ex=f'The Ch 8 run: GPT-2 XL ({G2 / 1e9:.1f}B) on {fmt(TR.training_tokens.magnitude / 1e9)}B tokens at {v("η")} = {fmt(TR.efficiency * 100)}% → '
                        f'{n("≈ " + fmt(TR.training_time.to("day").magnitude) + " days")} on one H100, {fmt(TR.energy.to("kWh").magnitude)} kWh, ${fmt(TR.rental_cost.to("USD").magnitude)} at ${USD_HR:.2f}/h.'),
            ], "Ch 6, App. Algorithm"),
            Col("2 Will it fit?", [
                Card(f'Weights: {v("M", "w")} = {v("N")} × bytes per parameter'),
                Table(["Llama 3 8B", "FP32 (4 B)", "FP16 (2 B)", "INT8 (1 B)", "INT4 (½ B)"], [["weights", gb(w["FP32"]), gb(w["FP16"]), gb(w["INT8"]), gb(w["INT4"])]]),
                Card("KV cache (inference)", f'{v("M", "KV")}({v("S")}) = 2 · {v("ℓ")} · {v("h", "kv")} · {v("d", "h")} · bytes · {v("S")} &nbsp;per sequence',
                     ex=f'Llama 3 8B FP16 ({v("h", "kv")} = {NKV}, {v("d", "h")} = {DH}): {n(fmt(kv_tok / 1024) + " KiB per token")}; {A}{v("S")} = {fmt(S["ctx"])} → '
                        f'{n(fmt(kv_seq / 2**20) + " MiB per sequence")}. With a K and V for all {NH} heads it would be {fmt(kv_mha / 1024)} KiB.'),
                Card("Largest batch that fits", f'{v("b", "max")} = {frac("capacity − " + v("M", "w") + " − overhead", v("M", "KV") + "(" + v("S") + ")")}',
                     ex=f'({CAP / 1e9:.1f} GB − {gb(w["FP16"])}) ÷ {kv_seq / 1e9:.3f} GB = {n(fmt(bmax) + " sequences")} at {v("S")} = {fmt(S["ctx"])}, an upper bound (no activations or allocator waste). Decode table →'),
                Card("Training state (mixed-precision Adam)", f'{v("M", "train")} ≈ (2 + 2 + 4 + 8) {v("N")} = 16{v("N")} bytes',
                     ex=f'FP16 weights, FP16 gradients, FP32 master copy, two FP32 Adam moments. Llama 3 8B: {n(fmt(16 * N / 1e9) + " GB", True)} &gt; {CAP / 1e9:.1f} GB before any activation.'),
                Card("Activations (training)", f'{v("M", "act")} ≈ 34 · {v("b")} · {v("S")} · {v("d")} bytes per layer',
                     holds="Ch 8's GPT-style FP16 constant, fused attention; an estimate for Llama.",
                     ex=f'Llama 3 8B, {v("b")} = 1, {v("S")} = {fmt(S["ctx"])}: {fmt(act_layer / 2**20)} MiB × {NL} layers = {n(f"{act_layer * NL / 1e9:.1f} GB")}.'),
            ], f"Ch 8, 10, 13 · H100 {fmt(CAP_GIB)} GiB = {CAP / 1e9:.1f} GB"),
            Col("3 How fast?", [
                Chart(rp.svg("H100 roofline")),
                Card(f'Roofline: attainable = min({v("R")}, {v("B")} · {v("I")}), ridge = {v("R")} / {v("B")}',
                     holds=f'Matrix multiply: {v("I")} = 2{v("MNK")} / bytes moved; {v("n")}/3 for square FP16 ({v("n")} = 4096 → {fmt(4096 / 3)}).'),
                Table(["H100 dense", "FP32", "TF32", "FP16", "FP8", "INT8"], [
                    ["peak (TFLOP/s, TOPS)", fmt(PEAK["fp32"] / 1e12), f"{PEAK['tf32'] / 1e12:.1f}", fmt(PEAK["fp16"] / 1e12), fmt(PEAK["fp8"] / 1e12), fmt(PEAK["int8"] / 1e12)],
                    ["ridge (FLOP/B)"] + [fmt(ridge[k]) for k in ("fp32", "tf32", "fp16", "fp8", "int8")]]),
                Card("Decode: every step reads the weights and the KV cache",
                     f'{v("t", "step")} ≈ {frac(v("M", "w") + " + " + v("b") + " · " + v("M", "KV") + "(" + v("S") + ")", v("B"))}, &nbsp; tokens/s = {v("b")} / {v("t", "step")}'),
                Table(["batch <i>b</i>", "ms/step", "tokens/s", "<i>I</i>", "$ / M tok"],
                      [[fmt(r["b"]), f"{r['ms']:.1f}", fmt(r["tps"]), fmt(r["I"]), f"${r['usd']:.2f}"] for r in rows],
                      caption=f'Llama 3 8B FP16, {v("S")} = {fmt(S["ctx"])}, ${USD_HR:.2f}/GPU-hour. {v("I")} stays below the ridge: batching buys throughput, not speed per token.'),
            ], f"Ch 8, 11, 13 · H100 {BW / 1e12:.2f} TB/s"),
        ],
    )

    # ---------------------------------------------------------------- page 2
    p2 = Page(
        question="Will it meet its target?",
        sub="Serving under load, deadlines on edge and tiny devices, and what shrinking the model buys.",
        meta="H100 · Jetson Orin Nano · STM32F4",
        notation=[
            ("Queues", f'{v("D")}, {v("O")}, {v("I")}, {v("L")} as on page 1 · {v("λ")} arrivals/s · {v("t", "s")} service time · {v("ρ")} = {v("λ")}{v("t", "s")} busy fraction · {v("W")} time in system · {v("Q")} requests in system'),
            ("Shrinking", f'{v("k")} bits per value · {v("Δ")} quantization step · {v("σ")} fraction of weights pruned'),
            ("Not the same “utilization”", f'{v("η")} (MFU) = useful FLOP/s ÷ peak · {v("ρ")} = fraction of time the server is busy · nvidia-smi “GPU-Util” = time any kernel was running'),
        ],
        cols=[
            Col("4 Can it serve the load?", [
                Card("Prefill (time to first token): compute-bound",
                     f'{v("t", "prefill")} ≈ (2{v("N")} · {v("S")} + 4{v("ℓ")}{v("S", sup="2")}{v("d")}) / ({v("R")} · {v("η")})',
                     ex=f'{fmt(S["prompt"])}-token prompt: {prefill_flop / 1e12:.1f} TFLOP at {A}{v("η")} = {fmt(S["eta_prefill"] * 100)}% → {n("≈ " + fmt(prefill_ms) + " ms")} (no causal saving).'),
                Card("Queueing (M/M/1: random arrivals, one server)",
                     f'{v("ρ")} = {v("λ")} · {v("t", "s")}, &nbsp; {v("W")} = {frac(v("t", "s"), "1 − " + v("ρ"))}, &nbsp; p99 ≈ {frac("4.6 · " + v("t", "s"), "1 − " + v("ρ"))}',
                     holds=f'4.6 = ln 100. Latency explodes as {v("ρ")} → 1; keep headroom for a tail target.'),
                Chart(qp.svg("Latency versus utilization")),
                f'<p class="ex">{A}{v("λ")} = {fmt(S["lam"])}/s, {v("t", "s")} = {fmt(ts)} ms → {v("ρ")} = {rho:.1f}: {v("W")} = {n(fmt(W_ms) + " ms")}, p99 = {n(fmt(p99_ms) + " ms")}. At {v("ρ")} = 0.5 the p99 is {fmt(p99_half)} ms.</p>',
                Card("Little's law", f'{v("Q")} = {v("λ")} · {v("W")}',
                     ex=f'{fmt(S["lam"])}/s × {fmt(W_ms)} ms = {n(fmt(inflight) + " requests")} in the system on average, {rho:.1f} in service and the rest waiting. '
                        f'Size queues with {v("Q")}; a batching server holds a KV cache per running request.'),
                Card("Cost per token", f'$ per token = {frac("$ per hour", "tokens/s × 3600")}',
                     ex=f'From the decode table: {n(usd_b1)} per million tokens at {v("b")} = 1, '
                        f'{n(usd_bmax)} at {v("b", "max")}. Batching is the cost lever.'),
            ], "Ch 13, 11"),
            Col("5 Does it meet the deadline?", [
                Card("Real-time frame budget", f'{v("t", "frame")} = 1 / fps &nbsp; ≥ &nbsp; Σ stage times (worst case)',
                     ex=f'{fmt(S["fps"])} fps → {n(f"{frame_ms:.1f} ms")} for capture, preprocessing, inference, and post-processing together.'),
                Card("Inference on an edge GPU", f'{v("t")} ≈ max( {frac(v("O"), v("R") + " · " + v("η"))}, {frac("weights + activations", v("B"))} ) + {v("L")}',
                     ex=f'YOLOv8-Nano ({Y_FLOP / 1e9:.1f} GFLOP, {Y_N / 1e6:.1f}M params) on Jetson Orin Nano ({fmt(J_PEAK / 1e12)} TFLOP/s FP16-equiv., {fmt(J_BW / 1e9)} GB/s) at {A}{v("η")} = {fmt(S["eta_edge"] * 100)}%: '
                        f'compute {n(f"{yolo_ms:.1f} ms")}; FP16 weights alone read in {yolo_w_ms:.2f} ms. Launches cost {J_LAUNCH_MS:g} ms each: run one compiled engine. '
                        f'Energy {v("E")} = {v("P")} · {v("t")} ≤ {fmt(J_TDP)} W × {yolo_ms:.1f} ms ≈ {n(fmt(J_TDP * yolo_ms) + " mJ")} per frame.'),
                Card("Will it fit on a microcontroller?", ["flash ≥ weights + code", "SRAM ≥ largest layer's (input + output) + runtime"],
                     ex=f'DS-CNN keyword spotter ({fmt(K_N / 1e3)}K params, INT8) on STM32F4: {n(fmt(K_N / 1024) + " KiB")} of {fmt(MCU_FLASH)} KiB flash; SRAM is {fmt(MCU_SRAM)} KiB, so the largest layer\'s activations decide.'),
                Card("Microcontroller time", f'{v("t")} ≈ {v("O")} / (FLOP per cycle × clock)',
                     ex=f'{fmt(K_FLOP / 1e6)} MFLOP at 1 FLOP/cycle ({fmt(MCU_CLK)} MHz) ≈ {n(fmt(kws_ms) + " ms")}; INT8 SIMD kernels at {A}{S["mcu_macs_per_cycle"]} MACs/cycle ≈ {n(fmt(kws_simd) + " ms")}.'),
                Card("Can the data pipeline keep up?", f'samples/s ≤ min(accel., CPU, {frac(v("B", "disk") + " (1 − loss)", "bytes/sample")})',
                     ex=f'{A}{fmt(S["ssd"] / 1e6)} MB/s SSD, {fmt(S["f_loss"] * 100)}% loss, {fmt(S["sample"] / 1e3)} KB images → disk supplies {n(fmt(supply) + " samples/s")}. CPU = workers × decode rate per worker. '
                        f'Loading {gb(w["FP16"])} of weights over PCIe ({fmt(PCIE / 1e9)} GB/s) takes ≥ {w["FP16"] / PCIE:.2f} s.'),
            ], "Ch 2, 4, 13"),
            Col("6 What does shrinking buy?", [
                Card("Convolution, standard vs depthwise-separable",
                     [f'standard: MACs = {v("H", "o")}{v("W", "o")} · {v("K", sup="2")} · {v("C", "in")} · {v("C", "out")}',
                      f'separable: MACs = {v("H", "o")}{v("W", "o")} · {v("C", "in")} · ({v("K", sup="2")} + {v("C", "out")})'],
                     holds=f'Per image, per layer; {v("H", "o")}×{v("W", "o")} output size. Savings ≈ 1 / (1/{v("C", "out")} + 1/{v("K", sup="2")}).',
                     ex=f'{hw_}×{hw_}, {v("K")} = {k_}, {ci}→{co} channels: {fmt(conv_std / 1e6)}M → {fmt(conv_dw / 1e6)}M MACs, {n(f"{conv_std / conv_dw:.1f}× fewer")}.'),
                Card("Quantization", f'bytes × {frac(v("k"), "16")} from FP16, &nbsp; {v("Δ")} = {frac("max − min", "2" + "<sup>" + v("k") + "</sup> − 1")}',
                     ex=f'Only quantized tensors shrink: INT4 weights cut decode bytes at {v("b", "max")} by just {fmt(int4_cut * 100)}%, since the FP16 KV cache dominates. '
                        f'Range [{lo:g}, {hi:g}]: {v("Δ")} = {n(f"{d8:.3f}")} at INT8, {n(f"{d4:.2f}")} at INT4. Error per value ≤ {v("Δ")}/2.'),
                Card("Pruning", f'bytes ≈ (1 − {v("σ")}) · {v("N")} · (value + index bytes)',
                     ex=f'{A}{v("σ")} = {fmt(S["prune"] * 100)}%, FP16: a {S["idx_bytes"]}-byte relative index per value gives {n(gb(prune_idx), True)}, no saving; a 1-bit mask gives {gb(prune_mask)}. '
                        f'2:4 sparsity doubles H100 peak math ({fmt(sparse_ratio)}×); bandwidth-bound decode gains only the fewer bytes.'),
                Card("Movement costs more than math", f'{v("E")} ≈ {v("D")} · {v("e", "byte")} + {v("O")} · {v("e", "op")}', chapter="Ch 1"),
                Table(["45 nm (Horowitz)", "INT8 mult", "FP16 mult", "FP32 mult", "DRAM, per byte"],
                      [["pJ", f"{E_INT8:g}", f"{E_FP16:g}", f"{E_FP32:g}", fmt(e_byte)]]),
                f'<p class="holds">A DRAM byte ≈ {fmt(e_byte / E_FP16)}× an FP16 multiply. At {v("I")} ≈ 1 (batch-1 decode), energy is almost all movement.</p>',
                Note("Before you trust the number:", f'every term in seconds or bytes? GB vs GiB? Compute-bound only if {v("I")} &gt; ridge? Memory counted with KV cache and activations? {v("ρ")} below about 0.7 for a tail target?'),
            ], "Ch 6, 10, 11"),
        ],
        bottom=Table(["Reference node", "peak", "memory bandwidth", "memory", "ridge (FLOP/B)", "launch", "power"], [
            ["NVIDIA H100 (cloud)", f"{fmt(PEAK['fp16'] / 1e12)} TFLOP/s FP16", f"{BW / 1e12:.2f} TB/s HBM", f"{fmt(CAP_GIB)} GiB", fmt(ridge["fp16"]), f"{fmt(LAUNCH_US)} µs", f"{fmt(TDP)} W"],
            ["Jetson Orin Nano (edge)", f"{fmt(J_PEAK / 1e12)} TFLOP/s FP16-equiv.", f"{fmt(J_BW / 1e9)} GB/s", f"{fmt(J.memory.capacity.to('GiB').magnitude)} GiB", fmt(J_PEAK / J_BW), f"{J_LAUNCH_MS:g} ms", f"{fmt(J_TDP)} W"],
            ["STM32F4, Cortex-M4F (tiny)", f"{fmt(MCU_PEAK / 1e6)} MFLOP/s FP32", f"{fmt(MCU_BW / 1e6)} MB/s", f"{fmt(MCU_SRAM)} KiB SRAM, {fmt(MCU_FLASH)} KiB flash", f"{MCU_PEAK / MCU_BW:.2f}", "—", "—"],
        ]),
    )
    return Sheet(volume=1, book="Introduction to Machine Learning Systems", scope="One Node", pages=[p1, p2],
                 sources=f'GB = 10<sup>9</sup> B, GiB = 2<sup>30</sup> B · 1 MAC = 2 FLOPs · {TAG}= input chosen for the example · values from the mlsysim registry; Llama 3 vocabulary from its model card.')

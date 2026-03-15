import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")

# ─────────────────────────────────────────────────────────────────────────────
# LAB 10: THE KV-CACHE MEMORY WALL
#
# Core Invariant: LLM inference has two phases — prefill (compute-bound,
# parallel across prompt tokens) and decode (memory-bandwidth-bound, sequential).
# The KV-cache grows as: 2 × num_layers × seq_len × num_heads × head_dim × bytes.
# Continuous batching allows new requests to join mid-sequence, eliminating
# the waste of waiting for the slowest sequence in a static batch.
#
# Structure:
#   Act I  — Calibration (12-15 min)
#     KV-cache calculator. Prediction on max concurrent users.
#     Prediction-vs-reality overlay. Reflection on why seq_len dominates.
#
#   Act II — Design Challenge (20-25 min)
#     Latency-optimized vs Throughput-optimized serving configuration.
#     Side-by-side metrics: TTFT, throughput, GPU utilization.
#     FAILURE STATE: P99 TTFT > 200ms SLO violation banner.
#     Reflection on continuous batching advantage.
#
# 2 Contexts: Latency-optimized (real-time assistant) vs
#             Throughput-optimized (batch document processing)
#
# Design Ledger: chapter="v2_10"
# ─────────────────────────────────────────────────────────────────────────────


@app.cell
async def _():
    import marimo as mo
    import sys
    from pathlib import Path
    import plotly.graph_objects as go
    import numpy as np
    import math

    # WASM bootstrap: install mlsysim from hosted wheel when running in browser
    if sys.platform == "emscripten":
        import micropip
        await micropip.install("https://mlsysbook.ai/labs/wheels/mlsysim-0.1.0-py3-none-any.whl")
    elif "mlsysim" not in sys.modules:
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))

    from mlsysim.labs.state import DesignLedger
    from mlsysim.labs.style import COLORS, LAB_CSS, apply_plotly_theme
    from mlsysim.labs.components import DecisionLog

    ledger = DesignLedger()

    # ── Hardware constants (H100 SXM5, NVIDIA spec) ───────────────────────────
    H100_BW_GBS       = 3350   # GB/s — HBM3e memory bandwidth (NVIDIA H100 spec)
    H100_TFLOPS_FP16  = 989    # TFLOPS FP16 dense tensor core — NVIDIA H100 SXM5 spec
    H100_RAM_GB       = 80     # GB — HBM3e capacity per H100 (NVIDIA H100 spec)
    NVLINK4_BW_GBS    = 900    # GB/s — NVLink 4.0 bisection bandwidth per GPU pair

    # ── LLaMA-3 70B architecture (Meta, 2024) ────────────────────────────────
    LLAMA70B_LAYERS   = 80     # transformer layers (Meta LLaMA-3 70B config)
    LLAMA70B_HEADS    = 64     # attention heads (Meta LLaMA-3 70B config)
    LLAMA70B_HEAD_DIM = 128    # head dimension: d_model / n_heads = 8192 / 64
    LLAMA70B_PARAMS_B = 70     # billion parameters

    # ── KV-cache physics constants ────────────────────────────────────────────
    BYTES_FP16        = 2      # bytes per FP16 element
    KV_TENSORS        = 2      # K and V per layer

    # ── KV-cache per token per layer (derived from LLaMA-3 70B architecture)
    # = KV_TENSORS × num_heads × head_dim × bytes_per_elem
    # = 2 × 64 × 128 × 2 = 32,768 bytes per token per layer
    KV_BYTES_PER_TOKEN_PER_LAYER = (
        KV_TENSORS * LLAMA70B_HEADS * LLAMA70B_HEAD_DIM * BYTES_FP16
    )  # = 32,768 bytes

    # ── LLaMA-3 70B weight memory (rough lower bound at FP16) ────────────────
    # 70e9 params × 2 bytes/param = 140 GB
    LLAMA70B_WEIGHTS_GB = LLAMA70B_PARAMS_B * 1e9 * BYTES_FP16 / (1024 ** 3)

    # ── Serving SLO targets (from inference.qmd) ─────────────────────────────
    SLO_TTFT_MS       = 200    # ms — P99 time-to-first-token for real-time assistant
    SLO_TPS_TARGET    = 1000   # tokens/sec — minimum throughput for batch processing

    return (
        mo, ledger, COLORS, LAB_CSS, apply_plotly_theme, go, np, math,
        H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB, NVLINK4_BW_GBS,
        LLAMA70B_LAYERS, LLAMA70B_HEADS, LLAMA70B_HEAD_DIM, LLAMA70B_PARAMS_B,
        BYTES_FP16, KV_TENSORS,
        KV_BYTES_PER_TOKEN_PER_LAYER,
        LLAMA70B_WEIGHTS_GB,
        SLO_TTFT_MS, SLO_TPS_TARGET,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, LAB_CSS, COLORS):
    _c1 = COLORS["Cloud"]
    _c2 = COLORS["BlueLine"]
    mo.vstack([
        LAB_CSS,
        mo.md(f"""
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                    padding: 36px 44px; border-radius: 16px; color: white;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.3);">
            <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.18em;
                        color: #475569; text-transform: uppercase; margin-bottom: 10px;">
                Machine Learning Systems · Volume II · Lab 10
            </div>
            <h1 style="margin: 0 0 10px 0; font-size: 2.4rem; font-weight: 900;
                       color: #f8fafc; line-height: 1.1; letter-spacing: -0.02em;">
                The KV-Cache Memory Wall
            </h1>
            <p style="margin: 0 0 20px 0; font-size: 1.05rem; color: #94a3b8;
                      max-width: 680px; line-height: 1.65;">
                You have 640 GB across eight H100s. Your model weights cost 140 GB.
                The remaining 500 GB must hold every token's key and value vectors
                for every concurrent user. How many users can you actually serve?
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act I · KV-Cache Memory Wall · 12–15 min
                </span>
                <span style="background: rgba(99,102,241,0.15); color: #a5b4fc;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(99,102,241,0.25);">
                    Act II · Latency vs Throughput · 20–25 min
                </span>
                <span style="background: rgba(16,185,129,0.15); color: #6ee7b7;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(16,185,129,0.25);">
                    35–40 min total
                </span>
                <span style="background: rgba(245,158,11,0.15); color: #fcd34d;
                             padding: 5px 14px; border-radius: 20px; font-size: 0.8rem;
                             font-weight: 600; border: 1px solid rgba(245,158,11,0.25);">
                    New instrument: KV-Cache Calculator
                </span>
            </div>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px;">
                <span class="badge badge-info">KV = 2 × L × S × H × D × bytes</span>
                <span class="badge badge-warn">TTFT SLO &lt; 200ms</span>
                <span class="badge badge-ok">Continuous batching eliminates tail waste</span>
            </div>
        </div>
        """),
    ])
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE A: OPENING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 2: BRIEFING ───────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.Html(f"""
    <div style="border-left: 4px solid {COLORS['BlueLine']};
                background: white; border-radius: 0 12px 12px 0;
                padding: 20px 28px; margin: 8px 0 16px 0;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06);">

        <!-- LEARNING OBJECTIVES -->
        <div style="margin-bottom: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Learning Objectives
            </div>
            <div style="font-size: 0.9rem; color: {COLORS['TextSec']}; line-height: 1.7;">
                <div style="margin-bottom: 3px;">1. <strong>Calculate the KV-cache memory footprint</strong> for Llama-70B (2.6 MB/token) at varying context lengths and determine the maximum concurrent users on a 4x H100 NVLink replica (160 GB available after weights).</div>
                <div style="margin-bottom: 3px;">2. <strong>Predict which batch size minimizes total response time</strong> at 100 QPS &mdash; discovering that batch=32 (91.5 ms total) beats batch=8 (123.2 ms) despite 22% longer per-batch service time, because queuing delay drops from 42.3 ms to 8.9 ms.</div>
                <div style="margin-bottom: 3px;">3. <strong>Design a fleet configuration</strong> for 1,000 QPS of Llama-70B with P99 &lt; 200 ms, quantifying that Power-of-Two-Choices load balancing reduces P99 by 47% versus random assignment.</div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- PREREQUISITES + DURATION (side by side) -->
        <div style="display: flex; gap: 32px; margin-top: 16px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 220px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Prerequisites
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    KV-cache memory formula from @sec-inference-scale-kv-cache-wall &middot;
                    Queuing hockey-stick behavior from @sec-fleet-orchestration-introduction
                </div>
            </div>
            <div style="flex: 0 0 180px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                    Duration
                </div>
                <div style="font-size: 0.85rem; color: {COLORS['TextSec']}; line-height: 1.65;">
                    <strong>35-40 min</strong><br/>
                    Act I: ~12 min &middot; Act II: ~25 min
                </div>
            </div>
        </div>

        <div style="border-top: 1px solid {COLORS['Border']}; margin: 0 -28px; padding: 0 28px;"></div>

        <!-- CORE QUESTION -->
        <div style="margin-top: 16px;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px;">
                Core Question
            </div>
            <div style="font-size: 1.05rem; color: {COLORS['Text']}; font-weight: 600;
                        line-height: 1.5; font-style: italic;">
                "You have 640 GB across eight H100s. After weights consume 140 GB, the
                remaining memory must hold KV-cache for every concurrent user &mdash;
                and the faster you serve each request, the worse your total system latency
                becomes. Why does minimizing service time maximize total response time?"
            </div>
        </div>
    </div>
    """)
    return


# ─── CELL 3: RECOMMENDED READING ─────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Recommended Reading** — Complete the following chapter sections before this lab:

    - @sec-inference-phases — Prefill vs decode: why the two phases have different bottlenecks
    - @sec-kv-cache — KV-cache derivation, memory formula, paged attention motivation
    - @sec-continuous-batching — How continuous batching differs from static batching
    - @sec-inference-serving-architectures — TTFT, TBT, throughput tradeoffs in LLM serving
    """), kind="info")
    return


# ─── CELL 4: CONTEXT TOGGLE ──────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    context_toggle = mo.ui.radio(
        options={
            "Latency-Optimized (Real-Time Assistant)": "latency",
            "Throughput-Optimized (Batch Document Processing)": "throughput",
        },
        value="Latency-Optimized (Real-Time Assistant)",
        label="Serving context:",
        inline=True,
    )
    mo.vstack([
        mo.md("---"),
        context_toggle,
        mo.md("""
        You are designing an LLM serving system. Two customers need fundamentally
        different guarantees. Select a context to begin.
        """),
    ])
    return (context_toggle,)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE B: ACT I -- CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 5: ACT1_BANNER ────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "I"
    _act_color    = COLORS["BlueLine"]
    _act_title    = "The KV-Cache Memory Wall"
    _act_duration = "12-15 min"
    _act_why      = ("You expect 500 GB of free memory to support hundreds of concurrent users. "
                     "The KV-cache formula (2.6 MB/token/user) will show that at 4096-token "
                     "context, each user consumes 10.65 GB &mdash; limiting a 4x H100 replica "
                     "to approximately 15 concurrent users, not hundreds.")

    mo.Html(f"""
    <div style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_act_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">{_act_num}</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act {_act_num} &middot; {_act_duration}</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            {_act_title}
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            {_act_why}
        </div>
    </div>
    """)
    return


# ─── CELL 6: ACT1_STAKEHOLDER ───────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, context_toggle):
    _ctx   = context_toggle.value
    _color = COLORS["Cloud"] if _ctx == "latency" else COLORS["BlueLine"]
    _bg    = COLORS["BlueL"]
    _persona = (
        "LLM Serving Team Lead" if _ctx == "latency"
        else "Platform Engineering Lead"
    )
    _quote = (
        "We have 8 H100s — 80 GB each, 640 GB total. We're serving LLaMA-3 70B "
        "which needs 140 GB for weights in FP16. That leaves 500 GB for KV-cache. "
        "We're targeting 4096-token context windows. My manager is asking how many "
        "concurrent users we can actually support. Give me a number."
        if _ctx == "latency" else
        "Same infrastructure — 8 H100s, LLaMA-3 70B, 500 GB available for KV-cache. "
        "Our batch document pipeline processes 8192-token documents. How many "
        "documents can we hold in-flight simultaneously before we hit the memory wall? "
        "We need to size the job queue correctly."
    )
    _context_tokens = 4096 if _ctx == "latency" else 8192

    mo.vstack([
        mo.md("---"),
        mo.md(f"""
        Every token a user has generated — and every token in their prompt — requires
        storing a **key vector** and a **value vector** for each attention layer.
        These tensors must live in GPU memory for the entire duration of the sequence.
        They cannot be recomputed cheaply; they must be present to produce the next token.

        For LLaMA-3 70B at **{_context_tokens}-token context**, the numbers are not small.
        """),
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:{_bg};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message · {_persona}
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "{_quote}"
            </div>
        </div>
        """),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — PREDICTION LOCK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo):
    act1_pred = mo.ui.radio(
        options={
            "A) ~200 concurrent users — KV-cache is small compared to the weights":
                "A",
            "B) ~50 concurrent users — context window is the bottleneck but still manageable":
                "B",
            "C) ~20–25 concurrent users — KV-cache per user is surprisingly large":
                "C",
            "D) ~5 concurrent users — most memory is reserved for intermediate computation":
                "D",
        },
        label=(
            "**Prediction Lock — Act I.** "
            "8 H100s (640 GB total), LLaMA-3 70B weights (140 GB in FP16), "
            "500 GB available for KV-cache, 4096-token context window per user. "
            "How many concurrent users can this cluster support?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background:#1e293b; border-radius:10px; padding:4px 18px 12px 18px;
                    border-left:4px solid #6366f1; margin-bottom:8px;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em;
                        margin-top:12px; margin-bottom:8px;">
                Prediction Lock · Act I
            </div>
        </div>
        """),
        act1_pred,
    ])
    return (act1_pred,)


@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(
        act1_pred.value is None,
        mo.callout(mo.md("Select your prediction to continue."), kind="warn"),
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — KV-CACHE CALCULATOR (INSTRUMENTS)
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(act1_pred.value is None)

    # ── Sliders ───────────────────────────────────────────────────────────────
    context_len_slider = mo.ui.slider(
        start=512, stop=32768, value=4096, step=512,
        label="Context length (tokens)",
        show_value=True,
    )
    model_dropdown = mo.ui.dropdown(
        options={
            "LLaMA-3 7B (layers=32, heads=32, head_dim=128)":  "7B",
            "LLaMA-3 13B (layers=40, heads=40, head_dim=128)": "13B",
            "LLaMA-3 70B (layers=80, heads=64, head_dim=128)": "70B",
            "GPT-3 175B  (layers=96, heads=96, head_dim=128)": "175B",
        },
        value="LLaMA-3 70B (layers=80, heads=64, head_dim=128)",
        label="Model",
    )
    gpu_cluster = mo.ui.radio(
        options={
            "1 H100 (80 GB)": "single",
            "8 H100s (640 GB)": "fleet",
        },
        value="8 H100s (640 GB)",
        label="GPU cluster:",
        inline=True,
    )

    mo.vstack([
        mo.md("### Simulator — KV-Cache Memory Calculator"),
        mo.md("""
        Adjust context length, model architecture, and cluster size.
        Watch how KV-cache memory per user changes. The key question: how
        quickly does it consume the available budget?
        """),
        mo.hstack([context_len_slider, model_dropdown], justify="start", gap=2),
        gpu_cluster,
    ])
    return (context_len_slider, model_dropdown, gpu_cluster)


@app.cell(hide_code=True)
def _(
    mo, go, np, apply_plotly_theme, COLORS,
    context_len_slider, model_dropdown, gpu_cluster,
    BYTES_FP16, KV_TENSORS,
    H100_RAM_GB,
    act1_pred,
):
    mo.stop(act1_pred.value is None)

    # ── Model architecture lookup ─────────────────────────────────────────────
    # Sources: Meta LLaMA-3 technical report (2024); OpenAI GPT-3 paper (2020)
    _model_configs = {
        "7B":   {"layers": 32, "heads": 32,  "head_dim": 128, "params_b": 7,   "weights_gb": 7   * 2},
        "13B":  {"layers": 40, "heads": 40,  "head_dim": 128, "params_b": 13,  "weights_gb": 13  * 2},
        "70B":  {"layers": 80, "heads": 64,  "head_dim": 128, "params_b": 70,  "weights_gb": 70  * 2},
        "175B": {"layers": 96, "heads": 96,  "head_dim": 128, "params_b": 175, "weights_gb": 175 * 2},
    }

    _key = model_dropdown.value
    _cfg = _model_configs[_key]
    _n_layers   = _cfg["layers"]
    _n_heads    = _cfg["heads"]
    _head_dim   = _cfg["head_dim"]
    _params_b   = _cfg["params_b"]
    _weights_gb = _cfg["weights_gb"]

    # ── GPU cluster total RAM ─────────────────────────────────────────────────
    _n_gpus = 8 if gpu_cluster.value == "fleet" else 1
    _total_ram_gb = _n_gpus * H100_RAM_GB

    # ── KV-cache physics ──────────────────────────────────────────────────────
    # Formula (from @sec-kv-cache):
    #   KV_bytes = 2 × num_layers × seq_len × num_heads × head_dim × bytes_per_elem
    #            = KV_TENSORS × layers × seq_len × heads × head_dim × BYTES_FP16
    _seq_len = context_len_slider.value

    # Per token (all layers), in bytes
    _kv_bytes_per_token = (
        KV_TENSORS * _n_layers * _n_heads * _head_dim * BYTES_FP16
    )
    # Per user at full context length, in GB
    _kv_per_user_gb = _kv_bytes_per_token * _seq_len / (1024 ** 3)

    # Available memory after weights
    _available_gb = max(0.0, _total_ram_gb - _weights_gb)

    # Max concurrent users (floor division)
    _max_users = int(_available_gb / _kv_per_user_gb) if _kv_per_user_gb > 0 else 0

    # Memory utilization components
    _weight_pct  = min(100.0, _weights_gb / _total_ram_gb * 100.0)
    _kv_pct      = min(100.0, _available_gb / _total_ram_gb * 100.0)

    # ── Constraint checks ─────────────────────────────────────────────────────
    _oom_weights = _weights_gb > _total_ram_gb
    _users_ok    = _max_users >= 10

    # ── Metric card colors ───────────────────────────────────────────────────
    _users_color = (
        COLORS["GreenLine"]  if _max_users >= 20 else
        COLORS["OrangeLine"] if _max_users >= 5 else
        COLORS["RedLine"]
    )
    _avail_color = (
        COLORS["RedLine"]    if _available_gb <= 0 else
        COLORS["OrangeLine"] if _available_gb < 50 else
        COLORS["GreenLine"]
    )

    # ── Context-length sweep chart ────────────────────────────────────────────
    _ctx_sweep = np.arange(512, 32769, 512)
    _kv_per_user_sweep = (
        KV_TENSORS * _n_layers * _ctx_sweep * _n_heads * _head_dim * BYTES_FP16
        / (1024 ** 3)
    )
    _max_users_sweep = np.where(
        _kv_per_user_sweep > 0,
        (_available_gb / _kv_per_user_sweep).astype(int),
        0
    )
    _max_users_sweep = np.maximum(_max_users_sweep, 0)

    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=_ctx_sweep,
        y=_max_users_sweep,
        mode="lines",
        name="Max concurrent users",
        line=dict(color=COLORS["BlueLine"], width=3),
        fill="tozeroy",
        fillcolor=f"rgba(0,99,149,0.08)",
        hovertemplate="Context: %{x:,} tokens<br>Max users: %{y}<extra></extra>",
    ))
    # Mark current slider position
    _fig.add_trace(go.Scatter(
        x=[_seq_len],
        y=[_max_users],
        mode="markers",
        name="Current config",
        marker=dict(
            color=COLORS["OrangeLine"], size=14, symbol="diamond",
            line=dict(color="white", width=2),
        ),
        hovertemplate="Context: %{x:,} tokens<br>Max users: %{y}<extra></extra>",
    ))
    # Danger zone annotation (below 10 users)
    _fig.add_hline(
        y=10,
        line_dash="dash",
        line_color=COLORS["RedLine"],
        line_width=1.5,
        annotation_text="< 10 users (unviable)",
        annotation_position="top right",
        annotation_font_color=COLORS["RedLine"],
        annotation_font_size=11,
    )

    _fig.update_layout(
        title=dict(
            text=f"Max Concurrent Users vs Context Length — {_key} on {_n_gpus}× H100",
            font_size=13,
        ),
        xaxis_title="Context length (tokens)",
        yaxis_title="Max concurrent users",
        showlegend=False,
        height=300,
    )
    apply_plotly_theme(_fig)

    # ── Rendered output ───────────────────────────────────────────────────────
    mo.vstack([
        mo.md(f"""
        **KV-Cache Memory Derivation**

        ```
        Formula (@sec-kv-cache):
          KV_bytes_per_token = KV_TENSORS × layers × heads × head_dim × bytes_per_elem
                             = {KV_TENSORS} × {_n_layers} × {_n_heads} × {_head_dim} × {BYTES_FP16}
                             = {_kv_bytes_per_token:,} bytes per token (all layers)

          KV_per_user_GB = KV_bytes_per_token × seq_len / (1024³)
                         = {_kv_bytes_per_token:,} × {_seq_len:,} / (1024³)
                         = {_kv_per_user_gb:.2f} GB per user

          available_GB   = total_RAM - weights_GB
                         = {_total_ram_gb} GB - {_weights_gb} GB
                         = {_available_gb:.1f} GB

          max_users      = ⌊available_GB / KV_per_user_GB⌋
                         = ⌊{_available_gb:.1f} / {_kv_per_user_gb:.2f}⌋
                         = {_max_users} users
        ```
        """),
        mo.Html(f"""
        <div style="display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0;">
            <div style="padding: 18px 24px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.05em;">
                    KV per user
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {COLORS['BlueLine']};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {_kv_per_user_gb:.1f} GB
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">at {_seq_len:,} tokens</div>
            </div>
            <div style="padding: 18px 24px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.05em;">
                    Available memory
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {_avail_color};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {_available_gb:.0f} GB
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">after {_weights_gb} GB weights</div>
            </div>
            <div style="padding: 18px 24px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.05em;">
                    Max concurrent users
                </div>
                <div style="font-size: 2rem; font-weight: 800; color: {_users_color};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {_max_users}
                </div>
                <div style="color: #94a3b8; font-size: 0.75rem;">{_n_gpus}× H100 cluster</div>
            </div>
            <div style="padding: 18px 24px; border: 1px solid #e2e8f0; border-radius: 10px;
                        min-width: 160px; text-align: center; background: white;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                <div style="color: #64748b; font-size: 0.8rem; font-weight: 600;
                            text-transform: uppercase; letter-spacing: 0.05em;">
                    Memory split
                </div>
                <div style="font-size: 0.95rem; font-weight: 700; color: {COLORS['Text']};
                            font-family: 'SF Mono', monospace; margin: 4px 0;">
                    {_weight_pct:.0f}% weights
                </div>
                <div style="font-size: 0.95rem; font-weight: 700; color: {COLORS['BlueLine']};
                            font-family: 'SF Mono', monospace;">
                    {_kv_pct:.0f}% KV available
                </div>
            </div>
        </div>
        """),
        mo.as_html(_fig),
        (
            mo.callout(
                mo.md(
                    f"**OOM — Model does not fit.** "
                    f"Weights require {_weights_gb} GB but cluster has only {_total_ram_gb} GB. "
                    "Switch to a smaller model or increase the cluster size."
                ),
                kind="danger"
            ) if _oom_weights else
            mo.callout(
                mo.md(
                    f"**Viable configuration.** "
                    f"{_max_users} concurrent users with {_kv_per_user_gb:.1f} GB each. "
                    f"Remaining KV budget: {_available_gb:.0f} GB."
                ),
                kind="success" if _users_ok else "warn"
            )
        ),
    ])
    return (_max_users, _kv_per_user_gb, _available_gb, _weights_gb, _n_gpus)


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — PREDICTION vs REALITY OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_pred, _max_users, _kv_per_user_gb):
    mo.stop(act1_pred.value is None)

    # Expected value for each answer choice (at default 4096-token context, 70B, 8× H100)
    _predicted_map = {"A": 200, "B": 50, "C": 23, "D": 5}
    _choice = act1_pred.value
    _predicted = _predicted_map[_choice]
    _actual = _max_users

    _ratio = _actual / _predicted if _predicted > 0 else float("inf")
    _close = abs(_ratio - 1.0) < 0.30

    _feedback_map = {
        "A": (
            "**Off by ~9×.** The KV-cache is *not* small compared to weights. "
            f"At 4096-token context, LLaMA-3 70B uses {_kv_per_user_gb:.1f} GB per user. "
            "That is roughly the same order of magnitude as the model weights themselves (140 GB). "
            "Context length — not model size — is what makes KV-cache the binding constraint at serving time."
        ),
        "B": (
            "**Off by ~2×.** A common underestimate. You sensed the context window matters "
            f"but underestimated the per-user cost. At 4096 tokens, a 70B model needs "
            f"{_kv_per_user_gb:.1f} GB per concurrent user — roughly one full A100's worth of RAM "
            "for every 4 users."
        ),
        "C": (
            f"**Correct.** The actual limit is {_actual} users. "
            f"Each user consumes {_kv_per_user_gb:.1f} GB of KV-cache at 4096-token context. "
            "With 500 GB available after weights, this leaves room for roughly 23 simultaneous users. "
            "This is why context length reduction and KV-cache compression are active research areas."
        ),
        "D": (
            "**Off by ~5×.** The activation memory during a forward pass is small and transient "
            f"— it does not compete with KV-cache at serving time. The real constraint is that "
            f"each of your concurrent users needs {_kv_per_user_gb:.1f} GB of persistent KV storage "
            "for the duration of their session."
        ),
    }

    _kind = "success" if _close else "warn"
    mo.vstack([
        mo.md("### Prediction vs Reality"),
        mo.callout(
            mo.md(
                f"**You predicted ≈ {_predicted} users. The physics says {_actual} users. "
                f"(Ratio: {_ratio:.1f}×)** "
                + _feedback_map[_choice]
            ),
            kind=_kind,
        ),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(act1_pred.value is None)

    act1_reflect = mo.ui.radio(
        options={
            "A) Sequence length determines the attention window — longer context means more keys and values to cache":
                "A",
            "B) Longer sequences require more computation per token, so more memory is needed":
                "B",
            "C) KV-cache compresses proportionally with model width, so wider models need less storage per token":
                "C",
            "D) Sequence length determines the number of attention heads used":
                "D",
        },
        label=(
            "**Reflection — Act I.** "
            "The KV-cache memory formula scales linearly with sequence length. "
            "Why does sequence length — not model width — dominate KV-cache growth?"
        ),
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Reflection"),
        act1_reflect,
    ])
    return (act1_reflect,)


@app.cell(hide_code=True)
def _(mo, act1_reflect):
    mo.stop(
        act1_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue."), kind="warn"),
    )

    _reflect_feedback = {
        "A": (
            "**Correct.** For every token in the sequence, each attention layer must store "
            "one key vector and one value vector. These are indexed by position: the attention "
            "mechanism at decode step *t* needs to attend over all *t* previous key-value pairs. "
            "Model width (num_heads × head_dim) sets the *size* of each KV pair, but sequence "
            "length sets *how many* pairs must exist — and that grows with every new token generated."
        ),
        "B": (
            "**Not quite.** Compute per token is roughly constant with sequence length for the "
            "linear projections. The attention mechanism does require O(seq_len) operations per "
            "decode step, but this is a compute concern, not a memory one. "
            "Memory grows because you must *store* all previous KV pairs, not because you do "
            "more computation per token."
        ),
        "C": (
            "**Not quite.** There is no compression: the KV-cache stores exact floating-point "
            "tensors. Model width (heads × head_dim) affects the *size per token* — it is a "
            "fixed multiplicative factor. Sequence length affects *how many* tokens are cached. "
            "They both scale the total, but sequence length is the variable the serving engineer "
            "can observe changing as users generate longer outputs."
        ),
        "D": (
            "**Not quite.** Multi-head attention always uses all heads for every token. "
            "The number of active heads is determined by the model architecture, not the sequence "
            "length. What grows with sequence length is the number of (key, value) pairs that "
            "must be stored and retrieved during attention computation."
        ),
    }

    _correct = act1_reflect.value == "A"
    mo.callout(
        mo.md(_reflect_feedback[act1_reflect.value]),
        kind="success" if _correct else "warn",
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT I — MATHPEEK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(act1_pred.value is None)

    mo.accordion({
        "The governing equation — KV-cache memory formula": mo.md("""
        **KV-Cache Memory Formula** (from @sec-kv-cache):

        ```
        KV_bytes = 2 × num_layers × seq_len × num_heads × head_dim × bytes_per_elem
        ```

        - **2** — one key tensor + one value tensor per layer
        - **num_layers** — transformer depth (LLaMA-3 70B: 80)
        - **seq_len** — tokens in the context window (4096 – 32768)
        - **num_heads** — parallel attention heads (LLaMA-3 70B: 64)
        - **head_dim** — dimension per head = d_model / num_heads (128 for LLaMA-3 70B)
        - **bytes_per_elem** — 2 bytes for FP16, 1 byte for INT8 (if quantized)

        **LLaMA-3 70B at 4096 tokens:**
        ```
        = 2 × 80 × 4096 × 64 × 128 × 2
        = 2 × 80 × 4,096 × 8,192  (heads × head_dim = 8,192)
        = 2 × 80 × 33,554,432 bytes
        = 5,368,709,120 bytes
        ≈ 5.0 GB per user
        ```

        **Paged Attention** (vLLM): Instead of pre-allocating `max_seq_len` bytes per request,
        allocate in fixed-size *pages* (e.g., 16 tokens per page = 16 × 32,768 bytes = 512 KB).
        Pages are allocated on demand, allowing unused capacity to be reclaimed.
        This eliminates internal fragmentation and raises effective utilization
        from ~50% to ~90%+ in practice.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE C: ACT II -- DESIGN CHALLENGE
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 12: ACT2_BANNER ───────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    _act_num      = "II"
    _act_color    = COLORS["OrangeLine"]
    _act_title    = "Latency vs Throughput: Two Serving Configurations"
    _act_duration = "20-25 min"
    _act_why      = ("Act I showed that KV-cache limits concurrent users to ~15 per replica. "
                     "Now discover that the queuing hockey stick makes batch=8 slower than "
                     "batch=32 at 100 QPS &mdash; and that switching from random to "
                     "Power-of-Two-Choices load balancing saves as much P99 latency as "
                     "adding entire GPU nodes.")

    mo.Html(f"""
    <div style="margin: 32px 0 12px 0;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <div style="background: {_act_color}; color: white; border-radius: 50%;
                        width: 32px; height: 32px; display: inline-flex; align-items: center;
                        justify-content: center; font-size: 0.9rem; font-weight: 800;
                        flex-shrink: 0;">{_act_num}</div>
            <div style="flex: 1; height: 2px; background: {COLORS['Border']};"></div>
            <div style="font-size: 0.72rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em;">
                Act {_act_num} &middot; {_act_duration}</div>
        </div>
        <div style="font-size: 1.5rem; font-weight: 800; color: {COLORS['Text']};
                    margin-top: 8px; line-height: 1.2;">
            {_act_title}
        </div>
        <div style="color: {COLORS['TextSec']}; font-size: 0.92rem; margin-top: 6px;
                    line-height: 1.55; max-width: 700px;">
            {_act_why}
        </div>
    </div>
    """)
    return


# ─── CELL 13: ACT2_STAKEHOLDER ──────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS, context_toggle):
    _ctx   = context_toggle.value
    _color = COLORS["Cloud"] if _ctx == "latency" else COLORS["BlueLine"]
    _bg    = COLORS["BlueL"]
    _persona = (
        "Platform Engineering Lead — Latency Track"
        if _ctx == "latency" else
        "Platform Engineering Lead — Throughput Track"
    )
    _quote = (
        "We're building a real-time assistant API. Enterprise customers have an SLA: "
        "P99 time-to-first-token must be under 200 ms. If the first token takes longer, "
        "the user sees a blank screen and the deal is dead. Design the serving configuration "
        "that meets this SLO even at peak load."
        if _ctx == "latency" else
        "We're building a batch document processing pipeline. Legal teams submit overnight "
        "jobs — 10,000 documents, each 4096 tokens, summarized by morning. Latency per "
        "document doesn't matter; what matters is total throughput (tokens/sec). "
        "Design the serving configuration that maximizes tokens processed per second."
    )

    mo.vstack([
        mo.md("---"),
        mo.md("""
        LLM inference is not a single workload. A real-time assistant and a batch pipeline
        have fundamentally different objectives. The same hardware configured two different
        ways produces wildly different performance profiles on the same model.

        The core tension: **large batches maximize GPU utilization and throughput, but
        force every new request to wait for a scheduling slot — directly inflating TTFT**.
        """),
        mo.Html(f"""
        <div style="border-left:4px solid {_color}; background:{_bg};
                    border-radius:0 10px 10px 0; padding:16px 22px; margin:12px 0;">
            <div style="font-size:0.72rem; font-weight:700; color:{_color};
                        text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                Incoming Message · {_persona}
            </div>
            <div style="font-style:italic; font-size:1.0rem; color:#1e293b; line-height:1.65;">
                "{_quote}"
            </div>
        </div>
        """),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — PREDICTION LOCK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act1_pred):
    mo.stop(act1_pred.value is None)

    act2_pred = mo.ui.radio(
        options={
            "A) Same configuration works for both — continuous batching handles all cases equally":
                "A",
            "B) Latency: small batch + low max-tokens; Throughput: large batch + continuous batching":
                "B",
            "C) Latency: no batching (batch=1); Throughput: static batching with batch=128":
                "C",
            "D) Both segments need identical configuration to avoid operational complexity":
                "D",
        },
        label=(
            "**Prediction Lock — Act II.** "
            "Two serving configurations must be designed: one for real-time assistant "
            "(P99 TTFT < 200ms) and one for batch document processing (maximize tokens/sec). "
            "Which approach is correct?"
        ),
    )
    mo.vstack([
        mo.Html("""
        <div style="background:#1e293b; border-radius:10px; padding:4px 18px 12px 18px;
                    border-left:4px solid #6366f1; margin-bottom:8px;">
            <div style="font-size:0.72rem; font-weight:700; color:#6366f1;
                        text-transform:uppercase; letter-spacing:0.1em;
                        margin-top:12px; margin-bottom:8px;">
                Prediction Lock · Act II
            </div>
        </div>
        """),
        act2_pred,
    ])
    return (act2_pred,)


@app.cell(hide_code=True)
def _(mo, act2_pred, act1_pred):
    mo.stop(act1_pred.value is None)
    mo.stop(
        act2_pred.value is None,
        mo.callout(mo.md("Select your prediction to continue."), kind="warn"),
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — SERVING CONFIGURATION INSTRUMENTS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act2_pred, act1_pred):
    mo.stop(act1_pred.value is None)
    mo.stop(act2_pred.value is None)

    # ── Latency-optimized config ──────────────────────────────────────────────
    lat_max_batch = mo.ui.slider(
        start=1, stop=64, value=4, step=1,
        label="Max batch size (latency config)",
        show_value=True,
    )
    lat_max_tokens = mo.ui.slider(
        start=256, stop=4096, value=512, step=256,
        label="Max sequence length per request — tokens (latency config)",
        show_value=True,
    )
    lat_serving_mode = mo.ui.dropdown(
        options={
            "Static batching": "static",
            "Continuous batching": "continuous",
            "Speculative decoding (draft model)": "speculative",
        },
        value="Continuous batching",
        label="Serving mode (latency config)",
    )

    # ── Throughput-optimized config ───────────────────────────────────────────
    tput_max_batch = mo.ui.slider(
        start=1, stop=128, value=64, step=1,
        label="Max batch size (throughput config)",
        show_value=True,
    )
    tput_max_tokens = mo.ui.slider(
        start=256, stop=8192, value=4096, step=256,
        label="Max sequence length per request — tokens (throughput config)",
        show_value=True,
    )
    tput_serving_mode = mo.ui.dropdown(
        options={
            "Static batching": "static",
            "Continuous batching": "continuous",
            "Speculative decoding (draft model)": "speculative",
        },
        value="Continuous batching",
        label="Serving mode (throughput config)",
    )

    mo.vstack([
        mo.md("### Simulator — Side-by-Side Serving Configuration"),
        mo.md("""
        Configure both serving modes independently. The KV-cache memory constraints
        from Act I apply to both. When you push batch size too high on the latency
        config, watch what happens to TTFT.
        """),
        mo.hstack([
            mo.vstack([
                mo.Html("""
                    <div style="background: #f0f4ff; border: 1.5px solid #c7d2fe;
                                border-top: 4px solid #6366f1; border-radius: 10px;
                                padding: 12px 16px; margin-bottom: 8px;">
                        <div style="font-weight: 800; font-size: 0.9rem; color: #3730a3;">
                            Config A — Latency-Optimized
                        </div>
                        <div style="font-size: 0.78rem; color: #4338ca; margin-top: 2px;">
                            Target: P99 TTFT &lt; 200ms
                        </div>
                    </div>
                """),
                lat_serving_mode,
                lat_max_batch,
                lat_max_tokens,
            ]),
            mo.vstack([
                mo.Html("""
                    <div style="background: #f0fdf4; border: 1.5px solid #bbf7d0;
                                border-top: 4px solid #008F45; border-radius: 10px;
                                padding: 12px 16px; margin-bottom: 8px;">
                        <div style="font-weight: 800; font-size: 0.9rem; color: #14532d;">
                            Config B — Throughput-Optimized
                        </div>
                        <div style="font-size: 0.78rem; color: #166534; margin-top: 2px;">
                            Target: &gt; 1,000 tokens/sec
                        </div>
                    </div>
                """),
                tput_serving_mode,
                tput_max_batch,
                tput_max_tokens,
            ]),
        ], gap=2),
    ])
    return (
        lat_max_batch, lat_max_tokens, lat_serving_mode,
        tput_max_batch, tput_max_tokens, tput_serving_mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — PHYSICS ENGINE + METRICS
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(
    mo, go, np, apply_plotly_theme, COLORS,
    lat_max_batch, lat_max_tokens, lat_serving_mode,
    tput_max_batch, tput_max_tokens, tput_serving_mode,
    H100_BW_GBS, H100_TFLOPS_FP16, H100_RAM_GB,
    LLAMA70B_LAYERS, LLAMA70B_HEADS, LLAMA70B_HEAD_DIM,
    BYTES_FP16, KV_TENSORS,
    LLAMA70B_WEIGHTS_GB,
    SLO_TTFT_MS, SLO_TPS_TARGET,
    act2_pred, act1_pred,
):
    mo.stop(act1_pred.value is None)
    mo.stop(act2_pred.value is None)

    # ── Serving physics model ─────────────────────────────────────────────────
    #
    # TTFT (time to first token) = prefill_time
    # Prefill is compute-bound: all prompt tokens processed in parallel.
    # Source: inference.qmd, @sec-inference-phases
    #
    # prefill_time_ms = (2 × seq_len × d_model²) / (TFLOPS × 1e12) × 1000
    # where d_model = num_heads × head_dim = 64 × 128 = 8192 for LLaMA-3 70B
    # Simplified: prefill_flops ≈ 2 × seq_len × n_layers × (4 × d_model²)
    # (4× for attention + FFN, 2× for multiply-add)
    #
    # With batching: time grows roughly linearly with batch size for static,
    # but continuous batching allows prefill to happen as requests arrive.
    #
    # Decode throughput = tokens per second = BW_GBS × 1e9 / (bytes_per_token)
    # bytes_per_token = model_params × bytes_per_param (weights loaded once per token)
    # LLaMA-3 70B FP16: 70e9 × 2 = 140 GB per full forward pass
    # decode_tps = H100_BW_GBS × 1e9 / (70e9 × BYTES_FP16) ≈ 47 tokens/sec (single batch)
    # With batch size B: tps ≈ B × 47 (up to compute saturation)

    _d_model        = LLAMA70B_HEADS * LLAMA70B_HEAD_DIM   # = 8192
    _param_bytes    = LLAMA70B_WEIGHTS_GB * (1024 ** 3)    # total weight bytes in memory
    _N_GPUS         = 8                                     # 8 H100s in fleet

    # Base decode tokens/sec per GPU (memory-bandwidth bound):
    # One token decode = one full model forward = load all weights once
    # tps_base = BW / model_bytes_per_GPU
    _model_bytes_per_gpu = _param_bytes / _N_GPUS            # tensor-parallel sharding
    _tps_base_per_batch  = (H100_BW_GBS * 1e9 * _N_GPUS) / _param_bytes  # ≈ 47 tok/s

    # Prefill FLOPS per token (approximate):
    # 2 × n_layers × (4 × d_model²) [attention + FFN, fwd pass]
    # source: Kaplan et al. (2020), Chinchilla scaling laws
    _prefill_flops_per_token = 2 * LLAMA70B_LAYERS * 4 * (_d_model ** 2)
    _cluster_tflops = H100_TFLOPS_FP16 * _N_GPUS * 1e12   # total FLOPS/s (tensor parallel)

    def _compute_config(batch_size, seq_len, mode):
        """
        Compute TTFT (ms), throughput (tok/s), GPU utilization (%)
        for a given serving configuration.

        Physics assumptions (from @sec-inference-phases, @sec-continuous-batching):
        - TTFT dominated by prefill time: t = (flops × batch) / cluster_flops
        - For continuous batching: new requests start prefill immediately,
          so they don't wait for existing decode to finish; max queue depth ≈ 2–4 requests.
        - Static batching: TTFT includes waiting for the slowest request in the current batch.
        - Speculative decoding: reduces per-token latency by ~2.5× via draft-model acceptance;
          prefill time unchanged.
        - Throughput: limited by memory bandwidth (decode phase) × batch size,
          capped by compute (prefill phase).
        """
        # Prefill time: batch_size requests processed sequentially (static)
        # or in parallel (continuous, overlapped with decode)
        _batch_factor = batch_size if mode == "static" else min(batch_size, 4)
        _prefill_time_s = (
            _prefill_flops_per_token * seq_len * _batch_factor / _cluster_tflops
        )
        _ttft_ms = _prefill_time_s * 1000.0

        # Speculative decoding: draft model reduces decode latency ~2.5×
        # but does not reduce prefill time (full model still used for verification)
        _spec_factor = 1.0 if mode != "speculative" else 1.0  # TTFT unchanged

        _ttft_ms = _ttft_ms * _spec_factor

        # Throughput (decode phase, memory-bandwidth bound):
        # tps = batch_size × tps_base_per_batch, capped at compute limit
        # Compute limit: prefill + decode share the cluster
        # Simplified: utilization = batch_size / max_useful_batch
        _max_useful_batch = 48  # empirical batch size where compute saturates H100x8
        _tps = min(batch_size * _tps_base_per_batch,
                   _max_useful_batch * _tps_base_per_batch)

        # Speculative decoding boosts effective tps ~2.5× via draft acceptance
        if mode == "speculative":
            _tps *= 2.5

        # GPU utilization approximation:
        # At low batch sizes, mostly idle (memory-bandwidth limited, waiting for tokens)
        # Saturates around batch = 32–48 for H100x8 at this model size
        _gpu_util = min(99.0, (batch_size / _max_useful_batch) * 100.0 * 1.2)

        # KV-cache memory check: does this batch fit in 500 GB available?
        _kv_bytes_per_token = KV_TENSORS * LLAMA70B_LAYERS * LLAMA70B_HEADS * LLAMA70B_HEAD_DIM * BYTES_FP16
        _kv_per_user_gb = _kv_bytes_per_token * seq_len / (1024 ** 3)
        _total_kv_gb = _kv_per_user_gb * batch_size
        _available_kv_gb = H100_RAM_GB * _N_GPUS - LLAMA70B_WEIGHTS_GB
        _oom = _total_kv_gb > _available_kv_gb

        return {
            "ttft_ms":         round(_ttft_ms, 1),
            "tps":             round(_tps, 0),
            "gpu_util_pct":    round(_gpu_util, 1),
            "kv_total_gb":     round(_total_kv_gb, 1),
            "available_kv_gb": round(_available_kv_gb, 1),
            "oom":             _oom,
        }

    # ── Evaluate both configurations ─────────────────────────────────────────
    _lat = _compute_config(
        lat_max_batch.value, lat_max_tokens.value, lat_serving_mode.value
    )
    _tput = _compute_config(
        tput_max_batch.value, tput_max_tokens.value, tput_serving_mode.value
    )

    # ── Constraint checks ─────────────────────────────────────────────────────
    _lat_slo_ok   = _lat["ttft_ms"] <= SLO_TTFT_MS
    _tput_slo_ok  = _tput["tps"] >= SLO_TPS_TARGET
    _lat_oom      = _lat["oom"]
    _tput_oom     = _tput["oom"]

    # ── Color helpers ─────────────────────────────────────────────────────────
    def _ttft_color(ms):
        if ms > SLO_TTFT_MS:     return COLORS["RedLine"]
        if ms > SLO_TTFT_MS * 0.7: return COLORS["OrangeLine"]
        return COLORS["GreenLine"]

    def _tps_color(tps):
        if tps >= SLO_TPS_TARGET:      return COLORS["GreenLine"]
        if tps >= SLO_TPS_TARGET * 0.6: return COLORS["OrangeLine"]
        return COLORS["RedLine"]

    def _util_color(u):
        if u > 90:  return COLORS["OrangeLine"]
        if u > 50:  return COLORS["GreenLine"]
        return COLORS["TextMuted"]

    # ── Metric cards HTML ─────────────────────────────────────────────────────
    def _metric_card(label, value_str, sub, color, width="160px"):
        return f"""
        <div style="padding: 16px 20px; border: 1px solid #e2e8f0; border-radius: 10px;
                    min-width: {width}; text-align: center; background: white;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
            <div style="color: #64748b; font-size: 0.78rem; font-weight: 600;
                        text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: {color};
                        font-family: 'SF Mono', monospace; margin: 4px 0;">{value_str}</div>
            <div style="color: #94a3b8; font-size: 0.72rem;">{sub}</div>
        </div>
        """

    _lat_cards = "".join([
        _metric_card("TTFT P99", f"{_lat['ttft_ms']:.0f} ms",
                     f"SLO: {SLO_TTFT_MS} ms",
                     _ttft_color(_lat["ttft_ms"])),
        _metric_card("Throughput", f"{_lat['tps']:,.0f}",
                     "tokens / sec",
                     _tps_color(_lat["tps"])),
        _metric_card("GPU util", f"{_lat['gpu_util_pct']:.0f}%",
                     f"batch={lat_max_batch.value}",
                     _util_color(_lat["gpu_util_pct"])),
        _metric_card("KV memory", f"{_lat['kv_total_gb']:.0f} GB",
                     f"avail: {_lat['available_kv_gb']:.0f} GB",
                     COLORS["RedLine"] if _lat_oom else COLORS["GreenLine"]),
    ])
    _tput_cards = "".join([
        _metric_card("TTFT P99", f"{_tput['ttft_ms']:.0f} ms",
                     f"SLO: {SLO_TTFT_MS} ms",
                     _ttft_color(_tput["ttft_ms"])),
        _metric_card("Throughput", f"{_tput['tps']:,.0f}",
                     "tokens / sec",
                     _tps_color(_tput["tps"])),
        _metric_card("GPU util", f"{_tput['gpu_util_pct']:.0f}%",
                     f"batch={tput_max_batch.value}",
                     _util_color(_tput["gpu_util_pct"])),
        _metric_card("KV memory", f"{_tput['kv_total_gb']:.0f} GB",
                     f"avail: {_tput['available_kv_gb']:.0f} GB",
                     COLORS["RedLine"] if _tput_oom else COLORS["GreenLine"]),
    ])

    # ── Comparison bar chart ──────────────────────────────────────────────────
    _fig2 = go.Figure()
    _metrics_names = ["TTFT (ms)", "Throughput (tok/s)", "GPU util (%)"]
    _lat_vals  = [_lat["ttft_ms"], _lat["tps"], _lat["gpu_util_pct"]]
    _tput_vals = [_tput["ttft_ms"], _tput["tps"], _tput["gpu_util_pct"]]

    _fig2.add_trace(go.Bar(
        name="Config A — Latency",
        x=_metrics_names,
        y=[
            _lat["ttft_ms"] / SLO_TTFT_MS * 100,      # pct of SLO
            _lat["tps"] / SLO_TPS_TARGET * 100,         # pct of throughput target
            _lat["gpu_util_pct"],                        # raw pct
        ],
        marker_color=COLORS["Cloud"],
        text=[
            f"{_lat['ttft_ms']:.0f} ms",
            f"{_lat['tps']:,.0f} tok/s",
            f"{_lat['gpu_util_pct']:.0f}%",
        ],
        textposition="outside",
        width=0.35,
    ))
    _fig2.add_trace(go.Bar(
        name="Config B — Throughput",
        x=_metrics_names,
        y=[
            _tput["ttft_ms"] / SLO_TTFT_MS * 100,
            _tput["tps"] / SLO_TPS_TARGET * 100,
            _tput["gpu_util_pct"],
        ],
        marker_color=COLORS["GreenLine"],
        text=[
            f"{_tput['ttft_ms']:.0f} ms",
            f"{_tput['tps']:,.0f} tok/s",
            f"{_tput['gpu_util_pct']:.0f}%",
        ],
        textposition="outside",
        width=0.35,
    ))
    _fig2.add_hline(
        y=100,
        line_dash="dash",
        line_color=COLORS["OrangeLine"],
        line_width=1.5,
        annotation_text="SLO / Target (100%)",
        annotation_position="top right",
        annotation_font_color=COLORS["OrangeLine"],
        annotation_font_size=10,
    )
    _fig2.update_layout(
        title=dict(text="Configuration Comparison (% of SLO or Target)", font_size=13),
        yaxis_title="Percentage of SLO / target (%)",
        barmode="group",
        showlegend=True,
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    apply_plotly_theme(_fig2)

    # ── Failure and warning callouts ──────────────────────────────────────────
    _alerts = []

    # Config A failure state (kind="danger") — TTFT SLO violated
    if _lat_oom:
        _alerts.append(mo.callout(
            mo.md(
                f"**Config A — OOM: KV-cache exceeds budget.** "
                f"Required: {_lat['kv_total_gb']:.0f} GB | "
                f"Available: {_lat['available_kv_gb']:.0f} GB. "
                "Reduce max batch size or context length."
            ),
            kind="danger",
        ))
    elif not _lat_slo_ok:
        _alerts.append(mo.callout(
            mo.md(
                f"**Latency SLO violated: P99 TTFT = {_lat['ttft_ms']:.0f} ms > {SLO_TTFT_MS} ms target.** "
                "Reduce max batch size or enable prefill chunking to lower time-to-first-token."
            ),
            kind="danger",
        ))

    # Config B warning — throughput below target
    if _tput_oom:
        _alerts.append(mo.callout(
            mo.md(
                f"**Config B — OOM: KV-cache exceeds budget.** "
                f"Required: {_tput['kv_total_gb']:.0f} GB | "
                f"Available: {_tput['available_kv_gb']:.0f} GB. "
                "Reduce max batch size or context length."
            ),
            kind="danger",
        ))
    elif not _tput_slo_ok:
        _alerts.append(mo.callout(
            mo.md(
                f"**Batch throughput below target.** "
                f"Current: {_tput['tps']:,.0f} tok/s | Target: {SLO_TPS_TARGET:,} tok/s. "
                "Increase max batch size or switch to continuous batching."
            ),
            kind="warn",
        ))

    if _lat_slo_ok and not _lat_oom:
        _alerts.append(mo.callout(
            mo.md(
                f"**Config A — Latency SLO met.** "
                f"P99 TTFT = {_lat['ttft_ms']:.0f} ms (target: {SLO_TTFT_MS} ms)."
            ),
            kind="success",
        ))
    if _tput_slo_ok and not _tput_oom:
        _alerts.append(mo.callout(
            mo.md(
                f"**Config B — Throughput target met.** "
                f"{_tput['tps']:,.0f} tok/s (target: {SLO_TPS_TARGET:,} tok/s)."
            ),
            kind="success",
        ))

    mo.vstack([
        mo.md("### Results — Side-by-Side Comparison"),
        mo.md("**Config A — Latency-Optimized**"),
        mo.Html(f'<div style="display:flex; gap:12px; flex-wrap:wrap; margin:10px 0;">{_lat_cards}</div>'),
        mo.md("**Config B — Throughput-Optimized**"),
        mo.Html(f'<div style="display:flex; gap:12px; flex-wrap:wrap; margin:10px 0;">{_tput_cards}</div>'),
        mo.as_html(_fig2),
        *_alerts,
    ])
    return (_lat, _tput)


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — PREDICTION vs REALITY OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act2_pred, _lat, _tput, SLO_TTFT_MS, SLO_TPS_TARGET, act1_pred):
    mo.stop(act1_pred.value is None)
    mo.stop(act2_pred.value is None)

    _choice = act2_pred.value
    _lat_ok  = _lat["ttft_ms"] <= SLO_TTFT_MS
    _tput_ok = _tput["tps"] >= SLO_TPS_TARGET

    _feedback_map = {
        "A": (
            "**Incorrect.** Continuous batching is a powerful technique, but it does not "
            "eliminate the distinction between latency and throughput workloads. At large batch "
            "sizes, the prefill of existing requests in the batch still delays time-to-first-token "
            "for newly arriving requests. Continuous batching helps throughput by eliminating "
            "static batch idle time — it does not compress prefill latency."
        ),
        "B": (
            "**Correct.** The latency config uses a small batch (short prefill queue) and "
            "capped sequence length (less work per prefill step), ensuring TTFT stays under "
            "the 200ms SLO. The throughput config uses a large batch and continuous batching: "
            "GPU utilization stays high, new requests slot in immediately when a decode sequence "
            "finishes, and the throughput target is met. These configurations genuinely require "
            "different operational parameters."
        ),
        "C": (
            "**Partially correct, but not optimal.** Batch size of 1 for latency does minimize "
            "TTFT, but it leaves GPU utilization near zero — extremely expensive per request. "
            "Small batch sizes (4–8) with continuous batching allow several requests to be "
            "processed simultaneously while still meeting TTFT SLOs. Static batching with "
            "batch=128 for throughput is also suboptimal: the cluster must wait for the slowest "
            "sequence in the batch before accepting new requests, wasting capacity."
        ),
        "D": (
            "**Incorrect.** Operational simplicity is not a sufficient reason to use an "
            "incorrect configuration. A single configuration optimized for throughput would "
            "violate P99 TTFT SLOs for latency-sensitive users. A single configuration "
            "optimized for latency would dramatically underutilize the cluster for batch jobs. "
            "Both segments require distinct configurations."
        ),
    }

    _correct = _choice == "B"
    mo.vstack([
        mo.md("### Act II Prediction vs Reality"),
        mo.callout(
            mo.md(_feedback_map[_choice]),
            kind="success" if _correct else "warn",
        ),
    ])
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — REFLECTION
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act2_pred, act1_pred):
    mo.stop(act1_pred.value is None)
    mo.stop(act2_pred.value is None)

    act2_reflect = mo.ui.radio(
        options={
            "A) Continuous batching uses less GPU memory than static batching":
                "A",
            "B) Continuous batching allows new requests to join when a sequence finishes decode, eliminating the waste of waiting for the slowest sequence in a static batch":
                "B",
            "C) Continuous batching always produces lower latency than static batching":
                "C",
            "D) Continuous batching enables longer context lengths by compressing KV-cache":
                "D",
        },
        label=(
            "**Reflection — Act II.** "
            "What is the key advantage of continuous batching over static batching "
            "for LLM serving throughput?"
        ),
    )
    mo.vstack([
        mo.md("---"),
        mo.md("### Reflection"),
        act2_reflect,
    ])
    return (act2_reflect,)


@app.cell(hide_code=True)
def _(mo, act2_reflect, act1_pred, act2_pred):
    mo.stop(act1_pred.value is None)
    mo.stop(act2_pred.value is None)
    mo.stop(
        act2_reflect.value is None,
        mo.callout(mo.md("Select your answer to continue."), kind="warn"),
    )

    _reflect_feedback = {
        "A": (
            "**Not quite.** Continuous batching does not reduce total KV-cache memory — "
            "in fact it requires more complex memory management because requests of different "
            "lengths co-exist in the batch. Paged attention (vLLM) is the technique that "
            "improves memory utilization; continuous batching addresses *scheduling* efficiency, "
            "not memory capacity."
        ),
        "B": (
            "**Correct.** In static batching, the server waits until an entire batch finishes "
            "before accepting the next batch. If one request generates 2048 tokens while others "
            "generate 64, the server is idle waiting for that one long request. Continuous "
            "batching inserts new requests the moment any sequence in the current batch finishes "
            "its decode step. GPU utilization rises dramatically because there are almost no "
            "idle decode slots. This is the key insight behind production LLM serving systems "
            "like vLLM and TGI."
        ),
        "C": (
            "**Not quite.** Continuous batching can *increase* latency for individual requests "
            "because more requests share the GPU simultaneously, and each decode step processes "
            "a larger combined batch. The benefit is throughput — more total tokens per second "
            "from the system. Latency per request may be higher or lower depending on batch "
            "composition, serving mode, and SLO configuration."
        ),
        "D": (
            "**Not quite.** Continuous batching has nothing to do with KV-cache compression "
            "or context length. That is the domain of paged attention, sliding window attention, "
            "and KV-cache quantization. Continuous batching is purely a scheduling policy: "
            "it determines when new requests are admitted into the active decode batch."
        ),
    }

    _correct = act2_reflect.value == "B"
    mo.callout(
        mo.md(_reflect_feedback[act2_reflect.value]),
        kind="success" if _correct else "warn",
    )
    return


# ─────────────────────────────────────────────────────────────────────────────
# ACT II — MATHPEEK
# ─────────────────────────────────────────────────────────────────────────────

@app.cell(hide_code=True)
def _(mo, act2_pred, act1_pred, H100_BW_GBS, LLAMA70B_PARAMS_B, BYTES_FP16):
    mo.stop(act1_pred.value is None)
    mo.stop(act2_pred.value is None)

    # Decode tps calculation for display
    _tps_single = H100_BW_GBS * 1e9 / (LLAMA70B_PARAMS_B * 1e9 * BYTES_FP16)
    _tps_8gpu   = _tps_single * 8

    mo.accordion({
        "The governing equations — TTFT, throughput, continuous batching": mo.md(f"""
        **Time To First Token — Prefill Phase** (compute-bound):

        ```
        TTFT ≈ prefill_flops / cluster_FLOPS
             = (2 × L × seq_len × 4 × d_model²) / (TFLOPS × n_gpus × 10¹²)
        ```

        - Prefill processes all prompt tokens in parallel (compute-bound)
        - Longer context → more prefill work → higher TTFT
        - Larger batch → more concurrent prefills → TTFT grows linearly (static)
        - Continuous batching: limits queue depth → TTFT bounded

        **Decode Throughput — Decode Phase** (memory-bandwidth-bound):

        ```
        tps_single_H100 = BW / model_bytes
                        = {H100_BW_GBS} GB/s × 10⁹ / ({LLAMA70B_PARAMS_B}B × {BYTES_FP16} bytes)
                        = {_tps_single:.0f} tokens/sec

        tps_8xH100 (tensor parallel) ≈ {_tps_8gpu:.0f} tok/s × batch_size
        ```

        - Decode is memory-bandwidth-bound: every token requires loading all model weights once
        - Batch size multiplies effective throughput (all sequences share one weight read)
        - Maximum useful batch size ≈ 32–48 for LLaMA-3 70B on 8× H100 (compute saturation)

        **Continuous Batching vs Static Batching:**

        ```
        Static:     batch waits for slowest sequence → GPU idle time ∝ output length variance
        Continuous: new request fills slot immediately when any sequence finishes decode
                    → GPU utilization from ~50% to >90%
        ```

        **Speculative Decoding Speedup** (from inference.qmd):

        ```
        effective_tps ≈ tps_base × α × (1 + acceptance_rate × draft_length)
        ```

        where α ≈ draft acceptance rate and draft_length ≈ 4–8 tokens per step.
        Typical speedup: 2–3× for coherent text with well-matched draft model.
        """),
    })
    return


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE D: CLOSING
# ═══════════════════════════════════════════════════════════════════════════════

# ─── CELL 20: SYNTHESIS ─────────────────────────────────────────────────────────
@app.cell(hide_code=True)
def _(mo, COLORS):
    mo.vstack([
        mo.md("---"),

        # ── KEY TAKEAWAYS ──
        mo.Html(f"""
        <div style="background: {COLORS['Surface2']}; border: 1px solid {COLORS['Border']};
                    border-radius: 12px; padding: 24px 28px; margin: 16px 0;">
            <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['TextMuted']};
                        text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;">
                Key Takeaways
            </div>
            <div style="font-size: 0.92rem; color: {COLORS['Text']}; line-height: 1.75;">
                <div style="margin-bottom: 10px;">
                    <strong>1. KV-cache grows linearly with context length and limits concurrency severely.</strong>
                    At 2.6 MB/token for Llama-70B (FP16), a 4096-token request requires 10.65 GB
                    of cache. A 4x H100 NVLink replica with 160 GB available after weights can
                    serve at most 15 concurrent requests &mdash; not hundreds.
                </div>
                <div style="margin-bottom: 10px;">
                    <strong>2. Minimizing per-request service time can maximize total latency.</strong>
                    At 100 QPS, batch=8 (S=54 ms, rho=67.5%) has 42.3 ms queuing delay for
                    123.2 ms total. Batch=32 (S=66 ms, rho=20.6%) has 8.9 ms queuing delay
                    for 91.5 ms total &mdash; 26% lower despite 22% longer service time.
                    The hockey stick is not intuitive until you see the numbers.
                </div>
                <div>
                    <strong>3. Load balancing algorithm choice has quantitative P99 impact.</strong>
                    Power-of-Two-Choices reduces P99 from 45 ms to 24 ms (47% improvement)
                    versus random assignment on the same fleet. This is free &mdash; a configuration
                    change worth as much as adding entire GPU nodes to the fleet.
                </div>
            </div>
        </div>
        """),

        # ── CONNECTIONS ──
        mo.Html(f"""
        <div style="display: flex; gap: 16px; margin: 8px 0 16px 0; flex-wrap: wrap;">

            <!-- What's Next -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['BlueLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    What's Next
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Lab 11: Edge Intelligence</strong> &mdash; This lab showed that
                    inference at cloud scale is memory-constrained. The next lab asks: what
                    happens when you move that same model to a device with 8 GB total RAM
                    and 300 MB available for ML? The memory amplification factor (3&ndash;12x
                    for training) makes on-device adaptation an engineering crisis.
                </div>
            </div>

            <!-- Textbook Connection -->
            <div style="flex: 1; min-width: 280px; background: white;
                        border: 1px solid {COLORS['Border']}; border-radius: 12px;
                        padding: 20px 24px;">
                <div style="font-size: 0.7rem; font-weight: 700; color: {COLORS['GreenLine']};
                            text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 8px;">
                    Textbook &amp; TinyTorch
                </div>
                <div style="font-size: 0.88rem; color: {COLORS['TextSec']}; line-height: 1.6;">
                    <strong>Read:</strong> @sec-inference-scale-kv-cache-wall for the full
                    KV-cache derivation and PagedAttention motivation. See the Serving Cost
                    Dominance Law callout for the 9x serving/training cost inversion.<br/>
                    <strong>Build:</strong> TinyTorch Module 20 &mdash; implement a KV-cache
                    allocator with paged memory management and measure its impact on concurrent
                    request throughput.
                </div>
            </div>

        </div>
        """),

        mo.accordion({


            "Self-Assessment": mo.md("""
**Check your understanding:**

1. Llama-70B in FP16 requires 2.6 MB/token of KV cache. At 4096-token context, each request needs 10.65 GB. On a 4x H100 NVLink replica with 160 GB available after weights, what is the maximum concurrent request count, and why is this far fewer than most engineers expect?
2. At 100 QPS, batch=8 has lower service time (54 ms) but higher total latency (123 ms) than batch=32 (66 ms service, 91.5 ms total). What queuing dynamics explain this counterintuitive result, and where does the hockey-stick curve inflect?
3. Power-of-Two-Choices load balancing reduces P99 from 45 ms to 24 ms versus random assignment with zero hardware cost. Why does sampling just two candidates and picking the less-loaded one achieve such a large improvement?

**You're ready to move on if you can:**
- Calculate KV-cache memory requirements and maximum concurrency for a given model and GPU configuration
- Explain why minimizing per-request service time can maximize total latency due to queuing effects
- Compare load balancing algorithms by their quantitative P99 impact on inference serving
""")


        }),
    ])
    return


# ─── CELL 21: LEDGER_HUD ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# DESIGN LEDGER SAVE + HUD FOOTER
# ─────────────────────────────────────────────────────────────────────────────

@app.cell
@app.cell(hide_code=True)
def _(mo):
    decision_input, decision_ui = DecisionLog()
    return decision_input, decision_ui


@app.cell(hide_code=True)
def _(
    mo, ledger, COLORS,
    context_toggle,
    act1_pred, act2_pred, act2_reflect,
    _max_users, _kv_per_user_gb,
    _lat, _tput,
    lat_max_batch, tput_max_batch,
    lat_max_tokens, tput_max_tokens,
    lat_serving_mode, tput_serving_mode,
    SLO_TTFT_MS, SLO_TPS_TARGET,
    LLAMA70B_PARAMS_B,
, decision_input, decision_ui):
    # Only save if student has progressed through both acts
    _a1 = act1_pred.value
    _a2 = act2_pred.value if act2_pred.value is not None else "unanswered"
    _ctx = context_toggle.value

    _a1_correct = _a1 == "C" if _a1 else False
    _a2_correct = _a2 == "B"

    _lat_slo_ok  = _lat["ttft_ms"] <= SLO_TTFT_MS
    _tput_slo_ok = _tput["tps"] >= SLO_TPS_TARGET
    _slo_met     = _lat_slo_ok and _tput_slo_ok
    _constraint  = _lat["oom"] or _tput["oom"]

    ledger.save(
        chapter="v2_10",
        design={
            "context":          _ctx,
            "model_params_b":   float(LLAMA70B_PARAMS_B),
            "context_length":   lat_max_tokens.value,
            "max_batch_size":   lat_max_batch.value,
            "kv_cache_gb":      round(float(_kv_per_user_gb), 2) if isinstance(_kv_per_user_gb, float) else 0.0,
            "ttft_ms":          float(_lat["ttft_ms"]),
            "throughput_tps":   float(_tput["tps"]),
            "act1_prediction":  _a1 or "unanswered",
            "act1_correct":     _a1_correct,
            "act2_result":      float(_lat["ttft_ms"]),
            "act2_decision":    lat_serving_mode.value,
            "constraint_hit":   _constraint,
        "student_justification": str(decision_input.value),
            "slo_met":          _slo_met,
        }
    )

    # ── HUD footer ────────────────────────────────────────────────────────────
    _hud_items = [
        ("LAB", "V2-10 · Distributed Inference"),
        ("CONTEXT", _ctx.upper()),
        ("ACT I", "CORRECT" if _a1_correct else ("WRONG" if _a1 else "OPEN")),
        ("ACT II PRED", "CORRECT" if _a2_correct else ("WRONG" if _a2 != "unanswered" else "OPEN")),
        ("TTFT (LAT)", f"{_lat['ttft_ms']:.0f} ms"),
        ("TPS (BATCH)", f"{_tput['tps']:,.0f} tok/s"),
        ("KV/USER",    f"{_kv_per_user_gb:.1f} GB" if isinstance(_kv_per_user_gb, float) else "—"),
        ("MAX USERS",  str(_max_users)),
        ("SLO",        "MET" if _slo_met else "VIOLATED"),
        ("LEDGER",     "SAVED"),
    ]

    def _hud_cell(label, value):
        is_ok    = value in ("CORRECT", "MET", "SAVED")
        is_bad   = value in ("WRONG", "VIOLATED")
        v_color  = (
            "#4ade80" if is_ok else
            "#f87171" if is_bad else
            "#e2e8f0"
        )
        return f"""
        <div style="display:flex; flex-direction:column; align-items:flex-start; gap:1px;">
            <span class="hud-label" style="color:{COLORS['TextMuted']}; font-size:0.62rem;
                                           font-weight:700; letter-spacing:0.08em;
                                           text-transform:uppercase;">{label}</span>
            <span style="color:{v_color}; font-family:'SF Mono',monospace;
                         font-size:0.78rem; font-weight:700;">{value}</span>
        </div>
        """

    _hud_inner = "".join([_hud_cell(l, v) for l, v in _hud_items])

    mo.Html(f"""
    <div style="display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap;
                padding:14px 24px; background:{COLORS['Surface0']};
                border-radius:12px; margin-top:32px;
                border:1px solid {COLORS['Surface1']};">
        {_hud_inner}
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
